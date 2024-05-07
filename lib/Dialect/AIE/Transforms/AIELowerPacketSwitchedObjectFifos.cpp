//===- AIELowerPacketSwitchedObjectFifos.cpp ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>
#include <set>
#include <optional>

#define DEBUG_TYPE "aie-lower-packet-sw-objectFifos"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// Lock Analysis
//===----------------------------------------------------------------------===//
class LockAnalysis {
  DenseMap<std::pair<Value, int>, int> locksPerTile;

public:
  LockAnalysis(DeviceOp &device) {
    // go over the locks created for each tile and update the index in
    // locksPerTile
    for (auto lockOp : device.getOps<LockOp>()) {
      auto tile = lockOp.getTile();
      auto lockID = lockOp.getLockIDValue();
      locksPerTile[{tile, lockID}] = 1;
    }
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(TileOp &tileOp) {
    const auto &targetModel = getTargetModel(tileOp);
    for (unsigned i = 0;
         i < targetModel.getNumLocks(tileOp.getCol(), tileOp.getRow()); i++)
      if (int usageCnt = locksPerTile[{tileOp, i}]; usageCnt == 0) {
        locksPerTile[{tileOp, i}] = 1;
        return i;
      }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// TileDMA Channel Analysis
//===----------------------------------------------------------------------===//
class DMAChannelAnalysis {
  DenseMap<Value, int> masterChannelsPerTile;
  DenseMap<Value, int> slaveChannelsPerTile;
  int packetId;

public:
  DMAChannelAnalysis(DeviceOp &device) {
    // go over the channels used for each tile and update the master/slave
    // channel maps
    for (auto memOp : device.getOps<MemOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          if (op.isSend())
            getMasterDMAChannel(memOp.getTile());
          else
            getSlaveDMAChannel(memOp.getTile());
        }
      }
    }
    packetId = 0; // TODO: check if other IDs have already been used
  }

  // Returns a DMABDPacket with the next available unique packet ID.
  DMABDPacket getDMABDPacket() {
    int lastPacketId = packetId;
    packetId++;
    return {lastPacketId, lastPacketId};
  }

  /// Given an AIE tile, returns its next usable master channel.
  DMAChannel getMasterDMAChannel(Value tile) {
    if (masterChannelsPerTile.find(tile) == masterChannelsPerTile.end()) {
      masterChannelsPerTile[tile] = 0;
    } else {
      assert([&] {
        auto tileOp = tile.getDefiningOp<TileOp>();
        int numChannels = tileOp.getNumSourceConnections(WireBundle::DMA);
        if (masterChannelsPerTile[tile] >= numChannels - 1) {
          printf("All tile DMA master channels are already in use.\n");
          return false;
        }
        return true;
      }());
      masterChannelsPerTile[tile]++;
    }
    DMAChannel dmaChan = {DMAChannelDir::MM2S, masterChannelsPerTile[tile]};
    return dmaChan;
  }

  /// Given an AIE tile, returns its next usable slave channel.
  DMAChannel getSlaveDMAChannel(Value tile) {
    if (slaveChannelsPerTile.find(tile) == slaveChannelsPerTile.end()) {
      slaveChannelsPerTile[tile] = 0;
    } else {
      assert([&] {
        auto tileOp = tile.getDefiningOp<TileOp>();
        int numChannels = tileOp.getNumDestConnections(WireBundle::DMA);
        if (slaveChannelsPerTile[tile] >= numChannels - 1) {
          printf("All tile DMA slave channels are already in use.\n");
          return false;
        }
        return true;
      }());
      slaveChannelsPerTile[tile]++;
    }
    DMAChannel dmaChan = {DMAChannelDir::S2MM, slaveChannelsPerTile[tile]};
    return dmaChan;
  }
};

struct AIELowerPacketSwitchedObjectFifosPass
    : AIELowerPacketSwitchedObjectFifosBase<AIELowerPacketSwitchedObjectFifosPass> {
  DenseMap<PacketSwitchedObjectFifoOp, std::vector<BufferOp>>
      buffersPerFifo; // maps each objFifo to its corresponding buffer
  DenseMap<PacketSwitchedObjectFifoOp, std::vector<ExternalBufferOp>>
      externalBuffersPerFifo; // maps each objFifo to its corresponding
  // external buffers
  DenseMap<PacketSwitchedObjectFifoOp, std::vector<LockOp>>
      locksPerFifo; // maps each objFifo to its corresponding locks
  std::vector<std::pair<PacketSwitchedObjectFifoOp, std::vector<PacketSwitchedObjectFifoOp>>>
      splitFifos; // maps each objFifo between non-adjacent tiles to its
  // corresponding consumer objectFifos

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * -1 if the shared memory module is that of the first input tile,
  ///   * 1 if it is that of the second input tile,
  ///   * 0 is no memory module is shared.
  bool isSharedMemory(TileOp a, TileOp b, int *share_direction) {
    const auto &targetModel = getTargetModel(a.getOperation());

    if ((a.isShimTile() && !b.isShimTile()) ||
        (!a.isShimTile() && b.isShimTile())) {
      *share_direction = 0;
      return false;
    }
    if ((targetModel.isMemTile(a.getCol(), a.getRow()) &&
         !targetModel.isMemTile(b.getCol(), b.getRow())) ||
        (!targetModel.isMemTile(a.getCol(), a.getRow()) &&
         targetModel.isMemTile(b.getCol(), b.getRow()))) {
      *share_direction = 0;
      return false;
    }
    bool rightShared = targetModel.isLegalMemAffinity(
        a.colIndex(), a.rowIndex(), b.colIndex(), b.rowIndex());

    bool leftShared = targetModel.isLegalMemAffinity(
        b.colIndex(), b.rowIndex(), a.colIndex(), a.rowIndex());

    if (leftShared)
      *share_direction = -1;
    else if (rightShared)
      *share_direction = 1;
    else
      *share_direction = 0;

    return leftShared || rightShared;
  }

  // Return true if the objectFifo created by createOp requires a DMA to be set
  // up. This is the case if the tiles are not adjacent (no shared memory), if
  // the objectFifo broadcasts to multiple tiles, or if one of the consumers
  // or the producer wants to use the multi-dimensional address generation
  // features of the DMA.
  bool requiresDMAs(PacketSwitchedObjectFifoOp createOp, int &share_direction) {
    bool hasSharedMemory = false;

    if (createOp.getConsumerTiles().size() == 1) {

      // Test for shared memory
      for (auto consumerTile : createOp.getConsumerTiles()) {
        if (auto consumerTileOp =
                dyn_cast<TileOp>(consumerTile.getDefiningOp());
            isSharedMemory(createOp.getProducerTileOp(), consumerTileOp,
                           &share_direction))
          hasSharedMemory = true;
      }
    }

    return !hasSharedMemory;
  }

  PacketSwitchedObjectFifoOp
  createObjectFifo(OpBuilder &builder, AIEObjectFifoType datatype,
                   std::string name, Value prodTile, Value consTile,
                   Attribute depth) {
    auto ofName = builder.getStringAttr(name);
    auto fifo = builder.create<PacketSwitchedObjectFifoOp>(
        builder.getUnknownLoc(), ofName, prodTile, consTile, depth, datatype);
    return fifo;
  }

  /// Function used to create objectFifo locks based on target architecture.
  /// Called by createObjectFifoElements().
  std::vector<LockOp> createObjectFifoLocks(OpBuilder &builder,
                                            LockAnalysis &lockAnalysis,
                                            PacketSwitchedObjectFifoOp op, int numElem,
                                            TileOp creation_tile) {
    std::vector<LockOp> locks;
    auto dev = op->getParentOfType<DeviceOp>();
    auto &target = dev.getTargetModel();
    if (creation_tile.isShimTile())
      numElem = externalBuffersPerFifo[op].size();
    if (target.getTargetArch() == AIEArch::AIE1) {
      int of_elem_index = 0; // used to give objectFifo elements a symbolic name
      for (int i = 0; i < numElem; i++) {
        // create corresponding aie1 locks
        int lockID = lockAnalysis.getLockID(creation_tile);
        assert(lockID >= 0 && "No more locks to allocate!");
        auto lock = builder.create<LockOp>(builder.getUnknownLoc(),
                                           creation_tile, lockID, 0);
        lock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_lock_" +
                                  std::to_string(of_elem_index)));
        locks.push_back(lock);
        of_elem_index++;
      }
    } else {
      // create corresponding aie2 locks
      int prodLockID = lockAnalysis.getLockID(creation_tile);
      assert(prodLockID >= 0 && "No more locks to allocate!");
      auto prodLock = builder.create<LockOp>(
          builder.getUnknownLoc(), creation_tile, prodLockID, numElem);
      prodLock.getOperation()->setAttr(
          SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name().str() + "_prod_lock"));
      locks.push_back(prodLock);

      int consLockID = lockAnalysis.getLockID(creation_tile);
      assert(consLockID >= 0 && "No more locks to allocate!");
      auto consLock = builder.create<LockOp>(builder.getUnknownLoc(),
                                             creation_tile, consLockID, 0);
      consLock.getOperation()->setAttr(
          SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name().str() + "_cons_lock"));
      locks.push_back(consLock);
    }
    return locks;
  }

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, LockAnalysis &lockAnalysis,
                                PacketSwitchedObjectFifoOp op, int share_direction) {
    if (!op.size())
      return;

    std::vector<BufferOp> buffers;
    auto fifo = op.getElemType().cast<AIEObjectFifoType>();
    auto elemType = fifo.getElementType().cast<MemRefType>();
    int numElem = op.size();
    int of_elem_index = 0; // used to give objectFifo elements a symbolic name

    TileOp creation_tile;
    if (share_direction == 0 || share_direction == -1)
      creation_tile = op.getProducerTileOp();
    else {
      auto consumerTileOp =
          dyn_cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
      creation_tile = consumerTileOp;
    }

    // Reset opbuilder location to after the last tile declaration
    Operation *t = nullptr;
    auto dev = op->getParentOfType<DeviceOp>();
    for (auto tile_op : dev.getBody()->getOps<TileOp>()) {
      t = tile_op.getOperation();
    }
    builder.setInsertionPointAfter(t);
    for (int i = 0; i < numElem; i++) {
      // if shimTile external buffers are collected from input code
      // create as many locks as there are external buffers
      if (!creation_tile.isShimTile()) {
        auto buff = builder.create<BufferOp>(
            builder.getUnknownLoc(), elemType, creation_tile,
            builder.getStringAttr(op.name().str() + "_buff_" +
                                  std::to_string(of_elem_index)),
            /*address*/ nullptr, /*initial_value*/ nullptr,
            /*mem_bank*/ nullptr);
        buffers.push_back(buff);
      }
      of_elem_index++;
    }
    std::vector<LockOp> locks = createObjectFifoLocks(builder, lockAnalysis, op,
                                                      numElem, creation_tile);
    buffersPerFifo[op] = buffers;
    locksPerFifo[op] = locks;
  }

  /// Function that returns a pointer to the block of a Region
  /// that contains the AIEEndOp.
  Block *findEndOpBlock(Region &r) {
    Block *endBlock = nullptr;
    for (auto &bl : r.getBlocks())
      if (!bl.getOps<EndOp>().empty())
        endBlock = &bl;
    return endBlock;
  }

  /// Function used to create a Bd block.
  template <typename MyOp>
  void createBd(OpBuilder &builder, LockOp acqLock, int acqMode,
                LockAction acqLockAction, LockOp relLock, int relMode,
                MyOp buff, int offset, int len, Block *succ,
                std::optional<DMABDPacket> bdPacket) {
    builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock, acqLockAction,
                              acqMode);
    if (bdPacket) {
      builder.create<DMABDPACKETOp>(builder.getUnknownLoc(), bdPacket.value().packet_type,
                                    bdPacket.value().packet_id);
    }
    builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len);
    builder.create<UseLockOp>(builder.getUnknownLoc(), relLock,
                              LockAction::Release, relMode);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  template <typename MyOp>
  void createBdBlock(OpBuilder &builder, PacketSwitchedObjectFifoOp op, int lockMode,
                     int acqNum, int relNum, MyOp buff, int offset, int len,
                     DMAChannelDir channelDir, size_t blockIndex, Block *succ,
                     std::optional<DMABDPacket> bdPacket) {
    LockOp acqLock;
    LockOp relLock;
    int acqMode = 1;
    int relMode = 1;
    auto acqLockAction = LockAction::Acquire;
    auto dev = op->getParentOfType<DeviceOp>();
    if (auto &target = dev.getTargetModel();
        target.getTargetArch() == AIEArch::AIE1) {
      acqMode = lockMode == 0 ? 1 : 0;
      relMode = lockMode == 0 ? 0 : 1;
      acqLock = locksPerFifo[op][blockIndex];
      relLock = locksPerFifo[op][blockIndex];
    } else {
      acqMode = acqNum;
      relMode = relNum;
      acqLockAction = LockAction::AcquireGreaterEqual;
      acqLock = channelDir == DMAChannelDir::S2MM ? locksPerFifo[op][0]
                                                  : locksPerFifo[op][1];
      relLock = channelDir == DMAChannelDir::S2MM ? locksPerFifo[op][1]
                                                  : locksPerFifo[op][0];
    }
    createBd(builder, acqLock, acqMode, acqLockAction, relLock, relMode,
             buff, offset, len, succ, bdPacket);
  }

  /// Function that either calls createAIETileDMA(), createShimDMA() or
  /// createMemTileDMA() based on op tile row value.
  void createDMA(DeviceOp &device, OpBuilder &builder, PacketSwitchedObjectFifoOp op,
                 DMAChannelDir channelDir, int channelIndex, int lockMode,
                 std::optional<DMABDPacket> bdPacket) {
    if (op.getProducerTileOp().isShimTile()) {
      createShimDMA(device, builder, op, channelDir, channelIndex, lockMode,
                    bdPacket);
    } else if (op.getProducerTileOp().isMemTile()) {
      createMemTileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       bdPacket);
    } else {
      createAIETileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       bdPacket);
    }
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createAIETileDMA(DeviceOp &device, OpBuilder &builder,
                        PacketSwitchedObjectFifoOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        std::optional<DMABDPacket> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;
    int offset = 0;

    auto fifo = op.getElemType().cast<AIEObjectFifoType>();
    auto elemType = fifo.getElementType().cast<MemRefType>();
    int len = elemType.getNumElements();

    PacketSwitchedObjectFifoOp target = op;

    // search for MemOp
    Operation *producerMem = nullptr;
    for (auto memOp : device.getOps<MemOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = memOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerMem == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      auto newMemOp =
          builder.create<MemOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerMem = newMemOp.getOperation();
    }
    Block *endBlock = findEndOpBlock(producerMem->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCount*/ 1, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t blockIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (blockIndex >= buffersPerFifo[target].size())
        break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      builder.setInsertionPointToStart(curr);
      createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                              buffersPerFifo[target][blockIndex], offset, len,
                              channelDir, blockIndex, succ, bdPacket);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a ShimDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createShimDMA(DeviceOp &device, OpBuilder &builder,
                     PacketSwitchedObjectFifoOp op, DMAChannelDir channelDir,
                     int channelIndex, int lockMode,
                     std::optional<DMABDPacket> bdPacket) {
    size_t numBlocks = externalBuffersPerFifo[op].size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;
    int offset = 0;

    // search for ShimDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<ShimDMAOp>()) {
      if (dmaOp.getTile() == op.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = op.getProducerTileOp();
    if (producerDMA == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      auto newDMAOp = builder.create<ShimDMAOp>(
          builder.getUnknownLoc(), builder.getIndexType(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCout*/ 1, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t blockIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (blockIndex >= externalBuffersPerFifo[op].size())
        break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      MemRefType buffer = externalBuffersPerFifo[op][blockIndex].getType();
      int len = buffer.getNumElements();
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(builder, op, lockMode, acqNum, relNum,
                                      externalBuffersPerFifo[op][blockIndex],
                                      offset, len, channelDir, blockIndex, succ,
                                      bdPacket);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a MemTileDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createMemTileDMA(DeviceOp &device, OpBuilder &builder,
                        PacketSwitchedObjectFifoOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        std::optional<DMABDPacket> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int offset = 0;
    auto fifo = op.getElemType().cast<AIEObjectFifoType>();
    auto elemType = fifo.getElementType().cast<MemRefType>();
    int lenOut = elemType.getNumElements();
    int acqNum = 1;
    int relNum = 1;

    PacketSwitchedObjectFifoOp target = op;

    // search for MemTileDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<MemTileDMAOp>()) {
      if (dmaOp.getTile() == target.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerDMA == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      auto newDMAOp =
          builder.create<MemTileDMAOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCount*/ 1, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t blockIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (blockIndex >= buffersPerFifo[target].size())
        break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      builder.setInsertionPointToStart(curr);
      createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                              buffersPerFifo[target][blockIndex], offset,
                              lenOut, channelDir, blockIndex, succ,
                              bdPacket);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a UseLockOp based on input parameters.
  /// acc is an accumulator map that tracks the indices of the next locks to
  /// acquire (or release). Uses op to find index of acc for next lockID.
  /// Updates acc.
  void createUseLocks(OpBuilder &builder, PacketSwitchedObjectFifoOp op,
                      ObjectFifoPort port,
                      DenseMap<std::pair<PacketSwitchedObjectFifoOp, int>, int> &acc,
                      int numLocks, LockAction lockAction) {
    PacketSwitchedObjectFifoOp target = op;
    auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;

    auto dev = op->getParentOfType<DeviceOp>();
    if (auto &targetArch = dev.getTargetModel();
        targetArch.getTargetArch() == AIEArch::AIE1) {
      int lockMode = 0;
      if ((port == ObjectFifoPort::Produce &&
           lockAction == LockAction::Release) ||
          (port == ObjectFifoPort::Consume &&
           lockAction == LockAction::Acquire))
        lockMode = 1;
      for (int i = 0; i < numLocks; i++) {
        int lockID = acc[{op, portNum}];
        builder.create<UseLockOp>(builder.getUnknownLoc(),
                                  locksPerFifo[target][lockID], lockAction,
                                  lockMode);
        acc[{op, portNum}] =
            (lockID + 1) % op.size(); // update to next objFifo elem
      }
    } else {
      if (numLocks == 0)
        return;
      // search for the correct lock based on the port of the acq/rel
      // operation e.g. acq as consumer is the read lock (second)
      LockOp lock;
      if (lockAction == LockAction::AcquireGreaterEqual) {
        if (port == ObjectFifoPort::Produce)
          lock = locksPerFifo[target][0];
        else
          lock = locksPerFifo[target][1];
      } else {
        if (port == ObjectFifoPort::Produce)
          lock = locksPerFifo[target][1];
        else
          lock = locksPerFifo[target][0];
      }
      builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockAction,
                                numLocks);
      acc[{op, portNum}] = (acc[{op, portNum}] + numLocks) %
                           op.size(); // update to next objFifo elem
    }
  }

  /// Function used to check whether op is already contained in map.
  /// If it is then return the associated int, if not create new entry and
  /// return 0.
  int updateAndReturnIndex(
      DenseMap<std::pair<PacketSwitchedObjectFifoOp, int>, int> &map,
      std::pair<PacketSwitchedObjectFifoOp, int> pair) {
    if (map.find(pair) == map.end()) {
      map[pair] = 0;
      return 0;
    }
    return map[pair];
  }

  /// Function used to add an external buffer to the externalBuffersPerFifo map.
  void addExternalBuffer(PacketSwitchedObjectFifoOp fifo, ExternalBufferOp buff) {
    if (externalBuffersPerFifo.find(fifo) == externalBuffersPerFifo.end()) {
      std::vector<ExternalBufferOp> buffs;
      externalBuffersPerFifo[fifo] = buffs;
    }
    externalBuffersPerFifo[fifo].push_back(buff);
  }

  /// Function used to detect all external buffers associated with parent
  /// objectFifo and tile then map them to child objectFifo.
  void detectExternalBuffers(DeviceOp &device, PacketSwitchedObjectFifoOp parent,
                             PacketSwitchedObjectFifoOp child, Value tile) {
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
      if (auto objFifo = regOp.getObjectFifo();
          regOp.getTile() == tile && objFifo == parent)
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>());
  }

  /// Function used to replace uses of split objectFifos.
  void replaceSplitFifo(PacketSwitchedObjectFifoOp originalOp, PacketSwitchedObjectFifoOp newOp,
                        TileOp tile) {
    auto original =
        originalOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newSymbol =
        newOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    for (auto user : tile->getUsers())
      if (isa<CoreOp>(user))
        if (auto res =
                SymbolTable::replaceAllSymbolUses(original, newSymbol, user);
            res.failed())
          llvm_unreachable("unreachable");
  }

  /// Function used to find the size of an objectFifo after split based on
  /// the maximum number of elements (of the original objectFifo) acquired
  /// by a process running on given tile. If no CoreOp exists for this tile
  /// return 0.
  int findObjectFifoSize(DeviceOp &device, Value tile,
                         PacketSwitchedObjectFifoOp objFifo) {
    if (objFifo.size() == 0)
      return 0;

    // if memTile, size is equal to objFifo size
    if (tile.getDefiningOp<TileOp>().isMemTile())
      return objFifo.size();

    // if shimTile, size is equal to number of external buffers
    if (tile.getDefiningOp<TileOp>().isShimTile())
      for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
        if (regOp.getTile() == tile)
          return regOp.getExternalBuffers().size();
      }

    return objFifo.size();
  }

  /// Function used to generate, from an objectFifo with a shimTile endpoint, a
  /// shimDMAAllocationOp containing the channelDir, channelIndex and
  /// shimTile col assigned by the objectFifo lowering.
  void createObjectFifoAllocationInfo(OpBuilder &builder, MLIRContext *ctx,
                                      FlatSymbolRefAttr obj_fifo, int colIndex,
                                      DMAChannelDir channelDir,
                                      int channelIndex) {
    builder.create<ShimDMAAllocationOp>(builder.getUnknownLoc(), obj_fifo,
                                        DMAChannelDirAttr::get(ctx, channelDir),
                                        builder.getI64IntegerAttr(channelIndex),
                                        builder.getI64IntegerAttr(colIndex));
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
    auto ctx = device->getContext();
    std::set<TileOp>
        objectFifoTiles; // track cores to check for loops during unrolling

    //===------------------------------------------------------------------===//
    // Split objectFifos into a consumer end and producer end if needed
    //===------------------------------------------------------------------===//
    // We are going to create additional createObjectFifoOps, so get a copy of
    // all "original" ones before the loop to avoid looping over newly created
    // ones.
    std::vector<PacketSwitchedObjectFifoOp> createFifoOps;
    auto range = device.getOps<PacketSwitchedObjectFifoOp>();
    createFifoOps.insert(createFifoOps.end(), range.begin(), range.end());
    for (auto createOp : createFifoOps) {
      std::vector<PacketSwitchedObjectFifoOp> splitConsumerFifos;
      int consumerIndex = 0;
      int consumerDepth = createOp.size();

      // Only FIFOs using DMA are split into two ends;
      // skip in shared memory case
      if (int share_direction = 0; !requiresDMAs(createOp, share_direction))
        continue;

      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());

        if (isa<ArrayAttr>(createOp.getElemNumber())) {
          // +1 to account for 1st depth (producer)
          consumerDepth = createOp.size(consumerIndex + 1);
        } else {
          consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);
        }

        builder.setInsertionPointAfter(createOp);
        auto datatype = createOp.getElemType().cast<AIEObjectFifoType>();
        auto consumerObjFifoSize =
            builder.getIntegerAttr(builder.getI32Type(), consumerDepth);
        // rename and replace split objectFifo
        std::string consumerFifoName;
        if (createOp.getConsumerTiles().size() > 1) {
          consumerFifoName = createOp.name().str() + "_" +
                             std::to_string(consumerIndex) + "_cons";
        } else {
          consumerFifoName = createOp.name().str() + "_cons";
        }

        PacketSwitchedObjectFifoOp consumerFifo = createObjectFifo(
            builder, datatype, consumerFifoName, consumerTile, consumerTile,
            consumerObjFifoSize);
        replaceSplitFifo(createOp, consumerFifo, consumerTileOp);

        // identify external buffers that were registered to the consumer fifo
        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile);

        // record that this objectFifo was split; it will require DMA config
        splitConsumerFifos.push_back(consumerFifo);

        consumerIndex++;
      }

      if (!splitConsumerFifos.empty()) {
        splitFifos.emplace_back(createOp, splitConsumerFifos);
      }
    }

    //===------------------------------------------------------------------===//
    // - Create objectFifo buffers and locks.
    // - Populate a list of tiles containing objectFifos for later processing of
    //   the acquires/releases (uses of the FIFO).
    //===------------------------------------------------------------------===//
    for (auto createOp : device.getOps<PacketSwitchedObjectFifoOp>()) {
      int share_direction = 0;
      bool shared = !requiresDMAs(createOp, share_direction);

      // add all tiles that contain an objectFifo to objectFifoTiles for later
      // loop unrolling pass
      objectFifoTiles.insert(createOp.getProducerTileOp());
      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());
        objectFifoTiles.insert(consumerTileOp);
      }

      // identify external buffers that were registered to
      // the producer objectFifo
      if (createOp.getProducerTileOp().isShimTile())
        detectExternalBuffers(device, createOp, createOp,
                              createOp.getProducerTile());

      // if split, the necessary size for producer fifo might change
      if (shared)
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      else {
        if (isa<ArrayAttr>(createOp.getElemNumber()))
          createOp.setElemNumberAttr(
              builder.getI32IntegerAttr(createOp.size()));
        else {
          int prodMaxAcquire = findObjectFifoSize(
              device, createOp.getProducerTileOp(), createOp);
          createOp.setElemNumberAttr(builder.getI32IntegerAttr(prodMaxAcquire));
        }
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      }
    }

    //===------------------------------------------------------------------===//
    // Create flows and tile DMAs
    //===------------------------------------------------------------------===//
    // Only the objectFifos we split above require DMA communication; the others
    // rely on shared memory and share the same buffers.
    for (auto &[producer, consumers] : splitFifos) {
      // create producer tile DMA
      DMAChannel producerChan =
          dmaAnalysis.getMasterDMAChannel(producer.getProducerTile());
      DMABDPacket bdPacket = dmaAnalysis.getDMABDPacket();
      createDMA(device, builder, producer, producerChan.direction,
                producerChan.channel, 0, {bdPacket});
      // generate objectFifo allocation info
      builder.setInsertionPoint(&device.getBody()->back());
      if (producer.getProducerTileOp().isShimTile())
        createObjectFifoAllocationInfo(
            builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
            producer.getProducerTileOp().colIndex(), producerChan.direction,
            producerChan.channel);

      // create packet flow
      builder.setInsertionPointAfter(producer);
      auto packetflow = builder.create<PacketFlowOp>(builder.getUnknownLoc(), builder.getIntegerAttr(builder.getI8Type(), bdPacket.packet_id), nullptr);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&packetflow.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }

      for (auto consumer : consumers) {
        DMAChannel consumerChan =
            dmaAnalysis.getSlaveDMAChannel(consumer.getProducerTile());
        
        builder.setInsertionPointToStart(&packetflow.getPorts().front());
        builder.create<PacketDestOp>(builder.getUnknownLoc(),
                                     consumer.getProducerTile(),
                                     WireBundle::DMA, consumerChan.channel);

        // create consumer tile DMA
        createDMA(device, builder, consumer, consumerChan.direction,
                  consumerChan.channel, 1, {});
        // generate objectFifo allocation info
        builder.setInsertionPoint(&device.getBody()->back());
        if (consumer.getProducerTileOp().isShimTile())
          createObjectFifoAllocationInfo(
              builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
              consumer.getProducerTileOp().colIndex(), consumerChan.direction,
              consumerChan.channel);
      }

      builder.setInsertionPointToStart(&packetflow.getPorts().front());
      builder.create<PacketSourceOp>(builder.getUnknownLoc(),
                                     producer.getProducerTile(),
                                     WireBundle::DMA, producerChan.channel);
    }

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<PacketSwitchedObjectFifoOp,
              ObjectFifoRegisterExternalBuffersOp>(op))
        opsToErase.insert(op);
    });
    topologicalSort(opsToErase);
    IRRewriter rewriter(&getContext());
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it)
      (*it)->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIELowerPacketSwitchedObjectFifosPass() {
  return std::make_unique<AIELowerPacketSwitchedObjectFifosPass>();
}
