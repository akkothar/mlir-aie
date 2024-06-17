#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

InW1 = 1
InH1 = 1
InC = 16

InW2 = 1
InH2 = 1
OutC = 960
WeightChunks=2

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

def mobilenetBottleneckB():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            # define types
            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)
            uint32_ty = IntegerType.get_unsigned(32)

        # ************************ bneck13 ************************
  
            ty_in = MemRefType.get(
                (
                    InW2,
                    1,
                    InC,
                ),
                int8_ty,
            )
         
            # define wts
            ty_wts = MemRefType.get(
                (InC//WeightChunks * OutC,), int8_ty
            )
            ty_all_wts= MemRefType.get(
                (
                    InC * OutC,
                ),
                int8_ty,
            )

            ty_out = MemRefType.get(
                (
                    InW2,
                    1,
                    OutC,
                ),
                uint8_ty,
            )
            
# HERE
            
   
            conv2dk1_get = external_func(
                "conv2dk1_i8_ui8_partial_get",
                inputs=[
                    ty_in,
                    ty_wts,
                    ty_out,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )
            conv2dk1_put = external_func(
                "conv2dk1_i8_partial_put",
                inputs=[
                    ty_in,
                    ty_wts,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)

            MemTile01 = tile(0, 1)

            ComputeTile05 = tile(0, 5)
            ComputeTile04 = tile(0, 4)
            # ComputeTile15 = tile(1, 5)

            cascade_flow(ComputeTile05, ComputeTile04)
            # AIE-array data movement with object fifos
            # ************************ bneck13 ************************
            # Input
            inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2",
                ShimTile00,
                [ComputeTile05,ComputeTile04],
                [2, 2, 2],
                ty_in,
            )
        
            # # wts
            OF_wts_L3L2 = object_fifo(
                "OF_wts_L3L2", ShimTile00, MemTile01, 1, ty_all_wts
            )
            OF_wts_memtile_put = object_fifo(
                "OF_wts_memtile_put", MemTile01, ComputeTile05, 1, ty_wts
            )
            OF_wts_memtile_get = object_fifo(
                "OF_wts_memtile_get",
                MemTile01,
                ComputeTile04,
                1,
                ty_wts,
            )
           
            object_fifo_link(OF_wts_L3L2, [OF_wts_memtile_put,OF_wts_memtile_get],[],[0,InC//2 * OutC])

        
            # Set up compute tiles

            # rtp02 = Buffer(ComputeTile02, [16], T.i32(), "rtp02")
            # rtp03 = Buffer(ComputeTile03, [16], T.i32(), "rtp03")
            rtp04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")


            out_04_L2 = object_fifo("out_04_L2", ComputeTile04, [MemTile01], 2, ty_out)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile01, [ShimTile00], 2, ty_out)
            object_fifo_link(out_04_L2, OF_outOFL2L3)


            # Compute tile 4
            @core(ComputeTile05, "conv2dk1_put.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(InH2):
                        elemIn = inOF_act_L3L2.acquire(ObjectFifoPort.Consume, 1)
                        for oc in range(0,InW2):
                            for WeightIndex in range (0,WeightChunks//2):
                                elemWts = OF_wts_memtile_put.acquire(ObjectFifoPort.Consume, 1)
                                call(
                                    conv2dk1_put,
                                    [
                                        elemIn,
                                        elemWts,

                                        arith.constant(InW2),
                                        arith.constant(InC),
                                        arith.constant(OutC),

                                        WeightChunks,
                                        WeightIndex,
                                        oc
                                    ],
                                )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_wts_memtile_put", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)
                        
                        yield_([])
                    yield_([])

            # Compute tile 4
            @core(ComputeTile04, "conv2dk1_get.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(InH2):
                        elemIn = inOF_act_L3L2.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = out_04_L2.acquire(ObjectFifoPort.Produce, 1)

                        
                        scale = memref.load(rtp04, [0])
                        
                        for oc in range(0,InW2):
                            for WeightIndex in range (WeightChunks//2,WeightChunks ):
                                elemWts = OF_wts_memtile_get.acquire(ObjectFifoPort.Consume, 1)
                                call(
                                    conv2dk1_get,
                                    [
                                        elemIn,
                                        elemWts,
                                        elemOut0,
                                        arith.constant(InW2),
                                        arith.constant(InC),
                                        arith.constant(OutC),
                                        scale,
                                        WeightChunks,
                                        WeightIndex,
                                        oc
                                    ],
                                )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_wts_memtile_get", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "out_04_L2", 1)
                        
                        yield_([])
                    yield_([])

            # # instruction stream generation
            activationsInSize32b = (InW1 * InH1 * InC) // 4

            acitivationsOutSize32b = (InW1 * InH1 * OutC) // 4

        
            totalWeightsSize32b = (
               InC*OutC
            ) // 4


            totalWeightsSize32b_complete = (
                totalWeightsSize32b
            )

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                # NpuWriteRTPOp("rtp02", col=0, row=2, index=0, value=9)
                # NpuWriteRTPOp("rtp03", col=0, row=3, index=0, value=8)
                NpuWriteRTPOp("rtp04", col=0, row=4, index=0, value=8)

                
                npu_dma_memcpy_nd(
                    metadata="inOF_act_L3L2",
                    bd_id=0,
                    mem=inputFromL3,
                    sizes=[1, 1, 1, activationsInSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="outOFL2L3",
                    bd_id=2,
                    mem=outputToL3,
                    sizes=[1, 1, 1, acitivationsOutSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="OF_wts_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeightsSize32b],
                )

                npu_sync(column=0, row=0, direction=0, channel=0)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


mobilenetBottleneckB()