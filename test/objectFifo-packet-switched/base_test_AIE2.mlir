//===- base_test_AIE2.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-packet-sw-objectFifos %s | FileCheck %s

// CHECK: module @packet_switch {
// CHECK:   aie.device(xcvc1902) {
// CHECK:     %tile_1_3 = aie.tile(1, 3)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %of1_cons_buff_0 = aie.buffer(%tile_3_3) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of1_cons_lock_0 = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     %of1_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %of1_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:     %of0_cons_buff_0 = aie.buffer(%tile_3_3) {sym_name = "of0_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of0_cons_lock_0 = aie.lock(%tile_3_3, 0) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:     %of0_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of0_buff_0"} : memref<16xi32> 
// CHECK:     %of0_lock_0 = aie.lock(%tile_1_3, 0) {init = 0 : i32, sym_name = "of0_lock_0"}
// CHECK:     aie.packet_flow(0) {
// CHECK:       aie.packet_source<%tile_1_3, DMA : 0>
// CHECK:       aie.packet_dest<%tile_3_3, DMA : 0>
// CHECK:     }
// CHECK:     aie.packet_flow(1) {
// CHECK:       aie.packet_source<%tile_1_3, DMA : 1>
// CHECK:       aie.packet_dest<%tile_3_3, DMA : 1>
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%tile_1_3) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of0_lock_0, Acquire, 1)
// CHECK:       aie.dma_bd_packet(0, 0)
// CHECK:       aie.dma_bd(%of0_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of0_lock_0, Release, 0)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 1, ^bb3, ^bb4, repeat_count = 1)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       aie.use_lock(%of1_lock_0, Acquire, 1)
// CHECK:       aie.dma_bd_packet(1, 1)
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_lock_0, Release, 0)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%tile_3_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 1)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%of0_cons_lock_0, Acquire, 0)
// CHECK:       aie.dma_bd(%of0_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of0_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4, repeat_count = 1)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       aie.use_lock(%of1_cons_lock_0, Acquire, 0)
// CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @packet_switch {
   aie.device(xcvc1902) {
      %tile13 = aie.tile(1, 3)
      %tile33 = aie.tile(3, 3)

      aie.packet_sw_objectfifo @of0 (%tile13, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.packet_sw_objectfifo @of1 (%tile13, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
