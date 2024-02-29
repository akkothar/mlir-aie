//===- base_test_AIE2.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Xilinx Inc.
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-packet-objectFifo %s | FileCheck %s

// CHECK: 

module @packet_switch {
   aie.device(xcvc1902) {
      %tile10 = aie.tile(1, 3)
      %tile13 = aie.tile(3, 3)

      aiex.packet_objectfifo @of0 (%tile10, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      aiex.packet_objectfifo @of1 (%tile10, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      aiex.packet_objectfifo @of2 (%tile10, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
