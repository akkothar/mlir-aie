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

// CHECK: 

module @packet_switch {
   aie.device(xcvc1902) {
      %tile13 = aie.tile(1, 3)
      %tile33 = aie.tile(3, 3)

      aie.packet_sw_objectfifo @of0 (%tile13, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.packet_sw_objectfifo @of1 (%tile13, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      //aie.packet_sw_objectfifo @of2 (%tile13, {%tile33}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
