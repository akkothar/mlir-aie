//===- base_objectfifo.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// CHECK: error{{.*}}'aie.mem' op uses more output channels than available on this tile

module @packet_switch {
   aie.device(xcve2302) {
      %tile10 = aie.tile(1, 0)
      %tile13 = aie.tile(1, 3)

      aie.objectfifo @of0 (%tile10, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.objectfifo @of1 (%tile10, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
      aie.objectfifo @of2 (%tile10, {%tile13}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
