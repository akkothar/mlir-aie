//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xf32>

  %core13 = AIE.core(%tile13) {
    %val1 = arith.constant 7.0 : f32
    %idx1 = arith.constant 3 : index
    %2 = arith.addf %val1, %val1 : f32
    memref.store %2, %buf13_0[%idx1] : memref<256xf32>
    %val2 = arith.constant 8.0 : f32
    %idx2 = arith.constant 5 : index
    memref.store %val2, %buf13_0[%idx2] : memref<256xf32>
    %val3 = memref.load %buf13_0[%idx1] : memref<256xf32>
    %idx3 = arith.constant 9 : index
    memref.store %val3,%buf13_0[%idx3] : memref<256xf32>
    AIE.end
  }
}