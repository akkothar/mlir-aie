# memtile_repeat/distribute_repeat/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

N = 4096
dev = AIEDevice.npu1_1col
col = 0

if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def distribute_repeat():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            memRef_ty = T.memref(1024, T.i32())
            memRef_512_ty = T.memref(512, T.i32())
            memRef_3072_ty = T.memref(3072, T.i32())

            # Tile declarations
            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)
            ComputeTile2 = tile(col, 2)
            ComputeTile3 = tile(col, 3)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, MemTile, 1, memRef_ty)
            of_in2 = object_fifo("in2", MemTile, ComputeTile2, [1, 1], memRef_512_ty)
            of_in3 = object_fifo("in3", MemTile, ComputeTile3, [1, 1], memRef_512_ty)
            object_fifo_link(of_in, [of_in2, of_in3])

            of_out2 = object_fifo("out2", ComputeTile2, MemTile, [1, 1], memRef_512_ty)
            of_out3 = object_fifo("out3", ComputeTile3, MemTile, [1, 1], memRef_512_ty)
            of_out = object_fifo("out", MemTile, ShimTile, 1, memRef_ty)
            object_fifo_link([of_out2, of_out3], of_out)

            # Set up compute tiles

            # Compute tile 2
            @core(ComputeTile2)
            def core_body():
                for _ in for_(sys.maxsize):
                    elemOut = of_out2.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in2.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(512):
                        v0 = memref.load(elemIn, [i])
                        v1 = arith.addi(v0, arith.constant(1, T.i32()))
                        memref.store(v1, elemOut, [i])
                        yield_([])
                    of_in2.release(ObjectFifoPort.Consume, 1)
                    of_out2.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # Compute tile 3
            @core(ComputeTile3)
            def core_body():
                for _ in for_(sys.maxsize):
                    elemOut = of_out3.acquire(ObjectFifoPort.Produce, 1)
                    elemIn = of_in3.acquire(ObjectFifoPort.Consume, 1)
                    for i in for_(512):
                        v0 = memref.load(elemIn, [i])
                        v1 = arith.addi(v0, arith.constant(2, T.i32()))
                        memref.store(v1, elemOut, [i])
                        yield_([])
                    of_in3.release(ObjectFifoPort.Consume, 1)
                    of_out3.release(ObjectFifoPort.Produce, 1)
                    yield_([])

            # To/from AIE-array data movement
            tensor_ty = T.memref(N, T.i32())
            tensor_in_ty = T.memref(N // 4, T.i32())

            @FuncOp.from_py_func(tensor_in_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
                npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N // 4])
                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


distribute_repeat()