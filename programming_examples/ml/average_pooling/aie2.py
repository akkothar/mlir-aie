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

in_channels = 256
out_channels = in_channels

in_width = 7
in_height = 7
kernel_size = 7
out_width = in_width // kernel_size
out_height = in_height // kernel_size


if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])


def avg_pooling():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            # actIn_ty = T.memref(actIn, T.i8())
            # bufIn_ty = T.memref(bufIn, T.i8())

            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)

            in_ty = MemRefType.get((in_width, in_height, in_channels, ), uint8_ty, )
            out_ty = MemRefType.get((out_width, out_height, out_channels ), uint8_ty, )

            # AIE Core Function declarations
            average_pooling = external_func(
                "average_pooling",
                inputs=[
                    in_ty,
                    out_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)


            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile, MemTile, 1, in_ty )
            of_act_L2_02 = object_fifo("act_L2_02", MemTile, ComputeTile2, 1, in_ty)
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

           
            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 1, out_ty)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 1, out_ty)
            object_fifo_link(of_out_02_L2, of_outOFL2L3)

            # Set up compute tiles


            @core(ComputeTile2, "avg_pool.o")
            def core_body():
                for _ in for_(sys.maxsize):
                    elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                    elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)

                    res = call(
                        average_pooling,
                        [
                            elemIn,
                            elemOut0,
                            in_width,
                            in_height,
                            in_channels,
                            kernel_size,
                        ],
                    )

                    objectfifo_release(ObjectFifoPort.Consume, "act_L2_02", 1)
                    objectfifo_release(ObjectFifoPort.Produce, "out_02_L2", 1)

                    yield_([])

            # To/from AIE-array data movement

            activationsInSize32b = (in_width * in_height * in_channels) // 4
            activationsOutSize32b = (out_width * out_height * out_channels) // 4
        
            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)


            @FuncOp.from_py_func(activationsInL3_ty, activationsOutL3_ty)
            def sequence(I, O):

                npu_dma_memcpy_nd(
                    metadata="inOF_act_L3L2",
                    bd_id=0,
                    mem=I,
                    sizes=[1, 1, 1, activationsInSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="outOFL2L3",
                    bd_id=1,
                    mem=O,
                    sizes=[1, 1, 1, activationsOutSize32b],
                )
                npu_sync(column=0, row=0, direction=0, channel=0)


    print(ctx.module)


avg_pooling()
