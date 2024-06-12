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
import math




if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])



InC = 960
InW2 = 1
InH2 = 1
OutC = 16
WeightChunks=2
RepeatChannels=math.floor(InW2)
# WeightIndex=0
# WeightSplitPerCore=WeightSplit//2


def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)
            uint32_ty = IntegerType.get_unsigned(32)
            indx_ty=IndexType.get()

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
     
            # AIE Core Function declarations
            conv2dk1_i8_ui8_partial = external_func(
                "conv2dk1_i8_ui8_partial",
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

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)


            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, ty_in
            )
            of_act_L2_02 = object_fifo("act_L2_02", MemTile, ComputeTile2, 2, ty_in)
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, MemTile, 1, ty_all_wts
            )
            of_inOF_wts_L2_02 = object_fifo(
                "inOF_wts_L2_02", MemTile, [ComputeTile2], [1,1], ty_wts
            )
            object_fifo_link(of_inOF_wts_0_L3L2, of_inOF_wts_L2_02)
            of_inOF_wts_L2_02.set_memtile_repeat(RepeatChannels)
            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 2, ty_out)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, ty_out)
            object_fifo_link(of_out_02_L2, of_outOFL2L3)
            

            # Set up compute tiles
            rtp2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")

            # Compute tile 2
            @core(ComputeTile2, "conv2dk1.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(InH2):
                        elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)

                        
                        scale = memref.load(rtp2, [0])
                        
                        for oc in range(0,InW2):
                            for WeightIndex in range (0,WeightChunks):
                                elemWts = of_inOF_wts_L2_02.acquire(ObjectFifoPort.Consume, 1)
                                call(
                                    conv2dk1_i8_ui8_partial,
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
                                objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_L2_02", 1)
                    # second iteration
                                # elemWts = of_inOF_wts_L2_02.acquire(ObjectFifoPort.Consume, 1)
                                
                #             call(
                #                 conv2dk1_i8_ui8_partial,
                #                 [
                #                     elemIn,
                #                     elemWts,
                #                     elemOut0,
                #                     arith.constant(InW2),
                #                     arith.constant(InC),
                #                     arith.constant(OutC),
                #                     scale,
                #                     WeightChunks,
                #                     1,
                #                     oc
                #                 ],
                #             )
                    
                #             objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_L2_02", 1)

                # # third iteration
                #             elemWts = of_inOF_wts_L2_02.acquire(ObjectFifoPort.Consume, 1)
                #             call(
                #                 conv2dk1_i8_ui8_partial,
                #                 [
                #                     elemIn,
                #                     elemWts,
                #                     elemOut0,
                #                     arith.constant(InW2),
                #                     arith.constant(InC),
                #                     arith.constant(OutC),
                #                     scale,
                #                     WeightChunks,
                #                     2,
                #                     oc
                #                 ],
                #             )
                    
                #             objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_L2_02", 1)


                # # fourth iteration
                #             elemWts = of_inOF_wts_L2_02.acquire(ObjectFifoPort.Consume, 1)
                            
                #             call(
                #                 conv2dk1_i8_ui8_partial,
                #                 [
                #                     elemIn,
                #                     elemWts,
                #                     elemOut0,
                #                     arith.constant(InW2),
                #                     arith.constant(InC),
                #                     arith.constant(OutC),
                #                     scale,
                #                     WeightChunks,
                #                     3,
                #                     oc
                #                 ],
                #             )
                    
                #             objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_L2_02", 1)
                       
                        # for _ in for_(WeightChunks):
                        #     elemWts = of_inOF_wts_L2_02.acquire(ObjectFifoPort.Consume, 1)
                        #     scale = memref.load(rtp2, [0])
                        #     for oc in range(0,OutC//8):
                        #         call(
                        #             conv2dk1_i8_ui8_partial,
                        #             [
                        #                 elemIn,
                        #                 elemWts,
                        #                 elemOut0,
                        #                 arith.constant(InW2),
                        #                 arith.constant(InC),
                        #                 arith.constant(OutC),
                        #                 scale,
                        #                 WeightChunks,
                        #                 WeightIndex,
                        #                 oc
                        #             ],
                        #         )
                        #     WeightIndex+=1
                        #     objectfifo_release(ObjectFifoPort.Consume, "inOF_wts_L2_02", 1)
                        #     yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "act_L2_02", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "out_02_L2", 1)
                        
                        yield_([])
                    yield_([])

            # To/from AIE-array data movement
            activationsInSize32b = (InW2 * InH2 * InC) // 4

            acitivationsOutSize32b = (InW2 * InH2 * OutC) // 4
            
            totalWeightsSize32b = (
               InC*OutC
            ) // 4


            totalWeightsSize32b_complete = (
                totalWeightsSize32b
            )



            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)
            # memRef_16x16_ty = T.memref(16, 16, T.i32())

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                # NpuWriteRTPOp("rtp02", col=0, row=2, index=0, value=9)
                # NpuWriteRTPOp("rtp03", col=0, row=3, index=0, value=8)
                NpuWriteRTPOp("rtp2", col=0, row=2, index=0, value=10)

                
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
                    metadata="inOF_wts_0_L3L2",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, totalWeightsSize32b],
                )

                npu_sync(column=0, row=0, direction=0, channel=0)

    #    print(ctx.module.operation.verify())
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


conv2dk1()
