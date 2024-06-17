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

bneck_13_InW1 = 8
bneck_13_InH1 = 1
bneck_13_InC1 = 160
bneck_13_OutC1 = 120
InputSplit=2
OutputSplit=bneck_13_OutC1//8

bneck_13_InW2 = bneck_13_InW1
bneck_13_InH2 = bneck_13_InH1
bneck_13_OutC2 = bneck_13_OutC1

bneck_13_InW3 = bneck_13_InW2
bneck_13_InH3 = bneck_13_InH2
bneck_13_OutC3 = 160

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])

enableTrace = False
trace_size = 16384
traceSizeInInt32s = trace_size // 4


def mobilenetBottleneckC():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_2col)
        def device_body():

            # define types
            uint8_ty = IntegerType.get_unsigned(8)
            int8_ty = IntegerType.get_signless(8)
            int32_ty = IntegerType.get_signless(32)
            uint32_ty = IntegerType.get_unsigned(32)

        # ************************ bneck13 ************************
            # input
            ty_bneck_13_layer1_in = MemRefType.get(
                (
                    bneck_13_InW1,
                    1,
                    bneck_13_InC1,
                ),
                int8_ty,
            )
            ty_bneck_13_layer2_in = MemRefType.get(
                (
                    bneck_13_InW2,
                    1,
                    bneck_13_OutC1,
                ),
                uint8_ty,
            )
            ty_bneck_13_layer3_in = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC2,
                ),
                uint8_ty,
            )
         
            # define wts
            ty_bneck_13_layer1_wts_split = MemRefType.get(
                ((bneck_13_InC1 * bneck_13_OutC1)//(InputSplit*OutputSplit),), int8_ty
            )
            ty_bneck_13_layer1_wts_full = MemRefType.get(
                ((bneck_13_InC1 * bneck_13_OutC1),), int8_ty
            )
            ty_bneck_13_layer2_wts = MemRefType.get(
                (3 * 3 * bneck_13_OutC2 * 1,), int8_ty
            )
            ty_bneck_13_layer3_wts = MemRefType.get(
                (bneck_13_OutC2 * bneck_13_OutC3,), int8_ty
            )
            ty_bneck_13_all_wts= MemRefType.get(
                (
                    bneck_13_InC1 * bneck_13_OutC1
                    + 3 * 3 * bneck_13_OutC2 * 1,
                    # + bneck_13_OutC2 * bneck_13_OutC3,
                ),
                int8_ty,
            )

            # output
            ty_bneck_13_layer1_out = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC1,
                ),
                uint8_ty,
            )
            ty_bneck_13_layer2_out = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC2,
                ),
                uint8_ty,
            )
            ty_bneck_13_layer3_out = MemRefType.get(
                (
                    bneck_13_InW3,
                    1,
                    bneck_13_OutC3,
                ),
                int8_ty,
            )
      
            
            # ************************ bneck13 ************************
            bn13_conv2dk1_fused_relu_get = external_func(
                "conv2dk1_i8_ui8_partial_width_get",
                inputs=[
                    ty_bneck_13_layer1_in,
                    ty_bneck_13_layer1_wts_split,
                    ty_bneck_13_layer1_out,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )
            bn13_conv2dk1_fused_relu_put = external_func(
                "conv2dk1_i8_ui8_partial_width_put",
                inputs=[
                    ty_bneck_13_layer1_in,
                    ty_bneck_13_layer1_wts_split,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )


            bn13_conv2dk3_dw = external_func(
                "bn10_conv2dk3_ui8",
                inputs=[
                    ty_bneck_13_layer2_in,
                    ty_bneck_13_layer2_in,
                    ty_bneck_13_layer2_in,
                    ty_bneck_13_layer2_wts,
                    ty_bneck_13_layer2_out,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )
            # bn13_conv2dk1_skip = external_func(
            #     "conv2dk1_skip_ui8_i8_i8",
            #     inputs=[
            #         ty_bneck_13_layer3_in,
            #         ty_bneck_13_layer3_wts,
            #         ty_bneck_13_layer3_out,
            #         ty_bneck_13_layer1_in,
            #         int32_ty,
            #         int32_ty,
            #         int32_ty,
            #         int32_ty,
            #         int32_ty,
            #     ],
            # )

            # Tile declarations
            ShimTile00 = tile(0, 0)
            ShimTile10 = tile(1, 0)

            MemTile01 = tile(0, 1)
            MemTile11 = tile(1, 1)
            

            ComputeTile05 = tile(0, 5)
            ComputeTile04 = tile(0, 4)
            ComputeTile03 = tile(0, 3)
            ComputeTile02 = tile(0, 2)
            
            
            # bn11
            
            # ComputeTile15 = tile(1, 5)

            cascade_flow(ComputeTile05, ComputeTile04)

            # AIE-array data movement with object fifos
            # ************************ bneck13 ************************
            # Input
            inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2",
                ShimTile00,
                [ComputeTile05,ComputeTile04, MemTile01],
                [2, 2, 2, 4],
                ty_bneck_13_layer1_in,
            )
            # OF_bneck_13_skip = object_fifo(
            #     "OF_bneck_13_skip", MemTile01, ComputeTile05, 2, ty_bneck_13_layer1_in
            # )
            # object_fifo_link(inOF_act_L3L2, OF_bneck_13_skip)
            
            
            # # wts
            OF_bneck_13_wts_L3L2_layer1 = object_fifo(
                "OF_bneck_13_wts_L3L2_layer1", ShimTile00, MemTile01, 1, ty_bneck_13_layer1_wts_full
            )
       
            OF_bneck_13_wts_memtile_layer1_put = object_fifo(
                "OF_bneck_13_wts_memtile_layer1_put", MemTile01, ComputeTile05, 1, ty_bneck_13_layer1_wts_split
            )
            OF_bneck_13_wts_memtile_layer1_get = object_fifo(
                "OF_bneck_13_wts_memtile_layer1_get", MemTile01, ComputeTile04, 1, ty_bneck_13_layer1_wts_split
            )

            object_fifo_link(OF_bneck_13_wts_L3L2_layer1, [OF_bneck_13_wts_memtile_layer1_put,OF_bneck_13_wts_memtile_layer1_get],[],\
                             [0,bneck_13_InC1 * bneck_13_OutC1//2])
            # OF_bneck_13_wts_memtile_layer1_put.set_memtile_repeat(1)
            # OF_bneck_13_wts_memtile_layer1_get.set_memtile_repeat(1)

            OF_bneck_13_wts_L3L2_layer2 = object_fifo(
                "OF_bneck_13_wts_L3L2_layer2", ShimTile10, MemTile11, 1, ty_bneck_13_layer2_wts
            )
            OF_bneck_13_wts_memtile_layer2 = object_fifo(
                "OF_bneck_13_wts_memtile_layer2",
                MemTile11,
                ComputeTile03,
                1,
                ty_bneck_13_layer2_wts,
            )
            object_fifo_link(OF_bneck_13_wts_L3L2_layer2, [OF_bneck_13_wts_memtile_layer2],[],[0])
            # OF_bneck_13_wts_memtile_layer3 = object_fifo(
            #     "OF_bneck_13_wts_memtile_layer3",
            #     MemTile01,
            #     ComputeTile05,
            #     1,
            #     ty_bneck_13_layer3_wts,
            # )
            # object_fifo_link(OF_bneck_13_wts_L3L2_layer1, [OF_bneck_13_wts_memtile_layer1_put,OF_bneck_13_wts_memtile_layer1_get, OF_bneck_13_wts_memtile_layer2],[],\
            #                  [0,\
            #                   (bneck_13_InC1 * bneck_13_OutC1//2),\
            #                     bneck_13_InC1 * bneck_13_OutC1])

            
            # Set up compute tiles

            # rtp02 = Buffer(ComputeTile02, [16], T.i32(), "rtp02")
            # rtp03 = Buffer(ComputeTile04, [16], T.i32(), "rtp03")
            
            OF_bneck_13_act_layer1_layer2 = object_fifo("OF_bneck_13_act_layer1_layer2", ComputeTile04, [MemTile01], 2,ty_bneck_13_layer2_in)

            # OF_bneck_13_act_layer1_layer2 = object_fifo("OF_bneck_13_act_layer1_layer2", ComputeTile04, [ComputeTile03], 4,ty_bneck_13_layer2_in,via_DMA=True)
            # OF_bneck_13_act_layer2_layer3 = object_fifo("OF_bneck_13_act_layer2_layer3", ComputeTile04, [ComputeTile05], 2,ty_bneck_13_layer3_in)


            # OF_bneck_13_layer3_bn_12_layer2 = object_fifo("OF_bneck_13_layer3_bn_12_layer2", ComputeTile03, [MemTile01], 2, ty_bneck_13_layer3_in)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile01, [ShimTile00], 2, ty_bneck_13_layer3_in)
            object_fifo_link(OF_bneck_13_act_layer1_layer2, OF_outOFL2L3)
            rtp04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")
        # ************************ bneck13 ************************
             # 1x1 conv2d
            # Compute tile 5
            @core(ComputeTile05, "conv2dk1_put.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(bneck_13_InH1):
                        elemIn = inOF_act_L3L2.acquire(ObjectFifoPort.Consume, 1)
                        for oc in range(0,OutputSplit):
                            for WeightIndex in range (0,InputSplit//2):
                                elemWts = OF_bneck_13_wts_memtile_layer1_put.acquire(ObjectFifoPort.Consume, 1)
                                for x_start in range(0,bneck_13_InW1,8):
                                    call(
                                        bn13_conv2dk1_fused_relu_put,
                                        [
                                            elemIn,
                                            elemWts,
                                            arith.constant(bneck_13_InW1),
                                            arith.constant(bneck_13_InC1),
                                            arith.constant(bneck_13_OutC1),
                                            InputSplit,
                                            WeightIndex,
                                            x_start,
                                            oc
                                        ],
                                    )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer1_put", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)
                        
                        yield_([])
                    yield_([])

            # Compute tile 4
            @core(ComputeTile04, "conv2dk1_get.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(bneck_13_InH1):
                        elemIn = inOF_act_L3L2.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = OF_bneck_13_act_layer1_layer2.acquire(ObjectFifoPort.Produce, 1)
                        
                        scale = memref.load(rtp04, [0])
                        for oc in range(0,OutputSplit):
                            for WeightIndex in range (InputSplit//2,InputSplit ):
                                elemWts = OF_bneck_13_wts_memtile_layer1_get.acquire(ObjectFifoPort.Consume, 1)
                                for x_start in range(0,bneck_13_InW1,8):
                                    call(
                                        bn13_conv2dk1_fused_relu_get,
                                        [
                                            elemIn,
                                            elemWts,
                                            elemOut0,
                                            arith.constant(bneck_13_InW1),
                                            arith.constant(bneck_13_InC1),
                                            arith.constant(bneck_13_OutC1),
                                            scale,
                                            InputSplit,
                                            WeightIndex,
                                            x_start,
                                            oc
                                        ],
                                    )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer1_get", 1)
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L3L2", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_act_layer1_layer2", 1)
                        
                        yield_([])
                    yield_([])

            # # # # Compute tile 3
            # @core(ComputeTile03, "bn10_conv2dk3_dw.o")
            # def core_body():
            #     scale = 8
            #     for _ in for_(sys.maxsize):

            #         # acquire weights and rtps once
            #         element0Weights = OF_bneck_13_wts_memtile_layer2.acquire(ObjectFifoPort.Consume, 1)
            #         # scale = memref.load(rtpComputeTile04, 0)

            #         # pre-amble: top row
            #         elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
            #             ObjectFifoPort.Consume, 2
            #         )
            #         element0ActivactionsOut = OF_bneck_13_layer3_bn_12_layer2.acquire(ObjectFifoPort.Produce, 1)
            #         res = call(
            #             bn13_conv2dk3_dw,
            #             [
            #                 elementActivactionsIn[0],
            #                 elementActivactionsIn[0],
            #                 elementActivactionsIn[1],
            #                 element0Weights,
            #                 element0ActivactionsOut,
            #                 bneck_13_InW2,
            #                 1,
            #                 bneck_13_OutC2,
            #                 3,
            #                 3,
            #                 0,
            #                 scale,
            #                 0,
            #             ],
            #         )
            #         objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_layer3_bn_12_layer2", 1)

            #         # middle
            #         for _ in for_(bneck_13_InH2 - 2):
            #             elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
            #                 ObjectFifoPort.Consume, 3
            #             )
            #             element0ActivactionsOut = OF_bneck_13_layer3_bn_12_layer2.acquire(
            #                 ObjectFifoPort.Produce, 1
            #             )
            #             res = call(
            #                 bn13_conv2dk3_dw,
            #                 [
            #                     elementActivactionsIn[0],
            #                     elementActivactionsIn[1],
            #                     elementActivactionsIn[2],
            #                     element0Weights,
            #                     element0ActivactionsOut,
            #                     bneck_13_InW2,
            #                     1,
            #                     bneck_13_OutC2,
            #                     3,
            #                     3,
            #                     1,
            #                     scale,
            #                     0,
            #                 ],
            #             )

            #             objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer1_layer2", 1)
            #             objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_layer3_bn_12_layer2", 1)
            #             yield_([])

            #         # last part
            #         elementActivactionsIn = OF_bneck_13_act_layer1_layer2.acquire(
            #             ObjectFifoPort.Consume, 2
            #         )
            #         element0ActivactionsOut = OF_bneck_13_layer3_bn_12_layer2.acquire(ObjectFifoPort.Produce, 1)
            #         res = call(
            #             bn13_conv2dk3_dw,
            #             [
            #                 elementActivactionsIn[0],
            #                 elementActivactionsIn[1],
            #                 elementActivactionsIn[1],
            #                 element0Weights,
            #                 element0ActivactionsOut,
            #                 bneck_13_InW2,
            #                 1,
            #                 bneck_13_OutC2,
            #                 3,
            #                 3,
            #                 2,
            #                 scale,
            #                 0,
            #             ],
            #         )

            #         objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer1_layer2", 2)
            #         objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_layer3_bn_12_layer2", 1)

            #         objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer2", 1)
            #         yield_([])

            # # # Compute tile 4
            # # @core(ComputeTile05, "bn_conv2dk1_skip.o")
            # # def core_body():

            # #     for _ in for_(0xFFFFFFFF):
            # #         elemWts = OF_bneck_13_wts_memtile_layer3.acquire(ObjectFifoPort.Consume, 1)

            # #         scale = memref.load(rtp04, [0])
            # #         skipScale = memref.load(rtp04, [1])
            # #         # scale = memref.load(rtpComputeTile02, [0])

            # #         for _ in for_(bneck_13_InH3):
            # #             elemIn = OF_bneck_13_act_layer2_layer3.acquire(ObjectFifoPort.Consume, 1)
            # #             elemOut0 = OF_bneck_13_layer3_bn_12_layer1.acquire(ObjectFifoPort.Produce, 1)
            # #             elementSkipsIn = OF_bneck_13_skip.acquire(
            # #                     ObjectFifoPort.Consume, 1
            # #                 )

            # #             call(
            # #                 bn13_conv2dk1_skip,
            # #                 [
            # #                     elemIn,
            # #                     elemWts,
            # #                     elemOut0,
            # #                     elementSkipsIn,
            # #                     bneck_13_InW3,
            # #                     bneck_13_OutC2,
            # #                     bneck_13_OutC3,
            # #                     scale,
            # #                     skipScale,
            # #                 ],
            # #             )

            # #             objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_act_layer2_layer3", 1)
            # #             objectfifo_release(ObjectFifoPort.Produce, "OF_bneck_13_layer3_bn_12_layer1", 1)
            # #             objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_skip", 1)
            # #             yield_([])
            # #         objectfifo_release(ObjectFifoPort.Consume, "OF_bneck_13_wts_memtile_layer3", 1)
            # #         yield_([])
            

            # # instruction stream generation
            activationsInSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_InC1) // 4
            # acitivationsOutSize32b = (bneck_12_InW2 * bneck_12_InH2 * bneck_12_OutC3) // 4
            acitivationsOutSize32b = (bneck_13_InW1 * bneck_13_InH1 * bneck_13_OutC1) // 4

            bn13_layer_1_totalWeightsSize32b = (
            bneck_13_InC1*bneck_13_OutC1

            #    +bneck_13_OutC2*bneck_13_OutC3
            ) // 4

            bn13_layer_2_totalWeightsSize32b = (
               3 * 3 * bneck_13_OutC2 * 1
            #    +bneck_13_OutC2*bneck_13_OutC3
            ) // 4

            bn13_totalWeightsSize32b = (
            bneck_13_InC1*bneck_13_OutC1+
               3 * 3 * bneck_13_OutC2 * 1
            #    +bneck_13_OutC2*bneck_13_OutC3
            ) // 4



            totalWeightsSize32b_complete = (
                bn13_totalWeightsSize32b 
            )

            activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
            weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
            activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                # NpuWriteRTPOp("rtp02", col=0, row=2, index=0, value=9)
                # NpuWriteRTPOp("rtp03", col=0, row=3, index=0, value=8)
                NpuWriteRTPOp("rtp04", col=0, row=4, index=0, value=9)
                # NpuWriteRTPOp("rtp04", col=0, row=4, index=1, value=0)
                
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
                    metadata="OF_bneck_13_wts_L3L2_layer1",
                    bd_id=1,
                    mem=weightsFromL3,
                    sizes=[1, 1, 1, bn13_layer_1_totalWeightsSize32b],
                )
                npu_dma_memcpy_nd(
                    metadata="OF_bneck_13_wts_L3L2_layer2",
                    bd_id=1,
                    mem=weightsFromL3,
                    offsets=[0, 0, 0, bn13_layer_1_totalWeightsSize32b],
                    sizes=[1, 1, 1, bn13_layer_2_totalWeightsSize32b],
                )

                npu_sync(column=0, row=0, direction=0, channel=0)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


mobilenetBottleneckC()