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

InW2 = 7

InH2 = 7
InC = 960
OutC = 160
InputSplit=2
OutputSplit=OutC//8 #calculate 8 OCs at a time, should increase to more


RepeatChannels=math.floor(InH2)

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
  
            ty_in_full = MemRefType.get((InW2, 1, InC, ), uint8_ty, ) 
            ty_in_split = MemRefType.get((InW2, 1, InC//InputSplit, ), uint8_ty, )
            # define wts
            ty_wts = MemRefType.get(((InC * 8)//(InputSplit),), int8_ty )
            ty_all_wts= MemRefType.get((InC * OutC, ), int8_ty, )
            ty_out = MemRefType.get((InW2, 1, OutC, ), uint8_ty, )
                    
            # HERE            
            conv2dk1_put = external_func(
                "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put",
                inputs=[
                    ty_in_split,
                    ty_wts,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                    int32_ty,
                ],
            )

            conv2dk1_get = external_func(
                "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_get",
                inputs=[
                    ty_in_split,
                    ty_wts,
                    ty_out,
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
            inOF_act_L3L2 = object_fifo("inOF_act_L3L2", ShimTile00, MemTile01, 2, ty_in_full, )

            inOF_act_L2_put = object_fifo("inOF_act_L2_put", MemTile01, ComputeTile05, 2, ty_in_split, )
            inOF_act_L2_get = object_fifo("inOF_act_L2_get", MemTile01, ComputeTile04, 2, ty_in_split, )
            object_fifo_link(inOF_act_L3L2, [inOF_act_L2_put,inOF_act_L2_get],[],[0,(InW2*InC//InputSplit)])
            # # wts
            OF_wts_L3L2 = object_fifo("OF_wts_L3L2", ShimTile00, MemTile01, 1, ty_all_wts )
            OF_wts_memtile_put = object_fifo("OF_wts_memtile_put", MemTile01, ComputeTile05, 1, ty_wts)
            OF_wts_memtile_get = object_fifo("OF_wts_memtile_get", MemTile01, ComputeTile04, 1, ty_wts)
           
            object_fifo_link(OF_wts_L3L2, [OF_wts_memtile_put,OF_wts_memtile_get],[],[0,(InC * OutC)//2])
            OF_wts_memtile_put.set_memtile_repeat(RepeatChannels)
            OF_wts_memtile_get.set_memtile_repeat(RepeatChannels)
        
            # Set up compute tiles
            rtp04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")


            out_04_L2 = object_fifo("out_04_L2", ComputeTile04, [MemTile01], 2, ty_out)
            OF_outOFL2L3 = object_fifo("outOFL2L3", MemTile01, [ShimTile00], 2, ty_out)
            object_fifo_link(out_04_L2, OF_outOFL2L3)


            # Compute tile 4
            @core(ComputeTile05, "conv2dk1_put.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(InH2):
                        elemIn = inOF_act_L2_put.acquire(ObjectFifoPort.Consume, 1)
                        # for oc in range(0,OutputSplit):
                        for oc in for_(OutputSplit):
                            oc_cast= arith.IndexCastOp(T.i32(), oc)
                            # for WeightIndex in range (0,InputSplit//2):
                            for WeightIndex in for_(0,InputSplit//2): #how many input channel splits, 1 in case InputSplit is 2
                                WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                                elemWts = OF_wts_memtile_put.acquire(ObjectFifoPort.Consume, 1)
                                x_start=0  
                                call(
                                    conv2dk1_put,
                                    [
                                        elemIn,
                                        elemWts,
                                        arith.constant(InW2),
                                        arith.constant(InC),
                                        arith.constant(OutC),
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_wts_memtile_put", 1)
                                yield_([])
                            yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L2_put", 1)
                        
                        yield_([])
                    yield_([])

            # Compute tile 4
            @core(ComputeTile04, "conv2dk1_get.o")
            def core_body():
                for _ in for_(0xFFFFFFFF):
                    
                    for _ in for_(InH2):
                        
                        elemIn = inOF_act_L2_get.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = out_04_L2.acquire(ObjectFifoPort.Produce, 1)
                        
                        scale = 11
                        # scale = memref.load(rtp04, [0])
                        # for oc in range(0,OutputSplit):
                        for oc in for_(OutputSplit):
                            
                            oc_cast= arith.IndexCastOp(T.i32(), oc)
                            for WeightIndex in for_(0,InputSplit//2):
                                WeightIndex_cast= arith.IndexCastOp(T.i32(), WeightIndex)
                                elemWts = OF_wts_memtile_get.acquire(ObjectFifoPort.Consume, 1)
                                x_start=0 

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
                                        InputSplit,
                                        WeightIndex_cast,
                                        x_start,
                                        oc_cast
                                    ],
                                )
                                objectfifo_release(ObjectFifoPort.Consume, "OF_wts_memtile_get", 1)
                                yield_([])
                            yield_([])
                        objectfifo_release(ObjectFifoPort.Consume, "inOF_act_L2_get", 1)
                        objectfifo_release(ObjectFifoPort.Produce, "out_04_L2", 1)
                        
                        yield_([])
                    yield_([])

            # # instruction stream generation
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

            @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                NpuWriteRTPOp("rtp04", col=0, row=4, index=0, value=11)

                
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