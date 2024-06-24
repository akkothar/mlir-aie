#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

from aie2_bottleneckA import bottleneckACore

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import *
from aie.extras.dialects.ext.memref import view as memref_view

def mobilenetV3_bn_6_7(tileColIndex = 0,tensorInW = 112, tensorInH = 112, tensorInC = 16, 
                       bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 9, bn6_scaleFactor3 = 11, 
                       bn7_depthWiseChannels = 200, bn7_depthWiseStride = 1, bn7_withSkip = True, bn7_tensorOutC = 80, bn7_scaleFactor1 = 9, bn7_scaleFactor2 = 8, bn7_scaleFactor3 = 11, bn7_scaleFactorAdd = 0,
                       enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    tensorL6_1InC = tensorInC
    tensorL6_1InW = tensorInW
    tensorL6_1InH = tensorInH

    tensorL6_2InC = bn6_depthWiseChannels
    tensorL6_2InW = tensorL6_1InW
    tensorL6_2InH = tensorL6_1InH

    tensorL6_3InC = tensorL6_2InC
    tensorL6_3InW = tensorL6_2InW // bn6_depthWiseStride
    tensorL6_3InH = tensorL6_2InH // bn6_depthWiseStride
    tensorL6_3OutC = bn6_tensorOutC

    tensorL7_1InC = tensorL6_3OutC
    tensorL7_1InW = tensorL6_3InW
    tensorL7_1InH = tensorL6_3InH

    tensorL7_2InC = bn7_depthWiseChannels
    tensorL7_2InW = tensorL7_1InW
    tensorL7_2InH = tensorL7_1InH

    tensorL7_3InC = tensorL7_2InC
    tensorL7_3InW = tensorL7_2InW // bn7_depthWiseStride
    tensorL7_3InH = tensorL7_2InH // bn7_depthWiseStride
    tensorL7_3OutC = bn7_tensorOutC

    # final output
    tensorOutW = tensorL7_3InW
    tensorOutH = tensorL7_3InH
    tensorOutC = tensorL7_3OutC

    @device(AIEDevice.npu1_1col)
    def device_body():
        
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        tensorLayerIn_ty = MemRefType.get((tensorInW, 1, tensorInC), int8_ty)
        tensorLayerOut_ty = MemRefType.get((tensorOutW, 1, tensorOutC), int8_ty)

        # setup all the weights here
        bn6_weights_size=1 * 1 * tensorL6_1InC * tensorL6_2InC + 3 * 3 * tensorL6_3InC * 1 + 1 * 1 * tensorL6_3InC * tensorL6_3OutC
        bn6_weightsAllLayers_ty = MemRefType.get((bn6_weights_size,), int8_ty)
        bn7_weights_size=1 * 1 * tensorL7_1InC * tensorL7_2InC + 3 * 3 * tensorL7_3InC * 1 + 1 * 1 * tensorL7_3InC * tensorL7_3OutC
        bn7_weightsAllLayers_ty = MemRefType.get((bn7_weights_size,), int8_ty)
        total_weights=bn6_weights_size+bn7_weights_size
        total_weights_ty = MemRefType.get((total_weights,), int8_ty)

        # temporary types for tensor to enable intial test
        bn6_tensorLayer1In_ty = MemRefType.get((tensorL6_1InW, 1, tensorL6_1InC), int8_ty)
        bn6_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL6_1InC * tensorL6_2InC,), int8_ty)
        bn6_tensorLayer2In_ty = MemRefType.get((tensorL6_2InW, 1, tensorL6_2InC), uint8_ty)
        bn6_tensorLayer1Out_ty = bn6_tensorLayer2In_ty
        bn6_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL6_3InC * 1,), int8_ty)
        bn6_tensorLayer3In_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3InC), uint8_ty)
        bn6_tensorLayer2Out_ty = bn6_tensorLayer3In_ty
        bn6_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL6_3InC * tensorL6_3OutC,), int8_ty)
        bn6_tensorLayer3Out_ty = MemRefType.get((tensorL7_1InW, 1, tensorL6_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn6_conv2dk1_relu_i8_ui8 = external_func("bn6_conv2dk1_relu_i8_ui8",inputs=[bn6_tensorLayer1In_ty, bn6_weightsLayer1_ty, bn6_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn6_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty, bn6_weightsLayer2_ty, bn6_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn6_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty,bn6_tensorLayer2In_ty, bn6_weightsLayer2_ty, bn6_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk1_skip_ui8_i8_i8 = external_func("bn6_conv2dk1_skip_ui8_i8_i8",inputs=[bn6_tensorLayer3In_ty, bn6_weightsLayer3_ty, bn6_tensorLayer3Out_ty, bn6_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn6_conv2dk1_ui8_i8 = external_func("bn6_conv2dk1_ui8_i8",inputs=[bn6_tensorLayer3In_ty, bn6_weightsLayer3_ty, bn6_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        ShimTile = tile(tileColIndex, 0)
        MemTile = tile(tileColIndex, 1)
        ComputeTile2 = tile(tileColIndex, 2)
        ComputeTile3 = tile(tileColIndex, 3)

        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile, ComputeTile2, 2, tensorLayerIn_ty)

        # wts
        wts_OF_L3L2 = object_fifo("wts_OF_L3L2", ShimTile, MemTile, 1, total_weights_ty)
        bn6_wts_OF_L3L1 = object_fifo("bn6_wts_OF_L2L1", MemTile, ComputeTile2, [1,1], bn6_weightsAllLayers_ty)
        bn7_wts_OF_L3L1 = object_fifo("bn7_wts_OF_L2L1", MemTile, ComputeTile3, [1,1], bn7_weightsAllLayers_ty)
        object_fifo_link(wts_OF_L3L2, [bn6_wts_OF_L3L1,bn7_wts_OF_L3L1],[],[0,bn6_weights_size])

        # Output
        act_out = object_fifo("act_out", ComputeTile3, [ShimTile], 1, tensorLayerOut_ty)
                
        # Set up compute tiles
        rtpComputeTile2 = Buffer(ComputeTile2, [16], T.i32(), "rtp2")
        rtpComputeTile3 = Buffer(ComputeTile2, [16], T.i32(), "rtp3")
        
        # Compute tile 6
        bn6_objectArchiveName = "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn6_depthWiseStride, "skip" if (bn6_withSkip) else "")
        bn6_tensorLayer1Out_ty = MemRefType.get((tensorL6_2InW, 1, tensorL6_2InC),uint8_ty)
        bn6_tensorLayer2Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3InC),uint8_ty)
        bn6_tensorLayer3Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3OutC),int8_ty)

        # between compute tiles
        act_bn6_bn7 = object_fifo("act_bn6_bn7", ComputeTile2, ComputeTile3, 2, bn6_tensorLayer3Out_ty)

        bottleneckACore("bn6", ComputeTile2, act_in, bn6_wts_OF_L3L1, act_bn6_bn7, rtpComputeTile2, bn6_objectArchiveName,
                         bn6_conv2dk1_relu_i8_ui8, bn6_conv2dk3_dw_stride1_relu_ui8_ui8, bn6_conv2dk3_dw_stride2_relu_ui8_ui8, bn6_conv2dk1_ui8_i8, bn6_conv2dk1_skip_ui8_i8_i8,
                           bn6_tensorLayer1Out_ty, bn6_tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, bn6_depthWiseStride, bn6_depthWiseChannels, tensorOutC, bn6_withSkip)

        # ******************************************************************************************************************************
        
        bn7_tensorLayer1In_ty = MemRefType.get((tensorL7_1InW, 1, tensorL7_1InC), int8_ty)
        bn7_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL7_1InC * tensorL7_2InC,), int8_ty)
        bn7_tensorLayer2In_ty = MemRefType.get((tensorL7_2InW, 1, tensorL7_2InC), uint8_ty)
        bn7_tensorLayer1Out_ty = bn7_tensorLayer2In_ty
        bn7_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL7_3InC * 1,), int8_ty)
        bn7_tensorLayer3In_ty = MemRefType.get((tensorL7_3InW, 1, tensorL7_3InC), uint8_ty)
        bn7_tensorLayer2Out_ty = bn7_tensorLayer3In_ty
        bn7_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL7_3InC * tensorL7_3OutC,), int8_ty)
        bn7_tensorLayer3Out_ty = MemRefType.get((tensorL7_3InW, 1, tensorL7_3OutC),int8_ty)
        
        

        # AIE Core Function declarations
        bn7_conv2dk1_relu_i8_ui8 = external_func("bn7_conv2dk1_relu_i8_ui8",inputs=[bn7_tensorLayer1In_ty, bn7_weightsLayer1_ty, bn7_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn7_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty, bn7_weightsLayer2_ty, bn7_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn7_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty,bn7_tensorLayer2In_ty, bn7_weightsLayer2_ty, bn7_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk1_skip_ui8_i8_i8 = external_func("bn7_conv2dk1_skip_ui8_i8_i8",inputs=[bn7_tensorLayer3In_ty, bn7_weightsLayer3_ty, bn7_tensorLayer3Out_ty, bn7_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn7_conv2dk1_ui8_i8 = external_func("bn7_conv2dk1_ui8_i8",inputs=[bn7_tensorLayer3In_ty, bn7_weightsLayer3_ty, bn7_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        bn7_objectArchiveName = "bn7_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn7_depthWiseStride, "skip" if (bn7_withSkip) else "")

        bottleneckACore("bn7", ComputeTile3, act_bn6_bn7, bn7_wts_OF_L3L1, act_out, rtpComputeTile3, bn7_objectArchiveName, 
                        bn7_conv2dk1_relu_i8_ui8, bn7_conv2dk3_dw_stride1_relu_ui8_ui8, bn7_conv2dk3_dw_stride2_relu_ui8_ui8, bn7_conv2dk1_ui8_i8, bn7_conv2dk1_skip_ui8_i8_i8, 
                        bn7_tensorLayer1Out_ty, bn7_tensorLayer2Out_ty, 
                        tensorL7_1InW, tensorL7_1InH, tensorL7_1InC, bn7_depthWiseStride, bn7_depthWiseChannels, tensorOutC, bn7_withSkip)

with mlir_mod_ctx() as ctx:
    mobilenetV3_bn_6_7(tileColIndex = 0,tensorInW = 28, tensorInH = 28, tensorInC = 40, 
                    bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 9, bn6_scaleFactor3 = 11, 
                    bn7_depthWiseStride = 1, bn7_depthWiseChannels = 200, bn7_withSkip = True, bn7_tensorOutC = 80, bn7_scaleFactor1 = 9, bn7_scaleFactor2 = 8, bn7_scaleFactor3 = 11, bn7_scaleFactorAdd = 0,
                    enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)

