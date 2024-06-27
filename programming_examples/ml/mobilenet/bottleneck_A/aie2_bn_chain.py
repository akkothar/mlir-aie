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

def mobilenetV3_bn_4_5_6_7_8(tileColIndex = 0,tensorInW = 28, tensorInH = 28, tensorInC = 40, 
                    # bn2_depthWiseStride = 2, bn2_depthWiseChannels = 240, bn2_withSkip = False, bn2_tensorOutC = 80, bn2_scaleFactor1 = 8, bn2_scaleFactor2 = 8, bn2_scaleFactor3 = 11,  bn2_scaleFactorAdd = 0,
                    #    bn3_depthWiseStride = 2, bn3_depthWiseChannels = 240, bn3_withSkip = False, bn3_tensorOutC = 80, bn3_scaleFactor1 = 8, bn3_scaleFactor2 = 8, bn3_scaleFactor3 = 11,  bn3_scaleFactorAdd = 0,
                       bn4_depthWiseStride = 1, bn4_depthWiseChannels = 120, bn4_withSkip = False, bn4_tensorOutC = 80, bn4_scaleFactor1 = 8, bn4_scaleFactor2 = 8, bn4_scaleFactor3 = 11,  bn4_scaleFactorAdd = 0,
                       bn5_depthWiseStride = 1, bn5_depthWiseChannels = 120, bn5_withSkip = True, bn5_tensorOutC = 80, bn5_scaleFactor1 = 8, bn5_scaleFactor2 = 8, bn5_scaleFactor3 = 11,  bn5_scaleFactorAdd = 0,
                       bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 8, bn6_scaleFactor3 = 11,  bn6_scaleFactorAdd = 0,
                       bn7_depthWiseStride = 1, bn7_depthWiseChannels = 200, bn7_withSkip = True, bn7_tensorOutC = 80, bn7_scaleFactor1 = 9, bn7_scaleFactor2 = 8, bn7_scaleFactor3 = 11, bn7_scaleFactorAdd = 0,
                       bn8_depthWiseStride = 1, bn8_depthWiseChannels = 184, bn8_withSkip = True, bn8_tensorOutC = 80, bn8_scaleFactor1 = 9, bn8_scaleFactor2 = 8, bn8_scaleFactor3 = 11, bn8_scaleFactorAdd = 0,
                       enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096):

    tensorL4_1InC = tensorInC
    tensorL4_1InW = tensorInW
    tensorL4_1InH = tensorInH
    
    tensorL4_2InC = bn4_depthWiseChannels 
    tensorL4_2InW = tensorL4_1InW
    tensorL4_2InH = tensorL4_1InH

    tensorL4_3InC = tensorL4_2InC
    tensorL4_3InW = tensorL4_2InW // bn5_depthWiseStride
    tensorL4_3InH = tensorL4_2InH // bn5_depthWiseStride
    tensorL4_3OutC = bn4_tensorOutC
    # 

    tensorL5_1InC = tensorL4_3OutC
    tensorL5_1InW = tensorL4_3InW
    tensorL5_1InH = tensorL4_3InH

    tensorL5_2InC = bn5_depthWiseChannels
    tensorL5_2InW = tensorL5_1InW
    tensorL5_2InH = tensorL5_1InH

    tensorL5_3InC = tensorL5_2InC
    tensorL5_3InW = tensorL5_2InW // bn5_depthWiseStride
    tensorL5_3InH = tensorL5_2InH // bn5_depthWiseStride
    tensorL5_3OutC = bn5_tensorOutC
    # 
    tensorL6_1InC = tensorL5_3OutC
    tensorL6_1InW = tensorL5_3InW
    tensorL6_1InH = tensorL5_3InH

    tensorL6_2InC = bn6_depthWiseChannels
    tensorL6_2InW = tensorL6_1InW
    tensorL6_2InH = tensorL6_1InH

    tensorL6_3InC = tensorL6_2InC
    tensorL6_3InW = tensorL6_2InW // bn6_depthWiseStride
    tensorL6_3InH = tensorL6_2InH // bn6_depthWiseStride
    tensorL6_3OutC = bn6_tensorOutC
# 
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
# 
    tensorL8_1InC = tensorL7_3OutC
    tensorL8_1InW = tensorL7_3InW
    tensorL8_1InH = tensorL7_3InH

    tensorL8_2InC = bn8_depthWiseChannels
    tensorL8_2InW = tensorL8_1InW
    tensorL8_2InH = tensorL8_1InH

    tensorL8_3InC = tensorL8_2InC
    tensorL8_3InW = tensorL8_2InW // bn8_depthWiseStride
    tensorL8_3InH = tensorL8_2InH // bn8_depthWiseStride
    tensorL8_3OutC = bn8_tensorOutC

    # final output
    tensorOutW = tensorL8_3InW
    tensorOutH = tensorL8_3InH
    tensorOutC = tensorL8_3OutC

    tensorOutW = tensorL8_3InW
    tensorOutH = tensorL8_3InH
    tensorOutC = tensorL8_3OutC

    @device(AIEDevice.npu1_2col)
    def device_body():
        
        # define types
        uint8_ty = IntegerType.get_unsigned(8)
        int8_ty = IntegerType.get_signless(8)
        int16_ty = IntegerType.get_signless(16)
        int32_ty = IntegerType.get_signless(32)

        tensorLayerIn_ty = MemRefType.get((tensorInW, 1, tensorInC), int8_ty)
        tensorLayerOut_ty = MemRefType.get((tensorOutW, 1, tensorOutC), int8_ty)

        # setup all the weights here
        bn4_weights_size=1 * 1 * tensorL4_1InC * tensorL4_2InC + 3 * 3 * tensorL4_3InC * 1 + 1 * 1 * tensorL4_3InC * tensorL4_3OutC
        bn4_weightsAllLayers_ty = MemRefType.get((bn4_weights_size,), int8_ty)
        bn5_weights_size=1 * 1 * tensorL5_1InC * tensorL5_2InC + 3 * 3 * tensorL5_3InC * 1 + 1 * 1 * tensorL5_3InC * tensorL5_3OutC
        bn5_weightsAllLayers_ty = MemRefType.get((bn5_weights_size,), int8_ty)
        bn6_weights_size=1 * 1 * tensorL6_1InC * tensorL6_2InC + 3 * 3 * tensorL6_3InC * 1 + 1 * 1 * tensorL6_3InC * tensorL6_3OutC
        bn6_weightsAllLayers_ty = MemRefType.get((bn6_weights_size,), int8_ty)
        bn7_weights_size=1 * 1 * tensorL7_1InC * tensorL7_2InC + 3 * 3 * tensorL7_3InC * 1 + 1 * 1 * tensorL7_3InC * tensorL7_3OutC
        bn7_weightsAllLayers_ty = MemRefType.get((bn7_weights_size,), int8_ty)
        bn8_weights_size=1 * 1 * tensorL8_1InC * tensorL8_2InC + 3 * 3 * tensorL8_3InC * 1 + 1 * 1 * tensorL8_3InC * tensorL8_3OutC
        bn8_weightsAllLayers_ty = MemRefType.get((bn8_weights_size,), int8_ty)
 
        memtile_01_wts=bn4_weights_size+bn5_weights_size
        memtile_01_wts_ty = MemRefType.get((memtile_01_wts,), int8_ty)
        
        memtile_11_wts=bn6_weights_size+bn7_weights_size+bn8_weights_size
        memtile_11_wts_ty = MemRefType.get((memtile_11_wts,), int8_ty)

        total_weights=memtile_01_wts+memtile_11_wts
        total_weights_ty = MemRefType.get((total_weights,), int8_ty)


        ShimTile00 = tile(tileColIndex, 0)
        ShimTile10 = tile(tileColIndex+1, 0)
        
        MemTile01 = tile(tileColIndex, 1)
        MemTile11 = tile(tileColIndex+1, 1)

        ComputeTile03 = tile(tileColIndex, 3)
        ComputeTile04 = tile(tileColIndex, 4)
        ComputeTile12 = tile(tileColIndex+1, 2)
        ComputeTile13 = tile(tileColIndex+1, 3)
        ComputeTile14 = tile(tileColIndex+1, 4)

                
        # Set up compute tiles
        rtpComputeTile03 = Buffer(ComputeTile03, [16], T.i32(), "rtp03")
        rtpComputeTile04 = Buffer(ComputeTile04, [16], T.i32(), "rtp04")
        rtpComputeTile12 = Buffer(ComputeTile12, [16], T.i32(), "rtp12")
        rtpComputeTile13 = Buffer(ComputeTile13, [16], T.i32(), "rtp13")
        rtpComputeTile14 = Buffer(ComputeTile14, [16], T.i32(), "rtp14")
        
        # AIE-array data movement with object fifos
        
        # Input
        act_in = object_fifo("act_in", ShimTile00, ComputeTile03, 2, tensorLayerIn_ty)

        # wts
        wts_OF_01_L3L2 = object_fifo("wts_OF_01_L3L2", ShimTile00, MemTile01, 1, memtile_01_wts_ty)
        bn4_wts_OF_L3L1 = object_fifo("bn4_wts_OF_L2L1", MemTile01, ComputeTile03, [1,1], bn4_weightsAllLayers_ty)
        bn5_wts_OF_L3L1 = object_fifo("bn5_wts_OF_L2L1", MemTile01, ComputeTile04, [1,1], bn5_weightsAllLayers_ty)
        object_fifo_link(wts_OF_01_L3L2, [bn4_wts_OF_L3L1,bn5_wts_OF_L3L1],[],[0,bn4_weights_size])

        #  # wts
        wts_OF_11_L3L2 = object_fifo("wts_OF_11_L3L2", ShimTile10, MemTile11, 1, memtile_11_wts_ty)
        bn6_wts_OF_L3L1 = object_fifo("bn6_wts_OF_L2L1", MemTile11, ComputeTile12, [1,1], bn6_weightsAllLayers_ty)
        bn7_wts_OF_L3L1 = object_fifo("bn7_wts_OF_L2L1", MemTile11, ComputeTile13, [1,1], bn7_weightsAllLayers_ty)
        bn8_wts_OF_L3L1 = object_fifo("bn8_wts_OF_L2L1", MemTile11, ComputeTile14, [1,1], bn8_weightsAllLayers_ty)
        object_fifo_link(wts_OF_11_L3L2, [bn6_wts_OF_L3L1,bn7_wts_OF_L3L1,bn8_wts_OF_L3L1],[],[0,bn6_weights_size,bn6_weights_size+bn7_weights_size])

         # temporary types for tensor to enable intial test
        bn4_tensorLayer1In_ty = MemRefType.get((tensorL4_1InW, 1, tensorL4_1InC), int8_ty)
        bn4_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL4_1InC * tensorL4_2InC,), int8_ty)
        bn4_tensorLayer2In_ty = MemRefType.get((tensorL4_2InW, 1, tensorL4_2InC), uint8_ty)
        bn4_tensorLayer1Out_ty = bn4_tensorLayer2In_ty
        bn4_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL4_3InC * 1,), int8_ty)
        bn4_tensorLayer3In_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3InC), uint8_ty)
        bn4_tensorLayer2Out_ty = bn4_tensorLayer3In_ty
        bn4_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL4_3InC * tensorL4_3OutC,), int8_ty)
        bn4_tensorLayer3Out_ty = MemRefType.get((tensorL6_1InW, 1, tensorL4_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn4_conv2dk1_relu_i8_ui8 = external_func("bn4_conv2dk1_relu_i8_ui8",inputs=[bn4_tensorLayer1In_ty, bn4_weightsLayer1_ty, bn4_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn4_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty, bn4_weightsLayer2_ty, bn4_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn4_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty,bn4_tensorLayer2In_ty, bn4_weightsLayer2_ty, bn4_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk1_skip_ui8_i8_i8 = external_func("bn4_conv2dk1_skip_ui8_i8_i8",inputs=[bn4_tensorLayer3In_ty, bn4_weightsLayer3_ty, bn4_tensorLayer3Out_ty, bn4_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn4_conv2dk1_ui8_i8 = external_func("bn4_conv2dk1_ui8_i8",inputs=[bn4_tensorLayer3In_ty, bn4_weightsLayer3_ty, bn4_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 6
        bn4_objectArchiveName = "bn4_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn4_depthWiseStride, "skip" if (bn4_withSkip) else "")
        bn4_tensorLayer1Out_ty = MemRefType.get((tensorL4_2InW, 1, tensorL4_2InC),uint8_ty)
        bn4_tensorLayer2Out_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3InC),uint8_ty)
        bn4_tensorLayer3Out_ty = MemRefType.get((tensorL4_3InW, 1, tensorL4_3OutC),int8_ty)        

       

        # between compute tiles
        act_bn4_bn5 = object_fifo("act_bn4_bn5", ComputeTile03, ComputeTile04, 2, bn4_tensorLayer3Out_ty)

        bottleneckACore("bn4", ComputeTile03, act_in, bn4_wts_OF_L3L1, act_bn4_bn5, rtpComputeTile03, bn4_objectArchiveName,
                         bn4_conv2dk1_relu_i8_ui8, bn4_conv2dk3_dw_stride1_relu_ui8_ui8, bn4_conv2dk3_dw_stride2_relu_ui8_ui8, bn4_conv2dk1_ui8_i8, bn4_conv2dk1_skip_ui8_i8_i8,
                           bn4_tensorLayer1Out_ty, bn4_tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, bn4_depthWiseStride, bn4_depthWiseChannels, tensorL4_3OutC, bn4_withSkip)

        # # ******************************************************************bn5******************************************************************

        # temporary types for tensor to enable intial test
        bn5_tensorLayer1In_ty = MemRefType.get((tensorL5_1InW, 1, tensorL5_1InC), int8_ty)
        bn5_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL5_1InC * tensorL5_2InC,), int8_ty)
        bn5_tensorLayer2In_ty = MemRefType.get((tensorL5_2InW, 1, tensorL5_2InC), uint8_ty)
        bn5_tensorLayer1Out_ty = bn5_tensorLayer2In_ty
        bn5_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL5_3InC * 1,), int8_ty)
        bn5_tensorLayer3In_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3InC), uint8_ty)
        bn5_tensorLayer2Out_ty = bn5_tensorLayer3In_ty
        bn5_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL5_3InC * tensorL5_3OutC,), int8_ty)
        bn5_tensorLayer3Out_ty = MemRefType.get((tensorL6_1InW, 1, tensorL5_3OutC),int8_ty)
        
        # AIE Core Function declarations
        bn5_conv2dk1_relu_i8_ui8 = external_func("bn5_conv2dk1_relu_i8_ui8",inputs=[bn5_tensorLayer1In_ty, bn5_weightsLayer1_ty, bn5_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn5_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty, bn5_weightsLayer2_ty, bn5_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn5_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty,bn5_tensorLayer2In_ty, bn5_weightsLayer2_ty, bn5_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk1_skip_ui8_i8_i8 = external_func("bn5_conv2dk1_skip_ui8_i8_i8",inputs=[bn5_tensorLayer3In_ty, bn5_weightsLayer3_ty, bn5_tensorLayer3Out_ty, bn5_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn5_conv2dk1_ui8_i8 = external_func("bn5_conv2dk1_ui8_i8",inputs=[bn5_tensorLayer3In_ty, bn5_weightsLayer3_ty, bn5_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        # Compute tile 6
        bn5_objectArchiveName = "bn5_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn5_depthWiseStride, "skip" if (bn5_withSkip) else "")
        bn5_tensorLayer1Out_ty = MemRefType.get((tensorL5_2InW, 1, tensorL5_2InC),uint8_ty)
        bn5_tensorLayer2Out_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3InC),uint8_ty)
        bn5_tensorLayer3Out_ty = MemRefType.get((tensorL5_3InW, 1, tensorL5_3OutC),int8_ty)        

       

        # between compute tiles
        act_bn5_bn6 = object_fifo("act_bn5_bn6", ComputeTile04, ComputeTile12, 2, bn5_tensorLayer3Out_ty)

        bottleneckACore("bn5", ComputeTile04, act_bn4_bn5, bn5_wts_OF_L3L1, act_bn5_bn6, rtpComputeTile04, bn5_objectArchiveName,
                         bn5_conv2dk1_relu_i8_ui8, bn5_conv2dk3_dw_stride1_relu_ui8_ui8, bn5_conv2dk3_dw_stride2_relu_ui8_ui8, bn5_conv2dk1_ui8_i8, bn5_conv2dk1_skip_ui8_i8_i8,
                           bn5_tensorLayer1Out_ty, bn5_tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, bn5_depthWiseStride, bn5_depthWiseChannels, tensorL5_3OutC, bn5_withSkip)


        # # temporary types for tensor to enable intial test
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

        
        # # Compute tile 6
        bn6_objectArchiveName = "bn6_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn6_depthWiseStride, "skip" if (bn6_withSkip) else "")
        bn6_tensorLayer1Out_ty = MemRefType.get((tensorL6_2InW, 1, tensorL6_2InC),uint8_ty)
        bn6_tensorLayer2Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3InC),uint8_ty)
        bn6_tensorLayer3Out_ty = MemRefType.get((tensorL6_3InW, 1, tensorL6_3OutC),int8_ty)

        # between compute tiles
        act_bn6_bn7 = object_fifo("act_bn6_bn7", ComputeTile12, ComputeTile13, 2, bn6_tensorLayer3Out_ty)


        bottleneckACore("bn6", ComputeTile12, act_bn5_bn6, bn6_wts_OF_L3L1, act_bn6_bn7, rtpComputeTile12, bn6_objectArchiveName,
                         bn6_conv2dk1_relu_i8_ui8, bn6_conv2dk3_dw_stride1_relu_ui8_ui8, bn6_conv2dk3_dw_stride2_relu_ui8_ui8, bn6_conv2dk1_ui8_i8, bn6_conv2dk1_skip_ui8_i8_i8,
                           bn6_tensorLayer1Out_ty, bn6_tensorLayer2Out_ty, tensorInW, tensorInH, tensorInC, bn6_depthWiseStride, bn6_depthWiseChannels, tensorL6_3OutC, bn6_withSkip)

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

        # between compute tiles
        act_bn7_bn8 = object_fifo("act_bn7_bn8", ComputeTile13, ComputeTile14, 2, bn7_tensorLayer3Out_ty)

        bottleneckACore("bn7", ComputeTile13, act_bn6_bn7, bn7_wts_OF_L3L1, act_bn7_bn8, rtpComputeTile13, bn7_objectArchiveName, 
                        bn7_conv2dk1_relu_i8_ui8, bn7_conv2dk3_dw_stride1_relu_ui8_ui8, bn7_conv2dk3_dw_stride2_relu_ui8_ui8, bn7_conv2dk1_ui8_i8, bn7_conv2dk1_skip_ui8_i8_i8, 
                        bn7_tensorLayer1Out_ty, bn7_tensorLayer2Out_ty, tensorL7_1InW, tensorL7_1InH, tensorL7_1InC, bn7_depthWiseStride, bn7_depthWiseChannels, tensorL7_3OutC, bn7_withSkip)


        # # ******************************************************************************************************************************        
        bn8_tensorLayer1In_ty = MemRefType.get((tensorL8_1InW, 1, tensorL8_1InC), int8_ty)
        bn8_weightsLayer1_ty = MemRefType.get((1 * 1 * tensorL8_1InC * tensorL8_2InC,), int8_ty)
        bn8_tensorLayer2In_ty = MemRefType.get((tensorL8_2InW, 1, tensorL8_2InC), uint8_ty)
        bn8_tensorLayer1Out_ty = bn8_tensorLayer2In_ty
        bn8_weightsLayer2_ty = MemRefType.get((3 * 3 * tensorL8_3InC * 1,), int8_ty)
        bn8_tensorLayer3In_ty = MemRefType.get((tensorL8_3InW, 1, tensorL8_3InC), uint8_ty)
        bn8_tensorLayer2Out_ty = bn8_tensorLayer3In_ty
        bn8_weightsLayer3_ty = MemRefType.get((1 * 1 * tensorL8_3InC * tensorL8_3OutC,), int8_ty)
        bn8_tensorLayer3Out_ty = MemRefType.get((tensorL8_3InW, 1, tensorL8_3OutC),int8_ty)
        
        # Output
        act_out = object_fifo("act_out", ComputeTile14, ShimTile10, 2, tensorLayerOut_ty)

        # AIE Core Function declarations
        bn8_conv2dk1_relu_i8_ui8 = external_func("bn8_conv2dk1_relu_i8_ui8",inputs=[bn8_tensorLayer1In_ty, bn8_weightsLayer1_ty, bn8_tensorLayer1Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk3_dw_stride2_relu_ui8_ui8 = external_func("bn8_conv2dk3_dw_stride2_relu_ui8_ui8",inputs=[bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty, bn8_weightsLayer2_ty, bn8_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk3_dw_stride1_relu_ui8_ui8 = external_func("bn8_conv2dk3_dw_stride1_relu_ui8_ui8",inputs=[bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty,bn8_tensorLayer2In_ty, bn8_weightsLayer2_ty, bn8_tensorLayer2Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk1_skip_ui8_i8_i8 = external_func("bn8_conv2dk1_skip_ui8_i8_i8",inputs=[bn8_tensorLayer3In_ty, bn8_weightsLayer3_ty, bn8_tensorLayer3Out_ty, bn8_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty, int32_ty])
        bn8_conv2dk1_ui8_i8 = external_func("bn8_conv2dk1_ui8_i8",inputs=[bn8_tensorLayer3In_ty, bn8_weightsLayer3_ty, bn8_tensorLayer3Out_ty, int32_ty, int32_ty, int32_ty, int32_ty])

        bn8_objectArchiveName = "bn8_combined_con2dk1fusedrelu_conv2dk3dwstride%s_conv2dk1%s.a" % (bn8_depthWiseStride, "skip" if (bn8_withSkip) else "")

        bottleneckACore("bn8", ComputeTile14, act_bn7_bn8, bn8_wts_OF_L3L1, act_out, rtpComputeTile14, bn8_objectArchiveName, 
                        bn8_conv2dk1_relu_i8_ui8, bn8_conv2dk3_dw_stride1_relu_ui8_ui8, bn8_conv2dk3_dw_stride2_relu_ui8_ui8, bn8_conv2dk1_ui8_i8, bn8_conv2dk1_skip_ui8_i8_i8, 
                        bn8_tensorLayer1Out_ty, bn8_tensorLayer2Out_ty, tensorL8_1InW, tensorL8_1InH, tensorL8_1InC, bn8_depthWiseStride, bn8_depthWiseChannels, tensorL8_3OutC, bn8_withSkip)


         # instruction stream generation
        activationsInSize32b = (tensorInW * tensorInH * tensorInC) // 4
        activationsOutSize32b = (tensorOutW * tensorOutH * tensorOutC) // 4
        activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
        weightsInL3_ty = MemRefType.get((total_weights,), int32_ty)
        activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

        memtile_01_wts32b=(memtile_01_wts)//4
        memtile_11_wts32b=(memtile_11_wts)//4

        @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
        def sequence(inputFromL3, weightsFromL3, outputToL3):
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=0, value=bn4_scaleFactor1)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=1, value=bn4_scaleFactor2)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=2, value=bn4_scaleFactor3)
            NpuWriteRTPOp("rtp03", col=tileColIndex, row=3, index=3, value=bn4_scaleFactorAdd)

            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=0, value=bn5_scaleFactor1)
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=1, value=bn5_scaleFactor2)
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=2, value=bn5_scaleFactor3)
            NpuWriteRTPOp("rtp04", col=tileColIndex, row=4, index=3, value=bn5_scaleFactorAdd)

            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=0, value=bn6_scaleFactor1)
            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=1, value=bn6_scaleFactor2)
            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=2, value=bn6_scaleFactor3)
            NpuWriteRTPOp("rtp12", col=tileColIndex+1, row=2, index=3, value=bn6_scaleFactorAdd)


            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=0, value=bn7_scaleFactor1)
            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=1, value=bn7_scaleFactor2)
            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=2, value=bn7_scaleFactor3)
            NpuWriteRTPOp("rtp13", col=tileColIndex+1, row=3, index=3, value=bn7_scaleFactorAdd)

            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=0, value=bn8_scaleFactor1)
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=1, value=bn8_scaleFactor2)
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=2, value=bn8_scaleFactor3)
            NpuWriteRTPOp("rtp14", col=tileColIndex+1, row=4, index=3, value=bn8_scaleFactorAdd)
            
            npu_dma_memcpy_nd(
                metadata="act_in",
                bd_id=0,
                mem=inputFromL3,
                sizes=[1, 1, 1, activationsInSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="act_out",
                bd_id=2,
                mem=outputToL3,
                sizes=[1, 1, 1, activationsOutSize32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_OF_01_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                sizes=[1, 1, 1, memtile_01_wts32b],
            )
            npu_dma_memcpy_nd(
                metadata="wts_OF_11_L3L2",
                bd_id=1,
                mem=weightsFromL3,
                offsets=[0, 0, 0, memtile_01_wts32b],
                sizes=[1, 1, 1, memtile_11_wts32b],
            )
            npu_sync(column=1, row=0, direction=0, channel=0)

with mlir_mod_ctx() as ctx:
    mobilenetV3_bn_4_5_6_7_8(tileColIndex = 0, tensorInW = 28, tensorInH = 28, tensorInC = 40, 
                    bn4_depthWiseStride = 1, bn4_depthWiseChannels = 120, bn4_withSkip = True, bn4_tensorOutC = 40, bn4_scaleFactor1 = 8, bn4_scaleFactor2 = 8, bn4_scaleFactor3 = 11,  bn4_scaleFactorAdd = 0,
                    bn5_depthWiseStride = 1, bn5_depthWiseChannels = 120, bn5_withSkip = True, bn5_tensorOutC = 40, bn5_scaleFactor1 = 8, bn5_scaleFactor2 = 8, bn5_scaleFactor3 = 11,  bn5_scaleFactorAdd = 0,
                    bn6_depthWiseStride = 2, bn6_depthWiseChannels = 240, bn6_withSkip = False, bn6_tensorOutC = 80, bn6_scaleFactor1 = 8, bn6_scaleFactor2 = 8, bn6_scaleFactor3 = 11,  bn6_scaleFactorAdd = 0,
                    bn7_depthWiseStride = 1, bn7_depthWiseChannels = 200, bn7_withSkip = True, bn7_tensorOutC = 80, bn7_scaleFactor1 = 9, bn7_scaleFactor2 = 8, bn7_scaleFactor3 = 11, bn7_scaleFactorAdd = 0,
                    bn8_depthWiseStride = 1, bn8_depthWiseChannels = 184,  bn8_withSkip = True, bn8_tensorOutC = 80, bn8_scaleFactor1 = 9, bn8_scaleFactor2 = 8, bn8_scaleFactor3 = 11, bn8_scaleFactorAdd = 0,
                    enableTrace = False, trace_size = 16384, traceSizeInInt32s = 4096)
    
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
