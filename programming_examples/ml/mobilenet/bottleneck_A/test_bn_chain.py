#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils
from dolphin import print_dolphin,print_three_dolphins
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
vectorSize=8


tensorInW = 56
tensorInH = 56 
tensorInC = 24

# config for bn2
bn2_depthwiseStride = 1
bn2_depthWiseChannels = 72
bneck_2_OutC=24

# each layer
bneck_2_InW1 = tensorInW
bneck_2_InH1 = tensorInH
bneck_2_InC1 = tensorInC
bneck_2_OutC1 = bn2_depthWiseChannels

bneck_2_InW2 = bneck_2_InW1
bneck_2_InH2 = bneck_2_InH1
bneck_2_OutC2 = bneck_2_OutC1

bneck_2_InW3 = bneck_2_InW2 // bn2_depthwiseStride
bneck_2_InH3 = bneck_2_InH2 // bn2_depthwiseStride
bneck_2_OutC3 = bneck_2_OutC

# config for bn3
bn3_depthwiseStride = 2
bn3_depthWiseChannels = 72
bneck_3_OutC=40

# each layer
bneck_3_InW1 = bneck_2_InW3
bneck_3_InH1 = bneck_2_InH3
bneck_3_InC1 = bneck_2_OutC3
bneck_3_OutC1 = bn3_depthWiseChannels

bneck_3_InW2 = bneck_3_InW1
bneck_3_InH2 = bneck_3_InH1
bneck_3_OutC2 = bneck_3_OutC1

bneck_3_InW3 = bneck_3_InW2 // bn3_depthwiseStride
bneck_3_InH3 = bneck_3_InH2 // bn3_depthwiseStride
bneck_3_OutC3 = bneck_3_OutC


# config for bn5
bn4_depthwiseStride = 1
bn4_depthWiseChannels = 120
bneck_4_OutC=40

# each layer
bneck_4_InW1 = bneck_3_InW3
bneck_4_InH1 = bneck_3_InH3
bneck_4_InC1 = bneck_3_OutC3
bneck_4_OutC1 = bn4_depthWiseChannels

bneck_4_InW2 = bneck_4_InW1
bneck_4_InH2 = bneck_4_InH1
bneck_4_OutC2 = bneck_4_OutC1

bneck_4_InW3 = bneck_4_InW2 // bn4_depthwiseStride
bneck_4_InH3 = bneck_4_InH2 // bn4_depthwiseStride
bneck_4_OutC3 = bneck_4_OutC

# config for bn5
bn5_depthwiseStride = 1
bn5_depthWiseChannels = 120
bneck_5_OutC=40

# each layer
bneck_5_InW1 = bneck_4_InW3
bneck_5_InH1 = bneck_4_InH3
bneck_5_InC1 = bneck_4_OutC3
bneck_5_OutC1 = bn5_depthWiseChannels

bneck_5_InW2 = bneck_5_InW1
bneck_5_InH2 = bneck_5_InH1
bneck_5_OutC2 = bneck_5_OutC1

bneck_5_InW3 = bneck_5_InW2 // bn5_depthwiseStride
bneck_5_InH3 = bneck_5_InH2 // bn5_depthwiseStride
bneck_5_OutC3 = bneck_5_OutC

# config for bn6
bneck_6_tensorInW = bneck_5_InW3
bneck_6_tensorInH = bneck_5_InH3
bneck_6_tensorInC = bneck_5_OutC3
bneck_6_tensorOutC = 80
bn6_depthwiseStride = 2
bn6_depthWiseChannels = 240

bneck_6_InW1 = bneck_6_tensorInW
bneck_6_InH1 = bneck_6_tensorInH
bneck_6_InC1 = bneck_6_tensorInC
bneck_6_OutC1 = bn6_depthWiseChannels

bneck_6_InW2 = bneck_6_InW1 
bneck_6_InH2 = bneck_6_InH1 
bneck_6_OutC2 = bneck_6_OutC1

bneck_6_InW3 = bneck_6_InW2 // bn6_depthwiseStride
bneck_6_InH3 = bneck_6_InH2 // bn6_depthwiseStride
bneck_6_OutC3 = bneck_6_tensorOutC

# config for bn7
bneck_7_tensorInW = bneck_6_InW3
bneck_7_tensorInH = bneck_6_InH3 
bneck_7_tensorInC = bneck_6_OutC3
bneck_7_tensorOutC = 80

bn7_depthwiseStride = 1
bn7_depthWiseChannels = 200

bneck_7_InW1 = bneck_7_tensorInW
bneck_7_InH1 = bneck_7_tensorInH
bneck_7_InC1 = bneck_7_tensorInC
bneck_7_OutC1 = bn7_depthWiseChannels

bneck_7_InW2 = bneck_7_InW1
bneck_7_InH2 = bneck_7_InH1
bneck_7_OutC2 = bneck_7_OutC1

bneck_7_InW3 = bneck_7_InW2
bneck_7_InH3 = bneck_7_InH2
bneck_7_OutC3 = bneck_7_tensorOutC

# config for bn8
bneck_8_tensorInW = bneck_7_InW3
bneck_8_tensorInH = bneck_7_InH3 
bneck_8_tensorInC = bneck_7_OutC3
bneck_8_tensorOutC = 80
bneck_8_depthwiseStride = 1
bneck_8_depthWiseChannels = 184

bneck_8_InW1 = bneck_8_tensorInW
bneck_8_InH1 = bneck_8_tensorInH
bneck_8_InC1 = bneck_8_tensorInC
bneck_8_OutC1 = bneck_8_depthWiseChannels

bneck_8_InW2 = bneck_8_InW1
bneck_8_InH2 = bneck_8_InH1
bneck_8_OutC2 = bneck_8_OutC1

bneck_8_InW3 = bneck_8_InW2
bneck_8_InH3 = bneck_8_InH2
bneck_8_OutC3 = bneck_8_tensorOutC


# config for bn8
bneck_9_tensorInW = bneck_8_InW3
bneck_9_tensorInH = bneck_8_InH3 
bneck_9_tensorInC = bneck_8_OutC3
bneck_9_tensorOutC = 80
bneck_9_depthwiseStride = 1
bneck_9_depthWiseChannels = 184

bneck_9_InW1 = bneck_9_tensorInW
bneck_9_InH1 = bneck_9_tensorInH
bneck_9_InC1 = bneck_9_tensorInC
bneck_9_OutC1 = bneck_9_depthWiseChannels

bneck_9_InW2 = bneck_9_InW1
bneck_9_InH2 = bneck_9_InH1
bneck_9_OutC2 = bneck_9_OutC1

bneck_9_InW3 = bneck_9_InW2
bneck_9_InH3 = bneck_9_InH2
bneck_9_OutC3 = bneck_9_tensorOutC

tensorOutW = bneck_9_InW3 
tensorOutH = bneck_9_InH3
tensorOutC = bneck_9_OutC3


InC_vec =  math.floor(tensorInC/vectorSize)
OutC_vec =  math.floor(tensorOutC/vectorSize)


def main(opts):
    design = "mobilenet_bottleneck_A_bn3"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = 16384
    enable_trace = False
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts =((bneck_2_InC1*bneck_2_OutC1 + 3*3*bneck_2_OutC2 + bneck_2_OutC2*bneck_2_OutC3)+
                      (bneck_3_InC1*bneck_3_OutC1 + 3*3*bneck_3_OutC2 + bneck_3_OutC2*bneck_3_OutC3)+
                      (bneck_4_InC1*bneck_4_OutC1 + 3*3*bneck_4_OutC2 + bneck_4_OutC2*bneck_4_OutC3)+
                      (bneck_5_InC1*bneck_5_OutC1 + 3*3*bneck_5_OutC2 + bneck_5_OutC2*bneck_5_OutC3)+
                      (bneck_6_InC1*bneck_6_OutC1 + 3*3*bneck_6_OutC2 + bneck_6_OutC2*bneck_6_OutC3)+
                      (bneck_7_InC1*bneck_7_OutC1 + 3*3*bneck_7_OutC2 + bneck_7_OutC2*bneck_7_OutC3)+
                      (bneck_8_InC1*bneck_8_OutC1 + 3*3*bneck_8_OutC2 + bneck_8_OutC2*bneck_8_OutC3)+
                      (bneck_9_InC1*bneck_9_OutC1 + 3*3*bneck_9_OutC2 + bneck_9_OutC2*bneck_9_OutC3),1)
    
    print("total weights:::",shape_total_wts)
    shape_in_act = (tensorInH, InC_vec, tensorInW, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (tensorOutH, OutC_vec, tensorOutW, vectorSize) # HCWC8
    shape_out_final = (OutC_vec*vectorSize, tensorOutH, tensorOutW) # CHW
    
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, InC_vec*vectorSize, tensorInH, tensorInW)
    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in_act,
        dtype_in,
        shape_total_wts,
        dtype_wts,
        shape_out,
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )
    class QuantBottleneckA(nn.Module):
        def __init__(self, in_planes=16,
                     bn2_expand=16,bn2_project=16,
                     bn3_expand=16,bn3_project=16,
                     bn4_expand=16,bn4_project=16, 
                     bn5_expand=16,bn5_project=16, 
                     bn6_expand=16,bn6_project=16, 
                     bn7_expand=16,bn7_project=16, 
                     bn8_expand=16,bn8_project=16,
                     bn9_expand=16,bn9_project=16):
            super(QuantBottleneckA, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            # bn2
            self.bn2_quant_conv1 = QuantConv2d(
                in_planes,
                bn2_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn2_quant_conv2 = QuantConv2d(
                bn2_expand,
                bn2_expand,
                kernel_size=3,
                stride=bn2_depthWiseChannels,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn2_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn2_quant_conv3 = QuantConv2d(
                bn2_expand,
                bn2_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn2_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn2_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn2_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# bn3
            self.bn3_quant_conv1 = QuantConv2d(
                bn2_project,
                bn3_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn3_quant_conv2 = QuantConv2d(
                bn3_expand,
                bn3_expand,
                kernel_size=3,
                stride=bn3_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn3_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn3_quant_conv3 = QuantConv2d(
                bn3_expand,
                bn3_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn3_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn3_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn3_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# 
            self.bn4_quant_conv1 = QuantConv2d(
                bn3_project,
                bn4_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn4_quant_conv2 = QuantConv2d(
                bn4_expand,
                bn4_expand,
                kernel_size=3,
                stride=bn4_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn4_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn4_quant_conv3 = QuantConv2d(
                bn4_expand,
                bn4_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn4_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn4_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn4_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
# 
            self.bn5_quant_conv1 = QuantConv2d(
                bn4_project,
                bn5_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn5_quant_conv2 = QuantConv2d(
                bn5_expand,
                bn5_expand,
                kernel_size=3,
                stride=bn5_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn5_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn5_quant_conv3 = QuantConv2d(
                bn5_expand,
                bn5_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn5_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn5_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn5_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn6_quant_conv1 = QuantConv2d(
                bn5_project,
                bn6_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn6_quant_conv2 = QuantConv2d(
                bn6_expand,
                bn6_expand,
                kernel_size=3,
                stride=bn6_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn6_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn6_quant_conv3 = QuantConv2d(
                bn6_expand,
                bn6_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn6_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn6_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn6_quant_id_2 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            # bn7
            self.bn7_quant_conv1 = QuantConv2d(
                bn6_project,
                bn7_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn7_quant_conv2 = QuantConv2d(
                bn7_expand,
                bn7_expand,
                kernel_size=3,
                stride=bn7_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn7_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn7_quant_conv3 = QuantConv2d(
                bn7_expand,
                bn7_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn7_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn7_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn7_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            # bn8
            self.bn8_quant_conv1 = QuantConv2d(
                bn7_project,
                bn8_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn8_quant_conv2 = QuantConv2d(
                bn8_expand,
                bn8_expand,
                kernel_size=3,
                stride=bneck_8_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn8_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn8_quant_conv3 = QuantConv2d(
                bn8_expand,
                bn8_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn8_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn8_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn8_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            # bn9
            self.bn9_quant_conv1 = QuantConv2d(
                bn8_project,
                bn9_expand,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn9_quant_conv2 = QuantConv2d(
                bn9_expand,
                bn9_expand,
                kernel_size=3,
                stride=bneck_8_depthwiseStride,
                padding=1,
                padding_mode="zeros",
                bit_width=8,
                groups=bn9_expand,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn9_quant_conv3 = QuantConv2d(
                bn9_expand,
                bn9_project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            self.bn9_quant_relu1 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn9_quant_relu2 = QuantReLU(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.bn9_add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )


        def forward(self, x):
            out_q = self.quant_id_1(x)

            out = self.bn2_quant_conv1(out_q)
            out = self.bn2_quant_relu1(out)
            out = self.bn2_quant_conv2(out)
            out = self.bn2_quant_relu2(out)
            out = self.bn2_quant_conv3(out)
            out = self.quant_id_1(out)
            out = out+out_q
            out_q = self.bn2_add(out)

            # bn3
            out = self.bn3_quant_conv1(out)
            out = self.bn3_quant_relu1(out)
            out = self.bn3_quant_conv2(out)
            out = self.bn3_quant_relu2(out)
            out = self.bn3_quant_conv3(out)
            out_q = self.bn3_quant_id_2(out)

            # # bn4
            out = self.bn4_quant_conv1(out_q)
            out = self.bn4_quant_relu1(out)
            out = self.bn4_quant_conv2(out)
            out = self.bn4_quant_relu2(out)
            out = self.bn4_quant_conv3(out)
            out = self.bn3_quant_id_2(out)
            out = out+out_q
            out_q = self.bn4_add(out)

            # # # bn5
            out = self.bn5_quant_conv1(out_q)
            out = self.bn5_quant_relu1(out)
            out = self.bn5_quant_conv2(out)
            out = self.bn5_quant_relu2(out)
            out = self.bn5_quant_conv3(out)
            out = self.bn4_add(out)
            out = out+out_q
            out = self.bn5_add(out)
            
            # bn6
            out = self.bn6_quant_conv1(out)
            out = self.bn6_quant_relu1(out)
            out = self.bn6_quant_conv2(out)
            out = self.bn6_quant_relu2(out)
            out = self.bn6_quant_conv3(out)
            out_q = self.bn6_quant_id_2(out)

            
            # # bn7
            out = self.bn7_quant_conv1(out_q)
            out = self.bn7_quant_relu1(out)
            out = self.bn7_quant_conv2(out)
            out = self.bn7_quant_relu2(out)
            out = self.bn7_quant_conv3(out)
            out = self.bn6_quant_id_2(out)
            out = out+out_q
            out_q = self.bn7_add(out)

            # # bn8

            out = self.bn8_quant_conv1(out_q)
            out = self.bn8_quant_relu1(out)
            out = self.bn8_quant_conv2(out)
            out = self.bn8_quant_relu2(out)
            out = self.bn8_quant_conv3(out)
            out = self.bn7_add(out)
            out = out+out_q
            out_q = self.bn8_add(out)

            # # bn9

            out = self.bn9_quant_conv1(out_q)
            out = self.bn9_quant_relu1(out)
            out = self.bn9_quant_conv2(out)
            out = self.bn9_quant_relu2(out)
            out = self.bn9_quant_conv3(out)
            out = self.bn8_add(out)
            out = out+out_q
            out_q = self.bn9_add(out)
            return out_q

    quant_bottleneck_model = QuantBottleneckA(in_planes=tensorInC, 
                                            bn2_expand=bneck_2_OutC1,bn2_project=bneck_2_OutC3,
                                            bn3_expand=bneck_3_OutC1,bn3_project=bneck_3_OutC3, 
                                            bn4_expand=bneck_4_OutC1,bn4_project=bneck_4_OutC3, 
                                            bn5_expand=bneck_5_OutC1,bn5_project=bneck_5_OutC3, 
                                            bn6_expand=bneck_6_OutC1,bn6_project=bneck_6_OutC3,
                                            bn7_expand=bneck_7_OutC1,bn7_project=bneck_7_OutC3, 
                                            bn8_expand=bneck_8_OutC1,bn8_project=bneck_8_OutC3,
                                            bn9_expand=bneck_9_OutC1,bn9_project=bneck_9_OutC3)
    quant_bottleneck_model.eval()
    
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    block_2_inp_scale1= quant_bottleneck_model.quant_id_1.quant_act_scale()

    block_2_relu_1 = quant_bottleneck_model.bn2_quant_relu1.quant_act_scale()
    block_2_relu_2 = quant_bottleneck_model.bn2_quant_relu2.quant_act_scale()
    block_2_skip_add = quant_bottleneck_model.bn2_add.quant_act_scale()

    block_2_weight_scale1 = quant_bottleneck_model.bn2_quant_conv1.quant_weight_scale()
    block_2_weight_scale2 = quant_bottleneck_model.bn2_quant_conv2.quant_weight_scale()
    block_2_weight_scale3 = quant_bottleneck_model.bn2_quant_conv3.quant_weight_scale()
    block_2_combined_scale1 = -torch.log2(
        block_2_inp_scale1 * block_2_weight_scale1 / block_2_relu_1
    )
    block_2_combined_scale2 = -torch.log2(
        block_2_relu_1 * block_2_weight_scale2 / block_2_relu_2
    )  
    block_2_combined_scale3 = -torch.log2(
        block_2_relu_2 * block_2_weight_scale3/block_2_inp_scale1
    )   
    block_2_combined_scale_skip = -torch.log2(
        block_2_inp_scale1 / block_2_skip_add
    )  # After addition | clip -128-->127



    print("********************BN2*******************************")
    print("combined_scale after conv1x1:", block_2_combined_scale1.item())
    print("combined_scale after conv3x3:", block_2_combined_scale2.item())
    print("combined_scale after conv1x1:", block_2_combined_scale3.item())
    print("combined_scale after skip add:", block_2_combined_scale_skip.item())
    print("********************BN2*******************************")
    
    init_scale = block_2_skip_add
    block_3_relu_1 = quant_bottleneck_model.bn3_quant_relu1.quant_act_scale()
    block_3_relu_2 = quant_bottleneck_model.bn3_quant_relu2.quant_act_scale()
    block_3_final_scale = quant_bottleneck_model.bn3_quant_id_2.quant_act_scale()

    block_3_weight_scale1 = quant_bottleneck_model.bn3_quant_conv1.quant_weight_scale()
    block_3_weight_scale2 = quant_bottleneck_model.bn3_quant_conv2.quant_weight_scale()
    block_3_weight_scale3 = quant_bottleneck_model.bn3_quant_conv3.quant_weight_scale()
    block_3_combined_scale1 = -torch.log2(
        init_scale * block_3_weight_scale1 / block_3_relu_1
    )
    block_3_combined_scale2 = -torch.log2(
        block_3_relu_1 * block_3_weight_scale2 / block_3_relu_2
    )  
    block_3_combined_scale3 = -torch.log2(
        block_3_relu_2 * block_3_weight_scale3/block_3_final_scale
    )   

    print("********************bn3*******************************")
    print("combined_scale after conv1x1:", block_3_combined_scale1.item())
    print("combined_scale after conv3x3:", block_3_combined_scale2.item())
    print("combined_scale after conv1x1:", block_3_combined_scale3.item())
    print("********************bn3*******************************")


    block_4_inp_scale1= block_3_final_scale
    block_4_relu_1 = quant_bottleneck_model.bn4_quant_relu1.quant_act_scale()
    block_4_relu_2 = quant_bottleneck_model.bn4_quant_relu2.quant_act_scale()
    block_4_skip_add = quant_bottleneck_model.bn4_add.quant_act_scale()

    block_4_weight_scale1 = quant_bottleneck_model.bn4_quant_conv1.quant_weight_scale()
    block_4_weight_scale2 = quant_bottleneck_model.bn4_quant_conv2.quant_weight_scale()
    block_4_weight_scale3 = quant_bottleneck_model.bn4_quant_conv3.quant_weight_scale()
    block_4_combined_scale1 = -torch.log2(
        block_4_inp_scale1 * block_4_weight_scale1 / block_4_relu_1
    )
    block_4_combined_scale2 = -torch.log2(
        block_4_relu_1 * block_4_weight_scale2 / block_4_relu_2
    )  
    block_4_combined_scale3 = -torch.log2(
        block_4_relu_2 * block_4_weight_scale3/block_4_inp_scale1
    )   
    block_4_combined_scale_skip = -torch.log2(
        block_4_inp_scale1 / block_4_skip_add
    )  # After addition | clip -128-->127



    print("********************bn4*******************************")
    print("combined_scale after conv1x1:", block_4_combined_scale1.item())
    print("combined_scale after conv3x3:", block_4_combined_scale2.item())
    print("combined_scale after conv1x1:", block_4_combined_scale3.item())
    print("combined_scale after skip add:", block_4_combined_scale_skip.item())
    print("********************bn4*******************************")



    block_5_inp_scale1= block_4_skip_add
    block_5_relu_1 = quant_bottleneck_model.bn5_quant_relu1.quant_act_scale()
    block_5_relu_2 = quant_bottleneck_model.bn5_quant_relu2.quant_act_scale()
    block_5_skip_add = quant_bottleneck_model.bn5_add.quant_act_scale()

    block_5_weight_scale1 = quant_bottleneck_model.bn5_quant_conv1.quant_weight_scale()
    block_5_weight_scale2 = quant_bottleneck_model.bn5_quant_conv2.quant_weight_scale()
    block_5_weight_scale3 = quant_bottleneck_model.bn5_quant_conv3.quant_weight_scale()
    block_5_combined_scale1 = -torch.log2(
        block_5_inp_scale1 * block_5_weight_scale1 / block_5_relu_1
    )
    block_5_combined_scale2 = -torch.log2(
        block_5_relu_1 * block_5_weight_scale2 / block_5_relu_2
    )  
    block_5_combined_scale3 = -torch.log2(
        block_5_relu_2 * block_5_weight_scale3/block_5_inp_scale1
    )   
    block_5_combined_scale_skip = -torch.log2(
        block_5_inp_scale1 / block_5_skip_add
    )  # After addition | clip -128-->127



    print("********************bn5*******************************")
    print("combined_scale after conv1x1:", block_5_combined_scale1.item())
    print("combined_scale after conv3x3:", block_5_combined_scale2.item())
    print("combined_scale after conv1x1:", block_5_combined_scale3.item())
    print("combined_scale after skip add:", block_5_combined_scale_skip.item())
    print("********************bn5*******************************")




    block_6_relu_1 = quant_bottleneck_model.bn6_quant_relu1.quant_act_scale()
    block_6_relu_2 = quant_bottleneck_model.bn6_quant_relu2.quant_act_scale()
    block_6_final_scale = quant_bottleneck_model.bn6_quant_id_2.quant_act_scale()

    block_6_weight_scale1 = quant_bottleneck_model.bn6_quant_conv1.quant_weight_scale()
    block_6_weight_scale2 = quant_bottleneck_model.bn6_quant_conv2.quant_weight_scale()
    block_6_weight_scale3 = quant_bottleneck_model.bn6_quant_conv3.quant_weight_scale()
    block_6_combined_scale1 = -torch.log2(
        block_5_skip_add * block_6_weight_scale1 / block_6_relu_1
    )
    block_6_combined_scale2 = -torch.log2(
        block_6_relu_1 * block_6_weight_scale2 / block_6_relu_2
    )  
    block_6_combined_scale3 = -torch.log2(
        block_6_relu_2 * block_6_weight_scale3/block_6_final_scale
    )   

    print("********************BN6*******************************")
    print("combined_scale after conv1x1:", block_6_combined_scale1.item())
    print("combined_scale after conv3x3:", block_6_combined_scale2.item())
    print("combined_scale after conv1x1:", block_6_combined_scale3.item())
    print("********************BN6*******************************")

    block_7_inp_scale1= block_6_final_scale

    block_7_relu_1 = quant_bottleneck_model.bn7_quant_relu1.quant_act_scale()
    block_7_relu_2 = quant_bottleneck_model.bn7_quant_relu2.quant_act_scale()
    block_7_skip_add = quant_bottleneck_model.bn7_add.quant_act_scale()

    block_7_weight_scale1 = quant_bottleneck_model.bn7_quant_conv1.quant_weight_scale()
    block_7_weight_scale2 = quant_bottleneck_model.bn7_quant_conv2.quant_weight_scale()
    block_7_weight_scale3 = quant_bottleneck_model.bn7_quant_conv3.quant_weight_scale()
    block_7_combined_scale1 = -torch.log2(
        block_7_inp_scale1 * block_7_weight_scale1 / block_7_relu_1
    )
    block_7_combined_scale2 = -torch.log2(
        block_7_relu_1 * block_7_weight_scale2 / block_7_relu_2
    )  
    block_7_combined_scale3 = -torch.log2(
        block_7_relu_2 * block_7_weight_scale3/block_7_inp_scale1
    )   
    block_7_combined_scale_skip = -torch.log2(
        block_7_inp_scale1 / block_7_skip_add
    )  # After addition | clip -128-->127

    print("********************BN7*******************************")
    print("combined_scale after conv1x1:", block_7_combined_scale1.item())
    print("combined_scale after conv3x3:", block_7_combined_scale2.item())
    print("combined_scale after conv1x1:", block_7_combined_scale3.item())
    print("combined_scale after skip add:", block_7_combined_scale_skip.item())
    print("********************BN7*******************************")

    block_8_inp_scale1= block_7_skip_add
    block_8_relu_1 = quant_bottleneck_model.bn8_quant_relu1.quant_act_scale()
    block_8_relu_2 = quant_bottleneck_model.bn8_quant_relu2.quant_act_scale()
    block_8_skip_add = quant_bottleneck_model.bn8_add.quant_act_scale()
    block_8_weight_scale1 = quant_bottleneck_model.bn8_quant_conv1.quant_weight_scale()
    block_8_weight_scale2 = quant_bottleneck_model.bn8_quant_conv2.quant_weight_scale()
    block_8_weight_scale3 = quant_bottleneck_model.bn8_quant_conv3.quant_weight_scale()
    block_8_combined_scale1 = -torch.log2(
        block_8_inp_scale1 * block_8_weight_scale1 / block_8_relu_1
    )
    block_8_combined_scale2 = -torch.log2(
        block_8_relu_1 * block_8_weight_scale2 / block_8_relu_2
    )  
    block_8_combined_scale3 = -torch.log2(
        block_8_relu_2 * block_8_weight_scale3/block_8_inp_scale1
    )   
    block_8_combined_scale_skip = -torch.log2(
        block_8_inp_scale1 / block_8_skip_add
    )  # After addition | clip -128-->127

    print("********************BN8*******************************")
    print("combined_scale after conv1x1:", block_8_combined_scale1.item())
    print("combined_scale after conv3x3:", block_8_combined_scale2.item())
    print("combined_scale after conv1x1:", block_8_combined_scale3.item())
    print("combined_scale after skip add:", block_8_combined_scale_skip.item())
    print("********************BN8*******************************")

    block_9_inp_scale1= block_8_skip_add
    block_9_relu_1 = quant_bottleneck_model.bn9_quant_relu1.quant_act_scale()
    block_9_relu_2 = quant_bottleneck_model.bn9_quant_relu2.quant_act_scale()
    block_9_skip_add = quant_bottleneck_model.bn9_add.quant_act_scale()
    block_9_weight_scale1 = quant_bottleneck_model.bn9_quant_conv1.quant_weight_scale()
    block_9_weight_scale2 = quant_bottleneck_model.bn9_quant_conv2.quant_weight_scale()
    block_9_weight_scale3 = quant_bottleneck_model.bn9_quant_conv3.quant_weight_scale()
    block_9_combined_scale1 = -torch.log2(
        block_9_inp_scale1 * block_9_weight_scale1 / block_9_relu_1
    )
    block_9_combined_scale2 = -torch.log2(
        block_9_relu_1 * block_9_weight_scale2 / block_9_relu_2
    )  
    block_9_combined_scale3 = -torch.log2(
        block_9_relu_2 * block_9_weight_scale3/block_9_inp_scale1
    )   
    block_9_combined_scale_skip = -torch.log2(
        block_9_inp_scale1 / block_9_skip_add
    )  # After addition | clip -128-->127

    print("********************BN9*******************************")
    print("combined_scale after conv1x1:", block_9_combined_scale1.item())
    print("combined_scale after conv3x3:", block_9_combined_scale2.item())
    print("combined_scale after conv1x1:", block_9_combined_scale3.item())
    print("combined_scale after skip add:", block_9_combined_scale_skip.item())
    print("********************BN9*******************************")
    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    block_2_int_weight_1 = quant_bottleneck_model.bn2_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_2_int_weight_2 = quant_bottleneck_model.bn2_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_2_int_weight_3 = quant_bottleneck_model.bn2_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_3_int_weight_1 = quant_bottleneck_model.bn3_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_3_int_weight_2 = quant_bottleneck_model.bn3_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_3_int_weight_3 = quant_bottleneck_model.bn3_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_4_int_weight_1 = quant_bottleneck_model.bn4_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_4_int_weight_2 = quant_bottleneck_model.bn4_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_4_int_weight_3 = quant_bottleneck_model.bn4_quant_conv3.quant_weight().int(
        float_datatype=True
    )


    block_5_int_weight_1 = quant_bottleneck_model.bn5_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_5_int_weight_2 = quant_bottleneck_model.bn5_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_5_int_weight_3 = quant_bottleneck_model.bn5_quant_conv3.quant_weight().int(
        float_datatype=True
    )


    block_6_int_weight_1 = quant_bottleneck_model.bn6_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_6_int_weight_2 = quant_bottleneck_model.bn6_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_6_int_weight_3 = quant_bottleneck_model.bn6_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_7_int_weight_1 = quant_bottleneck_model.bn7_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_7_int_weight_2 = quant_bottleneck_model.bn7_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_7_int_weight_3 = quant_bottleneck_model.bn7_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_8_int_weight_1 = quant_bottleneck_model.bn8_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_8_int_weight_2 = quant_bottleneck_model.bn8_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_8_int_weight_3 = quant_bottleneck_model.bn8_quant_conv3.quant_weight().int(
        float_datatype=True
    )

    block_9_int_weight_1 = quant_bottleneck_model.bn9_quant_conv1.quant_weight().int(
        float_datatype=True
    )
    block_9_int_weight_2 = quant_bottleneck_model.bn9_quant_conv2.quant_weight().int(
        float_datatype=True
    )
    block_9_int_weight_3 = quant_bottleneck_model.bn9_quant_conv3.quant_weight().int(
        float_datatype=True
    )


  
    golden_output.tofile(
        log_folder + "/golden_output.txt", sep=",", format="%d"
    )
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt.txt", sep=",", format="%d")
    # **************************** bn2 ****************************
    bn2_wts1 = ds.reorder_mat(
        block_2_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn2_wts2 = ds.reorder_mat(
        block_2_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn2_wts3 = ds.reorder_mat(
        block_2_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn2_total_wts = np.concatenate((bn2_wts1, bn2_wts2, bn2_wts3), axis=None)
    # **************************** bn3 ****************************
    bn3_wts1 = ds.reorder_mat(
        block_3_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn3_wts2 = ds.reorder_mat(
        block_3_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn3_wts3 = ds.reorder_mat(
        block_3_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn3_total_wts = np.concatenate((bn3_wts1, bn3_wts2, bn3_wts3), axis=None)

    # **************************** bn4 ****************************
    bn4_wts1 = ds.reorder_mat(
        block_4_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn4_wts2 = ds.reorder_mat(
        block_4_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn4_wts3 = ds.reorder_mat(
        block_4_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn4_total_wts = np.concatenate((bn4_wts1, bn4_wts2, bn4_wts3), axis=None)


    # **************************** bn5 ****************************
    bn5_wts1 = ds.reorder_mat(
        block_5_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn5_wts2 = ds.reorder_mat(
        block_5_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn5_wts3 = ds.reorder_mat(
        block_5_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn5_total_wts = np.concatenate((bn5_wts1, bn5_wts2, bn5_wts3), axis=None)


    # **************************** bn6 ****************************
    bn6_wts1 = ds.reorder_mat(
        block_6_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn6_wts2 = ds.reorder_mat(
        block_6_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn6_wts3 = ds.reorder_mat(
        block_6_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn6_total_wts = np.concatenate((bn6_wts1, bn6_wts2, bn6_wts3), axis=None)
    # **************************** bn7 ****************************
    bn7_wts1 = ds.reorder_mat(
        block_7_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn7_wts2 = ds.reorder_mat(
        block_7_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn7_wts3 = ds.reorder_mat(
        block_7_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn7_total_wts = np.concatenate((bn7_wts1, bn7_wts2, bn7_wts3), axis=None)

     # **************************** bn8 ****************************
    bn8_wts1 = ds.reorder_mat(
        block_8_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn8_wts2 = ds.reorder_mat(
        block_8_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn8_wts3 = ds.reorder_mat(
        block_8_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn8_total_wts = np.concatenate((bn8_wts1, bn8_wts2, bn8_wts3), axis=None)

     # **************************** bn9 ****************************
    bn9_wts1 = ds.reorder_mat(
        block_9_int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    bn9_wts2 = ds.reorder_mat(
        block_9_int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    )
    bn9_wts3 = ds.reorder_mat(
        block_9_int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    bn9_total_wts = np.concatenate((bn9_wts1, bn9_wts2, bn9_wts3), axis=None)

    total_wts = np.concatenate((bn2_total_wts,bn3_total_wts,bn4_total_wts,bn5_total_wts,bn6_total_wts, bn7_total_wts,bn8_total_wts,bn9_total_wts), axis=None)

    total_wts.tofile(log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d")
    # print("{}+{}+{}".format(bn6_wts1.shape, bn6_wts2.shape, bn6_wts3.shape))
    print(shape_total_wts)
    print(total_wts.shape)
    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute(app, ifm_mem_fmt, total_wts) 
        stop = time.time_ns()
        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout
    # ------------------------------------------------------
    temp_out = aie_output.reshape(shape_out)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(shape_out_final)
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0)
    print(ofm_mem_fmt_out)
    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------
    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    if np.allclose(
        ofm_mem_fmt_out,
        golden_output,
        rtol=0,
        atol=2,
    ):
        print("\nPASS!\n")
        print_three_dolphins()
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)