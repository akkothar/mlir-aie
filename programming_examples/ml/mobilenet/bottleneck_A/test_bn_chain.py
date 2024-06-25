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
from dolphin import print_dolphin
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
    Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
vectorSize=8

bneck_6_tensorInW = 28
bneck_6_tensorInH = 28 
bneck_6_tensorInC = 40
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

bneck_7_tensorInW = 14
bneck_7_tensorInH = 14 
bneck_7_tensorInC = 80
bneck_7_tensorOutC = bneck_7_tensorInC

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
bneck_7_OutC3 = bneck_7_InC1


bneck_8_depthwiseStride = 1
bneck_8_depthWiseChannels = 184

bneck_8_InW1 = bneck_7_InW3
bneck_8_InH1 = bneck_7_InH3
bneck_8_InC1 = bneck_7_OutC3
bneck_8_OutC1 = bneck_8_depthWiseChannels

bneck_8_InW2 = bneck_8_InW1
bneck_8_InH2 = bneck_8_InH1
bneck_8_OutC2 = bneck_8_OutC1

bneck_8_InW3 = bneck_8_InW2
bneck_8_InH3 = bneck_8_InH2
bneck_8_OutC3 = bneck_8_InC1




bneck_6_InC1_vec =  math.floor(bneck_6_InC1/vectorSize)
bneck_8_OutC3_vec =  math.floor(bneck_8_OutC3/vectorSize)




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

    shape_total_wts =((bneck_6_InC1*bneck_6_OutC1 + 3*3*bneck_6_OutC2 + bneck_6_OutC2*bneck_6_OutC3)+(bneck_7_InC1*bneck_7_OutC1 + 3*3*bneck_7_OutC2 + bneck_7_OutC2*bneck_7_OutC3)+ (bneck_8_InC1*bneck_8_OutC1 + 3*3*bneck_8_OutC2 + bneck_8_OutC2*bneck_8_OutC3),1)
    print("total weights:::",shape_total_wts)
    shape_in_act = (bneck_6_InH1, bneck_6_InC1_vec, bneck_6_InW1, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (bneck_8_InH3, bneck_8_OutC3_vec, bneck_8_InW3, vectorSize) # HCWC8
    shape_out_final = (bneck_8_OutC3_vec*vectorSize, bneck_8_InH3, bneck_8_InW3) # CHW
    
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, bneck_6_InC1_vec*vectorSize, bneck_6_InH1, bneck_6_InW1)
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
        def __init__(self, in_planes=16, bn6_expand=16,bn6_project=16, bn7_expand=16,bn7_project=16, bn8_expand=16,bn8_project=16):
            super(QuantBottleneckA, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.bn6_quant_conv1 = QuantConv2d(
                in_planes,
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


        def forward(self, x):
            out = self.quant_id_1(x)
            out = self.bn6_quant_conv1(out)
            out = self.bn6_quant_relu1(out)
            out = self.bn6_quant_conv2(out)
            out = self.bn6_quant_relu2(out)
            out = self.bn6_quant_conv3(out)
            out_q = self.bn6_quant_id_2(out)

            # bn7
            out = self.bn7_quant_conv1(out_q)
            out = self.bn7_quant_relu1(out)
            out = self.bn7_quant_conv2(out)
            out = self.bn7_quant_relu2(out)
            out = self.bn7_quant_conv3(out)
            out = self.bn6_quant_id_2(out)
            out = out+out_q
            out_q = self.bn7_add(out)

            # bn8

            out = self.bn8_quant_conv1(out_q)
            out = self.bn8_quant_relu1(out)
            out = self.bn8_quant_conv2(out)
            out = self.bn8_quant_relu2(out)
            out = self.bn8_quant_conv3(out)
            out = self.bn7_add(out)
            out = out+out_q
            out = self.bn8_add(out)

            return out

    quant_bottleneck_model = QuantBottleneckA(in_planes=bneck_6_InC1, bn6_expand=bneck_6_OutC1,bn6_project=bneck_6_OutC3,bn7_expand=bneck_7_OutC1,bn7_project=bneck_7_OutC3, bn8_expand=bneck_8_OutC1,bn8_project=bneck_8_OutC3)
    quant_bottleneck_model.eval()
    
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    init_scale = quant_bottleneck_model.quant_id_1.quant_act_scale()
    block_6_relu_1 = quant_bottleneck_model.bn6_quant_relu1.quant_act_scale()
    block_6_relu_2 = quant_bottleneck_model.bn6_quant_relu2.quant_act_scale()
    block_6_final_scale = quant_bottleneck_model.bn6_quant_id_2.quant_act_scale()

    block_6_weight_scale1 = quant_bottleneck_model.bn6_quant_conv1.quant_weight_scale()
    block_6_weight_scale2 = quant_bottleneck_model.bn6_quant_conv2.quant_weight_scale()
    block_6_weight_scale3 = quant_bottleneck_model.bn6_quant_conv3.quant_weight_scale()
    block_6_combined_scale1 = -torch.log2(
        init_scale * block_6_weight_scale1 / block_6_relu_1
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
    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

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

    total_wts = np.concatenate((bn6_total_wts, bn7_total_wts,bn8_total_wts), axis=None)

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
        print_dolphin()
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
