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


InW2 = 7
InH2 = 7
# OutC2 = OutC1
OutC2 = 64

OutC3 = 64

InC_vec =  math.floor(OutC2/vectorSize)
OutC_vec =  math.floor(OutC3/vectorSize)

wts_size=OutC2*OutC3
def main(opts):
    design = "mobilenet_bottleneck_C"
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
    dtype_in = np.dtype("uint8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    shape_total_wts = (wts_size, 1)
    shape_in_act = (InH2, InC_vec, InW2, vectorSize)  #'YCXC8' , 'CYX'
    shape_out = (InH2, OutC_vec, InW2, vectorSize) #bneck_12_OutC3/8
    shape_out_final = (OutC_vec*vectorSize, InH2, InW2) #bneck_12_OutC3/8
    
    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    input = torch.randn(1, InC_vec*vectorSize, InH2, InW2)
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
    class QuantBottleneck(nn.Module):
        def __init__(self, in_planes=16, expand=16,project=16,bn14_expand=16,bn11_project=16,bn12_expand=16,bn12_project=16):
            super(QuantBottleneck, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.quant_conv3 = QuantConv2d(
                expand,
                project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            
            self.add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            # out = self.quant_conv1(out_q)
            # out = self.quant_relu1(out)
            # out = self.quant_conv2(out)
            # out = self.quant_relu2(out)
            out = self.quant_conv3(out_q)
            # out = self.quant_id_1(out)
            # out=out+out_q
            out = self.add(out)
            return out

    class QuantBottleneck_HALF(nn.Module):
        def __init__(self, in_planes=16, expand=16,project=16):
            super(QuantBottleneck_HALF, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

            self.quant_conv3 = QuantConv2d(
                expand,
                project,
                kernel_size=1,
                bit_width=8,
                weight_bit_width=8,
                bias=False,
                weight_quant=Int8WeightPerTensorFixedPoint,
                return_quant_tensor=True,
            )
            
            self.add = QuantIdentity(
                act_quant=Int8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out_q = self.quant_id_1(x)
            # out = self.quant_conv1(out_q)
            # out = self.quant_relu1(out)
            # out = self.quant_conv2(out)
            # out = self.quant_relu2(out)
            out = self.quant_conv3(out_q)
            # out = self.quant_id_1(out)
            # out=out+out_q
            out = self.add(out)
            return out

    quant_bottleneck_model = QuantBottleneck(expand=OutC2,project=OutC3)
    
    quant_bottleneck_model.eval()
    
    
    q_bottleneck_out = quant_bottleneck_model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    print(input.shape)
  
    # extract int input
    q_inp = quant_bottleneck_model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)


 
    inp_scale1= quant_bottleneck_model.quant_id_1.quant_act_scale()
    skip_add = quant_bottleneck_model.add.quant_act_scale()
    weight_scale3 = quant_bottleneck_model.quant_conv3.quant_weight_scale()
    combined_scale3 = -torch.log2(
        inp_scale1 * weight_scale3/skip_add
    )   
    
    print("********************BN13*******************************")
    print("combined_scale after conv1x1:", combined_scale3.item())
    print("**************************************************")

    quant_bottleneck_model_HALF = QuantBottleneck_HALF(expand=OutC2//2,project=OutC3)
    quant_bottleneck_model_HALF.eval()
    # q_bottleneck_out_HALF = quant_bottleneck_model_HALF(input[:,0:32,:,:])
    # golden_output_HALF = q_bottleneck_out_HALF.int(float_datatype=True).data.numpy().astype(dtype_out)
    # print("Golden_HALF::Brevitas::", golden_output_HALF)

    # inp_scale1_HALF= quant_bottleneck_model_HALF.quant_id_1.quant_act_scale()
    # skip_add_HALF = quant_bottleneck_model_HALF.add.quant_act_scale()
    # weight_scale3_HALF = quant_bottleneck_model_HALF.quant_conv3.quant_weight_scale()
    # combined_scale3_HALF = -torch.log2(
    #     inp_scale1_HALF * weight_scale3_HALF/skip_add_HALF
    # )   
    
    # print("********************BN13*******************************")
    # print("combined_scale_HALF after conv1x1:", combined_scale3_HALF.item())
    # print("*************************************************")

    # print("combined_scale after conv1x1:", ( block_0_relu_2 * block_0_weight_scale3).item())
    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------

 
    int_weight_3 = quant_bottleneck_model.quant_conv3.quant_weight().int(
        float_datatype=True
    )
    int_weight_3_HALF = quant_bottleneck_model_HALF.quant_conv3.quant_weight().int(
        float_datatype=True
    )

    int_weight_3_put=int_weight_3[:,0:32,:,:]
    int_weight_3_get=int_weight_3[:,32:64,:,:]
    print(int_weight_3.shape)
    print(int_weight_3_HALF.shape)
    print(int_weight_3[:,0:32,:,:].shape)
    print(int_weight_3[:,32:64,:,:].shape)

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

    # **************************** bn13 ****************************
    # wts1 = ds.reorder_mat(
    #     int_weight_1.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    # wts2 = ds.reorder_mat(
    #     int_weight_2.data.numpy().astype(dtype_wts), "OIYXI1O8", "OIYX"
    # )
    wts3 = ds.reorder_mat(
        int_weight_3.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    wts3_put = ds.reorder_mat(
        int_weight_3_put.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )
    
    wts3_get = ds.reorder_mat(
        int_weight_3_get.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    )

    # wts3_put_HALF = ds.reorder_mat(
    #     int_weight_3_HALF.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX"
    # )
    total_wts = np.concatenate((wts3_put,wts3_get), axis=None)

    total_wts.tofile(log_folder + "/after_weights_mem_fmt_final.txt", sep=",", format="%d")
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
        atol=1,
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