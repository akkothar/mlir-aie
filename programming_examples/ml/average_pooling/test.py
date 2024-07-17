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
from aie.utils.xrt import setup_aie_single, extract_trace, write_out_trace, execute_single
import aie.utils.test as test_utils
import torch
import torch.nn.functional as F
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
# In this case, we will assume a batch size of 1 for simplicity
from brevitas.quant.fixed_point import (
    Uint8ActPerTensorFixedPoint,
)
from brevitas.nn import QuantConv2d, QuantIdentity, QuantReLU

tensorInC = 256

vectorSize=8
tensorInW = 7
tensorInH = 7

tensorOutC = tensorInC
kdim=7
stride=1

tensorOutW = tensorInW // kdim
tensorOutH = tensorInH // kdim


InC_vec =  math.floor(tensorInC/vectorSize)
OutC_vec =  math.floor(tensorOutC/vectorSize)


def main(opts):
    design = "average_pool"
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

    dtype_out = np.dtype("uint8")
    # Create a random input tensor
    

    shape_in_act = (tensorInH, InC_vec, tensorInW, vectorSize)  #'YCXC8' , 'CYX'

    shape_out = (tensorOutH, OutC_vec, tensorOutW, vectorSize)
    shape_out_final = (OutC_vec*vectorSize, tensorOutH, tensorOutW) # CHW
 
    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie_single(
        xclbin_path=xclbin_path,
        insts_path=insts_path,
        in_0_shape=shape_in_act,
        in_0_dtype=dtype_in,
        out_buf_shape=shape_out,
        out_buf_dtype=dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
    )

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
     # Perform average pooling
    class QuantModel(nn.Module):
        def __init__(self):
            super(QuantModel, self).__init__()
            self.quant_id_1 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )
            self.quant_id_2 = QuantIdentity(
                act_quant=Uint8ActPerTensorFixedPoint,
                bit_width=8,
                return_quant_tensor=True,
            )

        def forward(self, x):
            out = self.quant_id_1(x)
            out=  F.avg_pool2d(out, kdim, stride)
            out = self.quant_id_2(out)
            return out
    # golden_output = F.avg_pool2d(int_inp, kdim, stride)

    # ------------------------------------------------------
    # Reorder input data-layout
    # ------------------------------------------------------
    input = torch.randn(1, tensorInC, tensorInH, tensorInW)
    model = QuantModel()
    model.eval()
    

    q_bottleneck_out = model(input)
    golden_output = q_bottleneck_out.int(float_datatype=True).data.numpy().astype(dtype_out)
    print("Golden::Brevitas::", golden_output)
    q_inp = model.quant_id_1(input)
    int_inp = q_inp.int(float_datatype=True)

    inp_scale1 = model.quant_id_1.quant_act_scale()
    inp_scale2 = model.quant_id_2.quant_act_scale()
    combined_scale1 = -torch.log2(inp_scale1 / inp_scale2)
    

    
    ds = DataShaper()
    before_input = int_inp.squeeze().data.numpy().astype(dtype_in)
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt_1x1.txt", sep=",", format="%d")

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        start = time.time_ns()
        aie_output = execute_single(app, ifm_mem_fmt) 
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
    # print(int_inp)

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
        exit(0)
    else:
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
