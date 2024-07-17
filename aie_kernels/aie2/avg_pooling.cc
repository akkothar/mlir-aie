//===- conv2dk3.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>


//*****************************************************************************
// softmax for stride 1
// act: int8, wts: int8, out: uint8
//*****************************************************************************

 void average_pooling_scalar(uint8_t *input, uint8_t *output, 
                            int width, int height,int channels, int kernelSize) {
  event0();
  int stride = 1;
  int kernelArea = kernelSize * kernelSize;
    int outputHeight = 1;
    int outputWidth = 1;


    // Iterate over channels and c8
    for (int c = 0; c < channels/8; ++c) {
        for (int c8 = 0; c8 < 8; ++c8) {
            float sum = 0.0;
            // Sum all elements within the kernel for the given channel and sub-channel
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                  // YCXC8
                    int inputIndex =
                              (y*width*8*(channels / 8))
                              +(width*8*c) 
                              + x*8
                              + c8; 
                    sum += input[inputIndex];
                }
            }

            // Calculate the average and store it in the output
            int outputIndex = (c * 8 + c8);
            output[outputIndex] = sum / kernelArea;
        }
    }

  event1();
}

 

extern "C" {


  void average_pooling(uint8_t *input, uint8_t *output, int in_width, int in_height,int in_channels, int kernelSize)  {
  average_pooling_scalar(input, output, in_width, in_height, in_channels, kernelSize) ;
}


}