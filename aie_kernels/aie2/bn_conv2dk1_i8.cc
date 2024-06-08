//===- conv2dk1.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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

#define REL_WRITE 0
#define REL_READ 1

const int32_t MAX_VALUES = 16;


//*****************************************************************************
// conv2d 1x1_PUT - scalar
// act: int8, wts: int8, cascade: uint8
//*****************************************************************************

#ifdef PUT_I8_CAS
void conv2dk1_i8_scalar_cascade_put(
    int8_t *input0, int8_t *kernels, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int32_t input_split,const int32_t weight_index) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;

  // Calculate half the input channels
  const int input_channel_chunk_size = input_channels / input_split;

  // Determine the start and end of the loop based on the chunk index
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum[MAX_VALUES];
      for (x = 0; x < input_width; x++) { // col of output image
        if(weight_index==0)
          sum[x] = 0;

        for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            
            sum[x] += val * k;
          }
        }
        
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        // sum_srs = (sum_srs > MAX)    ? MAX
        //           : (sum_srs < -MIN) ? -MIN
        //                              : sum_srs; // clip
        v16vec_partial=upd_elem(v16vec_partial, value_index, sum[x]);
        value_index++;
        if (value_index == MAX_VALUES) {
                // Transfer the values from vec to acc 
                v16acc_partial= lups(v16vec_partial,0);
                put_mcd(v16acc_partial); //push over cascade
                // Reset the index
                value_index = 0;
        }
      }
    }
  }

  event1();
}
#endif


#ifdef PUT_UI8_CAS
void conv2dk1_ui8_scalar_cascade_put(
    uint8_t *input0, int8_t *kernels, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;

  // Calculate half the input channels
  const int half_input_channels = input_channels / 2;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs=0;
        for (ic = 0; ic < half_input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (half_input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            
            sum += val * k;
          }
        }
        
        // sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        // sum_srs = (sum_srs > MAX)    ? MAX
        //           : (sum_srs < -MIN) ? -MIN
        //                              : sum_srs; // clip
        v16vec_partial=upd_elem(v16vec_partial, value_index, sum);
        value_index++;
        if (value_index == MAX_VALUES) {
                // Transfer the values from vec to acc 
                v16acc_partial= lups(v16vec_partial,0);
                put_mcd(v16acc_partial); //push over cascade
                // Reset the index
                value_index = 0;
        }
      }
    }
  }

  event1();
}
#endif



#ifdef GET
void conv2dk1_i8_ui8_scalar_cascade_get(
    int8_t *input0, int8_t *kernels, uint8_t *output,
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;
  
  const int scaleT = scale;
  const int half_input_channels = input_channels / 2;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;

        // Extract cascade sum values when starting a new block
        if (value_index == 0) {
                v16acc_partial=get_scd_v16acc64(); // Get the accumulated values
                v16vec_partial= lsrs(v16acc_partial,0,0); // Convert accumulator to vector
                
        }

        // Extract the specific cascade sum for the current index
        int partial_sum=ext_elem(v16vec_partial, value_index);
        value_index++;

        for (ic = half_input_channels/8; ic < input_channels / 8; ic++) {
          
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (half_input_channels / 8) * 64) + ((ic - half_input_channels / 8) * 64) + (ic8 * 8) + oc8];
            
            sum += val * k;
          }
        }
        
        if (value_index == MAX_VALUES) {
                value_index = 0;
        }
        // scale for convolution
        

        sum=sum+partial_sum;
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > UMAX)    ? UMAX
                  : (sum_srs < 0) ? 0
                                     : sum_srs; // clip
        //clip

        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif

#ifdef SCALAR

const int32_t SMAX = 127;
const int32_t SMIN = 128;
//*****************************************************************************
// conv2d 1x1 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, int8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            uint8_t val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int8_t k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                               (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > SMAX) ? SMAX : (sum_srs < -SMIN) ? -SMIN : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }
event1();
}

#endif

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef PUT_I8_CAS
void conv2dk1_i8_put(int8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels,const int32_t input_split,const int32_t weight_index) {
  conv2dk1_i8_scalar_cascade_put(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels, input_split,weight_index);
}
#endif // PUT

#ifdef PUT_UI8_CAS
void conv2dk1_ui8_put(uint8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels) {
  conv2dk1_ui8_scalar_cascade_put(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels);
}
#endif // PUT


#ifdef GET
void conv2dk1_i8_ui8_get(int8_t *input0,int8_t *kernels,
                       uint8_t *output,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale
                       ) {
  conv2dk1_i8_ui8_scalar_cascade_get(input0,  kernels, output, input_width,
                           input_channels, output_channels, scale);
}

#endif // GET

#ifdef BN10
    void bn10_conv2dk1_ui8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }
#endif //BN12

#ifdef BN12
    void bn12_conv2dk1_ui8(uint8_t *input, int8_t *kernels, int8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

#endif // BN10
} // extern "C"