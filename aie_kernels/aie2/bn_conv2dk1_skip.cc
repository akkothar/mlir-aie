//===- conv2dk1_skip_init.cc -------------------------------------------------*-
// C++
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

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;
//*****************************************************************************
// conv2d 1x1 skip - scalar
// act: uint8, wts: int8, skip: int8, out: int8
//*****************************************************************************
#ifdef REGULAR
void conv2dk1_skip_ui8_i8_i8_scalar(
    uint8_t *input0, int8_t *kernels, int8_t *output, int8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int64_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // scale for convolution
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs + skip_temp;

        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                                 : (skip_sum_srs_final < -MIN)
                                     ? -MIN
                                     : skip_sum_srs_final; // clip
        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;
      }
    }
  }

  event1();
}
#endif

#ifdef PUT
void conv2dk1_skip_ui8_i8_scalar_cascade_put(
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
void conv2dk1_skip_ui8_i8_i8_scalar_cascade_get(
    uint8_t *input0, int8_t *kernels, int8_t *output,int8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;
  
  const int scaleT = scale;
  const int skip_scaleT = skip_scale;

  const int half_input_channels = input_channels / 2;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int64_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;

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
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        //clip

        // skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        // skip_sum = sum_srs + skip_temp;

        // skip_sum_srs_final =
        //     (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        // skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
        //                          : (skip_sum_srs_final < -MIN)
        //                              ? -MIN
        //                              : skip_sum_srs_final; // clip

        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}
#endif
//*****************************************************************************
// conv2d 1x1 skip wrappers
//*****************************************************************************
extern "C" {

#ifdef REGULAR


void conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                       int8_t *output, int8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // REGULAR
#ifdef PUT
void conv2dk1_skip_ui8_i8_put(uint8_t *input0,int8_t *kernels,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels) {
  conv2dk1_skip_ui8_i8_scalar_cascade_put(input0,  kernels,
                                            input_width,  input_channels, 
                                            output_channels);
}
#endif // PUT

#ifdef GET


void conv2dk1_skip_ui8_i8_i8_get(uint8_t *input0,int8_t *kernels,
                       int8_t *output, int8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_i8_i8_scalar_cascade_get(input0,  kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // GET

} // extern "C"