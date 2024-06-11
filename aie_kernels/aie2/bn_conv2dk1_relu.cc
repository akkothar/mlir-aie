//===- conv2dk1.cc -------------------------------------------------*- C++
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

#define REL_WRITE 0
#define REL_READ 1
const int32_t UMAX = 255;
const int32_t MAX_VALUES = 16;


#ifdef PARTIAL

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_ui8_scalar_partial(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale,
                        int32_t input_split,int32_t weight_index,int32_t oc ) {
  
  event0();
  int x, ic, ic8, oc8;

  //  v16acc64 chess_storage(cm0) v16acc_partial0;
  //  v16acc64 chess_storage(cm1) v16acc_partial1;
  //  v16acc64 chess_storage(cm2) v16acc_partial2;
  //  v16acc64 chess_storage(cm3) v16acc_partial3;
  //  v16acc64 chess_storage(cm4) v16acc_partial4;
  //  v16acc64 chess_storage(cm5) v16acc_partial5;
  //  v16acc64 chess_storage(cm6) v16acc_partial6;
  //  v16acc64 chess_storage(cm7) v16acc_partial7;
  //  v16acc64 chess_storage(cm8) v16acc_partial8;

  static v16acc64 v16acc_partial0;
  static v16acc64 v16acc_partial1;
  static v16acc64 v16acc_partial2;
  static v16acc64 v16acc_partial3;
  static v16acc64 v16acc_partial4;
  static v16acc64 v16acc_partial5;
  static v16acc64 v16acc_partial6;
  static v16acc64 v16acc_partial7;
  static v16acc64 v16acc_partial8;

  // Array of pointers to the accumulators
  v16acc64* accumulators[] = {
      &v16acc_partial0, &v16acc_partial1, &v16acc_partial2, &v16acc_partial3,
      &v16acc_partial4, &v16acc_partial5, &v16acc_partial6, &v16acc_partial7,
      &v16acc_partial8
  };

  // static v16acc64 v16acc_partial;

  // Determine the start and end of the loop based on the chunk index for weights
  const int input_channel_chunk_size = input_channels / input_split;
  const int start_ic = weight_index * input_channel_chunk_size;
  const int end_ic =  start_ic + input_channel_chunk_size;
  // for (oc = 0; oc < output_channels / 8; oc++) {
  for (x = 0; x < input_width; x++) { // col of output image
    v16acc64& accumulator = *accumulators[x];
    v16int32 v16vec_partial = lsrs(accumulator,0,0); 
    int value_index = 0;

    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum = 0;
      int current_sum = 0;
      int sum_srs = 0;
      int last_sum = 0;
      int final_sum=0;

      //Current iteration: go over all the input channels
      for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) + ((ic - start_ic / 8) * 64) + (ic8 * 8) + oc8];
            current_sum += val * k;
          }
      }
    
      if (weight_index != 0){  // Extract the partial sum 
        last_sum=ext_elem(v16vec_partial, value_index);
      }

      sum=current_sum+last_sum;

      // Transfer scalar sum to vector
      v16vec_partial=upd_elem(v16vec_partial, value_index, sum); 
      value_index++; 

      if(end_ic == input_channels){ //if final set of input channels, scale the final output
            // Transfer the values from acc to vect 
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
            // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
            output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
      
      
      if (oc8 == 7) { //end of vectorization
            // // Transfer the values from vec to acc 
          accumulator= lups(v16vec_partial,0);
          value_index = 0;
      }
    }
    
  }
  // }
  event1();



  // event0();
  // int x, ic,  ic8, oc8;
  // // scale=-17;
  // // for (oc = 0; oc < output_channels / 8; oc++) {
  //   for (x = 0; x < input_width; x++) { // col of output image
  //     for (oc8 = 0; oc8 < 8; oc8++) {
  //       int sum = 0;
  //       int sum_srs = 0;

  //       for (ic = 0; ic < input_channels / 8; ic++) {
  //         for (ic8 = 0; ic8 < 8; ic8++) {
  //           int val = input[(ic * input_width * 8) + (x * 8) + ic8];
  //           int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
  //                           (ic8 * 8) + oc8];
  //           sum += val * k;
  //         }
  //       }

  //       // sum_srs=sum>>scale;
  //       sum_srs = (sum + (1 << (scale - 1))) >> scale;
  //       sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
  //       // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
  //       output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
  //     }
  //   }
  // // }

  // event1();


}
#endif
//*****************************************************************************
// conv2d 1x1_GET - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************

#ifdef GET
void conv2dk1_i8_ui8_scalar_cascade_get(
    int8_t *input0, int8_t *kernels, uint8_t *output,
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int32_t input_split,const int32_t weight_index,
    const int scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;
  
  const int scaleT = scale;
  const int input_channel_chunk_size = input_channels / input_split;

  // Determine the start and end of the loop based on the chunk index
  const int start_ic = input_channels/2 + weight_index * input_channel_chunk_size;
  const int end_ic = start_ic + input_channel_chunk_size;

  v16int32 v16vec_partial = undef_v16int32();
  v16acc64 v16acc_partial = undef_v16acc64();
  int value_index = 0;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      int sum[MAX_VALUES];
      for (x = 0; x < input_width; x++) { // col of output image
       if(weight_index==0)
          sum[x] = 0;
        int sum_srs = 0;

        // Extract cascade sum values when starting a new block
        if (value_index == 0) {
                v16acc_partial=get_scd_v16acc64(); // Get the accumulated values
                v16vec_partial= lsrs(v16acc_partial,0,0); // Convert accumulator to vector
                
        }

        // Extract the specific cascade sum for the current index
        int partial_sum=ext_elem(v16vec_partial, value_index);
        value_index++;

        for (ic = start_ic/8; ic < end_ic / 8; ic++) {
          
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channel_chunk_size / 8) * 64) + ((ic - input_channel_chunk_size / 8) * 64) + (ic8 * 8) + oc8];
            
            sum[x] += val * k;
          }
        }
        
        if (value_index == MAX_VALUES) {
                value_index = 0;
        }
        // scale for convolution
        sum[x]=sum[x]+partial_sum;
        // sum=partial_sum;
        if(end_ic == input_channels){
          sum_srs = (sum[x] + (1 << (scaleT - 1))) >> scaleT;
          sum_srs = (sum_srs > UMAX)    ? UMAX
                    : (sum_srs < 0) ? 0
                                      : sum_srs; // clip
          //clip

          output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
        }
      }
    }
  }

  event1();
}
#endif

#ifdef SCALAR

const int32_t UMAX = 255;

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
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
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

#else // UINT8_ACT

//*****************************************************************************
// conv2d 1x1 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_ui8_scalar(uint8_t *input, int8_t *kernels, uint8_t *output,
                         const int32_t input_width,
                         const int32_t input_channels,
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
        sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

#endif // UINT8_ACT

#else // Vector

#endif // Vector

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

#ifdef BN10
    #ifdef SCALAR

    #ifdef INT8_ACT

    void bn10_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                    const int32_t input_width, const int32_t input_channels,
                    const int32_t output_channels, const int scale) {
      conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                        output_channels, scale);
    }

    #else // UINT8_ACT

    void bn10_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

    #endif // UINT8_ACT

    #endif // Vector
  #endif // Vector
  #ifdef BN12
    #ifdef SCALAR

    #ifdef INT8_ACT

    void bn12_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                    const int32_t input_width, const int32_t input_channels,
                    const int32_t output_channels, const int scale) {
      conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                        output_channels, scale);
    }

    #else // UINT8_ACT

    void bn12_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
      conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
    }

    #endif // UINT8_ACT

    #endif // Vector
   #endif // Vector
  
  
 #ifdef BN11
      #ifdef SCALAR

      #ifdef INT8_ACT

      void bn11_conv2dk1_i8(int8_t *input, int8_t *kernels, uint8_t *output,
                      const int32_t input_width, const int32_t input_channels,
                      const int32_t output_channels, const int scale) {
        conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                          output_channels, scale);
      }

      #else // UINT8_ACT

      void bn11_conv2dk1_ui8(uint8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
        conv2dk1_ui8_scalar(input, kernels, output, input_width, input_channels,
                            output_channels, scale);
      }

      #endif // UINT8_ACT


      #endif // Vector
#endif



#ifdef GET


void conv2dk1_i8_ui8_get(int8_t *input0,int8_t *kernels,
                       uint8_t *output,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels,const int32_t input_split,
                       const int32_t weight_index, const int scale
                       ) {
  conv2dk1_i8_ui8_scalar_cascade_get(input0,  kernels, output, input_width,
                           input_channels, output_channels,input_split,weight_index, scale);
}

#endif // GET

#ifdef PARTIAL

void conv2dk1_i8_ui8_partial(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale,
                         int32_t input_split, int32_t weight_index, int32_t oc ) {
  conv2dk1_i8_ui8_scalar_partial(input, kernels, output, input_width, input_channels,
                     output_channels, scale,input_split,weight_index,oc);
}
#endif

} // extern "C"