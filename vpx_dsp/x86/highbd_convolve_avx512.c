/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX-512

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_dsp/x86/convolve.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

// High bit-depth 8-tap horizontal convolution using AVX-512
void vpx_highbd_convolve8_horiz_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                        uint16_t *dst, ptrdiff_t dst_stride,
                                        const InterpKernel *filter, int x0_q4,
                                        int x_step_q4, int y0_q4, int y_step_q4,
                                        int w, int h, int bd) {
  const int16_t *filter_x = filter[x0_q4];
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  
  // Load filter coefficients and replicate
  const __m512i f0 = _mm512_set1_epi16(filter_x[0]);
  const __m512i f1 = _mm512_set1_epi16(filter_x[1]);
  const __m512i f2 = _mm512_set1_epi16(filter_x[2]);
  const __m512i f3 = _mm512_set1_epi16(filter_x[3]);
  const __m512i f4 = _mm512_set1_epi16(filter_x[4]);
  const __m512i f5 = _mm512_set1_epi16(filter_x[5]);
  const __m512i f6 = _mm512_set1_epi16(filter_x[6]);
  const __m512i f7 = _mm512_set1_epi16(filter_x[7]);
  
  const __m512i round_const = _mm512_set1_epi32(1 << (FILTER_BITS - 1));
  const int clamp_high = (1 << bd) - 1;
  const __m512i clamp_high_vec = _mm512_set1_epi16(clamp_high);
  
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; x += 16) {
      // Process 16 pixels at a time with AVX-512
      const int remaining = VPXMIN(16, w - x);
      const __mmask16 mask = (1 << remaining) - 1;
      
      // Load 16 + 7 = 23 pixels (need 7 extra for 8-tap filter)
      __m512i s0, s1, s2, s3, s4, s5, s6, s7;
      
      if (x + 23 <= w + 7) {
        // Can load full vectors
        s0 = _mm512_loadu_si512((const __m512i *)(src + x - 3));
        s1 = _mm512_loadu_si512((const __m512i *)(src + x - 2));
        s2 = _mm512_loadu_si512((const __m512i *)(src + x - 1));
        s3 = _mm512_loadu_si512((const __m512i *)(src + x + 0));
        s4 = _mm512_loadu_si512((const __m512i *)(src + x + 1));
        s5 = _mm512_loadu_si512((const __m512i *)(src + x + 2));
        s6 = _mm512_loadu_si512((const __m512i *)(src + x + 3));
        s7 = _mm512_loadu_si512((const __m512i *)(src + x + 4));
      } else {
        // Handle edge cases with masked loads
        s0 = _mm512_maskz_loadu_epi16(mask, src + x - 3);
        s1 = _mm512_maskz_loadu_epi16(mask, src + x - 2);
        s2 = _mm512_maskz_loadu_epi16(mask, src + x - 1);
        s3 = _mm512_maskz_loadu_epi16(mask, src + x + 0);
        s4 = _mm512_maskz_loadu_epi16(mask, src + x + 1);
        s5 = _mm512_maskz_loadu_epi16(mask, src + x + 2);
        s6 = _mm512_maskz_loadu_epi16(mask, src + x + 3);
        s7 = _mm512_maskz_loadu_epi16(mask, src + x + 4);
      }
      
      // Apply 8-tap filter
      __m512i sum = _mm512_mullo_epi16(s0, f0);
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s1, f1));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s2, f2));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s3, f3));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s4, f4));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s5, f5));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s6, f6));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s7, f7));
      
      // Convert to 32-bit for rounding and shift
      const __m512i sum_lo = _mm512_unpacklo_epi16(sum, _mm512_setzero_si512());
      const __m512i sum_hi = _mm512_unpackhi_epi16(sum, _mm512_srai_epi16(sum, 15));
      
      const __m512i rounded_lo = _mm512_add_epi32(sum_lo, round_const);
      const __m512i rounded_hi = _mm512_add_epi32(sum_hi, round_const);
      
      const __m512i shifted_lo = _mm512_srai_epi32(rounded_lo, FILTER_BITS);
      const __m512i shifted_hi = _mm512_srai_epi32(rounded_hi, FILTER_BITS);
      
      // Pack back to 16-bit and clamp
      __m512i result = _mm512_packs_epi32(shifted_lo, shifted_hi);
      result = _mm512_max_epi16(result, _mm512_setzero_si512());
      result = _mm512_min_epi16(result, clamp_high_vec);
      
      // Store result
      _mm512_mask_storeu_epi16(dst + x, mask, result);
    }
    
    src += src_stride;
    dst += dst_stride;
  }
}

// High bit-depth 8-tap vertical convolution using AVX-512
void vpx_highbd_convolve8_vert_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                       uint16_t *dst, ptrdiff_t dst_stride,
                                       const InterpKernel *filter, int x0_q4,
                                       int x_step_q4, int y0_q4, int y_step_q4,
                                       int w, int h, int bd) {
  const int16_t *filter_y = filter[y0_q4];
  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  
  // Load filter coefficients
  const __m512i f0 = _mm512_set1_epi16(filter_y[0]);
  const __m512i f1 = _mm512_set1_epi16(filter_y[1]);
  const __m512i f2 = _mm512_set1_epi16(filter_y[2]);
  const __m512i f3 = _mm512_set1_epi16(filter_y[3]);
  const __m512i f4 = _mm512_set1_epi16(filter_y[4]);
  const __m512i f5 = _mm512_set1_epi16(filter_y[5]);
  const __m512i f6 = _mm512_set1_epi16(filter_y[6]);
  const __m512i f7 = _mm512_set1_epi16(filter_y[7]);
  
  const __m512i round_const = _mm512_set1_epi32(1 << (FILTER_BITS - 1));
  const int clamp_high = (1 << bd) - 1;
  const __m512i clamp_high_vec = _mm512_set1_epi16(clamp_high);
  
  // Process vertical filtering
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; x += 16) {
      const int remaining = VPXMIN(16, w - x);
      const __mmask16 mask = (1 << remaining) - 1;
      
      // Load 8 rows of pixels for vertical filtering
      const __m512i s0 = _mm512_maskz_loadu_epi16(mask, src + x - 3 * src_stride);
      const __m512i s1 = _mm512_maskz_loadu_epi16(mask, src + x - 2 * src_stride);
      const __m512i s2 = _mm512_maskz_loadu_epi16(mask, src + x - 1 * src_stride);
      const __m512i s3 = _mm512_maskz_loadu_epi16(mask, src + x + 0 * src_stride);
      const __m512i s4 = _mm512_maskz_loadu_epi16(mask, src + x + 1 * src_stride);
      const __m512i s5 = _mm512_maskz_loadu_epi16(mask, src + x + 2 * src_stride);
      const __m512i s6 = _mm512_maskz_loadu_epi16(mask, src + x + 3 * src_stride);
      const __m512i s7 = _mm512_maskz_loadu_epi16(mask, src + x + 4 * src_stride);
      
      // Apply 8-tap vertical filter
      __m512i sum = _mm512_mullo_epi16(s0, f0);
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s1, f1));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s2, f2));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s3, f3));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s4, f4));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s5, f5));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s6, f6));
      sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s7, f7));
      
      // Round and shift with proper bit-depth handling
      const __m512i sum_lo = _mm512_unpacklo_epi16(sum, _mm512_setzero_si512());
      const __m512i sum_hi = _mm512_unpackhi_epi16(sum, _mm512_srai_epi16(sum, 15));
      
      const __m512i rounded_lo = _mm512_add_epi32(sum_lo, round_const);
      const __m512i rounded_hi = _mm512_add_epi32(sum_hi, round_const);
      
      const __m512i shifted_lo = _mm512_srai_epi32(rounded_lo, FILTER_BITS);
      const __m512i shifted_hi = _mm512_srai_epi32(rounded_hi, FILTER_BITS);
      
      // Pack and clamp
      __m512i result = _mm512_packs_epi32(shifted_lo, shifted_hi);
      result = _mm512_max_epi16(result, _mm512_setzero_si512());
      result = _mm512_min_epi16(result, clamp_high_vec);
      
      // Store result
      _mm512_mask_storeu_epi16(dst + x, mask, result);
    }
    
    src += src_stride;
    dst += dst_stride;
  }
}

// High bit-depth 8-tap 2D convolution using AVX-512
void vpx_highbd_convolve8_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                  uint16_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel *filter, int x0_q4,
                                  int x_step_q4, int y0_q4, int y_step_q4,
                                  int w, int h, int bd) {
  // For 2D convolution, we need intermediate buffer
  // Use stack buffer for small blocks, heap for large ones
  const int intermediate_height = h + 7;  // Need 7 extra rows for 8-tap
  uint16_t temp_buffer[64 * 71];  // Stack buffer for up to 64x64 blocks
  uint16_t *temp = temp_buffer;
  int temp_allocated = 0;
  
  const int temp_size = w * intermediate_height;
  if (temp_size > sizeof(temp_buffer) / sizeof(temp_buffer[0])) {
    temp = (uint16_t *)vpx_malloc(temp_size * sizeof(uint16_t));
    temp_allocated = 1;
  }
  
  // First pass: horizontal filtering
  vpx_highbd_convolve8_horiz_avx512(src - 3 * src_stride, src_stride,
                                     temp, w,
                                     filter, x0_q4, x_step_q4, 0, 16,
                                     w, intermediate_height, bd);
  
  // Second pass: vertical filtering
  vpx_highbd_convolve8_vert_avx512(temp + 3 * w, w,
                                    dst, dst_stride,
                                    filter, 0, 16, y0_q4, y_step_q4,
                                    w, h, bd);
  
  if (temp_allocated) {
    vpx_free(temp);
  }
}

// High bit-depth copy function optimized with AVX-512
void vpx_highbd_convolve_copy_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                      uint16_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel *filter, int x0_q4,
                                      int x_step_q4, int y0_q4, int y_step_q4,
                                      int w, int h, int bd) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  (void)bd;
  
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; x += 16) {
      const int remaining = VPXMIN(16, w - x);
      const __mmask16 mask = (1 << remaining) - 1;
      
      const __m512i data = _mm512_maskz_loadu_epi16(mask, src + x);
      _mm512_mask_storeu_epi16(dst + x, mask, data);
    }
    
    src += src_stride;
    dst += dst_stride;
  }
}

// High bit-depth averaging function with AVX-512
void vpx_highbd_convolve_avg_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                     uint16_t *dst, ptrdiff_t dst_stride,
                                     const InterpKernel *filter, int x0_q4,
                                     int x_step_q4, int y0_q4, int y_step_q4,
                                     int w, int h, int bd) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  (void)bd;
  
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; x += 16) {
      const int remaining = VPXMIN(16, w - x);
      const __mmask16 mask = (1 << remaining) - 1;
      
      const __m512i src_data = _mm512_maskz_loadu_epi16(mask, src + x);
      const __m512i dst_data = _mm512_maskz_loadu_epi16(mask, dst + x);
      
      // Average with proper rounding
      const __m512i sum = _mm512_add_epi16(src_data, dst_data);
      const __m512i avg = _mm512_add_epi16(sum, _mm512_set1_epi16(1));
      const __m512i result = _mm512_srli_epi16(avg, 1);
      
      _mm512_mask_storeu_epi16(dst + x, mask, result);
    }
    
    src += src_stride;
    dst += dst_stride;
  }
}