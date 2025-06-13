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
#include <string.h>     // for memcpy

#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

// Forward declarations
void vpx_scaled_horiz_avx512(const uint8_t *src, ptrdiff_t src_stride,
                             uint8_t *dst, ptrdiff_t dst_stride,
                             const InterpKernel *filter,
                             int x0_q4, int x_step_q4,
                             int w, int h);

void vpx_scaled_vert_avx512(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter,
                            int y0_q4, int y_step_q4,
                            int w, int h);

// Optimized scaled 8-tap convolution using AVX-512
void vpx_scaled_2d_avx512(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *filter,
                          int x0_q4, int x_step_q4,
                          int y0_q4, int y_step_q4,
                          int w, int h) {
  // For optimal performance with AVX-512, we process in chunks
  const int temp_h = (((h - 1) * y_step_q4 + y0_q4) >> 4) + 8;
  
  // Use stack buffer for small blocks, heap for large ones
  uint8_t temp_buffer[64 * 72];  // Maximum 64-wide by 72-high
  uint8_t *temp = temp_buffer;
  int temp_allocated = 0;
  
  const int temp_size = w * temp_h;
  if (temp_size > sizeof(temp_buffer)) {
    temp = (uint8_t *)vpx_malloc(temp_size);
    temp_allocated = 1;
  }
  
  // Horizontal pass with scaling
  vpx_scaled_horiz_avx512(src, src_stride, temp, w, filter,
                          x0_q4, x_step_q4, w, temp_h);
  
  // Vertical pass with scaling
  vpx_scaled_vert_avx512(temp, w, dst, dst_stride, filter,
                         y0_q4, y_step_q4, w, h);
  
  if (temp_allocated) {
    vpx_free(temp);
  }
}

// Horizontal scaled convolution using AVX-512
void vpx_scaled_horiz_avx512(const uint8_t *src, ptrdiff_t src_stride,
                             uint8_t *dst, ptrdiff_t dst_stride,
                             const InterpKernel *filter,
                             int x0_q4, int x_step_q4,
                             int w, int h) {
  const __m512i round_const = _mm512_set1_epi32(1 << (FILTER_BITS - 1));
  
  for (int y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    
    for (int x = 0; x < w; x += 16) {
      // Process up to 16 pixels at once
      const int pixels_to_process = VPXMIN(16, w - x);
      const __mmask16 mask = (1 << pixels_to_process) - 1;
      
      __m512i results = _mm512_setzero_si512();
      
      // Process each output pixel
      for (int i = 0; i < pixels_to_process; ++i) {
        const int x_src = x_q4 >> 4;
        const int x_frac = x_q4 & 15;
        
        // Load filter coefficients for this fractional position
        const int16_t *f = filter[x_frac];
        const __m512i f0 = _mm512_set1_epi16(f[0]);
        const __m512i f1 = _mm512_set1_epi16(f[1]);
        const __m512i f2 = _mm512_set1_epi16(f[2]);
        const __m512i f3 = _mm512_set1_epi16(f[3]);
        const __m512i f4 = _mm512_set1_epi16(f[4]);
        const __m512i f5 = _mm512_set1_epi16(f[5]);
        const __m512i f6 = _mm512_set1_epi16(f[6]);
        const __m512i f7 = _mm512_set1_epi16(f[7]);
        
        // Load 8 source pixels for 8-tap filter
        const __m512i s0 = _mm512_set1_epi16(src[x_src - 3]);
        const __m512i s1 = _mm512_set1_epi16(src[x_src - 2]);
        const __m512i s2 = _mm512_set1_epi16(src[x_src - 1]);
        const __m512i s3 = _mm512_set1_epi16(src[x_src + 0]);
        const __m512i s4 = _mm512_set1_epi16(src[x_src + 1]);
        const __m512i s5 = _mm512_set1_epi16(src[x_src + 2]);
        const __m512i s6 = _mm512_set1_epi16(src[x_src + 3]);
        const __m512i s7 = _mm512_set1_epi16(src[x_src + 4]);
        
        // Apply 8-tap filter
        __m512i sum = _mm512_mullo_epi16(s0, f0);
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s1, f1));
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s2, f2));
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s3, f3));
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s4, f4));
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s5, f5));
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s6, f6));
        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(s7, f7));
        
        // Round and shift
        const __m512i sum_32 = _mm512_add_epi32(
            _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum, 0)),
            round_const);
        const __m512i shifted = _mm512_srai_epi32(sum_32, FILTER_BITS);
        const __m512i clamped = _mm512_packus_epi32(shifted, shifted);
        
        // Insert result at position i
        results = _mm512_mask_blend_epi8((1 << i), results,
                    _mm512_shuffle_epi8(clamped, _mm512_set1_epi8(i)));
        
        x_q4 += x_step_q4;
      }
      
      // Store results
      _mm512_mask_storeu_epi8(dst + x, mask, results);
    }
    
    src += src_stride;
    dst += dst_stride;
  }
}

// Vertical scaled convolution using AVX-512
void vpx_scaled_vert_avx512(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter,
                            int y0_q4, int y_step_q4,
                            int w, int h) {
  const __m512i round_const = _mm512_set1_epi32(1 << (FILTER_BITS - 1));
  int y_q4 = y0_q4;
  
  for (int y = 0; y < h; ++y) {
    const int y_src = y_q4 >> 4;
    const int y_frac = y_q4 & 15;
    
    // Load filter coefficients for this fractional position
    const int16_t *f = filter[y_frac];
    const __m512i f0 = _mm512_set1_epi16(f[0]);
    const __m512i f1 = _mm512_set1_epi16(f[1]);
    const __m512i f2 = _mm512_set1_epi16(f[2]);
    const __m512i f3 = _mm512_set1_epi16(f[3]);
    const __m512i f4 = _mm512_set1_epi16(f[4]);
    const __m512i f5 = _mm512_set1_epi16(f[5]);
    const __m512i f6 = _mm512_set1_epi16(f[6]);
    const __m512i f7 = _mm512_set1_epi16(f[7]);
    
    for (int x = 0; x < w; x += 64) {
      const int pixels_to_process = VPXMIN(64, w - x);
      const __mmask64 mask = (1ULL << pixels_to_process) - 1;
      
      // Load 8 rows of pixels for vertical filtering
      const __m512i s0 = _mm512_maskz_loadu_epi8(mask, src + (y_src - 3) * src_stride + x);
      const __m512i s1 = _mm512_maskz_loadu_epi8(mask, src + (y_src - 2) * src_stride + x);
      const __m512i s2 = _mm512_maskz_loadu_epi8(mask, src + (y_src - 1) * src_stride + x);
      const __m512i s3 = _mm512_maskz_loadu_epi8(mask, src + (y_src + 0) * src_stride + x);
      const __m512i s4 = _mm512_maskz_loadu_epi8(mask, src + (y_src + 1) * src_stride + x);
      const __m512i s5 = _mm512_maskz_loadu_epi8(mask, src + (y_src + 2) * src_stride + x);
      const __m512i s6 = _mm512_maskz_loadu_epi8(mask, src + (y_src + 3) * src_stride + x);
      const __m512i s7 = _mm512_maskz_loadu_epi8(mask, src + (y_src + 4) * src_stride + x);
      
      // Convert to 16-bit for multiplication
      const __m512i s0_lo = _mm512_unpacklo_epi8(s0, _mm512_setzero_si512());
      const __m512i s0_hi = _mm512_unpackhi_epi8(s0, _mm512_setzero_si512());
      const __m512i s1_lo = _mm512_unpacklo_epi8(s1, _mm512_setzero_si512());
      const __m512i s1_hi = _mm512_unpackhi_epi8(s1, _mm512_setzero_si512());
      const __m512i s2_lo = _mm512_unpacklo_epi8(s2, _mm512_setzero_si512());
      const __m512i s2_hi = _mm512_unpackhi_epi8(s2, _mm512_setzero_si512());
      const __m512i s3_lo = _mm512_unpacklo_epi8(s3, _mm512_setzero_si512());
      const __m512i s3_hi = _mm512_unpackhi_epi8(s3, _mm512_setzero_si512());
      const __m512i s4_lo = _mm512_unpacklo_epi8(s4, _mm512_setzero_si512());
      const __m512i s4_hi = _mm512_unpackhi_epi8(s4, _mm512_setzero_si512());
      const __m512i s5_lo = _mm512_unpacklo_epi8(s5, _mm512_setzero_si512());
      const __m512i s5_hi = _mm512_unpackhi_epi8(s5, _mm512_setzero_si512());
      const __m512i s6_lo = _mm512_unpacklo_epi8(s6, _mm512_setzero_si512());
      const __m512i s6_hi = _mm512_unpackhi_epi8(s6, _mm512_setzero_si512());
      const __m512i s7_lo = _mm512_unpacklo_epi8(s7, _mm512_setzero_si512());
      const __m512i s7_hi = _mm512_unpackhi_epi8(s7, _mm512_setzero_si512());
      
      // Apply 8-tap vertical filter
      __m512i sum_lo = _mm512_mullo_epi16(s0_lo, f0);
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s1_lo, f1));
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s2_lo, f2));
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s3_lo, f3));
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s4_lo, f4));
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s5_lo, f5));
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s6_lo, f6));
      sum_lo = _mm512_add_epi16(sum_lo, _mm512_mullo_epi16(s7_lo, f7));
      
      __m512i sum_hi = _mm512_mullo_epi16(s0_hi, f0);
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s1_hi, f1));
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s2_hi, f2));
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s3_hi, f3));
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s4_hi, f4));
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s5_hi, f5));
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s6_hi, f6));
      sum_hi = _mm512_add_epi16(sum_hi, _mm512_mullo_epi16(s7_hi, f7));
      
      // Round and shift to 32-bit
      const __m512i sum_lo_32_0 = _mm512_unpacklo_epi16(sum_lo, _mm512_setzero_si512());
      const __m512i sum_lo_32_1 = _mm512_unpackhi_epi16(sum_lo, _mm512_setzero_si512());
      const __m512i sum_hi_32_0 = _mm512_unpacklo_epi16(sum_hi, _mm512_setzero_si512());
      const __m512i sum_hi_32_1 = _mm512_unpackhi_epi16(sum_hi, _mm512_setzero_si512());
      
      const __m512i rounded_lo_0 = _mm512_add_epi32(sum_lo_32_0, round_const);
      const __m512i rounded_lo_1 = _mm512_add_epi32(sum_lo_32_1, round_const);
      const __m512i rounded_hi_0 = _mm512_add_epi32(sum_hi_32_0, round_const);
      const __m512i rounded_hi_1 = _mm512_add_epi32(sum_hi_32_1, round_const);
      
      const __m512i shifted_lo_0 = _mm512_srai_epi32(rounded_lo_0, FILTER_BITS);
      const __m512i shifted_lo_1 = _mm512_srai_epi32(rounded_lo_1, FILTER_BITS);
      const __m512i shifted_hi_0 = _mm512_srai_epi32(rounded_hi_0, FILTER_BITS);
      const __m512i shifted_hi_1 = _mm512_srai_epi32(rounded_hi_1, FILTER_BITS);
      
      // Pack back to 8-bit and clamp
      const __m512i packed_lo = _mm512_packus_epi32(shifted_lo_0, shifted_lo_1);
      const __m512i packed_hi = _mm512_packus_epi32(shifted_hi_0, shifted_hi_1);
      const __m512i result = _mm512_packus_epi16(packed_lo, packed_hi);
      
      // Store result
      _mm512_mask_storeu_epi8(dst + x, mask, result);
    }
    
    y_q4 += y_step_q4;
    dst += dst_stride;
  }
}

// Optimized averaging scaled convolution using AVX-512
void vpx_scaled_avg_2d_avx512(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter,
                              int x0_q4, int x_step_q4,
                              int y0_q4, int y_step_q4,
                              int w, int h) {
  // First perform standard scaled convolution to temporary buffer
  const int temp_h = (((h - 1) * y_step_q4 + y0_q4) >> 4) + 8;
  
  uint8_t temp_buffer[64 * 72];
  uint8_t *temp = temp_buffer;
  int temp_allocated = 0;
  
  const int temp_size = w * h;  // Only need h rows for final result
  if (temp_size > sizeof(temp_buffer)) {
    temp = (uint8_t *)vpx_malloc(temp_size);
    temp_allocated = 1;
  }
  
  // Perform scaled convolution to temp buffer
  vpx_scaled_2d_avx512(src, src_stride, temp, w, filter,
                       x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
  
  // Average with existing dst content using AVX-512
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; x += 64) {
      const int pixels_to_process = VPXMIN(64, w - x);
      const __mmask64 mask = (1ULL << pixels_to_process) - 1;
      
      const __m512i temp_data = _mm512_maskz_loadu_epi8(mask, temp + y * w + x);
      const __m512i dst_data = _mm512_maskz_loadu_epi8(mask, dst + y * dst_stride + x);
      
      // Average with proper rounding
      const __m512i sum = _mm512_add_epi8(temp_data, dst_data);
      const __m512i avg = _mm512_add_epi8(sum, _mm512_set1_epi8(1));
      const __m512i result = _mm512_srli_epi16(avg, 1);
      
      _mm512_mask_storeu_epi8(dst + y * dst_stride + x, mask, result);
    }
  }
  
  if (temp_allocated) {
    vpx_free(temp);
  }
}