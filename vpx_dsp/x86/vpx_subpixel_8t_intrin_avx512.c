/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX-512

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_mem/vpx_mem.h"

// AVX-512 specific 8-tap convolution implementation

static INLINE void vpx_filter_block1d16_h8_avx512(
    const uint8_t *src_ptr, ptrdiff_t src_stride, uint8_t *output_ptr,
    ptrdiff_t output_stride, uint32_t output_height,
    const int16_t *filter) {
  
  // Load filter coefficients
  const __m512i k_256 = _mm512_set1_epi16(256);
  const __m512i f_values = _mm512_set1_epi64(
      (((int64_t)filter[7]) << 48) | (((int64_t)filter[6]) << 32) |
      (((int64_t)filter[5]) << 16) | ((int64_t)filter[4]));
  const __m512i f_values2 = _mm512_set1_epi64(
      (((int64_t)filter[3]) << 48) | (((int64_t)filter[2]) << 32) |
      (((int64_t)filter[1]) << 16) | ((int64_t)filter[0]));
  
  for (unsigned int i = 0; i < output_height; ++i) {
    // Load 16 pixels + 7 extra for filtering
    const __m512i src_reg = _mm512_loadu_si512((const __m512i *)(src_ptr - 3));
    
    // Convert to 16-bit for processing
    const __m512i src_lo = _mm512_unpacklo_epi8(src_reg, _mm512_setzero_si512());
    const __m512i src_hi = _mm512_unpackhi_epi8(src_reg, _mm512_setzero_si512());
    
    // Apply 8-tap horizontal filter
    __m512i res_lo = _mm512_setzero_si512();
    __m512i res_hi = _mm512_setzero_si512();
    
    for (int k = 0; k < 8; k++) {
      const __m512i f_val = _mm512_set1_epi16(filter[k]);
      
      // Shift source data by k positions
      const __m512i src_shifted_lo = _mm512_bsrli_epi128(src_lo, k * 2);
      const __m512i src_shifted_hi = _mm512_bsrli_epi128(src_hi, k * 2);
      
      res_lo = _mm512_add_epi16(res_lo, _mm512_mullo_epi16(src_shifted_lo, f_val));
      res_hi = _mm512_add_epi16(res_hi, _mm512_mullo_epi16(src_shifted_hi, f_val));
    }
    
    // Round and shift
    res_lo = _mm512_add_epi16(res_lo, k_256);
    res_hi = _mm512_add_epi16(res_hi, k_256);
    res_lo = _mm512_srai_epi16(res_lo, 9);
    res_hi = _mm512_srai_epi16(res_hi, 9);
    
    // Pack back to 8-bit and store
    const __m512i result = _mm512_packus_epi16(res_lo, res_hi);
    _mm256_storeu_si256((__m256i *)output_ptr, 
                        _mm512_extracti64x4_epi64(result, 0));
    
    src_ptr += src_stride;
    output_ptr += output_stride;
  }
}

static INLINE void vpx_filter_block1d16_v8_avx512(
    const uint8_t *src_ptr, ptrdiff_t src_stride, uint8_t *output_ptr,
    ptrdiff_t output_stride, uint32_t output_height,
    const int16_t *filter) {
  
  // Load filter coefficients
  const __m512i k_256 = _mm512_set1_epi16(256);
  
  for (unsigned int i = 0; i < output_height; ++i) {
    // Load 8 lines of 16 pixels each for vertical filtering
    __m512i src_lines[8];
    for (int k = 0; k < 8; k++) {
      const __m256i src_line = _mm256_loadu_si256(
          (const __m256i *)(src_ptr + (k - 3) * src_stride));
      src_lines[k] = _mm512_cvtepu8_epi16(src_line);
    }
    
    // Apply 8-tap vertical filter
    __m512i result = _mm512_setzero_si512();
    for (int k = 0; k < 8; k++) {
      const __m512i f_val = _mm512_set1_epi16(filter[k]);
      result = _mm512_add_epi16(result, _mm512_mullo_epi16(src_lines[k], f_val));
    }
    
    // Round and shift
    result = _mm512_add_epi16(result, k_256);
    result = _mm512_srai_epi16(result, 9);
    
    // Pack to 8-bit and store
    const __m256i result_8bit = _mm256_packus_epi16(
        _mm512_extracti64x4_epi64(result, 0),
        _mm512_extracti64x4_epi64(result, 1));
    _mm256_storeu_si256((__m256i *)output_ptr, result_8bit);
    
    src_ptr += src_stride;
    output_ptr += output_stride;
  }
}

void vpx_convolve8_horiz_avx512(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel *filter, int x0_q4,
                                int x_step_q4, int y0_q4, int y_step_q4,
                                int w, int h) {
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  
  const int16_t *const filter_x = filter[x0_q4];
  
  if (w == 16) {
    vpx_filter_block1d16_h8_avx512(src, src_stride, dst, dst_stride, h, filter_x);
  } else if (w == 32) {
    // Process 32-pixel width using two 16-pixel operations
    vpx_filter_block1d16_h8_avx512(src, src_stride, dst, dst_stride, h, filter_x);
    vpx_filter_block1d16_h8_avx512(src + 16, src_stride, dst + 16, dst_stride, h, filter_x);
  } else if (w == 64) {
    // Process 64-pixel width using four 16-pixel operations
    vpx_filter_block1d16_h8_avx512(src, src_stride, dst, dst_stride, h, filter_x);
    vpx_filter_block1d16_h8_avx512(src + 16, src_stride, dst + 16, dst_stride, h, filter_x);
    vpx_filter_block1d16_h8_avx512(src + 32, src_stride, dst + 32, dst_stride, h, filter_x);
    vpx_filter_block1d16_h8_avx512(src + 48, src_stride, dst + 48, dst_stride, h, filter_x);
  } else {
    // Fall back to reference implementation for other sizes
    vpx_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                          x_step_q4, y0_q4, y_step_q4, w, h);
  }
}

void vpx_convolve8_vert_avx512(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int x_step_q4, int y0_q4, int y_step_q4,
                               int w, int h) {
  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  
  const int16_t *const filter_y = filter[y0_q4];
  
  if (w == 16) {
    vpx_filter_block1d16_v8_avx512(src, src_stride, dst, dst_stride, h, filter_y);
  } else if (w == 32) {
    // Process 32-pixel width using two 16-pixel operations
    vpx_filter_block1d16_v8_avx512(src, src_stride, dst, dst_stride, h, filter_y);
    vpx_filter_block1d16_v8_avx512(src + 16, src_stride, dst + 16, dst_stride, h, filter_y);
  } else if (w == 64) {
    // Process 64-pixel width using four 16-pixel operations
    vpx_filter_block1d16_v8_avx512(src, src_stride, dst, dst_stride, h, filter_y);
    vpx_filter_block1d16_v8_avx512(src + 16, src_stride, dst + 16, dst_stride, h, filter_y);
    vpx_filter_block1d16_v8_avx512(src + 32, src_stride, dst + 32, dst_stride, h, filter_y);
    vpx_filter_block1d16_v8_avx512(src + 48, src_stride, dst + 48, dst_stride, h, filter_y);
  } else {
    // Fall back to reference implementation for other sizes
    vpx_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                         x_step_q4, y0_q4, y_step_q4, w, h);
  }
}

void vpx_convolve8_avx512(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *filter, int x0_q4,
                          int x_step_q4, int y0_q4, int y_step_q4,
                          int w, int h) {
  
  // Use temporary buffer for separable filtering
  DECLARE_ALIGNED(64, uint8_t, temp[64 * (64 + 7)]);
  const int temp_stride = 64;
  
  // Apply horizontal filter first
  vpx_convolve8_horiz_avx512(src - 3 * src_stride, src_stride, temp, temp_stride,
                             filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h + 7);
  
  // Apply vertical filter
  vpx_convolve8_vert_avx512(temp + 3 * temp_stride, temp_stride, dst, dst_stride,
                            filter, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
}

void vpx_convolve8_avg_avx512(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int x0_q4,
                              int x_step_q4, int y0_q4, int y_step_q4,
                              int w, int h) {
  
  // Use temporary buffer for convolution result
  DECLARE_ALIGNED(64, uint8_t, temp[64 * 64]);
  
  // Apply convolution
  vpx_convolve8_avx512(src, src_stride, temp, w, filter, x0_q4, x_step_q4,
                       y0_q4, y_step_q4, w, h);
  
  // Average with destination
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; j += 64) {
      const int actual_w = (w - j) >= 64 ? 64 : (w - j);
      
      if (actual_w >= 32) {
        const __m512i temp_vec = _mm512_loadu_si512((const __m512i *)(temp + i * w + j));
        const __m512i dst_vec = _mm512_loadu_si512((const __m512i *)(dst + i * dst_stride + j));
        const __m512i avg = _mm512_avg_epu8(temp_vec, dst_vec);
        _mm512_storeu_si512((__m512i *)(dst + i * dst_stride + j), avg);
      } else {
        // Fall back to smaller vector operations or scalar
        for (int k = 0; k < actual_w; ++k) {
          dst[i * dst_stride + j + k] = 
              (temp[i * w + j + k] + dst[i * dst_stride + j + k] + 1) >> 1;
        }
      }
    }
  }
}