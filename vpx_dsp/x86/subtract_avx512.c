/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
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

void vpx_subtract_block_avx512(int rows, int cols, int16_t *diff_ptr,
                               ptrdiff_t diff_stride, const uint8_t *src_ptr,
                               ptrdiff_t src_stride, const uint8_t *pred_ptr,
                               ptrdiff_t pred_stride) {
  int i, j;

  if (cols == 64) {
    // Process 64-column blocks using full 512-bit vectors
    for (i = 0; i < rows; ++i) {
      // Load 64 bytes from source and prediction
      const __m512i src_vec = _mm512_loadu_si512((const __m512i *)src_ptr);
      const __m512i pred_vec = _mm512_loadu_si512((const __m512i *)pred_ptr);
      
      // Convert to 16-bit and subtract
      const __m512i src_lo = _mm512_unpacklo_epi8(src_vec, _mm512_setzero_si512());
      const __m512i src_hi = _mm512_unpackhi_epi8(src_vec, _mm512_setzero_si512());
      const __m512i pred_lo = _mm512_unpacklo_epi8(pred_vec, _mm512_setzero_si512());
      const __m512i pred_hi = _mm512_unpackhi_epi8(pred_vec, _mm512_setzero_si512());
      
      const __m512i diff_lo = _mm512_sub_epi16(src_lo, pred_lo);
      const __m512i diff_hi = _mm512_sub_epi16(src_hi, pred_hi);
      
      // Store results
      _mm512_storeu_si512((__m512i *)diff_ptr, diff_lo);
      _mm512_storeu_si512((__m512i *)(diff_ptr + 32), diff_hi);
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  } else if (cols == 32) {
    // Process 32-column blocks using half 512-bit vectors
    for (i = 0; i < rows; ++i) {
      // Load 32 bytes from source and prediction
      const __m256i src_vec = _mm256_loadu_si256((const __m256i *)src_ptr);
      const __m256i pred_vec = _mm256_loadu_si256((const __m256i *)pred_ptr);
      
      // Convert to 512-bit and extend to 16-bit
      const __m512i src_512 = _mm512_cvtepu8_epi16(src_vec);
      const __m512i pred_512 = _mm512_cvtepu8_epi16(pred_vec);
      
      // Subtract
      const __m512i diff = _mm512_sub_epi16(src_512, pred_512);
      
      // Store result
      _mm512_storeu_si512((__m512i *)diff_ptr, diff);
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  } else if (cols == 16) {
    // Process 16-column blocks
    for (i = 0; i < rows; ++i) {
      // Load 16 bytes from source and prediction
      const __m128i src_vec = _mm_loadu_si128((const __m128i *)src_ptr);
      const __m128i pred_vec = _mm_loadu_si128((const __m128i *)pred_ptr);
      
      // Convert to 512-bit and extend to 16-bit
      const __m512i src_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src_vec));
      const __m512i pred_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(pred_vec));
      
      // Subtract (only use lower 256 bits)
      const __m512i diff = _mm512_sub_epi16(src_512, pred_512);
      
      // Store result (only lower 16 elements)
      _mm256_storeu_si256((__m256i *)diff_ptr, _mm512_extracti64x4_epi64(diff, 0));
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  } else {
    // Fall back to processing smaller blocks or odd sizes
    for (i = 0; i < rows; ++i) {
      for (j = 0; j < cols; j += 8) {
        const int remaining = cols - j;
        const int process_cols = remaining >= 8 ? 8 : remaining;
        
        if (process_cols == 8) {
          // Load 8 bytes from source and prediction
          const __m128i src_vec = _mm_loadl_epi64((const __m128i *)(src_ptr + j));
          const __m128i pred_vec = _mm_loadl_epi64((const __m128i *)(pred_ptr + j));
          
          // Convert to 16-bit
          const __m128i src_16 = _mm_unpacklo_epi8(src_vec, _mm_setzero_si128());
          const __m128i pred_16 = _mm_unpacklo_epi8(pred_vec, _mm_setzero_si128());
          
          // Subtract
          const __m128i diff = _mm_sub_epi16(src_16, pred_16);
          
          // Store result
          _mm_storeu_si128((__m128i *)(diff_ptr + j), diff);
        } else {
          // Handle remaining elements scalar
          for (int k = 0; k < process_cols; ++k) {
            diff_ptr[j + k] = src_ptr[j + k] - pred_ptr[j + k];
          }
        }
      }
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_subtract_block_avx512(int rows, int cols, int16_t *diff_ptr,
                                      ptrdiff_t diff_stride,
                                      const uint8_t *src8_ptr,
                                      ptrdiff_t src_stride,
                                      const uint8_t *pred8_ptr,
                                      ptrdiff_t pred_stride, int bd) {
  int i, j;
  const uint16_t *src_ptr = CONVERT_TO_SHORTPTR(src8_ptr);
  const uint16_t *pred_ptr = CONVERT_TO_SHORTPTR(pred8_ptr);
  (void)bd;

  if (cols == 32) {
    // Process 32 16-bit elements using full 512-bit vectors
    for (i = 0; i < rows; ++i) {
      // Load 32 16-bit elements from source and prediction
      const __m512i src_vec = _mm512_loadu_si512((const __m512i *)src_ptr);
      const __m512i pred_vec = _mm512_loadu_si512((const __m512i *)pred_ptr);
      
      // Subtract
      const __m512i diff = _mm512_sub_epi16(src_vec, pred_vec);
      
      // Store result
      _mm512_storeu_si512((__m512i *)diff_ptr, diff);
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  } else if (cols == 16) {
    // Process 16 16-bit elements using half 512-bit vectors
    for (i = 0; i < rows; ++i) {
      // Load 16 16-bit elements from source and prediction
      const __m256i src_vec = _mm256_loadu_si256((const __m256i *)src_ptr);
      const __m256i pred_vec = _mm256_loadu_si256((const __m256i *)pred_ptr);
      
      // Convert to 512-bit
      const __m512i src_512 = _mm512_castsi256_si512(src_vec);
      const __m512i pred_512 = _mm512_castsi256_si512(pred_vec);
      
      // Subtract
      const __m512i diff = _mm512_sub_epi16(src_512, pred_512);
      
      // Store result (only lower 256 bits)
      _mm256_storeu_si256((__m256i *)diff_ptr, _mm512_extracti64x4_epi64(diff, 0));
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  } else {
    // Fall back to processing smaller blocks
    for (i = 0; i < rows; ++i) {
      for (j = 0; j < cols; j += 8) {
        const int remaining = cols - j;
        const int process_cols = remaining >= 8 ? 8 : remaining;
        
        if (process_cols == 8) {
          // Load 8 16-bit elements from source and prediction
          const __m128i src_vec = _mm_loadu_si128((const __m128i *)(src_ptr + j));
          const __m128i pred_vec = _mm_loadu_si128((const __m128i *)(pred_ptr + j));
          
          // Subtract
          const __m128i diff = _mm_sub_epi16(src_vec, pred_vec);
          
          // Store result
          _mm_storeu_si128((__m128i *)(diff_ptr + j), diff);
        } else {
          // Handle remaining elements scalar
          for (int k = 0; k < process_cols; ++k) {
            diff_ptr[j + k] = src_ptr[j + k] - pred_ptr[j + k];
          }
        }
      }
      
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    }
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH