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

#include "./vpx_dsp_rtcd.h"

// Manual reduction for 16-bit values since manual_reduce_epi16 doesn't exist
static INLINE int manual_reduce_epi16(__m512i vsum) {
  __m256i sum_256 = _mm512_extracti64x4_epi64(vsum, 0);
  sum_256 = _mm256_add_epi16(sum_256, _mm512_extracti64x4_epi64(vsum, 1));
  __m128i sum_128 = _mm256_extracti128_si256(sum_256, 0);
  sum_128 = _mm_add_epi16(sum_128, _mm256_extracti128_si256(sum_256, 1));
  sum_128 = _mm_add_epi16(sum_128, _mm_srli_si128(sum_128, 8));
  sum_128 = _mm_add_epi16(sum_128, _mm_srli_si128(sum_128, 4));
  sum_128 = _mm_add_epi16(sum_128, _mm_srli_si128(sum_128, 2));
  return _mm_extract_epi16(sum_128, 0);
}

// DC Predictor for 32x32 blocks
void vpx_dc_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  __m512i sum = _mm512_setzero_si512();
  
  // Load and sum above pixels (32 bytes)
  const __m256i above_vec = _mm256_loadu_si256((const __m256i *)above);
  const __m512i above_512 = _mm512_cvtepu8_epi16(above_vec);
  sum = _mm512_add_epi16(sum, above_512);
  
  // Load and sum left pixels (32 bytes)  
  const __m256i left_vec = _mm256_loadu_si256((const __m256i *)left);
  const __m512i left_512 = _mm512_cvtepu8_epi16(left_vec);
  sum = _mm512_add_epi16(sum, left_512);
  
  // Horizontal reduction to get total sum
  const uint32_t total_sum = manual_reduce_epi16(sum);
  const uint8_t dc_val = (total_sum + 32) >> 6;  // Divide by 64 with rounding
  
  // Broadcast DC value to 512-bit vector
  const __m512i dc_vec = _mm512_set1_epi8(dc_val);
  
  // Fill 32x32 block
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512((__m512i *)(dst + i * stride), dc_vec);
  }
}

// Horizontal Predictor for 32x32 blocks
void vpx_h_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)above;
  
  for (int i = 0; i < 32; ++i) {
    // Broadcast left pixel value across entire row
    const __m512i left_val = _mm512_set1_epi8(left[i]);
    _mm512_storeu_si512((__m512i *)(dst + i * stride), left_val);
  }
}

// Vertical Predictor for 32x32 blocks  
void vpx_v_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)left;
  
  // Load above row once
  const __m512i above_vec = _mm512_loadu_si512((const __m512i *)above);
  
  // Replicate above row to all 32 rows
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512((__m512i *)(dst + i * stride), above_vec);
  }
}

// TM (True Motion) Predictor for 32x32 blocks
void vpx_tm_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint8_t top_left = above[-1];
  const __m512i top_left_vec = _mm512_set1_epi16(top_left);
  
  // Load above row and convert to 16-bit
  const __m256i above_8bit = _mm256_loadu_si256((const __m256i *)above);
  const __m512i above_vec = _mm512_cvtepu8_epi16(above_8bit);
  
  // Calculate above - top_left once
  const __m512i above_diff = _mm512_sub_epi16(above_vec, top_left_vec);
  
  for (int i = 0; i < 32; ++i) {
    // Broadcast left pixel and convert to 16-bit
    const __m512i left_val = _mm512_set1_epi16(left[i]);
    
    // Calculate left[i] + (above[j] - top_left) for each j
    const __m512i result_16 = _mm512_add_epi16(left_val, above_diff);
    
    // Convert back to 8-bit with saturation
    const __m256i result_8 = _mm256_packus_epi16(
        _mm512_extracti64x4_epi64(result_16, 0),
        _mm512_extracti64x4_epi64(result_16, 1));
    
    _mm256_storeu_si256((__m256i *)(dst + i * stride), result_8);
  }
}

// Left DC Predictor for 32x32 blocks (when above is not available)
void vpx_dc_left_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above, const uint8_t *left) {
  __m512i sum = _mm512_setzero_si512();
  (void)above;
  
  // Load and sum left pixels (32 bytes)
  const __m256i left_vec = _mm256_loadu_si256((const __m256i *)left);
  const __m512i left_512 = _mm512_cvtepu8_epi16(left_vec);
  sum = _mm512_add_epi16(sum, left_512);
  
  // Horizontal reduction to get total sum
  const uint32_t total_sum = manual_reduce_epi16(sum);
  const uint8_t dc_val = (total_sum + 16) >> 5;  // Divide by 32 with rounding
  
  // Broadcast DC value to 512-bit vector
  const __m512i dc_vec = _mm512_set1_epi8(dc_val);
  
  // Fill 32x32 block
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512((__m512i *)(dst + i * stride), dc_vec);
  }
}

// Top DC Predictor for 32x32 blocks (when left is not available)
void vpx_dc_top_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                       const uint8_t *above, const uint8_t *left) {
  __m512i sum = _mm512_setzero_si512();
  (void)left;
  
  // Load and sum above pixels (32 bytes)
  const __m256i above_vec = _mm256_loadu_si256((const __m256i *)above);
  const __m512i above_512 = _mm512_cvtepu8_epi16(above_vec);
  sum = _mm512_add_epi16(sum, above_512);
  
  // Horizontal reduction to get total sum  
  const uint32_t total_sum = manual_reduce_epi16(sum);
  const uint8_t dc_val = (total_sum + 16) >> 5;  // Divide by 32 with rounding
  
  // Broadcast DC value to 512-bit vector
  const __m512i dc_vec = _mm512_set1_epi8(dc_val);
  
  // Fill 32x32 block
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512((__m512i *)(dst + i * stride), dc_vec);
  }
}

// 128 DC Predictor for 32x32 blocks (when neither above nor left is available)
void vpx_dc_128_predictor_32x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                       const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;
  
  // Fill with constant 128 value
  const __m512i dc_vec = _mm512_set1_epi8(128);
  
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512((__m512i *)(dst + i * stride), dc_vec);
  }
}

// Additional optimized predictor for 16x16 blocks using AVX-512
void vpx_dc_predictor_16x16_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  __m256i sum = _mm256_setzero_si256();
  
  // Load and sum above pixels (16 bytes)
  const __m128i above_vec = _mm_loadu_si128((const __m128i *)above);
  const __m256i above_256 = _mm256_cvtepu8_epi16(above_vec);
  sum = _mm256_add_epi16(sum, above_256);
  
  // Load and sum left pixels (16 bytes)
  const __m128i left_vec = _mm_loadu_si128((const __m128i *)left);
  const __m256i left_256 = _mm256_cvtepu8_epi16(left_vec);
  sum = _mm256_add_epi16(sum, left_256);
  
  // Horizontal reduction using AVX-512 reduction
  const __m512i sum_512 = _mm512_castsi256_si512(sum);
  const uint32_t total_sum = manual_reduce_epi16(sum_512);
  const uint8_t dc_val = (total_sum + 16) >> 5;  // Divide by 32 with rounding
  
  // Broadcast DC value to 128-bit vector (16 bytes)
  const __m128i dc_vec = _mm_set1_epi8(dc_val);
  
  // Fill 16x16 block
  for (int i = 0; i < 16; ++i) {
    _mm_storeu_si128((__m128i *)(dst + i * stride), dc_vec);
  }
}