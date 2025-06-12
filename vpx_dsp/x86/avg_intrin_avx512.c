/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX-512

#include "./vpx_dsp_rtcd.h"

unsigned int vpx_avg_8x8_avx512(const uint8_t *s, int p) {
  __m512i sum = _mm512_setzero_si512();
  
  // Load 8 rows of 8 bytes each and accumulate
  for (int i = 0; i < 8; ++i) {
    // Load 8 bytes and broadcast to fill 64 bytes for processing
    const __m128i row = _mm_loadl_epi64((const __m128i *)(s + i * p));
    const __m512i row_512 = _mm512_broadcastq_epi64(row);
    
    // Add to sum using SAD instruction for efficient horizontal sum
    sum = _mm512_add_epi64(sum, _mm512_sad_epu8(row_512, _mm512_setzero_si512()));
  }
  
  // Horizontal reduction of the sum
  const __m256i sum_256 = _mm256_add_epi64(_mm512_extracti64x4_epi64(sum, 0),
                                           _mm512_extracti64x4_epi64(sum, 1));
  const __m128i sum_128 = _mm_add_epi64(_mm256_extracti128_si256(sum_256, 0),
                                        _mm256_extracti128_si256(sum_256, 1));
  const uint64_t total = _mm_extract_epi64(sum_128, 0) + _mm_extract_epi64(sum_128, 1);
  
  return (unsigned int)((total + 32) >> 6);  // Divide by 64 with rounding
}

unsigned int vpx_avg_4x4_avx512(const uint8_t *s, int p) {
  __m512i sum = _mm512_setzero_si512();
  
  // Load 4 rows of 4 bytes each
  for (int i = 0; i < 4; ++i) {
    // Load 4 bytes and broadcast for processing
    const uint32_t row_data = *(const uint32_t *)(s + i * p);
    const __m512i row = _mm512_set1_epi32(row_data);
    
    // Add to sum using SAD instruction
    sum = _mm512_add_epi64(sum, _mm512_sad_epu8(row, _mm512_setzero_si512()));
  }
  
  // Horizontal reduction of the sum  
  const __m256i sum_256 = _mm256_add_epi64(_mm512_extracti64x4_epi64(sum, 0),
                                           _mm512_extracti64x4_epi64(sum, 1));
  const __m128i sum_128 = _mm_add_epi64(_mm256_extracti128_si256(sum_256, 0),
                                        _mm256_extracti128_si256(sum_256, 1));
  const uint64_t total = _mm_extract_epi64(sum_128, 0) + _mm_extract_epi64(sum_128, 1);
  
  return (unsigned int)((total + 8) >> 4);  // Divide by 16 with rounding
}

unsigned int vpx_get_mb_ss_avx512(const int16_t *src) {
  __m512i sum = _mm512_setzero_si512();
  
  // Process 256 16-bit values (16x16 macroblock) in chunks of 32
  for (int i = 0; i < 256; i += 32) {
    // Load 32 16-bit values
    const __m512i data = _mm512_loadu_si512((const __m512i *)(src + i));
    
    // Square each value: convert to 32-bit, square, and accumulate
    const __m512i data_lo = _mm512_unpacklo_epi16(data, _mm512_setzero_si512());
    const __m512i data_hi = _mm512_unpackhi_epi16(data, _mm512_setzero_si512());
    
    const __m512i squared_lo = _mm512_mullo_epi32(data_lo, data_lo);
    const __m512i squared_hi = _mm512_mullo_epi32(data_hi, data_hi);
    
    // Accumulate squares
    sum = _mm512_add_epi32(sum, squared_lo);
    sum = _mm512_add_epi32(sum, squared_hi);
  }
  
  // Horizontal reduction
  return _mm512_reduce_add_epi32(sum);
}

// Optimized version for 64-byte aligned averaging operations
static INLINE unsigned int avg_block_avx512(const uint8_t *s, int stride,
                                            int width, int height) {
  __m512i sum = _mm512_setzero_si512();
  
  if (width == 64) {
    // Process full 64-byte rows
    for (int i = 0; i < height; ++i) {
      const __m512i row = _mm512_loadu_si512((const __m512i *)(s + i * stride));
      sum = _mm512_add_epi64(sum, _mm512_sad_epu8(row, _mm512_setzero_si512()));
    }
  } else if (width == 32) {
    // Process 32-byte rows
    for (int i = 0; i < height; ++i) {
      const __m256i row = _mm256_loadu_si256((const __m256i *)(s + i * stride));
      const __m512i row_512 = _mm512_castsi256_si512(row);
      sum = _mm512_add_epi64(sum, _mm512_sad_epu8(row_512, _mm512_setzero_si512()));
    }
  } else if (width == 16) {
    // Process 16-byte rows
    for (int i = 0; i < height; ++i) {
      const __m128i row = _mm_loadu_si128((const __m128i *)(s + i * stride));
      const __m512i row_512 = _mm512_castsi128_si512(row);
      sum = _mm512_add_epi64(sum, _mm512_sad_epu8(row_512, _mm512_setzero_si512()));
    }
  }
  
  // Horizontal reduction
  const unsigned int total_pixels = width * height;
  const uint64_t total_sum = _mm512_reduce_add_epi64(sum);
  
  return (unsigned int)((total_sum + (total_pixels >> 1)) / total_pixels);
}

// Extended averaging functions for larger blocks
unsigned int vpx_avg_16x16_avx512(const uint8_t *s, int p) {
  return avg_block_avx512(s, p, 16, 16);
}

unsigned int vpx_avg_32x32_avx512(const uint8_t *s, int p) {
  return avg_block_avx512(s, p, 32, 32);
}

unsigned int vpx_avg_64x64_avx512(const uint8_t *s, int p) {
  return avg_block_avx512(s, p, 64, 64);
}