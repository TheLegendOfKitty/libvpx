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

#include "vpx_dsp/variance.h"

// Manual reduction for 16-bit values since _mm512_reduce_add_epi16 doesn't exist
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

static INLINE void variance_kernel_avx512(const __m512i src, const __m512i ref,
                                          __m512i *const sse,
                                          __m512i *const sum) {
  const __m512i adj_sub = _mm512_set1_epi16((short)0xff01);  // (1,-1)
  
  // unpack into pairs and subtract  
  const __m512i src_ref0 = _mm512_unpacklo_epi8(src, ref);
  const __m512i src_ref1 = _mm512_unpackhi_epi8(src, ref);
  const __m512i diff0 = _mm512_maddubs_epi16(src_ref0, adj_sub);
  const __m512i diff1 = _mm512_maddubs_epi16(src_ref1, adj_sub);
  
  // accumulate squared differences and sums
  const __m512i madd0 = _mm512_madd_epi16(diff0, diff0);
  const __m512i madd1 = _mm512_madd_epi16(diff1, diff1);
  *sum = _mm512_add_epi16(*sum, _mm512_add_epi16(diff0, diff1));
  *sse = _mm512_add_epi32(*sse, _mm512_add_epi32(madd0, madd1));
}

static INLINE void variance64_avx512(const uint8_t *src, int src_stride,
                                     const uint8_t *ref, int ref_stride,
                                     int h, uint64_t *sse, int64_t *sum) {
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m512i v_s = _mm512_loadu_si512((const __m512i *)(src));
    const __m512i v_r = _mm512_loadu_si512((const __m512i *)(ref));
    variance_kernel_avx512(v_s, v_r, &vsse, &vsum);
    src += src_stride;
    ref += ref_stride;
  }
  
  // Horizontal reduction for SSE (32-bit values)
  *sse = _mm512_reduce_add_epi32(vsse);
  
  // Horizontal reduction for sum (16-bit values) - manual since reduce_add_epi16 doesn't exist
  __m256i sum_256 = _mm512_extracti64x4_epi64(vsum, 0);
  sum_256 = _mm256_add_epi16(sum_256, _mm512_extracti64x4_epi64(vsum, 1));
  __m128i sum_128 = _mm256_extracti128_si256(sum_256, 0);
  sum_128 = _mm_add_epi16(sum_128, _mm256_extracti128_si256(sum_256, 1));
  sum_128 = _mm_add_epi16(sum_128, _mm_srli_si128(sum_128, 8));
  sum_128 = _mm_add_epi16(sum_128, _mm_srli_si128(sum_128, 4));
  sum_128 = _mm_add_epi16(sum_128, _mm_srli_si128(sum_128, 2));
  *sum = _mm_extract_epi16(sum_128, 0);
}

unsigned int vpx_variance64x64_avx512(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      unsigned int *sse) {
  uint64_t sse64;
  int64_t sum64;
  variance64_avx512(src, src_stride, ref, ref_stride, 64, &sse64, &sum64);
  *sse = (unsigned int)sse64;
  return *sse - (unsigned int)(((int64_t)sum64 * sum64) >> 12);
}

unsigned int vpx_variance64x32_avx512(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      unsigned int *sse) {
  uint64_t sse64;
  int64_t sum64;
  variance64_avx512(src, src_stride, ref, ref_stride, 32, &sse64, &sum64);
  *sse = (unsigned int)sse64;
  return *sse - (unsigned int)(((int64_t)sum64 * sum64) >> 11);
}

unsigned int vpx_variance32x64_avx512(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      unsigned int *sse) {
  // Process two 32-byte chunks per row
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < 64; ++i) {
    // Load 32 bytes by combining two 16-byte loads into one 512-bit vector
    const __m256i v_s0 = _mm256_loadu_si256((const __m256i *)(src));
    const __m256i v_r0 = _mm256_loadu_si256((const __m256i *)(ref));
    const __m512i v_s = _mm512_castsi256_si512(v_s0);
    const __m512i v_r = _mm512_castsi256_si512(v_r0);
    
    variance_kernel_avx512(v_s, v_r, &vsse, &vsum);
    src += src_stride;
    ref += ref_stride;
  }
  
  uint64_t sse64 = _mm512_reduce_add_epi32(vsse);
  int64_t sum64 = manual_reduce_epi16(vsum);
  
  *sse = (unsigned int)sse64;
  return *sse - (unsigned int)(((int64_t)sum64 * sum64) >> 11);
}

unsigned int vpx_variance32x32_avx512(const uint8_t *src, int src_stride,
                                      const uint8_t *ref, int ref_stride,
                                      unsigned int *sse) {
  // Process two 32-byte chunks per row
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < 32; ++i) {
    const __m256i v_s0 = _mm256_loadu_si256((const __m256i *)(src));
    const __m256i v_r0 = _mm256_loadu_si256((const __m256i *)(ref));
    const __m512i v_s = _mm512_castsi256_si512(v_s0);
    const __m512i v_r = _mm512_castsi256_si512(v_r0);
    
    variance_kernel_avx512(v_s, v_r, &vsse, &vsum);
    src += src_stride;
    ref += ref_stride;
  }
  
  uint64_t sse64 = _mm512_reduce_add_epi32(vsse);
  int64_t sum64 = manual_reduce_epi16(vsum);
  
  *sse = (unsigned int)sse64;
  return *sse - (unsigned int)(((int64_t)sum64 * sum64) >> 10);
}

static INLINE void sub_pixel_variance64x_h_avx512(
    const uint8_t *src, int src_stride, int x_offset, const uint8_t *ref,
    int ref_stride, int height, unsigned int *sse, int *sum) {
  
  // Simple filter coefficients for AVX-512
  const uint8_t filter1 = 16 - x_offset * 2;
  const uint8_t filter2 = x_offset * 2;
  const __m512i filter = _mm512_set1_epi16(
      (filter2 << 8) + filter1);
  const __m512i pw_8 = _mm512_set1_epi16(8);
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < height; i++) {
    // Load 64+1 bytes for horizontal filtering
    const __m512i src0 = _mm512_loadu_si512((const __m512i *)(src));
    const __m512i src1 = _mm512_loadu_si512((const __m512i *)(src + 1));
    
    // Apply horizontal bilinear filter
    const __m512i src_lo = _mm512_unpacklo_epi8(src0, src1);
    const __m512i src_hi = _mm512_unpackhi_epi8(src0, src1);
    
    const __m512i filtered_lo = _mm512_maddubs_epi16(src_lo, filter);
    const __m512i filtered_hi = _mm512_maddubs_epi16(src_hi, filter);
    
    const __m512i filtered = _mm512_packus_epi16(
        _mm512_srli_epi16(_mm512_add_epi16(filtered_lo, pw_8), 4),
        _mm512_srli_epi16(_mm512_add_epi16(filtered_hi, pw_8), 4));
    
    const __m512i ref_vec = _mm512_loadu_si512((const __m512i *)(ref));
    
    variance_kernel_avx512(filtered, ref_vec, &vsse, &vsum);
    
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = _mm512_reduce_add_epi32(vsse);
  *sum = manual_reduce_epi16(vsum);
}

unsigned int vpx_sub_pixel_variance64x64_avx512(const uint8_t *src,
                                                int src_stride,
                                                int x_offset, int y_offset,
                                                const uint8_t *ref,
                                                int ref_stride,
                                                unsigned int *sse) {
  unsigned int se;
  int s;
  if (x_offset == 0) {
    if (y_offset == 0) {
      return vpx_variance64x64_avx512(src, src_stride, ref, ref_stride, sse);
    } else {
      // Vertical subpixel - can be optimized with AVX-512
      return vpx_sub_pixel_variance64x64_c(src, src_stride, x_offset, y_offset,
                                           ref, ref_stride, sse);
    }
  } else {
    if (y_offset == 0) {
      sub_pixel_variance64x_h_avx512(src, src_stride, x_offset, ref, ref_stride,
                                     64, &se, &s);
      *sse = se;
      return se - (unsigned int)(((int64_t)s * s) >> 12);
    } else {
      // Both horizontal and vertical subpixel
      return vpx_sub_pixel_variance64x64_c(src, src_stride, x_offset, y_offset,
                                           ref, ref_stride, sse);
    }
  }
}

unsigned int vpx_sub_pixel_variance64x32_avx512(const uint8_t *src,
                                                int src_stride,
                                                int x_offset, int y_offset,
                                                const uint8_t *ref,
                                                int ref_stride,
                                                unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance64x32_avx512(src, src_stride, ref, ref_stride, sse);
  } else if (y_offset == 0) {
    unsigned int se;
    int s;
    sub_pixel_variance64x_h_avx512(src, src_stride, x_offset, ref, ref_stride,
                                   32, &se, &s);
    *sse = se;
    return se - (unsigned int)(((int64_t)s * s) >> 11);
  } else {
    return vpx_sub_pixel_variance64x32_c(src, src_stride, x_offset, y_offset,
                                         ref, ref_stride, sse);
  }
}

// Additional MSE Functions with AVX-512 optimization
unsigned int vpx_mse16x16_avx512(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride,
                                 unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  
  for (int i = 0; i < 16; ++i) {
    // Load 16 bytes from source and reference
    const __m128i src = _mm_loadu_si128((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadu_si128((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Convert to 512-bit for processing
    const __m512i src_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    
    // Calculate difference and square
    const __m512i diff = _mm512_sub_epi16(src_512, ref_512);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    // Accumulate squared differences
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    const __m512i squared_hi = _mm512_unpackhi_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, _mm512_add_epi32(squared_lo, squared_hi));
  }
  
  *sse = _mm512_reduce_add_epi32(sum_sse);
  return *sse;
}

unsigned int vpx_mse16x8_avx512(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride,
                                unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  
  for (int i = 0; i < 8; ++i) {
    // Load 16 bytes from source and reference
    const __m128i src = _mm_loadu_si128((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadu_si128((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Convert to 512-bit for processing
    const __m512i src_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    
    // Calculate difference and square
    const __m512i diff = _mm512_sub_epi16(src_512, ref_512);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    // Accumulate squared differences
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    const __m512i squared_hi = _mm512_unpackhi_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, _mm512_add_epi32(squared_lo, squared_hi));
  }
  
  *sse = _mm512_reduce_add_epi32(sum_sse);
  return *sse;
}

unsigned int vpx_mse8x16_avx512(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride,
                                unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  
  for (int i = 0; i < 16; ++i) {
    // Load 8 bytes from source and reference
    const __m128i src = _mm_loadl_epi64((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadl_epi64((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Convert to 512-bit for processing
    const __m512i src_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    
    // Calculate difference and square
    const __m512i diff = _mm512_sub_epi16(src_512, ref_512);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    // Accumulate squared differences (only first 8 elements are valid)
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, squared_lo);
  }
  
  *sse = _mm512_reduce_add_epi32(sum_sse);
  return *sse;
}

unsigned int vpx_mse8x8_avx512(const uint8_t *src_ptr, int src_stride,
                               const uint8_t *ref_ptr, int ref_stride,
                               unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  
  for (int i = 0; i < 8; ++i) {
    // Load 8 bytes from source and reference
    const __m128i src = _mm_loadl_epi64((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadl_epi64((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Convert to 512-bit for processing
    const __m512i src_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_512 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    
    // Calculate difference and square
    const __m512i diff = _mm512_sub_epi16(src_512, ref_512);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    // Accumulate squared differences (only first 8 elements are valid)
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, squared_lo);
  }
  
  *sse = _mm512_reduce_add_epi32(sum_sse);
  return *sse;
}

// Additional variance functions for missing block sizes
unsigned int vpx_variance16x16_avx512(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride,
                                      unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  __m512i sum_diff = _mm512_setzero_si512();
  
  for (int i = 0; i < 16; ++i) {
    // Load 16 bytes from source and reference
    const __m128i src = _mm_loadu_si128((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadu_si128((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Use SAD for efficient difference calculation
    const __m512i src_512 = _mm512_castsi128_si512(src);
    const __m512i ref_512 = _mm512_castsi128_si512(ref);
    
    // Calculate SAD and accumulate
    sum_diff = _mm512_add_epi64(sum_diff, _mm512_sad_epu8(src_512, ref_512));
    
    // Calculate SSE
    const __m512i src_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    const __m512i diff = _mm512_sub_epi16(src_16, ref_16);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    const __m512i squared_hi = _mm512_unpackhi_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, _mm512_add_epi32(squared_lo, squared_hi));
  }
  
  const int64_t sum = _mm512_reduce_add_epi64(sum_diff);
  *sse = _mm512_reduce_add_epi32(sum_sse);
  
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 8);  // Divide by 256
}

unsigned int vpx_variance16x32_avx512(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride,
                                      unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  __m512i sum_diff = _mm512_setzero_si512();
  
  for (int i = 0; i < 32; ++i) {
    // Load 16 bytes from source and reference
    const __m128i src = _mm_loadu_si128((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadu_si128((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Use SAD for efficient difference calculation
    const __m512i src_512 = _mm512_castsi128_si512(src);
    const __m512i ref_512 = _mm512_castsi128_si512(ref);
    
    // Calculate SAD and accumulate
    sum_diff = _mm512_add_epi64(sum_diff, _mm512_sad_epu8(src_512, ref_512));
    
    // Calculate SSE
    const __m512i src_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    const __m512i diff = _mm512_sub_epi16(src_16, ref_16);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    const __m512i squared_hi = _mm512_unpackhi_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, _mm512_add_epi32(squared_lo, squared_hi));
  }
  
  const int64_t sum = _mm512_reduce_add_epi64(sum_diff);
  *sse = _mm512_reduce_add_epi32(sum_sse);
  
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 9);  // Divide by 512
}

// Additional subpixel variance functions
unsigned int vpx_sub_pixel_variance32x32_avx512(const uint8_t *src,
                                                int src_stride,
                                                int x_offset, int y_offset,
                                                const uint8_t *ref,
                                                int ref_stride,
                                                unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance32x32_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance32x32_c(src, src_stride, x_offset, y_offset,
                                         ref, ref_stride, sse);
  }
}

unsigned int vpx_sub_pixel_variance16x16_avx512(const uint8_t *src,
                                                int src_stride,
                                                int x_offset, int y_offset,
                                                const uint8_t *ref,
                                                int ref_stride,
                                                unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance16x16_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance16x16_c(src, src_stride, x_offset, y_offset,
                                         ref, ref_stride, sse);
  }
}

// Additional variance functions for missing block sizes
unsigned int vpx_variance8x16_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  __m512i sum_diff = _mm512_setzero_si512();
  
  for (int i = 0; i < 16; ++i) {
    // Load 8 bytes from source and reference
    const __m128i src = _mm_loadl_epi64((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadl_epi64((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Use SAD for efficient difference calculation
    const __m512i src_512 = _mm512_castsi128_si512(src);
    const __m512i ref_512 = _mm512_castsi128_si512(ref);
    
    // Calculate SAD and accumulate
    sum_diff = _mm512_add_epi64(sum_diff, _mm512_sad_epu8(src_512, ref_512));
    
    // Calculate SSE
    const __m512i src_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    const __m512i diff = _mm512_sub_epi16(src_16, ref_16);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, squared_lo);
  }
  
  const int64_t sum = _mm512_reduce_add_epi64(sum_diff);
  *sse = _mm512_reduce_add_epi32(sum_sse);
  
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 7);  // Divide by 128
}

unsigned int vpx_variance8x8_avx512(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  __m512i sum_diff = _mm512_setzero_si512();
  
  for (int i = 0; i < 8; ++i) {
    // Load 8 bytes from source and reference
    const __m128i src = _mm_loadl_epi64((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadl_epi64((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Use SAD for efficient difference calculation
    const __m512i src_512 = _mm512_castsi128_si512(src);
    const __m512i ref_512 = _mm512_castsi128_si512(ref);
    
    // Calculate SAD and accumulate
    sum_diff = _mm512_add_epi64(sum_diff, _mm512_sad_epu8(src_512, ref_512));
    
    // Calculate SSE
    const __m512i src_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    const __m512i diff = _mm512_sub_epi16(src_16, ref_16);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, squared_lo);
  }
  
  const int64_t sum = _mm512_reduce_add_epi64(sum_diff);
  *sse = _mm512_reduce_add_epi32(sum_sse);
  
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 6);  // Divide by 64
}

unsigned int vpx_variance8x4_avx512(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m512i sum_sse = _mm512_setzero_si512();
  __m512i sum_diff = _mm512_setzero_si512();
  
  for (int i = 0; i < 4; ++i) {
    // Load 8 bytes from source and reference
    const __m128i src = _mm_loadl_epi64((const __m128i *)(src_ptr + i * src_stride));
    const __m128i ref = _mm_loadl_epi64((const __m128i *)(ref_ptr + i * ref_stride));
    
    // Use SAD for efficient difference calculation
    const __m512i src_512 = _mm512_castsi128_si512(src);
    const __m512i ref_512 = _mm512_castsi128_si512(ref);
    
    // Calculate SAD and accumulate
    sum_diff = _mm512_add_epi64(sum_diff, _mm512_sad_epu8(src_512, ref_512));
    
    // Calculate SSE
    const __m512i src_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(src));
    const __m512i ref_16 = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(ref));
    const __m512i diff = _mm512_sub_epi16(src_16, ref_16);
    const __m512i squared = _mm512_mullo_epi16(diff, diff);
    
    const __m512i squared_lo = _mm512_unpacklo_epi16(squared, _mm512_setzero_si512());
    sum_sse = _mm512_add_epi32(sum_sse, squared_lo);
  }
  
  const int64_t sum = _mm512_reduce_add_epi64(sum_diff);
  *sse = _mm512_reduce_add_epi32(sum_sse);
  
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 5);  // Divide by 32
}

unsigned int vpx_variance4x8_avx512(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  unsigned int sum = 0;
  unsigned int sum_sq = 0;
  
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int diff = src_ptr[i * src_stride + j] - ref_ptr[i * ref_stride + j];
      sum += diff;
      sum_sq += diff * diff;
    }
  }
  
  *sse = sum_sq;
  return sum_sq - (unsigned int)(((int64_t)sum * sum) >> 5);  // Divide by 32
}

unsigned int vpx_variance4x4_avx512(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  unsigned int sum = 0;
  unsigned int sum_sq = 0;
  
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int diff = src_ptr[i * src_stride + j] - ref_ptr[i * ref_stride + j];
      sum += diff;
      sum_sq += diff * diff;
    }
  }
  
  *sse = sum_sq;
  return sum_sq - (unsigned int)(((int64_t)sum * sum) >> 4);  // Divide by 16
}

// Additional subpixel variance functions
unsigned int vpx_sub_pixel_variance8x16_avx512(const uint8_t *src,
                                               int src_stride,
                                               int x_offset, int y_offset,
                                               const uint8_t *ref,
                                               int ref_stride,
                                               unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance8x16_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance8x16_c(src, src_stride, x_offset, y_offset,
                                        ref, ref_stride, sse);
  }
}

unsigned int vpx_sub_pixel_variance8x8_avx512(const uint8_t *src,
                                              int src_stride,
                                              int x_offset, int y_offset,
                                              const uint8_t *ref,
                                              int ref_stride,
                                              unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance8x8_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance8x8_c(src, src_stride, x_offset, y_offset,
                                       ref, ref_stride, sse);
  }
}

unsigned int vpx_sub_pixel_variance8x4_avx512(const uint8_t *src,
                                              int src_stride,
                                              int x_offset, int y_offset,
                                              const uint8_t *ref,
                                              int ref_stride,
                                              unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance8x4_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance8x4_c(src, src_stride, x_offset, y_offset,
                                       ref, ref_stride, sse);
  }
}

unsigned int vpx_sub_pixel_variance4x8_avx512(const uint8_t *src,
                                              int src_stride,
                                              int x_offset, int y_offset,
                                              const uint8_t *ref,
                                              int ref_stride,
                                              unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance4x8_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance4x8_c(src, src_stride, x_offset, y_offset,
                                       ref, ref_stride, sse);
  }
}

unsigned int vpx_sub_pixel_variance4x4_avx512(const uint8_t *src,
                                              int src_stride,
                                              int x_offset, int y_offset,
                                              const uint8_t *ref,
                                              int ref_stride,
                                              unsigned int *sse) {
  if (x_offset == 0 && y_offset == 0) {
    return vpx_variance4x4_avx512(src, src_stride, ref, ref_stride, sse);
  } else {
    // Fall back to reference implementation for subpixel cases
    return vpx_sub_pixel_variance4x4_c(src, src_stride, x_offset, y_offset,
                                       ref, ref_stride, sse);
  }
}