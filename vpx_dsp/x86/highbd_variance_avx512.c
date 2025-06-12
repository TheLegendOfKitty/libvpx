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

#include "vpx_dsp/variance.h"
#include "vpx_dsp/x86/bitdepth_conversion_avx2.h"

// High bit-depth variance kernel using AVX-512
static INLINE void highbd_variance_kernel_avx512(const __m512i src, const __m512i ref,
                                                  __m512i *const sse, __m512i *const sum) {
  const __m512i diff = _mm512_sub_epi16(src, ref);
  *sum = _mm512_add_epi32(*sum, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(diff, 0)));
  *sum = _mm512_add_epi32(*sum, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(diff, 1)));
  
  // Calculate squared differences
  const __m512i diff_sq = _mm512_mullo_epi16(diff, diff);
  *sse = _mm512_add_epi32(*sse, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 0)));
  *sse = _mm512_add_epi32(*sse, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
}

// High bit-depth variance for 64-width blocks
static INLINE void highbd_variance64_avx512(const uint16_t *src, int src_stride,
                                             const uint16_t *ref, int ref_stride,
                                             int h, uint64_t *sse, int64_t *sum) {
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < 64; j += 16) {
      const __m512i v_s = _mm512_loadu_si512((const __m512i *)(src + j));
      const __m512i v_r = _mm512_loadu_si512((const __m512i *)(ref + j));
      highbd_variance_kernel_avx512(v_s, v_r, &vsse, &vsum);
    }
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = _mm512_reduce_add_epi32(vsse);
  *sum = _mm512_reduce_add_epi32(vsum);
}

// High bit-depth variance for 32-width blocks
static INLINE void highbd_variance32_avx512(const uint16_t *src, int src_stride,
                                             const uint16_t *ref, int ref_stride,
                                             int h, uint64_t *sse, int64_t *sum) {
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m512i v_s = _mm512_loadu_si512((const __m512i *)(src));
    const __m512i v_r = _mm512_loadu_si512((const __m512i *)(ref));
    highbd_variance_kernel_avx512(v_s, v_r, &vsse, &vsum);
    
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = _mm512_reduce_add_epi32(vsse);
  *sum = _mm512_reduce_add_epi32(vsum);
}

// High bit-depth variance for 16-width blocks
static INLINE void highbd_variance16_avx512(const uint16_t *src, int src_stride,
                                             const uint16_t *ref, int ref_stride,
                                             int h, uint64_t *sse, int64_t *sum) {
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m256i v_s_256 = _mm256_loadu_si256((const __m256i *)(src));
    const __m256i v_r_256 = _mm256_loadu_si256((const __m256i *)(ref));
    const __m512i v_s = _mm512_cvtepu16_epi32(v_s_256);
    const __m512i v_r = _mm512_cvtepu16_epi32(v_r_256);
    
    const __m512i diff = _mm512_sub_epi32(v_s, v_r);
    vsum = _mm512_add_epi32(vsum, diff);
    
    const __m512i diff_sq = _mm512_mullo_epi32(diff, diff);
    vsse = _mm512_add_epi32(vsse, diff_sq);
    
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = _mm512_reduce_add_epi32(vsse);
  *sum = _mm512_reduce_add_epi32(vsum);
}

// High bit-depth variance for 8-width blocks
static INLINE void highbd_variance8_avx512(const uint16_t *src, int src_stride,
                                            const uint16_t *ref, int ref_stride,
                                            int h, uint64_t *sse, int64_t *sum) {
  __m512i vsse = _mm512_setzero_si512();
  __m512i vsum = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m128i v_s_128 = _mm_loadu_si128((const __m128i *)(src));
    const __m128i v_r_128 = _mm_loadu_si128((const __m128i *)(ref));
    const __m256i v_s_256 = _mm256_cvtepu16_epi32(v_s_128);
    const __m256i v_r_256 = _mm256_cvtepu16_epi32(v_r_128);
    const __m512i v_s = _mm512_cvtepi32_epi64(v_s_256);
    const __m512i v_r = _mm512_cvtepi32_epi64(v_r_256);
    
    const __m512i diff = _mm512_sub_epi64(v_s, v_r);
    vsum = _mm512_add_epi64(vsum, diff);
    
    const __m512i diff_sq = _mm512_mullo_epi64(diff, diff);
    vsse = _mm512_add_epi64(vsse, diff_sq);
    
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = _mm512_reduce_add_epi64(vsse);
  *sum = _mm512_reduce_add_epi64(vsum);
}

// High bit-depth variance implementations
uint32_t vpx_highbd_variance64x64_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance64_avx512(src, src_stride, ref, ref_stride, 64, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 12);
}

uint32_t vpx_highbd_variance64x32_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance64_avx512(src, src_stride, ref, ref_stride, 32, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 11);
}

uint32_t vpx_highbd_variance32x64_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance32_avx512(src, src_stride, ref, ref_stride, 64, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 11);
}

uint32_t vpx_highbd_variance32x32_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance32_avx512(src, src_stride, ref, ref_stride, 32, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 10);
}

uint32_t vpx_highbd_variance32x16_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance32_avx512(src, src_stride, ref, ref_stride, 16, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 9);
}

uint32_t vpx_highbd_variance16x32_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance16_avx512(src, src_stride, ref, ref_stride, 32, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 9);
}

uint32_t vpx_highbd_variance16x16_avx512(const uint8_t *src8, int src_stride,
                                          const uint8_t *ref8, int ref_stride,
                                          uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance16_avx512(src, src_stride, ref, ref_stride, 16, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 8);
}

uint32_t vpx_highbd_variance16x8_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride,
                                         uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance16_avx512(src, src_stride, ref, ref_stride, 8, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 7);
}

uint32_t vpx_highbd_variance8x16_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride,
                                         uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance8_avx512(src, src_stride, ref, ref_stride, 16, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 7);
}

uint32_t vpx_highbd_variance8x8_avx512(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride,
                                        uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance8_avx512(src, src_stride, ref, ref_stride, 8, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 6);
}

uint32_t vpx_highbd_variance8x4_avx512(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride,
                                        uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance8_avx512(src, src_stride, ref, ref_stride, 4, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 5);
}

uint32_t vpx_highbd_variance4x8_avx512(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride,
                                        uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64 = 0;
  int64_t sum64 = 0;
  
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int32_t diff = src[j] - ref[j];
      sum64 += diff;
      sse64 += diff * diff;
    }
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 5);
}

uint32_t vpx_highbd_variance4x4_avx512(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride,
                                        uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64 = 0;
  int64_t sum64 = 0;
  
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int32_t diff = src[j] - ref[j];
      sum64 += diff;
      sse64 += diff * diff;
    }
    src += src_stride;
    ref += ref_stride;
  }
  
  *sse = (uint32_t)sse64;
  return *sse - (uint32_t)(((int64_t)sum64 * sum64) >> 4);
}

// High bit-depth MSE functions
uint32_t vpx_highbd_mse16x16_avx512(const uint8_t *src8, int src_stride,
                                     const uint8_t *ref8, int ref_stride,
                                     uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance16_avx512(src, src_stride, ref, ref_stride, 16, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse;
}

uint32_t vpx_highbd_mse16x8_avx512(const uint8_t *src8, int src_stride,
                                    const uint8_t *ref8, int ref_stride,
                                    uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance16_avx512(src, src_stride, ref, ref_stride, 8, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse;
}

uint32_t vpx_highbd_mse8x16_avx512(const uint8_t *src8, int src_stride,
                                    const uint8_t *ref8, int ref_stride,
                                    uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance8_avx512(src, src_stride, ref, ref_stride, 16, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse;
}

uint32_t vpx_highbd_mse8x8_avx512(const uint8_t *src8, int src_stride,
                                   const uint8_t *ref8, int ref_stride,
                                   uint32_t *sse) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  uint64_t sse64;
  int64_t sum64;
  
  highbd_variance8_avx512(src, src_stride, ref, ref_stride, 8, &sse64, &sum64);
  *sse = (uint32_t)sse64;
  return *sse;
}