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

#include "vpx_dsp/x86/bitdepth_conversion_avx2.h"

// High bit-depth SAD for 64-width blocks using AVX-512
static INLINE unsigned int highbd_sad64xh_avx512(const uint16_t *src_ptr,
                                                  int src_stride,
                                                  const uint16_t *ref_ptr,
                                                  int ref_stride, int h) {
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < 64; j += 16) {
      const __m512i src = _mm512_loadu_si512((const __m512i *)(src_ptr + j));
      const __m512i ref = _mm512_loadu_si512((const __m512i *)(ref_ptr + j));
      
      // Calculate absolute differences for 16-bit values
      const __m512i diff = _mm512_abs_epi16(_mm512_sub_epi16(src, ref));
      
      // Accumulate differences by converting to 32-bit and adding
      const __m512i diff_lo = _mm512_unpacklo_epi16(diff, _mm512_setzero_si512());
      const __m512i diff_hi = _mm512_unpackhi_epi16(diff, _mm512_setzero_si512());
      
      sad_acc = _mm512_add_epi32(sad_acc, diff_lo);
      sad_acc = _mm512_add_epi32(sad_acc, diff_hi);
    }
    
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  
  return _mm512_reduce_add_epi32(sad_acc);
}

// High bit-depth SAD for 32-width blocks using AVX-512
static INLINE unsigned int highbd_sad32xh_avx512(const uint16_t *src_ptr,
                                                  int src_stride,
                                                  const uint16_t *ref_ptr,
                                                  int ref_stride, int h) {
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m512i src = _mm512_loadu_si512((const __m512i *)(src_ptr));
    const __m512i ref = _mm512_loadu_si512((const __m512i *)(ref_ptr));
    
    // Calculate absolute differences
    const __m512i diff = _mm512_abs_epi16(_mm512_sub_epi16(src, ref));
    
    // Accumulate differences
    const __m512i diff_lo = _mm512_unpacklo_epi16(diff, _mm512_setzero_si512());
    const __m512i diff_hi = _mm512_unpackhi_epi16(diff, _mm512_setzero_si512());
    
    sad_acc = _mm512_add_epi32(sad_acc, diff_lo);
    sad_acc = _mm512_add_epi32(sad_acc, diff_hi);
    
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  
  return _mm512_reduce_add_epi32(sad_acc);
}

// High bit-depth SAD for 16-width blocks using AVX-512
static INLINE unsigned int highbd_sad16xh_avx512(const uint16_t *src_ptr,
                                                  int src_stride,
                                                  const uint16_t *ref_ptr,
                                                  int ref_stride, int h) {
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m256i src = _mm256_loadu_si256((const __m256i *)(src_ptr));
    const __m256i ref = _mm256_loadu_si256((const __m256i *)(ref_ptr));
    
    // Calculate absolute differences
    const __m256i diff = _mm256_abs_epi16(_mm256_sub_epi16(src, ref));
    
    // Convert to 32-bit and accumulate
    const __m512i diff_32 = _mm512_cvtepu16_epi32(diff);
    sad_acc = _mm512_add_epi32(sad_acc, diff_32);
    
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  
  return _mm512_reduce_add_epi32(sad_acc);
}

// High bit-depth SAD for 8-width blocks using AVX-512
static INLINE unsigned int highbd_sad8xh_avx512(const uint16_t *src_ptr,
                                                 int src_stride,
                                                 const uint16_t *ref_ptr,
                                                 int ref_stride, int h) {
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < h; ++i) {
    const __m128i src = _mm_loadu_si128((const __m128i *)(src_ptr));
    const __m128i ref = _mm_loadu_si128((const __m128i *)(ref_ptr));
    
    // Calculate absolute differences
    const __m128i diff = _mm_abs_epi16(_mm_sub_epi16(src, ref));
    
    // Convert to 32-bit and accumulate
    const __m256i diff_32 = _mm256_cvtepu16_epi32(diff);
    const __m512i diff_512 = _mm512_cvtepu32_epi64(diff_32);
    sad_acc = _mm512_add_epi64(sad_acc, diff_512);
    
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  
  return _mm512_reduce_add_epi64(sad_acc);
}

// High bit-depth SAD implementations
unsigned int vpx_highbd_sad64x64_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad64xh_avx512(src, src_stride, ref, ref_stride, 64);
}

unsigned int vpx_highbd_sad64x32_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad64xh_avx512(src, src_stride, ref, ref_stride, 32);
}

unsigned int vpx_highbd_sad32x64_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad32xh_avx512(src, src_stride, ref, ref_stride, 64);
}

unsigned int vpx_highbd_sad32x32_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad32xh_avx512(src, src_stride, ref, ref_stride, 32);
}

unsigned int vpx_highbd_sad32x16_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad32xh_avx512(src, src_stride, ref, ref_stride, 16);
}

unsigned int vpx_highbd_sad16x32_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad16xh_avx512(src, src_stride, ref, ref_stride, 32);
}

unsigned int vpx_highbd_sad16x16_avx512(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad16xh_avx512(src, src_stride, ref, ref_stride, 16);
}

unsigned int vpx_highbd_sad16x8_avx512(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad16xh_avx512(src, src_stride, ref, ref_stride, 8);
}

unsigned int vpx_highbd_sad8x16_avx512(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad8xh_avx512(src, src_stride, ref, ref_stride, 16);
}

unsigned int vpx_highbd_sad8x8_avx512(const uint8_t *src8, int src_stride,
                                       const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad8xh_avx512(src, src_stride, ref, ref_stride, 8);
}

unsigned int vpx_highbd_sad8x4_avx512(const uint8_t *src8, int src_stride,
                                       const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  return highbd_sad8xh_avx512(src, src_stride, ref, ref_stride, 4);
}

unsigned int vpx_highbd_sad4x8_avx512(const uint8_t *src8, int src_stride,
                                       const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  unsigned int sad = 0;
  
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      sad += abs(src[j] - ref[j]);
    }
    src += src_stride;
    ref += ref_stride;
  }
  
  return sad;
}

unsigned int vpx_highbd_sad4x4_avx512(const uint8_t *src8, int src_stride,
                                       const uint8_t *ref8, int ref_stride) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  unsigned int sad = 0;
  
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      sad += abs(src[j] - ref[j]);
    }
    src += src_stride;
    ref += ref_stride;
  }
  
  return sad;
}

// High bit-depth SAD averaging implementations
unsigned int vpx_highbd_sad64x64_avg_avx512(const uint8_t *src8, int src_stride,
                                             const uint8_t *ref8, int ref_stride,
                                             const uint8_t *second_pred8) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  const uint16_t *second_pred = CONVERT_TO_SHORTPTR(second_pred8);
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; j += 16) {
      const __m512i src_vec = _mm512_loadu_si512((const __m512i *)(src + j));
      const __m512i ref_vec = _mm512_loadu_si512((const __m512i *)(ref + j));
      const __m512i pred_vec = _mm512_loadu_si512((const __m512i *)(second_pred + j));
      
      // Average ref and second_pred
      const __m512i avg = _mm512_avg_epu16(ref_vec, pred_vec);
      
      // Calculate absolute differences
      const __m512i diff = _mm512_abs_epi16(_mm512_sub_epi16(src_vec, avg));
      
      // Accumulate
      const __m512i diff_lo = _mm512_unpacklo_epi16(diff, _mm512_setzero_si512());
      const __m512i diff_hi = _mm512_unpackhi_epi16(diff, _mm512_setzero_si512());
      
      sad_acc = _mm512_add_epi32(sad_acc, diff_lo);
      sad_acc = _mm512_add_epi32(sad_acc, diff_hi);
    }
    
    src += src_stride;
    ref += ref_stride;
    second_pred += 64;
  }
  
  return _mm512_reduce_add_epi32(sad_acc);
}

unsigned int vpx_highbd_sad32x32_avg_avx512(const uint8_t *src8, int src_stride,
                                             const uint8_t *ref8, int ref_stride,
                                             const uint8_t *second_pred8) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  const uint16_t *second_pred = CONVERT_TO_SHORTPTR(second_pred8);
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < 32; ++i) {
    const __m512i src_vec = _mm512_loadu_si512((const __m512i *)(src));
    const __m512i ref_vec = _mm512_loadu_si512((const __m512i *)(ref));
    const __m512i pred_vec = _mm512_loadu_si512((const __m512i *)(second_pred));
    
    // Average ref and second_pred
    const __m512i avg = _mm512_avg_epu16(ref_vec, pred_vec);
    
    // Calculate absolute differences
    const __m512i diff = _mm512_abs_epi16(_mm512_sub_epi16(src_vec, avg));
    
    // Accumulate
    const __m512i diff_lo = _mm512_unpacklo_epi16(diff, _mm512_setzero_si512());
    const __m512i diff_hi = _mm512_unpackhi_epi16(diff, _mm512_setzero_si512());
    
    sad_acc = _mm512_add_epi32(sad_acc, diff_lo);
    sad_acc = _mm512_add_epi32(sad_acc, diff_hi);
    
    src += src_stride;
    ref += ref_stride;
    second_pred += 32;
  }
  
  return _mm512_reduce_add_epi32(sad_acc);
}

unsigned int vpx_highbd_sad16x16_avg_avx512(const uint8_t *src8, int src_stride,
                                             const uint8_t *ref8, int ref_stride,
                                             const uint8_t *second_pred8) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  const uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  const uint16_t *second_pred = CONVERT_TO_SHORTPTR(second_pred8);
  __m512i sad_acc = _mm512_setzero_si512();
  
  for (int i = 0; i < 16; ++i) {
    const __m256i src_vec = _mm256_loadu_si256((const __m256i *)(src));
    const __m256i ref_vec = _mm256_loadu_si256((const __m256i *)(ref));
    const __m256i pred_vec = _mm256_loadu_si256((const __m256i *)(second_pred));
    
    // Average ref and second_pred
    const __m256i avg = _mm256_avg_epu16(ref_vec, pred_vec);
    
    // Calculate absolute differences
    const __m256i diff = _mm256_abs_epi16(_mm256_sub_epi16(src_vec, avg));
    
    // Convert to 32-bit and accumulate
    const __m512i diff_32 = _mm512_cvtepu16_epi32(diff);
    sad_acc = _mm512_add_epi32(sad_acc, diff_32);
    
    src += src_stride;
    ref += ref_stride;
    second_pred += 16;
  }
  
  return _mm512_reduce_add_epi32(sad_acc);
}

// High bit-depth 4D SAD functions
void vpx_highbd_sad64x64x4d_avx512(const uint8_t *src8, int src_stride,
                                    const uint8_t *const ref_array[4],
                                    int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  
  for (int i = 0; i < 4; ++i) {
    const uint16_t *ref = CONVERT_TO_SHORTPTR(ref_array[i]);
    sad_array[i] = highbd_sad64xh_avx512(src, src_stride, ref, ref_stride, 64);
  }
}

void vpx_highbd_sad32x32x4d_avx512(const uint8_t *src8, int src_stride,
                                    const uint8_t *const ref_array[4],
                                    int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  
  for (int i = 0; i < 4; ++i) {
    const uint16_t *ref = CONVERT_TO_SHORTPTR(ref_array[i]);
    sad_array[i] = highbd_sad32xh_avx512(src, src_stride, ref, ref_stride, 32);
  }
}

void vpx_highbd_sad16x16x4d_avx512(const uint8_t *src8, int src_stride,
                                    const uint8_t *const ref_array[4],
                                    int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  
  for (int i = 0; i < 4; ++i) {
    const uint16_t *ref = CONVERT_TO_SHORTPTR(ref_array[i]);
    sad_array[i] = highbd_sad16xh_avx512(src, src_stride, ref, ref_stride, 16);
  }
}

void vpx_highbd_sad8x8x4d_avx512(const uint8_t *src8, int src_stride,
                                  const uint8_t *const ref_array[4],
                                  int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  
  for (int i = 0; i < 4; ++i) {
    const uint16_t *ref = CONVERT_TO_SHORTPTR(ref_array[i]);
    sad_array[i] = highbd_sad8xh_avx512(src, src_stride, ref, ref_stride, 8);
  }
}

void vpx_highbd_sad4x4x4d_avx512(const uint8_t *src8, int src_stride,
                                  const uint8_t *const ref_array[4],
                                  int ref_stride, uint32_t sad_array[4]) {
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  
  for (int i = 0; i < 4; ++i) {
    const uint16_t *ref = CONVERT_TO_SHORTPTR(ref_array[i]);
    unsigned int sad = 0;
    
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        sad += abs(src[x] - ref[x]);
      }
      src += src_stride;
      ref += ref_stride;
    }
    src -= 4 * src_stride;  // Reset src pointer for next reference
    
    sad_array[i] = sad;
  }
}