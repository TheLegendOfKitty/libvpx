/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <immintrin.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"

static INLINE unsigned int sad64xh_avx512(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int h) {
  int i, res;
  __m512i sad_reg, ref_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  for (i = 0; i < h; i++) {
    ref_reg = _mm512_loadu_si512((const __m512i *)ref_ptr);
    sad_reg =
        _mm512_sad_epu8(ref_reg, _mm512_loadu_si512((__m512 const *)src_ptr));
    sum_sad = _mm512_add_epi32(sum_sad, sad_reg);
    ref_ptr += ref_stride;
    src_ptr += src_stride;
  }
  res = _mm512_reduce_add_epi32(sum_sad);
  return res;
}

#define FSAD64_H(h)                                                           \
  unsigned int vpx_sad64x##h##_avx512(const uint8_t *src_ptr, int src_stride, \
                                      const uint8_t *ref_ptr,                 \
                                      int ref_stride) {                       \
    return sad64xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, h);       \
  }

#define FSADS64_H(h)                                                  \
  unsigned int vpx_sad_skip_64x##h##_avx512(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride) {                                               \
    return 2 * sad64xh_avx512(src_ptr, src_stride * 2, ref_ptr,       \
                              ref_stride * 2, h / 2);                 \
  }

#define FSAD64  \
  FSAD64_H(64)  \
  FSAD64_H(32)  \
  FSADS64_H(64) \
  FSADS64_H(32)

FSAD64

#undef FSAD64
#undef FSAD64_H
#undef FSADS64_H

#define FSADAVG64_H(h)                                                         \
  unsigned int vpx_sad64x##h##_avg_avx512(                                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    int i;                                                                     \
    __m512i sad_reg, ref_reg;                                                  \
    __m512i sum_sad = _mm512_setzero_si512();                                  \
    for (i = 0; i < h; i++) {                                                  \
      ref_reg = _mm512_loadu_si512((const __m512i *)ref_ptr);                  \
      ref_reg = _mm512_avg_epu8(                                               \
          ref_reg, _mm512_loadu_si512((const __m512i *)second_pred));          \
      sad_reg = _mm512_sad_epu8(ref_reg,                                       \
                                _mm512_loadu_si512((const __m512i *)src_ptr)); \
      sum_sad = _mm512_add_epi32(sum_sad, sad_reg);                            \
      ref_ptr += ref_stride;                                                   \
      src_ptr += src_stride;                                                   \
      second_pred += 64;                                                       \
    }                                                                          \
    return (unsigned int)_mm512_reduce_add_epi32(sum_sad);                     \
  }

#define FSADAVG64 \
  FSADAVG64_H(64) \
  FSADAVG64_H(32)

FSADAVG64

#undef FSADAVG64
#undef FSADAVG64_H

// 32x32, 32x16 block SAD functions
static INLINE unsigned int sad32xh_avx512(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int h) {
  int i;
  __m256i sad_reg, ref_reg, src_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  
  for (i = 0; i < h; i++) {
    src_reg = _mm256_loadu_si256((const __m256i *)src_ptr);
    ref_reg = _mm256_loadu_si256((const __m256i *)ref_ptr);
    sad_reg = _mm256_sad_epu8(src_reg, ref_reg);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(_mm256_castsi256_si128(sad_reg))));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(_mm256_extracti128_si256(sad_reg, 1))));
    
    ref_ptr += ref_stride;
    src_ptr += src_stride;
  }
  return _mm512_reduce_add_epi32(sum_sad);
}

unsigned int vpx_sad32x32_avx512(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride) {
  return sad32xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 32);
}

unsigned int vpx_sad32x16_avx512(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride) {
  return sad32xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 16);
}

unsigned int vpx_sad_skip_32x32_avx512(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride) {
  return 2 * sad32xh_avx512(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, 16);
}

unsigned int vpx_sad_skip_32x16_avx512(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride) {
  return 2 * sad32xh_avx512(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, 8);
}

// 16x32, 16x16, 16x8 block SAD functions
static INLINE unsigned int sad16xh_avx512(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int h) {
  int i;
  __m128i sad_reg, ref_reg, src_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  
  for (i = 0; i < h; i++) {
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    ref_reg = _mm_loadu_si128((const __m128i *)ref_ptr);
    sad_reg = _mm_sad_epu8(src_reg, ref_reg);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(sad_reg)));
    
    ref_ptr += ref_stride;
    src_ptr += src_stride;
  }
  return _mm512_reduce_add_epi32(sum_sad);
}

unsigned int vpx_sad16x32_avx512(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 32);
}

unsigned int vpx_sad16x16_avx512(const uint8_t *src_ptr, int src_stride,
                                 const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 16);
}

unsigned int vpx_sad16x8_avx512(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_avx512(src_ptr, src_stride, ref_ptr, ref_stride, 8);
}

unsigned int vpx_sad_skip_16x32_avx512(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride) {
  return 2 * sad16xh_avx512(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, 16);
}

unsigned int vpx_sad_skip_16x16_avx512(const uint8_t *src_ptr, int src_stride,
                                       const uint8_t *ref_ptr, int ref_stride) {
  return 2 * sad16xh_avx512(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, 8);
}

unsigned int vpx_sad_skip_16x8_avx512(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride) {
  return 2 * sad16xh_avx512(src_ptr, src_stride * 2, ref_ptr, ref_stride * 2, 4);
}

// AVG versions for 32x and 16x blocks
unsigned int vpx_sad32x32_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     const uint8_t *second_pred) {
  int i;
  __m256i sad_reg, ref_reg, src_reg, pred_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  
  for (i = 0; i < 32; i++) {
    src_reg = _mm256_loadu_si256((const __m256i *)src_ptr);
    ref_reg = _mm256_loadu_si256((const __m256i *)ref_ptr);
    pred_reg = _mm256_loadu_si256((const __m256i *)second_pred);
    ref_reg = _mm256_avg_epu8(ref_reg, pred_reg);
    sad_reg = _mm256_sad_epu8(src_reg, ref_reg);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(_mm256_castsi256_si128(sad_reg))));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(_mm256_extracti128_si256(sad_reg, 1))));
    
    ref_ptr += ref_stride;
    src_ptr += src_stride;
    second_pred += 32;
  }
  return _mm512_reduce_add_epi32(sum_sad);
}

unsigned int vpx_sad32x16_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     const uint8_t *second_pred) {
  int i;
  __m256i sad_reg, ref_reg, src_reg, pred_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  
  for (i = 0; i < 16; i++) {
    src_reg = _mm256_loadu_si256((const __m256i *)src_ptr);
    ref_reg = _mm256_loadu_si256((const __m256i *)ref_ptr);
    pred_reg = _mm256_loadu_si256((const __m256i *)second_pred);
    ref_reg = _mm256_avg_epu8(ref_reg, pred_reg);
    sad_reg = _mm256_sad_epu8(src_reg, ref_reg);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(_mm256_castsi256_si128(sad_reg))));
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(_mm256_extracti128_si256(sad_reg, 1))));
    
    ref_ptr += ref_stride;
    src_ptr += src_stride;
    second_pred += 32;
  }
  return _mm512_reduce_add_epi32(sum_sad);
}

unsigned int vpx_sad16x32_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     const uint8_t *second_pred) {
  int i;
  __m128i sad_reg, ref_reg, src_reg, pred_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  
  for (i = 0; i < 32; i++) {
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    ref_reg = _mm_loadu_si128((const __m128i *)ref_ptr);
    pred_reg = _mm_loadu_si128((const __m128i *)second_pred);
    ref_reg = _mm_avg_epu8(ref_reg, pred_reg);
    sad_reg = _mm_sad_epu8(src_reg, ref_reg);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(sad_reg)));
    
    ref_ptr += ref_stride;
    src_ptr += src_stride;
    second_pred += 16;
  }
  return _mm512_reduce_add_epi32(sum_sad);
}

unsigned int vpx_sad16x16_avg_avx512(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride,
                                     const uint8_t *second_pred) {
  int i;
  __m128i sad_reg, ref_reg, src_reg, pred_reg;
  __m512i sum_sad = _mm512_setzero_si512();
  
  for (i = 0; i < 16; i++) {
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    ref_reg = _mm_loadu_si128((const __m128i *)ref_ptr);
    pred_reg = _mm_loadu_si128((const __m128i *)second_pred);
    ref_reg = _mm_avg_epu8(ref_reg, pred_reg);
    sad_reg = _mm_sad_epu8(src_reg, ref_reg);
    sum_sad = _mm512_add_epi32(sum_sad, _mm512_cvtepi32_epi64(_mm256_castsi128_si256(sad_reg)));
    
    ref_ptr += ref_stride;
    src_ptr += src_stride;
    second_pred += 16;
  }
  return _mm512_reduce_add_epi32(sum_sad);
}
