/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <immintrin.h>

#include "./vp9_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

// Load 32 tran_low_t elements into a 512-bit register
static INLINE __m512i load_tran_low_avx512(const tran_low_t *a) {
#if CONFIG_VP9_HIGHBITDEPTH
  return _mm512_load_si512((const __m512i *)a);
#else
  // Convert 16-bit to 32-bit and load into 512-bit register
  const __m256i a_256 = _mm256_load_si256((const __m256i *)a);
  return _mm512_cvtepi16_epi32(a_256);
#endif
}

int64_t vp9_block_error_avx512(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                                intptr_t block_size, int64_t *ssz) {
  __m512i sse_512, ssz_512;
  const __m512i zero = _mm512_setzero_si512();
  int64_t sse;

  // If the block size is 32 then the results will fit in 32 bits.
  if (block_size == 32) {
    __m512i coeff_512, dqcoeff_512;
    // Load 32 elements for coeff and dqcoeff.
    coeff_512 = load_tran_low_avx512(coeff);
    dqcoeff_512 = load_tran_low_avx512(dqcoeff);
    
    // dqcoeff - coeff
    dqcoeff_512 = _mm512_sub_epi32(dqcoeff_512, coeff_512);
    // Square the differences
    dqcoeff_512 = _mm512_mullo_epi32(dqcoeff_512, dqcoeff_512);
    // Square the coefficients
    coeff_512 = _mm512_mullo_epi32(coeff_512, coeff_512);
    
    // Convert to 64-bit for accumulation
    sse_512 = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(dqcoeff_512, 0));
    sse_512 = _mm512_add_epi64(sse_512, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(dqcoeff_512, 1)));
    
    ssz_512 = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(coeff_512, 0));
    ssz_512 = _mm512_add_epi64(ssz_512, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(coeff_512, 1)));
  } else {
    int i;
    assert(block_size % 32 == 0);
    sse_512 = zero;
    ssz_512 = zero;

    for (i = 0; i < block_size; i += 32) {
      __m512i coeff_0, dqcoeff_0;
      __m512i diff_squared, coeff_squared;
      
      // Load 32 elements for coeff and dqcoeff.
      coeff_0 = load_tran_low_avx512(coeff + i);
      dqcoeff_0 = load_tran_low_avx512(dqcoeff + i);
      
      // Calculate differences and square them
      diff_squared = _mm512_sub_epi32(dqcoeff_0, coeff_0);
      diff_squared = _mm512_mullo_epi32(diff_squared, diff_squared);
      
      // Square the coefficients
      coeff_squared = _mm512_mullo_epi32(coeff_0, coeff_0);
      
      // Convert to 64-bit and accumulate
      // Process lower 256 bits (8 elements)
      __m256i diff_lo = _mm512_extracti64x4_epi64(diff_squared, 0);
      __m256i diff_hi = _mm512_extracti64x4_epi64(diff_squared, 1);
      __m256i coeff_lo = _mm512_extracti64x4_epi64(coeff_squared, 0);
      __m256i coeff_hi = _mm512_extracti64x4_epi64(coeff_squared, 1);
      
      sse_512 = _mm512_add_epi64(sse_512, _mm512_cvtepi32_epi64(diff_lo));
      sse_512 = _mm512_add_epi64(sse_512, _mm512_cvtepi32_epi64(diff_hi));
      
      ssz_512 = _mm512_add_epi64(ssz_512, _mm512_cvtepi32_epi64(coeff_lo));
      ssz_512 = _mm512_add_epi64(ssz_512, _mm512_cvtepi32_epi64(coeff_hi));
    }
  }

  // Horizontal sum of 64-bit values in 512-bit register
  sse = _mm512_reduce_add_epi64(sse_512);
  *ssz = _mm512_reduce_add_epi64(ssz_512);

  return sse;
}

int64_t vp9_block_error_fp_avx512(const tran_low_t *coeff,
                                   const tran_low_t *dqcoeff, int block_size) {
  const __m512i zero = _mm512_setzero_si512();
  __m512i sse_512 = zero;
  int64_t sse;

  if (block_size == 32) {
    // Load 32 elements for coeff and dqcoeff.
    const __m512i _coeff = load_tran_low_avx512(coeff);
    const __m512i _dqcoeff = load_tran_low_avx512(dqcoeff);
    
    // dqcoeff - coeff
    const __m512i diff = _mm512_sub_epi32(_dqcoeff, _coeff);
    // Square the differences
    const __m512i error = _mm512_mullo_epi32(diff, diff);
    
    // Convert to 64-bit for accumulation
    __m256i error_lo = _mm512_extracti64x4_epi64(error, 0);
    __m256i error_hi = _mm512_extracti64x4_epi64(error, 1);
    
    sse_512 = _mm512_cvtepi32_epi64(error_lo);
    sse_512 = _mm512_add_epi64(sse_512, _mm512_cvtepi32_epi64(error_hi));
  } else {
    int i;
    for (i = 0; i < block_size; i += 32) {
      // Load 32 elements for coeff and dqcoeff.
      const __m512i _coeff = load_tran_low_avx512(coeff + i);
      const __m512i _dqcoeff = load_tran_low_avx512(dqcoeff + i);
      
      const __m512i diff = _mm512_sub_epi32(_dqcoeff, _coeff);
      const __m512i error = _mm512_mullo_epi32(diff, diff);
      
      // Convert to 64-bit and accumulate
      __m256i error_lo = _mm512_extracti64x4_epi64(error, 0);
      __m256i error_hi = _mm512_extracti64x4_epi64(error, 1);
      
      sse_512 = _mm512_add_epi64(sse_512, _mm512_cvtepi32_epi64(error_lo));
      sse_512 = _mm512_add_epi64(sse_512, _mm512_cvtepi32_epi64(error_hi));
    }
  }

  // Horizontal sum of all 64-bit values in the 512-bit register
  sse = _mm512_reduce_add_epi64(sse_512);
  return sse;
}

// Sum of squares for 2D blocks - optimized for RD calculations
uint64_t vp9_sum_squares_2d_avx512(const int16_t *src, int src_stride, int size) {
  __m512i sum_512 = _mm512_setzero_si512();
  const __m512i zero = _mm512_setzero_si512();
  
  if (size == 4) {
    // 4x4 block - load 4 rows of 4 elements each
    for (int i = 0; i < 4; i++) {
      __m128i row = _mm_loadl_epi64((const __m128i *)(src + i * src_stride));
      __m512i row_512 = _mm512_cvtepi16_epi32(_mm256_castsi128_si256(row));
      row_512 = _mm512_mullo_epi32(row_512, row_512);
      
      // Convert to 64-bit and accumulate
      __m256i row_lo = _mm512_extracti64x4_epi64(row_512, 0);
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_lo));
    }
  } else if (size == 8) {
    // 8x8 block - load 8 rows of 8 elements each
    for (int i = 0; i < 8; i++) {
      __m128i row = _mm_load_si128((const __m128i *)(src + i * src_stride));
      __m512i row_512 = _mm512_cvtepi16_epi32(_mm256_castsi128_si256(row));
      row_512 = _mm512_mullo_epi32(row_512, row_512);
      
      // Convert to 64-bit and accumulate
      __m256i row_lo = _mm512_extracti64x4_epi64(row_512, 0);
      __m256i row_hi = _mm512_extracti64x4_epi64(row_512, 1);
      
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_lo));
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_hi));
    }
  } else if (size == 16) {
    // 16x16 block - load 16 rows of 16 elements each
    for (int i = 0; i < 16; i++) {
      __m256i row = _mm256_load_si256((const __m256i *)(src + i * src_stride));
      __m512i row_512 = _mm512_cvtepi16_epi32(row);
      row_512 = _mm512_mullo_epi32(row_512, row_512);
      
      // Convert to 64-bit and accumulate
      __m256i row_lo = _mm512_extracti64x4_epi64(row_512, 0);
      __m256i row_hi = _mm512_extracti64x4_epi64(row_512, 1);
      
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_lo));
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_hi));
    }
  } else if (size == 32) {
    // 32x32 block - load 32 rows of 32 elements each, processing 16 at a time
    for (int i = 0; i < 32; i++) {
      // Process first 16 elements of the row
      __m256i row_0 = _mm256_load_si256((const __m256i *)(src + i * src_stride));
      __m512i row_512_0 = _mm512_cvtepi16_epi32(row_0);
      row_512_0 = _mm512_mullo_epi32(row_512_0, row_512_0);
      
      // Process second 16 elements of the row
      __m256i row_1 = _mm256_load_si256((const __m256i *)(src + i * src_stride + 16));
      __m512i row_512_1 = _mm512_cvtepi16_epi32(row_1);
      row_512_1 = _mm512_mullo_epi32(row_512_1, row_512_1);
      
      // Convert to 64-bit and accumulate
      __m256i row_0_lo = _mm512_extracti64x4_epi64(row_512_0, 0);
      __m256i row_0_hi = _mm512_extracti64x4_epi64(row_512_0, 1);
      __m256i row_1_lo = _mm512_extracti64x4_epi64(row_512_1, 0);
      __m256i row_1_hi = _mm512_extracti64x4_epi64(row_512_1, 1);
      
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_0_lo));
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_0_hi));
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_1_lo));
      sum_512 = _mm512_add_epi64(sum_512, _mm512_cvtepi32_epi64(row_1_hi));
    }
  }
  
  // Horizontal sum of all 64-bit values
  return _mm512_reduce_add_epi64(sum_512);
}

// Block difference SSE calculation for RD optimization
uint64_t vp9_block_diff_sse_avx512(const uint8_t *src, int src_stride,
                                    const uint8_t *pred, int pred_stride,
                                    int block_size) {
  __m512i sse_512 = _mm512_setzero_si512();
  
  if (block_size == 4) {
    // 4x4 block
    for (int i = 0; i < 4; i++) {
      __m128i src_row = _mm_cvtsi32_si128(*(const uint32_t *)(src + i * src_stride));
      __m128i pred_row = _mm_cvtsi32_si128(*(const uint32_t *)(pred + i * pred_stride));
      
      // Convert to 16-bit
      __m128i src_16 = _mm_unpacklo_epi8(src_row, _mm_setzero_si128());
      __m128i pred_16 = _mm_unpacklo_epi8(pred_row, _mm_setzero_si128());
      
      // Calculate differences and square them
      __m128i diff = _mm_sub_epi16(src_16, pred_16);
      __m128i diff_sq = _mm_madd_epi16(diff, diff);
      
      // Accumulate to 64-bit
      __m512i diff_64 = _mm512_cvtepi32_epi64(_mm256_castsi128_si256(diff_sq));
      sse_512 = _mm512_add_epi64(sse_512, diff_64);
    }
  } else if (block_size == 8) {
    // 8x8 block
    for (int i = 0; i < 8; i++) {
      __m128i src_row = _mm_loadl_epi64((const __m128i *)(src + i * src_stride));
      __m128i pred_row = _mm_loadl_epi64((const __m128i *)(pred + i * pred_stride));
      
      // Convert to 16-bit
      __m128i src_16 = _mm_unpacklo_epi8(src_row, _mm_setzero_si128());
      __m128i pred_16 = _mm_unpacklo_epi8(pred_row, _mm_setzero_si128());
      
      // Calculate differences and square them
      __m128i diff = _mm_sub_epi16(src_16, pred_16);
      __m128i diff_sq = _mm_madd_epi16(diff, diff);
      
      // Accumulate to 64-bit
      __m512i diff_64 = _mm512_cvtepi32_epi64(_mm256_castsi128_si256(diff_sq));
      sse_512 = _mm512_add_epi64(sse_512, diff_64);
    }
  } else if (block_size == 16) {
    // 16x16 block
    for (int i = 0; i < 16; i++) {
      __m128i src_row = _mm_load_si128((const __m128i *)(src + i * src_stride));
      __m128i pred_row = _mm_load_si128((const __m128i *)(pred + i * pred_stride));
      
      // Convert to 16-bit (split into low and high 8 bytes)
      __m128i src_lo = _mm_unpacklo_epi8(src_row, _mm_setzero_si128());
      __m128i src_hi = _mm_unpackhi_epi8(src_row, _mm_setzero_si128());
      __m128i pred_lo = _mm_unpacklo_epi8(pred_row, _mm_setzero_si128());
      __m128i pred_hi = _mm_unpackhi_epi8(pred_row, _mm_setzero_si128());
      
      // Calculate differences and square them
      __m128i diff_lo = _mm_sub_epi16(src_lo, pred_lo);
      __m128i diff_hi = _mm_sub_epi16(src_hi, pred_hi);
      __m128i diff_sq_lo = _mm_madd_epi16(diff_lo, diff_lo);
      __m128i diff_sq_hi = _mm_madd_epi16(diff_hi, diff_hi);
      
      // Combine and accumulate to 64-bit
      __m256i diff_sq = _mm256_insertf128_si256(_mm256_castsi128_si256(diff_sq_lo), diff_sq_hi, 1);
      __m512i diff_64 = _mm512_cvtepi32_epi64(diff_sq);
      sse_512 = _mm512_add_epi64(sse_512, diff_64);
    }
  } else if (block_size == 32) {
    // 32x32 block
    for (int i = 0; i < 32; i++) {
      // Process 32 bytes per row (split into two 16-byte chunks)
      __m128i src_0 = _mm_load_si128((const __m128i *)(src + i * src_stride));
      __m128i src_1 = _mm_load_si128((const __m128i *)(src + i * src_stride + 16));
      __m128i pred_0 = _mm_load_si128((const __m128i *)(pred + i * pred_stride));
      __m128i pred_1 = _mm_load_si128((const __m128i *)(pred + i * pred_stride + 16));
      
      // Process first 16 bytes
      __m128i src_0_lo = _mm_unpacklo_epi8(src_0, _mm_setzero_si128());
      __m128i src_0_hi = _mm_unpackhi_epi8(src_0, _mm_setzero_si128());
      __m128i pred_0_lo = _mm_unpacklo_epi8(pred_0, _mm_setzero_si128());
      __m128i pred_0_hi = _mm_unpackhi_epi8(pred_0, _mm_setzero_si128());
      
      __m128i diff_0_lo = _mm_sub_epi16(src_0_lo, pred_0_lo);
      __m128i diff_0_hi = _mm_sub_epi16(src_0_hi, pred_0_hi);
      __m128i diff_sq_0_lo = _mm_madd_epi16(diff_0_lo, diff_0_lo);
      __m128i diff_sq_0_hi = _mm_madd_epi16(diff_0_hi, diff_0_hi);
      
      // Process second 16 bytes
      __m128i src_1_lo = _mm_unpacklo_epi8(src_1, _mm_setzero_si128());
      __m128i src_1_hi = _mm_unpackhi_epi8(src_1, _mm_setzero_si128());
      __m128i pred_1_lo = _mm_unpacklo_epi8(pred_1, _mm_setzero_si128());
      __m128i pred_1_hi = _mm_unpackhi_epi8(pred_1, _mm_setzero_si128());
      
      __m128i diff_1_lo = _mm_sub_epi16(src_1_lo, pred_1_lo);
      __m128i diff_1_hi = _mm_sub_epi16(src_1_hi, pred_1_hi);
      __m128i diff_sq_1_lo = _mm_madd_epi16(diff_1_lo, diff_1_lo);
      __m128i diff_sq_1_hi = _mm_madd_epi16(diff_1_hi, diff_1_hi);
      
      // Combine and accumulate to 64-bit
      __m256i diff_sq_0 = _mm256_insertf128_si256(_mm256_castsi128_si256(diff_sq_0_lo), diff_sq_0_hi, 1);
      __m256i diff_sq_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(diff_sq_1_lo), diff_sq_1_hi, 1);
      
      __m512i diff_64_0 = _mm512_cvtepi32_epi64(diff_sq_0);
      __m512i diff_64_1 = _mm512_cvtepi32_epi64(diff_sq_1);
      
      sse_512 = _mm512_add_epi64(sse_512, diff_64_0);
      sse_512 = _mm512_add_epi64(sse_512, diff_64_1);
    }
  }
  
  // Horizontal sum of all 64-bit values
  return _mm512_reduce_add_epi64(sse_512);
}