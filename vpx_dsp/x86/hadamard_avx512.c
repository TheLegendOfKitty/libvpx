/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
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
#include "vpx/vpx_integer.h"

// Perform 1D Hadamard transform on 16 elements using AVX-512
static INLINE void hadamard_1d_16_avx512(__m512i *a0, __m512i *a1) {
  const __m512i b0 = _mm512_add_epi16(*a0, *a1);
  const __m512i b1 = _mm512_sub_epi16(*a0, *a1);
  *a0 = b0;
  *a1 = b1;
}

// Perform 8x8 Hadamard transform using AVX-512 for improved performance
static INLINE void hadamard_8x8_avx512(const int16_t *src_diff, 
                                       ptrdiff_t src_stride, __m512i *coeff) {
  __m512i rows[8];
  
  // Load 8 rows of 8 16-bit elements each, packed into 512-bit vectors
  for (int i = 0; i < 8; ++i) {
    // Load 8 16-bit elements and broadcast across the 512-bit vector
    const __m128i row = _mm_loadu_si128((const __m128i *)(src_diff + i * src_stride));
    rows[i] = _mm512_broadcast_i32x4(row);
  }
  
  // Perform horizontal Hadamard transform (rows)
  hadamard_1d_16_avx512(&rows[0], &rows[1]);
  hadamard_1d_16_avx512(&rows[2], &rows[3]);
  hadamard_1d_16_avx512(&rows[4], &rows[5]);
  hadamard_1d_16_avx512(&rows[6], &rows[7]);
  
  hadamard_1d_16_avx512(&rows[0], &rows[2]);
  hadamard_1d_16_avx512(&rows[1], &rows[3]);
  hadamard_1d_16_avx512(&rows[4], &rows[6]);
  hadamard_1d_16_avx512(&rows[5], &rows[7]);
  
  hadamard_1d_16_avx512(&rows[0], &rows[4]);
  hadamard_1d_16_avx512(&rows[1], &rows[5]);
  hadamard_1d_16_avx512(&rows[2], &rows[6]);
  hadamard_1d_16_avx512(&rows[3], &rows[7]);
  
  // Transpose for vertical transform
  __m512i cols[8];
  
  // Efficient transpose using AVX-512 permute operations
  const __m512i idx_lo = _mm512_set_epi64(6, 4, 2, 0, 14, 12, 10, 8);
  const __m512i idx_hi = _mm512_set_epi64(7, 5, 3, 1, 15, 13, 11, 9);
  
  for (int i = 0; i < 4; ++i) {
    cols[i] = _mm512_permutex2var_epi64(rows[2*i], idx_lo, rows[2*i + 1]);
    cols[i + 4] = _mm512_permutex2var_epi64(rows[2*i], idx_hi, rows[2*i + 1]);
  }
  
  // Perform vertical Hadamard transform (columns)
  hadamard_1d_16_avx512(&cols[0], &cols[1]);
  hadamard_1d_16_avx512(&cols[2], &cols[3]);
  hadamard_1d_16_avx512(&cols[4], &cols[5]);
  hadamard_1d_16_avx512(&cols[6], &cols[7]);
  
  hadamard_1d_16_avx512(&cols[0], &cols[2]);
  hadamard_1d_16_avx512(&cols[1], &cols[3]);
  hadamard_1d_16_avx512(&cols[4], &cols[6]);
  hadamard_1d_16_avx512(&cols[5], &cols[7]);
  
  hadamard_1d_16_avx512(&cols[0], &cols[4]);
  hadamard_1d_16_avx512(&cols[1], &cols[5]);
  hadamard_1d_16_avx512(&cols[2], &cols[6]);
  hadamard_1d_16_avx512(&cols[3], &cols[7]);
  
  // Store results
  for (int i = 0; i < 8; ++i) {
    coeff[i] = cols[i];
  }
}

void vpx_hadamard_16x16_avx512(const int16_t *src_diff, ptrdiff_t src_stride,
                               tran_low_t *coeff) {
  __m512i coeff_vecs[32];  // 4 8x8 blocks * 8 vectors each
  
  // Process four 8x8 blocks
  hadamard_8x8_avx512(src_diff, src_stride, &coeff_vecs[0]);
  hadamard_8x8_avx512(src_diff + 8, src_stride, &coeff_vecs[8]);
  hadamard_8x8_avx512(src_diff + 8 * src_stride, src_stride, &coeff_vecs[16]);
  hadamard_8x8_avx512(src_diff + 8 * src_stride + 8, src_stride, &coeff_vecs[24]);
  
  // Combine the four 8x8 blocks with additional Hadamard operations
  for (int i = 0; i < 8; ++i) {
    // Process corresponding vectors from each 8x8 block
    __m512i a = coeff_vecs[i];      // Top-left 8x8
    __m512i b = coeff_vecs[i + 8];  // Top-right 8x8
    __m512i c = coeff_vecs[i + 16]; // Bottom-left 8x8
    __m512i d = coeff_vecs[i + 24]; // Bottom-right 8x8
    
    // 2x2 Hadamard on the four 8x8 blocks
    const __m512i ab_sum = _mm512_add_epi16(a, b);
    const __m512i ab_diff = _mm512_sub_epi16(a, b);
    const __m512i cd_sum = _mm512_add_epi16(c, d);
    const __m512i cd_diff = _mm512_sub_epi16(c, d);
    
    const __m512i result0 = _mm512_add_epi16(ab_sum, cd_sum);
    const __m512i result1 = _mm512_sub_epi16(ab_sum, cd_sum);
    const __m512i result2 = _mm512_add_epi16(ab_diff, cd_diff);
    const __m512i result3 = _mm512_sub_epi16(ab_diff, cd_diff);
    
    // Store results - convert to tran_low_t format
    if (sizeof(tran_low_t) == sizeof(int32_t)) {
      // Sign extend to 32-bit
      const __m512i result0_lo = _mm512_unpacklo_epi16(result0, _mm512_srai_epi16(result0, 15));
      const __m512i result0_hi = _mm512_unpackhi_epi16(result0, _mm512_srai_epi16(result0, 15));
      _mm512_storeu_si512((__m512i *)(coeff + i * 32), result0_lo);
      _mm512_storeu_si512((__m512i *)(coeff + i * 32 + 16), result0_hi);
      
      const __m512i result1_lo = _mm512_unpacklo_epi16(result1, _mm512_srai_epi16(result1, 15));
      const __m512i result1_hi = _mm512_unpackhi_epi16(result1, _mm512_srai_epi16(result1, 15));
      _mm512_storeu_si512((__m512i *)(coeff + (i + 8) * 32), result1_lo);
      _mm512_storeu_si512((__m512i *)(coeff + (i + 8) * 32 + 16), result1_hi);
      
      const __m512i result2_lo = _mm512_unpacklo_epi16(result2, _mm512_srai_epi16(result2, 15));
      const __m512i result2_hi = _mm512_unpackhi_epi16(result2, _mm512_srai_epi16(result2, 15));
      _mm512_storeu_si512((__m512i *)(coeff + (i + 16) * 32), result2_lo);
      _mm512_storeu_si512((__m512i *)(coeff + (i + 16) * 32 + 16), result2_hi);
      
      const __m512i result3_lo = _mm512_unpacklo_epi16(result3, _mm512_srai_epi16(result3, 15));
      const __m512i result3_hi = _mm512_unpackhi_epi16(result3, _mm512_srai_epi16(result3, 15));
      _mm512_storeu_si512((__m512i *)(coeff + (i + 24) * 32), result3_lo);
      _mm512_storeu_si512((__m512i *)(coeff + (i + 24) * 32 + 16), result3_hi);
    } else {
      // Store as 16-bit (low bit-depth case)
      _mm256_storeu_si256((__m256i *)(coeff + i * 16), _mm512_extracti64x4_epi64(result0, 0));
      _mm256_storeu_si256((__m256i *)(coeff + (i + 8) * 16), _mm512_extracti64x4_epi64(result1, 0));
      _mm256_storeu_si256((__m256i *)(coeff + (i + 16) * 16), _mm512_extracti64x4_epi64(result2, 0));
      _mm256_storeu_si256((__m256i *)(coeff + (i + 24) * 16), _mm512_extracti64x4_epi64(result3, 0));
    }
  }
}

void vpx_hadamard_32x32_avx512(const int16_t *src_diff, ptrdiff_t src_stride,
                               tran_low_t *coeff) {
  // Process sixteen 8x8 blocks arranged in a 4x4 grid
  __m512i coeff_8x8[16][8];  // 16 8x8 blocks, 8 vectors each
  
  // Compute Hadamard transform for each 8x8 block
  for (int block_row = 0; block_row < 4; ++block_row) {
    for (int block_col = 0; block_col < 4; ++block_col) {
      const int block_idx = block_row * 4 + block_col;
      const int16_t *block_src = src_diff + block_row * 8 * src_stride + block_col * 8;
      hadamard_8x8_avx512(block_src, src_stride, coeff_8x8[block_idx]);
    }
  }
  
  // Combine 8x8 blocks hierarchically
  // First level: combine into 16x16 blocks (2x2 grid of 8x8 blocks)
  __m512i coeff_16x16[4][32];
  
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const int block16_idx = i * 2 + j;
      const int block_tl = (i * 2) * 4 + (j * 2);      // Top-left 8x8
      const int block_tr = (i * 2) * 4 + (j * 2 + 1);  // Top-right 8x8
      const int block_bl = (i * 2 + 1) * 4 + (j * 2);  // Bottom-left 8x8
      const int block_br = (i * 2 + 1) * 4 + (j * 2 + 1); // Bottom-right 8x8
      
      for (int k = 0; k < 8; ++k) {
        // 2x2 Hadamard on four 8x8 blocks
        const __m512i a = coeff_8x8[block_tl][k];
        const __m512i b = coeff_8x8[block_tr][k];
        const __m512i c = coeff_8x8[block_bl][k];
        const __m512i d = coeff_8x8[block_br][k];
        
        const __m512i ab_sum = _mm512_add_epi16(a, b);
        const __m512i ab_diff = _mm512_sub_epi16(a, b);
        const __m512i cd_sum = _mm512_add_epi16(c, d);
        const __m512i cd_diff = _mm512_sub_epi16(c, d);
        
        coeff_16x16[block16_idx][k] = _mm512_add_epi16(ab_sum, cd_sum);
        coeff_16x16[block16_idx][k + 8] = _mm512_sub_epi16(ab_sum, cd_sum);
        coeff_16x16[block16_idx][k + 16] = _mm512_add_epi16(ab_diff, cd_diff);
        coeff_16x16[block16_idx][k + 24] = _mm512_sub_epi16(ab_diff, cd_diff);
      }
    }
  }
  
  // Second level: combine four 16x16 blocks into final 32x32 result
  for (int k = 0; k < 32; ++k) {
    // 2x2 Hadamard on four 16x16 blocks
    const __m512i a = coeff_16x16[0][k];  // Top-left 16x16
    const __m512i b = coeff_16x16[1][k];  // Top-right 16x16
    const __m512i c = coeff_16x16[2][k];  // Bottom-left 16x16
    const __m512i d = coeff_16x16[3][k];  // Bottom-right 16x16
    
    const __m512i ab_sum = _mm512_add_epi16(a, b);
    const __m512i ab_diff = _mm512_sub_epi16(a, b);
    const __m512i cd_sum = _mm512_add_epi16(c, d);
    const __m512i cd_diff = _mm512_sub_epi16(c, d);
    
    const __m512i result0 = _mm512_add_epi16(ab_sum, cd_sum);
    const __m512i result1 = _mm512_sub_epi16(ab_sum, cd_sum);
    const __m512i result2 = _mm512_add_epi16(ab_diff, cd_diff);
    const __m512i result3 = _mm512_sub_epi16(ab_diff, cd_diff);
    
    // Store results
    if (sizeof(tran_low_t) == sizeof(int32_t)) {
      // Convert to 32-bit and store
      const __m512i result0_lo = _mm512_unpacklo_epi16(result0, _mm512_srai_epi16(result0, 15));
      const __m512i result0_hi = _mm512_unpackhi_epi16(result0, _mm512_srai_epi16(result0, 15));
      _mm512_storeu_si512((__m512i *)(coeff + k * 64), result0_lo);
      _mm512_storeu_si512((__m512i *)(coeff + k * 64 + 16), result0_hi);
      _mm512_storeu_si512((__m512i *)(coeff + k * 64 + 32), 
                         _mm512_unpacklo_epi16(result1, _mm512_srai_epi16(result1, 15)));
      _mm512_storeu_si512((__m512i *)(coeff + k * 64 + 48), 
                         _mm512_unpackhi_epi16(result1, _mm512_srai_epi16(result1, 15)));
    } else {
      // Store as 16-bit
      _mm256_storeu_si256((__m256i *)(coeff + k * 32), _mm512_extracti64x4_epi64(result0, 0));
      _mm256_storeu_si256((__m256i *)(coeff + k * 32 + 16), _mm512_extracti64x4_epi64(result1, 0));
    }
  }
}