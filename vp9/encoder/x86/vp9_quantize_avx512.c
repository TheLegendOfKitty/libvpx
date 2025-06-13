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

#include "./vp9_rtcd.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void quantize_b_16x16_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, int skip_block,
    const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan_ptr, const int16_t *iscan_ptr, int is_32x32) {
  
  const __m512i zero = _mm512_setzero_si512();
  int index = 16;
  int eob = -1;
  const int zbins[2] = {zbin_ptr[0], zbin_ptr[1]};
  const int rounds[2] = {round_ptr[0], round_ptr[1]};
  const int quants[2] = {quant_ptr[0], quant_ptr[1]};
  const int dequants[2] = {dequant_ptr[0], dequant_ptr[1]};
  const int shift = is_32x32 ? 1 : 0;
  
  (void)skip_block;
  (void)quant_shift_ptr;

  // Setup DC constants
  const __m512i zbin_dc = _mm512_set1_epi32(zbins[0] >> shift);
  const __m512i round_dc = _mm512_set1_epi32(rounds[0] >> shift);
  const __m512i quant_dc = _mm512_set1_epi32(quants[0]);
  const __m512i dequant_dc = _mm512_set1_epi32(dequants[0]);
  
  // Setup AC constants
  const __m512i zbin_ac = _mm512_set1_epi32(zbins[1] >> shift);
  const __m512i round_ac = _mm512_set1_epi32(rounds[1] >> shift);
  const __m512i quant_ac = _mm512_set1_epi32(quants[1]);
  const __m512i dequant_ac = _mm512_set1_epi32(dequants[1]);

  // Handle DC coefficient
  {
    const int rc = 0;
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    if (abs_coeff >= zbins[0]) {
      const int64_t tmp = abs_coeff + rounds[0];
      const int tmp_qc = (int)((tmp * quants[0]) >> 16);
      qcoeff_ptr[rc] = (tmp_qc ^ coeff_sign) - coeff_sign;
      dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequants[0];
      if (tmp_qc) eob = iscan_ptr[rc];
    } else {
      qcoeff_ptr[rc] = 0;
      dqcoeff_ptr[rc] = 0;
    }
  }

  // Process coefficients in groups of 16 using AVX-512
  for (index = 16; index < n_coeffs; index += 16) {
    __m512i coeff, qcoeff, dqcoeff;
    
    // Load coefficients
    coeff = _mm512_loadu_si512((const __m512i *)(coeff_ptr + index));
    
    // Calculate absolute values
    const __m512i coeff_abs = _mm512_abs_epi32(coeff);
    
    // Compare with zero bin threshold
    const __mmask16 cmp_mask = _mm512_cmpge_epi32_mask(coeff_abs, zbin_ac);
    
    if (_mm512_kortestz(cmp_mask, cmp_mask)) {
      // All coefficients are below threshold - set to zero
      _mm512_storeu_si512((__m512i *)(qcoeff_ptr + index), zero);
      _mm512_storeu_si512((__m512i *)(dqcoeff_ptr + index), zero);
      continue;
    }
    
    // Add rounding
    const __m512i coeff_abs_round = _mm512_add_epi32(coeff_abs, round_ac);
    
    // Multiply by quantizer (with 64-bit intermediate to prevent overflow)
    const __m512i coeff_abs_round_lo = _mm512_unpacklo_epi32(coeff_abs_round, zero);
    const __m512i coeff_abs_round_hi = _mm512_unpackhi_epi32(coeff_abs_round, zero);
    const __m512i quant_lo = _mm512_unpacklo_epi32(quant_ac, zero);
    const __m512i quant_hi = _mm512_unpackhi_epi32(quant_ac, zero);
    
    const __m512i qcoeff_abs_lo = _mm512_srli_epi64(
        _mm512_mul_epu32(coeff_abs_round_lo, quant_lo), 16);
    const __m512i qcoeff_abs_hi = _mm512_srli_epi64(
        _mm512_mul_epu32(coeff_abs_round_hi, quant_hi), 16);
    
    // Manual pack since _mm512_packus_epi64 doesn't exist
    // The result should already be in 32-bit range, so we can use shuffle/blend
    const __m512i qcoeff_abs = _mm512_castps_si512(_mm512_shuffle_ps(
        _mm512_castsi512_ps(qcoeff_abs_lo),
        _mm512_castsi512_ps(qcoeff_abs_hi), 0x88));
    
    // Apply sign manually (since _mm512_sign_epi32 doesn't exist)
    const __mmask16 sign_mask = _mm512_cmplt_epi32_mask(coeff, zero);
    qcoeff = _mm512_mask_sub_epi32(qcoeff_abs, sign_mask, zero, qcoeff_abs);
    
    // Mask out coefficients below threshold
    qcoeff = _mm512_mask_blend_epi32(cmp_mask, zero, qcoeff);
    
    // Calculate dequantized values
    dqcoeff = _mm512_mullo_epi32(qcoeff_abs, dequant_ac);
    // Apply sign manually (since _mm512_sign_epi32 doesn't exist)
    dqcoeff = _mm512_mask_sub_epi32(dqcoeff, sign_mask, zero, dqcoeff);
    dqcoeff = _mm512_mask_blend_epi32(cmp_mask, zero, dqcoeff);
    
    // Store results
    _mm512_storeu_si512((__m512i *)(qcoeff_ptr + index), qcoeff);
    _mm512_storeu_si512((__m512i *)(dqcoeff_ptr + index), dqcoeff);
    
    // Update EOB
    if (!_mm512_kortestz(cmp_mask, cmp_mask)) {
      // Find last non-zero coefficient
      for (int i = 15; i >= 0; i--) {
        const int rc = index + i;
        if (qcoeff_ptr[rc]) {
          eob = iscan_ptr[rc];
          break;
        }
      }
    }
  }
  
  *eob_ptr = eob + 1;
}

void vp9_quantize_fp_avx512(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                            int skip_block, const int16_t *zbin_ptr,
                            const int16_t *round_ptr, const int16_t *quant_ptr,
                            const int16_t *quant_shift_ptr,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t *dequant_ptr, uint16_t *eob_ptr,
                            const int16_t *scan_ptr, const int16_t *iscan_ptr) {
  quantize_b_16x16_avx512(coeff_ptr, n_coeffs, skip_block, zbin_ptr, round_ptr,
                          quant_ptr, quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,
                          dequant_ptr, eob_ptr, scan_ptr, iscan_ptr, 0);
}

void vp9_quantize_fp_32x32_avx512(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                  int skip_block, const int16_t *zbin_ptr,
                                  const int16_t *round_ptr,
                                  const int16_t *quant_ptr,
                                  const int16_t *quant_shift_ptr,
                                  tran_low_t *qcoeff_ptr,
                                  tran_low_t *dqcoeff_ptr,
                                  const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                  const int16_t *scan_ptr,
                                  const int16_t *iscan_ptr) {
  quantize_b_16x16_avx512(coeff_ptr, n_coeffs, skip_block, zbin_ptr, round_ptr,
                          quant_ptr, quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,
                          dequant_ptr, eob_ptr, scan_ptr, iscan_ptr, 1);
}