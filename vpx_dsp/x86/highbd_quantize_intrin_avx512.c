/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX-512

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/x86/bitdepth_conversion_avx2.h"

static INLINE void highbd_load_coeffs_avx512(const tran_low_t *coeff_ptr,
                                             __m512i *coeff0, __m512i *coeff1) {
  *coeff0 = _mm512_loadu_si512((const __m512i *)coeff_ptr);
  *coeff1 = _mm512_loadu_si512((const __m512i *)(coeff_ptr + 16));
}

static INLINE void highbd_store_coeffs_avx512(tran_low_t *coeff_ptr,
                                              __m512i *coeff0, __m512i *coeff1) {
  _mm512_storeu_si512((__m512i *)coeff_ptr, *coeff0);
  _mm512_storeu_si512((__m512i *)(coeff_ptr + 16), *coeff1);
}

static INLINE __m512i highbd_calculate_qcoeff_avx512(__m512i coeff_abs,
                                                     __m512i round,
                                                     __m512i quant) {
  const __m512i coeff_abs_round = _mm512_add_epi32(coeff_abs, round);
  // Use 64-bit multiplication to prevent overflow
  const __m512i coeff_abs_round_lo = _mm512_unpacklo_epi32(coeff_abs_round, _mm512_setzero_si512());
  const __m512i coeff_abs_round_hi = _mm512_unpackhi_epi32(coeff_abs_round, _mm512_setzero_si512());
  const __m512i quant_lo = _mm512_unpacklo_epi32(quant, _mm512_setzero_si512());
  const __m512i quant_hi = _mm512_unpackhi_epi32(quant, _mm512_setzero_si512());
  
  const __m512i qcoeff_lo = _mm512_srli_epi64(_mm512_mul_epu32(coeff_abs_round_lo, quant_lo), 16);
  const __m512i qcoeff_hi = _mm512_srli_epi64(_mm512_mul_epu32(coeff_abs_round_hi, quant_hi), 16);
  
  return _mm512_packus_epi64(qcoeff_lo, qcoeff_hi);
}

static INLINE __m512i highbd_calculate_dqcoeff_avx512(__m512i qcoeff,
                                                      __m512i dequant) {
  // Multiply qcoeff by dequant
  const __m512i qcoeff_lo = _mm512_unpacklo_epi32(qcoeff, _mm512_setzero_si512());
  const __m512i qcoeff_hi = _mm512_unpackhi_epi32(qcoeff, _mm512_setzero_si512());
  const __m512i dequant_lo = _mm512_unpacklo_epi32(dequant, _mm512_setzero_si512());
  const __m512i dequant_hi = _mm512_unpackhi_epi32(dequant, _mm512_setzero_si512());
  
  const __m512i dqcoeff_lo = _mm512_mul_epu32(qcoeff_lo, dequant_lo);
  const __m512i dqcoeff_hi = _mm512_mul_epu32(qcoeff_hi, dequant_hi);
  
  return _mm512_packus_epi64(dqcoeff_lo, dqcoeff_hi);
}

static INLINE void highbd_quantize_16_avx512(
    __m512i *coeff0, __m512i *coeff1, const __m512i *round0,
    const __m512i *round1, const __m512i *quant0, const __m512i *quant1,
    const __m512i *dequant0, const __m512i *dequant1, __m512i *qcoeff0,
    __m512i *qcoeff1, __m512i *dqcoeff0, __m512i *dqcoeff1,
    __m512i *eob_max) {
  
  // Calculate absolute values
  const __m512i coeff_abs0 = _mm512_abs_epi32(*coeff0);
  const __m512i coeff_abs1 = _mm512_abs_epi32(*coeff1);
  
  // Calculate quantized coefficients
  const __m512i qcoeff_abs0 = highbd_calculate_qcoeff_avx512(coeff_abs0, *round0, *quant0);
  const __m512i qcoeff_abs1 = highbd_calculate_qcoeff_avx512(coeff_abs1, *round1, *quant1);
  
  // Apply signs
  *qcoeff0 = _mm512_sign_epi32(qcoeff_abs0, *coeff0);
  *qcoeff1 = _mm512_sign_epi32(qcoeff_abs1, *coeff1);
  
  // Calculate dequantized coefficients
  *dqcoeff0 = highbd_calculate_dqcoeff_avx512(qcoeff_abs0, *dequant0);
  *dqcoeff1 = highbd_calculate_dqcoeff_avx512(qcoeff_abs1, *dequant1);
  
  // Apply signs to dequantized coefficients
  *dqcoeff0 = _mm512_sign_epi32(*dqcoeff0, *coeff0);
  *dqcoeff1 = _mm512_sign_epi32(*dqcoeff1, *coeff1);
  
  // Update EOB mask
  const __m512i nz_mask0 = _mm512_cmpgt_epi32_mask(qcoeff_abs0, _mm512_setzero_si512());
  const __m512i nz_mask1 = _mm512_cmpgt_epi32_mask(qcoeff_abs1, _mm512_setzero_si512());
  *eob_max = _mm512_or_si512(*eob_max, _mm512_or_si512(nz_mask0, nz_mask1));
}

void vpx_highbd_quantize_b_avx512(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                  int skip_block, const int16_t *zbin_ptr,
                                  const int16_t *round_ptr,
                                  const int16_t *quant_ptr,
                                  const int16_t *quant_shift_ptr,
                                  tran_low_t *qcoeff_ptr,
                                  tran_low_t *dqcoeff_ptr,
                                  const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                  const int16_t *scan, const int16_t *iscan) {
  int index = 16;
  int non_zero_count = 0;
  int eob = 0;
  (void)skip_block;
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  (void)scan;

  // Load quantization parameters
  const __m512i round = _mm512_set1_epi32((int32_t)round_ptr[1]);
  const __m512i quant = _mm512_set1_epi32((int32_t)quant_ptr[1]);
  const __m512i dequant = _mm512_set1_epi32((int32_t)dequant_ptr[1]);
  const __m512i zero = _mm512_setzero_si512();
  __m512i eob_max = zero;

  // DC coefficient (handle separately)
  {
    const __m512i round_dc = _mm512_set1_epi32((int32_t)round_ptr[0]);
    const __m512i quant_dc = _mm512_set1_epi32((int32_t)quant_ptr[0]);
    const __m512i dequant_dc = _mm512_set1_epi32((int32_t)dequant_ptr[0]);
    
    __m512i coeff0, coeff1;
    __m512i qcoeff0, qcoeff1;
    __m512i dqcoeff0, dqcoeff1;
    
    highbd_load_coeffs_avx512(coeff_ptr, &coeff0, &coeff1);
    
    // Process DC coefficient in first position
    const tran_low_t dc_coeff = coeff_ptr[0];
    const int32_t dc_abs = abs(dc_coeff);
    const int32_t dc_qcoeff = (dc_abs + round_ptr[0]) * quant_ptr[0] >> 16;
    const int32_t dc_dqcoeff = dc_qcoeff * dequant_ptr[0];
    
    qcoeff_ptr[0] = (dc_coeff < 0) ? -dc_qcoeff : dc_qcoeff;
    dqcoeff_ptr[0] = (dc_coeff < 0) ? -dc_dqcoeff : dc_dqcoeff;
    
    if (dc_qcoeff) eob = 1;
    
    // Process remaining coefficients in first vector (skip DC)
    coeff0 = _mm512_mask_blend_epi32(0x0001, coeff0, zero);
    
    highbd_quantize_16_avx512(&coeff0, &coeff1, &round, &round,
                              &quant, &quant, &dequant, &dequant,
                              &qcoeff0, &qcoeff1, &dqcoeff0, &dqcoeff1,
                              &eob_max);
    
    // Store results (preserving DC)
    qcoeff0 = _mm512_mask_blend_epi32(0x0001, qcoeff0, _mm512_set1_epi32(qcoeff_ptr[0]));
    dqcoeff0 = _mm512_mask_blend_epi32(0x0001, dqcoeff0, _mm512_set1_epi32(dqcoeff_ptr[0]));
    
    highbd_store_coeffs_avx512(qcoeff_ptr, &qcoeff0, &qcoeff1);
    highbd_store_coeffs_avx512(dqcoeff_ptr, &dqcoeff0, &dqcoeff1);
  }

  // Process remaining coefficients in blocks of 32
  for (index = 32; index < n_coeffs; index += 32) {
    __m512i coeff0, coeff1;
    __m512i qcoeff0, qcoeff1;
    __m512i dqcoeff0, dqcoeff1;
    
    highbd_load_coeffs_avx512(coeff_ptr + index, &coeff0, &coeff1);
    
    highbd_quantize_16_avx512(&coeff0, &coeff1, &round, &round,
                              &quant, &quant, &dequant, &dequant,
                              &qcoeff0, &qcoeff1, &dqcoeff0, &dqcoeff1,
                              &eob_max);
    
    highbd_store_coeffs_avx512(qcoeff_ptr + index, &qcoeff0, &qcoeff1);
    highbd_store_coeffs_avx512(dqcoeff_ptr + index, &dqcoeff0, &dqcoeff1);
  }

  // Calculate EOB
  if (!_mm512_test_epi32_mask(eob_max, eob_max)) {
    *eob_ptr = 0;
  } else {
    for (index = n_coeffs - 1; index >= 0; --index) {
      if (qcoeff_ptr[index]) {
        eob = index + 1;
        break;
      }
    }
    *eob_ptr = eob;
  }
}

void vpx_highbd_quantize_b_32x32_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, int skip_block,
    const int16_t *zbin_ptr, const int16_t *round_ptr,
    const int16_t *quant_ptr, const int16_t *quant_shift_ptr,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
    const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan,
    const int16_t *iscan) {
  
  // For 32x32 blocks, use modified rounding
  int16_t round_32x32[2];
  round_32x32[0] = ROUND_POWER_OF_TWO(round_ptr[0], 1);
  round_32x32[1] = ROUND_POWER_OF_TWO(round_ptr[1], 1);
  
  vpx_highbd_quantize_b_avx512(coeff_ptr, n_coeffs, skip_block, zbin_ptr,
                               round_32x32, quant_ptr, quant_shift_ptr,
                               qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr,
                               scan, iscan);
}