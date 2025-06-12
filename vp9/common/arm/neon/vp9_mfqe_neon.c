/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "vp9/common/vp9_postproc.h"

static INLINE void filter_by_weight_8x1_neon(const uint8_t *src, uint8_t *dst, 
                                              int src_weight) {
  const int dst_weight = (1 << MFQE_PRECISION) - src_weight;
  const int rounding_bit = 1 << (MFQE_PRECISION - 1);
  
  // Load 8 bytes from source and destination
  const uint8x8_t v_src = vld1_u8(src);
  const uint8x8_t v_dst = vld1_u8(dst);
  
  // Convert to 16-bit for multiplication
  const uint16x8_t v_src_16 = vmovl_u8(v_src);
  const uint16x8_t v_dst_16 = vmovl_u8(v_dst);
  
  // Multiply by weights
  const uint16x8_t v_src_weighted = vmulq_n_u16(v_src_16, src_weight);
  const uint16x8_t v_dst_weighted = vmulq_n_u16(v_dst_16, dst_weight);
  
  // Add together with rounding
  const uint16x8_t v_sum = vaddq_u16(v_src_weighted, v_dst_weighted);
  const uint16x8_t v_rounding = vdupq_n_u16(rounding_bit);
  const uint16x8_t v_sum_rounded = vaddq_u16(v_sum, v_rounding);
  
  // Shift down and narrow back to 8-bit
  const uint8x8_t v_result = vqrshrun_n_s16(vreinterpretq_s16_u16(v_sum_rounded), MFQE_PRECISION);
  
  // Store result
  vst1_u8(dst, v_result);
}

static INLINE void filter_by_weight_16x1_neon(const uint8_t *src, uint8_t *dst,
                                               int src_weight) {
  const int dst_weight = (1 << MFQE_PRECISION) - src_weight;
  const int rounding_bit = 1 << (MFQE_PRECISION - 1);
  
  // Load 16 bytes from source and destination
  const uint8x16_t v_src = vld1q_u8(src);
  const uint8x16_t v_dst = vld1q_u8(dst);
  
  // Convert to 16-bit for multiplication
  const uint16x8_t v_src_lo = vmovl_u8(vget_low_u8(v_src));
  const uint16x8_t v_src_hi = vmovl_u8(vget_high_u8(v_src));
  const uint16x8_t v_dst_lo = vmovl_u8(vget_low_u8(v_dst));
  const uint16x8_t v_dst_hi = vmovl_u8(vget_high_u8(v_dst));
  
  // Multiply by weights
  const uint16x8_t v_src_weighted_lo = vmulq_n_u16(v_src_lo, src_weight);
  const uint16x8_t v_src_weighted_hi = vmulq_n_u16(v_src_hi, src_weight);
  const uint16x8_t v_dst_weighted_lo = vmulq_n_u16(v_dst_lo, dst_weight);
  const uint16x8_t v_dst_weighted_hi = vmulq_n_u16(v_dst_hi, dst_weight);
  
  // Add together with rounding
  const uint16x8_t v_sum_lo = vaddq_u16(v_src_weighted_lo, v_dst_weighted_lo);
  const uint16x8_t v_sum_hi = vaddq_u16(v_src_weighted_hi, v_dst_weighted_hi);
  const uint16x8_t v_rounding = vdupq_n_u16(rounding_bit);
  const uint16x8_t v_sum_rounded_lo = vaddq_u16(v_sum_lo, v_rounding);
  const uint16x8_t v_sum_rounded_hi = vaddq_u16(v_sum_hi, v_rounding);
  
  // Shift down and narrow back to 8-bit
  const uint8x8_t v_result_lo = vqrshrun_n_s16(vreinterpretq_s16_u16(v_sum_rounded_lo), MFQE_PRECISION);
  const uint8x8_t v_result_hi = vqrshrun_n_s16(vreinterpretq_s16_u16(v_sum_rounded_hi), MFQE_PRECISION);
  
  // Store result
  vst1q_u8(dst, vcombine_u8(v_result_lo, v_result_hi));
}

void vp9_filter_by_weight8x8_neon(const uint8_t *src, int src_stride,
                                  uint8_t *dst, int dst_stride, int src_weight) {
  int i;
  
  for (i = 0; i < 8; i++) {
    filter_by_weight_8x1_neon(src, dst, src_weight);
    src += src_stride;
    dst += dst_stride;
  }
}

void vp9_filter_by_weight16x16_neon(const uint8_t *src, int src_stride,
                                    uint8_t *dst, int dst_stride, int src_weight) {
  int i;
  
  for (i = 0; i < 16; i++) {
    filter_by_weight_16x1_neon(src, dst, src_weight);
    src += src_stride;
    dst += dst_stride;
  }
}