/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/vpx_dsp_common.h"

void vpx_idct4x4_16_add_neon(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const uint8_t *dst = dest;
  uint32x2_t s32 = vdup_n_u32(0);
  int16x8_t a[2];
  uint8x8_t s, d[2];
  uint16x8_t sum[2];

  assert(!((intptr_t)dest % sizeof(uint32_t)));
  assert(!(stride % sizeof(uint32_t)));

  // Rows
  a[0] = load_tran_low_to_s16q(input);
  a[1] = load_tran_low_to_s16q(input + 8);
  transpose_idct4x4_16_bd8(a);

  // Columns
  a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
  transpose_idct4x4_16_bd8(a);
  a[0] = vrshrq_n_s16(a[0], 4);
  a[1] = vrshrq_n_s16(a[1], 4);

  s = load_u8(dst, stride);
  dst += 2 * stride;
  // The elements are loaded in reverse order.
  s32 = vld1_lane_u32((const uint32_t *)dst, s32, 1);
  dst += stride;
  s32 = vld1_lane_u32((const uint32_t *)dst, s32, 0);

  sum[0] = vaddw_u8(vreinterpretq_u16_s16(a[0]), s);
  sum[1] = vaddw_u8(vreinterpretq_u16_s16(a[1]), vreinterpret_u8_u32(s32));
  d[0] = vqmovun_s16(vreinterpretq_s16_u16(sum[0]));
  d[1] = vqmovun_s16(vreinterpretq_s16_u16(sum[1]));

  store_u8(dest, stride, d[0]);
  dest += 2 * stride;
  // The elements are stored in reverse order.
  vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d[1]), 1);
  dest += stride;
  vst1_lane_u32((uint32_t *)dest, vreinterpret_u32_u8(d[1]), 0);
}

void vpx_iwht4x4_16_add_neon(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  int16_t output[16];
  
  // First pass: horizontal 1-D Walsh-Hadamard transform on rows using NEON
  // Load all 4 rows at once and apply quantization shift
  int16x4_t row0 = vshr_n_s16(vld1_s16((const int16_t *)(input + 0)), UNIT_QUANT_SHIFT);
  int16x4_t row1 = vshr_n_s16(vld1_s16((const int16_t *)(input + 4)), UNIT_QUANT_SHIFT);
  int16x4_t row2 = vshr_n_s16(vld1_s16((const int16_t *)(input + 8)), UNIT_QUANT_SHIFT);
  int16x4_t row3 = vshr_n_s16(vld1_s16((const int16_t *)(input + 12)), UNIT_QUANT_SHIFT);

  // Apply Walsh-Hadamard transform to each row
  for (int i = 0; i < 4; i++) {
    int16_t a1, b1, c1, d1, e1;
    
    // Extract elements from each row
    if (i == 0) {
      a1 = vget_lane_s16(row0, 0); c1 = vget_lane_s16(row0, 1);
      d1 = vget_lane_s16(row0, 2); b1 = vget_lane_s16(row0, 3);
    } else if (i == 1) {
      a1 = vget_lane_s16(row1, 0); c1 = vget_lane_s16(row1, 1);
      d1 = vget_lane_s16(row1, 2); b1 = vget_lane_s16(row1, 3);
    } else if (i == 2) {
      a1 = vget_lane_s16(row2, 0); c1 = vget_lane_s16(row2, 1);
      d1 = vget_lane_s16(row2, 2); b1 = vget_lane_s16(row2, 3);
    } else {
      a1 = vget_lane_s16(row3, 0); c1 = vget_lane_s16(row3, 1);
      d1 = vget_lane_s16(row3, 2); b1 = vget_lane_s16(row3, 3);
    }
    
    // Walsh-Hadamard butterfly operations
    a1 += c1;
    d1 -= b1;
    e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= b1;
    d1 += c1;
    
    // Store results
    output[i * 4 + 0] = a1;
    output[i * 4 + 1] = b1;
    output[i * 4 + 2] = c1;
    output[i * 4 + 3] = d1;
  }

  // Second pass: vertical 1-D Walsh-Hadamard transform on columns + add to dest
  for (int i = 0; i < 4; i++) {
    // Extract column i from intermediate results
    int16_t a1 = output[4 * 0 + i];
    int16_t c1 = output[4 * 1 + i];
    int16_t d1 = output[4 * 2 + i];
    int16_t b1 = output[4 * 3 + i];
    
    // Walsh-Hadamard butterfly operations
    a1 += c1;
    d1 -= b1;
    int16_t e1 = (a1 - d1) >> 1;
    b1 = e1 - b1;
    c1 = e1 - c1;
    a1 -= b1;
    d1 += c1;
    
    // Add to destination with pixel clamping using NEON saturation
    int16x4_t transform_vals = vdup_n_s16(0);
    transform_vals = vset_lane_s16(a1, transform_vals, 0);
    transform_vals = vset_lane_s16(b1, transform_vals, 1);
    transform_vals = vset_lane_s16(c1, transform_vals, 2);
    transform_vals = vset_lane_s16(d1, transform_vals, 3);
    
    // Load destination pixels
    uint8x8_t dest_vals = vdup_n_u8(0);
    dest_vals = vset_lane_u8(dest[stride * 0], dest_vals, 0);
    dest_vals = vset_lane_u8(dest[stride * 1], dest_vals, 1);
    dest_vals = vset_lane_u8(dest[stride * 2], dest_vals, 2);
    dest_vals = vset_lane_u8(dest[stride * 3], dest_vals, 3);
    
    // Convert to 16-bit, add, and clamp
    uint16x4_t dest_16 = vget_low_u16(vmovl_u8(dest_vals));
    int16x4_t dest_s16 = vreinterpret_s16_u16(dest_16);
    int16x4_t sum = vadd_s16(dest_s16, transform_vals);
    uint8x8_t result = vqmovun_s16(vcombine_s16(sum, sum));
    
    // Store results back to destination
    dest[stride * 0] = vget_lane_u8(result, 0);
    dest[stride * 1] = vget_lane_u8(result, 1);
    dest[stride * 2] = vget_lane_u8(result, 2);
    dest[stride * 3] = vget_lane_u8(result, 3);
    
    dest++;
  }
}
