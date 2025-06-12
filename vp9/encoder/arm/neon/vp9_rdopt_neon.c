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

#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_mem/vpx_mem.h"
#include "vp9/common/vp9_common.h"

// NEON-optimized sum of squared differences for multiple 4x4 blocks
// This is useful for RD optimization when multiple error values need accumulation
uint64_t vp9_sum_squares_2d_neon(const int16_t *src, int src_stride, int size) {
  uint64x2_t sum_sq = vdupq_n_u64(0);
  
  for (int i = 0; i < size; i++) {
    const int16_t *row = src + i * src_stride;
    int j = 0;
    
    // Process 8 elements at a time
    for (; j <= size - 8; j += 8) {
      int16x8_t data = vld1q_s16(row + j);
      int32x4_t sq_lo = vmull_s16(vget_low_s16(data), vget_low_s16(data));
      int32x4_t sq_hi = vmull_s16(vget_high_s16(data), vget_high_s16(data));
      sum_sq = vpadalq_u32(sum_sq, vreinterpretq_u32_s32(sq_lo));
      sum_sq = vpadalq_u32(sum_sq, vreinterpretq_u32_s32(sq_hi));
    }
    
    // Handle remaining elements
    for (; j < size; j++) {
      const int16_t val = row[j];
      sum_sq = vaddq_u64(sum_sq, vdupq_n_u64((uint64_t)(val * val)));
    }
  }
  
  return horizontal_add_uint64x2(sum_sq);
}

// NEON-optimized block difference accumulation for RD calculations
// Computes sum of squared differences between prediction and source
uint64_t vp9_block_diff_sse_neon(const uint8_t *src, int src_stride,
                                 const uint8_t *pred, int pred_stride,
                                 int block_size) {
  uint64x2_t sse = vdupq_n_u64(0);
  
  for (int i = 0; i < block_size; i++) {
    const uint8_t *src_row = src + i * src_stride;
    const uint8_t *pred_row = pred + i * pred_stride;
    int j = 0;
    
    // Process 16 pixels at a time
    for (; j <= block_size - 16; j += 16) {
      uint8x16_t src_vec = vld1q_u8(src_row + j);
      uint8x16_t pred_vec = vld1q_u8(pred_row + j);
      uint8x16_t diff = vabdq_u8(src_vec, pred_vec);
      
      // Split into low and high parts
      uint8x8_t diff_lo = vget_low_u8(diff);
      uint8x8_t diff_hi = vget_high_u8(diff);
      
      // Square and accumulate
      uint16x8_t sq_lo = vmull_u8(diff_lo, diff_lo);
      uint16x8_t sq_hi = vmull_u8(diff_hi, diff_hi);
      
      sse = vpadalq_u16(sse, sq_lo);
      sse = vpadalq_u16(sse, sq_hi);
    }
    
    // Process 8 pixels at a time
    for (; j <= block_size - 8; j += 8) {
      uint8x8_t src_vec = vld1_u8(src_row + j);
      uint8x8_t pred_vec = vld1_u8(pred_row + j);
      uint8x8_t diff = vabd_u8(src_vec, pred_vec);
      uint16x8_t sq = vmull_u8(diff, diff);
      sse = vpadalq_u16(sse, sq);
    }
    
    // Handle remaining pixels
    for (; j < block_size; j++) {
      int diff = src_row[j] - pred_row[j];
      sse = vaddq_u64(sse, vdupq_n_u64((uint64_t)(diff * diff)));
    }
  }
  
  return horizontal_add_uint64x2(sse);
}

// NEON-optimized multiple variance calculation for RD optimization
// Useful when comparing multiple prediction modes
static void vp9_multi_block_variance_neon(const uint8_t *src, int src_stride,
                                          const uint8_t *pred[], int pred_stride,
                                          int block_size, int num_preds,
                                          uint32_t *sse_array, uint32_t *var_array) {
  for (int p = 0; p < num_preds; p++) {
    uint64x2_t sum_sq = vdupq_n_u64(0);
    uint32x4_t sum = vdupq_n_u32(0);
    
    for (int i = 0; i < block_size; i++) {
      const uint8_t *src_row = src + i * src_stride;
      const uint8_t *pred_row = pred[p] + i * pred_stride;
      int j = 0;
      
      // Process 16 pixels at a time
      for (; j <= block_size - 16; j += 16) {
        uint8x16_t src_vec = vld1q_u8(src_row + j);
        uint8x16_t pred_vec = vld1q_u8(pred_row + j);
        
        // Calculate differences
        int16x8_t diff_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(src_vec), 
                                                           vget_low_u8(pred_vec)));
        int16x8_t diff_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(src_vec), 
                                                           vget_high_u8(pred_vec)));
        
        // Accumulate sum for mean calculation
        int32x4_t sum_s32 = vreinterpretq_s32_u32(sum);
        sum_s32 = vpadalq_s16(sum_s32, diff_lo);
        sum_s32 = vpadalq_s16(sum_s32, diff_hi);
        sum = vreinterpretq_u32_s32(sum_s32);
        
        // Accumulate sum of squares
        int32x4_t sq_lo = vmull_s16(vget_low_s16(diff_lo), vget_low_s16(diff_lo));
        int32x4_t sq_hi = vmull_s16(vget_high_s16(diff_lo), vget_high_s16(diff_lo));
        sum_sq = vpadalq_u32(sum_sq, vreinterpretq_u32_s32(sq_lo));
        sum_sq = vpadalq_u32(sum_sq, vreinterpretq_u32_s32(sq_hi));
        
        sq_lo = vmull_s16(vget_low_s16(diff_hi), vget_low_s16(diff_hi));
        sq_hi = vmull_s16(vget_high_s16(diff_hi), vget_high_s16(diff_hi));
        sum_sq = vpadalq_u32(sum_sq, vreinterpretq_u32_s32(sq_lo));
        sum_sq = vpadalq_u32(sum_sq, vreinterpretq_u32_s32(sq_hi));
      }
      
      // Handle remaining pixels
      for (; j < block_size; j++) {
        int diff = src_row[j] - pred_row[j];
        sum = vaddq_u32(sum, vdupq_n_u32(diff));
        sum_sq = vaddq_u64(sum_sq, vdupq_n_u64((uint64_t)(diff * diff)));
      }
    }
    
    uint64_t total_sum_sq = horizontal_add_uint64x2(sum_sq);
    int32_t total_sum = horizontal_add_int32x4(vreinterpretq_s32_u32(sum));
    
    sse_array[p] = (uint32_t)total_sum_sq;
    
    // Calculate variance: sse - (sum^2 / n)
    uint64_t mean_sq = ((uint64_t)total_sum * total_sum) / (block_size * block_size);
    var_array[p] = (uint32_t)(total_sum_sq - mean_sq);
  }
}