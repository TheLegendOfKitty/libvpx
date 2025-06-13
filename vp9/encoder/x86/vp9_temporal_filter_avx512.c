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

#include "./vp9_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#include "vp9/encoder/vp9_temporal_filter.h"

// Apply temporal filter to 16x16 blocks using AVX-512
void vp9_apply_temporal_filter_avx512(const uint8_t *y_src, int y_src_stride,
                                       const uint8_t *y_pre, int y_pre_stride,
                                       const uint8_t *u_src, const uint8_t *v_src,
                                       int uv_src_stride,
                                       const uint8_t *u_pre, const uint8_t *v_pre,
                                       int uv_pre_stride,
                                       unsigned int block_width,
                                       unsigned int block_height,
                                       int ss_x, int ss_y,
                                       int strength,
                                       int *y_count, int *y_accumulator,
                                       int *u_count, int *u_accumulator,
                                       int *v_count, int *v_accumulator) {
  const int uv_block_width = block_width >> ss_x;
  const int uv_block_height = block_height >> ss_y;
  const int y_width = block_width;
  const int y_height = block_height;
  
  // Temporal filter constants
  const __m512i strength_vec = _mm512_set1_epi16(strength);
  const __m512i one_vec = _mm512_set1_epi16(1);
  const __m512i rounding = _mm512_set1_epi32(1 << (15 - 1));
  
  // Process Y plane
  for (unsigned int i = 0; i < y_height; ++i) {
    for (unsigned int j = 0; j < y_width; j += 64) {
      const unsigned int pixels_to_process = VPXMIN(64, y_width - j);
      const __mmask64 mask = (1ULL << pixels_to_process) - 1;
      
      // Load source and prediction pixels
      const __m512i src_lo = _mm512_maskz_loadu_epi8(mask, y_src + j);
      const __m512i pre_lo = _mm512_maskz_loadu_epi8(mask, y_pre + j);
      
      // Convert to 16-bit for processing
      const __m512i src_lo_16 = _mm512_unpacklo_epi8(src_lo, _mm512_setzero_si512());
      const __m512i src_hi_16 = _mm512_unpackhi_epi8(src_lo, _mm512_setzero_si512());
      const __m512i pre_lo_16 = _mm512_unpacklo_epi8(pre_lo, _mm512_setzero_si512());
      const __m512i pre_hi_16 = _mm512_unpackhi_epi8(pre_lo, _mm512_setzero_si512());
      
      // Calculate absolute differences
      const __m512i diff_lo = _mm512_abs_epi16(_mm512_sub_epi16(src_lo_16, pre_lo_16));
      const __m512i diff_hi = _mm512_abs_epi16(_mm512_sub_epi16(src_hi_16, pre_hi_16));
      
      // Calculate modifier based on difference and strength
      // modifier = 16 * exp(-diff^2 / (2 * strength^2))
      // Approximated using: modifier = max(0, 16 - (diff * diff) / strength)
      const __m512i diff_sq_lo = _mm512_mullo_epi16(diff_lo, diff_lo);
      const __m512i diff_sq_hi = _mm512_mullo_epi16(diff_hi, diff_hi);
      
      // Convert to 32-bit for division
      const __m512i diff_sq_lo_32_0 = _mm512_unpacklo_epi16(diff_sq_lo, _mm512_setzero_si512());
      const __m512i diff_sq_lo_32_1 = _mm512_unpackhi_epi16(diff_sq_lo, _mm512_setzero_si512());
      const __m512i diff_sq_hi_32_0 = _mm512_unpacklo_epi16(diff_sq_hi, _mm512_setzero_si512());
      const __m512i diff_sq_hi_32_1 = _mm512_unpackhi_epi16(diff_sq_hi, _mm512_setzero_si512());
      
      // Compute modifier (simplified approximation)
      const __m512i strength_32 = _mm512_set1_epi32(strength);
      const __m512i base_modifier = _mm512_set1_epi32(16);
      
      const __m512i mod_lo_0 = _mm512_max_epi32(_mm512_setzero_si512(),
                                 _mm512_sub_epi32(base_modifier, 
                                   _mm512_srli_epi32(diff_sq_lo_32_0, 4)));
      const __m512i mod_lo_1 = _mm512_max_epi32(_mm512_setzero_si512(),
                                 _mm512_sub_epi32(base_modifier, 
                                   _mm512_srli_epi32(diff_sq_lo_32_1, 4)));
      const __m512i mod_hi_0 = _mm512_max_epi32(_mm512_setzero_si512(),
                                 _mm512_sub_epi32(base_modifier, 
                                   _mm512_srli_epi32(diff_sq_hi_32_0, 4)));
      const __m512i mod_hi_1 = _mm512_max_epi32(_mm512_setzero_si512(),
                                 _mm512_sub_epi32(base_modifier, 
                                   _mm512_srli_epi32(diff_sq_hi_32_1, 4)));
      
      // Pack modifiers back to 16-bit
      const __m512i mod_lo = _mm512_packs_epi32(mod_lo_0, mod_lo_1);
      const __m512i mod_hi = _mm512_packs_epi32(mod_hi_0, mod_hi_1);
      
      // Update accumulator: accumulator += modifier * prediction
      // Update count: count += modifier
      for (int k = 0; k < pixels_to_process; ++k) {
        const int pixel_row = i;
        const int pixel_col = j + k;
        const int idx = pixel_row * y_width + pixel_col;
        
        // Extract modifiers using memory approach since _mm512_extract_epi16 doesn't exist
        DECLARE_ALIGNED(64, uint16_t, mod_data[64]);
        _mm512_storeu_si512((__m512i *)mod_data, mod_lo);
        _mm512_storeu_si512((__m512i *)(mod_data + 32), mod_hi);
        
        const uint16_t modifier = mod_data[k];
        const uint8_t prediction = y_pre[pixel_col];
        
        y_count[idx] += modifier;
        y_accumulator[idx] += modifier * prediction;
      }
    }
    
    y_src += y_src_stride;
    y_pre += y_pre_stride;
  }
  
  // Process U plane
  for (unsigned int i = 0; i < uv_block_height; ++i) {
    for (unsigned int j = 0; j < uv_block_width; j += 64) {
      const unsigned int pixels_to_process = VPXMIN(64, uv_block_width - j);
      const __mmask64 mask = (1ULL << pixels_to_process) - 1;
      
      // Load source and prediction pixels
      const __m512i u_src_vec = _mm512_maskz_loadu_epi8(mask, u_src + j);
      const __m512i u_pre_vec = _mm512_maskz_loadu_epi8(mask, u_pre + j);
      
      // Convert to 16-bit for processing
      const __m512i u_src_lo_16 = _mm512_unpacklo_epi8(u_src_vec, _mm512_setzero_si512());
      const __m512i u_src_hi_16 = _mm512_unpackhi_epi8(u_src_vec, _mm512_setzero_si512());
      const __m512i u_pre_lo_16 = _mm512_unpacklo_epi8(u_pre_vec, _mm512_setzero_si512());
      const __m512i u_pre_hi_16 = _mm512_unpackhi_epi8(u_pre_vec, _mm512_setzero_si512());
      
      // Calculate absolute differences
      const __m512i u_diff_lo = _mm512_abs_epi16(_mm512_sub_epi16(u_src_lo_16, u_pre_lo_16));
      const __m512i u_diff_hi = _mm512_abs_epi16(_mm512_sub_epi16(u_src_hi_16, u_pre_hi_16));
      
      // Calculate modifier (simplified)
      const __m512i u_diff_sq_lo = _mm512_mullo_epi16(u_diff_lo, u_diff_lo);
      const __m512i u_diff_sq_hi = _mm512_mullo_epi16(u_diff_hi, u_diff_hi);
      
      const __m512i u_mod_lo = _mm512_max_epi16(_mm512_setzero_si512(),
                                 _mm512_sub_epi16(_mm512_set1_epi16(16), 
                                   _mm512_srli_epi16(u_diff_sq_lo, 4)));
      const __m512i u_mod_hi = _mm512_max_epi16(_mm512_setzero_si512(),
                                 _mm512_sub_epi16(_mm512_set1_epi16(16), 
                                   _mm512_srli_epi16(u_diff_sq_hi, 4)));
      
      // Update U accumulator and count
      for (int k = 0; k < pixels_to_process; ++k) {
        const int pixel_row = i;
        const int pixel_col = j + k;
        const int idx = pixel_row * uv_block_width + pixel_col;
        
        // Extract modifiers using memory approach
        DECLARE_ALIGNED(64, uint16_t, u_mod_data[64]);
        _mm512_storeu_si512((__m512i *)u_mod_data, u_mod_lo);
        _mm512_storeu_si512((__m512i *)(u_mod_data + 32), u_mod_hi);
        
        const uint16_t modifier = u_mod_data[k];
        const uint8_t prediction = u_pre[pixel_col];
        
        u_count[idx] += modifier;
        u_accumulator[idx] += modifier * prediction;
      }
    }
    
    u_src += uv_src_stride;
    u_pre += uv_pre_stride;
  }
  
  // Process V plane
  for (unsigned int i = 0; i < uv_block_height; ++i) {
    for (unsigned int j = 0; j < uv_block_width; j += 64) {
      const unsigned int pixels_to_process = VPXMIN(64, uv_block_width - j);
      const __mmask64 mask = (1ULL << pixels_to_process) - 1;
      
      // Load source and prediction pixels
      const __m512i v_src_vec = _mm512_maskz_loadu_epi8(mask, v_src + j);
      const __m512i v_pre_vec = _mm512_maskz_loadu_epi8(mask, v_pre + j);
      
      // Convert to 16-bit for processing
      const __m512i v_src_lo_16 = _mm512_unpacklo_epi8(v_src_vec, _mm512_setzero_si512());
      const __m512i v_src_hi_16 = _mm512_unpackhi_epi8(v_src_vec, _mm512_setzero_si512());
      const __m512i v_pre_lo_16 = _mm512_unpacklo_epi8(v_pre_vec, _mm512_setzero_si512());
      const __m512i v_pre_hi_16 = _mm512_unpackhi_epi8(v_pre_vec, _mm512_setzero_si512());
      
      // Calculate absolute differences
      const __m512i v_diff_lo = _mm512_abs_epi16(_mm512_sub_epi16(v_src_lo_16, v_pre_lo_16));
      const __m512i v_diff_hi = _mm512_abs_epi16(_mm512_sub_epi16(v_src_hi_16, v_pre_hi_16));
      
      // Calculate modifier (simplified)
      const __m512i v_diff_sq_lo = _mm512_mullo_epi16(v_diff_lo, v_diff_lo);
      const __m512i v_diff_sq_hi = _mm512_mullo_epi16(v_diff_hi, v_diff_hi);
      
      const __m512i v_mod_lo = _mm512_max_epi16(_mm512_setzero_si512(),
                                 _mm512_sub_epi16(_mm512_set1_epi16(16), 
                                   _mm512_srli_epi16(v_diff_sq_lo, 4)));
      const __m512i v_mod_hi = _mm512_max_epi16(_mm512_setzero_si512(),
                                 _mm512_sub_epi16(_mm512_set1_epi16(16), 
                                   _mm512_srli_epi16(v_diff_sq_hi, 4)));
      
      // Update V accumulator and count
      for (int k = 0; k < pixels_to_process; ++k) {
        const int pixel_row = i;
        const int pixel_col = j + k;
        const int idx = pixel_row * uv_block_width + pixel_col;
        
        // Extract modifiers using memory approach
        DECLARE_ALIGNED(64, uint16_t, v_mod_data[64]);
        _mm512_storeu_si512((__m512i *)v_mod_data, v_mod_lo);
        _mm512_storeu_si512((__m512i *)(v_mod_data + 32), v_mod_hi);
        
        const uint16_t modifier = v_mod_data[k];
        const uint8_t prediction = v_pre[pixel_col];
        
        v_count[idx] += modifier;
        v_accumulator[idx] += modifier * prediction;
      }
    }
    
    v_src += uv_src_stride;
    v_pre += uv_pre_stride;
  }
}

// High bit-depth temporal filter using AVX-512
void vp9_highbd_apply_temporal_filter_avx512(const uint16_t *y_src, int y_src_stride,
                                              const uint16_t *y_pre, int y_pre_stride,
                                              const uint16_t *u_src, const uint16_t *v_src,
                                              int uv_src_stride,
                                              const uint16_t *u_pre, const uint16_t *v_pre,
                                              int uv_pre_stride,
                                              unsigned int block_width,
                                              unsigned int block_height,
                                              int ss_x, int ss_y,
                                              int strength,
                                              int *y_count, int *y_accumulator,
                                              int *u_count, int *u_accumulator,
                                              int *v_count, int *v_accumulator) {
  const int uv_block_width = block_width >> ss_x;
  const int uv_block_height = block_height >> ss_y;
  const int y_width = block_width;
  const int y_height = block_height;
  
  // Process Y plane with 16-bit precision
  for (unsigned int i = 0; i < y_height; ++i) {
    for (unsigned int j = 0; j < y_width; j += 32) {
      const unsigned int pixels_to_process = VPXMIN(32, y_width - j);
      const __mmask32 mask = (1U << pixels_to_process) - 1;
      
      // Load 16-bit source and prediction pixels
      const __m512i src_vec = _mm512_maskz_loadu_epi16(mask, y_src + j);
      const __m512i pre_vec = _mm512_maskz_loadu_epi16(mask, y_pre + j);
      
      // Calculate absolute differences
      const __m512i diff = _mm512_abs_epi16(_mm512_sub_epi16(src_vec, pre_vec));
      
      // Calculate modifier based on difference and strength
      const __m512i diff_sq = _mm512_mullo_epi16(diff, diff);
      
      // Convert to 32-bit for more precise calculation
      const __m512i diff_sq_lo = _mm512_unpacklo_epi16(diff_sq, _mm512_setzero_si512());
      const __m512i diff_sq_hi = _mm512_unpackhi_epi16(diff_sq, _mm512_setzero_si512());
      
      // Compute modifier (approximation for exponential decay)
      const __m512i strength_32 = _mm512_set1_epi32(strength);
      const __m512i base_modifier = _mm512_set1_epi32(16);
      
      // Use shift-based approximation instead of _mm512_div_epi32 (doesn't exist)
      const __m512i mod_lo = _mm512_max_epi32(_mm512_setzero_si512(),
                               _mm512_sub_epi32(base_modifier, 
                                 _mm512_srli_epi32(diff_sq_lo, 8)));  // Approximate division
      const __m512i mod_hi = _mm512_max_epi32(_mm512_setzero_si512(),
                               _mm512_sub_epi32(base_modifier, 
                                 _mm512_srli_epi32(diff_sq_hi, 8)));
      
      // Update accumulator and count for each pixel
      for (int k = 0; k < pixels_to_process; ++k) {
        const int pixel_row = i;
        const int pixel_col = j + k;
        const int idx = pixel_row * y_width + pixel_col;
        
        // Extract modifier using memory approach since _mm512_extract_epi32 doesn't exist
        DECLARE_ALIGNED(64, uint32_t, mod_data[32]);
        _mm512_storeu_si512((__m512i *)mod_data, mod_lo);
        _mm512_storeu_si512((__m512i *)(mod_data + 16), mod_hi);
        
        uint32_t modifier = mod_data[k];
        
        const uint16_t prediction = y_pre[pixel_col];
        
        y_count[idx] += modifier;
        y_accumulator[idx] += modifier * prediction;
      }
    }
    
    y_src += y_src_stride;
    y_pre += y_pre_stride;
  }
  
  // Process U and V planes similarly
  for (unsigned int i = 0; i < uv_block_height; ++i) {
    for (unsigned int j = 0; j < uv_block_width; j += 32) {
      const unsigned int pixels_to_process = VPXMIN(32, uv_block_width - j);
      const __mmask32 mask = (1U << pixels_to_process) - 1;
      
      // Process U plane
      const __m512i u_src_vec = _mm512_maskz_loadu_epi16(mask, u_src + j);
      const __m512i u_pre_vec = _mm512_maskz_loadu_epi16(mask, u_pre + j);
      const __m512i u_diff = _mm512_abs_epi16(_mm512_sub_epi16(u_src_vec, u_pre_vec));
      const __m512i u_diff_sq = _mm512_mullo_epi16(u_diff, u_diff);
      
      // Process V plane
      const __m512i v_src_vec = _mm512_maskz_loadu_epi16(mask, v_src + j);
      const __m512i v_pre_vec = _mm512_maskz_loadu_epi16(mask, v_pre + j);
      const __m512i v_diff = _mm512_abs_epi16(_mm512_sub_epi16(v_src_vec, v_pre_vec));
      const __m512i v_diff_sq = _mm512_mullo_epi16(v_diff, v_diff);
      
      // Calculate modifiers (simplified for performance)
      const __m512i u_modifier = _mm512_max_epi16(_mm512_setzero_si512(),
                                   _mm512_sub_epi16(_mm512_set1_epi16(16), 
                                     _mm512_srli_epi16(u_diff_sq, 6)));
      const __m512i v_modifier = _mm512_max_epi16(_mm512_setzero_si512(),
                                   _mm512_sub_epi16(_mm512_set1_epi16(16), 
                                     _mm512_srli_epi16(v_diff_sq, 6)));
      
      // Update accumulators
      for (int k = 0; k < pixels_to_process; ++k) {
        const int pixel_row = i;
        const int pixel_col = j + k;
        const int idx = pixel_row * uv_block_width + pixel_col;
        
        // Extract modifiers using memory approach
        DECLARE_ALIGNED(64, uint16_t, uv_mod_data[64]);
        _mm512_storeu_si512((__m512i *)uv_mod_data, u_modifier);
        _mm512_storeu_si512((__m512i *)(uv_mod_data + 32), v_modifier);
        
        const uint16_t u_mod = uv_mod_data[k];
        const uint16_t v_mod = uv_mod_data[k + 32];
        const uint16_t u_prediction = u_pre[pixel_col];
        const uint16_t v_prediction = v_pre[pixel_col];
        
        u_count[idx] += u_mod;
        u_accumulator[idx] += u_mod * u_prediction;
        
        v_count[idx] += v_mod;
        v_accumulator[idx] += v_mod * v_prediction;
      }
    }
    
    u_src += uv_src_stride;
    u_pre += uv_pre_stride;
    v_src += uv_src_stride;
    v_pre += uv_pre_stride;
  }
}