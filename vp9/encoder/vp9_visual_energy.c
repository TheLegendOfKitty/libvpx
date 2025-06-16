/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/encoder/vp9_visual_energy.h"
#include "vp9/encoder/vp9_encoder.h"      // For VP9_COMP definition, vpx_malloc/vpx_free
#include "vpx_dsp/vpx_dsp_common.h"       // For vpx_variance_fn_ptr_t
#include "vp9/common/vp9_common.h"        // For BLOCK_SIZE definition
#include "vpx_mem/vpx_mem.h"              // For vpx_malloc/vpx_free

// Define a block size for visual energy calculation (e.g., 16x16)
#define VISUAL_ENERGY_BLOCK_SIZE BLOCK_16X16

void vp9_calculate_visual_energy(VP9_COMP *cpi,
                                 const YV12_BUFFER_CONFIG *source) {
  const int block_w = block_size_wide[VISUAL_ENERGY_BLOCK_SIZE];
  const int block_h = block_size_high[VISUAL_ENERGY_BLOCK_SIZE];
  const int source_w = source->y_crop_width;
  const int source_h = source->y_crop_height;
  const int stride = source->y_stride;
  const uint8_t *src = source->y_buffer;
  vpx_variance_fn_ptr_t vf; // Variance function pointer
  unsigned int sse;
  int map_idx = 0;

  const int map_width = (source_w + block_w - 1) / block_w;
  const int map_height = (source_h + block_h - 1) / block_h;
  const int map_size = map_width * map_height;

  // Get the appropriate variance function (assuming 8-bit depth for now)
  // TODO: Handle high bit depth if necessary
  vf = cpi->fn_ptr[VISUAL_ENERGY_BLOCK_SIZE].vf;

  if (!vf) {
    // Fallback or error handling if variance function is not available
    // For now, just return without doing anything
    // Also free the map if it was allocated and vf is not found (e.g. unsupported block size)
    if (cpi->visual_energy_map) {
        vpx_free(cpi->visual_energy_map);
        cpi->visual_energy_map = NULL;
    }
    return;
  }

  // Allocate or reallocate visual_energy_map if necessary
  // TODO(yaowu): This allocation should ideally be managed elsewhere,
  // e.g., during VP9_COMP creation or when frame dimensions change.
  // For now, handle it here for simplicity.
  if (!cpi->visual_energy_map || cpi->common.width != source_w || cpi->common.height != source_h) {
    vpx_free(cpi->visual_energy_map); // free existing map if dimensions changed
    cpi->visual_energy_map = (unsigned char *)vpx_malloc(map_size * sizeof(unsigned char));
    if (!cpi->visual_energy_map) {
      // Allocation failed, handle error (e.g., log and return)
      return;
    }
    // Update dimensions in common if we reallocated due to dimension change,
    // though this might be better handled at a higher level.
    // cpi->common.width = source_w; // This should be done when cpi is updated
    // cpi->common.height = source_h;
  }


  for (int r = 0; r < source_h; r += block_h) {
    for (int c = 0; c < source_w; c += block_w) {
      const uint8_t *block_src = src + r * stride + c;
      unsigned int variance;

      // Ensure we don't read past buffer boundaries for partial blocks
      const int current_block_w = VPXMIN(block_w, source_w - c);
      const int current_block_h = VPXMIN(block_h, source_h - r);

      // If the block is smaller than the target size, we might need to handle it differently
      // For now, only calculate variance for full blocks
      if (current_block_w == block_w && current_block_h == block_h) {
        variance = vf(block_src, stride, block_src, stride, &sse);
      } else {
        // For partial blocks, set variance to a default low value or skip
        variance = 0; // Placeholder
      }

      // Store the variance. Clamping or normalization might be needed.
      // For now, directly store the lower 8 bits if variance > 255.
      if (map_idx < map_size) { // Ensure we don't write out of bounds
        cpi->visual_energy_map[map_idx++] = (unsigned char)(VPXMIN(variance, 255));
      } else {
        // Should not happen if map_size is calculated correctly
        break;
      }
    }
    if (map_idx >= map_size && r < source_h - block_h) {
        // Should not happen if map_size is calculated correctly
        break;
    }
  }
}
