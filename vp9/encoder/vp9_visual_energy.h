/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_VISUAL_ENERGY_H_
#define VP9_ENCODER_VP9_VISUAL_ENERGY_H_

#include "vpx_scale/yv12config.h"

// Forward declaration of VP9_COMP
struct VP9_COMP;

#ifdef __cplusplus
extern "C" {
#endif

// Function to calculate visual energy for a source frame.
// The visual energy map is stored within the cpi structure.
// The dimensions and interpretation of this map will depend on the specific
// algorithm used.
void vp9_calculate_visual_energy(struct VP9_COMP *cpi,
                                 const YV12_BUFFER_CONFIG *source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_VISUAL_ENERGY_H_
