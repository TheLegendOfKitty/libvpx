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
#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_convolve.h"

void vpx_scaled_horiz_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                           ptrdiff_t dst_stride, const InterpKernel *filter,
                           int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                           int w, int h) {
  vpx_convolve8_horiz_neon(src, src_stride, dst, dst_stride, filter, x0_q4,
                           x_step_q4, y0_q4, y_step_q4, w, h);
}

void vpx_scaled_vert_neon(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                          ptrdiff_t dst_stride, const InterpKernel *filter,
                          int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                          int w, int h) {
  vpx_convolve8_vert_neon(src, src_stride, dst, dst_stride, filter, x0_q4,
                          x_step_q4, y0_q4, y_step_q4, w, h);
}


void vpx_scaled_avg_horiz_neon(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int x_step_q4, int y0_q4, int y_step_q4,
                               int w, int h) {
  vpx_convolve8_avg_horiz_neon(src, src_stride, dst, dst_stride, filter, x0_q4,
                               x_step_q4, y0_q4, y_step_q4, w, h);
}

void vpx_scaled_avg_vert_neon(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int x0_q4,
                              int x_step_q4, int y0_q4, int y_step_q4,
                              int w, int h) {
  vpx_convolve8_avg_vert_neon(src, src_stride, dst, dst_stride, filter, x0_q4,
                              x_step_q4, y0_q4, y_step_q4, w, h);
}

void vpx_scaled_avg_2d_neon(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter, int x0_q4,
                            int x_step_q4, int y0_q4, int y_step_q4,
                            int w, int h) {
  vpx_convolve8_avg_neon(src, src_stride, dst, dst_stride, filter, x0_q4,
                         x_step_q4, y0_q4, y_step_q4, w, h);
}