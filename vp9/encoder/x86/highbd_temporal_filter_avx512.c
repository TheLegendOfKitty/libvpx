/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <immintrin.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "vp9/encoder/vp9_temporal_filter.h"

static INLINE void highbd_shuffle_12tap_filter_avx512(const int16_t *filter,
                                                      __m512i *f) {
  // Load filter coefficients and broadcast to 512-bit registers
  const __m256i f_256_low = 
      _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)filter));
  const __m256i f_256_high = _mm256_broadcastsi128_si256(
      _mm_loadl_epi64((const __m128i *)(filter + 8)));

  // Convert to 512-bit and shuffle using insert
  const __m512i f_low = _mm512_inserti64x4(_mm512_castsi256_si512(f_256_low), f_256_low, 1);
  const __m512i f_high = _mm512_inserti64x4(_mm512_castsi256_si512(f_256_high), f_256_high, 1);

  f[0] = _mm512_shuffle_epi32(f_low, 0x00);
  f[1] = _mm512_shuffle_epi32(f_low, 0x55);
  f[2] = _mm512_shuffle_epi32(f_low, 0xaa);
  f[3] = _mm512_shuffle_epi32(f_low, 0xff);
  f[4] = _mm512_shuffle_epi32(f_high, 0x00);
  f[5] = _mm512_shuffle_epi32(f_high, 0x55);
}

static INLINE __m512i highbd_convolve_12tap_avx512(const __m512i *s,
                                                   const __m512i *f) {
  const __m512i res_0 = _mm512_madd_epi16(s[0], f[0]);
  const __m512i res_1 = _mm512_madd_epi16(s[1], f[1]);
  const __m512i res_2 = _mm512_madd_epi16(s[2], f[2]);
  const __m512i res_3 = _mm512_madd_epi16(s[3], f[3]);
  const __m512i res_4 = _mm512_madd_epi16(s[4], f[4]);
  const __m512i res_5 = _mm512_madd_epi16(s[5], f[5]);

  const __m512i res =
      _mm512_add_epi32(_mm512_add_epi32(res_0, res_1),
                       _mm512_add_epi32(_mm512_add_epi32(res_2, res_3),
                                        _mm512_add_epi32(res_4, res_5)));
  return res;
}

static INLINE void reuse_src_data_avx512(const __m512i *src, __m512i *des) {
  des[0] = src[0];
  des[1] = src[1];
  des[2] = src[2];
  des[3] = src[3];
  des[4] = src[4];
}

void vpx_highbd_convolve12_horiz_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                        uint16_t *dst, ptrdiff_t dst_stride,
                                        const InterpKernel12 *filter, int x0_q4,
                                        int x_step_q4, int y0_q4, int y_step_q4,
                                        int w, int h, int bd) {
  assert(x_step_q4 == 16);
  (void)y0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint16_t *src_ptr = src;
  src_ptr -= MAX_FILTER_TAP / 2 - 1;
  __m512i s[6], f[6];
  const __m512i rounding = _mm512_set1_epi32(1 << (FILTER_BITS - 1));
  const __m512i max = _mm512_set1_epi16((1 << bd) - 1);
  highbd_shuffle_12tap_filter_avx512(filter[x0_q4], f);

  for (int j = 0; j < w; j += 16) {
    for (int i = 0; i < h; i += 2) {
      // Load 16 pixels per row for 512-bit processing
      const __m512i row0 =
          _mm512_loadu_si512((const __m512i *)&src_ptr[i * src_stride + j]);
      const __m512i row1 = 
          _mm512_loadu_si512((const __m512i *)&src_ptr[(i + 1) * src_stride + j]);
      
      // Load additional pixels for filter taps
      const __m256i row0_32 =
          _mm256_loadu_si256((const __m256i *)&src_ptr[i * src_stride + j + 16]);
      const __m256i row1_32 = 
          _mm256_loadu_si256((const __m256i *)&src_ptr[(i + 1) * src_stride + j + 16]);

      // Combine data for convolution
      const __m512i r0 = row0;
      const __m512i r1 = row1;
      const __m512i r2 = _mm512_inserti64x4(_mm512_castsi256_si512(row0_32), row1_32, 1);

      // Setup source data for even pixels
      s[0] = r0;
      s[1] = _mm512_alignr_epi8(r1, r0, 4);
      s[2] = _mm512_alignr_epi8(r1, r0, 8);
      s[3] = _mm512_alignr_epi8(r1, r0, 12);
      s[4] = r1;
      s[5] = _mm512_alignr_epi8(r2, r1, 4);

      // Process even pixels (16 elements)
      __m512i res_even = highbd_convolve_12tap_avx512(s, f);
      res_even = _mm512_srai_epi32(_mm512_add_epi32(res_even, rounding), FILTER_BITS);

      // Setup source data for odd pixels
      s[0] = _mm512_alignr_epi8(r1, r0, 2);
      s[1] = _mm512_alignr_epi8(r1, r0, 6);
      s[2] = _mm512_alignr_epi8(r1, r0, 10);
      s[3] = _mm512_alignr_epi8(r1, r0, 14);
      s[4] = _mm512_alignr_epi8(r2, r1, 2);
      s[5] = _mm512_alignr_epi8(r2, r1, 6);

      // Process odd pixels (16 elements)
      __m512i res_odd = highbd_convolve_12tap_avx512(s, f);
      res_odd = _mm512_srai_epi32(_mm512_add_epi32(res_odd, rounding), FILTER_BITS);

      // Interleave even and odd results
      const __m512i res_0 = _mm512_unpacklo_epi32(res_even, res_odd);
      const __m512i res_1 = _mm512_unpackhi_epi32(res_even, res_odd);
      
      // Pack to 16-bit and clamp
      const __m512i res_2 = _mm512_packus_epi32(res_0, res_1);
      const __m512i res = _mm512_min_epi16(res_2, max);
      
      // Store results (16 pixels per row)
      _mm256_storeu_si256((__m256i *)&dst[i * dst_stride + j],
                         _mm512_castsi512_si256(res));
      if (i + 1 < h) {
        _mm256_storeu_si256((__m256i *)(&dst[(i + 1) * dst_stride + j]),
                           _mm512_extracti64x4_epi64(res, 1));
      }
    }
  }
}

void vpx_highbd_convolve12_vert_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                       uint16_t *dst, ptrdiff_t dst_stride,
                                       const InterpKernel12 *filter, int x0_q4,
                                       int x_step_q4, int y0_q4, int y_step_q4,
                                       int w, int h, int bd) {
  assert(y_step_q4 == 16);
  (void)x0_q4;
  (void)x_step_q4;
  (void)y_step_q4;
  const uint16_t *src_ptr = src;
  src_ptr -= src_stride * (MAX_FILTER_TAP / 2 - 1);
  __m512i s[12], f[6];
  const __m512i rounding = _mm512_set1_epi32(((1 << FILTER_BITS) >> 1));
  const __m512i max = _mm512_set1_epi16((1 << bd) - 1);
  highbd_shuffle_12tap_filter_avx512(filter[y0_q4], f);

  for (int j = 0; j < w; j += 16) {
    // Load initial rows (16 pixels per row using 512-bit loads)
    __m256i s0 = _mm256_loadu_si256((const __m256i *)(src_ptr + 0 * src_stride + j));
    __m256i s1 = _mm256_loadu_si256((const __m256i *)(src_ptr + 1 * src_stride + j));
    __m256i s2 = _mm256_loadu_si256((const __m256i *)(src_ptr + 2 * src_stride + j));
    __m256i s3 = _mm256_loadu_si256((const __m256i *)(src_ptr + 3 * src_stride + j));
    __m256i s4 = _mm256_loadu_si256((const __m256i *)(src_ptr + 4 * src_stride + j));
    __m256i s5 = _mm256_loadu_si256((const __m256i *)(src_ptr + 5 * src_stride + j));
    __m256i s6 = _mm256_loadu_si256((const __m256i *)(src_ptr + 6 * src_stride + j));
    __m256i s7 = _mm256_loadu_si256((const __m256i *)(src_ptr + 7 * src_stride + j));
    __m256i s8 = _mm256_loadu_si256((const __m256i *)(src_ptr + 8 * src_stride + j));
    __m256i s9 = _mm256_loadu_si256((const __m256i *)(src_ptr + 9 * src_stride + j));
    __m256i s10t = _mm256_loadu_si256((const __m256i *)(src_ptr + 10 * src_stride + j));

    // Combine pairs of rows into 512-bit registers
    __m512i r01 = _mm512_inserti64x4(_mm512_castsi256_si512(s0), s1, 1);
    __m512i r12 = _mm512_inserti64x4(_mm512_castsi256_si512(s1), s2, 1);
    __m512i r23 = _mm512_inserti64x4(_mm512_castsi256_si512(s2), s3, 1);
    __m512i r34 = _mm512_inserti64x4(_mm512_castsi256_si512(s3), s4, 1);
    __m512i r45 = _mm512_inserti64x4(_mm512_castsi256_si512(s4), s5, 1);
    __m512i r56 = _mm512_inserti64x4(_mm512_castsi256_si512(s5), s6, 1);
    __m512i r67 = _mm512_inserti64x4(_mm512_castsi256_si512(s6), s7, 1);
    __m512i r78 = _mm512_inserti64x4(_mm512_castsi256_si512(s7), s8, 1);
    __m512i r89 = _mm512_inserti64x4(_mm512_castsi256_si512(s8), s9, 1);
    __m512i r910 = _mm512_inserti64x4(_mm512_castsi256_si512(s9), s10t, 1);

    // Interleave for vertical filtering
    s[0] = _mm512_unpacklo_epi16(r01, r12);
    s[1] = _mm512_unpacklo_epi16(r23, r34);
    s[2] = _mm512_unpacklo_epi16(r45, r56);
    s[3] = _mm512_unpacklo_epi16(r67, r78);
    s[4] = _mm512_unpacklo_epi16(r89, r910);

    s[6] = _mm512_unpackhi_epi16(r01, r12);
    s[7] = _mm512_unpackhi_epi16(r23, r34);
    s[8] = _mm512_unpackhi_epi16(r45, r56);
    s[9] = _mm512_unpackhi_epi16(r67, r78);
    s[10] = _mm512_unpackhi_epi16(r89, r910);
    
    for (int i = 0; i < h; i += 2) {
      // Load new rows for sliding window
      const __m256i s10 = _mm256_loadu_si256(
          (const __m256i *)(src_ptr + (i + 10) * src_stride + j));
      const __m256i s11 = _mm256_loadu_si256(
          (const __m256i *)(src_ptr + (i + 11) * src_stride + j));
      const __m256i s12 = _mm256_loadu_si256(
          (const __m256i *)(src_ptr + (i + 12) * src_stride + j));
      
      __m512i r1011 = _mm512_inserti64x4(_mm512_castsi256_si512(s10), s11, 1);
      __m512i r1112 = _mm512_inserti64x4(_mm512_castsi256_si512(s11), s12, 1);

      s[5] = _mm512_unpacklo_epi16(r1011, r1112);
      s[11] = _mm512_unpackhi_epi16(r1011, r1112);

      // Process lower 8 pixels of each row
      const __m512i res_a = highbd_convolve_12tap_avx512(s, f);
      __m512i res_a_round = _mm512_srai_epi32(_mm512_add_epi32(res_a, rounding), FILTER_BITS);
      
      // Process upper 8 pixels of each row
      const __m512i res_b = highbd_convolve_12tap_avx512(s + 6, f);
      __m512i res_b_round = _mm512_srai_epi32(_mm512_add_epi32(res_b, rounding), FILTER_BITS);

      // Pack and clamp results using manual shuffle since _mm512_packus_epi64 doesn't exist
      const __m512i res_0 = _mm512_packus_epi32(res_a_round, res_b_round);
      const __m512i res = _mm512_min_epi16(res_0, max);
      
      // Store results (16 pixels per row)
      _mm256_storeu_si256((__m256i *)&dst[i * dst_stride + j],
                         _mm512_castsi512_si256(res));
      _mm256_storeu_si256((__m256i *)(&dst[(i + 1) * dst_stride + j]),
                         _mm512_extracti64x4_epi64(res, 1));

      // Shift data for next iteration
      reuse_src_data_avx512(s + 1, s);
      reuse_src_data_avx512(s + 7, s + 6);
    }
  }
}

void vpx_highbd_convolve12_avx512(const uint16_t *src, ptrdiff_t src_stride,
                                  uint16_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel12 *filter, int x0_q4,
                                  int x_step_q4, int y0_q4, int y_step_q4, int w,
                                  int h, int bd) {
  assert(x_step_q4 == 16 && y_step_q4 == 16);
  assert(h == 32 || h == 16 || h == 8);
  assert(w == 32 || w == 16 || w == 8);
  DECLARE_ALIGNED(64, uint16_t, temp[BW * (BH + MAX_FILTER_TAP - 1)]);
  const int temp_stride = BW;
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + MAX_FILTER_TAP;

  vpx_highbd_convolve12_horiz_avx512(src - src_stride * (MAX_FILTER_TAP / 2 - 1),
                                     src_stride, temp, temp_stride, filter, x0_q4,
                                     x_step_q4, y0_q4, y_step_q4, w,
                                     intermediate_height, bd);
  vpx_highbd_convolve12_vert_avx512(temp + temp_stride * (MAX_FILTER_TAP / 2 - 1),
                                    temp_stride, dst, dst_stride, filter, x0_q4,
                                    x_step_q4, y0_q4, y_step_q4, w, h, bd);
}