/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "third_party/googletest/src/include/gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/acm_random.h"
#include "vpx/vpx_encoder.h"

extern "C" {
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ratectrl.h"
// To test static functions, we include the .c file.
#include "vp9/encoder/vp9_rdopt.c"
// For VISUAL_ENERGY_BLOCK_SIZE, if needed, though not directly used here.
// #include "vp9/encoder/vp9_visual_energy.h"
} // extern "C"

namespace {

// Test fixture for psychovisual RD tests
class PsyRdTest : public ::libvpx_test::EncoderTest,
                  public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  PsyRdTest() : EncoderTest(&libvpx_test::kVP9) {}
  ~PsyRdTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(0));
    // Basic config
    cfg_.g_w = 128;
    cfg_.g_h = 128;
    cfg_.rc_target_bitrate = 1000;
    cfg_.g_error_resilient = 0;
    cfg_.g_profile = 0;
    cfg_.g_pass = VPX_RC_ONE_PASS;
    cfg_.g_input_bit_depth = 8;
#if CONFIG_VP9_HIGHBITDEPTH
    cfg_.g_bit_depth = VPX_BITS_8;
#endif
    // Default to psy-rd off for baseline comparisons unless specified in test
    cfg_.enable_psy_rd = 0;
  }
};


TEST_P(PsyRdTest, ApplyVisualEnergyWeight) {
  // Test the vp9_apply_visual_energy_weight function from vp9_rdopt.c
  // Formula used in implementation: (int64_t)(distortion * (1.0 + (double)visual_energy / 255.0))

  const int64_t base_distortion = 10000;

  // Visual energy = 0 (no effect)
  int visual_energy_0 = 0;
  int64_t weighted_dist_0 = vp9_apply_visual_energy_weight(base_distortion, visual_energy_0);
  EXPECT_EQ(weighted_dist_0, base_distortion);

  // Visual energy = 255 (max effect, distortion * 2.0)
  int visual_energy_max = 255;
  int64_t weighted_dist_max = vp9_apply_visual_energy_weight(base_distortion, visual_energy_max);
  EXPECT_EQ(weighted_dist_max, (int64_t)(base_distortion * (1.0 + 255.0 / 255.0)));
  EXPECT_EQ(weighted_dist_max, base_distortion * 2);


  // Visual energy = 128 (approx mid-range)
  int visual_energy_mid = 128;
  int64_t weighted_dist_mid = vp9_apply_visual_energy_weight(base_distortion, visual_energy_mid);
  EXPECT_EQ(weighted_dist_mid, (int64_t)(base_distortion * (1.0 + 128.0 / 255.0)));
}

TEST_P(PsyRdTest, RegulateQAdjustment) {
  vpx_codec_ctx_t enc;
  // Initialize with default psy_rd = 0 from fixture SetUp
  ASSERT_EQ(VPX_CODEC_OK, vpx_codec_enc_init(&enc, CodecInterface(), &cfg_, 0));
  VP9_COMP *cpi = reinterpret_cast<VP9_COMP *>(enc.priv);
  ASSERT_NE(cpi, nullptr);

  RATE_CONTROL *rc = &cpi->rc;
  VP9_COMMON *cm = &cpi->common;

  // Setup basic RC parameters for a predictable baseline
  rc->best_quality = 10; // Arbitrary but within typical range
  rc->worst_quality = 200; // Arbitrary
  rc->avg_frame_bandwidth = 10000;
  rc->this_frame_target = 10000;
  cm->MBs = (cfg_.g_w * cfg_.g_h) / (16 * 16);
#if CONFIG_VP9_HIGHBITDEPTH
  cm->bit_depth = cfg_.g_bit_depth;
#else
  cm->bit_depth = VPX_BITS_8;
#endif
  cm->frame_type = INTER_FRAME; // typical case for this function
  vp9_rc_init_minq_luts();

  // Save original function pointers that might be changed by speed features
  vp9_variance_fn_ptr_t original_fn_ptr[BLOCK_SIZES];
  memcpy(original_fn_ptr, cpi->fn_ptr, sizeof(original_fn_ptr));
  SPEED_FEATURES original_sf = cpi->sf;
  vp9_set_speed_features_framesize_independent(cpi, 0); // Speed 0 for predictability
  vp9_set_speed_features_framesize_dependent(cpi, 0);

  // Scenario 1: Psy RD disabled
  cpi->oxcf.enable_psy_rd = 0;
  rc->avg_visual_energy = 128; // Some mid-range energy, should have no effect
  int bottom_idx_disabled, top_idx_disabled;
  // We need to ensure other RC params that affect bounds are somewhat stable/known
  // For one-pass CBR, which vp9_rc_pick_q_and_bounds covers:
  rc->avg_frame_qindex[INTER_FRAME] = 100; // A typical mid-range Q
  rc->last_q[INTER_FRAME] = 100;
  rc->buffer_level = rc->optimal_buffer_level; // Neutral buffer state

  vp9_rc_pick_q_and_bounds(cpi, &bottom_idx_disabled, &top_idx_disabled);

  // Scenario 2: Psy RD enabled, avg_visual_energy = 0 (low energy)
  cpi->oxcf.enable_psy_rd = 1;
  rc->avg_visual_energy = 0.0;
  int bottom_idx_low_energy, top_idx_low_energy;
  vp9_rc_pick_q_and_bounds(cpi, &bottom_idx_low_energy, &top_idx_low_energy);
  // With energy_factor = 1.0, bounds should be identical to disabled case
  EXPECT_EQ(bottom_idx_low_energy, bottom_idx_disabled);
  EXPECT_EQ(top_idx_low_energy, top_idx_disabled);

  // Scenario 3: Psy RD enabled, avg_visual_energy = 255 (high energy)
  // energy_factor = 1.0 - (255.0 / 255.0) * 0.25 = 0.75
  rc->avg_visual_energy = 255.0;
  int bottom_idx_high_energy, top_idx_high_energy;
  vp9_rc_pick_q_and_bounds(cpi, &bottom_idx_high_energy, &top_idx_high_energy);
  EXPECT_LE(top_idx_high_energy, top_idx_disabled);   // Should be lower or equal (due to clamping)
  EXPECT_LE(bottom_idx_high_energy, bottom_idx_disabled); // Should be lower or equal

  // Check if it's significantly lower (e.g. at least 10-20% for high energy if not clamped)
  if (bottom_idx_disabled > rc->best_quality + 10) { // Avoid issues if already at best_quality
    EXPECT_LT(bottom_idx_high_energy, bottom_idx_disabled);
  }
   if (top_idx_disabled > rc->best_quality + 10) {
    EXPECT_LT(top_idx_high_energy, top_idx_disabled);
  }


  // Scenario 4: Psy RD enabled, avg_visual_energy = 128 (mid energy)
  // energy_factor = 1.0 - (128.0 / 255.0) * 0.25 approx 1.0 - 0.5 * 0.25 = 0.875
  rc->avg_visual_energy = 128.0;
  int bottom_idx_mid_energy, top_idx_mid_energy;
  vp9_rc_pick_q_and_bounds(cpi, &bottom_idx_mid_energy, &top_idx_mid_energy);
  EXPECT_LE(top_idx_mid_energy, top_idx_disabled);
  EXPECT_GE(top_idx_mid_energy, top_idx_high_energy);
  EXPECT_LE(bottom_idx_mid_energy, bottom_idx_disabled);
  EXPECT_GE(bottom_idx_mid_energy, bottom_idx_high_energy);

  // Restore original function pointers and speed features
  memcpy(cpi->fn_ptr, original_fn_ptr, sizeof(original_fn_ptr));
  cpi->sf = original_sf;

  ASSERT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&enc));
}

VP9_INSTANTIATE_TEST_CASE(PsyRdTest, ALL_TEST_MODES);

}  // namespace
