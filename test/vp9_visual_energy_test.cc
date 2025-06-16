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

// Needs to be included after vpx_encoder.h to avoid issues with VP9_COMP on some platforms
extern "C" {
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_visual_energy.h"
#include "vp9/common/vp9_common.h" // For block_size_wide, block_size_high
}

namespace {

const int kTestBlockSize = 16; // VISUAL_ENERGY_BLOCK_SIZE is BLOCK_16X16

class VisualEnergyTest : public ::libvpx_test::EncoderTest,
                         public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  VisualEnergyTest() : EncoderTest(&libvpx_test::kVP9) {}
  ~VisualEnergyTest() override = default;

  void SetUp() override {
    InitializeConfig(GET_PARAM(0));
    // Default configuration parameters for the test.
    cfg_.g_w = 64;
    cfg_.g_h = 32; // Test non-square
    cfg_.rc_target_bitrate = 1000;
    cfg_.g_lag_in_frames = 0;
    cfg_.g_error_resilient = 0;
    cfg_.g_profile = 0;
    cfg_.g_pass = VPX_RC_ONE_PASS;
    cfg_.g_input_bit_depth = 8;
#if CONFIG_VP9_HIGHBITDEPTH
    cfg_.g_bit_depth = VPX_BITS_8; // Keep it simple for this test
#endif

    // Enable psychovisual RD (though it's directly calling the function)
    cfg_.enable_psy_rd = 1;
  }

  void CreateSourceFrame(int width, int height, libvpx_test::ACMRandom *rnd, bool specific_pattern = false) {
    source_ = new libvpx_test::Source;
    ASSERT_NE(source_, nullptr);
    source_->Init(width, height, cfg_.g_timebase.den, cfg_.g_timebase.num, 0, 1);
    source_->frame_ = vpx_img_alloc(nullptr, VPX_IMG_FMT_I420, width, height, 16);
    ASSERT_NE(source_->frame_, nullptr);

    for (int r = 0; r < height; ++r) {
      for (int c = 0; c < width; ++c) {
        if (specific_pattern && r < kTestBlockSize && c < kTestBlockSize) {
          // First block with a constant value for predictable variance
          source_->frame_->planes[VPX_PLANE_Y][r * source_->frame_->stride[VPX_PLANE_Y] + c] = 100;
        } else {
          source_->frame_->planes[VPX_PLANE_Y][r * source_->frame_->stride[VPX_PLANE_Y] + c] = rnd->Rand8();
        }
      }
    }
    // Initialize UV planes too, though not used by current visual energy function
    for (int plane = VPX_PLANE_U; plane <= VPX_PLANE_V; ++plane) {
        const int uv_height = (height + source_->frame_->y_chroma_shift) >> source_->frame_->y_chroma_shift;
        const int uv_width = (width + source_->frame_->x_chroma_shift) >> source_->frame_->x_chroma_shift;
        for (int r = 0; r < uv_height; ++r) {
            for (int c = 0; c < uv_width; ++c) {
                source_->frame_->planes[plane][r * source_->frame_->stride[plane] + c] = rnd->Rand8();
            }
        }
    }
  }

  void TearDown() override {
    if (source_ != nullptr) {
      vpx_img_free(source_->frame_);
      delete source_;
      source_ = nullptr;
    }
  }

  libvpx_test::Source *source_ = nullptr;
};

TEST_P(VisualEnergyTest, CalculateVisualEnergyMap) {
  libvpx_test::ACMRandom rnd(0); // Use a fixed seed for reproducibility if needed
  const int width = cfg_.g_w;
  const int height = cfg_.g_h;
  CreateSourceFrame(width, height, &rnd, true); // Use specific pattern for first block

  vpx_codec_ctx_t enc;
  ASSERT_EQ(VPX_CODEC_OK, vpx_codec_enc_init(&enc, CodecInterface(), &cfg_, 0));

  VP9_COMP *cpi = reinterpret_cast<VP9_COMP *>(enc.priv);
  ASSERT_NE(cpi, nullptr);

  // Simulate the state where cpi->Source is set
  cpi->Source = source_->frame_;
  cpi->common.width = width;
  cpi->common.height = height;
  cpi->common.subsampling_x = source_->frame_->x_chroma_shift;
  cpi->common.subsampling_y = source_->frame_->y_chroma_shift;
#if CONFIG_VP9_HIGHBITDEPTH
  cpi->common.use_highbitdepth = (source_->frame_->fmt & VPX_IMG_FMT_HIGHBITDEPTH) != 0;
  cpi->td.mb.e_mbd.bd = cfg_.g_bit_depth;
  cpi->common.bit_depth = cfg_.g_bit_depth;
#endif

  // The variance functions should be initialized by vpx_codec_enc_init
  // through init_config -> vp9_change_config -> BFP macros.
  // If not, vp9_initialize_fn_ptr(cpi); might be needed, but it's better if init handles it.

  // Call the function to test
  vp9_calculate_visual_energy(cpi, cpi->Source);

  ASSERT_NE(cpi->visual_energy_map, nullptr) << "visual_energy_map is NULL";

  const int map_block_w_pixels = block_size_wide[VISUAL_ENERGY_BLOCK_SIZE];
  const int map_block_h_pixels = block_size_high[VISUAL_ENERGY_BLOCK_SIZE];
  const int map_w = (width + map_block_w_pixels - 1) / map_block_w_pixels;
  const int map_h = (height + map_block_h_pixels - 1) / map_block_h_pixels;
  const int map_size = map_w * map_h;

  // Verification:
  // 1. First block (0,0) was filled with a constant value (100). Its variance should be 0.
  if (map_size > 0) {
    EXPECT_EQ(cpi->visual_energy_map[0], 0) << "Variance of constant block (0,0) should be 0.";
  }

  // 2. For other blocks (filled with random data), check if values are within [0, 255]
  for (int i = 1; i < map_size; ++i) { // Start from 1 if first block is special
    EXPECT_GE(cpi->visual_energy_map[i], 0);
    EXPECT_LE(cpi->visual_energy_map[i], 255);
  }

  // 3. Test partial blocks if frame size is not a multiple of block size
  // Example: If width = 60, height = 30, kTestBlockSize = 16
  // map_w = (60+15)/16 = 4
  // map_h = (30+15)/16 = 2
  // Block (3,0) would be partial width. Block (0,1) would be partial height.
  // Block (3,1) would be partial width and height.
  // Current implementation sets variance to 0 for partial blocks.
  if (width % kTestBlockSize != 0 && map_w > 0 && map_h > 0) {
      EXPECT_EQ(cpi->visual_energy_map[map_w - 1], 0) << "Variance of partial width block should be 0.";
  }
  if (height % kTestBlockSize != 0 && map_h > 0 && map_w > 0) {
       EXPECT_EQ(cpi->visual_energy_map[(map_h - 1) * map_w], 0) << "Variance of partial height block should be 0.";
  }
   if (width % kTestBlockSize != 0 && height % kTestBlockSize != 0 && map_w > 0 && map_h > 0) {
       EXPECT_EQ(cpi->visual_energy_map[map_size - 1], 0) << "Variance of partial width/height block should be 0.";
   }


  // Cleanup
  ASSERT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&enc));
  // cpi->visual_energy_map is freed by dealloc_compressor_data via vpx_codec_destroy
}

VP9_INSTANTIATE_TEST_CASE(VisualEnergyTest, ALL_TEST_MODES);

}  // namespace
