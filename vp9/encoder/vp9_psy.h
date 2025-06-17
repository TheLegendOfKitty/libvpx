#ifndef VP9_ENCODER_VP9_PSY_H_
#define VP9_ENCODER_VP9_PSY_H_

#include "vp9/encoder/vp9_encoder.h"

int64_t vp9_psy_rd_dist(const uint8_t *src, int src_stride, const uint8_t *dst,
                        int dst_stride, int width, int height, int use_hbd,
                        double psy_strength, double bitrate_bias);

#endif  // VP9_ENCODER_VP9_PSY_H_
