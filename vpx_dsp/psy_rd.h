#ifndef VPX_VPX_DSP_PSY_RD_H_
#define VPX_VPX_DSP_PSY_RD_H_

#include <stdint.h>
#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

uint64_t vp9_get_psy_full_dist(const void *s, uint32_t so, uint32_t sp,
                               const void *r, uint32_t ro, uint32_t rp,
                               uint32_t w, uint32_t h, int is_hbd,
                               double psy_rd_strength);

#ifdef __cplusplus
}
#endif

#endif  // VPX_VPX_DSP_PSY_RD_H_
