#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "vpx_dsp/psy_rd.h"
#include "vpx_dsp/vpx_dsp_common.h"

typedef int32_t sum_t;
typedef int64_t sum2_t;

#define BITS_PER_SUM (8 * sizeof(sum_t))

#define HADAMARD4(d0, d1, d2, d3, s0, s1, s2, s3) { \
        sum2_t t0 = s0 + s1; \
        sum2_t t1 = s0 - s1; \
        sum2_t t2 = s2 + s3; \
        sum2_t t3 = s2 - s3; \
        d0 = t0 + t2; \
        d2 = t0 - t2; \
        d1 = t1 + t3; \
        d3 = t1 - t3; \
}

static inline sum2_t abs2(sum2_t a)
{
    const sum2_t mask = (a >> (BITS_PER_SUM - 1)) & (((sum2_t)1 << BITS_PER_SUM) + 1);
    const sum2_t s = (mask << BITS_PER_SUM) - mask;
    return (a + s) ^ s;
}

static uint64_t vp9_sa8d_8x8(const uint8_t* s, uint32_t sp, const uint8_t* r, uint32_t rp) {
    sum2_t tmp[8][4];
    sum2_t a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3;
    sum2_t sum = 0;

    for (int i = 0; i < 8; i++, s += sp, r += rp) {
        a0 = s[0] - r[0];
        a1 = s[1] - r[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM);
        a2 = s[2] - r[2];
        a3 = s[3] - r[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM);
        a4 = s[4] - r[4];
        a5 = s[5] - r[5];
        b2 = (a4 + a5) + ((a4 - a5) << BITS_PER_SUM);
        a6 = s[6] - r[6];
        a7 = s[7] - r[7];
        b3 = (a6 + a7) + ((a6 - a7) << BITS_PER_SUM);
        HADAMARD4(tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], b0, b1, b2, b3)
    }
    for (int i = 0; i < 4; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i])
        HADAMARD4(a4, a5, a6, a7, tmp[4][i], tmp[5][i], tmp[6][i], tmp[7][i])
        b0  = abs2(a0 + a4) + abs2(a0 - a4);
        b0 += abs2(a1 + a5) + abs2(a1 - a5);
        b0 += abs2(a2 + a6) + abs2(a2 - a6);
        b0 += abs2(a3 + a7) + abs2(a3 - a7);
        sum += (sum_t)b0 + (b0 >> BITS_PER_SUM);
    }

    return (uint64_t)((sum + 2) >> 2);
}

static uint64_t vp9_satd_4x4(const uint8_t* s, uint32_t sp, const uint8_t* r, uint32_t rp) {
    sum2_t tmp[4][2];
    sum2_t a0, a1, a2, a3, b0, b1;
    sum2_t sum = 0;

    for (int i = 0; i < 4; i++, s += sp, r += rp) {
        a0 = s[0] - r[0];
        a1 = s[1] - r[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM);
        a2 = s[2] - r[2];
        a3 = s[3] - r[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM);
        tmp[i][0] = b0 + b1;
        tmp[i][1] = b0 - b1;
    }
    for (int i = 0; i < 2; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i])
        a0 = abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
        sum += ((sum_t)a0) + (a0 >> BITS_PER_SUM);
    }

    return (uint64_t)(sum >> 1);
}

static uint64_t vp9_psy_sad_nxn(uint32_t w, uint32_t h, const uint8_t* s,
                                uint32_t sp, const uint8_t* r, uint32_t rp) {
    int sum = 0;

    for (uint32_t i = 0; i < h; i++) {
        for (uint32_t j = 0; j < w; j++) {
            sum += abs(s[j] - r[j]);
        }
        s += sp;
        r += rp;
    }

    return sum;
}

static uint64_t vp9_psy_distortion(const uint8_t* input, uint32_t input_stride,
                            const uint8_t* recon, uint32_t recon_stride,
                            uint32_t width, uint32_t height) {

    static uint8_t zero_buffer[8] = { 0 };
    uint64_t total_nrg = 0;

    if (width >= 8 && height >= 8) {
        for (uint32_t i = 0; i < height; i += 8) {
            for (uint32_t j = 0; j < width; j += 8) {
                int32_t input_nrg = (int32_t)(vp9_sa8d_8x8(input + i * input_stride + j, input_stride, zero_buffer, 0)) -
                    (int32_t)(vp9_psy_sad_nxn(8, 8, input + i * input_stride + j, input_stride, zero_buffer, 0) >> 2);
                int32_t recon_nrg = (int32_t)(vp9_sa8d_8x8(recon + i * recon_stride + j, recon_stride, zero_buffer, 0)) -
                    (int32_t)(vp9_psy_sad_nxn(8, 8, recon + i * recon_stride + j, recon_stride, zero_buffer, 0) >> 2);
                total_nrg += abs(input_nrg - recon_nrg);
            }
        }
    } else {
        for (uint32_t i = 0; i < height; i += 4) {
            for (uint32_t j = 0; j < width; j += 4) {
                int32_t input_nrg = (int32_t)vp9_satd_4x4(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                    (int32_t)(vp9_psy_sad_nxn(4, 4, input + i * input_stride + j, input_stride, zero_buffer, 0) >> 2);
                int32_t recon_nrg = (int32_t)vp9_satd_4x4(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                    (int32_t)(vp9_psy_sad_nxn(4, 4, recon + i * recon_stride + j, recon_stride, zero_buffer, 0) >> 2);
                total_nrg += abs(input_nrg - recon_nrg);
            }
        }
    }
    return (total_nrg >> 1);
}

uint64_t vp9_get_psy_full_dist(const void* s, uint32_t so, uint32_t sp,
                               const void* r, uint32_t ro, uint32_t rp,
                               uint32_t w, uint32_t h, int is_hbd,
                               double psy_rd_strength) {
    uint64_t dist;

    if (is_hbd) {
        // HBD not supported yet
        dist = 0;
    } else {
        dist = vp9_psy_distortion((const uint8_t*)s + so, sp, (const uint8_t*)r + ro, rp, w, h);
    }

    return (uint64_t)(dist * psy_rd_strength);
}
