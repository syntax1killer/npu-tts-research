// LayerNorm kernel for AIE2P
//
// LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Kernel signature: layernorm_bf16(x_in, params, x_out)
//   x_in:   [LN_DIM] bf16 input
//   params: [2 * LN_DIM] bf16 — first half is gamma, second half is beta
//   x_out:  [LN_DIM] bf16 output
//
// Based on IRON reference layer_norm.cc patterns.
// Uses bf16 vectors of 16 for computation.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LN_DIM
#define LN_DIM 256
#endif

constexpr unsigned VEC = 16;

extern "C" {

void layernorm_bf16(bfloat16 *__restrict x_in,
                    bfloat16 *__restrict params,
                    bfloat16 *__restrict x_out) {
    constexpr int dim = LN_DIM;
    constexpr float eps = 1e-5f;
    constexpr int chunks = dim / VEC;

    // params layout: [gamma(dim) | beta(dim)]
    bfloat16 *gamma = params;
    bfloat16 *beta = params + dim;

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    event0();

    // === Pass 1: Accumulate sum and sum-of-squares ===
    ::aie::vector<bfloat16, VEC> sum_acc = ::aie::zeros<bfloat16, VEC>();
    ::aie::vector<float, VEC> sum_sq_acc = ::aie::zeros<float, VEC>();

    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> v = ::aie::load_v<VEC>(x_in + i * VEC);
        sum_acc = ::aie::add(sum_acc, v);
        ::aie::vector<float, VEC> sq = ::aie::mul(v, v);
        sum_sq_acc = ::aie::add(sum_sq_acc, sq);
    }

    float sum_vals = ::aie::reduce_add(sum_acc);
    float sum_sq_vals = ::aie::reduce_add(sum_sq_acc);

    float mean = sum_vals / (float)dim;
    float variance = (sum_sq_vals / (float)dim) - (mean * mean);
    float inv_std = ::aie::invsqrt(variance + eps);

    // === Pass 2: Normalize, scale, shift ===
    ::aie::vector<bfloat16, VEC> mean_v = ::aie::broadcast<bfloat16, VEC>((bfloat16)mean);
    ::aie::vector<bfloat16, VEC> inv_std_v = ::aie::broadcast<bfloat16, VEC>((bfloat16)inv_std);

    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> vx = ::aie::load_v<VEC>(x_in + i * VEC);
        ::aie::vector<bfloat16, VEC> vg = ::aie::load_v<VEC>(gamma + i * VEC);
        ::aie::vector<bfloat16, VEC> vb = ::aie::load_v<VEC>(beta + i * VEC);

        ::aie::vector<bfloat16, VEC> centered = ::aie::sub(vx, mean_v);
        ::aie::vector<bfloat16, VEC> normed = ::aie::mul(centered, inv_std_v);
        ::aie::vector<bfloat16, VEC> scaled = ::aie::mul(normed, vg);
        ::aie::vector<bfloat16, VEC> result = ::aie::add(scaled, vb);
        ::aie::store_v(x_out + i * VEC, result);
    }

    event1();
}

} // extern "C"
