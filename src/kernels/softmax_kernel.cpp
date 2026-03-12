// Softmax kernel for AIE2P
//
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
//
// Three-pass algorithm (based on IRON reference):
//   Pass 1: find max (scaled by log2e for numerical stability)
//   Pass 2: compute exp2(log2e * x - max), accumulate sum
//   Pass 3: divide by sum
//
// Separate input/output buffers, fixed SM_DIM elements per invocation.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SM_DIM
#define SM_DIM 128
#endif

constexpr unsigned VEC = 16;
constexpr float log2e_f = 1.4453125f;  // log2(e) in bf16-friendly approx

extern "C" {

void softmax_bf16(bfloat16 *__restrict input,
                  bfloat16 *__restrict output) {
    constexpr int dim = SM_DIM;
    constexpr int chunks = dim / VEC;

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    event0();

    ::aie::vector<bfloat16, VEC> log2e_vec = ::aie::broadcast<bfloat16, VEC>((bfloat16)log2e_f);

    // === Pass 1: Find max(log2e * x) ===
    float max_val = -1e30f;
    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> v = ::aie::load_v<VEC>(input + i * VEC);
        auto scaled = ::aie::mul(v, log2e_vec);
        float chunk_max = ::aie::reduce_max(scaled.to_vector<bfloat16>());
        if (chunk_max > max_val) max_val = chunk_max;
    }

    ::aie::vector<bfloat16, VEC> max_vec = ::aie::broadcast<bfloat16, VEC>((bfloat16)max_val);

    // === Pass 2: Compute exp2(log2e*x - max), accumulate sum ===
    ::aie::accum<accfloat, VEC> sum_acc;
    sum_acc.from_vector(::aie::zeros<float, VEC>(), 0);

    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> v = ::aie::load_v<VEC>(input + i * VEC);
        auto scaled = ::aie::mul(v, log2e_vec);
        auto shifted = ::aie::sub(scaled, max_vec);
        ::aie::vector<bfloat16, VEC> exp_val = ::aie::exp2<bfloat16>(shifted.to_vector<float>());
        sum_acc = ::aie::add(sum_acc, exp_val);
        ::aie::store_v(output + i * VEC, exp_val);
    }

    // === Pass 3: Divide by sum ===
    float total = ::aie::reduce_add(sum_acc.to_vector<float>());
    bfloat16 inv_sum = (bfloat16)::aie::inv(total);

    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> v = ::aie::load_v<VEC>(output + i * VEC);
        auto result = ::aie::mul(v, inv_sum);
        ::aie::store_v(output + i * VEC, result.to_vector<bfloat16>());
    }

    event1();
}

} // extern "C"
