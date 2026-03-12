// GELU activation kernel for AIE2P
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Based on IRON reference gelu.cc.
// Uses bf16 vectors of 16 with built-in aie::tanh.
// Separate input/output buffers, fixed size per invocation.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GELU_SIZE
#define GELU_SIZE 256
#endif

constexpr unsigned VEC = 16;

extern "C" {

void gelu_bf16(bfloat16 *__restrict input,
               bfloat16 *__restrict output) {
    constexpr int size = GELU_SIZE;
    constexpr int chunks = size / VEC;

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    event0();

    const bfloat16 k0_5 = 0.5f;
    const bfloat16 k1 = 1.0f;
    const bfloat16 sqrt_2_over_pi = 0.79788456f;
    const bfloat16 kBeta = 0.044715f;

    auto v05 = ::aie::broadcast<bfloat16, VEC>(k0_5);
    auto v1 = ::aie::broadcast<bfloat16, VEC>(k1);
    auto vs2opi = ::aie::broadcast<bfloat16, VEC>(sqrt_2_over_pi);
    auto vBeta = ::aie::broadcast<bfloat16, VEC>(kBeta);

    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> x = ::aie::load_v<VEC>(input + i * VEC);

        // x^3
        ::aie::vector<bfloat16, VEC> x2 = ::aie::mul(x, x);
        ::aie::vector<bfloat16, VEC> x3 = ::aie::mul(x, x2);

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        ::aie::vector<bfloat16, VEC> x3_beta = ::aie::mul(x3, vBeta);
        ::aie::vector<bfloat16, VEC> inner = ::aie::add(x, x3_beta);
        auto inner1_acc = ::aie::mul(inner, vs2opi);

        // tanh approximation (built-in) — needs f32 input
        ::aie::vector<bfloat16, VEC> tanh_out = ::aie::tanh<bfloat16>(inner1_acc.to_vector<float>());

        // 0.5 * x * (1 + tanh)
        ::aie::vector<bfloat16, VEC> one_plus_tanh = ::aie::add(tanh_out, v1);
        ::aie::vector<bfloat16, VEC> mul_v05 = ::aie::mul(v05, one_plus_tanh);
        ::aie::vector<bfloat16, VEC> result = ::aie::mul(x, mul_v05);

        ::aie::store_v(output + i * VEC, result);
    }

    event1();
}

} // extern "C"
