// Element-wise add kernel for AIE2P
//
// out = a + b  (bf16, element-wise)
// Used for residual connections in transformer layers.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ADD_SIZE
#define ADD_SIZE 256
#endif

constexpr unsigned VEC = 16;

extern "C" {

void add_bf16(bfloat16 *__restrict a,
              bfloat16 *__restrict b,
              bfloat16 *__restrict out) {
    constexpr int size = ADD_SIZE;
    constexpr int chunks = size / VEC;

    event0();
    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> va = ::aie::load_v<VEC>(a + i * VEC);
        ::aie::vector<bfloat16, VEC> vb = ::aie::load_v<VEC>(b + i * VEC);
        ::aie::vector<bfloat16, VEC> result = ::aie::add(va, vb);
        ::aie::store_v(out + i * VEC, result);
    }
    event1();
}

} // extern "C"
