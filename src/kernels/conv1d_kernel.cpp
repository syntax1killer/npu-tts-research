// Conv1d GEMM kernel for AIE2P — row-major input/output
//
// Uses element-by-element vector loading (proven to work on aie2p)
// followed by aie::mmul<4,8,8> hardware matmul intrinsic.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 32
#endif

// Allow customizing function names for multi-kernel designs
// (avoids symbol clashes when different tiles use different DIM_* values)
#ifndef FN_MATMUL
#define FN_MATMUL matmul_bf16_bf16
#endif
#ifndef FN_ZERO
#define FN_ZERO zero_bf16
#endif

extern "C" {

void FN_ZERO(bfloat16 *__restrict c) {
    constexpr int total = DIM_M * DIM_N;
    constexpr int vec_len = 32; // 512-bit / 16-bit
    const aie::vector<bfloat16, vec_len> zeros = aie::zeros<bfloat16, vec_len>();
    for (int i = 0; i < total; i += vec_len) {
        aie::store_v(c + i, zeros);
    }
}

void FN_MATMUL(bfloat16 *__restrict pA,
                      bfloat16 *__restrict pB,
                      bfloat16 *__restrict pC) {
    constexpr int M = DIM_M;
    constexpr int K = DIM_K;
    constexpr int N = DIM_N;
    constexpr int r = 4;   // mmul M dim
    constexpr int s = 8;   // mmul K dim
    constexpr int t = 8;   // mmul N dim

    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

    event0();

    // Process output in r×t blocks
    for (int mb = 0; mb < M; mb += r) {
        for (int nb = 0; nb < N; nb += t) {
            // Load existing C accumulator element by element
            aie::vector<bfloat16, r * t> c_vec;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < t; j++)
                    c_vec[i * t + j] = pC[(mb + i) * N + nb + j];

            MMUL acc(c_vec);

            // K-dimension loop
            for (int kb = 0; kb < K; kb += s) {
                // Load A[mb:mb+r, kb:kb+s]
                aie::vector<bfloat16, r * s> va;
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < s; j++)
                        va[i * s + j] = pA[(mb + i) * K + kb + j];

                // Load B[kb:kb+s, nb:nb+t]
                aie::vector<bfloat16, s * t> vb;
                for (int i = 0; i < s; i++)
                    for (int j = 0; j < t; j++)
                        vb[i * t + j] = pB[(kb + i) * N + nb + j];

                acc.mac(va, vb);
            }

            // Store result to C
            aie::vector<bfloat16, r * t> result = acc.template to_vector<bfloat16>();
            for (int i = 0; i < r; i++)
                for (int j = 0; j < t; j++)
                    pC[(mb + i) * N + nb + j] = result[i * t + j];
        }
    }

    event1();
}

} // extern "C"
