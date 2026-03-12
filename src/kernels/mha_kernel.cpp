// Multi-Head Attention kernel for AIE2P (Strix Point)
//
// Uses auto-vectorized C++ loops instead of explicit aie:: intrinsics
// to avoid legalization failures on the 512-bit aie2p data path.
// See research doc §4.1-4.3 for details on why.
//
// Kokoro TTS uses 8-head attention with hidden_dim=512, head_dim=64.
// Per head: Q[128,64], K[128,64], V[128,64]
//
// Attention:
//   1. QK = Q @ K^T → [128, 128] per head
//   2. QK_scaled = QK / sqrt(64)
//   3. Attn = softmax(QK_scaled, dim=-1)
//   4. Out = Attn @ V → [128, 64] per head
//
// Memory: QK[128,128] BF16 = 32KB — fits in L1 but tight.
// Better: tile the sequence dim, compute softmax in blocks.
//
// Softmax uses LUT-based exp (same pattern as GELU kernel).

#include <stdint.h>

typedef _Float16 bfloat16_t;

extern "C" {

// Scaled dot-product: QK_tile = Q_block @ K_block^T / sqrt(d)
// Q_block: [tile_m, d] BF16 (row-major)
// K_block: [tile_n, d] BF16 (row-major — transposed during compute)
// QK_tile: [tile_m, tile_n] BF16 (output)
// scale: scalar 1/sqrt(d) as float
void qk_scaled(bfloat16_t *__restrict Q,
               bfloat16_t *__restrict K,
               bfloat16_t *__restrict QK,
               int32_t tile_m,
               int32_t tile_n,
               int32_t d,
               float scale) {
    // C = A @ B^T where B is K stored as [tile_n, d]
    // C[m, n] = sum_k(Q[m, k] * K[n, k]) * scale
    for (int m = 0; m < tile_m; m++) {
        for (int n = 0; n < tile_n; n++) {
            float acc = 0.0f;
            // Inner reduction — auto-vectorized
            #pragma clang loop vectorize(enable)
            for (int k = 0; k < d; k++) {
                acc += (float)Q[m * d + k] * (float)K[n * d + k];
            }
            QK[m * tile_n + n] = (bfloat16_t)(acc * scale);
        }
    }
}

// Softmax along the last dimension (row-wise)
// Uses 3-pass algorithm: max → exp-sum → normalize
// All in FP32 for numerical stability
void softmax_row(bfloat16_t *__restrict in,
                 bfloat16_t *__restrict out,
                 int32_t rows,
                 int32_t cols,
                 const float *__restrict exp_lut) {
    for (int r = 0; r < rows; r++) {
        bfloat16_t *row_in = in + r * cols;
        bfloat16_t *row_out = out + r * cols;

        // Pass 1: find max
        float max_val = -1e30f;
        for (int c = 0; c < cols; c++) {
            float v = (float)row_in[c];
            if (v > max_val) max_val = v;
        }

        // Pass 2: exp(x - max) and sum
        // If exp_lut is provided, use it; otherwise use polynomial
        float sum = 0.0f;
        // Temporary storage in output buffer (reused)
        for (int c = 0; c < cols; c++) {
            float val = (float)row_in[c] - max_val;
            // Clamp to prevent overflow
            if (val < -20.0f) val = -20.0f;

            // exp via Horner (degree 6) — avoids bitwise ops that fail on aie2p
            float ex = 1.0f + val * (1.0f + val * (0.5f + val *
                       (0.1667f + val * (0.0417f + val *
                       (0.00833f + val * 0.00139f)))));
            if (ex < 0.0f) ex = 0.0f;  // safety clamp

            row_out[c] = (bfloat16_t)ex;  // temporary store
            sum += ex;
        }

        // Pass 3: normalize
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        #pragma clang loop vectorize(enable)
        for (int c = 0; c < cols; c++) {
            float v = (float)row_out[c] * inv_sum;
            row_out[c] = (bfloat16_t)v;
        }
    }
}

// General GEMM: C[M,N] = A[M,K] @ B[K,N]
// All BF16 with FP32 accumulation
// This is the workhorse for both Q*K^T and Attn*V
void gemm_bf16(bfloat16_t *__restrict A,
               bfloat16_t *__restrict B,
               bfloat16_t *__restrict C,
               int32_t M, int32_t K, int32_t N,
               int32_t accumulate) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = accumulate ? (float)C[m * N + n] : 0.0f;
            #pragma clang loop vectorize(enable)
            for (int k = 0; k < K; k++) {
                acc += (float)A[m * K + k] * (float)B[k * N + n];
            }
            C[m * N + n] = (bfloat16_t)acc;
        }
    }
}

} // extern "C"
