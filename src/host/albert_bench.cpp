// Phase D: Full ALBERT Pass Benchmark (Hybrid CPU/NPU)
//
// Correct ALBERT POST-norm pass order:
//   GEMM(Q,K,V) → MHA → GEMM(AttnDense) → Add+LN → GEMM(FFNup) → GELU → GEMM(FFNdown) → Add+LN
//
// NPU: IRON GEMM for all matmuls, IRON MHA for attention
// CPU: LayerNorm, GELU, Add, bias application
//
// Optimizations:
//   - Pre-allocated XRT buffers (BOs allocated once, reused across all 12 passes)
//   - Weights uploaded once at init (ALBERT shared weights never change)
//   - Fast GELU with polynomial tanh approximation
//   - Supports real ONNX weights via --weights-dir (from extract_weights.py)
//
// Build: see build_albert.bat
// Run:   albert_bench.exe --kernel MLIR_AIE [--weights-dir <path>] [--iron-mha] ...

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include <immintrin.h>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"

// ALBERT dimensions
constexpr int SEQ   = 128;
constexpr int DIM   = 768;
constexpr int FFN   = 2048;
constexpr int HEADS = 12;
constexpr int HD    = DIM / HEADS;  // 64

// BF16 conversion
static uint16_t f2bf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t rb = ((bits >> 16) & 1) + 0x7FFF;
    return static_cast<uint16_t>((bits + rb) >> 16);
}
static float bf2f(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

// ---- File loading ----

static std::vector<uint16_t> load_bf16_bin(const std::string &path, int expected_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "ERROR: cannot open " << path << "\n"; exit(1); }
    std::vector<uint16_t> v(expected_elems);
    f.read(reinterpret_cast<char *>(v.data()), expected_elems * 2);
    if (!f) { std::cerr << "ERROR: short read " << path << "\n"; exit(1); }
    return v;
}

static std::vector<float> load_f32_bin(const std::string &path, int expected_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "ERROR: cannot open " << path << "\n"; exit(1); }
    std::vector<float> v(expected_elems);
    f.read(reinterpret_cast<char *>(v.data()), expected_elems * 4);
    if (!f) { std::cerr << "ERROR: short read " << path << "\n"; exit(1); }
    return v;
}

// ---- CPU element-wise ops ----

static void cpu_layernorm(const uint16_t *in, const float *gamma, const float *beta,
                          uint16_t *out, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        float mean = 0.0f;
        for (int j = 0; j < dim; j++) mean += bf2f(in[r * dim + j]);
        mean /= dim;
        float var = 0.0f;
        for (int j = 0; j < dim; j++) {
            float d = bf2f(in[r * dim + j]) - mean;
            var += d * d;
        }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < dim; j++) {
            float normed = (bf2f(in[r * dim + j]) - mean) * inv_std;
            out[r * dim + j] = f2bf(gamma[j] * normed + beta[j]);
        }
    }
}

static inline float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static void cpu_gelu(const uint16_t *in, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        float x = bf2f(in[i]);
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i] = f2bf(0.5f * x * (1.0f + fast_tanh(inner)));
    }
}

static void cpu_add(const uint16_t *a, const uint16_t *b, uint16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}

static void cpu_add_bias(uint16_t *data, const float *bias, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            data[r * cols + c] = f2bf(bf2f(data[r * cols + c]) + bias[c]);
}

// ---- CPU multi-head attention (FP32, AVX2 optimized) ----
// Dimensions: seq=128, hd=64, heads=12
// Working set per head: ~160KB (fits in L2 cache)

static inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void cpu_mha(const uint16_t *Q, const uint16_t *K, const uint16_t *V,
                    uint16_t *out, int seq, int heads, int hd) {
    const int dim = heads * hd;
    const float scale = 1.0f / std::sqrt((float)hd);
    const int hd_vecs = hd / 8;  // 64/8 = 8 AVX2 vectors

    // Per-head working buffers (aligned for AVX2)
    alignas(32) float Qh[128 * 64];
    alignas(32) float Kh[128 * 64];
    alignas(32) float Vh[128 * 64];
    alignas(32) float scores[128 * 128];

    for (int h = 0; h < heads; h++) {
        // 1. Extract and convert bf16 -> f32
        for (int s = 0; s < seq; s++) {
            const int base = s * dim + h * hd;
            for (int d = 0; d < hd; d++) {
                Qh[s * hd + d] = bf2f(Q[base + d]);
                Kh[s * hd + d] = bf2f(K[base + d]);
                Vh[s * hd + d] = bf2f(V[base + d]);
            }
        }

        // 2. Scores = Q @ K^T * scale  [seq, seq]
        // AVX2 FMA dot product: hd=64 = 8 vectors of 8 floats
        for (int i = 0; i < seq; i++) {
            const float *qi = &Qh[i * hd];
            for (int j = 0; j < seq; j++) {
                const float *kj = &Kh[j * hd];
                __m256 acc = _mm256_setzero_ps();
                for (int d = 0; d < hd; d += 8) {
                    __m256 q = _mm256_load_ps(qi + d);
                    __m256 k = _mm256_load_ps(kj + d);
                    acc = _mm256_fmadd_ps(q, k, acc);
                }
                scores[i * seq + j] = hsum256(acc) * scale;
            }
        }

        // 3. Softmax per row (scalar — exp dominates)
        for (int i = 0; i < seq; i++) {
            float *row = &scores[i * seq];
            float max_val = row[0];
            for (int j = 1; j < seq; j++)
                max_val = std::max(max_val, row[j]);
            float sum = 0;
            for (int j = 0; j < seq; j++) {
                row[j] = std::exp(row[j] - max_val);
                sum += row[j];
            }
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < seq; j++)
                row[j] *= inv_sum;
        }

        // 4. Output = scores @ V  [seq, hd]
        // Broadcast score, multiply V row, accumulate across all j
        for (int i = 0; i < seq; i++) {
            __m256 acc[8];  // hd/8 = 8 accumulators
            for (int a = 0; a < hd_vecs; a++)
                acc[a] = _mm256_setzero_ps();

            for (int j = 0; j < seq; j++) {
                __m256 s = _mm256_set1_ps(scores[i * seq + j]);
                const float *vj = &Vh[j * hd];
                for (int d = 0; d < hd; d += 8)
                    acc[d / 8] = _mm256_fmadd_ps(s, _mm256_load_ps(vj + d), acc[d / 8]);
            }

            // Store as bf16
            alignas(32) float tmp[64];
            for (int d = 0; d < hd; d += 8)
                _mm256_store_ps(tmp + d, acc[d / 8]);
            for (int d = 0; d < hd; d++)
                out[i * dim + h * hd + d] = f2bf(tmp[d]);
        }
    }
}

// ---- NPU op wrapper ----

struct NpuOp {
    xrt::hw_context ctx;
    xrt::kernel kernel;
    std::vector<uint32_t> instr;
    xrt::bo bo_instr;

    NpuOp() = default;

    void load(xrt::device &dev, const std::string &xclbin_path,
              const std::string &instr_path, const std::string &kernel_name) {
        instr = test_utils::load_instr_binary(instr_path);
        auto xclbin = xrt::xclbin(xclbin_path);
        dev.register_xclbin(xclbin);
        ctx = xrt::hw_context(dev, xclbin.get_uuid());
        kernel = xrt::kernel(ctx, kernel_name);
        bo_instr = xrt::bo(dev, instr.size() * sizeof(uint32_t),
                           XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        void *p = bo_instr.map<void *>();
        memcpy(p, instr.data(), instr.size() * sizeof(uint32_t));
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    xrt::bo alloc(xrt::device &dev, size_t bytes, int arg_idx) {
        return xrt::bo(dev, std::max(bytes, (size_t)4096),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(arg_idx));
    }

    void run3(xrt::bo &a, xrt::bo &b, xrt::bo &c) {
        auto r = kernel(3, bo_instr, instr.size(), a, b, c);
        r.wait();
    }

    void run4(xrt::bo &a, xrt::bo &b, xrt::bo &c, xrt::bo &d) {
        auto r = kernel(3, bo_instr, instr.size(), a, b, c, d);
        r.wait();
    }

    // Legacy: per-call MHA for Phase 3 fallback
    void run_mha_head(xrt::device &dev, const uint16_t *qkt, const uint16_t *v,
                      uint16_t *out, int seq, int hd) {
        int qkt_elems = seq * hd + hd * seq;
        size_t qkt_sz = std::max((size_t)(qkt_elems * 2), (size_t)4096);
        size_t v_sz   = std::max((size_t)(seq * hd * 2), (size_t)4096);
        size_t o_sz   = std::max((size_t)(seq * hd * 2), (size_t)4096);
        auto bo_qkt = xrt::bo(dev, qkt_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_v   = xrt::bo(dev, v_sz,   XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_out = xrt::bo(dev, o_sz,   XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        memcpy(bo_qkt.map<void *>(), qkt, qkt_elems * 2);
        memcpy(bo_v.map<void *>(), v, seq * hd * 2);
        memset(bo_out.map<void *>(), 0, o_sz);
        bo_qkt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = kernel(3, bo_instr, instr.size(), bo_qkt, bo_v, bo_out);
        run.wait();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(out, bo_out.map<void *>(), seq * hd * 2);
    }
};

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

int main(int argc, const char *argv[]) {
    cxxopts::Options options("albert_bench");
    options.add_options()
        ("kernel", "Kernel name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("gemm768-xclbin", "GEMM 768x768 xclbin", cxxopts::value<std::string>())
        ("gemm768-instr", "GEMM 768x768 instr.bin", cxxopts::value<std::string>())
        ("gemm2048up-xclbin", "GEMM 768x2048 xclbin", cxxopts::value<std::string>())
        ("gemm2048up-instr", "GEMM 768x2048 instr.bin", cxxopts::value<std::string>())
        ("gemm2048down-xclbin", "GEMM 2048x768 xclbin", cxxopts::value<std::string>())
        ("gemm2048down-instr", "GEMM 2048x768 instr.bin", cxxopts::value<std::string>())
        ("mha-xclbin", "MHA xclbin", cxxopts::value<std::string>())
        ("mha-instr", "MHA instr.bin", cxxopts::value<std::string>())
        ("iron-mha", "Use IRON MHA (4-arg kernel)", cxxopts::value<bool>()->default_value("false"))
        ("cpu-mha", "Use CPU FP32 MHA (for precision isolation)", cxxopts::value<bool>()->default_value("false"))
        ("fused-qkv", "Use fused QKV GEMM (768->2304)", cxxopts::value<bool>()->default_value("false"))
        ("gemm2304-xclbin", "Fused QKV GEMM 768x2304 xclbin", cxxopts::value<std::string>())
        ("gemm2304-instr", "Fused QKV GEMM 768x2304 instr.bin", cxxopts::value<std::string>())
        ("npu-gelu", "Run GELU on NPU instead of CPU", cxxopts::value<bool>()->default_value("false"))
        ("gelu-xclbin", "GELU xclbin", cxxopts::value<std::string>())
        ("gelu-instr", "GELU instr.bin", cxxopts::value<std::string>())
        ("weights-dir", "Load real weights from directory (from extract_weights.py)", cxxopts::value<std::string>())
        ("real-input", "Load real ALBERT input (bf16 .bin) instead of random", cxxopts::value<std::string>())
        ("dump-dir", "Dump per-pass outputs for precision validation", cxxopts::value<std::string>())
        ("passes", "Number of ALBERT passes", cxxopts::value<int>()->default_value("12"))
        ("verbosity,v", "Verbosity", cxxopts::value<int>()->default_value("0"))
        ("help,h", "Print help");

    auto vm = options.parse(argc, argv);
    if (vm.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int n_passes = vm["passes"].as<int>();
    int verbosity = vm["verbosity"].as<int>();
    std::string kernel_name = vm["kernel"].as<std::string>();
    bool use_iron_mha = vm["iron-mha"].as<bool>();
    bool use_cpu_mha = vm["cpu-mha"].as<bool>();
    bool use_fused_qkv = vm["fused-qkv"].as<bool>();
    bool use_npu_gelu = vm["npu-gelu"].as<bool>();
    bool use_real_weights = vm.count("weights-dir") > 0;
    std::string weights_dir = use_real_weights ? vm["weights-dir"].as<std::string>() : "";
    bool dump = vm.count("dump-dir") > 0;
    std::string dump_dir = dump ? vm["dump-dir"].as<std::string>() : "";

    auto save_bf16 = [](const std::string &path, const uint16_t *data, int n) {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(data), n * 2);
    };

    std::cout << "=== ALBERT Benchmark (Hybrid CPU/NPU) ===\n"
              << "  SEQ=" << SEQ << " DIM=" << DIM << " FFN=" << FFN
              << " HEADS=" << HEADS << " HD=" << HD << "\n"
              << "  Passes: " << n_passes
              << "  MHA: " << (use_cpu_mha ? "CPU-FP32" : (use_iron_mha ? "IRON" : "Phase3"))
              << "  QKV: " << (use_fused_qkv ? "fused" : "separate")
              << "  GELU: " << (use_npu_gelu ? "NPU" : "CPU")
              << "  Weights: " << (use_real_weights ? "real" : "random")
              << "\n\n";

    // Open device
    xrt::device device(0);
    std::cout << "Device opened.\n";

    // Load NPU ops
    NpuOp gemm768, gemm2304, gemm2048up, gemm2048down, mha, gelu_op;

    std::cout << "Loading GEMM 768x768..."; std::cout.flush();
    gemm768.load(device, vm["gemm768-xclbin"].as<std::string>(),
                 vm["gemm768-instr"].as<std::string>(), kernel_name);
    std::cout << " OK\n";

    std::cout << "Loading GEMM 768x2048..."; std::cout.flush();
    gemm2048up.load(device, vm["gemm2048up-xclbin"].as<std::string>(),
                    vm["gemm2048up-instr"].as<std::string>(), kernel_name);
    std::cout << " OK\n";

    std::cout << "Loading GEMM 2048x768..."; std::cout.flush();
    gemm2048down.load(device, vm["gemm2048down-xclbin"].as<std::string>(),
                      vm["gemm2048down-instr"].as<std::string>(), kernel_name);
    std::cout << " OK\n";

    std::cout << "Loading MHA..."; std::cout.flush();
    mha.load(device, vm["mha-xclbin"].as<std::string>(),
             vm["mha-instr"].as<std::string>(), kernel_name);
    std::cout << " OK\n";

    if (use_fused_qkv) {
        std::cout << "Loading GEMM 768x2304 (fused QKV)..."; std::cout.flush();
        gemm2304.load(device, vm["gemm2304-xclbin"].as<std::string>(),
                      vm["gemm2304-instr"].as<std::string>(), kernel_name);
        std::cout << " OK\n";
    }

    if (use_npu_gelu) {
        std::cout << "Loading GELU..."; std::cout.flush();
        gelu_op.load(device, vm["gelu-xclbin"].as<std::string>(),
                     vm["gelu-instr"].as<std::string>(), kernel_name);
        std::cout << " OK\n";
    }

    // ---- Load or generate weights ----
    std::vector<uint16_t> W_Q, W_K, W_V, W_attn_dense, W_ffn_up, W_ffn_down;
    std::vector<float> B_Q, B_K, B_V, B_attn_dense, B_ffn_up, B_ffn_down;
    std::vector<float> LN_attn_gamma, LN_attn_beta, LN_ffn_gamma, LN_ffn_beta;

    if (use_real_weights) {
        std::cout << "\nLoading weights from " << weights_dir << "..."; std::cout.flush();
        auto p = [&](const std::string &name) { return weights_dir + "/" + name; };

        W_Q          = load_bf16_bin(p("W_Q.bin"), DIM * DIM);
        W_K          = load_bf16_bin(p("W_K.bin"), DIM * DIM);
        W_V          = load_bf16_bin(p("W_V.bin"), DIM * DIM);
        W_attn_dense = load_bf16_bin(p("W_attn_dense.bin"), DIM * DIM);
        W_ffn_up     = load_bf16_bin(p("W_ffn_up.bin"), DIM * FFN);
        W_ffn_down   = load_bf16_bin(p("W_ffn_down.bin"), FFN * DIM);

        B_Q          = load_f32_bin(p("B_Q.bin"), DIM);
        B_K          = load_f32_bin(p("B_K.bin"), DIM);
        B_V          = load_f32_bin(p("B_V.bin"), DIM);
        B_attn_dense = load_f32_bin(p("B_attn_dense.bin"), DIM);
        B_ffn_up     = load_f32_bin(p("B_ffn_up.bin"), FFN);
        B_ffn_down   = load_f32_bin(p("B_ffn_down.bin"), DIM);

        LN_attn_gamma = load_f32_bin(p("LN_attn_gamma.bin"), DIM);
        LN_attn_beta  = load_f32_bin(p("LN_attn_beta.bin"), DIM);
        LN_ffn_gamma  = load_f32_bin(p("LN_ffn_gamma.bin"), DIM);
        LN_ffn_beta   = load_f32_bin(p("LN_ffn_beta.bin"), DIM);
        std::cout << " OK\n";
    } else {
        srand(42);
        auto rand_bf16 = [](int n) {
            std::vector<uint16_t> v(n);
            for (auto &x : v) x = f2bf(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
            return v;
        };

        W_Q          = rand_bf16(DIM * DIM);
        W_K          = rand_bf16(DIM * DIM);
        W_V          = rand_bf16(DIM * DIM);
        W_attn_dense = rand_bf16(DIM * DIM);
        W_ffn_up     = rand_bf16(DIM * FFN);
        W_ffn_down   = rand_bf16(FFN * DIM);

        // Zero biases for random-weight mode (no bias effect on timing)
        B_Q.assign(DIM, 0.0f);
        B_K.assign(DIM, 0.0f);
        B_V.assign(DIM, 0.0f);
        B_attn_dense.assign(DIM, 0.0f);
        B_ffn_up.assign(FFN, 0.0f);
        B_ffn_down.assign(DIM, 0.0f);

        // Random LayerNorm params
        LN_attn_gamma.resize(DIM); LN_attn_beta.resize(DIM);
        LN_ffn_gamma.resize(DIM);  LN_ffn_beta.resize(DIM);
        for (int i = 0; i < DIM; i++) {
            LN_attn_gamma[i] = 0.9f + 0.2f * (float)rand() / RAND_MAX;
            LN_attn_beta[i]  = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            LN_ffn_gamma[i]  = 0.9f + 0.2f * (float)rand() / RAND_MAX;
            LN_ffn_beta[i]   = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    // ---- Pre-allocate XRT buffers ----
    std::cout << "\nPre-allocating buffers..."; std::cout.flush();

    // GEMM 768: input + 4 weights (Q/K/V/AttnDense) + output
    auto g768_a  = gemm768.alloc(device, SEQ * DIM * 2, 3);
    auto g768_wq = gemm768.alloc(device, DIM * DIM * 2, 4);
    auto g768_wk = gemm768.alloc(device, DIM * DIM * 2, 4);
    auto g768_wv = gemm768.alloc(device, DIM * DIM * 2, 4);
    auto g768_wd = gemm768.alloc(device, DIM * DIM * 2, 4);  // attn dense
    auto g768_c  = gemm768.alloc(device, SEQ * DIM * 2, 5);

    // GEMM 2048up/down: input + weight + output
    auto gup_a = gemm2048up.alloc(device, SEQ * DIM * 2, 3);
    auto gup_w = gemm2048up.alloc(device, DIM * FFN * 2, 4);
    auto gup_c = gemm2048up.alloc(device, SEQ * FFN * 2, 5);
    auto gdn_a = gemm2048down.alloc(device, SEQ * FFN * 2, 3);
    auto gdn_w = gemm2048down.alloc(device, FFN * DIM * 2, 4);
    auto gdn_c = gemm2048down.alloc(device, SEQ * DIM * 2, 5);

    // Cache mapped pointers
    auto *p_g768_a = (uint16_t *)g768_a.map<void *>();
    auto *p_g768_c = (uint16_t *)g768_c.map<void *>();
    auto *p_gup_a  = (uint16_t *)gup_a.map<void *>();
    auto *p_gup_c  = (uint16_t *)gup_c.map<void *>();
    auto *p_gdn_a  = (uint16_t *)gdn_a.map<void *>();
    auto *p_gdn_c  = (uint16_t *)gdn_c.map<void *>();

    // Upload weights ONCE
    memcpy(g768_wq.map<void *>(), W_Q.data(), DIM * DIM * 2);
    g768_wq.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(g768_wk.map<void *>(), W_K.data(), DIM * DIM * 2);
    g768_wk.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(g768_wv.map<void *>(), W_V.data(), DIM * DIM * 2);
    g768_wv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(g768_wd.map<void *>(), W_attn_dense.data(), DIM * DIM * 2);
    g768_wd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(gup_w.map<void *>(), W_ffn_up.data(), DIM * FFN * 2);
    gup_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(gdn_w.map<void *>(), W_ffn_down.data(), FFN * DIM * 2);
    gdn_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Fused QKV GEMM buffers (768 -> 2304)
    constexpr int DIM_QKV = DIM * 3;  // 2304
    xrt::bo g2304_a, g2304_w, g2304_c;
    uint16_t *p_g2304_a = nullptr, *p_g2304_c = nullptr;
    if (use_fused_qkv) {
        g2304_a = gemm2304.alloc(device, SEQ * DIM * 2, 3);
        g2304_w = gemm2304.alloc(device, DIM * DIM_QKV * 2, 4);
        g2304_c = gemm2304.alloc(device, SEQ * DIM_QKV * 2, 5);
        p_g2304_a = (uint16_t *)g2304_a.map<void *>();
        p_g2304_c = (uint16_t *)g2304_c.map<void *>();

        // Concatenate W_Q | W_K | W_V into W_QKV [768, 2304]
        auto *p_w = (uint16_t *)g2304_w.map<void *>();
        for (int r = 0; r < DIM; r++) {
            memcpy(&p_w[r * DIM_QKV + 0],    &W_Q[r * DIM], DIM * 2);
            memcpy(&p_w[r * DIM_QKV + DIM],  &W_K[r * DIM], DIM * 2);
            memcpy(&p_w[r * DIM_QKV + 2*DIM], &W_V[r * DIM], DIM * 2);
        }
        g2304_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::cout << "  Fused QKV weight [768,2304] uploaded.\n";
    }

    // NPU GELU buffers
    xrt::bo gelu_in, gelu_unused, gelu_out;
    uint16_t *p_gelu_in = nullptr, *p_gelu_out = nullptr;
    if (use_npu_gelu) {
        gelu_in     = gelu_op.alloc(device, SEQ * FFN * 2, 3);
        gelu_unused = gelu_op.alloc(device, 4096, 4);  // design.py has 3-arg sequence
        gelu_out    = gelu_op.alloc(device, SEQ * FFN * 2, 5);
        p_gelu_in   = (uint16_t *)gelu_in.map<void *>();
        p_gelu_out  = (uint16_t *)gelu_out.map<void *>();
    }

    // IRON MHA buffers
    xrt::bo mha_q, mha_k, mha_v, mha_o;
    uint16_t *p_mha_q = nullptr, *p_mha_k = nullptr;
    uint16_t *p_mha_v = nullptr, *p_mha_o = nullptr;
    if (use_iron_mha) {
        mha_q = mha.alloc(device, HEADS * SEQ * HD * 2, 3);
        mha_k = mha.alloc(device, HEADS * SEQ * HD * 2, 4);
        mha_v = mha.alloc(device, HEADS * SEQ * HD * 2, 5);
        mha_o = mha.alloc(device, HEADS * SEQ * HD * 2, 6);
        p_mha_q = (uint16_t *)mha_q.map<void *>();
        p_mha_k = (uint16_t *)mha_k.map<void *>();
        p_mha_v = (uint16_t *)mha_v.map<void *>();
        p_mha_o = (uint16_t *)mha_o.map<void *>();
    }

    std::cout << " OK (6 weight matrices uploaded once)\n\n";

    // Working buffers
    std::vector<uint16_t> X(SEQ * DIM);          // Current hidden state
    std::vector<uint16_t> Q(SEQ * DIM);
    std::vector<uint16_t> K(SEQ * DIM);
    std::vector<uint16_t> V(SEQ * DIM);
    std::vector<uint16_t> attn_out(SEQ * DIM);    // MHA output
    std::vector<uint16_t> attn_proj(SEQ * DIM);   // After attn dense
    std::vector<uint16_t> X_attn(SEQ * DIM);      // After attn residual + LN
    std::vector<uint16_t> ffn_hidden(SEQ * FFN);
    std::vector<uint16_t> ffn_act(SEQ * FFN);
    std::vector<uint16_t> ffn_out(SEQ * DIM);
    // Phase 3 MHA per-head buffers
    std::vector<uint16_t> qkt_packed(SEQ * HD + HD * SEQ);
    std::vector<uint16_t> head_v(SEQ * HD);
    std::vector<uint16_t> head_out(SEQ * HD);

    // Initialize X
    if (vm.count("real-input")) {
        std::string input_path = vm["real-input"].as<std::string>();
        std::cout << "Loading real ALBERT input from " << input_path << "...";
        std::cout.flush();
        auto loaded = load_bf16_bin(input_path, SEQ * DIM);
        memcpy(X.data(), loaded.data(), SEQ * DIM * 2);
        std::cout << " OK\n";
    } else {
        srand(42);
        for (auto &x : X) x = f2bf(((float)rand() / RAND_MAX - 0.5f) * 0.5f);
    }

    if (dump) {
        save_bf16(dump_dir + "/X_init.bin", X.data(), SEQ * DIM);
        std::cout << "Dumped X_init.bin\n";
    }

    // Timing accumulators
    double t_gemm_qkv = 0, t_mha = 0, t_attn_dense = 0, t_add_ln1 = 0;
    double t_gemm_ffn_up = 0, t_gelu = 0, t_gemm_ffn_down = 0, t_add_ln2 = 0;
    Timer timer;

    std::cout << "Running " << n_passes << " ALBERT passes...\n\n";
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int pass = 0; pass < n_passes; pass++) {
        if (verbosity >= 1)
            std::cout << "  Pass " << pass << ":\n";

        // 1. GEMM Q, K, V (NPU) — input is X directly (ALBERT post-norm: no pre-LN)
        timer.start();
        double dt;
        if (use_fused_qkv) {
            // Single fused GEMM: X[128,768] * W_QKV[768,2304] = QKV[128,2304]
            memcpy(p_g2304_a, X.data(), SEQ * DIM * 2);
            g2304_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            gemm2304.run3(g2304_a, g2304_w, g2304_c);
            g2304_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // Split output [128,2304] -> Q[128,768], K[128,768], V[128,768]
            for (int s = 0; s < SEQ; s++) {
                memcpy(&Q[s * DIM], &p_g2304_c[s * DIM_QKV + 0],     DIM * 2);
                memcpy(&K[s * DIM], &p_g2304_c[s * DIM_QKV + DIM],   DIM * 2);
                memcpy(&V[s * DIM], &p_g2304_c[s * DIM_QKV + 2*DIM], DIM * 2);
            }
            cpu_add_bias(Q.data(), B_Q.data(), SEQ, DIM);
            cpu_add_bias(K.data(), B_K.data(), SEQ, DIM);
            cpu_add_bias(V.data(), B_V.data(), SEQ, DIM);
        } else {
            // 3 separate GEMMs
            memcpy(p_g768_a, X.data(), SEQ * DIM * 2);
            g768_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            gemm768.run3(g768_a, g768_wq, g768_c);
            g768_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(Q.data(), p_g768_c, SEQ * DIM * 2);
            cpu_add_bias(Q.data(), B_Q.data(), SEQ, DIM);

            gemm768.run3(g768_a, g768_wk, g768_c);
            g768_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(K.data(), p_g768_c, SEQ * DIM * 2);
            cpu_add_bias(K.data(), B_K.data(), SEQ, DIM);

            gemm768.run3(g768_a, g768_wv, g768_c);
            g768_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(V.data(), p_g768_c, SEQ * DIM * 2);
            cpu_add_bias(V.data(), B_V.data(), SEQ, DIM);
        }
        dt = timer.stop_ms();
        t_gemm_qkv += dt;
        if (verbosity >= 1) std::cout << "    GEMM Q/K/V" << (use_fused_qkv ? " (fused)" : "") << ": " << dt << " ms\n";

        // 2. MHA (NPU or CPU)
        timer.start();
        if (use_cpu_mha) {
            cpu_mha(Q.data(), K.data(), V.data(), attn_out.data(), SEQ, HEADS, HD);
        } else if (use_iron_mha) {
            for (int h = 0; h < HEADS; h++)
                for (int s = 0; s < SEQ; s++)
                    for (int d = 0; d < HD; d++) {
                        int iron_idx = h * SEQ * HD + s * HD + d;
                        int flat_idx = s * DIM + h * HD + d;
                        p_mha_q[iron_idx] = Q[flat_idx];
                        p_mha_k[iron_idx] = K[flat_idx];
                        p_mha_v[iron_idx] = V[flat_idx];
                    }
            mha_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            mha_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            mha_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            memset(p_mha_o, 0, HEADS * SEQ * HD * 2);
            mha_o.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            mha.run4(mha_q, mha_k, mha_v, mha_o);

            mha_o.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            for (int h = 0; h < HEADS; h++)
                for (int s = 0; s < SEQ; s++)
                    for (int d = 0; d < HD; d++)
                        attn_out[s * DIM + h * HD + d] =
                            p_mha_o[h * SEQ * HD + s * HD + d];
        } else {
            for (int h = 0; h < HEADS; h++) {
                for (int i = 0; i < SEQ; i++)
                    for (int j = 0; j < HD; j++)
                        qkt_packed[i * HD + j] = Q[i * DIM + h * HD + j];
                for (int i = 0; i < HD; i++)
                    for (int j = 0; j < SEQ; j++)
                        qkt_packed[SEQ * HD + i * SEQ + j] = K[j * DIM + h * HD + i];
                for (int i = 0; i < SEQ; i++)
                    for (int j = 0; j < HD; j++)
                        head_v[i * HD + j] = V[i * DIM + h * HD + j];
                mha.run_mha_head(device, qkt_packed.data(), head_v.data(),
                                 head_out.data(), SEQ, HD);
                for (int i = 0; i < SEQ; i++)
                    for (int j = 0; j < HD; j++)
                        attn_out[i * DIM + h * HD + j] = head_out[i * HD + j];
            }
        }
        dt = timer.stop_ms();
        t_mha += dt;
        if (verbosity >= 1) {
            const char *mha_mode = use_cpu_mha ? "CPU-FP32" : (use_iron_mha ? "IRON" : "12 heads");
            std::cout << "    MHA (" << mha_mode << "): " << dt << " ms\n";
        }

        // 3. Attention output dense GEMM (NPU) + bias
        timer.start();
        memcpy(p_g768_a, attn_out.data(), SEQ * DIM * 2);
        g768_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        gemm768.run3(g768_a, g768_wd, g768_c);
        g768_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(attn_proj.data(), p_g768_c, SEQ * DIM * 2);
        cpu_add_bias(attn_proj.data(), B_attn_dense.data(), SEQ, DIM);
        dt = timer.stop_ms();
        t_attn_dense += dt;
        if (verbosity >= 1) std::cout << "    Attn Dense: " << dt << " ms\n";

        // 4. Residual add + Attention LayerNorm (CPU) — POST-norm
        timer.start();
        cpu_add(attn_proj.data(), X.data(), X_attn.data(), SEQ * DIM);
        cpu_layernorm(X_attn.data(), LN_attn_gamma.data(), LN_attn_beta.data(),
                      X_attn.data(), SEQ, DIM);
        dt = timer.stop_ms();
        t_add_ln1 += dt;
        if (verbosity >= 1) std::cout << "    Add+LN(attn): " << dt << " ms\n";

        // 5. GEMM FFN up (NPU) + bias
        timer.start();
        memcpy(p_gup_a, X_attn.data(), SEQ * DIM * 2);
        gup_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        gemm2048up.run3(gup_a, gup_w, gup_c);
        gup_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(ffn_hidden.data(), p_gup_c, SEQ * FFN * 2);
        cpu_add_bias(ffn_hidden.data(), B_ffn_up.data(), SEQ, FFN);
        dt = timer.stop_ms();
        t_gemm_ffn_up += dt;
        if (verbosity >= 1) std::cout << "    GEMM FFN up: " << dt << " ms\n";

        // 6. GELU (CPU or NPU)
        timer.start();
        if (use_npu_gelu) {
            // Copy biased FFN_up output to GELU input BO
            memcpy(p_gelu_in, ffn_hidden.data(), SEQ * FFN * 2);
            gelu_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            gelu_op.run3(gelu_in, gelu_unused, gelu_out);
            gelu_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            memcpy(ffn_act.data(), p_gelu_out, SEQ * FFN * 2);
        } else {
            cpu_gelu(ffn_hidden.data(), ffn_act.data(), SEQ * FFN);
        }
        dt = timer.stop_ms();
        t_gelu += dt;
        if (verbosity >= 1) std::cout << "    GELU" << (use_npu_gelu ? " (NPU)" : "") << ": " << dt << " ms\n";

        // 7. GEMM FFN down (NPU) + bias
        timer.start();
        memcpy(p_gdn_a, ffn_act.data(), SEQ * FFN * 2);
        gdn_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        gemm2048down.run3(gdn_a, gdn_w, gdn_c);
        gdn_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(ffn_out.data(), p_gdn_c, SEQ * DIM * 2);
        cpu_add_bias(ffn_out.data(), B_ffn_down.data(), SEQ, DIM);
        dt = timer.stop_ms();
        t_gemm_ffn_down += dt;
        if (verbosity >= 1) std::cout << "    GEMM FFN down: " << dt << " ms\n";

        // 8. Residual add + FFN LayerNorm (CPU) — POST-norm, output -> X
        timer.start();
        cpu_add(ffn_out.data(), X_attn.data(), X.data(), SEQ * DIM);
        cpu_layernorm(X.data(), LN_ffn_gamma.data(), LN_ffn_beta.data(),
                      X.data(), SEQ, DIM);
        dt = timer.stop_ms();
        t_add_ln2 += dt;
        if (verbosity >= 1) std::cout << "    Add+LN(FFN): " << dt << " ms\n";

        if (dump) {
            save_bf16(dump_dir + "/X_pass_" + std::to_string(pass) + ".bin",
                      X.data(), SEQ * DIM);
            if (verbosity >= 1)
                std::cout << "    Dumped X_pass_" << pass << ".bin\n";
        }

        if (verbosity >= 1) std::cout << "\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Report
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== ALBERT Benchmark Results (" << n_passes << " passes) ===\n\n";
    std::cout << "  Op              | Total (ms) | Per-Pass (ms) | Device\n";
    std::cout << "  ----------------+------------+---------------+-------\n";
    std::cout << "  GEMM Q/K/V " << (use_fused_qkv ? "fsd" : "sep") << " | " << std::setw(10) << t_gemm_qkv << " | " << std::setw(13) << t_gemm_qkv/n_passes << " | NPU\n";
    std::cout << "  MHA             | " << std::setw(10) << t_mha << " | " << std::setw(13) << t_mha/n_passes << " | " << (use_cpu_mha ? "CPU" : "NPU") << "\n";
    std::cout << "  Attn Dense      | " << std::setw(10) << t_attn_dense << " | " << std::setw(13) << t_attn_dense/n_passes << " | NPU\n";
    std::cout << "  Add+LN (attn)   | " << std::setw(10) << t_add_ln1 << " | " << std::setw(13) << t_add_ln1/n_passes << " | CPU\n";
    std::cout << "  GEMM FFN up     | " << std::setw(10) << t_gemm_ffn_up << " | " << std::setw(13) << t_gemm_ffn_up/n_passes << " | NPU\n";
    std::cout << "  GELU            | " << std::setw(10) << t_gelu << " | " << std::setw(13) << t_gelu/n_passes << " | " << (use_npu_gelu ? "NPU" : "CPU") << "\n";
    std::cout << "  GEMM FFN down   | " << std::setw(10) << t_gemm_ffn_down << " | " << std::setw(13) << t_gemm_ffn_down/n_passes << " | NPU\n";
    std::cout << "  Add+LN (FFN)    | " << std::setw(10) << t_add_ln2 << " | " << std::setw(13) << t_add_ln2/n_passes << " | CPU\n";
    std::cout << "  ----------------+------------+---------------+-------\n";
    std::cout << "  TOTAL           | " << std::setw(10) << total_ms << " | " << std::setw(13) << total_ms/n_passes << " |\n";

    double cpu_total = t_add_ln1 + t_add_ln2 + (use_npu_gelu ? 0 : t_gelu)
                     + (use_cpu_mha ? t_mha : 0);
    double npu_total = t_gemm_qkv + t_attn_dense + t_gemm_ffn_up + t_gemm_ffn_down
                     + (use_cpu_mha ? 0 : t_mha)
                     + (use_npu_gelu ? t_gelu : 0);
    std::cout << "\n  CPU ops: " << cpu_total << " ms (" << 100*cpu_total/total_ms << "%)\n";
    std::cout << "  NPU ops: " << npu_total << " ms (" << 100*npu_total/total_ms << "%)\n";
    std::cout << "\n  Per-pass: " << total_ms/n_passes << " ms\n";
    std::cout << "  " << n_passes << " passes: " << total_ms << " ms\n";

    // Compare against CPU baseline
    std::cout << "\n=== vs CPU 300ms Baseline ===\n";
    if (total_ms < 300.0) {
        std::cout << "  " << total_ms << "ms < 300ms: NPU WINS ("
                  << std::setprecision(2) << (300.0 / total_ms) << "x faster)\n";
    } else {
        std::cout << "  " << total_ms << "ms >= 300ms: CPU faster ("
                  << std::setprecision(2) << (total_ms / 300.0) << "x slower)\n";
    }

    return 0;
}
