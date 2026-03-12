// Benchmark host for IRON GEMM at Kokoro TTS dimensions
// Measures latency, throughput, and correctness vs CPU reference.
//
// Usage: benchmark_iron_gemm.exe --xclbin <path> --instr <path>
//        --kernel MLIR_AIE -M 128 -K 768 -N 768
//        [--warmup 3] [--iters 10] [--no-verify]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

static float bf16_to_float(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

struct BenchmarkResult {
    double mean_ms;
    double min_ms;
    double max_ms;
    double median_ms;
    double stddev_ms;
    double gflops;
    double bw_gbps;
    int errors;
    float max_abs_err;
    float max_rel_err;
};

int main(int argc, const char *argv[]) {
    cxxopts::Options options("benchmark_iron_gemm",
        "Benchmark IRON GEMM on NPU at Kokoro TTS dimensions");
    test_utils::add_default_options(options);
    options.add_options()
        ("M", "M dimension", cxxopts::value<int>()->default_value("128"))
        ("K", "K dimension", cxxopts::value<int>()->default_value("768"))
        ("N", "N dimension", cxxopts::value<int>()->default_value("768"))
        ("warmup", "Warmup iterations", cxxopts::value<int>()->default_value("3"))
        ("iters", "Timed iterations", cxxopts::value<int>()->default_value("10"))
        ("no-verify", "Skip CPU verification")
        ;

    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();

    const int M = vm["M"].as<int>();
    const int K = vm["K"].as<int>();
    const int N = vm["N"].as<int>();
    const int warmup_iters = vm["warmup"].as<int>();
    const int timed_iters = vm["iters"].as<int>();
    const bool verify = !vm.count("no-verify");

    std::cout << "=== IRON GEMM Benchmark ===" << std::endl;
    std::cout << "Dimensions: " << M << " x " << K << " x " << N << std::endl;
    std::cout << "Warmup: " << warmup_iters << ", Timed: " << timed_iters << std::endl;

    // Load instructions
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());

    // Init XRT
    xrt::device device;
    xrt::kernel kernel;
    test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                     vm["xclbin"].as<std::string>(),
                                     vm["kernel"].as<std::string>());

    // Buffer sizes (minimum 4096 for XRT alignment)
    const size_t A_sz = std::max((size_t)(M * K * sizeof(uint16_t)), (size_t)4096);
    const size_t B_sz = std::max((size_t)(K * N * sizeof(uint16_t)), (size_t)4096);
    const size_t C_sz = std::max((size_t)(M * N * sizeof(uint16_t)), (size_t)4096);

    // Allocate buffers
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_A = xrt::bo(device, A_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_B = xrt::bo(device, B_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_C = xrt::bo(device, C_sz, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Upload instructions
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    // Initialize A with small random values
    uint16_t *bufA = bo_A.map<uint16_t *>();
    srand(42);
    for (int i = 0; i < M * K; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        bufA[i] = float_to_bf16(val);
    }

    // Initialize B
    uint16_t *bufB = bo_B.map<uint16_t *>();
    for (int i = 0; i < K * N; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        bufB[i] = float_to_bf16(val);
    }

    uint16_t *bufC = bo_C.map<uint16_t *>();

    // Sync inputs to device
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ── Warmup ──────────────────────────────────────────────────────
    std::cout << "\nWarming up (" << warmup_iters << " iters)..." << std::flush;
    for (int i = 0; i < warmup_iters; i++) {
        memset(bufC, 0, C_sz);
        bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = kernel(3, bo_instr, instr_v.size(), bo_A, bo_B, bo_C);
        run.wait();
    }
    std::cout << " done" << std::endl;

    // ── Timed iterations ────────────────────────────────────────────
    std::vector<double> latencies;
    latencies.reserve(timed_iters);

    std::cout << "Running " << timed_iters << " timed iterations..." << std::endl;
    for (int i = 0; i < timed_iters; i++) {
        memset(bufC, 0, C_sz);
        bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto run = kernel(3, bo_instr, instr_v.size(), bo_A, bo_B, bo_C);
        run.wait();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);

        if (verbosity >= 1) {
            std::cout << "  iter " << i << ": " << std::fixed
                      << std::setprecision(2) << ms << " ms" << std::endl;
        }
    }

    // ── Statistics ──────────────────────────────────────────────────
    BenchmarkResult result = {};
    std::sort(latencies.begin(), latencies.end());
    result.min_ms = latencies.front();
    result.max_ms = latencies.back();
    result.median_ms = latencies[timed_iters / 2];

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    result.mean_ms = sum / timed_iters;

    double sq_sum = 0;
    for (auto &l : latencies)
        sq_sum += (l - result.mean_ms) * (l - result.mean_ms);
    result.stddev_ms = std::sqrt(sq_sum / timed_iters);

    // FLOPS: 2*M*K*N (multiply-accumulate)
    double flops = 2.0 * M * K * N;
    result.gflops = flops / (result.median_ms * 1e-3) / 1e9;

    // Bandwidth: A + B + C in bytes
    double total_bytes = (double)(M * K + K * N + M * N) * 2.0; // bf16 = 2 bytes
    result.bw_gbps = total_bytes / (result.median_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Results: GEMM " << M << "x" << K << "x" << N << " ===" << std::endl;
    std::cout << "  Latency (median): " << result.median_ms << " ms" << std::endl;
    std::cout << "  Latency (mean):   " << result.mean_ms << " ms" << std::endl;
    std::cout << "  Latency (min):    " << result.min_ms << " ms" << std::endl;
    std::cout << "  Latency (max):    " << result.max_ms << " ms" << std::endl;
    std::cout << "  Latency (stddev): " << result.stddev_ms << " ms" << std::endl;
    std::cout << "  Throughput:       " << result.gflops << " GFLOPS" << std::endl;
    std::cout << "  Eff. Bandwidth:   " << result.bw_gbps << " GB/s" << std::endl;

    // ── Verification ────────────────────────────────────────────────
    if (verify) {
        std::cout << "\nVerifying against CPU reference..." << std::flush;
        bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        int errors = 0;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float ref = 0.0f;
                for (int kk = 0; kk < K; kk++) {
                    float a = bf16_to_float(bufA[m * K + kk]);
                    float b = bf16_to_float(bufB[kk * N + n]);
                    ref += a * b;
                }
                float npu = bf16_to_float(bufC[m * N + n]);
                float err = std::abs(npu - ref);
                float rel = (std::abs(ref) > 1e-6f) ? err / std::abs(ref) : err;
                max_abs_err = std::max(max_abs_err, err);
                max_rel_err = std::max(max_rel_err, rel);

                // Generous tolerance for BFP16 with FP32 accum
                if (rel > 0.15f && err > 5.0f) {
                    if (errors < 5) {
                        std::cout << "\n  MISMATCH [" << m << "," << n
                                  << "] npu=" << npu << " ref=" << ref
                                  << " err=" << err;
                    }
                    errors++;
                }
            }
        }

        result.errors = errors;
        result.max_abs_err = max_abs_err;
        result.max_rel_err = max_rel_err;

        std::cout << " done" << std::endl;
        std::cout << "  Max absolute error: " << max_abs_err << std::endl;
        std::cout << "  Max relative error: " << max_rel_err << std::endl;
        if (errors == 0) {
            std::cout << "  PASS" << std::endl;
        } else {
            std::cout << "  " << errors << " mismatches out of " << M * N
                      << " FAIL" << std::endl;
        }
    }

    // ── Summary line (easy to grep) ─────────────────────────────────
    std::cout << "\nSUMMARY: " << M << "x" << K << "x" << N
              << " median=" << result.median_ms << "ms"
              << " min=" << result.min_ms << "ms"
              << " gflops=" << result.gflops
              << " bw=" << result.bw_gbps << "GB/s"
              << " err=" << result.max_abs_err
              << (result.errors == 0 ? " PASS" : " FAIL")
              << std::endl;

    return result.errors > 0 ? 1 : 0;
}
