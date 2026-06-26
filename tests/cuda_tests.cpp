// cuda_tests.cpp — GPU backend tests. Built and linked only when SUB0_BUILD_CUDA is ON
// (see tests/CMakeLists.txt), since they require nvcc-built code and a CUDA device. They
// drive the backend across the extern "C" seam and check it against CPU references: the
// device parameter mirror must round-trip the weight blob, and the dense-linear kernel
// must match a CPU recomputation (the parity gate every GPU kernel is held to).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"   // trainable_floats()

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

extern "C" int  sub0_cuda_selftest();
extern "C" int  sub0_cuda_init();
extern "C" void sub0_cuda_shutdown();
extern "C" int  sub0_cuda_upload_params(const float* host);
extern "C" int  sub0_cuda_download_params(float* host);
extern "C" int  sub0_cuda_linear(const float* X, int T, int in, int out,
                                 const float* W, const float* bias, float* Y);
extern "C" int  sub0_cuda_forward(const int* ids, int batch, int T, float* out_logits);
extern "C" void sub0_cuda_set_tf32(int on);

TEST_CASE("CUDA backend self-test runs on the device", "[cuda]") {
    REQUIRE(sub0_cuda_selftest() == 0);
}

TEST_CASE("CUDA param mirror round-trips the weight blob", "[cuda]") {
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> up(n), down(n, 0.0f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : up) x = dist(rng);

    REQUIRE(sub0_cuda_upload_params(up.data()) == 0);
    REQUIRE(sub0_cuda_download_params(down.data()) == 0);
    for (std::size_t i = 0; i < n; ++i) REQUIRE(down[i] == up[i]);   // exact: it is just a memcpy
    sub0_cuda_shutdown();
}

TEST_CASE("CUDA dense linear matches a CPU reference", "[cuda]") {
    sub0_cuda_set_tf32(0);   // full FP32 for a tight parity gate
    const int T = 12, in = 96, out = 128;
    std::vector<float> X(static_cast<std::size_t>(T) * in);
    std::vector<float> W(static_cast<std::size_t>(in) * out);
    std::vector<float> B(static_cast<std::size_t>(out));
    std::vector<float> Yg(static_cast<std::size_t>(T) * out, 0.0f);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : X) v = nd(rng);
    for (float& v : W) v = nd(rng);
    for (float& v : B) v = nd(rng);

    REQUIRE(sub0_cuda_linear(X.data(), T, in, out, W.data(), B.data(), Yg.data()) == 0);

    // CPU reference: Y[t,o] = sum_p X[t,p]*W[p,o] + bias[o] (same p-order as the kernel).
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < out; ++o) {
            float acc = B[static_cast<std::size_t>(o)];
            for (int p = 0; p < in; ++p)
                acc += X[static_cast<std::size_t>(t) * in + p] * W[static_cast<std::size_t>(p) * out + o];
            REQUIRE(Yg[static_cast<std::size_t>(t) * out + o] == Catch::Approx(acc).margin(1e-2));
        }
}

TEST_CASE("CUDA forward matches the CPU engine logits", "[cuda]") {
    sub0::build_model();                                     // random-init params
    sub0_cuda_set_tf32(0);                                   // full FP32 for a tight parity gate
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);  // mirror them to the device

    const int T = 16;
    std::vector<int> ids(static_cast<std::size_t>(T));
    std::mt19937 rng(21);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : ids) x = tok(rng);

    sub0::graph_reset();
    sub0::Node* lg = sub0::forward(ids.data(), T);           // CPU logits [T, VOCAB]
    const std::vector<float> cpu(lg->data.begin(), lg->data.end());

    std::vector<float> gpu(static_cast<std::size_t>(T) * VOCAB, 0.0f);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, gpu.data()) == 0);

    double max_abs = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i)
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(cpu[i] - gpu[i])));
    INFO("max abs logit diff = " << max_abs);
    REQUIRE(max_abs < 1e-2);          // same op sequence; GPU uses CUDA fast-math (__expf/rsqrtf)
    sub0_cuda_shutdown();
}

TEST_CASE("CUDA batched forward matches per-window CPU logits", "[cuda]") {
    sub0::build_model();
    sub0_cuda_set_tf32(0);                                   // full FP32 for a tight parity gate
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 12;
    std::vector<int> ids(static_cast<std::size_t>(batch) * T);
    std::mt19937 rng(33);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : ids) x = tok(rng);

    // CPU reference: forward each window independently into its row block.
    std::vector<float> cpu(static_cast<std::size_t>(batch) * T * VOCAB);
    for (int b = 0; b < batch; ++b) {
        sub0::graph_reset();
        sub0::Node* lg = sub0::forward(ids.data() + static_cast<std::size_t>(b) * T, T);
        std::copy(lg->data.begin(), lg->data.end(),
                  cpu.begin() + static_cast<std::size_t>(b) * T * VOCAB);
    }

    std::vector<float> gpu(static_cast<std::size_t>(batch) * T * VOCAB, 0.0f);
    REQUIRE(sub0_cuda_forward(ids.data(), batch, T, gpu.data()) == 0);

    double max_abs = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i)
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(cpu[i] - gpu[i])));
    INFO("max abs logit diff (batched) = " << max_abs);
    REQUIRE(max_abs < 1e-2);          // batched M=B*T must match per-window CPU
    sub0_cuda_shutdown();
}

TEST_CASE("CUDA TF32 forward stays close to the CPU logits", "[cuda]") {
    sub0::build_model();
    sub0_cuda_set_tf32(1);                                   // TF32 tensor-core GEMM math
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int T = 16;
    std::vector<int> ids(static_cast<std::size_t>(T));
    std::mt19937 rng(44);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : ids) x = tok(rng);

    sub0::graph_reset();
    sub0::Node* lg = sub0::forward(ids.data(), T);
    const std::vector<float> cpu(lg->data.begin(), lg->data.end());

    std::vector<float> gpu(static_cast<std::size_t>(T) * VOCAB, 0.0f);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, gpu.data()) == 0);

    double max_abs = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i)
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(cpu[i] - gpu[i])));
    INFO("max abs logit diff (TF32) = " << max_abs);
    REQUIRE(max_abs < 5e-2);          // TF32 trades mantissa bits: looser but still bounded
    sub0_cuda_set_tf32(0);            // restore FP32 for any later tests
    sub0_cuda_shutdown();
}

