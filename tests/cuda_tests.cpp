// cuda_tests.cpp — GPU backend tests. Built and linked only when SUB0_BUILD_CUDA is ON
// (see tests/CMakeLists.txt), since they require nvcc-built code and a CUDA device. They
// drive the backend across the extern "C" seam and check it against CPU references: the
// device parameter mirror must round-trip the weight blob, and the dense-linear kernel
// must match a CPU recomputation (the parity gate every GPU kernel is held to).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"     // trainable_floats()
#include "sub0/memplan.hpp"  // the pure footprint model under test

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
extern "C" int  sub0_cuda_backward(const int* ids, const int* targets, int batch, int T,
                                   float* out_grad, double* out_loss, const int* lengths = nullptr);
extern "C" int  sub0_cuda_adam_step(float lr, long t);
extern "C" int  sub0_cuda_train_predicted_mb(int batch);
extern "C" int  sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb);

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

// Build a contiguous random token stream and the matching per-window starts so the CPU
// train_batch and the GPU backward see the identical (ids, targets) over batch*T rows.
static void make_windows(int batch, int T, unsigned seed, std::vector<int>& data,
                         std::vector<std::size_t>& starts, std::vector<int>& ids,
                         std::vector<int>& targets) {
    const int M = batch * T;
    data.resize(static_cast<std::size_t>(M) + 1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : data) x = tok(rng);
    starts.resize(batch);
    for (int b = 0; b < batch; ++b) starts[b] = static_cast<std::size_t>(b) * T;
    ids.assign(data.begin(), data.begin() + M);            // window b inputs  = data[b*T + t]
    targets.assign(data.begin() + 1, data.begin() + M + 1);// window b targets = data[b*T + t + 1]
}

TEST_CASE("CUDA backward matches the CPU reduced gradient", "[cuda]") {
    sub0::build_model();
    sub0_cuda_set_tf32(0);                                  // full FP32 for a tight parity gate
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 55, data, starts, ids, targets);

    // CPU reference: the reduced gradient the optimizer consumes after a minibatch.
    sub0::train_batch(data.data(), starts.data(), batch, T);
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    std::vector<float> gpu_grad(n, 0.0f);
    double loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &loss) == 0);

    // Relative L2 over the whole gradient: the GPU's CUDA fast-math differs from the CPU's at
    // ~1e-6 per op and those differences accumulate through the reverse pass, so compare norms.
    double num = 0.0, den = 0.0, maxabs = 0.0, dot = 0.0, gn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
        num += d * d;
        den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
        dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i];
        gn  += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
        maxabs = std::max(maxabs, std::fabs(d));
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
    INFO("grad rel-L2 = " << rel << "  cos = " << cos << "  max abs = " << maxabs << "  loss = " << loss);
    // F32 storage parity is tight; BF16 storage cannot match raw magnitudes (3 decimal digits) so we
    // assert it points the SAME direction as the CPU gradient and stays finite -- behaviour, not bits.
    REQUIRE(std::isfinite(loss));
    if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(cos > 0.7);
    else                                    REQUIRE(rel < 1e-2);
    sub0_cuda_shutdown();
}

TEST_CASE("CUDA backward matches the CPU gradient with short padded windows", "[cuda]") {
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    // A batch of mixed lengths: some windows shorter than T (a short document padded up to T). The
    // CPU trains each window at its own length; the GPU pads and masks the loss via `lengths`. The
    // reduced gradients must still agree -- the padding must contribute nothing.
    const int batch = 4, T = 8;
    const std::vector<int> lengths = {8, 5, 3, 7};
    std::vector<int> data(static_cast<std::size_t>(batch) * T + 1);
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : data) x = tok(rng);
    std::vector<std::size_t> starts(batch);
    for (int b = 0; b < batch; ++b) starts[b] = static_cast<std::size_t>(b) * T;

    sub0::train_batch(data.data(), starts.data(), batch, T, lengths.data());     // CPU reference
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    std::vector<int> ids(static_cast<std::size_t>(batch) * T), targets(static_cast<std::size_t>(batch) * T);
    for (int b = 0; b < batch; ++b)
        for (int s = 0; s < T; ++s) {
            const bool real = s < lengths[b];
            ids[static_cast<std::size_t>(b) * T + s]     = real ? data[starts[b] + s]     : 0;
            targets[static_cast<std::size_t>(b) * T + s] = real ? data[starts[b] + s + 1] : 0;
        }
    std::vector<float> gpu_grad(n, 0.0f);
    double loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &loss,
                              lengths.data()) == 0);

    double num = 0.0, den = 0.0, dot = 0.0, gn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
        num += d * d;
        den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
        dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i];
        gn  += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
    INFO("padded grad rel-L2 = " << rel << "  cos = " << cos << "  loss = " << loss);
    REQUIRE(std::isfinite(loss));
    if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(cos > 0.7);   // direction matches; padding still inert
    else                                    REQUIRE(rel < 1e-2);
    sub0_cuda_shutdown();
}

TEST_CASE("CUDA AdamW step matches the CPU optimizer update", "[cuda]") {
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 66, data, starts, ids, targets);

    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> p0(sub0::params_ptr(), sub0::params_ptr() + n);   // shared start point

    // CPU: one AdamW step from zeroed moments. Capture the parameter delta it produces.
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.0f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.0f);
    sub0::train_batch(data.data(), starts.data(), batch, T);
    sub0::AdamW opt(0.001f);
    opt.step();
    const std::vector<float> p_cpu(sub0::params_ptr(), sub0::params_ptr() + n);

    // GPU: backward (fills the device grad, zeroes moments) then one AdamW step at t=1.
    std::vector<float> tmp(n);
    double loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, tmp.data(), &loss) == 0);
    REQUIRE(sub0_cuda_adam_step(0.001f, 1) == 0);
    std::vector<float> p_gpu(n);
    REQUIRE(sub0_cuda_download_params(p_gpu.data()) == 0);

    // Compare the UPDATE deltas (p - p0): the parameters are dominated by p0, so comparing the
    // small steps is the meaningful gate (otherwise grad differences would be masked).
    double num = 0.0, den = 0.0, maxabs = 0.0, dot = 0.0, gn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dc = static_cast<double>(p_cpu[i]) - p0[i];
        const double dg = static_cast<double>(p_gpu[i]) - p0[i];
        const double d  = dg - dc;
        num += d * d;
        den += dc * dc;
        dot += dg * dc;
        gn  += dg * dg;
        maxabs = std::max(maxabs, std::fabs(d));
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
    INFO("param-delta rel-L2 = " << rel << "  cos = " << cos << "  max abs = " << maxabs);
    if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(cos > 0.7);   // same update direction
    else                                    REQUIRE(rel < 2e-2);
    sub0_cuda_shutdown();
}

// Regression for the GPU-training divergence: amplify the weights so GELU inputs reach the
// saturation regime where the device tanh's __expf used to overflow to NaN (harmless at small
// init, but it poisoned the weights once training grew the activations). The forward must stay
// finite and track the CPU's clamped fast-math saturation.
TEST_CASE("CUDA forward stays finite under saturating GELU activations", "[cuda]") {
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> saved(sub0::params_ptr(), sub0::params_ptr() + n);
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = saved[i] * 6.0f;   // push into saturation
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int T = 16;
    std::vector<int> ids(static_cast<std::size_t>(T));
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : ids) x = tok(rng);

    sub0::graph_reset();
    sub0::Node* lg = sub0::forward(ids.data(), T);
    const std::vector<float> cpu(lg->data.begin(), lg->data.end());
    std::vector<float> gpu(static_cast<std::size_t>(T) * VOCAB, 0.0f);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, gpu.data()) == 0);

    bool   finite = true;
    double maxabs = 0.0, maxcpu = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        finite = finite && std::isfinite(gpu[i]);
        maxabs = std::max(maxabs, static_cast<double>(std::fabs(cpu[i] - gpu[i])));
        maxcpu = std::max(maxcpu, static_cast<double>(std::fabs(cpu[i])));
    }
    INFO("saturated: max abs diff = " << maxabs << "  max|cpu| = " << maxcpu);
    REQUIRE(finite);                          // dev_tanh must not overflow to NaN/Inf
    REQUIRE(maxabs < 0.05 * maxcpu + 1e-2);   // matches the CPU's clamped saturation

    std::copy(saved.begin(), saved.end(), sub0::params_ptr());   // restore for later tests
    sub0_cuda_shutdown();
}

// This build's dimensions for the pure footprint model (sub0/memplan.hpp), straight from the config.
static constexpr sub0::memplan::Dims kTestDims{ D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB };

// The footprint model must agree with the canonical parameter layout: param_floats() re-derives
// PARAM_FLOATS from dims (so the configurator can call it without layout.hpp). If they diverge,
// every footprint prediction is wrong by a constant -- catch it here, decoupled from the device.
TEST_CASE("memplan param_floats matches the canonical layout", "[cuda][memplan]") {
    REQUIRE(sub0::memplan::param_floats(kTestDims) == sub0::trainable_floats());
}

// The drift check the whole footprint scheme rests on: predict the resident training VRAM from the
// pure model, then ALLOCATE it for real and measure the device delta (cudaMemGetInfo). They must
// agree to within allocation-rounding noise. If a buffer is added/removed/resized in
// backend_cuda.cu without updating memplan.hpp, the gap blows past the tolerance and this fails --
// exactly the "we didn't maintain the calculation" regression we want to catch automatically.
TEST_CASE("memplan prediction matches measured device usage", "[cuda][memplan]") {
    for (const int batch : {32, 64, 128}) {
        double predicted_mb = 0.0, actual_mb = 0.0;
        REQUIRE(sub0_cuda_train_footprint(batch, &predicted_mb, &actual_mb) == 0);
        WARN("batch=" << batch << "  predicted=" << predicted_mb << " MiB  measured=" << actual_mb
             << " MiB  gap=" << (actual_mb - predicted_mb) << " MiB");
        REQUIRE(actual_mb > 0.0);
        // The cudaMemGetInfo measurement only counts DEDICATED VRAM. Once the footprint approaches
        // the dedicated budget it spills to WDDM shared memory, which the free-memory delta does NOT
        // see -- so a too-large prediction is structurally un-measurable (measured plateaus while
        // predicted keeps growing). Only validate parity for batches comfortably inside VRAM; the
        // larger batches still print their gap above for inspection. (For a small model all three
        // batches fit and are checked; the guard only excuses the genuinely over-budget ones.)
        if (GPU_VRAM_MB > 0 && predicted_mb > 0.8 * static_cast<double>(GPU_VRAM_MB)) {
            WARN("batch=" << batch << " predicted " << predicted_mb
                 << " MiB is near/over the VRAM budget; measurement spills to shared, skipping parity");
            continue;
        }
        // The measured delta is our exact byte sum plus per-cudaMalloc rounding and WDDM free-memory
        // noise -- a bounded, batch-independent offset, NOT a percentage. So we allow a fixed MiB
        // slack (sub0::memplan::FOOTPRINT_TOLERANCE_MB). A missing/extra/resized buffer shifts the
        // gap by hundreds of MiB -- far outside this band -- which is the regression we want to catch.
        CHECK(predicted_mb == Catch::Approx(actual_mb).margin(sub0::memplan::FOOTPRINT_TOLERANCE_MB));
    }
    sub0_cuda_shutdown();
}

