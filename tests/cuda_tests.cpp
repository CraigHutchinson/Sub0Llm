// cuda_tests.cpp — GPU backend tests. Built and linked only when SUB0_BUILD_CUDA is ON
// (see tests/CMakeLists.txt), since they require nvcc-built code and a CUDA device. They
// drive the backend across the extern "C" seam and check it against CPU references: the
// device parameter mirror must round-trip the weight blob, and the dense-linear kernel
// must match a CPU recomputation (the parity gate every GPU kernel is held to).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"     // trainable_floats()
#include "sub0/layout.hpp"   // PARAM_LAYOUT / PKind — attn-only grad slice for the bisection probe
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
extern "C" int  sub0_cuda_free_vram_mb();
extern "C" int  sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms);
extern "C" int  sub0_cuda_train_profile(int batch, int T, int iters,
                                        double* fwd_ms, double* bwd_ms, double* adam_ms);
extern "C" int  sub0_cuda_attn_check(int batch, int T, int iters, double* out_maxreldiff, double* out_speedup);
extern "C" int  sub0_cuda_attn_bwd_check(int batch, int T, int iters, double* out_maxreldiff, double* out_speedup);
extern "C" int  sub0_cuda_kv_reset();
extern "C" int  sub0_cuda_forward_one(int id, int pos, float* out_logits);
extern "C" int  sub0_cuda_attn_regcheck(int* stats_regs, int* stats_spill, int* dq_regs, int* dq_spill,
                                        int* dv_regs, int* dv_spill, int* dk_regs, int* dk_spill,
                                        int* fwd_regs, int* fwd_spill);
extern "C" int  sub0_cuda_train_reserve(int batch);
extern "C" int  sub0_cuda_scratch_stats(long long* fwd_rows, long long* tr_rows,
                                        long long* fwd_grows, long long* tr_grows);

namespace {
// RAII: guarantees sub0_cuda_shutdown() runs even when a REQUIRE above throws mid-test (Catch2's
// REQUIRE unwinds the test body via exception on failure, unlike CHECK). Without this, a test that
// fails BEFORE its own tail-end shutdown() call -- exactly the shape most tests here have -- leaves
// the cuBLAS handle (and whatever math mode / captured graph it holds) alive for whichever test
// Catch2 runs next in this same process, discovered 2026-07-02 chasing an elevated logit diff in the
// batch-grow test that only appeared when run after other [cuda] tests, not in isolation. Declare one
// at the top of any test that touches CUDA state (init/upload_params/set_tf32/...); do not also call
// sub0_cuda_shutdown() manually in that test (harmless if you do -- shutdown is idempotent -- but the
// guard already covers every exit path).
struct CudaGuard {
    ~CudaGuard() { sub0_cuda_shutdown(); }
};
}  // namespace

TEST_CASE("CUDA backend self-test runs on the device", "[cuda]") {
    REQUIRE(sub0_cuda_selftest() == 0);
}

TEST_CASE("CUDA param mirror round-trips the weight blob", "[cuda]") {
    CudaGuard _cuda_guard;
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> up(n), down(n, 0.0f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : up) x = dist(rng);

    REQUIRE(sub0_cuda_upload_params(up.data()) == 0);
    REQUIRE(sub0_cuda_download_params(down.data()) == 0);
    for (std::size_t i = 0; i < n; ++i) REQUIRE(down[i] == up[i]);   // exact: it is just a memcpy
}

TEST_CASE("CUDA dense linear matches a CPU reference", "[cuda]") {
    CudaGuard _cuda_guard;
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
    CudaGuard _cuda_guard;
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
}

TEST_CASE("CUDA batched forward matches per-window CPU logits", "[cuda]") {
    CudaGuard _cuda_guard;
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
}

TEST_CASE("CUDA TF32 forward stays close to the CPU logits", "[cuda]") {
    CudaGuard _cuda_guard;
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
    CudaGuard _cuda_guard;
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
}

// Bisection probe: cosine of ONLY the attention-projection grads (Wq/Wk/Wv/Wo) against the CPU.
// The attention saved buffers (a/qkv/att) and their grads (da/dqkv/datt) feed exactly these four
// weights; isolating them attributes a BF16 regression to the attention path rather than the FFN
// (which the reduced-gradient test already covers). Tighter than the full-grad gate so a single
// flipped buffer is visible: F32 must match closely, BF16 must still point the same direction.
TEST_CASE("CUDA attention-only gradient stays aligned with the CPU", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 55, data, starts, ids, targets);

    sub0::train_batch(data.data(), starts.data(), batch, T);
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    std::vector<float> gpu_grad(n, 0.0f);
    double loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &loss) == 0);

    // Accumulate cosine/rel-L2 over only the Q/K/V/O weight tensors (every layer).
    double num = 0.0, den = 0.0, dot = 0.0, gn = 0.0;
    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
        const bool attn = p.kind == sub0::PKind::Wq || p.kind == sub0::PKind::Wk ||
                          p.kind == sub0::PKind::Wv || p.kind == sub0::PKind::Wo;
        if (!attn) continue;
        for (std::size_t i = p.off; i < p.off + p.n(); ++i) {
            const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
            num += d * d;
            den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
            dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i];
            gn  += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
        }
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
    INFO("attn grad rel-L2 = " << rel << "  cos = " << cos);
    if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(cos > 0.7);
    else                                    REQUIRE(rel < 1e-2);
}

// Flash attention FORWARD guard: the tiled kernel (attn_fwd_tiled_kernel, the hot path) must (1)
// produce the SAME output as the naive reference (attn_train_act_kernel) -- they share the
// increasing-j online-softmax recurrence, so agreement is to ~bf16 rounding -- and (2) be a good
// deal faster. The speedup is a RATIO (tiled vs naive on identical inputs, same GPU state), so it is
// dimensionless: it does NOT drift with clocks/thermals/host load the way an absolute-ms floor
// would, which makes it a stable structural check that the shared-memory tiling stayed intact. The
// 2x floor is far below the ~8.5x seen at d448/T256 -- it exists to fail loudly if a future edit
// silently reverts the kernel to streaming K/V from global (ratio would collapse toward 1x).
TEST_CASE("CUDA flash-attention forward matches the naive kernel and is faster", "[cuda]") {
    CudaGuard _cuda_guard;
    double reldiff = 1.0, speedup = 0.0;
    const int rc = sub0_cuda_attn_check(128, SEQ_LEN, 50, &reldiff, &speedup);
    WARN("flash attn forward: max rel diff = " << reldiff << "   speedup = " << speedup << "x");
    REQUIRE(rc == 0);              // returns nonzero only on a parity failure
    REQUIRE(reldiff < 5e-2);       // tiled == naive to bf16 rounding
    REQUIRE(speedup > 2.0);        // tiling intact (regression guard; observed ~8.5x at d448)
}

// Flash attention BACKWARD guard. Correctness is gated tightly by the CPU-fp32 gradient tests above;
// this is a speed + gross-sanity guard. The naive reference accumulates dq/dk/dv in bf16, so it
// carries ~sqrt(T)*bf16_eps accumulation noise the FP32-accumulating tiled kernels do not -- hence the
// looser (0.15) parity bound here vs the bit-exact forward. The dimensionless speedup ratio is the
// stable regression signal (huge: the naive backward is low-parallelism, ~45x slower at d448).
TEST_CASE("CUDA flash-attention backward matches the naive kernel and is much faster", "[cuda]") {
    CudaGuard _cuda_guard;
    double reldiff = 1.0, speedup = 0.0;
    const int rc = sub0_cuda_attn_bwd_check(128, SEQ_LEN, 30, &reldiff, &speedup);
    WARN("flash attn backward: max rel diff = " << reldiff << "   speedup = " << speedup << "x");
    REQUIRE(rc == 0);              // returns nonzero only on a gross parity failure
    REQUIRE(reldiff < 1.5e-1);     // within bf16-accumulation noise of the naive reference
    REQUIRE(speedup > 3.0);        // tiling + parallelism intact (observed ~45x at d448)
}

// Register/local-memory-spill regression guard for the five flash-attention kernels (forward tile +
// the four backward kernels). Queries the ACTUAL compiled kernel attributes via cudaFuncGetAttributes
// (deterministic, load-independent -- same philosophy as the speedup ratio above, not wall-clock), so
// a future change to any of these kernels' resident per-thread state is caught here, not discovered
// later via a slow, easy-to-miss manual ptxas -v audit. All five are driven to ZERO register spill:
// dq and dk both via a warp-cooperative channel split (2 threads share a query/key, each holding
// HD/LANES channels, the two half-width dot products combined via warp shuffle -- attn_dq_lanes /
// attn_dk_lanes in backend_cuda.cu), dv by dropping the v/stat_dot state dv doesn't need. The naive
// "3*HD > 255" register-count formula undercounts real pressure (loop scalars, staging pointers): dq
// and dk both measured a real spill at HD=64 despite 3*HD=192 looking safe on paper, which is why the
// split threshold is empirically set at 3*HD >= 192, not > 255 (see attn_dq_lanes's comment).
TEST_CASE("CUDA attention kernels stay within their register/spill budget", "[cuda]") {
    CudaGuard _cuda_guard;
    int stats_regs = 0, stats_spill = 0, dq_regs = 0, dq_spill = 0, dv_regs = 0, dv_spill = 0,
        dk_regs = 0, dk_spill = 0, fwd_regs = 0, fwd_spill = 0;
    REQUIRE(sub0_cuda_attn_regcheck(&stats_regs, &stats_spill, &dq_regs, &dq_spill, &dv_regs, &dv_spill,
                                    &dk_regs, &dk_spill, &fwd_regs, &fwd_spill) == 0);
    INFO("stats: " << stats_regs << " regs, " << stats_spill << "B spill");
    INFO("dq:    " << dq_regs << " regs, " << dq_spill << "B spill");
    INFO("dv:    " << dv_regs << " regs, " << dv_spill << "B spill");
    INFO("dk:    " << dk_regs << " regs, " << dk_spill << "B spill");
    INFO("fwd:   " << fwd_regs << " regs, " << fwd_spill << "B spill");
    CHECK(stats_spill == 0);
    CHECK(dq_spill    == 0);
    CHECK(dv_spill    == 0);
    CHECK(fwd_spill   == 0);
    CHECK(dk_spill    == 0);
}

// GPU forward_one (KV-cache decode) parity, validated against a SHARPENED (non-random) model rather
// than raw random init. Random-init logits are near-flat, so the decode path's two-pass softmax and
// the full forward's flash online-softmax can each round differently and FLIP the argmax on
// essentially a coin toss, amplifying over layers -- a known benign artifact, not a kernel bug (see
// engine memory notes on the d448 random-weight divergence). ANY real training moves the model off
// that knife-edge. Overfitting one short, fully-deterministic sequence (next-token has a unique
// correct answer at every position, so a real gradient signal exists from step 1) sharpens the logits
// without the cost of a real corpus. If decode still disagrees with the full forward here, the
// mismatch is a genuine kernel bug, not argmax noise under near-uniform logits.
// Hidden (like [.bench]): 150 single-threaded CPU training steps (batch=1, no thread parallelism
// across a single window) make this the slowest case in the suite (~4min at d768) despite being a
// correctness gate, not a benchmark. Run explicitly when touching GPU decode: `sub0_tests
// "*forward_one decode*"`.
TEST_CASE("CUDA forward_one decode matches full forward once the model is trained (non-random weights)",
         "[cuda][.slow]") {
    CudaGuard _cuda_guard;
    sub0::build_model();

    // ids cycle 0..K-1 repeating: given the causal prefix, next-token is uniquely determined at every
    // position (zero label noise), so the loss has real signal to descend from step 1.
    const int T = std::min(16, SEQ_LEN);
    const int K = 6;
    std::vector<int> data(static_cast<std::size_t>(T) + 1);
    for (int i = 0; i < static_cast<int>(data.size()); ++i) data[i] = i % K;
    const std::vector<std::size_t> starts(1, std::size_t{0});

    sub0::AdamW opt(0.01f);
    float loss = 0.0f;
    for (int s = 0; s < 150; ++s) {
        loss = sub0::train_batch(data.data(), starts.data(), 1, T);
        opt.step();
    }
    const float uniform = std::log(static_cast<float>(VOCAB));   // untrained loss floor
    INFO("post-training loss = " << loss << "  (uniform baseline ~= " << uniform << ")");
    REQUIRE(loss < 0.85f * uniform);      // real learning happened, not still at the random-init floor

    sub0_cuda_set_tf32(0);                                    // full FP32 for a tight parity gate
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const std::vector<int> ids(data.begin(), data.begin() + T);
    std::vector<float> full(static_cast<std::size_t>(T) * VOCAB);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, full.data()) == 0);

    REQUIRE(sub0_cuda_kv_reset() == 0);
    auto argmax = [](const float* p, int n) { int b = 0; for (int j = 1; j < n; ++j) if (p[j] > p[b]) b = j; return b; };
    std::vector<float> one(static_cast<std::size_t>(VOCAB));
    int agree = 0;
    double maxrel = 0.0;
    for (int pos = 0; pos < T; ++pos) {
        REQUIRE(sub0_cuda_forward_one(ids[pos], pos, one.data()) == 0);
        const float* ref = full.data() + static_cast<std::size_t>(pos) * VOCAB;
        double maxabs = 0.0, maxmag = 1e-30;
        for (int j = 0; j < VOCAB; ++j) {
            maxabs = std::max(maxabs, static_cast<double>(std::fabs(one[j] - ref[j])));
            maxmag = std::max(maxmag, static_cast<double>(std::fabs(ref[j])));
        }
        maxrel = std::max(maxrel, maxabs / maxmag);
        if (argmax(one.data(), VOCAB) == argmax(ref, VOCAB)) ++agree;
    }
    const double top1_pct = 100.0 * agree / T;
    INFO("decode vs full-forward top-1 agreement = " << top1_pct << "%  max rel diff = " << maxrel);
    REQUIRE(top1_pct >= 98.0);   // trained (sharp) logits: no amplification floor -- expect near-exact
}

TEST_CASE("CUDA backward matches the CPU gradient with short padded windows", "[cuda]") {
    CudaGuard _cuda_guard;
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
}

TEST_CASE("CUDA AdamW step matches the CPU optimizer update", "[cuda]") {
    CudaGuard _cuda_guard;
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
}

// Regression for the GPU-training divergence: amplify the weights so GELU inputs reach the
// saturation regime where the device tanh's __expf used to overflow to NaN (harmless at small
// init, but it poisoned the weights once training grew the activations). The forward must stay
// finite and track the CPU's clamped fast-math saturation.
TEST_CASE("CUDA forward stays finite under saturating GELU activations", "[cuda]") {
    CudaGuard _cuda_guard;
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
    CudaGuard _cuda_guard;
    for (const int batch : {32, 64, 128, 256}) {           // 256 = the training batch (~4.7 GB); within VRAM
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
}

// VRAM leak / smoke trace: allocate the full resident TRAINING set for a batch, then shut down and
// confirm the device buffers are released. Catches a buffer that sub0_cuda_shutdown forgets to free
// (the "2.7 GB during the run -> 0 idle" question -- proves the footprint is the process's own and is
// reclaimed, not leaked). cudaMemGetInfo is noisy (driver/WDDM), so we assert RECLAMATION, not bits.
TEST_CASE("CUDA shutdown releases the training footprint (no VRAM leak)", "[cuda][memplan]") {
    // Explicit mid-test shutdown() call below is the thing under test (reclaim behaviour), not a
    // cleanup call -- CudaGuard is still declared as a safety net for the REQUIREs above it, which
    // would otherwise skip both the mid-test and the final shutdown() on failure.
    CudaGuard _cuda_guard;
    double pred = 0.0, act = 0.0;
    REQUIRE(sub0_cuda_train_footprint(128, &pred, &act) == 0);   // allocs fwd+train+opt @128, leaves resident
    REQUIRE(act > 0.0);
    const int free_loaded = sub0_cuda_free_vram_mb();            // scratch resident -> low free
    REQUIRE(free_loaded > 0);
    sub0_cuda_shutdown();                                        // release scratch + params + opt
    const int free_after = sub0_cuda_free_vram_mb();             // context persists; buffers gone
    WARN("VRAM trace: footprint=" << act << " MiB | free loaded=" << free_loaded
         << " | free after shutdown=" << free_after << " MiB");
    CHECK(free_after >= free_loaded);                            // shutdown gave memory back
    // Reclaimed >= most of the footprint we allocated -> nothing of consequence leaked.
    CHECK(static_cast<double>(free_after - free_loaded) >= 0.5 * act);
    sub0_cuda_shutdown();
}

// Batch-grow + inference-graph re-capture: forward at batch 4, then a LARGER batch 8. The grow path
// (fwd_free_batch -> invalidate_graph -> reallocate -> recapture) must produce correct logits at both
// sizes. Guards the session's grow-on-demand + graph-capture logic against a stale-buffer/graph bug.
TEST_CASE("CUDA forward stays correct across a batch grow (graph re-capture)", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int T = 10;
    for (const int batch : {4, 8}) {                            // 4 captures; 8 grows + recaptures
        std::vector<int> ids(static_cast<std::size_t>(batch) * T);
        std::mt19937 rng(101 + batch);
        std::uniform_int_distribution<int> tok(0, VOCAB - 1);
        for (int& x : ids) x = tok(rng);

        std::vector<float> cpu(static_cast<std::size_t>(batch) * T * VOCAB);
        for (int b = 0; b < batch; ++b) {
            sub0::graph_reset();
            sub0::Node* lg = sub0::forward(ids.data() + static_cast<std::size_t>(b) * T, T);
            std::copy(lg->data.begin(), lg->data.end(), cpu.begin() + static_cast<std::size_t>(b) * T * VOCAB);
        }
        std::vector<float> gpu(static_cast<std::size_t>(batch) * T * VOCAB, 0.0f);
        REQUIRE(sub0_cuda_forward(ids.data(), batch, T, gpu.data()) == 0);

        double max_abs = 0.0;
        for (std::size_t i = 0; i < cpu.size(); ++i)
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(cpu[i] - gpu[i])));
        INFO("batch " << batch << " max abs logit diff = " << max_abs);
        // The lm_head GEMM forces bf16-tensor-core compute (force_tc in launch_linear, backend_cuda.cu)
        // independent of the CudaTf32 knob this test disables above -- measured (nsys, 2026-07-02): its
        // awkward VOCAB-wide N was the one GEMM in the whole forward/backward pass NOT already landing
        // on a tensor-core kernel under cuBLAS's own heuristic, unlike every other GEMM here (QKV/attn-
        // out/FFN), which already run at this same precision level regardless of the knob. So the CPU
        // reference (exact FP32) and the GPU logits are expected to diverge by a bit more than the old
        // near-bit-exact bound -- 3e-2 comfortably covers it (measures ~3.3e-3 in practice) with margin
        // to spare, while still catching a real correctness bug (NaN, a wrong axis, stale graph state),
        // which would blow well past this. (This threshold was briefly chasing a ~2e-2 value seen when
        // this test ran after others in the same process -- that was CudaGuard's actual bug, a cuBLAS
        // handle/math-mode leak from an earlier test's REQUIRE throwing before its own cleanup; fixed
        // now, not a property of this GEMM.)
        REQUIRE(max_abs < 3e-2);
    }
}

// Row-budget growth semantics (token-budget batching under vary_seq): after reserving
// `kBase * SEQ_LEN` rows, EVERY (batch, T) pair whose product fits that budget -- including batches
// LARGER than the reserving batch at shorter T, the exact trade the training loop makes -- must be
// served by the existing buffers with NO reallocation, and exceeding the budget must grow. This is
// the guard against the old `g_tr_cap >= batch` check silently resizing the whole training scratch
// to batch_t * SEQ_LEN rows (a VRAM blowout) the first time a short-T step scaled its batch up.
TEST_CASE("CUDA training scratch trades batch against T inside one row budget", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_init() == 0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    constexpr int kBase = 8;                              // reserve = kBase * SEQ_LEN rows
    REQUIRE(sub0_cuda_train_reserve(kBase) == 0);
    long long fwd_rows = 0, tr_rows = 0, fwd_grows0 = 0, tr_grows0 = 0;
    REQUIRE(sub0_cuda_scratch_stats(&fwd_rows, &tr_rows, &fwd_grows0, &tr_grows0) == 0);
    REQUIRE(tr_rows  == static_cast<long long>(kBase) * SEQ_LEN);
    REQUIRE(fwd_rows == static_cast<long long>(kBase) * SEQ_LEN);

    const std::size_t n = sub0::trainable_floats();
    std::vector<float> grad(n);
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;

    // In-budget pairs: full shape, batch scaled 2x/4x/8x at T/2 / T/4 / T/8, and a tiny step.
    for (const int k : {1, 2, 4, 8}) {
        const int batch = kBase * k, T = std::max(1, SEQ_LEN / k);
        INFO("in-budget step batch=" << batch << " T=" << T);
        make_windows(batch, T, 900u + static_cast<unsigned>(k), data, starts, ids, targets);
        double loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, grad.data(), &loss) == 0);
        REQUIRE(std::isfinite(loss));
    }
    {
        make_windows(1, 8, 990, data, starts, ids, targets);   // far under budget: also no realloc
        double loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), 1, 8, grad.data(), &loss) == 0);
    }
    long long fwd_grows1 = 0, tr_grows1 = 0;
    REQUIRE(sub0_cuda_scratch_stats(nullptr, nullptr, &fwd_grows1, &tr_grows1) == 0);
    CHECK(tr_grows1  == tr_grows0);                        // zero reallocations inside the budget
    CHECK(fwd_grows1 == fwd_grows0);

    // One row past the budget (batch = kBase+1 at full T) must grow, exactly once, to the new product.
    {
        const int batch = kBase + 1;
        make_windows(batch, SEQ_LEN, 991, data, starts, ids, targets);
        double loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, SEQ_LEN, grad.data(), &loss) == 0);
        REQUIRE(std::isfinite(loss));
    }
    long long tr_rows2 = 0, fwd_grows2 = 0, tr_grows2 = 0;
    REQUIRE(sub0_cuda_scratch_stats(nullptr, &tr_rows2, &fwd_grows2, &tr_grows2) == 0);
    CHECK(tr_grows2  == tr_grows1 + 1);
    CHECK(fwd_grows2 == fwd_grows1 + 1);
    CHECK(tr_rows2   == static_cast<long long>(kBase + 1) * SEQ_LEN);
}

// Gradient-scale pin for the effective-batch trade: presenting the SAME windows twice at 2x the
// batch must leave the mean loss AND the reduced gradient unchanged -- ce_backward_kernel weights
// each row 1/(batch*len[b]) with the ACTUAL per-call batch, so a step that scales batch_t up at
// short T keeps gradient magnitude identical to the nominal-batch step. A wrong denominator (e.g.
// still using the tuned batch) would show up here as an exact 2x, far outside the tolerance.
TEST_CASE("CUDA gradient scale is invariant to duplicating the batch", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);                                 // full FP32 for a tight gate
    // Suite-order independence: build_model randomizes the shared host params only ONCE per
    // process, and earlier tests (the AdamW parity case) step them -- a stepped model here sits
    // near softmax saturation (loss ~19.4), where shape-dependent GEMM rounding between the M and
    // 2M calls amplifies far past the tolerance bands below (measured cos 0.987 / ratio 0.87 in
    // full-suite order vs 0.99997 / 0.9993 in isolation). Re-seed deterministically small instead.
    {
        std::mt19937 prng(4242);
        std::normal_distribution<float> w(0.0f, 0.02f);
        float* p = sub0::params_ptr();
        for (std::size_t i = 0; i < sub0::trainable_floats(); ++i) p[i] = w(prng);
    }
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 16;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 77, data, starts, ids, targets);

    const std::size_t n = sub0::trainable_floats();
    std::vector<float> g1(n), g2(n);
    double loss1 = 0.0, loss2 = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, g1.data(), &loss1) == 0);

    std::vector<int> ids2(ids), targets2(targets);         // the same windows, twice
    ids2.insert(ids2.end(), ids.begin(), ids.end());
    targets2.insert(targets2.end(), targets.begin(), targets.end());
    REQUIRE(sub0_cuda_backward(ids2.data(), targets2.data(), 2 * batch, T, g2.data(), &loss2) == 0);

    REQUIRE(std::isfinite(loss1));
    // The two calls run DIFFERENT GEMM shapes (M vs 2M), so cuBLAS tiles/reduces in a different
    // order and -- under bf16 activation storage -- the logits themselves differ at ~1e-4 rel
    // (measured: loss 8.39651 vs 8.39559 on the d192 build). That rounding shows up as a few 1e-3
    // of gradient rel-L2, so the SCALE assertions are direction + norm-ratio, where a wrong batch
    // denominator is an exact 2x (ratio 0.5, cos unchanged) -- far outside these bands.
    CHECK(loss2 == Catch::Approx(loss1).epsilon(1e-3));
    double dot = 0.0, n1 = 0.0, n2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(g1[i]) * g2[i];
        n1  += static_cast<double>(g1[i]) * g1[i];
        n2  += static_cast<double>(g2[i]) * g2[i];
    }
    const double cos   = dot / std::max(std::sqrt(n1 * n2), 1e-30);
    const double ratio = std::sqrt(n2 / std::max(n1, 1e-30));
    INFO("duplicated-batch grad cos = " << cos << "  |g2|/|g1| = " << ratio
         << "  loss1 = " << loss1 << "  loss2 = " << loss2);
    CHECK(cos > 0.995);
    CHECK(ratio > 0.95);
    CHECK(ratio < 1.05);
}

// Per-phase profile: attribute the step time to forward / backward / adam. The backward RECOMPUTES
// the checkpointed activations (a memory-for-compute trade) so it is expected to be heavy; this
// surfaces the split so further throughput rework (optional checkpointing) is data-driven. The
// grad-clip scale is computed on-device (grad_clip_scale_kernel, 2026-07) instead of a host
// round-trip, so "adam" no longer carries that bubble -- this profile is how the ~10% drop in the
// adam phase's elapsed time was confirmed.
TEST_CASE("CUDA per-phase profile attributes the step (forward/backward/adam)", "[cuda][.bench]") {
    CudaGuard _cuda_guard;
    double f = 0.0, b = 0.0, a = 0.0;
    REQUIRE(sub0_cuda_train_profile(32, 128, 10, &f, &b, &a) == 0);
    const double tot = f + b + a;
    WARN("step phases (ms): forward=" << f << "  backward=" << b << "  adam=" << a << "  | total=" << tot
         << "  (backward recomputes checkpointed activations; grad-clip scale is on-device, no host "
            "round-trip)");
    REQUIRE(tot > 0.0);
    CHECK(std::isfinite(f)); CHECK(std::isfinite(b)); CHECK(std::isfinite(a));
}

// Cold vs sustained step: report ms/step for a 2-iter (cold) and a many-iter (warm) measurement of the
// SAME small config. Documents the cold-clock/cuBLAS-autotune gap that makes the tuner's 2-warmup
// samples over-report step time vs sustained training; a small config keeps the test fast.
TEST_CASE("CUDA train step warms up (sustained < cold)", "[cuda][.bench]") {
    CudaGuard _cuda_guard;
    const int batch = 16, T = 64;                               // small -> fast, but enough to clock-ramp
    double cold = 0.0, warm = 0.0;
    REQUIRE(sub0_cuda_train_benchmark(batch, T, 2,  &cold) == 0);
    REQUIRE(sub0_cuda_train_benchmark(batch, T, 40, &warm) == 0);
    WARN("step ms: cold(2 iters)=" << cold << "  warm(40 iters)=" << warm
         << "  (sustained is faster once the clock + cuBLAS warm)");
    REQUIRE(std::isfinite(cold));
    REQUIRE(std::isfinite(warm));
    CHECK(warm <= cold * 1.5);                                  // warming must not make it slower
}

