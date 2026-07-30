// cuda_tests.cpp — GPU backend tests. Built and linked only when SUB0_BUILD_CUDA is ON
// (see tests/CMakeLists.txt), since they require nvcc-built code and a CUDA device. They
// drive the backend across the extern "C" seam and check it against CPU references: the
// device parameter mirror must round-trip the weight blob, and the dense-linear kernel
// must match a CPU recomputation (the parity gate every GPU kernel is held to).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/blend.hpp"     // doc_of -- the hybrid router's own per-window document resolution
#include "sub0/core.hpp"     // trainable_floats()
#include "sub0/decode.hpp"   // gpu_decode_try_enable -- must honour the supports_decode cap
#include "sub0/eval.hpp"     // sub0::eval -- the consumer under test for the device eval seam
#include "sub0/layout.hpp"   // PARAM_LAYOUT / PKind — attn-only grad slice for the bisection probe
#include "sub0/memplan.hpp"  // the pure footprint model under test
#include "sub0/muon.hpp"     // sub0::muon::newton_schulz5 -- the CPU reference the GPU Muon tests check against
#include "sub0/scratch_slots.hpp"  // ScratchBindings/SlotEncoding + SCRATCH_SLOT_BASE -- the binding-compose
                                   // parity tests build CPU-side bindings and the GPU-side override table

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

extern "C" int  sub0_cuda_selftest();
extern "C" int  sub0_cuda_init();
extern "C" void sub0_cuda_shutdown();
extern "C" int  sub0_cuda_upload_params(const float* host);
extern "C" int  sub0_cuda_download_params(float* host);
extern "C" int  sub0_cuda_download_opt(float* host_m, float* host_v);
extern "C" int  sub0_cuda_upload_opt(const float* host_m, const float* host_v);
extern "C" int  sub0_cuda_linear(const float* X, int T, int in, int out,
                                 const float* W, const float* bias, float* Y);
extern "C" int  sub0_cuda_forward(const int* ids, int batch, int T, float* out_logits);
extern "C" void sub0_cuda_set_tf32(int on);
extern "C" int  sub0_cuda_backward(const int* ids, const int* targets, int batch, int T,
                                   float* out_grad, double* out_loss, const int* lengths = nullptr,
                                   double* out_win_loss = nullptr);
// sub0_cuda_forward_loss is declared canonically by sub0/device_backend.hpp (included above), which
// this target now compiles in its device configuration -- no local extern needed.
extern "C" int  sub0_cuda_adam_step(float lr, long t, float muon_lr);
// Forces the logits row-chunk COUNT (0 = memplan's derivation). Test-only lever -- see the
// chunk-invariance cases below and docs/MEMORY_AUDIT.md 4a.
extern "C" void sub0_cuda_set_logits_chunks(int n);
extern "C" int  sub0_cuda_muon_ns_check(const float* in, int rows, int cols, int force_tf32, float* out);
extern "C" int  sub0_cuda_train_predicted_mb(int batch);
extern "C" int  sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb);
extern "C" int  sub0_cuda_free_vram_mb();
extern "C" int  sub0_cuda_seq_len();
extern "C" int  sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms);
extern "C" int  sub0_cuda_train_profile(int batch, int T, int iters,
                                        double* fwd_ms, double* bwd_ms, double* adam_ms);
extern "C" int  sub0_cuda_attn_check(int batch, int T, int iters, double* out_maxreldiff, double* out_speedup);
extern "C" int  sub0_cuda_attn_bwd_check(int batch, int T, int iters, double* out_maxreldiff, double* out_speedup);
extern "C" int  sub0_cuda_test_accumulate_check(int M, int in, int out, double* out_maxreldiff);
extern "C" int  sub0_cuda_tied_head_check(int M, int C, int V,
                                          double* out_relL2_fwd, double* out_relL2_bwd);
extern "C" int  sub0_cuda_swiglu_check(int M, int F, double* out_relL2_fwd, double* out_relL2_bwd);
extern "C" int  sub0_cuda_ce_chunk_check(int M, int T, int batch, int chunk_cap,
                                         double* out_relL2_dlogits, double* out_reldiff_loss);
extern "C" int  sub0_cuda_qknorm_check(int shape_sel, int rows,
                                       double* out_relL2_fwd, double* out_relL2_bwd);
extern "C" int  sub0_cuda_kv_reset();
extern "C" int  sub0_cuda_forward_one(int id, int pos, float* out_logits);
extern "C" int  sub0_cuda_attn_regcheck(int* stats_regs, int* stats_spill, int* dq_regs, int* dq_spill,
                                        int* dv_regs, int* dv_spill, int* dk_regs, int* dk_spill,
                                        int* fwd_regs, int* fwd_spill);
extern "C" int  sub0_cuda_train_reserve(int batch);
extern "C" int  sub0_cuda_scratch_stats(long long* fwd_rows, long long* tr_rows,
                                        long long* fwd_grows, long long* tr_grows);
extern "C" int  sub0_cuda_set_window_bindings(const int* override_idx, int n_positions,
                                              const int* entries, int n_entries,
                                              const int* frags, int n_frags);
extern "C" int  sub0_cuda_binding_compose_check(int enc_sel, unsigned seed,
                                                double* out_fwd_maxabs, double* out_fwd_maxrel,
                                                double* out_fwd_act_maxrel,
                                                double* out_bwd_maxabs, double* out_bwd_maxrel);

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

    // CPU reference: Y[t,o] = sum_p X[t,p]*W[p,o] + bias[o] (same p-order as the kernel). launch_linear
    // is the PRODUCTION GEMM path, which on a BF16 build (GEMM_DTYPE == BF16) rounds its inputs to BF16
    // -- so a near-zero output element (heavy cancellation of ~O(in) terms) carries large RELATIVE error
    // and a tight per-element absolute gate is the wrong metric. Compare the whole-vector relative L2
    // (magnitude-weighted, robust to those near-zero elements), BF16-aware like the gradient/tied-head
    // parity checks. set_tf32(0) above pins the TF32 path off; it does NOT turn the BF16 GEMM into FP32.
    double num = 0.0, den = 0.0;
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < out; ++o) {
            float acc = B[static_cast<std::size_t>(o)];
            for (int p = 0; p < in; ++p)
                acc += X[static_cast<std::size_t>(t) * in + p] * W[static_cast<std::size_t>(p) * out + o];
            const double d = static_cast<double>(Yg[static_cast<std::size_t>(t) * out + o]) - acc;
            num += d * d; den += static_cast<double>(acc) * acc;
        }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    INFO("dense linear rel-L2 = " << rel);
    if constexpr (GEMM_DTYPE == Dtype::BF16) REQUIRE(rel < 3e-2);   // BF16 GEMM: behaviour, not bit-parity
    else                                     REQUIRE(rel < 1e-3);
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

// ---------------------------------------------------------------------------------------------
// The EVAL seam (sub0_cuda_forward_loss): the number `report` and every A/B comparison are scored
// with. This is the gate the whole device-eval path rests on -- an eval that silently disagreed with
// the CPU would not fail loudly, it would just quietly shift every model's val_nelbo by some amount
// and corrupt comparisons between a device-scored model and a CPU-scored one.
//
// Two independent checks, because they can fail separately:
//   1. against sub0_cuda_backward's loss on the SAME input -- both go through ce_backward_kernel, so a
//      disagreement means the loss-only branch (dlogits == nullptr) diverged from the gradient branch;
//   2. against the CPU engine end to end through sub0::eval -- which additionally covers the forward
//      itself (forward_device vs sub0::forward) and all the consumer-side window/batch plumbing.
// The gate is TIGHT (not the loose cosine the gradient tests use) because this path is FP32
// throughout: it runs the inference forward, not the bf16 training forward.
TEST_CASE("CUDA forward_loss matches the backward path's loss", "[cuda][eval]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 77, data, starts, ids, targets);

    std::vector<float> grad(sub0::trainable_floats(), 0.0f);
    double bwd_loss = 0.0, fwd_loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, grad.data(), &bwd_loss) == 0);
    REQUIRE(sub0_cuda_forward_loss(ids.data(), targets.data(), batch, T, &fwd_loss, nullptr) == 0);
    INFO("forward_loss = " << fwd_loss << "  backward loss = " << bwd_loss);
    REQUIRE(std::isfinite(fwd_loss));
    // The two forwards are not bit-identical (bf16 training activations vs FP32 inference), so this
    // is a behavioural gate, not an exact one -- but a real defect in the loss-only branch (a wrong
    // normalizer, a skipped mask, the ptgt shortcut picking the wrong element) moves it far more.
    REQUIRE(fwd_loss == Catch::Approx(bwd_loss).epsilon(0.05));

    // Masked (LOSS_IGNORE_INDEX) targets must be excluded from BOTH the loss and its per-window
    // denominator. Masking rows without renormalizing would shift the mean by ~len/active.
    std::vector<int> masked = targets;
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < T; t += 3) masked[static_cast<std::size_t>(b) * T + t] = sub0::LOSS_IGNORE_INDEX;
    double masked_fwd = 0.0, masked_bwd = 0.0;
    REQUIRE(sub0_cuda_forward_loss(ids.data(), masked.data(), batch, T, &masked_fwd, nullptr) == 0);
    REQUIRE(sub0_cuda_backward(ids.data(), masked.data(), batch, T, grad.data(), &masked_bwd) == 0);
    INFO("masked forward_loss = " << masked_fwd << "  masked backward loss = " << masked_bwd);
    REQUIRE(std::isfinite(masked_fwd));
    REQUIRE(masked_fwd == Catch::Approx(masked_bwd).epsilon(0.05));

    // Short-window padding: `lengths` marks the real extent, and positions past it must contribute
    // nothing. A window trained on only its first half must not score the same as the full one.
    std::vector<int> lengths(static_cast<std::size_t>(batch), T / 2);
    double short_loss = 0.0;
    REQUIRE(sub0_cuda_forward_loss(ids.data(), targets.data(), batch, T, &short_loss, lengths.data()) == 0);
    REQUIRE(std::isfinite(short_loss));
    REQUIRE(short_loss != Catch::Approx(fwd_loss).epsilon(1e-9));
}

// The PER-WINDOW loss readout on real hardware -- the device half of tests/win_loss_tests.cpp, which
// pins the same identity on the CPU engine. This is the measurement docs/TUTOR.md's mastery surface is
// built from, and it is only affordable because it is a readout of a value ce_backward_kernel already
// computes; the identity below is what proves the readout and the reported scalar are the SAME quantity:
//
//     mean over b of win[b]  ==  the batch mean the same call returned
//
// Unlike the cross-backend comparisons above, this one is INTERNAL to the device: both numbers come out
// of ONE kernel launch over ONE set of logits, so nothing but arithmetic separates them.
//
// The tolerance is 1e-6 relative rather than exact, and the reason is worth stating because it also
// bounds what the mastery surface can resolve. The scalar accumulator forms `w * nll` in FLOAT (w =
// 1/(batch*denom), the pre-existing expression, deliberately left alone so the reported loss stays
// bit-identical), while the per-window readout divides in double. Measured divergence at a masked batch
// is ~6e-9 relative -- fp32 rounding of the summands, not a defect. A real defect in this readout is
// nowhere near that small: a wrong denominator moves a window by the ratio len/active (~1.5x here), and
// a chunk-local row index corrupts whole windows.
TEST_CASE("CUDA per-window loss readout averages to the batch mean", "[cuda][eval][winloss]") {
    // Relative agreement floor between the float-accumulated scalar and the double-accumulated readout.
    constexpr double kReadoutEps = 1e-6;
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 78, data, starts, ids, targets);

    // 1. the loss-only (eval) branch.
    double fwd_loss = 0.0;
    std::vector<double> win(static_cast<std::size_t>(batch), -1.0);
    REQUIRE(sub0_cuda_forward_loss(ids.data(), targets.data(), batch, T, &fwd_loss, nullptr,
                                   win.data()) == 0);
    double sum = 0.0;
    for (double w : win) { REQUIRE(std::isfinite(w)); REQUIRE(w > 0.0); sum += w; }
    INFO("mean of per-window = " << sum / batch << "  batch mean = " << fwd_loss);
    REQUIRE(sum / batch == Catch::Approx(fwd_loss).epsilon(kReadoutEps));

    // 2. the gradient branch (dlogits != nullptr), which reaches the same kernel down the CHUNKED
    //    head_ce_chunked path -- where a per-window accumulator indexed by a chunk-local row instead of
    //    an absolute one would corrupt every window past the first chunk boundary and still return a
    //    plausible batch mean.
    std::vector<float> grad(sub0::trainable_floats(), 0.0f);
    double bwd_loss = 0.0;
    std::vector<double> bwin(static_cast<std::size_t>(batch), -1.0);
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, grad.data(), &bwd_loss, nullptr,
                               bwin.data()) == 0);
    double bsum = 0.0;
    for (double w : bwin) { REQUIRE(std::isfinite(w)); REQUIRE(w > 0.0); bsum += w; }
    REQUIRE(bsum / batch == Catch::Approx(bwd_loss).epsilon(kReadoutEps));

    // 3. masked + ragged: the per-window figure must use the ACTIVE-count denominator, not the raw
    //    length. Masking two thirds of window 0 only, so a wrong denominator shows up as ONE window
    //    disagreeing while the others stay right -- invisible in any batch-level number.
    std::vector<int> masked = targets;
    for (int t = 0; t < T; t += 3) masked[static_cast<std::size_t>(t)] = sub0::LOSS_IGNORE_INDEX;
    double masked_loss = 0.0;
    std::vector<double> mwin(static_cast<std::size_t>(batch), -1.0);
    REQUIRE(sub0_cuda_forward_loss(ids.data(), masked.data(), batch, T, &masked_loss, nullptr,
                                   mwin.data()) == 0);
    double msum = 0.0;
    for (double w : mwin) { REQUIRE(std::isfinite(w)); msum += w; }
    REQUIRE(msum / batch == Catch::Approx(masked_loss).epsilon(kReadoutEps));
    // Only window 0 was masked, so only window 0 may have moved.
    for (int b = 1; b < batch; ++b)
        REQUIRE(mwin[static_cast<std::size_t>(b)] == Catch::Approx(win[static_cast<std::size_t>(b)]).epsilon(kReadoutEps));

    // 4. the readout must not perturb what it reads: same call without it, same scalar.
    double again = 0.0;
    REQUIRE(sub0_cuda_forward_loss(ids.data(), targets.data(), batch, T, &again, nullptr) == 0);
    REQUIRE(again == Catch::Approx(fwd_loss).epsilon(1e-12));
}

TEST_CASE("CUDA device eval reproduces the CPU eval end to end", "[cuda][eval]") {
    // The consumer-level property, on real hardware: sub0::eval routed through the device must give
    // the same NELBO as the CPU route over the identical window set. tests/eval_seam_tests.cpp pins
    // the same property against a mock backend (so it runs everywhere); this one additionally proves
    // the CUDA kernels agree, which the mock by construction cannot.
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);

    std::mt19937 rng(97);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> corpus(static_cast<std::size_t>(SEQ_LEN) * 12);
    for (int& x : corpus) x = tok(rng);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0, 8);
    REQUIRE(ws.count > 1);

    const sub0::eval::Session s(/*allow=*/true);
    REQUIRE(s.use_device);

    const double cpu = sub0::eval::nelbo_cpu(data, ws, SEQ_LEN, 2);
    const double dev = sub0::eval::nelbo_device(data, ws, SEQ_LEN);
    INFO("cpu nelbo = " << cpu << "  device nelbo = " << dev);
    REQUIRE(std::isfinite(cpu));
    REQUIRE(std::isfinite(dev));
    // 1% -- comfortably inside FP32 forward drift, and far tighter than any difference an A/B is
    // asked to resolve (the LoopSplit arms sit ~4% apart, seed noise ~1%).
    REQUIRE(dev == Catch::Approx(cpu).epsilon(0.01));

    // The same must hold at a short context width, which is what the context-length curve scores.
    const double cpu64 = sub0::eval::nelbo_cpu(data, ws, 64, 2);
    const double dev64 = sub0::eval::nelbo_device(data, ws, 64);
    INFO("cpu nelbo@64 = " << cpu64 << "  device nelbo@64 = " << dev64);
    REQUIRE(dev64 == Catch::Approx(cpu64).epsilon(0.01));
}

// The ignore-index CE path (loss-masked training -- e.g. the uncombine curriculum's harness-injected
// spans): a target < 0 (LOSS_IGNORE_INDEX) trains NO gradient and the per-window loss/grad normalizes
// over the ACTIVE (non-ignored) count, not the raw length. Masks ~1/3 of the rows and checks the GPU
// reduced gradient AND mean loss match the CPU train_batch loss_mask path -- so the device active[]
// normalizer agrees with op_cross_entropy's ce_active. If the kernel wrongly normalized by length
// instead of the active count, the grad magnitudes (and loss) would be off by ~len/active (~1.6x here)
// -- far outside the gate below. (The dense test above already covers the active == length case.)
TEST_CASE("CUDA masked (ignore-index) backward matches the CPU loss-masked gradient", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);                                  // full FP32 for a tight parity gate
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8, M = batch * T;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 55, data, starts, ids, targets);

    // Row mask (1 = trained, 0 = ignored): every 3rd row, so each 8-row window keeps >= 5 active and
    // none is fully masked. The CPU loss_mask is parallel to `data`, read at the TARGET position
    // p = b*T + t + 1; the GPU instead sees those positions as LOSS_IGNORE_INDEX targets directly.
    std::vector<std::uint8_t> row_train(static_cast<std::size_t>(M));
    for (int m = 0; m < M; ++m) row_train[static_cast<std::size_t>(m)] = (m % 3 == 0) ? 0 : 1;
    std::vector<std::uint8_t> loss_mask(static_cast<std::size_t>(M) + 1, 1);   // loss_mask[0] unused (p >= 1)
    for (int m = 0; m < M; ++m) loss_mask[static_cast<std::size_t>(m) + 1] = row_train[static_cast<std::size_t>(m)];
    std::vector<int> targets_masked(targets);
    for (int m = 0; m < M; ++m)
        if (!row_train[static_cast<std::size_t>(m)]) targets_masked[static_cast<std::size_t>(m)] = sub0::LOSS_IGNORE_INDEX;

    // CPU reference: loss-masked reduced gradient + mean loss.
    const float cpu_loss = sub0::train_batch(data.data(), starts.data(), batch, T, nullptr, loss_mask.data());
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    std::vector<float> gpu_grad(n, 0.0f);
    double gpu_loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets_masked.data(), batch, T, gpu_grad.data(), &gpu_loss) == 0);

    double num = 0.0, den = 0.0, dot = 0.0, gn = 0.0, maxabs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
        num += d * d; den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
        dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i]; gn += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
        maxabs = std::max(maxabs, std::fabs(d));
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
    INFO("masked grad rel-L2 = " << rel << "  cos = " << cos << "  max abs = " << maxabs
         << "  cpu_loss = " << cpu_loss << "  gpu_loss = " << gpu_loss);
    REQUIRE(std::isfinite(gpu_loss));
    REQUIRE(std::fabs(gpu_loss - static_cast<double>(cpu_loss)) < 1e-2);   // active-count normalization agrees
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

// Bisection probe for tied embeddings specifically: isolate JUST the TokEmb slice of the full
// gradient vector, the same style as the attention-only bisection above. TokEmb is the ONE tensor
// that receives TWO distinct backward contributions when USE_TIED_EMBEDDINGS (the embedding-lookup
// scatter-add AND the tied head's own dY^T.a term, accumulated via launch_tied_head_bwd -- see
// backward_device in backend_cuda.cu) -- every other tensor gets exactly one. The whole-vector test
// above ("CUDA backward matches the CPU reduced gradient") already covers this generically (it would
// catch a badly wrong contribution), but a bug where just ONE of the two contributions is dropped or
// double-counted could in principle be diluted by the L2 norm of the rest of the gradient; isolating
// TokEmb directly removes that risk. `if constexpr`-gated so it contributes ZERO assertions (and
// costs nothing) in the default untied CUDA test build -- only meaningful once a --tie-embeddings 1
// --compute 1 build actually exercises the tied path.
TEST_CASE("CUDA tied-embedding gradient (tok_emb) stays aligned with the CPU", "[cuda]") {
    if constexpr (USE_TIED_EMBEDDINGS) {
        CudaGuard _cuda_guard;
        sub0::build_model();
        sub0_cuda_set_tf32(0);
        REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

        const int batch = 4, T = 8;
        std::vector<int> data, ids, targets;
        std::vector<std::size_t> starts;
        make_windows(batch, T, 88, data, starts, ids, targets);

        sub0::train_batch(data.data(), starts.data(), batch, T);
        const std::size_t n = sub0::trainable_floats();
        const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

        std::vector<float> gpu_grad(n, 0.0f);
        double loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &loss) == 0);

        double num = 0.0, den = 0.0, maxabs = 0.0;
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
            if (p.kind != sub0::PKind::TokEmb) continue;
            for (std::size_t i = p.off; i < p.off + p.n(); ++i) {
                const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
                num += d * d;
                den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
                maxabs = std::max(maxabs, std::fabs(d));
            }
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        INFO("tok_emb grad rel-L2 = " << rel << "  max abs = " << maxabs << "  loss = " << loss);
        REQUIRE(std::isfinite(loss));
        if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(rel < 0.3);   // BF16: behaviour, not bit-parity
        else                                    REQUIRE(rel < 1e-2);
    }
}

// Bisection probe for QK-norm specifically, the same style as the tied-embedding bisection above:
// isolate JUST the QNorm/KNorm gamma slices of the full gradient vector. These are the newest tensors
// in the layout (inserted between Wo and the FFN block -- see layout.hpp) and the whole-vector test
// above ("CUDA backward matches the CPU reduced gradient") already covers them generically, but a bug
// specific to qknorm_backward_act_kernel's atomicAdd dgamma accumulation (e.g. one layer's gamma
// picking up another layer's contribution, or the in-place dy->dx overwrite corrupting a later read)
// could in principle be diluted by the L2 norm of the rest of the gradient; isolating these two
// per-layer [1,D_HEAD] slices directly removes that risk. `if constexpr`-gated so it contributes ZERO
// assertions (and costs nothing) in the default qk-norm-off CUDA test build -- only meaningful once a
// --qk-norm 1 --compute 1 build actually exercises the QK-norm path.
TEST_CASE("CUDA QK-norm gradient (q_norm/k_norm) stays aligned with the CPU", "[cuda]") {
    if constexpr (USE_QK_NORM) {
        CudaGuard _cuda_guard;
        sub0::build_model();
        sub0_cuda_set_tf32(0);
        REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

        const int batch = 4, T = 8;
        std::vector<int> data, ids, targets;
        std::vector<std::size_t> starts;
        make_windows(batch, T, 99, data, starts, ids, targets);

        sub0::train_batch(data.data(), starts.data(), batch, T);
        const std::size_t n = sub0::trainable_floats();
        const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

        std::vector<float> gpu_grad(n, 0.0f);
        double loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &loss) == 0);

        double num = 0.0, den = 0.0, maxabs = 0.0;
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
            if (p.kind != sub0::PKind::QNorm && p.kind != sub0::PKind::KNorm) continue;
            for (std::size_t i = p.off; i < p.off + p.n(); ++i) {
                const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
                num += d * d;
                den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
                maxabs = std::max(maxabs, std::fabs(d));
            }
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        INFO("q_norm/k_norm grad rel-L2 = " << rel << "  max abs = " << maxabs << "  loss = " << loss);
        REQUIRE(std::isfinite(loss));
        if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(rel < 0.3);   // BF16: behaviour, not bit-parity
        else                                    REQUIRE(rel < 1e-2);
    }
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

// gemm()'s beta=1 accumulate mode (and bias_grad_kernel's/launch_linear_bwd's accumulate flag riding
// on it) has no real call site yet -- added as groundwork for a future row-chunked GEMM (e.g.
// splitting the lm_head backward over M to shrink its dominant per-window scratch buffer). This is
// its ONLY coverage: a row-chunked dW/dbias computation must match a single full-M reference call,
// since the underlying math (dW = sum_m X[m]*dY[m], dbias = sum_m dY[m]) splits cleanly across any
// row boundary -- a real accumulate-mode bug would show up as a wrong scale/shape, not GEMM
// reassociation-level noise, so this bound is tight.
TEST_CASE("CUDA gemm/bias_grad/launch_linear_bwd accumulate mode matches a full-M reference", "[cuda]") {
    CudaGuard _cuda_guard;
    double reldiff = 1.0;
    const int rc = sub0_cuda_test_accumulate_check(96, 40, 56, &reldiff);
    WARN("accumulate check: max rel diff = " << reldiff);
    REQUIRE(rc == 0);
    REQUIRE(reldiff < 1e-3);
}

// Dims-independent GEMM check for the tied-embedding head's GPU forward+backward
// (launch_tied_head / launch_tied_head_bwd -- see backend_cuda.cu). Runs at TWO scales in this one
// binary (no rebuild needed, unlike the whole-model parity tests below, since C/V here are runtime
// params, not the baked D_MODEL/VOCAB): a toy scale and a production-like scale matching this
// project's actual production d448/vocab config, per the project's "test more than one scale"
// standing rule (a real precedent exists of a bug reproducing only at production dims). This is
// independent of whether the CURRENT build has USE_TIED_EMBEDDINGS on -- it exercises the raw GEMM
// primitive directly, so it is meaningful (and cheap) even in the default untied CUDA test build.
TEST_CASE("CUDA tied-head GEMM forward+backward matches a hand-computed reference", "[cuda]") {
    CudaGuard _cuda_guard;
    for (const auto& [M, C, V] : {std::tuple{4, 8, 16}, std::tuple{64, 448, 4306}}) {
        double fwd_reldiff = 1.0, bwd_reldiff = 1.0;
        const int rc = sub0_cuda_tied_head_check(M, C, V, &fwd_reldiff, &bwd_reldiff);
        WARN("tied-head check: M=" << M << " C=" << C << " V=" << V
             << " | fwd rel-L2 = " << fwd_reldiff << " | bwd rel-L2 = " << bwd_reldiff);
        REQUIRE(rc == 0);
        // launch_tied_head forces tensor cores and, on a BF16 build (GEMM_DTYPE == BF16), rounds inputs
        // to BF16 -- so the production-scale rel-L2 (deep V accumulation) sits at a few e-3, above the
        // tight FP32-era 1e-3 gate. Widen for BF16 (still far below the >>1e-1 a real GEMM regression
        // shows), the same regime-aware pattern as the gradient parity checks above.
        if constexpr (GEMM_DTYPE == Dtype::BF16) {
            REQUIRE(fwd_reldiff < 8e-3);
            REQUIRE(bwd_reldiff < 8e-3);
        } else {
            REQUIRE(fwd_reldiff < 1e-3);
            REQUIRE(bwd_reldiff < 1e-3);
        }
    }
}

// Dims-independent CPU-vs-GPU parity check for SwiGLU (swiglu_kernel/swiglu_act_kernel/
// swiglu_backward_act_kernel -- USE_GATED_FFN's FFN nonlinearity, see backend_cuda.cu). Runs at TWO
// scales in this one binary (M/F are runtime params, not baked dims): a toy shape and a production-
// like shape (M=64,F=1792 -- this project's actual production D_FF). Meaningful (and cheap) in ANY
// CUDA build regardless of whether USE_GATED_FFN is on -- it exercises the raw kernels directly, same
// reasoning as the tied-head check above being independent of USE_TIED_EMBEDDINGS. Internally exercises
// BOTH the non-aliased forward AND the in-place (y==up_pre) forward every real call site uses, plus
// BOTH the non-aliased backward AND the dup-aliased-onto-dy backward backward_device's role-remapped
// buffers rely on -- see sub0_cuda_swiglu_check's own comment.
TEST_CASE("CUDA SwiGLU kernels forward+backward match a hand-computed reference", "[cuda]") {
    CudaGuard _cuda_guard;
    for (const auto& [M, F] : {std::pair{4, 8}, std::pair{64, 1792}}) {
        double fwd_relL2 = 1.0, bwd_relL2 = 1.0;
        const int rc = sub0_cuda_swiglu_check(M, F, &fwd_relL2, &bwd_relL2);
        WARN("swiglu check: M=" << M << " F=" << F
             << " | fwd rel-L2 = " << fwd_relL2 << " | bwd rel-L2 = " << bwd_relL2);
        REQUIRE(rc == 0);
        REQUIRE(fwd_relL2 < 1e-5);
        REQUIRE(bwd_relL2 < 1e-5);
    }
}

// Bisection probe for gated-FFN specifically, the same style as the tied-embedding/QK-norm bisection
// probes above: isolate JUST the Wg/W1/W2 gradient slices of the full gradient vector (USE_GATED_FFN's
// three FFN weight tensors -- the ones backward_device's role-remapped ff1/gact/dff1/dgact buffer reuse
// (see that function's own comment) is riskiest for) against the CPU reference. The whole-vector test
// above ("CUDA backward matches the CPU reduced gradient") already covers this generically, but a bug
// specific to the buffer-role remapping (e.g. reading a stale gate_pre/up_pre after it was overwritten
// by the wrong step) could in principle be diluted by the L2 norm of the rest of the gradient;
// isolating these three per-layer weight tensors directly removes that risk. `if constexpr`-gated so it
// contributes ZERO assertions (and costs nothing) in the default plain-FFN CUDA test build -- only
// meaningful once a --gated-ffn 1 --compute 1 build actually exercises the gated path.
TEST_CASE("CUDA gated-FFN gradient (Wg/W1/W2) stays aligned with the CPU", "[cuda]") {
    if constexpr (USE_GATED_FFN) {
        CudaGuard _cuda_guard;
        sub0::build_model();
        sub0_cuda_set_tf32(0);
        REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

        const int batch = 4, T = 8;
        std::vector<int> data, ids, targets;
        std::vector<std::size_t> starts;
        make_windows(batch, T, 111, data, starts, ids, targets);

        sub0::train_batch(data.data(), starts.data(), batch, T);
        const std::size_t n = sub0::trainable_floats();
        const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

        std::vector<float> gpu_grad(n, 0.0f);
        double loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &loss) == 0);

        double num = 0.0, den = 0.0, maxabs = 0.0, dot = 0.0, gn = 0.0;
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
            const bool ffn = p.kind == sub0::PKind::Wg || p.kind == sub0::PKind::W1 ||
                             p.kind == sub0::PKind::W2;
            if (!ffn) continue;
            for (std::size_t i = p.off; i < p.off + p.n(); ++i) {
                const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
                num += d * d;
                den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
                dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i];
                gn  += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
                maxabs = std::max(maxabs, std::fabs(d));
            }
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
        INFO("gated-FFN grad rel-L2 = " << rel << "  cos = " << cos << "  max abs = " << maxabs << "  loss = " << loss);
        REQUIRE(std::isfinite(loss));
        if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(cos > 0.7);
        else                                    REQUIRE(rel < 1e-2);
    }
}

// Direct correctness proof for ce_backward_kernel's row_offset parameter (head_ce_chunked's chunked
// lm_head/tied-head/cross-entropy lever, "B1"): chunking a [M,V] logits/dlogits/loss computation over
// row-chunks must match a single full-M call exactly. Two scales (toy + production-like V), and at
// EACH scale chunk_cap is chosen to NOT evenly divide M, so the last chunk is deliberately short --
// stressing exactly the boundary risk (a wrong absolute row index past the first chunk would silently
// corrupt every subsequent row's window/position lookup, not fail loudly).
TEST_CASE("CUDA cross-entropy chunking (row_offset) matches a single full-M call", "[cuda]") {
    CudaGuard _cuda_guard;
    struct Case { int T, batch, chunk_cap; };
    for (const auto& c : {Case{8, 3, 7}, Case{256, 3, 300}}) {
        const int M = c.batch * c.T;
        double relL2_dlogits = 1.0, reldiff_loss = 1.0;
        const int rc = sub0_cuda_ce_chunk_check(M, c.T, c.batch, c.chunk_cap, &relL2_dlogits, &reldiff_loss);
        INFO("last chunk rows = " << (M % c.chunk_cap == 0 ? c.chunk_cap : M % c.chunk_cap));
        WARN("ce chunk check: M=" << M << " T=" << c.T << " batch=" << c.batch << " chunk_cap=" << c.chunk_cap
             << " | dlogits rel-L2 = " << relL2_dlogits << " | loss reldiff = " << reldiff_loss);
        REQUIRE(rc == 0);
        REQUIRE(relL2_dlogits < 1e-6);
        REQUIRE(reldiff_loss < 1e-6);
    }
}

// Dims-independent CPU-vs-GPU parity check for QK-norm's forward+backward kernels (qknorm_act_kernel /
// qknorm_save_act_kernel / qknorm_backward_act_kernel), the same convention as the tied-head GEMM check
// above: TWO shapes in this one binary (H/DH are template params on the kernels, so sub0_cuda_qknorm_check
// dispatches via an explicit shape_sel rather than taking H/DH at runtime -- see its own comment), a toy
// scale and a production-like scale (H=7,DH=64 -- this project's actual production d448/N_HEADS=7
// config), each run at two row counts. Meaningful (and cheap) in ANY CUDA build regardless of whether
// USE_QK_NORM is on -- it exercises the raw kernels directly, same reasoning as the tied-head check
// being independent of USE_TIED_EMBEDDINGS. The backward check internally pre-seeds dgamma with a
// random NONZERO pattern before calling the kernel, proving the atomicAdd ACCUMULATE semantics
// backward_device's per-layer q_norm/k_norm grad slots rely on (see qknorm_check_impl's own comment).
TEST_CASE("CUDA QK-norm kernels forward+backward match a hand-computed reference", "[cuda]") {
    CudaGuard _cuda_guard;
    struct Case { int shape_sel; int H, DH; int rows; };
    for (const auto& c : {Case{0, 2, 8, 5}, Case{0, 2, 8, 37}, Case{1, 7, 64, 3}, Case{1, 7, 64, 64}}) {
        double fwd_relL2 = 1.0, bwd_relL2 = 1.0;
        const int rc = sub0_cuda_qknorm_check(c.shape_sel, c.rows, &fwd_relL2, &bwd_relL2);
        WARN("qknorm check: H=" << c.H << " DH=" << c.DH << " rows=" << c.rows
             << " | fwd rel-L2 = " << fwd_relL2 << " | bwd rel-L2 = " << bwd_relL2);
        REQUIRE(rc == 0);
        REQUIRE(fwd_relL2 < 1e-5);   // includes the V-sub-block-untouched check (1e30 sentinel on failure)
        REQUIRE(bwd_relL2 < 1e-5);
    }
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

    // On a depth-attention build the decode graph does not implement the depth mix, so the ONLY correct
    // behaviour is refusal -- and that is what this case must check. Previously it ran the parity loop
    // regardless and PASSED at 98% top-1 while comparing a depth-mixed full forward against a non-mixed
    // decode: top-1 agreement is far too coarse an instrument to detect a missing V-rewrite. Asserting
    // the refusal is a real presence test; the old comparison was not.
    if constexpr (sub0::USE_DEPTH_ATTN) {
        REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);
        CHECK(sub0_cuda_kv_reset() != 0);
        std::vector<float> sink(static_cast<std::size_t>(VOCAB));
        CHECK(sub0_cuda_forward_one(0, 0, sink.data()) != 0);
        CHECK(sub0_dev_caps().supports_decode == 0);      // the bit and the seam must agree
        return;
    }

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
    REQUIRE(sub0_cuda_adam_step(0.001f, 1, 0.0f) == 0);   // muon_lr=0 -> pure AdamW, unchanged behavior
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

// GPU Newton-Schulz self-test: sub0_cuda_muon_ns_check runs the EXACT device pipeline
// device_adam_step's per-matrix Muon loop uses (backend_cuda.cu's muon_newton_schulz_device) in
// isolation, comparable directly against sub0::muon::newton_schulz5 (the CPU reference,
// include/sub0/muon.hpp) -- the single riskiest, most novel piece of this GPU port (Phase 1
// audit): the CPU's physical transpose-for-compute-efficiency step is replaced entirely by cuBLAS
// op-flag algebra on the GPU side, so this deliberately covers BOTH rows>cols and rows<cols at toy
// AND this BUILD's actual production shape (D_MODEL x D_FF and its transpose) -- same shape-list
// idea as muon_tests.cpp's own CPU test. Shapes are deliberately expressed relative to D_MODEL/D_FF
// rather than hardcoded absolute numbers: sub0_cuda_muon_ns_check reuses ensure_muon_scratch's
// buffers, which are sized to THIS build's own sub0::MUON_MAX_MN/MUON_MAX_MM (a compile-time
// constant derived from the CURRENT build's PARAM_LAYOUT, not a fixed number) -- a hardcoded
// {448,1792} would silently exceed a smaller test build's scratch and fail with rc!=0 (caught
// exactly this way when this test was first run against a d32 build). Using D_MODEL/D_FF means
// this test is production-shape-exact whenever compiled into a production-dims build, and still
// exercises "this build's own biggest matrix, both orientations" at any other scale. Two checks per
// shape: (1) rel-L2 against the CPU reference (WARNed, informational -- 5-step Newton-Schulz
// amplifies perturbations by design, and CPU-double vs GPU-float accumulation order legitimately
// differs, so this is not the hard gate); (2) the CPU-INDEPENDENT Gram-property check (off-diagonal
// collapse, bounded diagonal -- identical bounds to muon_tests.cpp's CPU test) as the TRUE
// correctness authority, per the Phase 1 precision recommendation ("treat the Gram-property check
// as the true correctness authority, not just the parity number").
TEST_CASE("CUDA Muon Newton-Schulz matches the CPU reference and the Gram property", "[cuda]") {
    CudaGuard _cuda_guard;
    struct { int rows, cols; } shapes[] = {{4, 4}, {4, 16}, {16, 4},
                                           {D_MODEL, D_MODEL}, {D_MODEL, D_FF}, {D_FF, D_MODEL}};
    for (auto [rows, cols] : shapes) {
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> in(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
        for (float& v : in) v = nd(rng);

        std::vector<float> cpu_out(in.size());
        sub0::muon::newton_schulz5(in.data(), rows, cols, cpu_out.data(), 5);

        std::vector<float> gpu_out(in.size());
        REQUIRE(sub0_cuda_muon_ns_check(in.data(), rows, cols, 0, gpu_out.data()) == 0);

        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < in.size(); ++i) {
            const double d = static_cast<double>(gpu_out[i]) - cpu_out[i];
            num += d * d;
            den += static_cast<double>(cpu_out[i]) * cpu_out[i];
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        WARN("muon NS check: rows=" << rows << " cols=" << cols << " | rel-L2 vs CPU reference = " << rel);

        // Independent re-derivation of the Gram-matrix property check (not reusing any GPU or CPU
        // newton_schulz5 code) -- same working-orientation convention as muon_tests.cpp's gram_stats.
        const int m = std::min(rows, cols), n = std::max(rows, cols);
        std::vector<float> Xw(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        if (rows > cols) {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    Xw[static_cast<std::size_t>(j) * n + i] = gpu_out[static_cast<std::size_t>(i) * cols + j];
        } else {
            Xw = gpu_out;
        }
        double max_offdiag = 0.0, min_diag = 1e30, max_diag = -1e30;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k)
                    s += static_cast<double>(Xw[static_cast<std::size_t>(i) * n + k]) *
                         static_cast<double>(Xw[static_cast<std::size_t>(j) * n + k]);
                if (i == j) { min_diag = std::min(min_diag, s); max_diag = std::max(max_diag, s); }
                else        max_offdiag = std::max(max_offdiag, std::fabs(s));
            }
        }
        INFO("shape [" << rows << "," << cols << "] max_offdiag=" << max_offdiag
             << " min_diag=" << min_diag << " max_diag=" << max_diag);
        CHECK(max_offdiag < 0.3);
        CHECK(min_diag > 0.3);
        CHECK(max_diag < 2.0);
    }
}

// TF32-leakage regression, the direct reproduction case for the #1 hazard this port's Phase 1
// audit flagged: run_fwd_bwd sets the SHARED cuBLAS handle to TF32 tensor-op math on EVERY training
// step (set_handle_tf32), so a Muon GEMM going through the ordinary gemm()/gemm_compute() path
// could silently inherit reduced-precision math from an unrelated forward-pass knob.
// gemm_muon's CUBLAS_COMPUTE_32F_PEDANTIC compute type exists specifically to be immune to this.
// Direct proof: run the SAME input through muon_newton_schulz_device with the handle's math mode
// forced to TF32 vs left at its default, and require the two outputs are nearly identical -- a
// tolerance far tighter than TF32's characteristic ~1e-3 relative error (so this test would FAIL
// LOUDLY if PEDANTIC ever stopped being honored), but not asserted bit-exact (cuBLAS's own
// algorithm-selection heuristic is permitted to legitimately vary its summation order between the
// two handle states even under an identical, non-reduced-precision compute type).
TEST_CASE("CUDA Muon Newton-Schulz is immune to the shared handle's TF32 math mode", "[cuda]") {
    CudaGuard _cuda_guard;
    // This build's own actual production shape (D_MODEL x D_FF) -- see the NS self-test's comment
    // above for why this can't be a hardcoded absolute shape (bounded by ensure_muon_scratch's
    // build-specific MUON_MAX_MN/MUON_MAX_MM capacity).
    const int rows = D_MODEL, cols = D_FF;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> in(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (float& v : in) v = nd(rng);

    std::vector<float> out_plain(in.size()), out_tf32(in.size());
    REQUIRE(sub0_cuda_muon_ns_check(in.data(), rows, cols, 0, out_plain.data()) == 0);
    REQUIRE(sub0_cuda_muon_ns_check(in.data(), rows, cols, 1, out_tf32.data()) == 0);

    double maxabsdiff = 0.0, maxrel = 0.0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out_plain[i]) - out_tf32[i]);
        maxabsdiff = std::max(maxabsdiff, d);
        maxrel = std::max(maxrel, d / std::max(std::fabs(static_cast<double>(out_plain[i])), 1e-6));
    }
    WARN("muon NS TF32-forced-on max abs diff = " << maxabsdiff << " | max rel diff = " << maxrel);
    REQUIRE(maxrel < 1e-5);   // >=1000x tighter than TF32's ~1e-3 characteristic error -- proves immunity
}

// Whole-step bisection probe for Muon specifically, the same style as the tied-embedding/QK-norm/
// gated-FFN bisection probes elsewhere in this file: start CPU and GPU from an IDENTICAL parameter
// state, take one hybrid Muon+AdamW step on each, and compare the resulting parameter DELTAS on
// the Muon-eligible slices separately from the non-Muon slices (must match plain-AdamW parity
// exactly, unaffected by this feature) and the momentum state itself. Runtime-flagged (NOT
// `if constexpr`-gated, unlike the gated-FFN/QK-norm/tied-embedding probes -- Muon eligibility is a
// runtime --optimizer choice, not a compile-time config), so this runs in EVERY CUDA build.
//
// Includes the two direct hazard-reproduction assertions this port's Phase 1 audit specifically
// flagged: (1) g_dev_vel's Muon-eligible slices must be EXACTLY 0.0f after the step -- the direct
// proof that adam_step_kernel's is_muon_param skip is working (a bug here would silently corrupt
// Muon's own momentum, which lives in the SAME m[] arena under a different update rule -- see
// device_adam_step's own comment, and the coordinator's explicit approval of rejecting the
// "zero the Muon grad slices" alternative for exactly this reason); (2) the Muon-eligible slices of
// g_dev_m (Muon's momentum) match the CPU's own g_param_m for the same slices, confirming the
// momentum EMA itself -- not just the final parameter delta -- is correct.
TEST_CASE("CUDA Muon step matches the CPU hybrid optimizer update", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 8;
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(batch, T, 77, data, starts, ids, targets);

    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> p0(sub0::params_ptr(), sub0::params_ptr() + n);

    // CPU: one hybrid Muon+AdamW step from zeroed moments.
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.0f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.0f);
    sub0::train_batch(data.data(), starts.data(), batch, T);
    sub0::AdamW opt(0.001f, /*use_muon=*/true);
    opt.set_muon_lr(0.02f);
    opt.step();
    const std::vector<float> p_cpu(sub0::params_ptr(), sub0::params_ptr() + n);
    const std::vector<float> m_cpu(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n);

    // GPU: backward (fills the device grad, zeroes moments) then one hybrid step at t=1, SAME
    // muon_lr as the CPU side above.
    std::vector<float> tmp(n);
    double loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, tmp.data(), &loss) == 0);
    REQUIRE(sub0_cuda_adam_step(0.001f, 1, 0.02f) == 0);
    std::vector<float> p_gpu(n), m_gpu(n), v_gpu(n);
    REQUIRE(sub0_cuda_download_params(p_gpu.data()) == 0);
    REQUIRE(sub0_cuda_download_opt(m_gpu.data(), v_gpu.data()) == 0);

    // Two independent metrics per group (Muon-slice / non-Muon-slice / momentum), same reasoning as
    // "CUDA AdamW step matches the CPU optimizer update" and the gated-FFN bisection probe above:
    // rel-L2 on the raw delta is the tight, direct gate under FP32 activations, but is NOT the
    // right metric under this build's BF16 activation/GEMM storage -- bf16's ~3-decimal-digit
    // mantissa means CPU-vs-GPU deltas on individual near-zero elements can differ by a large
    // RELATIVE amount while still pointing the same direction overall; cosine similarity (does the
    // GPU delta point the same way as the CPU delta, in aggregate) is this file's own established
    // BF16 gate for exactly this reason. Confirmed empirically, not assumed: an earlier version of
    // this test used rel-L2 unconditionally under BF16 too and passed at tiny (d32/L2) and
    // gated-FFN (d32/L2) scale by chance, then failed hard at production scale (d448/L11) -- non-
    // muon-slice rel-L2 = 0.52 -- once there were enough BF16-rounded elements for that measurement
    // artifact to dominate; cosine similarity is stable across all three scales (see this test's own
    // INFO line for the measured numbers).
    double muon_num = 0.0, muon_den = 0.0, muon_dot = 0.0, muon_gn = 0.0;
    double adam_num = 0.0, adam_den = 0.0, adam_dot = 0.0, adam_gn = 0.0;
    double m_num = 0.0, m_den = 0.0, m_dot = 0.0, m_gn = 0.0;
    float max_vel_muon = 0.0f;
    for (const sub0::ParamDesc& pd : sub0::PARAM_LAYOUT) {
        const bool muon = sub0::is_muon_kind(pd.kind);
        for (std::size_t i = pd.off; i < pd.off + pd.n(); ++i) {
            const double dc = static_cast<double>(p_cpu[i]) - p0[i];
            const double dg = static_cast<double>(p_gpu[i]) - p0[i];
            if (muon) {
                const double dm = dg - dc;
                muon_num += dm * dm; muon_den += dc * dc; muon_dot += dg * dc; muon_gn += dg * dg;
                const double mc = static_cast<double>(m_cpu[i]), mg = static_cast<double>(m_gpu[i]);
                const double dmm = mg - mc;
                m_num += dmm * dmm; m_den += mc * mc; m_dot += mg * mc; m_gn += mg * mg;
                max_vel_muon = std::max(max_vel_muon, std::fabs(v_gpu[i]));   // tripwire: must stay exactly 0
            } else {
                const double da = dg - dc;
                adam_num += da * da; adam_den += dc * dc; adam_dot += dg * dc; adam_gn += dg * dg;
            }
        }
    }
    const double muon_rel = std::sqrt(muon_num / std::max(muon_den, 1e-30));
    const double adam_rel = std::sqrt(adam_num / std::max(adam_den, 1e-30));
    const double m_rel    = std::sqrt(m_num / std::max(m_den, 1e-30));
    const double muon_cos = muon_dot / std::max(std::sqrt(muon_gn * muon_den), 1e-30);
    const double adam_cos = adam_dot / std::max(std::sqrt(adam_gn * adam_den), 1e-30);
    const double m_cos    = m_dot / std::max(std::sqrt(m_gn * m_den), 1e-30);
    INFO("muon-slice delta rel-L2 = " << muon_rel << " cos = " << muon_cos
         << " | non-muon-slice delta rel-L2 = " << adam_rel << " cos = " << adam_cos
         << " | muon momentum (m) rel-L2 = " << m_rel << " cos = " << m_cos
         << " | max |vel| on muon slices = " << max_vel_muon);
    REQUIRE(std::isfinite(loss));
    REQUIRE(max_vel_muon == 0.0f);        // direct hazard-reproduction: no double-update/momentum corruption
    if constexpr (ACT_DTYPE == Dtype::BF16) {
        REQUIRE(adam_cos > 0.7);          // same gate/threshold as the plain-AdamW BF16 precedent
        REQUIRE(m_cos > 0.7);
        REQUIRE(muon_cos > 0.5);          // looser than the AdamW bar: Newton-Schulz's quintic
                                           // amplification (Phase 1 audit) means small CPU/GPU float
                                           // differences compound over 5 iterations more than a plain
                                           // elementwise AdamW update does; tighten once more scales'
                                           // worth of measured numbers (this test's own INFO line) are in.
    } else {
        REQUIRE(adam_rel < 1e-2);         // non-Muon slices: unaffected, same bar as the plain-AdamW test
        REQUIRE(m_rel < 1e-2);            // Muon's own momentum EMA matches the CPU reference
        REQUIRE(muon_rel < 1e-1);         // looser than the AdamW bar, same reasoning as the BF16
                                           // muon_cos bound above.
    }
}

// Checkpoint/state round-trip, confirmed EMPIRICALLY (not just by tracing code -- per explicit
// Phase 3 review guidance: "cheap to verify directly, and checkpoint compatibility bugs are
// exactly the kind of thing that look obviously-fine on paper and aren't"). Mimics
// train_stage.cpp's actual save_checkpoint/load_checkpoint round-trip mechanism -- a raw memcpy of
// params_ptr()/adam_m_ptr()/adam_v_ptr() via sub0_cuda_download_params/download_opt on save, then
// upload_params/upload_opt on resume -- without depending on train_stage.cpp's own file-I/O
// internals (private to that translation unit): two hybrid Muon+AdamW steps run UNINTERRUPTED
// should land on very nearly the IDENTICAL final (params, m, vel) as the same two steps with a
// full device-state download+upload round trip (i.e. exactly what a save+resume does) injected
// between them.
//
// NOT asserted bit-exact, even though the round trip itself is a lossless memcpy: the two paths'
// SECOND backward() call is a genuinely SEPARATE kernel-launch sequence (not a replay), and CUDA's
// atomicAdd-based reductions (embed_backward_kernel's scatter-add, the grad-clip/Frobenius-norm
// block-reduces) are not guaranteed bit-stable in summation order across separate launches even
// for byte-identical inputs -- confirmed empirically here (measured ~1e-10-scale absolute drift on
// m/vel, exactly zero on params, first run), not assumed. The tolerance below is set far above
// that measured noise floor and far below the scale a genuine round-trip bug (a dropped field, a
// wrong buffer, byte-order corruption) would produce.
constexpr double CKPT_ROUNDTRIP_TOL = 1e-6;   // absolute; see comment above for the noise-floor evidence
TEST_CASE("CUDA Muon momentum state round-trips a save/resume cycle exactly", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> p0(sub0::params_ptr(), sub0::params_ptr() + n);
    const std::vector<float> zero(n, 0.0f);

    const int batch = 4, T = 8;
    std::vector<int> data1, ids1, targets1; std::vector<std::size_t> starts1;
    make_windows(batch, T, 21, data1, starts1, ids1, targets1);
    std::vector<int> data2, ids2, targets2; std::vector<std::size_t> starts2;
    make_windows(batch, T, 22, data2, starts2, ids2, targets2);
    std::vector<float> tmp(n);
    double loss = 0.0;

    // Reference: two steps, uninterrupted.
    REQUIRE(sub0_cuda_upload_params(p0.data()) == 0);
    REQUIRE(sub0_cuda_upload_opt(zero.data(), zero.data()) == 0);
    REQUIRE(sub0_cuda_backward(ids1.data(), targets1.data(), batch, T, tmp.data(), &loss) == 0);
    REQUIRE(sub0_cuda_adam_step(0.001f, 1, 0.02f) == 0);
    REQUIRE(sub0_cuda_backward(ids2.data(), targets2.data(), batch, T, tmp.data(), &loss) == 0);
    REQUIRE(sub0_cuda_adam_step(0.001f, 2, 0.02f) == 0);
    std::vector<float> p_ref(n), m_ref(n), v_ref(n);
    REQUIRE(sub0_cuda_download_params(p_ref.data()) == 0);
    REQUIRE(sub0_cuda_download_opt(m_ref.data(), v_ref.data()) == 0);

    // Round-tripped: step 1, download EVERYTHING (== save_checkpoint's param+m+vel write), re-upload
    // (== load_checkpoint's param+m+vel read, simulating a resumed process), step 2.
    REQUIRE(sub0_cuda_upload_params(p0.data()) == 0);
    REQUIRE(sub0_cuda_upload_opt(zero.data(), zero.data()) == 0);
    REQUIRE(sub0_cuda_backward(ids1.data(), targets1.data(), batch, T, tmp.data(), &loss) == 0);
    REQUIRE(sub0_cuda_adam_step(0.001f, 1, 0.02f) == 0);
    std::vector<float> p_mid(n), m_mid(n), v_mid(n);
    REQUIRE(sub0_cuda_download_params(p_mid.data()) == 0);
    REQUIRE(sub0_cuda_download_opt(m_mid.data(), v_mid.data()) == 0);
    REQUIRE(sub0_cuda_upload_params(p_mid.data()) == 0);
    REQUIRE(sub0_cuda_upload_opt(m_mid.data(), v_mid.data()) == 0);
    REQUIRE(sub0_cuda_backward(ids2.data(), targets2.data(), batch, T, tmp.data(), &loss) == 0);
    REQUIRE(sub0_cuda_adam_step(0.001f, 2, 0.02f) == 0);
    std::vector<float> p_rt(n), m_rt(n), v_rt(n);
    REQUIRE(sub0_cuda_download_params(p_rt.data()) == 0);
    REQUIRE(sub0_cuda_download_opt(m_rt.data(), v_rt.data()) == 0);

    double maxdiff_p = 0.0, maxdiff_m = 0.0, maxdiff_v = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        maxdiff_p = std::max(maxdiff_p, std::fabs(static_cast<double>(p_ref[i]) - p_rt[i]));
        maxdiff_m = std::max(maxdiff_m, std::fabs(static_cast<double>(m_ref[i]) - m_rt[i]));
        maxdiff_v = std::max(maxdiff_v, std::fabs(static_cast<double>(v_ref[i]) - v_rt[i]));
    }
    INFO("checkpoint round-trip max abs diff: params=" << maxdiff_p << " m=" << maxdiff_m << " vel=" << maxdiff_v
         << " (tolerance " << CKPT_ROUNDTRIP_TOL << " -- see this test's own comment for why this isn't 0.0)");
    REQUIRE(maxdiff_p < CKPT_ROUNDTRIP_TOL);
    REQUIRE(maxdiff_m < CKPT_ROUNDTRIP_TOL);
    REQUIRE(maxdiff_v < CKPT_ROUNDTRIP_TOL);
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
// `tied` must match USE_TIED_EMBEDDINGS -- param_floats()'s head term differs (ln_f alone vs
// ln_f+lm_head+lm_bias), so a tied build with this left false would fail the very next test.
// `qk_norm` must match USE_QK_NORM the same way (adds the q_norm/k_norm gamma floats). `gated` must
// match USE_GATED_FFN the same way (Wg replaces the b1/b2 bias floats).
static constexpr sub0::memplan::Dims kTestDims = sub0::current_build_dims();

// The footprint model must agree with the canonical parameter layout: param_floats() re-derives
// PARAM_FLOATS from dims (so the configurator can call it without layout.hpp). If they diverge,
// every footprint prediction is wrong by a constant -- catch it here, decoupled from the device.
TEST_CASE("memplan param_floats matches the canonical layout", "[cuda][memplan]") {
    REQUIRE(sub0::memplan::param_floats(kTestDims) == sub0::trainable_floats());
}

// Pins logits_n_chunks/logits_chunk_rows' derivation (the "B1" chunked lm_head/CE lever) against
// hand-computed expectations, independent of any specific build's baked VOCAB/D_FF, so a future change
// to the ceil(vocab/d_ff)-clamped-to-[1,8] formula is caught here rather than only showing up as a
// throughput/memory regression later.
TEST_CASE("memplan logits_n_chunks/logits_chunk_rows match their derivation", "[cuda][memplan]") {
    CHECK(sub0::memplan::logits_n_chunks(100, 256) == 1);      // vocab < d_ff -> no chunking needed
    CHECK(sub0::memplan::logits_n_chunks(256, 256) == 1);      // exactly equal -> ceil(1) = 1
    CHECK(sub0::memplan::logits_n_chunks(257, 256) == 2);      // just over -> ceil(257/256) = 2
    CHECK(sub0::memplan::logits_n_chunks(2048, 256) == 8);     // ceil(8) = 8, right at the clamp
    CHECK(sub0::memplan::logits_n_chunks(1000000, 256) == 8);  // far beyond the clamp -> stays 8
    CHECK(sub0::memplan::logits_n_chunks(16517, 1792) == 8);   // this project's real production dims

    CHECK(sub0::memplan::logits_chunk_rows(2048, 256, 1000) == 125);  // 8 chunks: ceil(1000/8) = 125
    CHECK(sub0::memplan::logits_chunk_rows(100, 256, 1000) == 1000);  // 1 chunk: the whole thing

    // General property, checked against THIS build's own baked VOCAB/D_FF: chunk_rows * n_chunks must
    // always cover total_rows (no row left unallocated), and chunk_rows must be no smaller than the
    // exact (unrounded) even split.
    const int n = sub0::memplan::logits_n_chunks(VOCAB, D_FF);
    const auto rows = sub0::memplan::logits_chunk_rows(VOCAB, D_FF, 12345);
    CHECK(rows * static_cast<sub0::memplan::u64>(n) >= 12345);
    CHECK(rows >= 12345 / static_cast<sub0::memplan::u64>(n));
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
        // Asymmetric band (see memplan.hpp's FOOTPRINT_*_TOLERANCE_MB): UNDER-prediction is the
        // dangerous, OOM-risk direction and stays tight; OVER-prediction is safe (reserve more than
        // used) and gets a wider band that still catches a gross buffer drift but tolerates the known,
        // documented steady d768 over-estimate. gap > 0 = over-predict (safe), < 0 = under-predict (risky).
        const double gap = predicted_mb - actual_mb;
        CHECK(gap > -sub0::memplan::FOOTPRINT_TOLERANCE_MB);             // never under-reserve beyond tolerance
        CHECK(gap <  sub0::memplan::FOOTPRINT_OVERPREDICT_TOLERANCE_MB); // over-reserve stays bounded
    }
}

// Regression test for a real predictor/allocator divergence. train_alloc used to branch on its own
// runtime chunk override while sub0_cuda_train_footprint always re-derived the count from memplan, so
// forcing a chunk count moved the ALLOCATION without moving the PREDICTION -- up to 1.35 GiB apart at the
// LoopSplit arms' shape, four times FOOTPRINT_OVERPREDICT_TOLERANCE_MB. It never fired because the only
// test that set the override was declared after the footprint test in this file, so the ordering hid it;
// under --order rand, sharding, or the first production use of the lever it would have surfaced as a
// mis-sized batch. Both now read Dims::logits_chunks, and this pins that they agree.
TEST_CASE("memplan tracks a FORCED logits chunk count (predictor/allocator agree)", "[cuda][memplan]") {
    CudaGuard _cuda_guard;
    constexpr int kBatch = 64;
    double base_pred = 0.0, base_act = 0.0;
    REQUIRE(sub0_cuda_train_footprint(kBatch, &base_pred, &base_act) == 0);
    REQUIRE(base_act > 0.0);

    // Force MORE chunks than the derivation would pick, so the logits buffer genuinely shrinks. Anything
    // above the derived count exercises the path where the old code diverged (the override was also not
    // clamped to LOGITS_MAX_CHUNKS, which is the case that diverged furthest).
    const int forced = sub0::memplan::logits_n_chunks(VOCAB, D_FF) * 4;
    sub0_cuda_set_logits_chunks(forced);
    double pred = 0.0, act = 0.0;
    const int rc = sub0_cuda_train_footprint(kBatch, &pred, &act);
    sub0_cuda_set_logits_chunks(0);        // restore BEFORE any assertion can abort the test
    REQUIRE(rc == 0);
    REQUIRE(act > 0.0);
    WARN("forced " << forced << " chunks: predicted " << pred << " MiB, measured " << act
         << " MiB (unforced: " << base_pred << " / " << base_act << ")");

    // The PREDICTION must follow the override down -- this is the assertion that fails on the old code,
    // where pred would have equalled base_pred exactly while act dropped.
    CHECK(pred < base_pred);
    CHECK(act <= base_act);
    // ...and it must still match reality, to the same asymmetric band the unforced prediction is held to.
    const double gap = pred - act;
    CHECK(gap > -sub0::memplan::FOOTPRINT_TOLERANCE_MB);
    CHECK(gap <  sub0::memplan::FOOTPRINT_OVERPREDICT_TOLERANCE_MB);
}

// memplan.hpp justifies a FIXED absolute footprint tolerance (rather than a percentage) by asserting in
// prose that the predicted-vs-measured gap is "bounded and near-batch-independent" -- per-cudaMalloc
// rounding over a fixed number of allocations plus WDDM noise. That claim is load-bearing: if the gap
// actually scaled with the batch, a tolerance calibrated at batch 64 would be meaningless at the 448 the
// production arms run, which is exactly the regime where an under-prediction turns into a failed reserve.
// Measure it instead of trusting the comment.
TEST_CASE("memplan: the prediction gap does not scale with batch", "[cuda][memplan]") {
    CudaGuard _cuda_guard;
    double p_lo = 0.0, a_lo = 0.0, p_hi = 0.0, a_hi = 0.0;
    REQUIRE(sub0_cuda_train_footprint(64, &p_lo, &a_lo) == 0);
    REQUIRE(sub0_cuda_train_footprint(256, &p_hi, &a_hi) == 0);
    REQUIRE(a_lo > 0.0);
    REQUIRE(a_hi > 0.0);
    if (GPU_VRAM_MB > 0 && p_hi > 0.8 * static_cast<double>(GPU_VRAM_MB)) {
        WARN("batch 256 predicted " << p_hi << " MiB is near the VRAM budget; skipping the scaling check");
        return;
    }
    const double gap_lo = p_lo - a_lo, gap_hi = p_hi - a_hi;
    WARN("gap at batch 64: " << gap_lo << " MiB | at batch 256: " << gap_hi << " MiB (4x the rows)");
    // The footprint itself grew ~4x; the gap must NOT. Allowing it to move by one whole tolerance band is
    // generous -- the point is to catch a gap that tracks the batch, which would land near 4x gap_lo.
    CHECK(std::fabs(gap_hi - gap_lo) < sub0::memplan::FOOTPRINT_TOLERANCE_MB);
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

// Proves the gradient-merge math for a SOURCE-ROUTED hybrid CPU/GPU training step (design: project
// memory hybrid-cpu-gpu-execution-design -- not yet wired into train_stage.cpp; this test exists to
// verify the merge algorithm in isolation BEFORE touching the live training loop). content-embed forces
// an entire step onto CPU today; the hybrid idea is to route ONLY the content-embed-needing windows to
// CPU while the rest of the SAME step's batch runs on GPU, then combine the two independently-computed
// gradients into one update. Both train_batch (CPU) and sub0_cuda_backward (GPU) are proven per-call
// MEAN-normalized over THEIR OWN batch size ("CUDA gradient scale is invariant to duplicating the batch"
// above, and backward_device's "invM = 1/M... makes the result equal the CPU train_batch grad" comment
// in backend_cuda.cu) -- so combining two sub-batch means into one full-batch-equivalent mean requires
// weighting each by its OWN share of the combined batch: combined = (n_cpu/n_total)*cpu_grad +
// (n_gpu/n_total)*gpu_grad. This test partitions ONE batch of windows in half, computes each half's
// gradient on its respective backend, combines them with that weighting, and checks the result against
// a REFERENCE: the plain CPU train_batch gradient over the SAME windows as ONE undivided batch.
TEST_CASE("CUDA+CPU hybrid split: weighted-merged gradient matches an undivided CPU batch", "[cuda]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    sub0_cuda_set_tf32(0);                                  // full FP32 for a tight parity gate
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int total = 8, cpu_n = 3, gpu_n = total - cpu_n, T = 8;   // uneven split -- exercises the weighting, not just a 50/50 average
    std::vector<int> data, ids, targets;
    std::vector<std::size_t> starts;
    make_windows(total, T, 91, data, starts, ids, targets);

    const std::size_t n = sub0::trainable_floats();

    // Reference: the WHOLE batch as one undivided CPU train_batch call.
    sub0::train_batch(data.data(), starts.data(), total, T);
    const std::vector<float> ref_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    // Split: windows [0,cpu_n) on the CPU, windows [cpu_n,total) on the GPU -- same underlying windows,
    // same weights, just partitioned (mirroring how a hybrid step would route by blend source).
    sub0::train_batch(data.data(), starts.data(), cpu_n, T);
    const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    std::vector<float> gpu_grad(n, 0.0f);
    double gpu_loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data() + static_cast<std::size_t>(cpu_n) * T,
                               targets.data() + static_cast<std::size_t>(cpu_n) * T,
                               gpu_n, T, gpu_grad.data(), &gpu_loss) == 0);

    std::vector<float> combined(n);
    const float w_cpu = static_cast<float>(cpu_n) / static_cast<float>(total);
    const float w_gpu = static_cast<float>(gpu_n) / static_cast<float>(total);
    for (std::size_t i = 0; i < n; ++i) combined[i] = w_cpu * cpu_grad[i] + w_gpu * gpu_grad[i];

    // Same rel-L2/cos convention as "CUDA backward matches the CPU reduced gradient" above -- cross-
    // backend fast-math differences accumulate through the reverse pass, so compare direction/magnitude,
    // not bits.
    double num = 0.0, den = 0.0, dot = 0.0, cn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(combined[i]) - ref_grad[i];
        num += d * d;
        den += static_cast<double>(ref_grad[i]) * ref_grad[i];
        dot += static_cast<double>(combined[i]) * ref_grad[i];
        cn  += static_cast<double>(combined[i]) * combined[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(cn * den), 1e-30);
    INFO("hybrid-merge grad rel-L2 = " << rel << "  cos = " << cos << "  gpu_loss = " << gpu_loss);
    if constexpr (ACT_DTYPE == Dtype::BF16) REQUIRE(cos > 0.7);
    else                                    REQUIRE(rel < 1e-2);
}

// ============================================================================
//  Binding-compose parity suite (docs/BACKENDS.md "Design: binding-compose on CUDA")
// ============================================================================
// The first caps flip: the device composes overridden embed rows from fragment tok_emb rows
// (param-free MeanPool/Hash/HRR arms of sub0::encode_slot). Four gates, per the design doc:
// per-arm forward differential, per-arm backward-scatter differential (both via
// sub0_cuda_binding_compose_check, which drives the PRODUCTION kernels through the real install
// path against the in-process encode_slot/encode_slot_bwd reference), inertness (no table
// installed = bit-identical logits), and an end-to-end train-step differential on a
// binding-heavy batch. supports_binding_compose flips to 1 only with all four green on hardware.

TEST_CASE("CUDA binding-compose kernels match encode_slot/encode_slot_bwd per encoder arm", "[cuda]") {
    CudaGuard _cuda_guard;
    if constexpr (POS_ENCODING != PosEncoding::Rope) {
        SUCCEED("binding-compose install is RoPE-path only; Absolute build skips (install rejects)");
        return;
    }
    struct Arm { int enc; const char* name; };
    for (const Arm arm : { Arm{ static_cast<int>(sub0::SlotEncoding::MeanPool), "MeanPool" },
                           Arm{ static_cast<int>(sub0::SlotEncoding::Hash),     "Hash" },
                           Arm{ static_cast<int>(sub0::SlotEncoding::HRR),      "HRR" } }) {
        double fa = 0, fr = 0, ar = 0, ba = 0, br = 0;
        REQUIRE(sub0_cuda_binding_compose_check(arm.enc, 1234u, &fa, &fr, &ar, &ba, &br) == 0);
        INFO(arm.name << ": fwd maxabs=" << fa << " maxrel=" << fr
             << "  fwd(act) maxrel=" << ar << "  bwd maxabs=" << ba << " maxrel=" << br);
        // FP32 compose math on both sides (same per-fragment accumulation order); differences are
        // powf/cosf/sinf ulps + FMA contraction, so the FP32-store paths gate tight. The act-store
        // path only differs by the final rounding: BF16 storage adds ~2^-8 relative, F32 nothing.
        CHECK(fr < 5e-4);
        CHECK(br < 5e-4);
        if constexpr (ACT_DTYPE == Dtype::BF16) CHECK(ar < 1e-2);
        else                                    CHECK(ar < 5e-4);
        // Not vacuous: the composed rows genuinely differ from the plain rows they replace, so a
        // kernel that ignored the override table entirely would fail the fwd gate by ~O(1), not ulps.
    }
    // Unsupported encodings must be REJECTED at install (return nonzero), never silently
    // mis-composed -- the learned-enc_w arms and Scalar have no device kernels.
    const int ovr[1] = { 0 };
    for (const int bad_enc : { static_cast<int>(sub0::SlotEncoding::CharEncoder),
                               static_cast<int>(sub0::SlotEncoding::ConvPool),
                               static_cast<int>(sub0::SlotEncoding::Scalar) }) {
        const int entry[3] = { 0, 1, bad_enc };
        const int frag[1]  = { 65 };
        CHECK(sub0_cuda_set_window_bindings(ovr, 1, entry, 1, frag, 1) != 0);
    }
    REQUIRE(sub0_cuda_set_window_bindings(nullptr, 0, nullptr, 0, nullptr, 0) == 0);  // leave cleared
}

TEST_CASE("CUDA binding-compose is inert when no table is installed (bit-identical logits)", "[cuda]") {
    CudaGuard _cuda_guard;
    if constexpr (POS_ENCODING != PosEncoding::Rope) {
        SUCCEED("binding-compose install is RoPE-path only; Absolute build skips");
        return;
    }
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int T = 16;
    std::vector<int> ids(static_cast<std::size_t>(T));
    std::mt19937 rng(58);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int& x : ids) x = tok(rng);

    const std::size_t n = static_cast<std::size_t>(T) * VOCAB;
    std::vector<float> before(n), during(n), after(n);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, before.data()) == 0);   // never-installed baseline

    // Install: position 3 composes as the MEAN of two other tokens' rows (differs from its own row
    // under random weights), through the SAME captured graph -- proving the override branch fires
    // inside a replay, not only on fresh launches.
    std::vector<int> ovr(static_cast<std::size_t>(T), -1);
    ovr[3] = 0;
    const int entry[3] = { 0, 2, static_cast<int>(sub0::SlotEncoding::MeanPool) };
    const int frag[2]  = { ids[5], ids[9] };
    REQUIRE(sub0_cuda_set_window_bindings(ovr.data(), T, entry, 1, frag, 2) == 0);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, during.data()) == 0);
    REQUIRE(std::memcmp(before.data(), during.data(), n * sizeof(float)) != 0);   // it DID compose

    // Clear: the per-step default state must reproduce the never-installed logits BIT-identically
    // (the kernels' no-table path is the plain lookup, untouched math -- the inertness contract).
    REQUIRE(sub0_cuda_set_window_bindings(nullptr, 0, nullptr, 0, nullptr, 0) == 0);
    REQUIRE(sub0_cuda_forward(ids.data(), 1, T, after.data()) == 0);
    REQUIRE(std::memcmp(before.data(), after.data(), n * sizeof(float)) == 0);
}

// End-to-end: a binding-heavy batch (scratch-slot tokens bound to per-window fragment lists,
// ~1/4 of positions composed) trained through the DEVICE step must produce the same reduced
// gradient as the CPU train_batch content-embed path (win_binds) -- the exact differential shape
// of "CUDA backward matches the CPU reduced gradient" above, with the override table computed
// host-side the way the hybrid router's follow-up will (walk ids, consult the binding views).
TEST_CASE("CUDA binding-compose train step matches the CPU content-embed gradient", "[cuda]") {
    CudaGuard _cuda_guard;
    if constexpr (POS_ENCODING != PosEncoding::Rope) {
        SUCCEED("binding-compose install is RoPE-path only; Absolute build skips");
        return;
    }
    static_assert(sub0::SCRATCH_SLOT_BASE + 3 <= VOCAB, "slot ids must be in-vocab");
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int batch = 4, T = 16, M = batch * T;
    const int nslots = 3;                                   // slots 0..2 bound per window
    for (const sub0::SlotEncoding enc : { sub0::SlotEncoding::MeanPool, sub0::SlotEncoding::Hash,
                                          sub0::SlotEncoding::HRR }) {
        const int enc_sel = static_cast<int>(enc);
        std::vector<int> data, ids, targets;
        std::vector<std::size_t> starts;
        make_windows(batch, T, 137u + static_cast<unsigned>(enc_sel), data, starts, ids, targets);

        // Per-window fragment tables (window-specific contents, lens 1..6 across (b, slot)) and
        // slot tokens at 4 of every window's 16 positions -> binding-heavy, multiple slots reused.
        std::mt19937 rng(211u + static_cast<unsigned>(enc_sel));
        std::uniform_int_distribution<int> byte_tok(0, 255);       // byte-range ids, always in-vocab
        std::vector<std::vector<std::vector<int>>> slots(static_cast<std::size_t>(batch));
        for (int b = 0; b < batch; ++b) {
            slots[static_cast<std::size_t>(b)].resize(sub0::SCRATCH_SLOT_COUNT);
            for (int s = 0; s < nslots; ++s) {
                const int len = 1 + (b + s) % 6;                   // covers len 1 .. 6
                auto& fr = slots[static_cast<std::size_t>(b)][static_cast<std::size_t>(s)];
                for (int p = 0; p < len; ++p) fr.push_back(byte_tok(rng));
            }
            for (int t = 1; t < T; t += 4)                          // positions 1,5,9,13
                data[static_cast<std::size_t>(b) * T + t] = sub0::scratch_slot_id((t / 4) % nslots);
        }
        ids.assign(data.begin(), data.begin() + M);                 // refresh views over mutated data
        targets.assign(data.begin() + 1, data.begin() + M + 1);

        // CPU reference: train_batch's per-window content-embed path.
        std::vector<sub0::ScratchBindings> binds(static_cast<std::size_t>(batch));
        std::vector<const sub0::ScratchBindings*> winb(static_cast<std::size_t>(batch));
        for (int b = 0; b < batch; ++b) {
            binds[static_cast<std::size_t>(b)].slots    = slots[static_cast<std::size_t>(b)];
            binds[static_cast<std::size_t>(b)].encoding = enc;
            winb[static_cast<std::size_t>(b)] = &binds[static_cast<std::size_t>(b)];
        }
        const float cpu_loss = sub0::train_batch(data.data(), starts.data(), batch, T,
                                                 nullptr, nullptr, winb.data());
        const std::size_t n = sub0::trainable_floats();
        const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

        // GPU: host-computed override table mirroring op_embed's dispatch (bound slot ids compose,
        // everything else -- including an unbound slot id -- stays a plain lookup), then the device
        // backward with the table installed.
        std::vector<int> ovr(static_cast<std::size_t>(M), -1), entries, frags;
        int ne = 0;
        for (int m = 0; m < M; ++m) {
            const int tokid = ids[static_cast<std::size_t>(m)];
            if (!sub0::is_scratch_slot(tokid)) continue;
            const auto& fr = slots[static_cast<std::size_t>(m / T)]
                                  [static_cast<std::size_t>(tokid - sub0::SCRATCH_SLOT_BASE)];
            if (fr.empty()) continue;
            entries.push_back(static_cast<int>(frags.size()));
            entries.push_back(static_cast<int>(fr.size()));
            entries.push_back(enc_sel);
            frags.insert(frags.end(), fr.begin(), fr.end());
            ovr[static_cast<std::size_t>(m)] = ne++;
        }
        REQUIRE(ne == batch * 4);                                   // every slot position overrides
        REQUIRE(sub0_cuda_set_window_bindings(ovr.data(), M, entries.data(), ne,
                                              frags.data(), static_cast<int>(frags.size())) == 0);
        std::vector<float> gpu_grad(n, 0.0f);
        double gpu_loss = 0.0;
        REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &gpu_loss) == 0);
        REQUIRE(sub0_cuda_set_window_bindings(nullptr, 0, nullptr, 0, nullptr, 0) == 0);  // per-step clear

        double num = 0.0, den = 0.0, dot = 0.0, gn = 0.0, maxabs = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
            num += d * d; den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
            dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i];
            gn  += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
            maxabs = std::max(maxabs, std::fabs(d));
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
        INFO("enc=" << static_cast<int>(enc) << "  grad rel-L2 = " << rel << "  cos = " << cos
             << "  max abs = " << maxabs << "  cpu_loss = " << cpu_loss << "  gpu_loss = " << gpu_loss);
        REQUIRE(std::isfinite(gpu_loss));
        CHECK(std::fabs(gpu_loss - static_cast<double>(cpu_loss)) < 1e-2);
        // Same convention as the reduced-gradient test above: F32 storage gates tight; BF16 storage
        // asserts direction (a wrong/missing compose adjoint would send the tok_emb slice sideways,
        // far below these bands -- the per-arm scatter differential above is the tight bwd gate).
        if constexpr (ACT_DTYPE == Dtype::BF16) CHECK(cos > 0.7);
        else                                    CHECK(rel < 1e-2);
    }
}

// train_stage.cpp's hybrid router (needs_cpu &= !supports_binding_compose, docs/BACKENDS.md's
// documented follow-up) now routes content-embed windows to the GPU sub-batch too, building its
// override table from doc_of(src.docs, starts[b]) + doc_bindings[doc] -- i.e. several windows drawn
// from the SAME document share one binding table, resolved by document rather than by window. The
// end-to-end test above never exercises that: each of its windows carries its own distinct,
// directly-assigned slot table with no doc_of lookup in the loop at all. This closes that specific
// gap: two documents, two windows per document, the SAME slot index bound to DIFFERENT content in
// each document, so a doc_of mix-up (e.g. resolving every window against document 0) would fail
// this test even though the per-window-only test above would stay green.
TEST_CASE("CUDA binding-compose: doc_of-resolved bindings across multiple windows per document match CPU",
          "[cuda]") {
    CudaGuard _cuda_guard;
    if constexpr (POS_ENCODING != PosEncoding::Rope) {
        SUCCEED("binding-compose install is RoPE-path only; Absolute build skips (install rejects)");
        return;
    }
    static_assert(sub0::SCRATCH_SLOT_BASE + 2 <= VOCAB, "slot ids must be in-vocab");
    sub0::build_model();
    sub0_cuda_set_tf32(0);
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const int T = 12, batch = 4, M = batch * T;
    const std::vector<std::uint64_t> docs = { 0, 60, 120 };       // 2 documents, 60 tokens each
    const std::vector<std::size_t> starts = { 5, 25, 65, 90 };    // windows 0,1 -> doc0; 2,3 -> doc1

    std::mt19937 rng(4242u);
    // Filler must exclude the scratch-slot ids. They are LOW ids (SCRATCH_SLOT_BASE is
    // casing::TOK_RESERVED_4, a marker id), so a plain uniform draw over the whole vocab lands on one
    // roughly SCRATCH_SLOT_COUNT/VOCAB of the time -- about a 20-30% chance of at least one collision
    // across these 48 positions. A collision creates an unintended binding and breaks the
    // "exactly one bound position per window" invariant the test asserts below, so the test's
    // correctness silently depended on the seed and on VOCAB. It surfaced when a rebuild moved VOCAB.
    std::uniform_int_distribution<int> tok(0, sub0::SCRATCH_SLOT_BASE - 1);
    std::vector<int> data(121);
    for (int& x : data) x = tok(rng);
    // One bound position per window (t=3); the SAME slot index (b%2) alternates within each doc, so
    // a correct doc_of resolution must pull DIFFERENT fragment content for window 0 vs window 2 even
    // though both bind slot 0.
    for (int b = 0; b < batch; ++b)
        data[starts[static_cast<std::size_t>(b)] + 3] = sub0::scratch_slot_id(b % 2);

    std::uniform_int_distribution<int> byte_tok(0, 255);
    std::vector<std::vector<std::vector<int>>> doc_bindings(2);
    for (auto& doc : doc_bindings) {
        doc.resize(sub0::SCRATCH_SLOT_COUNT);
        for (int s = 0; s < 2; ++s) {
            const int len = 1 + s;
            for (int p = 0; p < len; ++p) doc[static_cast<std::size_t>(s)].push_back(byte_tok(rng));
        }
    }

    std::vector<int> ids(static_cast<std::size_t>(M)), targets(static_cast<std::size_t>(M));
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < T; ++t) {
            const std::size_t src = starts[static_cast<std::size_t>(b)] + static_cast<std::size_t>(t);
            ids[static_cast<std::size_t>(b) * T + t]     = data[src];
            targets[static_cast<std::size_t>(b) * T + t] = data[src + 1];
        }

    // CPU reference: train_batch's per-window content-embed path, bindings resolved via doc_of --
    // exactly materialize_cpu_window's own construction (train_stage.cpp).
    std::vector<sub0::ScratchBindings> cpu_binds(static_cast<std::size_t>(batch));
    std::vector<const sub0::ScratchBindings*> winb(static_cast<std::size_t>(batch));
    for (int b = 0; b < batch; ++b) {
        const std::size_t doc = sub0::doc_of(std::span<const std::uint64_t>(docs),
                                             starts[static_cast<std::size_t>(b)]);
        cpu_binds[static_cast<std::size_t>(b)] = sub0::ScratchBindings{
            std::span<const std::vector<int>>(doc_bindings[doc]), sub0::SlotEncoding::MeanPool };
        winb[static_cast<std::size_t>(b)] = &cpu_binds[static_cast<std::size_t>(b)];
    }
    const float cpu_loss = sub0::train_batch(data.data(), starts.data(), batch, T,
                                             nullptr, nullptr, winb.data());
    const std::size_t n = sub0::trainable_floats();
    const std::vector<float> cpu_grad(sub0::grad_ptr(), sub0::grad_ptr() + n);

    // GPU: host-computed override table via the SAME doc_of + ScratchBindings::bound/fragments calls
    // install_gpu_bindings (train_stage.cpp) makes, flattened into the device wire format.
    std::vector<int> ovr(static_cast<std::size_t>(M), -1), entries, frags;
    for (int b = 0; b < batch; ++b) {
        const std::size_t doc = sub0::doc_of(std::span<const std::uint64_t>(docs),
                                             starts[static_cast<std::size_t>(b)]);
        const sub0::ScratchBindings sb{
            std::span<const std::vector<int>>(doc_bindings[doc]), sub0::SlotEncoding::MeanPool };
        for (int t = 0; t < T; ++t) {
            const int tokid = ids[static_cast<std::size_t>(b) * T + t];
            if (!sb.bound(tokid)) continue;
            const auto fr = sb.fragments(tokid);
            ovr[static_cast<std::size_t>(b) * T + t] = static_cast<int>(entries.size() / 3);
            entries.push_back(static_cast<int>(frags.size()));
            entries.push_back(static_cast<int>(fr.size()));
            entries.push_back(static_cast<int>(sub0::SlotEncoding::MeanPool));
            frags.insert(frags.end(), fr.begin(), fr.end());
        }
    }
    REQUIRE(static_cast<int>(entries.size() / 3) == batch);   // exactly one bound position per window
    REQUIRE(sub0_cuda_set_window_bindings(ovr.data(), M, entries.data(),
                                          static_cast<int>(entries.size() / 3),
                                          frags.data(), static_cast<int>(frags.size())) == 0);
    std::vector<float> gpu_grad(n, 0.0f);
    double gpu_loss = 0.0;
    REQUIRE(sub0_cuda_backward(ids.data(), targets.data(), batch, T, gpu_grad.data(), &gpu_loss) == 0);
    REQUIRE(sub0_cuda_set_window_bindings(nullptr, 0, nullptr, 0, nullptr, 0) == 0);   // per-step clear

    double num = 0.0, den = 0.0, dot = 0.0, gn = 0.0, maxabs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(gpu_grad[i]) - cpu_grad[i];
        num += d * d; den += static_cast<double>(cpu_grad[i]) * cpu_grad[i];
        dot += static_cast<double>(gpu_grad[i]) * cpu_grad[i];
        gn  += static_cast<double>(gpu_grad[i]) * gpu_grad[i];
        maxabs = std::max(maxabs, std::fabs(d));
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double cos = dot / std::max(std::sqrt(gn * den), 1e-30);
    INFO("doc_of hybrid-router grad rel-L2 = " << rel << "  cos = " << cos << "  max abs = " << maxabs
         << "  cpu_loss = " << cpu_loss << "  gpu_loss = " << gpu_loss);
    REQUIRE(std::isfinite(gpu_loss));
    CHECK(std::fabs(gpu_loss - static_cast<double>(cpu_loss)) < 1e-2);
    if constexpr (ACT_DTYPE == Dtype::BF16) CHECK(cos > 0.7);
    else                                    CHECK(rel < 1e-2);
}

// Per-phase profile: attribute the step time to forward / backward / adam. The backward RECOMPUTES
// the checkpointed activations (a memory-for-compute trade) so it is expected to be heavy; this
// surfaces the split so further throughput rework (optional checkpointing) is data-driven. The
// grad-clip scale is computed on-device (grad_clip_scale_kernel, 2026-07) instead of a host
// round-trip, so "adam" no longer carries that bubble -- this profile is how the ~10% drop in the
// adam phase's elapsed time was confirmed.
// LOGITS ROW-CHUNKING IS RESULT-NEUTRAL. head_ce_chunked processes the lm_head -> cross-entropy ->
// head-backward chain in row-chunks and its comment claims the result is "mathematically IDENTICAL to the
// old unchunked path". That was asserted, never tested -- and it is load-bearing: the chunk count is the
// largest single VRAM lever available (at the LoopSplit arms' shape the logits buffer is 1806 MiB at the
// default 8 chunks, 452 MiB at 32), so it will be varied per build. A chunking bug would surface as a
// slightly wrong gradient, which no loss curve would reveal.
//
// Drives the SAME backward at several chunk counts and compares the reduced gradient against the
// unchunked (1-chunk) reference. Exact equality is not the bar -- chunking changes float SUMMATION ORDER
// for the dW/dtok_emb accumulations, so the tolerance is reduction noise, not algorithmic slack.
TEST_CASE("logits row-chunking does not change the gradient", "[cuda][chunk]") {
    CudaGuard _cuda_guard;
    sub0::build_model();
    const int batch = 4, T = std::min(64, SEQ_LEN);
    std::mt19937 rng(4242);
    std::vector<int> ids(static_cast<size_t>(batch) * T), tgt(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int>(rng() % VOCAB);
        tgt[i] = static_cast<int>(rng() % VOCAB);
    }
    REQUIRE(sub0_cuda_upload_params(sub0::params_ptr()) == 0);

    const std::size_t n = sub0::trainable_floats();
    auto grad_at = [&](int chunks, double* out_loss) {
        sub0_cuda_set_logits_chunks(chunks);
        REQUIRE(sub0_cuda_train_reserve(batch) == 0);
        std::vector<float> g(n, 0.f);
        REQUIRE(sub0_cuda_backward(ids.data(), tgt.data(), batch, T, g.data(), out_loss) == 0);
        return g;
    };

    double loss_ref = 0.0;
    const std::vector<float> ref = grad_at(1, &loss_ref);   // unchunked reference
    REQUIRE(std::isfinite(loss_ref));

    // 7 and 13 do NOT divide M (= batch*T), so they exercise head_ce_chunked's RAGGED last chunk
    // (rows = min(chunk_cap, M - m0)) and its row_offset arithmetic -- the path where an off-by-one
    // would corrupt only the final partial chunk and stay invisible in an evenly-dividing test.
    for (int chunks : {2, 4, 7, 8, 13, 32}) {
        double loss = 0.0;
        const std::vector<float> g = grad_at(chunks, &loss);
        // The loss is a plain sum over rows, so it should agree very tightly.
        INFO("chunks=" << chunks << " loss=" << loss << " ref=" << loss_ref);
        REQUIRE(loss == Catch::Approx(loss_ref).epsilon(1e-6));
        double dot = 0.0, na = 0.0, nb = 0.0, worst = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            dot += static_cast<double>(ref[i]) * g[i];
            na  += static_cast<double>(ref[i]) * ref[i];
            nb  += static_cast<double>(g[i])   * g[i];
            worst = std::max(worst, std::fabs(static_cast<double>(g[i]) - ref[i]));
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        INFO("chunks=" << chunks << " cos=" << cos << " worst abs delta=" << worst);
        REQUIRE(cos > 0.9999);
    }
    sub0_cuda_set_logits_chunks(0);   // restore memplan's derivation for every later test
}

// Properties the CHUNKING DERIVATION must hold regardless of shape. Pure memplan, no device work.
//
// These exist because the current derivation is a CHUNK COUNT (ceil(vocab/d_ff), clamped), and a count
// makes the resulting buffer scale with total_rows -- i.e. with BATCH. That is the wrong invariant for a
// memory cap: doubling the batch doubles the logits buffer at the same chunk count, which is exactly what
// the lever is supposed to prevent. The generalized target is a chunk SIZE (equivalently, a byte budget:
// chunk_rows * vocab * 4 <= target), which is batch-invariant. Until that lands, these tests pin what the
// count-based form does and does not guarantee, so the gap is visible rather than assumed away.
TEST_CASE("logits chunk derivation: coverage, bounds, and the batch-scaling gap", "[cuda][chunk]") {
    using sub0::memplan::logits_n_chunks;
    using sub0::memplan::logits_chunk_rows;

    // COVERAGE: the chunks must span every row, including when the count does not divide them. A
    // shortfall here would silently drop the tail rows from the loss and its gradient.
    for (unsigned long long rows : {1ull, 7ull, 255ull, 256ull, 229376ull}) {
        for (int vocab : {512, 16508, 49152}) {
            for (int dff : {512, 1792}) {
                const int k = logits_n_chunks(vocab, dff);
                const auto cr = logits_chunk_rows(vocab, dff, rows);
                REQUIRE(k >= 1);
                REQUIRE(cr >= 1);                              // never a zero-row chunk
                REQUIRE(static_cast<unsigned long long>(k) * cr >= rows);   // spans every row
                REQUIRE(cr <= rows);                           // never over-allocates past the budget
            }
        }
    }

    // BOUNDS: honours the configured cap, and degenerates to a single chunk when vocab is small
    // relative to d_ff (zero overhead, exactly the unchunked path).
    REQUIRE(logits_n_chunks(16508, 512) == std::min(33, sub0::memplan::LOGITS_MAX_CHUNKS));
    REQUIRE(logits_n_chunks(256, 1792) == 1);
    REQUIRE(logits_chunk_rows(256, 1792, 1000) == 1000);       // one chunk covers everything

    // THE GAP, pinned rather than hidden: the buffer scales with batch at a fixed chunk count, so the
    // count-based form does NOT bound the buffer's size. Doubling the rows doubles the chunk. When the
    // derivation moves to a chunk SIZE / byte budget this assertion should INVERT -- which is the point
    // of writing it down.
    const auto at_1x = logits_chunk_rows(16508, 512, 229376);
    const auto at_2x = logits_chunk_rows(16508, 512, 2 * 229376ull);
    REQUIRE(at_2x == 2 * at_1x);
}

// The caps struct exists so consumers degrade around a MISSING capability instead of being driven into
// an unimplemented path. supports_decode became conditional when depth attention landed: training and
// batched inference/eval support it, the single-token decode path does not. This pins both halves --
// that the bit tells the truth, and that the decode consumer actually reads it.
//
// Regression test for a real defect, not a hypothetical: gpu_decode_try_enable() gated only on HAS_CUDA
// plus init success, so on a depth-attention build `sub0llm report` would have driven its sample battery
// straight into forward_one_device's refusal -- mid-run, at the first eval, not at an explicit `gen`.
TEST_CASE("CUDA caps: supports_decode is honest, and the decode consumer honours it", "[cuda]") {
    REQUIRE(sub0_dev_caps().supports_decode == (DEPTH_ATTN_STRIDE == 0 ? 1 : 0));
    if constexpr (sub0::USE_DEPTH_ATTN) {
        // Must return false WITHOUT touching the device -- the point is that no decode call is made.
        REQUIRE(sub0::gpu_decode_try_enable() == false);
    }
    // Whatever decode reports, the paths depth attention DOES implement stay advertised.
    REQUIRE(sub0_dev_caps().supports_train == 1);
    REQUIRE(sub0_dev_caps().supports_eval  == 1);
}

// The measurement entry points REJECT an over-long T rather than clamping it, because silently
// profiling a different shape than the caller asked for would make every number they report
// untrustworthy. That puts the burden on callers to size their request to the build -- and both
// callers got it wrong: the [.bench] cases below hardcoded T=128/64, and tools/cuda_selftest.cpp
// hardcoded the fineweb T=256 while DISCARDING every return code, so on a SEQ_LEN=64 build six of
// its eight checks no-oped and it still exited 0.
//
// This pins both halves of the contract: the bound is queryable, and exceeding it is refused. If a
// future change makes these clamp instead, the second REQUIRE fails and the callers' std::min goes
// from necessary to merely harmless -- a deliberate decision rather than a silent drift.
TEST_CASE("CUDA measurement entry points refuse T > SEQ_LEN, and the bound is queryable", "[cuda]") {
    CudaGuard _cuda_guard;                       // the accepted call below allocates fwd/train/opt state
    REQUIRE(sub0_cuda_seq_len() == SEQ_LEN);
    double f = 0.0, b = 0.0, a = 0.0;
    CHECK(sub0_cuda_train_profile(4, SEQ_LEN + 1, 1, &f, &b, &a) != 0);
    CHECK(sub0_cuda_train_benchmark(4, SEQ_LEN + 1, 1, nullptr) != 0);
    CHECK(sub0_cuda_attn_check(4, SEQ_LEN + 1, 1, nullptr, nullptr) != 0);
    // ...and the very same call at the queried bound is accepted, so the guard rejects only what it
    // must. Cheap at these dims; this is the shape a correctly-written caller produces.
    CHECK(sub0_cuda_train_profile(4, sub0_cuda_seq_len(), 1, &f, &b, &a) == 0);
}

TEST_CASE("CUDA per-phase profile attributes the step (forward/backward/adam)", "[cuda][.bench]") {
    CudaGuard _cuda_guard;
    double f = 0.0, b = 0.0, a = 0.0;
    // T clamped to the build: train_profile REJECTS T > SEQ_LEN rather than clamping, so a hardcoded
    // 128 made this REQUIRE fail on any shorter build. It went unnoticed because [.bench] is hidden
    // unless selected by name.
    const int T = std::min(128, SEQ_LEN);
    REQUIRE(sub0_cuda_train_profile(32, T, 10, &f, &b, &a) == 0);
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
    const int batch = 16, T = std::min(64, SEQ_LEN);            // small -> fast, but enough to clock-ramp
    double cold = 0.0, warm = 0.0;
    REQUIRE(sub0_cuda_train_benchmark(batch, T, 2,  &cold) == 0);
    REQUIRE(sub0_cuda_train_benchmark(batch, T, 40, &warm) == 0);
    WARN("step ms: cold(2 iters)=" << cold << "  warm(40 iters)=" << warm
         << "  (sustained is faster once the clock + cuBLAS warm)");
    REQUIRE(std::isfinite(cold));
    REQUIRE(std::isfinite(warm));
    CHECK(warm <= cold * 1.5);                                  // warming must not make it slower
}

