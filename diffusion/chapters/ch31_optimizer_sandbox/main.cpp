// Chapter 31 — The Optimizer Sandbox: why diffusion training is optimizer-bound
//
// Ch29's grad-probe revealed the shape of the problem: the diffusion gradient is an average
// over near-ORTHOGONAL noise levels (easy↔hard cosine ≈ 0.1) with low WITHIN-level consistency
// (≈ 0.3 raw, ≈ 0.5 after exact-count masking). And the headline recall tracked that consistency
// number almost 1:1 (0.31→0.51 ⇒ ~15%→19%). That points the finger at the optimizer: the step is
// a noisy average of conflicting signals, and how an optimizer DIGESTS that noise sets the ceiling.
//
// This is a SANDBOX, not a model. It strips the LLM away and keeps only the gradient pathology in
// a setting where we know the exact optimum, so we can run the *production* Adam / AdamW / Muon
// (the very code ch29 ships) on a controlled gradient and answer concrete questions FAST:
//
//   Q1. As within-level consistency drops, which optimizer degrades, and how fast?
//   Q2. WHY does Muon keep failing on this task (ch29: full-coverage NELBO 5.68, word-START 0.2%)
//       even though it's designed to converge faster/lower on transformer matrices?
//   Q3. Does shared-t (one noise level per step) actually help, and does it help Muon specifically?
//
// The synthetic problem (faithful to the probe, trivial to reason about):
//   • Parameter W ∈ R^{m×n} — a MATRIX, so Muon orthogonalizes it (its real operating regime).
//   • Target W*. The full objective is L(W) = ½‖W−W*‖²_F, optimum at W=W*, residual we can read off.
//   • T "noise levels": level t owns a DISJOINT block of columns. Its gradient is (W−W*) restricted
//     to those columns ⇒ per-level gradients are EXACTLY orthogonal across levels (the probe's ≈0
//     cross-level cosine), and they SUM to the true gradient (so covering all t over time is unbiased).
//   • Within-level noise σ·R added over the level's columns sets the consistency knob:
//     consistency ≈ |signal| / sqrt(|signal|² + σ²·d).  Swept to the empirical {1.0,0.9,0.7,0.5,0.3}.
//   • shared-t: one level per step (signal sharp, noise/√B).  independent-t: B workers, own levels
//     each (diluted across orthogonal blocks) — exactly ch29's --shared-t vs the per-worker baseline.
//
// The hypothesis under test (Q2): Muon's Newton-Schulz drives ALL singular values of the momentum
// matrix to ≈1. When consistency is high the momentum's large singular directions are signal and
// the small ones are negligible; orthogonalizing is a well-conditioned win. When consistency is low
// the small singular directions ARE noise — and Muon AMPLIFIES them to full step magnitude, spending
// as much step on noise as on signal. So Muon should degrade FASTER than Adam as consistency falls,
// and only win in the clean / high-consistency regime.

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include <cmath>
#include <cstdint>
#include <print>
#include <random>
#include <string>
#include <vector>

using namespace sub0llm;

namespace {

constexpr std::int64_t M = 64;     // param rows
constexpr std::int64_t N = 64;     // param cols
constexpr std::int64_t T = 8;      // noise levels (disjoint column blocks)
constexpr std::int64_t BLK = N / T;// columns per level
constexpr int          B   = 4;    // samples averaged per step (the data-parallel width)
constexpr int          STEPS = 1500;

// Flat row-major index into an M×N matrix.
inline std::size_t idx(std::int64_t r, std::int64_t c) { return static_cast<std::size_t>(r * N + c); }

// Frobenius distance between two flat M×N buffers.
double residual(const std::vector<float>& a, const float* b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

// Write the per-step synthetic gradient into `g` (M*N, row-major), given the current weights `w`.
//   shared_t : one level for all B samples (noise averaged → σ/√B over that one block).
//   else     : B samples, each its OWN random level (diluted across orthogonal blocks).
// Returns nothing; `g` is fully overwritten each step (no accumulation — the optimizer holds state).
void make_grad(std::vector<float>& g, const float* w, const float* wstar,
               float sigma, bool shared_t, std::mt19937& rng) {
    std::fill(g.begin(), g.end(), 0.0f);
    std::uniform_int_distribution<std::int64_t> pick_level(0, T - 1);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    auto add_level = [&](std::int64_t t, float gscale, float nscale) {
        const std::int64_t c0 = t * BLK;
        for (std::int64_t r = 0; r < M; ++r)
            for (std::int64_t c = c0; c < c0 + BLK; ++c) {
                const std::size_t k = idx(r, c);
                g[k] += gscale * (w[k] - wstar[k]) + nscale * noise(rng);
            }
    };

    if (shared_t) {
        const std::int64_t t = pick_level(rng);
        // One block: full signal, noise averaged over B independent draws ⇒ σ/√B.
        add_level(t, 1.0f, sigma / std::sqrt(static_cast<float>(B)));
    } else {
        // B workers, independent levels: average the per-worker gradients (1/B each). Different
        // blocks rarely coincide, so each block gets ~1/B of the signal — the dilution we measured.
        for (int b = 0; b < B; ++b)
            add_level(pick_level(rng), 1.0f / static_cast<float>(B), sigma / static_cast<float>(B));
    }
}

// Within-level consistency at the given weights: mean over levels of E[cos(single-sample grad,
// mean grad)] at that level. Mirrors ch29 --grad-probe's consistency metric exactly.
double measure_consistency(const float* w, const float* wstar, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.0f, 1.0f);
    constexpr int K = 64;
    double acc = 0;
    for (std::int64_t t = 0; t < T; ++t) {
        const std::int64_t c0 = t * BLK;
        // Signal = mean grad at this level (noise is zero-mean) = (w−w*) on the block.
        double sig_norm2 = 0;
        for (std::int64_t r = 0; r < M; ++r)
            for (std::int64_t c = c0; c < c0 + BLK; ++c) {
                const double s = w[idx(r, c)] - wstar[idx(r, c)];
                sig_norm2 += s * s;
            }
        double cos_acc = 0;
        for (int k = 0; k < K; ++k) {
            double dot = 0, gn2 = 0;
            for (std::int64_t r = 0; r < M; ++r)
                for (std::int64_t c = c0; c < c0 + BLK; ++c) {
                    const double s = w[idx(r, c)] - wstar[idx(r, c)];
                    const double gi = s + sigma * noise(rng);
                    dot += gi * s; gn2 += gi * gi;
                }
            cos_acc += dot / (std::sqrt(gn2) * std::sqrt(sig_norm2) + 1e-30);
        }
        acc += cos_acc / K;
    }
    return acc / static_cast<double>(T);
}

struct RunResult { double final_res; double cons_start; double cons_end; float lr; };

// One training run: fresh optimizer + fresh W0 (same seed across optimizers so the gradient-noise
// sequence is IDENTICAL — differences are purely the optimizer). Returns the relative residual.
RunResult run(const std::string& opt_name, float lr, float sigma, bool shared_t,
              const std::vector<float>& w0, const std::vector<float>& wstar) {
    Tensor wt({M, N}, DType::Float32);
    std::copy(w0.begin(), w0.end(), wt.data_as<float>().begin());
    autograd::Variable W(std::move(wt), /*requires_grad=*/true);
    W.grad() = zeros({M, N});                       // optimizer reads p->grad() in place

    std::vector<autograd::Variable*> params{&W};
    auto opt = nn::make_optimizer(opt_name, params, lr);

    float*       wp = W.data().data_as<float>().data();
    float*       gp = W.grad().data_as<float>().data();
    std::vector<float> gbuf(static_cast<std::size_t>(M * N));

    std::mt19937 rng(12345u);                        // matched noise across optimizers
    const double res0 = residual(w0, wstar.data());
    const double cs   = measure_consistency(wp, wstar.data(), sigma, rng);

    for (int s = 0; s < STEPS; ++s) {
        make_grad(gbuf, wp, wstar.data(), sigma, shared_t, rng);
        std::copy(gbuf.begin(), gbuf.end(), gp);
        opt->step();
    }
    const double ce = measure_consistency(wp, wstar.data(), sigma, rng);
    const std::vector<float> wfinal(wp, wp + M * N);    // wp is stable (in-place updates)
    return {residual(wfinal, wstar.data()) / res0, cs, ce, lr};
}

// Best (lowest) final residual over an LR grid — fair comparison, no optimizer handicapped by a
// single fixed LR. Returns the winning run.
RunResult best_over_lrs(const std::string& opt_name, const std::vector<float>& lrs, float sigma,
                        bool shared_t, const std::vector<float>& w0,
                        const std::vector<float>& wstar) {
    RunResult best{1e30, 0, 0, 0};
    for (float lr : lrs) {
        RunResult r = run(opt_name, lr, sigma, shared_t, w0, wstar);
        if (r.final_res < best.final_res) best = r;
    }
    return best;
}

// ── DEPTH-2 control ────────────────────────────────────────────────────────────────────────
// The single-matrix problem above is CONVEX and well-conditioned — Muon's home turf. Real
// transformers are neither. This control composes two layers: Y = W2·W1 (M×N), target Y* a
// realizable product. The loss ½‖Y−Y*‖² is now BILINEAR (non-convex) and the two layers are
// COUPLED — and Muon orthogonalizes BOTH simultaneously. If Muon's real-world collapse is a
// depth/coupling effect rather than a noise effect, it should appear here and not above.
//   grads:  E = (Y−Y*) masked to the level's columns (+noise);  dW2 = E·W1ᵀ ;  dW1 = W2ᵀ·E.

double deep_residual(const Tensor& W2, const Tensor& W1, const std::vector<float>& ystar) {
    const Tensor Y = ops::matmul(W2, W1);
    const float* y = Y.data_as<float>().data();
    double s = 0;
    for (std::size_t i = 0; i < ystar.size(); ++i) { const double d = y[i] - ystar[i]; s += d * d; }
    return std::sqrt(s);
}

// Build the masked+noised output error E (M×N) for one step into `E`.
void make_error(Tensor& E, const Tensor& Y, const std::vector<float>& ystar,
                float sigma, bool shared_t, std::mt19937& rng) {
    float* e = E.data_as<float>().data();
    const float* y = Y.data_as<float>().data();
    std::fill(e, e + M * N, 0.0f);
    std::uniform_int_distribution<std::int64_t> pick_level(0, T - 1);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    auto add_level = [&](std::int64_t t, float gscale, float nscale) {
        const std::int64_t c0 = t * BLK;
        for (std::int64_t r = 0; r < M; ++r)
            for (std::int64_t c = c0; c < c0 + BLK; ++c) {
                const std::size_t k = idx(r, c);
                e[k] += gscale * (y[k] - ystar[k]) + nscale * noise(rng);
            }
    };
    if (shared_t) add_level(pick_level(rng), 1.0f, sigma / std::sqrt(static_cast<float>(B)));
    else for (int b = 0; b < B; ++b)
        add_level(pick_level(rng), 1.0f / static_cast<float>(B), sigma / static_cast<float>(B));
}

double run_deep(const std::string& opt_name, float lr, float sigma, bool shared_t,
                const std::vector<float>& w1_0, const std::vector<float>& w2_0,
                const std::vector<float>& ystar) {
    Tensor w1t({M, N}, DType::Float32), w2t({M, M}, DType::Float32);
    std::copy(w1_0.begin(), w1_0.end(), w1t.data_as<float>().begin());
    std::copy(w2_0.begin(), w2_0.end(), w2t.data_as<float>().begin());
    autograd::Variable W1(std::move(w1t), true), W2(std::move(w2t), true);
    W1.grad() = zeros({M, N});
    W2.grad() = zeros({M, M});

    std::vector<autograd::Variable*> params{&W2, &W1};   // both 2D ⇒ Muon orthogonalizes both
    auto opt = nn::make_optimizer(opt_name, params, lr);

    const double res0 = deep_residual(W2.data(), W1.data(), ystar);
    Tensor E({M, N}, DType::Float32);
    std::mt19937 rng(12345u);
    for (int s = 0; s < STEPS; ++s) {
        const Tensor Y = ops::matmul(W2.data(), W1.data());
        make_error(E, Y, ystar, sigma, shared_t, rng);
        W2.grad() = ops::matmul_bt(E, W1.data());        // dW2 = E·W1ᵀ  (M×M)
        W1.grad() = ops::matmul_tb(W2.data(), E);        // dW1 = W2ᵀ·E  (M×N)
        opt->step();
    }
    return deep_residual(W2.data(), W1.data(), ystar) / res0;
}

double best_deep(const std::string& opt_name, const std::vector<float>& lrs, float sigma,
                 bool shared_t, const std::vector<float>& w1_0, const std::vector<float>& w2_0,
                 const std::vector<float>& ystar) {
    double best = 1e30;
    for (float lr : lrs) best = std::min(best, run_deep(opt_name, lr, sigma, shared_t, w1_0, w2_0, ystar));
    return best;
}

} // namespace

int main() {
    std::println("Ch31 — Optimizer Sandbox: digesting the diffusion gradient");
    std::println("  param {}x{}, {} noise levels (disjoint {}-col blocks), B={} samples/step, {} steps",
                 M, N, T, BLK, B, STEPS);
    std::println("  per-optimizer LR is swept; we report the BEST (lowest) final residual.\n");

    // Fixed problem instance (same W0, W* for every cell so the table is comparable).
    std::mt19937 init(7u);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> w0(M * N), wstar(M * N);
    for (auto& x : wstar) x = g(init);
    for (auto& x : w0)    x = g(init);

    const std::vector<float> adam_lrs{3e-2f, 1e-2f, 3e-3f, 1e-3f, 3e-4f};
    const std::vector<float> muon_lrs{5e-2f, 2e-2f, 1e-2f, 5e-3f, 2e-3f};

    // Pick σ to hit a target INITIAL within-level consistency c0:
    //   c0 ≈ |s|/sqrt(|s|²+σ²d)  ⇒  σ = (|s|/√d)·sqrt(1/c0²−1), |s|,d measured on the start blocks.
    auto sigma_for = [&](double c0) -> float {
        if (c0 >= 0.999) return 0.0f;
        // mean per-block signal norm and block element count at W0
        double sig2 = 0; for (std::size_t i = 0; i < w0.size(); ++i) { const double d = w0[i] - wstar[i]; sig2 += d * d; }
        const double per_block_sig2 = sig2 / static_cast<double>(T);
        const double d_block = static_cast<double>(M * BLK);
        const double s_rms = std::sqrt(per_block_sig2 / d_block);
        return static_cast<float>(s_rms * std::sqrt(1.0 / (c0 * c0) - 1.0));
    };

    const std::vector<double> targets{1.0, 0.9, 0.7, 0.5, 0.3};

    for (bool shared : {true, false}) {
        std::println("════ t-mode: {} ════════════════════════════════════════════",
                     shared ? "shared-t   (one noise level / step — ch29 --shared-t)"
                            : "independent-t (B workers, own levels — diluted baseline)");
        std::println("  target-c0   meas-c0(start→end)     adam      adamw      muon     muon/adam");
        for (double c0 : targets) {
            const float sigma = sigma_for(c0);
            RunResult a  = best_over_lrs("adam",  adam_lrs, sigma, shared, w0, wstar);
            RunResult aw = best_over_lrs("adamw", adam_lrs, sigma, shared, w0, wstar);
            RunResult mu = best_over_lrs("muon",  muon_lrs, sigma, shared, w0, wstar);
            std::println("    {:.2f}      {:.2f} → {:.2f}        {:.4f}    {:.4f}    {:.4f}     {:.2f}x",
                         c0, a.cons_start, a.cons_end, a.final_res, aw.final_res, mu.final_res,
                         mu.final_res / (a.final_res + 1e-12));
        }
        std::println("");
    }

    std::println("Reading: residual = ‖W−W*‖/‖W0−W*‖ (lower=better; 0 = solved). muon/adam > 1 ⇒ Muon worse.");
    std::println("  If Muon's ratio climbs as c0 falls, orthogonalization is amplifying the within-level");
    std::println("  noise — the mechanism behind ch29's Muon failure. shared-t should help all three, Muon most.\n");

    // ── DEPTH-2 control: Y = W2·W1, bilinear + coupled, Muon orthogonalizes both layers ──────
    std::println("════ DEPTH-2 control: Y = W2·W1 (non-convex, layer-coupled) ════════════════");
    std::mt19937 dinit(11u);
    const float scale = 1.0f / std::sqrt(static_cast<float>(M));
    std::normal_distribution<float> dg(0.0f, scale);
    std::vector<float> w1_0(M * N), w2_0(M * M), w1s(M * N), w2s(M * M);
    for (auto& x : w1_0) x = dg(dinit);
    for (auto& x : w2_0) x = dg(dinit);
    for (auto& x : w1s)  x = dg(dinit);
    for (auto& x : w2s)  x = dg(dinit);
    {   // realizable target Y* = W2*·W1*
        Tensor a({M, M}, DType::Float32), b({M, N}, DType::Float32);
        std::copy(w2s.begin(), w2s.end(), a.data_as<float>().begin());
        std::copy(w1s.begin(), w1s.end(), b.data_as<float>().begin());
        const Tensor Ys = ops::matmul(a, b);
        std::vector<float> ystar(Ys.data_as<float>().begin(), Ys.data_as<float>().end());

        // σ relative to the init output-error RMS, to hit a target consistency on the output grad.
        Tensor a0({M, M}, DType::Float32), b0({M, N}, DType::Float32);
        std::copy(w2_0.begin(), w2_0.end(), a0.data_as<float>().begin());
        std::copy(w1_0.begin(), w1_0.end(), b0.data_as<float>().begin());
        const Tensor Y0 = ops::matmul(a0, b0);
        const float* y0 = Y0.data_as<float>().data();
        double e2 = 0; for (std::size_t i = 0; i < ystar.size(); ++i) { const double d = y0[i] - ystar[i]; e2 += d * d; }
        const double e_rms = std::sqrt(e2 / static_cast<double>(M * N));
        auto dsigma = [&](double c0) { return c0 >= 0.999 ? 0.0f
                          : static_cast<float>(e_rms * std::sqrt(1.0 / (c0 * c0) - 1.0)); };

        for (bool shared : {true, false}) {
            std::println("  t-mode {}:", shared ? "shared-t" : "independent-t");
            std::println("    target-c0      adam      adamw      muon     muon/adam");
            for (double c0 : targets) {
                const float sg = dsigma(c0);
                const double a_r  = best_deep("adam",  adam_lrs, sg, shared, w1_0, w2_0, ystar);
                const double aw_r = best_deep("adamw", adam_lrs, sg, shared, w1_0, w2_0, ystar);
                const double mu_r = best_deep("muon",  muon_lrs, sg, shared, w1_0, w2_0, ystar);
                std::println("      {:.2f}        {:.4f}    {:.4f}    {:.4f}     {:.2f}x",
                             c0, a_r, aw_r, mu_r, mu_r / (a_r + 1e-12));
            }
        }
    }
    std::println("\nReading (depth-2): output residual ‖Y−Y*‖/‖Y0−Y*‖. If Muon's ratio is >1 here but <1");
    std::println("  in the single-matrix case, the failure is the depth/coupling, not the gradient noise.\n");

    // ── LR SENSITIVITY: the tables above report BEST-over-LR. Real ch29 ran ONE Muon LR (0.02,
    // no warmup, no sweep). If Muon is great at its best LR but craters elsewhere, the real-world
    // failure is TUNING, not a fundamental incompatibility — the hypothesis the controls leave. ──
    std::println("════ LR sensitivity at c0=0.50, shared-t (single-matrix) ═══════════════════");
    const float sig50 = sigma_for(0.5);
    std::println("  adam:");
    for (float lr : adam_lrs)
        std::println("    lr={:.0e}   residual {:.4f}", lr, run("adam", lr, sig50, true, w0, wstar).final_res);
    std::println("  muon:");
    for (float lr : muon_lrs)
        std::println("    lr={:.0e}   residual {:.4f}", lr, run("muon", lr, sig50, true, w0, wstar).final_res);
    std::println("\nReading (sensitivity): a wide Muon spread (great at one LR, poor at others) ⇒ ch29's");
    std::println("  single untuned Muon run failed on TUNING, not on the gradient pathology we blamed.");
    return 0;
}
