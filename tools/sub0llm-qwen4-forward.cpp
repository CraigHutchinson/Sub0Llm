// sub0llm-qwen4-forward -- WP4d's harness: the FIRST time this engine is built at Qwen3.8-Flash-Next's
// real axes, loads the real transplanted weights through the real load_model(), and runs Model::forward
// on them (docs/WP4_SCOPE.md S6, WP4d).
//
// WHY A TOOL AND NOT A TEST. Every other real-axes target in this repo (sub0_qwen4_shape_tests,
// sub0llm-transplant) deliberately links NO engine, because layout.hpp is closed over one
// sub0_config.hpp and two definitions of sub0::PARAM_LAYOUT in one binary is an ODR violation. This one
// is the opposite: it links sub0_core, so it must be built against the SAME generated config the engine
// was -- i.e. against a real `sub0llm-configure` run at the real axes, not the hand-written
// tests/qwen4_real_axes/sub0_config.hpp. It cannot join sub0_tests for the same reason sub0_tests cannot
// be built at these axes: 43.4 GiB of resident weights is not a unit-test fixture.
//
// WHAT IT CHECKS, in order, each printed with its actual number rather than a pass/fail:
//   1. load_model() accepts the artifact -- header fields, PARAM_FLOATS, and both architecture
//      fingerprint trailers. Nothing had ever fed this file to the engine's own reader before.
//   2. The two SYNTHESIZED destinations are what the transplant said they were: LnF all-1.0, LmBias
//      all-0.0 (docs/WP4_SCOPE.md WP4c finding 3). Checked against the bytes that actually loaded.
//   3. Model::forward on real tokens: finite, and its per-layer-free summary statistics.
//   4. forward vs forward_one parity at the real dims -- the check docs/WP4_SCOPE.md S6 names as having
//      caught a real bug in every one of WP1-3.
//   5. The LnF question, answered with numbers rather than in the abstract: how much the final
//      op_rmsnorm(ln_f) -- which the real model does NOT have, and which was synthesized to the RMSNorm
//      identity GAIN even though RMSNorm itself is not an identity -- moves the logits, and how much
//      op_rmsnorm's eps (1e-5 here, 1e-6 in every real Qwen4ExpTextRMSNorm) moves them.
//   6. Peak working set, because docs/QWEN4_MEMORY_ORCHESTRATION.md's budget is a prediction until
//      something measures it.
//
// The input tokens are the six hand-chosen REAL Qwen vocabulary ids the n-gram fixture already uses
// (tests/fixtures/qwen4_preview/ngram_embedding_manifest.json, "test_tokens") -- the only real-vocab
// token ids anywhere in this repo's fixture set. Every other qwen4_preview fixture is a SLICED layer
// (hidden_size 32 for GDN, 16 for QSA), so its inputs and reference outputs do not exist at the real
// hidden_size at all and cannot be fed to a full-dims Model::forward -- see the report for WP4d.

#include "sub0/core.hpp"
#include "sub0/layout.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <print>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

using namespace sub0;

namespace {

// The six real Qwen token ids (see the header comment). All are < VOCAB by construction: the real
// unigram_vocab_size IS this build's VOCAB.
constexpr int kTokens[] = {1543, 88123, 245000, 7, 99999, 156789};
constexpr int kT = static_cast<int>(std::size(kTokens));

struct Stats { double mean = 0, rms = 0, min = 0, max = 0; std::size_t nonfinite = 0; };

Stats stats_of(const float* p, std::size_t n) {
    Stats s;
    if (n == 0) return s;
    s.min = s.max = p[0];
    double sum = 0, sq = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = p[i];
        if (!std::isfinite(v)) { ++s.nonfinite; continue; }
        sum += v; sq += v * v;
        s.min = std::min(s.min, v); s.max = std::max(s.max, v);
    }
    s.mean = sum / static_cast<double>(n);
    s.rms  = std::sqrt(sq / static_cast<double>(n));
    return s;
}

// Locate one PARAM_LAYOUT entry by kind (the FIRST match, which for the model-level tensors below is
// the only one). Returns nullptr rather than asserting so the caller can report the absence.
const ParamDesc* find_kind(PKind k) {
    for (const ParamDesc& p : PARAM_LAYOUT) if (p.kind == k) return &p;
    return nullptr;
}

double peak_working_set_gib() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0 * 1024.0);
#endif
    return 0.0;
}

void report_memory(const char* when) {
    std::println("[mem] {:<28} peak working set {:.2f} GiB", when, peak_working_set_gib());
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"sub0llm-qwen4-forward: WP4d -- load the real transplanted Qwen4 sub-stack and run "
                 "Model::forward on it"};
    std::string model_path;
    int T = kT;
    app.add_option("--model", model_path, "the transplanted S0L5 artifact (qwen4_sub4.bin)")->required();
    app.add_option("--tokens", T, "how many of the six fixture tokens to run (1..6)")
       ->capture_default_str()->check(CLI::Range(1, kT));
    CLI11_PARSE(app, argc, argv);
    // Unbuffered: this run is minutes long and holds 43 GiB resident, so if it dies the partial output
    // IS the finding. A buffered stdout redirected to a file loses all of it.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::println("--- build --------------------------------------------------------------");
    print_config();
    print_host_memplan();
    std::println("PARAM_FLOATS {} | NUM_PARAMS {} | D_Q {} | D_KV {} | ROTARY_DIM {}",
                 PARAM_FLOATS, NUM_PARAMS, D_Q, D_KV, ROTARY_DIM);
    report_memory("before load");

    std::println("\n--- 1. load_model ------------------------------------------------------");
    const auto t0 = std::chrono::steady_clock::now();
    if (!load_model(model_path.c_str())) {
        std::println(stderr, "FAIL: load_model rejected '{}' (see the reason printed above)", model_path);
        return 2;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::println("load_model: ACCEPTED in {:.1f}s ({:.2f} GiB of f32 parameters)", load_s,
                 static_cast<double>(PARAM_FLOATS) * 4.0 / (1024.0 * 1024.0 * 1024.0));
    report_memory("after load");

    const float* P = params_ptr();

    std::println("\n--- 2. the two synthesized destinations ---------------------------------");
    const ParamDesc* lnf = find_kind(PKind::LnF);
    const ParamDesc* lmh = find_kind(PKind::LmHead);
    const ParamDesc* lmb = find_kind(PKind::LmBias);
    if (!lnf || !lmh || !lmb) { std::println(stderr, "FAIL: LnF/LmHead/LmBias missing from PARAM_LAYOUT"); return 3; }
    {
        const Stats a = stats_of(P + lnf->off, lnf->n());
        const Stats b = stats_of(P + lmb->off, lmb->n());
        std::println("LnF    [{}x{}]: min {:.9g} max {:.9g}  (transplant synthesized 1.0 -- the RMSNorm "
                     "identity GAIN)", lnf->rows, lnf->cols, a.min, a.max);
        std::println("LmBias [{}x{}]: min {:.9g} max {:.9g}  (transplant synthesized 0.0)",
                     lmb->rows, lmb->cols, b.min, b.max);
    }
    {   // A cheap "these really are the real weights, not a zeroed arena" check on the two biggest
        // tensors -- AGENTS.md S9's own real-file discipline applied to what LANDED, not what was written.
        const ParamDesc* emb = find_kind(PKind::TokEmb);
        const Stats e = stats_of(P + emb->off, 4096);          // first rows only: 2.5 GiB is not free
        const Stats h = stats_of(P + lmh->off, 4096);
        std::println("TokEmb  first 4096 floats: mean {:+.6f} rms {:.6f} [{:+.4f}, {:+.4f}]",
                     e.mean, e.rms, e.min, e.max);
        std::println("LmHead  first 4096 floats: mean {:+.6f} rms {:.6f} [{:+.4f}, {:+.4f}]",
                     h.mean, h.rms, h.min, h.max);
    }

    std::println("\n--- 3. Model::forward on the real weights -------------------------------");
    std::println("tokens ({}): {}", T, [&] {
        std::string s;
        for (int i = 0; i < T; ++i) s += (i ? ", " : "") + std::to_string(kTokens[i]);
        return s;
    }());
    graph_reset();          // lays out this thread's parameter nodes + allocates its Worker
    report_memory("after graph_reset");
    const auto t1 = std::chrono::steady_clock::now();
    Node* out = forward(kTokens, T);
    const double fwd_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (!out) { std::println(stderr, "FAIL: forward returned null"); return 4; }
    std::println("forward: [{} x {}] in {:.2f}s", out->rows, out->cols, fwd_s);
    report_memory("after forward");
    std::vector<float> batched(out->data.begin(), out->data.end());
    for (int t = 0; t < T; ++t) {
        const Stats s = stats_of(batched.data() + static_cast<std::size_t>(t) * VOCAB, VOCAB);
        const auto  am = std::max_element(batched.begin() + static_cast<std::ptrdiff_t>(t) * VOCAB,
                                          batched.begin() + static_cast<std::ptrdiff_t>(t + 1) * VOCAB);
        std::println("  row {}: mean {:+.6f} rms {:.4f} [{:+.4f}, {:+.4f}] argmax {} nonfinite {}",
                     t, s.mean, s.rms, s.min, s.max,
                     std::distance(batched.begin() + static_cast<std::ptrdiff_t>(t) * VOCAB, am), s.nonfinite);
    }

    std::println("\n--- 4. forward vs forward_one parity at the real dims --------------------");
    kv_reset();
    double max_abs = 0.0, max_rel = 0.0;
    int worst_t = -1, worst_v = -1;
    std::vector<float> hidden_last(D_MODEL, 0.f);
    const auto t2 = std::chrono::steady_clock::now();
    for (int t = 0; t < T; ++t) {
        const float* one = forward_one(kTokens[t], t);
        if (t == T - 1) std::copy_n(last_hidden_ptr(), D_MODEL, hidden_last.begin());
        const float* ref = batched.data() + static_cast<std::size_t>(t) * VOCAB;
        for (int v = 0; v < VOCAB; ++v) {
            const double d = std::abs(static_cast<double>(one[v]) - ref[v]);
            const double r = d / (std::abs(static_cast<double>(ref[v])) + 1e-6);
            if (d > max_abs) { max_abs = d; worst_t = t; worst_v = v; }
            max_rel = std::max(max_rel, r);
        }
    }
    const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    std::println("forward_one over {} positions in {:.2f}s", T, dec_s);
    std::println("max |forward - forward_one| = {:.6g}  (at row {}, vocab id {}); max relative = {:.6g}",
                 max_abs, worst_t, worst_v, max_rel);
    report_memory("after forward_one");

    std::println("\n--- 5. the LnF / eps question, measured ---------------------------------");
    // hidden_last is the GR EXIT-COLLAPSED, pre-ln_f representation at the final position (that is what
    // last_hidden_ptr captures under USE_GATED_RESIDUAL -- backend_cpu.cpp's forward_one). The real
    // model has NO tensor at this site at all (WP4c finding 3: no output_norm.weight under any name),
    // so the question is not "is the gain right" but "does applying an RMSNorm here at all change the
    // answer". Three readouts off the SAME hidden state, differing only in that step.
    const Stats hs = stats_of(hidden_last.data(), D_MODEL);
    std::println("pre-ln_f hidden (last position): rms {:.6f} mean {:+.6f} [{:+.4f}, {:+.4f}]",
                 hs.rms, hs.mean, hs.min, hs.max);
    {
        double ms = 0.0;
        for (int j = 0; j < D_MODEL; ++j) ms += static_cast<double>(hidden_last[j]) * hidden_last[j];
        ms /= D_MODEL;
        const double r5 = 1.0 / std::sqrt(ms + 1e-5), r6 = 1.0 / std::sqrt(ms + 1e-6);
        std::println("mean-square {:.9g}; 1/sqrt(ms+eps): eps=1e-5 -> {:.9g}, eps=1e-6 -> {:.9g} "
                     "(relative gap {:.3g})", ms, r5, r6, std::abs(r5 - r6) / r5);
        // Readouts: y = h * r * G (the engine's ln_f), y = h * r6 * G (the real eps), and y = h (no norm
        // at all -- the real model's actual structure). One lm_head GEMM each.
        const float* G  = P + lnf->off;
        const float* Wh = P + lmh->off;         // [D_MODEL, VOCAB], rows=in
        const float* B  = P + lmb->off;
        std::vector<double> a5(D_MODEL), a6(D_MODEL), an(D_MODEL);
        for (int j = 0; j < D_MODEL; ++j) {
            a5[j] = hidden_last[j] * r5 * G[j];
            a6[j] = hidden_last[j] * r6 * G[j];
            an[j] = hidden_last[j];
        }
        std::vector<double> l5(VOCAB, 0.0), l6(VOCAB, 0.0), ln(VOCAB, 0.0);
        for (int j = 0; j < D_MODEL; ++j) {
            const float* row = Wh + static_cast<std::size_t>(j) * VOCAB;
            const double x5 = a5[j], x6 = a6[j], xn = an[j];
            for (int v = 0; v < VOCAB; ++v) { l5[v] += x5 * row[v]; l6[v] += x6 * row[v]; ln[v] += xn * row[v]; }
        }
        for (int v = 0; v < VOCAB; ++v) { l5[v] += B[v]; l6[v] += B[v]; ln[v] += B[v]; }
        double d56 = 0.0, d5n = 0.0;
        for (int v = 0; v < VOCAB; ++v) {
            d56 = std::max(d56, std::abs(l5[v] - l6[v]));
            d5n = std::max(d5n, std::abs(l5[v] - ln[v]));
        }
        const auto arg = [](const std::vector<double>& l) {
            return static_cast<int>(std::distance(l.begin(), std::max_element(l.begin(), l.end())));
        };
        std::println("logits max|eps1e-5 - eps1e-6| = {:.6g}   (argmax {} vs {})", d56, arg(l5), arg(l6));
        std::println("logits max|with ln_f - WITHOUT any final norm| = {:.6g}   (argmax {} vs {})",
                     d5n, arg(l5), arg(ln));
        std::println("cross-check: this row's forward_one argmax was {}", arg(l5));
    }

    std::println("\n--- peak resource use ---------------------------------------------------");
    report_memory("final");
    return 0;
}
