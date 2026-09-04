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
//   2. The tail is what the corrected layout says: NO LnF slot at all (the real model has no final
//      norm), and one synthesized destination, LmBias all-0.0. Checked against the bytes that loaded.
//   3. Model::forward on real tokens: finite, and its per-layer-free summary statistics.
//   4. forward vs forward_one parity at the real dims -- the check docs/WP4_SCOPE.md S6 names as having
//      caught a real bug in every one of WP1-3.
//   5. The LnF question, now RESOLVED rather than open: that lm_head really does read the GR exit
//      collapse's output un-normed (an independent double-precision readout must reproduce the engine's
//      own logits), plus the size of what the removed op_rmsnorm(ln_f) had been costing -- the same
//      measurement WP4d made when the site still existed, kept so the fix has a before/after number.
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
#include "sub0/gated_residual_math.hpp"
#include "sub0/gdn_math.hpp"
#include "sub0/moe_math.hpp"
#include "sub0/moe_quant.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <fstream>
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
    // WP4e's gate (docs/WP4_SCOPE.md WP4e) is a BITWISE comparison of the logits an all-f32-resident
    // build produces against a quantized-resident one's. The two are different builds -- MOE_QUANT_EXPERTS
    // changes PARAM_LAYOUT, so they cannot be one binary -- so the comparison has to happen outside the
    // process, on the raw [T x VOCAB] f32 array each writes here. Raw floats, no header: the only thing
    // the gate asks of the file is that two of them be byte-identical.
    std::string dump_path;
    app.add_option("--dump-logits", dump_path,
                   "write forward()'s raw [T x VOCAB] f32 logits to this path (WP4e's bitwise gate)");
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

    std::println("\n--- 2. the tail: no LnF slot, one synthesized destination ----------------");
    // LnF is GONE from the layout (the real model has no final norm -- the GR exit collapse's own
    // hc_norm is it), so the only synthesized destination left is LmBias. Asserted as an ABSENCE here,
    // not merely omitted, because "the slot quietly came back" is exactly what this check is for.
    const ParamDesc* lnf = find_kind(PKind::LnF);
    const ParamDesc* lmh = find_kind(PKind::LmHead);
    const ParamDesc* lmb = find_kind(PKind::LmBias);
    if (lnf) { std::println(stderr, "FAIL: PARAM_LAYOUT still carries an LnF tensor under GR"); return 3; }
    if (!lmh || !lmb) { std::println(stderr, "FAIL: LmHead/LmBias missing from PARAM_LAYOUT"); return 3; }
    {
        const Stats b = stats_of(P + lmb->off, lmb->n());
        std::println("LnF: ABSENT from PARAM_LAYOUT, as the real model is (no output_norm.weight in the "
                     "GGUF, and the safetensors index's only model-level norm is the GR exit's hc_norm)");
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
    if (!dump_path.empty()) {
        std::ofstream os(dump_path, std::ios::binary | std::ios::trunc);
        os.write(reinterpret_cast<const char*>(batched.data()),
                 static_cast<std::streamsize>(batched.size() * sizeof(float)));
        if (!os) { std::println(stderr, "FAIL: could not write {}", dump_path); return 5; }
        std::println("wrote {} ({} floats = {} bytes) for WP4e's bitwise comparison", dump_path,
                     batched.size(), batched.size() * sizeof(float));
    }
    for (int t = 0; t < T; ++t) {
        const Stats s = stats_of(batched.data() + static_cast<std::size_t>(t) * VOCAB, VOCAB);
        const auto  am = std::max_element(batched.begin() + static_cast<std::ptrdiff_t>(t) * VOCAB,
                                          batched.begin() + static_cast<std::ptrdiff_t>(t + 1) * VOCAB);
        std::println("  row {}: mean {:+.6f} rms {:.4f} [{:+.4f}, {:+.4f}] argmax {} nonfinite {}",
                     t, s.mean, s.rms, s.min, s.max,
                     std::distance(batched.begin() + static_cast<std::ptrdiff_t>(t) * VOCAB, am), s.nonfinite);
    }

    std::println("\n--- 3b. Model::forward vs an INDEPENDENT math-core replay ----------------");
    // The gate docs/WP4_SCOPE.md S6 asks for is "layer 0's output matches the existing real fixture
    // through the full engine path". That gate as written is not reachable, and the reason is worth
    // stating precisely rather than working around: every qwen4_preview fixture is a SLICED layer
    // (gdn_layer0_small is hidden_size 32 / 1 key head / 3 value heads; qsa_layer3_small is hidden_size
    // 16 / 2 heads / head_dim 8), so its input and its reference output do not exist at hidden_size
    // 2560 at all. WP4c's levels 3/4 replayed those fixtures at the FIXTURE's dims through the
    // transplant's own mapping; nothing anywhere produces a reference for a real-dims layer.
    //
    // What IS both reachable and the actual thing WP4d adds over WP4c: does the ENGINE path -- the Node
    // graph, GR's real wrapping, MIXER_SCHEDULE, the PARAM_LAYOUT offsets the weights landed at -- agree
    // with the math cores WP4c already validated, driven from the SAME loaded weights? That is the
    // divergence this stage could localize, and it is checked below by rebuilding the first three (GDN)
    // layers straight out of params_ptr() with gr::/gdn::/moe:: calls and comparing against the engine's
    // own per-execution residual-stream norms (loop_pass_stats). Layer 3 (QSA) is excluded only because
    // its op needs the backend's internal precomputed rope table; the check still covers embed, the GR
    // entry tile, 6 GR instances, 3 GDN mixers and 3 MoE blocks composed in the engine's own order.
    {
        std::vector<float> eng_delta(LOOP_EXEC_COUNT, 0.f), eng_hnorm(LOOP_EXEC_COUNT, 0.f);
        loop_pass_stats(kTokens, T, eng_delta.data(), eng_hnorm.data());

        constexpr int WIDE = HC_COUNT * D_MODEL;
        const std::size_t tw = static_cast<std::size_t>(T) * WIDE;
        std::vector<float> wide(tw), prev(tw), normed(tw), mixed(static_cast<std::size_t>(T) * D_MODEL),
                            mixer_out(static_cast<std::size_t>(T) * D_MODEL),
                            inj(static_cast<std::size_t>(T) * HC_COUNT),
                            h0(static_cast<std::size_t>(T) * D_MODEL);
        std::vector<float> gr_scr(gr::mix_scratch_floats(GR_DIMS, T));
        std::vector<float> gdn_state(gdn::state_floats(GDN_DIMS)), gdn_conv(gdn::conv_hist_floats(GDN_DIMS));
        std::vector<float> gdn_scr(gdn::scratch_floats(GDN_DIMS, T)), moe_scr(moe::scratch_floats(MOE_DIMS));

        const ParamDesc* emb = find_kind(PKind::TokEmb);
        for (int t = 0; t < T; ++t)
            std::copy_n(P + emb->off + static_cast<std::size_t>(kTokens[t]) * D_MODEL, D_MODEL,
                        h0.begin() + static_cast<std::ptrdiff_t>(t) * D_MODEL);
        gr::tile(GR_DIMS, T, h0.data(), wide.data());

        // Walk PARAM_LAYOUT with a cursor and CHECK each kind as it goes, rather than hard-coding
        // offsets: a silent disagreement about the layout order is precisely one of the things this
        // comparison exists to detect, so it must not be assumed by the checker itself.
        std::size_t pi = 1;   // 0 == TokEmb
        bool layout_ok = true;
        const auto take = [&](PKind want) -> const float* {
            if (pi >= static_cast<std::size_t>(NUM_PARAMS) || PARAM_LAYOUT[pi].kind != want) {
                if (layout_ok)
                    std::println(stderr, "LAYOUT MISMATCH at PARAM_LAYOUT[{}]: expected kind {}, found {}",
                                 pi, static_cast<int>(want),
                                 pi < static_cast<std::size_t>(NUM_PARAMS)
                                     ? static_cast<int>(PARAM_LAYOUT[pi].kind) : -1);
                layout_ok = false;
                return P;
            }
            return P + PARAM_LAYOUT[pi++].off;
        };
        std::vector<const float*> eg(NUM_EXPERTS), eu(NUM_EXPERTS), ed(NUM_EXPERTS);
        // WP4e: in a quantized-resident build the routed experts are not PARAM_LAYOUT tensors, so this
        // replay opens the sidecar ITSELF rather than borrowing the engine's Store -- keeping it the
        // independent second derivation it exists to be (it already walks PARAM_LAYOUT with its own
        // cursor rather than trusting the engine's offsets, for the same reason).
        moeq::Store replay_store;
        moeq::ExpertCache<2, static_cast<std::size_t>(D_MODEL) * D_FF> replay_cache;
        if constexpr (USE_MOE_QUANT) {
            std::string err;
            if (!replay_store.open(model_path + ".moeq", err)) {
                std::println(stderr, "FAIL: replay cannot open the sidecar: {}", err);
                return 6;
            }
            replay_cache.allocate();
        }
        double worst_h = 0.0, worst_d = 0.0;
        for (int l = 0; l < N_LAYERS; ++l) {
            if (MIXER_SCHEDULE[static_cast<std::size_t>(l)] != LayerMixer::Gdn) break;   // layer 3 (QSA)
            std::copy(wide.begin(), wide.end(), prev.begin());
            const float* an = take(PKind::GrHcNorm);
            const float* ad = take(PKind::GrMixDown);
            const float* au = take(PKind::GrMixUp);
            const float* ai = take(PKind::GrBlockInject);
            const float* qkv = take(PKind::GdnInProjQkv);
            const float* wz  = take(PKind::GdnInProjZ);
            const float* wb  = take(PKind::GdnInProjB);
            const float* wa  = take(PKind::GdnInProjA);
            const float* cw  = take(PKind::GdnConv);
            const float* al  = take(PKind::GdnALog);
            const float* dtb = take(PKind::GdnDtBias);
            const float* nw  = take(PKind::GdnNorm);
            const float* op  = take(PKind::GdnOutProj);
            const float* fn = take(PKind::GrHcNorm);
            const float* fd = take(PKind::GrMixDown);
            const float* fu = take(PKind::GrMixUp);
            const float* fi = take(PKind::GrBlockInject);
            const float* rt = take(PKind::MoeRouter);
            if constexpr (!USE_MOE_QUANT) {
                for (int e = 0; e < NUM_EXPERTS; ++e) {
                    eg[static_cast<std::size_t>(e)] = take(PKind::MoeGate);
                    eu[static_cast<std::size_t>(e)] = take(PKind::MoeUp);
                    ed[static_cast<std::size_t>(e)] = take(PKind::MoeDown);
                }
            }
            const float* sg = take(PKind::MoeSharedGate);
            const float* su = take(PKind::MoeSharedUp);
            const float* sd = take(PKind::MoeSharedDown);
            const float* sp = take(PKind::MoeSharedGateProj);
            if (!layout_ok) break;

            // attention half: GR read -> GDN -> GR write
            gr::hc_norm(GR_DIMS, T, wide.data(), an, normed.data());
            gr::mix(GR_DIMS, T, normed.data(), ad, au, mixed.data(), gr_scr.data());
            gr::gate(GR_DIMS, T, normed.data(), ai, inj.data());
            std::fill(gdn_state.begin(), gdn_state.end(), 0.f);
            std::fill(gdn_conv.begin(), gdn_conv.end(), 0.f);
            gdn::forward(GDN_DIMS, T, mixed.data(), qkv, wz, wb, wa, cw, dtb, al, nw, op,
                         gdn_state.data(), gdn_conv.data(), mixer_out.data(), gdn_scr.data());
            gr::combine(GR_DIMS, T, wide.data(), mixer_out.data(), inj.data(), wide.data());
            // mlp half: GR read -> MoE -> GR write
            gr::hc_norm(GR_DIMS, T, wide.data(), fn, normed.data());
            gr::mix(GR_DIMS, T, normed.data(), fd, fu, mixed.data(), gr_scr.data());
            gr::gate(GR_DIMS, T, normed.data(), fi, inj.data());
            moe::forward_via(MOE_DIMS, T, mixed.data(), rt,
                             [&](int e) -> moe::ExpertWeights {
                                 if constexpr (USE_MOE_QUANT) {
                                     const auto r = replay_cache.resolve(replay_store, l, e);
                                     return {r.gate, r.up, r.down};
                                 } else {
                                     return {eg[static_cast<std::size_t>(e)],
                                             eu[static_cast<std::size_t>(e)],
                                             ed[static_cast<std::size_t>(e)]};
                                 }
                             },
                             sg, su, sd, sp, mixer_out.data(), moe_scr.data());
            gr::combine(GR_DIMS, T, wide.data(), mixer_out.data(), inj.data(), wide.data());

            double hn = 0.0, dl = 0.0;
            for (std::size_t i = 0; i < tw; ++i) {
                hn += static_cast<double>(prev[i]) * prev[i];
                const double d = static_cast<double>(wide[i]) - prev[i];
                dl += d * d;
            }
            hn = std::sqrt(hn); dl = std::sqrt(dl);
            const double rh = std::abs(hn - eng_hnorm[static_cast<std::size_t>(l)]) / (hn + 1e-12);
            const double rd = std::abs(dl - eng_delta[static_cast<std::size_t>(l)]) / (dl + 1e-12);
            worst_h = std::max(worst_h, rh); worst_d = std::max(worst_d, rd);
            std::println("  layer {} ({}): ||h_in|| engine {:.7g} vs replay {:.7g} (rel {:.3g}) | "
                         "||delta|| engine {:.7g} vs replay {:.7g} (rel {:.3g})",
                         l, "GDN", static_cast<double>(eng_hnorm[static_cast<std::size_t>(l)]), hn, rh,
                         static_cast<double>(eng_delta[static_cast<std::size_t>(l)]), dl, rd);
        }
        if (layout_ok)
            std::println("worst relative disagreement, engine path vs math-core replay: "
                         "||h_in|| {:.3g}, ||delta|| {:.3g}", worst_h, worst_d);
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

    std::println("\n--- 5. the readout is un-normed, and what the removed LnF was costing ----");
    // hidden_last is the GR EXIT-COLLAPSED representation at the final position -- and, since the LnF
    // removal, it is EXACTLY what lm_head reads (backend_cpu.cpp's forward_one no longer calls
    // rmsnorm_row under USE_GATED_RESIDUAL). Two claims are checked here with numbers:
    //   (a) the engine really is un-normed now: recomputing lm_head(hidden_last) + bias in double
    //       reproduces forward_one's own logits row to float-rounding, which it cannot do if any
    //       normalization still sits between them;
    //   (b) how much the removed op_rmsnorm(h, ln_f) WAS moving the answer -- computed against the
    //       exact gain the old artifact carried (synthesized 1.0), so this is the size of the bug that
    //       was fixed, not a hypothetical.
    const Stats hs = stats_of(hidden_last.data(), D_MODEL);
    std::println("final hidden (last position, what lm_head reads): rms {:.6f} mean {:+.6f} "
                 "[{:+.4f}, {:+.4f}]", hs.rms, hs.mean, hs.min, hs.max);
    {
        double ms = 0.0;
        for (int j = 0; j < D_MODEL; ++j) ms += static_cast<double>(hidden_last[j]) * hidden_last[j];
        ms /= D_MODEL;
        const double r5 = 1.0 / std::sqrt(ms + 1e-5), r6 = 1.0 / std::sqrt(ms + 1e-6);
        std::println("mean-square {:.9g}; the RMSNorm this site no longer applies would have scaled by "
                     "{:.9g} (eps 1e-5) / {:.9g} (eps 1e-6)", ms, r5, r6);
        const float* Wh = P + lmh->off;         // [D_MODEL, VOCAB], rows=in
        const float* B  = P + lmb->off;
        // ln = the real structure (no final norm, gain-free); l5 = what the pre-fix build computed
        // (RMSNorm at eps 1e-5 with the synthesized identity gain 1.0); l6 = same at the real eps.
        std::vector<double> l5(VOCAB, 0.0), l6(VOCAB, 0.0), ln(VOCAB, 0.0);
        for (int j = 0; j < D_MODEL; ++j) {
            const float* row = Wh + static_cast<std::size_t>(j) * VOCAB;
            const double xn = hidden_last[j], x5 = xn * r5, x6 = xn * r6;
            for (int v = 0; v < VOCAB; ++v) { l5[v] += x5 * row[v]; l6[v] += x6 * row[v]; ln[v] += xn * row[v]; }
        }
        for (int v = 0; v < VOCAB; ++v) { l5[v] += B[v]; l6[v] += B[v]; ln[v] += B[v]; }
        double d56 = 0.0, d5n = 0.0, dnn = 0.0, lrms = 0.0;
        // forward()'s own logits for the SAME (last) position -- the engine's answer to compare against.
        const float* eng = batched.data() + static_cast<std::size_t>(T - 1) * VOCAB;
        for (int v = 0; v < VOCAB; ++v) {
            d56 = std::max(d56, std::abs(l5[v] - l6[v]));
            d5n = std::max(d5n, std::abs(l5[v] - ln[v]));
            dnn = std::max(dnn, std::abs(ln[v] - static_cast<double>(eng[v])));
            lrms += ln[v] * ln[v];
        }
        lrms = std::sqrt(lrms / VOCAB);
        const auto arg = [](const std::vector<double>& l) {
            return static_cast<int>(std::distance(l.begin(), std::max_element(l.begin(), l.end())));
        };
        std::println("(a) max |independent un-normed readout - engine logits| = {:.6g}  (argmax {} vs "
                     "engine's own)", dnn, arg(ln));
        std::println("(b) max |WITH the removed ln_f - without it| = {:.6g}  against a logit rms of "
                     "{:.6g}  ({:.1f}% of scale; argmax {} vs {})",
                     d5n, lrms, 100.0 * d5n / (lrms > 0 ? lrms : 1.0), arg(l5), arg(ln));
        std::println("    and the eps question that removal made moot: max|1e-5 - 1e-6| = {:.6g}", d56);
    }

    std::println("\n--- peak resource use ---------------------------------------------------");
    report_memory("final");
    return 0;
}
