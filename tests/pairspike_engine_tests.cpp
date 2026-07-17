// pairspike_engine_tests.cpp -- SPIKE (Phase 2 of docs/SCRATCH_TOKENS.md's word-level plan): can a tiny
// model CHOOSE which of K bound entities to reference -- content-select answered by EMITTING a
// `<sigil> h` pair -- when the pair's index position carries the entity's composed content vector
// (SentinelBindings' embedding override)? K=12 deliberately exceeds the bounded pool's 6-id ceiling, and
// the entities' spellings are NEVER in the token stream: the override is the only channel their content
// can reach the model through.
//
// Arms (each K=12, 3 training seeds, EpochWalk sampling -- the validated spike default):
//   OVERRIDE/ConvPool -- the shootout-selected default encoder (learned enc_w, single-threaded AdamW).
//   OVERRIDE/HRR      -- the parameter-free order-sensitive fallback (first-letter queries need ORDER,
//                        so permutation-invariant MeanPool is not a meaningful override arm here).
//   NO-OVERRIDE       -- pairs embed as their plain rows: the preamble then carries NO information
//                        linking a handle to its entity's spelling, so first-letter queries are
//                        unanswerable IN PRINCIPLE -- this pins the chance floor (~1/K) and proves any
//                        override-arm gap is carried by the embedding override, not the pair syntax.
//
// The spike commandeers casing::TOK_RESERVED_9 as the sigil (no tokenizer-format change; the permanent
// id comes from the bounded pool's deprecation -- see SentinelBindings' comment). Training drives the
// low-level forward/backward API single-threaded (set_sentinel_bindings is per-window state, and
// ConvPool's enc_w_grad has no per-thread reduction), same shape as the persistent spike's own loop.
// Tagged [.pairspike] (hidden): trains 9 models. Invoke: `sub0_tests "[pairspike]"`.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/pairspike.hpp"
#include "sub0/scratch_slots.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ps  = sub0::pairspike;
namespace ss  = sub0::scratchspike;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int    kSigil        = cas::TOK_RESERVED_9;   // spike-commandeered; see file header
constexpr int    kBatch        = 16;
constexpr int    kEvalRounds   = 10;
constexpr int    kStepsPerEval = 300;
constexpr int    kOovPool      = 400;
constexpr double kDrilledFrac  = 0.7;
constexpr float  kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

struct EncAdam { std::vector<float> m, v; long t = 0; };

// Exactly-once-per-epoch shuffled document walk -- the validated spike-training default (see the
// epoch-walk variance experiment in persistent_scratchspike_engine_tests.cpp / project docs).
struct DocWalk {
    std::vector<std::size_t> perm;
    std::size_t pos = 0;
    explicit DocWalk(std::size_t n_docs) : perm(n_docs) {
        for (std::size_t i = 0; i < n_docs; ++i) perm[i] = i;
    }
    std::size_t next(std::mt19937& rng) {
        if (pos == 0) std::shuffle(perm.begin(), perm.end(), rng);
        const std::size_t d = perm[pos];
        pos = (pos + 1) % perm.size();
        return d;
    }
};

struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

// The answer pair the model emitted: the two tokens right after SEP ('='), or (-1,-1).
std::pair<int, int> answer_pair(const std::vector<int>& out) {
    for (std::size_t i = 0; i + 2 < out.size(); ++i)
        if (out[i] == ps::SEP) return { out[i + 1], out[i + 2] };
    return { -1, -1 };
}

// One eval task via live greedy decode: install the task's bindings (override arms) or nothing
// (control), prime the prompt, decode, grade the emitted pair STRICTLY (sigil AND correct handle).
bool run_task(const ps::Task& k, bool use_override, sub0::SlotEncoding enc, const float* enc_w) {
    std::vector<std::vector<int>> tbl;
    for (const std::string& oov : k.binds) tbl.push_back(ss::oov_bytes(oov));
    sub0::SentinelBindings sb{ std::span<const std::vector<int>>(tbl), kSigil, ps::HANDLE_BASE, enc };
    sb.enc_w = enc_w;
    if (use_override) sub0::set_sentinel_bindings(&sb);
    std::vector<int> ctx = k.prompt;
    std::mt19937 rng(0);
    const int n = std::min(8, SEQ_LEN - static_cast<int>(ctx.size()) - 1);   // pair + EOS needs ~4
    sub0::kv_decode_generate(ctx, n, /*temp=*/1.f, /*topk=*/1, rng, cas::TOK_EOS, /*use_gpu=*/false);
    if (use_override) sub0::set_sentinel_bindings(nullptr);
    const auto [sig, handle] = answer_pair(ctx);
    return sig == kSigil && handle == k.answer_handle;
}

// Eval task GENERATION stays serial (rng consumption order defines the eval set -- must stay identical
// across runs/threads); SCORING parallelizes over tasks (g_kv, the binding installs, and run_task's
// state are all per-thread -- the pragma degrades to serial if OpenMP is off for this TU).
Acc eval_select(int K, const std::vector<std::string>& oovs, bool use_override, sub0::SlotEncoding enc,
                const float* enc_w, unsigned seed) {
    std::vector<ps::Task> tasks;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi)
        tasks.push_back(ps::pick_select_pair_task(kSigil, oovs, qi, K, rng));
    int ok = 0;
    #pragma omp parallel for reduction(+ : ok) schedule(dynamic)
    for (int i = 0; i < static_cast<int>(tasks.size()); ++i)
        ok += run_task(tasks[static_cast<std::size_t>(i)], use_override, enc, enc_w);
    return { ok, static_cast<int>(tasks.size()) };
}

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

// Training loop. PARAM-FREE encoders (all Route-A arms + the one-hop MeanPool/Hash/HRR arms) take the
// MULTI-THREADED path: train_batch's win_sentinel per-window binding arrays (2026-07-17 -- see
// core.hpp's train_batch comment), all cores instead of one, ~10-15x faster iteration. LEARNED-encoder
// arms (ConvPool: enc_w_grad has no per-thread reduction) keep the single-threaded manual loop -- the
// documented CharEncoder-class limit. EpochWalk drives window selection identically in both paths.
void train_steps(const ps::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng, int window,
                 bool use_override, sub0::SlotEncoding enc, DocWalk& walk,
                 std::vector<float>* enc_w, std::vector<float>* enc_w_grad, EncAdam* ea) {
    const bool learned = (enc_w != nullptr);
    std::vector<int> masked_tgt(static_cast<std::size_t>(window));
    std::vector<std::size_t> starts(static_cast<std::size_t>(kBatch));
    std::vector<int>         lens(static_cast<std::size_t>(kBatch));
    std::vector<sub0::SentinelBindings>        sbs(static_cast<std::size_t>(kBatch));
    std::vector<const sub0::SentinelBindings*> sb_ptrs(static_cast<std::size_t>(kBatch));
    for (int s = 0; s < steps; ++s) {
        opt.zero_grad();
        if (enc_w_grad) std::fill(enc_w_grad->begin(), enc_w_grad->end(), 0.f);
        if (!learned) {   // multi-threaded: prepare the whole batch, hand it to train_batch
            for (int b = 0; b < kBatch; ++b) {
                const std::size_t d   = walk.next(rng);
                const std::size_t ds_ = ds.doc_starts[d], de = ds.doc_starts[d + 1];
                starts[static_cast<std::size_t>(b)] = ds_;
                lens[static_cast<std::size_t>(b)]   = static_cast<int>(std::min<std::size_t>(
                                                          static_cast<std::size_t>(window), de - ds_ - 1));
                sbs[static_cast<std::size_t>(b)] = sub0::SentinelBindings{
                    std::span<const std::vector<int>>(ds.doc_bindings[d]), kSigil, ps::HANDLE_BASE, enc };
                sb_ptrs[static_cast<std::size_t>(b)] = &sbs[static_cast<std::size_t>(b)];
            }
            (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, window, lens.data(),
                                    ds.mask.data(), nullptr, use_override ? sb_ptrs.data() : nullptr,
                                    nullptr);
            opt.step();
            continue;
        }
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t d   = walk.next(rng);
            const std::size_t ds_ = ds.doc_starts[d], de = ds.doc_starts[d + 1];
            const int Tb = static_cast<int>(std::min<std::size_t>(
                               static_cast<std::size_t>(window), de - ds_ - 1));
            for (int i = 0; i < Tb; ++i) {
                const std::size_t p = ds_ + static_cast<std::size_t>(i) + 1;
                masked_tgt[static_cast<std::size_t>(i)] = ds.mask[p] ? ds.tokens[p] : sub0::LOSS_IGNORE_INDEX;
            }
            sub0::SentinelBindings sb{ std::span<const std::vector<int>>(ds.doc_bindings[d]),
                                       kSigil, ps::HANDLE_BASE, enc };
            if (enc_w)      sb.enc_w      = enc_w->data();
            if (enc_w_grad) sb.enc_w_grad = enc_w_grad->data();
            if (use_override) sub0::set_sentinel_bindings(&sb);
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(ds.tokens.data() + ds_, Tb);
            sub0::Node* loss   = sub0::cross_entropy(logits, masked_tgt.data());
            sub0::backward(loss, 1.f / static_cast<float>(kBatch));
            if (use_override) sub0::set_sentinel_bindings(nullptr);
        }
        sub0::reduce_gradients();
        opt.step();
        if (ea && enc_w && enc_w_grad) {
            ++ea->t;
            const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
            const float bc1 = 1.f - std::pow(b1, static_cast<float>(ea->t));
            const float bc2 = 1.f - std::pow(b2, static_cast<float>(ea->t));
            for (std::size_t i = 0; i < enc_w->size(); ++i) {
                const float g = (*enc_w_grad)[i];
                ea->m[i] = b1 * ea->m[i] + (1.f - b1) * g;
                ea->v[i] = b2 * ea->v[i] + (1.f - b2) * g * g;
                (*enc_w)[i] -= kLr * (ea->m[i] / bc1) / (std::sqrt(ea->v[i] / bc2) + eps);
            }
        }
    }
}

void run_pair_experiment(const char* arm_name, int kK, bool use_override, sub0::SlotEncoding enc,
                         std::initializer_list<unsigned> seeds) {
    // Trace length: 2K (pair preamble) + 2 (query) + 4 (SEP + pair + EOS) = 2K+6; generous margin.
    const int window = std::min(SEQ_LEN, 2 * kK + 16);
    REQUIRE(window >= 2 * kK + 6);

    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    REQUIRE(static_cast<int>(split.drilled.size()) > kK);

    ps::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ps::Dataset ds = ps::build_dataset_select_pair(kSigil, split, kK, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));

    const bool learned = (enc == sub0::SlotEncoding::ConvPool) && use_override;
    const int C = D_MODEL;

    std::string report = "\n=== pairspike SELECT-via-pair (arm=" + std::string(arm_name) +
                         ", K=" + std::to_string(kK) + " entities, sigil=TOK_RESERVED_9, sampling=epoch-walk"
                         ", VOCAB=" + std::to_string(VOCAB) + " D_MODEL=" + std::to_string(D_MODEL) +
                         " window=" + std::to_string(window) +
                         " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) +
                         " seeds=" + std::to_string(seeds.size()) + ") ===\n"
                         "  (first-letter query over K pair-bound entities; the model answers by EMITTING\n"
                         "   the right <sigil> <handle> pair -- chance ~= 1/K; entity spellings\n"
                         "   are NEVER in the stream, only in the override embedding)\n";

    double sum_final_held = 0.0;
    for (unsigned seed : seeds) {
        sub0::build_model();
        reset_opt_state();
        std::vector<float> enc_w, enc_w_grad;
        EncAdam ea;
        if (learned) {
            enc_w.resize(static_cast<std::size_t>(2) * C * C);
            enc_w_grad.resize(enc_w.size());
            std::mt19937 wr(seed * 1000u + 7u);
            std::normal_distribution<float> wnd(0.f, 1.f / std::sqrt(static_cast<float>(C)));
            for (float& x : enc_w) x = wnd(wr);
            ea.m.assign(enc_w.size(), 0.f); ea.v.assign(enc_w.size(), 0.f);
        }

        report += "  -- seed " + std::to_string(seed) + " --\n";
        sub0::AdamW opt(kLr);
        std::mt19937 rng(seed);
        DocWalk walk(ds.doc_starts.size() - 1);
        double last_held = 0.0;
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(ds, opt, kStepsPerEval, rng, window, use_override, enc, walk,
                        learned ? &enc_w : nullptr, learned ? &enc_w_grad : nullptr,
                        learned ? &ea : nullptr);
            const Acc d = eval_select(kK, split.drilled, use_override, enc,
                                      learned ? enc_w.data() : nullptr, /*seed=*/7);
            const Acc h = eval_select(kK, split.held_out, use_override, enc,
                                      learned ? enc_w.data() : nullptr, /*seed=*/11);
            last_held = h.rate();
            char line[192];
            std::snprintf(line, sizeof line, "  step %5d | DRILLED sel=%.3f | HELD-OUT sel=%.3f\n",
                          (r + 1) * kStepsPerEval, d.rate(), h.rate());
            report += line;
        }
        REQUIRE(std::isfinite(last_held));
        sum_final_held += last_held;
    }
    char mean_line[128];
    std::snprintf(mean_line, sizeof mean_line, "  MEAN final held-out over %zu seed(s): %.3f\n",
                  seeds.size(), sum_final_held / static_cast<double>(seeds.size()));
    report += mean_line;
    WARN(report);
}

}  // namespace

TEST_CASE("pairspike: select-via-pair K=12, OVERRIDE/ConvPool (the shootout-selected default)",
         "[.pairspike]") {
    run_pair_experiment("OVERRIDE/ConvPool", /*kK=*/12, /*use_override=*/true, sub0::SlotEncoding::ConvPool,
                        {1, 2, 3});
}

TEST_CASE("pairspike: select-via-pair K=12, OVERRIDE/HRR (parameter-free order-sensitive fallback)",
         "[.pairspike]") {
    run_pair_experiment("OVERRIDE/HRR", /*kK=*/12, /*use_override=*/true, sub0::SlotEncoding::HRR, {1, 2, 3});
}

TEST_CASE("pairspike: select-via-pair K=12, NO-OVERRIDE control (chance floor -- the preamble carries "
         "no handle<->content information without the embedding override)",
         "[.pairspike]") {
    run_pair_experiment("NO-OVERRIDE", /*kK=*/12, /*use_override=*/false, sub0::SlotEncoding::MeanPool,
                        {1, 2, 3});
}

// ============================================================================================================
// ROUTE-A (resolve-through-uncombine) arms -- user-directed after the one-hop K=12 null: "first letter
// queries should track through the uncombine." See pairspike.hpp's select_pair_cot_task for the trace
// shape (the proven local-grounded CoT scaffold, applied to pairs). Content enters via interceptor-
// injected bytes, so the pair needs NO embedding override for correctness -- the override arm here
// measures whether it ADDS anything on top of the working scaffold, not whether it is load-bearing.

// The eval-side binding table the uncombine interceptor consults: expand(handle) -> the entity's bytes.
// The sigil preceding the handle is irrelevant to expand (UNCOMBINE targets the immediately-preceding
// token, which is the handle itself).
struct PairTable {
    std::vector<std::vector<int>> bindings;   // bindings[i] = fragments of handle_token(i)
    void reset() { bindings.clear(); }
    std::vector<int> expand(int token) const {
        const int i = token - ps::HANDLE_BASE;
        if (i >= 0 && i < static_cast<int>(bindings.size()) && !bindings[static_cast<std::size_t>(i)].empty())
            return bindings[static_cast<std::size_t>(i)];
        return {token};
    }
    std::vector<int> combine(const std::vector<int>& frags) const { return frags; }   // never minted here
};

bool run_task_cot(const ps::Task& k, bool use_override, sub0::SlotEncoding enc) {
    PairTable ops;
    for (const std::string& oov : k.binds) ops.bindings.push_back(ss::oov_bytes(oov));
    std::vector<std::vector<int>> tbl = ops.bindings;
    sub0::SentinelBindings sb{ std::span<const std::vector<int>>(tbl), kSigil, ps::HANDLE_BASE, enc };
    if (use_override) sub0::set_sentinel_bindings(&sb);
    std::vector<int> ctx = k.prompt;
    std::mt19937 rng(0);
    const int n = std::min(SEQ_LEN - static_cast<int>(ctx.size()) - 1,
                           static_cast<int>(k.binds.size()) * 12 + 16);   // full resolve trace + answer
    sub0::kv_decode_generate(ctx, n, /*temp=*/1.f, /*topk=*/1, rng, cas::TOK_EOS, /*use_gpu=*/false,
                             /*on_token=*/{},
                             [&](int t) { return ops.expand(t); },
                             [&](const std::vector<int>& f) { return ops.combine(f); });
    if (use_override) sub0::set_sentinel_bindings(nullptr);
    const auto [sig, handle] = answer_pair(ctx);
    return sig == kSigil && handle == k.answer_handle;
}

Acc eval_select_cot(int K, const std::vector<std::string>& oovs, bool use_override, sub0::SlotEncoding enc,
                    unsigned seed) {
    std::vector<ps::Task> tasks;   // serial generation, parallel scoring -- see eval_select's comment
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi)
        tasks.push_back(ps::pick_select_pair_cot_task(kSigil, oovs, qi, K, rng));
    int ok = 0;
    #pragma omp parallel for reduction(+ : ok) schedule(dynamic)
    for (int i = 0; i < static_cast<int>(tasks.size()); ++i)
        ok += run_task_cot(tasks[static_cast<std::size_t>(i)], use_override, enc);
    return { ok, static_cast<int>(tasks.size()) };
}

void run_pair_cot_experiment(const char* arm_name, int kK, bool use_override, sub0::SlotEncoding enc,
                             std::initializer_list<unsigned> seeds) {
    // Per candidate: pair(2) + UNCOMBINE + up to 6 bytes + END + restate + verdict ~= 12; plus query(2)
    // and answer(4). Generous margin, capped at SEQ_LEN.
    const int window = std::min(SEQ_LEN, kK * 12 + 24);
    REQUIRE(window >= kK * 12 + 6);

    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    REQUIRE(static_cast<int>(split.drilled.size()) > kK);

    ps::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ps::Dataset ds = ps::build_dataset_select_pair_cot(kSigil, split, kK, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));

    std::string report = "\n=== pairspike SELECT-via-pair ROUTE-A/uncombine-CoT (arm=" + std::string(arm_name) +
                         ", K=" + std::to_string(kK) + " entities, sigil=TOK_RESERVED_9, sampling=epoch-walk"
                         ", VOCAB=" + std::to_string(VOCAB) + " D_MODEL=" + std::to_string(D_MODEL) +
                         " window=" + std::to_string(window) +
                         " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) +
                         " seeds=" + std::to_string(seeds.size()) + ") ===\n"
                         "  (resolve every pair via live UNCOMBINE, restate the query locally, verdict,\n"
                         "   then COPY the matching pair -- the proven local-CoT scaffold; chance ~= 1/K)\n";

    // THREE-PILLAR reporting (correctness / performance / memory -- standing shootout policy,
    // 2026-07-17): accuracy alone can crown an arm that pays a hidden step-cost or memory tax, so every
    // arm reports wall-clock alongside accuracy; memory is a static per-encoder fact (param-free arms
    // add 0 bytes; a learned encoder adds enc_w + grads + moments) stated in the summary.
    double sum_final_held = 0.0, sum_train_s = 0.0, sum_eval_s = 0.0;
    for (unsigned seed : seeds) {
        sub0::build_model();
        reset_opt_state();
        report += "  -- seed " + std::to_string(seed) + " --\n";
        sub0::AdamW opt(kLr);
        std::mt19937 rng(seed);
        DocWalk walk(ds.doc_starts.size() - 1);
        double last_held = 0.0;
        for (int r = 0; r < kEvalRounds; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            train_steps(ds, opt, kStepsPerEval, rng, window, use_override, enc, walk,
                        nullptr, nullptr, nullptr);
            const auto t1 = std::chrono::steady_clock::now();
            const Acc d = eval_select_cot(kK, split.drilled,  use_override, enc, /*seed=*/7);
            const Acc h = eval_select_cot(kK, split.held_out, use_override, enc, /*seed=*/11);
            const auto t2 = std::chrono::steady_clock::now();
            sum_train_s += std::chrono::duration<double>(t1 - t0).count();
            sum_eval_s  += std::chrono::duration<double>(t2 - t1).count();
            last_held = h.rate();
            char line[192];
            std::snprintf(line, sizeof line, "  step %5d | DRILLED sel=%.3f | HELD-OUT sel=%.3f\n",
                          (r + 1) * kStepsPerEval, d.rate(), h.rate());
            report += line;
        }
        REQUIRE(std::isfinite(last_held));
        sum_final_held += last_held;
    }
    const double n_seeds = static_cast<double>(seeds.size());
    char mean_line[256];
    std::snprintf(mean_line, sizeof mean_line,
                  "  MEAN final held-out over %zu seed(s): %.3f | train %.0fs/seed, eval %.0fs/seed | "
                  "encoder mem: +0 B (param-free)\n",
                  seeds.size(), sum_final_held / n_seeds, sum_train_s / n_seeds, sum_eval_s / n_seeds);
    report += mean_line;
    WARN(report);
}

// K=3 DIAGNOSTIC (added after the K=12 NULL result -- every arm at chance, drilled included): the
// dedicated-id content_select + HRR precedent scored 0.744 held-out at K=3 (chance 0.333). If pairs ALSO
// clear chance decisively here, the pair mechanism works and K=12 was a budget/K-scaling problem (the
// basin-entry lesson: capability-edge tasks sit at chance until a late transition); if pairs sit at
// chance where dedicated ids succeeded, the pair OUTPUT interface (handle prediction via static lm_head
// rows) is the wall, and the pointer-style-head fallback becomes the path. Own tag, runs separately.
TEST_CASE("pairspike: select-via-pair K=3 DIAGNOSTIC, OVERRIDE/HRR (vs content_select's 0.744 precedent)",
         "[.pairspike_k3]") {
    run_pair_experiment("OVERRIDE/HRR-K3", /*kK=*/3, /*use_override=*/true, sub0::SlotEncoding::HRR, {1, 2, 3});
}

TEST_CASE("pairspike: select-via-pair K=3 DIAGNOSTIC, NO-OVERRIDE control",
         "[.pairspike_k3]") {
    run_pair_experiment("NO-OVERRIDE-K3", /*kK=*/3, /*use_override=*/false, sub0::SlotEncoding::MeanPool,
                        {1, 2, 3});
}

TEST_CASE("pairspike ROUTE-A: resolve-through-uncombine CoT, K=12, NO override (the pair as a plain "
         "symbol -- content enters via injected bytes, the proven scaffold)",
         "[.pairspike_cot]") {
    run_pair_cot_experiment("ROUTE-A/no-override", /*kK=*/12, /*use_override=*/false,
                            sub0::SlotEncoding::MeanPool, {1, 2, 3});
}

TEST_CASE("pairspike ROUTE-A: resolve-through-uncombine CoT, K=12, WITH HRR override (does the "
         "content-carrying embedding ADD anything on top of the working scaffold?)",
         "[.pairspike_cot]") {
    run_pair_cot_experiment("ROUTE-A/HRR-override", /*kK=*/12, /*use_override=*/true,
                            sub0::SlotEncoding::HRR, {1, 2, 3});
}
