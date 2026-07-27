// arch_identity_tests.cpp -- numerical fingerprints that prove an "inert at its neutral setting"
// architecture axis really is inert.
//
// AGENTS.md §4 requires every new capability to default OFF and leave the default build byte-identical,
// verified by comparing the suite's actual assertion counts before and after -- not merely "no new
// failures". Counts catch a test that stopped running; they do NOT catch a forward pass that silently
// started producing different numbers while still passing every tolerance-based assertion. This file
// closes that gap for the GQA / LoopSplit / RoPE-scaling work (docs plan: Nanbeige-4.5-style features),
// each of which threads new branches through op_attn, the layer loop and the KV cache.
//
// Mechanism: fixed seed (build_model() is deterministic), fixed inputs, then a compact fingerprint of
// forward logits, reduced gradients, and a greedy decode trace. The fingerprints are PRINTED, because
// their absolute values are config-dependent (D_MODEL/N_LAYERS/VOCAB all move them) and hardcoding a
// golden constant per build dir would be brittle. The workflow is: record the printed values on the
// neutral build before a change, re-run after, diff. A drift with every new axis at its default is a
// leak into the default path.
//
// The assertions here are the config-independent half: finiteness, non-degeneracy, and run-to-run
// determinism. Those hold at ANY config, so they fail loudly on a genuine breakage without needing a
// golden file. Deliberately NOT duplicated here: the forward_one-vs-forward equivalence check and the
// finite-difference gradient check, which already live in engine_tests.cpp and are the primary guards
// for decode drift and backward correctness respectively.
//
// --- RECORDED NEUTRAL BASELINES ------------------------------------------------------------------
// Re-run with every axis at its default and compare. A drift here is a leak into the default path --
// UNLESS a commit legitimately changed the neutral configuration, in which case re-record and say so
// here. Exactly that happened once already, and leaving the old numbers in place would have made the
// next reader mistake a legitimate change for a leak:
//
//   aa64107 dropped pos_emb under RoPE. That removes a real [SEQ_LEN, D_MODEL] parameter, so both
//   suites' assertion counts fell (factspike96 by 122794 == ~5 per removed float, the per-float
//   assertion loops over the table) and the GRAD hash moved. forward and decode were unaffected:
//   the table was allocated, zeroed and never read, so it never influenced a forward value.
//
// Current, at LOOP/GQA/RoPE-scaling defaults, with parked-spike tests OFF (the default):
//
//   d196check    (d196 L11 H7 seq256 vocab16535) suite: 44009991 assertions / 125 cases
//   hd96_check   (d384 L2  H4 seq256, CUDA)      suite: 4126369 assertions / 42 cases
//
// factspike96 (d96 L8 H2 vocab493) is NO LONGER a gate and is deliberately not listed. Its 493-token
// toy vocabulary makes engine_tests' finite-difference check read ~3.8% low while every realistic-vocab
// config reads 0.9999+ (see AGENTS.md 7), and its spike is parked with its tests off by default. For a
// fast second scale use d96 L8 H2 at vocab 16535 -- d196check's corpus with dims overridden.
//
// Pre-aa64107 values, kept only to explain the delta above (do NOT compare against these):
//
//   factspike96  suite: 4918240 / 126   grad hash=74d8266eb0d3669e
//   d196check    suite: 44260967 / 127  grad hash=7b560d163d630c59
//
// The per-tensor forward/grad/decode fingerprints are printed by the test itself -- record them from a
// clean run at the config you are about to change, not from this comment, since they move with D_MODEL/
// N_LAYERS/VOCAB and so cannot be stated once for every build dir.
//
// (factspike96's suite count excludes the pre-existing, unrelated "case markers stay atomic" tokenizer
// failure -- it fails identically on a clean checkout of that toy config, verified by stash+rebuild.)

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

// FNV-1a over the raw bytes of a float span. A checksum rather than a sum: sums are insensitive to
// permutation, so a reordered layer loop (exactly what LoopSplit changes) could leave a sum untouched.
std::uint64_t fnv1a(std::span<const float> v) {
    std::uint64_t h = 1469598103934665603ull;
    for (float f : v) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(f));
        std::memcpy(&bits, &f, sizeof(bits));
        for (int b = 0; b < 4; ++b) {
            h ^= static_cast<std::uint64_t>((bits >> (8 * b)) & 0xFFu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

// Summary statistics alongside the checksum: a checksum alone says "something moved" but not how much,
// and a one-ulp fast-math difference is not the same finding as a restructured forward pass.
struct Stats {
    double sum = 0.0, absmax = 0.0;
    std::uint64_t hash = 0;
};

Stats summarize(std::span<const float> v) {
    Stats s;
    for (float f : v) {
        s.sum += static_cast<double>(f);
        s.absmax = std::max(s.absmax, static_cast<double>(std::fabs(f)));
    }
    s.hash = fnv1a(v);
    return s;
}

std::vector<int> fixed_window(int T, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> ids(static_cast<std::size_t>(T));
    for (int& x : ids) x = tok(rng);
    return ids;
}

}  // namespace

TEST_CASE("arch identity: forward/backward/decode fingerprints for the neutral configuration",
         "[arch][identity]") {
    constexpr int T = 16;
    const std::vector<int> ids = fixed_window(T, 4242u);
    std::vector<int> tgt = fixed_window(T, 2424u);

    sub0::build_model();

    // --- forward logits -----------------------------------------------------------------------
    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(ids.data(), T);
    const Stats fwd = summarize(std::span<const float>(logits->data));

    // --- reduced gradients --------------------------------------------------------------------
    // train_batch's own single-window path, so this covers the same reduction the real training loop
    // uses rather than a hand-rolled forward/backward that could drift from it.
    std::vector<int> data(ids.begin(), ids.end());
    data.push_back(tgt[0]);                       // train_batch reads targets as data[start+1 ...]
    std::size_t start = 0;
    (void)sub0::train_batch(data.data(), &start, /*batch=*/1, T - 1);
    const Stats grad = summarize(std::span<const float>(sub0::grad_ptr(), sub0::trainable_floats()));

    // --- greedy decode trace ------------------------------------------------------------------
    // Exercises forward_one + the KV cache, the path GQA's KV stride and LoopSplit's per-execution
    // cache slots both change.
    sub0::kv_reset();
    const float* dec = nullptr;
    for (int i = 0; i < T; ++i) dec = sub0::forward_one(ids[static_cast<std::size_t>(i)], i);
    const Stats decode = summarize(std::span<const float>(dec, VOCAB));

    char buf[512];
    std::snprintf(buf, sizeof buf,
                 "\n=== arch identity fingerprint (d%d L%d H%d seq%d vocab%d) ===\n"
                 "  forward: sum=%.6f absmax=%.6f hash=%016llx\n"
                 "  grad:    sum=%.6f absmax=%.6f hash=%016llx\n"
                 "  decode:  sum=%.6f absmax=%.6f hash=%016llx\n"
                 "  (record these on the neutral build; a drift after adding an axis that defaults\n"
                 "   OFF means it leaked into the default path -- see AGENTS.md 4)\n",
                 D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                 fwd.sum, fwd.absmax, static_cast<unsigned long long>(fwd.hash),
                 grad.sum, grad.absmax, static_cast<unsigned long long>(grad.hash),
                 decode.sum, decode.absmax, static_cast<unsigned long long>(decode.hash));
    WARN(std::string(buf));

    // Config-independent assertions: these hold at any dims, so they catch a real breakage without a
    // golden file. A zero absmax means the pass produced nothing; a non-finite value means it diverged.
    CHECK(std::isfinite(fwd.sum));    CHECK(fwd.absmax > 0.0);
    CHECK(std::isfinite(grad.sum));   CHECK(grad.absmax > 0.0);
    CHECK(std::isfinite(decode.sum)); CHECK(decode.absmax > 0.0);
}

TEST_CASE("arch identity: repeated forward and decode are bit-identical", "[arch][identity]") {
    constexpr int T = 12;
    const std::vector<int> ids = fixed_window(T, 77u);
    sub0::build_model();

    sub0::graph_reset();
    sub0::Node* a = sub0::forward(ids.data(), T);
    const std::uint64_t h1 = fnv1a(std::span<const float>(a->data));
    sub0::graph_reset();
    sub0::Node* b = sub0::forward(ids.data(), T);
    const std::uint64_t h2 = fnv1a(std::span<const float>(b->data));
    CHECK(h1 == h2);   // bit-identical, not approx: same inputs, same weights, same op order

    sub0::kv_reset();
    const float* d1 = nullptr;
    for (int i = 0; i < T; ++i) d1 = sub0::forward_one(ids[static_cast<std::size_t>(i)], i);
    const std::uint64_t k1 = fnv1a(std::span<const float>(d1, VOCAB));
    sub0::kv_reset();
    const float* d2 = nullptr;
    for (int i = 0; i < T; ++i) d2 = sub0::forward_one(ids[static_cast<std::size_t>(i)], i);
    const std::uint64_t k2 = fnv1a(std::span<const float>(d2, VOCAB));
    CHECK(k1 == k2);   // kv_reset() must fully clear per-generation state
}
