// sentinel_pair_engine_tests.cpp -- engine differentials for the sentinel-PAIR embedding override
// (scratch_slots.hpp's SentinelBindings + backend_cpu.cpp's `ids[t-1] == sigil` dispatch in
// op_embed/backward/forward_one). Mirrors persistent_slots_engine_tests.cpp's own methodology: fast,
// training-free, in the DEFAULT suite -- proving the dispatch fires (and only fires) where designed,
// independently of any spike curriculum's training outcome. Added after the pairspike K=3 one-hop A/B
// produced bit-identical override-vs-control trajectories (suspicious enough to demand a direct proof
// the mechanism engages -- see the pairspike logs / project history 2026-07-17).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"
#include "sub0/casing.hpp"
#include "sub0/scratch_slots.hpp"

#include <cmath>
#include <span>
#include <vector>

using sub0::SentinelBindings;
using sub0::SlotEncoding;
namespace cas = sub0::casing;

namespace {
constexpr int kSigil = cas::TOK_RESERVED_9;   // the pairspike sigil (spike-commandeered reserved id)
}

TEST_CASE("sentinel pair: bound/fragments are pure and index correctly", "[sentinel_pair]") {
    std::vector<std::vector<int>> tbl(2);
    tbl[0] = {'x', 'y', 'z'};
    // tbl[1] left empty -> handle 'b' is declared-but-unbound
    const SentinelBindings sb{ std::span<const std::vector<int>>(tbl), kSigil, 'a', SlotEncoding::MeanPool };
    CHECK(sb.bound('a'));
    CHECK_FALSE(sb.bound('b'));       // in-range index but empty fragments
    CHECK_FALSE(sb.bound('c'));       // past the table
    CHECK_FALSE(sb.bound('a' - 1));   // below the base
    CHECK(sb.fragments('a').size() == 3);
    CHECK(sb.fragments('c').empty());
}

TEST_CASE("sentinel pair: the token AFTER the sigil embeds from its binding (forward differential)",
         "[sentinel_pair]") {
    sub0::build_model();
    std::vector<std::vector<int>> tbl(1);
    tbl[0] = {'x', 'y', 'z'};
    const SentinelBindings sb{ std::span<const std::vector<int>>(tbl), kSigil, 'a', SlotEncoding::MeanPool };

    // The pair [sigil 'a'] mid-sequence: with the table installed, position 2 ('a') must embed as the
    // fragment mean -- so the logits MUST differ from the uninstalled run of the SAME sequence.
    const std::vector<int> seq = {'q', kSigil, 'a', 'w'};

    sub0::set_sentinel_bindings(&sb);
    sub0::graph_reset();
    sub0::Node* with = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> l_with(with->data.begin(), with->data.end());
    sub0::set_sentinel_bindings(nullptr);

    sub0::graph_reset();
    sub0::Node* without = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> l_without(without->data.begin(), without->data.end());

    REQUIRE(l_with.size() == l_without.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < l_with.size(); ++i)
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(l_with[i] - l_without[i])));
    REQUIRE(max_diff > 1e-4);   // the override engaged: same tokens, different computation

    // Same sequence WITHOUT a preceding sigil for the 'a': the override must NOT fire ('a' at position 0,
    // and 'a' preceded by an ordinary byte) -- installed table, bit-identical to uninstalled.
    const std::vector<int> plain = {'a', 'q', 'a', 'w'};
    sub0::set_sentinel_bindings(&sb);
    sub0::graph_reset();
    sub0::Node* p1 = sub0::forward(plain.data(), static_cast<int>(plain.size()));
    const std::vector<float> lp1(p1->data.begin(), p1->data.end());
    sub0::set_sentinel_bindings(nullptr);
    sub0::graph_reset();
    sub0::Node* p2 = sub0::forward(plain.data(), static_cast<int>(plain.size()));
    const std::vector<float> lp2(p2->data.begin(), p2->data.end());
    REQUIRE(lp1.size() == lp2.size());
    for (std::size_t i = 0; i < lp1.size(); ++i) REQUIRE(lp1[i] == lp2[i]);
}

TEST_CASE("sentinel pair: backward scatters the overridden position's grad into the fragment rows",
         "[sentinel_pair]") {
    sub0::build_model();
    std::vector<std::vector<int>> tbl(1);
    tbl[0] = {'x', 'y'};
    const SentinelBindings sb{ std::span<const std::vector<int>>(tbl), kSigil, 'a', SlotEncoding::MeanPool };

    // 'x'/'y' appear ONLY as fragments (never as sequence tokens), so any gradient in their rows can
    // only have arrived through encode_slot_bwd's scatter from the overridden 'a' position.
    const std::vector<int> seq = {kSigil, 'a', 'q', 'w'};
    const std::vector<int> tgt = {seq[1], seq[2], seq[3]};

    sub0::set_sentinel_bindings(&sb);
    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(seq.data(), 3);
    sub0::backward(sub0::cross_entropy(logits, tgt.data()), 1.f);
    sub0::reduce_gradients();
    sub0::set_sentinel_bindings(nullptr);

    const float* g = sub0::grad_ptr();   // [VOCAB, C] tok_emb grads at param offset 0
    bool any = false;
    for (int f : {int('x'), int('y')})
        for (int j = 0; j < D_MODEL; ++j)
            if (g[static_cast<std::size_t>(f) * D_MODEL + j] != 0.f) any = true;
    REQUIRE(any);
}

TEST_CASE("sentinel pair: forward_one applies the override with sequential prev-token tracking",
         "[sentinel_pair]") {
    sub0::build_model();
    std::vector<std::vector<int>> tbl(1);
    tbl[0] = {'x', 'y', 'z'};
    const SentinelBindings sb{ std::span<const std::vector<int>>(tbl), kSigil, 'a', SlotEncoding::MeanPool };

    // Decode the pair sequence once WITH and once WITHOUT the table: the final logits row must differ
    // (the 'a' fed at position 1 follows the sigil fed at position 0).
    auto decode_last = [&](bool install) {
        if (install) sub0::set_sentinel_bindings(&sb);
        sub0::kv_reset();
        const std::vector<int> seq = {kSigil, 'a', 'w'};
        const float* logits = nullptr;
        for (int p = 0; p < static_cast<int>(seq.size()); ++p)
            logits = sub0::forward_one(seq[static_cast<std::size_t>(p)], p);
        std::vector<float> out(logits, logits + VOCAB);
        if (install) sub0::set_sentinel_bindings(nullptr);
        return out;
    };
    const std::vector<float> with    = decode_last(true);
    const std::vector<float> without = decode_last(false);
    double max_diff = 0.0;
    for (int v = 0; v < VOCAB; ++v)
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(with[v] - without[v])));
    REQUIRE(max_diff > 1e-4);

    // A fresh generation starting with 'a' at position 0 must NOT treat the previous generation's last
    // token as its predecessor (the sequential-tracking guarantee): identical with and without the table.
    auto decode_fresh = [&](bool install) {
        if (install) sub0::set_sentinel_bindings(&sb);
        sub0::kv_reset();
        const std::vector<int> seq = {'a', 'w'};
        const float* logits = nullptr;
        for (int p = 0; p < static_cast<int>(seq.size()); ++p)
            logits = sub0::forward_one(seq[static_cast<std::size_t>(p)], p);
        std::vector<float> out(logits, logits + VOCAB);
        if (install) sub0::set_sentinel_bindings(nullptr);
        return out;
    };
    const std::vector<float> f1 = decode_fresh(true);
    const std::vector<float> f2 = decode_fresh(false);
    for (int v = 0; v < VOCAB; ++v) REQUIRE(f1[v] == f2[v]);
}
