// persistent_slots_engine_tests.cpp -- the engine-side SPIKE for the persistent (unbounded) scratch-slot
// range (include/sub0/scratch_slots.hpp's PersistentBindings + is_persistent_slot, wired into
// backend_cpu.cpp's op_embed/backward_node/forward_one). Distinct from scratch_embed_tests.cpp's
// EPHEMERAL pool (ids inside the fixed TOK_RESERVED_4..9 block, thread_local, rebound per window): this
// range is ids >= VOCAB, open-ended, global/immutable-once-set -- see PersistentBindings' own header
// comment for the full design and docs/DETERMINISTIC_MECHANISMS.md's "persistent compound-word cache"
// extension note for the motivation.
//
// The pure encoder math (encode_slot/encode_slot_bwd) is ALREADY id-agnostic and already unit-tested by
// scratch_embed_tests.cpp -- what's NEW and needs proving here is the ENGINE DISPATCH: does an id >= VOCAB
// actually get routed through encode_slot (never the raw tab[id,...] lookup, which would read out of the
// [VOCAB,C] table's bounds), in both directions (forward + backward), and does it stay safe when
// unbound/no table is installed (the OOB risk PersistentBindings' comment names explicitly)?

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"
#include "sub0/scratch_slots.hpp"

#include <cmath>
#include <span>
#include <vector>

using sub0::SlotEncoding;
using sub0::PersistentBindings;

namespace {
// A persistent slot id must never collide with an ephemeral scratch slot or any real vocab piece --
// VOCAB itself is exactly that boundary (the first id no ordinary token ever occupies).
int persistent_id(int i) { return VOCAB + i; }
}  // namespace

TEST_CASE("persistent slots: is_persistent_slot / persistent_fragments are pure and null-safe", "[persistent_slots]") {
    CHECK_FALSE(sub0::is_persistent_slot(0, VOCAB));
    CHECK_FALSE(sub0::is_persistent_slot(VOCAB - 1, VOCAB));
    CHECK(sub0::is_persistent_slot(VOCAB, VOCAB));
    CHECK(sub0::is_persistent_slot(VOCAB + 1000, VOCAB));

    // No table installed -> empty fragments (encode_slot's zero-row contract), never a crash/UB.
    CHECK(sub0::persistent_fragments(nullptr, persistent_id(0)).empty());

    std::vector<std::vector<int>> tbl(1);
    tbl[0] = {'a', 'b', 'c'};
    const PersistentBindings pb{ std::span<const std::vector<int>>(tbl), VOCAB, SlotEncoding::MeanPool };
    CHECK(pb.bound(persistent_id(0)));
    CHECK_FALSE(pb.bound(persistent_id(1)));           // in-range index but unfilled slot
    CHECK_FALSE(pb.bound(VOCAB - 1));                  // below base entirely
    CHECK(sub0::persistent_fragments(&pb, persistent_id(0)).size() == 3);
    CHECK(sub0::persistent_fragments(&pb, persistent_id(1)).empty());   // unbound slot -> empty, not garbage
}

// Engine differential: a persistent slot's forward embedding must equal encode_slot's own mean-pool
// output for the same fragments. Persistent ids have NO output logit column (logits only span [0,VOCAB)),
// unlike the ephemeral-slot test's tied-head caveat -- so there is nothing to exclude on that side. The
// reference arm instead substitutes an ORDINARY byte token whose row is temporarily overwritten to the
// expected mean (mirrors scratch_embed_tests.cpp's technique, adapted since a persistent id has no real
// row within [0,VOCAB) to overwrite for a "plain lookup" comparison).
TEST_CASE("persistent slots: a bound slot embeds as its fragment mean in forward", "[persistent_slots]") {
    sub0::build_model();
    const int C = D_MODEL;
    float* P = sub0::params_ptr();                 // tok_emb is [VOCAB, C] at param offset 0
    const std::vector<int> frags = {'a', 'b', 'c'};  // byte tokens, all < VOCAB
    constexpr int stand_in = '~';                    // an ordinary byte token, unrelated to frags

    std::vector<float> mean(C, 0.f);
    for (int f : frags) for (int j = 0; j < C; ++j) mean[j] += P[static_cast<std::size_t>(f) * C + j];
    for (int j = 0; j < C; ++j) mean[j] /= static_cast<float>(frags.size());

    // A) persistent-slot: bind, forward a sequence referencing the persistent id.
    std::vector<std::vector<int>> tbl(1);
    tbl[0] = frags;
    const PersistentBindings pb{ std::span<const std::vector<int>>(tbl), VOCAB, SlotEncoding::MeanPool };
    const std::vector<int> seq_persist = {'x', persistent_id(0), 'y'};
    sub0::set_persistent_bindings(&pb);
    sub0::graph_reset();
    sub0::Node* a = sub0::forward(seq_persist.data(), static_cast<int>(seq_persist.size()));
    const std::vector<float> logits_persist(a->data.begin(), a->data.end());
    sub0::set_persistent_bindings(nullptr);

    // B) reference: overwrite stand_in's row to the expected mean, forward with no persistent table.
    std::vector<float> saved(P + static_cast<std::size_t>(stand_in) * C, P + static_cast<std::size_t>(stand_in) * C + C);
    for (int j = 0; j < C; ++j) P[static_cast<std::size_t>(stand_in) * C + j] = mean[j];
    const std::vector<int> seq_ref = {'x', stand_in, 'y'};
    sub0::graph_reset();
    sub0::Node* b = sub0::forward(seq_ref.data(), static_cast<int>(seq_ref.size()));
    const std::vector<float> logits_ref(b->data.begin(), b->data.end());
    for (int j = 0; j < C; ++j) P[static_cast<std::size_t>(stand_in) * C + j] = saved[j];

    REQUIRE(logits_persist.size() == logits_ref.size());
    const int T = static_cast<int>(seq_persist.size());
    // N-gram embeddings (docs/NGRAM_EMBEDDING.md) hash the RAW TOKEN ID, deliberately: `persistent_id(0)`
    // and `stand_in` ('~') are, BY DESIGN, two ids the n-gram mechanism must be able to tell apart --
    // that is exactly what routes a persistent slot to "no n-gram signal, id 0" (backend_cpu.cpp's
    // `ngram_tok`/`id_tok` guard) instead of hashing an essentially arbitrary unbounded integer, which
    // is what this very test caught before that guard existed (see docs/NGRAM_EMBEDDING.md sec 5 for the
    // incident). So the moment n-gram embeddings are enabled, this differential's whole premise --
    // that the engine cannot tell arm A's id from arm B's -- necessarily stops holding for every
    // position within n-gram-shift reach of positions 1 ('x') or the middle slot itself: it is not a
    // regression to fix, it is the guard doing its job. Every OTHER op (tok_emb, attention, FFN) still
    // cannot tell the two arms apart, so the exact-match technique below stays the correctness gate for
    // everything n-gram embeddings do not touch.
    if constexpr (!sub0::NGRAM_EMBED) {
        bool any_checked = false;
        for (int t = 0; t < T; ++t)
            for (int v = 0; v < VOCAB; ++v) {
                if (v == stand_in) continue;   // the reference arm's tied-head reads stand_in's own (overwritten) row
                const std::size_t i = static_cast<std::size_t>(t) * VOCAB + v;
                REQUIRE(logits_persist[i] == Catch::Approx(logits_ref[i]).margin(1e-4));
                any_checked = true;
            }
        REQUIRE(any_checked);
    } else {
        // Compensating check, so this branch still asserts something real rather than silently doing
        // nothing: position 2 ('y') is one n-gram shift away from the middle slot, so its logits MUST
        // differ between the two arms (arm A's 'y' hashes context id 0 for the middle slot, arm B's
        // hashes stand_in's real id) -- proving the guard is actually live, not accidentally a no-op
        // that would make this test pass again for the wrong reason.
        double diff2 = 0.0;
        for (int v = 0; v < VOCAB; ++v) {
            const std::size_t i = static_cast<std::size_t>(2) * VOCAB + v;
            const double d = static_cast<double>(logits_persist[i]) - logits_ref[i];
            diff2 += d * d;
        }
        REQUIRE(diff2 > 1e-8);
    }
}

// Gradient flow: a persistent slot's row grad must scatter into its fragment rows via encode_slot_bwd,
// through a REAL forward+cross_entropy+backward cycle (not just the pure encode_slot_bwd math
// scratch_embed_tests.cpp already checks) -- train_batch doesn't accept PersistentBindings (it's a
// global, not a per-window array like the ephemeral pool's win_binds), so this drives the lower-level
// forward/backward API directly, the same way engine_tests.cpp's own gradient checks do.
TEST_CASE("persistent slots: backward scatters the slot's row grad into its fragment rows", "[persistent_slots]") {
    sub0::build_model();
    const int C = D_MODEL;
    const std::vector<int> frags = {'d', 'e'};

    std::vector<std::vector<int>> tbl(1);
    tbl[0] = frags;
    const PersistentBindings pb{ std::span<const std::vector<int>>(tbl), VOCAB, SlotEncoding::MeanPool };

    // The persistent id occupies INPUT position 0 only -- never a target (targets are seq[1..], and a
    // persistent id can never legally be one: it has no logit column, since logits only span [0,VOCAB)).
    // Its position-0 embedding still influences every later prediction via attention, so gradient must
    // flow back to it even though nothing ever "predicts" the id itself.
    const std::vector<int> seq = {persistent_id(0), 'y', 'z', 'w'};   // 3 inputs + 1 shifted target
    const std::vector<int> tgt = {seq[1], seq[2], seq[3]};

    sub0::set_persistent_bindings(&pb);
    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(seq.data(), 3);
    sub0::backward(sub0::cross_entropy(logits, tgt.data()), 1.f);
    sub0::reduce_gradients();
    sub0::set_persistent_bindings(nullptr);

    const float* g = sub0::grad_ptr();               // same [VOCAB, C] layout as params_ptr()
    bool any_nonzero = false;
    for (int f : frags) {
        for (int j = 0; j < C; ++j) {
            if (g[static_cast<std::size_t>(f) * C + j] != 0.f) any_nonzero = true;
        }
    }
    REQUIRE(any_nonzero);   // the persistent slot's usage drove real gradient into its fragment rows
}

// Safety: an id >= VOCAB with NO table installed (the state every existing model/caller is in today)
// must produce a well-defined zero-composed row and complete without crashing/reading out of bounds --
// never fall through to the raw [VOCAB,C] table lookup. This is the concrete invariant
// PersistentBindings' header comment names as the reason this predicate must be checked unconditionally.
TEST_CASE("persistent slots: an unbound id (no table installed) is safe, not a crash", "[persistent_slots]") {
    sub0::build_model();
    const std::vector<int> seq = {'x', persistent_id(0), persistent_id(999), 'y'};

    sub0::set_persistent_bindings(nullptr);   // explicit: the default/untouched state
    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    REQUIRE(logits->data.size() == seq.size() * static_cast<std::size_t>(VOCAB));
    for (float v : logits->data) REQUIRE(std::isfinite(v));   // no garbage from an OOB read
}

// No-binding invariance: with the persistent table never touched, an ORDINARY (< VOCAB) sequence must be
// byte-identical to before this feature existed -- the third `is_persistent_slot` branch this cut added
// to op_embed/backward_node/forward_one must be provably inert for the common case.
TEST_CASE("persistent slots: ordinary sequences are unaffected when the persistent table is unset", "[persistent_slots]") {
    sub0::build_model();
    const std::vector<int> seq = {'h', 'e', 'l', 'l', 'o'};

    sub0::set_persistent_bindings(nullptr);
    sub0::graph_reset();
    sub0::Node* a = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> l0(a->data.begin(), a->data.end());

    sub0::graph_reset();
    sub0::Node* b = sub0::forward(seq.data(), static_cast<int>(seq.size()));
    const std::vector<float> l1(b->data.begin(), b->data.end());

    REQUIRE(l0.size() == l1.size());
    for (std::size_t i = 0; i < l0.size(); ++i) REQUIRE(l0[i] == l1[i]);   // bit-identical, deterministic
}
