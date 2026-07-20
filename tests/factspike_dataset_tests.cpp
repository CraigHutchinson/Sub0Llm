// factspike_dataset_tests.cpp -- Phase 0 (docs/FACTSPIKE.md): a free, model-free, always-run leakage
// pre-check on the factspike FactPair generation scheme, gating any real training. Learns a SMALL
// (forced-multi-piece) in-memory tokenizer from the factspike corpus text itself -- self-contained, no
// external build config needed -- then verifies no single vocab piece disproportionately predicts a
// drilled subject's assigned fact color (the real risk a design review surfaced: since content_embed
// composes a slot's embedding from the SUBJECT's OWN piece rows, if one piece happened to be shared by
// most/all subjects of one color and absent from every other color, the "scratch arm" of the real
// capstone could "succeed" via that spurious correlation instead of genuine representational transfer --
// and a held-out subject sharing that piece would falsely score well too, quietly breaking the negative
// control). Also verifies held-out subjects don't cluster near one color's piece set.

#include <catch2/catch_test_macros.hpp>

#include "sub0/factspike.hpp"
#include "sub0/scratch.hpp"
#include "sub0/unigram.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fs = sub0::factspike;
using sub0::tok::Tokenizer;

// The clean piece-id set a subject's own spelling tokenizes to (markers stripped) -- the exact basis
// content_embed would compose a slot's embedding from at eval time (sub0::detail::word_span is the same
// helper corpus_collapse/prefill_collapse use to extract a word's piece span).
std::set<int> subject_pieces(const Tokenizer& tk, const std::string& subject) {
    const std::vector<int> ctx = sub0::tok::encode(tk, subject);
    std::set<int> pieces;
    for (std::size_t i = 0; i < ctx.size(); ) {
        const auto [span_len, ids] = sub0::detail::word_span(ctx, i);
        for (int id : ids) pieces.insert(id);
        i += span_len;
    }
    return pieces;
}

}  // namespace

TEST_CASE("factspike Phase 0: no vocab piece leaks a drilled subject's fact color", "[factspike]") {
    std::mt19937_64 rng(20260719);
    constexpr int kNSubjects = 24;   // 8 colors x 3 drilled each, on average
    constexpr double kDrilledFrac = 0.75;
    const fs::FactSplit split = fs::make_fact_split(rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    const std::string corpus = fs::build_corpus_text(split.drilled, /*docs_per_fact=*/20, /*seed=*/7);
    REQUIRE_FALSE(corpus.empty());

    sub0::tok::LearnOptions opt;
    opt.vocab_target = 320;   // deliberately small: forces multi-piece subjects (Phase A verifies for real)
    const Tokenizer tk = sub0::tok::learn(corpus, opt);

    // piece id -> which colors' drilled subjects contain it, and how many distinct subjects per color.
    std::map<int, std::map<std::string, int>> piece_color_count;
    std::map<std::string, int> subjects_per_color;
    for (const fs::FactPair& fp : split.drilled) {
        ++subjects_per_color[fp.fact];
        for (int p : subject_pieces(tk, fp.subject)) ++piece_color_count[p][fp.fact];
    }

    // A piece is a LEAK if it appears in >=2 subjects of exactly one color and zero subjects of every
    // other color that also has >=2 drilled subjects (a coincidental single-subject overlap isn't
    // evidence of anything; a piece consistently tied to one color's whole drilled set is).
    std::vector<std::string> leaks;
    for (const auto& [piece, by_color] : piece_color_count) {
        if (by_color.size() != 1) continue;   // appears across >1 color -- not color-specific
        const auto& [color, count] = *by_color.begin();
        if (count >= 2 && count == subjects_per_color[color] && subjects_per_color[color] >= 2)
            leaks.push_back("piece " + std::to_string(piece) + " -> every drilled '" + color + "' subject, no other color");
    }
    if (!leaks.empty()) {
        std::string msg = "factspike Phase 0: possible color-leaking piece(s) found:\n";
        for (const std::string& l : leaks) msg += "  " + l + "\n";
        WARN(msg);
    }
    CHECK(leaks.empty());

    // Held-out subjects shouldn't share their ENTIRE piece set with one color's drilled subjects either
    // (that would let the held-out "negative control" arm falsely score well via the same leak).
    std::vector<std::string> held_out_overlaps;
    for (const fs::FactPair& ho : split.held_out) {
        const std::set<int> ho_pieces = subject_pieces(tk, ho.subject);
        for (const auto& [color, n] : subjects_per_color) {
            if (n < 2) continue;
            bool all_shared = !ho_pieces.empty();
            for (int p : ho_pieces) {
                const auto it = piece_color_count.find(p);
                if (it == piece_color_count.end() || it->second.count(color) == 0) { all_shared = false; break; }
            }
            if (all_shared)
                held_out_overlaps.push_back("held-out '" + ho.subject + "' shares its whole piece set with '" + color + "'");
        }
    }
    if (!held_out_overlaps.empty()) {
        std::string msg = "factspike Phase 0: held-out subject(s) suspiciously overlap a drilled color:\n";
        for (const std::string& l : held_out_overlaps) msg += "  " + l + "\n";
        WARN(msg);
    }
    CHECK(held_out_overlaps.empty());
}
