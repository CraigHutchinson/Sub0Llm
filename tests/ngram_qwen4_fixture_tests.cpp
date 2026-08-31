// ngram_qwen4_fixture_tests.cpp -- structural cross-check of this project's n-gram embedding "concat"
// composition against REAL Qwen/Qwen3.8-Flash-Next weight values (tests/fixtures/qwen4_preview/, landed
// on `main` by a parallel research effort -- see docs/QWEN4_PREVIEW_REFERENCE.md and
// tests/fixtures/qwen4_preview/ngram_embedding_manifest.json for provenance).
//
// What this DOES verify: that "concat" — this project's fusion mode (docs/NGRAM_EMBEDDING.md sec 2) and
// Qwen4's `Qwen4ExpTextNGramEmbedding.forward()`'s own `.flatten(-2)` -- means the SAME thing in both
// places: per-table/per-head embedding rows laid out end-to-end, table/head index major, in-row index
// minor, with no cross-table interference. That is a real, checkable structural fact about REAL model
// output (`ngram_embedding_flattened.bin` is the real module's real return value for this real input),
// not a synthetic fixture.
//
// What this does NOT verify (an exact numeric match is NOT expected, and this file makes no attempt at
// one -- see docs/NGRAM_EMBEDDING.md sec 7's "explicitly deferred" list): Qwen4's row-index hashing
// (splitmix64-style layer multipliers, XOR-mixed shifted-token-history) is a DIFFERENT scheme from this
// project's polynomial-hash-with-modular-reduction (Nanbeige's `NanbeigeNgramEmbedding`, the verified
// reference this project's Stage 1 actually implements -- see docs/NGRAM_EMBEDDING.md sec 1). Different
// table sizes, different table counts (16 heads vs this project's k*(n-1) embedders), and Qwen4 injects
// via a hyper-connection stream rather than Nanbeige's single learned `concat_proj` linear -- so there is
// no `concat_proj`-equivalent stage in this fixture to compare against, and this file does not attempt
// one. Engine-free (this project's own `sub0::` concat convention is re-expressed inline below, not
// exercised through the compiled engine) -- part of sub0_frontend_tests, per its own SUB0_SOURCE_DIR use.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef SUB0_SOURCE_DIR

namespace {
namespace fs = std::filesystem;

std::vector<float> read_f32(const fs::path& p, std::size_t expect_n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> out(expect_n);
    if (!f) return {};
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expect_n * sizeof(float)));
    if (static_cast<std::size_t>(f.gcount()) != expect_n * sizeof(float)) return {};
    return out;
}

}  // namespace

// The manifest's own declared shapes (ngram_embedding_manifest.json): 6 positions, 16 heads
// (8 bigram + 8 trigram, ngram_size=3), 160 floats per head -> 2560 = 16*160 flattened.
constexpr std::size_t kPositions   = 6;
constexpr std::size_t kHeads       = 16;
constexpr std::size_t kHeadDim     = 160;
constexpr std::size_t kFlatDim     = kHeads * kHeadDim;

TEST_CASE("n-gram concat convention matches Qwen4's real flatten(-2) output, row-for-row",
          "[ngram][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    const fs::path per_head_path  = dir / "ngram_embedding_per_head.bin";
    const fs::path flattened_path = dir / "ngram_embedding_flattened.bin";
    if (!fs::exists(per_head_path) || !fs::exists(flattened_path)) {
        WARN("Qwen4 n-gram fixtures not present at " << dir.string() << " -- skipping (optional, see "
             "docs/NGRAM_EMBEDDING.md sec 8's fixture-check note)");
        return;
    }

    const std::vector<float> per_head  = read_f32(per_head_path,  kPositions * kHeads * kHeadDim);
    const std::vector<float> flattened = read_f32(flattened_path, kPositions * kFlatDim);
    REQUIRE(per_head.size()  == kPositions * kHeads * kHeadDim);
    REQUIRE(flattened.size() == kPositions * kFlatDim);

    // This project's own "concat" convention (docs/NGRAM_EMBEDDING.md sec 3/4): embedder `e`'s D-wide
    // row occupies columns [e*D, (e+1)*D) of the concatenated vector -- table/head index MAJOR, in-row
    // index MINOR. Re-expressed here (not called through the compiled engine, which has its own,
    // differently-sized NGRAM_EMB_DIM/NGRAM_NUM_EMBEDDERS for whatever model this test binary happens to
    // be built for) so the ORDERING CONVENTION itself is checked against real reference output.
    std::size_t mismatches = 0;
    double max_abs_diff = 0.0;
    for (std::size_t pos = 0; pos < kPositions; ++pos) {
        for (std::size_t head = 0; head < kHeads; ++head) {
            for (std::size_t d = 0; d < kHeadDim; ++d) {
                const float from_per_head = per_head[(pos * kHeads + head) * kHeadDim + d];
                const float from_flat     = flattened[pos * kFlatDim + head * kHeadDim + d];
                const double diff = static_cast<double>(from_per_head) - static_cast<double>(from_flat);
                max_abs_diff = std::max(max_abs_diff, std::abs(diff));
                if (from_per_head != from_flat) ++mismatches;
            }
        }
    }
    INFO("max |per_head - flattened[concat-ordered]| = " << max_abs_diff << ", mismatches = " << mismatches
         << " / " << (kPositions * kFlatDim));
    // Exact bit-for-bit: both files are the same real float32 values, just two different views the
    // fixture extraction saved of the SAME real module output -- see the manifest's own
    // "flattened_embed[pos][:] == per_head_embed[pos].flatten()" invariant. A mismatch here would mean
    // either this project's concat ordering convention is wrong, or the fixture itself is inconsistent.
    REQUIRE(mismatches == 0);

    // Sanity: the real table's values are not all zero/degenerate (a corrupted or placeholder fixture
    // would pass the ordering check above vacuously).
    double sumabs = 0.0;
    for (float v : flattened) sumabs += std::abs(static_cast<double>(v));
    REQUIRE(sumabs > 0.0);
}

#endif  // SUB0_SOURCE_DIR
