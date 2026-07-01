// config_tests.cpp — unit tests for the pure configuration-decision logic (sub0/config_util.hpp),
// factored out of the configurator's main() so it is testable without I/O. Linked into the engine-free
// sub0_tok_tests target. Covers: auto-sizing (the ladder + the 0=auto/pin resolution), the tune-cache
// parse (defaults + each key + the derived GPU batch), and precision resolution (codes + capability).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/config_util.hpp"

#include <sstream>
#include <vector>

using namespace sub0::config;

TEST_CASE("autosize: corpus scale picks the model ladder", "[config][autosize]") {
    const std::uintmax_t MB = 1000000;
    // TinyStories-scale (22 MB) -> the compact model the user pins as ~d196.
    const ModelDims tiny = autosize(22 * MB);
    CHECK(tiny.d_model == 192);
    CHECK(tiny.n_layers == 6);
    CHECK(tiny.vocab == 4096);
    CHECK(tiny.d_model % tiny.n_heads == 0);          // must stay head-divisible

    // FineWeb-smoke-scale (1 GB) -> the larger model.
    const ModelDims fw = autosize(1024 * MB);
    CHECK(fw.d_model == 448);
    CHECK(fw.n_layers == 11);
    CHECK(fw.vocab == 16384);

    // Monotonic non-decreasing in corpus size across the rungs (bigger corpus never shrinks the model).
    int prev_d = 0, prev_v = 0;
    for (std::uintmax_t mb : {1u, 200u, 2000u, 10000u, 60000u}) {
        const ModelDims m = autosize(mb * MB);
        CHECK(m.d_model >= prev_d);
        CHECK(m.vocab   >= prev_v);
        CHECK(m.d_model % m.n_heads == 0);
        prev_d = m.d_model; prev_v = m.vocab;
    }
}

TEST_CASE("apply_autosize: 0 fields auto, nonzero fields pinned", "[config][autosize]") {
    const std::uintmax_t bytes = 22ull * 1000000;       // -> auto d192 L6 H6 seq256 v4096

    // All auto.
    const ModelDims all = apply_autosize({}, bytes);
    CHECK(all.d_model == 192);
    CHECK(all.vocab == 4096);

    // Pin d_model + heads; the rest auto-size (and the pinned values are preserved exactly).
    ModelDims pinned;
    pinned.d_model = 256; pinned.n_heads = 8;
    const ModelDims r = apply_autosize(pinned, bytes);
    CHECK(r.d_model == 256);                              // pinned
    CHECK(r.n_heads == 8);                                // pinned
    CHECK(r.n_layers == 6);                               // auto
    CHECK(r.seq_len == 256);                              // auto
    CHECK(r.vocab == 4096);                               // auto
}

TEST_CASE("model sidecar: parse/format round-trip + fill_defaults precedence", "[config][sidecar]") {
    std::istringstream s("# a comment\nd_model=256\nn_layers=8\nn_heads=8\nseq_len=512\nvocab=16000\njunk\n");
    const ModelDims m = parse_model_sidecar(s);
    CHECK(m.d_model == 256);
    CHECK(m.n_layers == 8);
    CHECK(m.vocab == 16000);

    // format -> parse round-trips the values.
    std::istringstream s2(format_model_sidecar(m));
    const ModelDims m2 = parse_model_sidecar(s2);
    CHECK(m2.d_model == 256);
    CHECK(m2.seq_len == 512);
    CHECK(m2.vocab == 16000);

    // Precedence: a nonzero (CLI) field is preserved; a 0 field is filled from the sidecar.
    ModelDims cli;
    cli.d_model = 384;                                   // pinned on the CLI
    const ModelDims filled = fill_defaults(cli, m);
    CHECK(filled.d_model == 384);                        // CLI wins
    CHECK(filled.n_layers == 8);                         // sidecar fills the rest
    CHECK(filled.vocab == 16000);
}

TEST_CASE("parse_tune_cache: defaults, keys, and the derived GPU batch", "[config][tune]") {
    // Empty cache -> hardware threads, conservative wpt, GPU batch derived from the width.
    std::istringstream empty("");
    const TuneDefaults d0 = parse_tune_cache(empty, 24);
    CHECK(d0.threads == 24);
    CHECK(d0.windows_per_thread == 4);
    CHECK(d0.gpu_batch == 24 * 4);                        // derived (no tuned value)
    CHECK_FALSE(d0.tf32_from_cache);

    // A populated cache overrides each field; an explicit gpu_batch wins over the derived one. The
    // retired attn_bwd_per_query key (flash backward is now unconditional) must be ignored, not error.
    std::istringstream cache(
        "threads=8\nwindows_per_thread=2\ngpu_batch=293\nattn_bwd_per_query=1\ncuda_tf32=1\n");
    const TuneDefaults d = parse_tune_cache(cache, 24);
    CHECK(d.threads == 8);
    CHECK(d.windows_per_thread == 2);
    CHECK(d.gpu_batch == 293);                            // not 8*2
    CHECK(d.cuda_tf32);
    CHECK(d.tf32_from_cache);                             // so the caller knows to honour it

    // Malformed / non-positive values fall back; blank + comment-ish lines are ignored.
    std::istringstream bad("threads=0\nwindows_per_thread=-1\ngarbage\n\n=5\n");
    const TuneDefaults db = parse_tune_cache(bad, 12);
    CHECK(db.threads == 12);                              // 0 rejected -> hardware
    CHECK(db.windows_per_thread == 4);                   // -1 rejected -> default
}

TEST_CASE("vocab curve: the ideal-vocab knee + bytes/token", "[config][vocab]") {
    // Decreasing per-merge benefits (BPE picks the highest-count pair first) -> the compression curve.
    const std::vector<long long> counts = {1000, 500, 200, 100, 50, 25, 10, 5};   // total reduction 1890
    const int n_base = 269;

    // The knee = vocab capturing X% of the total reduction.
    CHECK(vocab_at_fraction(1890, counts, n_base, 0.50) == n_base + 1);   // 945 reached after merge 1
    CHECK(vocab_at_fraction(1890, counts, n_base, 0.90) == n_base + 4);   // 1701 reached after merge 4
    CHECK(vocab_at_fraction(1890, counts, n_base, 1.00) == n_base + 8);   // all merges
    CHECK(vocab_at_fraction(0,    {},     n_base, 0.90) == n_base);       // no reduction -> base only

    // bytes/token rises monotonically from the character-encoding floor as vocab grows.
    const long long total_bytes = 2000;                                  // = total_word_tokens at k=0
    CHECK(bytes_per_token_at(total_bytes, counts, 0) == Catch::Approx(1.0));
    CHECK(bytes_per_token_at(total_bytes, counts, 8) > bytes_per_token_at(total_bytes, counts, 1));
}

TEST_CASE("f16_capable + resolve_precision: codes and capability", "[config][precision]") {
    CHECK_FALSE(f16_capable(0, 120));                    // forced off
    CHECK(f16_capable(1, 0));                            // forced on (no GPU needed for the flag)
    CHECK(f16_capable(2, 120));                          // AUTO + sm_120 -> capable
    CHECK_FALSE(f16_capable(2, 70));                     // AUTO + sm_70 -> not capable

    CHECK(resolve_precision(0, true).dtype == "F32");
    CHECK(resolve_precision(0, false).dtype == "F32");
    CHECK(resolve_precision(1, true).dtype == "BF16");
    CHECK(resolve_precision(1, true).status == PrecStatus::Ok);
    CHECK(resolve_precision(1, false).status == PrecStatus::NeedsF16Hardware);   // BF16 forced, no HW
    CHECK(resolve_precision(2, true).status == PrecStatus::F16Unsupported);      // F16 not wired
    CHECK(resolve_precision(9, true).dtype == "BF16");   // AUTO + capable
    CHECK(resolve_precision(9, false).dtype == "F32");   // AUTO + not capable
}
