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

TEST_CASE("autosize: corpus scale drives a smooth, formula-based model shape", "[config][autosize]") {
    const std::uintmax_t MB = 1000000;

    // TinyStories-scale (22 MB): tiny enough that the params floor binds. d128/L8 matches the ORIGINAL
    // TinyStories paper's own ~2.5M-param reference config almost exactly (their real reference: 8
    // layers, embedding dim 128) -- a genuine prior-art match, not a coincidence of the floor clamp.
    const ModelDims tiny = autosize(22 * MB);
    CHECK(tiny.d_model == 128);
    CHECK(tiny.n_layers == 8);
    CHECK(tiny.n_heads == 2);
    CHECK(tiny.seq_len == 256);
    CHECK(tiny.vocab == 4096);
    CHECK(tiny.d_model % tiny.n_heads == 0);          // must stay head-divisible

    // 1 GB: comfortably past the floor, smoothly smaller than the old coarse-bucket value (which gave
    // d448 for BOTH 1 GB and this project's real ~2.2 GB tinystories.txt corpus despite them differing
    // more than 2x in scale).
    const ModelDims fw = autosize(1024 * MB);
    CHECK(fw.d_model == 192);
    CHECK(fw.n_layers == 8);
    CHECK(fw.n_heads == 3);
    CHECK(fw.seq_len == 256);
    CHECK(fw.vocab == 18432);

    // This project's real tinystories.txt corpus (~2227.75 MB, the actual on-disk full
    // TinyStoriesV2-GPT4 train split): d256/L8, ~19M params -- much closer to what real small-model
    // research on THIS KIND of data actually uses (see the formula's own doc comment for the TinyStories
    // paper / SmolLM2 / FineWeb-Edu-ablation reference points that informed this) than a naive
    // Chinchilla-strict formula would suggest.
    const ModelDims real = autosize(static_cast<std::uintmax_t>(2227.753162 * MB));
    CHECK(real.d_model == 256);
    CHECK(real.n_layers == 8);
    CHECK(real.n_heads == 4);
    CHECK(real.seq_len == 256);
    CHECK(real.vocab == 25088);

    // Monotonic non-decreasing in corpus size across a wide sweep (bigger corpus never shrinks the
    // model), and stays head-divisible everywhere -- including well past the OLD ladder's hard 768
    // ceiling, which this formula has no equivalent of (only the generous kMaxDModel safety clamp).
    int prev_d = 0, prev_l = 0, prev_v = 0, prev_seq = 0;
    for (std::uintmax_t mb : {1u, 22u, 200u, 1024u, 2000u, 10000u, 32768u, 60000u, 100000u}) {
        const ModelDims m = autosize(mb * MB);
        CHECK(m.d_model  >= prev_d);
        CHECK(m.n_layers >= prev_l);
        CHECK(m.vocab    >= prev_v);
        CHECK(m.seq_len  >= prev_seq);
        CHECK(m.d_model % m.n_heads == 0);
        CHECK(m.n_heads >= 2);
        prev_d = m.d_model; prev_l = m.n_layers; prev_v = m.vocab; prev_seq = m.seq_len;
    }
    // A genuinely huge corpus keeps scaling UP instead of saturating at the old fixed ceiling.
    CHECK(autosize(100000 * MB).d_model > autosize(10000 * MB).d_model);
}

TEST_CASE("autosize: hardware-aware VRAM clamp never suggests an unbuildable shape", "[config][autosize]") {
    const std::uintmax_t MB = 1000000;

    // This project's own REAL fineweb_edu.txt corpus (~46.1 GB) with no VRAM budget known (0 = CPU-only
    // or undetected) -- comfortably fits an 8 GB-class card already (~3.6 GiB), demonstrating the
    // conservative tokens/param ratio alone (not the clamp) is what keeps this reachable by default.
    const ModelDims unclamped = autosize(static_cast<std::uintmax_t>(46119.4394 * MB));
    CHECK(unclamped.d_model == 832);
    CHECK(unclamped.n_layers == 14);
    CHECK(unclamped.n_heads == 13);

    // A genuinely huge (hypothetical) corpus DOES exceed an 8151 MiB budget (this project's own real
    // detected VRAM) without the clamp -- the clamp then shrinks it to fit, one head at a time, holding
    // vocab fixed (a separate axis, per the formula's own doc comment) rather than starving it.
    const std::uintmax_t huge = 500000ull * MB;
    const ModelDims noclamp = autosize(huge, /*vram_mb=*/0);
    const ModelDims clamped = autosize(huge, /*vram_mb=*/8151);
    CHECK(clamped.d_model <= noclamp.d_model);
    CHECK(clamped.vocab == noclamp.vocab);            // vocab doesn't compete for the VRAM budget
    CHECK(clamped.d_model % clamped.n_heads == 0);
    CHECK(clamped.n_heads >= 2);                      // the shrink loop never goes below the floor

    // The realized shape's own approximate footprint (matching the formula's internal calibration:
    // 12*L*d^2 + 2*V*d params, ~18 bytes/param) actually fits the requested budget.
    const double params = 12.0 * clamped.n_layers * static_cast<double>(clamped.d_model) * clamped.d_model
                         + 2.0 * clamped.vocab * clamped.d_model;
    CHECK(params * 18.0 / 1e6 <= 8151.0);

    // A smaller card clamps further; monotonic in the budget itself (more VRAM never shrinks the model).
    const ModelDims smallCard = autosize(huge, /*vram_mb=*/4096);
    CHECK(smallCard.d_model <= clamped.d_model);

    // vram_mb=0 is a true no-op: identical to the un-clamped call, not "clamp to zero".
    CHECK(autosize(huge, 0).d_model == noclamp.d_model);
}

TEST_CASE("autosize: size_scale is a minimal<->ideal lever on the SAME formula", "[config][autosize]") {
    const std::uintmax_t bytes = static_cast<std::uintmax_t>(2227.753162 * 1000000);   // tinystories.txt

    const ModelDims minimal = autosize(bytes, 0, 0.5);
    const ModelDims base    = autosize(bytes, 0, 1.0);
    const ModelDims ideal   = autosize(bytes, 0, 2.0);

    CHECK(minimal.d_model <= base.d_model);
    CHECK(base.d_model    <= ideal.d_model);
    CHECK(minimal.d_model < ideal.d_model);      // a real difference, not a no-op
    // Vocab is corpus-token-count-derived, not capacity-derived -- the lever doesn't move it at all.
    CHECK(minimal.vocab == base.vocab);
    CHECK(base.vocab == ideal.vocab);
    CHECK(minimal.d_model % minimal.n_heads == 0);
    CHECK(ideal.d_model % ideal.n_heads == 0);
}

TEST_CASE("apply_autosize: 0 fields auto, nonzero fields pinned", "[config][autosize]") {
    const std::uintmax_t bytes = 22ull * 1000000;       // -> auto d128 L8 H2 seq256 v4096

    // All auto.
    const ModelDims all = apply_autosize({}, bytes);
    CHECK(all.d_model == 128);
    CHECK(all.vocab == 4096);

    // Pin d_model + heads; the rest auto-size (and the pinned values are preserved exactly).
    ModelDims pinned;
    pinned.d_model = 256; pinned.n_heads = 8;
    const ModelDims r = apply_autosize(pinned, bytes);
    CHECK(r.d_model == 256);                              // pinned
    CHECK(r.n_heads == 8);                                // pinned
    CHECK(r.n_layers == 8);                               // auto
    CHECK(r.seq_len == 256);                              // auto
    CHECK(r.vocab == 4096);                               // auto
}

TEST_CASE("token calibration: parse/format round-trip, sum-then-divide, and autosize() wiring", "[config][autosize][calibration]") {
    // Round-trip: format then re-parse recovers the exact totals.
    const TokenCalibration seed{48346287060ull, 14070509477ull};   // this project's real tinystories+fineweb_edu seed
    std::istringstream ss(format_token_calibration(seed));
    const TokenCalibration parsed = parse_token_calibration(ss);
    CHECK(parsed.total_bytes == seed.total_bytes);
    CHECK(parsed.total_tokens == seed.total_tokens);

    // Sum-then-divide, not an average of per-corpus averages: a huge corpus outweighs a tiny one.
    CHECK(bytes_per_token_calibrated(seed) == Catch::Approx(3.436).epsilon(0.001));

    // No accumulated data (a fresh checkout, or an empty/corrupt file) -> the caller's fallback, unchanged.
    CHECK(bytes_per_token_calibrated(TokenCalibration{}, 4.0) == 4.0);
    CHECK(bytes_per_token_calibrated(TokenCalibration{}, 3.5) == 3.5);

    // Ignores unknown keys and blank lines (forward-compatible, matches parse_model_sidecar's own leniency).
    std::istringstream messy("# a comment\ngarbage_key=999\ntotal_bytes=100\n\ntotal_tokens=25\n");
    const TokenCalibration m = parse_token_calibration(messy);
    CHECK(m.total_bytes == 100);
    CHECK(m.total_tokens == 25);
    CHECK(bytes_per_token_calibrated(m) == Catch::Approx(4.0));

    // A calibrated ratio actually changes autosize()'s output vs the generic 4.0 default (a smaller
    // bytes/token means MORE estimated tokens for the same corpus_bytes, so a bigger suggested model).
    const std::uintmax_t bytes = static_cast<std::uintmax_t>(2227.753162 * 1000000);   // tinystories.txt
    const ModelDims generic     = autosize(bytes, 0, 1.0, 4.0);
    const ModelDims calibrated  = autosize(bytes, 0, 1.0, bytes_per_token_calibrated(seed));
    CHECK(calibrated.d_model >= generic.d_model);
}

TEST_CASE("d_ff_for: plain FFN unchanged, gated FFN roughly param-matched to it", "[config][dims]") {
    // Plain: unchanged 4x convention, exactly, regardless of d_model.
    CHECK(d_ff_for(448, false) == 4 * 448);
    CHECK(d_ff_for(32, false) == 4 * 32);

    // Gated: rounded up to a multiple of 64, never below 64.
    CHECK(d_ff_for(448, true) == 1216);   // exact 8/3*448=1194.67 (int div: 1194) -> ceil to 1216
    CHECK(d_ff_for(32, true) == 128);     // exact 8/3*32=85.33 (int div: 85) -> ceil to 128
    CHECK(d_ff_for(0, true) == 64);       // degenerate d_model: floors at the minimum multiple

    // The actual point: total FFN params (excluding the negligible bias) should land close between
    // plain (2*D*F) and gated (3*D*F) at the SAME d_model -- not the pre-rebalance "gated costs 50%
    // more" outcome. Allow up to ~15% drift from the 64-rounding, but no more (this is the whole
    // reason d_ff_for exists, not an incidental property).
    for (int d : {128, 192, 256, 448, 640, 768}) {
        const long long plain_params = 2LL * d * d_ff_for(d, false);
        const long long gated_params = 3LL * d * d_ff_for(d, true);
        const double ratio = static_cast<double>(gated_params) / static_cast<double>(plain_params);
        INFO("d_model=" << d << " plain_ff=" << d_ff_for(d, false) << " gated_ff=" << d_ff_for(d, true)
                        << " ratio=" << ratio);
        CHECK(ratio > 0.85);
        CHECK(ratio < 1.15);
    }
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
