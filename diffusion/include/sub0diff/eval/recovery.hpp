#pragma once

// recovery.hpp (Ch29) — shared one-step recovery evaluation for denoisers.
//
// Generalizes Ch28's chapter-local evaluation: mask chosen positions of a clean
// window, run a single denoiser forward, count exact recoveries. The corpus-wide
// sweep produces a recall-vs-noise curve, and optional per-position accumulators
// expose window-edge effects (Ch28 measured ~40% edge vs ~62% interior recall —
// one-sided context; motivates Ch30/31 block overlap / committed-block context).

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::eval {

struct RecoveryResult {
    int hits   = 0;   // masked positions filled correctly
    int masked = 0;   // total masked positions
    // Word-start vs word-continuation breakdown — populated only when an `is_word_start`
    // table is supplied. A BPE word-start carries the Ġ leading-space marker; a
    // continuation is a mid-word subword. The split tests whether the headline recall is
    // inflated by *easy* continuation completion (`Ham`→`let`) vs *real* word prediction.
    int ws_hits = 0, ws_masked = 0;   // word-START tokens (the honest, hard targets)
    int wc_hits = 0, wc_masked = 0;   // word-CONTINUATION tokens (partial-word completion)
    // Word-level recall: a word containing ≥1 masked token counts as recovered only if
    // ALL its masked tokens are recovered. The more meaningful, tokenizer-robust number.
    int word_hits = 0, word_total = 0;

    [[nodiscard]] float recall() const noexcept { return frac(hits, masked); }
    [[nodiscard]] float ws_recall() const noexcept { return frac(ws_hits, ws_masked); }
    [[nodiscard]] float wc_recall() const noexcept { return frac(wc_hits, wc_masked); }
    [[nodiscard]] float word_recall() const noexcept { return frac(word_hits, word_total); }

    void accumulate(const RecoveryResult& r) noexcept {
        hits += r.hits; masked += r.masked;
        ws_hits += r.ws_hits; ws_masked += r.ws_masked;
        wc_hits += r.wc_hits; wc_masked += r.wc_masked;
        word_hits += r.word_hits; word_total += r.word_total;
    }

private:
    [[nodiscard]] static float frac(int a, int b) noexcept {
        return b > 0 ? static_cast<float>(a) / static_cast<float>(b) : 0.0f;
    }
};

// Per-window-position accumulators (index = position): reveals whether edge
// positions train/recover worse than the interior.
struct PositionStats {
    std::vector<int> hits, masked;
    explicit PositionStats(std::size_t T) : hits(T, 0), masked(T, 0) {}
    [[nodiscard]] float recall_at(std::size_t i) const noexcept {
        return masked[i] > 0 ? static_cast<float>(hits[i]) / static_cast<float>(masked[i]) : 0.0f;
    }
};

// Mask the flagged positions, denoise once, count exact recoveries (greedy argmax
// over REAL tokens only — the [MASK] row is never a valid prediction).
//   is_word_start — optional table indexed by token id (1 = the token carries the BPE
//   word-start marker). When non-empty, fills the word-start/continuation split and the
//   word-level recall fields of the result. A window's position 0 is always treated as a
//   word boundary regardless of the table.
[[nodiscard]] RecoveryResult evaluate_recovery(const nn::Denoiser& model,
                                               std::span<const std::int32_t> clean_ids,
                                               std::span<const std::uint8_t> masked,
                                               PositionStats* pos = nullptr,
                                               std::span<const std::uint8_t> is_word_start = {});

// Corrupt windows of `corpus_ids` at `noise` and accumulate recall.
// max_windows caps the cost: when the stream has more sliding positions than the
// budget, windows are sampled on a uniform stride across the whole stream (full
// coverage of the corpus span, bounded forward count). 0 = exhaustive (every
// position — on large corpora this can cost ORDERS more time than training did).
//   is_word_start — forwarded to evaluate_recovery for the word-level breakdown.
//   whole_word    — when true (and is_word_start supplied), corruption masks whole words
//                   (all subwords of a chosen word together) instead of independent tokens.
[[nodiscard]] RecoveryResult evaluate_corpus_recall(const nn::Denoiser& model,
                                                    std::span<const std::int32_t> corpus_ids,
                                                    std::int64_t T, float noise,
                                                    std::mt19937& rng,
                                                    PositionStats* pos = nullptr,
                                                    std::size_t max_windows = 0,
                                                    std::span<const std::uint8_t> is_word_start = {},
                                                    bool whole_word = false);

} // namespace sub0diff::eval
