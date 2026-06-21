#pragma once

// topic_drift.hpp (Ch32 Phase 0, metric M2) — the TOPIC-DRIFT metric.
//
// A flat word-level model loses the thread across a passage: it (a) LOOPS — repeats short n-grams
// degenerately — and (b) DRIFTS — the content words (entities/nouns) that anchor a story early stop
// recurring, so the topic dissolves. M2 quantifies both WITHOUT retraining, on any token passage
// (a held-out corpus passage OR a model-generated one), so corpus and model are A/B-comparable:
//
//   • distinct-n  (n=2,3,4) = unique n-grams / total n-grams.  HIGHER = less looping.
//   • persistence = mean Jaccard overlap of the CONTENT-word sets of two windows at distance d.
//       persistence_near (adjacent windows) vs persistence_far (distant windows). Coherent text keeps
//       content overlap at distance (the cat stays the cat); drifting text decays to background fast.
//   • drift = 1 − persistence_far / persistence_near.  HIGHER = more topic-drift.
//
// "Content" words are the frequency-split complement of the function-word head (the `stop_k` most
// frequent types are grammar; the rest carry topic) — the same split P2's `is_content` table will
// use, so M2 is the gate for P2: a trained gist conditioner (P2) should LOWER drift vs flat P1
// (BUILD_PLAN §Phase 0 / Phase 2). 4b already showed a crude untrained spatial-spread sampler cuts
// looping ~25–30% ([`4B_RESULTS.md`]); M2 is the metric that scores whether a trained gist does more.

#include <cstdint>
#include <span>
#include <vector>

namespace sub0diff::eval {

struct TopicDriftResult {
    double distinct2 = 0.0, distinct3 = 0.0, distinct4 = 0.0;  // unique n-grams / total (↑ = less looping)
    // content_recurrence = fraction of a passage's DISTINCT content types that occur ≥2× — does the
    // text reuse the entities it introduces? Coherent stories do (the cat stays the cat); a flat
    // model that scatters one-off content words does NOT. This is the M2 coherence signal that
    // VALIDATES on TinyStories (real > window-mixed control), unlike window-Jaccard persistence.
    double content_recurrence = 0.0;  // ↑ = entities persist (coherent)
    double persistence_near = 0.0;   // adjacent-window content-set Jaccard (weak on homogeneous corpora)
    double persistence_far  = 0.0;   // distant-window content-set Jaccard
    double drift            = 0.0;   // 1 − far/near (↓ = less topic-drift)
    std::uint64_t n_tokens  = 0;     // total tokens scored
    std::uint64_t n_passages = 0;    // passages averaged
};

// Mark the `stop_k` most-frequent token TYPES (by train-frequency) as FUNCTION words; every other
// type is CONTENT. Returns is_content indexed by token id (size = vocab). Mirrors rare_type_mask /
// is_word_start: a pure frequency split, no linguistic resource needed.
[[nodiscard]] std::vector<std::uint8_t>
content_type_mask(std::span<const std::int32_t> train_ids, std::int64_t vocab, std::int64_t stop_k);

// M2 over one or many passages (each a token span). distinct-n and persistence are computed
// per-passage and AVERAGED (so the score is comparable regardless of how many passages are pooled).
// window = tokens per content-window; near/far = window distances for the persistence buckets.
[[nodiscard]] TopicDriftResult
evaluate_topic_drift(std::span<const std::span<const std::int32_t>> passages,
                     std::span<const std::uint8_t> is_content,
                     std::int64_t window = 16, std::int64_t near_d = 1, std::int64_t far_d = 4);

// Convenience single-passage overload.
[[nodiscard]] TopicDriftResult
evaluate_topic_drift(std::span<const std::int32_t> passage,
                     std::span<const std::uint8_t> is_content,
                     std::int64_t window = 16, std::int64_t near_d = 1, std::int64_t far_d = 4);

}  // namespace sub0diff::eval
