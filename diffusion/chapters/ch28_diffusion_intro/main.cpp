// Chapter 28 — Text Diffusion, Part 1: Corruption, a Bidirectional Denoiser, One-Step Recovery
//
// Every chapter so far built an AUTOREGRESSIVE model: predict token t+1 from tokens ≤ t,
// left to right, one at a time. DiffusionGemma works completely differently — it starts from
// a canvas of placeholder tokens and refines ALL positions in parallel, many at once, using
// bidirectional attention. This chapter introduces the three ideas the rest of the diffusion
// arc builds on:
//
//   1. The FORWARD process (corruption): damage a clean sentence by masking a fraction of its
//      tokens. This is fixed and parameter-free (sub0diff/nn/noise_schedule.hpp).
//   2. NOISE-LEVEL conditioning: tell the model how corrupted its input is, injected like a
//      positional embedding (sub0diff/nn/time_embedding.hpp).
//   3. A BIDIRECTIONAL DENOISER: a transformer that predicts the clean token at every position
//      at once — reusing sub0llm's blocks with causal=false (sub0diff/nn/denoiser.hpp).
//
// We train a tiny denoiser to undo masking on a small corpus, then show it recover a masked
// sentence in a single forward pass. Ch29 formalises the training objective and scales it up;
// Ch30 turns this one-step recovery into the iterative reverse-diffusion sampling loop.

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"
#include "sub0diff/nn/time_embedding.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include "sub0llm/core/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace sub0llm;
namespace ag = sub0llm::autograd;
namespace dn = sub0diff::nn;

static void section(std::string_view title) { std::println("\n── {} ──", title); }

// ── A minimal char-level tokenizer (self-contained; the BPE tokenizer is overkill here) ──
struct CharVocab {
    std::map<char, std::int32_t> to_id;
    std::vector<char>            to_char;

    explicit CharVocab(const std::string& corpus) {
        for (char c : corpus)
            if (!to_id.contains(c)) { to_id[c] = static_cast<std::int32_t>(to_char.size()); to_char.push_back(c); }
    }
    [[nodiscard]] std::int64_t size() const { return static_cast<std::int64_t>(to_char.size()); }

    [[nodiscard]] std::vector<std::int32_t> encode(const std::string& s) const {
        std::vector<std::int32_t> v;
        v.reserve(s.size());
        for (char c : s) v.push_back(to_id.at(c));
        return v;
    }
    [[nodiscard]] std::string decode(std::span<const std::int32_t> ids) const {
        std::string s;
        for (std::int32_t id : ids)
            s.push_back(id >= 0 && id < size() ? to_char[static_cast<std::size_t>(id)] : '?');
        return s;
    }
};

static Tensor ids_tensor(const std::vector<std::int32_t>& v) {
    Tensor t({static_cast<std::int64_t>(v.size())}, DType::Int32);
    auto s = t.data_as<std::int32_t>();
    std::ranges::copy(v, s.begin());
    return t;
}

// Render a sequence with mask positions shown as '_'.
static std::string render_masked(const CharVocab& vocab,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint8_t> masked,
                                 std::int32_t mask_id) {
    std::string s;
    for (std::size_t i = 0; i < tokens.size(); ++i)
        s.push_back((masked.empty() ? tokens[i] == mask_id : masked[i]) ? '_'
                    : vocab.to_char[static_cast<std::size_t>(tokens[i])]);
    return s;
}

// One-step recovery: mask the flagged positions, run a single denoiser forward,
// and count how many masked positions are filled with the original token.
struct RecoveryResult {
    int hits   = 0;   // masked positions filled correctly
    int masked = 0;   // total masked positions
    [[nodiscard]] float recall() const { return masked > 0 ? static_cast<float>(hits) / static_cast<float>(masked) : 0.0f; }
};

// Optional per-position accumulators (index = position in the window): reveals
// whether window-edge positions train/recover worse than the interior.
struct PositionStats {
    std::vector<int> hits, masked;
    explicit PositionStats(std::size_t T) : hits(T, 0), masked(T, 0) {}
};

RecoveryResult evaluate_recovery(const dn::Denoiser& model,
                                 std::span<const std::int32_t> clean_ids,
                                 std::span<const std::uint8_t> masked,
                                 std::int32_t mask_id,
                                 PositionStats* pos = nullptr) {
    auto input = std::vector<std::int32_t>(clean_ids.begin(), clean_ids.end());
    int n_masked = 0;
    for (std::size_t i = 0; i < masked.size(); ++i)
        if (masked[i]) { input[i] = mask_id; ++n_masked; }

    const float noise = static_cast<float>(n_masked) / static_cast<float>(clean_ids.size());
    ag::Variable logits = model.forward(ids_tensor(input), noise);
    auto lz = logits.data().data_as<float>();
    const std::size_t C = static_cast<std::size_t>(model.model_vocab());
    const std::int64_t V = model.model_vocab() - 1;   // argmax over real tokens only

    RecoveryResult res{0, n_masked};
    for (std::size_t t = 0; t < masked.size(); ++t) {
        if (!masked[t]) continue;
        std::int32_t best = 0;
        float best_v = -1e30f;
        for (std::int64_t c = 0; c < V; ++c) {
            const float v = lz[t * C + static_cast<std::size_t>(c)];
            if (v > best_v) { best_v = v; best = static_cast<std::int32_t>(c); }
        }
        const bool hit = (best == clean_ids[t]);
        res.hits += hit;
        if (pos) { pos->masked[t] += 1; pos->hits[t] += hit; }
    }
    return res;
}

// Exhaustive sweep: corrupt EVERY corpus window at a given noise level and measure
// recall = fraction of masked positions recovered exactly. One number per noise level
// instead of a handful of binary spot checks.
RecoveryResult evaluate_corpus_recall(const dn::Denoiser& model,
                                      std::span<const std::int32_t> corpus_ids,
                                      std::int64_t T, float noise,
                                      std::int32_t mask_id, std::int64_t real_vocab,
                                      std::mt19937& rng,
                                      PositionStats* pos = nullptr) {
    RecoveryResult total;
    const std::size_t n_windows = corpus_ids.size() - static_cast<std::size_t>(T);
    for (std::size_t off = 0; off < n_windows; ++off) {
        auto window = corpus_ids.subspan(off, static_cast<std::size_t>(T));
        auto corr = dn::corrupt(window, noise, dn::NoiseSchedule::Absorbing,
                                mask_id, real_vocab, rng);
        auto r = evaluate_recovery(model, window, corr.corrupted, mask_id, pos);
        total.hits   += r.hits;
        total.masked += r.masked;
    }
    return total;
}

int main(int argc, char** argv) {
    sub0llm::init_cpu_compute();  // FTZ+DAZ — see core/runtime.hpp for the why (4.4× here)

    std::println("sub0llm — Chapter 28: Text Diffusion, Part 1 (corruption → denoiser → recovery)");

    // ── corpus + vocab ──────────────────────────────────────────────────────────
    const std::string corpus =
        "the quick brown fox jumps over the lazy dog. "
        "a stitch in time saves nine. "
        "the early bird catches the worm. "
        "actions speak louder than words. "
        "practice makes perfect every single day. "
        "all that glitters is not gold. "
        "better late than never. "
        "beauty is in the eye of the beholder. "
        "birds of a feather flock together. "
        "cat got your tongue. "
        "don't count your chickens before they hatch. "
        "every cloud has a silver lining. "
        "fair and square. "
        "go the extra mile. "
        "had a ball at the party last night. "
        "if the mountain won't come to Muhammad then Muhammad goes to the mountain. "
        "in the nick of time we arrived. "
        "jack of all trades but master of none. "
        "kiss of death ended the deal. "
        "let by and by. "
        "make a long story short. "
        "never put off till tomorrow what you can do today. "
        "once in a blue moon we see such beauty. "
        "play devil's advocate for a moment. "
        "quiet as the grave in the old house. "
        "raining cats and dogs all afternoon. "
        "the best of both worlds is what she found. "
        "the cake is a lie but the truth hurts. "
        "the pen is mightier than the sword. "
        "there is no place like home. "
        "this is the end of the road. "
        "to be or not to be that is the question. "
        "under the same roof they lived. "
        "we shall overcome the difficulty. "
        "you can't judge a book by its cover. "
        "a journey of a thousand miles begins with a single step. "
        "knowledge is power and wisdom is the light. "
        "the sun rises in the east and sets in the west. "
        "time flies when you are having fun. ";
    CharVocab vocab(corpus);
    const std::int64_t V = vocab.size();
    std::println("corpus: {} chars, vocab: {} unique characters", corpus.size(), V);

    const auto corpus_ids = vocab.encode(corpus);

    // ── 1. the forward (corruption) process ──────────────────────────────────────
    section("1. Forward process — corrupting a clean sentence");
    std::println("Absorbing-state diffusion replaces a fraction of tokens with [MASK] (shown as '_').");
    std::println("The model's job is to invert this. Higher noise = more masked positions.\n");
    {
        std::mt19937 rng(7);
        const std::string sentence = "the quick brown fox";
        const auto clean = vocab.encode(sentence);
        const std::int32_t mask_id = static_cast<std::int32_t>(V);  // == real vocab size
        for (float p : {0.25f, 0.5f, 0.75f}) {
            auto c = dn::corrupt(std::span<const std::int32_t>(clean), p,
                                 dn::NoiseSchedule::Absorbing, mask_id, V, rng);
            std::println("  noise={:.0f}%  \"{}\"", p * 100.0f,
                         render_masked(vocab, c.tokens, c.corrupted, mask_id));
        }
    }

    // ── 2. build the denoiser ─────────────────────────────────────────────────────
    section("2. A tiny bidirectional denoiser");
    const std::int64_t D = 96, n_layers = 3, d_ff = 256;
    const std::size_t  n_heads = 6, n_kv_heads = 6;
    dn::Denoiser model(V, D, n_heads, n_kv_heads, n_layers, d_ff, /*seed=*/42);
    auto params = model.parameters();
    std::int64_t n_params = 0;
    for (auto* p : params) n_params += p->data().numel();
    std::println("Denoiser: D={}, layers={}, heads={}, params={} (bidirectional attention)",
                 D, n_layers, n_heads, n_params);
    std::println("Model vocab = {} (real {} + 1 [MASK] row at id {}).",
                 model.model_vocab(), V, model.mask_id());

    // T is the window size we train on — how many tokens we corrupt and predict at once.
    const std::int64_t T = 24;

    const std::size_t corpus_windows = static_cast<std::size_t>(corpus_ids.size()) - static_cast<std::size_t>(T);

    // ── 3. train: frontier-point noise with a mastery-gated ceiling ─────────────
    // A/B sweeps on this corpus established two things:
    //   • CONCENTRATION beats dilution — training mass AT the ceiling ("point")
    //     outperforms uniform band sampling at every noise level, because a step's
    //     learning signal is its masked-token count: a 5%-noise step teaches ~1
    //     token, a 50% step ~12. Band sampling wastes half the steps on weak signal.
    //   • The signal-per-step imbalance also means dwell time per level should scale
    //     like 1/noise — so curriculum evidence is measured in MASKED TOKENS, not
    //     steps: low-noise levels automatically train longer to earn the same proof.
    // The default (auto) mode combines both with per-noise-bucket loss EMAs (each
    // bucket judged only against its own history, so "loss rose because noise rose"
    // can never read as regression) and SELF-TERMINATES once the target ceiling is
    // mastered — the step count becomes an output of training, not an input.
    section("3. Training — frontier-point noise, mastery-gated ceiling");
    const std::uint64_t steps = corpus_windows * 200LL;   // upper bound; auto stops early
    // argv[1]: auto (default) | point | linear | mix | adaptive   argv[2]: curriculum end (default 0.80)
    //   auto     — frontier-point sampling + token-gated adaptive ceiling + self-terminating
    //   point    — frontier-point sampling, linear ceiling ramp, fixed step count
    //   linear   — band sampling U[floor,ceiling], linear ramp (kept for comparison)
    //   mix      — 50% at ceiling, 50% band (comparison)
    //   adaptive — band sampling, token-gated adaptive ceiling (comparison)
    const std::string_view mode = argc > 1 ? argv[1] : "auto";
    const bool auto_curriculum     = mode == "auto";
    const bool adaptive_curriculum = mode == "adaptive" || auto_curriculum;
    const bool frontier_mix        = mode == "mix";
    const bool frontier_point      = mode == "point" || auto_curriculum;  // mass AT the ceiling
    const bool linear_curriculum   = !adaptive_curriculum;

    // Minimum corruption floor — deterministic guarantee, not random fallback
    // With T=24 tokens, 0.05 noise → ~1 token minimum (ceil(24*0.05) = 2)
    const float min_corruption = 1.0f / static_cast<float>(T-1);  // one token minimum

    // Curriculum parameters
    const float curriculum_start = min_corruption;   // start easy — only 5% masked
    const float curriculum_end   = argc > 2 ? std::stof(argv[2]) : 0.80f;
    const float noise_step       = min_corruption;   // smaller steps for smoother adaptation

    std::println("Training (max {} steps, {} windows), {} ceiling → {:.2f}:",
                 steps, corpus_windows,
                 auto_curriculum     ? "AUTO (point + token-gated, self-terminating)"
                 : adaptive_curriculum ? "ADAPTIVE (band + token-gated)"
                 : frontier_point    ? "LINEAR-POINT"
                 : frontier_mix      ? "LINEAR+FRONTIER-MIX" : "LINEAR",
                 curriculum_end);

    nn::Adam opt(params, /*lr=*/2e-3f);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<std::size_t> off_dist(0, corpus_ids.size() - static_cast<std::size_t>(T) - 1);
    const std::int32_t mask_id = model.mask_id();

    // Ceiling of the noise band (the floor is curriculum_start, fixed).
    float ceiling = curriculum_start;

    // ── Per-noise-bucket curriculum state (adaptive/auto modes) ─────────────────
    // One bucket per noise_step of the band. Each bucket keeps an EMA of the losses
    // of steps that LANDED in it, so buckets are only ever compared with themselves:
    // difficulty shifts between levels can't masquerade as learning or regression.
    // Evidence is counted in MASKED TOKENS (the true signal unit): the ceiling bucket
    // is judged once it has accumulated min_tokens since its last decision, which
    // makes dwell time per level ∝ 1/noise — token-linearized training.
    struct NoiseBucket {
        float         ema      = 0.0f;
        float         baseline = 0.0f;
        std::uint64_t tokens   = 0;       // masked tokens since this bucket's last decision
        bool          ema_init = false, baseline_set = false;
    };
    constexpr float         bucket_alpha = 0.02f;   // ~50-sample horizon per bucket
    constexpr std::uint64_t min_tokens   = 1500;    // evidence required per decision
    const auto bucket_of = [&](float noise) {
        return std::min(static_cast<std::size_t>(noise / noise_step + 0.5f),
                        static_cast<std::size_t>(curriculum_end / noise_step + 0.5f));
    };
    std::vector<NoiseBucket> buckets(bucket_of(curriculum_end) + 1);
    bool curriculum_converged = false;   // auto mode: target ceiling reached AND mastered

    std::uniform_real_distribution<float> noise_dist(0.0f, 1.0f);

    // ── Pre-allocated per-step state — the loop body performs zero heap allocations
    // beyond the autograd graph itself. The graph is freed at the end of each
    // iteration, before any buffer is reused; corrupt_into reuses corr's storage.
    Tensor ids_input({static_cast<std::int64_t>(T)}, DType::Int32);
    Tensor ids_clean({static_cast<std::int64_t>(T)}, DType::Int32);
    Tensor weights({static_cast<std::int64_t>(T)}, DType::Float32);
    const auto ids_input_s = ids_input.data_as<std::int32_t>();
    const auto ids_clean_s = ids_clean.data_as<std::int32_t>();
    const auto weights_s   = weights.data_as<float>();
    dn::Corruption corr;
    const auto corpus_span = std::span<const std::int32_t>(corpus_ids);

    // ── Timer for throughput measurement ───────────────────────────────────────
    auto train_start = std::chrono::high_resolution_clock::now();
    std::uint64_t steps_taken = 0;
    std::uint64_t reportinterval_steps = 0;
    std::uint64_t reportinterval_tokens = 0;
    auto last_timing_check = train_start;

    for (std::uint64_t step = 0; step < steps; ++step) {
        // ── Ceiling schedule ───────────────────────────────────────────────────────
        if (linear_curriculum) {
            const float ramp = static_cast<float>(step) / (0.8f * static_cast<float>(steps));
            ceiling = std::min(curriculum_end,
                               curriculum_start + (curriculum_end - curriculum_start) * std::min(ramp, 1.0f));
        }

        // ── Sample a window (a view into the corpus) and a noise level in the band ─
        // frontier_mix: half the steps train AT the ceiling (frontier pressure, like a
        // point curriculum), half uniformly below it (easy-regime coverage).
        const auto clean = corpus_span.subspan(off_dist(rng), static_cast<std::size_t>(T));
        const float u = noise_dist(rng);
        const float sampled_noise =
            frontier_point ? ceiling
          : frontier_mix   ? (u < 0.5f ? ceiling
                                       : curriculum_start + (ceiling - curriculum_start) * 2.0f * (u - 0.5f))
                           : curriculum_start + (ceiling - curriculum_start) * u;
        dn::corrupt_into(clean, sampled_noise, dn::NoiseSchedule::Absorbing,
                         mask_id, V, rng, corr);

        const float actual_noise = static_cast<float>(corr.n_corrupted) / static_cast<float>(T);

        std::ranges::copy(corr.tokens, ids_input_s.begin());
        std::ranges::copy(clean, ids_clean_s.begin());
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
            weights_s[i] = corr.corrupted[i];

        ag::Variable logits = model.forward(ids_input, actual_noise);  // (T, V+1)
        ag::Variable loss   = ag::weighted_cross_entropy(logits, ids_clean, weights);

        opt.zero_grad();
        loss.backward();
        (void)nn::clip_grad_norm(params, 5.0f);
        opt.step();

        // ── Adaptive ceiling: judge only the ceiling bucket, against itself ───────
        if (!linear_curriculum) {
            const float step_loss = loss.data().item<float>();
            NoiseBucket& b = buckets[bucket_of(actual_noise)];
            b.ema = b.ema_init ? bucket_alpha * step_loss + (1.0f - bucket_alpha) * b.ema
                               : (b.ema_init = true, step_loss);
            b.tokens += corr.n_corrupted;

            NoiseBucket& top = buckets[bucket_of(ceiling)];
            if (top.tokens >= min_tokens) {
                if (!top.baseline_set) {
                    top.baseline = top.ema;       // first full evidence window at this ceiling
                    top.baseline_set = true;
                    top.tokens = 0;
                } else {
                    const float improvement = top.baseline - top.ema;  // >0 = learning
                    const float prev = ceiling;
                    if (improvement > 0.08f)
                        ceiling = std::min(curriculum_end, ceiling + noise_step);
                    else if (improvement < -0.06f)
                        ceiling = std::max(curriculum_start, ceiling - noise_step);
                    else {
                        // Plateau. At the target ceiling this means the final level has
                        // stopped improving — the curriculum is complete (auto mode).
                        if (auto_curriculum && ceiling >= curriculum_end - 0.001f) {
                            curriculum_converged = true;
                            std::println("  curriculum converged at ceiling {:.2f} "
                                         "(bucket EMA {:.4f} plateaued, step {})",
                                         ceiling, top.ema, step);
                        }
                        top.tokens = 0;           // otherwise: re-judge after another window
                    }

                    if (std::abs(ceiling - prev) >= 0.001f) {
                        std::println("  ceiling: {:.2f} → {:.2f}  (bucket EMA {:.4f} → {:.4f}, step {})",
                                     prev, ceiling, top.baseline, top.ema, step);
                        top.tokens = 0;
                        // The new ceiling bucket is judged fresh against its own history.
                        NoiseBucket& nt = buckets[bucket_of(ceiling)];
                        nt.baseline_set = false;
                        nt.tokens = 0;
                    }
                }
            }
        }

        ++reportinterval_steps;
        reportinterval_tokens += corr.n_corrupted;

        // Throughput reporting at timed interval
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_since_check = std::chrono::duration<double>(now - last_timing_check).count();
        if (elapsed_since_check >= 30.0 || step == steps - 1) {
            double elapsed_total = std::chrono::duration<double>(now - train_start).count();
            double steps_per_sec = static_cast<double>(reportinterval_steps) / elapsed_since_check;
            double tokens_per_sec = static_cast<double>(reportinterval_tokens) / elapsed_since_check;
            std::println("{}s  step {:6}  loss={:.4f}  ceiling={:.2f}  [{} steps/s, {} tok/s]",
                         static_cast<int>(std::round(elapsed_total)),
                         step, loss.data().item<float>(), ceiling,
                         static_cast<int>(std::round(steps_per_sec)),
                         static_cast<int>(std::round(tokens_per_sec)));
            last_timing_check = now;
            reportinterval_steps = 0;  // reset counter for next interval
            reportinterval_tokens = 0; // reset token counter for next interval
        }

        if (curriculum_converged) { steps_taken = step + 1; break; }
        steps_taken = step + 1;
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_seconds = std::chrono::duration<double>(train_end - train_start).count();
    double overall_steps_per_sec = static_cast<double>(steps_taken) / train_seconds;
    std::println("Training complete: {:.1f}s, {:.1f} steps/s, {} steps{}", train_seconds,
                 overall_steps_per_sec, steps_taken,
                 curriculum_converged ? " (self-terminated: curriculum converged)"
                 : steps_taken == steps ? " (hit max-step bound)" : "");

    // ── 4a. qualitative spot checks — watch the model fill in the blanks ─────────
    section("4. Recovery evaluation");
    std::println("Unlike an autoregressive model, the denoiser predicts every blank at once, using");
    std::println("context from BOTH sides (bidirectional attention). A few illustrative fragments:\n");

    auto spot_check = [&](std::string_view sentence, std::initializer_list<std::size_t> mask_at) {
        const auto clean = vocab.encode(std::string(sentence));
        std::vector<std::uint8_t> masked(clean.size(), 0);
        for (std::size_t pos : mask_at) masked[pos] = 1;
        auto r = evaluate_recovery(model, clean, masked, mask_id);
        std::println("  [{}] \"{}\" — {}/{} correct",
                     r.hits == r.masked ? "PASS" : "MISS",
                     render_masked(vocab, clean, masked, mask_id), r.hits, r.masked);
        return r;
    };

    RecoveryResult spots;
    for (auto r : {spot_check("the quick brown fox", {5, 7, 12}),
                   spot_check("the lazy dog",        {4, 8}),
                   spot_check("a stitch in time",    {1, 3, 11}),
                   spot_check("knowledge is power and wisdom is the light", {3, 14, 25, 33})}) {
        spots.hits   += r.hits;
        spots.masked += r.masked;
    }
    std::println("\n  Spot checks: {}/{} masked positions correct ({:.0f}%)",
                 spots.hits, spots.masked, spots.recall() * 100.0f);

    // ── 4b. exhaustive recall sweep — every corpus window, across noise levels ───
    std::println("\nExhaustive sweep: corrupt EVERY one of the {} corpus windows at each noise", corpus_windows);
    std::println("level and measure recall (fraction of masked positions recovered exactly).");
    std::println("Recall should degrade gracefully as context is destroyed — not fall off a cliff");
    std::println("below the ceiling the curriculum reached ({:.2f}).\n", ceiling);

    std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
    std::mt19937 eval_rng(4242);
    RecoveryResult sweep_total;
    PositionStats pos_stats(static_cast<std::size_t>(T));
    for (float noise : {0.10f, 0.25f, 0.40f, 0.60f, 0.80f}) {
        auto r = evaluate_corpus_recall(model, std::span<const std::int32_t>(corpus_ids),
                                        T, noise, mask_id, V, eval_rng, &pos_stats);
        sweep_total.hits   += r.hits;
        sweep_total.masked += r.masked;
        std::println("  {:>5.0f}%  {:>8}  {:>10}  {:>6.1f}%",
                     noise * 100.0f, r.masked, r.hits, r.recall() * 100.0f);
    }
    std::println("\n  Overall recall across the sweep: {}/{} ({:.1f}%)",
                 sweep_total.hits, sweep_total.masked, sweep_total.recall() * 100.0f);

    // ── 4c. recall by window position — are the edges harder? ───────────────────
    // A masked token at position 0 or T-1 has context on only ONE side, so it should
    // recover worse than the interior; any other dips reveal under-trained positions.
    std::println("\nRecall by window position (aggregated across the sweep — edge effects):");
    std::print("  pos:    ");
    for (std::int64_t t = 0; t < T; ++t) std::print("{:>4}", t);
    std::print("\n  recall: ");
    float interior_sum = 0.0f; int interior_n = 0;
    for (std::int64_t t = 0; t < T; ++t) {
        const auto i = static_cast<std::size_t>(t);
        const float rc = pos_stats.masked[i] > 0
            ? static_cast<float>(pos_stats.hits[i]) / static_cast<float>(pos_stats.masked[i]) : 0.0f;
        std::print("{:>4.0f}", rc * 100.0f);
        if (t > 0 && t < T - 1) { interior_sum += rc; ++interior_n; }
    }
    const auto edge = [&](std::size_t i) {
        return pos_stats.masked[i] > 0
            ? static_cast<float>(pos_stats.hits[i]) / static_cast<float>(pos_stats.masked[i]) : 0.0f;
    };
    std::println("\n  edges: first {:.1f}%, last {:.1f}%  vs interior mean {:.1f}%",
                 edge(0) * 100.0f, edge(static_cast<std::size_t>(T - 1)) * 100.0f,
                 interior_sum / static_cast<float>(interior_n) * 100.0f);

    section("What's next");
    std::println("Ch29 — formalise the diffusion loss + a full bidirectional training loop with checkpoints.");
    std::println("Ch30 — the iterative reverse process: refine a whole canvas over many steps with remasking.");
    std::println("Ch31 — block-autoregressive chaining for sequences longer than one canvas.");
    return 0;
}
