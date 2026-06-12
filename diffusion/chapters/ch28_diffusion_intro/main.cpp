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
static Tensor f32_tensor(const std::vector<float>& v) {
    Tensor t({static_cast<std::int64_t>(v.size())});
    auto s = t.data_as<float>();
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

// Evaluate one-step recovery on a set of test fragments.
// Returns overall accuracy and prints per-fragment results.
struct RecoveryResult {
    float accuracy;   // fraction of masked positions filled correctly
    int   total_hits;
    int   total_masked;
};

RecoveryResult evaluate_recovery(const CharVocab& vocab,
                                 const dn::Denoiser& model,
                                 std::span<const std::int32_t> test_ids,
                                 std::span<const std::int32_t> clean_ids,
                                 std::vector<std::uint8_t> masked,
                                 std::int32_t mask_id,
                                 float noise) {
    auto input = std::vector<std::int32_t>(test_ids.begin(), test_ids.end());
    for (std::size_t i = 0; i < masked.size(); ++i)
        if (masked[i]) input[i] = mask_id;

    ag::Variable logits = model.forward(ids_tensor(input), noise);
    auto lz = logits.data().data_as<float>();
    const std::size_t C = static_cast<std::size_t>(model.model_vocab());

    auto recovered = input;
    for (std::size_t t = 0; t < masked.size(); ++t) {
        if (!masked[t]) continue;
        std::int32_t best = 0;
        float best_v = -1e30f;
        for (std::int64_t c = 0; c < vocab.size(); ++c) {
            const float v = lz[t * C + static_cast<std::size_t>(c)];
            if (v > best_v) { best_v = v; best = static_cast<std::int32_t>(c); }
        }
        recovered[t] = best;
    }

    int hits = 0, total = 0;
    for (std::size_t t = 0; t < masked.size(); ++t)
        if (masked[t]) { ++total; hits += (recovered[t] == clean_ids[t]); }

    RecoveryResult res{static_cast<float>(hits) / (total > 0 ? total : 1), hits, total};
    return res;
}

int main() {
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

    // Dynamic adapt interval: corpus-aware, not rigid.
    // We adapt every time we've seen ~5% of available corpus windows.
    // This scales with corpus size — small corpus adapts faster, large corpus adapts slower.
    const std::size_t corpus_windows = static_cast<std::size_t>(corpus_ids.size()) - static_cast<std::size_t>(T);
    const std::uint32_t adapt_interval = std::max(250U, static_cast<std::uint32_t>(corpus_windows*2));

    // ── 3. train: adaptive curriculum — noise level driven by performance ───────
    section("3. Training — adaptive curriculum: performance-driven noise level");
    std::println("The noise level moves up when the model masters the current difficulty");
    std::println("and eases back when it struggles — keeping the model at the edge of its ability.\n");
    const std::uint64_t steps = corpus_windows * 72LL;

    std::println("Training for {} steps ({} epochs, {} windows) with adaptive curriculum noise:",
                 steps, steps / corpus_windows, corpus_windows);

    // Minimum corruption floor — deterministic guarantee, not random fallback
    // With T=24 tokens, 0.05 noise → ~1 token minimum (ceil(24*0.05) = 2)
    const float min_corruption = 1.0f / static_cast<float>(T-1);  // one token minimum

    // Adaptive curriculum parameters
    const float curriculum_start = min_corruption;   // start easy — only 5% masked
    const float curriculum_end   = 0.80f;   // cap at 80% masked — push higher for robust denoising
    const float noise_step       = min_corruption;   // smaller steps for smoother adaptation

    nn::Adam opt(params, /*lr=*/2e-3f);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<std::size_t> off_dist(0, corpus_ids.size() - static_cast<std::size_t>(T) - 1);
    const std::int32_t mask_id = model.mask_id();

    // Current curriculum noise level (starts low, adapts based on performance)
    float current_noise = curriculum_start;

    // A FIXED held-out probe (same window every time) — corrupted with current_noise
    // so it's a realistic test of what the model will face at the current difficulty.
    std::vector<std::int32_t> probe(corpus_ids.begin(), corpus_ids.begin() + T);

    // Separate RNG for probe (fixed seed for reproducibility)
    std::mt19937 probe_rng(999);

    // EMA-smoothed probe loss — prevents oscillation when noise level changes
    // Raw probe loss rises when noise increases (harder task), causing false "getting worse"
    // signals. EMA smooths this so we measure actual learning, not task difficulty.
    // Baseline comparison: record EMA loss at last noise change, compare current EMA against it.
    float ema_probe_loss = 0.0f;
    constexpr float ema_alpha = 0.3f;  // aggressive smoothing — recent values weighted heavily
    bool ema_initialized = false;
    float ema_baseline = 0.0f;  // EMA loss recorded at last noise level change
    bool baseline_set = false;

    auto probe_loss = [&]() -> float {
        // Corrupt probe with actual_noise — realistic test of current difficulty
        auto corr = dn::corrupt(std::span<const std::int32_t>(probe),
                                current_noise,
                                dn::NoiseSchedule::Absorbing, mask_id, V, probe_rng);

        const float actual_noise = static_cast<float>(corr.n_corrupted) / static_cast<float>(T);
        std::vector<float> w(static_cast<std::size_t>(T));
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) w[i] = corr.corrupted[i];

        ag::Variable logits = model.forward(ids_tensor(corr.tokens), actual_noise);
        float raw_loss = ag::weighted_cross_entropy(logits, ids_tensor(probe), f32_tensor(w)).data().item<float>();

        // EMA smoothing — prevents oscillation from noise-level-induced loss spikes
        if (!ema_initialized) {
            ema_probe_loss = raw_loss;
            ema_initialized = true;
        } else {
            ema_probe_loss = ema_alpha * raw_loss + (1.0f - ema_alpha) * ema_probe_loss;
        }
        return ema_probe_loss;
    };

    // ── Timer for throughput measurement ───────────────────────────────────────
    auto train_start = std::chrono::high_resolution_clock::now();
    std::uint64_t reportinterval_steps = 0;
    std::uint64_t reportinterval_tokens = 0;
    auto last_timing_check = train_start;

    for (std::uint64_t step = 0; step < steps; ++step) {
        // ── Adaptive curriculum: adjust noise based on EMA probe loss vs baseline ───
        if (step > 0 && step % adapt_interval == 0) {
            float pl = probe_loss();

            // Initialize baseline on first adapt step
            if (!baseline_set) {
                ema_baseline = pl;
                baseline_set = true;
            }

            float prev_noise = current_noise;
            float improvement = ema_baseline - pl;  // positive = getting better (loss decreasing)

            // Thresholds tuned for EMA-smoothed losses (less noisy than raw)
            if (improvement > 0.08f) {
                // Sustained improvement — increase difficulty
                current_noise = std::min(curriculum_end, current_noise + noise_step);
                baseline_set = false;  // reset baseline at new difficulty
            } else if (improvement < -0.06f) {
                // Getting worse — ease back
                current_noise = std::max(curriculum_start, current_noise - noise_step * 0.5f);
                baseline_set = false;  // reset baseline at new difficulty
            }
            // else: plateau — hold steady

            if ( std::abs(current_noise - prev_noise) >= 0.001f)
                std::println("  curriculum noise: {:.2f} → {:.2f}  (EMA baseline {:.4f} → {:.4f})",
                             prev_noise, current_noise, ema_baseline, pl);
        }

        // ── Sample a training window and corrupt it with current noise level ──────
        const std::size_t off = off_dist(rng);
        std::vector<std::int32_t> clean(corpus_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                        corpus_ids.begin() + static_cast<std::ptrdiff_t>(off) + T);
        // Quantize noise to min_corruption multiples — deterministic curriculum levels.
        // corrupt() guarantees ≥1 corruption, so no fallback needed.
        auto corr = dn::corrupt(std::span<const std::int32_t>(clean),
                                current_noise,
                                dn::NoiseSchedule::Absorbing, mask_id, V, rng);

        const float actual_noise = static_cast<float>(corr.n_corrupted) / static_cast<float>(T);
        std::vector<float> w(static_cast<std::size_t>(T));
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) w[i] = corr.corrupted[i];

        ag::Variable logits = model.forward(ids_tensor(corr.tokens), actual_noise);  // (T, V+1)
        ag::Variable loss   = ag::weighted_cross_entropy(logits, ids_tensor(clean), f32_tensor(w));

        opt.zero_grad();
        loss.backward();
        (void)nn::clip_grad_norm(params, 5.0f);
        opt.step();

        ++reportinterval_steps;
        reportinterval_tokens += corr.n_corrupted;

        // Throughput reporting at timed interval
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_since_check = std::chrono::duration<double>(now - last_timing_check).count();
        if (elapsed_since_check >= 30.0 || step == steps - 1) {
            double elapsed_total = std::chrono::duration<double>(now - train_start).count();
            double steps_per_sec = static_cast<double>(reportinterval_steps) / elapsed_since_check;
            double tokens_per_sec = static_cast<double>(reportinterval_tokens) / elapsed_since_check;
            std::println("{}s  step {:6}  loss={:.4f}  noise={:.2f}  [{} steps/s, {} tok/s]",
                         static_cast<int>(std::round(elapsed_total)),
                         step, loss.data().item<float>(), current_noise,
                         static_cast<int>(std::round(steps_per_sec)),
                         static_cast<int>(std::round(tokens_per_sec)));
            last_timing_check = now;
            reportinterval_steps = 0;  // reset counter for next interval
            reportinterval_tokens = 0; // reset token counter for next interval
        }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_seconds = std::chrono::duration<double>(train_end - train_start).count();
    double overall_steps_per_sec = static_cast<double>(steps) / train_seconds;
    std::println("Training complete: {:.1f}s, {:.1f} steps/s, {} total steps", train_seconds,
                 overall_steps_per_sec, steps);

    // ── 4. multi-fragment recovery evaluation ───────────────────────────────────
    section("4. Recovery evaluation — testing across corpus fragments");
    std::println("Unlike an autoregressive model, the denoiser predicts every blank at once, using");
    std::println("context from BOTH sides (bidirectional attention). Testing on multiple fragments:\n");

    // Test fragments: mix of corpus windows and held-out sentences
    struct TestFragment {
        std::string description;
        std::vector<std::int32_t> clean_ids;
        std::vector<std::uint8_t> masked;
    };

    std::vector<TestFragment> test_fragments;

    // Fragment 1: "the quick brown fox" — appears verbatim in corpus
    {
        const std::string sentence = "the quick brown fox";
        const auto clean = vocab.encode(sentence);
        const auto Tn = static_cast<std::int64_t>(clean.size());
        std::vector<std::uint8_t> masked(static_cast<std::size_t>(Tn), 0);
        for (std::size_t pos : {5u, 7u, 12u}) {
            masked[pos] = 1;
        }
        test_fragments.push_back({"verbatim corpus (the quick brown fox)", clean, masked});
    }

    // Fragment 2: "the lazy dog" — appears verbatim in corpus, different positions
    {
        const std::string sentence = "the lazy dog";
        const auto clean = vocab.encode(sentence);
        const auto Tn = static_cast<std::int64_t>(clean.size());
        std::vector<std::uint8_t> masked(static_cast<std::size_t>(Tn), 0);
        for (std::size_t pos : {4u, 8u}) {
            masked[pos] = 1;
        }
        test_fragments.push_back({"verbatim corpus (the lazy dog)", clean, masked});
    }

    // Fragment 3: "a stitch in time" — appears verbatim in corpus
    {
        const std::string sentence = "a stitch in time";
        const auto clean = vocab.encode(sentence);
        const auto Tn = static_cast<std::int64_t>(clean.size());
        std::vector<std::uint8_t> masked(static_cast<std::size_t>(Tn), 0);
        for (std::size_t pos : {1u, 3u, 11u}) {
            masked[pos] = 1;
        }
        test_fragments.push_back({"verbatim corpus (a stitch in time)", clean, masked});
    }

    // Fragment 4: random corpus window at different offset
    {
        const std::size_t off = (corpus_ids.size() - static_cast<std::size_t>(T)) / 2;
        std::vector<std::int32_t> window(corpus_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                         corpus_ids.begin() + static_cast<std::ptrdiff_t>(off) + T);
        const auto Tn = static_cast<std::int64_t>(window.size());
        std::vector<std::uint8_t> masked(static_cast<std::size_t>(Tn), 0);
        // Mask positions 1, 5, 9, 13, 17, 21 (scattered, not just even)
        for (std::size_t i = 1u; i < masked.size(); i += 4) masked[i] = 1;
        test_fragments.push_back({"mid-corpus window (offset ~50%)", window, masked});
    }

    // Run recovery evaluation on all fragments
    RecoveryResult overall{0.0f, 0, 0};
    for (const auto& frag : test_fragments) {
        const float noise = static_cast<float>(std::count(frag.masked.begin(), frag.masked.end(), 1))
                          / static_cast<float>(frag.clean_ids.size());

        auto result = evaluate_recovery(vocab, model, frag.clean_ids, frag.clean_ids,
                                        frag.masked, mask_id, noise);
        overall.total_hits += result.total_hits;
        overall.total_masked += result.total_masked;

        std::println("  [{}] {} — {} correct",
                     result.accuracy > 0.9f ? "PASS" : (result.accuracy > 0.5f ? "PARTIAL" : "NEEDS WORK"),
                     frag.description, result.total_hits);
    }
    overall.accuracy = static_cast<float>(overall.total_hits) / (overall.total_masked > 0 ? overall.total_masked : 1);
    std::println("\n  Overall recovery: {}/{} masked positions correct ({:.0f}%)",
                 overall.total_hits, overall.total_masked, overall.accuracy * 100.0f);

    section("What's next");
    std::println("Ch29 — formalise the diffusion loss + a full bidirectional training loop with checkpoints.");
    std::println("Ch30 — the iterative reverse process: refine a whole canvas over many steps with remasking.");
    std::println("Ch31 — block-autoregressive chaining for sequences longer than one canvas.");
    return 0;
}
