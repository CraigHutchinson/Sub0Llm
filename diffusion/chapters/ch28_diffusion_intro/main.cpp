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

int main() {
    std::println("sub0llm — Chapter 28: Text Diffusion, Part 1 (corruption → denoiser → recovery)");

    // ── corpus + vocab ──────────────────────────────────────────────────────────
    const std::string corpus =
        "the quick brown fox jumps over the lazy dog. "
        "a stitch in time saves nine. "
        "the early bird catches the worm. "
        "actions speak louder than words. "
        "practice makes perfect every single day. ";
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

    // ── 3. train: adaptive curriculum — noise level driven by performance ───────
    section("3. Training — adaptive curriculum: performance-driven noise level");
    std::println("The noise level moves up when the model masters the current difficulty");
    std::println("and eases back when it struggles — keeping the model at the edge of its ability.\n");
    const std::int64_t T = 24, steps = 15000;

    // Adaptive curriculum parameters
    const float curriculum_start = 0.05f;   // start easy — only 5% masked
    const float curriculum_end   = 0.75f;   // cap at 75% masked
    const int   adapt_interval   = 500;     // evaluate performance every N steps
    const float noise_step       = 0.05f;   // how much to adjust noise per adaptation
    const int   improvement_window = 3;     // look at last N probe losses for trend

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

    // Track recent probe losses for trend detection
    std::vector<float> probe_history;
    constexpr int history_size = 6;

    auto probe_loss = [&]() -> float {
        // Corrupt probe with current_noise — realistic test of current difficulty
        auto corr = dn::corrupt(std::span<const std::int32_t>(probe), current_noise,
                                dn::NoiseSchedule::Absorbing, mask_id, V, probe_rng);
        if (corr.n_corrupted == 0) { corr.tokens[0] = mask_id; corr.corrupted[0] = 1; }

        std::vector<float> w(static_cast<std::size_t>(T));
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) w[i] = corr.corrupted[i];

        ag::Variable logits = model.forward(ids_tensor(corr.tokens), current_noise);
        return ag::weighted_cross_entropy(logits, ids_tensor(probe), f32_tensor(w)).data().item<float>();
    };

    for (std::int64_t step = 0; step < steps; ++step) {
        // ── Adaptive curriculum: adjust noise based on probe loss TREND ───────
        if (step > 0 && step % adapt_interval == 0) {
            float pl = probe_loss();
            probe_history.push_back(pl);
            if (probe_history.size() > static_cast<std::size_t>(history_size))
                probe_history.erase(probe_history.begin());

            // Need enough history to detect a trend
            if (probe_history.size() >= static_cast<std::size_t>(improvement_window)) {
                // Compare average of recent window vs older window
                std::size_t half = probe_history.size() / 2;
                float recent_avg = 0.0f, older_avg = 0.0f;
                for (std::size_t i = 0; i < half; ++i) {
                    older_avg += probe_history[i];
                    recent_avg += probe_history[half + i];
                }
                older_avg /= static_cast<float>(half);
                recent_avg /= static_cast<float>(half);

                float prev_noise = current_noise;
                float improvement = older_avg - recent_avg;  // positive = getting better

                if (improvement > 0.15f) {
                    // Sustained improvement — increase difficulty
                    current_noise = std::min(curriculum_end, current_noise + noise_step);
                } else if (improvement < -0.1f) {
                    // Getting worse — ease back
                    current_noise = std::max(curriculum_start, current_noise - noise_step * 0.5f);
                }
                // else: plateau — hold steady

                if (step % 500 == 0 || std::abs(current_noise - prev_noise) > 0.01f)
                    std::println("  curriculum noise: {:.2f} → {:.2f}  (improvement={:+.4f}, history={:.2f}→{:.2f})",
                                 prev_noise, current_noise, improvement, older_avg, recent_avg);
            }
        }

        // ── Sample a training window and corrupt it with current noise level ──────
        const std::size_t off = off_dist(rng);
        std::vector<std::int32_t> clean(corpus_ids.begin() + static_cast<std::ptrdiff_t>(off),
                                        corpus_ids.begin() + static_cast<std::ptrdiff_t>(off) + T);
        auto corr = dn::corrupt(std::span<const std::int32_t>(clean), current_noise,
                                dn::NoiseSchedule::Absorbing, mask_id, V, rng);
        if (corr.n_corrupted == 0) { corr.tokens[0] = mask_id; corr.corrupted[0] = 1; }  // need ≥1 target

        std::vector<float> w(static_cast<std::size_t>(T));
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) w[i] = corr.corrupted[i];

        ag::Variable logits = model.forward(ids_tensor(corr.tokens), current_noise);  // (T, V+1)
        ag::Variable loss   = ag::weighted_cross_entropy(logits, ids_tensor(clean), f32_tensor(w));

        opt.zero_grad();
        loss.backward();
        (void)nn::clip_grad_norm(params, 5.0f);
        opt.step();

        if (step % 150 == 0 || step == steps - 1) {
            std::println("  step {:4}  train masked-CE = {:.4f}   noise = {:.2f}",
                         step, loss.data().item<float>(), current_noise);
        }
    }

    // ── 4. one-step recovery ──────────────────────────────────────────────────────
    section("4. One-step recovery — fill in the blanks in a single forward pass");
    std::println("Unlike an autoregressive model, the denoiser predicts every blank at once, using");
    std::println("context from BOTH sides (bidirectional attention). We mask scattered characters:\n");
    {
        const std::string sentence = "the quick brown fox";   // appears verbatim in the corpus
        const auto clean = vocab.encode(sentence);
        const auto Tn = static_cast<std::int64_t>(clean.size());

        // mask a few individual characters scattered through the sentence (BERT-style cloze).
        auto input = clean;
        std::vector<std::uint8_t> masked(static_cast<std::size_t>(Tn), 0);
        for (std::size_t pos : {5u, 7u, 12u}) {  // 'u' in quick, 'c' in quick, 'o' in brown
            input[pos] = mask_id;
            masked[pos] = 1;
        }
        const float noise = 3.0f / static_cast<float>(Tn);

        std::println("  clean    : \"{}\"", sentence);
        std::println("  masked   : \"{}\"", render_masked(vocab, input, masked, mask_id));

        ag::Variable logits = model.forward(ids_tensor(input), noise);  // (Tn, V+1)
        auto lz = logits.data().data_as<float>();
        const std::size_t C = static_cast<std::size_t>(model.model_vocab());
        auto recovered = input;
        for (std::size_t t = 0; t < static_cast<std::size_t>(Tn); ++t) {
            if (!masked[t]) continue;
            std::int32_t best = 0;
            float best_v = -1e30f;
            for (std::int64_t c = 0; c < V; ++c) {   // argmax over REAL tokens (skip the mask col)
                const float v = lz[t * C + static_cast<std::size_t>(c)];
                if (v > best_v) { best_v = v; best = static_cast<std::int32_t>(c); }
            }
            recovered[t] = best;
        }
        std::println("  recovered: \"{}\"", vocab.decode(recovered));
        int hits = 0, total = 0;
        for (std::size_t t = 0; t < static_cast<std::size_t>(Tn); ++t)
            if (masked[t]) { ++total; hits += (recovered[t] == clean[t]); }
        std::println("\n  filled {}/{} blanks correctly{}.", hits, total,
                     hits == total ? " — exact one-step recovery ✓" : " (train longer for the rest)");
    }

    section("What's next");
    std::println("Ch29 — formalise the diffusion loss + a full bidirectional training loop with checkpoints.");
    std::println("Ch30 — the iterative reverse process: refine a whole canvas over many steps with remasking.");
    std::println("Ch31 — block-autoregressive chaining for sequences longer than one canvas.");
    return 0;
}
