// Chapter 30 — Text Diffusion, Part 3: The Reverse Process (Iterative Refinement)
//
// Ch29 trained a denoiser and measured ONE-STEP recovery: a single forward pass,
// argmax at every blank. That works when most of the canvas is context and fails
// when most is masked — recall fell from 20% @ 10% noise to 7% @ 75%. This chapter
// adds the reverse process that diffusion is named for:
//
//   canvas of [MASK] → denoise → COMMIT only confident predictions → denoise again
//
// Every committed token becomes bidirectional context for the rest, so the canvas
// "snaps into focus" over a handful of iterations (sub0diff/nn/sampler.hpp):
//   • confidence-threshold commitment with a guaranteed-progress floor
//   • DiffusionGemma-style entropy_bound early stop — iteration count is an
//     OUTPUT of generation, not an input (the Ch28 lesson, applied to inference)
//
// The chapter loads Ch29's model dir, shows free-form generation from a prompt,
// then measures iterative vs one-step recovery on IDENTICAL corrupted windows.
//
// MEASURED FINDING (857K-param model, shakespeare): iterative refinement LOSES to
// one-step (-0.5..-1.7pp recall) at every noise level, with or without low-
// confidence remasking, even committing one-token-at-a-time. The reverse process
// conditions on its own commitments; with ~10-15% top-1 accuracy most commits are
// wrong, and self-generated context is worse than the clean (masked) context the
// one-step pass sees — error compounding. The literature's iterative wins assume a
// base model whose confident predictions are usually RIGHT; that's a property to
// re-measure as model/data scale grows (the --conf/--min-commit/--remask knobs
// exist precisely to rerun this experiment against stronger checkpoints).
//
// Usage:
//   ch30_iterative_sampling --model-dir D:/tmp/sub0diff_ch29
//                           [--corpus data/shakespeare.txt] [--prompt "ROMEO:"]
//                           [--conf 0.9] [--entropy-bound 0.1] [--temperature 0]
//                           [--recall-windows 1500]

#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/model_io.hpp"
#include "sub0diff/nn/noise_schedule.hpp"
#include "sub0diff/nn/sampler.hpp"

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/runtime.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace dn = sub0diff::nn;
namespace de = sub0diff::eval;

static void section(std::string_view title) { std::println("\n── {} ──", title); }

struct Config {
    std::string model_dir;
    std::string corpus = "data/shakespeare.txt";
    std::string prompt = "and the king said ";
    dn::SamplerConfig sampler{};        // conf 0.9, entropy_bound 0.1, greedy
    std::size_t recall_windows = 1500;  // per noise level, strided over eval stream
};

static Config parse_args(int argc, char** argv) {
    Config c;
    // GENERATION temperature. Default >0 on purpose: temperature 0 (greedy) under the
    // confidence-commit sampler COLLAPSES a weak model to its single safest token — for a
    // char model that is the space token (~18% of all tokens), so the canvas fills with
    // blanks. A small temperature breaks that degeneracy and yields real text. (The recall
    // comparison in section 3 measures recovery GREEDILY via a temperature-0 copy — see below.)
    c.sampler.temperature = 0.9f;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
            return argv[++i];
        };
        if      (a == "--model-dir")      c.model_dir = next();
        else if (a == "--corpus")         c.corpus = next();
        else if (a == "--prompt")         c.prompt = next();
        else if (a == "--conf")           c.sampler.conf_threshold = std::stof(next());
        else if (a == "--min-commit")     c.sampler.min_commit_frac = std::stof(next());
        else if (a == "--remask")         c.sampler.remask_threshold = std::stof(next());
        else if (a == "--entropy-bound")  c.sampler.entropy_bound = std::stof(next());
        else if (a == "--temperature")    c.sampler.temperature = std::stof(next());
        else if (a == "--recall-windows") c.recall_windows = std::stoull(next());
        else throw std::runtime_error(std::format("unknown argument: {}", a));
    }
    if (c.model_dir.empty())
        throw std::runtime_error("--model-dir is required (a Ch29 checkpoint directory)");
    return c;
}

// Iterative recovery: seed the canvas with the corrupted window (masked positions
// stay [MASK], the rest is fixed context) and let the sampler refine. Directly
// comparable to one-step evaluate_recovery on identical corruption.
static de::RecoveryResult iterative_recovery(const dn::Denoiser& model,
                                             std::span<const std::int32_t> clean,
                                             std::span<const std::uint8_t> masked,
                                             const dn::SamplerConfig& scfg,
                                             std::mt19937& rng,
                                             dn::SamplerStats* stats_out = nullptr) {
    std::vector<std::int32_t> canvas(clean.begin(), clean.end());
    int n_masked = 0;
    for (std::size_t i = 0; i < masked.size(); ++i)
        if (masked[i]) { canvas[i] = model.mask_id(); ++n_masked; }

    auto stats = dn::refine_canvas(model, canvas, scfg, rng);
    if (stats_out) *stats_out = stats;

    de::RecoveryResult res{0, n_masked};
    for (std::size_t i = 0; i < masked.size(); ++i)
        if (masked[i]) res.hits += (canvas[i] == clean[i]);
    return res;
}

static int run(int argc, char** argv) {
    sub0llm::init_cpu_compute();
    std::println("sub0llm — Chapter 30: Text Diffusion, Part 3 (the reverse process)");

    Config cfg = parse_args(argc, argv);

    // ── 1. load the Ch29 model ──────────────────────────────────────────────────
    section("1. Loading the trained denoiser");
    auto lm = dn::load_model_dir(cfg.model_dir);
    const auto& model = *lm.model;
    const auto& tok   = *lm.tokenizer;
    const std::int64_t T = lm.seq_len;
    std::println("loaded {} (step {}): V={}, T={}", cfg.model_dir, lm.step,
                 model.real_vocab(), T);
    std::println("sampler: conf≥{:.2f} commits, entropy_bound {:.2f}, temperature {:.1f}",
                 cfg.sampler.conf_threshold, cfg.sampler.entropy_bound,
                 cfg.sampler.temperature);
    if (cfg.sampler.temperature <= 0.0f)
        std::println("  warning: temperature 0 (greedy) + confidence-commit can COLLAPSE a weak model to its\n"
                     "           single safest token (e.g. all spaces for a char model). Use --temperature 0.7+\n"
                     "           for real generation; this is the iterative-refinement weak-model precondition.");

    // ── 2. watch a canvas snap into focus ───────────────────────────────────────
    section("2. Generation — iterative refinement from a prompt");
    const auto prompt_ids = tok.encode(cfg.prompt);
    std::println("prompt: \"{}\" ({} tokens fixed, {} generated)\n",
                 cfg.prompt, prompt_ids.size(),
                 T - static_cast<std::int64_t>(prompt_ids.size()));
    {
        std::mt19937 rng(42);
        auto canvas = dn::make_canvas(model, T, prompt_ids);
        auto stats = dn::refine_canvas(
            model, canvas, cfg.sampler, rng,
            [&](std::span<const std::int32_t> c, std::size_t iter) {
                std::size_t remaining = 0;
                for (auto t : c) remaining += (t == model.mask_id());
                std::string text;
                for (auto t : c)
                    text += (t == model.mask_id()) ? std::string("·")
                            : std::string(tok.decode({t}));
                if (text.size() > 100) text.resize(100);
                std::println("  iter {:>2} ({:>2} masked left): {}", iter, remaining, text);
            });
        std::println("\n  {} iterations ({}), {} tokens committed, {:.2f}s "
                     "[{:.1f} tok/s incl. all refinement passes]",
                     stats.iterations,
                     stats.entropy_stopped ? "entropy-stopped" : "canvas complete",
                     stats.committed, stats.seconds,
                     static_cast<double>(stats.committed) / stats.seconds);
    }

    // ── 3. iterative vs one-step recovery on identical corruption ──────────────
    section("3. Iterative vs one-step recovery — same windows, same corruption");
    std::println("Iterative refinement conditions on its OWN commitments. Whether that beats");
    std::println("one-step depends on commit precision: when the base model's confident");
    std::println("predictions are usually right, every commit sharpens the next round; when");
    std::println("they are usually wrong (small models), self-conditioning COMPOUNDS errors");
    std::println("and one-step's clean context wins. Measure, don't assume:\n");

    // Held-out-ish stream: tail 5% of the corpus (matches Ch29's split convention).
    std::vector<std::int32_t> eval_ids;
    {
        std::ifstream f(cfg.corpus);
        if (!f) throw std::runtime_error(std::format("cannot open corpus: {}", cfg.corpus));
        std::string line, all;
        std::vector<std::string> paras;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) paras.push_back(line);
        }
        for (std::size_t i = paras.size() - paras.size() / 20; i < paras.size(); ++i) {
            auto v = tok.encode(paras[i]);
            eval_ids.insert(eval_ids.end(), v.begin(), v.end());
        }
    }
    const std::size_t n_positions = eval_ids.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t budget = std::min(cfg.recall_windows, n_positions);
    std::println("eval stream: {} tokens; {} strided windows per noise level\n",
                 eval_ids.size(), budget);

    std::println("  {:>6}  {:>10}  {:>10}  {:>12}  {:>10}", "noise", "one-step",
                 "iterative", "iters (avg)", "Δ recall");
    std::mt19937 rng_a(4242), rng_b(4242);   // identical corruption streams
    // Recovery is a DETERMINISTIC measurement (does iterative refinement recover the exact
    // masked token?), so force greedy here regardless of the generation temperature above —
    // one-step (evaluate_recovery) is argmax, and iterative must match it for a fair compare.
    dn::SamplerConfig recall_cfg = cfg.sampler;
    recall_cfg.temperature = 0.0f;
    const double stride = static_cast<double>(n_positions) / static_cast<double>(budget);
    for (float noise : {0.25f, 0.50f, 0.75f, 0.90f}) {
        de::RecoveryResult one, iter;
        std::size_t iter_sum = 0;
        dn::Corruption corr;
        for (std::size_t i = 0; i < budget; ++i) {
            const auto off = static_cast<std::size_t>(static_cast<double>(i) * stride);
            auto window = std::span<const std::int32_t>(eval_ids)
                              .subspan(off, static_cast<std::size_t>(T));
            dn::corrupt_into(window, noise, dn::NoiseSchedule::Absorbing,
                             model.mask_id(), model.real_vocab(), rng_a, corr);

            auto r1 = de::evaluate_recovery(model, window, corr.corrupted);
            one.hits += r1.hits; one.masked += r1.masked;

            dn::SamplerStats st;
            auto r2 = iterative_recovery(model, window, corr.corrupted,
                                         recall_cfg, rng_b, &st);
            iter.hits += r2.hits; iter.masked += r2.masked;
            iter_sum += st.iterations;
        }
        std::println("  {:>5.0f}%  {:>9.1f}%  {:>9.1f}%  {:>12.1f}  {:>+9.1f}pp",
                     noise * 100.0f, one.recall() * 100.0f, iter.recall() * 100.0f,
                     static_cast<double>(iter_sum) / static_cast<double>(budget),
                     (iter.recall() - one.recall()) * 100.0f);
    }

    section("What's next");
    std::println("Ch31 — block-autoregressive chaining: outputs longer than one canvas, with");
    std::println("       committed blocks providing the left context the edge profile is missing.");
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ch30: fatal: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    }
}
