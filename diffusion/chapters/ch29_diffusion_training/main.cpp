// Chapter 29 — Text Diffusion, Part 2: The Formal Objective and Real Training
//
// Ch28 trained a toy char-level denoiser with an ad-hoc masked loss and a hand-built
// curriculum. This chapter graduates to the real thing on real text:
//
//   1. The FORMAL objective (sub0diff/train/diffusion_loss.hpp): the diffusion NELBO
//      Rao-Blackwellizes to a 1/t-weighted masked cross-entropy — sample t ~ U(0,1],
//      mask with probability t, weight the masked CE by n_masked/(t·T). Ch28's
//      curriculum experiments were variants of choosing the distribution of t; the
//      formal objective integrates over ALL noise levels every epoch.
//   2. REAL data: BPE-tokenized Shakespeare via the Ch24 TextCorpus pipeline, with a
//      held-out split for honest evaluation.
//   3. CHECKPOINTS: save/resume via the Ch24 binary checkpoint format. The model dir
//      (config.json + tokenizer/ + step_*.ckpt) is what Ch30's iterative sampler loads.
//
// Usage:
//   ch29_diffusion_training [--corpus data/shakespeare.txt] [--paragraphs 600]
//                           [--vocab-size 512] [--seq-len 64] [--steps 200000]
//                           [--ckpt-dir /tmp/sub0diff_ch29] [--eval-every 2000]
//                           [--patience 5] [--t-max 1.0] [--eval-only]
//
// Training is SELF-TERMINATING (a Ch28 lesson): --steps is only a safety bound.
// Every --eval-every steps the held-out NELBO is measured; a checkpoint is saved on
// each improvement, and the run stops after --patience evaluations without one —
// classic early stopping, which on a small corpus also catches memorization (the
// Ch24 failure mode: train loss falls while eval loss climbs).

#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/runtime.hpp"
#include "sub0llm/data/text_corpus.hpp"
#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace sub0llm;
namespace ag = sub0llm::autograd;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
namespace de = sub0diff::eval;
namespace fs = std::filesystem;

static void section(std::string_view title) { std::println("\n── {} ──", title); }

// ── Configuration (CLI-overridable) ───────────────────────────────────────────
struct Config {
    std::string corpus     = "data/shakespeare.txt";
    std::string ckpt_dir   = "/tmp/sub0diff_ch29";
    std::size_t paragraphs = 7000;    // full shakespeare.txt (~7.2K paragraphs); BPE ~1 min
    std::size_t vocab_size = 512;
    std::int64_t seq_len   = 64;
    std::uint64_t steps    = 200000;   // safety BOUND — training self-terminates (see below)
    std::uint64_t eval_every = 2000;   // held-out NELBO cadence (the convergence signal)
    std::uint64_t patience   = 5;      // stop after this many evals without improvement
    float t_max            = 1.0f;    // <1.0 trains under a Ch28-style ceiling
    bool eval_only         = false;

    // Architecture — modest but real (≈1.6M params at V=512).
    std::int64_t embed_dim = 128, n_layers = 4, d_ff = 384;
    std::size_t  n_heads = 8, n_kv_heads = 4;   // GQA: 2 query heads per KV head
};

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
            return argv[++i];
        };
        if      (a == "--corpus")     c.corpus     = next();
        else if (a == "--ckpt-dir")   c.ckpt_dir   = next();
        else if (a == "--paragraphs") c.paragraphs = std::stoull(next());
        else if (a == "--vocab-size") c.vocab_size = std::stoull(next());
        else if (a == "--seq-len")    c.seq_len    = std::stoll(next());
        else if (a == "--steps")      c.steps      = std::stoull(next());
        else if (a == "--eval-every") c.eval_every = std::stoull(next());
        else if (a == "--patience")   c.patience   = std::stoull(next());
        else if (a == "--t-max")      c.t_max      = std::stof(next());
        else if (a == "--eval-only")  c.eval_only  = true;
        else throw std::runtime_error(std::format("unknown argument: {}", a));
    }
    return c;
}

// Paragraph reader: one paragraph per line (data/shakespeare.txt format, as Ch24).
// Strips trailing '\r' so CRLF corpora behave identically everywhere.
static std::vector<std::string> read_paragraphs(const std::string& path, std::size_t limit) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::format("cannot open corpus: {}", path));
    std::vector<std::string> out;
    std::string line;
    while (out.size() < limit && std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

static int run(int argc, char** argv);

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ch29: fatal: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    }
}

static int run(int argc, char** argv) {
    sub0llm::init_cpu_compute();   // FTZ+DAZ — prerequisite for stable training throughput
    std::println("sub0llm — Chapter 29: Text Diffusion, Part 2 (formal objective, real training)");

    Config cfg;
    try { cfg = parse_args(argc, argv); }
    catch (const std::exception& e) { std::println("error: {}", e.what()); return 2; }

    // ── 1. the objective, in words ──────────────────────────────────────────────
    section("1. The formal objective — diffusion NELBO as weighted masked LM");
    std::println("L = E[t~U(0,1]] E[mask~Bern(t)] (1/t)·Σ_masked -log p(x0[i]|x_t)");
    std::println("One Monte-Carlo sample per step: draw t, corrupt, weigh the masked CE by");
    std::println("n_masked/(t·T). Every noise level is trained EVERY epoch — Ch28's curriculum");
    std::println("is the special case of reshaping the distribution of t (here: U(0.02, {:.2f}]).",
                 cfg.t_max);

    // ── 2. data: BPE + TextCorpus with a held-out split ─────────────────────────
    section("2. Data — BPE-tokenized corpus, 90/10 train/eval split");
    auto paragraphs = read_paragraphs(cfg.corpus, cfg.paragraphs);
    if (paragraphs.size() < 20)
        throw std::runtime_error("corpus too small — need at least 20 paragraphs");
    const std::size_t n_eval = paragraphs.size() / 10;
    std::vector<std::string> eval_texts(paragraphs.end() - static_cast<std::ptrdiff_t>(n_eval),
                                        paragraphs.end());
    paragraphs.resize(paragraphs.size() - n_eval);

    std::println("{} train / {} eval paragraphs from {}", paragraphs.size(), n_eval, cfg.corpus);
    std::print("training BPE (vocab {})... ", cfg.vocab_size);
    auto bpe_start = std::chrono::steady_clock::now();
    auto tok = BPETokenizer::train(paragraphs, cfg.vocab_size);
    std::println("done in {:.1f}s",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - bpe_start).count());

    TextCorpus train_corpus(paragraphs, tok, cfg.seq_len, /*seed=*/42);
    std::println("train corpus: {} tokens, {} windows of {}",
                 train_corpus.total_tokens(), train_corpus.n_samples(), cfg.seq_len);

    // Flat eval token stream for the recall sweep + held-out NELBO.
    std::vector<std::int32_t> eval_ids;
    for (const auto& p : eval_texts) {
        auto ids = tok.encode(p);
        eval_ids.insert(eval_ids.end(), ids.begin(), ids.end());
    }
    std::println("eval stream: {} tokens", eval_ids.size());

    // ── 3. model + checkpoint resume ────────────────────────────────────────────
    section("3. Model — bidirectional denoiser over the BPE vocabulary");
    const auto V = static_cast<std::int64_t>(tok.vocab_size());
    dn::Denoiser model(V, cfg.embed_dim, cfg.n_heads, cfg.n_kv_heads,
                       cfg.n_layers, cfg.d_ff, /*seed=*/42);
    auto param_ptrs = model.parameters();
    std::vector<ag::Variable> params;                 // shared-impl handles for checkpointing
    params.reserve(param_ptrs.size());
    for (auto* p : param_ptrs) params.push_back(*p);
    std::int64_t n_params = 0;
    for (const auto& p : params) n_params += p.data().numel();
    std::println("Denoiser: V={} (+1 mask), D={}, layers={}, heads={}/{} (GQA), params={}",
                 V, cfg.embed_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, n_params);

    fs::create_directories(cfg.ckpt_dir);
    std::uint64_t start_step = 0;
    if (const auto path = latest_checkpoint_path(cfg.ckpt_dir); !path.empty()) {
        start_step = static_cast<std::uint64_t>(load_checkpoint(params, path));
        std::println("resumed from {} (step {})", path, start_step);
    } else {
        // Fresh run: persist tokenizer + config so Ch30's sampler can reload the model dir.
        tok.save(fs::path(cfg.ckpt_dir) / "tokenizer");
        std::ofstream cj(fs::path(cfg.ckpt_dir) / "config.json");
        cj << std::format(
            "{{\n  \"model\": \"sub0diff-denoiser\",\n  \"vocab_size\": {},\n"
            "  \"embed_dim\": {},\n  \"n_layers\": {},\n  \"n_heads\": {},\n"
            "  \"n_kv_heads\": {},\n  \"d_ff\": {},\n  \"seq_len\": {}\n}}\n",
            V, cfg.embed_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, cfg.seq_len);
        std::println("fresh run — wrote tokenizer/ and config.json to {}", cfg.ckpt_dir);
    }

    // Held-out NELBO: averaged diffusion loss over eval windows at fixed seeds —
    // the honest scalar to compare checkpoints with (lower = better).
    dt::DiffusionLossContext eval_ctx(cfg.seq_len);
    auto eval_nelbo = [&](std::size_t n_windows) {
        std::mt19937 erng(777);
        std::uniform_int_distribution<std::size_t> eoff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.seq_len) - 1);
        double sum = 0.0;
        for (std::size_t i = 0; i < n_windows; ++i) {
            auto w = std::span<const std::int32_t>(eval_ids)
                         .subspan(eoff(erng), static_cast<std::size_t>(cfg.seq_len));
            sum += dt::diffusion_loss(model, w, erng, eval_ctx).loss.data().item<float>();
        }
        return sum / static_cast<double>(n_windows);
    };

    // ── 4. training ─────────────────────────────────────────────────────────────
    if (!cfg.eval_only && start_step < cfg.steps) {
        section("4. Training — self-terminating on held-out NELBO (early stopping)");
        nn::Adam opt(param_ptrs, /*lr=*/1e-3f);
        std::mt19937 rng(1234 + static_cast<std::uint32_t>(start_step));
        dt::DiffusionLossContext ctx(cfg.seq_len);

        // NOTE: TextCorpus::next_sample REASSIGNS these tensors (fresh storage each
        // call) — the data span must be re-taken after every call, never cached.
        Tensor ids({cfg.seq_len}, DType::Int32), tgt({cfg.seq_len}, DType::Int32);

        auto t0 = std::chrono::steady_clock::now();
        auto last_report = t0;
        std::uint64_t interval_steps = 0, interval_tokens = 0;

        float best_nelbo = std::numeric_limits<float>::max();
        std::uint64_t evals_since_best = 0, steps_taken = start_step;
        constexpr float min_improvement = 0.01f;

        for (std::uint64_t step = start_step; step < cfg.steps; ++step) {
            if (!train_corpus.next_sample(ids, tgt)) {
                train_corpus.reset(static_cast<std::uint32_t>(step));
                (void)train_corpus.next_sample(ids, tgt);
            }
            const auto ids_s = ids.data_as<std::int32_t>();
            auto res = dt::diffusion_loss(
                model, std::span<const std::int32_t>(ids_s.data(), ids_s.size()),
                rng, ctx, 0.02f, cfg.t_max);

            opt.zero_grad();
            res.loss.backward();
            (void)nn::clip_grad_norm(param_ptrs, 5.0f);
            opt.step();
            ++interval_steps;
            interval_tokens += res.n_masked;
            steps_taken = step + 1;

            auto now = std::chrono::steady_clock::now();
            const double since_report = std::chrono::duration<double>(now - last_report).count();
            if (since_report >= 30.0) {
                const double el = std::chrono::duration<double>(now - t0).count();
                std::println("{:>5.0f}s  step {:>6}  nelbo={:.4f} (t={:.2f})  "
                             "[{:.0f} steps/s, {:.0f} masked-tok/s]",
                             el, step, res.loss.data().item<float>(), res.t,
                             static_cast<double>(interval_steps) / since_report,
                             static_cast<double>(interval_tokens) / since_report);
                last_report = now;
                interval_steps = 0;
                interval_tokens = 0;
            }

            // ── Convergence check: held-out NELBO every eval_every steps ───────────
            if ((step + 1) % cfg.eval_every == 0) {
                const float nelbo = static_cast<float>(eval_nelbo(64));
                if (nelbo < best_nelbo - min_improvement) {
                    best_nelbo = nelbo;
                    evals_since_best = 0;
                    save_checkpoint(params, cfg.ckpt_dir, static_cast<std::int64_t>(step + 1));
                    std::println("  eval @ {:>6}: held-out NELBO {:.4f}  ← best, checkpointed",
                                 step + 1, nelbo);
                } else {
                    ++evals_since_best;
                    std::println("  eval @ {:>6}: held-out NELBO {:.4f}  (best {:.4f}, {}/{} patience)",
                                 step + 1, nelbo, best_nelbo, evals_since_best, cfg.patience);
                    if (evals_since_best >= cfg.patience) {
                        std::println("  early stop: no improvement for {} evals", cfg.patience);
                        break;
                    }
                }
            }
        }
        const double total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::println("Training done: {:.1f}s, {} steps ({}), best held-out NELBO {:.4f}",
                     total, steps_taken,
                     steps_taken == cfg.steps ? "hit safety bound" : "self-terminated",
                     best_nelbo);
        // Reload the BEST checkpoint (the loop may have overfit past it) so §5
        // evaluates — and Ch30 inherits — the early-stopping winner, not the tail.
        if (const auto best = latest_checkpoint_path(cfg.ckpt_dir); !best.empty())
            (void)load_checkpoint(params, best);
    } else if (cfg.eval_only) {
        section("4. Training — skipped (--eval-only)");
    } else {
        section("4. Training — skipped (checkpoint already at target steps)");
    }

    // ── 5. held-out evaluation: NELBO + recall sweep + edge profile ─────────────
    section("5. Held-out evaluation");
    std::println("held-out NELBO (256 windows): {:.4f}\n", eval_nelbo(256));

    std::println("Recall sweep over the held-out stream ({} windows of {}):",
                 eval_ids.size() - static_cast<std::size_t>(cfg.seq_len), cfg.seq_len);
    std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
    std::mt19937 eval_rng(4242);
    de::PositionStats pos(static_cast<std::size_t>(cfg.seq_len));
    de::RecoveryResult sweep;
    for (float noise : {0.10f, 0.25f, 0.50f, 0.75f}) {
        auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.seq_len, noise, eval_rng, &pos);
        sweep.hits += r.hits; sweep.masked += r.masked;
        std::println("  {:>5.0f}%  {:>8}  {:>10}  {:>6.1f}%",
                     noise * 100.0f, r.masked, r.hits, r.recall() * 100.0f);
    }
    std::println("  overall: {}/{} ({:.1f}%)", sweep.hits, sweep.masked, sweep.recall() * 100.0f);

    float interior = 0.0f; int n_int = 0;
    for (std::int64_t t = 1; t + 1 < cfg.seq_len; ++t) {
        interior += pos.recall_at(static_cast<std::size_t>(t)); ++n_int;
    }
    std::println("\nEdge effect: first {:.1f}%, last {:.1f}% vs interior mean {:.1f}%",
                 pos.recall_at(0) * 100.0f,
                 pos.recall_at(static_cast<std::size_t>(cfg.seq_len - 1)) * 100.0f,
                 interior / static_cast<float>(n_int) * 100.0f);

    section("What's next");
    std::println("Ch30 — the reverse process: iterative canvas refinement with confidence remasking,");
    std::println("       loading this chapter's model dir ({}).", cfg.ckpt_dir);
    return 0;
}
