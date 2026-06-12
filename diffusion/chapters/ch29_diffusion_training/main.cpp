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
//   2. REAL data: BPE-tokenized Shakespeare as a flat token stream with random-offset
//      sliding windows (train and eval share one window distribution) + held-out split.
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
    std::int64_t paragraphs = 0;      // 0 or -1 = read all paragraphs from corpus
    std::size_t vocab_size = 512;
    std::int64_t seq_len   = 64;
    std::uint64_t steps    = 1000000;   // safety BOUND — training self-terminates (see below)
    std::uint64_t eval_every = 0;      // 0 = compute from corpus; override with --eval-every
    double eval_factor = 0.5;           // fraction of train positions per eval cadence (0.5 = 50%)
    std::uint64_t patience   = 5;      // stop after this many evals without improvement
    float t_max            = 1.0f;    // <1.0 trains under a Ch28-style ceiling
    bool eval_only         = false;
    bool profile           = false;   // report forward/backward/optimizer time split

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
        else if (a == "--paragraphs") c.paragraphs = std::stoll(next());
        else if (a == "--vocab-size") c.vocab_size = std::stoull(next());
        else if (a == "--seq-len")    c.seq_len    = std::stoll(next());
        else if (a == "--steps")      c.steps      = std::stoull(next());
        else if (a == "--eval-every") c.eval_every = std::stoull(next());
        else if (a == "--patience")   c.patience   = std::stoull(next());
        else if (a == "--eval-factor") c.eval_factor = std::stof(next());
        else if (a == "--t-max")      c.t_max      = std::stof(next());
        else if (a == "--eval-only")  c.eval_only  = true;
        else if (a == "--profile")    c.profile    = true;
        else throw std::runtime_error(std::format("unknown argument: {}", a));
    }
    return c;
}

// Paragraph reader: one paragraph per line (data/shakespeare.txt format, as Ch24).
// Strips trailing '\r', leading/trailing whitespace, and collapses internal
// runs of whitespace to a single space. Skips blank lines.
// limit == 0 or -1 means read ALL paragraphs from the corpus.
static std::vector<std::string> read_paragraphs(const std::string& path, std::int64_t limit) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::format("cannot open corpus: {}", path));
    std::vector<std::string> out;
    std::string line;
    const bool all = (limit <= 0);
    while (std::getline(f, line)) {
        // Strip trailing '\r' for CRLF files
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
               line.back() == '\v' || line.back() == '\f'))
            line.pop_back();
        // Strip leading whitespace
        auto start = line.begin();
        while (start != line.end() && std::isspace(static_cast<unsigned char>(*start)))
            ++start;
        // Strip trailing whitespace
        auto end = line.rbegin();
        while (end.base() != start && std::isspace(static_cast<unsigned char>(*end)))
            ++end;
        // Collapse internal whitespace to single space; skip blank lines
        if (start >= end.base()) continue;
        std::string cleaned;
        cleaned.reserve(std::distance(start, end.base()));
        bool prev_space = false;
        for (auto it = start; it != end.base(); ++it) {
            const char c = *it;
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!prev_space) { cleaned += ' '; prev_space = true; }
            } else {
                cleaned += c; prev_space = false;
            }
        }
        if (!cleaned.empty()) {
            if (!all && out.size() >= static_cast<std::size_t>(limit)) break;
            out.push_back(std::move(cleaned));
        }
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

    // ── 2. data: BPE + flat token streams with a held-out split ─────────────────
    section("2. Data — BPE-tokenized corpus, 95/5 train/eval split");
    auto paragraphs = read_paragraphs(cfg.corpus, cfg.paragraphs);
    if (paragraphs.size() < 20)
        throw std::runtime_error("corpus too small — need at least 20 paragraphs");
    const std::size_t n_eval = paragraphs.size() / 5;
    std::vector<std::string> eval_texts(paragraphs.end() - static_cast<std::ptrdiff_t>(n_eval),
                                        paragraphs.end());
    paragraphs.resize(paragraphs.size() - n_eval);

    std::println("{} train / {} eval paragraphs from {}", paragraphs.size(), n_eval, cfg.corpus);
    std::print("training BPE (vocab {})... ", cfg.vocab_size);
    auto bpe_start = std::chrono::steady_clock::now();
    auto tok = BPETokenizer::train(paragraphs, cfg.vocab_size);
    std::println("done in {:.1f}s",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - bpe_start).count());

    // Both splits are FLAT token streams sampled with sliding windows, so train and
    // eval see the same window distribution. Training draws windows at RANDOM offsets
    // (not disjoint blocks): every token gets trained at every window position, which
    // matters for the edge-recall deficit — a token at a window edge in one draw is
    // interior in the next. (TextCorpus's stride-(T+1) AR layout would pin each token
    // to one fixed position forever, and its +1 shift target is unused by diffusion.)
    auto flatten = [&](const std::vector<std::string>& texts) {
        std::vector<std::int32_t> ids;
        for (const auto& p : texts) {
            auto v = tok.encode(p);
            ids.insert(ids.end(), v.begin(), v.end());
        }
        return ids;
    };
    const std::vector<std::int32_t> train_ids = flatten(paragraphs);
    const std::vector<std::int32_t> eval_ids  = flatten(eval_texts);
    const auto win_count = [&](const std::vector<std::int32_t>& s) {
        return s.size() - static_cast<std::size_t>(cfg.seq_len) + 1;
    };
    std::println("train stream: {} tokens ({} sliding positions for {}-token windows)",
                 train_ids.size(), win_count(train_ids), cfg.seq_len);
    std::println("eval stream:  {} tokens ({} sliding positions for {}-token windows)",
                 eval_ids.size(), win_count(eval_ids), cfg.seq_len);

    // ── 2b. eval cadence: scale with non-overlapping corpus coverage ────────────
    // Sliding windows overlap heavily (T-1 per window), so use non-overlapping
    // windows for meaningful corpus coverage: total_train_positions / seq_len.
    const std::uint64_t total_train_windows = train_ids.size() / static_cast<std::size_t>(cfg.seq_len);
    if (cfg.eval_every == 0)
        cfg.eval_every = std::max(std::uint64_t(500),
                                  static_cast<std::uint64_t>(total_train_windows * cfg.eval_factor));
    std::println("eval cadence: every {} steps ({} non-overlapping windows, {} sliding)",
                 cfg.eval_every, total_train_windows, win_count(train_ids));

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

        // Random window offset per step — uniform over all sliding positions, the
        // same distribution the eval sweep measures.
        const std::span<const std::int32_t> train_span(train_ids);
        std::uniform_int_distribution<std::size_t> off_dist(0, win_count(train_ids) - 1);

        auto t0 = std::chrono::steady_clock::now();
        auto last_report = t0;
        std::uint64_t interval_steps = 0, interval_tokens = 0;

        float best_nelbo = std::numeric_limits<float>::max();
        std::uint64_t evals_since_best = 0, steps_taken = start_step;
        constexpr float min_improvement = 0.01f;

        // --profile: accumulated wall time per phase, reported with each timing line.
        double p_fwd = 0.0, p_bwd = 0.0, p_opt = 0.0;
        const auto tick = [] { return std::chrono::steady_clock::now(); };

        for (std::uint64_t step = start_step; step < cfg.steps; ++step) {
            const auto window = train_span.subspan(off_dist(rng),
                                                   static_cast<std::size_t>(cfg.seq_len));
            auto pt0 = tick();
            auto res = dt::diffusion_loss(model, window, rng, ctx, 0.02f, cfg.t_max);

            auto pt1 = tick();
            opt.zero_grad();
            res.loss.backward();
            auto pt2 = tick();
            (void)nn::clip_grad_norm(param_ptrs, 5.0f);
            opt.step();
            if (cfg.profile) {
                auto pt3 = tick();
                p_fwd += std::chrono::duration<double>(pt1 - pt0).count();
                p_bwd += std::chrono::duration<double>(pt2 - pt1).count();
                p_opt += std::chrono::duration<double>(pt3 - pt2).count();
            }
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
                if (cfg.profile) {
                    const double tot = p_fwd + p_bwd + p_opt;
                    std::println("        profile: fwd {:.0f}%  bwd {:.0f}%  opt {:.0f}%  "
                                 "({:.2f}/{:.2f}/{:.2f} ms/step)",
                                 100.0 * p_fwd / tot, 100.0 * p_bwd / tot, 100.0 * p_opt / tot,
                                 1e3 * p_fwd / static_cast<double>(interval_steps),
                                 1e3 * p_bwd / static_cast<double>(interval_steps),
                                 1e3 * p_opt / static_cast<double>(interval_steps));
                    p_fwd = p_bwd = p_opt = 0.0;
                }
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
        std::println("Training done: {:.1f}s, {} steps ({}), best held-out NELBO {}",
                     total, steps_taken,
                     steps_taken == cfg.steps ? "hit safety bound" : "self-terminated",
                     best_nelbo == std::numeric_limits<float>::max()
                         ? std::string("n/a (no eval ran)")
                         : std::format("{:.4f}", best_nelbo));
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

    std::println("Recall sweep over the held-out stream ({} sliding positions for {}-token windows):",
                 win_count(eval_ids), cfg.seq_len);
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
