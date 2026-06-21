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
//                           [--patience 3] [--t-max 1.0] [--eval-only]
//                           [--batch-size B] [--threads W] [--overfit N]
//
// MINIMAL-CASE diagnostic (--overfit N): train ONLY the first N non-overlapping windows
// and stop on TRAIN recall over them. A model with enough capacity MUST hit ~100% — N=1
// is degenerate (memorize one constant output), N=4/16/64 force more real modeling. If
// even N=1 can't fit, the bug is structural (capacity/optimizer/loss), not data. Bisect
// "do we need 8 layers / 3 heads?" by finding the smallest arch that still fits each N.
//
// Two ORTHOGONAL training knobs (see the unified trainer in train/parallel.hpp):
//   --batch-size B  effective batch = windows averaged per optimizer step. The QUALITY knob:
//                   gradient within-level consistency ∝ √B (Ch31 sandbox). Default B = W.
//   --threads W     parallel workers that split each step. The SPEED knob ONLY: the gradient
//                   (and the trained model) is IDENTICAL for any W — B windows split B/W per
//                   worker. So a given B (e.g. 16) trains the same on 1, 4, or 16 cores.
//   Mnemonic: pick B for how good the gradient is, W for how fast you get it.
//
// Training is SELF-TERMINATING (a Ch28 lesson): --steps is only a safety bound.
// Every --eval-every steps the held-out NELBO is measured; a checkpoint is saved on
// each improvement, and the run stops after --patience evaluations without one —
// classic early stopping, which on a small corpus also catches memorization (the
// Ch24 failure mode: train loss falls while eval loss climbs).

#include "sub0diff/config/run_config.hpp"
#include "sub0diff/data/token_cache.hpp"
#include "sub0diff/eval/inspect.hpp"
#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/curriculum.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/train/gpu_trainer.hpp"
#include "sub0diff/train/parallel.hpp"
#include "sub0diff/train/schedule.hpp"
#include "sub0diff/train/train_state.hpp"

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/runtime.hpp"

#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/scheduler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace sub0llm;
namespace ag = sub0llm::autograd;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
namespace de = sub0diff::eval;
namespace fs = std::filesystem;

// Short git SHA of the build, injected by CMake (see chapters/CMakeLists.txt). Stamped into
// run_config.json next to the config_sha so a checkpoint records exactly which code produced
// it — the code-sha half of the specialized-build tag.
#ifndef SUB0DIFF_CODE_SHA
#define SUB0DIFF_CODE_SHA "dev"
#endif

static void section(std::string_view title) { std::println("\n── {} ──", title); }

// ── Configuration ─────────────────────────────────────────────────────────────
// The run configuration is the reflected RunConfig module (sub0diff/config): grouped,
// scope-tagged sections with defaults; every --flag, the run_config.json schema, and the
// config-SHA derive from its field tables. Resolution layers defaults → run_config.json in
// the ckpt-dir → CLI, so `--ckpt-dir X` resumes a run with its exact settings.
namespace cfgm = sub0diff::config;
using Config = cfgm::RunConfig;

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

// Raw reader for char-level mode: returns the corpus as contiguous chunks with EVERY
// character preserved — spaces, indentation, and newlines included (only '\r' from CRLF
// is dropped). Unlike read_paragraphs (one line per paragraph, internal whitespace
// collapsed, blank lines skipped) this keeps the verse/line structure that the char
// tokenizer needs. Chunks break only at '\n', so no code point or line is split and
// concatenating them (the flatten step) reproduces the file's char stream exactly. We
// still chunk rather than return one big string so the 95/5 train/eval split and the
// sliding windows have many units to draw from. `limit` caps the chunk count (0 = all).
static std::vector<std::string> read_raw_chunks(const std::string& path, std::int64_t limit) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::format("cannot open corpus: {}", path));
    const std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::string text;
    text.reserve(raw.size());
    for (const char c : raw)
        if (c != '\r') text += c;   // normalise CRLF → LF; keep every other byte

    // ~400 chunks gives a ≥20-unit 5% eval tail; break at the next newline so lines and
    // multi-byte code points stay intact.
    constexpr std::size_t target_chunks = 400;
    const std::size_t chunk_len = std::max<std::size_t>(1, text.size() / target_chunks);
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t end = std::min(text.size(), i + chunk_len);
        while (end < text.size() && text[end] != '\n') ++end;
        if (end < text.size()) ++end;   // include the terminating newline
        out.emplace_back(text.substr(i, end - i));
        i = end;
        if (limit > 0 && out.size() >= static_cast<std::size_t>(limit)) break;
    }
    return out;
}

// ── Reusable support units — extracted to sub0diff for reuse across chapters ──────
//   inspect_recovery → sub0diff/eval/inspect.hpp        (call via de::inspect_recovery)
//   tokcache         → sub0diff/data/token_cache.hpp
//   trainstate       → sub0diff/train/train_state.hpp  (pairs with the run_config module)
namespace tokcache   = sub0diff::data::tokcache;
namespace trainstate = sub0diff::train::trainstate;

static int run(int argc, char** argv);

int main(int argc, char** argv) {
    // Unbuffered stdout so a redirected training log (nohup … > file) is monitorable in
    // real time. The print cadence here is low (a timing line every ~30 s + per-eval), so
    // dropping the block buffer costs nothing — and on Windows _IOLBF degrades to full
    // buffering, so _IONBF is the portable way to get line-by-line flushes.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
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
    try { cfg = cfgm::resolve(argc, argv); }
    catch (const std::exception& e) { std::println("error: {}", e.what()); return 2; }

    // ── 1. the objective, in words ──────────────────────────────────────────────
    section("1. The formal objective — diffusion NELBO as weighted masked LM");
    std::println("L = E[t~U(0,1]] E[mask~Bern(t)] (1/t)·Σ_masked -log p(x0[i]|x_t)");
    std::println("One Monte-Carlo sample per step: draw t, corrupt, weigh the masked CE by");
    std::println("n_masked/(t·T). Every noise level is trained EVERY epoch — Ch28's curriculum");
    std::println("is the special case of reshaping the distribution of t (here: U(0.02, {:.2f}]).",
                 cfg.optim.t_max);

    // ── 2. data: BPE + flat token streams with a held-out split ─────────────────
    if (cfg.data.char_level && cfg.data.word_level)
        throw std::runtime_error("--char-level and --word-level are mutually exclusive");
    const bool raw_mode = cfg.data.char_level || cfg.data.word_level;
    section(cfg.data.char_level ? "2. Data — char-tokenized corpus (whitespace preserved), 95/5 train/eval split"
            : cfg.data.word_level ? "2. Data — word-tokenized corpus (whitespace preserved), 95/5 train/eval split"
                                  : "2. Data — BPE-tokenized corpus, 95/5 train/eval split");
    // Char/word-level modes read the corpus RAW (newlines/indentation kept) so the model can
    // reproduce verse/line structure; BPE mode uses the paragraph-per-line reader.
    auto paragraphs = raw_mode
                          ? read_raw_chunks(cfg.data.corpus, cfg.data.paragraphs)
                          : read_paragraphs(cfg.data.corpus, cfg.data.paragraphs);
    if (paragraphs.size() < 20)
        throw std::runtime_error("corpus too small — need at least 20 paragraphs");
    const std::size_t n_eval = paragraphs.size() * 0.05;   // 5% held out for eval (early stopping, final metrics)
    std::vector<std::string> eval_texts(paragraphs.end() - static_cast<std::ptrdiff_t>(n_eval),
                                        paragraphs.end());
    paragraphs.resize(paragraphs.size() - n_eval);

    std::println("{} train / {} eval paragraphs from {}", paragraphs.size(), n_eval, cfg.data.corpus);
    // BPE caching: training BPE on the full corpus costs ~110s and is DETERMINISTIC,
    // so on resume/eval reload the tokenizer saved in the model dir instead of
    // retraining. This makes --eval-only and resumed training start near-instantly.
    const fs::path tok_dir    = fs::path(cfg.data.ckpt_dir) / "tokenizer";
    const fs::path vocab_json = tok_dir / "vocab.json";
    const fs::path merges_txt = tok_dir / "merges.txt";
    auto tok = [&] {
        if (fs::exists(vocab_json) && fs::exists(merges_txt)) {
            std::println("loaded cached tokenizer from {} (skipped tokenizer build)", tok_dir.string());
            return BPETokenizer::load(vocab_json, merges_txt);
        }
        if (cfg.data.char_level) {
            // Char-level: one token per code point, whitespace/newlines preserved, zero
            // merges — the graceful-degradation control (TRAINING_DESIGN §13.6). vocab is
            // tiny (~the unique chars in the corpus) and pinned into config.json below.
            std::print("building char-level tokenizer (whitespace preserved)... ");
            auto t = BPETokenizer::char_level(paragraphs);
            std::println("done — {} char vocab", t.vocab_size());
            return t;
        }
        if (cfg.data.word_level) {
            // Word-level: one token per whole word (the far end of the granularity spectrum).
            // Every token is a real word, so the model cannot emit a non-word — isolates the
            // §13.6 fragment-coordination mechanism. Large vocab (~unique words in corpus).
            std::print("building word-level tokenizer (one token per word)... ");
            auto t = BPETokenizer::word_level(paragraphs);
            std::println("done — {} word vocab", t.vocab_size());
            return t;
        }
        std::print("training BPE (vocab {})... ", cfg.model.vocab_size);
        auto bpe_start = std::chrono::steady_clock::now();
        auto t = BPETokenizer::train(paragraphs, cfg.model.vocab_size);
        std::println("done in {:.1f}s",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - bpe_start).count());
        return t;
    }();

    // Word-start table (indexed by token id): 1 iff the token carries the GPT-2 leading-
    // space marker Ġ (UTF-8 0xC4 0xA0) = the first subword of a word. Drives whole-word
    // masking and the word-start/continuation recall split. Built once, read-only, shared
    // with the trainer + eval (size = real vocab; clean ids are always < real vocab).
    std::vector<std::uint8_t> is_word_start(tok.vocab_size(), 0);
    for (std::size_t id = 0; id < tok.vocab_size(); ++id) {
        const auto s = tok.token_str(static_cast<BPETokenizer::TokenId>(id));
        is_word_start[id] = (s.size() >= 2 &&
                             static_cast<unsigned char>(s[0]) == 0xC4 &&
                             static_cast<unsigned char>(s[1]) == 0xA0) ? 1 : 0;
    }
    const std::span<const std::uint8_t> ws_span(is_word_start);

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
    // Cache the encoded streams: BPE-encoding the whole corpus is the startup bottleneck
    // and is deterministic, so reload it when (corpus, paragraph-limit, vocab) are unchanged.
    std::vector<std::int32_t> train_ids, eval_ids;
    const fs::path tokens_cache = fs::path(cfg.data.ckpt_dir) / "tokens.bin";
    const std::uint64_t corpus_size =
        fs::exists(cfg.data.corpus) ? static_cast<std::uint64_t>(fs::file_size(cfg.data.corpus)) : 0;
    if (tokcache::load(tokens_cache, corpus_size, cfg.data.paragraphs, tok.vocab_size(),
                       train_ids, eval_ids)) {
        std::println("loaded cached token streams from {} (skipped re-encoding {} paragraphs)",
                     tokens_cache.string(), paragraphs.size() + eval_texts.size());
    } else {
        const auto enc_t0 = std::chrono::steady_clock::now();
        train_ids = flatten(paragraphs);
        eval_ids  = flatten(eval_texts);
        tokcache::save(tokens_cache, corpus_size, cfg.data.paragraphs, tok.vocab_size(),
                       train_ids, eval_ids);
        std::println("encoded + cached token streams to {} ({:.1f}s)", tokens_cache.string(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - enc_t0).count());
    }
    const auto win_count = [&](const std::vector<std::int32_t>& s) {
        return s.size() - static_cast<std::size_t>(cfg.model.seq_len) + 1;
    };
    std::println("train stream: {} tokens ({} sliding positions for {}-token windows)",
                 train_ids.size(), win_count(train_ids), cfg.model.seq_len);
    std::println("eval stream:  {} tokens ({} sliding positions for {}-token windows)",
                 eval_ids.size(), win_count(eval_ids), cfg.model.seq_len);

    // ── 2b/2c. corpus-scaled schedule — ONE tested place (sub0diff/train/schedule.hpp) ──
    // Derives eval cadence, the safety step bound, and eval sample sizes. THE COVERAGE
    // RULE (Ch28): an eval is only a meaningful decision once the model has trained over
    // ≥50% of an epoch (non-overlapping windows) since the last one — below that, evals
    // are noise on partial data and patience trips on noise. The EFFECTIVE BATCH B (--batch,
    // windows/step) is folded in ONCE here — each step trains B windows, NOT W (B⊥W decoupled).
    const auto topo = sub0llm::detect_cpu_topology();
    if (cfg.optim.threads == 0) {
        // Default = ALL physical P-cores. The old P/2 default was tuned on the 1.6M model, which
        // was memory-bandwidth-bound (~flat W=3..8); the founded 6M default arch has ~3.7× the
        // compute per window, so it's COMPUTE-bound and scales: measured W=4→W=8 = ~63→~100+
        // windows/s (~1.6×) on this 6M model. Cap at P-cores, NOT logical: the step is barrier-
        // synced and the slower E-cores would straggle and gate every step (pin policy fills
        // P-cores first). Override with --threads. Shared topology: sub0llm/core/cpu_topology.hpp.
        const int pc = topo.n_perf_cores();
        cfg.optim.threads = std::max<std::size_t>(1, static_cast<std::size_t>(pc > 0 ? pc : topo.n_logical / 2));
    }
    // batch-size B (windows averaged per step → gradient consistency ∝ √B) is INDEPENDENT of the
    // worker count W (--threads, parallelism). Default B = W (the historical per-worker pool).
    // --batch-size B>1 raises consistency on any W; round up to a multiple of W (workers split B/W).
    if (cfg.optim.batch <= 1) cfg.optim.batch = std::max<std::size_t>(1, cfg.optim.threads);
    else cfg.optim.batch = ((cfg.optim.batch + cfg.optim.threads - 1) / cfg.optim.threads) * cfg.optim.threads;
    const bool eval_every_user = cfg.train.eval_every != 0;   // did the user fix the cadence/bound?
    const bool steps_user      = cfg.train.steps != 0;
    const double mean_t = 0.5 * (0.02 + cfg.optim.t_max);
    const dt::ScheduleConfig sc{.eval_factor = cfg.train.eval_factor};
    const auto sched = dt::make_schedule(
        train_ids.size(), eval_ids.size(), cfg.model.seq_len, /*windows_per_step=*/cfg.optim.batch,
        /*masked_per_window=*/mean_t * static_cast<double>(cfg.model.seq_len), sc,
        /*eval_every_override=*/cfg.train.eval_every, /*steps_override=*/cfg.train.steps,
        /*recall_override=*/cfg.diag.recall_windows);
    cfg.train.eval_every                       = sched.eval_every;
    cfg.train.steps                            = sched.steps_bound;
    cfg.diag.recall_windows                   = sched.recall_windows;
    const std::size_t eval_nelbo_windows = sched.eval_nelbo_windows;
    // Early-stopping floor: min_epochs full passes (in STEPS) before a stop may fire. A step trains
    // B = cfg.optim.batch windows (over W workers), so divide epoch windows by B to get steps.
    std::uint64_t min_stop_steps =
        cfg.train.min_epochs * sched.epoch_windows / std::max<std::size_t>(1, cfg.optim.batch);
    // --overfit overrides the corpus-scaled schedule: the train stream is unchanged but we only
    // visit N fixed windows, so the epoch/coverage logic doesn't apply. Watch train recall climb
    // (eval often), cap the run (a model that can't fit N in 20k steps has a structural bug), and
    // drop the min-epochs floor (stop the instant the N windows are fit).
    if (cfg.diag.overfit > 0) {
        if (!eval_every_user) cfg.train.eval_every = 200;
        if (!steps_user)      cfg.train.steps      = 20000;
        min_stop_steps = 0;
    }

    if (cfg.train.eval_factor < sc.min_coverage)
        std::println("note: --eval-factor {:.2f} clamped up to the {:.0f}% epoch-coverage floor "
                     "(below it, evals are noise on partial data — Ch28).",
                     cfg.train.eval_factor, 100.0 * sc.min_coverage);
    std::println("schedule: eval every {} steps = {:.0f}% epoch coverage "
                 "({} epoch windows, {} sliding); steps bound {} (~{:.0f} masked-token epochs)",
                 cfg.train.eval_every, 100.0 * sched.coverage_per_eval, sched.epoch_windows,
                 sched.sliding_train, cfg.train.steps, sc.coverage_epochs);
    std::println("early-stop floor: no stop before {} epochs (= {} steps); patience {} evals after",
                 cfg.train.min_epochs, min_stop_steps, cfg.train.patience);
    std::println("eval samples: nelbo {} windows, recall {}/level; {} worker thread{}",
                 eval_nelbo_windows,
                 cfg.diag.recall_windows == 0 ? std::string("exhaustive")
                                         : std::format("{}", cfg.diag.recall_windows),
                 cfg.optim.threads, cfg.optim.threads == 1 ? "" : "s");
    const auto worker_pins = sub0llm::resolve_pin_set(cfg.optim.pin, topo, cfg.optim.threads);
    if (cfg.optim.threads > 1) {
        std::string pinstr;
        for (std::size_t i = 0; i < worker_pins.size(); ++i)
            pinstr += (i ? "," : "") +
                      (worker_pins[i] < 0 ? std::string("-") : std::format("{}", worker_pins[i]));
        std::println("cpu topology: {} logical, {} physical P-cores ({} P-logical, {} E-logical); "
                     "pin policy '{}' → worker pins [{}]",
                     topo.n_logical, topo.n_perf_cores(), topo.perf.size(),
                     topo.efficiency.size(), cfg.optim.pin, pinstr);
    }

    // ── 3. model + checkpoint resume ────────────────────────────────────────────
    section("3. Model — bidirectional denoiser over the BPE vocabulary");
    const auto V = static_cast<std::int64_t>(tok.vocab_size());
    cfg.model.vocab_size = V;   // pin to the REAL post-BPE vocab so a resume rebuilds the exact arch
    dn::Denoiser model(V, cfg.model.embed_dim, cfg.model.n_heads, cfg.model.n_kv_heads,
                       cfg.model.n_layers, cfg.model.d_ff, static_cast<std::uint64_t>(cfg.optim.seed));
    auto param_ptrs = model.parameters();
    std::vector<ag::Variable> params;                 // shared-impl handles for checkpointing
    params.reserve(param_ptrs.size());
    for (auto* p : param_ptrs) params.push_back(*p);
    std::int64_t n_params = 0;
    for (const auto& p : params) n_params += p.data().numel();
    std::println("Denoiser: V={} (+1 mask), D={}, layers={}, heads={}/{} (GQA), params={}",
                 V, cfg.model.embed_dim, cfg.model.n_layers, cfg.model.n_heads, cfg.model.n_kv_heads, n_params);
    // Proportion sanity notes vs the founded references (head_dim ≥ 32, d_ff = 4·D). The founded
    // A/B (+11pt, §10) settled this: head_dim 32 + d_ff 4·D is the proportioned regime; the defaults
    // already satisfy it, so these only fire when a CLI override drops below the floor. The §12
    // capacity ladder also showed capacity/depth is NOT the binding constraint (data + gradient
    // quality are), so treat low-proportion warnings as advisory, not fatal.
    {
        const std::int64_t head_dim = cfg.model.embed_dim / static_cast<std::int64_t>(cfg.model.n_heads);
        if (head_dim < 32)
            std::println("  note: head_dim={} (< founded floor 32) — proportioning A/B favoured ≥32 "
                         "(advisory; capacity is not the bottleneck per §12).", head_dim);
        if (cfg.model.d_ff < 4 * cfg.model.embed_dim)
            std::println("  note: d_ff={} is below the founded 4·D={} (advisory).",
                         cfg.model.d_ff, 4 * cfg.model.embed_dim);
    }

    fs::create_directories(cfg.data.ckpt_dir);
    std::uint64_t start_step = 0;
    std::string resume_ckpt_path;            // the .ckpt we resumed from (→ matching .opt)
    trainstate::State resume_state;          // curriculum + early-stop state to rehydrate
    if (const auto path = latest_checkpoint_path(cfg.data.ckpt_dir); !path.empty()) {
        start_step = static_cast<std::uint64_t>(load_checkpoint(params, path));
        resume_ckpt_path = path;
        resume_state = trainstate::load(fs::path(cfg.data.ckpt_dir) / "train_state.txt", start_step);
        std::println("resumed from {} (step {}){}", path, start_step,
                     resume_state.have ? " + train_state.txt" : "");
    } else {
        // Fresh run: persist tokenizer + the model-only config.json that Ch30's sampler reads.
        tok.save(fs::path(cfg.data.ckpt_dir) / "tokenizer");
        std::ofstream cj(fs::path(cfg.data.ckpt_dir) / "config.json");
        cj << std::format(
            "{{\n  \"model\": \"sub0diff-denoiser\",\n  \"vocab_size\": {},\n"
            "  \"embed_dim\": {},\n  \"n_layers\": {},\n  \"n_heads\": {},\n"
            "  \"n_kv_heads\": {},\n  \"d_ff\": {},\n  \"seq_len\": {}\n}}\n",
            V, cfg.model.embed_dim, cfg.model.n_layers, cfg.model.n_heads, cfg.model.n_kv_heads, cfg.model.d_ff, cfg.model.seq_len);
        std::println("fresh run — wrote tokenizer/ and config.json to {}", cfg.data.ckpt_dir);
    }
    // Persist the FULL resolved run configuration (corpus, arch, every knob) on EVERY start,
    // so run_config.json always reflects the settings in force — `--ckpt-dir X` alone then
    // resumes this exact run. Stamped with code_sha + config_sha (the specialized-build tag).
    cfgm::write_run_config(cfg.data.ckpt_dir, cfg, SUB0DIFF_CODE_SHA);

    // ── device: move the model to the GPU AFTER the CPU checkpoint load (Stage 4 Phase 7) ───────
    // The optimizer + GpuTrainer are constructed below, so they see the GPU params (Variable::to is
    // in-place → param_ptrs stay valid; Adam's m/v then allocate on-device). Checkpoint save D2H's.
    const bool use_cuda = (cfg.optim.device == "cuda" || cfg.optim.device == "gpu");
    if (use_cuda) {
#ifdef SUB0LLM_CUDA
        model.to(sub0llm::Device::cuda());
        std::println("device: CUDA — single GPU stream (worker pool bypassed; --threads ignored)");
#else
        throw std::runtime_error("--device cuda requested but this binary was built without CUDA "
                                 "(configure the cuda preset)");
#endif
    }

    // ── Gradient-conflict probe (--grad-probe) — a Ch31-sandbox diagnostic ───────
    // At the CURRENT weights, does the gradient from EASY (low-t) maskings point the same way
    // as from HARD (high-t) maskings? Diffusion averages all noise levels in one step (unlike
    // AR's single-token signal); if easy/hard CONFLICT, the averaged step cancels useful signal
    // — a candidate root cause of the sticky unigram basin. Ch28's curriculum is the temporal,
    // BIASED answer to the same heterogeneity; this measures whether the conflict is even real.
    if (cfg.diag.grad_probe) {
        section("Gradient-conflict probe — do low-t (easy) and high-t (hard) maskings agree?");
        const std::size_t total = static_cast<std::size_t>(n_params);
        dt::DiffusionLossContext gctx(cfg.model.seq_len);
        std::mt19937 grng(2024);
        std::uniform_int_distribution<std::size_t> goff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.model.seq_len) - 1);
        const std::vector<float> ts = {0.10f, 0.25f, 0.50f, 0.75f};
        const int N = 128;                          // maskings averaged per noise level
        const std::size_t mid = 2;                  // t = 0.50 for the within-level consistency

        auto add_grads = [&](std::vector<double>& dst, double scale) {
            std::size_t off = 0;
            for (auto* p : param_ptrs) {
                const std::int64_t pn = p->data().numel();
                if (p->grad().numel() == pn) {
                    auto gd = p->grad().data_as<float>();
                    for (std::int64_t i = 0; i < pn; ++i)
                        dst[off + static_cast<std::size_t>(i)] += scale * gd[static_cast<std::size_t>(i)];
                }
                off += static_cast<std::size_t>(pn);
            }
        };
        auto cosd = [&](const std::vector<double>& a, const std::vector<double>& b) {
            double d = 0, na = 0, nb = 0;
            for (std::size_t i = 0; i < total; ++i) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
            return d / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        };

        std::vector<std::vector<double>> g(ts.size(), std::vector<double>(total, 0.0));
        for (std::size_t k = 0; k < ts.size(); ++k)
            for (int s = 0; s < N; ++s) {
                for (auto* p : param_ptrs) p->zero_grad();
                auto w = std::span<const std::int32_t>(eval_ids)
                             .subspan(goff(grng), static_cast<std::size_t>(cfg.model.seq_len));
                dt::diffusion_loss(model, w, grng, gctx, ts[k], ts[k], ws_span,
                                   cfg.optim.whole_word, cfg.optim.exact_noise).loss.backward();
                add_grads(g[k], 1.0 / static_cast<double>(N));
            }
        // Within-level consistency at t=mid: mean cos(single-sample grad, mean grad).
        double consist = 0.0; std::vector<double> sv(total, 0.0);
        for (int s = 0; s < N; ++s) {
            for (auto* p : param_ptrs) p->zero_grad();
            auto w = std::span<const std::int32_t>(eval_ids)
                         .subspan(goff(grng), static_cast<std::size_t>(cfg.model.seq_len));
            dt::diffusion_loss(model, w, grng, gctx, ts[mid], ts[mid], ws_span,
                               cfg.optim.whole_word, cfg.optim.exact_noise).loss.backward();
            std::fill(sv.begin(), sv.end(), 0.0); add_grads(sv, 1.0);
            consist += cosd(sv, g[mid]);
        }

        std::println("Cosine of the MEAN gradient between noise levels (N={} maskings each):", N);
        std::print("         "); for (float t : ts) std::print("  t={:.2f}", t); std::println("");
        for (std::size_t i = 0; i < ts.size(); ++i) {
            std::print("  t={:.2f} ", ts[i]);
            for (std::size_t j = 0; j < ts.size(); ++j) std::print("  {:+.2f} ", cosd(g[i], g[j]));
            std::println("");
        }
        std::println("\nWithin-level consistency at t={:.2f}: mean cos(sample, mean) = {:.3f} "
                     "(~1 low-variance, ~0 every masking differs)", ts[mid],
                     consist / static_cast<double>(N));
        std::println("\nReading: off-diagonal ~1 ⇒ all noise levels AGREE (curriculum unneeded; basin is elsewhere).");
        std::println("         off-diagonal ≤0 ⇒ easy/hard CONFLICT ⇒ averaging cancels signal");
        std::println("                          (curriculum / per-t reweighting / variance reduction justified).");
        return 0;
    }

    // Held-out NELBO: averaged diffusion loss over eval windows at fixed seeds —
    // the honest scalar to compare checkpoints with (lower = better).
    dt::DiffusionLossContext eval_ctx(cfg.model.seq_len);
    auto eval_nelbo = [&](std::size_t n_windows) {
        std::mt19937 erng(777);
        std::uniform_int_distribution<std::size_t> eoff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.model.seq_len) - 1);
        double sum = 0.0;
        for (std::size_t i = 0; i < n_windows; ++i) {
            auto w = std::span<const std::int32_t>(eval_ids)
                         .subspan(eoff(erng), static_cast<std::size_t>(cfg.model.seq_len));
            sum += dt::diffusion_loss(model, w, erng, eval_ctx, 0.02f, 1.0f,
                                      ws_span, cfg.optim.whole_word, cfg.optim.exact_noise,
                                      cfg.optim.contiguous)
                       .loss.data().item<float>();
        }
        return sum / static_cast<double>(n_windows);
    };

    // Held-out NELBO at a FIXED noise level (mask exactly round(t·T)) — the per-level mastery
    // signal the convergence-gated curriculum gates on. Fixed seed so it's comparable epoch to
    // epoch; exact-count so the masked count is exactly the frontier k.
    auto eval_nelbo_at = [&](float t_level, std::size_t n_windows) {
        std::mt19937 erng(778);
        std::uniform_int_distribution<std::size_t> eoff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.model.seq_len) - 1);
        double sum = 0.0;
        for (std::size_t i = 0; i < n_windows; ++i) {
            auto w = std::span<const std::int32_t>(eval_ids)
                         .subspan(eoff(erng), static_cast<std::size_t>(cfg.model.seq_len));
            sum += dt::diffusion_loss(model, w, erng, eval_ctx, t_level, t_level,
                                      ws_span, cfg.optim.whole_word, /*exact_count=*/true,
                                      cfg.optim.contiguous)
                       .loss.data().item<float>();
        }
        return sum / static_cast<double>(n_windows);
    };

    // Quick held-out recall (small fixed budget) — for --track-recall, to watch recall
    // alongside NELBO during training. Fixed seed so the number is comparable across steps.
    auto quick_recall = [&](std::size_t budget) {
        std::mt19937 rr(99);
        de::RecoveryResult agg;
        for (float n : {0.25f, 0.50f}) {
            auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.model.seq_len, n, rr,
                                                nullptr, budget, ws_span, cfg.optim.whole_word);
            agg.accumulate(r);
        }
        return agg;
    };

    // ── --overfit: the N fixed training windows + their exact train-recall ──────────
    // The minimal-case diagnostic (see Config::overfit). The N non-overlapping windows we
    // train on are the first N·seq_len tokens; recall is measured over EXACTLY those windows
    // (not held-out, not straddling slides) so "did the model fit what it was shown?" is a
    // clean yes/no. Same Bernoulli corruption as the held-out sweep for comparability.
    std::vector<std::size_t> overfit_offsets;
    if (cfg.diag.overfit > 0) {
        const auto N = static_cast<std::size_t>(cfg.diag.overfit);
        const auto T = static_cast<std::size_t>(cfg.model.seq_len);
        if (train_ids.size() < N * T)
            throw std::runtime_error(std::format(
                "--overfit {} needs ≥ {} train tokens ({} windows × {}), have {}",
                N, N * T, N, T, train_ids.size()));
        overfit_offsets.reserve(N);
        for (std::size_t i = 0; i < N; ++i) overfit_offsets.push_back(i * T);
    }
    auto overfit_recall_at = [&](float noise, std::mt19937& rr) {
        de::RecoveryResult agg;
        dn::Corruption corr;
        for (auto off : overfit_offsets) {
            auto w = std::span<const std::int32_t>(train_ids).subspan(off, static_cast<std::size_t>(cfg.model.seq_len));
            dn::corrupt_into(w, noise, sub0diff::spec::NoiseSchedule::Absorbing,
                             model.mask_id(), model.real_vocab(), rr, corr);
            agg.accumulate(de::evaluate_recovery(model, w, corr.corrupted, nullptr, ws_span));
        }
        return agg;
    };
    auto overfit_recall = [&] {
        std::mt19937 rr(99);
        de::RecoveryResult agg;
        for (float n : {0.10f, 0.25f, 0.50f, 0.75f}) agg.accumulate(overfit_recall_at(n, rr));
        return agg;
    };

    // ── 4. training ─────────────────────────────────────────────────────────────
    if (!cfg.diag.eval_only && start_step < cfg.train.steps) {
        section("4. Training — self-terminating on held-out NELBO (early stopping)");
        // Muon wants a larger LR than Adam — default to 0.02 if the user didn't pass --lr.
        std::unique_ptr<nn::Optimizer> opt;
        if (cfg.optim.optimizer == "muon") {
            if (!cfg.optim.lr_explicit) cfg.optim.lr = 0.02f;
            // Canonical Muon routing: the token embedding / output head (any 2D param with a
            // model_vocab=V+1 dimension) goes to AdamW, NOT Muon — orthogonalising their
            // frequency-skewed gradients collapses training. Hidden weight matrices get Muon.
            std::vector<char> force_adamw(param_ptrs.size(), 0);
            int n_emb = 0;
            for (std::size_t i = 0; i < param_ptrs.size(); ++i) {
                const auto& sh = param_ptrs[i]->data().shape();
                if (sh.size() == 2 && (sh[0] == V + 1 || sh[1] == V + 1)) {
                    force_adamw[i] = 1; ++n_emb;
                }
            }
            // AdamW-group LR = 1e-3 (the rate plain Adam learns the embedding well at here;
            // the modded-nanogpt 3e-4 was too slow for this small-model embedding).
            const float adamw_grp_lr = 1e-3f;
            opt = std::make_unique<nn::Muon>(param_ptrs, cfg.optim.lr, 0.95f, 5, true,
                                             adamw_grp_lr, 0.9f, 0.999f, 1e-8f, 0.0f,
                                             std::move(force_adamw));
            std::println("optimizer: muon (lr {:.1e}; AdamW-group lr {:.1e}; {} vocab-dim "
                         "params → AdamW)", cfg.optim.lr, adamw_grp_lr, n_emb);
        } else {
            opt = nn::make_optimizer(cfg.optim.optimizer, param_ptrs, cfg.optim.lr);
            std::println("optimizer: {} (lr {:.1e})", cfg.optim.optimizer, cfg.optim.lr);
        }
        // HONEST RESUME: restore Adam's (m, v, t) from the .opt next to the resumed .ckpt, so the
        // optimizer keeps its adaptive per-parameter rates instead of re-warming from zero. The
        // .opt path mirrors the weight checkpoint (step_*.ckpt → step_*.opt).
        if (!resume_ckpt_path.empty()) {
            fs::path opt_path = resume_ckpt_path;
            opt_path.replace_extension(".opt");
            if (fs::exists(opt_path) && opt->load_state(opt_path.string()))
                std::println("restored optimizer state from {}", opt_path.string());
            else
                std::println("note: no matching optimizer state ({}) — Adam moments start fresh "
                             "(a brief re-warm)", opt_path.string());
        }
        // LR schedule is OPT-IN (--warmup-steps N): a measured A/B showed warmup+cosine
        // HURT the known-good baseline (12.0% vs 14.8%) by decaying it into an earlier
        // early-stop, and did NOT rescue wider configs that get stuck in the unigram
        // basin (that is an optimization/init problem, not an LR-schedule one). Default
        // = constant cfg.optim.lr, the regime the baseline trains well at.
        std::optional<nn::CosineWithWarmup> lr_sched;
        if (cfg.optim.warmup_steps > 0) {
            lr_sched.emplace(cfg.optim.lr, cfg.optim.lr * 0.1f, static_cast<std::int64_t>(cfg.optim.warmup_steps),
                             static_cast<std::int64_t>(cfg.train.steps));
            std::println("lr schedule: warmup {} steps to {:.1e}, cosine decay to {:.1e}",
                         cfg.optim.warmup_steps, cfg.optim.lr, cfg.optim.lr * 0.1f);
        } else {
            std::println("lr: constant {:.1e} (no warmup; --warmup-steps N to enable)", cfg.optim.lr);
        }
        std::mt19937 rng(1234 + static_cast<std::uint32_t>(start_step));
        // Effective batch B windows/step (gradient consistency ∝ √B), split across W=threads
        // workers (B/W each). The master gradient is the mean over all B windows, identical for
        // any W — so single-thread (W=1) trains the SAME approach, just slower.
        const std::size_t windows_per_step = cfg.optim.batch;

        // Random window offset per step — uniform over all sliding positions, the
        // same distribution the eval sweep measures.
        const std::span<const std::int32_t> train_span(train_ids);
        std::uniform_int_distribution<std::size_t> off_dist(0, win_count(train_ids) - 1);

        auto t0 = std::chrono::steady_clock::now();
        auto last_report = t0;
        std::uint64_t interval_steps = 0, interval_tokens = 0;

        // Early-stop history rehydrated on an honest resume (else fresh).
        float best_nelbo = resume_state.have ? resume_state.best_nelbo
                                             : std::numeric_limits<float>::max();
        std::uint64_t evals_since_best = resume_state.have ? resume_state.evals_since_best : 0;
        std::uint64_t steps_taken = start_step;
        constexpr float min_improvement = 0.01f;

        // --profile: accumulated wall time per phase, reported with each timing line.
        double p_fwd = 0.0, p_bwd = 0.0, p_opt = 0.0;
        const auto tick = [] { return std::chrono::steady_clock::now(); };

        // Unified trainer: W=cfg.optim.threads workers cooperatively process B=cfg.optim.batch windows/step
        // (B/W each), master sees the MEAN gradient over all B. ALWAYS used (W=1 is a pool of one),
        // so single- and multi-threaded training run the identical fundamental step.
        // Back-end-agnostic trainer (ITrainer): the CPU worker pool, or — under --device cuda — a
        // single GPU stream (no threads/reduce; the GPU is the parallelism). main()'s loop below is
        // identical for both: set_t_range() + step().
        std::unique_ptr<dt::ITrainer> pool;
        if (use_cuda) {
            pool = std::make_unique<dt::GpuTrainer>(
                model, param_ptrs, cfg.model.seq_len, 0.02f, cfg.optim.t_max, 1234 + start_step,
                static_cast<std::int64_t>(cfg.optim.batch), cfg.optim.shared_t, cfg.optim.exact_noise,
                ws_span, cfg.optim.whole_word, cfg.optim.contiguous);
            std::println("trainer: batch-size {} windows/step on ONE GPU stream; shared-t {}, "
                         "exact-noise {}, masking {}",
                         cfg.optim.batch, cfg.optim.shared_t ? "on" : "off",
                         cfg.optim.exact_noise ? "on" : "off",
                         cfg.optim.contiguous ? "CONTIGUOUS-span" : "scattered");
        } else {
            pool = std::make_unique<dt::ParallelTrainer>(
                cfg.optim.threads,
                dt::ParallelTrainer::Arch{V, cfg.model.embed_dim, cfg.model.n_layers, cfg.model.d_ff,
                                          cfg.model.seq_len,
                                          static_cast<std::size_t>(cfg.model.n_heads),
                                          static_cast<std::size_t>(cfg.model.n_kv_heads)},
                param_ptrs, 0.02f, cfg.optim.t_max, 1234 + start_step,
                /*share_weights=*/true, ws_span, cfg.optim.whole_word, worker_pins, cfg.optim.exact_noise,
                static_cast<std::int64_t>(cfg.optim.batch), cfg.optim.shared_t, cfg.optim.contiguous);
            std::println("trainer: batch-size {} windows/step (consistency ∝ √{}) split across {} "
                         "worker{} (speed only — same gradient at any W); shared-t {}, exact-noise {}, masking {}",
                         cfg.optim.batch, cfg.optim.batch, cfg.optim.threads, cfg.optim.threads == 1 ? "" : "s",
                         cfg.optim.shared_t ? "on" : "off", cfg.optim.exact_noise ? "on" : "off",
                         cfg.optim.contiguous ? "CONTIGUOUS-span" : "scattered");
        }
        std::vector<std::size_t> offsets(cfg.optim.batch);
        std::size_t overfit_cursor = 0;   // cycles the N fixed windows across the B step slots
        if (cfg.diag.overfit > 0)
            std::println("OVERFIT mode: training only the first {} non-overlapping window{} "
                         "(stop on TRAIN recall ≥ 99%; cap {} steps)",
                         cfg.diag.overfit, cfg.diag.overfit == 1 ? "" : "s", cfg.train.steps);

        // Frontier-point curriculum (opt-in): train AT a noise ceiling that rises only
        // once mastered. The floor stays at 0.02; frontier-point trains exactly at the
        // ceiling. Disabled → uniform t ∈ (0.02, t_max] (the formal objective).
        std::unique_ptr<dt::NoiseCurriculum> curr;
        // Convergence-gated frontier curriculum (--curriculum-converge): integer-k, advanced
        // per-epoch on held-out NELBO plateau. Mutually exclusive with the token-gated one.
        std::unique_ptr<dt::FrontierCurriculum> fcurr;
        if (cfg.curriculum.curriculum && cfg.curriculum.curriculum_converge)
            throw std::runtime_error("--curriculum and --curriculum-converge are mutually exclusive");
        if (cfg.curriculum.curriculum) {
            curr = std::make_unique<dt::NoiseCurriculum>(
                dt::NoiseCurriculum::Config{.end = cfg.curriculum.curriculum_end,
                                            .min_tokens = cfg.curriculum.curriculum_min_tokens});
            std::println("curriculum: frontier-point, ceiling {:.2f}→{:.2f}, token-gated "
                         "(min_tokens={})",
                         0.05f, cfg.curriculum.curriculum_end, cfg.curriculum.curriculum_min_tokens);
        } else if (cfg.curriculum.curriculum_converge) {
            fcurr = std::make_unique<dt::FrontierCurriculum>(dt::FrontierCurriculum::Config{
                .seq_len = cfg.model.seq_len, .k_start = cfg.curriculum.curriculum_k_start, .k_max = 0,
                .k_step = cfg.curriculum.curriculum_k_step,
                .patience = static_cast<int>(cfg.curriculum.curriculum_patience)});
            // Honest resume: rehydrate the exact curriculum state from the sidecar (continue
            // mid-climb) — overrides --curriculum-k-start, which is the manual fallback.
            if (resume_state.have)
                fcurr->restore(resume_state.curr_k, resume_state.curr_best,
                               resume_state.curr_stalls, resume_state.curr_converged);
            std::println("curriculum: CONVERGENCE-gated, k={}→{} masked tokens (t={:.3f}→1.0), "
                         "+{}/level on per-epoch NELBO plateau (patience {} epochs); global "
                         "early-stop suspended until converged{}",
                         fcurr->level(), fcurr->max_level(), fcurr->frontier(),
                         cfg.curriculum.curriculum_k_step, cfg.curriculum.curriculum_patience,
                         fcurr->converged() ? " [resumed: CONVERGED]"
                       : resume_state.have ? std::format(" [resumed at k={}]", fcurr->level())
                                           : "");
        }

        // Honest checkpoint: weights (step_*.ckpt) + Adam state (step_*.opt) + the curriculum/
        // early-stop sidecar (train_state.txt), all stamped with the same step so a resume
        // restores the FULL training state. Replaces bare save_checkpoint at every save site.
        auto save_full = [&](std::uint64_t step) {
            save_checkpoint(params, cfg.data.ckpt_dir, static_cast<std::int64_t>(step));
            opt->save_state((fs::path(cfg.data.ckpt_dir) / std::format("step_{:09d}.opt", step)).string());
            trainstate::State s;
            s.step = step;
            if (fcurr) {
                s.curr_k = fcurr->level();       s.curr_best = fcurr->best();
                s.curr_stalls = fcurr->stalls(); s.curr_converged = fcurr->converged();
            }
            s.best_nelbo = best_nelbo;
            s.evals_since_best = evals_since_best;
            trainstate::save(fs::path(cfg.data.ckpt_dir) / "train_state.txt", s);
        };

        float last_loss = 0.0f, last_t = 0.0f;
        for (std::uint64_t step = start_step; step < cfg.train.steps; ++step) {
            auto pt0 = tick();
            std::uint64_t step_masked = 0;
            auto pt1 = pt0, pt2 = pt0;
            // Frontier-point DURING the ramp (sample t exactly at the ceiling — fast
            // per-bucket mastery, Ch28's winner). Once the ceiling CONVERGES at the top,
            // switch to the full band [0.02, ceiling] = the formal objective: training a
            // single t=1.0 (fully masked, no context) is degenerate and degrades the easy
            // regime, which made the uniform-noise eval climb and early-stop prematurely.
            float t_lo, t_hi;
            if (fcurr) {
                // Convergence-gated: train AT exactly k masked (t=k/T) until the curriculum
                // converges, then the full formal band. Advancement is decided per-EPOCH below.
                if (fcurr->converged()) { t_lo = 0.02f; t_hi = cfg.optim.t_max; }
                else                    { t_lo = t_hi = fcurr->frontier(); }
            } else {
                t_lo = (curr && !curr->converged()) ? curr->frontier() : 0.02f;
                t_hi = curr ? curr->frontier() : cfg.optim.t_max;
            }
            // Set the noise band the trainer samples from (curriculum ceiling, or the formal band).
            // shared-t sampling (one t for all B windows) happens inside pool->step() from this band.
            pool->set_t_range(t_lo, t_hi);
            // OVERFIT: cycle the N fixed windows across the B slots (full-batch GD over them).
            // Else: a fresh random sliding offset per slot (the held-out eval distribution).
            if (cfg.diag.overfit > 0)
                for (auto& o : offsets) o = overfit_offsets[overfit_cursor++ % overfit_offsets.size()];
            else
                for (auto& o : offsets) o = off_dist(rng);
            auto res = pool->step(train_span, offsets);
            last_loss   = res.mean_loss;
            last_t      = res.last_t;
            step_masked = res.masked_tokens;
            pt2 = pt1 = tick();   // fwd+bwd fused inside the pool
            // Token-gated mastery feedback: actual mask fraction ≈ the frontier.
            if (curr) curr->observe(last_t, last_loss, step_masked);
            (void)nn::clip_grad_norm(param_ptrs, 5.0f);
            if (lr_sched) opt->set_lr((*lr_sched)(static_cast<std::int64_t>(step)));
            opt->step();
            if (cfg.diag.profile) {
                auto pt3 = tick();
                p_fwd += std::chrono::duration<double>(pt1 - pt0).count();
                p_bwd += std::chrono::duration<double>(pt2 - pt1).count();
                p_opt += std::chrono::duration<double>(pt3 - pt2).count();
            }
            ++interval_steps;
            interval_tokens += step_masked;
            steps_taken = step + 1;

            auto now = std::chrono::steady_clock::now();
            const double since_report = std::chrono::duration<double>(now - last_report).count();
            if (since_report >= 30.0) {
                const double el = std::chrono::duration<double>(now - t0).count();
                // THROUGHPUT METRIC COHERENCE: a data-parallel "step" trains `threads`
                // windows (grad accumulation)
                const double onethread_steps_s = static_cast<double>(interval_steps) / since_report;
                const double windows_s   = onethread_steps_s * static_cast<double>(windows_per_step);
                const std::string curr_tag =
                    curr  ? std::format("  ceiling={:.2f}{}", curr->frontier(),
                                        curr->converged() ? " (converged)" : "")
                  : fcurr ? std::format("  k={}/{}{}", fcurr->level(), fcurr->max_level(),
                                        fcurr->converged() ? " (converged)" : "")
                          : std::string();
                std::println("{:>5.0f}s  step {:>6}  nelbo={:.4f} (t={:.2f}){}  "
                             "[{:.0f} windows/s, {:.0f} masked-tok/s]",
                             el, step, last_loss, last_t, curr_tag,
                             windows_s,
                             static_cast<double>(interval_tokens) / since_report);
                if (cfg.diag.profile) {
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

            // ── OVERFIT: stop on TRAIN recall over the N fixed windows (not held-out) ──
            if (cfg.diag.overfit > 0 && (step + 1) % cfg.train.eval_every == 0) {
                const auto r = overfit_recall();
                std::println("  overfit @ {:>6}: TRAIN recall {:.1f}% (word {:.1f}%, START {:.1f}%) "
                             "over {} window{} | train-nelbo {:.4f}",
                             step + 1, r.recall() * 100.0f, r.word_recall() * 100.0f,
                             r.ws_recall() * 100.0f, cfg.diag.overfit, cfg.diag.overfit == 1 ? "" : "s",
                             last_loss);
                if (r.recall() >= 0.99f) {
                    save_full(step + 1);
                    std::println("  overfit FIT ✓ — train recall {:.1f}% ≥ 99% at step {}: capacity "
                                 "SUFFICIENT for N={} (D={}, L={}, H={})",
                                 r.recall() * 100.0f, step + 1, cfg.diag.overfit,
                                 cfg.model.embed_dim, cfg.model.n_layers, cfg.model.n_heads);
                    steps_taken = step + 1;
                    break;
                }
                continue;   // overfit never runs the held-out/patience path below
            }

            // ── Convergence-gated curriculum: advance the frontier per EPOCH ───────
            // While the curriculum is progressing, the level-NELBO (held-out, exactly k masked)
            // is the mastery signal; global early-stop is SUSPENDED (the easy-level global NELBO
            // is dominated by the untrained high-k tail and would trip patience spuriously).
            if (fcurr && !fcurr->converged() && (step + 1) % cfg.train.eval_every == 0) {
                const float fnelbo   = static_cast<float>(eval_nelbo_at(fcurr->frontier(),
                                                                        eval_nelbo_windows));
                const float prev_best = fcurr->best();
                const std::int64_t k_before = fcurr->level();
                const bool improved  = fnelbo < prev_best - 0.01f;
                const bool advanced  = fcurr->observe_epoch(fnelbo);
                if (improved || advanced || fcurr->converged())
                    save_full(step + 1);
                // FORGETTING WATCH: frontier-point trains EXACTLY at k, so while climbing the model
                // no longer sees the easy k=1 regime. In theory it shouldn't forget (the post-
                // convergence full-objective phase revisits every level), but watch the base-level
                // (k=1) NELBO: if it CLIMBS as k rises, the easy levels are degrading and the
                // curriculum should be reworked as a cap/bias over [1,k] (see curriculum.hpp).
                const std::string base_tag = (k_before > 1)
                    ? std::format("  base(k=1)-NELBO {:.4f}",
                                  eval_nelbo_at(1.0f / static_cast<float>(cfg.model.seq_len),
                                                eval_nelbo_windows))
                    : std::string();
                std::println("  curriculum @ {:>6}: k={}/{} (t={:.3f}) level-NELBO {:.4f}{}  {}",
                             step + 1, k_before, fcurr->max_level(),
                             static_cast<double>(k_before) / static_cast<double>(cfg.model.seq_len),
                             fnelbo, base_tag,
                             advanced ? std::format("mastered → advance to k={}", fcurr->level())
                           : fcurr->converged() ? "mastered TOP → CONVERGED (→ full objective)"
                           : improved ? "learning"
                           : std::format("stall {}/{}", fcurr->stalls(), cfg.curriculum.curriculum_patience));
                if (fcurr->converged()) {   // hand off to global early-stop, fresh
                    best_nelbo = std::numeric_limits<float>::max();
                    evals_since_best = 0;
                }
                continue;
            }

            // ── Convergence check: held-out NELBO every eval_every steps ───────────
            if ((step + 1) % cfg.train.eval_every == 0) {
                const float nelbo = static_cast<float>(eval_nelbo(eval_nelbo_windows));
                if (cfg.diag.track_recall) {
                    const auto qr = quick_recall(800);
                    std::println("  track @ {:>6}: held-out recall {:.1f}% (word {:.1f}%, "
                                 "START {:.1f}%) | NELBO {:.4f}",
                                 step + 1, qr.recall() * 100.0f, qr.word_recall() * 100.0f,
                                 qr.ws_recall() * 100.0f, nelbo);
                }
                if (nelbo < best_nelbo - min_improvement) {
                    best_nelbo = nelbo;
                    evals_since_best = 0;
                    save_full(step + 1);
                    std::println("  eval @ {:>6}: held-out NELBO {:.4f}  ← best, checkpointed",
                                 step + 1, nelbo);
                } else {
                    ++evals_since_best;
                    // Don't stop before min_epochs full passes — a NELBO plateau this
                    // early is undertraining (often a unigram collapse), not convergence.
                    const bool floor_reached = steps_taken >= min_stop_steps;
                    std::println("  eval @ {:>6}: held-out NELBO {:.4f}  (best {:.4f}, {}/{} patience{})",
                                 step + 1, nelbo, best_nelbo, evals_since_best, cfg.train.patience,
                                 floor_reached ? "" : ", below min-epochs floor");
                    if (evals_since_best >= cfg.train.patience && floor_reached) {
                        std::println("  early stop: no improvement for {} evals (after {} min epochs)",
                                     cfg.train.patience, cfg.train.min_epochs);
                        break;
                    }
                }
            }
        }
        const double total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::println("Training done: {:.1f}s, {} steps ({}), best held-out NELBO {}",
                     total, steps_taken,
                     steps_taken == cfg.train.steps ? "hit safety bound" : "self-terminated",
                     best_nelbo == std::numeric_limits<float>::max()
                         ? std::string("n/a (no eval ran)")
                         : std::format("{:.4f}", best_nelbo));
        // Reload the BEST checkpoint (the loop may have overfit past it) so §5
        // evaluates — and Ch30 inherits — the early-stopping winner, not the tail.
        // SKIP in --overfit: we want the LIVE trained weights (a stale checkpoint from a
        // prior run in this dir must not clobber them), and the verdict is the train recall.
        if (cfg.diag.overfit == 0)
            if (const auto best = latest_checkpoint_path(cfg.data.ckpt_dir); !best.empty())
                (void)load_checkpoint(params, best);
    } else if (cfg.diag.eval_only) {
        section("4. Training — skipped (--eval-only)");
    } else {
        section("4. Training — skipped (checkpoint already at target steps)");
    }

    // Post-training diagnostics (§5 held-out eval, recall sweep, per-t) now run on the model's
    // device: held-out NELBO is diffusion_loss (CUDA), and evaluate_recovery runs the forward on
    // the GPU and D2H's only the (T,vocab) logits for its host-side argmax. So a GPU-trained model
    // keeps its diagnostics on the GPU — the recall sweep was the dominant wall-time cost of a full
    // run, and the forward is the expensive part. (--inspect is opt-in and not exercised here.)

    // ── 5'. OVERFIT verdict: per-noise TRAIN recall over the N fixed windows ────────
    // Held-out eval is meaningless here (we deliberately trained on N windows only), so
    // report the diagnostic — can the model reproduce exactly what it was shown? — and exit.
    if (cfg.diag.overfit > 0) {
        section("5. Overfit verdict — TRAIN recall over the fixed windows");
        std::println("Model: D={}, layers={}, heads={}/{} (GQA), d_ff={}, params={} | N={} window{}",
                     cfg.model.embed_dim, cfg.model.n_layers, cfg.model.n_heads, cfg.model.n_kv_heads, cfg.model.d_ff,
                     n_params, cfg.diag.overfit, cfg.diag.overfit == 1 ? "" : "s");
        std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
        std::mt19937 orng(4242);
        de::RecoveryResult agg;
        for (float noise : {0.10f, 0.25f, 0.50f, 0.75f}) {
            auto r = overfit_recall_at(noise, orng);
            agg.accumulate(r);
            std::println("  {:>5.0f}%  {:>8}  {:>10}  {:>6.1f}%",
                         noise * 100.0f, r.masked, r.hits, r.recall() * 100.0f);
        }
        const float rec = agg.recall();
        std::println("  overall TRAIN recall: {}/{} ({:.1f}%) | word-level {:.1f}% | word-START {:.1f}%",
                     agg.hits, agg.masked, rec * 100.0f,
                     agg.word_recall() * 100.0f, agg.ws_recall() * 100.0f);
        // FIT bar (0.98) sits just below the loop's stop bar (0.99): the final table re-rolls
        // corruption at a fresh seed, and single-window recall at high noise has real boundary
        // variance — 98%+ is unambiguously "fit", so don't flip a stopped run to PARTIAL on noise.
        std::println("\nVerdict: {}",
                     rec >= 0.98f
                         ? std::format("FIT ✓ — this arch can reproduce {} window{}; capacity is "
                                       "NOT the bottleneck at N={}.", cfg.diag.overfit,
                                       cfg.diag.overfit == 1 ? "" : "s", cfg.diag.overfit)
                     : rec >= 0.80f
                         ? std::format("PARTIAL ({:.0f}%) — close; try more steps (--steps), a wider "
                                       "model, or a higher LR (--lr).", rec * 100.0f)
                         : std::format("FAIL ({:.0f}%) — this arch could not even MEMORIZE {} "
                                       "window{} in {} steps. Structural issue (capacity / optimizer "
                                       "/ loss / conditioning), not data — investigate here, cheaply, "
                                       "before any corpus-scale run.", rec * 100.0f, cfg.diag.overfit,
                                       cfg.diag.overfit == 1 ? "" : "s", cfg.train.steps));
        return 0;
    }

    // ── 5. held-out evaluation: NELBO + recall sweep + edge profile ─────────────
    section("5. Held-out evaluation");
    std::println("held-out NELBO ({} windows): {:.4f}\n",
                 4 * eval_nelbo_windows, eval_nelbo(4 * eval_nelbo_windows));

    // 5a. Per-t diagnostic (--per-t): raw per-token NLL + recall vs noise level. The headline
    // NELBO is this curve AVERAGED over t — a single scalar hides whether the model is t-aware.
    if (cfg.diag.per_t) {
        section("5a. Per-t diagnostic — is the model t-aware?");
        // Unigram-entropy floor H0 = the irreducible t→1 loss (predict from an empty canvas).
        // A t-AGNOSTIC model's masked-CE sits near H0 at EVERY t (flat curve); a healthy model
        // rises from well below 1 nat at low t up toward H0 — that's "t-awareness".
        std::vector<std::uint64_t> freq(static_cast<std::size_t>(model.model_vocab()), 0);
        for (auto id : eval_ids)
            if (id >= 0 && id < model.model_vocab()) ++freq[static_cast<std::size_t>(id)];
        double H0 = 0.0; const double Ntok = static_cast<double>(eval_ids.size());
        for (auto f : freq) if (f) { const double p = static_cast<double>(f) / Ntok; H0 -= p * std::log(p); }
        std::println("  unigram-entropy floor H0 = {:.3f} nats (flat curve ≈ H0 ⇒ t-agnostic; "
                     "rising from <1 nat ⇒ t-aware / near floor)", H0);
        std::println("  {:>5}  {:>16}  {:>8}", "t", "masked-CE (nats)", "recall");
        std::mt19937 prng(2025);
        std::uniform_int_distribution<std::size_t> poff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.model.seq_len) - 1);
        constexpr std::size_t ce_windows = 512;
        const std::size_t rbudget = std::min<std::size_t>(2000, win_count(eval_ids));
        for (float t : {0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.40f,
                        0.50f, 0.60f, 0.70f, 0.80f, 0.90f, 0.95f}) {
            double ce = 0.0;
            for (std::size_t i = 0; i < ce_windows; ++i) {
                auto w = std::span<const std::int32_t>(eval_ids)
                             .subspan(poff(prng), static_cast<std::size_t>(cfg.model.seq_len));
                ce += dt::diffusion_loss(model, w, prng, eval_ctx, t, t,
                                         ws_span, /*whole_word=*/false, /*exact_count=*/true).mean_ce;
            }
            auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.model.seq_len, t, prng,
                                                nullptr, rbudget, ws_span, /*whole_word=*/false);
            std::println("  {:>5.2f}  {:>16.3f}  {:>7.1f}%",
                         t, ce / static_cast<double>(ce_windows), r.recall() * 100.0f);
        }
        std::println("");
    }

    // Budgeted sweep: a few thousand uniformly-strided windows per noise level
    // estimate recall to ~±1%; an exhaustive sweep of a large eval stream can cost
    // ORDERS more wall time than training did (467K positions × 4 levels ≈ 75 min
    // on complete_shakespeare). --recall-windows 0 restores the exhaustive sweep.
    const std::size_t sweep_budget =
        cfg.diag.recall_windows == 0 ? win_count(eval_ids)
                                : std::min<std::size_t>(cfg.diag.recall_windows, win_count(eval_ids));
    std::println("Recall sweep over the held-out stream: {} of {} sliding positions per noise "
                 "level ({}-token windows, uniform stride){}:",
                 sweep_budget, win_count(eval_ids), cfg.model.seq_len,
                 cfg.optim.whole_word ? " [WHOLE-WORD masking]" : "");
    std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
    std::mt19937 eval_rng(4242);
    de::PositionStats pos(static_cast<std::size_t>(cfg.model.seq_len));
    de::RecoveryResult sweep;
    const auto sweep_t0 = std::chrono::steady_clock::now();
    for (float noise : {0.10f, 0.25f, 0.50f, 0.75f}) {
        auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.model.seq_len, noise, eval_rng,
                                            &pos, cfg.diag.recall_windows, ws_span, cfg.optim.whole_word);
        sweep.accumulate(r);
        std::println("  {:>5.0f}%  {:>8}  {:>10}  {:>6.1f}%",
                     noise * 100.0f, r.masked, r.hits, r.recall() * 100.0f);
    }
    const double sweep_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sweep_t0).count();
    std::println("  overall: {}/{} ({:.1f}%)  [sweep took {:.1f}s]",
                 sweep.hits, sweep.masked, sweep.recall() * 100.0f, sweep_s);
    // What the headline number is made of (axes H/I): the honest hard targets are
    // word-START tokens; word-level recall credits a whole word only if all its masked
    // subwords are recovered. A big START≪CONTINUATION gap means per-token recall is
    // inflated by partial-word completion (and whole-word masking is the fix).
    std::println("  breakdown: word-level {:.1f}% ({}/{}) | word-START {:.1f}% ({}/{}) | "
                 "continuation {:.1f}% ({}/{})",
                 sweep.word_recall() * 100.0f, sweep.word_hits, sweep.word_total,
                 sweep.ws_recall() * 100.0f, sweep.ws_hits, sweep.ws_masked,
                 sweep.wc_recall() * 100.0f, sweep.wc_hits, sweep.wc_masked);

    float interior = 0.0f; int n_int = 0;
    for (std::int64_t t = 1; t + 1 < cfg.model.seq_len; ++t) {
        interior += pos.recall_at(static_cast<std::size_t>(t)); ++n_int;
    }
    std::println("\nEdge effect: first {:.1f}%, last {:.1f}% vs interior mean {:.1f}%",
                 pos.recall_at(0) * 100.0f,
                 pos.recall_at(static_cast<std::size_t>(cfg.model.seq_len - 1)) * 100.0f,
                 interior / static_cast<float>(n_int) * 100.0f);

    // ── 5b. MEMORIZATION check: the SAME sweep on the TRAIN stream (--eval-train) ──
    // train≫held-out ⇒ the model fits/memorizes training fine and the limit is
    // GENERALIZATION (the Ch28-comparable number is THIS one). train≈held-out ⇒ the
    // model isn't even fitting training (undertraining / optimization / data shape).
    if (cfg.diag.eval_train) {
        std::mt19937 train_rng(4242);
        de::RecoveryResult tsweep;
        std::println("\nTrain-stream recall (MEMORIZATION; held-out above is generalization):");
        std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
        for (float noise : {0.10f, 0.25f, 0.50f, 0.75f}) {
            auto r = de::evaluate_corpus_recall(model, train_ids, cfg.model.seq_len, noise, train_rng,
                                                nullptr, cfg.diag.recall_windows, ws_span, cfg.optim.whole_word);
            tsweep.accumulate(r);
            std::println("  {:>5.0f}%  {:>8}  {:>10}  {:>6.1f}%",
                         noise * 100.0f, r.masked, r.hits, r.recall() * 100.0f);
        }
        std::println("  TRAIN overall: {}/{} ({:.1f}%)  vs  held-out {:.1f}%  → gap {:.1f}pts",
                     tsweep.hits, tsweep.masked, tsweep.recall() * 100.0f,
                     sweep.recall() * 100.0f,
                     (tsweep.recall() - sweep.recall()) * 100.0f);
    }

    // ── 5c. per-sample inspection (--inspect N) ──────────────────────────────────
    if (cfg.diag.inspect > 0) {
        section("5c. Per-sample recovery inspection (--inspect)");
        const int n = static_cast<int>(cfg.diag.inspect);
        de::inspect_recovery(model, tok, eval_ids, cfg.model.seq_len, ws_span, cfg.optim.whole_word, n, 0.25f);
        de::inspect_recovery(model, tok, eval_ids, cfg.model.seq_len, ws_span, cfg.optim.whole_word, n, 0.50f);
    }

    section("What's next");
    std::println("Ch30 — the reverse process: iterative canvas refinement with confidence remasking,");
    std::println("       loading this chapter's model dir ({}).", cfg.data.ckpt_dir);
    return 0;
}
