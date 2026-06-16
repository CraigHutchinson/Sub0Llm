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

#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/train/curriculum.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/train/parallel.hpp"
#include "sub0diff/train/schedule.hpp"

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

static void section(std::string_view title) { std::println("\n── {} ──", title); }

// ── Configuration (CLI-overridable) ───────────────────────────────────────────
struct Config {
    std::string corpus     = "data/shakespeare.txt";
    std::string ckpt_dir   = "/tmp/sub0diff_ch29";
    std::int64_t paragraphs = 0;      // 0 or -1 = read all paragraphs from corpus
    std::size_t vocab_size = 512;
    std::int64_t seq_len   = 64;
    std::uint64_t steps    = 0;        // 0 = corpus-scaled safety bound (see §2c)
    std::uint64_t eval_every = 0;      // 0 = compute from corpus; override with --eval-every
    // Epochs of train coverage per eval. 1.0 = eval once per FULL epoch (was 0.5): a full-epoch-
    // separated held-out NELBO is a much cleaner decision signal than a half-epoch one, and it
    // halves the eval churn (each eval costs a recall+NELBO sweep). The hard floor stays 0.5.
    double eval_factor = 1.0;
    // Stop after this many evals without improvement - larger patience makes trainign less noise sensitive.
    std::uint64_t patience   = 10;
    // Minimum full passes (epochs) before early-stopping may fire. Guards against the
    // failure seen on small corpora: the held-out NELBO plateaus within the patience
    // window after only a few passes (the model has barely left the unigram collapse),
    // so the run stops undertrained. No meaningful decision before the corpus has been
    // seen enough times. Rarely binds on a large corpus (it naturally trains far longer).
    std::uint64_t min_epochs = 20;
    // LR schedule. A fixed LR makes small-model training fragile: some configs jump into
    // the unigram basin at step 0 and never escape (seen in the D=192 A/B). Linear warmup
    // lets every config start gently, then cosine-decay. warmup_steps 0 = derive (~2% of
    // the step bound, capped). seed controls model init (was hardcoded 42).
    float         lr           = 1e-3f;   // peak LR after warmup
    bool          lr_explicit  = false;   // did the user pass --lr? (else optimizer picks a default)
    std::uint64_t warmup_steps = 0;       // 0 = derive
    std::uint64_t seed         = 42;
    // Optimizer: adamw (DEFAULT — A/B winner, +1.4pt word-level over plain adam: decoupled
    // weight decay improves conditioning on the high-variance diffusion gradient) | adam |
    // muon (orthogonalised-momentum on 2D weights, AdamW on the rest; LOSES here — it amplifies
    // the gradient noise; wants a larger LR ~0.02, auto-selected). See sub0llm/nn/optimizer.hpp.
    std::string   optimizer    = "adamw";
    float t_max            = 1.0f;    // <1.0 trains under a Ch28-style ceiling
    // Frontier-point curriculum (Ch28's winner): start easy, raise the noise ceiling
    // only once the current frontier is mastered (token-gated). Fixes the uniform-t
    // formal objective's underfitting of the easy regime on small models.
    bool curriculum        = false;
    float curriculum_end   = 1.0f;    // target ceiling (1.0 = eventually the full objective)
    // Masked-token evidence required per ceiling decision — the ramp-rate knob. Higher
    // = each level is mastered more thoroughly before the ceiling rises (slower, safer).
    std::uint64_t curriculum_min_tokens = 1500;
    bool eval_only         = false;
    // WHOLE-WORD masking (axis H): corrupt whole words (all BPE subwords of a chosen word
    // together) instead of independent tokens. The honest, harder task — the model can no
    // longer "complete" a partially-visible word (Ham→let). Affects BOTH training and eval.
    bool whole_word        = false;
    // --inspect N: after eval, dump the N best- and N worst-recovered held-out windows with
    // per-masked-position truth→prediction detail (what the headline recall actually means).
    std::int64_t inspect   = 0;
    // --grad-probe: at the CURRENT weights, measure whether low-t (easy) and high-t (hard)
    // maskings pull the gradient in CONFLICTING directions (cosine of the mean gradient per
    // noise level) + within-level consistency. The diagnostic for "is the diffusion gradient a
    // high-variance average of conflicting signals?" — the hypothesis for the sticky unigram
    // basin and why each step is ineffective. Standalone: runs the probe and exits.
    bool grad_probe        = false;
    // --shared-t: data-parallel VARIANCE REDUCTION. Default samples an independent t per
    // worker (4 windows × 4 different t), which dilutes the step across the near-orthogonal
    // noise levels (grad-probe: easy↔hard cosine ≈0.1, within-level consistency only ≈0.3).
    // --shared-t samples ONE t per step so all workers mask at the SAME level → the 4-window
    // average sharpens the per-t gradient. Still unbiased (t ~ U over steps). Pool path only.
    // DEFAULT ON: exact+shared was the variance-reduction 2×2 winner (+1.0pt word-level).
    // Pass --no-shared-t to restore the independent-per-worker baseline.
    bool shared_t          = true;
    // --exact-noise: mask EXACTLY round(t·T) positions per window (shuffle-select) instead of
    // an independent Bernoulli per position. Realised noise = nominal exactly (exact model
    // conditioning), removes the Bernoulli COUNT variance from the gradient, and drops the
    // min-1 floor hack. A deterministic, even, unbiased corruption count (only WHICH stays
    // random). Affects training + the held-out NELBO; the recall sweep stays Bernoulli for a
    // stable cross-experiment metric.
    bool exact_noise       = true;
    // --track-recall: at every held-out NELBO eval, ALSO measure a quick held-out recall and
    // print it. Decouples the early-stop signal (NELBO) from the metric we care about (recall)
    // — the test for "are we stopping too early / is NELBO-plateau actually recall-plateau".
    bool track_recall      = false;
    // --per-t: after eval, print the per-noise-level masked-CE (raw per-token NLL, nats) + recall
    // curve over a fine t grid, plus the unigram-entropy floor. The #1 diagnostic for "is the model
    // t-AWARE?": a flat curve ≈ the t-averaged NELBO ⇒ t-agnostic (conditioning broken); a curve
    // rising from <1 nat at low t up to the entropy floor ⇒ healthy / near objective floor.
    bool per_t             = false;
    // Also sweep recall on the TRAIN stream — measures MEMORIZATION (fit) vs the held-out
    // GENERALIZATION. A large train≫held-out gap means the ceiling is generalization, not
    // coverage/undertraining (and is the apples-to-apples comparison with Ch28, which only
    // ever scored on its training corpus).
    bool eval_train        = false;
    // --overfit N: the MINIMAL-CASE capacity diagnostic. Restrict TRAINING to the first N
    // NON-OVERLAPPING windows (the first N·seq_len tokens of the train stream) and stop on
    // TRAIN recall over those EXACT windows, not held-out NELBO. A model with adequate
    // capacity MUST drive train recall to ~100%: N=1 is degenerate (memorize one constant
    // output → 100% at every noise level), N=4/16/64 force progressively more real modeling.
    // If even N=1 won't reach ~100%, the bug is STRUCTURAL (capacity / optimizer / loss /
    // conditioning), not data/coverage — investigate before any more corpus-scale runs. It
    // also answers "do we need 8 layers / 3 heads?" empirically: find the SMALLEST arch
    // (--n-layers/--embed-dim/--n-heads) that still fits each N. Uses the IDENTICAL training
    // path (ParallelTrainer + diffusion_loss) — that's the point: stress the real loop on a
    // case whose right answer we know. Default eval-every 200 / steps 20000 unless overridden.
    std::int64_t overfit   = 0;
    bool profile           = false;   // report forward/backward/optimizer time split
    // --threads W = WORKERS: how many parallel threads split each step's work. The SPEED knob
    // ONLY — the gradient (and thus the trained model) is identical for any W. 0 = auto (P/2),
    // 1 = serial. Each worker is a model replica sharing the master weights (see parallel.hpp).
    std::size_t threads    = 0;
    // --batch-size B = effective BATCH: windows averaged per optimizer step. The QUALITY knob —
    // with --shared-t (default) all B mask at one noise level, so the gradient's within-level
    // consistency rises ∝ √B (the Ch31-sandbox lever). INDEPENDENT of --threads: the B windows
    // are split B/W across the W workers, so the SAME √B gradient runs on 1, 4, or 16 cores —
    // only wall-time changes. Default B = W (the historical per-worker pool); rounded up to a
    // multiple of W. Rule of thumb: pick B for quality, W for the speed you want.
    std::size_t batch      = 1;
    // Worker pin policy (shared cpu_topology::resolve_pin_set): auto | all | P | E | "lo-hi"
    // | "a,b,c". Lets concurrent training jobs take DISJOINT cores (e.g. one run --pin
    // 0,1,10,11 and another --pin 12,13,22,23) instead of all oversubscribing the same
    // P-cores — the dominant cause of multi-job slowdown.
    std::string pin        = "auto";
    // SIZE_MAX = corpus-scaled (§2c); explicit N = fixed budget; 0 = exhaustive sweep.
    std::size_t recall_windows = std::numeric_limits<std::size_t>::max();

    // ── Architecture — proportions FOUNDED on working tiny-Shakespeare references ──
    // Two reference points bracket this regime (head_dim ≈ n_embd/n_heads):
    //   paulseet/tiny_llm  : D=96,  L=3, H=3 → head_dim 32, d_ff=4·D=384, block 64, 354K params
    //   nanoGPT char (Karpathy): D=384, L=6, H=6 → head_dim 64, d_ff=4·D=1536, block 256, 10.6M
    // Both keep head_dim ≥ 32 (64 is "typical") and d_ff = 4·D. The 8-layer completeb
    // run (D=128, L=8, H=8 → head_dim=16, d_ff=3·D) was MIS-PROPORTIONED: too deep+narrow,
    // heads too small, FFN too thin — and it underfit (33%/27%/17%/8% recall, plateaued).
    // Founded fix = wider, fewer layers, head_dim≥32, d_ff=4·D (see --founded preset and
    // the sanity warnings in run()). All overridable via CLI so experiments need no rebuild.
    std::int64_t embed_dim = 128, n_layers = 8, d_ff = 384;
    std::size_t  n_heads = 8, n_kv_heads = 4;   // GQA: 2 query heads per KV head
    bool         founded   = false;   // --founded: apply the founded reference proportions
};

// Founded reference config (nanoGPT-proportioned, CPU-feasible): wider D, head_dim 32,
// d_ff = 4·D, balanced depth. Applied by --founded unless the user set arch flags.
static void apply_founded(Config& c) {
    c.embed_dim  = 256;   // between paulseet(96) and nanoGPT(384); 2× the underfit run
    c.n_layers   = 6;     // nanoGPT's depth — 8 was too deep for D=128
    c.n_heads    = 8;     // head_dim = 256/8 = 32 (founded floor; was 16)
    c.n_kv_heads = 4;     // keep the 2:1 GQA ratio
    c.d_ff       = 1024;  // 4·D (was 3·D)
}

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
        else if (a == "--min-epochs") c.min_epochs = std::stoull(next());
        else if (a == "--lr")       { c.lr = std::stof(next()); c.lr_explicit = true; }
        else if (a == "--optimizer")  c.optimizer  = next();
        else if (a == "--warmup-steps") c.warmup_steps = std::stoull(next());
        else if (a == "--seed")       c.seed       = std::stoull(next());
        else if (a == "--eval-factor") c.eval_factor = std::stof(next());
        else if (a == "--t-max")      c.t_max      = std::stof(next());
        else if (a == "--curriculum") c.curriculum = true;
        else if (a == "--curriculum-end") c.curriculum_end = std::stof(next());
        else if (a == "--curriculum-min-tokens") c.curriculum_min_tokens = std::stoull(next());
        else if (a == "--founded")  { c.founded = true; apply_founded(c); }  // later flags still override
        // Architecture overrides (so capacity experiments need no recompile).
        else if (a == "--embed-dim")  c.embed_dim  = std::stoll(next());
        else if (a == "--n-layers")   c.n_layers   = std::stoll(next());
        else if (a == "--d-ff")       c.d_ff       = std::stoll(next());
        else if (a == "--n-heads")    c.n_heads    = std::stoull(next());
        else if (a == "--n-kv-heads") c.n_kv_heads = std::stoull(next());
        else if (a == "--eval-only")  c.eval_only  = true;
        else if (a == "--whole-word") c.whole_word = true;
        else if (a == "--inspect")    c.inspect    = std::stoll(next());
        else if (a == "--track-recall") c.track_recall = true;
        else if (a == "--per-t")        c.per_t        = true;
        else if (a == "--grad-probe")   c.grad_probe   = true;
        else if (a == "--shared-t")     c.shared_t     = true;
        else if (a == "--no-shared-t")  c.shared_t     = false;   // restore independent-per-worker t
        else if (a == "--exact-noise")  c.exact_noise  = true;
        else if (a == "--no-exact-noise") c.exact_noise = false;  // restore Bernoulli-per-position
        else if (a == "--eval-train") c.eval_train = true;
        else if (a == "--overfit")    c.overfit    = std::stoll(next());
        else if (a == "--profile")    c.profile    = true;
        else if (a == "--threads" || a == "--workers") c.threads = std::stoull(next());
        else if (a == "--batch-size" || a == "--batch") c.batch  = std::stoull(next());
        else if (a == "--pin")        c.pin        = next();
        else if (a == "--recall-windows") c.recall_windows = std::stoull(next());
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

// ── Per-sample recovery inspection (--inspect N) ───────────────────────────────
// Dumps the N best- and N worst-recovered held-out windows at a fixed noise level with
// per-masked-position truth→prediction detail, so "15% recall" becomes legible: which
// words the model nails, which it misses, and whether misses are word-STARTS (real
// prediction) or continuations (partial-word completion).
static void inspect_recovery(const dn::Denoiser& model, const BPETokenizer& tok,
                             std::span<const std::int32_t> stream, std::int64_t T,
                             std::span<const std::uint8_t> ws, bool whole_word,
                             int n_show, float noise) {
    // Render a single token id legibly: the Ġ word-start marker (UTF-8 C4 A0) → '·'.
    auto disp = [&](std::int32_t id) {
        std::string s(tok.token_str(id));
        if (s.size() >= 2 && static_cast<unsigned char>(s[0]) == 0xC4 &&
            static_cast<unsigned char>(s[1]) == 0xA0)
            return "·" + s.substr(2);
        return s;
    };
    struct Miss { std::int32_t pos, truth, pred; bool hit, word_start; };
    struct Sample { std::size_t off; float recall; std::vector<Miss> masked; };

    const std::size_t n_positions = stream.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t budget = std::min<std::size_t>(400, n_positions);
    const double step = static_cast<double>(n_positions) / static_cast<double>(budget);
    std::mt19937 rng(20240614);
    dn::Corruption corr;
    std::vector<Sample> samples;
    samples.reserve(budget);

    for (std::size_t i = 0; i < budget; ++i) {
        const auto off = static_cast<std::size_t>(static_cast<double>(i) * step);
        auto window = stream.subspan(off, static_cast<std::size_t>(T));
        if (whole_word)
            dn::corrupt_whole_word_into(window, ws, noise, sub0diff::spec::NoiseSchedule::Absorbing,
                                        model.mask_id(), model.real_vocab(), rng, corr);
        else
            dn::corrupt_into(window, noise, sub0diff::spec::NoiseSchedule::Absorbing,
                             model.mask_id(), model.real_vocab(), rng, corr);
        // Forward on the corrupted window, argmax over real tokens at each masked position.
        Tensor input({T}, DType::Int32);
        std::ranges::copy(corr.tokens, input.data_as<std::int32_t>().begin());
        const float actual = static_cast<float>(corr.n_corrupted) / static_cast<float>(T);
        auto logits = model.forward(input, actual);
        const auto lz = logits.data().data_as<float>();
        const auto C  = static_cast<std::size_t>(model.model_vocab());
        const auto V  = model.real_vocab();
        Sample s{off, 0.0f, {}};
        int hits = 0;
        for (std::int64_t t = 0; t < T; ++t) {
            if (!corr.corrupted[static_cast<std::size_t>(t)]) continue;
            std::int32_t best = 0; float bv = -1e30f;
            for (std::int64_t c = 0; c < V; ++c) {
                const float v = lz[static_cast<std::size_t>(t) * C + static_cast<std::size_t>(c)];
                if (v > bv) { bv = v; best = static_cast<std::int32_t>(c); }
            }
            const auto truth = window[static_cast<std::size_t>(t)];
            const bool hit = (best == truth);
            hits += hit;
            const bool wstart = (t == 0) || ws[static_cast<std::size_t>(truth)] != 0;
            s.masked.push_back({static_cast<std::int32_t>(t), truth, best, hit, wstart});
        }
        s.recall = s.masked.empty() ? 0.0f
                                    : static_cast<float>(hits) / static_cast<float>(s.masked.size());
        samples.push_back(std::move(s));
    }

    std::ranges::sort(samples, [](const Sample& a, const Sample& b) { return a.recall > b.recall; });

    auto dump = [&](const Sample& s) {
        auto window = stream.subspan(s.off, static_cast<std::size_t>(T));
        std::vector<BPETokenizer::TokenId> ids(window.begin(), window.end());
        std::string text = tok.decode(ids);
        if (text.size() > 200) text = text.substr(0, 200) + "…";
        std::println("  ── window @ {}  recall {:.0f}% ({} masked) ──",
                     s.off, s.recall * 100.0f, s.masked.size());
        std::println("    text: {}", text);
        for (const auto& m : s.masked) {
            std::println("      [{:>2}] {:<4} {:>10} → {:<10} {}",
                         m.pos, m.word_start ? "WORD" : "cont",
                         "'" + disp(m.truth) + "'", "'" + disp(m.pred) + "'",
                         m.hit ? "✓" : "✗");
        }
    };

    std::println("\n── --inspect: per-sample recovery at noise {:.0f}%{} ──",
                 noise * 100.0f, whole_word ? " (whole-word)" : "");
    const int n = std::min(n_show, static_cast<int>(samples.size()));
    std::println("Best {} recovered windows:", n);
    for (int i = 0; i < n; ++i) dump(samples[static_cast<std::size_t>(i)]);
    std::println("\nWorst {} recovered windows:", n);
    for (int i = 0; i < n; ++i)
        dump(samples[samples.size() - 1 - static_cast<std::size_t>(i)]);
}

// ── Token-stream cache ─────────────────────────────────────────────────────────
// BPE-encoding the full corpus (146K paragraphs) is the dominant STARTUP cost and is
// DETERMINISTIC given a fixed (corpus, paragraph-limit, tokenizer). Cache the post-split
// flat token streams to `ckpt_dir/tokens.bin` so resumes and --eval-only start near-
// instantly. The header fingerprints (corpus byte-size, paragraph limit, vocab size); any
// mismatch re-encodes and rewrites. (The vocab sweep is safe: each V has its own ckpt_dir.)
namespace tokcache {
constexpr std::uint32_t kMagic = 0x53304454u;  // "S0DT"
constexpr std::uint32_t kVersion = 1u;
struct Header {
    std::uint32_t magic = kMagic, version = kVersion;
    std::uint64_t corpus_size = 0;
    std::int64_t  paragraphs = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t n_train = 0, n_eval = 0;
};

[[nodiscard]] static bool load(const fs::path& path, std::uint64_t corpus_size,
                               std::int64_t paragraphs, std::uint64_t vocab_size,
                               std::vector<std::int32_t>& train_ids,
                               std::vector<std::int32_t>& eval_ids) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    Header h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || h.magic != kMagic || h.version != kVersion || h.corpus_size != corpus_size ||
        h.paragraphs != paragraphs || h.vocab_size != vocab_size)
        return false;
    train_ids.resize(h.n_train);
    eval_ids.resize(h.n_eval);
    if (h.n_train) f.read(reinterpret_cast<char*>(train_ids.data()),
                          static_cast<std::streamsize>(h.n_train * sizeof(std::int32_t)));
    if (h.n_eval)  f.read(reinterpret_cast<char*>(eval_ids.data()),
                          static_cast<std::streamsize>(h.n_eval * sizeof(std::int32_t)));
    return static_cast<bool>(f);
}

static void save(const fs::path& path, std::uint64_t corpus_size, std::int64_t paragraphs,
                 std::uint64_t vocab_size, const std::vector<std::int32_t>& train_ids,
                 const std::vector<std::int32_t>& eval_ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    Header h{kMagic, kVersion, corpus_size, paragraphs, vocab_size,
             train_ids.size(), eval_ids.size()};
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    if (!train_ids.empty()) f.write(reinterpret_cast<const char*>(train_ids.data()),
                                    static_cast<std::streamsize>(train_ids.size() * sizeof(std::int32_t)));
    if (!eval_ids.empty())  f.write(reinterpret_cast<const char*>(eval_ids.data()),
                                    static_cast<std::streamsize>(eval_ids.size() * sizeof(std::int32_t)));
}
} // namespace tokcache

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
    const std::size_t n_eval = paragraphs.size() * 0.05;   // 5% held out for eval (early stopping, final metrics)
    std::vector<std::string> eval_texts(paragraphs.end() - static_cast<std::ptrdiff_t>(n_eval),
                                        paragraphs.end());
    paragraphs.resize(paragraphs.size() - n_eval);

    std::println("{} train / {} eval paragraphs from {}", paragraphs.size(), n_eval, cfg.corpus);
    // BPE caching: training BPE on the full corpus costs ~110s and is DETERMINISTIC,
    // so on resume/eval reload the tokenizer saved in the model dir instead of
    // retraining. This makes --eval-only and resumed training start near-instantly.
    const fs::path tok_dir    = fs::path(cfg.ckpt_dir) / "tokenizer";
    const fs::path vocab_json = tok_dir / "vocab.json";
    const fs::path merges_txt = tok_dir / "merges.txt";
    auto tok = [&] {
        if (fs::exists(vocab_json) && fs::exists(merges_txt)) {
            std::println("loaded cached tokenizer from {} (skipped BPE training)", tok_dir.string());
            return BPETokenizer::load(vocab_json, merges_txt);
        }
        std::print("training BPE (vocab {})... ", cfg.vocab_size);
        auto bpe_start = std::chrono::steady_clock::now();
        auto t = BPETokenizer::train(paragraphs, cfg.vocab_size);
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
    const fs::path tokens_cache = fs::path(cfg.ckpt_dir) / "tokens.bin";
    const std::uint64_t corpus_size =
        fs::exists(cfg.corpus) ? static_cast<std::uint64_t>(fs::file_size(cfg.corpus)) : 0;
    if (tokcache::load(tokens_cache, corpus_size, cfg.paragraphs, tok.vocab_size(),
                       train_ids, eval_ids)) {
        std::println("loaded cached token streams from {} (skipped re-encoding {} paragraphs)",
                     tokens_cache.string(), paragraphs.size() + eval_texts.size());
    } else {
        const auto enc_t0 = std::chrono::steady_clock::now();
        train_ids = flatten(paragraphs);
        eval_ids  = flatten(eval_texts);
        tokcache::save(tokens_cache, corpus_size, cfg.paragraphs, tok.vocab_size(),
                       train_ids, eval_ids);
        std::println("encoded + cached token streams to {} ({:.1f}s)", tokens_cache.string(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - enc_t0).count());
    }
    const auto win_count = [&](const std::vector<std::int32_t>& s) {
        return s.size() - static_cast<std::size_t>(cfg.seq_len) + 1;
    };
    std::println("train stream: {} tokens ({} sliding positions for {}-token windows)",
                 train_ids.size(), win_count(train_ids), cfg.seq_len);
    std::println("eval stream:  {} tokens ({} sliding positions for {}-token windows)",
                 eval_ids.size(), win_count(eval_ids), cfg.seq_len);

    // ── 2b/2c. corpus-scaled schedule — ONE tested place (sub0diff/train/schedule.hpp) ──
    // Derives eval cadence, the safety step bound, and eval sample sizes. THE COVERAGE
    // RULE (Ch28): an eval is only a meaningful decision once the model has trained over
    // ≥50% of an epoch (non-overlapping windows) since the last one — below that, evals
    // are noise on partial data and patience trips on noise. The thread count is folded
    // in ONCE here (each step trains `threads` windows) — the prior split mis-reported it.
    const auto topo = sub0llm::detect_cpu_topology();
    if (cfg.threads == 0) {
        // Default = HALF the physical P-cores. Per-window data-parallel training is
        // MEMORY-BANDWIDTH bound for this small model (one P-core already pulls ~⅓ of
        // the DRAM bus; bench_membw: read scales only 1t→4t ≈2.0×, 1t→8t ≈2.4×). So
        // ~4 workers hit the bus wall — 8 is no faster and just burns cores. Measured
        // throughput plateaus flat across W=3..8 at ~2.5× serial. Override with
        // --threads. Shared topology model: sub0llm/core/cpu_topology.hpp.
        const int pc = topo.n_perf_cores();
        const int base = pc > 0 ? pc : topo.n_logical / 2;
        cfg.threads = std::max<std::size_t>(1, static_cast<std::size_t>((base + 1) / 2));
    }
    // batch-size B (windows averaged per step → gradient consistency ∝ √B) is INDEPENDENT of the
    // worker count W (--threads, parallelism). Default B = W (the historical per-worker pool).
    // --batch-size B>1 raises consistency on any W; round up to a multiple of W (workers split B/W).
    if (cfg.batch <= 1) cfg.batch = std::max<std::size_t>(1, cfg.threads);
    else cfg.batch = ((cfg.batch + cfg.threads - 1) / cfg.threads) * cfg.threads;
    const bool eval_every_user = cfg.eval_every != 0;   // did the user fix the cadence/bound?
    const bool steps_user      = cfg.steps != 0;
    const double mean_t = 0.5 * (0.02 + cfg.t_max);
    const dt::ScheduleConfig sc{.eval_factor = cfg.eval_factor};
    const auto sched = dt::make_schedule(
        train_ids.size(), eval_ids.size(), cfg.seq_len, cfg.threads,
        /*masked_per_window=*/mean_t * static_cast<double>(cfg.seq_len), sc,
        /*eval_every_override=*/cfg.eval_every, /*steps_override=*/cfg.steps,
        /*recall_override=*/cfg.recall_windows);
    cfg.eval_every                       = sched.eval_every;
    cfg.steps                            = sched.steps_bound;
    cfg.recall_windows                   = sched.recall_windows;
    const std::size_t eval_nelbo_windows = sched.eval_nelbo_windows;
    // Early-stopping floor: min_epochs full passes (in STEPS) before a stop may fire. A step trains
    // B = cfg.batch windows (over W workers), so divide epoch windows by B to get steps.
    std::uint64_t min_stop_steps =
        cfg.min_epochs * sched.epoch_windows / std::max<std::size_t>(1, cfg.batch);
    // --overfit overrides the corpus-scaled schedule: the train stream is unchanged but we only
    // visit N fixed windows, so the epoch/coverage logic doesn't apply. Watch train recall climb
    // (eval often), cap the run (a model that can't fit N in 20k steps has a structural bug), and
    // drop the min-epochs floor (stop the instant the N windows are fit).
    if (cfg.overfit > 0) {
        if (!eval_every_user) cfg.eval_every = 200;
        if (!steps_user)      cfg.steps      = 20000;
        min_stop_steps = 0;
    }

    if (cfg.eval_factor < sc.min_coverage)
        std::println("note: --eval-factor {:.2f} clamped up to the {:.0f}% epoch-coverage floor "
                     "(below it, evals are noise on partial data — Ch28).",
                     cfg.eval_factor, 100.0 * sc.min_coverage);
    std::println("schedule: eval every {} steps = {:.0f}% epoch coverage "
                 "({} epoch windows, {} sliding); steps bound {} (~{:.0f} masked-token epochs)",
                 cfg.eval_every, 100.0 * sched.coverage_per_eval, sched.epoch_windows,
                 sched.sliding_train, cfg.steps, sc.coverage_epochs);
    std::println("early-stop floor: no stop before {} epochs (= {} steps); patience {} evals after",
                 cfg.min_epochs, min_stop_steps, cfg.patience);
    std::println("eval samples: nelbo {} windows, recall {}/level; {} worker thread{}",
                 eval_nelbo_windows,
                 cfg.recall_windows == 0 ? std::string("exhaustive")
                                         : std::format("{}", cfg.recall_windows),
                 cfg.threads, cfg.threads == 1 ? "" : "s");
    const auto worker_pins = sub0llm::resolve_pin_set(cfg.pin, topo, cfg.threads);
    if (cfg.threads > 1) {
        std::string pinstr;
        for (std::size_t i = 0; i < worker_pins.size(); ++i)
            pinstr += (i ? "," : "") +
                      (worker_pins[i] < 0 ? std::string("-") : std::format("{}", worker_pins[i]));
        std::println("cpu topology: {} logical, {} physical P-cores ({} P-logical, {} E-logical); "
                     "pin policy '{}' → worker pins [{}]",
                     topo.n_logical, topo.n_perf_cores(), topo.perf.size(),
                     topo.efficiency.size(), cfg.pin, pinstr);
    }

    // ── 3. model + checkpoint resume ────────────────────────────────────────────
    section("3. Model — bidirectional denoiser over the BPE vocabulary");
    const auto V = static_cast<std::int64_t>(tok.vocab_size());
    dn::Denoiser model(V, cfg.embed_dim, cfg.n_heads, cfg.n_kv_heads,
                       cfg.n_layers, cfg.d_ff, static_cast<std::uint64_t>(cfg.seed));
    auto param_ptrs = model.parameters();
    std::vector<ag::Variable> params;                 // shared-impl handles for checkpointing
    params.reserve(param_ptrs.size());
    for (auto* p : param_ptrs) params.push_back(*p);
    std::int64_t n_params = 0;
    for (const auto& p : params) n_params += p.data().numel();
    std::println("Denoiser: V={} (+1 mask), D={}, layers={}, heads={}/{} (GQA), params={}",
                 V, cfg.embed_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, n_params);
    // Proportion notes vs the tiny-Shakespeare references (head_dim≈32-64, d_ff=4·D).
    // ADVISORY ONLY: those are char-level AUTOREGRESSIVE refs, and a Ch29 A/B REFUTED
    // them for this BPE bidirectional-diffusion model — D=128 with 8 heads/8 layers
    // (head_dim 16) beat 4 heads/6 layers (head_dim 32) by 14.8% vs 3.9% recall. So
    // more heads + depth win here; the warnings are informational, not prescriptive.
    {
        const std::int64_t head_dim = cfg.embed_dim / static_cast<std::int64_t>(cfg.n_heads);
        if (head_dim < 32)
            std::println("  note: head_dim={} (< the char-AR ref's 32) — but A/B shows MORE heads "
                         "win here, so this is likely fine.", head_dim);
        if (cfg.d_ff < 4 * cfg.embed_dim)
            std::println("  note: d_ff={} is below the references' 4·D={} (advisory).",
                         cfg.d_ff, 4 * cfg.embed_dim);
    }

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

    // ── Gradient-conflict probe (--grad-probe) — a Ch31-sandbox diagnostic ───────
    // At the CURRENT weights, does the gradient from EASY (low-t) maskings point the same way
    // as from HARD (high-t) maskings? Diffusion averages all noise levels in one step (unlike
    // AR's single-token signal); if easy/hard CONFLICT, the averaged step cancels useful signal
    // — a candidate root cause of the sticky unigram basin. Ch28's curriculum is the temporal,
    // BIASED answer to the same heterogeneity; this measures whether the conflict is even real.
    if (cfg.grad_probe) {
        section("Gradient-conflict probe — do low-t (easy) and high-t (hard) maskings agree?");
        const std::size_t total = static_cast<std::size_t>(n_params);
        dt::DiffusionLossContext gctx(cfg.seq_len);
        std::mt19937 grng(2024);
        std::uniform_int_distribution<std::size_t> goff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.seq_len) - 1);
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
                             .subspan(goff(grng), static_cast<std::size_t>(cfg.seq_len));
                dt::diffusion_loss(model, w, grng, gctx, ts[k], ts[k], ws_span,
                                   cfg.whole_word, cfg.exact_noise).loss.backward();
                add_grads(g[k], 1.0 / static_cast<double>(N));
            }
        // Within-level consistency at t=mid: mean cos(single-sample grad, mean grad).
        double consist = 0.0; std::vector<double> sv(total, 0.0);
        for (int s = 0; s < N; ++s) {
            for (auto* p : param_ptrs) p->zero_grad();
            auto w = std::span<const std::int32_t>(eval_ids)
                         .subspan(goff(grng), static_cast<std::size_t>(cfg.seq_len));
            dt::diffusion_loss(model, w, grng, gctx, ts[mid], ts[mid], ws_span,
                               cfg.whole_word, cfg.exact_noise).loss.backward();
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
    dt::DiffusionLossContext eval_ctx(cfg.seq_len);
    auto eval_nelbo = [&](std::size_t n_windows) {
        std::mt19937 erng(777);
        std::uniform_int_distribution<std::size_t> eoff(
            0, eval_ids.size() - static_cast<std::size_t>(cfg.seq_len) - 1);
        double sum = 0.0;
        for (std::size_t i = 0; i < n_windows; ++i) {
            auto w = std::span<const std::int32_t>(eval_ids)
                         .subspan(eoff(erng), static_cast<std::size_t>(cfg.seq_len));
            sum += dt::diffusion_loss(model, w, erng, eval_ctx, 0.02f, 1.0f,
                                      ws_span, cfg.whole_word, cfg.exact_noise)
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
            auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.seq_len, n, rr,
                                                nullptr, budget, ws_span, cfg.whole_word);
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
    if (cfg.overfit > 0) {
        const auto N = static_cast<std::size_t>(cfg.overfit);
        const auto T = static_cast<std::size_t>(cfg.seq_len);
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
            auto w = std::span<const std::int32_t>(train_ids).subspan(off, static_cast<std::size_t>(cfg.seq_len));
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
    if (!cfg.eval_only && start_step < cfg.steps) {
        section("4. Training — self-terminating on held-out NELBO (early stopping)");
        // Muon wants a larger LR than Adam — default to 0.02 if the user didn't pass --lr.
        std::unique_ptr<nn::Optimizer> opt;
        if (cfg.optimizer == "muon") {
            if (!cfg.lr_explicit) cfg.lr = 0.02f;
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
            opt = std::make_unique<nn::Muon>(param_ptrs, cfg.lr, 0.95f, 5, true,
                                             adamw_grp_lr, 0.9f, 0.999f, 1e-8f, 0.0f,
                                             std::move(force_adamw));
            std::println("optimizer: muon (lr {:.1e}; AdamW-group lr {:.1e}; {} vocab-dim "
                         "params → AdamW)", cfg.lr, adamw_grp_lr, n_emb);
        } else {
            opt = nn::make_optimizer(cfg.optimizer, param_ptrs, cfg.lr);
            std::println("optimizer: {} (lr {:.1e})", cfg.optimizer, cfg.lr);
        }
        // LR schedule is OPT-IN (--warmup-steps N): a measured A/B showed warmup+cosine
        // HURT the known-good baseline (12.0% vs 14.8%) by decaying it into an earlier
        // early-stop, and did NOT rescue wider configs that get stuck in the unigram
        // basin (that is an optimization/init problem, not an LR-schedule one). Default
        // = constant cfg.lr, the regime the baseline trains well at.
        std::optional<nn::CosineWithWarmup> lr_sched;
        if (cfg.warmup_steps > 0) {
            lr_sched.emplace(cfg.lr, cfg.lr * 0.1f, static_cast<std::int64_t>(cfg.warmup_steps),
                             static_cast<std::int64_t>(cfg.steps));
            std::println("lr schedule: warmup {} steps to {:.1e}, cosine decay to {:.1e}",
                         cfg.warmup_steps, cfg.lr, cfg.lr * 0.1f);
        } else {
            std::println("lr: constant {:.1e} (no warmup; --warmup-steps N to enable)", cfg.lr);
        }
        std::mt19937 rng(1234 + static_cast<std::uint32_t>(start_step));
        // Effective batch B windows/step (gradient consistency ∝ √B), split across W=threads
        // workers (B/W each). The master gradient is the mean over all B windows, identical for
        // any W — so single-thread (W=1) trains the SAME approach, just slower.
        const std::size_t windows_per_step = cfg.batch;

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

        // Unified trainer: W=cfg.threads workers cooperatively process B=cfg.batch windows/step
        // (B/W each), master sees the MEAN gradient over all B. ALWAYS used (W=1 is a pool of one),
        // so single- and multi-threaded training run the identical fundamental step.
        auto pool = std::make_unique<dt::ParallelTrainer>(
            cfg.threads,
            dt::ParallelTrainer::Arch{V, cfg.embed_dim, cfg.n_layers, cfg.d_ff,
                                      cfg.seq_len, cfg.n_heads, cfg.n_kv_heads},
            param_ptrs, 0.02f, cfg.t_max, 1234 + start_step,
            /*share_weights=*/true, ws_span, cfg.whole_word, worker_pins, cfg.exact_noise,
            static_cast<std::int64_t>(cfg.batch), cfg.shared_t);
        std::println("trainer: batch-size {} windows/step (consistency ∝ √{}) split across {} "
                     "worker{} (speed only — same gradient at any W); shared-t {}, exact-noise {}",
                     cfg.batch, cfg.batch, cfg.threads, cfg.threads == 1 ? "" : "s",
                     cfg.shared_t ? "on" : "off", cfg.exact_noise ? "on" : "off");
        std::vector<std::size_t> offsets(cfg.batch);
        std::size_t overfit_cursor = 0;   // cycles the N fixed windows across the B step slots
        if (cfg.overfit > 0)
            std::println("OVERFIT mode: training only the first {} non-overlapping window{} "
                         "(stop on TRAIN recall ≥ 99%; cap {} steps)",
                         cfg.overfit, cfg.overfit == 1 ? "" : "s", cfg.steps);

        // Frontier-point curriculum (opt-in): train AT a noise ceiling that rises only
        // once mastered. The floor stays at 0.02; frontier-point trains exactly at the
        // ceiling. Disabled → uniform t ∈ (0.02, t_max] (the formal objective).
        std::unique_ptr<dt::NoiseCurriculum> curr;
        if (cfg.curriculum) {
            curr = std::make_unique<dt::NoiseCurriculum>(
                dt::NoiseCurriculum::Config{.end = cfg.curriculum_end,
                                            .min_tokens = cfg.curriculum_min_tokens});
            std::println("curriculum: frontier-point, ceiling {:.2f}→{:.2f}, token-gated "
                         "(min_tokens={})",
                         0.05f, cfg.curriculum_end, cfg.curriculum_min_tokens);
        }

        float last_loss = 0.0f, last_t = 0.0f;
        for (std::uint64_t step = start_step; step < cfg.steps; ++step) {
            auto pt0 = tick();
            std::uint64_t step_masked = 0;
            auto pt1 = pt0, pt2 = pt0;
            // Frontier-point DURING the ramp (sample t exactly at the ceiling — fast
            // per-bucket mastery, Ch28's winner). Once the ceiling CONVERGES at the top,
            // switch to the full band [0.02, ceiling] = the formal objective: training a
            // single t=1.0 (fully masked, no context) is degenerate and degrades the easy
            // regime, which made the uniform-noise eval climb and early-stop prematurely.
            const float t_lo = (curr && !curr->converged()) ? curr->frontier() : 0.02f;
            const float t_hi = curr ? curr->frontier() : cfg.t_max;
            // Set the noise band the trainer samples from (curriculum ceiling, or the formal band).
            // shared-t sampling (one t for all B windows) happens inside pool->step() from this band.
            pool->set_t_range(t_lo, t_hi);
            // OVERFIT: cycle the N fixed windows across the B slots (full-batch GD over them).
            // Else: a fresh random sliding offset per slot (the held-out eval distribution).
            if (cfg.overfit > 0)
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
            if (cfg.profile) {
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
                std::println("{:>5.0f}s  step {:>6}  nelbo={:.4f} (t={:.2f}){}  "
                             "[{:.0f} windows/s, {:.0f} masked-tok/s]",
                             el, step, last_loss, last_t,
                             curr ? std::format("  ceiling={:.2f}{}", curr->frontier(),
                                                curr->converged() ? " (converged)" : "")
                                  : std::string(),
                             windows_s,
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

            // ── OVERFIT: stop on TRAIN recall over the N fixed windows (not held-out) ──
            if (cfg.overfit > 0 && (step + 1) % cfg.eval_every == 0) {
                const auto r = overfit_recall();
                std::println("  overfit @ {:>6}: TRAIN recall {:.1f}% (word {:.1f}%, START {:.1f}%) "
                             "over {} window{} | train-nelbo {:.4f}",
                             step + 1, r.recall() * 100.0f, r.word_recall() * 100.0f,
                             r.ws_recall() * 100.0f, cfg.overfit, cfg.overfit == 1 ? "" : "s",
                             last_loss);
                if (r.recall() >= 0.99f) {
                    save_checkpoint(params, cfg.ckpt_dir, static_cast<std::int64_t>(step + 1));
                    std::println("  overfit FIT ✓ — train recall {:.1f}% ≥ 99% at step {}: capacity "
                                 "SUFFICIENT for N={} (D={}, L={}, H={})",
                                 r.recall() * 100.0f, step + 1, cfg.overfit,
                                 cfg.embed_dim, cfg.n_layers, cfg.n_heads);
                    steps_taken = step + 1;
                    break;
                }
                continue;   // overfit never runs the held-out/patience path below
            }

            // ── Convergence check: held-out NELBO every eval_every steps ───────────
            if ((step + 1) % cfg.eval_every == 0) {
                const float nelbo = static_cast<float>(eval_nelbo(eval_nelbo_windows));
                if (cfg.track_recall) {
                    const auto qr = quick_recall(800);
                    std::println("  track @ {:>6}: held-out recall {:.1f}% (word {:.1f}%, "
                                 "START {:.1f}%) | NELBO {:.4f}",
                                 step + 1, qr.recall() * 100.0f, qr.word_recall() * 100.0f,
                                 qr.ws_recall() * 100.0f, nelbo);
                }
                if (nelbo < best_nelbo - min_improvement) {
                    best_nelbo = nelbo;
                    evals_since_best = 0;
                    save_checkpoint(params, cfg.ckpt_dir, static_cast<std::int64_t>(step + 1));
                    std::println("  eval @ {:>6}: held-out NELBO {:.4f}  ← best, checkpointed",
                                 step + 1, nelbo);
                } else {
                    ++evals_since_best;
                    // Don't stop before min_epochs full passes — a NELBO plateau this
                    // early is undertraining (often a unigram collapse), not convergence.
                    const bool floor_reached = steps_taken >= min_stop_steps;
                    std::println("  eval @ {:>6}: held-out NELBO {:.4f}  (best {:.4f}, {}/{} patience{})",
                                 step + 1, nelbo, best_nelbo, evals_since_best, cfg.patience,
                                 floor_reached ? "" : ", below min-epochs floor");
                    if (evals_since_best >= cfg.patience && floor_reached) {
                        std::println("  early stop: no improvement for {} evals (after {} min epochs)",
                                     cfg.patience, cfg.min_epochs);
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
        // SKIP in --overfit: we want the LIVE trained weights (a stale checkpoint from a
        // prior run in this dir must not clobber them), and the verdict is the train recall.
        if (cfg.overfit == 0)
            if (const auto best = latest_checkpoint_path(cfg.ckpt_dir); !best.empty())
                (void)load_checkpoint(params, best);
    } else if (cfg.eval_only) {
        section("4. Training — skipped (--eval-only)");
    } else {
        section("4. Training — skipped (checkpoint already at target steps)");
    }

    // ── 5'. OVERFIT verdict: per-noise TRAIN recall over the N fixed windows ────────
    // Held-out eval is meaningless here (we deliberately trained on N windows only), so
    // report the diagnostic — can the model reproduce exactly what it was shown? — and exit.
    if (cfg.overfit > 0) {
        section("5. Overfit verdict — TRAIN recall over the fixed windows");
        std::println("Model: D={}, layers={}, heads={}/{} (GQA), d_ff={}, params={} | N={} window{}",
                     cfg.embed_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff,
                     n_params, cfg.overfit, cfg.overfit == 1 ? "" : "s");
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
                                       "NOT the bottleneck at N={}.", cfg.overfit,
                                       cfg.overfit == 1 ? "" : "s", cfg.overfit)
                     : rec >= 0.80f
                         ? std::format("PARTIAL ({:.0f}%) — close; try more steps (--steps), a wider "
                                       "model, or a higher LR (--lr).", rec * 100.0f)
                         : std::format("FAIL ({:.0f}%) — this arch could not even MEMORIZE {} "
                                       "window{} in {} steps. Structural issue (capacity / optimizer "
                                       "/ loss / conditioning), not data — investigate here, cheaply, "
                                       "before any corpus-scale run.", rec * 100.0f, cfg.overfit,
                                       cfg.overfit == 1 ? "" : "s", cfg.steps));
        return 0;
    }

    // ── 5. held-out evaluation: NELBO + recall sweep + edge profile ─────────────
    section("5. Held-out evaluation");
    std::println("held-out NELBO ({} windows): {:.4f}\n",
                 4 * eval_nelbo_windows, eval_nelbo(4 * eval_nelbo_windows));

    // 5a. Per-t diagnostic (--per-t): raw per-token NLL + recall vs noise level. The headline
    // NELBO is this curve AVERAGED over t — a single scalar hides whether the model is t-aware.
    if (cfg.per_t) {
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
            0, eval_ids.size() - static_cast<std::size_t>(cfg.seq_len) - 1);
        constexpr std::size_t ce_windows = 512;
        const std::size_t rbudget = std::min<std::size_t>(2000, win_count(eval_ids));
        for (float t : {0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.40f,
                        0.50f, 0.60f, 0.70f, 0.80f, 0.90f, 0.95f}) {
            double ce = 0.0;
            for (std::size_t i = 0; i < ce_windows; ++i) {
                auto w = std::span<const std::int32_t>(eval_ids)
                             .subspan(poff(prng), static_cast<std::size_t>(cfg.seq_len));
                ce += dt::diffusion_loss(model, w, prng, eval_ctx, t, t,
                                         ws_span, /*whole_word=*/false, /*exact_count=*/true).mean_ce;
            }
            auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.seq_len, t, prng,
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
        cfg.recall_windows == 0 ? win_count(eval_ids)
                                : std::min<std::size_t>(cfg.recall_windows, win_count(eval_ids));
    std::println("Recall sweep over the held-out stream: {} of {} sliding positions per noise "
                 "level ({}-token windows, uniform stride){}:",
                 sweep_budget, win_count(eval_ids), cfg.seq_len,
                 cfg.whole_word ? " [WHOLE-WORD masking]" : "");
    std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
    std::mt19937 eval_rng(4242);
    de::PositionStats pos(static_cast<std::size_t>(cfg.seq_len));
    de::RecoveryResult sweep;
    const auto sweep_t0 = std::chrono::steady_clock::now();
    for (float noise : {0.10f, 0.25f, 0.50f, 0.75f}) {
        auto r = de::evaluate_corpus_recall(model, eval_ids, cfg.seq_len, noise, eval_rng,
                                            &pos, cfg.recall_windows, ws_span, cfg.whole_word);
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
    for (std::int64_t t = 1; t + 1 < cfg.seq_len; ++t) {
        interior += pos.recall_at(static_cast<std::size_t>(t)); ++n_int;
    }
    std::println("\nEdge effect: first {:.1f}%, last {:.1f}% vs interior mean {:.1f}%",
                 pos.recall_at(0) * 100.0f,
                 pos.recall_at(static_cast<std::size_t>(cfg.seq_len - 1)) * 100.0f,
                 interior / static_cast<float>(n_int) * 100.0f);

    // ── 5b. MEMORIZATION check: the SAME sweep on the TRAIN stream (--eval-train) ──
    // train≫held-out ⇒ the model fits/memorizes training fine and the limit is
    // GENERALIZATION (the Ch28-comparable number is THIS one). train≈held-out ⇒ the
    // model isn't even fitting training (undertraining / optimization / data shape).
    if (cfg.eval_train) {
        std::mt19937 train_rng(4242);
        de::RecoveryResult tsweep;
        std::println("\nTrain-stream recall (MEMORIZATION; held-out above is generalization):");
        std::println("  {:>6}  {:>8}  {:>10}  {:>7}", "noise", "masked", "recovered", "recall");
        for (float noise : {0.10f, 0.25f, 0.50f, 0.75f}) {
            auto r = de::evaluate_corpus_recall(model, train_ids, cfg.seq_len, noise, train_rng,
                                                nullptr, cfg.recall_windows, ws_span, cfg.whole_word);
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
    if (cfg.inspect > 0) {
        section("5c. Per-sample recovery inspection (--inspect)");
        const int n = static_cast<int>(cfg.inspect);
        inspect_recovery(model, tok, eval_ids, cfg.seq_len, ws_span, cfg.whole_word, n, 0.25f);
        inspect_recovery(model, tok, eval_ids, cfg.seq_len, ws_span, cfg.whole_word, n, 0.50f);
    }

    section("What's next");
    std::println("Ch30 — the reverse process: iterative canvas refinement with confidence remasking,");
    std::println("       loading this chapter's model dir ({}).", cfg.ckpt_dir);
    return 0;
}
