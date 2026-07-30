// train_stage.cpp — backend stage "train" (libsub0_train).
//
// Owns the training-time orchestration: corpus loading, a held-out validation
// split, minibatch sampling, the AdamW update schedule, automatic stopping and
// crash-safe checkpointing. The differentiable primitives (forward /
// cross_entropy / backward) and the optimizer come from the shared engine core.
//
// Schedule is corpus-relative, never a fixed step count:
//   * one "epoch" = train_tokens / (batch * SEQ_LEN) steps;
//   * validation NELBO (mean cross-entropy / token on the held-out tail) is not
//     measured until 50% of an epoch has been covered -- earlier numbers are
//     dominated by initialization noise and would trip a false plateau;
//   * after warmup we evaluate every ~10% of an epoch and stop on a plateau,
//     detected by a sign test over the last few evals (no magnitude threshold);
//   * a generous epoch-count backstop caps the run if no plateau is seen.
// A checkpoint with the full optimizer + loop state is written periodically so a
// crashed run resumes exactly.

#include "sub0/core.hpp"
#include "sub0/coherence.hpp"
#include "sub0/decode.hpp"   // shared KV-cache decode loop (GPU-first, CPU-fallback, EOS-stop) for
                             // preview_at/gen_self_stats -- also declares the init/shutdown/upload_params/
                             // set_tf32 seam GpuTrainer below reuses, so those aren't re-declared here
#include "sub0/config_util.hpp"  // sub0::config::kTokensPerParam — the SAME target ratio autosize() uses
#include "sub0/log.hpp"      // sub0::log — leveled diagnostics + the <model_dir>/train.log tee
#include "sub0/eval.hpp"     // held-out NELBO: window planning + the CPU/device dispatch (shared with tests)
#include "sub0/evalcache.hpp"
#include "sub0/registry.hpp"
#include "sub0/tokmap.hpp"
#include "sub0/tune.hpp"
#include "sub0/window.hpp"   // sample_window_start: keep each training window inside one document
#include "sub0/blend.hpp"    // BlendSource -- per-source data a corpus blend draws windows from
#include "sub0/blend_schedule.hpp" // ScheduleSpec/parse_blend_schedule_json + the staged epoch-fair scheduler
#include "sub0/spellspike.hpp"  // uncombine/combine curriculum generator (a "spellspike" schedule source)
#include "sub0/scratchspike.hpp" // scratch-token (context-translation) curriculum (a "scratchspike" schedule source)
#include "sub0/scratch_slots.hpp" // ScratchBindings / SlotEncoding -- content-derived scratch-slot embeddings
#include "sub0/op_curriculum.hpp" // op-delegation (math node routing) curriculum (an "op_curriculum" schedule source)
#include "sub0/wordspike.hpp"    // natural-prose word-collapse + op-delegation curriculum (a "wordspike" schedule source)
#include "sub0/corpus_collapse.hpp" // wordspike's mechanism over sampled REAL corpus docs (a "corpus_collapse" schedule source)
#include "sub0/bench.hpp"    // adaptive_time: budget-sized measurement shared with the GPU tuner
#include "sub0/memplan.hpp"  // train_resident_mb: predicted device footprint (guard + drift check)

// Code version + models root, baked in by CMake (configure-time) so a model records what
// produced it and lands in a structured directory. Fallbacks keep the file compilable alone.
#ifndef SUB0_GIT_SHA
#define SUB0_GIT_SHA "nogit"
#endif
#ifndef SUB0_MODELS_ROOT
#define SUB0_MODELS_ROOT "models"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <print>
#include <random>
#include <sstream>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(_WIN32)
#define NOMINMAX                 // keep std::min/std::max, not the windows.h macros
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// GPU training fast-path (Phase 2e). When a device backend is compiled in (SUB0_BUILD_CUDA today) and
// a device initializes, the training loop keeps the parameters resident on the GPU and runs each
// step (forward + backward + AdamW) there via sub0_dev_train_step, syncing back to the host
// param/optimizer arenas only at eval/checkpoint boundaries (where the CPU eval/save/preview code
// runs unchanged). The device path is gradient/AdamW parity-tested against this CPU backend.
//
// The canonical, backend-neutral seam declarations (docs/BACKENDS.md) -- also pulled in transitively
// via sub0/decode.hpp, included explicitly here too (IWYU: this file's own device calls shouldn't rely
// on a transitive include). Every device call in this stage goes through the neutral sub0_dev_* names
// (no local sub0_cuda_* extern declarations -- device_backend.hpp is the one place those live now);
// without a device backend linked, sub0_dev_* are fail-fast stubs, so the GpuTrainer methods below keep
// their own #if defined(SUB0_BUILD_CUDA) guards only where the CPU-only body would otherwise pointlessly
// touch never-used parameters ([[maybe_unused]] scaffolding), not because the neutral calls need it.
// [[nodiscard]] on every status-code return: a discarded return is exactly the class of bug that let
// the training loop keep "training" on a corrupted device context for thousands of steps after a real
// hardware fault -- see [[gpu-failure-detection-hardening]].
#include "sub0/device_backend.hpp"

namespace {

// Reference-cycle counter (rdtsc). The TSC is invariant on this class of CPU, so
// it ticks at a fixed rate independent of the core's current (thermally drifting)
// frequency. We therefore report the MINIMUM over many iterations: the fastest
// iteration is the one that ran un-throttled and un-preempted, which is the most
// reproducible measure of a code path's intrinsic cost.
inline std::uint64_t cpu_cycles() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ia32_rdtsc();
#else
    return __rdtsc();
#endif
}

// --- Schedule constants (all corpus-relative; see file header) --------------
constexpr double VAL_FRACTION        = 0.05;  // tail held out for validation NELBO
constexpr double EVAL_WARMUP_EPOCHS  = 0.50;  // no eval until this much coverage
constexpr double EVAL_INTERVAL_EPOCHS = 0.10; // eval/checkpoint cadence
constexpr double TICK_SECONDS        = 180.0;  // heartbeat: interim progress line between evals
constexpr double CKPT_SECONDS        = 3600.0; // crash-resistance checkpoint on a wall-clock tick,
                                               // independent of the (far rarer) eval cadence -- so an
                                               // unattended run never loses more than ~1h of progress
constexpr int    EVAL_WINDOWS_MAX    = sub0::eval::WINDOWS_MAX;   // bounded cost per eval (one source
                                              // of truth: tests plan the same window set from it)
constexpr int    MAX_EPOCHS_BACKSTOP = 30;    // ceiling if no plateau is detected
constexpr int    PLATEAU_WINDOW      = 6;     // evals fitted by the least-squares trend test
constexpr int    PLATEAU_PATIENCE    = 3;     // ...and a plateau ALSO requires no new best val-NELBO in
                                              // the last this-many evals (~0.3 epochs at the 0.10-epoch
                                              // eval cadence). Threshold-free, and the guard the trend
                                              // test structurally cannot provide: a run still producing
                                              // its best-ever model has not plateaued no matter how
                                              // shallow its fitted slope looks. Added after two arms
                                              // stopped "plateaued" while hitting a new best at EVERY
                                              // eval -- see coherence::improved_recently.
constexpr double PLATEAU_MIN_REL     = 0.005; // stop when the best-fit drop over the window < 0.5%
                                              // (~0.8%/epoch). 2% was too loose: it stopped on slow-
                                              // but-real tails (d96 @1.985 and d128 @2.68 both still
                                              // improving ~1.5%/window) well before the true floor.
// A hard floor before plateau-stop is even CONSIDERED, independent of any expected_plateau_epoch hint.
// Blending in any secondary schedule source at all dilutes the base corpus's own per-step gradient
// budget (fewer of each step's windows are base-corpus windows than an unblended run would draw),
// which can slow the base corpus's OWN val-NELBO improvement rate especially early in a run -- a real
// mechanism that could read as a premature plateau on a noisy early trend fit. This guard is bigger
// than EVAL_WARMUP_EPOCHS (which only gates whether eval RUNS at all, not whether a plateau-stop trusts
// its result) specifically to cover that early, blend-diluted stretch.
constexpr double PLATEAU_MIN_EPOCH   = 0.75;
// expected_plateau_epoch hint scaling (blend_schedule.hpp's ScheduleSpec::expected_plateau_epoch,
// optional): `trend_plateaued` declares victory when the fitted relative drop is BELOW min_rel, so a
// SMALLER min_rel is STRICTER (needs a flatter trend, more evidence) and a LARGER min_rel is LOOSER
// (confirms plateau sooner). The effective threshold is PLATEAU_MIN_REL_FAR (stricter) when the current
// epoch position is far from the hint, and PLATEAU_MIN_REL_AT_HINT (looser) once at or past it. Bounded,
// named constants -- not an open-ended "linear scale", which could otherwise blow up (immediate
// false-stop on any noisy read far past the hint) or shrink to zero (never stops far before it). See
// plateaued()'s own comment for the interpolation.
constexpr double PLATEAU_MIN_REL_FAR     = 0.002;  // >= 1 full epoch away from the hint: need a flatter trend
constexpr double PLATEAU_MIN_REL_AT_HINT = 0.02;   // at/past the hint: a looser trend already counts as done
// The DEFAULT expected_plateau_epoch when the blend schedule doesn't specify one (a schedule's own value
// always wins). Evidence, not a guess -- the full Muon main-corpus plateau ledger as of 2026-07-17:
// d128 2.00ep, d192 1.70, d256 2.00, d320 1.80, d448 1.90, d512 2.50, d768 2.30, ce256_prod_fixed
// (blend+content-embed) 2.80 -- every run confirmed in [1.7, 2.8], centered on ~2. Deliberately a HINT
// default, not a hard epoch cap: a hard stop at 2.0 would have cut d512/d768/ce256_prod_fixed short
// while val-NELBO was still improving (their best evals landed at 2.3-2.5+). The hint instead makes a
// genuinely flat trend CONFIRM FASTER near epoch 2 (d192's 1.7ep plateau would have confirmed earlier,
// saving tail compute) while staying strict far from it (a slow learner at 3+ epochs keeps training).
constexpr double DEFAULT_EXPECTED_PLATEAU_EPOCH = 2.0;
constexpr double LR_WARMUP_EPOCHS    = 0.25;  // linear LR warmup, then inverse-sqrt decay (lr_schedule)
// Muon's own reference peak lr (github.com/KellerJordan/Muon) -- unrelated in scale to AdamW's
// batch-derived peak_lr below (Muon's orthogonalized updates have a very different magnitude/
// geometry from AdamW's per-element-normalized ones), shares the SAME warmup/decay shape via
// lr_schedule() but at this separate peak. Not batch-scaled: Muon's own convention doesn't use
// Adam's sqrt(batch) heuristic.
constexpr float  MUON_LR_BASE        = 0.02f;


// Variable-length training: each step draws a window width T in [MIN_TRAIN_SEQ, SEQ_LEN], shared
// across the batch (so the GPU keeps a single M = batch*T GEMM). Exposing a range of context
// lengths stops the model overfitting to exactly SEQ_LEN and makes it robust to short prompts. The
// floor is kept low (SEQ_LEN/8) so most documents fit a full window AND so short documents -- which
// are padded up to T with the loss masked off -- waste little compute. For a tiny context there is
// no room to vary, so MIN_TRAIN_SEQ collapses to SEQ_LEN (fixed). SUB0_FIXED_SEQ=1 forces full T.
constexpr int    MIN_TRAIN_SEQ = SEQ_LEN > 16 ? std::max(8, SEQ_LEN / 8) : SEQ_LEN;

constexpr std::uint32_t CKPT_MAGIC   = 0x4B433053u;  // "S0CK"
// EXACT-match only. Older readers (v1-v4) are gone: this project has no production models, every
// checkpoint on disk is regenerable in minutes, and carrying migration branches for formats nothing
// writes is bloat that has to stay correct through every future change. A mismatched version is
// refused with a message saying to retrain -- which is the honest instruction anyway, since a resumed
// run under a format we no longer test is not something to trust.
// History, for the record: v2 added best_step, v3 drawn_tokens, v4 drawn_names, v5 the architecture
// fingerprint. Bump this on any layout change and the guard below does the rest.
constexpr std::uint32_t CKPT_VERSION = 5u;  // v2 added best_step (prune_ckpts best-checkpoint exemption);
                                            // v3 adds drawn_tokens (the blend scheduler's fairness state);
                                            // v4 adds drawn_names (index-aligned with drawn_tokens, so a
                                            // --blend-config-replace resume can reattribute progress BY
                                            // NAME -- see RunState::drawn_names' own comment);
                                            // v5 adds the architecture fingerprint -- axes that change
                                            // COMPUTATION but not shape (LoopSplit's schedule,
                                            // ROPE_THETA), which nfloat and every dim field are blind
                                            // to (layout.hpp ARCH_FINGERPRINT).
// NOT bumped for tokens_seen (2026-07-29): the resumed token count is now carried by state.json, which
// already records it, rather than by a new checkpoint field. A bump would have invalidated every
// existing .ckpt -- including the LoopSplit arms' checkpoints, which exist precisely so those runs can
// be continued past the false plateau this same change set fixes. Breaking them to improve the
// bookkeeping about them would have been a poor trade.

// --- corpus.tok access ------------------------------------------------------
// The tokenized corpus is consumed via a read-only memory map (sub0::TokMap): the OS
// pages the stream in on demand, so it may exceed RAM without a giant allocation -- the
// out-of-core path for a FineWeb-scale corpus. Random-window training and the fixed-
// window eval index the mapping directly. A diagnostic for each mapping failure mode.
const char* tokmap_error(sub0::TokMap::Err e) {
    using E = sub0::TokMap::Err;
    switch (e) {
        case E::Missing:   return "cannot open token file";
        case E::BadMagic:  return "not a corpus.tok file (bad magic)";
        case E::Truncated: return "truncated (token count exceeds file size)";
        default:           return "ok";
    }
}

// --- On-demand tokenization (out-of-core, no corpus.tok) --------------------
// When the build skipped corpus.tok (--corpus-pretok 0) the token copy (itself ~0.6x the source
// size, see should_pretokenize in config_util.hpp) is never written to disk. Training instead
// tokenizes contiguous regions of the RAW text corpus
// on the fly with the runtime tokenizer (sub0::encode, which reproduces the configurator's
// truecasing + BPE) into bounded in-memory buffers: a rotating training buffer refilled
// from new random regions each interval (so the run still traverses the whole corpus) and
// a fixed validation buffer (so the eval metric stays comparable). RAM is capped by the
// buffers; the corpus stays on disk. Region reads are contiguous, so encode() amortises.
constexpr std::size_t OD_REGION_BYTES   = 64u << 10;   // contiguous bytes read per region
constexpr std::size_t OD_TRAIN_BUF_TOK  = 8u  << 20;   // ~32 MB rotating shuffle buffer
constexpr std::size_t OD_VAL_BUF_TOK    = 512u << 10;  // ~2 MB fixed validation window

class TextCorpus {
public:
    bool open(const std::string& path) {
        is_.open(path, std::ios::binary | std::ios::ate);
        if (!is_) return false;
        size_ = static_cast<std::size_t>(is_.tellg());
        return size_ > 0;
    }
    std::size_t bytes() const { return size_; }

    // Read up to OD_REGION_BYTES starting at the line after `off` (so we don't begin
    // mid-word) and append its tokens to `out`. The trailing partial word is harmless --
    // training windows are taken from inside the buffer, and a partial word at a region
    // join is the same kind of boundary a packed corpus already has.
    void encode_region(std::size_t off, std::vector<int>& out) {
        if (off >= size_) return;
        is_.clear();
        is_.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        std::string buf(std::min(OD_REGION_BYTES, size_ - off), '\0');
        is_.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        buf.resize(static_cast<std::size_t>(is_.gcount()));
        std::size_t b = 0;
        if (off != 0) { const std::size_t nl = buf.find('\n'); if (nl != std::string::npos) b = nl + 1; }
        if (b >= buf.size()) return;
        const std::vector<int> toks = sub0::encode(buf.substr(b));
        out.insert(out.end(), toks.begin(), toks.end());
    }

    // Fill `out` with ~target tokens from random regions in [lo, hi) bytes (training).
    void fill_random(std::size_t lo, std::size_t hi, std::size_t target,
                     std::mt19937& rng, std::vector<int>& out) {
        out.clear();
        if (hi <= lo) return;
        std::uniform_int_distribution<std::size_t> pick(lo, hi - 1);
        while (out.size() < target) {
            const std::size_t before = out.size();
            encode_region(pick(rng), out);
            if (out.size() == before) break;   // no progress -> stop
        }
    }

    // Fill `out` with ~target tokens read sequentially from `start` bytes (validation).
    void fill_sequential(std::size_t start, std::size_t target, std::vector<int>& out) {
        out.clear();
        for (std::size_t off = start; out.size() < target && off < size_; off += OD_REGION_BYTES) {
            const std::size_t before = out.size();
            encode_region(off, out);
            if (out.size() == before) break;
        }
    }
private:
    std::ifstream is_;
    std::size_t size_ = 0;
};

// Load a usable token corpus for the single-buffer tools (tune / bench / autotemp): prefer the
// baked corpus.tok memory map, otherwise tokenize a bounded buffer on demand from the raw text
// corpus -- the SAME auto-fallback the trainer uses, so these tools work on an on-demand build
// (no corpus.tok) instead of bailing out. The returned span stays valid for as long as the
// caller's `tok` (the mmap) and `od_buf` (the on-demand storage) live; on failure it is empty
// and a diagnostic naming `tool` has already been printed.
sub0::TokView load_corpus_tokens(sub0::TokMap& tok, std::vector<int>& od_buf, const char* tool) {
    if (tok.ok() && tok.vocab() == VOCAB && tok.tokens().size() > static_cast<std::size_t>(SEQ_LEN) + 2)
        return tok.tokens();                       // baked corpus.tok (the fast path)

    // No usable corpus.tok -> tokenize on demand from the raw corpus (matches the train stage).
    if (!sub0::load_tokenizer(sub0::default_tokenizer())) {   // on-demand encode needs the runtime tokenizer
        sub0::log::error("{}: cannot load this build's tokenizer ('{}') -- on-demand tokenization "
                         "would silently encode against the wrong vocabulary", tool, sub0::default_tokenizer());
        return {};
    }
    TextCorpus text;
    if (text.open(sub0::default_corpus())) {
        std::mt19937 rng(123);
        text.fill_random(0, text.bytes(), OD_TRAIN_BUF_TOK, rng, od_buf);
    }
    if (od_buf.size() > static_cast<std::size_t>(SEQ_LEN) + 2)
        return sub0::TokView::over_int32(od_buf.data(), od_buf.size());   // on-demand int32 buffer as a TokView

    sub0::log::error("{}: no usable corpus.tok ('{}') and cannot tokenize the raw corpus '{}'",
                 tool, sub0::default_corpus_tok(), sub0::default_corpus());
    return {};
}

// --- Validation NELBO -------------------------------------------------------
// Mean cross-entropy per token over a fixed, evenly-spaced set of windows in the
// held-out tail. Fixed windows make the metric comparable across evals (so the
// plateau sign test sees signal, not resampling noise) and bound the cost.
// `session` selects the compute path: a Session brought up with the device gives the GPU forward-loss
// entry, a default-constructed one (or any build without a device backend) transparently uses the CPU.
double evaluate(sub0::TokView data, std::size_t val_start, const sub0::eval::Session& session) {
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, val_start, EVAL_WINDOWS_MAX);
    if (ws.empty()) return std::numeric_limits<double>::quiet_NaN();
    return sub0::eval::nelbo(data, ws, SEQ_LEN, session, DEFAULT_THREADS);
}

// CPU-only overload for the callers that have no session to hand (the training loop's own periodic
// eval -- see sub0::eval::Session's note on why the trainer deliberately stays on the CPU).
double evaluate(sub0::TokView data, std::size_t val_start) {
    const sub0::eval::Session cpu_only(/*allow=*/false);
    return evaluate(data, val_start, cpu_only);
}

// Is a sub0llm-train holding the machine-wide guard right now? A NON-INVASIVE probe: OpenMutex only
// tests for existence, it never acquires, so it cannot perturb or block the trainer.
//
// The read-only tools (report, autotemp) are deliberately NOT subject to SingleInstanceGuard --
// refusing to inspect a model while it trains would be obnoxious, and they write nothing a trainer
// owns. They are not free, though: both run generation grids that contend for the same CPU and GPU,
// so an innocuous-looking `report` quietly steals throughput from a multi-hour sweep and, worse,
// makes the sweep's own tok/s numbers wrong for that window. Detect it and do LESS, rather than
// refuse or silently compete.
#if defined(_WIN32)
inline bool trainer_active() {
    const std::string gname = std::string("Local") + char(92) + "sub0llm-train-global";
    HANDLE h = OpenMutexA(SYNCHRONIZE, FALSE, gname.c_str());
    if (h) { CloseHandle(h); return true; }
    return false;
}
#else
inline bool trainer_active() { return false; }   // no probe wired up off Windows; Windows-first build
#endif

// --- Context-length curve: NELBO as a function of how much context the model is given ----------
//
// Mean val_nelbo is a single number averaged over every position, and it is dominated by the easy
// early ones. That makes it nearly blind to the thing depth mechanisms (LoopSplit, depth attention)
// actually claim to improve: using LONG-RANGE context. Two arms can sit within seed noise on mean
// NELBO while differing sharply in how much they gain from a longer window.
//
// This scores the SAME fixed, evenly-spaced held-out windows at several context widths. The
// interesting quantity is not any single value but the DELTA from short to long: a model that
// genuinely exploits distant context improves a lot from 64 -> 512, one that has effectively learned
// an n-gram improves very little. Reported as an absolute nelbo per width plus the gain, so an A/B
// can compare the SHAPE of the curve rather than one averaged scalar.
//
// Reuses evaluate()'s window selection exactly (same val_start, same stride) so the widths are
// scored on identical text and differ only in how much of it the model may attend to.
struct ContextCurve {
    std::vector<int>    width;   // context width in tokens
    std::vector<double> nelbo;   // mean cross-entropy at that width
};
ContextCurve evaluate_context_curve(sub0::TokView data, std::size_t val_start,
                                    const sub0::eval::Session& session) {
    ContextCurve c;
    // SHORTEN when this eval has to run on the CPU: every extra width is another full CPU pass over
    // every eval window. Endpoints only (shortest and SEQ_LEN) still gives the gain figure -- which is
    // the number an A/B actually compares -- at half the cost, instead of the intermediate shape.
    //
    // With a device session the whole curve is cheap, so measure all of it. Note the two conditions
    // coincide by construction rather than by accident: report builds its Session with
    // `allow = !trainer_active()`, so "no device" and "a trainer owns this machine" are the same
    // state, and the shortening still triggers exactly when stealing CPU from a trainer would.
    const bool shorten = !session.use_device;
    if (shorten) {
        c.width.push_back(64);
        c.width.push_back(SEQ_LEN);
    } else {
        // Powers of two up to SEQ_LEN. A width needs at least 2 tokens (one input/target pair) and
        // the full window must fit, so anything larger than SEQ_LEN is skipped rather than clamped --
        // a silently clamped width would report a duplicate of SEQ_LEN and read as a flat curve.
        for (int w = 64; w <= SEQ_LEN; w *= 2) c.width.push_back(w);
        if (c.width.empty() || c.width.back() != SEQ_LEN) c.width.push_back(SEQ_LEN);
    }

    // ONE window plan shared by every width -- that is what makes the widths comparable: they score
    // identical text and differ only in how much of it the model may attend to.
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, val_start, EVAL_WINDOWS_MAX);
    if (ws.empty()) { c.width.clear(); return c; }
    for (int cw : c.width)
        c.nelbo.push_back(sub0::eval::nelbo(data, ws, cw, session, DEFAULT_THREADS));
    return c;
}

// Mean per-token predictive entropy (nats) of the model on the held-out tail: its
// average uncertainty when *reading* real text it never trained on, over the same fixed
// windows as evaluate(). exp() of this is the natural target for generation -- we match
// the model's sampling entropy to its reading entropy. Unlike cross-entropy (inflated by
// model error -> perplexity 5.17 here), entropy compares the model to ITSELF, so the
// matched temperature is not biased upward by an imperfect fit and centers near 1.
double mean_entropy(sub0::TokView data, std::size_t val_start) {
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, val_start, EVAL_WINDOWS_MAX);
    if (ws.empty()) return std::numeric_limits<double>::quiet_NaN();
    const int nw = ws.count;
    double total = 0.0; long n = 0;
    int win[SEQ_LEN + 1];                                   // materialize the window (token view may be uint16-packed)
    for (int w = 0; w < nw; ++w) {
        data.copy_to(ws.start_of(w), SEQ_LEN, win);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(win, SEQ_LEN);
        for (int r = 0; r < logits->rows; ++r) {
            const float* row = logits->data.data() + static_cast<std::size_t>(r) * VOCAB;
            float mx = -1e30f; for (int j = 0; j < VOCAB; ++j) mx = std::max(mx, row[j]);
            double Z = 0.0; for (int j = 0; j < VOCAB; ++j) Z += std::exp(static_cast<double>(row[j] - mx));
            double H = 0.0;
            for (int j = 0; j < VOCAB; ++j) {
                const double p = std::exp(static_cast<double>(row[j] - mx)) / Z;
                if (p > 0.0) H -= p * std::log(p);
            }
            total += H; ++n;
        }
    }
    sub0::graph_reset();
    return total / static_cast<double>(std::max(1L, n));
}

using sub0::coherence::ngram_repeat;   // pure n-gram repeat metric (see coherence.hpp)

// Plateau via the pure least-squares trend test (coherence::trend_plateaued): fit a line to the last
// PLATEAU_WINDOW+1 evals and stop only when that best-fit gradient implies the series is still
// shedding less than an effective min_rel threshold of the current level across the window. The fitted
// gradient is representative of the real downward trend and shrugs off the per-eval noise that tripped
// the old sign test (which false-stopped a strongly-but-noisily descending run -- see coherence_tests).
//
// `frac_epoch` gates two epoch-aware additions on top of the trend test itself: (1) a hard floor
// (PLATEAU_MIN_EPOCH) below which plateau-stop is never considered at all, regardless of hint presence
// -- see that constant's own comment for why an early blended run's val-NELBO trend can look falsely
// flat; (2) when `expected_plateau_epoch` is given (from the schedule JSON, e.g. informed by a prior
// size-sweep calibration on the same corpus), the effective min_rel threshold interpolates LINEARLY
// between PLATEAU_MIN_REL_FAR (>= 1 epoch away, stricter) and PLATEAU_MIN_REL_AT_HINT (at or past the
// hint, looser) over exactly that 1-epoch window -- bounded on both ends, never extrapolated past them.
// Without a hint, the threshold is PLATEAU_MIN_REL unchanged (today's behavior), still gated by the new
// minimum-epoch floor.
bool plateaued(const std::vector<double>& evals, double frac_epoch,
               std::optional<double> expected_plateau_epoch = std::nullopt) {
    if (frac_epoch < PLATEAU_MIN_EPOCH) return false;
    // PATIENCE, checked before the trend test and independent of every threshold below: a run that is
    // still setting new best val-NELBOs is not plateaued, whatever a fitted slope says. This is the
    // direct fix for the 2026-07-28 false stop, where two arms were cut at the same eval while
    // improving monotonically -- their fitted drop was real, just slightly under a threshold that the
    // expected_plateau_epoch hint had loosened. See coherence::improved_recently.
    if (sub0::coherence::improved_recently(evals, PLATEAU_PATIENCE)) return false;
    double min_rel = PLATEAU_MIN_REL;
    if (expected_plateau_epoch) {
        const double dist = std::abs(frac_epoch - *expected_plateau_epoch);   // epochs from the hint
        const double t = std::clamp(1.0 - dist, 0.0, 1.0);   // 0 at/beyond 1 epoch away, 1 AT the hint
        min_rel = PLATEAU_MIN_REL_FAR + t * (PLATEAU_MIN_REL_AT_HINT - PLATEAU_MIN_REL_FAR);
    }
    return sub0::coherence::trend_plateaued(evals, PLATEAU_WINDOW, min_rel);
}

// Per-step learning rate: a LINEAR WARMUP to the peak over `warmup` steps, then a horizon-free
// INVERSE-SQRT DECAY (lr proportional to 1/sqrt(step)). We previously trained at a CONSTANT lr
// (no warmup, no decay), which makes a model oscillate around a high floor instead of settling
// into the minimum -- the exact stall we saw (train+val flat ~3.28 with the lr bouncing the loss).
// Warmup avoids the early high-lr instability; the decay lets it descend past that floor. The
// inverse-sqrt form needs NO total-step horizon, so it fits the plateau-stopped, never-a-fixed-
// step-count schedule, and is recomputed from the global step so resume is exact.
inline float lr_schedule(long step, float peak, long warmup) {
    if (warmup < 1) warmup = 1;
    const float s = static_cast<float>(step), w = static_cast<float>(warmup);
    return step < warmup ? peak * (s / w) : peak * std::sqrt(w / s);
}

// Human-scaled duration for an ETA figure -- a raw 40000-second estimate is harder to parse at a
// glance than "11.1h". Negative/non-finite (wps not yet measurable, e.g. the very first interval)
// prints as "?" rather than a nonsense duration.
inline std::string format_eta(double secs) {
    if (!(secs >= 0.0) || !std::isfinite(secs)) return "?";
    if (secs < 60.0)   return std::format("{:.0f}s", secs);
    if (secs < 3600.0) return std::format("{:.1f}m", secs / 60.0);
    return std::format("{:.1f}h", secs / 3600.0);
}

// --- Checkpoint (full optimizer + loop state, for exact resume) -------------
struct RunState {
    long step = 0;
    double best_loss = std::numeric_limits<double>::infinity();
    long best_step = -1;          // step of the best eval so far -- prune_ckpts() exempts it
    // Cumulative tokens actually trained on (sum of batch_t*seq_t over every step), CARRIED across
    // resumes rather than re-derived. See CKPT_VERSION v6: the old step*batch*SEQ_LEN reconstruction
    // silently over-counted a resumed run, which made "same steps" and "same tokens" disagree between
    // two arms of the same A/B.
    long long tokens_seen = 0;
    std::vector<double> evals;
    // Per-source cumulative tokens drawn (sub0::BlendFairness::drawn_tokens) + the NAME each entry
    // belongs to (index-aligned with each other, not necessarily with the CURRENT run's `sources[]` --
    // see sub0::carry_forward_by_name). Persisting names, not just a bare index-aligned array, matters
    // concretely for --blend-config-replace: without them, a schedule swap that happens to keep the same
    // NUMBER of sources but changes what an index means (reordered, or one swapped for a differently-
    // named one at the same slot) would silently hand a new, unrelated source the old one's progress --
    // exactly the class of silent fairness-state corruption this whole redesign exists to prevent. Both
    // empty on a v1/v2 checkpoint (predates this field) -- the caller treats that as "no data", not a
    // resume failure.
    std::vector<double> drawn_tokens;
    std::vector<std::string> drawn_names;
};

template <class T> void wr(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
}
template <class T> T rd(std::istream& is) {
    T v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v;
}

// Returns false (and logs) if BOTH the plain rename and the remove-then-rename fallback fail -- e.g.
// dst is held open by an antivirus scan, a cloud-sync client, or another process (a `sub0llm-gen`
// reading model.bin). Previously silent on this path: a failed replace left `tmp` on disk with the
// caller having no idea the checkpoint didn't land.
bool atomic_replace(const std::filesystem::path& tmp, const std::filesystem::path& dst) {
    std::error_code ec;
    std::filesystem::rename(tmp, dst, ec);
    if (ec) {  // some platforms (Windows) won't clobber an existing target
        std::filesystem::remove(dst, ec);
        std::filesystem::rename(tmp, dst, ec);
    }
    if (ec) {
        sub0::log::warn("train: could not move '{}' -> '{}': {} (the write itself succeeded; '{}' is "
                        "left on disk)", tmp.string(), dst.string(), ec.message(), tmp.string());
        return false;
    }
    return true;
}

// Pin an artifact from the live build tree into the model directory: self-contained, and immune to a
// later `configure` (for another corpus, or a plain rebuild) overwriting the shared build-tree copy
// out from under a LIVE run -- the failure mode a stray build command hit mid-session (a targeted,
// unrelated build still cascaded into the auto-configure step and nearly clobbered a live train's
// corpus.tok; see the memory notes on this). Training then loads its tokenizer/corpus exclusively
// from this frozen copy, so the model.bin fingerprint stamped throughout the run stays consistent
// with what actually trained it, and the run survives a stray reconfigure entirely.
//
// Prefers a HARDLINK (near-instant, zero extra disk -- correct even for a many-GB corpus.tok, since
// the configurator's writer never mutates a file in place: it writes a .tmp then atomically renames
// it over the target, which repoints the build tree's directory entry to NEW data while this
// hardlink's entry keeps pointing at the OLD data, untouched). Falls back to a real copy only for
// files under `copy_fallback_max_bytes` (a full copy of a many-GB corpus.tok would be slow/costly
// exactly when hardlinking matters most); above that limit a failed hardlink just warns and the
// caller keeps using the live path (same risk as before this fix, not worse). Returns the path to
// use: the pinned copy on success, `src` unchanged on failure/skip.
std::filesystem::path bundle_into_model_dir(const std::filesystem::path& src,
                                            const std::filesystem::path& dst,
                                            std::uintmax_t copy_fallback_max_bytes) {
    std::error_code ec;
    if (!std::filesystem::exists(src, ec)) return src;
    if (std::filesystem::weakly_canonical(src, ec) == std::filesystem::weakly_canonical(dst, ec)) return src;
    std::filesystem::remove(dst, ec);                          // drop a stale pin from an earlier run
    ec.clear();
    std::filesystem::create_hard_link(src, dst, ec);
    if (!ec) return dst;
    const std::uintmax_t sz = std::filesystem::file_size(src, ec);
    if (!ec && sz <= copy_fallback_max_bytes) {
        ec.clear();
        std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) return dst;
    }
    sub0::log::warn("could not pin '{}' into the model dir: {} -- training will read the live build-tree "
                    "copy, which a later `configure` run could change underneath it", src.string(), ec.message());
    return src;
}

// Write magic + the constexpr config (validated on load) + continuation state +
// RNG + eval history + parameters + both Adam moments. Written to a temp file then
// renamed so a crash mid-write never corrupts a usable checkpoint. The step budget
// (max_steps) is deliberately NOT stored: it is a per-invocation policy, so a
// resumed run continues toward the budget the *current* call asks for.
//
// Returns false on any I/O failure (open/write/replace). Deliberately does NOT retry or block here:
// this is called from the training loop's own periodic cadence (every eval interval, or every
// CKPT_SECONDS crash-resistance tick), so a transient failure (an antivirus scan, a cloud-sync lock)
// self-heals at the NEXT call a few minutes later without training ever stalling on it -- see the
// call sites for the warn-and-continue handling. Retry-with-backoff-and-fallback-filename is reserved
// for the END-of-training save, the one point where there is no "next interval" to retry at.
bool save_checkpoint(const std::string& path, long adam_t, const std::mt19937& rng,
                     const RunState& rs, int batch, float lr, unsigned seed) {
    const std::string tmp = path + ".tmp";
    std::ofstream os(tmp, std::ios::binary);
    if (!os) { sub0::log::warn("train: cannot write checkpoint '{}'", tmp); return false; }

    wr(os, CKPT_MAGIC); wr(os, CKPT_VERSION);
    for (int c : {D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB, int(USE_TERNARY)}) wr(os, c);
    const std::uint64_t nfloat = sub0::trainable_floats();
    wr(os, nfloat);
    wr(os, sub0::ARCH_FINGERPRINT);   // v5+; see CKPT_VERSION

    wr(os, static_cast<std::int64_t>(rs.step));
    wr(os, static_cast<std::int64_t>(adam_t));
    wr(os, rs.best_loss);
    wr(os, static_cast<std::int64_t>(rs.best_step));
    wr(os, static_cast<std::int32_t>(batch));
    wr(os, static_cast<std::uint32_t>(seed));
    wr(os, lr);

    std::ostringstream rs_os; rs_os << rng;            // exact mt19937 state
    const std::string rng_state = rs_os.str();
    wr(os, static_cast<std::uint32_t>(rng_state.size()));
    os.write(rng_state.data(), static_cast<std::streamsize>(rng_state.size()));

    wr(os, static_cast<std::uint32_t>(rs.evals.size()));
    for (double e : rs.evals) wr(os, e);

    wr(os, static_cast<std::uint32_t>(rs.drawn_tokens.size()));
    for (double d : rs.drawn_tokens) wr(os, d);
    // drawn_names: index-aligned with drawn_tokens above, so a --blend-config-replace resume can
    // reattribute progress BY NAME (sub0::carry_forward_by_name) instead of blindly trusting position.
    wr(os, static_cast<std::uint32_t>(rs.drawn_names.size()));
    for (const std::string& nm : rs.drawn_names) {
        wr(os, static_cast<std::uint32_t>(nm.size()));
        os.write(nm.data(), static_cast<std::streamsize>(nm.size()));
    }

    const auto bytes = static_cast<std::streamsize>(nfloat * sizeof(float));
    sub0::sync_params_to_host();   // device backends: stage live params/moments into the *_ptr() buffers
    os.write(reinterpret_cast<const char*>(sub0::params_ptr()), bytes);
    os.write(reinterpret_cast<const char*>(sub0::adam_m_ptr()), bytes);
    os.write(reinterpret_cast<const char*>(sub0::adam_v_ptr()), bytes);
    os.flush();
    if (!os) { sub0::log::warn("train: checkpoint write failed '{}'", tmp); return false; }
    os.close();
    return atomic_replace(tmp, path);
}

// Bounded retry-with-backoff for a save action that MUST eventually succeed -- used only at the very
// end of training, where (unlike the periodic saves above) there is no later checkpoint interval to
// naturally retry at. `what` names the action for the log. Blocking here is fine: training is already
// finished, so a few seconds' delay costs nothing the way it would mid-run.
template <class F>
bool retry_with_backoff(F&& attempt, const char* what) {
    constexpr int kMaxAttempts = 5;
    constexpr auto kRetryDelay = std::chrono::seconds(3);
    for (int i = 0; i < kMaxAttempts; ++i) {
        if (attempt()) return true;
        if (i + 1 < kMaxAttempts) {
            sub0::log::warn("{} failed (attempt {}/{}) -- retrying in {}s...",
                            what, i + 1, kMaxAttempts, kRetryDelay.count());
            std::this_thread::sleep_for(kRetryDelay);
        }
    }
    return false;
}

// Restore a checkpoint onto an already-built model. Returns false (leaving the
// fresh model untouched) if the file is absent, malformed, or built for a different
// config -- so a stale checkpoint never silently trains the wrong thing. The restored
// batch/lr/seed and Adam counter (adam_t) flow out so the caller can continue the run
// exactly as it was configured.
bool load_checkpoint(const std::string& path, std::mt19937& rng, RunState& rs,
                     int& batch, float& lr, unsigned& seed, long& adam_t) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    const std::uint32_t magic = rd<std::uint32_t>(is);
    const std::uint32_t version = rd<std::uint32_t>(is);
    // v1->v2->v3->v4 each only ADDED a field (best_step, then drawn_tokens, then drawn_names) -- the
    // weights/optimizer/RNG bytes underneath are byte-identical at every step, unlike e.g. the RoPE
    // convention bump (a genuine math change, where reading an old checkpoint with new code would
    // silently mean something different). Reading every prior version here (defaulting best_step to
    // "unknown", drawn_tokens/drawn_names to empty) matters in practice, not just in principle: a live
    // training run keeps writing checkpoints
    // with its own already-running (pre-bump) binary until it is actually restarted, and this is also
    // the resume path -- rejecting an older version outright would force a fresh restart the next time
    // that run is resumed after a rebuild, discarding real GPU-hours.
    if (magic != CKPT_MAGIC || version != CKPT_VERSION) {
        sub0::log::warn("ignoring checkpoint '{}' (bad magic/version)", path);
        return false;
    }
    // SEQ_LEN is skipped under RoPE for the same reason load_model skips it (engine_core.cpp): with
    // no pos_emb table, no parameter shape depends on the window, so a checkpoint stays loadable
    // across window sizes. The nfloat comparison below still rejects any real shape mismatch. The
    // read must happen regardless -- the field is in the stream either way.
    bool ok = true;
    int field = 0;
    for (int ref : {D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB, int(USE_TERNARY)}) {
        const bool is_seq_len = (field++ == 4);
        const int got = rd<int>(is);
        if (got != ref && !(is_seq_len && !sub0::HAS_POS_EMB)) ok = false;
    }
    const std::uint64_t nfloat = rd<std::uint64_t>(is);
    // v5+ carries the execution-schedule id. A pre-v5 checkpoint predates LoopSplit and was therefore
    // trained un-looped, so that is the correct assumption rather than "unknown, skip the check" --
    // resuming un-looped weights into a looped build would silently train a different architecture.
    const std::uint64_t arch = rd<std::uint64_t>(is);
    if (!ok || nfloat != sub0::trainable_floats()) {
        sub0::log::warn("ignoring checkpoint '{}' (built for a different config)", path);
        return false;
    }
    if (arch != sub0::ARCH_FINGERPRINT) {
        const sub0::ArchAxes got = sub0::arch_axes_of(arch);
        sub0::log::warn("ignoring checkpoint '{}': built with a different architecture in a way nfloat "
                        "cannot catch -- loop schedule {}x{} vs {}x{}, rope theta {:g} vs {:g}, "
                        "depth-attn stride {} vs {}.",
                        path, got.middle_layers, got.repeats, LOOP_MIDDLE_LAYERS, LOOP_REPEATS,
                        got.rope_theta, ROPE_THETA, got.depth_attn_stride, DEPTH_ATTN_STRIDE);
        return false;
    }

    rs.step      = static_cast<long>(rd<std::int64_t>(is));
    adam_t       = static_cast<long>(rd<std::int64_t>(is));
    rs.best_loss = rd<double>(is);
    rs.best_step = static_cast<long>(rd<std::int64_t>(is));
    batch        = rd<std::int32_t>(is);
    seed         = rd<std::uint32_t>(is);
    lr           = rd<float>(is);

    const std::uint32_t rng_len = rd<std::uint32_t>(is);
    std::string rng_state(rng_len, '\0');
    is.read(rng_state.data(), rng_len);
    std::istringstream(rng_state) >> rng;

    const std::uint32_t nev = rd<std::uint32_t>(is);
    rs.evals.resize(nev);
    for (auto& e : rs.evals) e = rd<double>(is);

    rs.drawn_tokens.clear();
    rs.drawn_names.clear();
    {
        const std::uint32_t ndrawn = rd<std::uint32_t>(is);
        rs.drawn_tokens.resize(ndrawn);
        for (auto& d : rs.drawn_tokens) d = rd<double>(is);
    }
    {
        const std::uint32_t nnames = rd<std::uint32_t>(is);
        rs.drawn_names.resize(nnames);
        for (auto& nm : rs.drawn_names) {
            const std::uint32_t len = rd<std::uint32_t>(is);
            nm.resize(len);
            is.read(nm.data(), len);
        }
    }

    const auto bytes = static_cast<std::streamsize>(nfloat * sizeof(float));
    is.read(reinterpret_cast<char*>(sub0::params_ptr()), bytes);
    is.read(reinterpret_cast<char*>(sub0::adam_m_ptr()), bytes);
    is.read(reinterpret_cast<char*>(sub0::adam_v_ptr()), bytes);
    if (!is) { sub0::log::warn("checkpoint '{}' truncated -- starting fresh", path); return false; }
    sub0::sync_params_to_device();   // device backends: push loaded params/moments to the live copy
    return true;
}

// --- Progress-named checkpoint retention ------------------------------------
// Checkpoints are saved as "<model>.step<NNNNNNNNN>.ckpt" (zero-padded so a lexicographic sort is
// also temporal), so successive saves never overwrite each other and any stage stays resumable.
// `--keep N` bounds how many are retained (the newest N by step; keep<0 = keep ALL, a dev option).
// Resume picks the highest-step checkpoint, falling back to a legacy "<model>.ckpt" from older builds.
std::string ckpt_step_path(const std::string& model_path, long step) {
    return std::format("{}.step{:09d}.ckpt", model_path, step);
}
std::string legacy_ckpt_path(const std::string& model_path) { return model_path + ".ckpt"; }

// (step, path) for every progress-named checkpoint of this model, ascending by step.
std::vector<std::pair<long, std::filesystem::path>> list_step_ckpts(const std::string& model_path) {
    std::vector<std::pair<long, std::filesystem::path>> out;
    const std::filesystem::path mp(model_path);
    const std::filesystem::path dir = mp.parent_path();
    const std::string prefix = mp.filename().string() + ".step";           // "model.bin.step"
    const std::string suffix = ".ckpt";
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(dir.empty() ? std::filesystem::path(".") : dir, ec)) {
        if (ec) break;
        const std::string name = de.path().filename().string();
        if (name.size() <= prefix.size() + suffix.size()) continue;
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        const std::string digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        char* end = nullptr; const long st = std::strtol(digits.c_str(), &end, 10);
        if (end && *end == '\0') out.emplace_back(st, de.path());
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

// The newest checkpoint to resume from: highest-step progress file, else the legacy path (or "").
std::string latest_ckpt_path(const std::string& model_path) {
    const auto v = list_step_ckpts(model_path);
    if (!v.empty()) return v.back().second.string();
    const std::string legacy = legacy_ckpt_path(model_path);
    return std::filesystem::exists(legacy) ? legacy : std::string();
}

// Keep the newest `keep` progress checkpoints (keep<0 = keep all), and ALWAYS additionally keep
// best_step (the checkpoint at the lowest val_nelbo seen so far), even if it has aged out of the
// newest-`keep` window -- otherwise a plateau developing right after the best eval rotates the best
// checkpoint out before the plateau detector (which needs PLATEAU_WINDOW more evals of evidence)
// ever gets to stop training on it, silently losing the best model this run ever produced in favor
// of a later, worse one. Prunes the oldest first, then removes any legacy single-file
// "<model>.ckpt" once progress files exist (it is then redundant).
void prune_ckpts(const std::string& model_path, int keep, long best_step) {
    if (keep < 0) return;
    const auto v = list_step_ckpts(model_path);
    std::error_code ec;
    std::vector<std::pair<long, std::filesystem::path>> prunable;
    for (const auto& e : v) if (e.first != best_step) prunable.push_back(e);
    if (static_cast<int>(prunable.size()) > keep)
        for (size_t i = 0; i + static_cast<size_t>(keep) < prunable.size(); ++i)   // drop the oldest
            std::filesystem::remove(prunable[i].second, ec);
    if (!v.empty()) std::filesystem::remove(legacy_ckpt_path(model_path), ec);
}

// A short text sample at a given temperature/top-k. Uses the SAME sampler as `gen`
// so the output reflects real generation quality -- the old greedy+noise hack made a
// coherent model look like word-salad.
//
// Fast path: the same KV-cache decode gen_stage.cpp uses (sub0/decode.hpp), whenever the whole
// generation fits the trained window and the weights are dense -- `use_gpu` selects a GPU session an
// earlier sub0::DecodeSession already brought up (pass sess.use_gpu; callers making several preview_at
// calls in a row -- report's sample battery, autotemp's final sample -- should share ONE session
// rather than re-paying CUDA init/upload per call).
//
// `n` is "up to n more tokens, BOUNDED BY THE TRAINED WINDOW" -- it is clamped to SEQ_LEN - ctx.size()
// below. That clamp is what keeps every caller on the KV-cache path, and it fixes a severe, long-
// mystifying performance cliff: the fast path's precondition is `ctx.size() + n <= SEQ_LEN`, so asking
// for a full SEQ_LEN-token sample after ANY non-empty prompt missed it by exactly the prompt's length
// and silently fell into the loop below -- a FULL FORWARD PER TOKEN, ~100x slower. report's sample
// battery asks for exactly that, so every sample in every report took the slow path.
//
// It is also the root cause of the "sub0llm-gen freezes at --n >= SEQ_LEN while --n 500 is fine"
// report: nothing was hung: --n 500 after a 4-token prompt fits the window and decodes in a second,
// --n 512 does not fit and grinds through 512 full forwards.
//
// The loop below is NOT dead -- it is the only path under USE_TERNARY, where KV-cache decode is
// unavailable. It is a plain last-SEQ_LEN slice; preview_at never needs attention sinks (a gen-only
// --attn-sinks feature).
std::string preview_at(const std::string& prompt, int n, float temp, int topk, std::mt19937& rng,
                       bool use_gpu) {
    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    n = std::min(n, SEQ_LEN - static_cast<int>(ctx.size()));
    if (n <= 0) return sub0::detokenize(ctx);   // the prompt already fills the window
    // Same learned stop signal gen_stage.cpp's decode loops check (see core.hpp's eos_token_id()
    // doc comment): stop BEFORE pushing so the marker is never printed, instead of always running
    // to the fixed token budget `n`. Without this, previews/report.txt samples ran straight through
    // EOS and kept sampling past it -- a context the model was never trained to continue from (every
    // training document ends there), producing the `<|endoftext|>`-chained rambling this fixes.
    const int eos_id = sub0::eos_token_id();
    if constexpr (!USE_TERNARY) {
        if (static_cast<int>(ctx.size()) + n <= SEQ_LEN) {
            sub0::kv_decode_generate(ctx, n, temp, topk, rng, eos_id, use_gpu);
            return sub0::detokenize(ctx);
        }
    }
    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
        const int last = logits->rows - 1;
        const int next = sub0::sample_token(logits->data.data() + (size_t)last * VOCAB, temp, topk, rng);
        if (next == eos_id) break;
        ctx.push_back(next);
    }
    sub0::graph_reset();
    return sub0::detokenize(ctx);
}

// End-of-training preview (the only caller is the "reached max steps" print below, not a
// mid-training eval tick) at `gen`'s own defaults -- temp 0.8, top-k 20 (cli_stages.hpp's
// gen_temp/gen_topk). Was 0.7 here (a stale mismatch, not a deliberate choice: this project has
// no single shared constant for the pair, so the two drifted apart) -- fixed so the printed
// sample is representative of what `sub0llm gen model.bin <prompt>` actually produces. Owns its own
// (single-shot) DecodeSession: called once, right before the training run's own gpu.shutdown() --
// harmless even when a GPU training session is still nominally "active" at that point, since
// gpu_decode_try_enable()/gpu_decode_shutdown() are idempotent/safe to call on top of it (see
// sub0/decode.hpp) and nothing device-side happens after this in sub0_train_stage.
std::string preview(const std::string& prompt, int n, std::mt19937& rng) {
    sub0::DecodeSession sess;
    return preview_at(prompt, n, 0.8f, 20, rng, sess.use_gpu);
}

}  // namespace

// Device training session (Phase 2e): keeps the parameters resident on the GPU and runs each
// forward+backward+AdamW step there, syncing back to the host param/optimizer arenas only at the
// eval/checkpoint boundaries where the unchanged CPU eval/save/preview code runs. No-op (active
// stays false) unless the CUDA backend is compiled in and a device initializes.
namespace {
struct GpuTrainer {
    bool active = false;
    // Set by the caller once enable() succeeds under content_embed_active (source-routed hybrid
    // CPU/GPU training -- see the hybrid_train branch in sub0_train_stage). In hybrid mode the HOST
    // arenas are the only ones an optimizer step ever advances (opt.step() runs on host; the device
    // only ever gets a one-way re-upload of the resulting params before its next forward pass, via
    // sub0_dev_upload_params) -- device optimizer state is never touched past enable()'s initial
    // upload, so sync_to_host() downloading it would clobber the host's real Adam moments with stale
    // frozen values. Never set by GpuTrainer itself; the caller is the only one who knows which mode
    // this run is in.
    bool hybrid = false;
    long t = 0;                       // AdamW step counter driving the device bias correction
    std::vector<int> ids, targets;    // per-step [batch*SEQ_LEN] window buffers
    // Set by step()/sync_to_host() on their most recent call. A device fault (illegal memory access
    // or similar -- see [[gpu-illegal-access-hardware-fault-not-code-bug]], a real hardware fault hit
    // this exact session) leaves the CUDA context permanently corrupted for the rest of the process
    // (CUDA errors are sticky); the caller must stop touching the device entirely once this is false,
    // not just log and keep training on stale/garbage state.
    bool ok = true;

    // Enable the device path: requires the CUDA backend, a device, and a batch within the
    // resident scratch width. Uploads the current host params (+ optimizer moments, zero on a
    // fresh run / restored on resume). Returns false to fall back to the CPU loop.
    //
    // `batch` is IN/OUT: optimistic-allocate-then-fallback. Tries the FULL requested batch (a
    // checkpoint's saved value, or the tuned DEFAULT_GPU_BATCH) via a REAL cudaMalloc attempt first,
    // rather than pre-emptively shrinking it from a VRAM *prediction* -- the card, driver, or a
    // concurrent GPU load can differ from whenever this batch was last tuned/written, in EITHER
    // direction, so trusting the live allocator's actual answer is more reliable (and less
    // needlessly conservative) than a predicted budget. Only falls back to a smaller,
    // conservatively-predicted floor if the optimistic attempt genuinely fails; if even the floor
    // doesn't fit, gives up on GPU entirely (returns false, caller uses the CPU path).
    //
    // `ids`/`targets` stay sized at (the FINAL, possibly-fallen-back) batch*SEQ_LEN (the token
    // budget) regardless of how the caller's per-step (batch_t, seq_t) pairs vary under the
    // token-budget scheduler below: every admitted pair satisfies batch_t*seq_t <= batch*SEQ_LEN by
    // construction (that IS the token budget), so this buffer is already big enough for any of them
    // -- no separate resize needed even though batch_t can individually exceed `batch` at a short
    // seq_t.
    [[nodiscard]] bool enable(int& batch, long resume_t) {
#if defined(SUB0_BUILD_CUDA)
        constexpr int kCap = 4096;                  // matches MAX_FWD_BATCH (the device scratch ceiling)
        //TODO: Remove getenv calls - these are tmeporary and we should use commandline/compiletime
        if (std::getenv("SUB0_TRAIN_CPU")) return false;   // measurement / fallback override
        if (!HAS_CUDA || batch > kCap) return false;
        // Honour the capability bits rather than discovering the gap at the first call. docs/BACKENDS.md
        // promises an inference-only backend ("supports_train=0, supports_decode=1") makes "the train
        // stage keep the CPU without any consumer special-casing" -- that was documentation only, since
        // nothing read supports_train. Same class of gap as the supports_decode one fixed in ec1d8cf: a
        // caps bit no consumer branches on is not a guard. No behaviour change on the CUDA backend, which
        // reports 1 for both.
        if (!sub0_dev_caps().supports_train) return false;
        if (sub0_dev_init() != 0) return false;
        if (sub0_dev_upload_params(sub0::params_ptr()) != 0) return false;
        // Device-resident optimizer moments. A backend without them cannot run this device-resident step
        // shape at all (opt.step() would advance host state the device never sees), so decline rather
        // than half-configure -- the hybrid path is a deliberate, caller-selected mode, not a fallback.
        if (!sub0_dev_caps().supports_opt_state) return false;
        if (sub0_dev_upload_opt(sub0::adam_m_ptr(), sub0::adam_v_ptr()) != 0) return false;
        // Pin the row budget up front (batch*SEQ_LEN rows) so no per-step (batch_t, seq_t) pair
        // ever triggers a grow-realloc mid-run. A failed sub0_dev_train_reserve leaves the backend's
        // scratch in a consistent (if partial) state -- the retry below's OWN reserve call correctly
        // frees and rebuilds from scratch, no special cleanup needed here.
        if (sub0_dev_train_reserve(batch) != 0) {
            // tied/qk_norm/gated must match USE_TIED_EMBEDDINGS/USE_QK_NORM/USE_GATED_FFN -- see
            // memplan.hpp's Dims comments.
            const sub0::memplan::Dims dims = sub0::current_build_dims();
            const int act_b = ACT_DTYPE == Dtype::BF16 ? 2 : 4;
            constexpr int kVramHeadroomMB = 512;   // cuBLAS workspace + allocator fragmentation slack
            const int free_mb = sub0_dev_free_mem_mb();
            const int vram_budget = (free_mb > 0 ? std::min(free_mb, static_cast<int>(GPU_VRAM_MB)) : GPU_VRAM_MB)
                                    - kVramHeadroomMB;
            const int floor = sub0::memplan::max_batch_for_vram(dims, vram_budget, batch, act_b);
            if (floor < 1 || sub0_dev_train_reserve(floor) != 0) return false;   // even the floor doesn't fit
            sub0::log::warn("batch {} did not fit VRAM -- falling back to {} ({} MiB usable of {} free)",
                            batch, floor, vram_budget, free_mb);
            batch = floor;
        }
        t = resume_t;
        active = true;
        ids.resize(static_cast<std::size_t>(batch) * SEQ_LEN);
        targets.resize(ids.size());
        return true;
#else
        (void)batch; (void)resume_t; return false;
#endif
    }

    // One resident device step over the sampled windows; returns the mean loss. Each window b is
    // materialized from its blend source `sources[src_idx[b]]` (x = view[start+s], y = view[start+s+1]);
    // a window shorter than T (a short document) fills its leading `lengths[b]` rows and PADS the rest,
    // with the trailing positions loss-masked on the device via the same `lengths` array -- so no
    // document is dropped and the padding contributes no gradient (causal attention keeps it out of the
    // real tokens). A masked source additionally writes LOSS_IGNORE_INDEX into the masked target rows
    // (the interior mask), read by the device's ignore-index cross-entropy -- parity-gated against the
    // CPU loss-masked path (tests/cuda_tests.cpp "CUDA masked (ignore-index) backward..."), so masked
    // sources run on GPU exactly like unmasked ones; no separate gating needed here.
    float step([[maybe_unused]] const std::vector<sub0::BlendSource>& sources,
               [[maybe_unused]] const int* src_idx, [[maybe_unused]] const std::size_t* starts,
               [[maybe_unused]] const int* lengths, [[maybe_unused]] int batch,
               [[maybe_unused]] int T, [[maybe_unused]] float lr,
               [[maybe_unused]] float muon_lr = 0.f) {
#if defined(SUB0_BUILD_CUDA)
        for (int b = 0; b < batch; ++b) {
            const sub0::BlendSource& src = sources[static_cast<std::size_t>(src_idx[b])];
            const std::size_t w = starts[b];               // window base into the source's token view
            const int  len = lengths ? lengths[b] : T;
            int s = 0;
            for (; s < len; ++s) {
                const std::size_t tp = w + static_cast<std::size_t>(s) + 1;   // target token index
                ids[static_cast<std::size_t>(b) * T + s]     = src.view[w + static_cast<std::size_t>(s)];
                targets[static_cast<std::size_t>(b) * T + s] =
                    (src.masked() && !src.mask[tp]) ? sub0::LOSS_IGNORE_INDEX : src.view[tp];
            }
            for (; s < T; ++s) {                       // pad the tail of a short window (loss-masked)
                ids[static_cast<std::size_t>(b) * T + s]     = 0;
                targets[static_cast<std::size_t>(b) * T + s] = 0;
            }
        }
        double loss = 0.0;
        // muon_lr <= 0 is pure AdamW on the GPU path (bit-identical to before this feature existed);
        // > 0 hybridizes with Muon on the Muon-eligible matrices -- see sub0_dev_train_step's own
        // comment (backend_cuda.cu) and the call site below (mirrors the CPU branch's own
        // opt.use_muon() ? lr_schedule(..., MUON_LR_BASE, ...) : implicit-AdamW-only computation).
        ok = (sub0_dev_train_step(ids.data(), targets.data(), batch, T, lr, ++t, &loss, lengths, muon_lr) == 0);
        return static_cast<float>(loss);
#else
        return 0.0f;
#endif
    }

    // Forward+backward ONLY -- no optimizer step applied (unlike step()). Fills out_grad[PARAM_FLOATS]
    // with the reduced device gradient so the caller can weighted-combine it with a CPU sub-batch's
    // gradient into ONE update instead of each backend independently updating its own resident parameter
    // copy -- the source-routed hybrid CPU/GPU training path (project memory
    // hybrid-cpu-gpu-execution-design; algorithm proven in tests/cuda_tests.cpp "CUDA+CPU hybrid split:
    // weighted-merged gradient matches an undivided CPU batch"). Same window-materialization as step()
    // (this call's OWN `batch`/`src_idx`/`starts`/`lengths` describe just the caller's chosen SUBSET of
    // a step's windows -- step() already treats its own `batch` this way, not necessarily a full step's);
    // reuses the same `ids`/`targets` scratch step() uses (the two are never called in the same step, so
    // no aliasing risk). Returns the sub-batch's own mean loss (informational -- the caller's real
    // combined loss uses the same weighting the gradient merge does).
    float backward_only([[maybe_unused]] const std::vector<sub0::BlendSource>& sources,
                        [[maybe_unused]] const int* src_idx, [[maybe_unused]] const std::size_t* starts,
                        [[maybe_unused]] const int* lengths, [[maybe_unused]] int batch,
                        [[maybe_unused]] int T, [[maybe_unused]] float* out_grad) {
#if defined(SUB0_BUILD_CUDA)
        for (int b = 0; b < batch; ++b) {
            const sub0::BlendSource& src = sources[static_cast<std::size_t>(src_idx[b])];
            const std::size_t w = starts[b];
            const int  len = lengths ? lengths[b] : T;
            int s = 0;
            for (; s < len; ++s) {
                const std::size_t tp = w + static_cast<std::size_t>(s) + 1;
                ids[static_cast<std::size_t>(b) * T + s]     = src.view[w + static_cast<std::size_t>(s)];
                targets[static_cast<std::size_t>(b) * T + s] =
                    (src.masked() && !src.mask[tp]) ? sub0::LOSS_IGNORE_INDEX : src.view[tp];
            }
            for (; s < T; ++s) {
                ids[static_cast<std::size_t>(b) * T + s]     = 0;
                targets[static_cast<std::size_t>(b) * T + s] = 0;
            }
        }
        double loss = 0.0;
        ok = (sub0_dev_backward(ids.data(), targets.data(), batch, T, out_grad, &loss, lengths) == 0);
        return static_cast<float>(loss);
#else
        return 0.0f;
#endif
    }

    // Refresh the host param + optimizer arenas from the device (before eval / checkpoint / save).
    // Returns false on a device failure -- the caller must NOT trust the host arenas (or proceed to
    // save/eval from them) when this happens; see `ok`'s own comment above.
    [[nodiscard]] bool sync_to_host() {
#if defined(SUB0_BUILD_CUDA)
        // hybrid: host is ALWAYS already canonical (see `hybrid`'s own comment) -- a download here
        // would overwrite correct host state with a stale/frozen device mirror, not refresh it.
        if (!active || hybrid) return true;
        ok = sub0_dev_download_params(sub0::params_ptr()) == 0 &&
             sub0_dev_download_opt(sub0::adam_m_ptr(), sub0::adam_v_ptr()) == 0;
        return ok;
#else
        return true;
#endif
    }

    void shutdown() {
#if defined(SUB0_BUILD_CUDA)
        if (active) sub0_dev_shutdown();
#endif
        active = false;
    }
};

// A persistent worker for the hybrid_train branch's CPU sub-batch, so the per-step split doesn't pay
// std::thread's OS-level construction cost (stack + control block) every training step -- AGENTS.md
// #1 treats that class of per-step cost the same as a per-step heap allocation. Runs ONLY
// sub0::train_batch over the CPU-routed window subset the caller fills into the referenced buffers
// before each run(); the referenced vectors are never resized concurrently with a pending job (the
// caller always wait()s before touching them again -- see the hybrid_train branch below). Not
// thread-safe against multiple concurrent run() callers; this run's step loop is the only caller.
struct CpuSubBatchWorker {
    std::vector<int>& win; std::vector<std::size_t>& starts; std::vector<int>& lens;
    std::vector<std::uint8_t>& mask; std::vector<const sub0::ScratchBindings*>& binds;
    int n = 0, T = 0; bool masked = false;
    float loss = 0.f;   // valid once wait() returns

    std::mutex m;
    std::condition_variable_any cv_go, cv_done;
    bool has_job = false, done = true;
    std::jthread thread;

    CpuSubBatchWorker(std::vector<int>& win_, std::vector<std::size_t>& starts_, std::vector<int>& lens_,
                      std::vector<std::uint8_t>& mask_, std::vector<const sub0::ScratchBindings*>& binds_)
        : win(win_), starts(starts_), lens(lens_), mask(mask_), binds(binds_) {
        thread = std::jthread([this](std::stop_token st) {
            std::unique_lock lock(m);
            for (;;) {
                const bool got_job = cv_go.wait(lock, st, [this] { return has_job; });
                if (!got_job) return;   // stop requested, no pending job -- exit
                has_job = false;
                lock.unlock();
                loss = sub0::train_batch(win.data(), starts.data(), n, T, lens.data(),
                                         masked ? mask.data() : nullptr, binds.data());
                lock.lock();
                done = true;
                cv_done.notify_one();
            }
        });
    }
    // Submit this step's CPU sub-batch (n windows, already materialized into win/starts/lens/mask/binds
    // by the caller); returns immediately, result is ready once wait() returns.
    void run(int n_, int T_, bool masked_) {
        n = n_; T = T_; masked = masked_;
        { std::lock_guard lock(m); has_job = true; done = false; }
        cv_go.notify_one();
    }
    void wait() {
        std::unique_lock lock(m);
        cv_done.wait(lock, [this] { return done; });
    }
    // ~CpuSubBatchWorker(): std::jthread's destructor requests stop then joins; the stop_token-aware
    // cv_go.wait() above wakes on that request even with no job pending, so the worker exits cleanly.
};
}  // namespace

// Graceful stop (Ctrl+C, a console window closing, or `taskkill` WITHOUT /F -- all deliver the same
// console control event on Windows) so an interrupted run saves a checkpoint at the step it reached
// instead of losing everything back to the last periodic save. The training loop polls this flag once
// per step (cheap) rather than trying to save directly from the OS callback, which can fire on a
// separate thread while a step is mid-flight.
namespace {
std::atomic<bool> g_graceful_stop{false};

#if defined(_WIN32)
BOOL WINAPI console_ctrl_handler(DWORD /*ctrl_type*/) {
    g_graceful_stop.store(true);
    return TRUE;   // handled -- without this Windows terminates immediately instead of giving the
                   // process a chance to notice the flag and save (it still grants only a few
                   // seconds before force-killing, comfortably more than one training step takes).
}
// RAII: registers the handler for this scope's lifetime and unregisters it on the way out, so a
// training run never leaves a dangling handler pointing at a function whose enclosing call has
// already returned.
struct ConsoleCtrlGuard {
    ConsoleCtrlGuard()  { SetConsoleCtrlHandler(console_ctrl_handler, TRUE); }
    ~ConsoleCtrlGuard() { SetConsoleCtrlHandler(console_ctrl_handler, FALSE); }
};
#else
struct ConsoleCtrlGuard {};   // no-op off Windows (this project builds Windows-first; SIGINT/SIGTERM
                              // via std::signal would be the POSIX equivalent if ever needed there)
#endif
}  // namespace

// Single-instance guards. A named OS mutex fails a duplicate launch fast, with a clear message, instead
// of letting it do damage. TWO scopes, because the two failure modes are different:
//
//   PER-MODEL-DIRECTORY -- always enforced, never overridable. Two writers on one checkpoint corrupt it.
//     Scope is the DIRECTORY, not model_path's exact string, so `models/x` and `models/x/model.bin`
//     (both accepted as --out) collide correctly.
//
//   GLOBAL -- enforced by default, released by --allow-concurrent. Two trainers on DIFFERENT model dirs
//     corrupt nothing, so this was previously allowed outright ("training two different models
//     concurrently is legitimate and common"). That reasoning ignored the SHARED resources: one GPU and
//     one page cache. Concurrent runs silently halve each other's throughput and inflate each other's
//     memory pressure -- and, the reason this is now a hard default, they produce measurements that are
//     wrong in a way no log line reveals. A leaked trainer and a genuine pipeline bottleneck look
//     identical in a tok/s trace; three throughput measurements were published and retracted here before
//     the second process was spotted. Correctness of MEASUREMENT is what this scope protects.
//
// --allow-concurrent exists because deliberate concurrency is legitimate: a small CPU sample-train
// alongside a long GPU run, say. It relaxes ONLY the global scope -- the per-directory guard still
// holds, because no flag makes two writers on one checkpoint safe.
#if defined(_WIN32)
struct SingleInstanceGuard {
    HANDLE mutex_ = nullptr;
    bool   held_  = false;

    // scope_key: a model-directory path for the per-directory guard, or an EMPTY path for the
    // machine-wide (global) one.
    explicit SingleInstanceGuard(const std::filesystem::path& scope_key) {
        if (scope_key.empty()) {
            const std::string gname = std::string("Local") + char(92) + "sub0llm-train-global";
            mutex_ = CreateMutexA(nullptr, TRUE, gname.c_str());
            held_  = mutex_ != nullptr && GetLastError() != ERROR_ALREADY_EXISTS;
            return;
        }
        std::error_code ec;
        const std::string canon = std::filesystem::weakly_canonical(scope_key, ec).string();
        const std::size_t h = std::hash<std::string>{}(canon.empty() ? scope_key.string() : canon);
        // "Local\" (not "Global\"): scoped to this user session, matching every other artifact this
        // tool touches; mutex names must also avoid backslashes, which the hash sidesteps entirely.
        const std::string name = std::format("Local\\sub0llm-train-{:016x}", h);
        mutex_ = CreateMutexA(nullptr, TRUE, name.c_str());
        held_  = mutex_ != nullptr && GetLastError() != ERROR_ALREADY_EXISTS;
    }
    ~SingleInstanceGuard() { if (mutex_) { if (held_) ReleaseMutex(mutex_); CloseHandle(mutex_); } }
    bool held() const { return held_; }
};
#else
struct SingleInstanceGuard {
    explicit SingleInstanceGuard(const std::filesystem::path&) {}
    bool held() const { return true; }   // no-op off Windows (see ConsoleCtrlGuard above). Both scopes
                                         // degrade to "always permitted" here; a POSIX build wanting the
                                         // guard would use an O_EXCL lockfile or flock on the same keys.
};
#endif

// Shared run-context banner (model/compute/training-backend), defined below; train and tune share it.
static void report_run_context(bool gpu_train, bool hybrid_train = false);

extern "C" SUB0_API int sub0_train_stage(const char* corpus_path, const char* model_out,
                                          int steps, double epochs, int batch, float lr, unsigned seed, int keep,
                                          int optimizer, int resume_mode, const char* blend_config_path,
                                          int replace_schedule, int allow_concurrent,
                                          double corpus_fraction, unsigned subset_seed) {
    const ConsoleCtrlGuard _console_ctrl_guard;   // Ctrl+C / window close -> save before exit, see above
    // Model storage: an explicit path is honoured as-is; otherwise lay the model out in a
    // structured, identity-named directory (corpus + dims) under the models root and register it
    // with a state.json, so `models` can discover it and prune incompatible ones. Resolved FIRST
    // (before touching the tokenizer/corpus) so the pinning below has somewhere to pin into.
    std::string model_path;
    std::filesystem::path meta_dir;
    const std::string created = sub0::registry::now_iso();
    long epoch_steps = 1;   // real value set with the schedule below; meta writes read it live
    if (model_out && *model_out) {
        // Second line of defence for --model naming something that is already a file.
        //
        // The FIRST line is the CLI shape itself: --model and --corpus are both named options now (see
        // cli_stages.hpp), so nothing can bind to the write target by position any more. That shape is
        // what actually closed this hole; the old usage was `train [model] [corpus]`, so typing just a
        // corpus -- the obvious reading -- made it the MODEL, and save_model wrote a model.bin straight
        // over it. It destroyed data/fineweb_smoke.txt and then data/tinystories.txt.
        //
        // This check stays anyway, because `--model <a corpus>` is still typeable, and the cost of a
        // false negative is a destroyed corpus. An existing regular file that is not a model.bin is a
        // corpus by any reasonable reading: refuse rather than write there.
        {
            const std::filesystem::path mp(model_out);
            std::error_code ec_;
            if (std::filesystem::is_regular_file(mp, ec_) && mp.filename() != "model.bin"
                && mp.extension() != ".bin") {
                sub0::log::error(
                    "train: --model '{}' is an existing file that is not a model.bin, so it looks like a "
                    "CORPUS -- training would OVERWRITE it with model weights. Did you mean --corpus "
                    "'{}'? Omit --model entirely to auto-name one under the models root.",
                    model_out, model_out);
                return 1;
            }
        }
        model_path = model_out;
        // Accept a model DIRECTORY too (e.g. resuming `models/<name>`): use its model.bin, so a dir path
        // doesn't mis-resolve the checkpoint to "<dir>.ckpt", silently start fresh, and nest a new dir.
        if (std::filesystem::is_directory(model_path))
            model_path = (std::filesystem::path(model_path) / "model.bin").string();
        meta_dir = std::filesystem::path(model_path).parent_path();   // state.json/train.log beside the model
        // A BARE filename (no directory component, e.g. "myrun" instead of "myrun/model") resolves
        // meta_dir to empty -- every meta_dir-gated side effect below (train.log's own set_file tee,
        // config.json/state.json/blend_schedule.json writes) then silently no-ops. The run itself is
        // unaffected (model.bin + checkpoints still save fine), but the result is invisible to `models`,
        // loses its recorded recipe, and -- for a content-embed/scratch run specifically -- can't be
        // regenerated with the right interceptor later without hand-reconstructing config.json from
        // memory (hit for real: see project history around 2026-07-16's ce256_prod_fixed relaunch,
        // which needed exactly that manual reconstruction afterward). Warn loudly instead of letting it
        // pass silently -- this is a one-line, easy-to-miss CLI mistake, not a design choice to protect.
        if (meta_dir.empty())
            sub0::log::warn("train: '{}' has no directory component -- config.json/state.json/"
                            "blend_schedule.json will NOT be written (only model.bin + checkpoints save). "
                            "This model won't show up in `models`, won't carry its recipe on resume, and "
                            "can't auto-enable content-embed/scratch on `gen` later. Use a directory-style "
                            "path instead (e.g. '{}/model') to get the full model-directory layout.",
                            model_path, model_path);
        // Two more explicit-path footguns, both hit for real this session (project memory
        // blended-scratch-op-capstone-validated): an explicit path whose directory component
        // resolves to the shared models ROOT itself, or to a directory another model already
        // occupies. Either way config.json/state.json/blend_schedule.json/corpus.tok/tokenizer.tok --
        // all written per-DIRECTORY, not per-basename -- get silently shared/overwritten between
        // unrelated model names: launch model A here, then model B here too (different basename,
        // same directory), and B's run reads/writes A's config.json and pinned blend_schedule.json.
        // The real incident this closes: a 3-source --blend-config silently trained only 2 sources
        // because an earlier smoke test's 2-source schedule was still pinned in a directory a second,
        // differently-named run also pointed at -- no error, no warning, just quietly wrong training
        // data until someone reads the log closely enough to notice a missing "blend: '...'" line.
        // Unlike the bare-filename case above (an unusual but legitimate choice to skip the directory
        // layout entirely), there is no legitimate reason for two different model names to share a
        // directory -- refuse outright rather than warn, matching SingleInstanceGuard's own precedent
        // of refusing an unambiguous conflict rather than merely logging it.
        if (!meta_dir.empty()) {
            std::error_code ec;
            const std::filesystem::path canon_meta = std::filesystem::weakly_canonical(meta_dir, ec);
            const std::filesystem::path canon_root =
                ec ? std::filesystem::path() : std::filesystem::weakly_canonical(
                                                    std::filesystem::path(SUB0_MODELS_ROOT), ec);
            if (!ec && canon_meta == canon_root) {
                sub0::log::error(
                    "train: '{}' resolves to the shared models root itself ('{}'), not a dedicated "
                    "model directory -- every model needs its OWN subdirectory (e.g. '{}/{}/model') "
                    "so its config.json/blend_schedule.json/corpus.tok aren't silently shared with "
                    "every other model. Refusing to start.",
                    model_path, SUB0_MODELS_ROOT, SUB0_MODELS_ROOT,
                    std::filesystem::path(model_path).stem().string());
                return 1;
            }
            const std::string our_stem = std::filesystem::path(model_path).filename().string();
            std::string foreign;
            for (const auto& de : std::filesystem::directory_iterator(meta_dir, ec)) {
                if (ec) break;
                const std::string name = de.path().filename().string();
                if (name.compare(0, our_stem.size(), our_stem) == 0) continue;   // ours (weights/.stepN.ckpt)
                if (name == "config.json" || name == "state.json" || name == "blend_schedule.json" ||
                    name == "train.log" || name == "corpus.tok" || name == "tokenizer.tok") continue;
                if (name.find(".ckpt") != std::string::npos) { foreign = name; break; }   // unambiguous
            }
            if (!foreign.empty()) {
                sub0::log::error(
                    "train: '{}' already holds a DIFFERENT model's checkpoint ('{}' in '{}') -- "
                    "starting here would silently share/overwrite that model's config.json/"
                    "blend_schedule.json/corpus.tok. Point at a dedicated subdirectory instead.",
                    model_path, foreign, meta_dir.string());
                return 1;
            }
        }
    } else {
        // No explicit path: find the most recent model dir with the SAME corpus + architecture
        // (registry::compatible() -- dims/flags, deliberately NOT the git SHA, see registry.hpp's
        // own doc comment) and decide whether to resume it or start a fresh, dated one. Architecture
        // match is the real "can these weights load" gate; whether the code version changed since
        // is a separate, softer question this block resolves via --resume/--fresh when ambiguous.
        namespace reg = sub0::registry;
        const std::string corpus_key = reg::corpus_tag(sub0::default_corpus());
        const std::vector<reg::ModelMeta> candidates = reg::scan(SUB0_MODELS_ROOT);
        const reg::ModelMeta* best = nullptr;
        for (const reg::ModelMeta& m : candidates) {
            if (m.corpus != corpus_key) continue;
            if (!reg::compatible(m, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                                 static_cast<int>(USE_TERNARY), static_cast<int>(POS_ENCODING),
                                 static_cast<int>(USE_GATED_FFN), static_cast<int>(USE_TIED_EMBEDDINGS),
                                 static_cast<int>(USE_QK_NORM), sub0::MODEL_ARCH_ID))
                continue;
            if (!best || m.updated > best->updated) best = &m;   // fixed-width ISO -> lexicographic = chronological
        }

        if (resume_mode == 1 && !best) {   // --resume with nothing to resume
            sub0::log::error("train: --resume requested but no existing model dir matches this "
                             "build's corpus+architecture yet.");
            return 1;
        }
        bool do_resume = false;
        if (best && resume_mode != 2) {   // resume_mode 2 = --fresh always skips this entirely
            const bool in_progress = best->status == "training";   // the only non-terminal status
            const bool sha_matches = best->git_sha == SUB0_GIT_SHA;
            if (resume_mode == 1) {
                do_resume = true;
            } else if (in_progress && sha_matches) {
                do_resume = true;   // unambiguous: same code, run didn't reach a clean stop
            } else if (in_progress && !sha_matches) {
                sub0::log::error(
                    "train: '{}' looks unfinished (status=training) but was trained by a different "
                    "code version ({} vs this build's {}) -- pass --resume to continue it anyway, or "
                    "--fresh to start a new dated model instead.",
                    best->dir.string(), best->git_sha.empty() ? "nogit" : best->git_sha, SUB0_GIT_SHA);
                return 1;
            } else {
                sub0::log::line("train: previous matching run '{}' already {} -- starting a new "
                                "dated model (pass --resume to continue it instead)",
                                best->dir.filename().string(), best->status);
            }
        }

        meta_dir = do_resume ? best->dir
                             : reg::model_dir(SUB0_MODELS_ROOT, sub0::default_corpus(),
                                              D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                                              static_cast<int>(USE_TERNARY),
                                              static_cast<int>(POS_ENCODING), reg::now_datetag(),
                                              static_cast<int>(USE_GATED_FFN),
                                              static_cast<int>(USE_TIED_EMBEDDINGS),
                                              static_cast<int>(USE_QK_NORM), sub0::MODEL_ARCH_ID);
        model_path = (meta_dir / "model.bin").string();
    }
    // Both guards are taken before any directory/log/GPU work, so a rejected launch leaves no side
    // effects. Per-directory FIRST: "you are racing your own checkpoint" is the more specific and more
    // actionable message when both would fire.
    const SingleInstanceGuard _dir_guard(meta_dir);
    if (!_dir_guard.held()) {
        sub0::log::error("train: another sub0llm-train is already running against '{}' -- refusing to "
                         "start a second one (they would race on the same checkpoint)", meta_dir.string());
        return 1;
    }
    // Global guard: default-deny. A second trainer anywhere shares this machine's GPU and page cache --
    // see SingleInstanceGuard's comment for why measurement correctness makes that a hard error rather
    // than a warning. Under --allow-concurrent the guard is constructed on the model dir instead, so it
    // is a no-op second acquire of a mutex this process already holds.
    const SingleInstanceGuard _global_guard(allow_concurrent ? std::filesystem::path{meta_dir}
                                                             : std::filesystem::path{});
    if (!allow_concurrent && !_global_guard.held()) {
        sub0::log::error("train: another sub0llm-train is already running on this machine -- refusing to "
                         "start a second one. Concurrent trainers share one GPU and one page cache: they "
                         "halve each other's throughput and make any measurement taken while both run "
                         "meaningless, with nothing in either log to show it. Wait for it to finish, or "
                         "pass --allow-concurrent if you deliberately want both (e.g. a CPU sample-train "
                         "beside a GPU run) and are not measuring.");
        return 1;
    }
    // Create the model directory up front (an explicit --out path may name a directory that doesn't
    // exist yet -- previously only the auto-derived branch above did this, so train.log's set_file
    // silently failed to open for a fresh explicit path; the pinning below has the same requirement).
    if (!meta_dir.empty()) { std::error_code ec; std::filesystem::create_directories(meta_dir, ec); }
    // Tee this run's progress into <model_dir>/train.log (append: a resume continues the same log),
    // so a long/background run keeps its own trajectory next to the weights + state.json -- set up
    // before the tokenizer/corpus loading below so a startup failure there is captured too.
    if (!meta_dir.empty()) sub0::log::set_file((meta_dir / "train.log").string());
    sub0::log::line("model dir: {}", model_path);

    // Training consumes the pre-tokenized corpus produced by the configurator, not
    // raw text. Any argument that is not a .tok file (including the driver's default
    // .txt corpus path) is intentionally ignored in favour of this build's baked-in
    // corpus.tok: the engine's VOCAB is fixed at configure time, so only the .tok the
    // configurator emitted for it can be trained against. To train on different text,
    // reconfigure/rebuild so the configurator retokenizes and re-bakes VOCAB.
    //
    // Both the tokenizer and the resolved corpus.tok are PINNED into the model directory (hardlink;
    // see bundle_into_model_dir) before this run reads either one, so the model dir is self-contained
    // and the whole run -- including every model.bin fingerprint stamped along the way -- is immune
    // to a later `configure` (for another corpus, or a plain rebuild) changing the live build-tree
    // copies out from under it. Without this, a stray build in the SAME directory (even one that
    // doesn't touch the configurator) can cascade into a re-tokenize that clobbers the very corpus.tok
    // this run has open.
    std::string tok_path = corpus_path ? corpus_path : "";
    if (tok_path.size() < 4 || tok_path.compare(tok_path.size() - 4, 4, ".tok") != 0)
        tok_path = sub0::default_corpus_tok();
    std::string tok_tokenizer_path = sub0::default_tokenizer();
    if (!meta_dir.empty()) {
        tok_tokenizer_path = bundle_into_model_dir(tok_tokenizer_path, meta_dir / "tokenizer.tok",
                                                    1ull << 30).string();               // small: always copy-able
        tok_path = bundle_into_model_dir(tok_path, meta_dir / "corpus.tok",
                                         1ull << 30).string();                          // may be many GB: hardlink only
    }

    // Not fatal to training itself (corpus.tok is pre-tokenized; the loss loop below never calls
    // encode/decode) -- only the periodic preview prints would come out garbled -- so warn and
    // continue rather than aborting a run that would otherwise train correctly.
    if (!sub0::load_tokenizer(tok_tokenizer_path.c_str()))
        sub0::log::warn("train: cannot load tokenizer '{}' -- training will proceed, but preview "
                        "samples will be garbled", tok_tokenizer_path);
    sub0::build_model();

    // Where training windows come from. train_span is sampled for training windows; val_span
    // is the fixed held-out eval region. Backed either by the corpus.tok memory map, or --
    // when no corpus.tok was built (--corpus-pretok 0) -- by bounded in-memory buffers tokenized
    // on demand from the raw corpus (see TextCorpus): no token copy on disk, RAM capped by
    // the buffers. The rotating training buffer is refilled each interval inside the loop.
    sub0::TokMap tok(tok_path);
    const bool on_demand = !tok.ok();

    sub0::TokView train_span, val_span;
    std::span<const std::uint64_t> doc_index;  // document-start token indices (empty => flat sampling)
    std::vector<int> train_buf, val_buf;       // on-demand backing storage
    TextCorpus text;                           // on-demand raw-corpus reader
    std::mt19937 buf_rng;                      // separate stream: refilling never perturbs `rng`
    std::size_t train_byte_lo = 0, train_byte_hi = 0;
    long refresh_n = 0;
    std::size_t est_train_tokens = 0;
    std::string src_desc;

    if (!on_demand) {
        // The .tok stream carries token ids in [0, VOCAB). The configurator bakes the matching
        // VOCAB into the engine, so within a build they always agree -- but a stale, hand-built,
        // or foreign .tok would index the embedding table out of bounds. Reject it up front.
        if (tok.vocab() != VOCAB) {
            sub0::log::error(                         "train: '{}' was tokenized for vocab {} but this engine was built for VOCAB {}.\n"
                         "       Reconfigure/rebuild against this corpus, or pass a matching .tok.",
                         tok_path, tok.vocab(), VOCAB);
            return 1;
        }
        const sub0::TokView data = tok.tokens();
        const std::size_t min_tokens = 2 * (static_cast<std::size_t>(SEQ_LEN) + 2);
        if (data.size() < min_tokens) {
            sub0::log::error("train: '{}' has only {} tokens, too few for seq_len {} (need >= {})",
                         tok_path, data.size(), SEQ_LEN, min_tokens);
            return 1;
        }
        // Defense in depth: a corrupt file could hold an out-of-range id. One linear scan
        // turns a silent OOB gather into a clear diagnostic (and faults in the mapping once,
        // but training's random windows touch most of it across epochs anyway).
        for (std::size_t i = 0; i < data.size(); ++i)
            if (data[i] < 0 || data[i] >= VOCAB) {
                sub0::log::error("train: '{}' token {} = {} is out of range [0,{})",
                             tok_path, i, data[i], VOCAB);
                return 1;
            }
        // Hold out the tail for validation; train only on the head so NELBO is honest.
        const std::size_t val_tokens = std::max<std::size_t>(
            static_cast<std::size_t>(SEQ_LEN) + 2,
            static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(data.size())));
        const std::size_t val_start = data.size() - val_tokens;
        train_span = data.first(val_start);
        val_span   = data.subspan(val_start);
        doc_index  = tok.doc_starts();   // boundary-aware sampling when the corpus.tok carries it
        est_train_tokens = val_start;
        src_desc = std::format("{} ({} tokens; {} train / {} val)", tok_path, data.size(), val_start, val_tokens);
    } else {
        // No corpus.tok: tokenize on demand from the raw text corpus. Split the corpus by
        // BYTES; tokenize a fixed validation buffer once and the first shuffle buffer now.
        const char* raw = sub0::default_corpus();
        if (!text.open(raw)) {
            sub0::log::error("train: no corpus.tok ('{}') and cannot open raw corpus '{}'", tok_path, raw);
            return 1;
        }
        const std::size_t total = text.bytes();
        const std::size_t val_bytes = std::max<std::size_t>(
            1, static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(total)));
        const std::size_t val_byte_start = total - val_bytes;
        train_byte_lo = 0; train_byte_hi = val_byte_start;

        text.fill_sequential(val_byte_start, OD_VAL_BUF_TOK, val_buf);
        buf_rng.seed(static_cast<std::uint32_t>(seed) ^ 0x9E3779B9u);
        text.fill_random(train_byte_lo, train_byte_hi, OD_TRAIN_BUF_TOK, buf_rng, train_buf);
        const std::size_t need = static_cast<std::size_t>(SEQ_LEN) + 2;
        if (val_buf.size() < need || train_buf.size() < need) {
            sub0::log::error("train: on-demand corpus '{}' too small to tokenize a window (need >= {} tokens)",
                         raw, need);
            return 1;
        }
        train_span = sub0::TokView::over_int32(train_buf.data(), train_buf.size());
        val_span   = sub0::TokView::over_int32(val_buf.data(), val_buf.size());
        // Estimate tokens/byte to size the epoch schedule (heuristic only). Sampled from several
        // regions spread across the corpus rather than just the first one: token density is not
        // uniform (FineWeb-style corpora cluster document types), so a single region at offset 0
        // is a noisy, potentially biased sample -- this was measured to undershoot the true count
        // by ~4-5% on a real run. Averaging OD_TPB_SAMPLES evenly-spaced regions still reads only a
        // few hundred KB total (trivial next to a multi-GB corpus) but is far more representative.
        constexpr int OD_TPB_SAMPLES = 8;
        std::vector<int> probe;
        double tpb_sum = 0.0;
        int    tpb_n   = 0;
        for (int i = 0; i < OD_TPB_SAMPLES; ++i) {
            const std::size_t off = (train_byte_hi * static_cast<std::size_t>(i)) / OD_TPB_SAMPLES;
            probe.clear();
            text.encode_region(off, probe);
            if (!probe.empty()) {
                tpb_sum += static_cast<double>(probe.size()) / static_cast<double>(OD_REGION_BYTES);
                ++tpb_n;
            }
        }
        const double tpb = tpb_n > 0 ? tpb_sum / tpb_n : 0.5;
        est_train_tokens = static_cast<std::size_t>(static_cast<double>(train_byte_hi) * tpb);
        src_desc = std::format("{} (on-demand; ~{} train tok est / {}-tok shuffle buf / {}-tok val buf)",
                               raw, est_train_tokens, train_buf.size(), val_buf.size());
    }

    std::mt19937 rng(seed);
    RunState rs;
    // tokens_seen lives in RunState (rs.tokens_seen) so the checkpoint carries the MEASURED count
    // across resumes. It accumulates exactly batch_t*seq_t per step -- never reconstructed from
    // step*batch*SEQ_LEN, which assumed every historical step ran a full-length window at the current
    // batch and inflated a resumed run's reported tokens and epochs against an unresumed one at the
    // same step count. See CKPT_VERSION v6.

    // The single source of truth for this run's training recipe (see registry.hpp's RunConfig doc
    // comment) -- built fresh from live state each time so it always reflects whatever `optimizer`/
    // `batch`/`lr`/`seed` currently ARE (which the resume-reconciliation block below may have just
    // adjusted), never a stale snapshot. It is the sole writer of the architecture and the recipe:
    // `write_state` below records PROVENANCE and PROGRESS only, and `read_state` merges the recipe
    // back in from config.json -- so there is exactly one writer per field and nothing to drift.
    auto build_run_config = [&] {
        sub0::registry::RunConfig cfg;
        cfg.corpus = sub0::registry::corpus_tag(sub0::default_corpus());
        cfg.corpus_fraction = corpus_fraction;   // which SUBSET of the corpus this model actually saw
        cfg.subset_seed     = subset_seed;
        cfg.d_model = D_MODEL; cfg.n_layers = N_LAYERS; cfg.n_heads = N_HEADS; cfg.n_kv_heads = N_KV_HEADS;
        cfg.loop_middle = LOOP_MIDDLE_LAYERS; cfg.loop_repeats = LOOP_REPEATS;
        cfg.depth_attn_stride = DEPTH_ATTN_STRIDE;
        cfg.rope_scaling = ROPE_SCALING; cfg.rope_scale_fac = ROPE_SCALE_FACTOR;
        cfg.rope_theta = ROPE_THETA;   // changes what the model computes -- layout.hpp ARCH_FINGERPRINT
        cfg.seq_len = SEQ_LEN; cfg.vocab = VOCAB; cfg.ternary = static_cast<int>(USE_TERNARY);
        cfg.pos_encoding = static_cast<int>(POS_ENCODING);
        cfg.gated_ffn = static_cast<int>(USE_GATED_FFN);
        cfg.tied_embeddings = static_cast<int>(USE_TIED_EMBEDDINGS);
        cfg.qk_norm = static_cast<int>(USE_QK_NORM);
        cfg.optimizer = (optimizer == 1) ? 1 : 0;   // matches AdamW opt(lr, optimizer==1) below verbatim
        cfg.batch = batch; cfg.lr = lr; cfg.seed = seed;
        // The blend recipe itself lives in the separately-pinned blend_schedule.json (see
        // sub0/blend_schedule.hpp), not here -- config_schema=2 is the signal that this model dir uses
        // that scheme (a config_schema<2 dir predates it and is refused on resume, not silently
        // reinterpreted -- see the resume-reconciliation block below).
        // Whether this run pinned a blend_schedule.json at all. A plain single-corpus run does not,
        // and its ABSENCE is then the expected state -- not something to warn about. Without this,
        // gen/eval could not tell "never had one" from "had one, now unreadable" and warned on every
        // plain model, training readers to ignore the message that matters.
        cfg.has_blend_schedule = std::filesystem::exists(meta_dir / "blend_schedule.json") ? 1 : 0;
        cfg.config_schema = 2;
        return cfg;
    };

    auto write_state = [&](const char* status) {
        if (meta_dir.empty()) return;
        // config.json first: it is the architecture + recipe, and state.json's arch_id below is only
        // meaningful alongside it. Nothing here re-states a config field -- read_state merges them.
        sub0::registry::write_config_json(build_run_config(), meta_dir);
        sub0::registry::ModelMeta m;
        m.arch_id = sub0::MODEL_ARCH_ID;   // full architecture identity; see registry::compatible
        m.git_sha = SUB0_GIT_SHA; m.created = created;
        m.updated = sub0::registry::now_iso();
        m.steps = rs.step;
        m.epochs = static_cast<double>(rs.step) / static_cast<double>(epoch_steps);
        m.tokens_seen = rs.tokens_seen;
        m.best_val_nelbo = (rs.best_loss < std::numeric_limits<double>::infinity()) ? rs.best_loss : -1.0;
        m.status = status;
        sub0::registry::write_state(meta_dir, m);
    };
    // Resume from the newest checkpoint for this output (progress-named, or a legacy single file):
    // overwrites the fresh model and restores batch/lr/seed, so the schedule below is computed from
    // the resumed batch, not the command line. Snapshot what THIS invocation asked for (the CLI's
    // resolved --batch/--lr/--seed, e.g. DEFAULT_GPU_BATCH when --batch was left at its auto default)
    // before load_checkpoint overwrites them, so a silent override is at least made visible below --
    // previously a resumed run could differ from what was requested/tuned this time with zero log
    // line explaining why, which read as "the binary must not have rebuilt" rather than "this is an
    // intentional resume".
    const int requested_batch = batch;
    const float requested_lr = lr;
    const unsigned requested_seed = seed;
    const std::string resume_ckpt = latest_ckpt_path(model_path);
    long adam_t = 0;
    const bool ckpt_existed = !resume_ckpt.empty();
    const bool resumed = ckpt_existed && load_checkpoint(resume_ckpt, rng, rs, batch, lr, seed, adam_t);
    if (ckpt_existed && !resumed) {          // a real checkpoint this build can't read -- don't clobber it
        sub0::log::error("checkpoint '{}' exists but is incompatible with this build; refusing to overwrite "
                         "it. Remove it (or train to a fresh path) to start over.", resume_ckpt);
        return 1;
    }
    if (resumed) {                           // make the resume UNMISSABLE (not buried in the schedule line)
        sub0::log::info("RESUMING from step {} (best val_nelbo {:.4f}) -- continuing, NOT starting fresh",
                        rs.step, rs.best_loss);
        if (batch != requested_batch || lr != requested_lr || seed != requested_seed)
            sub0::log::info("  checkpoint overrides this invocation's settings: batch {} -> {}, "
                            "lr {:.2e} -> {:.2e}, seed {} -> {} (the checkpoint's values win on resume)",
                            requested_batch, batch, requested_lr, lr, requested_seed, seed);
        // VRAM re-validation against a possibly-different card/driver/GPU-load than whenever this
        // batch was tuned/written happens inside GpuTrainer::enable() below (optimistic-allocate,
        // fall back to a conservative estimate only if the real allocation genuinely fails) -- not
        // here as a pre-emptive prediction, which risked clamping a batch that would have actually
        // fit fine (the same "wasted capacity" question a live VRAM measurement can over-trigger on).
    }
    // Carry the MEASURED cumulative token count across the resume, from the state.json this model
    // directory already writes at every save. It used to be reconstructed here as
    // rs.step * batch * SEQ_LEN, which silently assumed every historical step ran a full-length window
    // at the batch THIS invocation settled on -- false whenever seq length varies (the token-budget
    // scheduler's normal mode) or the batch differs, and it made a resumed run's reported tokens and
    // epochs incomparable with an unresumed one at the same step count. Reading the recorded value
    // needs no checkpoint format change, so existing .ckpt files stay resumable.
    //
    // Best-effort by design: state.json tracks the LATEST save, so resuming from a deliberately older
    // checkpoint carries a count from slightly further along. That is bounded by one save interval and
    // still far closer than the old reconstruction; a fresh run has no state.json and starts at 0.
    if (rs.step > 0) {
        sub0::registry::ModelMeta prev;
        if (sub0::registry::read_state(meta_dir, prev) && prev.tokens_seen > 0)
            rs.tokens_seen = prev.tokens_seen;
    }

    // Config-recipe reconciliation: unlike batch/lr/seed above, `optimizer` has no home in the
    // `.ckpt` binary and was previously derived PURELY from this invocation's CLI flag on every
    // resume, with nothing to cross-check it against -- the exact gap that once let a GPU-fault
    // auto-resume silently continue a Muon-trained run under plain AdamW instead (see
    // registry.hpp's RunConfig doc comment for the full incident). If this model directory already
    // has a persisted config.json (written by an earlier run under this fix), its `optimizer` is
    // now authoritative on resume too, same discipline as batch/lr/seed just above. A directory
    // predating this fix has no config.json yet -- `read_config_json` returns false and this
    // invocation's own value is used unchanged, same as before (no regression for old models; they
    // gain the protection the moment they're next saved under this build).
    if (resumed && !meta_dir.empty()) {
        sub0::registry::RunConfig persisted;
        if (sub0::registry::read_config_json(persisted, meta_dir)) {
            // config_schema<2 means this model dir predates the blend-schedule redesign: its
            // config.json still carries the now-removed spell_mix/scratch_mix/op_mix/content_embed/
            // content_embed_kind fields (silently ignored by this build's RunConfig, per registry.hpp's
            // own forward-compatible-unrecognized-key tolerance), which is exactly the danger -- a
            // model genuinely trained with content-embed active would otherwise resume and decode
            // WITHOUT its interceptor, a confidently wrong result. Refuse loudly instead of guessing.
            if (persisted.config_schema < 2) {
                sub0::log::error("train: '{}' predates the blend-schedule redesign (config_schema {}) -- "
                                 "refusing to resume it with this build. Re-train from scratch with this "
                                 "binary, or use an older build to continue this specific model.",
                                 meta_dir.string(), persisted.config_schema);
                return 1;
            }
            if (persisted.optimizer != optimizer) {
                sub0::log::info("  config.json overrides this invocation's optimizer: {} -> {} "
                                "(the persisted recipe wins on resume, same as batch/lr/seed above)",
                                optimizer == 1 ? "muon" : "adamw", persisted.optimizer == 1 ? "muon" : "adamw");
                optimizer = persisted.optimizer;
            }
        }
    }

    // --- Blend schedule: parse (or synthesize the trivial single-source case) then pin into the model
    // dir. See sub0/blend_schedule.hpp's header comment for why this replaced --spell-mix/--scratch-mix/
    // --op-mix/--content-embed/--gsm8k: those meant "this fraction of every step's windows," with no
    // tracking of how much of a source had actually been covered -- a small curriculum blended against
    // a huge corpus got re-covered dozens of times before the corpus finished one epoch, producing a
    // real, observed loss cliff-then-plateau (project memory: the incident that motivated this file).
    //
    // Resume semantics are STRICTER than bundle_into_model_dir's tokenizer.tok/corpus.tok pinning above
    // (which re-syncs from the live path on every invocation): a schedule identity change mid-run
    // literally changes what data trains the model, so the default here is to read ONLY the existing
    // pin and ignore a live --blend-config path entirely, warning (never silently switching) if one was
    // given anyway. --blend-config-replace is the separate, explicitly-named override for the real but
    // deliberate workflow of continuing a run under a genuinely different schedule.
    sub0::ScheduleSpec schedule;
    const bool have_blend_config = blend_config_path && *blend_config_path;
    {
        std::filesystem::path schedule_path;   // what we actually parse below; empty => synthesize sugar
        if (!meta_dir.empty()) {
            const std::filesystem::path pin = meta_dir / "blend_schedule.json";
            const bool pin_exists = std::filesystem::exists(pin);
            if (have_blend_config && (!pin_exists || replace_schedule)) {
                if (replace_schedule && pin_exists)
                    sub0::log::info("  SCHEDULE CHANGED (--blend-config-replace): re-pinning '{}' over "
                                    "the previous schedule -- per-source progress carries forward BY "
                                    "NAME, a new source starts at 0, a removed source is dropped",
                                    blend_config_path);
                schedule_path = bundle_into_model_dir(std::filesystem::path(blend_config_path), pin,
                                                      1ull << 20);   // schedules are tiny: always copy-able
            } else if (pin_exists) {
                if (have_blend_config)
                    sub0::log::warn("train: --blend-config given but '{}' already has a pinned schedule "
                                    "-- using the PINNED one unchanged (pass --blend-config-replace to "
                                    "deliberately switch schedules on this resume)", meta_dir.string());
                schedule_path = pin;
            }
        } else if (have_blend_config) {
            schedule_path = blend_config_path;   // no model dir to pin into -- read the live path directly
        }

        if (!schedule_path.empty()) {
            std::string error; std::vector<std::string> warnings;
            if (!sub0::parse_blend_schedule_json(schedule_path, schedule, error, warnings)) {
                sub0::log::error("train: blend schedule '{}' is unreadable: {}", schedule_path.string(), error);
                return 1;
            }
            for (const std::string& w : warnings) sub0::log::warn("train: blend schedule: {}", w);
        } else {
            // Sugar for the common case: no --blend-config at all -- a single-source schedule wrapping
            // the base corpus, equivalent to "just train on this corpus." Fully reproducible from
            // corpus_path alone (no separate identity to protect), so nothing needs pinning here.
            sub0::SourceSpec base_spec;
            base_spec.name = "base";
            base_spec.corpus = corpus_path ? corpus_path : sub0::default_corpus_tok();
            schedule.sources.push_back(std::move(base_spec));
            sub0::ScheduleStage only_stage;
            only_stage.until_epoch = std::numeric_limits<double>::infinity();
            only_stage.weights.emplace_back("base", 1.0);
            schedule.stages.push_back(std::move(only_stage));
        }
    }
    // content_embed needs at least one schedule source backed by a generator that records structured
    // slot->fragment binding data (scratchspike/op_curriculum -- spellspike and the base corpus carry
    // no such table, so their windows always pass null bindings, unchanged plain reserved-id embedding).
    // "Ever" (any stage, not just the currently-active one): a schedule can legitimately bootstrap a
    // generator hard in an early stage then taper it to zero later (curriculum-learning-style), and the
    // model still needs its content-embed interceptor at gen time for what it learned in that stage --
    // see gen_stage.cpp's own union-across-stages rewiring for the matching gen-time concern.
    const bool have_binding_source = std::any_of(schedule.sources.begin(), schedule.sources.end(),
        [](const sub0::SourceSpec& s) { return s.generator == "scratchspike" || s.generator == "op_curriculum"
                                             || s.generator == "wordspike" || s.generator == "corpus_collapse"; });
    bool content_embed_active = schedule.content_embed.has_value() && have_binding_source;
    if (schedule.content_embed && !have_binding_source)
        sub0::log::warn("train: blend schedule sets content_embed but declares no scratchspike/"
                        "op_curriculum/wordspike/corpus_collapse source to bind from -- disabled");
    const int content_embed_kind = content_embed_active
        ? static_cast<int>(*schedule.content_embed) : static_cast<int>(sub0::ContentEmbedKind::MeanPool);

    sub0::AdamW opt(lr, optimizer == 1);
    if (resumed) opt.set_step_count(adam_t);

    // Phase 2e: try to run the training loop on the GPU (params resident on device). Falls back
    // to the CPU data-parallel path when the CUDA backend is absent, no device initializes, or the
    // batch exceeds the device's resident scratch width. Muon (--optimizer muon) is honored on
    // BOTH paths: the GPU path routes the Muon-eligible weight matrices through its own device
    // Newton-Schulz pipeline (backend_cuda.cu's device_adam_step / muon_step_matrix), the same
    // hybrid dispatch shape as the CPU's AdamW::step -- see the per-step call site below for the
    // muon_lr computation. (Previously this silently fell back to pure AdamW on GPU with a warning;
    // GPU Muon support removed the need for that fallback entirely.)
    GpuTrainer gpu;
    // content_embed's binding lookup (encode_slot) forces the WINDOWS that actually draw from a
    // content-embed source (scratch-mix/op-mix) onto the CPU UNLESS the device backend can compose
    // bound embedding rows itself (sub0_dev_caps().supports_binding_compose -- docs/BACKENDS.md
    // "Design: binding-compose on CUDA"; dev_binding_compose below) -- every other window in the same
    // blend (base corpus, spell-mix) has no such need regardless. gpu_available still enables the
    // device session so those windows can route to it: see hybrid_train below (the source-routed
    // split; algorithm proven in tests/cuda_tests.cpp "CUDA+CPU hybrid split..."), which activates
    // whenever a GPU is available AND content_embed forces a CPU-side subset -- the two are not
    // mutually exclusive. When dev_binding_compose is true, that CPU-side subset is empty in practice
    // (content_embed_kind's persisted enum only ever selects a device-composable encoding -- see
    // ContentEmbedKind in scratch_slots.hpp), so hybrid_train's CPU sub-batch simply goes unused; the
    // capability check stays per-window (needs_cpu below) rather than collapsing hybrid_train back to
    // gpu_train so a future partial-capability backend degrades gracefully instead of miscomposing.
    const bool gpu_available = gpu.enable(batch, opt.step_count());
    const bool gpu_train      = !content_embed_active && gpu_available;   // pure GPU: no CPU-only source active
    const bool hybrid_train   =  content_embed_active && gpu_available;   // source-routed split (see above)
    gpu.hybrid = hybrid_train;
    // Cached once (a static per-backend fact, not a hot-loop query) -- see needs_cpu and
    // install_gpu_bindings below.
    const bool dev_binding_compose = sub0_dev_caps().supports_binding_compose != 0;

    // Corpus-relative schedule (max_steps is a per-invocation budget, never restored). For
    // on-demand the token count is estimated from tokens/byte (the schedule is heuristic and
    // plateau-stopped, so an estimate is fine).
    const std::size_t tokens_per_step = static_cast<std::size_t>(batch) * SEQ_LEN;
    // est_train_tokens is 64-bit: a large corpus (FineWeb ~= 12 B tokens) overflows a 32-bit `long`
    // (LLP64 Windows) -- casting it to long wrapped NEGATIVE, collapsing epoch_steps to 1 (so the run
    // stopped after MAX_EPOCHS_BACKSTOP *steps* and evaluated every step). Compute in size_t, then
    // narrow the (bounded, < 2^31) step count.
    // An EPOCH is one pass over the data actually being TRAINED ON, so --corpus-fraction has to scale
    // it. Without this the clock stayed keyed to the whole corpus: at --corpus-fraction 0.8 the run
    // reported "1 epoch" after drawing enough windows for the FULL corpus, i.e. after ~1.25 real
    // passes over the selected 80%. Everything downstream keys off epoch_steps -- eval warmup, eval
    // interval, LR warmup, the MAX_EPOCHS_BACKSTOP budget and the plateau logic -- so all of them were
    // stretched by 1/fraction, and the wall-clock-per-epoch a run is planned around was simply wrong.
    //
    // Scaling by the fraction rather than counting the selected tokens exactly is deliberate and
    // unbiased: doc_in_subset() selects on a hash of the document INDEX, independently of that
    // document's length, so the expected selected-token share is exactly `fraction`. Counting exactly
    // would need the document index here, which is not built until the sources are constructed below,
    // and the comment above already establishes this schedule as heuristic and plateau-stopped.
    const std::size_t epoch_tokens =
        (corpus_fraction >= 1.0)
            ? est_train_tokens
            : static_cast<std::size_t>(static_cast<double>(est_train_tokens) * corpus_fraction);
    epoch_steps  = static_cast<long>(std::max<std::size_t>(
        1, (epoch_tokens + tokens_per_step - 1) / tokens_per_step));
    const long warmup_steps = std::max<long>(1, std::lround(EVAL_WARMUP_EPOCHS  * epoch_steps));
    const long eval_every   = std::max<long>(1, std::lround(EVAL_INTERVAL_EPOCHS * epoch_steps));
    // Budget resolution, in priority order: an explicit --steps, then an explicit --epochs, then the
    // plateau detector with MAX_EPOCHS_BACKSTOP as its ceiling.
    //
    // --epochs is the better unit for a controlled comparison and is resolved HERE, the first point
    // where it can be exact: epoch_steps already accounts for the batch (tokens_per_step) and for
    // --corpus-fraction, so `epochs * epoch_steps` is a TOKEN budget expressed in whatever steps this
    // arm's batch happens to make of it. Two arms at the same --epochs therefore see the same number
    // of tokens even when they fit different batches -- which is precisely the property matched
    // --steps does NOT have, and how the first LoopSplit sweep silently ran one arm on 14.3% more data
    // (see loopsplit-3arm-batch-confound). Fractional is meaningful and expected: --epochs 1.8 is a
    // legitimate budget, not a rounding artifact.
    const long epoch_budget_steps =
        (epochs > 0.0) ? std::max<long>(1, std::lround(epochs * static_cast<double>(epoch_steps))) : 0;
    const long max_steps = (steps > 0)              ? static_cast<long>(steps)
                         : (epoch_budget_steps > 0) ? epoch_budget_steps
                                                    : static_cast<long>(MAX_EPOCHS_BACKSTOP) * epoch_steps;
    // An EXPLICIT budget (either unit) is a deliberate choice, so the plateau detector must not cut it
    // short -- that is what makes a matched-budget A/B matched. Only an unspecified budget is
    // plateau-stopped. `steps` alone used to carry this meaning; `--epochs` has to join it or an
    // epoch-budgeted arm would stop early and silently stop being comparable.
    const bool fixed_budget = (steps > 0) || (epoch_budget_steps > 0);
    const long lr_warmup_steps = std::max<long>(10, std::lround(LR_WARMUP_EPOCHS * epoch_steps));
    const float peak_lr = lr;   // `lr` is the peak; lr_schedule(step) warms up to it then decays

    std::print("corpus: {} | ", src_desc);
    report_run_context(gpu_train, hybrid_train);
    // Say which unit the budget came from AND what it costs in the other one, plus the token count --
    // the quantity two arms actually have to match. A reader comparing two runs should not have to
    // multiply batch by SEQ_LEN by steps in their head to find out whether the comparison was fair.
    sub0::log::line("schedule: {} steps/epoch | eval warmup {} | eval every {} | max {} steps "
                    "({:.2f} epochs, {:.2f}B tokens; budget from {}){}",
         epoch_steps, warmup_steps, eval_every, max_steps,
         static_cast<double>(max_steps) / static_cast<double>(epoch_steps),
         static_cast<double>(max_steps) * static_cast<double>(tokens_per_step) / 1e9,
         (steps > 0) ? "--steps" : (epoch_budget_steps > 0) ? "--epochs" : "plateau detection",
         on_demand ? " | on-demand" : "");   // the resume is announced prominently above, not buried here
    sub0::log::line("lr: peak {:.2e} (batch {}) | warmup {} steps -> inverse-sqrt decay",
         peak_lr, batch, lr_warmup_steps);
    if (opt.use_muon())
        sub0::log::line("optimizer: Muon (hidden 2D weight matrices) + AdamW (embeddings, head, norms, "
                        "biases) | muon lr peak {:.2e}", MUON_LR_BASE);
    std::fflush(stdout);
    // Register/refresh the model now that epoch_steps is known -- so state.json shows the correct step +
    // epoch immediately (a resume no longer looks like steps=0), and an interrupted run stays discoverable.
    write_state("training");

    // Token-budget scheduling (see `batch_t` in the loop below): a short-T step trades batch UP to
    // hold batch_t*seq_t roughly constant at tokens_per_step, so the window-indexed arrays here need
    // capacity for the LARGEST batch_t the scheduler can produce -- achieved at the smallest allowed
    // seq_t (MIN_TRAIN_SEQ). Using the exact same integer-division formula as the per-step
    // computation (tokens_per_step / seq_t) rather than an approximation keeps this bound exact, not
    // just "probably enough". Capped at MAX_DEVICE_BATCH, the same hard ceiling
    // sub0_dev_train_reserve/fwd_alloc/train_alloc enforce on the device side.
    const int max_batch_t = static_cast<int>(std::min<std::size_t>(
        tokens_per_step / static_cast<std::size_t>(MIN_TRAIN_SEQ),
        static_cast<std::size_t>(sub0::memplan::MAX_DEVICE_BATCH)));
    std::vector<size_t> starts(max_batch_t);
    std::vector<int>    win_len(max_batch_t);   // per-window trained length (< seq_t for short documents)
    std::vector<int>    src_idx(max_batch_t);   // per-window blend source index (all 0 for a single-source run)
    std::vector<int>    cpu_win;          // CPU path: materialized window tokens (view may be uint16-packed)
    std::vector<size_t> cpu_starts(max_batch_t);
    std::vector<std::uint8_t> cpu_mask;   // CPU path: per-window loss mask parallel to cpu_win (only when a source is masked)
    // content_embed: per-window content-derived scratch-slot bindings (windows from a binding-capable
    // schedule source only -- see content_embed_active's guard above). step_binds owns each window's
    // ScratchBindings VALUE (rebuilt fresh every step from its source's already-resident doc_bindings, so
    // the pointers in win_binds stay valid for exactly as long as train_batch needs them); win_binds is
    // what's actually passed through.
    std::vector<sub0::ScratchBindings> step_binds(static_cast<std::size_t>(max_batch_t));
    std::vector<const sub0::ScratchBindings*> win_binds(static_cast<std::size_t>(max_batch_t), nullptr);
    // hybrid_train (see gpu_available/gpu_train/hybrid_train above): each step partitions this step's
    // batch_t window indices into hyb_cpu_idx (content-embed sources -- CPU-only) and hyb_gpu_idx
    // (everything else -- GPU-eligible). All sized/reserved to max_batch_t up front so the per-step
    // .clear()+push_back() in the loop below never grow-reallocates once steady state is reached.
    std::vector<int> hyb_cpu_idx, hyb_gpu_idx;
    hyb_cpu_idx.reserve(static_cast<std::size_t>(max_batch_t));
    hyb_gpu_idx.reserve(static_cast<std::size_t>(max_batch_t));
    // Compacted (dense, 0-indexed) GPU sub-batch descriptors -- GpuTrainer::backward_only needs its OWN
    // batch's src_idx/starts/lengths dense from [0, gpu_n), the same shape gpu.step() already requires.
    std::vector<int> hyb_gpu_src(static_cast<std::size_t>(max_batch_t));
    std::vector<std::size_t> hyb_gpu_starts(static_cast<std::size_t>(max_batch_t));
    std::vector<int> hyb_gpu_len(static_cast<std::size_t>(max_batch_t));
    // Compacted CPU sub-batch trained-length array (win_len[b] at the CPU sub-batch's own positions --
    // cpu_win/cpu_starts/win_binds are already reused for this from the plain-CPU branch above).
    std::vector<int> hyb_cpu_len(static_cast<std::size_t>(max_batch_t));
    // GPU sub-batch's gradient, read back host-side by backward_only(); the CPU sub-batch's gradient
    // lands in sub0::grad_ptr() directly (train_batch's own destination) so only ONE extra buffer is
    // needed here, not two.
    std::vector<float> hyb_gpu_grad(hybrid_train ? sub0::trainable_floats() : std::size_t{0});
    // Binding-compose override table for the GPU sub-batch (see install_gpu_bindings below) --
    // populated only when dev_binding_compose is true; wire layout is device_backend.hpp's
    // sub0_dev_set_window_bindings contract (flat override_idx[gpu_n*seq_t], entries triples, a
    // concatenated frags array). Sized once so the per-step assign()/clear()+push_back() below never
    // grow-reallocates once steady state is reached, same reasoning as hyb_cpu_idx/hyb_gpu_idx above.
    std::vector<int> gpu_bind_idx, gpu_bind_entries, gpu_bind_frags;
    if (hybrid_train && dev_binding_compose) {
        gpu_bind_idx.reserve(static_cast<std::size_t>(max_batch_t) * SEQ_LEN);
        gpu_bind_entries.reserve(static_cast<std::size_t>(max_batch_t) * SEQ_LEN * SUB0_DEV_BIND_ENTRY_INTS);
        gpu_bind_frags.reserve(static_cast<std::size_t>(max_batch_t) * SEQ_LEN * 4);   // ~contains_k frags/slot
    }
    // Spawned once (not per-step -- see CpuSubBatchWorker's own comment) only when actually needed;
    // every other run leaves this empty and pays no worker-thread cost at all.
    std::optional<CpuSubBatchWorker> cpu_worker;
    if (hybrid_train) cpu_worker.emplace(cpu_win, cpu_starts, hyb_cpu_len, cpu_mask, win_binds);
    long steps_since_refresh = 0;

    // The blend: realize each schedule source into a real BlendSource. Generalizes the OLD --spell-mix/
    // --scratch-mix/--op-mix blocks (three near-identical blocks collapse into one generic loop) but
    // changes NOTHING about how a generator's own Dataset is built -- same tokenizer, same seed
    // derivation, same hardcoded defaults unless a schedule `params` override is given. Datasets must
    // outlive the training loop (the BlendSource objects only borrow spans into them); kept in parallel
    // vectors index-aligned with `sources`/`schedule.sources` so each survives regardless of which
    // sources are actually declared (the two unused-generator-kind vectors for any given index just
    // stay default-constructed/empty).
    std::vector<sub0::BlendSource> sources;
    sources.reserve(schedule.sources.size());
    std::vector<sub0::spellspike::Dataset>    spell_dss(schedule.sources.size());
    std::vector<sub0::scratchspike::Dataset>  scratch_dss(schedule.sources.size());
    std::vector<sub0::op_curriculum::Dataset> op_dss(schedule.sources.size());
    std::vector<sub0::wordspike::Dataset>     word_dss(schedule.sources.size());
    std::vector<sub0::corpus_collapse::Dataset> corpus_collapse_dss(schedule.sources.size());
    std::size_t kBaseSource = static_cast<std::size_t>(-1);   // sentinel: no corpus-typed source this run

    for (std::size_t i = 0; i < schedule.sources.size(); ++i) {
        const sub0::SourceSpec& spec = schedule.sources[i];
        if (!spec.corpus.empty()) {
            kBaseSource = i;
            // --corpus-fraction applies to the BASE corpus only -- the generated curricula below are
            // already small, and thinning one would delete coverage of the very mechanism it teaches.
            sub0::BlendSource base{ spec.name, train_span, doc_index, {}, {} };
            base.subset_fraction = corpus_fraction;
            base.subset_seed     = subset_seed;
            sources.push_back(base);
            if (corpus_fraction < 1.0) {
                // Documents, not tokens: selection is per-document, so the token count only lands on
                // the fraction in expectation. Report both so the log says what was actually chosen.
                const std::size_t ndocs = doc_index.empty() ? 0 : doc_index.size() - 1;
                std::size_t sel = 0;
                for (std::size_t d = 0; d < ndocs; ++d)
                    if (sub0::doc_in_subset(d, subset_seed, corpus_fraction)) ++sel;
                sub0::log::line("blend: '{}' base corpus ({} tokens) | SUBSET {:.1f}% of documents "
                                "(seed {}): {} of {} selected, distributed across the corpus",
                                spec.name, train_span.size(), corpus_fraction * 100.0,
                                subset_seed, sel, ndocs);
                if (sel == 0) {
                    sub0::log::error("train: --corpus-fraction {} selected NO documents from {} -- "
                                     "nothing to train on. Raise the fraction.", corpus_fraction, ndocs);
                    return 1;
                }
            } else {
                sub0::log::line("blend: '{}' base corpus ({} tokens)", spec.name, train_span.size());
            }
            continue;
        }
        sub0::tok::Tokenizer tk;
        std::ifstream tis(sub0::default_tokenizer(), std::ios::binary);
        if (!tis.good() || !sub0::tok::deserialize(tk, tis) || tk.vocab != VOCAB) {
            sub0::log::error("train: blend source '{}' (generator {}) needs the tokenizer '{}' but it "
                             "could not be loaded (or vocab != engine VOCAB {})", spec.name,
                             spec.generator, sub0::default_tokenizer(), VOCAB);
            return 1;
        }
        if (spec.generator == "spellspike") {
            // Drill EVERY eligible word (drilled_frac 1.0): production wants full vocab coverage, and
            // the spike proved the mechanism generalizes, so there is no need to hold words out here.
            const sub0::spellspike::WordSplit split =
                sub0::spellspike::split_task_words(tk, /*drilled_frac=*/1.0, /*seed=*/seed);
            sub0::spellspike::DatasetOptions dopt;
            dopt.tasks_per_word = spec.tasks_per_word > 0 ? spec.tasks_per_word : 12;
            dopt.seed = static_cast<std::uint64_t>(seed) ^ 0x5C0113B1EULL;   // distinct stream from window sampling
            spell_dss[i] = sub0::spellspike::build_dataset(tk, split, dopt);
            const sub0::spellspike::Dataset& ds = spell_dss[i];
            sources.push_back(sub0::BlendSource{ spec.name,
                sub0::TokView::over_int32(ds.tokens.data(), ds.tokens.size()),
                std::span<const std::uint64_t>(ds.doc_starts), std::span<const std::uint8_t>(ds.mask), {} });
            sub0::log::line("blend: '{}' uncombine curriculum ({} task words, {} traces, {} tokens)",
                            spec.name, split.drilled.size(), ds.doc_starts.size() - 1, ds.tokens.size());
        } else if (spec.generator == "scratchspike") {
            constexpr int kScratchK = sub0::scratchspike::SCRATCH_POOL;   // slots/context the curriculum exercises
            const sub0::scratchspike::OovSplit split = sub0::scratchspike::make_oov_split(
                tk, /*n_total=*/spec.n_oov > 0 ? spec.n_oov : 400, /*drilled_frac=*/1.0, /*seed=*/seed);
            sub0::scratchspike::DatasetOptions dopt;
            dopt.tasks_per_oov = spec.tasks_per_oov > 0 ? spec.tasks_per_oov : 12;
            dopt.seed = static_cast<std::uint64_t>(seed) ^ 0x5C2A7C40ULL;   // distinct stream
            // The content trace is LONG (~5+11*K), so contains_k is sized to SEQ_LEN; a too-short
            // context omits it. Capped at 4: the maximal-K sweep found localization holds above random
            // up to the pool's K=6 but the margin degrades/gets noisy past K~4. A schedule override
            // still can't exceed this derived ceiling -- see blend_schedule.hpp's own SourceSpec comment
            // on why safety clamps must survive exposing a knob.
            const int derived_contains_k = std::min({kScratchK, 4, std::max(0, (SEQ_LEN - 8) / 11)});
            const int contains_k = spec.contains_k >= 0
                ? std::min(spec.contains_k, derived_contains_k) : derived_contains_k;
            scratch_dss[i] = sub0::scratchspike::build_dataset_scratch(tk, split, kScratchK, dopt,
                                                                       contains_k >= 2 ? contains_k : 0);
            const sub0::scratchspike::Dataset& ds = scratch_dss[i];
            sources.push_back(sub0::BlendSource{ spec.name,
                sub0::TokView::over_int32(ds.tokens.data(), ds.tokens.size()),
                std::span<const std::uint64_t>(ds.doc_starts), std::span<const std::uint8_t>(ds.mask),
                std::span<const std::vector<std::vector<int>>>(ds.doc_bindings) });
            sub0::log::line("blend: '{}' scratch curriculum (K={} slots, content-K={}, {} OOVs, {} "
                            "traces, {} tokens)", spec.name, kScratchK, contains_k >= 2 ? contains_k : 0,
                            split.drilled.size(), ds.doc_starts.size() - 1, ds.tokens.size());
        } else if (spec.generator == "op_curriculum") {
            if (!spec.gsm8k_path.empty()) {
                // Real GSM8K: convert its <<expr=result>> annotations to delegated [op math] frames.
                op_dss[i] = sub0::op_curriculum::gsm8k_file_dataset(spec.gsm8k_path, tk);
                if (op_dss[i].doc_starts.size() <= 1) {
                    sub0::log::error("train: blend source '{}': gsm8k '{}' yielded no verifiable op "
                                     "problems (empty/wrong format?)", spec.name, spec.gsm8k_path);
                    return 1;
                }
                sub0::log::line("blend: '{}' GSM8K op-delegation ({} problems, {} tokens) <- {}",
                                spec.name, op_dss[i].doc_starts.size() - 1, op_dss[i].tokens.size(),
                                spec.gsm8k_path);
            } else {
                sub0::op_curriculum::Options oopt;
                oopt.seed = static_cast<std::uint64_t>(seed) ^ 0x0B5C11ED0FF1CEULL;   // distinct stream
                oopt.n_examples = spec.n_examples > 0 ? spec.n_examples : 6000;
                oopt.chain_frac = spec.chain_frac >= 0.0 ? spec.chain_frac : 0.4;   // fraction of multi-step collapse chains
                const int derived_max_digits = std::max(2, std::min(4, (SEQ_LEN - 12) / 8));
                oopt.max_digits = spec.max_digits > 0
                    ? std::min(spec.max_digits, derived_max_digits) : derived_max_digits;
                op_dss[i] = sub0::op_curriculum::build_dataset(tk, oopt);
                sub0::log::line("blend: '{}' op-delegation curriculum ({} examples, {} tokens, "
                                "chain-frac {:.2f})", spec.name, op_dss[i].doc_starts.size() - 1,
                                op_dss[i].tokens.size(), oopt.chain_frac);
            }
            const sub0::op_curriculum::Dataset& ds = op_dss[i];
            sources.push_back(sub0::BlendSource{ spec.name,
                sub0::TokView::over_int32(ds.tokens.data(), ds.tokens.size()),
                std::span<const std::uint64_t>(ds.doc_starts), std::span<const std::uint8_t>(ds.mask),
                std::span<const std::vector<std::vector<int>>>(ds.doc_bindings) });
        } else if (spec.generator == "wordspike") {
            sub0::wordspike::Options wopt;
            wopt.seed = static_cast<std::uint64_t>(seed) ^ 0x0740D5B11CEULL;   // distinct stream
            wopt.n_examples = spec.n_examples > 0 ? spec.n_examples : 3000;
            const int derived_max_digits = std::max(2, std::min(4, (SEQ_LEN - 12) / 8));   // same derivation as op_curriculum
            wopt.max_digits = spec.max_digits > 0
                ? std::min(spec.max_digits, derived_max_digits) : derived_max_digits;
            word_dss[i] = sub0::wordspike::build_dataset(tk, wopt, sub0::wordspike::Arm::Collapse);
            const sub0::wordspike::Dataset& ds = word_dss[i];
            sources.push_back(sub0::BlendSource{ spec.name,
                sub0::TokView::over_int32(ds.tokens.data(), ds.tokens.size()),
                std::span<const std::uint64_t>(ds.doc_starts), std::span<const std::uint8_t>(ds.mask),
                std::span<const std::vector<std::vector<int>>>(ds.doc_bindings) });
            sub0::log::line("blend: '{}' natural-prose word-collapse + op-delegation curriculum "
                            "({} examples, {} tokens)", spec.name, ds.doc_starts.size() - 1, ds.tokens.size());
        } else if (spec.generator == "corpus_collapse") {
            // Scales wordspike's proven mechanism to a sampled subset of REAL corpus documents (see
            // docs/CORPUS_COLLAPSE.md). Document boundaries are load-bearing (a fresh ScratchTable resets
            // per document) -- the on-demand/no-corpus.tok path (--corpus-pretok 0) never populates
            // `doc_index`, so this generator cannot degrade gracefully the way flat sampling does
            // elsewhere; fail loudly rather than silently build a source whose windows could sample past
            // an empty/zero-length view (a real crash risk in sample_window, not just a quality concern).
            if (doc_index.empty()) {
                sub0::log::error("train: blend source '{}' (corpus_collapse) needs a real corpus.tok with "
                                 "document boundaries -- rebuild with --corpus-pretok 1, or drop this "
                                 "source from the schedule", spec.name);
                return 1;
            }
            sub0::corpus_collapse::Options copt;
            copt.seed = static_cast<std::uint64_t>(seed) ^ 0xC0125C011A95EULL;   // distinct stream
            copt.n_docs = spec.n_examples > 0 ? spec.n_examples : 3000;
            copt.max_doc_tokens = SEQ_LEN - 2;   // whole document must fit one training window
            corpus_collapse_dss[i] = sub0::corpus_collapse::build_dataset(tk, train_span, doc_index,
                                                                           est_train_tokens, copt);
            const sub0::corpus_collapse::Dataset& ds = corpus_collapse_dss[i];
            sources.push_back(sub0::BlendSource{ spec.name,
                sub0::TokView::over_int32(ds.tokens.data(), ds.tokens.size()),
                std::span<const std::uint64_t>(ds.doc_starts), std::span<const std::uint8_t>(ds.mask),
                std::span<const std::vector<std::vector<int>>>(ds.doc_bindings) });
            sub0::log::line("blend: '{}' corpus-scale word-collapse curriculum ({} real documents "
                            "sampled, {} tokens)", spec.name, ds.doc_starts.size() - 1, ds.tokens.size());
        } else {
            sub0::log::error("train: blend source '{}' has unrecognized generator '{}'",
                             spec.name, spec.generator);
            return 1;
        }
    }
    if (content_embed_active)
        sub0::log::line("blend: content-derived scratch embeddings ON ({}, {})",
                        sub0::content_embed_kind_name(content_embed_kind),
                        hybrid_train ? "CPU for content-embed windows, GPU for the rest"
                                     : "CPU-forced");
    // Masked-source blending normalizes the loss/grad over the ACTIVE (non-ignored) positions on
    // whichever backend runs it: the CPU via train_batch's loss_mask, the GPU via the ce_backward
    // kernel's ignore-index path (targets < 0 inert + per-window active[] normalization). Both are
    // parity-gated (tests/cuda_tests.cpp "CUDA masked (ignore-index) backward..."), so a masked blend
    // runs on either backend.
    const bool any_masked = std::any_of(sources.begin(), sources.end(),
                                        [](const sub0::BlendSource& s) { return s.masked(); });

    // Resolve each stage's per-source target rate against the ACTUAL sources[] order built above (name
    // -> index lookups happen ONCE here, not in the per-window hot path), then track per-source
    // cumulative progress across the run -- the deficit scheduler's fairness state (see
    // sub0/blend_schedule.hpp's header comment for the algorithm). Seeded from the checkpoint BY NAME
    // (sub0::carry_forward_by_name, CKPT_VERSION 4): correct even across --blend-config-replace, where
    // the source COUNT can coincidentally match the old run's while what each index MEANS has changed
    // (reordered, or one source swapped for a differently-named one at the same slot) -- a positional
    // copy there would silently hand a new, unrelated source the old one's progress.
    std::vector<std::string> source_names;
    source_names.reserve(sources.size());
    for (const sub0::BlendSource& s : sources) source_names.push_back(s.name);
    const sub0::ResolvedSchedule resolved_schedule = sub0::resolve_schedule(schedule, source_names);
    sub0::BlendFairness blend_fair(sources.size());
    if (!rs.drawn_names.empty() && rs.drawn_tokens.size() == rs.drawn_names.size()) {
        blend_fair.drawn_tokens = sub0::carry_forward_by_name(rs.drawn_names, rs.drawn_tokens, source_names);
        const auto dropped = std::count_if(rs.drawn_names.begin(), rs.drawn_names.end(),
            [&](const std::string& nm) {
                return std::find(source_names.begin(), source_names.end(), nm) == source_names.end();
            });
        if (dropped > 0)
            sub0::log::info("  blend scheduler: {} source(s) from the checkpoint's fairness state are no "
                            "longer in this run's schedule -- their progress is dropped", dropped);
    } else if (rs.drawn_names.empty() && rs.drawn_tokens.size() == sources.size()) {
        // A v3-without-names checkpoint (predates CKPT_VERSION 4). Positional copy is only safe here
        // because reaching this branch requires the exact source count AND order a routine (non-replace)
        // resume always reconstructs deterministically from the same pinned schedule.
        blend_fair.drawn_tokens = rs.drawn_tokens;
    } else if (!rs.drawn_tokens.empty()) {
        sub0::log::info("  blend scheduler fairness state does not match this run's sources -- starting "
                        "fresh ({} recorded vs {} now)", rs.drawn_tokens.size(), sources.size());
    }
    rs.drawn_names = source_names;   // now safe to overwrite: the reconciliation above already consumed
                                     // whatever names the checkpoint carried in. Constant for the rest of
                                     // the run (source identity never changes mid-run), so this is set
                                     // once here rather than every step alongside drawn_tokens below.

    using clock = std::chrono::steady_clock;
    auto win_t0 = clock::now();
    long win_steps0 = rs.step;
    // Heartbeat: the eval/checkpoint cadence is corpus-relative and can be tens of minutes
    // apart (and val NELBO waits out a half-epoch warmup), so a long run could print nothing
    // for ages. Emit a lightweight interim line every TICK_SECONDS so progress (and the train
    // loss trend) is always visible; `last_log` tracks the wall time of the last printed line,
    // be it an eval or a tick.
    auto last_log = win_t0;
    long last_log_step = win_steps0;
    auto last_ckpt = win_t0;   // wall-clock timer for the crash-resistance checkpoint tick (CKPT_SECONDS)
    double run_loss = 0.0; int run_n = 0;
    // Section timers (CPU path only, cumulative since the last eval, same reset points as run_loss/
    // run_n above) -- diagnostic-only, to see WHERE step wall-clock actually goes (window prep vs.
    // train_batch vs. the optimizer step) without needing an external profiler. See project memory
    // cpu-profiling-tooling-backlog for the real-profiler follow-up this is a stopgap for.
    double prep_secs = 0.0, train_secs = 0.0, opt_secs = 0.0;
    // Windows/sec used to key off (step - win_steps0)*batch, correct only when every step processed
    // exactly `batch` windows. The token-budget scheduler below varies batch_t per step, so
    // throughput is tracked directly as a running window count instead, snapshotted at each reset
    // point exactly like win_steps0/last_log_step are for steps.
    long long total_windows = 0, windows_at_win_t0 = 0, windows_at_last_log = 0;
    // Tokens are snapshotted alongside windows because tok/s CANNOT be derived from win/s. The
    // token-budget scheduler admits batch_t windows of seq_t tokens under batch_t*seq_t <= batch*SEQ_LEN,
    // so a corpus of SHORT documents yields more, shorter windows per step: win/s rises while tok/s does
    // not. Printing wps*SEQ_LEN assumed every window was a full SEQ_LEN and overstated throughput by
    // SEQ_LEN/mean(seq_t) -- measured 2.35x on FineWeb-Edu (257,675 reported vs ~109,000 real), which is
    // the number a run gets planned around. rs.tokens_seen already accumulates batch_t*seq_t exactly;
    // it just was not being differenced here.
    // Seeded from rs.tokens_seen, NOT from 0: on a resume that count is already the checkpoint's
    // carried total (1.97e9 at the LoopSplit arms' 8586 steps), so a zero baseline makes the first
    // interval's numerator the CUMULATIVE tokens instead of the interval's own. Measured on arm A's
    // resume: the first heartbeat printed 11,009,242 tok/s and the first eval 2,121,884 against a real
    // 111,660 -- both self-correct once the tracker resets, so it is exactly two wrong lines per
    // resume. Two too many: win/s and tok/s silently disagreeing by 19x is how a bad throughput number
    // gets planned around, and this codebase has already published and retracted three of those.
    long long tokens_at_win_t0 = rs.tokens_seen, tokens_at_last_log = rs.tokens_seen;
    bool stop = false, graceful_stop = false;
    // A device fault mid-run (a real hardware fault hit this exact codebase once already -- see
    // [[gpu-illegal-access-hardware-fault-not-code-bug]]): stops the loop WITHOUT touching the device
    // again (no sync/eval/save on corrupted or stale host state), leaving the last periodic checkpoint
    // on disk untouched and its state.json status still "training" -- a plain `sub0llm train` with no
    // path will detect that unfinished run and offer to resume it (see registry.hpp's auto-resume).
    bool device_failed = false;
    // Variable-length training is on by default; SUB0_FIXED_SEQ=1 forces full-length windows.
    //TODO: Remove getenv calls
    const bool vary_seq = (MIN_TRAIN_SEQ < SEQ_LEN) && (std::getenv("SUB0_FIXED_SEQ") == nullptr);
    // content_embed_kind is fixed for the whole run by this point (resume-reconciliation above already
    // settled it) -- resolve it to a SlotEncoding ONCE here rather than re-deriving it every window.
    const sub0::SlotEncoding content_embed_enc = sub0::content_embed_encoding_of(content_embed_kind);

    for (long step = rs.step + 1; step <= max_steps && !stop; ++step) {
        // On-demand: refill the rotating shuffle buffer once per eval interval from new
        // random regions, so the run traverses the whole corpus over time. buf_rng is a
        // separate stream, so this never perturbs the resume-critical `rng`. The refill may
        // reallocate train_buf, so re-bind the span (the sampler reads train_span.size() live).
        if (on_demand && steps_since_refresh >= eval_every) {
            buf_rng.seed(static_cast<std::uint32_t>(seed) ^ (0x9E3779B9u * static_cast<std::uint32_t>(++refresh_n)));
            text.fill_random(train_byte_lo, train_byte_hi, OD_TRAIN_BUF_TOK, buf_rng, train_buf);
            train_span = sub0::TokView::over_int32(train_buf.data(), train_buf.size());
            if (kBaseSource != static_cast<std::size_t>(-1))
                sources[kBaseSource].view = train_span;   // the refill reallocates the base buffer -- rebind its source
            steps_since_refresh = 0;
        }
        ++steps_since_refresh;

        // Draw the window starts on the main thread (keeps the RNG stream, hence
        // resume, deterministic), then run the batch -- on the GPU when enabled, else
        // data-parallel across CPU threads.
        const int seq_t = vary_seq ? std::uniform_int_distribution<int>(MIN_TRAIN_SEQ, SEQ_LEN)(rng) : SEQ_LEN;
        // Token-budget batching: trade batch UP as seq_t shrinks so batch_t*seq_t holds roughly
        // constant at tokens_per_step (batch*SEQ_LEN) instead of wasting the fixed per-step overhead
        // (AdamW traffic, grad memset, QKV-mirror rebuild, host syncs) on a short-T step that only
        // fills a fraction of the reserved row budget. When vary_seq is off, seq_t == SEQ_LEN always
        // and this reduces to batch_t == batch exactly (tokens_per_step / SEQ_LEN == batch, no
        // rounding loss) -- bit-for-bit the pre-scheduler behavior.
        const int batch_t = static_cast<int>(tokens_per_step / static_cast<std::size_t>(seq_t));
        // frac_epoch: which schedule stage is active right now, and each active source's target rate --
        // computed once per step (constant across all of this step's windows), not once per window.
        const double frac_epoch = static_cast<double>(step) / static_cast<double>(epoch_steps);
        for (int b = 0; b < batch_t; ++b) {
            const sub0::BlendDraw d = sub0::sample_blend_staged(rng, blend_fair, sources, resolved_schedule,
                                                                seq_t, frac_epoch);   // pick a source, window inside it
            src_idx[b] = d.src;
            starts[b]  = d.win.start;
            win_len[b] = d.win.len;
        }
        const float lr_t = lr_schedule(step, peak_lr, lr_warmup_steps);   // warmup -> inverse-sqrt decay
        // Materializes window b (from this step's src_idx/starts/win_len draw) into cpu_win/cpu_starts/
        // cpu_mask/win_binds at destination slot `dst` -- shared by the plain-CPU branch (dst==b, every
        // window) and hybrid_train's CPU sub-batch (dst is a compacted index, b is the source window
        // hyb_cpu_idx[dst] names). The content-embed branch below is a no-op for windows the caller
        // already knows aren't content-embed-sourced (hybrid_train's CPU sub-batch is ALL such windows
        // by construction), so one shared body covers both shapes.
        const auto materialize_cpu_window = [&](int dst, int b) {
            const sub0::BlendSource& src = sources[static_cast<std::size_t>(src_idx[b])];
            const std::size_t base = static_cast<std::size_t>(dst) * (seq_t + 1);
            const std::size_t n = static_cast<std::size_t>(win_len[b]) + 1;   // inputs + the last shifted target
            src.view.copy_to(starts[b], n, &cpu_win[base]);
            if (any_masked) {
                // Aligned to cpu_win: 1 = trained, 0 = masked. An unmasked source trains every position;
                // a masked source copies its per-token mask for this window. train_batch reads the mask
                // at the TARGET index (base+i+1), so this parallel copy suffices.
                for (std::size_t k = 0; k < n; ++k)
                    cpu_mask[base + k] = src.masked() ? src.mask[starts[b] + k] : std::uint8_t{1};
            }
            cpu_starts[static_cast<std::size_t>(dst)] = base;
            // content_embed: a window drawn from a source that carries structured doc_bindings (any
            // generator-backed schedule source -- scratchspike/op_curriculum) gets its document's real
            // slot->fragment bindings; every other window (base corpus, spellspike) stays nullptr --
            // unchanged, plain reserved-id embedding. Generic over WHICH source, unlike the old
            // scratch_source_idx/op_source_idx-hardcoded branches this replaced.
            if (content_embed_active && !src.doc_bindings.empty()) {
                const std::size_t doc = sub0::doc_of(src.docs, starts[b]);
                step_binds[static_cast<std::size_t>(dst)] = sub0::ScratchBindings{
                    std::span<const std::vector<int>>(src.doc_bindings[doc]), content_embed_enc };
                win_binds[static_cast<std::size_t>(dst)] = &step_binds[static_cast<std::size_t>(dst)];
            } else if (content_embed_active) {
                win_binds[static_cast<std::size_t>(dst)] = nullptr;
            }
        };
        float step_loss;
        if (gpu_train) {
            // Same warmup/decay SHAPE as the CPU branch below, Muon's own (much larger) peak --
            // 0.f when Muon isn't selected, which sub0_dev_train_step treats as pure AdamW.
            const float muon_lr_t = opt.use_muon() ? lr_schedule(step, MUON_LR_BASE, lr_warmup_steps) : 0.f;
            step_loss = gpu.step(sources, src_idx.data(), starts.data(), win_len.data(), batch_t, seq_t, lr_t, muon_lr_t);
            opt.set_step_count(gpu.t);          // keep the checkpoint's step counter in lockstep
            if (!gpu.ok) {
                sub0::log::error("training: GPU step failed (device fault) at step {} -- stopping "
                                 "immediately without a final save; the last periodic checkpoint on "
                                 "disk is untouched and resumable.", step);
                device_failed = true;
                break;   // don't touch the device again this run -- not even the bookkeeping below
            }
        } else if (hybrid_train) {
            // Source-routed split: windows drawn from a content-embed source (scratch-mix/op-mix --
            // encode_slot is CPU-only) route to the CPU path; every other window in this same blend
            // (base corpus, spell-mix) has no such need and routes to the GPU, so the device session
            // opened above isn't wasted just because content-embed forces a CPU-only subset. The CPU
            // sub-batch's train_batch() runs on the persistent cpu_worker thread while the GPU sub-batch's
            // backward_only() runs on the main thread (mostly blocked on the device, not the CPU, so
            // the two don't meaningfully contend for cores); their gradients are weighted-combined by
            // sub-batch size into ONE update -- the algorithm tests/cuda_tests.cpp "CUDA+CPU hybrid
            // split: weighted-merged gradient matches an undivided CPU batch" proved on real hardware
            // (rel-L2 0.0087, cos 0.999963). See project memory hybrid-cpu-gpu-execution-design.
            opt.set_lr(lr_t);
            if (opt.use_muon())
                opt.set_muon_lr(lr_schedule(step, MUON_LR_BASE, lr_warmup_steps));
            const auto t_prep0 = clock::now();
            hyb_cpu_idx.clear(); hyb_gpu_idx.clear();
            for (int b = 0; b < batch_t; ++b) {
                // Generic over WHICH source: any source carrying structured doc_bindings needs the
                // CPU-only slot-binding lookup, UNLESS this backend composes bound rows itself
                // (dev_binding_compose -- install_gpu_bindings below covers the GPU sub-batch's own
                // bound positions in that case).
                const bool needs_cpu = !sources[static_cast<std::size_t>(src_idx[b])].doc_bindings.empty()
                                      && !dev_binding_compose;
                (needs_cpu ? hyb_cpu_idx : hyb_gpu_idx).push_back(b);
            }
            const int cpu_n = static_cast<int>(hyb_cpu_idx.size());
            const int gpu_n = static_cast<int>(hyb_gpu_idx.size());

            if (cpu_n > 0) {   // materialize the CPU sub-batch at COMPACTED positions [0, cpu_n)
                cpu_win.resize(static_cast<std::size_t>(cpu_n) * (seq_t + 1));
                if (any_masked) cpu_mask.resize(cpu_win.size());
                for (int i = 0; i < cpu_n; ++i) {
                    const int b = hyb_cpu_idx[static_cast<std::size_t>(i)];
                    materialize_cpu_window(i, b);
                    hyb_cpu_len[static_cast<std::size_t>(i)] = win_len[b];
                }
            }
            if (gpu_n > 0) {   // compacted GPU sub-batch descriptors at [0, gpu_n)
                for (int i = 0; i < gpu_n; ++i) {
                    const int b = hyb_gpu_idx[static_cast<std::size_t>(i)];
                    hyb_gpu_src[static_cast<std::size_t>(i)]    = src_idx[b];
                    hyb_gpu_starts[static_cast<std::size_t>(i)] = starts[b];
                    hyb_gpu_len[static_cast<std::size_t>(i)]    = win_len[b];
                }
            }
            // Build the binding-compose override table covering the GPU sub-batch's flat rows
            // [0,gpu_n)x[0,seq_t) -- mirrors materialize_cpu_window's ScratchBindings construction
            // (same doc_of + doc_bindings lookup) but flattened into device_backend.hpp's wire format
            // (docs/BACKENDS.md "Design: binding-compose on CUDA"). Every bound position here is
            // guaranteed a device-supported encoding: content_embed_kind's persisted enum never
            // selects anything else (see gpu_available's comment above), so no per-encoding gate is
            // needed. CPU-side construction only -- counted under prep_secs like the CPU sub-batch's
            // own materialize_cpu_window loop above; the actual host->device upload happens under
            // train_secs below, alongside the other device round-trips.
            const bool gpu_has_bindings = gpu_n > 0 && dev_binding_compose;
            if (gpu_has_bindings) {
                gpu_bind_idx.assign(static_cast<std::size_t>(gpu_n) * static_cast<std::size_t>(seq_t), -1);
                gpu_bind_entries.clear();
                gpu_bind_frags.clear();
                for (int i = 0; i < gpu_n; ++i) {
                    const int b = hyb_gpu_idx[static_cast<std::size_t>(i)];
                    const sub0::BlendSource& src = sources[static_cast<std::size_t>(src_idx[b])];
                    if (src.doc_bindings.empty()) continue;
                    const std::size_t doc = sub0::doc_of(src.docs, starts[b]);
                    const sub0::ScratchBindings sb{
                        std::span<const std::vector<int>>(src.doc_bindings[doc]), content_embed_enc };
                    for (int t = 0; t < win_len[b]; ++t) {
                        const int tok = src.view[starts[b] + static_cast<std::size_t>(t)];
                        if (!sb.bound(tok)) continue;
                        const auto frags = sb.fragments(tok);
                        gpu_bind_idx[static_cast<std::size_t>(i) * static_cast<std::size_t>(seq_t)
                                    + static_cast<std::size_t>(t)] =
                            static_cast<int>(gpu_bind_entries.size() / SUB0_DEV_BIND_ENTRY_INTS);
                        gpu_bind_entries.push_back(static_cast<int>(gpu_bind_frags.size()));
                        gpu_bind_entries.push_back(static_cast<int>(frags.size()));
                        gpu_bind_entries.push_back(static_cast<int>(content_embed_enc));
                        gpu_bind_frags.insert(gpu_bind_frags.end(), frags.begin(), frags.end());
                    }
                }
            }
            prep_secs += std::chrono::duration<double>(clock::now() - t_prep0).count();

            const auto t_train0 = clock::now();
            if (gpu_has_bindings &&
                sub0_dev_set_window_bindings(gpu_bind_idx.data(), static_cast<int>(gpu_bind_idx.size()),
                                             gpu_bind_entries.data(),
                                             static_cast<int>(gpu_bind_entries.size() / SUB0_DEV_BIND_ENTRY_INTS),
                                             gpu_bind_frags.data(), static_cast<int>(gpu_bind_frags.size())) != 0) {
                sub0::log::error("training: GPU binding-compose table install failed at step {} -- "
                                 "stopping immediately without a final save; the last periodic "
                                 "checkpoint on disk is untouched and resumable.", step);
                device_failed = true;
                break;
            }
            float cpu_loss = 0.f, gpu_loss = 0.f;
            if (cpu_n > 0 && gpu_n > 0) {
                cpu_worker->run(cpu_n, seq_t, any_masked);      // dispatch to the persistent worker thread
                gpu_loss = gpu.backward_only(sources, hyb_gpu_src.data(), hyb_gpu_starts.data(),
                                             hyb_gpu_len.data(), gpu_n, seq_t, hyb_gpu_grad.data());
                cpu_worker->wait();
                cpu_loss = cpu_worker->loss;
            } else if (cpu_n > 0) {
                cpu_loss = sub0::train_batch(cpu_win.data(), cpu_starts.data(), cpu_n, seq_t,
                                             hyb_cpu_len.data(), any_masked ? cpu_mask.data() : nullptr,
                                             win_binds.data());
            } else {   // gpu_n > 0 -- batch_t >= 1 guarantees at least one of the two is nonempty
                gpu_loss = gpu.backward_only(sources, hyb_gpu_src.data(), hyb_gpu_starts.data(),
                                             hyb_gpu_len.data(), gpu_n, seq_t, hyb_gpu_grad.data());
            }
            // Clear immediately after use -- sub0_dev_set_window_bindings's own contract ("the trainer
            // clears after each step") -- so a later step with gpu_has_bindings==false (an all-unbound
            // draw) doesn't leave a stale table composing rows that should now be plain lookups.
            if (gpu_has_bindings &&
                sub0_dev_set_window_bindings(nullptr, 0, nullptr, 0, nullptr, 0) != 0) {
                sub0::log::error("training: GPU binding-compose table clear failed at step {} -- "
                                 "stopping immediately without a final save; the last periodic "
                                 "checkpoint on disk is untouched and resumable.", step);
                device_failed = true;
                break;
            }
            train_secs += std::chrono::duration<double>(clock::now() - t_train0).count();
            if (!gpu.ok) {
                sub0::log::error("training: GPU sub-batch failed (device fault) at step {} -- stopping "
                                 "immediately without a final save; the last periodic checkpoint on "
                                 "disk is untouched and resumable.", step);
                device_failed = true;
                break;
            }

            const auto t_opt0 = clock::now();
            // Weighted-merge into grad_ptr(): train_batch already left the CPU sub-batch's own
            // mean-normalized gradient there when cpu_n>0; when cpu_n==0 the GPU gradient replaces it
            // wholesale instead (there is nothing to weight against).
            if (gpu_n > 0) {
                float* g = sub0::grad_ptr();
                const std::size_t n = sub0::trainable_floats();
                if (cpu_n > 0) {
                    const float w_cpu = static_cast<float>(cpu_n) / static_cast<float>(batch_t);
                    const float w_gpu = static_cast<float>(gpu_n) / static_cast<float>(batch_t);
                    for (std::size_t i = 0; i < n; ++i) g[i] = w_cpu * g[i] + w_gpu * hyb_gpu_grad[i];
                } else {
                    std::copy_n(hyb_gpu_grad.data(), n, g);
                }
            }
            opt.step();
            step_loss = (static_cast<float>(cpu_n) * cpu_loss + static_cast<float>(gpu_n) * gpu_loss)
                       / static_cast<float>(batch_t);
            // Device params just went stale (host advanced via opt.step() above) -- re-upload UNCONDITIONALLY,
            // even on a step with gpu_n==0. A step that happens to draw an all-CPU batch_t still advances
            // the host params via opt.step(); skipping the re-upload here would leave the device holding
            // an EARLIER step's weights, so the next step that DOES have gpu_n>0 would compute its gradient
            // against stale params -- a real correctness bug (rare under the deficit scheduler, which
            // deliberately spreads a binding-capable source's draws out rather than clustering them, but
            // not impossible), not just a wasted upload. A failed re-upload is exactly as fatal as a
            // failed backward_only above (the device can no longer be trusted for the next step).
            if (sub0_dev_upload_params(sub0::params_ptr()) != 0) {
                sub0::log::error("training: GPU param re-upload failed at step {} -- stopping "
                                 "immediately without a final save; the last periodic checkpoint on "
                                 "disk is untouched and resumable.", step);
                device_failed = true;
                break;
            }
            opt_secs += std::chrono::duration<double>(clock::now() - t_opt0).count();
        } else {
            opt.set_lr(lr_t);                   // apply the scheduled lr to the CPU AdamW-routed update
            if (opt.use_muon())                 // same warmup/decay SHAPE, Muon's own (much larger) peak
                opt.set_muon_lr(lr_schedule(step, MUON_LR_BASE, lr_warmup_steps));
            const auto t_prep0 = clock::now();
            cpu_win.resize(static_cast<std::size_t>(batch_t) * (seq_t + 1));   // materialize windows for the CPU engine
            if (any_masked) cpu_mask.resize(cpu_win.size());   // parallel loss mask (only when a source carries one)
            for (int b = 0; b < batch_t; ++b) materialize_cpu_window(b, b);
            prep_secs += std::chrono::duration<double>(clock::now() - t_prep0).count();
            const auto t_train0 = clock::now();
            step_loss = sub0::train_batch(cpu_win.data(), cpu_starts.data(), batch_t, seq_t, win_len.data(),
                                          any_masked ? cpu_mask.data() : nullptr,
                                          content_embed_active ? win_binds.data() : nullptr);
            train_secs += std::chrono::duration<double>(clock::now() - t_train0).count();
            const auto t_opt0 = clock::now();
            opt.step();
            opt_secs += std::chrono::duration<double>(clock::now() - t_opt0).count();
        }
        run_loss += step_loss; ++run_n;
        total_windows += batch_t;                          // for wps: batch_t varies per step now
        rs.tokens_seen += static_cast<long long>(batch_t) * seq_t;
        rs.step = step;
        rs.drawn_tokens = blend_fair.drawn_tokens;   // keep the checkpoint's fairness snapshot current --
                                                     // any save_checkpoint call below reads it from `rs`

        if (g_graceful_stop.load()) {
            sub0::log::line("  [graceful stop requested @ step {} -- saving before exit]", step);
            std::fflush(stdout);
            stop = graceful_stop = true;
        }

        const bool is_eval = (step % eval_every == 0 || step == max_steps);
        if (is_eval) {
            // Refresh the host param/optimizer arenas from the device so the CPU eval/save/preview
            // code below sees the current weights (no-op on the CPU path). A failure here means the
            // eval/save below would run on stale or corrupted host state -- stop instead of risking a
            // confidently-wrong plateau/save on garbage data (see device_failed's own comment).
            if (!gpu.sync_to_host()) {
                sub0::log::error("training: device sync failed at step {} -- stopping immediately "
                                 "without a final save; the last periodic checkpoint on disk is "
                                 "untouched and resumable.", step);
                device_failed = true;
                break;
            }
            // Throughput over the interval just completed (training only).
            const double secs = std::chrono::duration<double>(clock::now() - win_t0).count();
            const double wps   = secs > 0 ? static_cast<double>(total_windows - windows_at_win_t0) / secs : 0.0;
            const double tps   = secs > 0 ? static_cast<double>(rs.tokens_seen - tokens_at_win_t0) / secs : 0.0;
            // frac_epoch: computed once per step, above the gpu_train/hybrid_train/CPU branch dispatch.

            // Validation NELBO only once enough of the corpus has been seen.
            std::string eval_str = "(warmup)";
            if (step >= warmup_steps) {
                const double nelbo = evaluate(val_span, 0);
                rs.evals.push_back(nelbo);
                if (nelbo < rs.best_loss) { rs.best_loss = nelbo; rs.best_step = step; }
                eval_str = std::format("val_nelbo {:.4f} (best {:.4f})", nelbo, rs.best_loss);
                // Plateau-stop only in AUTO mode (--steps 0). An explicit --steps N is a request to run
                // exactly N steps (matching the flag's help), so a curriculum whose signal isn't in the base
                // val split -- e.g. an op_curriculum blend source -- is not cut short by the base corpus
                // plateauing. frac_epoch (this step's schedule-clock position) gates plateaued()'s own
                // minimum-epoch floor + expected_plateau_epoch hint scaling -- the schedule's own value
                // when it sets one, else DEFAULT_EXPECTED_PLATEAU_EPOCH (the evidence-based ~2-epoch
                // Muon ledger center; see that constant's own comment for why a hint, not a hard cap).
                if (!fixed_budget && plateaued(rs.evals, frac_epoch,
                        schedule.expected_plateau_epoch.value_or(DEFAULT_EXPECTED_PLATEAU_EPOCH))) stop = true;
            }
            const long steps_to_next_epoch = (static_cast<long>(frac_epoch) + 1) * epoch_steps - step;
            // Step-rate directly (not via wps/batch): batch_t now varies per step, so a fixed
            // "windows per step" conversion factor no longer applies -- steps_to_next_epoch is
            // already in step units, so dividing by the interval's own step rate is exact.
            const long steps_completed = step - win_steps0;
            const double eta_next_epoch = (secs > 0 && steps_completed > 0)
                ? static_cast<double>(steps_to_next_epoch) * secs / steps_completed : -1.0;
            sub0::log::line("step {:>7}/{} [{:.2f} ep, next ep in {}]  train {:.4f}  {:.0f} win/s ({:.0f} tok/s)  {}",
                 step, max_steps, frac_epoch, format_eta(eta_next_epoch), run_loss / std::max(1, run_n),
                 wps, tps, eval_str);
            std::fflush(stdout);

            // Progress-named checkpoint + latest model every interval (covers warmup too); prune to
            // the newest `keep` so a long run doesn't fill the disk with 0.5 GB checkpoints. Neither
            // save blocks/retries here on failure (a transient lock -- antivirus, cloud sync, a
            // concurrent `sub0llm-gen` reading model.bin) -- training keeps going, and the NEXT
            // interval's save is the retry. Only the end-of-training save (below the loop) needs
            // real retry-with-backoff, since there is no later interval to fall back on there.
            if (!save_checkpoint(ckpt_step_path(model_path, step), opt.step_count(), rng, rs, batch, lr, seed))
                sub0::log::warn("checkpoint save failed at step {} -- will retry at the next interval", step);
            prune_ckpts(model_path, keep, rs.best_step);
            if (!sub0::save_model(model_path.c_str()))
                sub0::log::warn("model.bin save failed at step {} -- not fatal, retried at the next interval", step);
            write_state("training");          // refresh best_val_nelbo as it improves

            run_loss = 0.0; run_n = 0;
            prep_secs = 0.0; train_secs = 0.0; opt_secs = 0.0;
            win_t0 = clock::now(); win_steps0 = step; windows_at_win_t0 = total_windows;
            tokens_at_win_t0 = rs.tokens_seen;
            last_log = win_t0; last_log_step = step; windows_at_last_log = total_windows;  // an eval counts as a log: reset the tick timer
            tokens_at_last_log = rs.tokens_seen;
            last_ckpt = win_t0;                               // ...and it checkpointed: reset the ckpt tick
        } else if (std::chrono::duration<double>(clock::now() - last_log).count() >= TICK_SECONDS) {
            // Interim heartbeat between evals: train loss (running avg since the last eval) +
            // throughput since the last printed line. No val NELBO here -- that stays on the eval
            // cadence (it is the expensive, plateau-driving measurement).
            const double since = std::chrono::duration<double>(clock::now() - last_log).count();
            const double wps   = since > 0 ? static_cast<double>(total_windows - windows_at_last_log) / since : 0.0;
            const double tps   = since > 0 ? static_cast<double>(rs.tokens_seen - tokens_at_last_log) / since : 0.0;
            // frac_epoch: computed once per step, above the gpu_train/hybrid_train/CPU branch dispatch.
            const long steps_to_next_epoch = (static_cast<long>(frac_epoch) + 1) * epoch_steps - step;
            // ETA uses the rate over the CUMULATIVE window since the last eval (win_t0/win_steps0,
            // TICK_SECONDS * many ticks wide), not just the last TICK_SECONDS tick: a single 3-minute
            // tick's step rate is noisy enough (scheduling jitter, window-length variance) that keying
            // the ETA off it alone made the printed estimate visibly jump between ticks instead of
            // converging. The instant `wps` above is still worth showing as-is (it IS the current-tick
            // rate); only the ETA benefits from the wider, steadier sample.
            const double eta_secs = std::chrono::duration<double>(clock::now() - win_t0).count();
            const long   eta_steps_completed = step - win_steps0;
            const double eta_next_epoch = (eta_secs > 0 && eta_steps_completed > 0)
                ? static_cast<double>(steps_to_next_epoch) * eta_secs / eta_steps_completed : -1.0;
            // Section breakdown (CPU path only; all three stay 0 on GPU, so the suffix is empty there):
            // fraction of the cumulative since-last-eval wall-clock spent in window prep vs. train_batch
            // (forward+backward) vs. the optimizer step -- see prep_secs/train_secs/opt_secs above.
            std::string breakdown;
            const double section_total = prep_secs + train_secs + opt_secs;
            if (section_total > 0.0) {
                breakdown = std::format("  [prep {:.0f}% train {:.0f}% opt {:.0f}%]",
                    100.0 * prep_secs / section_total, 100.0 * train_secs / section_total,
                    100.0 * opt_secs / section_total);
            }
            sub0::log::line("  ~ step {:>7}/{} [{:.2f} ep, next ep in {}]  train {:.4f}  {:.0f} win/s ({:.0f} tok/s){}",
                 step, max_steps, frac_epoch, format_eta(eta_next_epoch), run_loss / std::max(1, run_n), wps, tps,
                 breakdown);
            std::fflush(stdout);
            last_log = clock::now(); last_log_step = step; windows_at_last_log = total_windows;
            tokens_at_last_log = rs.tokens_seen;
        }

        // Crash-resistance checkpoint on a wall-clock tick, independent of the (far rarer) eval
        // cadence -- with FineWeb's ~78k-step eval interval that would otherwise be the only save,
        // ~20h apart. An eval already saved + reset last_ckpt above, so this only fires on the long
        // stretches between evals. No val NELBO / plateau check here: it is purely a resume point.
        if (!is_eval && std::chrono::duration<double>(clock::now() - last_ckpt).count() >= CKPT_SECONDS) {
            if (!gpu.sync_to_host()) {                        // pull live device weights into the host arenas
                sub0::log::error("training: device sync failed at step {} (crash-resistance tick) -- "
                                 "stopping immediately without a final save; the last periodic "
                                 "checkpoint on disk is untouched and resumable.", step);
                device_failed = true;
                break;
            }
            if (!save_checkpoint(ckpt_step_path(model_path, step), opt.step_count(), rng, rs, batch, lr, seed))
                sub0::log::warn("checkpoint save failed at step {} -- will retry at the next interval", step);
            prune_ckpts(model_path, keep, rs.best_step);
            if (!sub0::save_model(model_path.c_str()))
                sub0::log::warn("model.bin save failed at step {} -- not fatal, retried at the next interval", step);
            write_state("training");
            last_ckpt = clock::now();
            sub0::log::line("  [checkpoint @ step {} ({:.0f}m crash-resistance tick)]", step, CKPT_SECONDS / 60.0);
            std::fflush(stdout);
        }
    }

    // A device failure already broke out of the loop above without touching the device again -- do
    // NOT attempt sync_to_host()/save here either, that would either fail again or (worse) silently
    // write stale/corrupted weights over the last GOOD checkpoint. Exit loudly with the last periodic
    // checkpoint's status ("training") left exactly as it was -- see device_failed's own comment for
    // why that's the right resumable state to leave behind.
    if (device_failed) {
        sub0::log::error("training: stopped at step {} due to a device failure -- no final save was "
                         "attempted. The most recent periodic checkpoint (from before this happened) "
                         "is untouched on disk; a bare `sub0llm train` will detect it as unfinished "
                         "and offer to resume it.", rs.step);
        gpu.shutdown();
        return 1;
    }
    if (!gpu.sync_to_host()) {   // ensure the host arenas hold the final device weights before saving
        sub0::log::error("training: final device sync failed at step {} -- refusing to overwrite "
                         "model.bin/checkpoint with what would be stale or corrupted weights. The "
                         "most recent periodic checkpoint on disk is untouched and resumable.", rs.step);
        gpu.shutdown();
        return 1;
    }
    if (graceful_stop) {
        // Time-sensitive: a Ctrl+C/console-close handler has a short OS grace period before a
        // force-kill, so this deliberately does NOT retry-with-backoff -- blocking here risks
        // blowing that grace period and getting force-killed mid-retry, losing MORE than a single
        // quick (possibly failed) attempt would. Still surfaces a failure instead of staying silent.
        if (!sub0::save_model(model_path.c_str()))
            sub0::log::error("model.bin save failed during graceful stop -- the checkpoint below is "
                             "the authoritative record");
        if (!save_checkpoint(ckpt_step_path(model_path, rs.step), opt.step_count(), rng, rs, batch, lr, seed))
            sub0::log::error("checkpoint save failed during graceful stop at step {} -- training "
                             "progress may not be on disk", rs.step);
        prune_ckpts(model_path, keep, rs.best_step);
        write_state("stopped");
        // The checkpoint above is already (best-effort) on disk -- skip the sample generation (a full
        // SEQ_LEN-token decode can take real time at a large model) so the exit stays prompt.
        sub0::log::line("stopped (graceful) at step {} (best val_nelbo {:.4f}) -> {}",
             rs.step, rs.best_loss, model_path);
        gpu.shutdown();
        return 0;
    }
    // Normal end of training (plateau or max-steps): no responsiveness pressure, so this is the one
    // save that gets a real retry-with-backoff -- and, if that still isn't enough, a fallback
    // filename -- since unlike the periodic saves during the loop, there is no later checkpoint
    // interval left to naturally retry at once training has actually finished.
    if (!retry_with_backoff([&] { return sub0::save_model(model_path.c_str()); }, "final model.bin save"))
        sub0::log::error("model.bin save failed after retries; the checkpoint below is the "
                         "authoritative record");
    const std::string final_ckpt = ckpt_step_path(model_path, rs.step);
    if (!retry_with_backoff([&] {
            return save_checkpoint(final_ckpt, opt.step_count(), rng, rs, batch, lr, seed);
        }, "final checkpoint save")) {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string fallback = final_ckpt + ".rescue-" + std::to_string(now);
        sub0::log::error("final checkpoint repeatedly failed to save to '{}' -- falling back to '{}'",
                         final_ckpt, fallback);
        if (!save_checkpoint(fallback, opt.step_count(), rng, rs, batch, lr, seed))
            sub0::log::error("fallback checkpoint save ALSO failed -- training progress may be lost");
    }
    prune_ckpts(model_path, keep, rs.best_step);
    write_state(stop ? "plateaued" : "trained");
    // Generate a full SEQ_LEN-token sample so the preview actually fills and slides the extended
    // context window (preview_at caps the model input at SEQ_LEN; a short 120-token run never
    // reaches it). This exercises the real long-context behaviour the trained window supports.
    sub0::log::line("  --- sample ({}-token context) ---\n  {}", SEQ_LEN, preview("the ", SEQ_LEN, rng));
    sub0::log::line("{} at step {} (best val_nelbo {:.4f}) -> {}",
         stop ? "plateaued" : "reached max steps", rs.step, rs.best_loss, model_path);
    gpu.shutdown();        // release the device session (no-op on the CPU path)
    return 0;
}

namespace {
// Min and median of a cycle-count sample, the two robust summaries we report.
struct Stat { std::uint64_t lo, med; };
Stat summarize(std::vector<std::uint64_t>& v) {
    std::sort(v.begin(), v.end());
    return { v.front(), v[v.size() / 2] };
}
}  // namespace

// Surface the threading mode loudly. A build that lost OpenMP runs every "thread"
// serially; without this banner that only shows up as mysteriously flat scaling in
// the numbers below, which is exactly the silent failure this guards against.
static void report_threading() {
#if defined(_OPENMP)
    std::println("threading: OpenMP enabled ({} hardware threads available)", omp_get_max_threads());
#else
    std::println("threading: !! OpenMP DISABLED -- every run is SINGLE-THREADED; thread");
    std::println("           scaling below is meaningless. Rebuild with OpenMP. !!");
#endif
    std::fflush(stdout);
}

// Shared run-context banner so `train`, `tune`, ... all report the same model/compute facts up
// front. Lines 1-2 (model dims + static memory + compute mode + CUDA availability) come from the
// backend's print_config(); the third names the training path that will actually execute, so the
// throughput numbers that follow are read against the right backend.
static void report_run_context(bool gpu_train, bool hybrid_train) {
    sub0::print_config();   // engine config line (console only; the same dims are in the model's config.json)
    const char* backend = hybrid_train ? "hybrid CPU+GPU (source-routed split: content-embed windows on "
                                          "CPU, the rest on GPU; host-canonical AdamW)"
                        : gpu_train    ? "GPU (resident device step: fwd+bwd+AdamW)"
                                       : "CPU (data-parallel minibatch)";
    sub0::log::line("training backend: {}", backend);
    std::fflush(stdout);
}

// Single-thread, cycle-accurate microbenchmark of the training hot path. It is the
// control baseline for optimization: pinned to one core, OpenMP forced to one
// thread, timed in rdtsc cycles, reported as the min (un-throttled cost) and median
// over many iterations so thermal drift and scheduling don't move the number.
extern "C" SUB0_API int sub0_bench_stage(int iters, int threads, int windows_per_thread) {
    if (iters <= 0) iters = 200;
#if defined(_WIN32)
    const DWORD_PTR prev_aff =
        SetThreadAffinityMask(GetCurrentThread(), 1ull);  // pin to logical CPU 0 (a P-core)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#endif

    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const sub0::TokView data = load_corpus_tokens(tok, od_buf, "bench");
    if (data.empty()) return 1;
    sub0::build_model();
    sub0::AdamW opt(0.001f);
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);

    std::vector<std::uint64_t> fwd, bwd, optc, step;
    fwd.reserve(iters); bwd.reserve(iters); optc.reserve(iters); step.reserve(iters);

    const int warmup = std::max(5, iters / 10);
    int win[SEQ_LEN + 1];                                   // materialize the window (token view may be uint16-packed)
    for (int it = -warmup; it < iters; ++it) {
        const size_t s = startd(rng);
        data.copy_to(s, SEQ_LEN + 1, win);
        const int* x = win;
        const int* y = win + 1;

        opt.zero_grad();
        const std::uint64_t s0 = cpu_cycles();
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(x, SEQ_LEN);
        const std::uint64_t s1 = cpu_cycles();
        sub0::Node* loss = sub0::cross_entropy(logits, y);
        const std::uint64_t s2 = cpu_cycles();
        sub0::backward(loss, 1.f);
        sub0::reduce_gradients();        // single-window: publish grad for the optimizer
        const std::uint64_t s3 = cpu_cycles();
        opt.step();
        const std::uint64_t s4 = cpu_cycles();

        if (it >= 0) {
            fwd.push_back(s1 - s0);
            bwd.push_back(s3 - s2);
            optc.push_back(s4 - s3);
            step.push_back(s4 - s0);
        }
    }

    // Wall time of one representative window to anchor cycles to a TSC frequency.
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t c0 = cpu_cycles();
    constexpr int CAL = 2000;
    for (int i = 0; i < CAL; ++i) { data.copy_to(startd(rng), SEQ_LEN, win); sub0::graph_reset(); (void)sub0::forward(win, SEQ_LEN); }
    const std::uint64_t c1 = cpu_cycles();
    const double cal_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double tsc_ghz = cal_s > 0 ? static_cast<double>(c1 - c0) / cal_s / 1e9 : 0.0;

    const Stat sf = summarize(fwd), sb = summarize(bwd), so = summarize(optc), ss = summarize(step);
    auto us = [&](std::uint64_t cyc) { return tsc_ghz > 0 ? cyc / (tsc_ghz * 1000.0) : 0.0; };

    report_threading();
    std::println("--- single-thread hot-path bench ({} iters, 1 thread, pinned) ---", iters);
    std::println("TSC ~{:.2f} GHz | min cycles/window (median in parens), us at TSC rate", tsc_ghz);
    std::println("  forward    {:>10}  ({:>10})   {:.1f} us", sf.lo, sf.med, us(sf.lo));
    std::println("  backward   {:>10}  ({:>10})   {:.1f} us", sb.lo, sb.med, us(sb.lo));
    std::println("  optimizer  {:>10}  ({:>10})   {:.1f} us", so.lo, so.med, us(so.lo));
    std::println("  full step  {:>10}  ({:>10})   {:.1f} us", ss.lo, ss.med, us(ss.lo));
    std::println("  -> min full-step throughput: {:.0f} window/s", ss.lo ? tsc_ghz * 1e9 / ss.lo : 0.0);

    // --- Data-parallel minibatch throughput ---------------------------------
    // Exercises train_batch end to end: each window's forward+backward runs on its
    // own thread into a private Worker slot, then the per-slot gradients are summed
    // into the shared gradient. This is the path the Worker memory layout affects --
    // the single cross-thread loop strides over slots -- so it is the control for
    // deciding whether to split the gradient accumulator out of the Worker struct.
    // Compare window/s here against the single-thread full step above to read off
    // both multi-thread scaling and the reduction overhead.
#if defined(_WIN32)
    if (prev_aff) SetThreadAffinityMask(GetCurrentThread(), prev_aff);  // unpin for multi-core
#endif
    {
        // The minibatch scales with the thread count: each thread carries a fixed
        // WINDOWS_PER_THREAD windows, so per-thread work is constant across thread
        // counts and the "Nx vs 1-thread" figure is a clean scaling read. (The old
        // max(active*4, 16) floor clamped the batch below 4 threads and skewed that
        // comparison.) biters = iters/4 keeps this section's wall time ~matched to the
        // single-thread section, since `active` threads chew through B windows per step.
        // windows/thread is tunable (the `tune` subcommand searches it) and passed in.
        const int WINDOWS_PER_THREAD = windows_per_thread > 0 ? windows_per_thread : 4;
        const int active = threads > 0 ? threads : 1;
        const int B = active * WINDOWS_PER_THREAD;
        std::vector<std::size_t> starts(static_cast<size_t>(B)), mstarts(static_cast<size_t>(B));
        std::vector<int> mbuf(static_cast<size_t>(B) * (SEQ_LEN + 1));   // materialized windows (uint16-packed view)
        std::vector<std::uint64_t> batch_cyc;
        const int bwarm = 3, biters = std::max(20, iters / 4);
        batch_cyc.reserve(static_cast<size_t>(biters));
        for (int it = -bwarm; it < biters; ++it) {
            for (int b = 0; b < B; ++b) {
                starts[static_cast<size_t>(b)] = startd(rng);
                data.copy_to(starts[static_cast<size_t>(b)], SEQ_LEN + 1, &mbuf[static_cast<size_t>(b) * (SEQ_LEN + 1)]);
                mstarts[static_cast<size_t>(b)] = static_cast<size_t>(b) * (SEQ_LEN + 1);
            }
            const std::uint64_t b0 = cpu_cycles();
            sub0::train_batch(mbuf.data(), mstarts.data(), B, SEQ_LEN);
            const std::uint64_t b1 = cpu_cycles();
            if (it >= 0) batch_cyc.push_back(b1 - b0);
        }
        const Stat sbatch = summarize(batch_cyc);
        const double per_window = static_cast<double>(sbatch.lo) / B;
        std::println("--- data-parallel minibatch ({} threads, batch {} = {} windows/thread) ---",
                     active, B, WINDOWS_PER_THREAD);
        std::println("  batch      {:>10}  ({:>10})   {:.1f} us", sbatch.lo, sbatch.med, us(sbatch.lo));
        std::println("  -> {:.0f} window/s  ({:.0f} cycles/window, {:.2f}x vs 1-thread full step)",
                     per_window > 0 ? tsc_ghz * 1e9 / per_window : 0.0, per_window,
                     per_window > 0 ? static_cast<double>(ss.lo) / per_window : 0.0);
    }
    return 0;
}

// --- Auto-tune ---------------------------------------------------------------
// `sub0llm tune` searches the runtime knobs that govern data-parallel throughput
// (thread count and per-thread batch granularity) and reports the configuration
// with the highest measured window/s. The search itself lives in the engine-free
// sub0::tune module; this stage only supplies the knobs and a measurement-backed
// objective, which keeps the math in tune.hpp unit testable in isolation.
//
// TODO(tune-train): a training auto-tuner is the natural next user of sub0::tune --
// build a Space over learning-rate / batch ladders and pass an objective of
// -validation_NELBO (best-of-N short runs) to maximize(). No change to tune.hpp is
// needed; only a new objective lambda and knob set here.
namespace {

// Wall-clock throughput (window/s) of consecutive train_batch calls at a given thread count and
// per-thread window count, measured with the SHARED budget-sized timer (sub0/bench.hpp) -- the
// same adaption the GPU sweep uses. Rather than a fixed step count, the run is sized to a wall-time
// budget: a quick burst while exploring broadly, a longer throttled run while pinning the winner
// (a longer budget lets the cores heat up and thermally throttle, so the number reflects the
// SUSTAINED rate a real training run sees). Returns the throughput AND the wall time spent
// profiling (so the live trace -- and a captured log -- shows where the tuner spent its time).
// Random run-to-run noise is handled upstream by the search's median-of-samples confirmation.
struct DpMeasure { double window_per_s = 0.0; double profile_ms = 0.0; int steps = 0; };

DpMeasure measure_dp_throughput(sub0::TokView data, std::mt19937& rng,
                                int threads, int windows_per_thread, double budget_ms) {
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#endif
    const int active = threads > 0 ? threads : 1;
    const int B = active * std::max(1, windows_per_thread);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);
    std::vector<std::size_t> starts(static_cast<size_t>(B)), mstarts(static_cast<size_t>(B));
    std::vector<int> mbuf(static_cast<size_t>(B) * (SEQ_LEN + 1));   // materialized windows (uint16-packed view)

    auto one_step = [&] {
        for (int b = 0; b < B; ++b) {
            starts[static_cast<size_t>(b)] = startd(rng);
            data.copy_to(starts[static_cast<size_t>(b)], SEQ_LEN + 1, &mbuf[static_cast<size_t>(b) * (SEQ_LEN + 1)]);
            mstarts[static_cast<size_t>(b)] = static_cast<size_t>(b) * (SEQ_LEN + 1);
        }
        sub0::train_batch(mbuf.data(), mstarts.data(), B, SEQ_LEN);
    };
    auto run_timed = [&](int n) -> double {
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < n; ++s) one_step();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    // CPU-flavoured bounds: one warmup pass primes caches; allow many short steps for the fast
    // multi-thread configs while keeping the budget in charge of the wall time.
    const sub0::bench::Budget b{ .budget_ms = budget_ms, .warmup = 1, .min_iters = 4, .max_iters = 4096 };
    const sub0::bench::Timing t = sub0::bench::adaptive_time(one_step, run_timed, b);
    const double wps = t.per_step_ms > 0.0 ? static_cast<double>(B) * 1000.0 / t.per_step_ms : 0.0;
    return { wps, t.total_ms, t.iters };
}

// Read an int value for `key` from the tune cache ("key=value" lines), leaving *out untouched if
// the cache or key is absent. Lets `tune --gpu` carry the existing CPU tuning (threads / windows)
// forward instead of clobbering it when it rewrites the cache.
bool read_tune_cache_value(const char* key, int* out) {
    if (DEFAULT_TUNE_CACHE[0] == '\0') return false;
    std::ifstream in(DEFAULT_TUNE_CACHE);
    if (!in) return false;
    const std::string want = std::string(key) + "=";
    for (std::string line; std::getline(in, line); )
        if (line.rfind(want, 0) == 0) { *out = std::atoi(line.c_str() + want.size()); return true; }
    return false;
}

}  // namespace

// Backend selector for sub0_tune_stage (kept an int for the C ABI): 0=auto, 1=all, 2=cpu, 3=gpu.
// `auto` and `all` both tune the CPU sweep plus the GPU device-step knobs when a CUDA device is
// present; `cpu` tunes only the data-parallel sweep; `gpu` tunes only the device-step knobs and
// carries the cached CPU tuning forward. The driver maps the --backend string onto these.
enum : int { TUNE_BACKEND_AUTO = 0, TUNE_BACKEND_ALL = 1, TUNE_BACKEND_CPU = 2, TUNE_BACKEND_GPU = 3 };

// This build's dimensions for the pure footprint model (sub0/memplan.hpp): lets the GPU sweep
// predict the resident VRAM a training batch needs BEFORE allocating it, and lets the run
// cross-check that prediction against the device's actual usage. `tied`/`qk_norm`/`gated` must match
// USE_TIED_EMBEDDINGS/USE_QK_NORM/USE_GATED_FFN -- see memplan.hpp's Dims comments.
static constexpr sub0::memplan::Dims kGpuDims = sub0::current_build_dims();

extern "C" SUB0_API int sub0_tune_stage(int max_threads, int verbose, int backend,
                                        int thorough, int budget_s) {
    // AUTO resolves to GPU-only when a CUDA device is available: that is the only backend this
    // project trains against today, and the CPU sweep is both slow (its own search loop) and its
    // result goes unused by a GPU run. ALL always runs both, for anyone who explicitly wants the CPU
    // numbers too. Once a real hybrid CPU-offload training path exists, AUTO should decide between
    // GPU-only and CPU+GPU based on whether tuning the CPU side is actually worthwhile then -- not a
    // concern yet, so this is deliberately the simple binary case, not a placeholder for it.
    if (backend == TUNE_BACKEND_AUTO && HAS_CUDA) backend = TUNE_BACKEND_GPU;
    const bool run_cpu = backend != TUNE_BACKEND_GPU;   // gpu-only skips the CPU sweep
    const bool run_gpu = backend != TUNE_BACKEND_CPU;   // cpu-only skips the device-step sweep

    // NO global timeout: each sample is bounded by its own PER-TEST budget (the Schedule's per-phase
    // budget_ms, enforced by bench::adaptive_time's per-test wall cap), so the whole knob grid is
    // measured -- no important point is skipped (a global deadline would skip whatever it didn't reach
    // first). `--seconds` is an optional hard SAFETY backstop only (0 = none, the default).
    const auto tune_start = std::chrono::steady_clock::now();
    std::function<bool()> safety_stop = nullptr;
    if (budget_s > 0)
        safety_stop = [end = tune_start + std::chrono::seconds(budget_s)] {
            return std::chrono::steady_clock::now() > end;
        };
    std::println("tune: per-test budget ({}){}", thorough ? "thorough" : "fast",
                 budget_s > 0 ? std::format(" | safety cap {}s", budget_s) : std::string{});

    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const sub0::TokView data = load_corpus_tokens(tok, od_buf, "tune");
    if (data.empty()) return 1;
    sub0::build_model();
    std::mt19937 rng(123);

    // Lead with the same model/compute context the training run prints, so a captured tune log
    // records the dimensions the throughput numbers below were measured against. The training
    // backend reflects the path this tune will exercise (the GPU device step when CUDA is present
    // and not excluded via --backend cpu).
    report_run_context(run_gpu && HAS_CUDA);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int cap = max_threads > 0 ? std::min<int>(max_threads, static_cast<int>(hw))
                                     : static_cast<int>(hw);

    // CPU data-parallel result. Filled by the CPU sweep below; when --backend gpu skips that sweep
    // we carry the existing CPU tuning forward (cache, else baked defaults) so the rewrite never
    // clobbers it.
    int best_threads = DEFAULT_THREADS;
    int best_wpt     = DEFAULT_WINDOWS_PER_THREAD;

    if (run_cpu) {
        // The knob space. Extensible: append a Knob and read it by index in the objective.
        std::vector<double> thread_vals;
        for (int t = 1; t <= cap; ++t) thread_vals.push_back(static_cast<double>(t));
        sub0::tune::Space space = {
            {"threads",        thread_vals},
            {"windows/thread", {1, 2, 4, 8, 16}},
        };

        // Objective: maximize measured window/s. tune::maximize never sees the engine. Reports each
        // configuration live (buffering the whole trace to the end would look like a long hang),
        // tracking the running best. The wall-time BUDGET grows as the search narrows (driven by
        // on_phase below) so the precise, narrowing-down measurements run long enough to thermally
        // throttle -- the regime a sustained training run actually operates in -- while every point
        // still costs ~the same wall time regardless of how fast that configuration is.
        double budget_ms = 200.0;
        double running_best = 0.0;
        sub0::tune::Objective objective = [&](const sub0::tune::Assignment& a) {
            const int threads = static_cast<int>(std::lround(a[0]));
            const int wpt     = static_cast<int>(std::lround(a[1]));
            const DpMeasure m = measure_dp_throughput(data, rng, threads, wpt, budget_ms);
            if (verbose) {
                const bool best = m.window_per_s > running_best;
                std::println("  threads={:>2}  windows/thread={:>2}  ->  {:>6.0f} window/s  "
                             "({:.2f}s / {} steps){}",
                             threads, wpt, m.window_per_s, m.profile_ms / 1000.0, m.steps,
                             best ? "   <- best" : "");
                std::fflush(stdout);   // stream progress as it happens, don't buffer to the end
            }
            running_best = std::max(running_best, m.window_per_s);
            return m.window_per_s;
        };

        std::println("--- auto-tune throughput (knobs: threads 1..{}, windows/thread {{1,2,4,8,16}}) ---", cap);
        report_threading();
        std::fflush(stdout);

        // Shared budgeted-sweep schedule (sub0::tune::schedule_for / apply): the fast vs thorough
        // effort + per-phase measurement budgets live in ONE tested place; here we add only the
        // CPU-specific deadline, cool-down and phase logging. The GPU sweep below reuses the same.
        sub0::tune::Options opt;
        opt.settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); };
        const sub0::tune::Schedule sched = sub0::tune::schedule_for(thorough != 0, /*gpu=*/false);
        sub0::tune::apply(opt, sched, safety_stop, &budget_ms,
            [&](sub0::tune::Phase p) {
                switch (p) {
                    case sub0::tune::Phase::Explore:
                        std::println("[explore] wide sweep, short noisy runs (~{:.1f}s/point)", sched.explore_ms / 1000.0); break;
                    case sub0::tune::Phase::Refine:
                        std::println("[refine]  narrowing in, longer runs (~{:.1f}s/point)", sched.refine_ms / 1000.0); break;
                    case sub0::tune::Phase::Confirm:
                        std::println("[confirm] re-measuring finalists (~{:.1f}s/point, dropping clear losers)", sched.confirm_ms / 1000.0); break;
                }
                std::fflush(stdout);
            });
        opt.separable = (thorough == 0);    // fast: coordinate-descent (threads/wpt separable); thorough: joint grid
        sub0::tune::Result r = sub0::tune::maximize(space, objective, opt);
        if (safety_stop && safety_stop())
            std::println("[budget] CPU sweep hit the --seconds safety cap -- reporting best so far");

        best_threads = static_cast<int>(std::lround(r.best[0]));
        best_wpt     = static_cast<int>(std::lround(r.best[1]));
        const double single = measure_dp_throughput(data, rng, 1, best_wpt, 1200.0).window_per_s;
        std::println("");
        std::println("evaluated {} measurements; winner confirmed over {} samples", r.evaluations, r.best_samples);
        std::println("best: threads={}  windows/thread={}  ->  {:.0f} window/s  ({:.2f}x vs 1 thread)",
                     best_threads, best_wpt, r.best_score, single > 0 ? r.best_score / single : 0.0);
        std::println("apply with:  sub0llm bench --threads {} --windows-per-thread {}", best_threads, best_wpt);
    } else {
        // --backend gpu: skip the CPU sweep. Carry the existing CPU tuning forward (cache wins over
        // the baked defaults) so we tune only the GPU knobs and rewrite the cache without losing it.
        read_tune_cache_value("threads", &best_threads);
        read_tune_cache_value("windows_per_thread", &best_wpt);
        std::println("--- GPU-only tune (CPU sweep skipped; carrying threads={} windows/thread={}) ---",
                     best_threads, best_wpt);
        std::fflush(stdout);
    }

    // GPU device-step tuning (Phase 2e+): when the CUDA backend is built and a device is present,
    // also tune the GPU throughput knobs by timing the resident training step. The BATCH is tuned
    // in any build (it is a direct dimension, not a Knob); TF32 and the attention-backward strategy
    // are runtime-mutable only under SUB0_TUNING, so they are swept there and baked otherwise.
    //
    // TODO(dynamic-training-mode): extend SUB0_TUNING into a "dynamic training" build where the MODEL
    // dimensions (d_model / n_layers / n_heads / d_ff) are runtime parameters instead of baked
    // constexpr. Today they are compile-time (best cache locality + folded hot loops), so comparing
    // e.g. n_heads = 3 vs 4 vs 5 needs a reconfigure+rebuild per point. A dynamic mode would trade
    // that speed for the flexibility to sweep architecture knobs in one process (a probe-train per
    // candidate, ranked by short-budget val NELBO -- see the `report` early-indicator discussion),
    // making automatic dimension search possible. Slower kernels (dims not known at compile time),
    // so gate it behind SUB0_TUNING and keep the baked path as the production default.
    int gpu_batch = best_threads * best_wpt;        // default: the CPU-tuned data-parallel width
    int gpu_tf32  = CUDA_TF32 ? 1 : 0;
    if (!run_gpu) {                                 // --backend cpu: keep any cached GPU knobs intact
        read_tune_cache_value("gpu_batch", &gpu_batch);
        read_tune_cache_value("cuda_tf32", &gpu_tf32);
    }
#if defined(SUB0_BUILD_CUDA)
    if (run_gpu && HAS_CUDA && sub0_dev_init() == 0) {
        std::println("");
#if defined(SUB0_TUNING)
        std::println("--- GPU device-step tuning (knobs: batch, TF32, attn-backward) ---");
#else
        std::println("--- GPU device-step tuning (knob: batch; TF32/attn-backward baked) ---");
#endif
        report_threading();

        // Auto-validate the footprint model against reality before trusting it to gate the sweep:
        // allocate the smallest grid batch for real and compare the measured VRAM delta to the pure
        // prediction. A meaningful gap means memplan.hpp has drifted from the allocations in
        // backend_cuda.cu -- warn loudly (and the CUDA footprint test fails) so the predictive guard
        // below, which trusts the prediction, is never silently steering on a stale formula.
        {
            double pred_mb = 0.0, act_mb = 0.0;
            if (sub0_dev_train_footprint(64, &pred_mb, &act_mb) == 0 && act_mb > 0.0) {
                // Asymmetric (see memplan.hpp): under-prediction beyond the tight tolerance is the
                // OOM-risk "stale" case; over-prediction is safe up to the wider band (tolerates the
                // known steady d768 over-estimate without crying wolf every run).
                const double gap = pred_mb - act_mb;   // >0 over-predict (safe), <0 under-predict (risky)
                const bool stale = gap < -sub0::memplan::FOOTPRINT_TOLERANCE_MB
                                || gap >  sub0::memplan::FOOTPRINT_OVERPREDICT_TOLERANCE_MB;
                std::println("footprint check @ batch 64: predicted {:.0f} MiB | measured {:.0f} MiB | gap {:+.0f} MiB{}",
                             pred_mb, act_mb, gap,
                             stale ? "  !! memplan.hpp is STALE -- update the device footprint model !!" : "");
                std::fflush(stdout);
            }
        }
        // The footprint probe above LEAVES its batch-64 set (~1.5 GB) resident. Free it before
        // measuring free VRAM below, else the budget -- and the VRAM-fit batch ceiling derived from
        // it -- is under-counted by that set (the conservative 293 we saw). The CUDA context survives
        // shutdown, so the sweep's first measurement re-allocs from a clean, accurately-budgeted slate.
        sub0_dev_shutdown();

        // The SAME robust search the CPU sweep uses (sub0::tune::maximize): a joint coarse grid +
        // top-basin refinement + median-of-samples confirmation. This replaces the old single-pass
        // batch sweep that stopped at the first sub-6% throughput gain -- a rule a single noisy
        // reading could trip early, and which silently settled on a LOCAL trough (the measured curve
        // dips at batch 512 below 384 before climbing higher again at 768, so "first plateau" picked
        // the worse of two peaks). Objective = measured tok/s; the device timer is budget-sized so
        // each sample costs ~the same wall time. (The attention backward is now the flash tiled path
        // unconditionally, so batch and TF32 are the only device knobs left to sweep.)
        sub0_dev_set_tf32(CUDA_TF32 ? 1 : 0);      // baseline for batch-only builds
        // Runtime VRAM-fit batch ladder instead of a baked list: compute the largest batch whose
        // resident footprint fits dedicated VRAM (memplan::max_batch_for_vram), then double 64..ceiling.
        // bf16's halved footprint lifts that ceiling well past the old static 1024, so larger batches
        // get TUNED automatically on bigger cards / smaller models; the guard below still skips misfits.
        constexpr int kTuneMaxBatch = 4096;         // sanity cap == MAX_FWD_BATCH (device scratch ceiling)
        const int act_b = ACT_DTYPE == Dtype::BF16 ? 2 : 4;
        // Budget against ACTUAL free VRAM, not the baked GPU_VRAM_MB spec: the spec is the card total,
        // but the usable budget is total minus the CUDA context/driver reservation (which the pure
        // footprint model cannot see -- that gap is why a batch the model said "fits" still spilled at
        // ~8.3 GB on an 8 GB card). Reserve a further headroom for the cuBLAS GEMM workspace (allocated
        // lazily on the first matmul, AFTER this measurement) plus allocator fragmentation.
        constexpr int kVramHeadroomMB = 512;        // cuBLAS workspace + fragmentation slack
        const int free_mb = sub0_dev_free_mem_mb();
        const int vram_budget = (free_mb > 0 ? std::min(free_mb, static_cast<int>(GPU_VRAM_MB)) : GPU_VRAM_MB)
                                - kVramHeadroomMB;
        const int cap   = sub0::memplan::max_batch_for_vram(kGpuDims, vram_budget, kTuneMaxBatch, act_b);
        std::vector<int> batch_ladder;
        for (int b = 64; b < cap; b *= 2) batch_ladder.push_back(b);
        if (cap >= 64) batch_ladder.push_back(cap);             // top rung = the exact VRAM ceiling
        if (batch_ladder.empty()) batch_ladder.push_back(std::max(1, cap));
        std::println("batch ladder (VRAM-fit, {} MiB usable of {} free / {} spec): {} rungs, max batch {}",
                     vram_budget, free_mb, GPU_VRAM_MB, batch_ladder.size(), cap);
        sub0::tune::Space gspace = { {"batch", std::vector<double>(batch_ladder.begin(), batch_ladder.end())} };
#if defined(SUB0_TUNING)
        gspace.push_back({"tf32",     {0, 1}});
#endif
        // Format the optional TF32 knob column only when it is in the space.
        auto knob_suffix = [](const sub0::tune::Assignment& a) -> std::string {
            return a.size() > 1 ? std::format("  tf32={}", static_cast<int>(std::lround(a[1]))) : std::string();
        };

        double gbudget_ms = 400.0;          // grown per phase via on_phase below
        double grunning_best = 0.0;
        sub0::tune::Objective gobjective = [&](const sub0::tune::Assignment& a) -> double {
            const int batch = static_cast<int>(std::lround(a[0]));
            // Predictive VRAM guard: never even TIME a batch whose resident footprint cannot fit the
            // usable VRAM budget (free minus context/cuBLAS headroom, computed above). On Windows an
            // over-budget cudaMalloc does NOT OOM -- it silently spills to WDDM shared memory and
            // THRASHES over PCIe, so the measurement would "succeed" at a ruinous ~10x-slower rate and
            // pollute the search. Predict up front and skip instead.
            if (vram_budget > 0) {
                const int need = sub0::memplan::train_resident_mb(kGpuDims, batch, ACT_DTYPE == Dtype::BF16 ? 2 : 4);
                if (need > vram_budget) {
                    if (verbose) {
                        std::println("  batch={:>4}{}  ->  predicted {} MiB > {} MiB usable, skipping (would spill)",
                                     batch, knob_suffix(a), need, vram_budget);
                        std::fflush(stdout);
                    }
                    return 0.0;
                }
            }
            if (a.size() > 1) sub0_dev_set_tf32(static_cast<int>(std::lround(a[1])));
            double ms = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            const int rc = sub0_dev_time_train_step(batch, SEQ_LEN, gbudget_ms, &ms);
            const double profile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // VRAM cap / timing failure -> worst score so the search steers away (and keeps probing
            // the smaller batches) instead of aborting the whole sweep on one allocation failure.
            if (rc != 0 || ms <= 0.0) {
                if (verbose) {
                    std::println("  batch={:>4}{}  ->  alloc/timing failed  ({:.2f}s)",
                                 batch, knob_suffix(a), profile_s);
                    std::fflush(stdout);
                }
                return 0.0;
            }
            const double tok = static_cast<double>(batch) * SEQ_LEN * 1000.0 / ms;
            if (verbose) {
                const bool best = tok > grunning_best;
                std::println("  batch={:>4}{}  ->  {:>8.0f} tok/s  ({:.2f} ms/step, {:.2f}s){}",
                             batch, knob_suffix(a), tok, ms, profile_s, best ? "   <- best" : "");
                std::fflush(stdout);
            }
            grunning_best = std::max(grunning_best, tok);
            return tok;
        };

        // One-time device warmup BEFORE the search. A fresh CUDA context pays its cold start --
        // cuBLAS handle/workspace + first-matmul autotuning (~100s) plus the resident-param malloc --
        // on the very first timed step. Left uncharged that single cold sample blew the WHOLE time
        // budget before the batch ladder was even explored (the tune then cached the one batch it
        // managed to measure as "best"). A couple of discarded steps at a mid-ladder batch realize the
        // context + prime cuBLAS so every measured point below reflects steady-state throughput.
        {
            const int warm_batch = batch_ladder[batch_ladder.size() / 2];
            double warm_ms = 0.0;
            const auto wt0 = std::chrono::steady_clock::now();
            const bool warm_ok = sub0_dev_time_train_step(warm_batch, SEQ_LEN, 100.0, &warm_ms) == 0;  // primes context + cuBLAS
            if (warm_ok && sub0_dev_time_train_step(warm_batch, SEQ_LEN, 100.0, &warm_ms) != 0)         // steady-state read
                sub0::log::warn("device warmup's steady-state read failed -- the tune sweep below will "
                                "likely surface the same problem, but timings up to that point may be off");
            if (!warm_ok)
                sub0::log::warn("device warmup failed at batch {} -- the tune sweep below will likely "
                                "hit the same problem for every batch it tries", warm_batch);
            std::println("device warmup @ batch {}: {:.0f} ms/step (primed in {:.1f}s)", warm_batch, warm_ms,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - wt0).count());
            std::fflush(stdout);
        }

        // Same shared schedule, GPU profile (device steps cost ~seconds: longer per phase, lighter
        // confirm). Only the device-step deadline and the per-phase log line differ from the CPU sweep.
        sub0::tune::Options gopt;
        gopt.settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); };
        const sub0::tune::Schedule gsched = sub0::tune::schedule_for(thorough != 0, /*gpu=*/true);
        sub0::tune::apply(gopt, gsched, safety_stop, &gbudget_ms,
            [&](sub0::tune::Phase p) {
                switch (p) {
                    case sub0::tune::Phase::Explore:
                        std::println("[explore] joint batch x knob grid (~{:.1f}s/sample)", gsched.explore_ms / 1000.0); break;
                    case sub0::tune::Phase::Refine:
                        std::println("[refine]  zooming the leading basins (~{:.1f}s/sample)", gsched.refine_ms / 1000.0); break;
                    case sub0::tune::Phase::Confirm:
                        std::println("[confirm] re-measuring finalists by median (~{:.1f}s/sample)", gsched.confirm_ms / 1000.0); break;
                }
                std::fflush(stdout);
            });
        gopt.separable = (thorough == 0);   // fast: coordinate-descent; thorough: joint batch x knob grid
        const sub0::tune::Result gr = sub0::tune::maximize(gspace, gobjective, gopt);
        if (safety_stop && safety_stop())
            std::println("[budget] GPU sweep hit the --seconds safety cap -- reporting best so far");

        gpu_batch = static_cast<int>(std::lround(gr.best[0]));
#if defined(SUB0_TUNING)
        gpu_tf32 = static_cast<int>(std::lround(gr.best[1]));
#else
        std::println("  (TF32 is baked in this build; rebuild with -DSUB0_TUNING=ON to tune it)");
#endif
        std::println("");
        std::println("evaluated {} device measurements; winner confirmed over {} samples",
                     gr.evaluations, gr.best_samples);
        std::println("best GPU: batch={}  tf32={}  ->  {:.0f} tok/s (median)",
                     gpu_batch, gpu_tf32, gr.best_score);
        sub0_dev_shutdown();
    }
#endif

    // Persist ALL tuned throughput knobs ONCE so the next build bakes them into the config header
    // (DEFAULT_THREADS / DEFAULT_WINDOWS_PER_THREAD / DEFAULT_GPU_BATCH / CUDA_TF32).
    if (DEFAULT_TUNE_CACHE[0] == '\0') {
        std::println("(no tune cache configured; tuned defaults not persisted)");
    } else if (std::ofstream cache(DEFAULT_TUNE_CACHE, std::ios::trunc); cache) {
        cache << "threads=" << best_threads << "\n"
              << "windows_per_thread=" << best_wpt << "\n";
#if defined(SUB0_BUILD_CUDA)
        cache << "gpu_batch=" << gpu_batch << "\n"
              << "cuda_tf32=" << gpu_tf32 << "\n";
#endif
        std::println("persisted tuned defaults to {}", DEFAULT_TUNE_CACHE);
        std::println("rebuild to bake them in:  cmake --build --preset native");
    } else {
        sub0::log::warn("could not write tune cache '{}'", DEFAULT_TUNE_CACHE);
    }
    return 0;
}

// --- Auto-temperature ("coherence tuner") -----------------------------------
// `sub0llm autotemp` picks the sampling temperature whose GENERATIONS are as
// in-distribution as real held-out text -- both judged by the model itself, so the
// loop is self-validating with no external grader. The held-out tail has perplexity
// exp(val_nelbo) under the model (its genuine uncertainty on text it never trained
// on). We generate at a candidate temperature and measure the chosen tokens' own
// perplexity, scored under the model's TRUE (T=1) distribution -- the same measure as
// real-text perplexity, so the two are directly comparable:
//   * greedy / low temp  -> self-perplexity far BELOW target: the model walks its own
//                           high-probability path and degenerates into repetition;
//   * high temp          -> self-perplexity far ABOVE target: increasingly random.
// Self-perplexity rises monotonically with temperature, so we bisect to the crossing.
// We MATCH the target, never minimise it -- minimising drives T->0 and straight back
// into the repetition failure. This is the coherence analogue of the throughput
// `tune` stage: same "search a knob against a self-measured objective" shape.
namespace {

struct GenStats { double ppl = 0.0; double rep4 = 0.0; };

// Generate from several real prefixes in the held-out tail and return (a) the sampled
// tokens' mean perplexity under the model's true (T=1) distribution and (b) the 4-gram
// repeat rate of the generations (the degeneration signal). A FIXED local RNG seed gives
// common random numbers across temperatures, so the perplexity-vs-temperature curve is
// smooth enough to bisect cleanly. Sampling uses temperature+top-k (the real gen path);
// scoring uses the full T=1 softmax so the number is comparable to real-text perplexity.
//
// Fast path: the same KV-cache decode gen_stage.cpp/preview_at use (sub0/decode.hpp) per seed, via
// kv_decode_generate's on_token hook -- it fires once per token actually emitted (never for eos_id),
// handing back the exact logits row it was sampled from, so the NLL below is scored from that same
// row instead of a second forward pass. `use_gpu` selects a GPU session the caller already brought up
// (compute_raw_metrics shares ONE across the whole temperature grid -- this runs n_seeds times PER
// temperature, so re-enabling per seed would re-pay CUDA init/upload/graph-capture ~10x over).
GenStats gen_self_stats(sub0::TokView data, std::size_t val_start,
                        float temp, int topk, int n_seeds, int gen_len, unsigned cr_seed, bool use_gpu) {
    const std::size_t last = data.size() - SEQ_LEN - 1;
    const std::size_t span = (last > val_start) ? last - val_start : 0;
    const int prefix_len = std::max(1, std::min(SEQ_LEN / 4, 16));
    std::mt19937 rng(cr_seed);                          // common random numbers across temps
    const int eos_id = sub0::eos_token_id();             // same learned stop signal as gen/preview_at

    double nll_sum = 0.0, rep_sum = 0.0; long nll_n = 0;
    std::vector<int> gen; gen.reserve(static_cast<std::size_t>(gen_len));
    // Surprise of the chosen token under the TRUE (T=1) model distribution. A natural EOS never
    // reaches here (kv_decode_generate/the fallback loop below both stop before calling this for
    // eos_id) -- sampling past it would score an out-of-distribution continuation (no document ever
    // trains on "what comes after EOS") as if it were representative degeneration/repetition at this
    // temperature, corrupting the exact statistic this tuner calibrates against.
    auto score_token = [&](const float* row, int tok) {
        float mx = -1e30f; for (int j = 0; j < VOCAB; ++j) mx = std::max(mx, row[j]);
        double Z = 0.0; for (int j = 0; j < VOCAB; ++j) Z += std::exp(static_cast<double>(row[j] - mx));
        nll_sum += -(static_cast<double>(row[tok] - mx) - std::log(Z));
        ++nll_n;
        gen.push_back(tok);
    };
    for (int k = 0; k < n_seeds; ++k) {
        std::size_t s = val_start +
            (n_seeds > 1 ? static_cast<std::size_t>(k) * span / static_cast<std::size_t>(n_seeds - 1) : 0);
        if (s > last) s = last;
        std::vector<int> ctx(static_cast<std::size_t>(prefix_len));   // materialize the seed prefix (uint16-packed view)
        data.copy_to(s, static_cast<std::size_t>(prefix_len), ctx.data());
        gen.clear();
        bool fast = false;
        if constexpr (!USE_TERNARY) {
            if (static_cast<int>(ctx.size()) + gen_len <= SEQ_LEN) {
                sub0::kv_decode_generate(ctx, gen_len, temp, topk, rng, eos_id, use_gpu, score_token);
                fast = true;
            }
        }
        if (!fast) {
            for (int g = 0; g < gen_len; ++g) {
                const int T = std::min(static_cast<int>(ctx.size()), SEQ_LEN);
                sub0::graph_reset();
                sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
                const float* row = logits->data.data() + static_cast<std::size_t>(logits->rows - 1) * VOCAB;
                const int tok = sub0::sample_token(row, temp, topk, rng);
                if (tok == eos_id) break;
                score_token(row, tok);
                ctx.push_back(tok);
            }
        }
        rep_sum += ngram_repeat(gen.data(), static_cast<int>(gen.size()));
    }
    sub0::graph_reset();
    GenStats out;
    out.ppl  = std::exp(nll_sum / static_cast<double>(std::max(1L, nll_n)));
    out.rep4 = n_seeds > 0 ? rep_sum / static_cast<double>(n_seeds) : 0.0;
    return out;
}

// Real-text 4-gram repeat measured EXACTLY like the generation metric: averaged over
// the same number of equal-length windows spread across the held-out tail. Measuring
// real text over one long contiguous span instead would inflate it (names/phrases recur
// across many sentences), making the gen-vs-real comparison apples-to-oranges.
double real_windowed_repeat(sub0::TokView data, std::size_t val_start,
                            int n_seeds, int win) {
    if (data.size() <= static_cast<std::size_t>(win) + 1) return 0.0;
    const std::size_t last = data.size() - 1 - static_cast<std::size_t>(win);
    const std::size_t span = (last > val_start) ? last - val_start : 0;
    double sum = 0.0;
    std::vector<int> wbuf(static_cast<std::size_t>(win));           // materialize the window (uint16-packed view)
    for (int k = 0; k < n_seeds; ++k) {
        std::size_t s = val_start +
            (n_seeds > 1 ? static_cast<std::size_t>(k) * span / static_cast<std::size_t>(n_seeds - 1) : 0);
        if (s > last) s = last;
        data.copy_to(s, static_cast<std::size_t>(win), wbuf.data());
        sum += ngram_repeat(wbuf.data(), win);
    }
    return n_seeds > 0 ? sum / static_cast<double>(n_seeds) : 0.0;
}

using sub0::coherence::Crossing;       // monotone-crossing interpolation (see coherence.hpp)
using sub0::coherence::interp_cross;

}  // namespace

// Shared by `report`, `autotemp`, and `models --refresh`: the RAW measurements that require actually
// running the model (a forward pass over held-out data, and -- if `want_autotemp_grid` -- a full
// generation sweep). See evalcache.hpp's own header comment for why this is split from the DERIVED
// values (bits/byte, tokens/param, autotemp's interpolated crossing temperatures) every caller
// recomputes fresh from these numbers instead of storing them too. Assumes `sub0::build_model()` was
// already called by the caller (so `models --refresh` can call it once and loop over many models);
// loads `model_in` itself. `EvalMetrics.steps` stays at its default -1 if the load fails -- callers
// check `sub0::load_model`'s own return directly rather than relying on that as the failure signal,
// since -1 is also a perfectly valid "never measured" value elsewhere in this cache.
// `session` routes the two NELBO evals (device when it has one, CPU otherwise). The autotemp grid
// below brings up its OWN DecodeSession and tears it down again, so `session` must not be used after
// this returns -- see sub0::eval::Session's note on overlapping bring-ups.
static sub0::evalcache::EvalMetrics compute_raw_metrics(sub0::TokView data, std::size_t val_start,
                                                         bool want_autotemp_grid, unsigned seed,
                                                         const sub0::eval::Session& session) {
    sub0::evalcache::EvalMetrics m;
    const sub0::TokView train_span = data.first(val_start);
    const sub0::TokView val_span   = data.subspan(val_start);
    m.train_nelbo = evaluate(train_span, 0, session);
    m.val_nelbo   = evaluate(val_span, 0, session);
    {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(sub0::default_corpus(), ec);
        if (!ec && !data.empty()) m.bytes_per_tok = static_cast<double>(sz) / static_cast<double>(data.size());
    }
    if (want_autotemp_grid) {
        constexpr int AUTOTEMP_TOPK = 20;
        constexpr int N_SEEDS = 10, GEN_LEN = 96;   // long windows so repetition/looping is visible
        m.target_ppl = std::exp(mean_entropy(data, val_start));
        m.target_rep = real_windowed_repeat(data, val_start, N_SEEDS, GEN_LEN);
        m.grid.temp = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.4f};
        m.grid.gen_ppl.resize(m.grid.temp.size());
        m.grid.rep4.resize(m.grid.temp.size());
        // One GPU decode session shared across the WHOLE grid (11 temps x N_SEEDS seeds each --
        // re-enabling per seed would re-pay CUDA init/upload/graph-capture ~110x over, see decode.hpp).
        sub0::DecodeSession sess;
        for (std::size_t i = 0; i < m.grid.temp.size(); ++i) {
            const GenStats g = gen_self_stats(data, val_start, m.grid.temp[i], AUTOTEMP_TOPK,
                                              N_SEEDS, GEN_LEN, seed, sess.use_gpu);
            m.grid.gen_ppl[i] = g.ppl; m.grid.rep4[i] = g.rep4;
        }
    }
    return m;
}

extern "C" SUB0_API int sub0_autotemp_stage(const char* model_in, unsigned seed, int verbose) {
    constexpr int AUTOTEMP_TOPK = 20;   // matches gen's default; also used by compute_raw_metrics' own sweep
    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const sub0::TokView data = load_corpus_tokens(tok, od_buf, "autotemp");
    if (data.empty()) return 1;

    sub0::build_model();
    if (!sub0::load_model(model_in)) {
        sub0::log::error("autotemp: cannot load model '{}'", model_in);
        return 1;
    }
    // Not just cosmetic here: gen_self_stats' EOS-stop check depends on eos_token_id(), which
    // returns -1 with no tokenizer loaded -- exactly the bug models --refresh had (fixed 4bf29fc)
    // before this same check was added here. A silent failure would corrupt the whole measurement,
    // not just the printed sample, so this fails loudly instead of warning-and-continuing.
    if (!sub0::load_tokenizer(sub0::default_tokenizer())) {
        sub0::log::error("autotemp: cannot load this build's tokenizer -- the EOS-stop signal "
                         "would be wrong for the whole measurement, not just the printed sample");
        return 1;
    }

    // The SAME held-out split the trainer uses, so the target perplexity is measured on
    // text the model never trained on.
    const std::size_t val_tokens = std::max<std::size_t>(
        static_cast<std::size_t>(SEQ_LEN) + 2,
        static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(data.size())));
    const std::size_t val_start = data.size() - val_tokens;

    // autotemp is the EXPENSIVE one: an 11-temperature x 10-seed generation grid, far heavier than
    // report's bounded NELBO eval. Running it against a live trainer distorts both measurements.
    if (trainer_active())
        sub0::log::warn("autotemp: a sub0llm-train is running -- the temperature grid is a heavy "
                        "generation sweep and will contend with it. Prefer running this when idle.");
    const sub0::eval::Session eval_sess(/*allow=*/!trainer_active());
    const sub0::evalcache::EvalMetrics raw =
        compute_raw_metrics(data, val_start, /*want_autotemp_grid=*/true, seed, eval_sess);
    const double ce_ppl     = std::exp(raw.val_nelbo);   // headline (inflated by model error) -- exp(val_nelbo), not a separate measurement
    const double target_ppl = raw.target_ppl;
    const double target_rep = raw.target_rep;

    std::println("autotemp: model {}", model_in);
    std::println("held-out (real text): cross-entropy perplexity {:.2f} | reading entropy {:.2f} | 4-gram repeat {:.1f}%",
                 ce_ppl, target_ppl, 100.0 * target_rep);
    std::fflush(stdout);

    // The grid IS the evidence (already computed above); this loop just prints it. gen_ppl rises
    // with temperature, 4-gram repeat falls, so each target has exactly one crossing -- recovered by
    // interpolation below (no noisy bisection, and both anchors from the one sweep already run).
    const std::vector<float>& grid = raw.grid.temp;
    const std::vector<double>& ppls = raw.grid.gen_ppl;
    const std::vector<double>& reps = raw.grid.rep4;
    if (verbose) {
        std::println("  {:>6}   {:>10}   {:>10}", "temp", "gen_ppl", "4gram-rep");
        for (std::size_t i = 0; i < grid.size(); ++i) {
            std::println("  {:>6.2f}   {:>10.3f}   {:>9.1f}%{}{}", grid[i], ppls[i], 100.0 * reps[i],
                         ppls[i] >= target_ppl ? "  ppl>=tgt" : "", reps[i] <= target_rep ? "  rep<=tgt" : "");
            std::fflush(stdout);
        }
    }

    // Cache this run's raw numbers, preserving whatever report_stage already wrote (train/val
    // NELBO, bytes_per_tok) rather than overwriting those fields with autotemp's own defaults.
    if (model_in && *model_in) {
        const std::filesystem::path metrics_dir = std::filesystem::path(model_in).parent_path();
        sub0::registry::ModelMeta meta;
        sub0::registry::read_state(metrics_dir, meta);   // for the steps stamp; ok if this fails (stays 0)
        sub0::evalcache::EvalMetrics cached;
        sub0::evalcache::read_metrics(metrics_dir, cached);   // ok if this returns false (nothing cached yet)
        cached.steps = meta.steps;
        cached.measured_at = sub0::registry::now_iso();
        cached.target_ppl = raw.target_ppl; cached.target_rep = raw.target_rep; cached.grid = raw.grid;
        sub0::evalcache::write_metrics(metrics_dir, cached);
    }

    const Crossing by_ppl = interp_cross(grid, ppls, target_ppl, /*increasing=*/true);
    const Crossing by_rep = interp_cross(grid, reps, target_rep, /*increasing=*/false);

    // Repetition is the robust anchor (a surface property of real text, independent of
    // the model's calibration); perplexity is the principled cross-check. Both match by
    // pushing temperature UP for this model, because it is under-fit -- it is MORE
    // repetitive and MORE over-confident than real text at every coherent temperature.
    //
    // We never recommend hotter than the training-natural ceiling temp 1.0: cross-entropy
    // training fits the model's distribution at temp 1, so sampling above it only helps a
    // genuinely well-calibrated model. When BOTH matches sit above 1.0 the model cannot
    // reach real-text diversity without losing coherence (heat turns repetition into
    // gibberish) -- that gap is the under-fit signal, and the fix is capacity/training,
    // not a hotter temperature. For a well-calibrated model the matches fall below 1.0 and
    // the cap does not bind, so autotemp returns the true crossing.
    constexpr float TEMP_CEILING = 1.0f;
    const float rec = std::min(by_rep.temp, TEMP_CEILING);
    const bool underfit = by_rep.temp > TEMP_CEILING && by_ppl.temp > TEMP_CEILING;

    std::mt19937 prng(seed ^ 0x9e3779b9u);
    std::println("");
    std::println("matched temperatures (both = 'as in-distribution as real text'):");
    std::println("  by repetition (gen 4-gram repeat -> real {:.1f}%):   temp {:.2f}{}",
                 100.0 * target_rep, by_rep.temp, by_rep.hit ? "" : " (off-grid)");
    std::println("  by perplexity (gen perplexity -> reading entropy):   temp {:.2f}{}",
                 by_ppl.temp, by_ppl.hit ? "" : " (off-grid)");
    if (underfit)
        std::println("  diagnosis: both matches exceed the training-natural ceiling 1.0 -- the model is\n"
                     "             under-fit (more repetitive AND less confident than real text), so it\n"
                     "             cannot match real-text diversity without losing coherence. Capping at\n"
                     "             1.0; the real lever is capacity/training, not temperature.");
    std::println("");
    std::println("recommended --temp {:.2f}", rec);
    sub0::DecodeSession preview_sess;
    std::println("  --- sample @ temp {:.2f} ---\n  {}", rec,
                 preview_at("the ", 80, rec, AUTOTEMP_TOPK, prng, preview_sess.use_gpu));
    std::println("apply with:  sub0llm gen {} \"<prompt>\" --temp {:.2f}", model_in, rec);
    return 0;
}

// --- Memory plan --------------------------------------------------------------
// `sub0llm memplan` reports the predicted device footprint for generation and training using the
// pure model (sub0/memplan.hpp), broken down by component, with a batch sweep vs the VRAM budget
// and guidance on the knobs that move it. No device needed -- this is the same prediction the
// configurator and tuner gate on, surfaced for planning a model/batch/context that fits.
extern "C" SUB0_API int sub0_memplan_stage() {
    namespace mp = sub0::memplan;
    // Every field, not the first six. Left short, this reported a memory plan for a DIFFERENT
    // architecture than the build: tied/qk_norm/gated/pos_emb/n_kv_heads/exec_layers all fell back to
    // their defaults, so a gated+tied+qk-norm RoPE build (i.e. the production shape) was costed as an
    // untied, non-gated, absolute-position MHA one. The tail three are new, but the first three were
    // already wrong -- this is the aggregate-init hazard the roadmap tracks under current_build_dims().
    const mp::Dims d = sub0::current_build_dims();
    auto mib = [](unsigned long long b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    const int vram = GPU_VRAM_MB;
    std::println("memory plan: d{} L{} H{} ff{} seq{} v{} | params {:.1f}M | VRAM {} MiB (+{} shared)",
                 D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
                 mp::param_floats(d) / 1e6, vram, GPU_SHARED_MEM_MB);
    std::println("precision: GEMM {} | activations {} | master FP32  (BF16_OK={}; FFN scratch BF16-stored, parity gated by direction)",
                 GEMM_DTYPE == Dtype::BF16 ? "BF16" : "F32", ACT_DTYPE == Dtype::BF16 ? "BF16" : "F32", BF16_OK);

    const double persist = mib(mp::persistent_bytes(d, ACT_DTYPE == Dtype::BF16 ? 2 : 4));
    std::println("");
    std::println("persistent (resident, batch-independent): {:.0f} MiB", persist);
    std::println("  params + grad + m + vel + decay (5x) + fused QKV weights{}",
                 ACT_DTYPE == Dtype::BF16 ? " + BF16 weight mirrors" : "");
    std::println("generation (batch 1): persistent + fwd scratch = {:.0f} MiB",
                 persist + mib(mp::fwd_scratch_bytes(d, 1)));

    std::println("");
    std::println("training footprint by batch (persistent + dids + train scratch):");
    for (const int b : {32, 64, 128, 256, 512, 1024}) {
        const auto aB = static_cast<sub0::memplan::u64>(ACT_DTYPE == Dtype::BF16 ? 2 : 4);
    const int need = mp::train_resident_mb(d, b, aB);
        std::println("  batch {:>5}: {:>7} MiB  {}", b, need,
                     vram <= 0 ? "" : need > vram ? "!! over VRAM (spills to shared, ~10x slower)" : "fits");
    }
    const int db = DEFAULT_GPU_BATCH;
    std::println("");
    std::println("breakdown @ DEFAULT_GPU_BATCH={}: persistent {:.0f} | dids {:.0f} | train-scratch {:.0f} = {:.0f} MiB",
                 db, persist, mib(mp::fwd_dids_bytes(d, db)), mib(mp::train_scratch_bytes(d, db, ACT_DTYPE == Dtype::BF16 ? 2 : 4)),
                 mib(mp::train_resident_bytes(d, db, ACT_DTYPE == Dtype::BF16 ? 2 : 4)));
    std::println("");
    std::println("knobs: train scratch ~ batch * seq (acts) ; attention is O(seq^2); params ~ d^2*layers.");
    std::println("  - halve seq -> ~halve activations + 4x less attention; raise/lower batch scales linearly;");
    std::println("  - BF16 activations would ~halve the train scratch; smaller d/layers cuts params + acts.");
    return 0;
}

// --- Model fitness report ----------------------------------------------------
// `sub0llm report [model]` interrogates how well the baked architecture is sized for its corpus
// and prints per-knob guidance for a retrain. It combines three signals that, together, separate
// the usual failure modes:
//   * train-vs-val gap     -> overfitting (large) vs capacity/optimization bound (small);
//   * bits-per-byte        -> absolute quality, normalized by tokenization (comparable across vocab);
//   * tokens-per-parameter -> Chinchilla compute-optimal sizing (~20 tokens/param is balanced).
// Small gap + high bits/byte + many tokens/param => "too small for the data" (grow capacity);
// a large gap => "too big / too little data" (shrink, or add data/regularization). Per-knob rules
// (head_dim, depth/width aspect, vocab compression, context) then point at WHICH knob to move. The
// loss signals need a model that loads into THIS build; without one, structural guidance still
// applies (it is a property of the dims and the corpus, not the weights).
namespace {
const char* verdict_word(int level) {   // 0 ok, 1 could, 2 should, 3 must
    switch (level) { case 3: return "MUST  "; case 2: return "SHOULD"; case 1: return "could "; default: return "ok    "; }
}
}  // namespace

extern "C" SUB0_API int sub0_report_stage(const char* model_in) {
    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const sub0::TokView data = load_corpus_tokens(tok, od_buf, "report");
    if (data.empty()) return 1;
    sub0::build_model();

    // Tee every report line to stdout AND a buffer, so the same text can be persisted next to the
    // model for retrospective per-version comparison (no rebuild/rerun needed to recall the metrics).
    std::string report_txt;
    auto emit = [&]<class... A>(std::format_string<A...> fmt, A&&... a) {
        std::string line = std::format(fmt, std::forward<A>(a)...);
        std::puts(line.c_str());
        report_txt += line;
        report_txt += '\n';
    };

    // The held-out split mirrors training (last VAL_FRACTION); train metrics use the head.
    const std::size_t val_tokens = std::max<std::size_t>(
        static_cast<std::size_t>(SEQ_LEN) + 2,
        static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(data.size())));
    const std::size_t val_start = data.size() - val_tokens;
    const sub0::TokView train_span = data.first(val_start);   // val_span lives inside compute_raw_metrics now

    const long long params       = static_cast<long long>(sub0::trainable_floats());
    const int       head_dim     = D_MODEL / N_HEADS;
    const double    aspect       = static_cast<double>(D_MODEL) / N_LAYERS;
    const long long train_tokens = static_cast<long long>(train_span.size());

    // bytes/token from the raw corpus size (approximate: raw vs normalized bytes differ slightly).
    double bytes_per_tok = 0.0;
    {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(sub0::default_corpus(), ec);
        if (!ec && !data.empty()) bytes_per_tok = static_cast<double>(sz) / static_cast<double>(data.size());
    }

    emit("model report");
    emit("architecture: d_model {} | n_layers {} | n_heads {} (head_dim {}) | seq_len {} | vocab {} | pos {}",
                 D_MODEL, N_LAYERS, N_HEADS, head_dim, SEQ_LEN, VOCAB,
                 POS_ENCODING == PosEncoding::Rope ? "RoPE" : "absolute");
    emit("parameters:   {:.2f}M trainable floats", params / 1e6);

    if (model_in && *model_in) {   // training provenance from the state.json next to the model
        sub0::registry::ModelMeta m;
        if (sub0::registry::read_state(std::filesystem::path(model_in).parent_path(), m) && m.steps > 0)
            emit("training:     steps {} | epochs {:.2f} | tokens_seen {} | seed {}",
                         m.steps, m.epochs, m.tokens_seen, m.seed);
    }

    // Loss metrics need a loaded model. Skip gracefully if it cannot load (e.g. a model trained
    // under a different config/scheme -- the header guard refuses it), keeping structural guidance.
    bool   have_loss = false, overfit = false, underfit_quality = false;
    double rel_gap = 0.0;
    bool   model_loaded = false;
    if (model_in && *model_in) {
        // A running trainer owns the machine's throughput budget; say so BEFORE doing the work, so a
        // slow report reads as "deliberately sharing" rather than "hung". `report` already skips the
        // autotemp grid (the expensive generation sweep) -- what remains is the bounded fixed-window
        // NELBO eval, which is cheap but not free.
        if (trainer_active())
            sub0::log::warn("report: a sub0llm-train is running -- this shares its CPU/GPU and will "
                            "depress the numbers in BOTH. Re-run when idle for a clean measurement.");
        if (sub0::load_model(model_in)) {
            model_loaded = true;
            // ONE device session for every NELBO number below (both splits + every context width).
            // Declined outright while a trainer holds the machine -- see sub0::eval::Session. Scoped
            // to this block so it is torn down before the sample battery brings up its own
            // DecodeSession further down: the two are independent bring-ups of the same context, and
            // the later one's teardown would leave this one pointing at a dead device.
            const sub0::eval::Session eval_sess(/*allow=*/!trainer_active());
            // Through emit(), not the log: WHICH backend produced these numbers is provenance, so it
            // belongs in the saved report.txt beside them, not only on the console of the run that
            // happened to produce it.
            if (eval_sess.use_device)
                emit("scoring:      {} device backend", sub0_dev_caps().name);
            else
                emit("scoring:      CPU ({}) -- same numbers, but slow enough that the "
                     "context-length curve is shortened to its endpoints", eval_sess.declined);
            const sub0::evalcache::EvalMetrics raw =
                compute_raw_metrics(data, val_start, /*want_autotemp_grid=*/false, 0, eval_sess);
            const double train_nelbo = raw.train_nelbo;
            const double val_nelbo   = raw.val_nelbo;
            const double gap = val_nelbo - train_nelbo;
            rel_gap = val_nelbo > 0 ? gap / val_nelbo : 0.0;
            const double bpb = bytes_per_tok > 0 ? val_nelbo / (0.6931471805599453 * bytes_per_tok) : 0.0;
            overfit          = rel_gap > 0.15;
            underfit_quality = bpb > 1.2;     // clean web text should approach ~1.0 bits/byte
            have_loss = true;
            emit("");
            emit("quality:");
            emit("  train nelbo {:.4f}  (ppl {:.2f})", train_nelbo, std::exp(train_nelbo));
            emit("  val   nelbo {:.4f}  (ppl {:.2f})", val_nelbo, std::exp(val_nelbo));
            // Context-length curve: how much the model actually GAINS from longer context. Mean
            // nelbo above cannot show this -- it averages every position together. For comparing two
            // architectures the shape here is the more informative number: a model exploiting
            // long-range structure improves markedly from the shortest width to SEQ_LEN; one that has
            // effectively learned an n-gram barely moves.
            {
                const ContextCurve cc = evaluate_context_curve(data, val_start, eval_sess);
                if (cc.nelbo.size() == cc.width.size() && !cc.nelbo.empty()) {
                    emit("  context-length curve (val nelbo by how much context the model may attend to){}:",
                         cc.width.size() <= 2 ? " [SHORTENED: CPU-only eval; endpoints only]" : "");
                    for (std::size_t i = 0; i < cc.width.size(); ++i)
                        emit("    ctx {:>4}: nelbo {:.4f}  (ppl {:.2f})",
                             cc.width[i], cc.nelbo[i], std::exp(cc.nelbo[i]));
                    const double gain = cc.nelbo.front() - cc.nelbo.back();
                    emit("    gain {:>4} -> {:<4}: {:.4f} nelbo ({:.1f}%) -- higher = better use of long context",
                         cc.width.front(), cc.width.back(), gain,
                         cc.nelbo.front() > 0 ? 100.0 * gain / cc.nelbo.front() : 0.0);
                }
            }
            // LoopSplit diagnostic: does each repeated pass still DO anything? A looped block applies
            // the same function repeatedly, so its passes differ only by their input; repeated
            // application of a contractive map converges to a fixed point, at which point the extra
            // executions cost throughput and buy no quality. That is the leading explanation for
            // looping under-delivering, and it is directly measurable rather than inferred: compare
            // what pass 2 adds to the residual stream against what pass 1 added, for the SAME layers.
            if constexpr (sub0::LOOP_EXEC_COUNT > N_LAYERS) {
                std::vector<float> dlt(sub0::LOOP_EXEC_COUNT, 0.0f), hn(sub0::LOOP_EXEC_COUNT, 0.0f);
                std::vector<int>   win(static_cast<std::size_t>(SEQ_LEN) + 1);
                const sub0::eval::WindowSet ws = sub0::eval::plan(data, val_start, EVAL_WINDOWS_MAX);
                // Averaged over several held-out windows: a single window's norms are noisy enough to
                // read a trend into that is not there.
                const int nprobe = std::min(8, std::max(1, ws.count));
                for (int w = 0; w < nprobe; ++w) {
                    data.copy_to(ws.start_of(w), static_cast<std::size_t>(SEQ_LEN) + 1, win.data());
                    std::vector<float> d1(sub0::LOOP_EXEC_COUNT), h1(sub0::LOOP_EXEC_COUNT);
                    sub0::graph_reset();
                    sub0::loop_pass_stats(win.data(), SEQ_LEN, d1.data(), h1.data());
                    for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) { dlt[e] += d1[e]; hn[e] += h1[e]; }
                }
                sub0::graph_reset();
                emit("");
                emit("loop diagnostic (is each repeated pass still doing work?):");
                emit("    {:>4}  {:>5}  {:>10}  {:>10}", "exec", "layer", "|dh|", "|dh|/|h|");
                for (int e = 0; e < sub0::LOOP_EXEC_COUNT; ++e) {
                    const double d = dlt[e] / nprobe, hh = hn[e] / nprobe;
                    emit("    {:>4}  {:>5}  {:>10.3f}  {:>10.4f}", e, sub0::LAYER_EXEC_ORDER[e], d,
                         hh > 0.0 ? d / hh : 0.0);
                }
                // Head/tail run once, so only the MIDDLE block's repeats are comparable. Ratio of the
                // last pass's total contribution to the first's: ~1.0 = every pass still contributing,
                // well under 1.0 = converging toward a fixed point (what depth attention targets).
                constexpr int kHead = (N_LAYERS - LOOP_MIDDLE_LAYERS) / 2;
                double first = 0.0, last = 0.0;
                for (int i = 0; i < LOOP_MIDDLE_LAYERS; ++i) {
                    first += dlt[kHead + i];
                    last  += dlt[kHead + (LOOP_REPEATS - 1) * LOOP_MIDDLE_LAYERS + i];
                }
                const double ratio = first > 0.0 ? last / first : 0.0;
                emit("    middle-block contribution, pass {} / pass 1: {:.3f}  -> {}", LOOP_REPEATS, ratio,
                     ratio > 0.85 ? "every pass still contributes (NOT a fixed point)"
                     : ratio > 0.5 ? "later passes contribute less -- partial convergence"
                                   : "STRONG convergence: the repeats are nearly inert");
            }
            emit("  train/val gap {:.4f}  ({:.1f}%)  -> {}", gap, 100.0 * rel_gap,
                         overfit ? "OVERFITTING (model too large / too little data)"
                                 : rel_gap < 0.03 ? "not overfitting (capacity / optimization bound)"
                                                  : "healthy generalization gap");
            if (bytes_per_tok > 0)
                emit("  bits/byte   {:.3f}  (corpus ~{:.2f} bytes/token)  -> {}", bpb, bytes_per_tok,
                             bpb > 1.2 ? "well above ~1.0: real headroom"
                                       : bpb > 0.9 ? "near a good small-model range" : "strong");

            // Cache this run's raw numbers, preserving whatever autotemp_stage already wrote
            // (target_ppl/target_rep/grid) rather than overwriting those fields with report's defaults.
            {
                const std::filesystem::path metrics_dir = std::filesystem::path(model_in).parent_path();
                sub0::registry::ModelMeta meta;
                sub0::registry::read_state(metrics_dir, meta);   // for the steps stamp; ok if this fails (stays 0)
                sub0::evalcache::EvalMetrics cached;
                sub0::evalcache::read_metrics(metrics_dir, cached);   // ok if this returns false (nothing cached yet)
                cached.steps = meta.steps;
                cached.measured_at = sub0::registry::now_iso();
                cached.train_nelbo = train_nelbo; cached.val_nelbo = val_nelbo; cached.bytes_per_tok = bytes_per_tok;
                sub0::evalcache::write_metrics(metrics_dir, cached);
            }
        } else {
            emit("note: '{}' did not load into this build (different config/scheme); "
                         "showing structural guidance only.", model_in);
        }
    } else {
        emit("note: no model given -- showing structural (corpus-fit) guidance only.");
    }

    // Grounding samples: actual generations at a fixed grid of (prompt, temperature) so a reader --
    // and, more importantly, a LATER cross-model-size comparison reading the saved report.txt files
    // side by side -- can judge quality directly under identical conditions, not just from the loss
    // numbers. Two prompts (the training preview's own "the ", and the common story-opening "Once upon
    // a time" also used in this project's own data/tinystories_findings.txt reference) x three
    // temperatures (0.4 near-deterministic, 0.7, and 0.8 -- gen's own real default, cli_stages.hpp's
    // gen_temp) -- top-k 20 matches gen's default throughout. One shared, fixed-seed RNG stream across
    // the whole grid (not independently reseeded per cell) keeps the WHOLE battery reproducible as a
    // unit, matching this function's existing convention. Full SEQ_LEN length on every sample, even
    // though that costs real CPU time on a big model, so every cell exercises the trained context
    // window identically -- consistency across the grid matters more than saving time here.
    if (model_loaded) {
        // Not fatal to the report as a whole (the NELBO/bits-per-byte quality numbers above are
        // already computed and printed) -- only these samples would come out garbled.
        if (!sub0::load_tokenizer(sub0::default_tokenizer()))
            sub0::log::warn("report: cannot load tokenizer -- the sample section below will be garbled");
        std::mt19937 rng(1234);                            // fixed seed -> reproducible report samples
        emit("");
        emit("samples ({}-token context, fixed seed 1234, top-k 20):", SEQ_LEN);
        // One GPU decode session shared across the whole 2-prompt x 3-temp battery (6 generations) --
        // re-enabling per generation would re-pay CUDA init/upload/graph-capture 6x over, see decode.hpp.
        sub0::DecodeSession sess;
        for (const char* prompt : {"the ", "Once upon a time"}) {
            emit("  prompt \"{}\":", prompt);
            for (float temp : {0.4f, 0.7f, 0.8f}) {
                emit("    [temp {:.1f}]  {}", temp, preview_at(prompt, SEQ_LEN, temp, 20, rng, sess.use_gpu));
            }
        }
    }

    // Same target ratio autosize() sizes fresh corpora against (sub0::config::kTokensPerParam,
    // config_util.hpp) -- NOT raw Chinchilla-optimal (20:1); see that constant's own doc comment for
    // why this project targets well above pure compute-optimal. The healthy band is a 0.25x-2x spread
    // around it, the same multiplicative width the old hardcoded 5-40 band used around its (then) 20.
    constexpr double kDataRichMul = 2.0, kDataLimitedMul = 0.25;
    const double tok_per_param = params > 0 ? static_cast<double>(train_tokens) / params : 0.0;
    emit("");
    emit("corpus fit (target ~{:.0f} tokens/param -- see sub0::config::kTokensPerParam):", sub0::config::kTokensPerParam);
    emit("  train tokens  {:.2f}M", train_tokens / 1e6);
    emit("  tokens/param  {:.1f}  -> {}", tok_per_param,
                 tok_per_param > sub0::config::kTokensPerParam * kDataRichMul
                     ? "data-rich: the model is UNDERSIZED for this corpus"
                 : tok_per_param < sub0::config::kTokensPerParam * kDataLimitedMul
                     ? "data-limited: model may be oversized / undertrained"
                     : "near the target ratio");

    // Grow capacity unless we are actually overfitting.
    const bool grow = !overfit && (tok_per_param > sub0::config::kTokensPerParam * kDataRichMul || underfit_quality);

    int v_d = 0, v_l = 0, v_h = 0, v_seq = 0, v_vocab = 0;
    std::string r_d, r_l, r_h, r_seq, r_vocab;
    // Head guidance: prefer RAISING d_model (keeping the head count) over lowering n_heads. Measured
    // on tinystories: dropping 4 heads -> 2 to reach head_dim 64 (d128 H2) converged WORSE (val 2.44)
    // than the smaller 4-head d96 (1.985); widening to 4x40 (d160 H4) won outright (1.461). Head count
    // matters more than head_dim here, so do NOT trade heads away for head_dim.
    if (head_dim < 32)       { v_h = 3; r_h = std::format("head_dim {} < 32: raise d_model, KEEP heads (dropping heads to widen head_dim measured worse)", head_dim); }
    else if (head_dim < 48)  { v_h = 2; r_h = std::format("head_dim {} < 48: raise d_model (keep heads)", head_dim); }
    else if (head_dim < 64)  { v_h = 1; r_h = std::format("head_dim {} just below 64 (fine; raise d_model if growing)", head_dim); }
    else if (head_dim > 160) { v_h = 1; r_h = std::format("head_dim {} large; more heads would add expressiveness", head_dim); }
    else                     { r_h = std::format("head_dim {} in the healthy 64-128 range", head_dim); }
    if (grow)                { v_d = head_dim < 64 ? 3 : 2; r_d = "capacity-bound: widen the residual stream (primary lever)"; }
    else if (overfit)        { v_d = 1; r_d = "could narrow to curb overfitting"; }
    else                     { r_d = "balanced for the current fit"; }
    if (grow && aspect > 48) { v_l = 2; r_l = std::format("aspect d/L={:.0f} high (shallow): add depth", aspect); }
    else if (grow)           { v_l = 1; r_l = std::format("aspect d/L={:.0f} ok; depth is a secondary lever", aspect); }
    else if (aspect < 8)     { v_l = 1; r_l = std::format("aspect d/L={:.0f} low (deep & thin)", aspect); }
    else                     { r_l = std::format("aspect d/L={:.0f} reasonable", aspect); }
    if (SEQ_LEN < 128)       { v_seq = 1; r_seq = "short context; RoPE extends cleanly to longer windows"; }
    else                     { r_seq = "adequate context"; }
    if (bytes_per_tok > 0 && bytes_per_tok < 2.5) { v_vocab = 1; r_vocab = std::format("bytes/token {:.2f} low: a larger vocab compresses better (cost: embedding params)", bytes_per_tok); }
    else                                          { r_vocab = "compression reasonable"; }

    emit("");
    emit("per-knob guidance (must / should / could / ok):");
    emit("  d_model  {:<5} [{}] {}", D_MODEL,  verdict_word(v_d),     r_d);
    emit("  n_layers {:<5} [{}] {}", N_LAYERS, verdict_word(v_l),     r_l);
    emit("  n_heads  {:<5} [{}] {}", N_HEADS,  verdict_word(v_h),     r_h);
    emit("  seq_len  {:<5} [{}] {}", SEQ_LEN,  verdict_word(v_seq),   r_seq);
    emit("  vocab    {:<5} [{}] {}", VOCAB,    verdict_word(v_vocab), r_vocab);

    // Concrete next-size suggestion when growing. Two grow drivers need different targets:
    //   * data-rich (tokens/param above the data-rich threshold): size UP to
    //     ~train_tokens/kTokensPerParam params (the SAME ratio autosize() targets for a fresh corpus);
    //   * quality-bound (poor bits/byte at sane tokens/param): spend more capacity than now.
    // Use the LARGER of the two so a "grow" never suggests fewer params than the current model.
    // Strategy: KEEP the current head count (it is the proven knob -- see the head guidance above)
    // and widen d_model, which raises head_dim naturally. Search widths that are a multiple of
    // n_heads (so head_dim is integral) starting above the current d_model, depth >= current.
    if (grow && train_tokens > 0) {
        const long long ratio_target = static_cast<long long>(train_tokens / sub0::config::kTokensPerParam);
        const long long target_p     = std::max<long long>(ratio_target, static_cast<long long>(params * 1.8));
        // Same aspect-ratio curve autosize() uses (config_util.hpp), pinned to the OVERALL target like
        // autosize()'s own shape search -- not recomputed per candidate C, and not the old independently
        // -hardcoded "C/40.0" this project used before the curve was made scale-dependent and shared.
        const double    aspect     = sub0::config::aspect_for_params(static_cast<double>(target_p));
        const int       step       = std::max(N_HEADS, 16);                         // keep C a head multiple
        const int       C_start    = ((D_MODEL / step) + 1) * step;                 // next multiple up
        long long best_p = 0; int best_C = C_start, best_L = N_LAYERS;
        for (int C = C_start; C <= 2048; C += step) {
            const int L = std::clamp(std::max(N_LAYERS, static_cast<int>(std::lround(C / aspect))), 2, 24);
            const long long p = static_cast<long long>(
                sub0::memplan::param_floats({C, L, N_HEADS, 4 * C, SEQ_LEN, VOCAB}));
            if (best_p == 0 || std::llabs(p - target_p) < std::llabs(best_p - target_p)) {
                best_p = p; best_C = C; best_L = L;
            }
        }
        emit("");
        emit("suggested next size (target ~{:.1f}M params; max of ratio-target {:.1f}M and 1.8x current):",
                     target_p / 1e6, ratio_target / 1e6);
        // sub0llm-configure owns dims now, not CMake (SUB0_AUTO_CONFIGURE was removed) -- pin via its
        // CLI flags, matching the README's own documented re-configure workflow.
        emit("  sub0llm-configure --corpus <corpus> --dmodel {} --layers {} --heads {}",
                     best_C, best_L, N_HEADS);
        emit("  cmake --build --preset native");
        emit("  -> d{} L{} H{} (head_dim {}, aspect {:.0f}) ~= {:.2f}M params; then `sub0llm train`",
                     best_C, best_L, N_HEADS, best_C / N_HEADS, static_cast<double>(best_C) / best_L, best_p / 1e6);
    }

    // Persist the report next to the model so each trained version keeps its own metrics + samples
    // for later comparison without re-running (the corpus eval + generation are the slow parts).
    if (model_in && *model_in) {
        const std::filesystem::path dir = std::filesystem::path(model_in).parent_path();
        const std::filesystem::path out = dir / "report.txt";
        if (std::ofstream f{out}; f) {
            f << report_txt;
            std::println("\nsaved report -> {}", out.string());
        } else {
            std::println("\nwarning: could not write report to {}", out.string());
        }
    }
    return 0;
}

// --- Model registry ----------------------------------------------------------
// `sub0llm models` discovers every trained model (scans the state.json under the models root --
// the registry is the set of those files, so it never drifts out of sync) and flags which load
// into THIS build (matching architecture dims). `--prune` reclaims the incompatible ones, whose
// checkpoints this engine could never load anyway. Destructive, so it is opt-in via the flag.
// The SAME "recommended temperature" derivation autotemp_stage prints (rec = min(by_rep.temp,
// TEMP_CEILING)) -- reused, not reimplemented, against a model's cached grid. Returns -1 when no
// autotemp grid has been cached yet (nothing to derive from).
static double recommended_temp_from_cache(const sub0::evalcache::EvalMetrics& m) {
    if (m.grid.temp.empty() || m.target_rep < 0) return -1.0;
    constexpr float TEMP_CEILING = 1.0f;
    const Crossing by_rep = interp_cross(m.grid.temp, m.grid.rep4, m.target_rep, /*increasing=*/false);
    return std::min(static_cast<double>(by_rep.temp), static_cast<double>(TEMP_CEILING));
}

extern "C" SUB0_API int sub0_models_stage(int prune, int verbose, const char* corpus_filter,
                                          const char* sha_filter, const char* since, const char* until,
                                          int metrics, int refresh, int force) {
    (void)verbose;
    namespace reg = sub0::registry;
    namespace ec  = sub0::evalcache;
    std::vector<reg::ModelMeta> models = reg::scan(SUB0_MODELS_ROOT);
    std::println("models root: {}  ({} model{})", SUB0_MODELS_ROOT, models.size(), models.size() == 1 ? "" : "s");
    std::println("this build:  d{} l{} h{} sq{} v{}{}{}{}{}{} @ {}",
                 D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB, USE_TERNARY ? "t" : "",
                 reg::pos_tag(static_cast<int>(POS_ENCODING)), USE_GATED_FFN ? "g" : "",
                 USE_TIED_EMBEDDINGS ? "w" : "", USE_QK_NORM ? "q" : "", SUB0_GIT_SHA);
    if (models.empty()) { std::println("(none yet -- `sub0llm train` creates one)"); return 0; }

    auto loadable = [&](const reg::ModelMeta& m) {
        return reg::compatible(m, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                               static_cast<int>(USE_TERNARY), static_cast<int>(POS_ENCODING),
                               static_cast<int>(USE_GATED_FFN), static_cast<int>(USE_TIED_EMBEDDINGS),
                               static_cast<int>(USE_QK_NORM), sub0::MODEL_ARCH_ID);
    };

    // Filter BEFORE anything else, so the plain listing, --prune, --metrics, and --refresh all agree
    // on the same selected set (an empty filter arg -> no restriction on that axis). --sha is a
    // prefix match (short shas are the norm); --since/--until compare against just the DATE portion
    // of state.json's fixed-width ISO `created` string (its first 10 chars, "YYYY-MM-DD") -- comparing
    // the full "YYYY-MM-DDTHH:MM:SSZ" timestamp against a bare date would make --until wrongly
    // exclude every model created ON that date (any same-day timestamp sorts lexicographically AFTER
    // the bare date string), while --since's lower-bound direction happens to work either way.
    auto matches = [&](const reg::ModelMeta& m) {
        if (corpus_filter && *corpus_filter && m.corpus.find(corpus_filter) == std::string::npos) return false;
        if (sha_filter && *sha_filter && m.git_sha.rfind(sha_filter, 0) != 0) return false;
        const std::string created_date = m.created.substr(0, std::min<std::size_t>(10, m.created.size()));
        if (since && *since && created_date < since) return false;
        if (until && *until && created_date > until) return false;
        return true;
    };
    std::vector<reg::ModelMeta> sel;
    for (reg::ModelMeta& m : models) if (matches(m)) sel.push_back(std::move(m));
    std::sort(sel.begin(), sel.end(),
              [](const reg::ModelMeta& a, const reg::ModelMeta& b) { return a.dir.filename() < b.dir.filename(); });
    if (sel.empty() && (corpus_filter && *corpus_filter || sha_filter && *sha_filter ||
                        (since && *since) || (until && *until))) {
        std::println("(no models match the given filter)");
        return 0;
    }

    if (metrics || refresh) {
        if (refresh) {
            int n_loadable = 0;
            for (const reg::ModelMeta& m : sel) if (loadable(m)) ++n_loadable;
            std::println("refreshing up to {} loadable, filtered model(s) ({} total filtered, {} skipped "
                         "as incompatible with this build)...", n_loadable, sel.size(), sel.size() - static_cast<std::size_t>(n_loadable));
            sub0::build_model();   // once, reused for every loadable model in the loop below
            // gen_self_stats resolves the EOS-stop signal via sub0::eos_token_id(), which returns -1
            // (no tokenizer loaded) unless this is called -- without it, generation never stops early,
            // silently inflating the measured repetition/perplexity exactly like the bug a016d17 fixed
            // (and the identical bug this exact call site had until 4bf29fc -- the return value wasn't
            // even checked then, so a failure here would have reintroduced the same corruption).
            if (!sub0::load_tokenizer(sub0::default_tokenizer())) {
                sub0::log::error("models --refresh: cannot load this build's tokenizer -- refusing to "
                                 "measure with a broken EOS-stop signal (see the comment above)");
                return 1;
            }
            sub0::TokMap tok(sub0::default_corpus_tok());
            std::vector<int> od_buf;
            const sub0::TokView data = load_corpus_tokens(tok, od_buf, "models --refresh");
            if (data.empty()) { sub0::log::error("models --refresh: could not load this build's corpus"); return 1; }
            const std::size_t val_tokens = std::max<std::size_t>(
                static_cast<std::size_t>(SEQ_LEN) + 2,
                static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(data.size())));
            const std::size_t val_start = data.size() - val_tokens;
            int n_done = 0;
            for (const reg::ModelMeta& m : sel) {
                if (!loadable(m)) continue;
                ec::EvalMetrics cached;
                const bool had_cache = ec::read_metrics(m.dir, cached);
                if (!force && had_cache && cached.steps == m.steps) continue;   // already fresh
                const std::string model_path = (m.dir / "model.bin").string();
                if (!sub0::load_model(model_path.c_str())) {
                    sub0::log::warn("models --refresh: '{}' did not load (skipped)", m.dir.filename().string());
                    continue;
                }
                // Per-model session: each iteration loads DIFFERENT weights, so the device upload has
                // to happen again anyway -- one long-lived session across the loop would score every
                // model after the first against the first one's uploaded params.
                const sub0::eval::Session eval_sess(/*allow=*/!trainer_active());
                const ec::EvalMetrics raw =
                    compute_raw_metrics(data, val_start, /*want_autotemp_grid=*/true, 42, eval_sess);
                ec::EvalMetrics out;
                out.steps = m.steps; out.measured_at = reg::now_iso();
                out.train_nelbo = raw.train_nelbo; out.val_nelbo = raw.val_nelbo; out.bytes_per_tok = raw.bytes_per_tok;
                out.target_ppl = raw.target_ppl; out.target_rep = raw.target_rep; out.grid = raw.grid;
                ec::write_metrics(m.dir, out);
                std::println("  refreshed {}  (val_nelbo {:.4f})", m.dir.filename().string(), out.val_nelbo);
                ++n_done;
            }
            std::println("refreshed {} model(s)", n_done);
        }

        std::println("");
        std::println("{:<3} {:<48} {:>5} {:>3} {:>3} {:>7} {:>9} {:>7} {:>5} {:>4}  {}",
                     "use", "model", "d", "L", "H", "params", "val_nelbo", "bits/B", "temp", "opt", "corpus");
        for (const reg::ModelMeta& m : sel) {
            ec::EvalMetrics cm;
            const bool has = ec::read_metrics(m.dir, cm);
            if (!has) {
                const bool has_bin = std::filesystem::exists(m.dir / "bin");
                std::println("{:<3} {:<48} {:>5} {:>3} {:>3}  -- not yet measured --{}",
                             loadable(m) ? " * " : " x ", m.dir.filename().string(),
                             m.d_model, m.n_layers, m.n_heads,
                             has_bin ? "  (bin/ present -- run report/autotemp via it)" : "");
                continue;
            }
            const int d_ff = sub0::config::d_ff_for(m.d_model, m.gated_ffn);
            const auto params = sub0::memplan::param_floats(sub0::memplan::Dims{
                m.d_model, m.n_layers, m.n_heads, d_ff, m.seq_len, m.vocab,
                static_cast<bool>(m.tied_embeddings), static_cast<bool>(m.qk_norm), static_cast<bool>(m.gated_ffn)});
            const double bpb = cm.val_nelbo >= 0 && cm.bytes_per_tok > 0
                ? cm.val_nelbo / (0.6931471805599453 * cm.bytes_per_tok) : -1.0;
            const double temp = recommended_temp_from_cache(cm);
            std::println("{:<3} {:<48} {:>5} {:>3} {:>3} {:>6.1f}M {:>9} {:>7} {:>5} {:>4}  {}",
                         loadable(m) ? " * " : " x ", m.dir.filename().string(),
                         m.d_model, m.n_layers, m.n_heads, params / 1e6,
                         cm.val_nelbo >= 0 ? std::format("{:.4f}", cm.val_nelbo) : "-",
                         bpb >= 0 ? std::format("{:.3f}", bpb) : "-",
                         temp >= 0 ? std::format("{:.2f}", temp) : "-",
                         m.optimizer ? "muon" : "adamw", m.corpus);
        }
        std::println("(* loadable by this build | x incompatible architecture)");
        return 0;
    }

    std::println("{:<3} {:<50} {:>9}  {:<10} {}", "use", "model", "val_nelbo", "status", "corpus");
    for (const reg::ModelMeta& m : sel)
        std::println("{:<3} {:<50} {:>9}  {:<10} {}", loadable(m) ? " * " : " x ",
                     m.dir.filename().string(),
                     m.best_val_nelbo >= 0 ? std::format("{:.4f}", m.best_val_nelbo) : "-",
                     m.status, m.corpus);
    std::println("(* loadable by this build | x incompatible architecture)");

    if (prune) {
        int removed = 0;
        for (const reg::ModelMeta& m : sel) {
            if (loadable(m)) continue;
            std::error_code ec2;
            const auto n = std::filesystem::remove_all(m.dir, ec2);
            if (!ec2) { std::println("pruned {} ({} entries)", m.dir.filename().string(), n); ++removed; }
            else      sub0::log::warn("could not prune {}", m.dir.string());
        }
        std::println("pruned {} incompatible model(s)", removed);
    }
    return 0;
}

// --- Bundle --------------------------------------------------------------------
// `sub0llm bundle <model>` copies THIS build's own runtime binaries (the umbrella exe + the
// libraries gen/report actually load -- core+gen+train, deliberately NOT the CUDA backend, since
// generation/eval always run on the CPU engine regardless of how the model was trained) into the
// model's own directory. Opt-in and separate from `train`: every model directory's dims are pinned
// to the exact build that produced it (see `sub0llm models`' own loadable/incompatible split), so a
// LATER, differently-configured build of this same repo can no longer load an old checkpoint.
// Bundling a copy of the exact binaries that CAN load it turns "reconfigure + rebuild to compare an
// old model" into "run the bundled copy from its own model dir" -- valuable for a cross-model-size
// research sweep (train several sizes, keep each buildable/runnable side by side), not a routine
// step every `train` run needs.
extern "C" SUB0_API int sub0_bundle_stage(const char* model_in) {
    if (!model_in || !*model_in) { sub0::log::error("bundle: a model path is required"); return 1; }
    const std::filesystem::path model_dir = std::filesystem::path(model_in).parent_path();
    if (model_dir.empty() || !std::filesystem::exists(model_dir)) {
        sub0::log::error("bundle: model directory not found: {}", model_dir.string());
        return 1;
    }

#if defined(_WIN32)
    char self_path[MAX_PATH]{};
    const DWORD n = GetModuleFileNameA(nullptr, self_path, MAX_PATH);
    if (n == 0 || n == MAX_PATH) {
        sub0::log::error("bundle: could not resolve this executable's own path");
        return 1;
    }
    const std::filesystem::path self_dir = std::filesystem::path(self_path).parent_path();
#else
    sub0::log::error("bundle: not implemented on this platform");
    return 1;
#endif

    static constexpr const char* kFiles[] = {
        "sub0llm.exe", "sub0_core.dll", "sub0_gen.dll", "sub0_train.dll",
    };
    const std::filesystem::path bundle_dir = model_dir / "bin";
    std::error_code ec;
    std::filesystem::create_directories(bundle_dir, ec);
    int copied = 0;
    for (const char* name : kFiles) {
        const std::filesystem::path src = self_dir / name;
        if (!std::filesystem::exists(src)) {
            sub0::log::warn("bundle: missing {} in {} (skipped)", name, self_dir.string());
            continue;
        }
        std::filesystem::copy_file(src, bundle_dir / name,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) { sub0::log::warn("bundle: could not copy {}: {}", name, ec.message()); continue; }
        ++copied;
    }
    std::println("bundled {} file(s) from {} -> {}", copied, self_dir.string(), bundle_dir.string());
    if (copied > 0)
        std::println("run later without rebuilding: {}\\sub0llm.exe report \"{}\"", bundle_dir.string(), model_in);
    return copied > 0 ? 0 : 1;
}

// --- Checkpoint -> model.bin conversion ---------------------------------------
// `sub0llm ckpt2model <ckpt> <model_out>` extracts just the weights from a full training
// checkpoint (which also carries optimizer moments, RNG state, and eval history -- a DIFFERENT
// binary layout from model.bin's, see save_model()/load_model() in engine_core.cpp) into a plain
// model.bin that gen/report can load directly. This is the missing piece that makes prune_ckpts'
// best-checkpoint exemption actually useful: keeping the best .ckpt on disk is pointless if the
// only way to run generation against it is resuming a full training session just to get model.bin
// re-saved from it (model.bin itself is overwritten with the LATEST weights on every eval tick, not
// the best).
extern "C" SUB0_API int sub0_ckpt2model_stage(const char* ckpt_in, const char* model_out) {
    if (!ckpt_in || !*ckpt_in || !model_out || !*model_out) {
        sub0::log::error("ckpt2model: both a checkpoint and an output model path are required");
        return 1;
    }
    // Prefer the tokenizer sitting next to the checkpoint (what training actually bundled in, see
    // bundle_into_model_dir) so the written model.bin's fingerprint trailer matches what gen/report
    // expect; fall back to this build's own default tokenizer only if the model dir has none.
    const std::filesystem::path sibling_tok = std::filesystem::path(ckpt_in).parent_path() / "tokenizer.tok";
    const std::string tok_path = std::filesystem::exists(sibling_tok) ? sibling_tok.string()
                                                                       : sub0::default_tokenizer();
    if (!sub0::load_tokenizer(tok_path.c_str())) {
        // Not cosmetic here: save_model() below stamps the loaded tokenizer's fingerprint into
        // model.bin's own trailer (see tokenizer_fingerprint()'s doc comment) -- writing one with no
        // tokenizer loaded produces a model.bin whose fingerprint gen/report will reject as
        // mismatched (or silently accept as "no tokenizer," worse), defeating this tool's one job.
        sub0::log::error("ckpt2model: cannot load tokenizer '{}' -- the written model.bin would carry "
                         "a wrong/missing fingerprint", tok_path);
        return 1;
    }
    sub0::build_model();   // allocates the param/Adam arenas load_checkpoint() reads into

    std::mt19937 rng; RunState rs; int batch = 0; float lr = 0.0f; unsigned seed = 0; long adam_t = 0;
    if (!load_checkpoint(ckpt_in, rng, rs, batch, lr, seed, adam_t)) {
        sub0::log::error("ckpt2model: could not load checkpoint '{}' (see the warning above)", ckpt_in);
        return 1;
    }
    if (!sub0::save_model(model_out)) {
        sub0::log::error("ckpt2model: could not write '{}'", model_out);
        return 1;
    }
    std::println("{} (step {}, best-so-far val_nelbo {}) -> {}", ckpt_in, rs.step,
                 rs.best_loss < std::numeric_limits<double>::infinity() ? std::format("{:.4f}", rs.best_loss) : "-",
                 model_out);
    return 0;
}

