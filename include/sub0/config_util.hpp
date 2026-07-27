// sub0/config_util.hpp — pure configuration-decision logic, factored out of the configurator's
// main() so it is unit-testable (no I/O, no globals). The configurator wires these to the corpus
// scan + header emission; tests drive them directly. Header-only, std-only.

#pragma once

#include "sub0/memplan.hpp"   // sub0::memplan::{Dims, max_batch_for_vram} -- gpu_batch_estimate() below

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <string>
#include <vector>

namespace sub0::config {

// --- Model auto-sizing -----------------------------------------------------
// The dimensions a corpus of a given size justifies, so a fresh corpus trains a sensibly-sized model
// as a STARTING POINT without hand-tuning -- always overridable (CLI flag > <corpus>.model sidecar >
// this). Formula-based, not a lookup table: every dimension is a smooth, monotonic function of corpus
// scale, grounded in established scaling-law understanding AND real reference models actually trained
// on comparable data, rather than a handful of hand-picked per-bucket values.
//
// 1) Token count is estimated from raw corpus bytes via a bytes/token ratio -- this whole function runs
//    BEFORE any real tokenization, so it can only ever be a starting estimate. The ratio itself is
//    CALIBRATED, not a one-off guess: `bytes_per_token` defaults to a generic 4.0 floor, but
//    `tools/configurator.cpp` passes in the REAL combined average from `data/tokenizer_calibration.txt`
//    (see TokenCalibration below) -- a per-corpus LEDGER, one (bytes, tokens) entry per corpus this
//    project's own sub0llm-configure has ever tokenized, upserted (not accumulated) each time a fresh
//    tokenization happens so re-measuring the same corpus from a different build dir never double-counts
//    it. Seeded from this project's own tinystories.txt (3.336 bytes/token) and fineweb_edu.txt (3.441, the
//    biggest real reference so far, and the dominant contributor since it's ~20x more tokens) --
//    combined: 3.436. The vocab-curve knee (`--dump-vocab`) is a separate, more PRECISE per-corpus
//    refinement once a specific corpus is actually scanned; this is the cross-corpus prior that seeds
//    the estimate before that happens.
// 2) The target trainable-parameter budget uses a tokens/param ratio well above Chinchilla's pure
//    compute-optimal 20:1 (Hoffmann et al. 2022) -- real small-model practice consistently trains far
//    past compute-optimal because, at this scale, DATA is cheap relative to keeping the model small
//    and deployable: SmolLM2-135M used ~14,800 tokens/param, SmolLM2-1.7B ~6,470, and even the
//    FineWeb-Edu paper's own from-scratch ablation model (a much closer match to this project's
//    regime -- a research-scale run, not a dedicated mega-lab training campaign) used ~192 tokens/param.
//    100:1 is a deliberately conservative middle ground given this project's corpus is FIXED and finite
//    (not an effectively-unlimited web-scale stream) -- its own empirical finding backs the same
//    DIRECTION even at 20:1: the dims_analysis.py capacity sweep on the (then) 22 MB tinystories corpus
//    found a train<val gap appearing at 5.8M params, ~1.3 tokens/param, drastically below even
//    Chinchilla's ratio -- i.e. the corpus, not capacity, was already the ceiling there.
// 3) That budget becomes a transformer body: N_body = 12 * n_layers * d_model^2 (the standard
//    attention (4d^2/layer) + 4x-FFN (8d^2/layer) parameter-count identity), decomposed via a
//    width/depth aspect ratio (d_model / n_layers) and snapped to head_dim=64-multiples (this
//    project's own "founded" head width, see project memory) -- head-divisibility and a non-degenerate
//    head_dim both fall out of that snapping automatically, not as a separate check. The aspect ratio
//    itself SCALES WITH MODEL SIZE rather than staying fixed: real small models are proportionally much
//    DEEPER than large ones -- the TinyStories paper's own reference configs keep n_layers=8 fixed from
//    1M to 28M params (varying only width: d=64/128/256, aspect ratio 8-32), while SmolLM2-1.7B widens
//    to d=2048/L=24 (aspect ratio ~85) and explicitly documents "prioritize depth over width" as the
//    strategy for its smaller 135M/360M variants. A single fixed ratio can't capture that -- it's
//    interpolated log-linearly in target_params between a small-scale floor and a large-scale ceiling,
//    both taken directly from those real reference points rather than picked arbitrarily.
// 4) Vocabulary follows Heaps'-law-style sublinear growth in token count (empirically, natural-language
//    vocabulary diversity grows roughly as tokens^0.4, not linearly with corpus size) -- deliberately a
//    SEPARATE axis from the capacity budget above, not competing with it for the same params (at very
//    small corpus scale, embedding+head params can otherwise dominate the naive budget entirely and
//    starve the transformer body to a degenerate size).
// 5) seq_len scales mildly with model width (a bigger model can productively use more context) rather
//    than with corpus bytes directly, since no per-document length statistic exists yet at this point
//    in the pipeline (only the configure-time corpus byte count, not a scan).
// 6) A hardware-aware clamp: if the detected GPU's VRAM (0 = unknown/CPU-only, in which case this is a
//    no-op) can't fit even batch=1 of the naive suggestion, the shape is shrunk step-by-step (dropping
//    one head at a time, recomputing everything derived from it, vocab held fixed since it doesn't
//    compete for the same budget) until it does. This is what makes "point it at a big corpus" never
//    outright FAIL on a modest GPU by default -- an explicit --dmodel override that intentionally
//    exceeds available VRAM still hits the configurator's existing hard-error path downstream (see
//    tools/configurator.cpp's own VRAM-clamp comment), which is correct: that's a deliberate choice,
//    not an auto-suggestion gone wrong. The byte-budget-per-parameter constant here is a coarse
//    approximation (real footprint also depends on batch/seq/activation precision, all handled exactly
//    by memplan.hpp downstream) calibrated against a real measurement, not guessed.
// 7) size_scale (default 1.0) is a caller-chosen multiplier on the target-parameter budget, from a
//    minimal/fast/safe starting point (< 1.0) to a more generous one (> 1.0) -- the SAME formula and
//    reasoning throughout, just a different point on it, rather than a second unrelated sizing scheme.
//
// Every dimension is monotonic non-decreasing in corpus_bytes (smooth functions composed with rounding,
// clamping and the VRAM shrink-loop all stay monotonic; property-tested), d_model stays exactly
// head-divisible by construction, and generous [min,max] clamps keep a pathologically tiny or huge
// corpus from producing a degenerate shape even with no VRAM budget known.
struct ModelDims { int d_model = 0, n_layers = 0, n_heads = 0, seq_len = 0, vocab = 0; };

// Target tokens/param ratio -- see point 2 above: conservative vs. real small-model practice
// (192-14,800:1), well above pure Chinchilla (20:1). NAMESPACE SCOPE (not local to autosize()) so
// `sub0llm report`'s corpus-fit diagnostic (train_stage.cpp) can reference the SAME constant instead
// of an independently-hardcoded copy that can silently drift from this one -- exactly the class of
// bug a stale `preview()` temperature default (0.7 vs gen's real 0.8) turned out to be elsewhere in
// this codebase, found and fixed the same session this constant was hoisted for that reason.
inline constexpr double kTokensPerParam = 100.0;

// Width/depth aspect ratio (d_model / n_layers) interpolates log-linearly in target_params between a
// small-scale floor (TinyStories' own ~1M-param reference configs, aspect 8-32) and a large-scale
// ceiling (SmolLM2-1.7B's own real aspect ratio, ~85) -- see autosize()'s own doc comment (point 3) for
// the full reasoning. NAMESPACE SCOPE for the same reason kTokensPerParam is: `sub0llm report`'s
// next-size search (train_stage.cpp) needs the identical curve, not an independently-hardcoded ratio
// that can silently go stale the way a fixed "40.0" here already had before this was shared.
inline constexpr double kAspectMin = 16.0, kAspectMax = 85.0, kAspectMinParams = 1e6, kAspectRefParams = 1.7e9;
inline double aspect_for_params(double params) {
    const double aspect_ref = std::log10(kAspectRefParams / kAspectMinParams);
    const double t = std::log10(std::max(params, kAspectMinParams) / kAspectMinParams);
    return std::clamp(kAspectMin + (kAspectMax - kAspectMin) * (t / aspect_ref), kAspectMin, kAspectMax);
}

inline ModelDims autosize(std::uintmax_t corpus_bytes, int vram_mb = 0, double size_scale = 1.0,
                           double bytes_per_token = 4.0) {
    constexpr int    kHeadDim         = 64;     // "founded" head width -- see project memory
    constexpr double kVocabScale = 8.0, kVocabBeta = 0.4;         // Heaps'-law-style vocab growth
    constexpr int    kMinVocab = 1024, kMaxVocab = 49152;         // token IDs must fit this project's
                                                                   // uint16 corpus.tok encoding (<=65536)
    constexpr int    kMinHeads = 2,    kMaxDModel = 2048;         // 2048 = 32 * kHeadDim, exact
    constexpr int    kMinLayers = 4,   kMaxLayers = 48;
    constexpr int    kMinSeq = 128,    kMaxSeq = 1024;
    // Calibrated against a real measurement (this project's own memplan.hpp, BF16 activations, batch=1):
    // a 651.9M-param shape needed ~11,158 MiB, i.e. ~17.1 bytes/param; 18 leaves a small margin so this
    // coarse pre-check stays on the safe side of memplan's own exact byte-level accounting downstream.
    constexpr double kBytesPerParamVram = 18.0;
    if (bytes_per_token <= 0.0) bytes_per_token = 4.0;   // guard a corrupt/zero calibration file

    const double tokens = static_cast<double>(corpus_bytes) / bytes_per_token;
    const double target_params = std::max(1.0, size_scale * tokens / kTokensPerParam);

    const double vocab_raw = kVocabScale * std::pow(std::max(1.0, tokens), kVocabBeta);
    const int    vocab = std::clamp(static_cast<int>(std::lround(vocab_raw / 512.0)) * 512,
                                     kMinVocab, kMaxVocab);

    // Shape (d_model, n_layers, seq_len, realized param count) for a given head count -- everything
    // downstream of n_heads is exactly determined, so the VRAM shrink-loop below just walks n_heads down.
    // The aspect ratio itself stays pinned to the ORIGINAL corpus-derived target_params throughout (not
    // recomputed per shrink step): it answers "how deep should a model of the scale THIS CORPUS
    // justifies be," a separate question from "how far does hardware then force us to shrink it."
    const double aspect = aspect_for_params(target_params);
    const auto shape_for = [&](int n_heads) {
        const int d_model  = std::min(kMaxDModel, n_heads * kHeadDim);
        n_heads             = d_model / kHeadDim;   // re-derive: exact by construction
        const int n_layers  = std::clamp(static_cast<int>(std::lround(d_model / aspect)),
                                          kMinLayers, kMaxLayers);
        const double seq_raw = 256.0 * std::sqrt(static_cast<double>(d_model) / 192.0);
        const int    seq_len = std::clamp(static_cast<int>(std::lround(seq_raw / 128.0)) * 128,
                                           kMinSeq, kMaxSeq);
        const double realized_params = 12.0 * n_layers * static_cast<double>(d_model) * d_model
                                      + 2.0 * vocab * d_model;
        return std::tuple{d_model, n_layers, n_heads, seq_len, realized_params};
    };

    // N_body = 12*L*d^2, L = d/aspect  =>  N_body = (12/aspect) * d^3.
    const double d_raw = std::cbrt(target_params * aspect / 12.0);
    int n_heads = std::max(kMinHeads, static_cast<int>(std::lround(d_raw / kHeadDim)));

    auto [d_model, n_layers, n_heads_final, seq_len, realized_params] = shape_for(n_heads);

    // Hardware-aware clamp (point 6 above): shrink one head at a time -- recomputing everything derived
    // from it, including vocab's own (fixed) contribution to realized_params -- until it fits, rather
    // than a one-shot pre-estimate that can't see vocab's cost until the shape is actually known.
    if (vram_mb > 0) {
        const double budget_params = static_cast<double>(vram_mb) * 1e6 / kBytesPerParamVram;
        while (realized_params > budget_params && n_heads_final > kMinHeads) {
            --n_heads_final;
            std::tie(d_model, n_layers, n_heads_final, seq_len, realized_params) = shape_for(n_heads_final);
        }
    }

    return {d_model, n_layers, n_heads_final, seq_len, vocab};
}

// Fill any field of `pinned` left 0 (the "auto" sentinel) from `fallback`; a nonzero field (an
// explicit --dmodel/... or a sidecar pin) is preserved. The resolution precedence is built by
// chaining: CLI -> sidecar -> auto-size.
inline ModelDims fill_defaults(ModelDims pinned, const ModelDims& fallback) {
    if (pinned.d_model  == 0) pinned.d_model  = fallback.d_model;
    if (pinned.n_layers == 0) pinned.n_layers = fallback.n_layers;
    if (pinned.n_heads  == 0) pinned.n_heads  = fallback.n_heads;
    if (pinned.seq_len  == 0) pinned.seq_len  = fallback.seq_len;
    if (pinned.vocab    == 0) pinned.vocab    = fallback.vocab;
    return pinned;
}
inline ModelDims apply_autosize(ModelDims pinned, std::uintmax_t corpus_bytes, int vram_mb = 0,
                                 double size_scale = 1.0, double bytes_per_token = 4.0) {
    return fill_defaults(pinned, autosize(corpus_bytes, vram_mb, size_scale, bytes_per_token));
}

// --- Cross-corpus tokenizer bytes/token calibration (data/tokenizer_calibration.txt) -----------
// A per-corpus LEDGER (one (bytes, tokens) entry per distinct corpus path), so autosize()'s a-priori
// bytes/token estimate keeps improving as more real corpora are processed instead of staying pinned
// to one generic guess forever. Keyed by corpus path rather than a single running total: the same
// named corpus (e.g. tinystories.txt) gets re-tokenized fresh in every new/reconfigured build
// directory this project uses (ts_smoke, ts_cuda, a one-off test dir, ...), and a flat accumulator
// would add that corpus's contribution again on every one of those -- silently inflating its weight
// in the combined estimate every time someone spins up a new build dir for the SAME corpus. An
// upsert (replace this corpus's entry, don't add to a total) makes re-measuring idempotent: the
// calibration reflects each corpus's real bytes/token exactly once, no matter how many times or
// where it gets tokenized. bytes_per_token_calibrated() sums bytes and tokens ACROSS entries
// (sum-then-divide, not an average of per-corpus averages) so a huge corpus still correctly
// outweighs a tiny one -- matching how the original two-corpus seed (tinystories.txt + the much
// bigger fineweb_edu.txt) was combined by hand before this ever ran automatically. Pure parse/format/
// upsert (matching the <corpus>.model sidecar's own convention below); the file I/O stays in the
// configurator.
struct TokCorpusEntry { std::string corpus; std::uintmax_t bytes = 0, tokens = 0; };
struct TokenCalibration { std::vector<TokCorpusEntry> entries; };

inline TokenCalibration parse_token_calibration(std::istream& is) {
    TokenCalibration c;
    for (std::string line; std::getline(is, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto t1 = line.find('\t');
        const auto t2 = (t1 == std::string::npos) ? std::string::npos : line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) continue;
        TokCorpusEntry e;
        e.corpus = line.substr(0, t1);
        e.bytes  = std::strtoull(line.substr(t1 + 1, t2 - t1 - 1).c_str(), nullptr, 10);
        e.tokens = std::strtoull(line.substr(t2 + 1).c_str(), nullptr, 10);
        if (!e.corpus.empty()) c.entries.push_back(std::move(e));
    }
    return c;
}
inline std::string format_token_calibration(const TokenCalibration& c) {
    std::string out =
        "# sub0 tokenizer bytes/token calibration ledger (auto-accumulated by sub0llm-configure --\n"
        "# see autosize()'s own doc comment, point 1). One line per DISTINCT corpus (re-tokenizing an\n"
        "# already-listed corpus REPLACES its entry, so this stays correct across many build dirs).\n"
        "# path\\tbytes\\ttokens\n";
    for (const TokCorpusEntry& e : c.entries)
        out += e.corpus + "\t" + std::to_string(e.bytes) + "\t" + std::to_string(e.tokens) + "\n";
    return out;
}
// Idempotent upsert: a fresh measurement of `corpus` replaces its existing entry (if any) rather
// than accumulating on top of it, so tokenizing the same corpus again -- in a different build dir,
// or after an unrelated config change forces a re-tokenize -- never double-counts it.
inline void upsert_token_calibration(TokenCalibration& c, const std::string& corpus,
                                     std::uintmax_t bytes, std::uintmax_t tokens) {
    for (TokCorpusEntry& e : c.entries)
        if (e.corpus == corpus) { e.bytes = bytes; e.tokens = tokens; return; }
    c.entries.push_back({corpus, bytes, tokens});
}
// The calibrated estimate, or `fallback` (autosize()'s own generic 4.0 default) if nothing has been
// accumulated yet (a fresh checkout with no calibration file, or a corrupt/empty one).
inline double bytes_per_token_calibrated(const TokenCalibration& c, double fallback = 4.0) {
    std::uintmax_t bytes = 0, tokens = 0;
    for (const TokCorpusEntry& e : c.entries) { bytes += e.bytes; tokens += e.tokens; }
    return tokens > 0 ? static_cast<double>(bytes) / static_cast<double>(tokens) : fallback;
}

// FFN hidden width: 4*D_MODEL for the plain (2-matrix, GELU+bias) FFN -- this project's long-standing
// default -- or, when gated (SwiGLU: 3 matrices, no bias), a width chosen so the two FFN styles land
// at roughly the SAME total param count at the same D_MODEL, rather than gating silently costing 50%
// more FFN params for a same-width swap. Derivation: plain params = 2*D*F_plain, gated params =
// 3*D*F_gated; equal when F_gated = (2/3)*F_plain = (2/3)*4*D = (8/3)*D -- this project's OWN 4x
// convention determines the 8/3 ratio, it is not copied from elsewhere (it happens to match the
// common "SwiGLU ~8/3" convention in the literature, which was derived the same way against ITS
// baseline's 4x). Rounded UP to the next multiple of 64 (never below 64) for GEMM-friendly shapes,
// matching the common convention of ceiling rather than rounding to nearest (never undershoots).
inline int d_ff_for(int d_model, bool gated_ffn) {
    if (!gated_ffn) return 4 * d_model;
    const int exact = d_model * 8 / 3;
    constexpr int kMultiple = 64;
    return std::max(kMultiple, ((exact + kMultiple - 1) / kMultiple) * kMultiple);
}

// --- Per-corpus model sidecar (<corpus>.model) -----------------------------
// A key=value sidecar so a chosen size PERSISTS across configurator re-runs (a build-time auto-regen
// then keeps the user's pins instead of re-auto-sizing). Pure parse/format; the file I/O stays in the
// configurator. A 0 field means "unset" (defer to auto-size).
inline ModelDims parse_model_sidecar(std::istream& is) {
    ModelDims m;
    for (std::string line; std::getline(is, line);) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq);
        const int         v = std::atoi(line.substr(eq + 1).c_str());
        if      (k == "d_model")  m.d_model  = v;
        else if (k == "n_layers") m.n_layers = v;
        else if (k == "n_heads")  m.n_heads  = v;
        else if (k == "seq_len")  m.seq_len  = v;
        else if (k == "vocab")    m.vocab    = v;
    }
    return m;
}
inline std::string format_model_sidecar(const ModelDims& m) {
    return "# sub0 per-corpus model config (auto-seeded; edit to pin a size, then rebuild)\n"
           "d_model="  + std::to_string(m.d_model)  + "\n"
           "n_layers=" + std::to_string(m.n_layers) + "\n"
           "n_heads="  + std::to_string(m.n_heads)  + "\n"
           "seq_len="  + std::to_string(m.seq_len)  + "\n"
           "vocab="    + std::to_string(m.vocab)    + "\n";
}

// --- Tuned runtime defaults (from the tune cache) --------------------------
// The threads / windows-per-thread / GPU-batch / TF32 / attn-backward a `tune` run persisted. Falls
// back to the hardware core count + conservative defaults when the cache is empty or a key is absent.
struct TuneDefaults {
    int  threads            = 1;
    int  windows_per_thread = 4;
    int  gpu_batch          = 0;   // 0 -> derived = threads * windows_per_thread (set on return)
    bool cuda_tf32          = false;
    bool tf32_from_cache    = false;
    bool gpu_batch_from_cache = false;   // true only if a REAL `tune --backend gpu` result was read
};

inline TuneDefaults parse_tune_cache(std::istream& is, int hw_concurrency) {
    TuneDefaults d;
    d.threads = hw_concurrency > 0 ? hw_concurrency : 1;
    for (std::string line; std::getline(is, line);) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const int         val = std::atoi(line.substr(eq + 1).c_str());
        if      (key == "threads"            && val > 0) d.threads            = val;
        else if (key == "windows_per_thread" && val > 0) d.windows_per_thread = val;
        else if (key == "gpu_batch"          && val > 0) { d.gpu_batch = val; d.gpu_batch_from_cache = true; }
        else if (key == "cuda_tf32")                   { d.cuda_tf32 = (val != 0); d.tf32_from_cache = true; }
    }
    if (d.gpu_batch <= 0) d.gpu_batch = d.threads * d.windows_per_thread;
    return d;
}

// A VRAM-aware starting batch for an UNTUNED GPU build (no real `tune --backend gpu` result exists
// yet) -- used only as the pre-tune default, never overriding a real cached tune result. Before this,
// an untuned GPU build's DEFAULT_GPU_BATCH fell back to threads*windows_per_thread (the CPU
// data-parallel width), which has nothing to do with GPU VRAM or throughput -- confirmed as a real,
// live gap: a production GPU training run measured only 31% of an 8 GB card's VRAM in use at exactly
// that CPU-derived batch (96), because a real tune had never been run for that build (see this
// project's own "GPU batch undertuned / VRAM headroom" notes). Estimates the largest batch that
// plausibly fits (`memplan::max_batch_for_vram`, the SAME primitive the tuner bounds its own search
// with, so the two never disagree in spirit) against a HEADROOM-reserved budget -- `headroom_mb`
// mirrors the tuner's own reservation for the cuBLAS GEMM workspace (allocated lazily on the first
// matmul, invisible to the pure footprint model) plus allocator fragmentation, without which an
// over-budget cudaMalloc doesn't fail loudly on Windows, it silently spills to WDDM shared memory and
// thrashes over PCIe (~10x slower) -- a silent performance cliff worse than under-using VRAM. Rounded
// DOWN to a multiple of `align` (default 32, the CUDA warp size and a safe divisor of the tensor-core
// tile shapes cuBLAS/CUTLASS actually use) so the estimate lands on a clean, GEMM-friendly boundary
// instead of whatever number a raw binary search happens to produce. This is a REASONABLE starting
// point, not a substitute for actually tuning: the tuner's own search found throughput is
// NON-monotonic in batch on this hardware (a real dip at 512 that recovers by 768), so the true
// optimum can land on either side of this estimate -- `sub0llm tune --backend gpu` is still how you
// find it exactly.
// SAFE_BATCH_CEILING caps what this function may RECOMMEND, independently of what fits in VRAM.
// sub0_cuda_train_step faults (SIGSEGV, no step line printed) above a batch that measures between 512
// and 704 on this project's hardware -- reproduced at d256 L6 H4 and again at d384 L6 H8, in both MHA
// and GQA builds, so it is batch-driven rather than shape-specific. Root cause is still open (project
// memory gpu-large-batch-access-violation).
//
// Until it is fixed, an unclamped estimate is not merely aggressive, it is WRONG: on an 8 GB card this
// function returned 704 for a d384 config, which means plain `sub0llm-train` with no --batch flag
// crashed on the first step. A recommendation that cannot run is worse than a conservative one, and a
// long unattended full-corpus run is exactly where it hurts most.
//
// Deliberately a CEILING on the recommendation only -- an explicit `--batch 640` is still honoured, so
// the cap cannot silently block investigating or fixing the fault. Remove it, with the A/B to show
// throughput actually improves past 512, once the fault is root-caused.
inline constexpr int SAFE_BATCH_CEILING = 512;

inline int gpu_batch_estimate(const sub0::memplan::Dims& dims, int vram_mb, int hard_cap,
                              sub0::memplan::u64 act_bytes, int headroom_mb = 512, int align = 32) {
    if (vram_mb <= 0) return 0;             // unknown VRAM -> caller keeps its own (CPU-width) fallback
    const int budget = vram_mb - headroom_mb;
    if (budget <= 0) return 0;
    const int fit_vram = sub0::memplan::max_batch_for_vram(dims, budget, hard_cap, act_bytes);
    const int fit = (fit_vram < SAFE_BATCH_CEILING) ? fit_vram : SAFE_BATCH_CEILING;
    return (fit >= align) ? (fit / align) * align : fit;   // too small to round without hitting 0
}

// --- Reduced-precision resolution ------------------------------------------
// A platform is BF16-capable iff BF16 is forced (flag 1) or AUTO (flag 2) and the GPU is sm_80+.
inline bool f16_capable(int bf16_flag, int cuda_arch) {
    return (bf16_flag == 1) || (bf16_flag == 2 && cuda_arch >= 80);
}

// Resolve a per-section precision code (0=F32, 1=BF16, 2=F16, 9=AUTO) to a Dtype name. `status`
// reports the not-yet-supported choices so the caller errors instead of std::exit-ing inside logic.
enum class PrecStatus { Ok, NeedsF16Hardware, F16Unsupported };
struct Precision { std::string dtype; PrecStatus status; };

inline Precision resolve_precision(int code, bool f16_ok) {
    switch (code) {
        case 0:  return {"F32",  PrecStatus::Ok};
        case 1:  return f16_ok ? Precision{"BF16", PrecStatus::Ok}
                               : Precision{"",     PrecStatus::NeedsF16Hardware};
        case 2:  return {"", PrecStatus::F16Unsupported};
        default: return {f16_ok ? "BF16" : "F32", PrecStatus::Ok};   // 9 = AUTO
    }
}

// --- Pretokenize-vs-on-demand decision --------------------------------------
// Pre-tokenizing the whole corpus to corpus.tok is a clear win for fast random-access training while
// that token copy stays page-cacheable; a corpus large enough that it can't stay resident is better
// served tokenizing windows on demand (out-of-core: no on-disk token copy, no RAM pressure). AUTO
// decides by SCALE rather than a fixed threshold: pre-tokenize when the estimated corpus.tok fits in
// half of physical RAM, on-demand otherwise. Pure (both sizes are already-known inputs) so it is
// unit-testable without a real filesystem/OS RAM query.
//
// corpus.tok is SMALLER than the source, not larger: the v2 format packs 2 bytes/token for any vocab
// up to 65536 (every autosize() dims ladder entry stays under that), and real corpora measured in this
// project compress to ~3.3-3.8 source bytes/token (JOIN + Unigram pieces), so corpus.tok lands around
// 0.55-0.6x the source size (measured: a 46.1 GB fineweb corpus produced a 26.7 GB corpus.tok; a 2.23 GB
// corpus produced a 1.34 GB one -- both ~0.6x). 3/4 is a deliberately conservative upper bound above
// that (covers a corpus that compresses worse than this project's prose corpora) while still being far
// closer to reality than treating corpus.tok as 2x the source, which used to make AUTO skip
// pre-tokenizing corpora that would in fact fit comfortably.
inline bool should_pretokenize(std::uintmax_t corpus_bytes, std::uintmax_t ram_bytes) {
    const std::uintmax_t est_tok_bytes = corpus_bytes * 3 / 4;
    return ram_bytes == 0 || est_tok_bytes <= ram_bytes / 2; // unknown RAM -> don't block on it, pretokenize
}

// --- Vocabulary-size curve (the "ideal vocab" knee) ------------------------
// A BPE merge i removes `merge_counts[i]` corpus tokens, so total_word_tokens(n_base+k) =
// total_word_bytes − Σ_{i<k} merge_counts[i] traces the whole bytes/token-vs-vocab curve. These pure
// summaries drive the --dump-vocab report; tested in isolation.

// The vocab size capturing `frac` (0..1) of the total achievable token reduction -- the
// diminishing-returns knee. Returns n_base when there is no reduction.
inline int vocab_at_fraction(long long total_reduction, const std::vector<long long>& merge_counts,
                             int n_base, double frac) {
    if (total_reduction <= 0) return n_base;
    const long long target = static_cast<long long>(frac * static_cast<double>(total_reduction));
    long long cum = 0;
    int k = 0;
    for (; k < static_cast<int>(merge_counts.size()) && cum < target; ++k) cum += merge_counts[k];
    return n_base + k;
}

// bytes/token at vocab n_base+k (1.0 at the character-encoding floor, rising as vocab grows).
inline double bytes_per_token_at(long long total_bytes, const std::vector<long long>& merge_counts, int k) {
    const int kk = std::min(k, static_cast<int>(merge_counts.size()));
    long long cum = 0;
    for (int i = 0; i < kk; ++i) cum += merge_counts[static_cast<std::size_t>(i)];
    const long long tok = total_bytes - cum;
    return tok > 0 ? static_cast<double>(total_bytes) / static_cast<double>(tok) : 0.0;
}

}  // namespace sub0::config
