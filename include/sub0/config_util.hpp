// sub0/config_util.hpp — pure configuration-decision logic, factored out of the configurator's
// main() so it is unit-testable (no I/O, no globals). The configurator wires these to the corpus
// scan + header emission; tests drive them directly. Header-only, std-only.

#pragma once

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
// 1) Token count is estimated from raw corpus bytes via a fixed bytes/token ratio -- this project's own
//    tokenizer lands around 3.3-4 bytes/token on real English corpora once the JOIN scheme's
//    space-folding applies (measured: 3.336 on the real tinystories.txt scan). This whole function runs
//    BEFORE any real tokenization, so it can only ever be a starting estimate anyway -- the vocab-curve
//    knee (`--dump-vocab`) refines it for real once the corpus is actually scanned.
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

inline ModelDims autosize(std::uintmax_t corpus_bytes, int vram_mb = 0, double size_scale = 1.0) {
    constexpr double kBytesPerToken   = 4.0;    // a-priori estimate; refined post-tokenization
    constexpr double kTokensPerParam  = 100.0;  // see point 2 above: conservative vs. real small-model
                                                 // practice (192-14,800:1), well above pure Chinchilla (20:1)
    // Width/depth aspect ratio (d_model / n_layers) interpolates log-linearly in target_params between
    // a small-scale floor (TinyStories' own ~1M-param reference configs, aspect 8-32) and a large-scale
    // ceiling (SmolLM2-1.7B's own real aspect ratio, ~85) -- see point 3 above.
    constexpr double kAspectMin = 16.0, kAspectMax = 85.0, kAspectMinParams = 1e6, kAspectRefParams = 1.7e9;
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

    const double tokens = static_cast<double>(corpus_bytes) / kBytesPerToken;
    const double target_params = std::max(1.0, size_scale * tokens / kTokensPerParam);

    const double aspect_ref = std::log10(kAspectRefParams / kAspectMinParams);
    const auto   aspect_for = [&](double params) {
        const double t = std::log10(std::max(params, kAspectMinParams) / kAspectMinParams);
        return std::clamp(kAspectMin + (kAspectMax - kAspectMin) * (t / aspect_ref), kAspectMin, kAspectMax);
    };

    const double vocab_raw = kVocabScale * std::pow(std::max(1.0, tokens), kVocabBeta);
    const int    vocab = std::clamp(static_cast<int>(std::lround(vocab_raw / 512.0)) * 512,
                                     kMinVocab, kMaxVocab);

    // Shape (d_model, n_layers, seq_len, realized param count) for a given head count -- everything
    // downstream of n_heads is exactly determined, so the VRAM shrink-loop below just walks n_heads down.
    // The aspect ratio itself stays pinned to the ORIGINAL corpus-derived target_params throughout (not
    // recomputed per shrink step): it answers "how deep should a model of the scale THIS CORPUS
    // justifies be," a separate question from "how far does hardware then force us to shrink it."
    const double aspect = aspect_for(target_params);
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
                                 double size_scale = 1.0) {
    return fill_defaults(pinned, autosize(corpus_bytes, vram_mb, size_scale));
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
        else if (key == "gpu_batch"          && val > 0) d.gpu_batch          = val;
        else if (key == "cuda_tf32")                   { d.cuda_tf32 = (val != 0); d.tf32_from_cache = true; }
    }
    if (d.gpu_batch <= 0) d.gpu_batch = d.threads * d.windows_per_thread;
    return d;
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
