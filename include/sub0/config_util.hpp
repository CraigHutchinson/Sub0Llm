// sub0/config_util.hpp — pure configuration-decision logic, factored out of the configurator's
// main() so it is unit-testable (no I/O, no globals). The configurator wires these to the corpus
// scan + header emission; tests drive them directly. Header-only, std-only.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <string>
#include <vector>

namespace sub0::config {

// --- Model auto-sizing -----------------------------------------------------
// The dimensions a corpus of a given size justifies, so a fresh corpus trains a sensibly-sized model
// without hand-tuning. A coarse ladder by corpus bytes (vocab tracks the --dump-vocab curve-knee
// findings); the exact ideal vocab is the curve knee, this is the default before analysis.
struct ModelDims { int d_model = 0, n_layers = 0, n_heads = 0, seq_len = 0, vocab = 0; };

inline ModelDims autosize(std::uintmax_t corpus_bytes) {
    const double mb = static_cast<double>(corpus_bytes) / 1e6;
    if (mb <    64.0) return {192,  6, 6, 256,  4096};   // ~tinystories (22 MB)
    if (mb <   512.0) return {320,  8, 8, 256,  8192};
    if (mb <  4096.0) return {448, 11, 7, 256, 16384};   // ~fineweb_smoke (1 GB)
    if (mb < 32768.0) return {640, 14, 8, 512, 24576};
    return                  {768, 16, 8, 512, 32768};    // ~full fineweb (45 GB)
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
inline ModelDims apply_autosize(ModelDims pinned, std::uintmax_t corpus_bytes) {
    return fill_defaults(pinned, autosize(corpus_bytes));
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
