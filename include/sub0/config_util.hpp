// sub0/config_util.hpp — pure configuration-decision logic, factored out of the configurator's
// main() so it is unit-testable (no I/O, no globals). The configurator wires these to the corpus
// scan + header emission; tests drive them directly. Header-only, std-only.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <istream>
#include <string>

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

// Fill any field left 0 (the "auto" sentinel) from the corpus-scaled default; a nonzero field (an
// explicit --dmodel/--vocab/... ) is preserved. Returns the resolved dims.
inline ModelDims apply_autosize(ModelDims pinned, std::uintmax_t corpus_bytes) {
    const ModelDims a = autosize(corpus_bytes);
    if (pinned.d_model  == 0) pinned.d_model  = a.d_model;
    if (pinned.n_layers == 0) pinned.n_layers = a.n_layers;
    if (pinned.n_heads  == 0) pinned.n_heads  = a.n_heads;
    if (pinned.seq_len  == 0) pinned.seq_len  = a.seq_len;
    if (pinned.vocab    == 0) pinned.vocab    = a.vocab;
    return pinned;
}

// --- Tuned runtime defaults (from the tune cache) --------------------------
// The threads / windows-per-thread / GPU-batch / TF32 / attn-backward a `tune` run persisted. Falls
// back to the hardware core count + conservative defaults when the cache is empty or a key is absent.
struct TuneDefaults {
    int  threads            = 1;
    int  windows_per_thread = 4;
    int  gpu_batch          = 0;   // 0 -> derived = threads * windows_per_thread (set on return)
    int  attn_bwd_per_query = 0;
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
        else if (key == "attn_bwd_per_query")            d.attn_bwd_per_query = (val != 0);
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

}  // namespace sub0::config
