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
#include "sub0/registry.hpp"
#include "sub0/tokmap.hpp"
#include "sub0/tune.hpp"
#include "sub0/window.hpp"   // sample_window_start: keep each training window inside one document
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <print>
#include <random>
#include <sstream>
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

// GPU training fast-path (Phase 2e). When the CUDA backend is compiled in (SUB0_BUILD_CUDA) and a
// device initializes, the training loop keeps the parameters resident on the GPU and runs each
// step (forward + backward + AdamW) there via sub0_cuda_train_step, syncing back to the host
// param/optimizer arenas only at eval/checkpoint boundaries (where the CPU eval/save/preview code
// runs unchanged). The device path is gradient/AdamW parity-tested against this CPU backend.
#if defined(SUB0_BUILD_CUDA)
extern "C" int  sub0_cuda_init();
extern "C" void sub0_cuda_shutdown();
extern "C" int  sub0_cuda_upload_params(const float* host);
extern "C" int  sub0_cuda_download_params(float* host);
extern "C" int  sub0_cuda_upload_opt(const float* host_m, const float* host_v);
extern "C" int  sub0_cuda_download_opt(float* host_m, float* host_v);
extern "C" int  sub0_cuda_train_step(const int* ids, const int* targets, int batch, int T,
                                     float lr, long t, double* out_loss, const int* lengths);
extern "C" void sub0_cuda_set_tf32(int on);
extern "C" void sub0_cuda_set_attn_bwd(int per_query);
extern "C" int  sub0_cuda_time_train_step(int batch, int T, double budget_ms, double* out_ms);
// Footprint validation: predicted (pure model) vs actual (measured cudaMemGetInfo delta) device MiB.
extern "C" int  sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb);
// Free dedicated VRAM (MiB) after the CUDA/cuBLAS context exists -- the usable budget for the ladder.
extern "C" int  sub0_cuda_free_vram_mb();
#endif

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
constexpr int    EVAL_WINDOWS_MAX    = 128;   // bounded cost per eval
constexpr int    MAX_EPOCHS_BACKSTOP = 30;    // ceiling if no plateau is detected
constexpr int    PLATEAU_WINDOW      = 6;     // evals fitted by the least-squares trend test
constexpr double PLATEAU_MIN_REL     = 0.005; // stop when the best-fit drop over the window < 0.5%
                                              // (~0.8%/epoch). 2% was too loose: it stopped on slow-
                                              // but-real tails (d96 @1.985 and d128 @2.68 both still
                                              // improving ~1.5%/window) well before the true floor.

// Variable-length training: each step draws a window width T in [MIN_TRAIN_SEQ, SEQ_LEN], shared
// across the batch (so the GPU keeps a single M = batch*T GEMM). Exposing a range of context
// lengths stops the model overfitting to exactly SEQ_LEN and makes it robust to short prompts. The
// floor is kept low (SEQ_LEN/8) so most documents fit a full window AND so short documents -- which
// are padded up to T with the loss masked off -- waste little compute. For a tiny context there is
// no room to vary, so MIN_TRAIN_SEQ collapses to SEQ_LEN (fixed). SUB0_FIXED_SEQ=1 forces full T.
constexpr int    MIN_TRAIN_SEQ = SEQ_LEN > 16 ? std::max(8, SEQ_LEN / 8) : SEQ_LEN;

constexpr std::uint32_t CKPT_MAGIC   = 0x4B433053u;  // "S0CK"
constexpr std::uint32_t CKPT_VERSION = 1u;

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
// When the build skipped corpus.tok (--corpus-tok 0) the token copy is never written to
// disk (~2x saved). Training instead tokenizes contiguous regions of the RAW text corpus
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
std::span<const int> load_corpus_tokens(sub0::TokMap& tok, std::vector<int>& od_buf, const char* tool) {
    if (tok.ok() && tok.vocab() == VOCAB && tok.tokens().size() > static_cast<std::size_t>(SEQ_LEN) + 2)
        return tok.tokens();                       // baked corpus.tok (the fast path)

    // No usable corpus.tok -> tokenize on demand from the raw corpus (matches the train stage).
    sub0::load_tokenizer(sub0::default_tokenizer());   // on-demand encode needs the runtime tokenizer
    TextCorpus text;
    if (text.open(sub0::default_corpus())) {
        std::mt19937 rng(123);
        text.fill_random(0, text.bytes(), OD_TRAIN_BUF_TOK, rng, od_buf);
    }
    if (od_buf.size() > static_cast<std::size_t>(SEQ_LEN) + 2) return od_buf;

    std::println(stderr, "{}: no usable corpus.tok ('{}') and cannot tokenize the raw corpus '{}'",
                 tool, sub0::default_corpus_tok(), sub0::default_corpus());
    return {};
}

// --- Validation NELBO -------------------------------------------------------
// Mean cross-entropy per token over a fixed, evenly-spaced set of windows in the
// held-out tail. Fixed windows make the metric comparable across evals (so the
// plateau sign test sees signal, not resampling noise) and bound the cost.
double evaluate(std::span<const int> data, std::size_t val_start) {
    const std::size_t last = data.size() - SEQ_LEN - 1;
    if (val_start > last) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t span = last - val_start;
    const int avail  = static_cast<int>(span / SEQ_LEN) + 1;
    const int nw     = std::min(EVAL_WINDOWS_MAX, std::max(1, avail));
    const std::size_t stride = (nw > 1) ? span / static_cast<std::size_t>(nw - 1) : 1;
    double total = 0.0;
    for (int w = 0; w < nw; ++w) {
        std::size_t s = val_start + static_cast<std::size_t>(w) * stride;
        if (s > last) s = last;
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(data.data() + s, SEQ_LEN);
        sub0::Node* loss   = sub0::cross_entropy(logits, data.data() + s + 1);
        total += loss->data[0];
    }
    sub0::graph_reset();
    return total / nw;
}

// Mean per-token predictive entropy (nats) of the model on the held-out tail: its
// average uncertainty when *reading* real text it never trained on, over the same fixed
// windows as evaluate(). exp() of this is the natural target for generation -- we match
// the model's sampling entropy to its reading entropy. Unlike cross-entropy (inflated by
// model error -> perplexity 5.17 here), entropy compares the model to ITSELF, so the
// matched temperature is not biased upward by an imperfect fit and centers near 1.
double mean_entropy(std::span<const int> data, std::size_t val_start) {
    const std::size_t last = data.size() - SEQ_LEN - 1;
    if (val_start > last) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t span = last - val_start;
    const int avail  = static_cast<int>(span / SEQ_LEN) + 1;
    const int nw     = std::min(EVAL_WINDOWS_MAX, std::max(1, avail));
    const std::size_t stride = (nw > 1) ? span / static_cast<std::size_t>(nw - 1) : 1;
    double total = 0.0; long n = 0;
    for (int w = 0; w < nw; ++w) {
        std::size_t s = val_start + static_cast<std::size_t>(w) * stride;
        if (s > last) s = last;
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(data.data() + s, SEQ_LEN);
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
// shedding less than PLATEAU_MIN_REL of the current level across the window. The fitted gradient is
// representative of the real downward trend and shrugs off the per-eval noise that tripped the old
// sign test (which false-stopped a strongly-but-noisily descending run -- see coherence_tests).
bool plateaued(const std::vector<double>& evals) {
    return sub0::coherence::trend_plateaued(evals, PLATEAU_WINDOW, PLATEAU_MIN_REL);
}

// --- Checkpoint (full optimizer + loop state, for exact resume) -------------
struct RunState {
    long step = 0;
    double best_loss = std::numeric_limits<double>::infinity();
    std::vector<double> evals;
};

template <class T> void wr(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof v);
}
template <class T> T rd(std::istream& is) {
    T v{}; is.read(reinterpret_cast<char*>(&v), sizeof v); return v;
}

void atomic_replace(const std::filesystem::path& tmp, const std::filesystem::path& dst) {
    std::error_code ec;
    std::filesystem::rename(tmp, dst, ec);
    if (ec) {  // some platforms (Windows) won't clobber an existing target
        std::filesystem::remove(dst, ec);
        std::filesystem::rename(tmp, dst, ec);
    }
}

// Write magic + the constexpr config (validated on load) + continuation state +
// RNG + eval history + parameters + both Adam moments. Written to a temp file then
// renamed so a crash mid-write never corrupts a usable checkpoint. The step budget
// (max_steps) is deliberately NOT stored: it is a per-invocation policy, so a
// resumed run continues toward the budget the *current* call asks for.
void save_checkpoint(const std::string& path, long adam_t, const std::mt19937& rng,
                     const RunState& rs, int batch, float lr, unsigned seed) {
    const std::string tmp = path + ".tmp";
    std::ofstream os(tmp, std::ios::binary);
    if (!os) { std::println(stderr, "train: cannot write checkpoint '{}'", tmp); return; }

    wr(os, CKPT_MAGIC); wr(os, CKPT_VERSION);
    for (int c : {D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB, int(USE_TERNARY)}) wr(os, c);
    const std::uint64_t nfloat = sub0::trainable_floats();
    wr(os, nfloat);

    wr(os, static_cast<std::int64_t>(rs.step));
    wr(os, static_cast<std::int64_t>(adam_t));
    wr(os, rs.best_loss);
    wr(os, static_cast<std::int32_t>(batch));
    wr(os, static_cast<std::uint32_t>(seed));
    wr(os, lr);

    std::ostringstream rs_os; rs_os << rng;            // exact mt19937 state
    const std::string rng_state = rs_os.str();
    wr(os, static_cast<std::uint32_t>(rng_state.size()));
    os.write(rng_state.data(), static_cast<std::streamsize>(rng_state.size()));

    wr(os, static_cast<std::uint32_t>(rs.evals.size()));
    for (double e : rs.evals) wr(os, e);

    const auto bytes = static_cast<std::streamsize>(nfloat * sizeof(float));
    sub0::sync_params_to_host();   // device backends: stage live params/moments into the *_ptr() buffers
    os.write(reinterpret_cast<const char*>(sub0::params_ptr()), bytes);
    os.write(reinterpret_cast<const char*>(sub0::adam_m_ptr()), bytes);
    os.write(reinterpret_cast<const char*>(sub0::adam_v_ptr()), bytes);
    os.flush();
    if (!os) { std::println(stderr, "train: checkpoint write failed '{}'", tmp); return; }
    os.close();
    atomic_replace(tmp, path);
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
    if (rd<std::uint32_t>(is) != CKPT_MAGIC || rd<std::uint32_t>(is) != CKPT_VERSION) {
        std::println(stderr, "train: ignoring checkpoint '{}' (bad magic/version)", path);
        return false;
    }
    bool ok = true;
    for (int ref : {D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB, int(USE_TERNARY)})
        if (rd<int>(is) != ref) ok = false;
    const std::uint64_t nfloat = rd<std::uint64_t>(is);
    if (!ok || nfloat != sub0::trainable_floats()) {
        std::println(stderr, "train: ignoring checkpoint '{}' (built for a different config)", path);
        return false;
    }

    rs.step      = static_cast<long>(rd<std::int64_t>(is));
    adam_t       = static_cast<long>(rd<std::int64_t>(is));
    rs.best_loss = rd<double>(is);
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

    const auto bytes = static_cast<std::streamsize>(nfloat * sizeof(float));
    is.read(reinterpret_cast<char*>(sub0::params_ptr()), bytes);
    is.read(reinterpret_cast<char*>(sub0::adam_m_ptr()), bytes);
    is.read(reinterpret_cast<char*>(sub0::adam_v_ptr()), bytes);
    if (!is) { std::println(stderr, "train: checkpoint '{}' truncated -- starting fresh", path); return false; }
    sub0::sync_params_to_device();   // device backends: push loaded params/moments to the live copy
    return true;
}

// A short text sample at a given temperature/top-k. Uses the SAME sampler as `gen`
// so the output reflects real generation quality -- the old greedy+noise hack made a
// coherent model look like word-salad.
std::string preview_at(const std::string& prompt, int n, float temp, int topk, std::mt19937& rng) {
    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
        const int last = logits->rows - 1;
        ctx.push_back(sub0::sample_token(logits->data.data() + (size_t)last * VOCAB, temp, topk, rng));
    }
    sub0::graph_reset();
    return sub0::detokenize(ctx);
}

// Mid-training preview at the generation defaults (temp 0.7, top-k 20).
std::string preview(const std::string& prompt, int n, std::mt19937& rng) {
    return preview_at(prompt, n, 0.7f, 20, rng);
}

}  // namespace

// Device training session (Phase 2e): keeps the parameters resident on the GPU and runs each
// forward+backward+AdamW step there, syncing back to the host param/optimizer arenas only at the
// eval/checkpoint boundaries where the unchanged CPU eval/save/preview code runs. No-op (active
// stays false) unless the CUDA backend is compiled in and a device initializes.
namespace {
struct GpuTrainer {
    bool active = false;
    long t = 0;                       // AdamW step counter driving the device bias correction
    std::vector<int> ids, targets;    // per-step [batch*SEQ_LEN] window buffers

    // Enable the device path: requires the CUDA backend, a device, and a batch within the
    // resident scratch width. Uploads the current host params (+ optimizer moments, zero on a
    // fresh run / restored on resume). Returns false to fall back to the CPU loop.
    bool enable(int batch, long resume_t) {
#if defined(SUB0_BUILD_CUDA)
        constexpr int kCap = 4096;                  // matches MAX_FWD_BATCH (the device scratch ceiling)
        //TODO: Remove getenv calls - these are tmeporary and we should use commandline/compiletime
        if (std::getenv("SUB0_TRAIN_CPU")) return false;   // measurement / fallback override
        if (!HAS_CUDA || batch > kCap) return false;
        if (sub0_cuda_init() != 0) return false;
        if (sub0_cuda_upload_params(sub0::params_ptr()) != 0) return false;
        sub0_cuda_upload_opt(sub0::adam_m_ptr(), sub0::adam_v_ptr());
        t = resume_t;
        active = true;
        ids.resize(static_cast<std::size_t>(batch) * SEQ_LEN);
        targets.resize(ids.size());
        return true;
#else
        (void)batch; (void)resume_t; return false;
#endif
    }

    // One resident device step over the sampled windows; returns the mean loss. Builds the
    // [batch*T] id/target windows (x = data[start+s], y = data[start+s+1]); a window shorter than T
    // (a short document) fills its leading `lengths[b]` rows and PADS the rest, with the trailing
    // positions loss-masked on the device via the same `lengths` array -- so no document is dropped
    // and the padding contributes no gradient (causal attention keeps it out of the real tokens).
    float step([[maybe_unused]] std::span<const int> data, [[maybe_unused]] const std::size_t* starts,
               [[maybe_unused]] const int* lengths, [[maybe_unused]] int batch,
               [[maybe_unused]] int T, [[maybe_unused]] float lr) {
#if defined(SUB0_BUILD_CUDA)
        for (int b = 0; b < batch; ++b) {
            const int* w   = data.data() + starts[b];
            const int  len = lengths ? lengths[b] : T;
            int s = 0;
            for (; s < len; ++s) {
                ids[static_cast<std::size_t>(b) * T + s]     = w[s];
                targets[static_cast<std::size_t>(b) * T + s] = w[s + 1];
            }
            for (; s < T; ++s) {                       // pad the tail of a short window (loss-masked)
                ids[static_cast<std::size_t>(b) * T + s]     = 0;
                targets[static_cast<std::size_t>(b) * T + s] = 0;
            }
        }
        double loss = 0.0;
        sub0_cuda_train_step(ids.data(), targets.data(), batch, T, lr, ++t, &loss, lengths);
        return static_cast<float>(loss);
#else
        return 0.0f;
#endif
    }

    // Refresh the host param + optimizer arenas from the device (before eval / checkpoint / save).
    void sync_to_host() {
#if defined(SUB0_BUILD_CUDA)
        if (!active) return;
        sub0_cuda_download_params(sub0::params_ptr());
        sub0_cuda_download_opt(sub0::adam_m_ptr(), sub0::adam_v_ptr());
#endif
    }

    void shutdown() {
#if defined(SUB0_BUILD_CUDA)
        if (active) sub0_cuda_shutdown();
#endif
        active = false;
    }
};
}  // namespace

// Shared run-context banner (model/compute/training-backend), defined below; train and tune share it.
static void report_run_context(bool gpu_train);

extern "C" SUB0_API int sub0_train_stage(const char* corpus_path, const char* model_out,
                                          int steps, int batch, float lr, unsigned seed) {
    // Training consumes the pre-tokenized corpus produced by the configurator, not
    // raw text. Any argument that is not a .tok file (including the driver's default
    // .txt corpus path) is intentionally ignored in favour of this build's baked-in
    // corpus.tok: the engine's VOCAB is fixed at configure time, so only the .tok the
    // configurator emitted for it can be trained against. To train on different text,
    // reconfigure/rebuild so the configurator retokenizes and re-bakes VOCAB.
    std::string tok_path = corpus_path ? corpus_path : "";
    if (tok_path.size() < 4 || tok_path.compare(tok_path.size() - 4, 4, ".tok") != 0)
        tok_path = sub0::default_corpus_tok();

    sub0::load_tokenizer(sub0::default_tokenizer());   // previews + on-demand encode
    sub0::build_model();

    // Where training windows come from. train_span is sampled for training windows; val_span
    // is the fixed held-out eval region. Backed either by the corpus.tok memory map, or --
    // when no corpus.tok was built (--corpus-tok 0) -- by bounded in-memory buffers tokenized
    // on demand from the raw corpus (see TextCorpus): no token copy on disk, RAM capped by
    // the buffers. The rotating training buffer is refilled each interval inside the loop.
    sub0::TokMap tok(tok_path);
    const bool on_demand = !tok.ok();

    std::span<const int> train_span, val_span;
    std::span<const std::uint32_t> doc_index;  // document-start token indices (empty => flat sampling)
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
            std::println(stderr,
                         "train: '{}' was tokenized for vocab {} but this engine was built for VOCAB {}.\n"
                         "       Reconfigure/rebuild against this corpus, or pass a matching .tok.",
                         tok_path, tok.vocab(), VOCAB);
            return 1;
        }
        const std::span<const int> data = tok.tokens();
        const std::size_t min_tokens = 2 * (static_cast<std::size_t>(SEQ_LEN) + 2);
        if (data.size() < min_tokens) {
            std::println(stderr, "train: '{}' has only {} tokens, too few for seq_len {} (need >= {})",
                         tok_path, data.size(), SEQ_LEN, min_tokens);
            return 1;
        }
        // Defense in depth: a corrupt file could hold an out-of-range id. One linear scan
        // turns a silent OOB gather into a clear diagnostic (and faults in the mapping once,
        // but training's random windows touch most of it across epochs anyway).
        for (std::size_t i = 0; i < data.size(); ++i)
            if (data[i] < 0 || data[i] >= VOCAB) {
                std::println(stderr, "train: '{}' token {} = {} is out of range [0,{})",
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
            std::println(stderr, "train: no corpus.tok ('{}') and cannot open raw corpus '{}'", tok_path, raw);
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
            std::println(stderr, "train: on-demand corpus '{}' too small to tokenize a window (need >= {} tokens)",
                         raw, need);
            return 1;
        }
        train_span = train_buf;
        val_span   = val_buf;
        // Estimate tokens/byte from one region to size the epoch schedule (heuristic only).
        std::vector<int> probe; text.encode_region(0, probe);
        const double tpb = probe.empty() ? 0.5 : static_cast<double>(probe.size()) / static_cast<double>(OD_REGION_BYTES);
        est_train_tokens = static_cast<std::size_t>(static_cast<double>(train_byte_hi) * tpb);
        src_desc = std::format("{} (on-demand; ~{} train tok est / {}-tok shuffle buf / {}-tok val buf)",
                               raw, est_train_tokens, train_buf.size(), val_buf.size());
    }

    std::mt19937 rng(seed);
    RunState rs;

    // Model storage: an explicit path is honoured as-is; otherwise lay the model out in a
    // structured, identity-named directory (corpus + dims + git SHA) under the models root and
    // register it with a meta.txt, so `models` can discover it and prune incompatible ones. The
    // derived path is deterministic, so re-running `train` with no path resumes the same model.
    std::string model_path;
    std::filesystem::path meta_dir;
    const std::string created = sub0::registry::now_iso();
    long epoch_steps = 1;   // real value set with the schedule below; meta writes read it live
    if (model_out && *model_out) {
        model_path = model_out;
        meta_dir   = std::filesystem::path(model_path).parent_path();   // keep meta.txt beside the
        // model so an explicit-path run (including a resume) still records provenance, not just the
        // auto-named layout below. A bare filename (no parent) leaves meta_dir empty -> write_meta skips.
    } else {
        meta_dir = sub0::registry::model_dir(SUB0_MODELS_ROOT, sub0::default_corpus(),
                                             D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                                             static_cast<int>(USE_TERNARY),
                                             static_cast<int>(POS_ENCODING), SUB0_GIT_SHA);
        std::error_code ec; std::filesystem::create_directories(meta_dir, ec);
        model_path = (meta_dir / "model.bin").string();
        std::println("model dir: {}", model_path);
    }
    auto write_meta = [&](const char* status) {
        if (meta_dir.empty()) return;
        sub0::registry::ModelMeta m;
        m.corpus = sub0::registry::corpus_tag(sub0::default_corpus());
        m.d_model = D_MODEL; m.n_layers = N_LAYERS; m.n_heads = N_HEADS;
        m.seq_len = SEQ_LEN; m.vocab = VOCAB; m.ternary = static_cast<int>(USE_TERNARY);
        m.pos_encoding = static_cast<int>(POS_ENCODING);
        m.git_sha = SUB0_GIT_SHA; m.created = created;
        m.updated = sub0::registry::now_iso();
        m.steps = rs.step;
        m.epochs = static_cast<double>(rs.step) / static_cast<double>(epoch_steps);
        m.tokens_seen = static_cast<long long>(rs.step) * batch * SEQ_LEN;  // approximate (variable T)
        m.batch = batch; m.lr = lr; m.seed = seed;
        m.best_val_nelbo = (rs.best_loss < std::numeric_limits<double>::infinity()) ? rs.best_loss : -1.0;
        m.status = status;
        sub0::registry::write_meta(meta_dir, m);
    };
    write_meta("training");   // register early so an interrupted run still leaves a discoverable model

    // Resume if a checkpoint for this output exists (overwrites the fresh model and
    // restores batch/lr/seed, so the schedule below is computed from the resumed
    // batch, not whatever the command line happened to pass).
    const std::string ckpt_path = model_path + ".ckpt";
    long adam_t = 0;
    const bool resumed = load_checkpoint(ckpt_path, rng, rs, batch, lr, seed, adam_t);

    sub0::AdamW opt(lr);
    if (resumed) opt.set_step_count(adam_t);

    // Phase 2e: try to run the training loop on the GPU (params resident on device). Falls back
    // to the CPU data-parallel path when the CUDA backend is absent, no device initializes, or the
    // batch exceeds the device's resident scratch width.
    GpuTrainer gpu;
    const bool gpu_train = gpu.enable(batch, opt.step_count());

    // Corpus-relative schedule (max_steps is a per-invocation budget, never restored). For
    // on-demand the token count is estimated from tokens/byte (the schedule is heuristic and
    // plateau-stopped, so an estimate is fine).
    const long tokens_per_step = static_cast<long>(batch) * SEQ_LEN;
    epoch_steps  = std::max<long>(1, (static_cast<long>(est_train_tokens) + tokens_per_step - 1) / tokens_per_step);
    const long warmup_steps = std::max<long>(1, std::lround(EVAL_WARMUP_EPOCHS  * epoch_steps));
    const long eval_every   = std::max<long>(1, std::lround(EVAL_INTERVAL_EPOCHS * epoch_steps));
    const long max_steps = (steps > 0) ? steps : static_cast<long>(MAX_EPOCHS_BACKSTOP) * epoch_steps;

    std::print("corpus: {} | ", src_desc);
    report_run_context(gpu_train);
    std::println("schedule: {} steps/epoch | warmup {} | eval every {} | max {} steps ({} epochs){}{}",
                 epoch_steps, warmup_steps, eval_every, max_steps,
                 (max_steps + epoch_steps - 1) / epoch_steps,
                 on_demand ? " | on-demand" : "",
                 resumed ? std::format(" | RESUMED at step {}", rs.step) : std::string{});
    std::fflush(stdout);

    std::vector<size_t> starts(batch);
    std::vector<int>    win_len(batch);   // per-window trained length (< seq_t for short documents)
    long steps_since_refresh = 0;

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
    double run_loss = 0.0; int run_n = 0;
    bool stop = false;
    // Variable-length training is on by default; SUB0_FIXED_SEQ=1 forces full-length windows.
    //TODO: Remove getenv calls
    const bool vary_seq = (MIN_TRAIN_SEQ < SEQ_LEN) && (std::getenv("SUB0_FIXED_SEQ") == nullptr);

    for (long step = rs.step + 1; step <= max_steps && !stop; ++step) {
        // On-demand: refill the rotating shuffle buffer once per eval interval from new
        // random regions, so the run traverses the whole corpus over time. buf_rng is a
        // separate stream, so this never perturbs the resume-critical `rng`. The refill may
        // reallocate train_buf, so re-bind the span (the sampler reads train_span.size() live).
        if (on_demand && steps_since_refresh >= eval_every) {
            buf_rng.seed(static_cast<std::uint32_t>(seed) ^ (0x9E3779B9u * static_cast<std::uint32_t>(++refresh_n)));
            text.fill_random(train_byte_lo, train_byte_hi, OD_TRAIN_BUF_TOK, buf_rng, train_buf);
            train_span = train_buf;
            steps_since_refresh = 0;
        }
        ++steps_since_refresh;

        // Draw the window starts on the main thread (keeps the RNG stream, hence
        // resume, deterministic), then run the batch -- on the GPU when enabled, else
        // data-parallel across CPU threads.
        const int seq_t = vary_seq ? std::uniform_int_distribution<int>(MIN_TRAIN_SEQ, SEQ_LEN)(rng) : SEQ_LEN;
        for (int b = 0; b < batch; ++b) {
            const sub0::Window win = sub0::sample_window(rng, seq_t, train_span.size(), doc_index);
            starts[b]  = win.start;
            win_len[b] = win.len;
        }
        float step_loss;
        if (gpu_train) {
            step_loss = gpu.step(train_span, starts.data(), win_len.data(), batch, seq_t, lr);
            opt.set_step_count(gpu.t);          // keep the checkpoint's step counter in lockstep
        } else {
            step_loss = sub0::train_batch(train_span.data(), starts.data(), batch, seq_t, win_len.data());
            opt.step();
        }
        run_loss += step_loss; ++run_n;
        rs.step = step;

        const bool is_eval = (step % eval_every == 0 || step == max_steps);
        if (is_eval) {
            // Refresh the host param/optimizer arenas from the device so the CPU eval/save/preview
            // code below sees the current weights (no-op on the CPU path).
            gpu.sync_to_host();
            // Throughput over the interval just completed (training only).
            const double secs = std::chrono::duration<double>(clock::now() - win_t0).count();
            const double wps   = secs > 0 ? static_cast<double>((step - win_steps0) * batch) / secs : 0.0;
            const double frac_epoch = static_cast<double>(step) / epoch_steps;

            // Validation NELBO only once enough of the corpus has been seen.
            std::string eval_str = "(warmup)";
            if (step >= warmup_steps) {
                const double nelbo = evaluate(val_span, 0);
                rs.evals.push_back(nelbo);
                rs.best_loss = std::min(rs.best_loss, nelbo);
                eval_str = std::format("val_nelbo {:.4f} (best {:.4f})", nelbo, rs.best_loss);
                if (plateaued(rs.evals)) stop = true;
            }
            std::println("step {:>7}/{} [{:.2f} ep]  train {:.4f}  {:.0f} win/s ({:.0f} tok/s)  {}",
                         step, max_steps, frac_epoch, run_loss / std::max(1, run_n),
                         wps, wps * SEQ_LEN, eval_str);
            std::fflush(stdout);

            // Checkpoint + latest model every interval (covers the warmup phase too).
            save_checkpoint(ckpt_path, opt.step_count(), rng, rs, batch, lr, seed);
            sub0::save_model(model_path.c_str());
            write_meta("training");          // refresh best_val_nelbo as it improves

            run_loss = 0.0; run_n = 0;
            win_t0 = clock::now(); win_steps0 = step;
            last_log = win_t0; last_log_step = step;          // an eval counts as a log: reset the tick timer
        } else if (std::chrono::duration<double>(clock::now() - last_log).count() >= TICK_SECONDS) {
            // Interim heartbeat between evals: train loss (running avg since the last eval) +
            // throughput since the last printed line. No val NELBO here -- that stays on the eval
            // cadence (it is the expensive, plateau-driving measurement).
            const double since = std::chrono::duration<double>(clock::now() - last_log).count();
            const double wps   = since > 0 ? static_cast<double>((step - last_log_step) * batch) / since : 0.0;
            const double frac_epoch = static_cast<double>(step) / epoch_steps;
            std::println("  ~ step {:>7}/{} [{:.2f} ep]  train {:.4f}  {:.0f} win/s ({:.0f} tok/s)",
                         step, max_steps, frac_epoch, run_loss / std::max(1, run_n), wps, wps * SEQ_LEN);
            std::fflush(stdout);
            last_log = clock::now(); last_log_step = step;
        }
    }

    gpu.sync_to_host();    // ensure the host arenas hold the final device weights before saving
    sub0::save_model(model_path.c_str());
    save_checkpoint(ckpt_path, opt.step_count(), rng, rs, batch, lr, seed);
    write_meta(stop ? "plateaued" : "trained");
    // Generate a full SEQ_LEN-token sample so the preview actually fills and slides the extended
    // context window (preview_at caps the model input at SEQ_LEN; a short 120-token run never
    // reaches it). This exercises the real long-context behaviour the trained window supports.
    std::println("  --- sample ({}-token context) ---\n  {}", SEQ_LEN, preview("the ", SEQ_LEN, rng));
    std::println("{} at step {} (best val_nelbo {:.4f}) -> {}",
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
static void report_run_context(bool gpu_train) {
    sub0::print_config();
    std::println("training backend: {}", gpu_train ? "GPU (resident device step: fwd+bwd+AdamW)"
                                                    : "CPU (data-parallel minibatch)");
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
    const std::span<const int> data = load_corpus_tokens(tok, od_buf, "bench");
    if (data.empty()) return 1;
    sub0::build_model();
    sub0::AdamW opt(0.001f);
    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);

    std::vector<std::uint64_t> fwd, bwd, optc, step;
    fwd.reserve(iters); bwd.reserve(iters); optc.reserve(iters); step.reserve(iters);

    const int warmup = std::max(5, iters / 10);
    for (int it = -warmup; it < iters; ++it) {
        const size_t s = startd(rng);
        const int* x = data.data() + s;
        const int* y = data.data() + s + 1;

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
    for (int i = 0; i < CAL; ++i) { sub0::graph_reset(); (void)sub0::forward(data.data() + startd(rng), SEQ_LEN); }
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
        std::vector<std::size_t> starts(static_cast<size_t>(B));
        std::vector<std::uint64_t> batch_cyc;
        const int bwarm = 3, biters = std::max(20, iters / 4);
        batch_cyc.reserve(static_cast<size_t>(biters));
        for (int it = -bwarm; it < biters; ++it) {
            for (int b = 0; b < B; ++b) starts[static_cast<size_t>(b)] = startd(rng);
            const std::uint64_t b0 = cpu_cycles();
            sub0::train_batch(data.data(), starts.data(), B, SEQ_LEN);
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

DpMeasure measure_dp_throughput(std::span<const int> data, std::mt19937& rng,
                                int threads, int windows_per_thread, double budget_ms) {
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#endif
    const int active = threads > 0 ? threads : 1;
    const int B = active * std::max(1, windows_per_thread);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);
    std::vector<std::size_t> starts(static_cast<size_t>(B));

    auto one_step = [&] {
        for (int b = 0; b < B; ++b) starts[static_cast<size_t>(b)] = startd(rng);
        sub0::train_batch(data.data(), starts.data(), B, SEQ_LEN);
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
// cross-check that prediction against the device's actual usage.
static constexpr sub0::memplan::Dims kGpuDims{ D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB };

extern "C" SUB0_API int sub0_tune_stage(int max_threads, int verbose, int backend) {
    const bool run_cpu = backend != TUNE_BACKEND_GPU;   // gpu-only skips the CPU sweep
    const bool run_gpu = backend != TUNE_BACKEND_CPU;   // cpu-only skips the device-step sweep

    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const std::span<const int> data = load_corpus_tokens(tok, od_buf, "tune");
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

        sub0::tune::Options opt;
        // After the hot coarse sweep, pause briefly before each refinement pass and confirmation
        // round so the cores shed heat and the narrowing-down readings are steadier.
        opt.settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); };
        // Lengthen each measurement's budget as the search narrows: a quick burst while exploring
        // broadly, then progressively longer throttled runs while pinning down the winner. The
        // phase header also makes it clear in the live output (and a captured log) which stage we
        // are in.
        opt.on_phase = [&](sub0::tune::Phase p) {
            switch (p) {
                case sub0::tune::Phase::Explore:
                    budget_ms = 200.0;
                    std::println("[explore] wide sweep, short runs (~0.2s/point)");
                    break;
                case sub0::tune::Phase::Refine:
                    budget_ms = 500.0;
                    std::println("[refine]  narrowing in, longer runs (~0.5s/point)");
                    break;
                case sub0::tune::Phase::Confirm:
                    budget_ms = 1200.0;
                    std::println("[confirm] re-measuring finalists, longest runs (~1.2s/point, dropping clear losers)");
                    break;
            }
            std::fflush(stdout);
        };
        sub0::tune::Result r = sub0::tune::maximize(space, objective, opt);

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
    int gpu_batch = best_threads * best_wpt;        // default: the CPU-tuned data-parallel width
    int gpu_tf32  = CUDA_TF32 ? 1 : 0;
    int gpu_attn  = 0;                              // per-head attention backward (the default)
    if (!run_gpu) {                                 // --backend cpu: keep any cached GPU knobs intact
        read_tune_cache_value("gpu_batch", &gpu_batch);
        read_tune_cache_value("cuda_tf32", &gpu_tf32);
        read_tune_cache_value("attn_bwd_per_query", &gpu_attn);
    }
#if defined(SUB0_BUILD_CUDA)
    if (run_gpu && HAS_CUDA && sub0_cuda_init() == 0) {
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
            if (sub0_cuda_train_footprint(64, &pred_mb, &act_mb) == 0 && act_mb > 0.0) {
                const double gap = std::fabs(pred_mb - act_mb);
                std::println("footprint check @ batch 64: predicted {:.0f} MiB | measured {:.0f} MiB | gap {:.0f} MiB{}",
                             pred_mb, act_mb, gap,
                             gap > sub0::memplan::FOOTPRINT_TOLERANCE_MB
                                 ? "  !! memplan.hpp is STALE -- update the device footprint model !!" : "");
                std::fflush(stdout);
            }
        }

        // The SAME robust search the CPU sweep uses (sub0::tune::maximize): a joint coarse grid +
        // top-basin refinement + median-of-samples confirmation. This replaces the old single-pass
        // batch sweep that stopped at the first sub-6% throughput gain -- a rule a single noisy
        // reading could trip early, and which silently settled on a LOCAL trough (the measured curve
        // dips at batch 512 below 384 before climbing higher again at 768, so "first plateau" picked
        // the worse of two peaks). The joint grid is the right tool because the knobs INTERACT: the
        // per-query attention backward wins at small batch but loses to per-head at large batch
        // (atomic contention), so the best attn strategy depends on the chosen batch. Objective =
        // measured tok/s; the device timer is budget-sized so each sample costs ~the same wall time.
        sub0_cuda_set_tf32(CUDA_TF32 ? 1 : 0); sub0_cuda_set_attn_bwd(0);   // baseline for batch-only builds
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
        const int free_mb = sub0_cuda_free_vram_mb();
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
        gspace.push_back({"attn_bwd", {0, 1}});
#endif
        // Format the optional knob columns (TF32 / attn strategy) only when they are in the space.
        auto knob_suffix = [](const sub0::tune::Assignment& a) -> std::string {
            std::string s;
            if (a.size() > 1) s += std::format("  tf32={}", static_cast<int>(std::lround(a[1])));
            if (a.size() > 2) s += std::format("  attn={:<5}", std::lround(a[2]) ? "query" : "head");
            return s;
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
            if (a.size() > 1) sub0_cuda_set_tf32(static_cast<int>(std::lround(a[1])));
            if (a.size() > 2) sub0_cuda_set_attn_bwd(static_cast<int>(std::lround(a[2])));
            double ms = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            const int rc = sub0_cuda_time_train_step(batch, SEQ_LEN, gbudget_ms, &ms);
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

        sub0::tune::Options gopt;
        gopt.settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); };
        gopt.confirm_rounds = 2;            // GPU steps cost seconds each: lighter confirmation than CPU
        gopt.on_phase = [&](sub0::tune::Phase p) {
            switch (p) {
                case sub0::tune::Phase::Explore:
                    gbudget_ms = 400.0;
                    std::println("[explore] joint batch x knob grid (~0.4s/sample)");
                    break;
                case sub0::tune::Phase::Refine:
                    gbudget_ms = 700.0;
                    std::println("[refine]  zooming the leading basins (~0.7s/sample)");
                    break;
                case sub0::tune::Phase::Confirm:
                    gbudget_ms = 1000.0;
                    std::println("[confirm] re-measuring finalists by median (~1.0s/sample)");
                    break;
            }
            std::fflush(stdout);
        };
        const sub0::tune::Result gr = sub0::tune::maximize(gspace, gobjective, gopt);

        gpu_batch = static_cast<int>(std::lround(gr.best[0]));
#if defined(SUB0_TUNING)
        gpu_tf32 = static_cast<int>(std::lround(gr.best[1]));
        gpu_attn = static_cast<int>(std::lround(gr.best[2]));
#else
        std::println("  (TF32 / attn-backward are baked in this build; rebuild with -DSUB0_TUNING=ON to tune them)");
#endif
        std::println("");
        std::println("evaluated {} device measurements; winner confirmed over {} samples",
                     gr.evaluations, gr.best_samples);
        std::println("best GPU: batch={}  tf32={}  attn_bwd={}  ->  {:.0f} tok/s (median)",
                     gpu_batch, gpu_tf32, gpu_attn ? "query" : "head", gr.best_score);
        sub0_cuda_shutdown();
    }
#endif

    // Persist ALL tuned throughput knobs ONCE so the next build bakes them into the config header
    // (DEFAULT_THREADS / DEFAULT_WINDOWS_PER_THREAD / DEFAULT_GPU_BATCH / CUDA_TF32 / ATTN_BWD_PER_QUERY).
    if (DEFAULT_TUNE_CACHE[0] == '\0') {
        std::println("(no tune cache configured; tuned defaults not persisted)");
    } else if (std::ofstream cache(DEFAULT_TUNE_CACHE, std::ios::trunc); cache) {
        cache << "threads=" << best_threads << "\n"
              << "windows_per_thread=" << best_wpt << "\n";
#if defined(SUB0_BUILD_CUDA)
        cache << "gpu_batch=" << gpu_batch << "\n"
              << "cuda_tf32=" << gpu_tf32 << "\n"
              << "attn_bwd_per_query=" << gpu_attn << "\n";
#endif
        std::println("persisted tuned defaults to {}", DEFAULT_TUNE_CACHE);
        std::println("rebuild to bake them in:  cmake --build --preset native");
    } else {
        std::println(stderr, "warning: could not write tune cache '{}'", DEFAULT_TUNE_CACHE);
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
GenStats gen_self_stats(std::span<const int> data, std::size_t val_start,
                        float temp, int topk, int n_seeds, int gen_len, unsigned cr_seed) {
    const std::size_t last = data.size() - SEQ_LEN - 1;
    const std::size_t span = (last > val_start) ? last - val_start : 0;
    const int prefix_len = std::max(1, std::min(SEQ_LEN / 4, 16));
    std::mt19937 rng(cr_seed);                          // common random numbers across temps

    double nll_sum = 0.0, rep_sum = 0.0; long nll_n = 0;
    std::vector<int> gen; gen.reserve(static_cast<std::size_t>(gen_len));
    for (int k = 0; k < n_seeds; ++k) {
        std::size_t s = val_start +
            (n_seeds > 1 ? static_cast<std::size_t>(k) * span / static_cast<std::size_t>(n_seeds - 1) : 0);
        if (s > last) s = last;
        std::vector<int> ctx(data.begin() + static_cast<std::ptrdiff_t>(s),
                             data.begin() + static_cast<std::ptrdiff_t>(s) + prefix_len);
        gen.clear();
        for (int g = 0; g < gen_len; ++g) {
            const int T = std::min(static_cast<int>(ctx.size()), SEQ_LEN);
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
            const float* row = logits->data.data() + static_cast<std::size_t>(logits->rows - 1) * VOCAB;
            const int tok = sub0::sample_token(row, temp, topk, rng);
            // Surprise of the chosen token under the TRUE (T=1) model distribution.
            float mx = -1e30f; for (int j = 0; j < VOCAB; ++j) mx = std::max(mx, row[j]);
            double Z = 0.0; for (int j = 0; j < VOCAB; ++j) Z += std::exp(static_cast<double>(row[j] - mx));
            nll_sum += -(static_cast<double>(row[tok] - mx) - std::log(Z));
            ++nll_n;
            gen.push_back(tok);
            ctx.push_back(tok);
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
double real_windowed_repeat(std::span<const int> data, std::size_t val_start,
                            int n_seeds, int win) {
    if (data.size() <= static_cast<std::size_t>(win) + 1) return 0.0;
    const std::size_t last = data.size() - 1 - static_cast<std::size_t>(win);
    const std::size_t span = (last > val_start) ? last - val_start : 0;
    double sum = 0.0;
    for (int k = 0; k < n_seeds; ++k) {
        std::size_t s = val_start +
            (n_seeds > 1 ? static_cast<std::size_t>(k) * span / static_cast<std::size_t>(n_seeds - 1) : 0);
        if (s > last) s = last;
        sum += ngram_repeat(data.data() + s, win);
    }
    return n_seeds > 0 ? sum / static_cast<double>(n_seeds) : 0.0;
}

using sub0::coherence::Crossing;       // monotone-crossing interpolation (see coherence.hpp)
using sub0::coherence::interp_cross;

}  // namespace

extern "C" SUB0_API int sub0_autotemp_stage(const char* model_in, unsigned seed, int verbose) {
    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const std::span<const int> data = load_corpus_tokens(tok, od_buf, "autotemp");
    if (data.empty()) return 1;

    sub0::build_model();
    if (!sub0::load_model(model_in)) {
        std::println(stderr, "autotemp: cannot load model '{}'", model_in);
        return 1;
    }
    sub0::load_tokenizer(sub0::default_tokenizer());    // for the sample print

    // The SAME held-out split the trainer uses, so the target perplexity is measured on
    // text the model never trained on.
    const std::size_t val_tokens = std::max<std::size_t>(
        static_cast<std::size_t>(SEQ_LEN) + 2,
        static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(data.size())));
    const std::size_t val_start = data.size() - val_tokens;

    // Sweep/scoring config. Top-k mirrors gen's default so the recommended temperature
    // transfers directly to the real generation path.
    constexpr int AUTOTEMP_TOPK = 20;
    constexpr int N_SEEDS = 10, GEN_LEN = 96;   // long windows so repetition/looping is visible

    // Two self-validating anchors from the real held-out text:
    //  * perplexity -- the model's reading entropy (compares the model to itself, so an
    //    imperfect fit does not bias it; we match generation perplexity to this);
    //  * repetition -- real text's 4-gram repeat rate, measured the SAME windowed way as
    //    the generations, so the comparison is fair. This is the robust anchor: it is a
    //    surface property of real text, independent of whether the model is well-calibrated.
    const double ce_ppl    = std::exp(evaluate(data, val_start));          // headline (inflated by model error)
    const double target_ppl = std::exp(mean_entropy(data, val_start));    // entropy-perplexity (match target)
    const double target_rep = real_windowed_repeat(data, val_start, N_SEEDS, GEN_LEN);

    std::println("autotemp: model {}", model_in);
    std::println("held-out (real text): cross-entropy perplexity {:.2f} | reading entropy {:.2f} | 4-gram repeat {:.1f}%",
                 ce_ppl, target_ppl, 100.0 * target_rep);
    std::fflush(stdout);

    // One honest sweep over a fixed temperature grid; the table IS the evidence. gen_ppl
    // rises with temperature, 4-gram repeat falls, so each target has exactly one crossing
    // -- recovered by interpolation (no noisy bisection, and both anchors from one pass).
    const std::vector<float> grid = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.4f};
    std::vector<double> ppls(grid.size()), reps(grid.size());
    if (verbose) std::println("  {:>6}   {:>10}   {:>10}", "temp", "gen_ppl", "4gram-rep");
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const GenStats g = gen_self_stats(data, val_start, grid[i], AUTOTEMP_TOPK, N_SEEDS, GEN_LEN, seed);
        ppls[i] = g.ppl; reps[i] = g.rep4;
        if (verbose) {
            std::println("  {:>6.2f}   {:>10.3f}   {:>9.1f}%{}{}", grid[i], g.ppl, 100.0 * g.rep4,
                         g.ppl >= target_ppl ? "  ppl>=tgt" : "", g.rep4 <= target_rep ? "  rep<=tgt" : "");
            std::fflush(stdout);
        }
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
    std::println("  --- sample @ temp {:.2f} ---\n  {}", rec, preview_at("the ", 80, rec, AUTOTEMP_TOPK, prng));
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
    const mp::Dims d{ D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB };
    auto mib = [](unsigned long long b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    const int vram = GPU_VRAM_MB;
    std::println("memory plan: d{} L{} H{} ff{} seq{} v{} | params {:.1f}M | VRAM {} MiB (+{} shared)",
                 D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
                 mp::param_floats(d) / 1e6, vram, GPU_SHARED_MEM_MB);
    std::println("precision: GEMM {} | activations {} | master FP32  (BF16_OK={}; FFN scratch BF16-stored, parity gated by direction)",
                 GEMM_DTYPE == Dtype::BF16 ? "BF16" : "F32", ACT_DTYPE == Dtype::BF16 ? "BF16" : "F32", BF16_OK);

    const double persist = mib(mp::persistent_bytes(d));
    std::println("");
    std::println("persistent (resident, batch-independent): {:.0f} MiB", persist);
    std::println("  params + grad + m + vel + decay (5x) + fused QKV weights");
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
    const std::span<const int> data = load_corpus_tokens(tok, od_buf, "report");
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
    const std::span<const int> train_span = data.first(val_start);
    const std::span<const int> val_span   = data.subspan(val_start);

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

    if (model_in && *model_in) {   // training provenance from the meta.txt next to the model
        sub0::registry::ModelMeta m;
        if (sub0::registry::read_meta(std::filesystem::path(model_in).parent_path(), m) && m.steps > 0)
            emit("training:     steps {} | epochs {:.2f} | tokens_seen {} | seed {}",
                         m.steps, m.epochs, m.tokens_seen, m.seed);
    }

    // Loss metrics need a loaded model. Skip gracefully if it cannot load (e.g. a model trained
    // under a different config/scheme -- the header guard refuses it), keeping structural guidance.
    bool   have_loss = false, overfit = false, underfit_quality = false;
    double rel_gap = 0.0;
    bool   model_loaded = false;
    if (model_in && *model_in) {
        if (sub0::load_model(model_in)) {
            model_loaded = true;
            const double train_nelbo = evaluate(train_span, 0);
            const double val_nelbo   = evaluate(val_span, 0);
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
            emit("  train/val gap {:.4f}  ({:.1f}%)  -> {}", gap, 100.0 * rel_gap,
                         overfit ? "OVERFITTING (model too large / too little data)"
                                 : rel_gap < 0.03 ? "not overfitting (capacity / optimization bound)"
                                                  : "healthy generalization gap");
            if (bytes_per_tok > 0)
                emit("  bits/byte   {:.3f}  (corpus ~{:.2f} bytes/token)  -> {}", bpb, bytes_per_tok,
                             bpb > 1.2 ? "well above ~1.0: real headroom"
                                       : bpb > 0.9 ? "near a good small-model range" : "strong");
        } else {
            emit("note: '{}' did not load into this build (different config/scheme); "
                         "showing structural guidance only.", model_in);
        }
    } else {
        emit("note: no model given -- showing structural (corpus-fit) guidance only.");
    }

    // Grounding samples: actual generations at the gen defaults so a reader (and the saved report)
    // can judge quality directly, not just by the loss numbers. Two temperatures bracket the
    // determinism/diversity trade-off; the full SEQ_LEN length exercises the trained context window.
    if (model_loaded) {
        sub0::load_tokenizer(sub0::default_tokenizer());   // encode/detokenize for the sample print
        std::mt19937 rng(1234);                            // fixed seed -> reproducible report samples
        emit("");
        emit("samples ({}-token context, prompt \"the \"):", SEQ_LEN);
        emit("  [temp 0.4]  {}", preview_at("the ", SEQ_LEN, 0.4f, 20, rng));
        emit("  [temp 0.7]  {}", preview_at("the ", SEQ_LEN, 0.7f, 20, rng));
    }

    const double tok_per_param = params > 0 ? static_cast<double>(train_tokens) / params : 0.0;
    emit("");
    emit("corpus fit (Chinchilla compute-optimal ~20 tokens/param):");
    emit("  train tokens  {:.2f}M", train_tokens / 1e6);
    emit("  tokens/param  {:.1f}  -> {}", tok_per_param,
                 tok_per_param > 40 ? "data-rich: the model is UNDERSIZED for this corpus"
                 : tok_per_param < 5 ? "data-limited: model may be oversized / undertrained"
                                     : "near compute-optimal");

    // Grow capacity unless we are actually overfitting.
    const bool grow = !overfit && (tok_per_param > 40 || underfit_quality);

    int v_d = 0, v_l = 0, v_h = 0, v_seq = 0, v_vocab = 0;
    std::string r_d, r_l, r_h, r_seq, r_vocab;
    if (head_dim < 32)       { v_h = 3; r_h = std::format("head_dim {} < 32: raise d_model or lower n_heads (target 64)", head_dim); }
    else if (head_dim < 48)  { v_h = 2; r_h = std::format("head_dim {} below the usual 64; prefer raising d_model", head_dim); }
    else if (head_dim < 64)  { v_h = 1; r_h = std::format("head_dim {} just below 64", head_dim); }
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
    //   * data-rich (tokens/param > 40): Chinchilla says size UP to ~train_tokens/20 params;
    //   * quality-bound (poor bits/byte at sane tokens/param): spend more capacity than now.
    // Use the LARGER of the two so a "grow" never suggests fewer params than the current model
    // (the old train_tokens/20-only target could undercut the current size and contradict the
    // "MUST widen" verdict). Search head_dim-64 widths STARTING ABOVE the current d_model, and keep
    // depth >= current (width is the primary lever; don't cut layers just to hit an aspect target).
    if (grow && train_tokens > 0) {
        const long long chinchilla = train_tokens / 20;
        const long long target_p   = std::max<long long>(chinchilla, static_cast<long long>(params * 1.8));
        const int       C_start    = ((D_MODEL / 64) + 1) * 64;                     // next head_dim-64 width up
        long long best_p = 0; int best_C = C_start, best_L = N_LAYERS, best_H = C_start / 64;
        for (int C = C_start; C <= 1024; C += 64) {
            const int H = C / 64;                                                   // head_dim = 64
            const int L = std::clamp(std::max(N_LAYERS, static_cast<int>(std::lround(C / 40.0))), 2, 24);
            const long long p = static_cast<long long>(
                sub0::memplan::param_floats({C, L, H, 4 * C, SEQ_LEN, VOCAB}));
            if (best_p == 0 || std::llabs(p - target_p) < std::llabs(best_p - target_p)) {
                best_p = p; best_C = C; best_L = L; best_H = H;
            }
        }
        emit("");
        emit("suggested next size (target ~{:.1f}M params; max of Chinchilla {:.1f}M and 1.8x current):",
                     target_p / 1e6, chinchilla / 1e6);
        emit("  cmake --preset native -DSUB0_D_MODEL={} -DSUB0_N_LAYERS={} -DSUB0_N_HEADS={}",
                     best_C, best_L, best_H);
        emit("  -> d{} L{} H{} (head_dim 64, aspect {:.0f}) ~= {:.2f}M params; then `sub0llm train`",
                     best_C, best_L, best_H, static_cast<double>(best_C) / best_L, best_p / 1e6);
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
// `sub0llm models` discovers every trained model (scans the meta.txt under the models root --
// the registry is the set of those files, so it never drifts out of sync) and flags which load
// into THIS build (matching architecture dims). `--prune` reclaims the incompatible ones, whose
// checkpoints this engine could never load anyway. Destructive, so it is opt-in via the flag.
extern "C" SUB0_API int sub0_models_stage(int prune, int verbose) {
    (void)verbose;
    namespace reg = sub0::registry;
    std::vector<reg::ModelMeta> models = reg::scan(SUB0_MODELS_ROOT);
    std::println("models root: {}  ({} model{})", SUB0_MODELS_ROOT, models.size(), models.size() == 1 ? "" : "s");
    std::println("this build:  d{} l{} h{} sq{} v{}{} @ {}",
                 D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB, USE_TERNARY ? "t" : "", SUB0_GIT_SHA);
    if (models.empty()) { std::println("(none yet -- `sub0llm train` creates one)"); return 0; }

    auto loadable = [&](const reg::ModelMeta& m) {
        return reg::compatible(m, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                               static_cast<int>(USE_TERNARY), static_cast<int>(POS_ENCODING));
    };
    std::sort(models.begin(), models.end(),
              [](const reg::ModelMeta& a, const reg::ModelMeta& b) { return a.dir.filename() < b.dir.filename(); });

    std::println("{:<3} {:<50} {:>9}  {:<10} {}", "use", "model", "val_nelbo", "status", "corpus");
    for (const reg::ModelMeta& m : models)
        std::println("{:<3} {:<50} {:>9}  {:<10} {}", loadable(m) ? " * " : " x ",
                     m.dir.filename().string(),
                     m.best_val_nelbo >= 0 ? std::format("{:.4f}", m.best_val_nelbo) : "-",
                     m.status, m.corpus);
    std::println("(* loadable by this build | x incompatible architecture)");

    if (prune) {
        int removed = 0;
        for (const reg::ModelMeta& m : models) {
            if (loadable(m)) continue;
            std::error_code ec;
            const auto n = std::filesystem::remove_all(m.dir, ec);
            if (!ec) { std::println("pruned {} ({} entries)", m.dir.filename().string(), n); ++removed; }
            else     std::println(stderr, "warning: could not prune {}", m.dir.string());
        }
        std::println("pruned {} incompatible model(s)", removed);
    }
    return 0;
}

