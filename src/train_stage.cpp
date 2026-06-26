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
                                     float lr, long t, double* out_loss);
extern "C" void sub0_cuda_set_tf32(int on);
extern "C" void sub0_cuda_set_attn_bwd(int per_query);
extern "C" int  sub0_cuda_time_train_step(int batch, int T, int iters, double* out_ms);
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
constexpr int    EVAL_WINDOWS_MAX    = 128;   // bounded cost per eval
constexpr int    MAX_EPOCHS_BACKSTOP = 30;    // ceiling if no plateau is detected
constexpr int    PLATEAU_WINDOW      = 6;     // deltas inspected by the sign test
constexpr int    PLATEAU_MIN_IMPROVE = 4;     // >= this many decreasing -> keep going

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

// Plateau as a sign test: among the last PLATEAU_WINDOW eval-to-eval deltas, count
// how many decreased. Still trending down if at least PLATEAU_MIN_IMPROVE did;
// otherwise the series is bouncing around a floor -> stop. Comparing only signs
// (not magnitudes) avoids a noise-sensitive improvement threshold.
bool plateaued(const std::vector<double>& evals) {
    if (static_cast<int>(evals.size()) < PLATEAU_WINDOW + 1) return false;
    const std::size_t n = evals.size();
    int improving = 0;
    for (int k = 0; k < PLATEAU_WINDOW; ++k)
        if (evals[n - 1 - k] < evals[n - 2 - k]) ++improving;
    return improving < PLATEAU_MIN_IMPROVE;
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
        constexpr int kCap = (DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD > 0)
                                 ? DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD : 8;
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

    // One resident device step over the sampled window starts; returns the mean loss. Builds the
    // [batch*SEQ_LEN] id/target windows (x = data[start+s], y = data[start+s+1]) like the CPU path.
    float step([[maybe_unused]] std::span<const int> data, [[maybe_unused]] const std::size_t* starts,
               [[maybe_unused]] int batch, [[maybe_unused]] float lr) {
#if defined(SUB0_BUILD_CUDA)
        for (int b = 0; b < batch; ++b) {
            const int* w = data.data() + starts[b];
            for (int s = 0; s < SEQ_LEN; ++s) {
                ids[static_cast<std::size_t>(b) * SEQ_LEN + s]     = w[s];
                targets[static_cast<std::size_t>(b) * SEQ_LEN + s] = w[s + 1];
            }
        }
        double loss = 0.0;
        sub0_cuda_train_step(ids.data(), targets.data(), batch, SEQ_LEN, lr, ++t, &loss);
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
    if (model_out && *model_out) {
        model_path = model_out;
    } else {
        meta_dir = sub0::registry::model_dir(SUB0_MODELS_ROOT, sub0::default_corpus(),
                                             D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB,
                                             static_cast<int>(USE_TERNARY), SUB0_GIT_SHA);
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
        m.git_sha = SUB0_GIT_SHA; m.created = created;
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
    const long epoch_steps  = std::max<long>(1, (static_cast<long>(est_train_tokens) + tokens_per_step - 1) / tokens_per_step);
    const long warmup_steps = std::max<long>(1, std::lround(EVAL_WARMUP_EPOCHS  * epoch_steps));
    const long eval_every   = std::max<long>(1, std::lround(EVAL_INTERVAL_EPOCHS * epoch_steps));
    const long max_steps = (steps > 0) ? steps : static_cast<long>(MAX_EPOCHS_BACKSTOP) * epoch_steps;

    std::print("corpus: {} | ", src_desc);
    sub0::print_config();
    std::println("training backend: {}", gpu_train ? "GPU (resident device step: fwd+bwd+AdamW)"
                                                    : "CPU (data-parallel minibatch)");
    std::println("schedule: {} steps/epoch | warmup {} | eval every {} | max {} steps ({} epochs){}{}",
                 epoch_steps, warmup_steps, eval_every, max_steps,
                 (max_steps + epoch_steps - 1) / epoch_steps,
                 on_demand ? " | on-demand" : "",
                 resumed ? std::format(" | RESUMED at step {}", rs.step) : std::string{});
    std::fflush(stdout);

    std::uniform_int_distribution<size_t> startd(0, train_span.size() - SEQ_LEN - 2);
    std::vector<size_t> starts(batch);
    long steps_since_refresh = 0;

    using clock = std::chrono::steady_clock;
    auto win_t0 = clock::now();
    long win_steps0 = rs.step;
    double run_loss = 0.0; int run_n = 0;
    bool stop = false;

    for (long step = rs.step + 1; step <= max_steps && !stop; ++step) {
        // On-demand: refill the rotating shuffle buffer once per eval interval from new
        // random regions, so the run traverses the whole corpus over time. buf_rng is a
        // separate stream, so this never perturbs the resume-critical `rng`. The refill may
        // reallocate train_buf, so re-bind the span and the sampler to the new storage.
        if (on_demand && steps_since_refresh >= eval_every) {
            buf_rng.seed(static_cast<std::uint32_t>(seed) ^ (0x9E3779B9u * static_cast<std::uint32_t>(++refresh_n)));
            text.fill_random(train_byte_lo, train_byte_hi, OD_TRAIN_BUF_TOK, buf_rng, train_buf);
            train_span = train_buf;
            startd = std::uniform_int_distribution<size_t>(0, train_span.size() - SEQ_LEN - 2);
            steps_since_refresh = 0;
        }
        ++steps_since_refresh;

        // Draw the window starts on the main thread (keeps the RNG stream, hence
        // resume, deterministic), then run the batch -- on the GPU when enabled, else
        // data-parallel across CPU threads.
        for (int b = 0; b < batch; ++b) starts[b] = startd(rng);
        float step_loss;
        if (gpu_train) {
            step_loss = gpu.step(train_span, starts.data(), batch, lr);
            opt.set_step_count(gpu.t);          // keep the checkpoint's step counter in lockstep
        } else {
            step_loss = sub0::train_batch(train_span.data(), starts.data(), batch, SEQ_LEN);
            opt.step();
        }
        run_loss += step_loss; ++run_n;
        rs.step = step;

        if (step % eval_every == 0 || step == max_steps) {
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
        }
    }

    gpu.sync_to_host();    // ensure the host arenas hold the final device weights before saving
    sub0::save_model(model_path.c_str());
    save_checkpoint(ckpt_path, opt.step_count(), rng, rs, batch, lr, seed);
    write_meta(stop ? "plateaued" : "trained");
    std::println("  --- sample ---\n  {}", preview("the ", 120, rng));
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

// Wall-clock throughput (window/s) of `steps` consecutive train_batch calls at a given
// thread count and per-thread window count. Throughput is the metric that matters long
// term, and it is deliberately NOT cycle-normalised: a longer `steps` lets the cores
// heat up and thermally throttle, so the number reflects the SUSTAINED rate a real
// training run would see. That is why the tuner raises `steps` as it narrows down --
// the coarse sweep uses a short burst for speed, the confirmation a long throttled run.
// Random run-to-run noise is handled upstream by the search's median-of-samples
// confirmation, so a single honest measurement is returned here (no optimistic best-of).
double measure_dp_throughput(std::span<const int> data, std::mt19937& rng,
                             int threads, int windows_per_thread, int steps) {
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#endif
    const int active = threads > 0 ? threads : 1;
    const int B = active * std::max(1, windows_per_thread);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);
    std::vector<std::size_t> starts(static_cast<size_t>(B));

    for (int b = 0; b < B; ++b) starts[static_cast<size_t>(b)] = startd(rng);  // warm caches
    sub0::train_batch(data.data(), starts.data(), B, SEQ_LEN);

    const int n = std::max(1, steps);
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n; ++s) {
        for (int b = 0; b < B; ++b) starts[static_cast<size_t>(b)] = startd(rng);
        sub0::train_batch(data.data(), starts.data(), B, SEQ_LEN);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return secs > 0 ? static_cast<double>(n) * B / secs : 0.0;
}

}  // namespace

extern "C" SUB0_API int sub0_tune_stage(int max_threads, int verbose) {
    sub0::TokMap tok(sub0::default_corpus_tok());
    std::vector<int> od_buf;
    const std::span<const int> data = load_corpus_tokens(tok, od_buf, "tune");
    if (data.empty()) return 1;
    sub0::build_model();
    std::mt19937 rng(123);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int cap = max_threads > 0 ? std::min<int>(max_threads, static_cast<int>(hw))
                                     : static_cast<int>(hw);

    // The knob space. Extensible: append a Knob and read it by index in the objective.
    std::vector<double> thread_vals;
    for (int t = 1; t <= cap; ++t) thread_vals.push_back(static_cast<double>(t));
    sub0::tune::Space space = {
        {"threads",        thread_vals},
        {"windows/thread", {1, 2, 4, 8, 16}},
    };

    // Objective: maximize measured window/s. tune::maximize never sees the engine.
    // Reports each configuration live (the search can run many evals; buffering the
    // whole trace to the end would look like a long hang), tracking the running best.
    // `steps` grows as the search narrows (driven by on_phase below) so the precise,
    // narrowing-down measurements run long enough to thermally throttle -- the regime a
    // sustained training run actually operates in.
    int steps = 8;
    double running_best = 0.0;
    sub0::tune::Objective objective = [&](const sub0::tune::Assignment& a) {
        const int threads = static_cast<int>(std::lround(a[0]));
        const int wpt     = static_cast<int>(std::lround(a[1]));
        const double wps  = measure_dp_throughput(data, rng, threads, wpt, steps);
        if (verbose) {
            const bool best = wps > running_best;
            std::println("  threads={:>2}  windows/thread={:>2}  ->  {:>6.0f} window/s{}",
                         threads, wpt, wps, best ? "   <- best" : "");
            std::fflush(stdout);   // stream progress as it happens, don't buffer to the end
        }
        running_best = std::max(running_best, wps);
        return wps;
    };

    std::println("--- auto-tune throughput (knobs: threads 1..{}, windows/thread {{1,2,4,8,16}}) ---", cap);
    report_threading();
    std::fflush(stdout);

    sub0::tune::Options opt;
    // After the hot coarse sweep, pause briefly before each refinement pass and
    // confirmation round so the cores shed heat and the narrowing-down readings are
    // steadier (the search core stays pure -- the pause lives here).
    opt.settle = [] { std::this_thread::sleep_for(std::chrono::milliseconds(150)); };
    // Lengthen each measurement as the search narrows: a quick burst while exploring
    // broadly, then progressively longer throttled runs while pinning down the winner.
    // The phase header also makes it clear in the live output which stage we are in.
    opt.on_phase = [&](sub0::tune::Phase p) {
        switch (p) {
            case sub0::tune::Phase::Explore:
                steps = 8;
                std::println("[explore] wide sweep, short runs");
                break;
            case sub0::tune::Phase::Refine:
                steps = 24;
                std::println("[refine]  narrowing in, longer runs");
                break;
            case sub0::tune::Phase::Confirm:
                steps = 64;
                std::println("[confirm] re-measuring finalists, longest runs (dropping clear losers)");
                break;
        }
        std::fflush(stdout);
    };
    sub0::tune::Result r = sub0::tune::maximize(space, objective, opt);

    const int best_threads = static_cast<int>(std::lround(r.best[0]));
    const int best_wpt     = static_cast<int>(std::lround(r.best[1]));
    const double single = measure_dp_throughput(data, rng, 1, best_wpt, 64);  // long run, matched to confirm
    std::println("");
    std::println("evaluated {} measurements; winner confirmed over {} samples", r.evaluations, r.best_samples);
    std::println("best: threads={}  windows/thread={}  ->  {:.0f} window/s  ({:.2f}x vs 1 thread)",
                 best_threads, best_wpt, r.best_score, single > 0 ? r.best_score / single : 0.0);
    std::println("apply with:  sub0llm bench --threads {} --windows-per-thread {}", best_threads, best_wpt);

    // Persist the winning configuration so the next build bakes it into the config
    // header as DEFAULT_THREADS / DEFAULT_WINDOWS_PER_THREAD (see sub0-configure).
    if (DEFAULT_TUNE_CACHE[0] == '\0') {
        std::println("(no tune cache configured; tuned defaults not persisted)");
    } else if (std::ofstream cache(DEFAULT_TUNE_CACHE, std::ios::trunc); cache) {
        cache << "threads=" << best_threads << "\n"
              << "windows_per_thread=" << best_wpt << "\n";
        std::println("persisted tuned defaults to {}", DEFAULT_TUNE_CACHE);
        std::println("rebuild to bake them in:  cmake --build --preset native");
    } else {
        std::println(stderr, "warning: could not write tune cache '{}'", DEFAULT_TUNE_CACHE);
    }

    // GPU device-step knob tuning (Phase 2e+): when the CUDA backend is built and a device is
    // present, also tune the GPU perf knobs (TF32 GEMM math, attention-backward strategy) by timing
    // the resident training step. Under SUB0_TUNING the knobs are runtime-mutable so we sweep them;
    // otherwise they are baked, so we report the baked config and point at the tuning build.
#if defined(SUB0_BUILD_CUDA)
    if (HAS_CUDA && sub0_cuda_init() == 0) {
        const int gbatch = DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD;   // the default training batch
        std::println("");
        std::println("--- GPU device-step tuning (knobs: TF32, attn-backward; batch {}) ---", gbatch);
        report_threading();
#if defined(SUB0_TUNING)
        double best_tok = 0.0; int best_tf32 = 0, best_attn = 0;
        for (int tf32 = 0; tf32 <= 1; ++tf32)
            for (int attn = 0; attn <= 1; ++attn) {
                sub0_cuda_set_tf32(tf32);
                sub0_cuda_set_attn_bwd(attn);
                double ms = 0.0;
                if (sub0_cuda_time_train_step(gbatch, SEQ_LEN, 30, &ms) != 0 || ms <= 0.0) continue;
                const double tok = static_cast<double>(gbatch) * SEQ_LEN * 1000.0 / ms;
                const bool best = tok > best_tok;
                std::println("  tf32={}  attn_bwd={:<5}  ->  {:>8.0f} tok/s  ({:.2f} ms/step){}",
                             tf32, attn ? "query" : "head", tok, ms, best ? "   <- best" : "");
                std::fflush(stdout);
                if (best) { best_tok = tok; best_tf32 = tf32; best_attn = attn; }
            }
        sub0_cuda_set_tf32(best_tf32); sub0_cuda_set_attn_bwd(best_attn);   // leave the winner applied
        std::println("best GPU: tf32={}  attn_bwd={}  ->  {:.0f} tok/s", best_tf32,
                     best_attn ? "query" : "head", best_tok);
        std::println("bake with:  cmake --preset native -DSUB0_CUDA_TF32={} ...  (and set the attn-backward",
                     best_tf32 ? "ON" : "OFF");
        std::println("            default to {} in backend_cuda.cu), then rebuild without -DSUB0_TUNING.",
                     best_attn ? "per-query" : "per-head");
#else
        double ms = 0.0;
        const double tok = (sub0_cuda_time_train_step(gbatch, SEQ_LEN, 30, &ms) == 0 && ms > 0.0)
                               ? static_cast<double>(gbatch) * SEQ_LEN * 1000.0 / ms : 0.0;
        std::println("  baked config  ->  {:.0f} tok/s  ({:.2f} ms/step)", tok, ms);
        std::println("  GPU knobs (TF32, attn-backward) are baked in this build; rebuild with");
        std::println("  -DSUB0_TUNING=ON to sweep and tune them.");
#endif
        sub0_cuda_shutdown();
    }
#endif
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
        return reg::compatible(m, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, VOCAB, static_cast<int>(USE_TERNARY));
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

