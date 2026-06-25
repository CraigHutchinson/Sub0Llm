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
#include "sub0/tune.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <expected>
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

// --- corpus.tok loading -----------------------------------------------------
enum class TokError { Missing, BadMagic, Truncated };

// Contents of corpus.tok: the token stream plus the vocabulary size it was produced
// for, so the caller can verify it matches the engine's compiled-in VOCAB.
struct TokData {
    int vocab = 0;                 // vocabulary the file was tokenized against
    std::vector<int> data;         // flat token-id stream
};

// Read the pre-tokenized corpus (corpus.tok, "S0TK") into a flat id array. Returning
// std::expected makes the data unreachable until the caller has handled the error.
std::expected<TokData, TokError> load_tokens(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return std::unexpected(TokError::Missing);
    auto rd = [&] { std::uint32_t v{}; is.read(reinterpret_cast<char*>(&v), 4); return v; };
    if (rd() != 0x4B543053u) return std::unexpected(TokError::BadMagic);  // "S0TK"
    TokData out;
    out.vocab = static_cast<int>(rd());
    const std::uint32_t ntok = rd();
    out.data.resize(ntok);
    is.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(ntok) * sizeof(int));
    if (!is) return std::unexpected(TokError::Truncated);
    return out;
}

// --- Validation NELBO -------------------------------------------------------
// Mean cross-entropy per token over a fixed, evenly-spaced set of windows in the
// held-out tail. Fixed windows make the metric comparable across evals (so the
// plateau sign test sees signal, not resampling noise) and bound the cost.
double evaluate(const std::vector<int>& data, std::size_t val_start) {
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
    return true;
}

// A short mid-training sample. Uses the SAME temperature/top-k sampler as `gen` so
// the preview reflects real generation quality -- the old greedy+noise hack made a
// coherent model look like word-salad.
std::string preview(const std::string& prompt, int n, std::mt19937& rng) {
    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
        const int last = logits->rows - 1;
        ctx.push_back(sub0::sample_token(logits->data.data() + (size_t)last * VOCAB, 0.7f, 20, rng));
    }
    sub0::graph_reset();
    return sub0::detokenize(ctx);
}

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

    std::expected<TokData, TokError> tok = load_tokens(tok_path);
    if (!tok) {
        switch (tok.error()) {
            case TokError::Missing:
                std::println(stderr, "train: cannot open token file '{}'", tok_path); break;
            case TokError::BadMagic:
                std::println(stderr, "train: '{}' is not a corpus.tok file (bad magic)", tok_path); break;
            case TokError::Truncated:
                std::println(stderr, "train: '{}' is truncated (token count exceeds file size)", tok_path); break;
        }
        return 1;
    }

    // The .tok stream carries token ids in [0, VOCAB). The configurator bakes the
    // matching VOCAB into the engine, so within a build they always agree -- but a
    // stale, hand-built, or foreign .tok passed on the command line would index the
    // embedding table out of bounds. Reject it up front rather than corrupt memory.
    if (tok->vocab != VOCAB) {
        std::println(stderr,
                     "train: '{}' was tokenized for vocab {} but this engine was built for VOCAB {}.\n"
                     "       Reconfigure/rebuild against this corpus, or pass a matching .tok.",
                     tok_path, tok->vocab, VOCAB);
        return 1;
    }

    std::vector<int>& data = tok->data;
    // Need a validation tail plus a training region each large enough for a window.
    const std::size_t min_tokens = 2 * (static_cast<std::size_t>(SEQ_LEN) + 2);
    if (data.size() < min_tokens) {
        std::println(stderr, "train: '{}' has only {} tokens, too few for seq_len {} (need >= {})",
                     tok_path, data.size(), SEQ_LEN, min_tokens);
        return 1;
    }

    // Defense in depth: even with a matching vocab field, a corrupt file could hold
    // an out-of-range id. One linear scan is cheap next to training and turns a
    // silent OOB gather into a clear diagnostic.
    for (size_t i = 0; i < data.size(); ++i)
        if (data[i] < 0 || data[i] >= VOCAB) {
            std::println(stderr, "train: '{}' token {} = {} is out of range [0,{})",
                         tok_path, i, data[i], VOCAB);
            return 1;
        }

    // Hold out the tail for validation; train only on the head so NELBO is honest.
    const std::size_t val_tokens = std::max<std::size_t>(
        static_cast<std::size_t>(SEQ_LEN) + 2,
        static_cast<std::size_t>(VAL_FRACTION * static_cast<double>(data.size())));
    const std::size_t val_start  = data.size() - val_tokens;
    const std::size_t train_tokens = val_start;  // [0, val_start) is the training region

    sub0::load_tokenizer(sub0::default_tokenizer());  // for previews
    sub0::build_model();

    std::mt19937 rng(seed);
    RunState rs;

    // Resume if a checkpoint for this output exists (overwrites the fresh model and
    // restores batch/lr/seed, so the schedule below is computed from the resumed
    // batch, not whatever the command line happened to pass).
    const std::string ckpt_path = std::string(model_out) + ".ckpt";
    long adam_t = 0;
    const bool resumed = load_checkpoint(ckpt_path, rng, rs, batch, lr, seed, adam_t);

    sub0::AdamW opt(lr);
    if (resumed) opt.set_step_count(adam_t);

    // Corpus-relative schedule (max_steps is a per-invocation budget, never restored).
    const long tokens_per_step = static_cast<long>(batch) * SEQ_LEN;
    const long epoch_steps  = std::max<long>(1, (static_cast<long>(train_tokens) + tokens_per_step - 1) / tokens_per_step);
    const long warmup_steps = std::max<long>(1, std::lround(EVAL_WARMUP_EPOCHS  * epoch_steps));
    const long eval_every   = std::max<long>(1, std::lround(EVAL_INTERVAL_EPOCHS * epoch_steps));
    const long max_steps = (steps > 0) ? steps : static_cast<long>(MAX_EPOCHS_BACKSTOP) * epoch_steps;

    std::print("corpus: {} ({} tokens; {} train / {} val) | ", tok_path, data.size(), train_tokens, val_tokens);
    sub0::print_config();
    std::println("schedule: {} steps/epoch | warmup {} | eval every {} | max {} steps ({} epochs){}",
                 epoch_steps, warmup_steps, eval_every, max_steps,
                 (max_steps + epoch_steps - 1) / epoch_steps,
                 resumed ? std::format(" | RESUMED at step {}", rs.step) : std::string{});
    std::fflush(stdout);

    std::uniform_int_distribution<size_t> startd(0, val_start - SEQ_LEN - 2);
    std::vector<size_t> starts(batch);

    using clock = std::chrono::steady_clock;
    auto win_t0 = clock::now();
    long win_steps0 = rs.step;
    double run_loss = 0.0; int run_n = 0;
    bool stop = false;

    for (long step = rs.step + 1; step <= max_steps && !stop; ++step) {
        // Draw the window starts on the main thread (keeps the RNG stream, hence
        // resume, deterministic), then run the batch data-parallel across threads.
        for (int b = 0; b < batch; ++b) starts[b] = startd(rng);
        const float step_loss = sub0::train_batch(data.data(), starts.data(), batch, SEQ_LEN);
        opt.step();
        run_loss += step_loss; ++run_n;
        rs.step = step;

        if (step % eval_every == 0 || step == max_steps) {
            // Throughput over the interval just completed (training only).
            const double secs = std::chrono::duration<double>(clock::now() - win_t0).count();
            const double wps   = secs > 0 ? static_cast<double>((step - win_steps0) * batch) / secs : 0.0;
            const double frac_epoch = static_cast<double>(step) / epoch_steps;

            // Validation NELBO only once enough of the corpus has been seen.
            std::string eval_str = "(warmup)";
            if (step >= warmup_steps) {
                const double nelbo = evaluate(data, val_start);
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
            sub0::save_model(model_out);

            run_loss = 0.0; run_n = 0;
            win_t0 = clock::now(); win_steps0 = step;
        }
    }

    sub0::save_model(model_out);
    save_checkpoint(ckpt_path, opt.step_count(), rng, rs, batch, lr, seed);
    std::println("  --- sample ---\n  {}", preview("the ", 120, rng));
    std::println("{} at step {} (best val_nelbo {:.4f}) -> {}",
                 stop ? "plateaued" : "reached max steps", rs.step, rs.best_loss, model_out);
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

    std::expected<TokData, TokError> tok = load_tokens(sub0::default_corpus_tok());
    if (!tok || tok->vocab != VOCAB || tok->data.size() <= static_cast<size_t>(SEQ_LEN) + 2) {
        std::println(stderr, "bench: cannot load a usable corpus.tok");
        return 1;
    }
    const std::vector<int>& data = tok->data;
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
double measure_dp_throughput(const std::vector<int>& data, std::mt19937& rng,
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
    std::expected<TokData, TokError> tok = load_tokens(sub0::default_corpus_tok());
    if (!tok || tok->vocab != VOCAB || tok->data.size() <= static_cast<size_t>(SEQ_LEN) + 2) {
        std::println(stderr, "tune: cannot load a usable corpus.tok");
        return 1;
    }
    const std::vector<int>& data = tok->data;
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
    return 0;
}

