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
#include <vector>

namespace {

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

// A short greedy-ish sample, used for mid-training previews.
std::string preview(const std::string& prompt, int n, std::mt19937& rng) {
    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
        const int last = logits->rows - 1;
        int best = 0; float bv = -1e30f;
        for (int j = 0; j < VOCAB; ++j) {
            float v = logits->data[(size_t)last * VOCAB + j];
            if (v > bv) { bv = v; best = j; }
        }
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        if (ud(rng) < 0.3f) best = std::min(best + 1, VOCAB - 1);
        ctx.push_back(best);
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

    using clock = std::chrono::steady_clock;
    auto win_t0 = clock::now();
    long win_steps0 = rs.step;
    double run_loss = 0.0; int run_n = 0;
    bool stop = false;

    for (long step = rs.step + 1; step <= max_steps && !stop; ++step) {
        opt.zero_grad();
        float step_loss = 0.f;
        for (int b = 0; b < batch; ++b) {
            size_t s = startd(rng);
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(data.data() + s, SEQ_LEN);
            sub0::Node* loss   = sub0::cross_entropy(logits, data.data() + s + 1);
            step_loss += loss->data[0] / batch;
            sub0::backward(loss, 1.f / batch);
        }
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
