// sub0/eval.hpp -- held-out NELBO evaluation: window selection + the CPU/device dispatch.
//
// Extracted from train_stage.cpp, where three call sites (evaluate, evaluate_context_curve,
// mean_entropy) had each open-coded the SAME window arithmetic. That duplication is not cosmetic:
// the whole point of a fixed, evenly-spaced window set is that every eval scores the IDENTICAL text,
// so the plateau sign test and any A/B comparison see model change rather than resampling noise. Three
// copies of that arithmetic is three chances for one of them to drift and silently compare two models
// on different windows.
//
// It also puts the eval where a test can reach it. Living inside a 4000-line tool TU, `evaluate()` was
// unreachable from the test suite, so the CPU and device paths could only ever be compared by running
// the real tool against a real model -- see tests/eval_seam_tests.cpp, which now drives both routes
// over the same windows and asserts they agree.

#pragma once

#include "sub0/core.hpp"
#include "sub0/tokmap.hpp"
#include "sub0/device_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace sub0::eval {

// Bounded cost per eval: the window count is capped here regardless of how much held-out text exists.
inline constexpr int WINDOWS_MAX = 128;

// The fixed, evenly-spaced set of window start offsets an evaluation scores. Deliberately a plain
// value (no data reference held): `plan()` resolves it once against a corpus and every consumer then
// asks it for start offsets, so two evaluations built from the same plan cannot diverge.
struct WindowSet {
    std::size_t first  = 0;   // offset of window 0
    std::size_t stride = 1;   // spacing between consecutive windows
    std::size_t last   = 0;   // largest legal start (a full SEQ_LEN+1 window must fit)
    int         count  = 0;   // number of windows (0 => nothing evaluable)

    [[nodiscard]] std::size_t start_of(int w) const {
        const std::size_t s = first + static_cast<std::size_t>(w) * stride;
        return s > last ? last : s;       // clamp, matching the original loops' `if (s > last) s = last`
    }
    [[nodiscard]] bool empty() const { return count <= 0; }
};

// Resolve the window set for `data` starting at `val_start`. `count == 0` means the span is too short
// to hold even one full window, which every caller must treat as "no measurement" (NaN), never as 0.
[[nodiscard]] inline WindowSet plan(TokView data, std::size_t val_start, int max_windows = WINDOWS_MAX) {
    WindowSet ws;
    // A window needs SEQ_LEN inputs plus one more token to be the last position's target, so
    // SEQ_LEN+1 is the smallest evaluable corpus -- and the `size - SEQ_LEN - 1` below UNDERFLOWS on
    // anything shorter (std::size_t), producing a huge `last` that the val_start bound check then
    // waves through. That underflow predates this header; it is guarded here rather than left for
    // each of the three former copies of this arithmetic to get right independently.
    if (data.size() < static_cast<std::size_t>(SEQ_LEN) + 1) return ws;
    ws.last = data.size() - SEQ_LEN - 1;
    if (val_start > ws.last) return ws;
    const std::size_t span = ws.last - val_start;
    const int avail = static_cast<int>(span / SEQ_LEN) + 1;
    ws.first  = val_start;
    ws.count  = std::min(max_windows, std::max(1, avail));
    ws.stride = (ws.count > 1) ? span / static_cast<std::size_t>(ws.count - 1) : 1;
    return ws;
}

// --- CPU path ---------------------------------------------------------------------------------
// Mean cross-entropy per token over the plan's windows at context width `ctx` (<= SEQ_LEN).
//
// Data-parallel over the windows, the same shape as train_batch's parallel region: forward() /
// graph_reset() / cross_entropy() all operate on a thread_local model arena, so this is a forward-only
// train_batch minus the backward/grad-reduction half. `win` is declared INSIDE the parallel region so
// each thread gets its own buffer -- a shared one would race between copy_to's write and forward's
// read. Static schedule keeps the SET of windows identical to a sequential run; only the evaluation
// and summation ORDER changes, which floating-point addition is not strictly invariant to (the same
// caveat train_batch's own cross-thread grad reduction already carries).
[[nodiscard]] inline double nelbo_cpu(TokView data, const WindowSet& ws, int ctx, int threads) {
    if (ws.empty() || ctx < 2 || ctx > SEQ_LEN) return std::numeric_limits<double>::quiet_NaN();
    const int nw = ws.count;
    double total = 0.0;
    #pragma omp parallel num_threads(threads)
    {
        std::vector<int> win(static_cast<std::size_t>(SEQ_LEN) + 1);
        #pragma omp for reduction(+ : total) schedule(static)
        for (int w = 0; w < nw; ++w) {
            data.copy_to(ws.start_of(w), static_cast<std::size_t>(ctx) + 1, win.data());
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(win.data(), ctx);
            sub0::Node* loss   = sub0::cross_entropy(logits, win.data() + 1);
            total += loss->data[0];
        }
    }
    sub0::graph_reset();
    return total / nw;
}

// --- Device path ------------------------------------------------------------------------------
// How many windows to submit per sub0_dev_forward_loss call. The device forward materializes the full
// [batch*ctx, VOCAB] logits buffer (it runs the graph-captured INFERENCE forward, which does not chunk
// the head the way training's head_ce_chunked does), and that buffer dominates the call's footprint --
// so bound IT, not the window count. Everything else the call touches is O(batch*ctx) ints.
//
// Derived from the shape rather than fixed, per the project's "never a fixed threshold when the scale
// is known" rule: at ctx=512/VOCAB=16.5k this yields ~16 windows, at ctx=64 it yields ~128.
inline constexpr std::size_t DEVICE_LOGITS_BUDGET_BYTES = 512ull << 20;   // 512 MiB
[[nodiscard]] inline int device_batch(int ctx, int nw) {
    if (nw < 1) return 1;                        // std::clamp is UB when lo > hi -- never hand it nw=0
    const std::size_t per_window = static_cast<std::size_t>(ctx) * VOCAB * sizeof(float);
    std::size_t b = per_window ? DEVICE_LOGITS_BUDGET_BYTES / per_window : 1;
    b = std::clamp<std::size_t>(b, 1, static_cast<std::size_t>(nw));
    // Even out the groups so the ragged last call is as close to the others as possible: each distinct
    // (batch, ctx) shape costs a CUDA graph capture, so 43/43/42 (two shapes) beats 50/50/28 (also two,
    // but with a much larger odd tail) and both beat letting the divisor fall where it may.
    const std::size_t groups = (static_cast<std::size_t>(nw) + b - 1) / b;
    b = (static_cast<std::size_t>(nw) + groups - 1) / groups;
    return static_cast<int>(b);
}

// Mean cross-entropy over the plan's windows at width `ctx`, computed on the device. Returns NaN if
// any device call fails, so a caller can fall back to nelbo_cpu rather than report a wrong number.
//
// Windows are submitted in groups; each call returns the mean over ITS group, so the groups are
// recombined weighted by window count -- NOT a plain mean of means, which a ragged final group would
// bias. Requires an active device session with the current params uploaded (see Session).
[[nodiscard]] inline double nelbo_device(TokView data, const WindowSet& ws, int ctx) {
    if (ws.empty() || ctx < 2 || ctx > SEQ_LEN) return std::numeric_limits<double>::quiet_NaN();
    const int nw = ws.count;
    const int b  = device_batch(ctx, nw);
    std::vector<int> ids(static_cast<std::size_t>(b) * ctx);
    std::vector<int> tgt(static_cast<std::size_t>(b) * ctx);
    std::vector<int> win(static_cast<std::size_t>(ctx) + 1);
    double weighted = 0.0;
    for (int w0 = 0; w0 < nw; w0 += b) {
        const int rows = std::min(b, nw - w0);
        for (int i = 0; i < rows; ++i) {
            data.copy_to(ws.start_of(w0 + i), static_cast<std::size_t>(ctx) + 1, win.data());
            int* id_row = ids.data() + static_cast<std::size_t>(i) * ctx;
            int* tg_row = tgt.data() + static_cast<std::size_t>(i) * ctx;
            for (int t = 0; t < ctx; ++t) { id_row[t] = win[t]; tg_row[t] = win[t + 1]; }
        }
        double group = 0.0;
        if (sub0_dev_forward_loss(ids.data(), tgt.data(), rows, ctx, &group, /*lengths=*/nullptr) != 0)
            return std::numeric_limits<double>::quiet_NaN();
        weighted += group * rows;    // each call returns the mean over ITS rows windows
    }
    return weighted / nw;
}

// --- Session ----------------------------------------------------------------------------------
// RAII device bring-up for a BATCH of evaluations (a whole context-length curve, both splits of a
// report), mirroring decode.hpp's DecodeSession and for the same reason: init + param upload + graph
// capture per call would cost more than the evaluation saves.
//
// `use_device` is false -- and every eval transparently stays on the CPU -- when there is no device
// backend, the device has no forward-loss entry, initialization fails, or `allow` is false. That last
// one is how the caller keeps a running trainer's machine to itself: report passes
// `!trainer_active()`, so an eval launched alongside a live training run neither competes for VRAM nor
// perturbs the trainer's throughput numbers.
//
// NOTE: a trainer's OWN periodic evals therefore stay on the CPU too -- the trainer holds that same
// mutex. That is deliberate for now: this path calls fwd_alloc(full=true), which would add the whole
// inference activation scratch (including a [batch*ctx, VOCAB] logits buffer) on top of a training
// allocation already tuned to fill VRAM. Giving the trainer a device eval needs a VRAM budget it can
// prove fits first; it is a follow-up, not a silent side effect of this one.
struct Session {
    bool use_device = false;
    // Why the device was NOT used, for a caller that wants to say so. Never null. A silent fallback
    // is the failure mode to avoid here: the CPU path gives the same numbers but can be two orders of
    // magnitude slower, so a report that quietly took it looks hung rather than degraded.
    const char* declined = "";

    explicit Session(bool allow = true) {
        // Gated on the CAPABILITY, not on HAS_CUDA: the seam is vendor-neutral by design, and the
        // no-device build's caps already report supports_eval = 0, so this short-circuits there
        // without naming a backend. It is also what lets the test-only mock backend be selected.
        if (!allow)                            { declined = "declined by the caller"; return; }
        if (!sub0_dev_caps().supports_eval)    { declined = "no device backend with a forward-loss entry"; return; }
        if (sub0_dev_init() != 0)              { declined = "device init failed"; return; }
        if (sub0_dev_upload_params(sub0::params_ptr()) != 0) {
            declined = "parameter upload failed";
            sub0_dev_shutdown();
            return;
        }
        use_device = true;
    }
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;
    ~Session() { if (use_device) sub0_dev_shutdown(); }
};

// The one entry every consumer should call: device when this session has one, CPU otherwise, and CPU
// as a FALLBACK if the device call fails mid-way (rather than propagating a NaN into a report).
[[nodiscard]] inline double nelbo(TokView data, const WindowSet& ws, int ctx, const Session& s,
                                  int threads) {
    if (s.use_device) {
        const double d = nelbo_device(data, ws, ctx);
        if (!std::isnan(d)) return d;
    }
    return nelbo_cpu(data, ws, ctx, threads);
}

}  // namespace sub0::eval
