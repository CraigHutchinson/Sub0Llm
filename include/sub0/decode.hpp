// sub0/decode.hpp — shared autoregressive KV-cache decode helper.
//
// gen_stage.cpp's fast path, train_stage.cpp's preview_at (training/report samples) and
// gen_self_stats (autotemp's generation sweep) all need the exact same shape: given a context that
// already fits inside SEQ_LEN, sample more tokens one at a time via the engine's KV-cache
// (forward_one), GPU-first with a CPU fallback, stopping BEFORE pushing the model's learned EOS
// marker so it is never emitted (see commit a016d17 -- sampling past EOS is out-of-distribution
// garbage, since no training document continues past its own end).
//
// Header-only, no library of its own: sub0_gen and sub0_train each already privately link
// sub0_backend_cuda and declare their own extern "C" seam into it (see CMakeLists.txt) -- moving the
// CUDA link edge into sub0_core would break its "ALWAYS the CPU backend" invariant (see sub0_core's
// own CMakeLists.txt comment), so this header instead declares the (small, decode-only) seam ONCE and
// drops into whichever TU includes it, compiled against that TU's own SUB0_BUILD_CUDA setting.
//
// Uses the neutral sub0_dev_* seam (device_backend.hpp, docs/BACKENDS.md) instead of its own
// sub0_cuda_* externs: the fail-fast stubs it provides without a device backend linked mean this file
// needs no #if scaffolding of its own -- gpu_decode_try_enable's HAS_CUDA check below still short-
// circuits the (harmless but pointless) stub call on a CPU-only build.
#pragma once

#include "sub0/core.hpp"
#include "sub0/casing.hpp"   // TOK_UNCOMBINE / TOK_COMBINE marker ids (kv_decode_generate interception)
#include "sub0/device_backend.hpp"   // sub0_dev_init/shutdown/upload_params/set_tf32/kv_reset/forward_one

#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace sub0 {

// Bring up the GPU KV-cache decode path: requires a device backend compiled in and a device to
// initialize, then uploads the CURRENT host params. Safe to call when a device context is already
// active elsewhere (sub0_dev_init is idempotent; re-uploading identical params is harmless) -- a live
// GPU training session's own final teardown (GpuTrainer::shutdown) still runs afterwards and is itself
// a no-op on an already-torn-down context. Returns false (caller uses the CPU path) on any failure: no
// device backend, no device, or an upload error -- generation still completes either way.
//
// Callers that intend to run MANY generations under one session (report's sample battery, autotemp's
// temperature grid) must call this ONCE and reuse the result across every kv_decode_generate call --
// re-enabling per generation would re-upload params and force a CUDA-graph recapture every time,
// undoing most of the speedup (see DecodeSession below, the RAII form of exactly that pattern).
inline bool gpu_decode_try_enable() {
    if (!HAS_CUDA) return false;   // short-circuits the (harmless but pointless) stub call otherwise
    // Honour the caps bit rather than assuming a linked backend decodes. docs/BACKENDS.md states
    // consumers "are DESIGNED to degrade around zeros", but this one did not read supports_decode at
    // all -- so a backend with train+eval but no decode (which the CUDA backend became once depth
    // attention landed) would have been driven straight into its unimplemented path. Cheap here, and it
    // is what makes `report` work on such a build: its sample battery silently uses the CPU instead.
    if (!sub0_dev_caps().supports_decode) return false;
    if (sub0_dev_init() != 0) return false;
    sub0_dev_set_tf32(0);   // tight FP32 inference math, matching the CUDA parity test's gate
    return sub0_dev_upload_params(sub0::params_ptr()) == 0;
}

// Tear down a GPU decode session brought up by gpu_decode_try_enable(). A no-op without a device
// backend linked (the stub), so callers can call it unconditionally (gated on their own success flag).
inline void gpu_decode_shutdown() {
    sub0_dev_shutdown();
}

// Autoregressively sample up to `n` more tokens onto `ctx` (already primed with a prompt/prefix) via
// KV-cache decode. Precondition: ctx.size() + n <= SEQ_LEN -- forward_one's own precondition (decode
// positions must stay < SEQ_LEN); callers whose context can run longer fall back to their own
// batched-forward sliding-window loop instead (e.g. gen_stage.cpp's attention-sink path -- preview_at/
// gen_self_stats never need sinks, a gen-only --attn-sinks feature, so they just fall back to a plain
// last-SEQ_LEN-tokens forward()). `use_gpu` selects the device path brought up by an earlier
// gpu_decode_try_enable()/DecodeSession (pass false to always use the CPU KV-cache).
//
// Stops BEFORE pushing `eos_id` -- the marker token itself is never appended to `ctx` -- matching
// gen_stage.cpp's decode loops exactly (commit a016d17). When given, `on_token(row, tok)` fires once
// per token actually appended (never for eos_id), with the [VOCAB] logits row it was sampled from, so
// a caller needing per-step statistics (gen_self_stats' NLL under the true T=1 distribution) gets them
// without a second forward pass.
//
// Combine/uncombine INTERCEPTION (folded in from the former standalone kv_decode_ops): when BOTH
// `expand` and `combine` are provided, the model can REQUEST deterministic tokenizer-layer operations
// that this loop fulfils (see sub0/spellspike.hpp + the TOK_UNCOMBINE/TOK_COMBINE comment in
// casing.hpp) -- the model learns to INVOKE, the tokenizer layer supplies the content:
//   expand(token)  -> the token's constituent fragments        (fulfils an emitted TOK_UNCOMBINE)
//   combine(frags) -> those fragments' minimal vocab token(s)   (fulfils a TOK_COMBINE region)
// The SAME seam generalises to any DETERMINISTIC COMPUTE NODE (see docs/DETERMINISTIC_MECHANISMS.md): when
// the model emits `compute_marker`, the loop calls `compute(ctx)` (given the context up to and including the
// marker -- the node parses its own operands from it) and injects the returned tokens. This is how the model
// learns to DELEGATE exact scalar computation (arithmetic, sort, ...) instead of approximating it -- routing
// is learned, the answer is exact. `compute` is independent of expand/combine (a compute-only caller works).
// Injected tokens ARE fed through the model (forward_one, so it attends to them) but never sampled.
// Interception runs on the CPU KV-cache path (the device-decode interceptor is a pending follow-on),
// so a caller that supplies the callbacks gets CPU decode regardless of `use_gpu`. With no callbacks
// (the default) this is byte-for-byte the original generate loop -- every existing caller is unchanged.
//
// `word_collapse`: HARNESS-DRIVEN (not model-requested) natural-word compaction (docs/SCRATCH_TOKENS.md,
// sub0::wordspike -- the mechanism repeatspike.hpp proved beats fuzzy re-spelling, applied live). Fires
// when the model finishes generating a multi-piece compound word (encode_join's own TOK_SPELL_START..END
// wrapping, tokenizer.cpp -- the SAME marker a model trained on any real text already learns to emit
// around a multi-piece word, no new signal invented). `word_collapse(span)` returns EITHER an empty
// vector ("not a recurrence, leave the just-generated span exactly as it is") or a non-empty replacement
// ("this recurred -- substitute this instead", typically one bound scratch slot). On a replacement, this
// retroactively SPLICES the KV-cache: `ctx.resize(open)` back to just before the word started, then
// `feed()`s the replacement at that (now-current) position. This needs no new low-level engine
// primitive: the CPU KVCache (backend_cpu.cpp) is a flat, position-indexed buffer with no separate
// length counter -- attention at any future forward_one call only ever reads `[0, pos]` for the `pos`
// it is given, so a smaller re-fed position is a plain overwrite, and RoPE (driven by that same explicit
// `pos` argument, not persistent state) is automatically correct afterward. Every multi-piece word
// (N>=2, schemeV3+, casing.hpp) is TOK_SPELL_START/END-wrapped, so this handles all of them live --
// an OLDER (schemeV2 and earlier) 2-piece word's bare TOK_JOIN had no closing marker to key a
// retroactive splice off; that gap closed when N==2 moved to SPELL-wrapping (see
// sub0::detail::word_span, scratch.hpp, and docs/CORPUS_COLLAPSE.md for the false-positive finding
// that motivated the change). PREFILL time (sub0::prefill_collapse, scratch.hpp) always handled every
// shape uniformly, since it has no live-splice asymmetry to begin with.
inline void kv_decode_generate(std::vector<int>& ctx, int n, float temp, int topk, std::mt19937& rng,
                               int eos_id, bool use_gpu,
                               const std::function<void(const float*, int)>& on_token = {},
                               const std::function<std::vector<int>(int)>& expand = {},
                               const std::function<std::vector<int>(const std::vector<int>&)>& combine = {},
                               const std::function<std::vector<int>(const std::vector<int>&)>& compute = {},
                               int compute_marker = -1,
                               const std::function<std::vector<int>(const std::vector<int>&)>& word_collapse = {}) {
    if (n <= 0) return;
    const bool intercept = static_cast<bool>(expand) && static_cast<bool>(combine);
    const bool do_compute = static_cast<bool>(compute) && compute_marker >= 0;
    const bool do_word_collapse = static_cast<bool>(word_collapse);
    // No #if here (device_backend.hpp's stubs make it unnecessary): use_gpu is only ever true after a
    // caller's own gpu_decode_try_enable() succeeded, which is already false-by-construction on a
    // build with no device backend linked (sub0_dev_init's stub returns -1) -- so this branch is
    // simply unreachable there, not compiled out.
    if (use_gpu && !intercept && !do_compute && !do_word_collapse) {   // interception is CPU-only for now -> fall through to CPU
        // A device fault here (illegal memory access or similar) leaves the CUDA context permanently
        // corrupted for the rest of the process (CUDA errors are sticky) -- every subsequent
        // forward_one call would return stale/garbage logits, which sample_token would then happily
        // sample from as if they were real, producing plausible-looking but meaningless continuation
        // text with no indication anything went wrong. Stop generating (keep whatever's already in
        // `ctx`, matching an early EOS stop) the moment any device call fails, rather than trusting
        // corrupted state -- same principle as GpuTrainer's step()/sync_to_host() failure handling in
        // train_stage.cpp, see [[gpu-illegal-access-hardware-fault-not-code-bug]].
        std::vector<float> logits(VOCAB);
        if (sub0_dev_kv_reset() != 0) {
            std::fprintf(stderr, "decode: GPU KV-cache reset failed -- stopping generation early\n");
            return;
        }
        for (int pos = 0; pos < static_cast<int>(ctx.size()); ++pos) {  // prefill the primed context
            if (sub0_dev_forward_one(ctx[pos], pos, logits.data()) != 0) {
                std::fprintf(stderr, "decode: GPU forward_one failed during prefill -- stopping "
                                     "generation early\n");
                return;
            }
        }
        for (int s = 0; s < n; ++s) {
            const int next = sub0::sample_token(logits.data(), temp, topk, rng);
            if (next == eos_id) break;         // learned stop signal -- never push the marker token
            if (on_token) on_token(logits.data(), next);
            ctx.push_back(next);
            if (sub0_dev_forward_one(ctx.back(), static_cast<int>(ctx.size()) - 1, logits.data()) != 0) {
                std::fprintf(stderr, "decode: GPU forward_one failed at generated token %d -- "
                                     "stopping generation early\n", s);
                return;
            }
        }
        return;
    }
    sub0::kv_reset();
    const float* logits = nullptr;
    for (int pos = 0; pos < static_cast<int>(ctx.size()); ++pos)       // prefill the primed context
        logits = sub0::forward_one(ctx[pos], pos);
    // Feed a token through the KV-cache (used for both sampled and harness-injected tokens).
    auto feed = [&](int tok) { ctx.push_back(tok); logits = sub0::forward_one(tok, static_cast<int>(ctx.size()) - 1); };
    // Resolve a just-processed marker `next` (the last token in ctx): a DISCRETE FORWARD-STEP delegation --
    // fulfil the requested deterministic op and INJECT its content (fed through the model so it attends to it,
    // but never sampled; guarded so injections never run past the window). Factored so it fires BOTH after
    // prefill (a prompt that ENDS in an op resolves in the forward pass -- no iterative generation needed) and
    // per generated token. See the former kv_decode_ops (now folded here) and sub0/spellspike.hpp.
    auto resolve = [&](int next) {
        if (do_compute && next == compute_marker) {   // deterministic node: inject its exact result
            for (int r : compute(ctx)) { if (static_cast<int>(ctx.size()) >= SEQ_LEN - 1) break; feed(r); }
        }
        if (do_word_collapse && next == casing::TOK_SPELL_END) {   // a compound word just finished
            std::size_t open = ctx.size();
            for (std::size_t i = ctx.size(); i-- > 0;) if (ctx[i] == casing::TOK_SPELL_START) { open = i; break; }
            if (open < ctx.size()) {
                const std::vector<int> span(ctx.begin() + static_cast<std::ptrdiff_t>(open) + 1, ctx.end() - 1);
                const std::vector<int> repl = word_collapse(span);
                if (!repl.empty()) {   // a recurrence -- retroactively splice (see this function's own header comment)
                    ctx.resize(open);
                    for (int r : repl) { if (static_cast<int>(ctx.size()) >= SEQ_LEN - 1) break; feed(r); }
                }
            }
        }
        if (!intercept) return;
        if (next == casing::TOK_UNCOMBINE) {
            const int tgt = ctx.size() >= 2 ? ctx[ctx.size() - 2] : -1;   // expand the PRECEDING token
            if (tgt >= 0) {
                for (int f : expand(tgt)) { if (static_cast<int>(ctx.size()) >= SEQ_LEN - 1) break; feed(f); }
                if (static_cast<int>(ctx.size()) < SEQ_LEN - 1) feed(casing::TOK_UNCOMBINE_END);
            }
        } else if (next == casing::TOK_COMBINE_END) {
            std::size_t open = ctx.size();                               // find the matching TOK_COMBINE
            for (std::size_t i = ctx.size(); i-- > 0;) if (ctx[i] == casing::TOK_COMBINE) { open = i; break; }
            if (open < ctx.size()) {
                const std::vector<int> frags(ctx.begin() + static_cast<std::ptrdiff_t>(open) + 1, ctx.end() - 1);
                for (int ct : combine(frags)) { if (static_cast<int>(ctx.size()) >= SEQ_LEN - 1) break; feed(ct); }
            }
        }
    };
    // FORWARD-PASS resolution of a prompt that ENDS in a completed op/marker: resolve it before generating, so
    // a prompt that merely POSES a computation is answered in the forward pass, not an iterative reasoning
    // loop. (Only the final marker -- a mid-prompt op would need a KV-cache-aware splice; backlog.) Also
    // covers word_collapse: a prompt happening to end exactly at a just-completed compound word gets the
    // same live-splice treatment here as a caller that pre-ran sub0::prefill_collapse over the WHOLE
    // prompt already gets for every earlier word -- defense in depth for a caller that doesn't.
    if ((do_compute || intercept || do_word_collapse) && !ctx.empty() && static_cast<int>(ctx.size()) < SEQ_LEN)
        resolve(ctx.back());
    // The extra `ctx.size() < SEQ_LEN` guard only matters in interception mode, where injected spans
    // can grow ctx past the n-token budget the non-intercept precondition (ctx.size()+n <= SEQ_LEN)
    // otherwise guarantees; it never fires without interception, so that path is byte-identical.
    for (int s = 0; s < n && static_cast<int>(ctx.size()) < SEQ_LEN; ++s) {
        const int next = sub0::sample_token(logits, temp, topk, rng);
        if (next == eos_id) break;             // learned stop signal -- never push the marker token
        if (on_token) on_token(logits, next);
        feed(next);
        resolve(next);
    }
}

// RAII: brings up a GPU decode session (if available) for a BATCH of kv_decode_generate calls made
// through the enclosing scope, and tears it down on the way out. Sharing one session across many
// generations -- report's whole sample battery, autotemp's whole temperature grid -- matters: a naive
// init+shutdown PER generation would make dozens/hundreds of short decodes dominated by CUDA context
// teardown/rebuild (and a forced graph recapture on every re-upload) instead of actual generation.
// `use_gpu` is what every kv_decode_generate call sharing this session should pass.
struct DecodeSession {
    bool use_gpu = false;
    DecodeSession() : use_gpu(gpu_decode_try_enable()) {}
    DecodeSession(const DecodeSession&) = delete;
    DecodeSession& operator=(const DecodeSession&) = delete;
    ~DecodeSession() { if (use_gpu) gpu_decode_shutdown(); }
};

}  // namespace sub0
