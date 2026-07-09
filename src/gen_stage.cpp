// gen_stage.cpp — backend stage "gen" (libsub0_gen).
//
// Owns autoregressive sampling: load a trained model, encode the prompt, and
// repeatedly run the engine forward pass, sampling the next token with
// temperature and top-k. Prints the continuation. Exposes one C entry point.

#include "sub0/core.hpp"
#include "sub0/decode.hpp"  // shared KV-cache decode loop (GPU-first, CPU-fallback, EOS-stop)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <print>
#include <random>
#include <string>
#include <vector>

// GPU KV-cache decode (Phase 2e follow-up to the training fast-path in train_stage.cpp). Parity-
// validated against a TRAINED model in cuda_tests.cpp ("CUDA forward_one decode matches full forward
// once the model is trained"): 100% top-1 agreement, ~1e-6 max rel diff at the d768/HD=96 config that
// diverges under random weights (a benign flash-vs-two-pass softmax amplification, not a kernel bug --
// see the memory notes on the d448 random-weight investigation). Device forward_one is FP32-only
// (matches the CPU path's dense assumption), so this mirrors USE_TERNARY gating below.

extern "C" SUB0_API int sub0_gen_stage(const char* model_in, const char* prompt,
                                        int n, float temp, int topk, unsigned seed, int attn_sinks) {
    sub0::build_model();                 // establish parameter-node layout
    if (!sub0::load_model(model_in)) {   // overwrite with trained weights (also reads the vocab fingerprint)
        std::println(stderr, "gen: cannot load model '{}'", model_in);
        return 1;
    }
    // Decode with the vocab this model was TRAINED against, not whatever the build tree currently holds:
    // prefer the tokenizer.tok bundled beside the model, falling back to the baked default. The
    // fingerprint guard in load_tokenizer then rejects a mismatch loudly rather than emitting garble.
    std::string tok_path = sub0::default_tokenizer();
    if (model_in && *model_in) {
        const std::filesystem::path bundled = std::filesystem::path(model_in).parent_path() / "tokenizer.tok";
        std::error_code ec;
        if (std::filesystem::exists(bundled, ec)) tok_path = bundled.string();
    }
    if (!sub0::load_tokenizer(tok_path.c_str())) {
        std::println(stderr, "gen: cannot load tokenizer '{}'", tok_path);
        return 1;
    }

    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::mt19937 rng(seed);
    // The model's learned stop signal (see casing.hpp's TOK_EOS): sampling loops below check for it
    // BEFORE pushing, so it never enters ctx / gets printed -- generation just ends naturally instead
    // of always running to the fixed token budget `n`. -1 (no tokenizer eos_id, e.g. a pre-EOS model)
    // never matches a real sampled id, so this degrades to today's fixed-budget behavior transparently.
    const int eos_id = sub0::eos_token_id();

    // Fast path: a KV-cache incremental decode (O(T) per token) whenever the whole context fits the
    // trained window and the weights are dense. forward_one's logits match the full forward to
    // fast-math tolerance, so the sampled continuation is the same -- just far cheaper. GPU decode is
    // tried first (device forward_one, faster per token); it falls back to the CPU KV-cache below on
    // any failure (no CUDA build, no device, or an upload error), so this always completes. The general
    // path (ternary weights, or a context that slides past SEQ_LEN) keeps the exact full-forward loop.
    if constexpr (!USE_TERNARY) {
        if (static_cast<int>(ctx.size()) + n <= SEQ_LEN) {
            sub0::DecodeSession sess;
            std::println(stderr, "gen: decode backend {}",
                        sess.use_gpu ? "GPU (device KV-cache)" : "CPU (KV-cache)");
            sub0::kv_decode_generate(ctx, n, temp, topk, rng, eos_id, sess.use_gpu);
            std::println("{}", sub0::detokenize(ctx));
            return 0;
        }
    }

    // Sink tokens: clamped so the "recent" half of the window never shrinks below SEQ_LEN/2 even if
    // the caller asks for an unreasonably large --attn-sinks.
    const int n_sink = attn_sinks > 0 ? std::min(attn_sinks, SEQ_LEN / 2) : 0;
    std::vector<int> window;   // scratch, rebuilt each step once sinks are active; reused to avoid realloc
    for (int s = 0; s < n; ++s) {
        int T;
        const int* win;
        // Once ctx has grown past the trained window, a plain last-SEQ_LEN slice drops the sequence's
        // opening tokens entirely -- StreamingLLM's observation is that those early tokens soak up a
        // disproportionate share of attention mass regardless of their content, so dropping them can
        // degrade quality more than their content alone would suggest. Keep the first n_sink tokens
        // resident alongside the most recent (SEQ_LEN - n_sink), forming a contiguous SEQ_LEN window.
        // RoPE only encodes RELATIVE row position within a forward() call (row index, not the token's
        // true absolute step), so reassigning the sinks to rows [0,n_sink) and the recent tail to
        // [n_sink,SEQ_LEN) needs no change to op_rope/forward() at all -- both halves land in-
        // distribution positions every step, with no artificial position gap between them.
        if (n_sink > 0 && static_cast<int>(ctx.size()) > SEQ_LEN) {
            const int n_recent = SEQ_LEN - n_sink;
            window.assign(ctx.begin(), ctx.begin() + n_sink);
            window.insert(window.end(), ctx.end() - n_recent, ctx.end());
            T = SEQ_LEN;
            win = window.data();
        } else {
            T = std::min((int)ctx.size(), SEQ_LEN);
            win = ctx.data() + (ctx.size() - T);
        }
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(win, T);
        const int last = logits->rows - 1;
        const int next = sub0::sample_token(logits->data.data() + (size_t)last * VOCAB, temp, topk, rng);
        if (next == eos_id) break;   // learned stop signal: end here, don't print the marker
        ctx.push_back(next);
    }
    sub0::graph_reset();
    std::println("{}", sub0::detokenize(ctx));
    return 0;
}

// Dump the build's BPE vocabulary as a readable table. Inspection only; no model
// is loaded. `tokenizer_path` may be null to use the baked-in tokenizer.tok, and
// `limit <= 0` prints every entry.
extern "C" SUB0_API int sub0_vocab_stage(const char* tokenizer_path, int limit) {
    const char* path = tokenizer_path ? tokenizer_path : sub0::default_tokenizer();
    if (!sub0::load_tokenizer(path)) {
        std::println(stderr, "vocab: cannot load tokenizer '{}'", path);
        return 1;
    }
    const std::vector<sub0::TokenEntry> rows = sub0::vocab_entries();

    int n_base = 0, n_merge = 0;
    for (const auto& e : rows)
        (e.kind == sub0::TokenEntry::Kind::Merge ? n_merge : n_base) += 1;
    std::println("tokenizer: {} | {} tokens ({} base, {} merges)",
                 path, rows.size(), n_base, n_merge);
    std::println("{:>6}  {:<6}  {:>3}  text", "id", "kind", "len");

    const std::size_t shown =
        (limit > 0) ? std::min(rows.size(), static_cast<std::size_t>(limit)) : rows.size();
    for (std::size_t i = 0; i < shown; ++i) {
        const sub0::TokenEntry& e = rows[i];
        const char* kind = e.kind == sub0::TokenEntry::Kind::Byte      ? "byte"
                         : e.kind == sub0::TokenEntry::Kind::CapMarker ? "cap"
                         : e.kind == sub0::TokenEntry::Kind::UpMarker  ? "up"
                                                                       : "merge";
        std::println("{:>6}  {:<6}  {:>3}  {}", e.id, kind, e.expansion_len, e.text);
    }
    if (shown < rows.size())
        std::println("... {} more (raise --limit to see all)", rows.size() - shown);
    return 0;
}
