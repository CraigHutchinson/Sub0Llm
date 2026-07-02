// gen_stage.cpp — backend stage "gen" (libsub0_gen).
//
// Owns autoregressive sampling: load a trained model, encode the prompt, and
// repeatedly run the engine forward pass, sampling the next token with
// temperature and top-k. Prints the continuation. Exposes one C entry point.

#include "sub0/core.hpp"

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
#if defined(SUB0_BUILD_CUDA)
extern "C" int  sub0_cuda_init();
extern "C" void sub0_cuda_shutdown();
extern "C" int  sub0_cuda_upload_params(const float* host);
extern "C" void sub0_cuda_set_tf32(int on);
extern "C" int  sub0_cuda_kv_reset();
extern "C" int  sub0_cuda_forward_one(int id, int pos, float* out_logits);
#endif

// Try to bring up the GPU KV-cache decode path for this generation: requires the CUDA backend
// compiled in and a device to initialize. Uploads the just-loaded weights once. Falls back to the CPU
// path (false) on any failure -- generation still completes, just at the CPU's per-token cost.
static bool gpu_decode_enable() {
#if defined(SUB0_BUILD_CUDA)
    if (!HAS_CUDA) return false;
    if (sub0_cuda_init() != 0) return false;
    sub0_cuda_set_tf32(0);   // match the FP32 config the parity test validated
    return sub0_cuda_upload_params(sub0::params_ptr()) == 0;
#else
    return false;
#endif
}

extern "C" SUB0_API int sub0_gen_stage(const char* model_in, const char* prompt,
                                        int n, float temp, int topk, unsigned seed) {
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
            std::vector<float> gpu_logits;              // only sized when the GPU path is active
            const float* logits = nullptr;
#if defined(SUB0_BUILD_CUDA)
            if (gpu_decode_enable()) {
                std::println(stderr, "gen: decode backend GPU (device KV-cache)");
                gpu_logits.resize(VOCAB);
                sub0_cuda_kv_reset();
                for (int pos = 0; pos < static_cast<int>(ctx.size()); ++pos)   // prefill the prompt
                    sub0_cuda_forward_one(ctx[pos], pos, gpu_logits.data());
                logits = gpu_logits.data();
                for (int s = 0; s < n; ++s) {
                    const int next = sub0::sample_token(logits, temp, topk, rng);
                    if (next == eos_id) break;   // learned stop signal: end here, don't print the marker
                    ctx.push_back(next);
                    sub0_cuda_forward_one(ctx.back(), static_cast<int>(ctx.size()) - 1, gpu_logits.data());
                }
                sub0_cuda_shutdown();
                std::println("{}", sub0::detokenize(ctx));
                return 0;
            }
#endif
            std::println(stderr, "gen: decode backend CPU (KV-cache)");
            sub0::kv_reset();
            for (int pos = 0; pos < static_cast<int>(ctx.size()); ++pos)   // prefill the prompt
                logits = sub0::forward_one(ctx[pos], pos);
            for (int s = 0; s < n; ++s) {
                const int next = sub0::sample_token(logits, temp, topk, rng);
                if (next == eos_id) break;       // learned stop signal: end here, don't print the marker
                ctx.push_back(next);
                logits = sub0::forward_one(ctx.back(), static_cast<int>(ctx.size()) - 1);
            }
            std::println("{}", sub0::detokenize(ctx));
            return 0;
        }
    }

    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
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
