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
#include <print>
#include <random>
#include <string>
#include <vector>

extern "C" SUB0_API int sub0_gen_stage(const char* model_in, const char* prompt,
                                        int n, float temp, int topk, unsigned seed) {
    sub0::build_model();                 // establish parameter-node layout
    if (!sub0::load_model(model_in)) {   // overwrite with trained weights
        std::println(stderr, "gen: cannot load model '{}'", model_in);
        return 1;
    }
    if (!sub0::load_tokenizer(sub0::default_tokenizer())) {
        std::println(stderr, "gen: cannot load tokenizer '{}'", sub0::default_tokenizer());
        return 1;
    }

    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::mt19937 rng(seed);

    // Fast path: a KV-cache incremental decode (O(T) per token) whenever the whole context fits the
    // trained window and the weights are dense. forward_one's logits match the full forward to
    // fast-math tolerance, so the sampled continuation is the same -- just far cheaper. The general
    // path (ternary weights, or a context that slides past SEQ_LEN) keeps the exact full-forward loop.
    if constexpr (!USE_TERNARY) {
        if (static_cast<int>(ctx.size()) + n <= SEQ_LEN) {
            sub0::kv_reset();
            const float* logits = nullptr;
            for (int pos = 0; pos < static_cast<int>(ctx.size()); ++pos)   // prefill the prompt
                logits = sub0::forward_one(ctx[pos], pos);
            for (int s = 0; s < n; ++s) {
                ctx.push_back(sub0::sample_token(logits, temp, topk, rng));
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
        ctx.push_back(sub0::sample_token(logits->data.data() + (size_t)last * VOCAB, temp, topk, rng));
    }
    sub0::graph_reset();
    std::println("{}", sub0::detokenize(ctx));
    return 0;
}

// Dump the build's BPE vocabulary as a readable table. Inspection only; no model
// is loaded. `tokenizer_path` may be null to use the baked-in tokenizer.bin, and
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
