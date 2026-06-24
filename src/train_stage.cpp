// train_stage.cpp — backend stage "train" (libsub0_train).
//
// Owns the training-time orchestration: corpus loading, minibatch sampling, the
// AdamW update schedule and progress reporting. The differentiable primitives
// (forward / cross_entropy / backward) and the optimizer itself come from the
// shared engine core. Exposes a single C entry point for the driver.

#include "sub0/core.hpp"

#include <cstdint>
#include <cstdio>
#include <expected>
#include <fstream>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace {

// Why a corpus.tok could not be loaded; the success payload is TokData.
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

// A short sample, used for mid-training previews.
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
        // mild sampling so previews aren't fully greedy
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
                std::println(stderr, "train: cannot open token file '{}'", tok_path);
                break;
            case TokError::BadMagic:
                std::println(stderr, "train: '{}' is not a corpus.tok file (bad magic)", tok_path);
                break;
            case TokError::Truncated:
                std::println(stderr, "train: '{}' is truncated (token count exceeds file size)", tok_path);
                break;
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
    if (data.size() <= static_cast<size_t>(SEQ_LEN) + 1) {
        std::println(stderr, "train: '{}' has only {} tokens, too few for seq_len {} (need > {})",
                     tok_path, data.size(), SEQ_LEN, SEQ_LEN + 1);
        return 1;
    }

    // Defense in depth: even with a matching vocab field, a corrupt file could hold
    // an out-of-range id. One linear scan is cheap next to training and turns a
    // silent OOB gather into a clear diagnostic.
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < 0 || data[i] >= VOCAB) {
            std::println(stderr, "train: '{}' token {} = {} is out of range [0,{})",
                         tok_path, i, data[i], VOCAB);
            return 1;
        }
    }

    // The tokenizer is only needed so mid-training previews can render text.
    sub0::load_tokenizer(sub0::default_tokenizer());

    sub0::build_model();
    std::print("corpus: {} ({} tokens) | ", tok_path, data.size());
    sub0::print_config();
    std::fflush(stdout);

    sub0::AdamW opt(lr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);

    double run_loss = 0.0; int run_n = 0;
    for (int step = 1; step <= steps; ++step) {
        opt.zero_grad();
        float step_loss = 0.f;
        for (int b = 0; b < batch; ++b) {
            size_t s = startd(rng);
            const int* x = data.data() + s;
            const int* y = data.data() + s + 1;
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(x, SEQ_LEN);
            sub0::Node* loss = sub0::cross_entropy(logits, y);
            step_loss += loss->data[0] / batch;
            sub0::backward(loss, 1.f / batch);
        }
        opt.step();
        run_loss += step_loss; ++run_n;

        if (step % 100 == 0 || step == 1) {
            std::println("step {:5}/{}  loss {:.4f}", step, steps, run_loss / run_n);
            std::fflush(stdout);
            run_loss = 0.0; run_n = 0;
        }
        if (step % 500 == 0 || step == steps) {
            std::println("  --- sample ---\n  {}", preview("the ", 120, rng));
            std::fflush(stdout);
        }
    }
    sub0::save_model(model_out);
    std::println("saved model to {}", model_out);
    return 0;
}
