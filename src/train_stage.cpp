// train_stage.cpp — backend stage "train" (libsub0_train).
//
// Owns the training-time orchestration: corpus loading, minibatch sampling, the
// AdamW update schedule and progress reporting. The differentiable primitives
// (forward / cross_entropy / backward) and the optimizer itself come from the
// shared engine core. Exposes a single C entry point for the driver.

#include "sub0/core.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

// Read the pre-tokenized corpus (corpus.tok, "S0TK") into a flat id array.
std::vector<int> load_tokens(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return {};
    auto rd = [&] { std::uint32_t v{}; is.read(reinterpret_cast<char*>(&v), 4); return v; };
    if (rd() != 0x4B543053u) return {};  // "S0TK"
    (void)rd();                          // vocab (checked by load_tokenizer)
    const std::uint32_t ntok = rd();
    std::vector<int> data(ntok);
    is.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(ntok) * sizeof(int));
    if (!is) return {};
    return data;
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
    // Training consumes the pre-tokenized corpus produced by the configurator.
    // If the driver passed a plain .txt path, fall back to the baked-in .tok.
    std::string tok_path = corpus_path ? corpus_path : "";
    if (tok_path.size() < 4 || tok_path.compare(tok_path.size() - 4, 4, ".tok") != 0)
        tok_path = sub0::default_corpus_tok();

    std::vector<int> data = load_tokens(tok_path);
    if ((int)data.size() <= SEQ_LEN + 1) {
        std::fprintf(stderr, "train: cannot read tokens '%s' (or too small for seq_len %d)\n",
                     tok_path.c_str(), SEQ_LEN);
        return 1;
    }

    // The tokenizer is only needed so mid-training previews can render text.
    sub0::load_tokenizer(sub0::default_tokenizer());

    sub0::build_model();
    std::printf("corpus: %s (%zu tokens) | ", tok_path.c_str(), data.size());
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
            std::printf("step %5d/%d  loss %.4f\n", step, steps, run_loss / run_n);
            std::fflush(stdout);
            run_loss = 0.0; run_n = 0;
        }
        if (step % 500 == 0 || step == steps) {
            std::printf("  --- sample ---\n  %s\n", preview("the ", 120, rng).c_str());
            std::fflush(stdout);
        }
    }
    sub0::save_model(model_out);
    std::printf("saved model to %s\n", model_out);
    return 0;
}
