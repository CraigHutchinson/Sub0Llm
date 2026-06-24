// train_stage.cpp — backend stage "train" (libsub0_train).
//
// Owns the training-time orchestration: corpus loading, minibatch sampling, the
// AdamW update schedule and progress reporting. The differentiable primitives
// (forward / cross_entropy / backward) and the optimizer itself come from the
// shared engine core. Exposes a single C entry point for the driver.

#include "sub0/core.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return {};
    return std::string((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
}

// A short sample, used for mid-training previews.
std::string preview(const std::string& prompt, int n, std::mt19937& rng) {
    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::string out = prompt;
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
        out.push_back(sub0::decode(best));
    }
    sub0::graph_reset();
    return out;
}

}  // namespace

extern "C" SUB0_API int sub0_train_stage(const char* corpus_path, const char* model_out,
                                          int steps, int batch, float lr, unsigned seed) {
    std::string text = read_file(corpus_path);
    if (text.empty()) { std::fprintf(stderr, "train: cannot read corpus '%s'\n", corpus_path); return 1; }
    std::vector<int> data = sub0::encode(text);
    if ((int)data.size() <= SEQ_LEN + 1) {
        std::fprintf(stderr, "train: corpus too small (%zu) for seq_len %d\n", data.size(), SEQ_LEN);
        return 1;
    }

    sub0::build_model();
    std::printf("corpus: %zu bytes (%zu tokens) | ", text.size(), data.size());
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
