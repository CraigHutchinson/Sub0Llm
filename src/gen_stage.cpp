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
#include <random>
#include <string>
#include <vector>

extern "C" SUB0_API int sub0_gen_stage(const char* model_in, const char* prompt,
                                        int n, float temp, int topk, unsigned seed) {
    sub0::build_model();                 // establish parameter-node layout
    if (!sub0::load_model(model_in)) {   // overwrite with trained weights
        std::fprintf(stderr, "gen: cannot load model '%s'\n", model_in);
        return 1;
    }
    if (!sub0::load_tokenizer(sub0::default_tokenizer())) {
        std::fprintf(stderr, "gen: cannot load tokenizer '%s'\n", sub0::default_tokenizer());
        return 1;
    }

    std::vector<int> ctx = sub0::encode(prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::mt19937 rng(seed);

    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ctx.data() + (ctx.size() - T), T);
        const int last = logits->rows - 1;

        std::array<float, VOCAB> l;
        for (int j = 0; j < VOCAB; ++j)
            l[j] = logits->data[(size_t)last * VOCAB + j] / std::max(1e-6f, temp);

        if (topk > 0 && topk < VOCAB) {
            std::array<int, VOCAB> idx;
            for (int j = 0; j < VOCAB; ++j) idx[j] = j;
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&](int a, int b) { return l[a] > l[b]; });
            std::array<float, VOCAB> keep;
            keep.fill(-1e30f);
            for (int t = 0; t < topk; ++t) keep[idx[t]] = l[idx[t]];
            l = keep;
        }
        float mx = -1e30f;
        for (float x : l) mx = std::max(mx, x);
        float Z = 0.f;
        for (float& x : l) { x = std::exp(x - mx); Z += x; }
        for (float& x : l) x /= Z;

        std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng), acc = 0.f;
        int pick = VOCAB - 1;
        for (int j = 0; j < VOCAB; ++j) { acc += l[j]; if (r <= acc) { pick = j; break; } }
        ctx.push_back(pick);
    }
    sub0::graph_reset();
    std::printf("%s\n", sub0::detokenize(ctx).c_str());
    return 0;
}
