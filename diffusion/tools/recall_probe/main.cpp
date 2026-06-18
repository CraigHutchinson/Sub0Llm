// recall_probe — single-forward structured-recall capability probe for a trained denoiser.
//
// The Ch29 recall sweep masks RANDOM positions (the training distribution). This probe instead
// "breaks" REAL corpus windows in STRUCTURED ways and measures how well a single denoiser forward
// recovers the masked tokens (argmax over the real vocab, exact match) — the capability behind
// real use cases:
//   continuation  give the START of a phrase, mask the rest      → "finish it"
//   prefix        give the END, mask the front                   → "what came before"
//   middle-infill give both ENDS, mask a centred gap             → "fill the middle"
//   scatter       random k (the in-distribution baseline)
//   anchors-only  give just the two endpoints (near-unconditional, the hardest)
// across a difficulty arc (how much is given). Contiguous masks are OUT-OF-DISTRIBUTION vs the
// random-mask training, so this also measures structured-infilling transfer. Single forward (not
// Ch30 iterative sampling), so it is the model's RAW recall, uncompounded by self-conditioning.
//
//   recall_probe --model-dir DIR [--corpus data/complete_shakespeare.txt] [--n-windows 300] [--seed 7]

#include "sub0diff/nn/model_io.hpp"

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace sub0llm;
namespace dn = sub0diff::nn;

namespace {
// One paragraph per line (data/shakespeare format): strip CR + leading/trailing ws, skip blanks.
std::vector<std::string> read_paragraphs(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("recall_probe: cannot open corpus: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        std::size_t b = 0;
        while (b < line.size() && std::isspace(static_cast<unsigned char>(line[b]))) ++b;
        if (b < line.size()) out.push_back(line.substr(b));
    }
    return out;
}
}

int main(int argc, char** argv) {
    std::string model_dir, corpus = "data/complete_shakespeare.txt";
    std::size_t n_windows = 300;
    std::uint64_t seed = 7;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if      (a == "--model-dir")  model_dir = next();
        else if (a == "--corpus")     corpus    = next();
        else if (a == "--n-windows")  n_windows = std::stoull(next());
        else if (a == "--seed")       seed      = std::stoull(next());
    }
    if (model_dir.empty()) { std::println("error: --model-dir required"); return 2; }

    const dn::LoadedModel lm = dn::load_model_dir(model_dir);
    const dn::Denoiser& model = *lm.model;
    const auto& tok = *lm.tokenizer;
    const std::int64_t T = lm.seq_len;
    const std::int32_t mask_id = model.mask_id();
    const std::int64_t V = model.real_vocab();
    const std::int64_t C = model.model_vocab();
    std::println("recall_probe: {} (step {}) — V={} T={}, {} corpus windows\n",
                 model_dir, lm.step, V, T, n_windows);

    // Flat token stream over the whole corpus (the same sliding-window layout training saw).
    std::vector<std::int32_t> stream;
    for (const auto& p : read_paragraphs(corpus)) {
        auto ids = tok.encode(p);
        stream.insert(stream.end(), ids.begin(), ids.end());
    }
    if (static_cast<std::int64_t>(stream.size()) < T + 1) { std::println("corpus too small"); return 2; }

    // Fixed window offsets (same windows for every scenario and every model → fair comparison).
    std::mt19937 orng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<std::size_t> off_dist(0, stream.size() - static_cast<std::size_t>(T));
    std::vector<std::size_t> offsets(n_windows);
    for (auto& o : offsets) o = off_dist(orng);

    // mask_fn(masked[T], rng) sets masked[t]=1 for positions to corrupt. Random scenarios use rng
    // (reseeded per scenario so both models see identical masks); contiguous ones are deterministic.
    auto run = [&](const char* name, std::uint32_t mask_seed,
                   const std::function<void(std::vector<char>&, std::mt19937&)>& mask_fn) {
        std::mt19937 mrng(mask_seed);
        long total = 0, hits = 0, nm_sum = 0;
        std::vector<char> masked(static_cast<std::size_t>(T));
        for (std::size_t off : offsets) {
            auto window = std::span<const std::int32_t>(stream).subspan(off, static_cast<std::size_t>(T));
            std::ranges::fill(masked, 0);
            mask_fn(masked, mrng);
            Tensor input({T}, DType::Int32);
            auto in = input.data_as<std::int32_t>();
            int nm = 0;
            for (std::int64_t t = 0; t < T; ++t) {
                in[static_cast<std::size_t>(t)] = window[static_cast<std::size_t>(t)];
                if (masked[static_cast<std::size_t>(t)]) { in[static_cast<std::size_t>(t)] = mask_id; ++nm; }
            }
            if (nm == 0) continue;
            const float noise = static_cast<float>(nm) / static_cast<float>(T);
            auto logits = model.forward(input, noise);
            const auto lz = logits.data().data_as<float>();
            for (std::int64_t t = 0; t < T; ++t) {
                if (!masked[static_cast<std::size_t>(t)]) continue;
                std::int32_t best = 0; float bv = -1e30f;
                for (std::int64_t c = 0; c < V; ++c) {
                    const float v = lz[static_cast<std::size_t>(t) * static_cast<std::size_t>(C) + static_cast<std::size_t>(c)];
                    if (v > bv) { bv = v; best = static_cast<std::int32_t>(c); }
                }
                hits += (best == window[static_cast<std::size_t>(t)]);
                ++total;
            }
            nm_sum += nm;
        }
        const double mask_pct = 100.0 * static_cast<double>(nm_sum) / (static_cast<double>(n_windows) * static_cast<double>(T));
        const double recall = total ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0;
        std::println("  {:<26} {:>5.1f}% masked   recall {:>5.1f}%", name, mask_pct, recall);
    };

    auto suffix = [T](double give) { return [T, give](std::vector<char>& m, std::mt19937&) {
        const std::int64_t g = std::llround(give * static_cast<double>(T));
        for (std::int64_t t = g; t < T; ++t) m[static_cast<std::size_t>(t)] = 1; }; };
    auto prefix = [T](double give) { return [T, give](std::vector<char>& m, std::mt19937&) {
        const std::int64_t g = std::llround(give * static_cast<double>(T));
        for (std::int64_t t = 0; t < T - g; ++t) m[static_cast<std::size_t>(t)] = 1; }; };
    auto middle = [T](double gap) { return [T, gap](std::vector<char>& m, std::mt19937&) {
        const std::int64_t w = std::llround(gap * static_cast<double>(T));
        const std::int64_t a = (T - w) / 2;
        for (std::int64_t t = a; t < a + w; ++t) m[static_cast<std::size_t>(t)] = 1; }; };
    auto scatter = [T](double frac) { return [T, frac](std::vector<char>& m, std::mt19937& r) {
        std::int64_t k = std::llround(frac * static_cast<double>(T));
        k = std::clamp<std::int64_t>(k, 1, T - 1);
        std::int64_t rem_k = k, rem_n = T; std::uniform_real_distribution<float> u(0, 1);
        for (std::int64_t t = 0; t < T && rem_k > 0; ++t, --rem_n)
            if (u(r) * static_cast<float>(rem_n) < static_cast<float>(rem_k)) { m[static_cast<std::size_t>(t)] = 1; --rem_k; } }; };

    std::println("  {:<26} {:>13}   {:>12}", "scenario (corpus phrase)", "given→masked", "exact recall");
    std::println("  ── easiest ─────────────────────────────────────────────");
    run("single token (1 masked)", seed + 1, [T](std::vector<char>& m, std::mt19937& r) {
        std::uniform_int_distribution<std::int64_t> d(0, T - 1); m[static_cast<std::size_t>(d(r))] = 1; });
    run("scatter 25% (train-dist)", seed + 2, scatter(0.25));
    run("scatter 50% (train-dist)", seed + 3, scatter(0.50));
    std::println("  ── continuation: give the start, finish the phrase ─────");
    run("continuation give 75%", seed, suffix(0.75));
    run("continuation give 50%", seed, suffix(0.50));
    run("continuation give 25%", seed, suffix(0.25));
    std::println("  ── prefix: give the end, recover the start ─────────────");
    run("prefix give-last 50%", seed, prefix(0.50));
    std::println("  ── middle-infill: give both ends, fill the gap ─────────");
    run("middle gap 25%", seed, middle(0.25));
    run("middle gap 50%", seed, middle(0.50));
    std::println("  ── hardest: only the two endpoints given ───────────────");
    run("2 anchors (ends only)", seed, [T](std::vector<char>& m, std::mt19937&) {
        for (std::int64_t t = 1; t < T - 1; ++t) m[static_cast<std::size_t>(t)] = 1; });
    return 0;
}
