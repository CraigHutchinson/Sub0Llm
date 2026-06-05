// sub0llm-verify — correctness harness (Ch27).
//
// Loads a GGUF model and emits outputs that can be diffed against a reference
// engine (llama.cpp) to gate any performance claim on numerical correctness:
//
//   --mode tokenize   print our BPE token ids   (vs llama-tokenize)
//   --mode greedy     greedy continuation ids+text (vs llama-cli --temp 0)
//   --mode logits     top-K next-token logits at the final position
//   --mode nll        mean NLL / perplexity over the text (vs llama-perplexity)
//
// "Performant but incorrect is not good": this is the oracle check the optimized
// (specialized) path must pass before its speedup means anything.

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace {

using sub0llm::Tensor;
using sub0llm::DType;

// Resident set size (working set) in MiB — to prove the Q8 RAM reclaim.
double rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
#endif
    return 0.0;
}

struct Args {
    std::string model;
    std::string text = "The capital of France is";
    std::string file;            // --file: read prompt text from a file
    std::string mode = "tokenize";
    int64_t     max_tokens = 20;
    int64_t     topk = 10;
    int64_t     cap = 0;         // --cap N: truncate to first N tokens (nll/logits)
    bool        pieces = false;
    bool        q8 = false;      // --q8: quantize attn+FFN+LM head to int8 (keep f32)
    bool        q8_only = false; // --q8-only: also drop f32 weights (RAM reclaim)
    bool        q8_load = false; // --q8-load: load raw Q8 directly (no f32 peak)
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", s));
            return argv[++i];
        };
        if (s == "--model")            a.model = next();
        else if (s == "--text")        a.text = next();
        else if (s == "--file")        a.file = next();
        else if (s == "--mode")        a.mode = next();
        else if (s == "-n" || s == "--max-tokens") a.max_tokens = std::stoll(next());
        else if (s == "--topk")        a.topk = std::stoll(next());
        else if (s == "--cap")         a.cap = std::stoll(next());
        else if (s == "--pieces")      a.pieces = true;
        else if (s == "--q8")          a.q8 = true;
        else if (s == "--q8-only")     a.q8_only = true;
        else if (s == "--q8-load")     a.q8_load = true;
        else if (s == "-h" || s == "--help") {
            std::cout << "usage: sub0llm-verify --model M.gguf --text \"...\" "
                         "--mode tokenize|greedy|logits|nll [-n N] [--topk K] [--pieces]\n";
            std::exit(0);
        } else throw std::runtime_error(std::format("unknown argument: {}", s));
    }
    if (a.model.empty()) throw std::runtime_error("--model is required");
    return a;
}

Tensor ids_tensor(const std::vector<int32_t>& ids) {
    Tensor t({static_cast<int64_t>(ids.size())}, DType::Int32);
    auto d = t.data_as<int32_t>();
    for (std::size_t i = 0; i < ids.size(); ++i) d[i] = ids[i];
    return t;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);

        sub0llm::nn::GGUFReader reader(args.model);
        auto tok   = sub0llm::BPETokenizer::from_vocab(reader.vocab().tokens,
                                                       reader.vocab().merges);

        std::string text = args.text;
        if (!args.file.empty()) {
            std::ifstream in(args.file, std::ios::binary);
            if (!in) throw std::runtime_error(std::format("cannot open --file {}", args.file));
            text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        auto ids = tok.encode(text);
        if (args.cap > 0 && static_cast<int64_t>(ids.size()) > args.cap)
            ids.resize(static_cast<std::size_t>(args.cap));

        if (args.mode == "tokenize") {
            for (std::size_t i = 0; i < ids.size(); ++i)
                std::cout << (i ? " " : "") << ids[i];
            std::cout << "\n";
            if (args.pieces)
                for (int32_t id : ids)
                    std::cout << "  " << id << " -> '" << tok.decode({id}) << "'\n";
            return 0;
        }

        // The remaining modes need the weights.
        auto model = args.q8_load ? sub0llm::nn::load_gguf_model_q8(reader)
                                  : sub0llm::nn::load_gguf_model(reader);
        const double rss_loaded = rss_mb();
        if (!args.q8_load && (args.q8 || args.q8_only))
            model.quantize_for_inference(/*drop_f32=*/args.q8_only);   // Ch27 int8 fast path
        if (rss_loaded > 0.0)
            std::cerr << std::format("[verify] RSS: {:.0f} MiB after load, {:.0f} MiB after "
                                     "quantize ({})\n", rss_loaded, rss_mb(),
                                     args.q8_load ? "q8-load, no f32 peak"
                                                  : args.q8_only ? "q8-only, f32 dropped"
                                                  : args.q8 ? "q8 + f32 kept" : "f32");

        if (args.mode == "greedy") {
            std::mt19937 rng(42);
            sub0llm::nn::SamplingConfig sc;  // default = Greedy (argmax)
            const auto all = sub0llm::nn::generate(model, ids, args.max_tokens, sc, rng);
            const std::vector<int32_t> gen(all.begin() + static_cast<std::ptrdiff_t>(ids.size()),
                                           all.end());
            std::cout << "prompt_ids:";
            for (int32_t id : ids) std::cout << ' ' << id;
            std::cout << "\ngen_ids:";
            for (int32_t id : gen) std::cout << ' ' << id;
            std::cout << "\ngen_text: " << tok.decode(gen) << "\n";
            return 0;
        }

        // forward once: logits (T, V)
        auto logits_var = model.forward(ids_tensor(ids));
        const Tensor& logits = logits_var.data();
        const int64_t T = logits.shape()[0];
        const int64_t V = logits.shape()[1];
        auto ld = logits.data_as<float>();

        if (args.mode == "logits") {
            const int64_t row = T - 1;  // next-token distribution after the prompt
            std::vector<int64_t> idx(static_cast<std::size_t>(V));
            std::iota(idx.begin(), idx.end(), 0);
            const std::size_t k = static_cast<std::size_t>(std::min<int64_t>(args.topk, V));
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                [&](int64_t x, int64_t y) {
                    return ld[static_cast<std::size_t>(row * V + x)] >
                           ld[static_cast<std::size_t>(row * V + y)];
                });
            std::cout << std::format("top-{} next-token logits after \"{}\":\n", k, args.text);
            for (std::size_t r = 0; r < k; ++r) {
                const int64_t id = idx[r];
                std::cout << std::format("  {:>8} {:>10.4f}  '{}'\n",
                                         id, ld[static_cast<std::size_t>(row * V + id)],
                                         tok.decode({static_cast<int32_t>(id)}));
            }
            return 0;
        }

        if (args.mode == "nll") {
            double total = 0.0; int64_t n = 0;
            for (int64_t i = 0; i < T - 1; ++i) {
                const int32_t target = ids[static_cast<std::size_t>(i + 1)];
                float mx = -std::numeric_limits<float>::infinity();
                for (int64_t v = 0; v < V; ++v)
                    mx = std::max(mx, ld[static_cast<std::size_t>(i * V + v)]);
                double se = 0.0;
                for (int64_t v = 0; v < V; ++v)
                    se += std::exp(ld[static_cast<std::size_t>(i * V + v)] - mx);
                const double logsum = mx + std::log(se);
                total += logsum - ld[static_cast<std::size_t>(i * V + target)];
                ++n;
            }
            const double mean = n ? total / static_cast<double>(n) : 0.0;
            std::cout << std::format("tokens={} mean_nll={:.5f} perplexity={:.4f}\n",
                                     ids.size(), mean, std::exp(mean));
            return 0;
        }

        throw std::runtime_error(std::format("unknown mode: {}", args.mode));
    } catch (const std::exception& e) {
        std::cerr << "[verify] error: " << e.what() << "\n";
        return 1;
    }
}
