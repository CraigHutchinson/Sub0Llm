// sub0llm-gemma — faithful Gemma 4 text inference + correctness harness (Ch27).
//
// Gemma 4 is per-layer heterogeneous (local/global attention, varying head_dim and
// kv-head count, dual RoPE base, layer-output scaling, GeGLU, logit soft-cap) and
// cannot run on the uniform ModernGPT — it uses the dedicated GemmaModel. This tool
// loads the Q8 GGUF, tokenizes with the SentencePiece tokenizer, and emits outputs
// to diff against llama.cpp (the correctness oracle):
//
//   --mode tokenize   SentencePiece token ids        (vs llama-tokenize)
//   --mode logits     top-K next-token logits        (vs llama logit dump)
//   --mode greedy     greedy continuation ids+text   (vs llama-completion -no-cnv)
//
// "Performant but incorrect is not good": correctness gates every perf claim.

#include "sub0llm/nn/gemma.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/tokenizer/sentencepiece.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
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
    std::string file;
    std::string mode = "tokenize";
    int64_t     max_tokens = 32;
    int64_t     topk = 10;
    int         threads = 0;      // 0 = auto; set to 1 for a single-core baseline
    int         repeat = 5;       // --mode bench: interleaved A/B rounds
    bool        pieces = false;
    bool        no_bos = false;   // for parity when the reference already added BOS
    bool        no_fuse = false;  // disable Q/K/V + gate/up GEMV fusion
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", s));
            return argv[++i];
        };
        if (s == "--model")             a.model = next();
        else if (s == "--text")         a.text = next();
        else if (s == "--file")         a.file = next();
        else if (s == "--mode")         a.mode = next();
        else if (s == "-n" || s == "--max-tokens") a.max_tokens = std::stoll(next());
        else if (s == "--topk")         a.topk = std::stoll(next());
        else if (s == "-t" || s == "--threads") a.threads = std::stoi(next());
        else if (s == "--repeat")       a.repeat = std::stoi(next());
        else if (s == "--pieces")       a.pieces = true;
        else if (s == "--no-bos")       a.no_bos = true;
        else if (s == "--no-fuse")      a.no_fuse = true;
        else if (s == "-h" || s == "--help") {
            std::cout << "usage: sub0llm-gemma --model G.gguf --text \"...\" "
                         "--mode tokenize|logits|greedy|bench [-n N] [-t N] [--no-fuse] "
                         "[--repeat N]\n";
            std::exit(0);
        } else throw std::runtime_error(std::format("unknown argument: {}", s));
    }
    if (a.model.empty()) throw std::runtime_error("--model is required");
    return a;
}

int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);

        sub0llm::nn::set_gemma_threads(args.threads);
        sub0llm::nn::set_gemma_fuse(!args.no_fuse);

        sub0llm::nn::GGUFReader reader(args.model);
        const auto& voc = reader.vocab();
        auto sp = sub0llm::SPTokenizer::from_vocab(voc.tokens, voc.scores,
                                                   voc.bos_id, voc.eos_id,
                                                   args.no_bos ? false : voc.add_bos);

        std::string text = args.text;
        if (!args.file.empty()) {
            std::ifstream in(args.file, std::ios::binary);
            if (!in) throw std::runtime_error(std::format("cannot open --file {}", args.file));
            text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }

        auto ids = sp.encode(text);

        if (args.mode == "tokenize") {
            for (std::size_t i = 0; i < ids.size(); ++i)
                std::cout << (i ? " " : "") << ids[i];
            std::cout << "\n";
            if (args.pieces)
                for (int32_t id : ids)
                    std::cout << "  " << id << " -> '" << sp.decode({id}) << "'\n";
            return 0;
        }

        // The remaining modes need the weights.
        const auto t_load0 = std::chrono::steady_clock::now();
        auto model = sub0llm::nn::GemmaModel::load_q8(reader);
        const double load_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_load0).count();
        std::cerr << std::format("[gemma] loaded {} layers, {:.2f}B params, {:.1f}s, RSS {:.0f} MiB\n",
                                 model.n_layers(),
                                 static_cast<double>(model.n_params()) / 1e9,
                                 load_s, rss_mb());

        const int64_t prompt_len = static_cast<int64_t>(ids.size());

        if (args.mode == "logits") {
            auto kv = model.make_cache(prompt_len + 1);
            std::vector<float> logits;
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t p = 0; p < prompt_len; ++p)
                logits = model.forward_one(ids[static_cast<std::size_t>(p)], p, kv);
            const double s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            const int64_t V = model.vocab_size();
            const std::size_t k = static_cast<std::size_t>(std::min<int64_t>(args.topk, V));
            std::vector<int64_t> idx(static_cast<std::size_t>(V));
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                [&](int64_t a, int64_t b) {
                    return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
                });
            std::cerr << std::format("[gemma] prompt forward {} tok in {:.2f}s ({:.2f} tok/s)\n",
                                     prompt_len, s, static_cast<double>(prompt_len) / s);
            std::cout << std::format("top-{} next-token logits after \"{}\":\n", k, text);
            for (std::size_t r = 0; r < k; ++r) {
                const int64_t id = idx[r];
                std::cout << std::format("  {:>8} {:>10.4f}  '{}'\n",
                                         id, logits[static_cast<std::size_t>(id)],
                                         sp.decode({static_cast<int32_t>(id)}));
            }
            return 0;
        }

        if (args.mode == "greedy") {
            auto kv = model.make_cache(prompt_len + args.max_tokens + 1);
            std::vector<float> logits;
            int64_t pos = 0;
            for (; pos < prompt_len; ++pos)
                logits = model.forward_one(ids[static_cast<std::size_t>(pos)], pos, kv);

            std::vector<int32_t> gen;
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t t = 0; t < args.max_tokens; ++t) {
                const int32_t next = argmax(logits);
                if (next == voc.eos_id) break;
                gen.push_back(next);
                logits = model.forward_one(next, pos, kv);
                ++pos;
            }
            const double s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            std::cout << "prompt_ids:";
            for (int32_t id : ids) std::cout << ' ' << id;
            std::cout << "\ngen_ids:";
            for (int32_t id : gen) std::cout << ' ' << id;
            std::cout << "\ngen_text: " << sp.decode(gen) << "\n";
            std::cerr << std::format("[gemma] generated {} tok in {:.2f}s ({:.2f} tok/s)\n",
                                     gen.size(), s, static_cast<double>(gen.size()) / s);
            return 0;
        }

        if (args.mode == "bench") {
            // Drift-free A/B: alternate fuse-ON / fuse-OFF within ONE process (model
            // loaded once, shared thermal state), interleaved over `repeat` rounds.
            // Single-token decode of `max_tokens` per measurement; report best+median.
            const int64_t G = args.max_tokens;
            auto decode_toks = [&](bool fuse) {
                sub0llm::nn::set_gemma_fuse(fuse);
                auto kv = model.make_cache(prompt_len + G + 1);
                std::vector<float> logits;
                int64_t pos = 0;
                for (; pos < prompt_len; ++pos)
                    logits = model.forward_one(ids[static_cast<std::size_t>(pos)], pos, kv);
                const auto t0 = std::chrono::steady_clock::now();
                for (int64_t t = 0; t < G; ++t) {
                    logits = model.forward_one(argmax(logits), pos, kv);
                    ++pos;
                }
                return static_cast<double>(G) / std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
            };
            decode_toks(true);   // warmup (page-in, settle clocks)

            std::vector<double> on, off;
            for (int r = 0; r < args.repeat; ++r) {
                const double a = decode_toks(true);
                const double b = decode_toks(false);
                on.push_back(a); off.push_back(b);
                std::cerr << std::format("[bench] round {}: fuse {:.2f} | no-fuse {:.2f} tok/s\n",
                                         r + 1, a, b);
            }
            auto stats = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                return std::pair{v.back(), v[v.size() / 2]};  // best, median
            };
            const auto [on_best, on_med]   = stats(on);
            const auto [off_best, off_med] = stats(off);
            std::cout << std::format(
                "FUSE    best {:.2f}  median {:.2f} tok/s\n"
                "NO-FUSE best {:.2f}  median {:.2f} tok/s\n"
                "fusion delta (median): {:+.1f}%\n",
                on_best, on_med, off_best, off_med,
                100.0 * (on_med - off_med) / off_med);
            return 0;
        }

        throw std::runtime_error(std::format("unknown mode: {}", args.mode));
    } catch (const std::exception& e) {
        std::cerr << "[gemma] error: " << e.what() << "\n";
        return 1;
    }
}
