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
#include <thread>
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
    std::string cores = "auto";   // auto | all | P | E | "lo-hi" | "a,b,c" (pin policy)
    int         repeat = 5;       // --mode bench: interleaved A/B rounds
    int         gpu_layers = 0;   // --mode hybrid*: first N layers run on the GPU
    bool        q8_kv = false;    // --q8-kv: q8 KV cache on the GPU layers (long context; lossy)
    int64_t     ctx = 0;          // --ctx N: KV-cache capacity (positions); 0 = prompt+max_tokens+1
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
        else if (s == "--cores")        a.cores = next();
        else if (s == "--repeat")       a.repeat = std::stoi(next());
        else if (s == "--gpu-layers")   a.gpu_layers = std::stoi(next());
        else if (s == "--q8-kv")        a.q8_kv = true;
        else if (s == "--ctx")          a.ctx = std::stoll(next());
        else if (s == "--pieces")       a.pieces = true;
        else if (s == "--no-bos")       a.no_bos = true;
        else if (s == "--no-fuse")      a.no_fuse = true;
        else if (s == "-h" || s == "--help") {
            std::cout << "usage: sub0llm-gemma --model G.gguf --text \"...\" "
                         "--mode tokenize|logits|greedy|bench|hybrid-check [-n N] [-t N] "
                         "[--no-fuse] [--repeat N] [--gpu-layers K]\n";
            std::exit(0);
        } else throw std::runtime_error(std::format("unknown argument: {}", s));
    }
    if (a.model.empty()) throw std::runtime_error("--model is required");
    return a;
}

// SIMD argmax over the V=262144 logits (std::max_element is scalar). First-max semantics.
int argmax(const std::vector<float>& v) {
    return static_cast<int>(
        sub0llm::backend::cpu::argmax_f32(v.data(), static_cast<int64_t>(v.size())));
}

// Parse an explicit CPU list: "lo-hi" range or "a,b,c" comma list.
std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    if (auto dash = s.find('-'); dash != std::string::npos && s.find(',') == std::string::npos) {
        const int lo = std::stoi(s.substr(0, dash)), hi = std::stoi(s.substr(dash + 1));
        for (int c = lo; c <= hi; ++c) out.push_back(c);
    } else {
        std::size_t i = 0;
        while (i < s.size()) {
            std::size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) out.push_back(std::stoi(s.substr(i, j - i)));
            i = j + 1;
        }
    }
    return out;
}

// Resolve a --cores policy to a logical-CPU pin set for a phase.
//   compute=true  → prompt-processing (compute-bound): prefers all cores / P-cores.
//   compute=false → token-generation (bandwidth-bound): prefers E-cores (frees P-cores).
// Returns empty = "no explicit set" (OS-default contiguous, i.e. all cores up to -t).
std::vector<int> resolve_cores(const std::string& policy,
                               const sub0llm::nn::GemmaCoreTopology& topo, bool compute) {
    if (policy == "all") return {};
    if (policy == "P")   return topo.perf;          // empty on non-hybrid → falls back to all
    if (policy == "E")   return topo.efficiency;
    if (policy == "auto") {
        if (!topo.hybrid()) return {};
        return compute ? std::vector<int>{} : topo.efficiency;   // PP: all cores; TG: E-cores
    }
    return parse_cpu_list(policy);                   // explicit "lo-hi" / "a,b,c"
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

        // Auto-detect P/E topology and resolve the --cores pin policy (no hardcoded index).
        const auto topo = sub0llm::nn::gemma_detect_cores();
        const auto pp_cpus = resolve_cores(args.cores, topo, /*compute=*/true);   // PP set
        const auto tg_cpus = resolve_cores(args.cores, topo, /*compute=*/false);  // TG set
        if (topo.hybrid())
            std::cerr << std::format("[gemma] cores: {} P-core + {} E-core (logical); "
                                     "policy '{}'\n", topo.perf.size(), topo.efficiency.size(),
                                     args.cores);

        const int64_t prompt_len = static_cast<int64_t>(ids.size());

        if (args.mode == "logits") {
            sub0llm::nn::set_gemma_cpus(pp_cpus);             // prefill = compute-bound
            auto kv = model.make_cache(prompt_len + 1);
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> logits = model.forward_prefill(ids.data(), prompt_len, 0, kv);
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
            {   // batched prefill (greedy: argmax only → skip softcap) — compute-bound → PP cores
                sub0llm::nn::set_gemma_cpus(pp_cpus);
                const auto pp0 = std::chrono::steady_clock::now();
                logits = model.forward_prefill(ids.data(), prompt_len, 0, kv, false);
                pos = prompt_len;
                const double pp_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pp0).count();
                std::cerr << std::format("[gemma] prompt forward {} tok in {:.2f}s ({:.2f} tok/s)\n",
                                         prompt_len, pp_s, static_cast<double>(prompt_len) / pp_s);
            }

            sub0llm::nn::set_gemma_cpus(tg_cpus);             // decode = bandwidth-bound → E-cores
            std::vector<int32_t> gen;
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t t = 0; t < args.max_tokens; ++t) {
                const int32_t next = argmax(logits);
                if (next == voc.eos_id) break;
                gen.push_back(next);
                logits = model.forward_one(next, pos, kv, false);
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

        if (args.mode == "hybrid") {
            // Like greedy, but the first --gpu-layers run on the GPU (forward_one_hybrid). Emits the
            // SAME PP/TG stderr lines as greedy so the interleaved bench parses both engines uniformly.
            // Token-by-token prefill (no batched GPU prefill yet) so PP is comparable to greedy's TTFT.
            const int64_t total = std::max<int64_t>(args.ctx, prompt_len + args.max_tokens + 1);
            model.enable_gpu_layers(args.gpu_layers, total, args.q8_kv);
            auto kv = model.make_cache(total);
            std::vector<float> logits;  int64_t pos = 0;
            {
                sub0llm::nn::set_gemma_cpus(pp_cpus);
                const auto pp0 = std::chrono::steady_clock::now();
                logits = model.forward_prefill_hybrid(ids.data(), prompt_len, 0, kv, false);  // batched GPU prefill
                pos = prompt_len;
                const double pp_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - pp0).count();
                std::cerr << std::format("[gemma] prompt forward {} tok in {:.2f}s ({:.2f} tok/s)\n",
                                         prompt_len, pp_s, static_cast<double>(prompt_len) / pp_s);
            }
            sub0llm::nn::set_gemma_cpus(tg_cpus);
            std::vector<int32_t> gen;
            const auto t0 = std::chrono::steady_clock::now();
            for (int64_t t = 0; t < args.max_tokens; ++t) {
                const int32_t next = argmax(logits);
                if (next == voc.eos_id) break;
                gen.push_back(next);
                logits = model.forward_one_hybrid(next, pos, kv, false);
                ++pos;
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "gen_ids:";
            for (int32_t id : gen) std::cout << ' ' << id;
            std::cout << "\ngen_text: " << sp.decode(gen) << "\n";
            std::cerr << std::format("[gemma] hybrid {} of {} layers on GPU\n", model.n_gpu_layers(), model.n_layers());
            std::cerr << std::format("[gemma] generated {} tok in {:.2f}s ({:.2f} tok/s)\n",
                                     gen.size(), s, static_cast<double>(gen.size()) / s);
            return 0;
        }

        if (args.mode == "bench") {
            // Drift-free A/B/C: cycle the cumulative orchestration levers within ONE
            // process (model loaded once, shared thermal state), interleaved over
            // `repeat` rounds. Single-token decode of `max_tokens` per measurement.
            sub0llm::nn::set_gemma_cpus(tg_cpus);            // decode benchmark = bandwidth set
            const int64_t G = args.max_tokens;
            auto decode_toks = [&](bool fuse, bool softcap) {
                sub0llm::nn::set_gemma_fuse(fuse);
                auto kv = model.make_cache(prompt_len + G + 1);
                std::vector<float> logits;
                int64_t pos = 0;
                for (; pos < prompt_len; ++pos)
                    logits = model.forward_one(ids[static_cast<std::size_t>(pos)], pos, kv, softcap);
                const auto t0 = std::chrono::steady_clock::now();
                for (int64_t t = 0; t < G; ++t) {
                    logits = model.forward_one(argmax(logits), pos, kv, softcap);
                    ++pos;
                }
                return static_cast<double>(G) / std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
            };
            struct Cfg { const char* name; bool fuse; bool softcap; std::vector<double> t; };
            Cfg cfgs[] = {
                {"baseline (no-fuse, softcap)", false, true,  {}},
                {"+fuse",                       true,  true,  {}},
                {"+fuse +no-softcap (best)",    true,  false, {}},
            };
            const int ncfg = static_cast<int>(std::size(cfgs));
            decode_toks(true, false);   // warmup
            // Rotate the evaluation order each round so every config samples each thermal
            // position equally — otherwise a fixed order biases later configs downward as
            // the chip heats within a round (the in-harness analogue of interleaving).
            for (int r = 0; r < args.repeat; ++r)
                for (int s = 0; s < ncfg; ++s) {
                    auto& c = cfgs[(r + s) % ncfg];
                    c.t.push_back(decode_toks(c.fuse, c.softcap));
                }

            auto median = [](std::vector<double> v) {
                std::sort(v.begin(), v.end()); return v[v.size() / 2];
            };
            const double base = median(cfgs[0].t);
            for (auto& c : cfgs) {
                std::sort(c.t.begin(), c.t.end());
                const double med = c.t[c.t.size() / 2];
                std::cout << std::format("{:<32s} median {:.2f} tok/s  best {:.2f}  ({:+.1f}% vs baseline)\n",
                                         c.name, med, c.t.back(), 100.0 * (med - base) / base);
            }
            return 0;
        }

        if (args.mode == "sweep") {
            // Thread-count sweep within ONE process (model loaded once): find OUR sweet
            // spot. Single-token decode at the best config (fused, no-softcap). Rotates
            // the thread-count order each round so thermal drift doesn't bias later T's.
            const int64_t G = args.max_tokens;
            sub0llm::nn::set_gemma_fuse(true);
            sub0llm::nn::set_gemma_cpus(pp_cpus);            // sweep across all cores (auto→{})
            auto decode_at = [&](int threads) {
                sub0llm::nn::set_gemma_threads(threads);
                auto kv = model.make_cache(prompt_len + G + 1);
                std::vector<float> logits;
                int64_t pos = 0;
                for (; pos < prompt_len; ++pos)
                    logits = model.forward_one(ids[static_cast<std::size_t>(pos)], pos, kv, false);
                const auto t0 = std::chrono::steady_clock::now();
                for (int64_t t = 0; t < G; ++t) {
                    logits = model.forward_one(argmax(logits), pos, kv, false);
                    ++pos;
                }
                return static_cast<double>(G) / std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
            };

            const unsigned hw = std::thread::hardware_concurrency();
            std::vector<int> Ts;
            for (int t : {1, 2, 4, 6, 8, 12, 16, 20, 24, 32})
                if (t <= static_cast<int>(hw)) Ts.push_back(t);
            std::vector<std::vector<double>> s(Ts.size());

            decode_at(static_cast<int>(hw > 4 ? hw - 4 : hw));   // warmup
            const int nt = static_cast<int>(Ts.size());
            for (int r = 0; r < args.repeat; ++r)
                for (int k = 0; k < nt; ++k) {
                    const int idx = (r + k) % nt;
                    s[static_cast<std::size_t>(idx)].push_back(decode_at(Ts[static_cast<std::size_t>(idx)]));
                }

            int best_t = Ts[0]; double best = 0.0;
            std::cout << "threads   median   best   tok/s\n";
            for (std::size_t k = 0; k < Ts.size(); ++k) {
                std::sort(s[k].begin(), s[k].end());
                const double med = s[k][s[k].size() / 2];
                if (med > best) { best = med; best_t = Ts[k]; }
                std::cout << std::format("  {:>4}   {:6.2f}  {:6.2f}\n", Ts[k], med, s[k].back());
            }
            std::cout << std::format("OUR SWEET SPOT: {} threads ({:.2f} tok/s median)\n", best_t, best);
            return 0;
        }

        if (args.mode == "hybrid-check") {
            // Full-model greedy PARITY gate: run the prompt+generation token-by-token two ways —
            // pure CPU (forward_one, our llama-validated oracle) and hybrid (first --gpu-layers on
            // the GPU, rest on CPU) — and report whether the GREEDY token sequences match. Both use
            // the token-by-token path so they are directly comparable (no batched-prefill skew).
            const int k = args.gpu_layers;
            // KV-cache capacity: --ctx sizes it independently of the token count, so we can measure
            // the VRAM pressure (and q8-vs-f32 fit) of a realistic context without generating it.
            const int64_t total = std::max<int64_t>(args.ctx, prompt_len + args.max_tokens + 1);

            auto greedy = [&](bool hybrid) {
                if (hybrid) model.enable_gpu_layers(k, total, args.q8_kv);
                auto kv = model.make_cache(total);
                std::vector<int32_t> gen;  std::vector<float> logits;  int64_t pos = 0;
                const auto fwd = [&](int32_t tok, bool want_logits) {
                    return hybrid ? model.forward_one_hybrid(tok, pos, kv, false, want_logits)
                                  : model.forward_one(tok, pos, kv, false, want_logits);
                };
                // Per-phase core policy (matches greedy mode): prompt prefill = compute cores;
                // decode = E-cores (bandwidth-bound). Time ONLY the decode loop → clean TG tok/s.
                sub0llm::nn::set_gemma_cpus(pp_cpus);
                for (; pos < prompt_len; ++pos) logits = fwd(ids[std::size_t(pos)], pos == prompt_len - 1);
                sub0llm::nn::set_gemma_cpus(tg_cpus);
                const auto t0 = std::chrono::steady_clock::now();
                for (int64_t t = 0; t < args.max_tokens; ++t) {
                    const int32_t nx = argmax(logits);
                    if (nx == voc.eos_id) break;
                    gen.push_back(nx);
                    logits = fwd(nx, true);
                    ++pos;
                }
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cerr << std::format("[gemma] {} decode: {} tok in {:.2f}s ({:.2f} tok/s)\n",
                                         hybrid ? "hybrid" : "cpu   ", gen.size(), s,
                                         static_cast<double>(gen.size()) / s);
                return gen;
            };

            const auto cpu_ids = greedy(false);
            std::cerr << std::format("[gemma] hybrid: first {} of {} layers on GPU\n", k, model.n_layers());
            const auto hyb_ids = greedy(true);

            std::size_t match = 0;
            const std::size_t n = std::min(cpu_ids.size(), hyb_ids.size());
            while (match < n && cpu_ids[match] == hyb_ids[match]) ++match;
            const bool ok = (cpu_ids == hyb_ids);
            std::cout << "cpu_ids   :"; for (int32_t id : cpu_ids) std::cout << ' ' << id; std::cout << "\n";
            std::cout << "hybrid_ids:"; for (int32_t id : hyb_ids) std::cout << ' ' << id; std::cout << "\n";
            std::cout << std::format("PARITY: {} — {}/{} greedy tokens match{}\n",
                                     ok ? "PASS (identical greedy sequence)" : "FAIL",
                                     match, std::max(cpu_ids.size(), hyb_ids.size()),
                                     ok ? "" : std::format(" (first divergence at index {})", match));
            return ok ? 0 : 2;
        }

        throw std::runtime_error(std::format("unknown mode: {}", args.mode));
    } catch (const std::exception& e) {
        std::cerr << "[gemma] error: " << e.what() << "\n";
        return 1;
    }
}
