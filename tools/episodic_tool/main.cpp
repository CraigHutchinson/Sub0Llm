// sub0llm-episodic — CLI tool for episodic memory operations on GGUF models.
//
// Subcommands:
//   write  --model MODEL.gguf --fact "TEXT" --delta DELTA.epis [--lr F] [--steps N]
//   recall --model MODEL.gguf --query "TEXT" --delta DELTA.epis
//   info   --model MODEL.gguf [--delta DELTA.epis]

#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ── simple_tokenize ────────────────────────────────────────────────────────────
//
// Greedy longest-match tokenizer over a vocabulary of token strings.
// Falls back to single-byte encoding if no match is found.
std::vector<int32_t> simple_tokenize(const std::string& text,
                                      const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int32_t> tok2id;
    tok2id.reserve(vocab.size());
    for (std::size_t i = 0; i < vocab.size(); ++i)
        tok2id[vocab[i]] = static_cast<int32_t>(i);

    std::vector<int32_t> ids;
    std::size_t pos = 0;
    while (pos < text.size()) {
        // Find longest match starting at pos
        std::size_t best_len = 0;
        int32_t     best_id  = -1;
        // Try lengths from longest possible down to 1
        const std::size_t max_len = std::min(text.size() - pos, std::size_t{32});
        for (std::size_t len = max_len; len >= 1; --len) {
            auto it = tok2id.find(text.substr(pos, len));
            if (it != tok2id.end()) {
                best_len = len;
                best_id  = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            pos += best_len;
        } else {
            // Unknown byte — use id 0 (or skip)
            ids.push_back(0);
            pos += 1;
        }
    }
    return ids;
}

// ── CLI arg parsing ────────────────────────────────────────────────────────────

struct Args {
    std::string subcmd;
    std::string model_path;
    std::string fact;
    std::string query;
    std::string delta_path;
    float lr        = 3e-2f;
    int   steps     = 10;
};

void print_usage() {
    std::cerr << R"(sub0llm-episodic — episodic memory operations on GGUF models

Usage:
  sub0llm-episodic write  --model MODEL.gguf --fact "TEXT" --delta DELTA.epis [--lr F] [--steps N]
  sub0llm-episodic recall --model MODEL.gguf --query "TEXT" --delta DELTA.epis
  sub0llm-episodic info   --model MODEL.gguf [--delta DELTA.epis]

Subcommands:
  write   Encode a fact into an episodic delta and save it
  recall  Load delta, run comprehension pass on query, print per-token NLL
  info    Show model architecture info from GGUF header

Options:
  --model  PATH   path to .gguf model file
  --fact   TEXT   text to encode (write subcommand)
  --query  TEXT   text to query (recall subcommand)
  --delta  PATH   path to episodic delta file (.epis)
  --lr     F      learning rate for episodic write (default: 0.03)
  --steps  N      elaborative rehearsal steps (default: 10)
)";
}

Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        std::exit(1);
    }

    Args args;
    args.subcmd = argv[1];
    if (args.subcmd == "--help" || args.subcmd == "-h") {
        print_usage();
        std::exit(0);
    }
    if (args.subcmd != "write" && args.subcmd != "recall" && args.subcmd != "info") {
        std::cerr << "Unknown subcommand: " << args.subcmd << "\n\n";
        print_usage();
        std::exit(1);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--model") args.model_path = next();
        else if (a == "--fact")  args.fact        = next();
        else if (a == "--query") args.query       = next();
        else if (a == "--delta") args.delta_path  = next();
        else if (a == "--lr")    args.lr          = std::stof(next());
        else if (a == "--steps") args.steps       = std::stoi(next());
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            print_usage();
            std::exit(1);
        }
    }

    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_usage();
        std::exit(1);
    }

    return args;
}

// ── subcommands ────────────────────────────────────────────────────────────────

int cmd_info(const Args& args) {
    sub0llm::nn::GGUFReader reader(args.model_path);
    const auto& cfg = reader.config();

    int64_t n_params = 0;
    for (const auto& [name, ti] : reader.tensors())
        n_params += ti.numel;

    std::cout << std::format("Model:       {}\n", args.model_path);
    std::cout << std::format("Arch:        {}\n", cfg.arch.empty() ? "(unknown)" : cfg.arch);
    std::cout << std::format("Vocab size:  {}\n", cfg.vocab_size);
    std::cout << std::format("Embed dim:   {}\n", cfg.embed_dim);
    std::cout << std::format("Layers:      {}\n", cfg.n_layers);
    std::cout << std::format("Heads:       {} (kv: {})\n", cfg.n_heads, cfg.n_kv_heads);
    std::cout << std::format("FFN dim:     {}\n", cfg.d_ff);
    std::cout << std::format("Context len: {}\n", cfg.context_len);
    std::cout << std::format("RoPE base:   {:.1f}\n", cfg.rope_base);
    std::cout << std::format("Tensors:     {}\n", reader.tensors().size());
    std::cout << std::format("Total elems: {:.2f}M\n",
                              static_cast<double>(n_params) / 1e6);

    if (!args.delta_path.empty() && std::filesystem::exists(args.delta_path)) {
        try {
            const auto delta = sub0llm::nn::load_episodic_state(args.delta_path);
            int64_t delta_elems = 0;
            for (const auto& d : delta.deltas) delta_elems += d.numel();
            std::cout << std::format("\nDelta file:  {}\n", args.delta_path);
            std::cout << std::format("Delta count: {}\n", delta.deltas.size());
            std::cout << std::format("Delta elems: {}\n", delta_elems);
            std::cout << std::format("Merged:      {}\n", delta.merged ? "yes" : "no");
        } catch (const std::exception& e) {
            std::cerr << std::format("Warning: could not load delta: {}\n", e.what());
        }
    }

    return 0;
}

int cmd_write(const Args& args) {
    if (args.fact.empty()) {
        std::cerr << "Error: --fact TEXT is required for write\n";
        return 1;
    }
    if (args.delta_path.empty()) {
        std::cerr << "Error: --delta PATH is required for write\n";
        return 1;
    }

    std::cerr << std::format("Loading model from '{}' ...\n", args.model_path);
    sub0llm::nn::GGUFReader reader(args.model_path);
    sub0llm::nn::ModernGPT model = sub0llm::nn::load_gguf_model(reader);

    const auto& vocab = reader.vocab();
    if (vocab.tokens.empty()) {
        std::cerr << "Warning: no vocabulary found in GGUF; using trivial byte tokenizer\n";
    }

    std::cerr << std::format("Tokenizing: '{}'\n", args.fact);
    std::vector<int32_t> tokens;
    if (!vocab.tokens.empty()) {
        tokens = simple_tokenize(args.fact, vocab.tokens);
    } else {
        // Byte-level fallback: clip to vocab_size
        const int32_t vsz = static_cast<int32_t>(model.vocab_size());
        for (unsigned char c : args.fact)
            tokens.push_back(static_cast<int32_t>(c) % vsz);
    }
    std::cerr << std::format("Tokens ({}):", tokens.size());
    for (int32_t t : tokens) std::cerr << " " << t;
    std::cerr << "\n";

    if (tokens.size() < 2) {
        std::cerr << "Warning: text too short for episodic encoding (need >= 2 tokens)\n";
    }

    auto state = sub0llm::nn::make_episodic_state(model);

    sub0llm::nn::EpisodicConfig cfg;
    cfg.learning_rate = args.lr;
    cfg.think_steps   = args.steps;

    std::cerr << std::format("Running episodic_encode (lr={:.4f}, steps={}) ...\n",
                              args.lr, args.steps);
    sub0llm::nn::episodic_encode(model, state, tokens, cfg);

    std::cerr << std::format("Saving delta to '{}' ...\n", args.delta_path);
    sub0llm::nn::save_episodic_state(state, args.delta_path);

    int64_t nonzero = 0;
    for (const auto& d : state.deltas) {
        for (float v : d.data_as<float>())
            if (v != 0.0f) ++nonzero;
    }
    std::cout << std::format("Done. Delta saved to '{}' ({} non-zero elements)\n",
                              args.delta_path, nonzero);
    return 0;
}

int cmd_recall(const Args& args) {
    if (args.query.empty()) {
        std::cerr << "Error: --query TEXT is required for recall\n";
        return 1;
    }
    if (args.delta_path.empty()) {
        std::cerr << "Error: --delta PATH is required for recall\n";
        return 1;
    }
    if (!std::filesystem::exists(args.delta_path)) {
        std::cerr << std::format("Error: delta file '{}' not found\n", args.delta_path);
        return 1;
    }

    std::cerr << std::format("Loading model from '{}' ...\n", args.model_path);
    sub0llm::nn::GGUFReader reader(args.model_path);
    sub0llm::nn::ModernGPT model = sub0llm::nn::load_gguf_model(reader);

    std::cerr << std::format("Loading delta from '{}' ...\n", args.delta_path);
    auto state = sub0llm::nn::load_episodic_state(args.delta_path);

    const auto& vocab = reader.vocab();

    std::cerr << std::format("Tokenizing query: '{}'\n", args.query);
    std::vector<int32_t> tokens;
    if (!vocab.tokens.empty()) {
        tokens = simple_tokenize(args.query, vocab.tokens);
    } else {
        const int32_t vsz = static_cast<int32_t>(model.vocab_size());
        for (unsigned char c : args.query)
            tokens.push_back(static_cast<int32_t>(c) % vsz);
    }
    std::cerr << std::format("Query tokens ({}):", tokens.size());
    for (int32_t t : tokens) std::cerr << " " << t;
    std::cerr << "\n";

    if (tokens.size() < 2) {
        std::cerr << "Warning: query too short for NLL computation (need >= 2 tokens)\n";
        return 0;
    }

    // Baseline NLL (without delta)
    const auto base_losses = sub0llm::nn::comprehension_pass(model, tokens);

    // NLL with delta merged
    state.merge(model);
    const auto delta_losses = sub0llm::nn::comprehension_pass(model, tokens);
    state.unmerge(model);

    float base_mean  = 0.0f, delta_mean = 0.0f;
    for (float v : base_losses)  base_mean  += v;
    for (float v : delta_losses) delta_mean += v;
    if (!base_losses.empty()) {
        base_mean  /= static_cast<float>(base_losses.size());
        delta_mean /= static_cast<float>(delta_losses.size());
    }

    std::cout << std::format("Per-token NLL (query: '{}')\n", args.query);
    std::cout << std::format("{:<6} {:>12} {:>12} {:>10}\n",
                              "pos", "base NLL", "delta NLL", "change");
    std::cout << std::string(44, '-') << "\n";
    for (std::size_t i = 0; i < base_losses.size(); ++i) {
        const float change = delta_losses[i] - base_losses[i];
        std::string tok_str;
        if (!vocab.tokens.empty() && i + 1 < tokens.size())
            tok_str = std::format(" '{}'", vocab.tokens[static_cast<std::size_t>(tokens[i + 1])]);
        std::cout << std::format("{:<6} {:>12.4f} {:>12.4f} {:>+10.4f}{}\n",
                                  i + 1, base_losses[i], delta_losses[i], change, tok_str);
    }
    std::cout << std::string(44, '-') << "\n";
    std::cout << std::format("{:<6} {:>12.4f} {:>12.4f} {:>+10.4f}  (mean)\n",
                              "", base_mean, delta_mean, delta_mean - base_mean);

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);

    try {
        if (args.subcmd == "write")  return cmd_write(args);
        if (args.subcmd == "recall") return cmd_recall(args);
        if (args.subcmd == "info")   return cmd_info(args);
    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }

    return 0;
}
