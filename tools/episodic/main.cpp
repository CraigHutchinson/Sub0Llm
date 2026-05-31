// sub0llm-episodic — write/recall facts into a model's episodic delta
//
// Usage:
//   sub0llm-episodic write  --model <model.gguf> --fact <text>   --delta <out.epis> [opts]
//   sub0llm-episodic recall --model <model.gguf> --query <text>  --delta <in.epis>
//   sub0llm-episodic info   --model <model.gguf>

#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;

// ── Args ──────────────────────────────────────────────────────────────────────

struct Args {
    std::string subcommand;
    std::string model_path;
    std::string delta_path;
    std::string fact;
    std::string query;
    float       lr    = 3e-2f;
    int         steps = 10;
};

static void usage(const char* argv0) {
    std::cerr <<
        "Usage:\n"
        "  " << argv0 << " write  --model <f.gguf> --fact <text>  --delta <out.epis> [--lr F] [--steps N]\n"
        "  " << argv0 << " recall --model <f.gguf> --query <text> --delta <in.epis>\n"
        "  " << argv0 << " info   --model <f.gguf>\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); std::exit(1); }
    Args a;
    a.subcommand = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error(std::format("flag {} requires an argument", flag));
            return argv[i];
        };
        if      (flag == "--model") a.model_path = next();
        else if (flag == "--delta") a.delta_path = next();
        else if (flag == "--fact")  a.fact  = next();
        else if (flag == "--query") a.query = next();
        else if (flag == "--lr")    a.lr    = std::stof(next());
        else if (flag == "--steps") a.steps = std::stoi(next());
        else { std::cerr << "Unknown flag: " << flag << "\n"; std::exit(1); }
    }
    return a;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Character-level tokenization: clamp each byte to [0, vocab_size).
static std::vector<int32_t> char_tokenize(const std::string& text, int64_t vocab_size) {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (auto byte : text)
        ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(byte)) %
                      static_cast<int32_t>(vocab_size));
    return ids;
}

static ModernGPT load_model(const Args& args) {
    if (args.model_path.empty())
        throw std::runtime_error("--model is required");
    if (!args.model_path.ends_with(".gguf"))
        throw std::runtime_error(std::format("Only .gguf files are supported, got: '{}'", args.model_path));
    std::cerr << "[info] loading GGUF: " << args.model_path << "\n";
    return load_gguf_model(args.model_path);
}

static float mean_nll(ModernGPT& model, const std::vector<int32_t>& tokens) {
    auto ls = comprehension_pass(model, tokens);
    return ls.empty() ? 0.0f :
           std::accumulate(ls.begin(), ls.end(), 0.0f) / static_cast<float>(ls.size());
}

// ── Subcommands ───────────────────────────────────────────────────────────────

static int cmd_info(const Args& args) {
    if (args.model_path.empty()) throw std::runtime_error("--model is required");
    GGUFReader reader(args.model_path);
    const GGUFModelConfig cfg = reader.config();
    std::cout << std::format(
        "arch:       {}\nvocab_size: {}\nembed_dim:  {}\n"
        "n_heads:    {}\nn_kv_heads: {}\nn_layers:   {}\n"
        "d_ff:       {}\ncontext:    {}\nrope_base:  {}\n"
        "lm_head:    {}\n",
        cfg.arch, cfg.vocab_size, cfg.embed_dim,
        cfg.n_heads, cfg.n_kv_heads, cfg.n_layers,
        cfg.d_ff, cfg.context_len, cfg.rope_base,
        cfg.has_separate_lm_head ? "separate (output.weight)" : "tied (token_embd.weight)");
    auto model = load_gguf_model(reader);
    std::cout << std::format("n_params:   {}\n", model.parameters().size());
    return 0;
}

static int cmd_write(const Args& args) {
    if (args.fact.empty())       throw std::runtime_error("--fact is required for write");
    if (args.delta_path.empty()) throw std::runtime_error("--delta is required for write");

    auto model = load_model(args);
    const int64_t V = model.parameters()[0]->data().shape()[0];

    EpisodicState state;
    if (std::filesystem::exists(args.delta_path)) {
        std::cerr << "[info] loading existing delta: " << args.delta_path << "\n";
        state = load_episodic_state(args.delta_path);
    } else {
        state = make_episodic_state(model);
    }

    auto tokens = char_tokenize(args.fact, V);
    std::cerr << std::format("[info] encoding {} tokens, lr={:.1e}, steps={}\n",
                             tokens.size(), args.lr, args.steps);

    float nll_before = mean_nll(model, tokens);

    EpisodicConfig cfg;
    cfg.learning_rate      = args.lr;
    cfg.think_steps        = args.steps;
    cfg.surprise_threshold = 0.0f;
    cfg.accumulate         = true;
    episodic_encode(model, state, tokens, cfg);

    // Verify: merge, measure, unmerge
    state.merge(model);
    float nll_after = mean_nll(model, tokens);
    state.unmerge(model);

    save_episodic_state(state, args.delta_path);

    std::cout << std::format("NLL before: {:.4f}\nNLL after:  {:.4f}\nReduction:  {:.1f}%\n"
                             "Delta saved to: {}\n",
                             nll_before, nll_after,
                             100.0f * (1.0f - nll_after / nll_before),
                             args.delta_path);
    return 0;
}

static int cmd_recall(const Args& args) {
    if (args.query.empty())      throw std::runtime_error("--query is required for recall");
    if (args.delta_path.empty()) throw std::runtime_error("--delta is required for recall");

    auto model = load_model(args);
    const int64_t V = model.parameters()[0]->data().shape()[0];

    auto state = load_episodic_state(args.delta_path);
    auto tokens = char_tokenize(args.query, V);

    float nll_base = mean_nll(model, tokens);

    state.merge(model);
    float nll_merged = mean_nll(model, tokens);
    state.unmerge(model);

    std::cout << std::format(
        "Query:     {}\nNLL (base): {:.4f}\nNLL (epis): {:.4f}\nDelta:      {:+.4f}  ({:+.1f}%)\n",
        args.query, nll_base, nll_merged,
        nll_merged - nll_base,
        100.0f * (nll_merged - nll_base) / nll_base);
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        if      (args.subcommand == "info")   return cmd_info(args);
        else if (args.subcommand == "write")  return cmd_write(args);
        else if (args.subcommand == "recall") return cmd_recall(args);
        else {
            std::cerr << "Unknown subcommand: " << args.subcommand << "\n";
            usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}
