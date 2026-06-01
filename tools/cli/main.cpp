// sub0llm-cli — interactive inference using a trained ModernGPT checkpoint or
// a GGUF model file.
//
// Checkpoint model-dir layout (written by ch24 training):
//   config.json          — architecture params
//   tokenizer/           — vocab.json + merges.txt
//   step_XXXXXXXXX.ckpt  — model weights
//
// GGUF: pass the .gguf file path directly as --model-dir (detected by extension).
//
// Usage:
//   sub0llm-cli --model-dir DIR   [options]    (checkpoint directory)
//   sub0llm-cli --model-dir F.gguf [options]   (GGUF model file)
//   sub0llm-cli --model-dir DIR --prompt "Once upon" --max-tokens 200
//   sub0llm-cli --model-dir DIR --interactive
//
// Options:
//   --model-dir DIR     path to model directory or .gguf file (required)
//   --prompt TEXT       initial prompt (default: empty → unconditional)
//   --max-tokens N      max new tokens to generate (default: 200)
//   --temperature F     sampling temperature >0 (default: 1.0)
//   --top-k N           top-k filtering; 0=disabled (default: 0)
//   --top-p F           nucleus sampling; 1.0=disabled (default: 1.0)
//   --greedy            force greedy decoding (overrides temperature/top-k/top-p)
//   --seed N            RNG seed (default: 42)
//   --interactive       read prompts from stdin, one per line

#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliConfig {
    std::string model_dir;
    std::string prompt;
    int64_t     max_tokens  = 200;
    float       temperature = 1.0f;
    int64_t     top_k       = 0;
    float       top_p       = 1.0f;
    bool        greedy      = false;
    bool        interactive = false;
    uint64_t    seed        = 42;
};

struct ModelArch {
    int64_t     vocab_size  = 0;
    int64_t     embed_dim   = 128;
    std::size_t n_heads     = 4;
    std::size_t n_kv_heads  = 4;
    int64_t     n_layers    = 4;
    int64_t     d_ff        = 0;
    int64_t     n_mtp_heads = 0;
};

void print_usage() {
    std::cerr << R"(sub0llm-cli — inference CLI for a trained ModernGPT model

Usage:
  sub0llm-cli --model-dir DIR     [options]   (checkpoint directory)
  sub0llm-cli --model-dir F.gguf  [options]   (GGUF model file)

Required:
  --model-dir PATH    directory with config.json/tokenizer/step_*.ckpt,
                      or a .gguf file (detected by .gguf extension)

Sampling:
  --prompt TEXT       initial prompt (default: empty)
  --max-tokens N      max new tokens (default: 200)
  --temperature F     sampling temperature >0 (default: 1.0)
  --top-k N           top-k; 0=disabled (default: 0)
  --top-p F           nucleus threshold in (0,1]; 1.0=disabled (default: 1.0)
  --greedy            force greedy decoding

Other:
  --interactive       read prompts from stdin line-by-line (Ctrl+D to exit)
  --seed N            RNG seed (default: 42)
  --help              show this message
)";
}

CliConfig parse_args(int argc, char** argv) {
    CliConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { print_usage(); std::exit(0); }
        else if (a == "--model-dir")   cfg.model_dir   = next();
        else if (a == "--prompt")      cfg.prompt      = next();
        else if (a == "--max-tokens")  cfg.max_tokens  = std::stoll(next());
        else if (a == "--temperature") cfg.temperature = std::stof(next());
        else if (a == "--top-k")       cfg.top_k       = std::stoll(next());
        else if (a == "--top-p")       cfg.top_p       = std::stof(next());
        else if (a == "--seed")        cfg.seed        = std::stoull(next());
        else if (a == "--greedy")      cfg.greedy      = true;
        else if (a == "--interactive") cfg.interactive = true;
        else { std::cerr << "Unknown argument: " << a << "\n"; print_usage(); std::exit(1); }
    }
    if (cfg.model_dir.empty()) {
        std::cerr << "Error: --model-dir is required\n\n";
        print_usage();
        std::exit(1);
    }
    return cfg;
}

ModelArch load_arch(const std::filesystem::path& dir) {
    const auto p = dir / "config.json";
    if (!std::filesystem::exists(p))
        throw std::runtime_error(std::format(
            "config.json not found in '{}'\n"
            "Train with ch24_real_training to generate it.", dir.string()));

    std::ifstream f(p);
    const auto j = nlohmann::json::parse(f);

    ModelArch a;
    a.vocab_size  = j.at("vocab_size").get<int64_t>();
    a.embed_dim   = j.at("embed_dim").get<int64_t>();
    a.n_heads     = j.at("n_heads").get<std::size_t>();
    a.n_kv_heads  = j.at("n_kv_heads").get<std::size_t>();
    a.n_layers    = j.at("n_layers").get<int64_t>();
    a.d_ff        = j.value("d_ff", int64_t{0});
    a.n_mtp_heads = j.value("n_mtp_heads", int64_t{0});
    return a;
}

sub0llm::nn::SamplingConfig make_sampling(const CliConfig& cfg) {
    using sub0llm::nn::SamplingConfig;
    using sub0llm::nn::SamplingMode;

    SamplingConfig sc;
    sc.temperature = cfg.temperature;
    if (cfg.greedy) {
        sc.mode = SamplingMode::Greedy;
    } else if (cfg.top_k > 0) {
        sc.mode  = SamplingMode::TopK;
        sc.top_k = cfg.top_k;
    } else if (cfg.top_p < 1.0f) {
        sc.mode  = SamplingMode::TopP;
        sc.top_p = cfg.top_p;
    } else if (cfg.temperature != 1.0f) {
        sc.mode = SamplingMode::Temperature;
    } else {
        sc.mode = SamplingMode::Greedy;
    }
    return sc;
}

std::string run_generation(sub0llm::nn::ModernGPT&  model,
                            sub0llm::BPETokenizer&   tok,
                            const std::string&       prompt,
                            const CliConfig&         cfg,
                            std::mt19937&            rng) {
    auto ids = tok.encode(prompt);
    if (ids.empty()) ids.push_back(0);

    const auto sc     = make_sampling(cfg);
    const auto all    = sub0llm::nn::generate(model, ids, cfg.max_tokens, sc, rng);
    const auto gen_begin = all.begin() + static_cast<std::ptrdiff_t>(ids.size());
    const std::vector<int32_t> generated(gen_begin, all.end());
    return tok.decode(generated);
}

// Returns true when path points to a GGUF file (detected by .gguf extension).
static bool is_gguf_path(const std::filesystem::path& p) {
    return p.extension() == ".gguf";
}

} // namespace

int main(int argc, char** argv) {
    const auto cfg = parse_args(argc, argv);
    const std::filesystem::path dir = cfg.model_dir;

    std::optional<sub0llm::nn::ModernGPT> model_storage;
    std::optional<sub0llm::BPETokenizer>  tok_storage;

    if (is_gguf_path(dir)) {
        // ── GGUF model file ───────────────────────────────────────────────────
        try {
            sub0llm::nn::GGUFReader reader(dir.string());
            const auto& gv = reader.vocab();
            tok_storage.emplace(
                sub0llm::BPETokenizer::from_vocab(gv.tokens, gv.merges));
            model_storage.emplace(sub0llm::nn::load_gguf_model(reader));

            const auto& cfg_gguf = reader.config();
            const int64_t n_params = [&] {
                int64_t n = 0;
                for (const auto* p : model_storage->parameters())
                    n += p->data().numel();
                return n;
            }();
            std::cerr << std::format(
                "GGUF | arch={} V={} D={} heads={}/{} layers={} | {:.2f}M params | {}\n",
                cfg_gguf.arch, cfg_gguf.vocab_size, cfg_gguf.embed_dim,
                cfg_gguf.n_heads, cfg_gguf.n_kv_heads, cfg_gguf.n_layers,
                static_cast<double>(n_params) / 1e6, dir.string());
        } catch (const std::exception& e) {
            std::cerr << "Error loading GGUF: " << e.what() << "\n";
            return 1;
        }
    } else {
        // ── Checkpoint directory ──────────────────────────────────────────────
        ModelArch arch;
        try {
            arch = load_arch(dir);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        try {
            const auto tok_dir = dir / "tokenizer";
            if (!std::filesystem::exists(tok_dir))
                throw std::runtime_error(std::format(
                    "tokenizer/ not found in '{}'", dir.string()));
            tok_storage.emplace(
                sub0llm::BPETokenizer::load(tok_dir / "vocab.json",
                                             tok_dir / "merges.txt"));
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        model_storage.emplace(arch.vocab_size, arch.embed_dim,
                              arch.n_heads,   arch.n_kv_heads,
                              arch.n_layers,  arch.d_ff,
                              arch.n_mtp_heads, /*seed=*/42);

        const auto ckpt = sub0llm::latest_checkpoint_path(dir.string());
        if (ckpt.empty()) {
            std::cerr << "Error: no checkpoint found in '" << dir.string() << "'\n";
            return 1;
        }

        std::vector<sub0llm::autograd::Variable> param_copies;
        for (auto* p : model_storage->parameters()) param_copies.push_back(*p);

        int64_t step = 0;
        try {
            step = sub0llm::load_checkpoint(param_copies, ckpt);
            auto raw = model_storage->parameters();
            for (std::size_t i = 0; i < raw.size(); ++i)
                raw[i]->data() = param_copies[i].data();
        } catch (const std::exception& e) {
            std::cerr << "Error loading checkpoint: " << e.what() << "\n";
            return 1;
        }

        const int64_t n_params = [&] {
            int64_t n = 0;
            for (const auto* p : model_storage->parameters()) n += p->data().numel();
            return n;
        }();
        std::cerr << std::format(
            "Loaded step {} | V={} D={} heads={}/{} layers={} | {:.2f}M params\n",
            step, arch.vocab_size, arch.embed_dim, arch.n_heads, arch.n_kv_heads,
            arch.n_layers, static_cast<double>(n_params) / 1e6);
    }

    auto& model = *model_storage;
    auto& tok   = *tok_storage;

    std::mt19937 rng(cfg.seed);

    if (cfg.interactive) {
        std::cerr << "Interactive mode — type a prompt and press Enter (Ctrl+D to exit)\n";
        std::string line;
        while (true) {
            std::cerr << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            try {
                std::cout << run_generation(model, tok, line, cfg, rng) << "\n" << std::flush;
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
        std::cerr << "\n";
    } else {
        try {
            std::cout << run_generation(model, tok, cfg.prompt, cfg, rng) << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}
