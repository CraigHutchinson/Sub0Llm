// sub0llm-cli — interactive inference using a trained ModernGPT checkpoint or GGUF model.
//
// Two model sources are supported:
//   --model-dir DIR     checkpoint directory written by ch24_real_training
//                         (config.json, tokenizer/, step_XXXXXXXXX.ckpt), OR a .gguf
//                         file (detected by extension)
//   --model FILE.gguf   GGUF file (tokenizer embedded; Q8 quantize-on-load by default)
//
// GGUF: pass the .gguf file path directly as --model-dir (detected by extension).
//
// Usage:
//   sub0llm-cli --model-dir DIR     [options]   (checkpoint dir or .gguf file)
//   sub0llm-cli --model FILE.gguf   [options]   (GGUF file)
//   sub0llm-cli --model FILE.gguf --prompt "Once upon" --max-tokens 200
//   sub0llm-cli --model FILE.gguf --interactive
//
// Options:
//   --model-dir PATH    model directory, or a .gguf file (detected by extension)
//   --model FILE.gguf   GGUF file (alias; mutually exclusive with --model-dir)
//   --f32               load full f32 weights (default: Q8 quantize-on-load)
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
    std::string model_dir;   // --model-dir: checkpoint directory
    std::string model_file;  // --model:     GGUF file path
    std::string prompt;
    int64_t     max_tokens  = 200;
    float       temperature = 1.0f;
    int64_t     top_k       = 0;
    float       top_p       = 1.0f;
    bool        greedy      = false;
    bool        interactive = false;
    bool        f32         = false;  // --f32: load full f32 weights (default: Q8 int8)
    uint32_t    seed        = 42;
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
  sub0llm-cli --model-dir DIR     [options]   (checkpoint dir or .gguf file)
  sub0llm-cli --model FILE.gguf   [options]   (GGUF file)

Model source (provide exactly one):
  --model-dir PATH    directory with config.json/tokenizer/step_*.ckpt,
                      or a .gguf file (detected by .gguf extension)
  --model FILE.gguf   GGUF model file (tokenizer embedded)
  --f32               load full f32 weights (default: Q8 quantize-on-load)

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
        else if (a == "--model")       cfg.model_file  = next();
        else if (a == "--prompt")      cfg.prompt      = next();
        else if (a == "--max-tokens")  cfg.max_tokens  = std::stoll(next());
        else if (a == "--temperature") cfg.temperature = std::stof(next());
        else if (a == "--top-k")       cfg.top_k       = std::stoll(next());
        else if (a == "--top-p")       cfg.top_p       = std::stof(next());
        else if (a == "--seed")        cfg.seed        = std::stoull(next());
        else if (a == "--greedy")      cfg.greedy      = true;
        else if (a == "--interactive") cfg.interactive = true;
        else if (a == "--f32")         cfg.f32         = true;
        else { std::cerr << "Unknown argument: " << a << "\n"; print_usage(); std::exit(1); }
    }
    if (cfg.model_dir.empty() && cfg.model_file.empty()) {
        std::cerr << "Error: --model-dir or --model is required\n\n";
        print_usage();
        std::exit(1);
    }
    if (!cfg.model_dir.empty() && !cfg.model_file.empty()) {
        std::cerr << "Error: --model-dir and --model are mutually exclusive\n\n";
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

    std::optional<sub0llm::nn::ModernGPT> model_opt;
    std::optional<sub0llm::BPETokenizer>  tok_opt;

    // Resolve the model source: an explicit --model GGUF, or a --model-dir that is
    // either a .gguf file (detected by extension) or a checkpoint directory.
    const std::filesystem::path dir = cfg.model_dir;
    const bool        is_gguf   = !cfg.model_file.empty() || is_gguf_path(dir);
    const std::string gguf_path = !cfg.model_file.empty() ? cfg.model_file : cfg.model_dir;

    try {
        if (is_gguf) {
            // ── GGUF path ──────────────────────────────────────────────────────
            std::cerr << std::format("[info] loading GGUF: {}  ({})\n", gguf_path,
                                     cfg.f32 ? "f32 weights" : "Q8 quantize-on-load");
            sub0llm::nn::GGUFReader reader(gguf_path);
            tok_opt   = sub0llm::BPETokenizer::from_vocab(reader.vocab().tokens,
                                                           reader.vocab().merges);
            // Default to Q8 quantize-on-load: faster generation and ~3.6× less RAM,
            // never materializing f32. Use --f32 for the full-precision path.
            model_opt = cfg.f32 ? sub0llm::nn::load_gguf_model(reader)
                                : sub0llm::nn::load_gguf_model_q8(reader);

            // n_params from the GGUF tensor table (robust whether or not f32 is elided).
            const int64_t n_params = [&] {
                int64_t n = 0;
                for (const auto& [name, ti] : reader.tensors()) n += ti.numel;
                return n;
            }();
            const auto& c = reader.config();
            std::cerr << std::format(
                "GGUF | arch={} V={} D={} heads={}/{} layers={} | {:.2f}M params\n",
                c.arch, c.vocab_size, c.embed_dim, c.n_heads, c.n_kv_heads,
                c.n_layers, static_cast<double>(n_params) / 1e6);
        } else {
            // ── Checkpoint directory path ──────────────────────────────────────
            const auto arch = load_arch(dir);

            const auto tok_dir = dir / "tokenizer";
            if (!std::filesystem::exists(tok_dir))
                throw std::runtime_error(std::format(
                    "tokenizer/ not found in '{}'", dir.string()));
            tok_opt = sub0llm::BPETokenizer::load(tok_dir / "vocab.json",
                                                   tok_dir / "merges.txt");

            model_opt.emplace(arch.vocab_size, arch.embed_dim,
                              arch.n_heads,   arch.n_kv_heads,
                              arch.n_layers,  arch.d_ff,
                              arch.n_mtp_heads, /*seed=*/42);

            const auto ckpt = sub0llm::latest_checkpoint_path(dir.string());
            if (ckpt.empty())
                throw std::runtime_error(std::format(
                    "no checkpoint found in '{}'", dir.string()));

            std::vector<sub0llm::autograd::Variable> param_copies;
            for (auto* p : model_opt->parameters()) param_copies.push_back(*p);
            const int64_t step = sub0llm::load_checkpoint(param_copies, ckpt);
            auto raw = model_opt->parameters();
            for (std::size_t i = 0; i < raw.size(); ++i)
                raw[i]->data() = param_copies[i].data();

            const int64_t n_params = [&] {
                int64_t n = 0;
                for (const auto* p : model_opt->parameters()) n += p->data().numel();
                return n;
            }();
            std::cerr << std::format(
                "Loaded step {} | V={} D={} heads={}/{} layers={} | {:.2f}M params\n",
                step, arch.vocab_size, arch.embed_dim, arch.n_heads, arch.n_kv_heads,
                arch.n_layers, static_cast<double>(n_params) / 1e6);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto& model = *model_opt;
    auto& tok   = *tok_opt;

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
