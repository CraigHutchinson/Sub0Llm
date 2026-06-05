// sub0llm-server — OpenAI-compatible HTTP inference server.
//
// Two model sources are supported:
//   --model-dir DIR     checkpoint directory (config.json, tokenizer/, step_*.ckpt)
//   --model FILE.gguf   GGUF file (tokenizer embedded, weights dequantised on load)
//
// Endpoints:
//   GET  /health                  → {"status":"ok","model":"sub0llm"}
//   GET  /v1/models               → list of available models
//   POST /v1/completions          → text completion (OpenAI schema)
//   POST /v1/chat/completions     → chat completion (messages → concatenated prompt)
//
// Usage:
//   sub0llm-server --model-dir DIR   [--host HOST] [--port PORT]
//   sub0llm-server --model FILE.gguf [--host HOST] [--port PORT]
//
// Options:
//   --model-dir DIR   directory with config.json, tokenizer/, step_*.ckpt
//   --model FILE.gguf GGUF file (mutually exclusive with --model-dir)
//   --host HOST       bind address (default: 0.0.0.0)
//   --port PORT       listen port  (default: 8080)
//   --threads N       worker threads for concurrent requests (default: 4)
//   --seed N          base RNG seed; each request uses seed+request_count (default: 42)

#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/nn/gguf_loader.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ServerConfig {
    std::string model_dir;   // --model-dir: checkpoint directory
    std::string model_file;  // --model:     GGUF file path
    std::string host     = "0.0.0.0";
    int         port     = 8080;
    int         threads  = 4;
    uint64_t    seed     = 42;
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
    std::cerr << R"(sub0llm-server — OpenAI-compatible inference server

Usage:
  sub0llm-server --model-dir DIR   [options]   (checkpoint directory)
  sub0llm-server --model FILE.gguf [options]   (GGUF file)

Model source (provide exactly one):
  --model-dir DIR   directory with config.json, tokenizer/, step_*.ckpt
  --model FILE.gguf GGUF model file (tokenizer embedded)

Options:
  --host HOST       bind address (default: 0.0.0.0)
  --port PORT       listen port  (default: 8080)
  --threads N       concurrent request handlers (default: 4)
  --seed N          base RNG seed (default: 42)
  --help            show this message

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/completions
  POST /v1/chat/completions
)";
}

ServerConfig parse_args(int argc, char** argv) {
    ServerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::cerr << a << " requires an argument\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { print_usage(); std::exit(0); }
        else if (a == "--model-dir") cfg.model_dir  = next();
        else if (a == "--model")     cfg.model_file = next();
        else if (a == "--host")      cfg.host       = next();
        else if (a == "--port")      cfg.port       = std::stoi(next());
        else if (a == "--threads")   cfg.threads    = std::stoi(next());
        else if (a == "--seed")      cfg.seed       = std::stoull(next());
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
            "config.json not found in '{}' — train with ch24_real_training first",
            dir.string()));

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

// ── InferenceEngine ───────────────────────────────────────────────────────────
// Thread-safe wrapper: ModernGPT::forward() uses mutable caches (RoPE, mask)
// so all inference must be serialised through the mutex.

class InferenceEngine {
public:
    // model_dir: checkpoint directory (config.json + tokenizer/ + *.ckpt)
    // gguf_file: GGUF file path — provide exactly one; leave the other empty
    InferenceEngine(const std::string& model_dir,
                    const std::string& gguf_file,
                    uint64_t seed)
        : seed_(seed) {
        if (!gguf_file.empty()) {
            // ── GGUF path ──────────────────────────────────────────────────
            std::cerr << "[info] loading GGUF: " << gguf_file << "  (Q8 quantize-on-load)\n";
            sub0llm::nn::GGUFReader reader(gguf_file);
            tok_   = std::make_unique<sub0llm::BPETokenizer>(
                sub0llm::BPETokenizer::from_vocab(reader.vocab().tokens,
                                                   reader.vocab().merges));
            // Q8 quantize-on-load by default: faster + ~3.6× less RAM, no f32 peak.
            model_ = std::make_unique<sub0llm::nn::ModernGPT>(
                sub0llm::nn::load_gguf_model_q8(reader));

            const auto& c = reader.config();
            vocab_size_ = c.vocab_size;
            embed_dim_  = c.embed_dim;
            n_layers_   = c.n_layers;
            // From the GGUF tensor table — parameters() is elided under Q8 load.
            for (const auto& [nm, ti] : reader.tensors()) n_params_ += ti.numel;
        } else {
            // ── Checkpoint directory path ──────────────────────────────────
            const std::filesystem::path dir = model_dir;
            const auto arch = load_arch(dir);

            model_ = std::make_unique<sub0llm::nn::ModernGPT>(
                arch.vocab_size, arch.embed_dim,
                arch.n_heads,   arch.n_kv_heads,
                arch.n_layers,  arch.d_ff,
                arch.n_mtp_heads, /*seed=*/42);

            const auto tok_dir = dir / "tokenizer";
            if (!std::filesystem::exists(tok_dir))
                throw std::runtime_error(std::format(
                    "tokenizer/ not found in '{}'", dir.string()));
            tok_ = std::make_unique<sub0llm::BPETokenizer>(
                sub0llm::BPETokenizer::load(tok_dir / "vocab.json",
                                             tok_dir / "merges.txt"));

            const auto ckpt = sub0llm::latest_checkpoint_path(dir.string());
            if (ckpt.empty())
                throw std::runtime_error(
                    std::format("No checkpoint found in '{}'", dir.string()));

            std::vector<sub0llm::autograd::Variable> copies;
            for (auto* p : model_->parameters()) copies.push_back(*p);
            step_ = sub0llm::load_checkpoint(copies, ckpt);
            auto raw = model_->parameters();
            for (std::size_t i = 0; i < raw.size(); ++i)
                raw[i]->data() = copies[i].data();

            vocab_size_ = arch.vocab_size;
            embed_dim_  = arch.embed_dim;
            n_layers_   = arch.n_layers;

            std::cerr << std::format(
                "Loaded step {} | V={} D={} layers={} | ckpt: {}\n",
                step_, vocab_size_, embed_dim_, n_layers_, ckpt);
        }

        if (n_params_ == 0)   // checkpoint path: count live parameters
            for (const auto* p : model_->parameters()) n_params_ += p->data().numel();
        std::cerr << std::format(
            "V={} D={} layers={} | {:.2f}M params\n",
            vocab_size_, embed_dim_, n_layers_,
            static_cast<double>(n_params_) / 1e6);
    }

    struct CompletionRequest {
        std::string prompt;
        int64_t     max_tokens  = 100;
        float       temperature = 1.0f;
        int64_t     top_k       = 0;
        float       top_p       = 1.0f;
        bool        greedy      = false;
    };

    struct CompletionResult {
        std::string text;
        int         prompt_tokens     = 0;
        int         completion_tokens = 0;
    };

    CompletionResult complete(const CompletionRequest& req) {
        using sub0llm::nn::SamplingConfig;
        using sub0llm::nn::SamplingMode;

        SamplingConfig sc;
        sc.temperature = req.temperature;
        if (req.greedy || (req.top_k == 0 && req.top_p >= 1.0f && req.temperature == 1.0f)) {
            sc.mode = SamplingMode::Greedy;
        } else if (req.top_k > 0) {
            sc.mode  = SamplingMode::TopK;
            sc.top_k = req.top_k;
        } else if (req.top_p < 1.0f) {
            sc.mode  = SamplingMode::TopP;
            sc.top_p = req.top_p;
        } else {
            sc.mode = SamplingMode::Temperature;
        }

        auto prompt_ids = tok_->encode(req.prompt);
        if (prompt_ids.empty()) prompt_ids.push_back(0);

        const uint64_t req_seed = seed_ + req_count_.fetch_add(1, std::memory_order_relaxed);
        std::mt19937 rng(req_seed);

        std::vector<int32_t> all;
        {
            std::lock_guard lock(mu_);
            all = sub0llm::nn::generate(*model_, prompt_ids, req.max_tokens, sc, rng);
        }

        const auto gen_begin = all.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size());
        const std::vector<int32_t> generated(gen_begin, all.end());

        CompletionResult r;
        r.text              = tok_->decode(generated);
        r.prompt_tokens     = static_cast<int>(prompt_ids.size());
        r.completion_tokens = static_cast<int>(generated.size());
        return r;
    }

    int64_t step()       const noexcept { return step_; }
    int64_t vocab_size() const noexcept { return vocab_size_; }
    int64_t n_params()   const noexcept { return n_params_; }

private:
    std::unique_ptr<sub0llm::nn::ModernGPT> model_;
    std::unique_ptr<sub0llm::BPETokenizer>  tok_;
    std::mutex                               mu_;
    std::atomic<uint64_t>                   req_count_{0};
    uint64_t seed_;
    int64_t  step_       = 0;
    int64_t  vocab_size_ = 0;
    int64_t  embed_dim_  = 0;
    int64_t  n_layers_   = 0;
    int64_t  n_params_   = 0;
};

// ── JSON helpers ──────────────────────────────────────────────────────────────

nlohmann::json error_body(int code, const std::string& msg) {
    return {{"error", {{"code", code}, {"message", msg}, {"type", "invalid_request_error"}}}};
}

std::string unix_time_str() {
    return std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

// ── Route handlers ────────────────────────────────────────────────────────────

void handle_health(const httplib::Request&, httplib::Response& res,
                   const InferenceEngine& eng) {
    const nlohmann::json body = {
        {"status",  "ok"},
        {"model",   "sub0llm"},
        {"step",    eng.step()},
        {"n_params", eng.n_params()},
    };
    res.set_content(body.dump(), "application/json");
}

void handle_models(const httplib::Request&, httplib::Response& res,
                   const InferenceEngine& eng) {
    const nlohmann::json body = {
        {"object", "list"},
        {"data",   nlohmann::json::array({
            {{"id",      "sub0llm"},
             {"object",  "model"},
             {"created", unix_time_str()},
             {"owned_by","sub0llm"},
             {"n_params", eng.n_params()}}
        })}
    };
    res.set_content(body.dump(), "application/json");
}

void handle_completions(const httplib::Request& req, httplib::Response& res,
                         InferenceEngine& eng) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content(error_body(400, "Invalid JSON body").dump(), "application/json");
        return;
    }

    InferenceEngine::CompletionRequest cr;
    try {
        cr.prompt      = j.value("prompt",      std::string{});
        cr.max_tokens  = j.value("max_tokens",  int64_t{100});
        cr.temperature = j.value("temperature", 1.0f);
        cr.top_k       = j.value("top_k",       int64_t{0});
        cr.top_p       = j.value("top_p",       1.0f);
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(error_body(400, e.what()).dump(), "application/json");
        return;
    }

    InferenceEngine::CompletionResult result;
    try {
        result = eng.complete(cr);
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(error_body(500, e.what()).dump(), "application/json");
        return;
    }

    const nlohmann::json body = {
        {"id",      std::format("cmpl-{}", unix_time_str())},
        {"object",  "text_completion"},
        {"created", unix_time_str()},
        {"model",   "sub0llm"},
        {"choices", nlohmann::json::array({
            {{"text",          result.text},
             {"index",         0},
             {"finish_reason", "length"}}
        })},
        {"usage", {
            {"prompt_tokens",     result.prompt_tokens},
            {"completion_tokens", result.completion_tokens},
            {"total_tokens",      result.prompt_tokens + result.completion_tokens}
        }}
    };
    res.set_content(body.dump(), "application/json");
}

void handle_chat_completions(const httplib::Request& req, httplib::Response& res,
                              InferenceEngine& eng) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content(error_body(400, "Invalid JSON body").dump(), "application/json");
        return;
    }

    // Concatenate messages into a single prompt string
    std::string prompt;
    try {
        for (const auto& msg : j.at("messages")) {
            const std::string role    = msg.value("role",    "user");
            const std::string content = msg.value("content", "");
            prompt += role + ": " + content + "\n";
        }
        prompt += "assistant: ";
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(error_body(400, std::format("messages: {}", e.what())).dump(),
                        "application/json");
        return;
    }

    InferenceEngine::CompletionRequest cr;
    cr.prompt      = prompt;
    cr.max_tokens  = j.value("max_tokens",  int64_t{100});
    cr.temperature = j.value("temperature", 1.0f);
    cr.top_k       = j.value("top_k",       int64_t{0});
    cr.top_p       = j.value("top_p",       1.0f);

    InferenceEngine::CompletionResult result;
    try {
        result = eng.complete(cr);
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(error_body(500, e.what()).dump(), "application/json");
        return;
    }

    const nlohmann::json body = {
        {"id",      std::format("chatcmpl-{}", unix_time_str())},
        {"object",  "chat.completion"},
        {"created", unix_time_str()},
        {"model",   "sub0llm"},
        {"choices", nlohmann::json::array({
            {{"message",       {{"role", "assistant"}, {"content", result.text}}},
             {"index",         0},
             {"finish_reason", "length"}}
        })},
        {"usage", {
            {"prompt_tokens",     result.prompt_tokens},
            {"completion_tokens", result.completion_tokens},
            {"total_tokens",      result.prompt_tokens + result.completion_tokens}
        }}
    };
    res.set_content(body.dump(), "application/json");
}

} // namespace

int main(int argc, char** argv) {
    const auto cfg = parse_args(argc, argv);

    // Load model
    InferenceEngine engine(cfg.model_dir, cfg.model_file, cfg.seed);

    // Build server
    httplib::Server svr;
    svr.new_task_queue = [&cfg] {
        return new httplib::ThreadPool(static_cast<std::size_t>(cfg.threads));
    };

    svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res, engine);
    });

    svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res, engine);
    });

    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        handle_completions(req, res, engine);
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res, engine);
    });

    // CORS for browser clients / curl --json
    svr.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::cerr << std::format("sub0llm-server listening on {}:{}\n", cfg.host, cfg.port);
    std::cerr << "  POST /v1/completions\n";
    std::cerr << "  POST /v1/chat/completions\n";
    std::cerr << "  GET  /health\n";
    std::cerr << "  GET  /v1/models\n";

    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << std::format("Failed to bind {}:{}\n", cfg.host, cfg.port);
        return 1;
    }

    return 0;
}
