// Chapter 24: Real-World Pretraining — Data, Resources, and Full Iterative Training
//
// Brings together every library component built across Chapters 1–23 into a
// practical guide to actually pretraining a language model.
//
//   §24.1  Data Landscape       — open datasets, synthetic data (Phi approach)
//   §24.2  Resource Estimation  — Chinchilla scaling, FLOPs, GPU-hours, memory
//   §24.3  Data Pipeline Demo   — TextCorpus streaming BPE-tokenised windows
//   §24.4  Synthetic Data via LLM API — Ollama HTTP client, graceful fallback
//   §24.5  Training Approaches  — decision tree comparing strategies
//   §24.6  Full Iterative Training — end-to-end loop with checkpoint save/resume
//
// Usage:
//   ./ch24_real_training [--phase <landscape|estimate|pipeline|synth|approaches|train|all>]
//                        [--ckpt-dir <dir>]
//                        [--steps <N>]

#include "sub0llm/data/text_corpus.hpp"
#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

// Cross-platform socket support for the minimal Ollama HTTP client
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
   using SocketFd = SOCKET;
   static constexpr SocketFd kBadSocket = INVALID_SOCKET;
   static inline void close_socket(SocketFd s) { ::closesocket(s); }
   static inline bool socket_valid(SocketFd s) { return s != INVALID_SOCKET; }
   // Initialise Winsock once at process start
   namespace { struct WsaGuard {
       WsaGuard()  { WSADATA d{}; ::WSAStartup(MAKEWORD(2,2), &d); }
       ~WsaGuard() { ::WSACleanup(); }
   } s_wsa; }
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
   using SocketFd = int;
   static constexpr SocketFd kBadSocket = -1;
   static inline void close_socket(SocketFd s) { ::close(s); }
   static inline bool socket_valid(SocketFd s) { return s >= 0; }
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── §24.1  Data Landscape ─────────────────────────────────────────────────────

static void section_landscape()
{
    std::cout << "\n=== §24.1  The Data Landscape ===\n\n";

    // Open pretraining datasets
    std::cout << "  Major open pretraining datasets:\n\n";
    std::cout << std::format("  {:<28} {:<20} {:<14} {}\n",
        "Dataset", "Organisation", "Scale", "Description");
    std::cout << "  " << std::string(88, '-') << "\n";

    struct DatasetRow {
        std::string_view name;
        std::string_view org;
        std::string_view scale;
        std::string_view desc;
    };
    const DatasetRow rows[] = {
        {"FineWeb-Edu",       "HuggingFaceFW",   "1.3T tokens",  "High-quality educational web text"},
        {"Dolma v1.7",        "Allen AI",         "3T tokens",    "Web + code + books + papers"},
        {"The Pile",          "EleutherAI",       "825 GB",       "22 domains: PubMed, GitHub, Wikipedia"},
        {"RedPajama-v2",      "Together AI",      "20T filtered", "84 CommonCrawl snapshots"},
        {"OpenWebText2",      "EleutherAI",       "65 GB",        "Reddit-upvoted web pages"},
        {"C4",                "Google",           "750 GB",       "Cleaned CommonCrawl"},
        {"The Stack v2",      "BigCode",          "600+ langs",   "Source code in many languages"},
        {"OpenWebMath",       "EleutherAI",       "14.7B tokens", "Mathematical web text"},
    };
    for (const auto& r : rows)
        std::cout << std::format("  {:<28} {:<20} {:<14} {}\n",
            r.name, r.org, r.scale, r.desc);

    std::cout << "\n  What major open models trained on:\n\n";
    std::cout << std::format("  {:<18} {:<22} {}\n", "Model", "Scale", "Data strategy");
    std::cout << "  " << std::string(80, '-') << "\n";

    struct ModelRow { std::string_view name, scale, strategy; };
    const ModelRow mrows[] = {
        {"Llama 3",     "15T tokens",       "Web, code, math (FineWeb-Edu + custom)"},
        {"Gemma 2/3",   "undisclosed",      "Web + code + math, proprietary CC processing"},
        {"Phi-3/4",     "3.3T tokens",      "Heavy synthetic \"textbook quality\" (GPT-4)"},
        {"Qwen 2.5",    "18T tokens",       "Multilingual web + code + math"},
    };
    for (const auto& r : mrows)
        std::cout << std::format("  {:<18} {:<22} {}\n", r.name, r.scale, r.strategy);

    std::cout << "\n  Key insight — the Phi approach:\n"
              << "    Generating synthetic training data using a powerful LLM is the most\n"
              << "    accessible path for small-model training without a data center.\n"
              << "    Microsoft Phi-3/4 achieved GPT-3.5-class performance on benchmarks\n"
              << "    at 3.8B params by training on GPT-4-generated \"textbook\" data.\n"
              << "    For sub0llm's demo scale (~25M params) this is the recommended path.\n";
}

// ── §24.2  Resource Estimation ────────────────────────────────────────────────

struct TrainingEstimate {
    int64_t n_params;
    int64_t chinchilla_tokens;  // 20 × n_params
    double  total_flops;        // 6 × n_params × n_tokens
    double  gpu_hours_a100;     // total_flops / (160e12 * 3600)
    double  gpu_hours_h100;     // total_flops / (500e12 * 3600)
    double  peak_memory_gb;     // ~16 bytes/param (param + grad + 2 Adam moments)
    double  dataset_gb_tokens;  // ~2 bytes/token
};

[[nodiscard]] static TrainingEstimate estimate_resources(int64_t n_params, int64_t n_tokens)
{
    TrainingEstimate e{};
    e.n_params           = n_params;
    e.chinchilla_tokens  = 20 * n_params;
    e.total_flops        = 6.0 * static_cast<double>(n_params) * static_cast<double>(n_tokens);
    e.gpu_hours_a100     = e.total_flops / (160e12 * 3600.0);
    e.gpu_hours_h100     = e.total_flops / (500e12 * 3600.0);
    e.peak_memory_gb     = 16.0 * static_cast<double>(n_params) / 1e9;
    e.dataset_gb_tokens  = 2.0  * static_cast<double>(n_tokens) / 1e9;
    return e;
}

static void section_estimate()
{
    std::cout << "\n=== §24.2  Resource Estimation (Chinchilla Scaling) ===\n\n";
    std::cout << "  Chinchilla law: optimal tokens = 20 × n_params\n"
              << "  Training FLOPs ≈ 6 × N × D   (N=params, D=tokens)\n"
              << "  Peak Adam memory ≈ 16 bytes/param (param+grad+2 moments)\n\n";

    struct ModelSpec {
        std::string_view name;
        int64_t n_params;
    };
    const ModelSpec specs[] = {
        {"demo (~25M)",    25'000'000LL},
        {"small (~85M)",   85'000'000LL},
        {"medium (~340M)", 340'000'000LL},
        {"large (~760M)",  760'000'000LL},
        {"gpt2-xl (~1.5B)",1'500'000'000LL},
        {"llama-7b (~7B)", 7'000'000'000LL},
    };

    std::cout << std::format("  {:<20} {:>12} {:>16} {:>18} {:>18} {:>12}\n",
        "Model", "Params", "Opt. Tokens", "Days A100×1", "Days H100×8", "Mem (GB)");
    std::cout << "  " << std::string(100, '-') << "\n";

    for (const auto& s : specs) {
        auto e  = estimate_resources(s.n_params, 20 * s.n_params);
        double days_a100   = e.gpu_hours_a100 / 24.0;
        double days_h100x8 = e.gpu_hours_h100 / (24.0 * 8.0);

        // Format tokens in human-readable form
        std::string tok_str;
        const double toks = static_cast<double>(e.chinchilla_tokens);
        if (toks >= 1e12)       tok_str = std::format("{:.1f}T", toks / 1e12);
        else if (toks >= 1e9)   tok_str = std::format("{:.1f}B", toks / 1e9);
        else if (toks >= 1e6)   tok_str = std::format("{:.1f}M", toks / 1e6);
        else                    tok_str = std::format("{:.0f}K", toks / 1e3);

        std::cout << std::format("  {:<20} {:>12} {:>16} {:>18.2f} {:>18.3f} {:>12.2f}\n",
            s.name,
            [&]{ std::string ps; const double p = static_cast<double>(s.n_params);
                 if (p>=1e9) ps=std::format("{:.1f}B",p/1e9);
                 else        ps=std::format("{:.0f}M",p/1e6);
                 return ps; }(),
            tok_str,
            days_a100,
            days_h100x8,
            e.peak_memory_gb);
    }

    std::cout << "\n  Key takeaways:\n"
              << "    • The 25M demo model reaches Chinchilla-optimal in minutes on one GPU.\n"
              << "    • A 340M model needs ~500M tokens and ~2 days on a single A100.\n"
              << "    • 7B models require thousands of GPU-days — a compute cluster.\n"
              << "    • For sub0llm: start with the demo scale, then scale as data allows.\n";
}

// ── §24.3  Data Pipeline Demo ─────────────────────────────────────────────────

static void section_pipeline()
{
    std::cout << "\n=== §24.3  Data Pipeline Demo ===\n\n";

    // Small in-memory corpus
    const std::vector<std::string> texts = {
        "The quick brown fox jumps over the lazy dog.",
        "Language models learn by predicting the next token in a sequence.",
        "Pretraining on diverse text teaches a model the structure of language.",
        "Tokenization splits raw text into subword units called tokens.",
        "The Transformer architecture uses attention to model long-range dependencies.",
        "Gradient descent updates model parameters to minimize the training loss.",
    };

    constexpr int64_t seq_len = 16;

    auto tok = BPETokenizer::train(texts, /*vocab_size=*/120);

    TextCorpus corpus(texts, tok, seq_len, /*seed=*/99);

    std::cout << std::format("  Corpus: {} tokens, {} samples (seq_len={})\n",
        corpus.total_tokens(), corpus.n_samples(), seq_len);

    Tensor ids, tgt;
    int sample_num = 0;
    while (corpus.next_sample(ids, tgt)) {
        ++sample_num;
        if (sample_num <= 4) {
            std::cout << std::format("  Sample {:2d}: ids[0..3]=[", sample_num);
            const auto sp = ids.data_as<int32_t>();
            for (int64_t i = 0; i < std::min(static_cast<int64_t>(4), seq_len); ++i)
                std::cout << sp[static_cast<std::size_t>(i)] << (i < 3 ? "," : "");
            std::cout << std::format(", ...]  tokens_consumed={}\n",
                corpus.tokens_consumed());
        }
    }
    std::cout << std::format("  Exhausted after {} samples.\n", sample_num);

    // Demonstrate reset
    std::cout << "\n  Reset — streaming again from start\n";
    corpus.reset(/*seed=*/42);
    int after_reset = 0;
    while (corpus.next_sample(ids, tgt)) ++after_reset;
    std::cout << std::format("  Streamed {} samples after reset.\n", after_reset);

    // Verify tgt is shifted ids (for consecutive non-shuffled windows)
    corpus.reset(/*seed=*/0);
    Tensor ids0, tgt0;
    if (corpus.next_sample(ids0, tgt0) && seq_len >= 2) {
        const auto id_sp  = ids0.data_as<int32_t>();
        const auto tgt_sp = tgt0.data_as<int32_t>();
        std::cout << std::format("  Sanity check: ids[0]={} tgt[0]={}  (tgt[i] is the next token)\n",
            id_sp[0], tgt_sp[0]);
    }

    std::cout << "\n  JSONL support: from_file() recognises lines starting with '{'\n"
              << "    and extracts the \"text\" field — ready for FineWeb-Edu, Dolma, etc.\n";
}

// ── §24.4  Synthetic Data via LLM API ─────────────────────────────────────────

// Minimal HTTP POST to localhost:port/path. Cross-platform (Windows + POSIX).
// Returns response body on success, empty string on error/timeout.
static std::string http_post_local(uint16_t port, std::string_view path,
                                    std::string_view json_body)
{
    const SocketFd sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (!socket_valid(sock)) return "";

    // 3-second send/recv timeout (platform-specific option type)
#ifdef _WIN32
    const DWORD timeout_ms = 3000;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
    struct timeval tv{};
    tv.tv_sec = 3;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");

    if (::connect(sock, reinterpret_cast<const struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close_socket(sock);
        return "";
    }

    // Build HTTP request
    std::string req = std::format(
        "POST {} HTTP/1.0\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: {}\r\n"
        "\r\n"
        "{}",
        path, json_body.size(), json_body);

    if (::send(sock, req.data(), static_cast<int>(req.size()), 0) < 0) {
        close_socket(sock);
        return "";
    }

    // Receive full response
    std::string response;
    char buf[4096];
    int n;
    while ((n = ::recv(sock, buf, static_cast<int>(sizeof(buf)), 0)) > 0)
        response.append(buf, static_cast<std::size_t>(n));
    close_socket(sock);

    if (response.empty()) return "";

    // Strip HTTP headers: find blank line separating headers from body
    const auto pos = response.find("\r\n\r\n");
    if (pos == std::string::npos) return "";
    return response.substr(pos + 4);
}

static std::vector<std::string> generate_synthetic_corpus(int n_samples)
{
    // 5 prompt templates
    const std::vector<std::string> prompts = {
        "Write a short factual paragraph about how neural networks learn from data.",
        "Explain in simple terms what a language model is and how it generates text.",
        "Describe the concept of gradient descent in one short paragraph.",
        "Write a brief explanation of what tokenization means in natural language processing.",
        "Explain why pretraining on large text corpora helps language models understand context.",
    };

    std::vector<std::string> results;
    bool ollama_ok = false;

    std::cout << "  Attempting Ollama at localhost:11434...\n";

    for (int i = 0; i < n_samples && i < static_cast<int>(prompts.size()); ++i) {
        nlohmann::json req;
        req["model"]  = "llama3";
        req["prompt"] = prompts[static_cast<std::size_t>(i)];
        req["stream"] = false;

        const std::string body = http_post_local(11434, "/api/generate", req.dump());
        if (body.empty()) break;

        try {
            auto j    = nlohmann::json::parse(body);
            auto text = j.at("response").get<std::string>();
            if (!text.empty()) {
                results.push_back(std::move(text));
                ollama_ok = true;
            }
        } catch (...) {
            break;
        }
    }

    if (ollama_ok) {
        std::cout << "  [OK] Ollama responded — using generated samples.\n";
    } else {
        std::cout << "  [NOT AVAILABLE] Ollama not running — using fallback corpus.\n";
    }

    if (results.empty()) {
        // Hardcoded fallback corpus (~20 diverse paragraphs)
        results = {
            "Artificial intelligence is the simulation of human intelligence by computer systems.",
            "Machine learning algorithms improve automatically through experience and exposure to data.",
            "Deep learning uses layered neural networks to model complex patterns in data.",
            "Natural language processing enables computers to understand and generate human language.",
            "The Transformer architecture revolutionised NLP with its self-attention mechanism.",
            "Tokenization converts raw text into discrete units that models can process.",
            "Byte-pair encoding is a subword tokenization algorithm widely used in modern LLMs.",
            "Gradient descent is an optimization algorithm that minimises loss by updating weights.",
            "Backpropagation computes gradients efficiently using the chain rule of calculus.",
            "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
            "The Chinchilla scaling law says optimal tokens should be about 20x the parameter count.",
            "Pretraining on large corpora gives models broad language understanding.",
            "Fine-tuning adapts a pretrained model to a specific downstream task.",
            "RLHF aligns language models with human preferences through reinforcement learning.",
            "Knowledge distillation transfers capabilities from a large teacher to a small student.",
            "The softmax function converts logits into a probability distribution over the vocabulary.",
            "Cross-entropy loss measures the divergence between predicted and true token distributions.",
            "Learning rate warmup helps stabilise training in the early gradient steps.",
            "Weight tying between token embeddings and the LM head reduces parameters.",
            "Rotary position embeddings encode token positions without adding learned parameters.",
            "RMSNorm normalises activations using only the root-mean-square, without mean subtraction.",
            "SwiGLU feedforward networks outperform standard GELU FFNs at the same compute budget.",
        };
    }

    // Fill remaining samples from fallback if Ollama produced fewer than requested
    while (static_cast<int>(results.size()) < n_samples &&
           results.size() < 22) {
        results.push_back(results[results.size() % 22]);
    }
    if (static_cast<int>(results.size()) > n_samples)
        results.resize(static_cast<std::size_t>(n_samples));

    return results;
}

static void section_synthetic()
{
    std::cout << "\n=== §24.4  Synthetic Data via LLM API ===\n\n";

    auto corpus = generate_synthetic_corpus(5);

    // Count average tokens with a small BPE
    auto tok = BPETokenizer::train(corpus, /*vocab_size=*/200);
    int64_t total_tok = 0;
    for (const auto& t : corpus)
        total_tok += static_cast<int64_t>(tok.encode(t).size());

    const double avg_tok = corpus.empty() ? 0.0
                         : static_cast<double>(total_tok) / static_cast<double>(corpus.size());

    std::cout << std::format("  Generated {} samples (avg {:.0f} tokens each)\n",
        corpus.size(), avg_tok);

    std::cout << "\n  Sample preview (first 80 chars):\n";
    for (std::size_t i = 0; i < std::min(corpus.size(), std::size_t{3}); ++i) {
        const auto& s = corpus[i];
        std::cout << std::format("    [{}] {}\n", i + 1,
            s.size() > 80 ? s.substr(0, 80) + "..." : s);
    }
}

// ── §24.5  Training Approaches ────────────────────────────────────────────────

static void section_training_approaches()
{
    std::cout << "\n=== §24.5  Training Approach Decision Tree ===\n\n";

    std::cout << std::format("  {:<28} {:<20} {:<12} {:<12} {}\n",
        "Approach", "Data needed", "Compute", "OOD risk", "Recommended when");
    std::cout << "  " << std::string(100, '-') << "\n";

    struct ApproachRow {
        std::string_view approach, data, compute, ood, when;
    };
    const ApproachRow rows[] = {
        {"From scratch",           "Chinchilla-optimal",   "Very high",   "Low",
         "Large compute budget + large dataset"},
        {"Continual pretrain",     "10–100M tokens",       "Medium",      "Low",
         "Fine-tuning an existing base model"},
        {"Distillation (Ch15)",    "Unlabelled text",      "Medium",      "Low",
         "Large teacher model available"},
        {"Synthetic-only (Phi)",   "None (LLM API)",       "Low",         "Medium",
         "No real data, small model target"},
        {"Mixed strategy",         "A few M tokens",       "Low–Medium",  "Low",
         "Best quality/cost ratio in practice"},
    };
    for (const auto& r : rows)
        std::cout << std::format("  {:<28} {:<20} {:<12} {:<12} {}\n",
            r.approach, r.data, r.compute, r.ood, r.when);

    std::cout << "\n  sub0llm's recommended path forward:\n"
              << "    1. Generate synthetic data with Ollama (§24.4) — no real data needed.\n"
              << "    2. Train demo-scale ModernGPT (D=128, 4 layers) on synthetic corpus.\n"
              << "    3. Attach the MathGPT head (Ch21) to the trained language backbone.\n"
              << "    4. Fine-tune on domain-specific data (math, code, etc.) if available.\n"
              << "    5. Apply LoRA (Ch12) for efficient task-specific adaptation.\n"
              << "    6. Use RLHF (Ch20) to align outputs with human preferences.\n"
              << "\n"
              << "  All 430+ sub0llm tests validate each of these components individually;\n"
              << "  §24.6 shows how they compose into a complete training loop.\n";
}

// ── §24.6  Full Iterative Training ────────────────────────────────────────────

static void section_full_training(int steps, const std::string& ckpt_dir, bool demo_mode)
{
    std::cout << "\n=== §24.6  Full Iterative Training ===\n\n";

    // 1. Get corpus
    std::vector<std::string> texts;
    if (demo_mode) {
        texts = generate_synthetic_corpus(50);
    } else {
        // In a real run, load from file:
        //   texts = load_texts_from_file(data_path);
        // For this demo, fall back to the same synthetic corpus
        std::cout << "  [real mode] would load from data file — using synthetic fallback\n";
        texts = generate_synthetic_corpus(50);
    }
    std::cout << std::format("  Corpus: {} documents\n", texts.size());

    // 2. Train BPE tokenizer
    const std::size_t vocab_size = 512;
    auto tok = BPETokenizer::train(texts, vocab_size);
    std::cout << std::format("  Tokenizer: vocab_size={}\n", tok.vocab_size());

    // 3. Build TextCorpus (hold out last 10% for eval)
    constexpr int64_t seq_len = 32;
    const std::size_t eval_split = std::max(std::size_t{1},
        texts.size() * 9 / 10);
    std::vector<std::string> train_texts(texts.begin(),
        texts.begin() + static_cast<std::ptrdiff_t>(eval_split));
    std::vector<std::string> eval_texts(
        texts.begin() + static_cast<std::ptrdiff_t>(eval_split), texts.end());

    TextCorpus train_corpus(train_texts, tok, seq_len, /*seed=*/42);
    TextCorpus eval_corpus(eval_texts,   tok, seq_len, /*seed=*/7);

    std::cout << std::format("  Train corpus: {} samples  Eval corpus: {} samples\n",
        train_corpus.n_samples(), eval_corpus.n_samples());

    if (train_corpus.n_samples() == 0) {
        std::cout << "  [SKIP] Not enough tokens for training (need seq_len+1 tokens minimum)\n";
        return;
    }

    // 4. Build ModernGPT
    const int64_t D       = demo_mode ? 128  : 256;
    const int64_t nlayers = demo_mode ? 4    : 6;
    const int64_t nheads  = demo_mode ? 4    : 8;
    const int64_t V       = static_cast<int64_t>(tok.vocab_size());

    ModernGPT model(V, D, static_cast<std::size_t>(nheads),
                    /*n_kv_heads=*/static_cast<std::size_t>(nheads),
                    nlayers, /*d_ff=*/0, /*n_mtp=*/0, /*seed=*/1337);

    auto params = model.parameters();
    const int64_t n_params = [&]{ int64_t n = 0;
        for (const auto* p : params) n += p->data().numel(); return n; }();
    std::cout << std::format("  ModernGPT: V={} D={} layers={} heads={}  ({:.2f}M params)\n",
        V, D, nlayers, nheads, static_cast<double>(n_params) / 1e6);

    // 5. Adam optimizer
    constexpr float lr = 3e-4f;
    Adam opt(params, lr);

    // 6. Resume from checkpoint if available
    int64_t start_step = 0;
    {
        const std::string latest = latest_checkpoint_path(ckpt_dir);
        if (!latest.empty()) {
            // Build Variable vector for loading (we need mutable Variables)
            std::vector<Variable> param_vars;
            param_vars.reserve(params.size());
            for (auto* p : params) param_vars.push_back(*p);
            try {
                start_step = load_checkpoint(param_vars, latest) + 1;
                // Copy loaded data back to the actual params
                for (std::size_t i = 0; i < params.size(); ++i)
                    params[i]->data() = param_vars[i].data();
                std::cout << std::format("  Resumed from checkpoint: {} (step {})\n",
                    latest, start_step - 1);
            } catch (const std::exception& e) {
                std::cout << std::format("  [WARN] Could not load checkpoint: {}\n", e.what());
                start_step = 0;
            }
        } else {
            std::cout << std::format("  No checkpoint found in '{}' — starting fresh\n", ckpt_dir);
        }
    }

    // 7. Training loop
    const auto wall_start = std::chrono::steady_clock::now();
    float ema_loss    = -1.f;
    constexpr float alpha = 0.95f;
    int64_t tokens_trained = 0;

    // Helper to fill a tensor from corpus, resetting on exhaustion
    Tensor ids_buf, tgt_buf;
    auto next_batch = [&]() -> bool {
        if (!train_corpus.next_sample(ids_buf, tgt_buf)) {
            train_corpus.reset();
            if (!train_corpus.next_sample(ids_buf, tgt_buf))
                return false;
        }
        return true;
    };

    std::cout << "\n  Training...\n";
    for (int64_t step = start_step; step < static_cast<int64_t>(steps); ++step) {
        if (!next_batch()) {
            std::cout << "  [WARN] Training corpus exhausted — ending early\n";
            break;
        }

        // Zero gradients
        opt.zero_grad();

        // Forward + loss
        auto logits = model.forward(ids_buf);

        // Targets: ids_buf shifted by 1 position = tgt_buf
        // logits is (T, V); we predict each token from 0..T-2, targeting 1..T-1
        // Use seq_len-1 tokens (drop the last position's logit)
        const int64_t T = seq_len;
        auto logits_trunc = narrow(logits, 0, T - 1);

        // Build target tensor from tgt_buf[0..T-2]
        Tensor targets({T - 1}, DType::Int32);
        {
            const auto src = tgt_buf.data_as<int32_t>();
            auto       dst = targets.data_as<int32_t>();
            for (int64_t i = 0; i < T - 1; ++i)
                dst[static_cast<std::size_t>(i)] = src[static_cast<std::size_t>(i)];
        }

        auto loss = cross_entropy(logits_trunc, targets);
        loss.backward();

        // Clip grads and step
        clip_grad_norm(params, 1.0f);
        opt.step();

        tokens_trained += seq_len;
        const float lv = loss.data().data_as<float>()[0];
        ema_loss = (ema_loss < 0.f) ? lv : alpha * ema_loss + (1.f - alpha) * lv;

        // Progress every 50 steps
        if ((step - start_step) % 50 == 0 || step == static_cast<int64_t>(steps) - 1) {
            const auto now     = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - wall_start).count();
            const double tok_s  = elapsed > 0 ? static_cast<double>(tokens_trained) / elapsed : 0.0;
            const int64_t remaining = static_cast<int64_t>(steps) - step - 1;
            const double eta_sec = (tok_s > 0 && remaining > 0)
                ? static_cast<double>(remaining) * seq_len / tok_s : 0.0;
            std::string eta_str = eta_sec > 3600 ? std::format("{:.1f}h", eta_sec / 3600)
                                : eta_sec > 60   ? std::format("{:.1f}m", eta_sec / 60)
                                :                  std::format("{:.0f}s", eta_sec);

            std::cout << std::format("  step {:5d}/{:d}  loss={:.4f}  {:.0f} tok/s  ETA {}\n",
                step, steps, ema_loss, tok_s, eta_str);
        }

        // Eval every 200 steps
        if ((step - start_step + 1) % 200 == 0 && eval_corpus.n_samples() > 0) {
            eval_corpus.reset();
            float eval_loss = 0.f;
            int   eval_n    = 0;
            Tensor eids, etgt;
            while (eval_corpus.next_sample(eids, etgt) && eval_n < 20) {
                auto elogits       = model.forward(eids);
                auto elogits_trunc = narrow(elogits, 0, T - 1);
                Tensor etargets({T - 1}, DType::Int32);
                {
                    const auto src = etgt.data_as<int32_t>();
                    auto       dst = etargets.data_as<int32_t>();
                    for (int64_t i = 0; i < T - 1; ++i)
                        dst[static_cast<std::size_t>(i)] = src[static_cast<std::size_t>(i)];
                }
                auto el = cross_entropy(elogits_trunc, etargets);
                eval_loss += el.data().data_as<float>()[0];
                ++eval_n;
            }
            if (eval_n > 0)
                std::cout << std::format("    [eval] step {} eval_loss={:.4f} (n={})\n",
                    step, eval_loss / static_cast<float>(eval_n), eval_n);
        }

        // Save checkpoint every 500 steps
        if ((step - start_step + 1) % 500 == 0) {
            std::vector<Variable> param_vars;
            param_vars.reserve(params.size());
            for (auto* p : params) param_vars.push_back(*p);
            save_checkpoint(param_vars, ckpt_dir, step);
            std::cout << std::format("    [ckpt] Saved checkpoint at step {}\n", step);
        }
    }

    // Save final checkpoint
    {
        const int64_t final_step = static_cast<int64_t>(steps) - 1;
        std::vector<Variable> param_vars;
        param_vars.reserve(params.size());
        for (auto* p : params) param_vars.push_back(*p);
        save_checkpoint(param_vars, ckpt_dir, final_step);
        std::cout << std::format("    [ckpt] Saved final checkpoint at step {}\n", final_step);
    }

    // 8. Summary
    const auto now     = std::chrono::steady_clock::now();
    const double total = std::chrono::duration<double>(now - wall_start).count();
    std::cout << std::format("\n  Summary:\n"
                             "    Final EMA loss : {:.4f}\n"
                             "    Tokens trained : {}\n"
                             "    Wall time      : {:.1f}s\n"
                             "    Throughput     : {:.0f} tok/s\n",
        ema_loss, tokens_trained, total,
        total > 0 ? static_cast<double>(tokens_trained) / total : 0.0);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::string phase    = "all";
    std::string ckpt_dir = "/tmp/ch24_ckpts";
    int         steps    = 200;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--phase" && i + 1 < argc)
            phase = argv[++i];
        else if (arg == "--ckpt-dir" && i + 1 < argc)
            ckpt_dir = argv[++i];
        else if (arg == "--steps" && i + 1 < argc)
            steps = std::stoi(argv[++i]);
    }

    const bool run_all = (phase == "all");

    if (run_all || phase == "landscape")  section_landscape();
    if (run_all || phase == "estimate")   section_estimate();
    if (run_all || phase == "pipeline")   section_pipeline();
    if (run_all || phase == "synth")      section_synthetic();
    if (run_all || phase == "approaches") section_training_approaches();
    if (run_all || phase == "train") {
        // 'train' phase: use provided steps (default 2000 when called explicitly)
        // 'all' phase: use shorter demo (200 steps or whatever --steps says)
        const int actual_steps = (phase == "train" && steps == 200) ? 2000 : steps;
        section_full_training(actual_steps, ckpt_dir, phase != "train");
    }

    if (!run_all &&
        phase != "landscape" && phase != "estimate" && phase != "pipeline" &&
        phase != "synth"     && phase != "approaches" && phase != "train")
    {
        throw std::runtime_error(
            std::format("unknown phase '{}' — valid: landscape, estimate, pipeline, "
                        "synth, approaches, train, all", phase));
    }

    return 0;
}
