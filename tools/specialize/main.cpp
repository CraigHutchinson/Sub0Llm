// sub0llm-specialize — the compile-time specialization front-end (Ch27).
//
// Reads a GGUF model's *metadata* (not its weights) and emits a C++23 header
// in which every architectural axis — vocab size, embed dim, head counts, head
// dim, layer count, FFN width, RoPE base, norm epsilon, activation, norm style,
// embedding scale, attention windowing — is a `static constexpr` member of a
// generated `Spec` struct, plus a JSON manifest of detected features.
//
// This is the "move runtime axes to compile time" engine: downstream binaries
// (gemma4-cli / gemma4-server) include the generated header and compile a model
// whose shapes are known to the optimizer, so loops unroll, buffers are sized,
// strides constant-fold, and -march=native vectorizes against fixed dimensions.
//
// Weights are still loaded from the GGUF at runtime — data is runtime, *shape*
// is compile time.
//
// Usage:
//   sub0llm-specialize --model path/to/model.gguf
//                      [--name Ident]        identifier for the Spec struct/file
//                      [--out-dir DIR]       where to write (default ".")
//                      [--namespace NS]      C++ namespace (default sub0llm::spec)

#include "sub0llm/nn/gguf_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using sub0llm::nn::GGUFReader;
using sub0llm::nn::GGUFModelConfig;

// ── feature manifest ──────────────────────────────────────────────────────────
//
// Everything a specialized build needs to know that isn't a raw dimension.
// Derived from the GGUF architecture string + tensor-name presence.
struct Manifest {
    std::string arch;            // lowercased general.architecture
    std::string family;          // "gemma" | "qwen" | "llama" | "generic"

    int64_t     vocab_size  = 0;
    int64_t     embed_dim   = 0;
    std::size_t n_heads     = 0;
    std::size_t n_kv_heads  = 0;
    std::size_t head_dim    = 0;
    int64_t     n_layers    = 0;
    int64_t     d_ff        = 0;
    int64_t     context_len = 0;
    float       rope_base   = 10000.0f;
    float       norm_eps    = 1e-6f;

    // Feature flags (the things that change the forward math).
    bool   use_qk_norm     = false;  // per-head RMSNorm on Q,K before RoPE (Qwen3, Gemma3)
    bool   tied_embeddings = true;   // lm_head == token_embd.T (no output.weight)
    bool   norm_plus_one   = false;  // RMSNorm uses (1+weight) gamma (Gemma)
    bool   attn_qkv_bias   = false;  // Q/K/V projections carry a bias (Qwen2)
    int    activation      = 0;      // 0 = SiLU/SwiGLU, 1 = GELU/GeGLU (Gemma)
    float  embed_scale     = 1.0f;   // multiply embeddings by this (Gemma: sqrt(D))

    // Attention windowing.
    int64_t sliding_window     = 0;  // 0 = full attention everywhere
    int     local_global_stride = 0; // 0 = uniform; N = every Nth layer is global (Gemma3: 6)
};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

// Turn a model filename into a valid C++ identifier: "Qwen3-0.6B-Q8_0" → "Qwen3_0_6B_Q8_0".
std::string sanitize_ident(std::string s) {
    for (char& c : s)
        if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
    if (!s.empty() && std::isdigit(static_cast<unsigned char>(s.front())))
        s.insert(s.begin(), '_');
    return s;
}

Manifest derive_manifest(const GGUFReader& reader) {
    const GGUFModelConfig& cfg = reader.config();
    Manifest m;

    m.arch        = lower(cfg.arch);
    m.vocab_size  = cfg.vocab_size;
    m.embed_dim   = cfg.embed_dim;
    m.n_heads     = cfg.n_heads;
    m.n_kv_heads  = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : cfg.n_heads;
    m.head_dim    = cfg.head_dim > 0
                        ? static_cast<std::size_t>(cfg.head_dim)
                        : (cfg.n_heads ? static_cast<std::size_t>(cfg.embed_dim) / cfg.n_heads : 0);
    m.n_layers    = cfg.n_layers;
    m.d_ff        = cfg.d_ff;
    m.context_len = cfg.context_len;
    m.rope_base   = cfg.rope_base;
    m.norm_eps    = cfg.norm_eps;

    // family classification
    if (starts_with(m.arch, "gemma"))      m.family = "gemma";
    else if (starts_with(m.arch, "qwen"))  m.family = "qwen";
    else if (starts_with(m.arch, "llama")) m.family = "llama";
    else                                   m.family = "generic";

    // feature detection from tensor presence (authoritative — reflects the actual file)
    m.use_qk_norm     = reader.has_tensor("blk.0.attn_q_norm.weight");
    m.tied_embeddings = !reader.has_tensor("output.weight");
    m.attn_qkv_bias   = reader.has_tensor("blk.0.attn_q.bias");

    // Gemma-family architectural choices (confirmed from the GGUF where possible;
    // otherwise the documented Gemma-2/3 lineage that Gemma 4 inherits).
    if (m.family == "gemma") {
        m.norm_plus_one = true;                                   // (1 + weight) RMSNorm
        m.activation    = 1;                                      // GeGLU
        m.embed_scale   = std::sqrt(static_cast<float>(m.embed_dim));
        m.sliding_window = cfg.sliding_window;
        // Gemma 2/3 interleave 5 local layers : 1 global (every 6th is global).
        if (m.sliding_window > 0) m.local_global_stride = 6;
    } else {
        m.sliding_window = cfg.sliding_window;
    }
    return m;
}

const char* activation_name(int a) { return a == 1 ? "GeGLU (GELU gate)" : "SwiGLU (SiLU gate)"; }

// Emit a valid C++ float literal. std::format("{}", 1.0f) yields "1", and "1f"
// is not a legal float literal — ensure a '.' or exponent before the 'f' suffix.
std::string float_lit(float v) {
    std::string s = std::format("{}", v);
    const bool has_dot_or_exp =
        s.find_first_of(".eEnN") != std::string::npos;  // n/N catches inf/nan
    if (!has_dot_or_exp) s += ".0";
    return s + "f";
}

// ── emitters ──────────────────────────────────────────────────────────────────

void emit_header(std::ostream& out, const Manifest& m,
                 const std::string& ident, const std::string& ns,
                 const std::string& src_name) {
    out << "// AUTO-GENERATED by sub0llm-specialize — DO NOT EDIT.\n";
    out << "// Source GGUF: " << src_name << "\n";
    out << "// Architecture: " << m.arch << "  (family: " << m.family << ")\n";
    out << "//\n";
    out << "// Every field is a compile-time constant: include this header in a\n";
    out << "// dedicated binary to monomorphize the model to these exact shapes.\n";
    out << "#pragma once\n\n";
    out << "#include <cstddef>\n#include <cstdint>\n\n";

    // open namespace (supports nested a::b)
    out << "namespace " << ns << " {\n\n";

    out << "struct " << ident << " {\n";
    out << "    static constexpr const char* arch        = \"" << m.arch << "\";\n";
    out << "    static constexpr const char* family      = \"" << m.family << "\";\n\n";

    out << "    // ── dimensions ──────────────────────────────────────────────\n";
    out << std::format("    static constexpr int64_t     vocab_size   = {};\n", m.vocab_size);
    out << std::format("    static constexpr int64_t     embed_dim    = {};\n", m.embed_dim);
    out << std::format("    static constexpr std::size_t n_heads      = {};\n", m.n_heads);
    out << std::format("    static constexpr std::size_t n_kv_heads   = {};\n", m.n_kv_heads);
    out << std::format("    static constexpr std::size_t head_dim     = {};\n", m.head_dim);
    out << std::format("    static constexpr int64_t     n_layers     = {};\n", m.n_layers);
    out << std::format("    static constexpr int64_t     d_ff         = {};\n", m.d_ff);
    out << std::format("    static constexpr int64_t     context_len  = {};\n", m.context_len);
    out << std::format("    static constexpr float       rope_base    = {};\n", float_lit(m.rope_base));
    out << std::format("    static constexpr float       norm_eps     = {};\n", float_lit(m.norm_eps));
    out << std::format("    static constexpr float       embed_scale  = {};\n\n", float_lit(m.embed_scale));

    out << "    // ── feature flags ───────────────────────────────────────────\n";
    out << std::format("    static constexpr bool use_qk_norm      = {};\n", m.use_qk_norm);
    out << std::format("    static constexpr bool tied_embeddings  = {};\n", m.tied_embeddings);
    out << std::format("    static constexpr bool norm_plus_one    = {};\n", m.norm_plus_one);
    out << std::format("    static constexpr bool attn_qkv_bias    = {};\n", m.attn_qkv_bias);
    out << "    // 0 = SwiGLU (SiLU gate), 1 = GeGLU (GELU gate)\n";
    out << std::format("    static constexpr int  activation       = {};\n", m.activation);
    out << std::format("    static constexpr int64_t sliding_window     = {};\n", m.sliding_window);
    out << "    // 0 = uniform attention; N = every Nth layer is global (else windowed)\n";
    out << std::format("    static constexpr int  local_global_stride  = {};\n\n", m.local_global_stride);

    out << "    // is layer l a global (full-attention) layer?\n";
    out << "    static constexpr bool is_global_layer(int64_t l) noexcept {\n";
    out << "        return local_global_stride == 0 || sliding_window <= 0\n";
    out << "            || ((l + 1) % local_global_stride == 0);\n";
    out << "    }\n";

    // a couple of compile-time sanity checks baked into the header
    out << "\n    static_assert(embed_dim > 0 && n_heads > 0 && n_layers > 0,\n";
    out << "                  \"specialized spec must have positive core dimensions\");\n";
    out << "    static_assert(n_heads % n_kv_heads == 0,\n";
    out << "                  \"n_heads must be a multiple of n_kv_heads (GQA)\");\n";
    out << "};\n\n";

    out << "} // namespace " << ns << "\n";
}

void emit_json(std::ostream& out, const Manifest& m, const std::string& src_name) {
    auto b = [](bool v) { return v ? "true" : "false"; };
    out << "{\n";
    out << std::format("  \"source\": \"{}\",\n", src_name);
    out << std::format("  \"arch\": \"{}\",\n", m.arch);
    out << std::format("  \"family\": \"{}\",\n", m.family);
    out << std::format("  \"vocab_size\": {},\n", m.vocab_size);
    out << std::format("  \"embed_dim\": {},\n", m.embed_dim);
    out << std::format("  \"n_heads\": {},\n", m.n_heads);
    out << std::format("  \"n_kv_heads\": {},\n", m.n_kv_heads);
    out << std::format("  \"head_dim\": {},\n", m.head_dim);
    out << std::format("  \"n_layers\": {},\n", m.n_layers);
    out << std::format("  \"d_ff\": {},\n", m.d_ff);
    out << std::format("  \"context_len\": {},\n", m.context_len);
    out << std::format("  \"rope_base\": {},\n", m.rope_base);
    out << std::format("  \"norm_eps\": {},\n", m.norm_eps);
    out << std::format("  \"embed_scale\": {},\n", m.embed_scale);
    out << std::format("  \"use_qk_norm\": {},\n", b(m.use_qk_norm));
    out << std::format("  \"tied_embeddings\": {},\n", b(m.tied_embeddings));
    out << std::format("  \"norm_plus_one\": {},\n", b(m.norm_plus_one));
    out << std::format("  \"attn_qkv_bias\": {},\n", b(m.attn_qkv_bias));
    out << std::format("  \"activation\": {},\n", m.activation);
    out << std::format("  \"sliding_window\": {},\n", m.sliding_window);
    out << std::format("  \"local_global_stride\": {}\n", m.local_global_stride);
    out << "}\n";
}

void print_summary(const Manifest& m, const std::string& ident) {
    std::cout << "\n  specialized spec: " << ident << "  (" << m.arch
              << ", family=" << m.family << ")\n";
    std::cout << "  ----------------------------------------------------------\n";
    std::cout << std::format("    vocab_size    {}\n", m.vocab_size);
    std::cout << std::format("    embed_dim     {}\n", m.embed_dim);
    std::cout << std::format("    n_heads/kv    {} / {}\n", m.n_heads, m.n_kv_heads);
    std::cout << std::format("    head_dim      {}\n", m.head_dim);
    std::cout << std::format("    n_layers      {}\n", m.n_layers);
    std::cout << std::format("    d_ff          {}\n", m.d_ff);
    std::cout << std::format("    context_len   {}\n", m.context_len);
    std::cout << std::format("    rope_base     {}\n", m.rope_base);
    std::cout << std::format("    norm_eps      {}\n", m.norm_eps);
    std::cout << std::format("    embed_scale   {}\n", m.embed_scale);
    std::cout << "    features:\n";
    std::cout << std::format("      qk_norm={}  tied_emb={}  (1+w)norm={}  qkv_bias={}\n",
                             m.use_qk_norm, m.tied_embeddings, m.norm_plus_one, m.attn_qkv_bias);
    std::cout << std::format("      activation={}\n", activation_name(m.activation));
    if (m.sliding_window > 0)
        std::cout << std::format("      sliding_window={}  global every {} layers\n",
                                 m.sliding_window, m.local_global_stride);
    else
        std::cout << "      full attention (no sliding window)\n";
    std::cout << "  ----------------------------------------------------------\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string model, name, out_dir = ".", ns = "sub0llm::spec";
    bool dump_meta = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
            return argv[++i];
        };
        if (a == "--model")          model   = next();
        else if (a == "--name")      name    = next();
        else if (a == "--out-dir")   out_dir = next();
        else if (a == "--namespace") ns      = next();
        else if (a == "--dump-meta") dump_meta = true;
        else if (a == "-h" || a == "--help") {
            std::cout << "usage: sub0llm-specialize --model M.gguf [--name N] "
                         "[--out-dir DIR] [--namespace NS] [--dump-meta]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    if (model.empty()) {
        std::cerr << "error: --model is required\n";
        return 2;
    }

    try {
        namespace fs = std::filesystem;
        const std::string stem = fs::path(model).stem().string();
        const std::string ident = name.empty() ? sanitize_ident(stem) : sanitize_ident(name);

        std::cout << "[specialize] reading metadata: " << model << "\n";
        GGUFReader reader(model);

        if (dump_meta) {
            std::cout << "\n  all metadata (" << reader.metadata().size() << " keys):\n";
            for (const auto& [k, v] : reader.metadata())
                std::cout << "    " << k << " = " << v << "\n";
            std::cout << "\n";
        }

        Manifest m = derive_manifest(reader);
        print_summary(m, ident);

        fs::create_directories(out_dir);
        const std::string lower_ident = lower(ident);
        const fs::path hdr  = fs::path(out_dir) / (lower_ident + "_spec.hpp");
        const fs::path json = fs::path(out_dir) / (lower_ident + "_arch.json");

        {
            std::ofstream f(hdr);
            if (!f) throw std::runtime_error(std::format("cannot write {}", hdr.string()));
            emit_header(f, m, ident, ns, fs::path(model).filename().string());
        }
        {
            std::ofstream f(json);
            if (!f) throw std::runtime_error(std::format("cannot write {}", json.string()));
            emit_json(f, m, fs::path(model).filename().string());
        }

        std::cout << "  wrote " << hdr.string()  << "\n";
        std::cout << "  wrote " << json.string() << "\n";
        std::cout << "\n  next: include \"" << lower_ident << "_spec.hpp\" and build a "
                     "dedicated binary against " << ns << "::" << ident << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[specialize] error: " << e.what() << "\n";
        return 1;
    }
}
