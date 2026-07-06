// registry.hpp -- model storage layout + a self-describing model index.
//
// Every trained model lives in its own directory under a models root, named for its
// IDENTITY -- corpus + architecture dims + the git SHA of the code that trained it:
//   models/sub0llm_<corpus>_d<D>l<L>h<H>sq<SEQ>v<VOCAB>[t][r]_<sha>/
//     model.bin        the parameters
//     model.bin.ckpt   the resumable optimizer/loop state
//     meta.txt         key=value provenance (below)
//
// The optional suffix letters encode build variants that change the weights' meaning without
// changing their shape: 't' = ternary block weights, 'r' = RoPE positional encoding (absolute
// learned positions, the legacy default, are untagged), 'g' = SwiGLU-gated FFN (the plain
// GELU+bias FFN, the legacy default, is untagged), 'w' = tied embeddings (the head reuses tok_emb;
// an untied head, its own matrix+bias, the legacy default, is untagged), 'q' = QK-norm (per-head
// RMSNorm on Q/K before RoPE; no norm, the legacy default, is untagged).
//
// The "registry" is just the set of meta.txt files: discovery scans them (no separate
// index to drift out of sync), and a model is COMPATIBLE with the current build iff its
// architecture dims match -- the checkpoint can only load into a matching engine, so
// dim-mismatched models are dead weight a `prune` can reclaim. The git SHA tags the code
// version for spotting obsolete-but-compatible models. Pure std + <filesystem>; the engine
// supplies the compile-time dims.
#pragma once

#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sub0::registry {

struct ModelMeta {
    std::string corpus, git_sha, created, updated, status;
    int d_model = 0, n_layers = 0, n_heads = 0, seq_len = 0, vocab = 0, ternary = 0;
    int pos_encoding = 0;                     // 0 = absolute learned (legacy default), 1 = RoPE
    int gated_ffn = 0;                        // 0 = plain GELU+bias FFN (legacy default), 1 = SwiGLU-gated
    int tied_embeddings = 0;                  // 0 = untied head (legacy default), 1 = head reuses tok_emb
    int qk_norm = 0;                          // 0 = no QK-norm (legacy default), 1 = per-head RMSNorm on Q/K
    // Training state (provenance of the run that produced this snapshot).
    long long steps = 0;                      // optimizer iterations completed
    double epochs = 0.0;                      // fractional epochs of the corpus covered
    long long tokens_seen = 0;                // approximate tokens consumed (steps * batch * seq)
    int batch = 0;                            // minibatch size
    double lr = 0.0;                          // learning rate
    unsigned seed = 0;                        // RNG seed (for reproducibility)
    double best_val_nelbo = -1.0;             // -1 = not yet evaluated
    std::filesystem::path dir;                // the model's directory (set by scan)
};

// Dir-name tag for the positional-encoding scheme (absolute is the legacy default -> untagged).
inline const char* pos_tag(int pos_enc) { return pos_enc == 1 ? "r" : ""; }

// A short, filesystem-safe tag for a corpus path: its stem, lowercased, non-alnum -> '_'.
inline std::string corpus_tag(const std::string& corpus_path) {
    std::string stem = std::filesystem::path(corpus_path).stem().string();
    std::string t;
    for (char c : stem) {
        const unsigned char u = static_cast<unsigned char>(c);
        t.push_back((u >= 'A' && u <= 'Z') ? static_cast<char>(u + 32)
                    : ((u >= 'a' && u <= 'z') || (u >= '0' && u <= '9')) ? c : '_');
    }
    if (t.empty()) t = "corpus";
    return t;
}

// The identity-encoding directory for a model (no I/O).
inline std::filesystem::path model_dir(const std::filesystem::path& models_root,
                                       const std::string& corpus, int d, int l, int h,
                                       int seq, int vocab, int ternary, int pos_enc,
                                       const std::string& sha, int gated_ffn = 0,
                                       int tied_embeddings = 0, int qk_norm = 0) {
    std::string name = "sub0llm_" + corpus_tag(corpus) +
                       "_d" + std::to_string(d) + "l" + std::to_string(l) + "h" + std::to_string(h) +
                       "sq" + std::to_string(seq) + "v" + std::to_string(vocab) +
                       (ternary ? "t" : "") + pos_tag(pos_enc) + (gated_ffn ? "g" : "") +
                       (tied_embeddings ? "w" : "") + (qk_norm ? "q" : "") +
                       "_" + (sha.empty() ? "nogit" : sha);
    return models_root / name;
}

inline std::string now_iso() {
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

inline void write_meta(const std::filesystem::path& dir, const ModelMeta& m) {
    std::error_code ec; std::filesystem::create_directories(dir, ec);
    std::ofstream os(dir / "meta.txt", std::ios::trunc);
    if (!os) return;
    os << "model=sub0llm\n"
       << "corpus="          << m.corpus    << "\n"
       << "d_model="         << m.d_model   << "\n"
       << "n_layers="        << m.n_layers  << "\n"
       << "n_heads="         << m.n_heads   << "\n"
       << "seq_len="         << m.seq_len   << "\n"
       << "vocab="           << m.vocab     << "\n"
       << "ternary="         << m.ternary   << "\n"
       << "pos_encoding="    << m.pos_encoding << "\n"
       << "gated_ffn="       << m.gated_ffn << "\n"
       << "tied_embeddings=" << m.tied_embeddings << "\n"
       << "qk_norm="         << m.qk_norm << "\n"
       << "git_sha="         << m.git_sha   << "\n"
       << "created="         << m.created   << "\n"
       << "updated="         << m.updated   << "\n"
       << "steps="           << m.steps     << "\n"
       << "epochs="          << m.epochs    << "\n"
       << "tokens_seen="     << m.tokens_seen << "\n"
       << "batch="           << m.batch     << "\n"
       << "lr="              << m.lr        << "\n"
       << "seed="            << m.seed      << "\n"
       << "best_val_nelbo="  << m.best_val_nelbo << "\n"
       << "status="          << m.status    << "\n";
}

inline bool read_meta(const std::filesystem::path& dir, ModelMeta& m) {
    std::ifstream is(dir / "meta.txt");
    if (!is) return false;
    auto as_int = [](const std::string& s) { int v = 0; std::from_chars(s.data(), s.data() + s.size(), v); return v; };
    for (std::string line; std::getline(is, line);) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if      (k == "corpus")         m.corpus = v;
        else if (k == "git_sha")        m.git_sha = v;
        else if (k == "created")        m.created = v;
        else if (k == "updated")        m.updated = v;
        else if (k == "status")         m.status = v;
        else if (k == "d_model")        m.d_model = as_int(v);
        else if (k == "n_layers")       m.n_layers = as_int(v);
        else if (k == "n_heads")        m.n_heads = as_int(v);
        else if (k == "seq_len")        m.seq_len = as_int(v);
        else if (k == "vocab")          m.vocab = as_int(v);
        else if (k == "ternary")        m.ternary = as_int(v);
        else if (k == "pos_encoding")   m.pos_encoding = as_int(v);
        else if (k == "gated_ffn")      m.gated_ffn = as_int(v);
        else if (k == "tied_embeddings") m.tied_embeddings = as_int(v);
        else if (k == "qk_norm")        m.qk_norm = as_int(v);
        else if (k == "steps")          m.steps = std::strtoll(v.c_str(), nullptr, 10);
        else if (k == "epochs")         m.epochs = std::strtod(v.c_str(), nullptr);
        else if (k == "tokens_seen")    m.tokens_seen = std::strtoll(v.c_str(), nullptr, 10);
        else if (k == "batch")          m.batch = as_int(v);
        else if (k == "lr")             m.lr = std::strtod(v.c_str(), nullptr);
        else if (k == "seed")           m.seed = static_cast<unsigned>(std::strtoul(v.c_str(), nullptr, 10));
        else if (k == "best_val_nelbo") m.best_val_nelbo = std::strtod(v.c_str(), nullptr);
    }
    m.dir = dir;
    return true;
}

// All models discovered under the root (each subdirectory holding a meta.txt).
inline std::vector<ModelMeta> scan(const std::filesystem::path& models_root) {
    std::vector<ModelMeta> out;
    std::error_code ec;
    if (!std::filesystem::exists(models_root, ec)) return out;
    for (const auto& e : std::filesystem::directory_iterator(models_root, ec)) {
        if (!e.is_directory()) continue;
        ModelMeta m;
        if (read_meta(e.path(), m)) out.push_back(std::move(m));
    }
    return out;
}

// A model loads into the current build only if its architecture dims AND weight-meaning variants
// (ternary, positional-encoding scheme, gated-FFN scheme, tied-embeddings scheme, tokenizer scheme)
// match exactly -- a same-shape mismatch would load silently but compute nonsense (the JOIN
// tokenizer's token ids mean different text). This is a DIAGNOSTIC check (`models`/`models --prune`);
// the actual load-time gate is engine_core.cpp's binary Header comparison, which is authoritative
// regardless of what this says.
inline bool compatible(const ModelMeta& m, int d, int l, int h, int seq, int vocab, int ternary,
                       int pos_enc, int gated_ffn = 0, int tied_embeddings = 0, int qk_norm = 0) {
    return m.d_model == d && m.n_layers == l && m.n_heads == h &&
           m.seq_len == seq && m.vocab == vocab && m.ternary == ternary &&
           m.pos_encoding == pos_enc && m.gated_ffn == gated_ffn &&
           m.tied_embeddings == tied_embeddings && m.qk_norm == qk_norm;
}

}  // namespace sub0::registry
