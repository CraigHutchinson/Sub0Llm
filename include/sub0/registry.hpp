// registry.hpp -- model storage layout + a self-describing model index.
//
// Every trained model lives in its own directory under a models root, named for its
// IDENTITY -- corpus + architecture dims + the git SHA of the code that trained it:
//   models/sub0llm_<corpus>_d<D>l<L>h<H>sq<SEQ>v<VOCAB>[t]_<sha>/
//     model.bin        the parameters
//     model.bin.ckpt   the resumable optimizer/loop state
//     meta.txt         key=value provenance (below)
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
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sub0::registry {

struct ModelMeta {
    std::string corpus, git_sha, created, status;
    int d_model = 0, n_layers = 0, n_heads = 0, seq_len = 0, vocab = 0, ternary = 0;
    double best_val_nelbo = -1.0;             // -1 = not yet evaluated
    std::filesystem::path dir;                // the model's directory (set by scan)
};

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
                                       int seq, int vocab, int ternary, const std::string& sha) {
    std::string name = "sub0llm_" + corpus_tag(corpus) +
                       "_d" + std::to_string(d) + "l" + std::to_string(l) + "h" + std::to_string(h) +
                       "sq" + std::to_string(seq) + "v" + std::to_string(vocab) +
                       (ternary ? "t" : "") + "_" + (sha.empty() ? "nogit" : sha);
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
       << "git_sha="         << m.git_sha   << "\n"
       << "created="         << m.created   << "\n"
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
        else if (k == "status")         m.status = v;
        else if (k == "d_model")        m.d_model = as_int(v);
        else if (k == "n_layers")       m.n_layers = as_int(v);
        else if (k == "n_heads")        m.n_heads = as_int(v);
        else if (k == "seq_len")        m.seq_len = as_int(v);
        else if (k == "vocab")          m.vocab = as_int(v);
        else if (k == "ternary")        m.ternary = as_int(v);
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

// A model loads into the current build only if its architecture dims match exactly.
inline bool compatible(const ModelMeta& m, int d, int l, int h, int seq, int vocab, int ternary) {
    return m.d_model == d && m.n_layers == l && m.n_heads == h &&
           m.seq_len == seq && m.vocab == vocab && m.ternary == ternary;
}

}  // namespace sub0::registry
