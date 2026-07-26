// registry.hpp -- model storage layout + a self-describing model index.
//
// Every trained model lives in its own directory under a models root, named for its
// IDENTITY -- corpus + architecture dims + a date+time tag for the training attempt:
//   models/sub0llm_<corpus>_d<D>l<L>h<H>sq<SEQ>v<VOCAB>[t][r]_<YYYYMMDD-HHMMSS>/
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
// The directory tag is a date+time, NOT the git SHA: the SHA changes on every commit (even an
// unrelated log-message fix), but architecture/weight-format compatibility is the thing that
// actually determines whether a checkpoint can resume -- see `compatible()` below, which already
// checks dims/flags and deliberately ignores the SHA. Keying the directory name to the SHA meant a
// trivial commit between stopping and resuming a multi-day run would silently start a fresh model
// instead of resuming it. The SHA is still recorded in meta.txt as a provenance/audit field (see
// ModelMeta::git_sha) -- just no longer part of directory identity or resume-matching.
//
// The "registry" is just the set of meta.txt files: discovery scans them (no separate
// index to drift out of sync), and a model is COMPATIBLE with the current build iff its
// architecture dims match -- the checkpoint can only load into a matching engine, so
// dim-mismatched models are dead weight a `prune` can reclaim. Pure std + <filesystem>; the
// engine supplies the compile-time dims.
#pragma once

#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Export macro for read_config_json's out-of-line definition (src/run_config.cpp, compiled once
// into sub0_core): every other function in this header is `inline` and compiled directly into
// each consumer's own translation unit, so this is the one symbol here that actually crosses the
// sub0_core.dll boundary and needs it. A self-contained copy of core.hpp's own SUB0_API (not an
// include of core.hpp itself) -- this header is deliberately pure std + <filesystem>, no engine
// dependency, matching its own doc comment above.
#ifndef SUB0_API
  #if defined(_WIN32)
    #define SUB0_API __declspec(dllexport)
  #else
    #define SUB0_API __attribute__((visibility("default")))
  #endif
#endif

namespace sub0::registry {

struct ModelMeta {
    std::string corpus, git_sha, created, updated, status;
    int d_model = 0, n_layers = 0, n_heads = 0, seq_len = 0, vocab = 0, ternary = 0;
    int pos_encoding = 0;                     // 0 = absolute learned (legacy default), 1 = RoPE
    int gated_ffn = 0;                        // 0 = plain GELU+bias FFN (legacy default), 1 = SwiGLU-gated
    int tied_embeddings = 0;                  // 0 = untied head (legacy default), 1 = head reuses tok_emb
    int qk_norm = 0;                          // 0 = no QK-norm (legacy default), 1 = per-head RMSNorm on Q/K
    int optimizer = 0;                        // 0 = AdamW (legacy default), 1 = Muon (hidden 2D matrices)
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

// --- RunConfig: the single source of truth for a model directory's TRAINING RECIPE -------------
//
// Born from a real incident: a GPU-fault auto-resume once silently continued training under plain
// AdamW instead of the Muon it had used for its first 7728 steps, because the optimizer choice was
// derived purely from the CURRENT invocation's CLI flag with no cross-check against what the model
// had actually been trained with -- `ModelMeta::optimizer` above was written to meta.txt every run
// but never read back. `RunConfig` fixes the CLASS of bug, not just that one field: it is the one
// place a new training-recipe option is declared, and by construction that declaration is *also*
// its serialization -- add a row to SUB0_RUN_CONFIG_FIELDS below and the field is written to and
// read from every model directory's config.json automatically (write_config_json/read_config_json
// are generated from this one list, not hand-maintained in three separate places that could drift
// out of sync with each other, which is exactly how the optimizer bug happened in the first place).
//
// Scope, deliberately not "every setting in the program": holds CHOICES that must not silently
// drift across a resume and have no other authoritative home.
//   - Architecture dims/flags ARE included, even though they're also `constexpr` at compile time
//     and also duplicated in `ModelMeta` -- see `compatible()`'s doc comment above for why that
//     redundancy is load-bearing (cross-build-config filtering before any model.bin is opened).
//   - `batch`/`lr`/`seed` ARE included for completeness/comparison, but are NOT re-enforced on
//     resume here -- they already have a correct, tightly-coupled authoritative home in the
//     `.ckpt` binary (see train_stage.cpp's `load_checkpoint`), which wins on resume with its own
//     log line. config.json's copy is informational, same role as meta.txt's copy.
//   - `optimizer` IS re-enforced on resume (see train_stage.cpp) -- it has no checkpoint-binary
//     home and no other read-back path, which is precisely the gap that caused the incident above.
//   - Deliberately EXCLUDED: per-invocation POLICY that has no "correct" persisted value to protect
//     (`steps`/`keep`/`resume_mode` -- see train_stage.cpp's own comment on why `steps` specifically
//     is never persisted) and per-STEP state that already lives in the checkpoint (RNG state, Adam
//     moment buffers, eval history) -- config.json is not a second copy of the checkpoint.
// `config_schema` guards the blend-schedule redesign (project memory: the blend scheduler used to be 5
// flat fields here -- spell_mix/scratch_mix/op_mix/content_embed/content_embed_kind -- replaced by a
// separately-pinned blend_schedule.json, since a staged multi-source schedule can't be expressed as flat
// scalars). Struct default 1 = "pre-redesign / unversioned": every config.json written before this field
// existed never wrote this key, so it decodes to 1 automatically (the SAME forward-compatible defaulting
// this file already relies on elsewhere -- an absent key just uses the struct default). New code always
// writes 2. `train`/`gen`/`eval` must refuse to operate on a `config_schema<2` model dir rather than
// silently reinterpreting the now-removed fields as absent/off -- a model actually trained with
// content-embed active would otherwise decode WITHOUT its interceptor, a confidently-wrong result, not a
// safe fallback.
#define SUB0_RUN_CONFIG_FIELDS(X)         \
    X(std::string, corpus,          "")   \
    X(int,         d_model,         0)    \
    X(int,         n_layers,        0)    \
    X(int,         n_heads,         0)    \
    X(int,         n_kv_heads,      0)    \
    X(int,         seq_len,         0)    \
    X(int,         vocab,           0)    \
    X(int,         ternary,         0)    \
    X(int,         pos_encoding,    0)    \
    X(int,         gated_ffn,       0)    \
    X(int,         tied_embeddings, 0)    \
    X(int,         qk_norm,         0)    \
    X(int,         optimizer,       0)    \
    X(int,         batch,           0)    \
    X(double,      lr,              0.0)  \
    X(unsigned,    seed,            0u)   \
    X(int,         config_schema,   1)

struct RunConfig {
#define SUB0_RUN_CONFIG_DECL(type, name, def) type name = def;
    SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_DECL)
#undef SUB0_RUN_CONFIG_DECL
};

namespace detail {
inline void json_write(std::ostream& os, const std::string& v) {
    os << '"';
    for (unsigned char c : v) {
        if (c == '"' || c == '\\')      { os << '\\' << static_cast<char>(c); }
        else if (c == '\n')             { os << "\\n"; }
        else if (c == '\r')             { os << "\\r"; }
        else if (c == '\t')             { os << "\\t"; }
        else if (c < 0x20 || c >= 0x7F) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); os << b; }
        else                             { os << static_cast<char>(c); }
    }
    os << '"';
}
inline void json_write(std::ostream& os, int v)      { os << v; }
inline void json_write(std::ostream& os, unsigned v) { os << v; }
inline void json_write(std::ostream& os, double v)   { os << v; }

// The enum-valued RunConfig fields print as NAMES (pos_encoding=rope, optimizer=muon, gated_ffn=on, ...)
// for readability; every other field stays numeric. The int<->name mapping lives HERE only; the reader
// (run_config.cpp) accepts EITHER form so old numeric config.json files still load. enum_name returns null
// for a non-enum field or an out-of-range value (-> numeric fallback).
inline const char* enum_name(std::string_view field, int v) {
    auto nth = [v](std::initializer_list<const char*> names) -> const char* {
        int i = 0; for (const char* n : names) if (i++ == v) return n; return nullptr; };
    if (field == "pos_encoding") return nth({"learned", "rope"});
    if (field == "optimizer")    return nth({"adamw", "muon"});
    if (field == "gated_ffn" || field == "tied_embeddings" || field == "qk_norm" || field == "ternary")
        return nth({"off", "on"});
    return nullptr;
}
inline bool enum_parse(std::string_view field, std::string_view s, int& out) {
    for (int v = 0;; ++v) { const char* n = enum_name(field, v); if (!n) return false; if (s == n) { out = v; return true; } }
}
// Field-aware write: an enum field emits its name (quoted string); everything else is numeric/string as-is.
template <class T> void json_write_field(std::ostream& os, std::string_view, const T& v) { json_write(os, v); }
inline void json_write_field(std::ostream& os, std::string_view field, int v) {
    if (const char* nm = enum_name(field, v)) json_write(os, std::string(nm));
    else json_write(os, v);
}
}  // namespace detail

// Writes <dir>/config.json, one field per SUB0_RUN_CONFIG_FIELDS row -- see RunConfig's own doc
// comment. Hand-rolled (not simdjson, which is a parser): a flat object of scalars needs no library
// to emit correctly, and this keeps the write path dependency-free for every caller.
inline void write_config_json(const RunConfig& c, const std::filesystem::path& dir) {
    std::error_code ec; std::filesystem::create_directories(dir, ec);
    std::ofstream os(dir / "config.json", std::ios::trunc);
    if (!os) return;
    os << "{\n";
    bool first = true;
#define SUB0_RUN_CONFIG_WRITE(type, name, def) \
    os << (first ? "" : ",\n") << "  \"" #name "\": "; detail::json_write_field(os, #name, c.name); first = false;
    SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_WRITE)
#undef SUB0_RUN_CONFIG_WRITE
    os << "\n}\n";
}

// Reads <dir>/config.json into `c`. Returns false if the file is missing or fails to parse -- an
// OLD model directory predating this feature has no config.json at all, and callers must treat
// that as "no persisted recipe available" (fall back to whatever they'd otherwise do), not an
// error. Implemented in src/run_config.cpp via simdjson::ondemand (single forward pass over the
// object's keys, one branch per SUB0_RUN_CONFIG_FIELDS row -- this project's JSON convention:
// register handlers, don't build a DOM tree) so declaring it here keeps this header simdjson-free
// for every consumer that only ever calls it, never simdjson types directly.
[[nodiscard]] SUB0_API bool read_config_json(RunConfig& c, const std::filesystem::path& dir);

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

// The identity-encoding directory for a model (no I/O). `datetime_tag` identifies this specific
// training ATTEMPT (see `now_datetag()`) -- it is NOT a compatibility check; that's `compatible()`
// below, driven by dims/flags alone.
inline std::filesystem::path model_dir(const std::filesystem::path& models_root,
                                       const std::string& corpus, int d, int l, int h,
                                       int seq, int vocab, int ternary, int pos_enc,
                                       const std::string& datetime_tag, int gated_ffn = 0,
                                       int tied_embeddings = 0, int qk_norm = 0) {
    std::string name = "sub0llm_" + corpus_tag(corpus) +
                       "_d" + std::to_string(d) + "l" + std::to_string(l) + "h" + std::to_string(h) +
                       "sq" + std::to_string(seq) + "v" + std::to_string(vocab) +
                       (ternary ? "t" : "") + pos_tag(pos_enc) + (gated_ffn ? "g" : "") +
                       (tied_embeddings ? "w" : "") + (qk_norm ? "q" : "") +
                       "_" + (datetime_tag.empty() ? "unknown" : datetime_tag);
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

// A filesystem-safe date+time tag for directory names: no ':' or 'T' (Windows disallows ':' in
// filenames), still fixed-width and lexicographically sortable like `now_iso()`.
inline std::string now_datetag() {
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    return buf;
}

inline void write_meta(const std::filesystem::path& dir, const ModelMeta& m) {
    std::error_code ec; std::filesystem::create_directories(dir, ec);
    std::ofstream os(dir / "meta.txt", std::ios::trunc);
    if (!os) return;
    // Enum-valued fields print as NAMES (gated_ffn=on, optimizer=muon, pos_encoding=rope, ...) for
    // readability, reusing config.json's OWN int<->name mapping (detail::enum_name) rather than a
    // second, hand-rolled one -- the two files can never disagree about what a value means. Falls
    // back to the raw int for a field/value enum_name doesn't recognize, matching json_write_field's
    // own fallback exactly.
    auto field = [&os](const char* name, int v) {
        os << name << '=';
        if (const char* nm = detail::enum_name(name, v)) os << nm; else os << v;
        os << '\n';
    };
    os << "model=sub0llm\n"
       << "corpus="          << m.corpus    << "\n"
       << "d_model="         << m.d_model   << "\n"
       << "n_layers="        << m.n_layers  << "\n"
       << "n_heads="         << m.n_heads   << "\n"
       << "seq_len="         << m.seq_len   << "\n"
       << "vocab="           << m.vocab     << "\n";
    field("ternary",         m.ternary);
    field("pos_encoding",    m.pos_encoding);
    field("gated_ffn",       m.gated_ffn);
    field("tied_embeddings", m.tied_embeddings);
    field("qk_norm",         m.qk_norm);
    field("optimizer",       m.optimizer);
    os << "git_sha="         << m.git_sha   << "\n"
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
    // Enum-valued fields accept EITHER form: a name (written by this build) or a raw int (an older
    // meta.txt written before this fix) -- same dual-acceptance contract config.json's reader already
    // has, via the SAME detail::enum_parse mapping so the two files never drift apart.
    auto as_enum = [&](const char* field, const std::string& s) {
        int v = 0; return detail::enum_parse(field, s, v) ? v : as_int(s);
    };
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
        else if (k == "ternary")        m.ternary = as_enum("ternary", v);
        else if (k == "pos_encoding")   m.pos_encoding = as_enum("pos_encoding", v);
        else if (k == "gated_ffn")      m.gated_ffn = as_enum("gated_ffn", v);
        else if (k == "tied_embeddings") m.tied_embeddings = as_enum("tied_embeddings", v);
        else if (k == "qk_norm")        m.qk_norm = as_enum("qk_norm", v);
        else if (k == "optimizer")      m.optimizer = as_enum("optimizer", v);
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
