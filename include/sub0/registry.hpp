// registry.hpp -- model storage layout + a self-describing model index.
//
// Every trained model lives in its own directory under a models root, named for its
// IDENTITY -- corpus + architecture dims + a date+time tag for the training attempt:
//   models/sub0llm_<corpus>_d<D>l<L>h<H>sq<SEQ>v<VOCAB>[t][r]_<YYYYMMDD-HHMMSS>/
//     model.bin        the parameters
//     model.bin.ckpt   the resumable optimizer/loop state
//     config.json      the architecture + the training recipe (RunConfig, below)
//     state.json       the run's provenance + progress (ModelMeta, below)
//
// config.json and state.json partition the metadata; they do not overlap. Every architecture and
// recipe value has exactly one writer (config.json, generated from SUB0_RUN_CONFIG_FIELDS) and every
// provenance/progress value has exactly one writer (state.json). `read_state` merges the two into one
// ModelMeta for display. The meta.txt these replaced duplicated FIFTEEN of config.json's fields, and
// the duplicate is the copy that went stale -- it never learned n_kv_heads, LoopSplit's schedule or
// the rope parameters, because those arrived through the X-macro and nothing updated the second,
// hand-written writer.
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
// instead of resuming it. The SHA is still recorded in state.json as a provenance/audit field (see
// ModelMeta::git_sha) -- just no longer part of directory identity or resume-matching.
//
// The "registry" is just the set of state.json files: discovery scans them (no separate
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
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// Export macro for this header's out-of-line half -- `read_config_json` and `read_state`, both
// defined in src/run_config.cpp, which is compiled into sub0_FRONTEND, not the engine. Everything
// else here is `inline` and lands in each consumer's own translation unit; these two are the symbols
// that cross a library boundary (they use simdjson, which this header must not pull in).
//
// The frontend is the right home precisely because registry is frontend surface: sub0_frontend_tests
// links it WITHOUT the engine, so a reader living in sub0_core would be unreachable from the very
// tests meant to cover it -- which is what forced the readers to stay inline in this header before.
//
// A self-contained copy of core.hpp's own SUB0_API (not an include of core.hpp itself) -- this
// header is deliberately pure std + <filesystem>, no engine dependency, matching its doc comment.
#ifndef SUB0_API
  #if defined(_WIN32)
    #define SUB0_API __declspec(dllexport)
  #else
    #define SUB0_API __attribute__((visibility("default")))
  #endif
#endif

namespace sub0::registry {

// --- RunState: the single source of truth for state.json, exactly as RunConfig is for config.json --
//
// One row per field, expanded three ways below (members, writer, reader) -- so the writer and the
// reader cannot disagree, and adding a field is one edit.
//
// This list used to be hand-written in BOTH write_state and read_state, which is the same shape of
// bug as the meta.txt/config.json duplication those two replaced: a second copy of a field list is
// the copy that goes stale. config.json was already safe (this X-macro pattern is why it learned
// n_kv_heads / LoopSplit / rope automatically); state.json was not, until now.
//
// `arch_id` is serialized as a hex STRING rather than a JSON number -- see write_state -- so it is
// special-cased by NAME in the field helpers, exactly as the enum-valued RunConfig fields already
// are. Same precedent, no new mechanism.
#define SUB0_RUN_STATE_FIELDS(X)                   \
    X(std::string,        git_sha,        "")      \
    X(std::string,        created,        "")      \
    X(std::string,        updated,        "")      \
    X(std::string,        status,         "")      \
    X(unsigned long long, arch_id,        0ull)    \
    X(long long,          steps,          0)       \
    X(double,             epochs,         0.0)     \
    X(long long,          tokens_seen,    0)       \
    X(double,             best_val_nelbo, -1.0)

struct RunState {
#define SUB0_RUN_STATE_DECL(type, name, def) type name = def;
    SUB0_RUN_STATE_FIELDS(SUB0_RUN_STATE_DECL)
#undef SUB0_RUN_STATE_DECL
};

// One model's metadata, as `read_state` assembles it from BOTH files in the directory. It INHERITS
// the state.json fields rather than restating them, so `m.steps` still reads exactly as before at
// every call site while there remains exactly one declaration of what state.json contains.
//
// The fields declared here are the [config.json] half: merged in by read_state for display and
// filtering, never written back. To add an architecture or recipe field, add a
// SUB0_RUN_CONFIG_FIELDS row and a merge line in read_state; do NOT add a second writer.
struct ModelMeta : RunState {
    // [config.json] architecture + recipe, merged in for display and filtering only.
    std::string corpus;
    int d_model = 0, n_layers = 0, n_heads = 0, seq_len = 0, vocab = 0, ternary = 0;
    int pos_encoding = 0;                     // 0 = absolute learned (legacy default), 1 = RoPE
    int gated_ffn = 0;                        // 0 = plain GELU+bias FFN (legacy default), 1 = SwiGLU-gated
    int tied_embeddings = 0;                  // 0 = untied head (legacy default), 1 = head reuses tok_emb
    int qk_norm = 0;                          // 0 = no QK-norm (legacy default), 1 = per-head RMSNorm on Q/K
    int optimizer = 0;                        // 0 = AdamW (legacy default), 1 = Muon (hidden 2D matrices)
    // [config.json] what it was asked to do.
    int batch = 0;                            // minibatch size
    double lr = 0.0;                          // learning rate
    unsigned seed = 0;                        // RNG seed (for reproducibility)
    // Not from either file: filled in by scan()/read_state from the path it just read.
    std::filesystem::path dir;
    // NOTE: git_sha/created/updated/status/arch_id/steps/epochs/tokens_seen/best_val_nelbo are
    // inherited from RunState above -- declared once, in SUB0_RUN_STATE_FIELDS. `arch_id` in
    // particular is the ONLY thing compatible() consults; 0 means "no such field in state.json",
    // which it treats as incompatible rather than guessing from the merged config fields.
};

// --- RunConfig: the single source of truth for a model directory's TRAINING RECIPE -------------
//
// Born from a real incident: a GPU-fault auto-resume once silently continued training under plain
// AdamW instead of the Muon it had used for its first 7728 steps, because the optimizer choice was
// derived purely from the CURRENT invocation's CLI flag with no cross-check against what the model
// had actually been trained with -- `ModelMeta::optimizer` above was written to the old meta.txt on
// every run but never read back. `RunConfig` fixes the CLASS of bug, not just that one field: it is the one
// place a new training-recipe option is declared, and by construction that declaration is *also*
// its serialization -- add a row to SUB0_RUN_CONFIG_FIELDS below and the field is written to and
// read from every model directory's config.json automatically (write_config_json/read_config_json
// are generated from this one list, not hand-maintained in three separate places that could drift
// out of sync with each other, which is exactly how the optimizer bug happened in the first place).
//
// Scope, deliberately not "every setting in the program": holds CHOICES that must not silently
// drift across a resume and have no other authoritative home.
//   - Architecture dims/flags ARE included, even though they're also `constexpr` at compile time:
//     `compatible()` filters across build configs before any model.bin is opened, and this file is
//     where the values it reports come from. They are NOT duplicated elsewhere on disk -- ModelMeta
//     holds them only as a read-side merge target, populated by `read_state` FROM here.
//   - `batch`/`lr`/`seed` ARE included for completeness/comparison, but are NOT re-enforced on
//     resume here -- they already have a correct, tightly-coupled authoritative home in the
//     `.ckpt` binary (see train_stage.cpp's `load_checkpoint`), which wins on resume with its own
//     log line. config.json's copy is informational.
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
    X(int,         loop_middle,     0)    \
    X(int,         loop_repeats,    1)    \
    X(int,         depth_attn_stride, 0)  \
    X(int,         gdn_full_attn_stride, 0)  \
    X(int,         ngram_max_n,     0)    \
    X(int,         ngram_tables,    1)    \
    X(int,         ngram_table_size, 0)   \
    X(int,         rope_scaling,    0)    \
    X(double,      rope_scale_fac,  1.0)  \
    X(double,      rope_theta,  10000.0)  \
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
    X(double,      corpus_fraction, 1.0)  \
    X(unsigned,    subset_seed,     0u)   \
    X(int,         has_blend_schedule, 0)  \
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
inline void json_write(std::ostream& os, int v)                { os << v; }
inline void json_write(std::ostream& os, unsigned v)           { os << v; }
inline void json_write(std::ostream& os, double v)             { os << v; }
inline void json_write(std::ostream& os, long long v)          { os << v; }
inline void json_write(std::ostream& os, unsigned long long v) { os << v; }

// One RunState field, dispatched by NAME so the writer macro stays uniform -- the same shape
// json_write_field already uses to print the enum-valued RunConfig fields as names.
//
// `arch_id` is the only special case: it is emitted as a hex STRING rather than a JSON number,
// because it is a 64-bit IDENTITY rather than a quantity and a great many JSON readers land integers
// in a double, where exactness stops at 2^53 -- a silently truncated arch_id would make two different
// architectures compare equal. Hex also reads directly against the model directory's own _a<hex> tag,
// which is this value's low 32 bits.
template <class T>
void json_write_state_field(std::ostream& os, std::string_view, const T& v) { json_write(os, v); }
inline void json_write_state_field(std::ostream& os, std::string_view field, unsigned long long v) {
    if (field == "arch_id") {
        char buf[24];
        std::snprintf(buf, sizeof buf, "%016llx", v);
        json_write(os, std::string(buf));
    } else {
        json_write(os, v);
    }
}

// The enum-valued RunConfig fields print as NAMES (pos_encoding=rope, optimizer=muon, gated_ffn=on, ...)
// for readability; every other field stays numeric. The int<->name mapping lives HERE only; the reader
// (run_config.cpp) accepts EITHER form so old numeric config.json files still load. enum_name returns null
// for a non-enum field or an out-of-range value (-> numeric fallback).
inline const char* enum_name(std::string_view field, int v) {
    auto nth = [v](std::initializer_list<const char*> names) -> const char* {
        int i = 0; for (const char* n : names) if (i++ == v) return n; return nullptr; };
    if (field == "pos_encoding") return nth({"learned", "rope"});
    if (field == "rope_scaling") return nth({"none", "linear"});
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

// One-line human summary of a RunConfig: every field that DIFFERS FROM ITS DEFAULT, enums rendered by
// name. Driven by SUB0_RUN_CONFIG_FIELDS, exactly like write_config_json above, so a new row appears in
// every banner automatically.
//
// Why this exists: three tools hand-rolled their own summary with three different, incomplete field
// sets. The configure banner listed d/L/H/seq plus four flags and mentioned NONE of n_kv_heads,
// loop_middle, loop_repeats, rope_scaling or rope_scale_fac -- so all three variants of a GQA A/B
// printed a BYTE-IDENTICAL banner and "three variants" was indistinguishable from "the same build three
// times" on screen. A banner whose job is reporting what you configured must not be able to omit the
// axis under test; deriving it from the same X-macro that already drives config.json makes omission
// structurally impossible rather than a thing to remember.
//
// Non-default rather than all-fields keeps the common case short: an unset axis is exactly the one the
// reader does not need told about. `corpus` is skipped -- it is a long path every caller already prints
// separately. Fields still at their default (loop_repeats=1, config_schema=1, an unset batch/lr/seed at
// configure time) drop out on their own, no exclusion list needed.
inline std::string describe_config(const RunConfig& c) {
    std::string out;
    auto add_field = [&out](std::string_view name, const std::string& rendered) {
        if (!out.empty()) out += ' ';
        out += name; out += '='; out += rendered;
    };
#define SUB0_RUN_CONFIG_DESCRIBE(type, name, def)                                           if constexpr (std::string_view(#name) != "corpus") {                                        if (!(c.name == static_cast<type>(def))) {                                                  std::ostringstream os_;                                                                 detail::json_write_field(os_, #name, c.name);                                           std::string v_ = os_.str();                                                             if (v_.size() >= 2 && v_.front() == '"') v_ = v_.substr(1, v_.size() - 2);              add_field(#name, v_);                                                               }                                                                                   }
    SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_DESCRIBE)
#undef SUB0_RUN_CONFIG_DESCRIBE
    return out;
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
// `arch_id` (sub0::MODEL_ARCH_ID) is appended as a short hex tag. The readable part stays readable --
// it is what a human scans an `ls` for -- but it is a hand-picked SUBSET and always will be, because a
// name carrying every axis would be unreadable. The tag closes that gap: two builds differing in ANY
// architecture axis get different directories even when the readable part collides, which is precisely
// what a GQA-vs-MHA pair at identical dims used to do.
inline std::filesystem::path model_dir(const std::filesystem::path& models_root,
                                       const std::string& corpus, int d, int l, int h,
                                       int seq, int vocab, int ternary, int pos_enc,
                                       const std::string& datetime_tag, int gated_ffn = 0,
                                       int tied_embeddings = 0, int qk_norm = 0,
                                       unsigned long long arch_id = 0) {
    char tag[24] = {};
    if (arch_id) std::snprintf(tag, sizeof(tag), "_a%08x",
                               static_cast<unsigned>(arch_id & 0xffffffffull));
    std::string name = "sub0llm_" + corpus_tag(corpus) +
                       "_d" + std::to_string(d) + "l" + std::to_string(l) + "h" + std::to_string(h) +
                       "sq" + std::to_string(seq) + "v" + std::to_string(vocab) +
                       (ternary ? "t" : "") + pos_tag(pos_enc) + (gated_ffn ? "g" : "") +
                       (tied_embeddings ? "w" : "") + (qk_norm ? "q" : "") + tag +
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

// Writes <dir>/state.json: the run's PROVENANCE and PROGRESS, and nothing else.
//
// It deliberately does NOT repeat the architecture or the training recipe. Those live in config.json,
// which is generated from SUB0_RUN_CONFIG_FIELDS and is therefore complete by construction. The two
// files used to overlap on FIFTEEN fields (corpus, every dim, all five arch enums, optimizer, batch,
// lr, seed) -- and this was the copy that went stale: it never learned n_kv_heads, LoopSplit's
// schedule or the rope parameters, because adding a field to the X-macro updated config.json alone.
// A second writer of the same fact does not stay in sync; it picks a moment to disagree.
//
// JSON, like config.json and blend_schedule.json, so every file in a model directory reads the same way.
inline void write_state(const std::filesystem::path& dir, const ModelMeta& m) {
    std::error_code ec; std::filesystem::create_directories(dir, ec);
    std::ofstream os(dir / "state.json", std::ios::trunc);
    if (!os) return;
    // Generated from SUB0_RUN_STATE_FIELDS -- the same list read_state reads, so the two cannot
    // disagree about what state.json contains. `first` handles the comma so field ORDER can change
    // (or a field be added anywhere in the list) without anyone having to move a trailing comma.
    os << "{\n";
    bool first = true;
#define SUB0_RUN_STATE_WRITE(type, name, def)                       \
    if (!first) os << ",\n";                                        \
    first = false;                                                  \
    os << "  \"" #name "\": ";                                      \
    detail::json_write_state_field(os, #name, m.name);
    SUB0_RUN_STATE_FIELDS(SUB0_RUN_STATE_WRITE)
#undef SUB0_RUN_STATE_WRITE
    os << "\n}\n";
}

// Reads <dir>/state.json (provenance + progress) AND <dir>/config.json (the architecture and the
// recipe) into one ModelMeta, for display, filtering and compatibility. Two files, but exactly ONE
// writer per field -- so unlike the meta.txt this replaced, the two cannot disagree.
//
// Out-of-line in run_config.cpp for the same reason read_config_json is: simdjson lives there, and
// this header stays pure std + <filesystem>. Returns false when state.json is missing or unparsable
// -- a directory without one is not a model directory. A missing config.json is NOT a failure: an
// interrupted first run leaves real provenance worth listing, so the architecture fields simply stay
// at their defaults rather than being invented.
SUB0_API bool read_state(const std::filesystem::path& dir, ModelMeta& m);

// All models discovered under the root: every subdirectory holding a state.json. Keying on
// state.json (not config.json) is what keeps a merely CONFIGURED directory out of the listing --
// `sub0llm configure` writes config.json before anything has been trained.
inline std::vector<ModelMeta> scan(const std::filesystem::path& models_root) {
    std::vector<ModelMeta> out;
    std::error_code ec;
    if (!std::filesystem::exists(models_root, ec)) return out;
    for (const auto& e : std::filesystem::directory_iterator(models_root, ec)) {
        if (!e.is_directory()) continue;
        ModelMeta m;
        if (read_state(e.path(), m)) out.push_back(std::move(m));
    }
    return out;
}

// A model loads into the current build only if its architecture dims AND weight-meaning variants
// (ternary, positional-encoding scheme, gated-FFN scheme, tied-embeddings scheme, tokenizer scheme)
// match exactly -- a same-shape mismatch would load silently but compute nonsense (the JOIN
// tokenizer's token ids mean different text). This is a DIAGNOSTIC check (`models`/`models --prune`);
// the actual load-time gate is engine_core.cpp's binary Header comparison, which is authoritative
// regardless of what this says.
// arch_id ALONE decides. It covers every axis, including the ones the individual fields cannot see
// (n_kv_heads, LoopSplit's schedule, rope theta/scaling). Answering "compatible" for a model that
// load_model then refuses is worse than answering "no" -- it sends `train` into a resume that fails.
//
// A state.json without an arch_id is treated as INCOMPATIBLE rather than falling back to a per-field
// comparison. That fallback existed for files written before arch_id and is gone with the rest of the
// legacy-format support: a per-field match is exactly the wrong answer for the axes it cannot see, so
// keeping it would mean the weaker check silently deciding whichever cases the stronger one was added
// to catch. The remaining parameters stay for callers that only have loose dims to hand.
inline bool compatible(const ModelMeta& m, int, int, int, int, int, int,
                       int, int = 0, int = 0, int = 0,
                       unsigned long long arch_id = 0) {
    return arch_id != 0 && m.arch_id == arch_id;
}

}  // namespace sub0::registry
