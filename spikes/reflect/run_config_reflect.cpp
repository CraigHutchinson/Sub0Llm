// run_config_reflect.cpp -- THROWAWAY SPIKE. Does P2996 reflection actually delete registry.hpp's
// serialization boilerplate, and what does the result read like?
//
// NOT built by CMake and NOT on any include path. It needs a compiler this project does not use --
// see README.md in this directory. Nothing in src/ or include/ may depend on it. Per AGENTS.md §11,
// a spike either gets merged into the mainline or retired; it does not sit here accruing rot.
//
// ---------------------------------------------------------------------------------------------
// WHAT IS BEING REPLACED
//
// registry.hpp today declares each RunConfig field FOUR times, via one X-macro:
//
//   #define SUB0_RUN_CONFIG_FIELDS(X)        \
//       X(std::string, corpus,          "")  \
//       X(int,         d_model,         0)   \
//       ... 24 rows ...
//
//   struct RunConfig { SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_DECL) };       // 1. the members
//   ... SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_WRITE)  in write_config_json  // 2. the writer
//   ... SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_READ)   in read_config_json   // 3. the reader
//   ... enum_name/enum_parse tables keyed by field NAME as a string           // 4. enum display
//
// The X-macro is genuinely good: it already makes those one source of truth, and it is the reason
// config.json never fell behind on n_kv_heads / LoopSplit / rope the way the old meta.txt did.
// So this spike is NOT fixing a correctness gap. The questions it exists to answer are narrower:
//
//   Q1. Does reflection remove the macro without giving up the single-source-of-truth property?
//   Q2. Is the result more readable than the X-macro, or merely different?
//   Q3. Does it also subsume registry.hpp's OTHER hand-written field list -- write_state /
//       read_state's 9 fields -- which the X-macro does NOT cover and which therefore CAN drift?
//
// Q3 is the real prize. The X-macro already solved config.json. state.json is still hand-listed in
// two places, and that is exactly the shape of the bug this codebase already ate once.
// ---------------------------------------------------------------------------------------------

#include <meta>

#include <charconv>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace spike {

// --- the struct, declared ONCE, as a plain struct ------------------------------------------------
// This is the whole point: no macro, no field list, no registration. Adding a field here is the
// entire edit -- the writer and the reader below pick it up with no second place to remember.
struct RunConfig {
    std::string corpus         = "";
    int         d_model        = 0;
    int         n_layers       = 0;
    int         n_heads        = 0;
    int         n_kv_heads     = 0;
    int         loop_middle    = 0;
    int         loop_repeats   = 1;
    int         rope_scaling   = 0;
    double      rope_scale_fac = 1.0;
    double      rope_theta     = 10000.0;
    int         seq_len        = 0;
    int         vocab          = 0;
    int         ternary        = 0;
    int         pos_encoding   = 0;
    int         gated_ffn      = 0;
    int         tied_embeddings = 0;
    int         qk_norm        = 0;
    int         optimizer      = 0;
    int         batch          = 0;
    double      lr             = 0.0;
    unsigned    seed           = 0u;
    double      corpus_fraction = 1.0;
    unsigned    subset_seed    = 0u;
    int         has_blend_schedule = 0;
    int         config_schema  = 3;
};

// And the OTHER list, the one the X-macro never covered (Q3). Today these 9 fields are written by
// hand in write_state and read by hand in read_state -- two places, no compiler check that they
// agree. Under reflection they are just a struct, and get the same treatment for free.
struct RunState {
    std::string   git_sha = "", created = "", updated = "", status = "";
    std::uint64_t arch_id = 0;
    long long     steps = 0;
    double        epochs = 0.0;
    long long     tokens_seen = 0;
    double        best_val_nelbo = -1.0;
};

// --- scalar leaves: the only thing that stays hand-written, one overload per TYPE, not per FIELD --
inline void emit(std::ostream& os, const std::string& v) {
    os << '"';
    for (char c : v) {
        if      (c == '"' ) os << "\\\"";
        else if (c == '\\') os << "\\\\";
        else if (c == '\n') os << "\\n";
        else                os << c;
    }
    os << '"';
}
inline void emit(std::ostream& os, int v)           { os << v; }
inline void emit(std::ostream& os, unsigned v)      { os << v; }
inline void emit(std::ostream& os, long long v)     { os << v; }
inline void emit(std::ostream& os, double v)        { os << v; }
inline void emit(std::ostream& os, std::uint64_t v) { os << v; }

inline bool parse_scalar(std::string_view s, std::string& out) { out.assign(s); return true; }
template <class T>
inline bool parse_scalar(std::string_view s, T& out) {
    return std::from_chars(s.data(), s.data() + s.size(), out).ec == std::errc{};
}

// --- the generic writer: ONE function, every struct, every field -------------------------------
// Compare against SUB0_RUN_CONFIG_WRITE + the 24-row macro list it expands over. This replaces
// both, and it is not specific to RunConfig -- RunState gets it at no extra cost, which is Q3.
template <class T>
std::string to_json(const T& obj) {
    std::ostringstream os;
    os << "{\n";
    bool first = true;
    // `template for` is an EXPANSION statement (P1306): the body is instantiated once per member,
    // with `mem` a constant in each instantiation -- so `obj.[:mem:]` below has a distinct, concrete
    // type per iteration. A plain runtime for-loop could not do this; that is why the X-macro was
    // needed at all.
    template for (constexpr auto mem : std::define_static_array(
                      std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::current()))) {
        if (!first) os << ",\n";
        first = false;
        os << "  \"" << std::meta::identifier_of(mem) << "\": ";
        emit(os, obj.[:mem:]);
    }
    os << "\n}\n";
    return os.str();
}

// --- the generic reader ------------------------------------------------------------------------
// Replaces SUB0_RUN_CONFIG_READ. Same forward-tolerance contract as the real reader: an
// unrecognized key is ignored rather than failing the parse, so a newer writer's file still loads.
template <class T>
bool from_json_field(T& obj, std::string_view key, std::string_view value) {
    bool matched = false;
    template for (constexpr auto mem : std::define_static_array(
                      std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::current()))) {
        if (!matched && key == std::meta::identifier_of(mem)) {
            matched = parse_scalar(value, obj.[:mem:]);
        }
    }
    return matched;
}

// --- what the X-macro could NOT do --------------------------------------------------------------
// A compile-time field count and name list. Today `describe_config()`'s banner and meta/config
// consistency are maintained by eye; here they are derivable. This is the argument that reflection
// is more than a macro replacement: it makes the field list QUERYABLE, not just expandable.
template <class T>
consteval std::size_t field_count() {
    return std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::current()).size();
}

}  // namespace spike

int main() {
    static_assert(spike::field_count<spike::RunConfig>() == 25, "RunConfig field count drifted");
    static_assert(spike::field_count<spike::RunState>()  == 9,  "RunState field count drifted");

    spike::RunConfig c;
    c.corpus = "tinystories";
    c.d_model = 448; c.n_layers = 11; c.n_heads = 7; c.n_kv_heads = 4;
    c.seq_len = 256; c.vocab = 16517; c.batch = 385; c.lr = 0.00693722; c.seed = 42u;
    c.corpus_fraction = 0.25; c.subset_seed = 7u;
    std::cout << "--- config.json (generated, zero per-field code) ---\n" << spike::to_json(c);

    spike::RunState s;
    s.git_sha = "8a01fff"; s.status = "plateaued";
    s.arch_id = 0xbf940c712c7017f0ULL; s.steps = 12236; s.best_val_nelbo = 1.19023;
    std::cout << "\n--- state.json (SAME two functions, no new code at all) ---\n" << spike::to_json(s);

    // Round-trip one field through the generic reader.
    spike::RunConfig back;
    const bool ok1 = spike::from_json_field(back, "d_model", "448");
    const bool ok2 = spike::from_json_field(back, "corpus_fraction", "0.25");
    const bool ok3 = spike::from_json_field(back, "no_such_field", "1");   // must be ignored, not fatal
    std::cout << "\nread d_model=" << back.d_model << " (ok=" << ok1 << ")"
              << "  corpus_fraction=" << back.corpus_fraction << " (ok=" << ok2 << ")"
              << "  unknown-key rejected=" << !ok3 << "\n";
    return 0;
}
