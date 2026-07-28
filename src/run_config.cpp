// run_config.cpp — implementation of sub0::registry::read_config_json (see registry.hpp's
// RunConfig doc comment for the "why"). simdjson::ondemand, single forward pass over the object's
// keys: this project's JSON convention is registered handlers over a DOM tree, matching how sparse
// config.json is (one flat object, no arrays/nesting).

#include "sub0/registry.hpp"

#include <simdjson.h>

#include <cstdlib>

namespace sub0::registry {

namespace {
// One overload per RunConfig field type; SUB0_RUN_CONFIG_READ below picks the right one purely by
// overload resolution on `c.name`'s actual type, so the macro itself never needs to know the type.
bool json_read(simdjson::ondemand::value v, std::string& out) {
    std::string_view sv;
    if (v.get_string().get(sv)) return false;
    out.assign(sv);
    return true;
}
bool json_read(simdjson::ondemand::value v, int& out) {
    int64_t i;
    if (v.get_int64().get(i)) return false;
    out = static_cast<int>(i);
    return true;
}
bool json_read(simdjson::ondemand::value v, unsigned& out) {
    uint64_t u;
    if (v.get_uint64().get(u)) return false;
    out = static_cast<unsigned>(u);
    return true;
}
bool json_read(simdjson::ondemand::value v, double& out) {
    double d;
    if (v.get_double().get(d)) return false;
    out = d;
    return true;
}
bool json_read(simdjson::ondemand::value v, long long& out) {
    int64_t i;
    if (v.get_int64().get(i)) return false;
    out = i;
    return true;
}
bool json_read(simdjson::ondemand::value v, unsigned long long& out) {
    uint64_t u;
    if (v.get_uint64().get(u)) return false;
    out = u;
    return true;
}

// One RunState field, dispatched by NAME, mirroring registry.hpp's json_write_state_field so the two
// halves special-case the same field for the same reason. `arch_id` is a hex STRING on disk (a 64-bit
// identity, not a quantity -- see the writer), so it is parsed from text rather than read as a number.
template <class T>
bool json_read_state_field(simdjson::ondemand::value v, std::string_view, T& out) {
    return json_read(v, out);
}
inline bool json_read_state_field(simdjson::ondemand::value v, std::string_view field,
                                  unsigned long long& out) {
    if (field != "arch_id") return json_read(v, out);
    std::string hex;
    if (!json_read(v, hex)) return false;
    out = std::strtoull(hex.c_str(), nullptr, 16);
    return true;
}

// Field-aware read: a non-int field defers to the type-dispatched json_read above; an INT field accepts
// EITHER a number OR a named enum string (pos_encoding="rope", optimizer="muon", gated_ffn="on", ...), so a
// named config.json AND an old numeric one both load. Peek the JSON type first (ondemand values are
// single-consume, so we must not speculatively get_int64 then get_string).
template <class T> bool json_read_field(simdjson::ondemand::value v, std::string_view, T& out) { return json_read(v, out); }
inline bool json_read_field(simdjson::ondemand::value v, std::string_view field, int& out) {
    simdjson::ondemand::json_type t;
    if (v.type().get(t)) return false;
    if (t == simdjson::ondemand::json_type::number) return json_read(v, out);
    if (t == simdjson::ondemand::json_type::string) {
        std::string_view sv;
        if (v.get_string().get(sv)) return false;
        return sub0::registry::detail::enum_parse(field, sv, out);
    }
    return false;
}
}  // namespace

bool read_config_json(RunConfig& c, const std::filesystem::path& dir) {
    simdjson::padded_string json;
    if (simdjson::padded_string::load((dir / "config.json").string()).get(json)) return false;

    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc)) return false;
    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj)) return false;

    for (auto field : obj) {
        std::string_view key;
        if (field.unescaped_key().get(key)) continue;   // a malformed key: skip this field, keep going
#define SUB0_RUN_CONFIG_READ(type, name, def) \
        if (key == #name) { json_read_field(field.value(), #name, c.name); continue; }
        SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_READ)
#undef SUB0_RUN_CONFIG_READ
        // An unrecognized key (a field from a future version of this schema, or a stray edit) is
        // silently ignored, so a newer writer's file still loads here.
    }
    return true;
}

// Reads state.json + config.json into one ModelMeta -- see the declaration in registry.hpp.
//
// The architecture and recipe fields come from config.json ALONE. The meta.txt this replaced carried
// its own copies of fifteen of them, and that copy was the one that rotted: it never learned
// n_kv_heads, LoopSplit's schedule or the rope parameters, because those were added to the
// SUB0_RUN_CONFIG_FIELDS X-macro (which generates config.json) and nothing updated the hand-written
// second writer. One writer per field is the fix; a more careful second writer is not.
bool read_state(const std::filesystem::path& dir, ModelMeta& m) {
    simdjson::padded_string json;
    if (simdjson::padded_string::load((dir / "state.json").string()).get(json)) return false;

    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc)) return false;
    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj)) return false;

    for (auto field : obj) {
        std::string_view key;
        if (field.unescaped_key().get(key)) continue;
        simdjson::ondemand::value v;
        if (field.value().get(v)) continue;
        // Generated from the SAME SUB0_RUN_STATE_FIELDS list write_state emits, so a field cannot be
        // written and then not read back (or vice versa) -- which is exactly what a second,
        // hand-maintained list eventually gets wrong.
#define SUB0_RUN_STATE_READ(type, name, def) \
        if (key == #name) { json_read_state_field(v, #name, m.name); continue; }
        SUB0_RUN_STATE_FIELDS(SUB0_RUN_STATE_READ)
#undef SUB0_RUN_STATE_READ
        // Unrecognized keys ignored, same forward tolerance read_config_json has.
    }

    // Architecture + recipe: config.json is the only source. A directory without one is incomplete
    // (an interrupted first run) rather than invalid -- the state above is still worth listing, so
    // the fields below just stay at their ModelMeta defaults.
    RunConfig c;
    if (read_config_json(c, dir)) {
        m.corpus          = c.corpus;
        m.d_model         = c.d_model;
        m.n_layers        = c.n_layers;
        m.n_heads         = c.n_heads;
        m.seq_len         = c.seq_len;
        m.vocab           = c.vocab;
        m.ternary         = c.ternary;
        m.pos_encoding    = c.pos_encoding;
        m.gated_ffn       = c.gated_ffn;
        m.tied_embeddings = c.tied_embeddings;
        m.qk_norm         = c.qk_norm;
        m.optimizer       = c.optimizer;
        m.batch           = c.batch;
        m.lr              = c.lr;
        m.seed            = c.seed;
    }
    m.dir = dir;
    return true;
}

}  // namespace sub0::registry
