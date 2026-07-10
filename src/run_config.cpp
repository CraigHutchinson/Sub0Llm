// run_config.cpp — implementation of sub0::registry::read_config_json (see registry.hpp's
// RunConfig doc comment for the "why"). simdjson::ondemand, single forward pass over the object's
// keys: this project's JSON convention is registered handlers over a DOM tree, matching how sparse
// config.json is (one flat object, no arrays/nesting).

#include "sub0/registry.hpp"

#include <simdjson.h>

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
        if (key == #name) { json_read(field.value(), c.name); continue; }
        SUB0_RUN_CONFIG_FIELDS(SUB0_RUN_CONFIG_READ)
#undef SUB0_RUN_CONFIG_READ
        // An unrecognized key (a field from a future version of this schema, or a stray edit) is
        // silently ignored -- same forward-compatible tolerance registry.hpp's read_meta already
        // has for meta.txt.
    }
    return true;
}

}  // namespace sub0::registry
