#pragma once

// schema.hpp — heap-free, extensible configuration via compile-time field reflection.
//
// A configuration "section" is a plain aggregate struct (value type, automatic
// lifetime — stack-friendly, no registry, no heap, no virtual) that exposes a single
// static reflection hook:
//
//   struct MySection {
//       std::int64_t embed_dim = 256;
//       bool         verbose   = false;
//       template <class Self, class V>
//       static constexpr void reflect(Self& self, V&& v) {
//           v("embed_dim", Scope::BuildTime, self.embed_dim, "residual width D");
//           v("verbose",   Scope::Runtime,   self.verbose,   "chatty logging");
//       }
//   };
//
// EVERY operation — CLI override, JSON load, JSON save, the config-SHA — is written
// ONCE here as a generic algorithm over that field list. A module EXTENDS the config
// by adding a member + one line to its own reflect(); it needs to touch nothing else
// (no parser, no serializer, no flag table). That is the "extended by the module
// itself via an API" requirement, satisfied without macros or RTTI.
//
// SCOPE is the build-time / run-time split: BuildTime fields are the model shape a
// future constexpr-specialized engine bakes in (and they alone form the config-SHA
// that tags such a build); Runtime fields are operational knobs (paths, threads,
// verbosity, eval cadence) that never change the engine binary. Keeping the tag on
// every field makes that split explicit and machine-checkable rather than a comment.
//
// Reading uses simdjson (behind config/json.hpp); writing is the dependency-free
// streamer below (simdjson does not emit JSON). Both are flat: field names are
// globally unique across sections, so the JSON object and CLI namespace stay flat
// and match the historical config.json / --flag layout.

#include <concepts>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>

#include "sub0llm/config/json.hpp"

namespace sub0llm::config {

enum class Scope : std::uint8_t {
    BuildTime,   // model shape — specialized into a constexpr engine; part of config_sha
    Runtime,     // operational knob — never affects the engine binary
};

enum class FieldKind : std::uint8_t { None, Bool, Int, Uint, Float, String };

// ── Per-type scalar plumbing (the only place each supported type is special-cased) ──
template <class T>
[[nodiscard]] constexpr FieldKind kind_of_type() noexcept {
    if      constexpr (std::same_as<T, bool>)        return FieldKind::Bool;
    else if constexpr (std::same_as<T, std::string>) return FieldKind::String;
    else if constexpr (std::floating_point<T>)       return FieldKind::Float;
    else if constexpr (std::is_signed_v<T>)          return FieldKind::Int;
    else                                             return FieldKind::Uint;   // unsigned integral
}

template <class T>
[[nodiscard]] bool assign_from_string(T& ref, std::string_view s) {
    try {
        if      constexpr (std::same_as<T, bool>)
            ref = (s == "1" || s == "true" || s == "True" || s == "on" || s == "yes");
        else if constexpr (std::same_as<T, std::string>) ref = std::string(s);
        else if constexpr (std::floating_point<T>)       ref = static_cast<T>(std::stod(std::string(s)));
        else if constexpr (std::is_signed_v<T>)          ref = static_cast<T>(std::stoll(std::string(s)));
        else                                             ref = static_cast<T>(std::stoull(std::string(s)));
        return true;
    } catch (...) { return false; }
}

template <class T>
void assign_from_json(T& ref, const JsonDoc& d, std::string_view key) {
    if      constexpr (std::same_as<T, bool>)        { if (auto v = d.boolean(key)) ref = *v; }
    else if constexpr (std::same_as<T, std::string>) { if (auto v = d.str(key))     ref = *v; }
    else if constexpr (std::floating_point<T>)       { if (auto v = d.f64(key))     ref = static_cast<T>(*v); }
    else if constexpr (std::is_signed_v<T>)          { if (auto v = d.i64(key))     ref = static_cast<T>(*v); }
    else                                             { if (auto v = d.u64(key))     ref = static_cast<T>(*v); }
}

inline void append_json_string(std::string& out, std::string_view s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                    out += std::format("\\u{:04x}", static_cast<unsigned char>(c));
                else out += c;
        }
    }
    out += '"';
}

template <class T>
void append_json_value(std::string& out, const T& v) {
    if      constexpr (std::same_as<T, bool>)        out += v ? "true" : "false";
    else if constexpr (std::same_as<T, std::string>) append_json_string(out, v);
    else if constexpr (std::floating_point<T>)       out += std::format("{:.9g}", v);
    else                                             out += std::format("{}", v);
}

// ── Visitors: each generic operation is one small stateful functor over the fields ──
namespace detail {

struct KindProbe {
    std::string_view want;
    FieldKind        found = FieldKind::None;
    template <class T>
    void operator()(std::string_view name, Scope, T&, std::string_view) {
        if (name == want) found = kind_of_type<T>();
    }
};

struct StringSetter {
    std::string_view want, val;
    bool             ok = false;
    template <class T>
    void operator()(std::string_view name, Scope, T& ref, std::string_view) {
        if (name == want) ok = assign_from_string(ref, val);
    }
};

struct BoolSetter {
    std::string_view want;
    bool             value, ok = false;
    template <class T>
    void operator()(std::string_view name, Scope, T& ref, std::string_view) {
        if (name == want) {
            if constexpr (std::same_as<T, bool>) { ref = value; ok = true; }
        }
    }
};

struct JsonLoader {
    const JsonDoc* doc;
    template <class T>
    void operator()(std::string_view name, Scope, T& ref, std::string_view) {
        assign_from_json(ref, *doc, name);
    }
};

struct JsonWriter {
    std::string* out;
    Scope        only;        // emit only fields of this scope …
    bool         all;         // … unless `all`, then every field
    bool         first = true;
    template <class T>
    void operator()(std::string_view name, Scope sc, const T& ref, std::string_view) {
        if (!all && sc != only) return;
        if (!first) *out += ",\n";
        first = false;
        *out += "  \"";
        *out += name;
        *out += "\": ";
        append_json_value(*out, ref);
    }
};

struct ShaFolder {
    std::uint64_t* h;
    Scope          only;
    static void fold(std::uint64_t& h, std::string_view s) noexcept {
        for (char c : s) { h ^= static_cast<unsigned char>(c); h *= 1099511628211ULL; }
    }
    template <class T>
    void operator()(std::string_view name, Scope sc, const T& ref, std::string_view) {
        if (sc != only) return;
        fold(*h, name);
        std::string v;
        append_json_value(v, ref);
        fold(*h, v);
    }
};

}  // namespace detail

// ── Generic algorithms over any Section (a type with static reflect(Self&, V&&)) ────

// Set a field by its (snake_case) name from a string value. Returns false if no field
// of that name exists or the value failed to parse.
template <class Section>
bool set_field(Section& s, std::string_view name, std::string_view value) {
    detail::StringSetter v{name, value};
    Section::reflect(s, v);
    return v.ok;
}

// Set a bool field by name (CLI presence / --no- negation). False if not a bool field.
template <class Section>
bool set_bool(Section& s, std::string_view name, bool value) {
    detail::BoolSetter v{name, value};
    Section::reflect(s, v);
    return v.ok;
}

// The kind of the named field (None if absent) — lets a CLI loop know whether a flag
// is a presence-bool (no value) or needs the next token as its value.
template <class Section>
[[nodiscard]] FieldKind field_kind(Section& s, std::string_view name) {
    detail::KindProbe v{name};
    Section::reflect(s, v);
    return v.found;
}

// Overwrite fields present in `doc`; absent keys keep their current value (so an old or
// partial config file degrades to the compiled-in defaults for whatever it omits).
template <class Section>
void load_json(Section& s, const JsonDoc& doc) {
    detail::JsonLoader v{&doc};
    Section::reflect(s, v);
}

// Serialize to a flat JSON object. `scope` selects which fields; `all=true` emits every
// field (the default for a full run_config.json), `all=false` emits only the given scope.
template <class Section>
[[nodiscard]] std::string to_json(const Section& s, bool all = true, Scope scope = Scope::Runtime) {
    std::string out = "{\n";
    detail::JsonWriter v{&out, scope, all};
    Section::reflect(s, v);
    out += "\n}\n";
    return out;
}

// FNV-1a over the (name,value) of every BuildTime field — the config_sha that, with the
// code_sha, tags a specialized engine build (same shape ⇒ same sha ⇒ reusable build).
template <class Section>
[[nodiscard]] std::uint64_t build_sha(const Section& s) {
    std::uint64_t h = 14695981039346656037ULL;
    detail::ShaFolder v{&h, Scope::BuildTime};
    Section::reflect(s, v);
    return h;
}

// Normalize a CLI flag ("--curriculum-k-step") to a field name ("curriculum_k_step").
[[nodiscard]] inline std::string flag_to_field(std::string_view flag) {
    while (!flag.empty() && flag.front() == '-') flag.remove_prefix(1);
    std::string s(flag);
    for (char& c : s) if (c == '-') c = '_';
    return s;
}

}  // namespace sub0llm::config
