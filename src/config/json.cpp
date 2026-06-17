// json.cpp — simdjson-backed implementation of the config JSON reader facade.
// This is the ONLY translation unit that includes simdjson; everything else in the
// config module depends only on json.hpp's plain-scalar interface.

#include "sub0llm/config/json.hpp"

#include <fstream>
#include <sstream>
#include <utility>

#include <simdjson.h>

namespace sub0llm::config {

struct JsonDoc::Impl {
    // The parser owns its internal buffers and the document references the source text,
    // so both must outlive every getter call — hence held together here, parsed once.
    simdjson::dom::parser parser;
    std::string           source;   // padded copy kept alive for the document's lifetime
    simdjson::dom::element root;
    bool                   is_object = false;
};

JsonDoc::JsonDoc() : impl_(std::make_unique<Impl>()) {}
JsonDoc::JsonDoc(JsonDoc&&) noexcept = default;
JsonDoc& JsonDoc::operator=(JsonDoc&&) noexcept = default;
JsonDoc::~JsonDoc() = default;

std::optional<JsonDoc> JsonDoc::parse(std::string text) {
    JsonDoc doc;
    doc.impl_->source = std::move(text);
    auto result = doc.impl_->parser.parse(doc.impl_->source);
    if (result.error()) return std::nullopt;
    doc.impl_->root = result.value();
    doc.impl_->is_object = doc.impl_->root.is_object();
    return doc;
}

std::optional<JsonDoc> JsonDoc::parse_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse(std::move(ss).str());
}

bool JsonDoc::has(std::string_view key) const {
    if (!impl_->is_object) return false;
    auto e = impl_->root.at_key(key);
    return !e.error();
}

std::optional<std::int64_t> JsonDoc::i64(std::string_view key) const {
    if (!impl_->is_object) return std::nullopt;
    auto e = impl_->root.at_key(key);
    if (e.error()) return std::nullopt;
    if (std::int64_t v; !e.get(v)) return v;
    // Permit an unsigned value that fits, and a float with no fractional part (configs
    // round-tripped through some writers stringify integers as 64.0).
    if (std::uint64_t u; !e.get(u) && u <= static_cast<std::uint64_t>(INT64_MAX))
        return static_cast<std::int64_t>(u);
    if (double d; !e.get(d) && d == static_cast<double>(static_cast<std::int64_t>(d)))
        return static_cast<std::int64_t>(d);
    return std::nullopt;
}

std::optional<std::uint64_t> JsonDoc::u64(std::string_view key) const {
    if (!impl_->is_object) return std::nullopt;
    auto e = impl_->root.at_key(key);
    if (e.error()) return std::nullopt;
    if (std::uint64_t v; !e.get(v)) return v;
    if (std::int64_t s; !e.get(s) && s >= 0) return static_cast<std::uint64_t>(s);
    if (double d; !e.get(d) && d >= 0.0 && d == static_cast<double>(static_cast<std::uint64_t>(d)))
        return static_cast<std::uint64_t>(d);
    return std::nullopt;
}

std::optional<double> JsonDoc::f64(std::string_view key) const {
    if (!impl_->is_object) return std::nullopt;
    auto e = impl_->root.at_key(key);
    if (e.error()) return std::nullopt;
    if (double v; !e.get(v)) return v;
    if (std::int64_t s; !e.get(s)) return static_cast<double>(s);
    if (std::uint64_t u; !e.get(u)) return static_cast<double>(u);
    return std::nullopt;
}

std::optional<bool> JsonDoc::boolean(std::string_view key) const {
    if (!impl_->is_object) return std::nullopt;
    auto e = impl_->root.at_key(key);
    if (e.error()) return std::nullopt;
    if (bool v; !e.get(v)) return v;
    return std::nullopt;
}

std::optional<std::string> JsonDoc::str(std::string_view key) const {
    if (!impl_->is_object) return std::nullopt;
    auto e = impl_->root.at_key(key);
    if (e.error()) return std::nullopt;
    if (std::string_view v; !e.get(v)) return std::string(v);
    return std::nullopt;
}

}  // namespace sub0llm::config
