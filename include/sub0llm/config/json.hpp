#pragma once

// json.hpp — a tiny random-access JSON *reader* facade over simdjson.
//
// The config module needs to look scalar values up by key, in any order, possibly
// several times (defaults → file → CLI layering reads the file once but the field
// table visits keys in struct order). simdjson's on-demand API is single-pass /
// forward-only, so this facade uses simdjson's DOM parser, which permits repeated
// random key access. simdjson is the *only* JSON dependency pulled in by config —
// and it lives entirely behind this facade (declared here, defined in src/config/
// json.cpp) so no header in the config module includes simdjson. Writing JSON is
// handled separately by the dependency-free streamer in schema.hpp (simdjson does
// not emit JSON).
//
// Lifetime: a JsonDoc owns the parsed buffer (one allocation at parse time — config
// is loaded ONCE at startup, never on the hot path, so this is not the heap-pressure
// the embedded buffers care about). Move-only. All getters return std::nullopt when
// the key is absent or holds the wrong type, so a partial/old config file degrades
// gracefully (missing keys keep their compiled-in defaults).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace sub0llm::config {

class JsonDoc {
public:
    // Parse from an in-memory string (copied/padded internally) or a file. Returns
    // std::nullopt on a parse error or unreadable file (caller falls back to defaults).
    [[nodiscard]] static std::optional<JsonDoc> parse(std::string text);
    [[nodiscard]] static std::optional<JsonDoc> parse_file(const std::filesystem::path& path);

    JsonDoc(JsonDoc&&) noexcept;
    JsonDoc& operator=(JsonDoc&&) noexcept;
    JsonDoc(const JsonDoc&) = delete;
    JsonDoc& operator=(const JsonDoc&) = delete;
    ~JsonDoc();

    // Scalar lookups by key on the root object. nullopt = absent or type-mismatched.
    [[nodiscard]] std::optional<std::int64_t>  i64(std::string_view key) const;
    [[nodiscard]] std::optional<std::uint64_t> u64(std::string_view key) const;
    [[nodiscard]] std::optional<double>        f64(std::string_view key) const;
    [[nodiscard]] std::optional<bool>          boolean(std::string_view key) const;
    [[nodiscard]] std::optional<std::string>   str(std::string_view key) const;

    [[nodiscard]] bool has(std::string_view key) const;

private:
    JsonDoc();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sub0llm::config
