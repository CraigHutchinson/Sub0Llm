#pragma once

// trace_json.hpp (Ch32 Viz, Phase A) — serialise a GenerationTrace to JSON for the web scrubber.
// Kept out of the header-template sampler so nlohmann/json stays out of the hot path.

#include "sub0diff/viz/trace.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace sub0diff::viz {

// Serialise `trace` to a JSON string. token_str(id) → the display string for a token id (decoded word);
// positions equal to mask_id render as a mask placeholder. Used by both the file writer and the
// Phase-C server endpoint (which returns it directly over HTTP).
std::string serialize_trace_json(const GenerationTrace& trace,
                                 const std::function<std::string(std::int32_t)>& token_str,
                                 std::int32_t mask_id);

// Write `trace` to `path` as JSON (calls serialize_trace_json). Returns the number of frames written.
std::size_t write_trace_json(const GenerationTrace& trace, const std::string& path,
                             const std::function<std::string(std::int32_t)>& token_str,
                             std::int32_t mask_id);

}  // namespace sub0diff::viz
