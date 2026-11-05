---
name: json-reads-forward-ondemand-not-dom-facade
description: "prefer simdjson on-demand FORWARD parsing (register handlers upfront, single pass) over the random-access DOM facade; deprecate JsonDoc"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

JSON *reads* should use simdjson's on-demand API **directly** in a forward / SAX style:
register a typed handler per expected key UP FRONT, then walk the object **once** in
document order, dispatching. Do NOT design around random access / a DOM facade.

**Why:** on-demand single-pass is faster and more memory-efficient than building a DOM and
doing one `at_key` lookup per field; the random access isn't needed if you register what you
want before reading. User stated this directly and added a `// TODO: Important DEPRECATE`
to `include/sub0llm/config/json.hpp` (the `JsonDoc` DOM facade) on 2026-06-22.

**How to apply:**
- New code: copy the `JsonFields` pattern in `diffusion/chapters/ch32_tree_predictor/viz_server.cpp`
  (`.f64/.i64/.u64/.str(key, ref)` registration → `read(body)` one forward pass; unknown keys
  skipped, absent keys keep defaults). `simdjson::simdjson` is PUBLIC on sub0llm so its include
  propagates — include `<simdjson.h>` directly in the TU (keep it OUT of headers).
- Writes stay nlohmann / hand-built strings (simdjson does not emit JSON).
- Pending refactor: deprecate `sub0llm::config::JsonDoc` (DOM `at_key` per field via the
  `schema.hpp` reflection loader) → a forward `JsonReader` that registers per-field typed sinks
  from `Section::reflect`, then does one on-demand pass. Touches json.hpp/cpp, schema.hpp
  (JsonLoader→registrar), run_config.cpp, test_config.cpp. Relates to [[simdjson-migration-backlog]].
