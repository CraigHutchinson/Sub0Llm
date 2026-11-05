---
name: single-responsibility-headers
description: "Standing rule for Sub0 libraries — one type/concept per header, grouped in coherent directories, umbrella headers for convenience; discoverability over fewer files"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 09648518-580b-4f7b-ba75-d6985e369c04
  modified: 2026-09-30T21:23:41.527Z
---

Split sources into small, single-responsibility headers grouped by concept, so a
reader can discover the available types from the file tree alone. Keep an umbrella
header per group (e.g. `fusion/executors.hpp` over `fusion/executors/*.hpp`) so
consumers still have one include.

**Why:** the user could not find `BasicWorld` inside Sub0ECS's 900-line `store.hpp`,
which also held Mask, Column, SidePool, Partition and meta helpers
(2026-09-30). The same complaint had already split `fusion/executors.hpp` and
`fusion/planner.hpp` into one file per alternative. The user asked to make this a
standing rule: "files having single responsibility, grouped coherent units".

**How to apply:** when a header holds more than one independently useful type, or
alternatives of one extension point, split it before landing (one executor, one
planner, one storage building block per file). Put internal helpers under `detail/`.
New code starts that way; don't wait for a complaint. Applies across the Sub0
family (Sub0ECS, Sub0Pipeline, Sub0Pub, ...). Related: [[boy-scout-rule-always-resolve-inconsistencies]].
