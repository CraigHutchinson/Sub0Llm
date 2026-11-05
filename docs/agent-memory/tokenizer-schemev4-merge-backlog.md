---
name: tokenizer-schemev4-merge-backlog
description: "DONE 2026-07-30 (merge 7792cbe + fix f0ae908) — schemeV4 landed on claude/llm-transformers-1bit-qf3t3i. Clean auto-merge, no conflicts; marker set UNCHANGED so the EOS fix survived. Found+fixed a silent cross-version load"
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T19:24:23.941Z
---

**LANDED 2026-07-30**, during the user-declared breaking-change window ("remove the need to backwards
compat -- lets make the breaking changes here"). Arms A-E are parked and will be **retrained**, not
resumed. Superseded the separate throughput-2 backlog item, which came along in the merge (it was NOT
already in mainline: the merge brought 12 commits, = 10 v2 + 2 throughput-2).

## How the merge actually went

**Clean auto-merge, zero conflicts** — none of the handoff's conflict-risk files (`casing.hpp`,
`src/tokenizer.cpp`, `tokenizer.hpp`, `node_frame.hpp`, `op_curriculum.hpp`, `configurator.cpp`)
collided, despite mainline being 45 commits past the merge-base.

**The predicted marker collision did NOT happen, and that is a real result rather than luck.** The
carried-forward marker-rendering fix (`1ea9f92`) was expected to break because "schemeV4 changes the
marker SET" — it does not. V4 changes the glue **TRANSITION RULE**; `casing.hpp` says so explicitly
("No marker/id change (n_base fixed)"). TOK_EOS/CAP/UP keep 256/257/258, `TOK_MARKER_COUNT - TOK_EOS`
stays 32, and `core.hpp` + `engine_core.cpp` are untouched by the merge. The EOS regression test passed
non-vacuously.

One working-tree hazard: `docs/TOKENIZER_V2_IDEAS.md` existed UNTRACKED locally (483-line earlier draft,
no IMPLEMENTED annotations) while the branch tracks a 559-line version. Merge would have refused. Draft
preserved at `<scratchpad>/pre-merge-backup/`.

## The defect the merge surfaced — [[caps-bit-nothing-reads-is-not-a-guard]] shape again

`deserialize()` read `kSchemeVersion` and **threw it away**. Because V4 kept magic, marker set and
`n_base` unchanged, a V3 `tokenizer.tok` passes every other guard and **loads**. It then falls back to
the "hardcoded glue floor" for its missing trailing table — but that floor IS `casing::glue_default`,
i.e. the V4 D1 rule. The encoder is V4 regardless. So a V3 vocab learned when `word,` cost a `TOK_JOIN`
gets driven by an encoder that emits none: **the learned merges stop matching the token stream**.
Nothing crashes; quality silently degrades. The old comment claimed such a file "still loads and encodes
identically" — it does not.

Fixed in `f0ae908`: exact-match on the version, **both directions**, with a regression test that rewrites
the version field of a REAL serialized file.

## What already worked (verified, do not re-solve)

`tools/configurator.cpp`'s reuse stamp **already carries `casing::kSchemeVersion`** (TOKSTAMP_VERSION 2),
so a re-configure regenerates BOTH `tokenizer.tok` and `corpus.tok`. Its own comment anticipates exactly
this hazard. No stale-corpus risk for the rerun.

## Gates (this tree's config, NOT the branch's 166/113425 — counts are config-dependent)

```
frontend  171 cases / 113471 assertions   (round-trip fuzz = the losslessness gate)
engine    132 cases / 44170069 assertions ("~[.]")
```

## Still open, carried forward

* **Truecasing "name" threshold** — 9.1% of vocab is short cased fragments, the root of
  [[repetition-is-a-proper-noun-defect]]. schemeV4 did NOT address this; it remains the live lead on
  repetition and is worth revisiting before the retrain.
* Design record: `docs/TOKENIZER_V2_IDEAS.md` (D1-D5, Points 2-4) + `docs/TOKENIZER_DESIGN.md` §4.
* Full pipeline smoke test (configure -> train -> eval -> report -> samples) before spending GPU-hours on
  the rerun, per [[caps-bit-nothing-reads-is-not-a-guard]].
