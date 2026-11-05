---
name: arch-features-off-by-default-in-production
description: "RESOLVED (067e9d9, 2026-07-14): gated FFN / tied embeddings / QK-norm / Muon now default ON in tools/configurator.cpp (int gated_ffn=1 etc.) -- confirmed still current 2026-07-18. Pre-flip 27GB fineweb model deleted (2026-07-18, user confirmed). meta.txt named-values gap also RESOLVED (28391dc, 2026-07-18): now reuses config.json's existing enum_name/enum_parse mapping instead of raw ints."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**RESOLVED.** The original finding (2026-07-14, below, kept for context) was fixed the same day by commit
`067e9d9` "feat(config): default the proven architecture stack ON (gated FFN + tied embeddings + QK-norm +
Muon)" -- `tools/configurator.cpp` now declares `int gated_ffn = 1; int tie_embeddings = 1; int qk_norm = 1;`
with a comment citing this exact memory item, and `--optimizer`'s CLI default is `muon`. Re-verified live in
code on 2026-07-18 (not just trusting the commit message) while doing a defaults/constexpr-adherence audit --
still current, nothing has regressed it. `models/ce256_prod_fixed` (created 2026-07-16, after the flip) has
`gated_ffn=1 tied_embeddings=1 qk_norm=1 optimizer=1` in its meta.txt, confirming a real production run
actually inherited the new defaults, not just the configurator's own declared values.

**DELETED (2026-07-18, user confirmed)**: `models/sub0llm_fineweb_edu_d448l11h7sq512v49275r_20260713-215514`
(27GB, created 2026-07-13, BEFORE the flip) was the one PRE-flip production run described below -- freed
back to the user rather than silently kept around superseded. If fineweb_edu production training is
revisited, re-run with the current (already-correct) defaults, no flags needed.

**RESOLVED (commit `28391dc`, 2026-07-18, same day)**: turned out `config.json` ALREADY printed named
values (`"gated_ffn": "on"`) via `registry.hpp`'s existing `detail::enum_name`/`enum_parse`/
`json_write_field` machinery -- only `meta.txt`'s separate, hand-rolled `write_meta`/`read_meta` still
used raw ints, so the two files could describe the same model differently. Fixed by pointing
`write_meta`/`read_meta` at the SAME mapping (no new one invented) -- `meta.txt` now writes
`gated_ffn=on`/`optimizer=muon`/`pos_encoding=rope` etc., and reads back either the new name form or an
old raw-int meta.txt correctly (verified: existing round-trip test green, a real CLI smoke train's
meta.txt inspected directly, full suite green on both CPU-only and CUDA configs). My earlier claim that
config.json ALSO had this gap was wrong -- I'd only actually read meta.txt directly and extrapolated;
checking `models/ce256_prod_fixed/config.json` directly during this fix showed it was already correct.

---

### Original finding (2026-07-14), superseded above except for the two "still open" items

The production fineweb_edu d448 run (models/sub0llm_fineweb_edu_d448l11h7sq512v49275r_20260713-215514,
meta.txt) trained with **all the proven architecture wins turned OFF**:

```
gated_ffn        = 0
tied_embeddings  = 0
qk_norm          = 0
optimizer        = 0   (AdamW, not Muon)
```

Each of these is DONE, verified, and a measured win (see [[weight-tying-tied-embeddings]], [[qk-norm]],
[[cuda-parity-backlog-deferred]] gated-FFN ca4649c, [[muon-optimizer]] / [[muon-cuda-parity-in-progress]]),
but they defaulted OFF -- the compile-time ones (gated_ffn / tied_embeddings / qk_norm are baked at
configure/CMake as USE_GATED_FFN / USE_TIED_EMBEDDINGS / USE_QK_NORM) and the runtime optimizer defaulted to
`adamw` unless `--optimizer muon`. So the workflow build didn't enable them and the run was effectively
weaker than it should have been.

**Also (same task, still open): meta.txt / config.json should print NAMED values, not raw numbers.** `pos_encoding=1`,
`optimizer=0`, `gated_ffn=0` etc. are ambiguous -- a reader can't tell what `1` means. Emit human-readable
tags: `pos_encoding=rope` (vs sinusoidal/alibi/...), `optimizer=adamw` (vs muon), `gated_ffn=off/on` (or
true/false), `tied_embeddings`/`qk_norm`/`ternary` as on/off. The RunConfig X-macro is the single place to add
a per-field enum->string mapping for the read/write + the meta.txt/report formatting. Keep the on-disk
config.json parseable (either a string enum, or keep the int + also print the name in meta.txt/report).
