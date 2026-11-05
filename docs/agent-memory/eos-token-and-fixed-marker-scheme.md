---
name: eos-token-and-fixed-marker-scheme
description: "Corpus had NO end-of-document signal (model never trained on 'last real token -> what comes next'); traced to scripts/get_fineweb.py's blank-line-only separator + a lost <|endoftext|> in local tinystories.txt. Added TOK_EOS (fixed id 256) + made ALL marker ids constexpr-verified, not scanned"
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Root cause (2026-07-02), found via user's question about why generation loops/rambles instead of
stopping:** the tokenizer had NO end-of-document token. `doc_starts` (configurator-computed, used only
by the training-window sampler to avoid cross-document windows) is invisible to the model, and
`sample_window`'s document-boundary cap means the LAST trainable target inside any document is the
document's own final token -- so the model was never trained on the pair "last real content token ->
what comes next" AT ALL, for any document, ever. That's a structural gap, not a data-quality issue.

**Traced to the source:** `scripts/get_fineweb.py` joined documents with a bare `"\n\n"` -- a heuristic
that ALSO fires on ordinary mid-document paragraph breaks, so it can't function as an unambiguous
stop signal even if the model attended to it. Fetched the canonical `roneneldan/TinyStories` dataset
directly from HuggingFace to check: it uses a literal `<|endoftext|>` marker (21,989 occurrences in the
19MB valid split alone, one per story) -- the standard GPT-2/3 convention. Our LOCAL `data/tinystories.txt`
had this marker STRIPPED at some point before this session (no `get_tinystories.py` existed; the file
predates this work) -- confirming "this predates the current pipeline," not a novel design gap.

**Fix, shipped this session (not yet committed -- see the [[gen-decode-kv-cache-optimization]]-style
pending list below):**

1. **Extraction scripts** now emit literal `<|endoftext|>` between documents: `scripts/get_fineweb.py`
   (join changed from `"\n\n"` to `f"\n{EOS}\n"`) and a NEW `scripts/get_tinystories.py` (fetches the
   canonical HF dataset, which already has the marker -- V2-GPT4 variant recommended as the default,
   the dataset's own README calls the original's GPT-3.5 stories "of lesser quality"). Both tested.

2. **Tokenizer**: `TOK_EOS` added to casing.hpp. Placement went through real discussion (recorded
   here to save re-deriving it): first tried id 0 (mirrors "\0 = string terminator" convention the
   user raised), but that requires shifting the ENTIRE byte alphabet by +1 (byte b -> base id b+1),
   which breaks "token id < 256 == that literal byte, no offset" -- a property useful for tooling
   that inspects corpus.tok raw and for future id-range masking. **Settled on id 256** (the FIRST
   marker, right after the untouched 0..255 byte range) -- keeps the byte range byte-for-byte
   identical to before, EOS is still the lowest marker id. `encode_join` (tokenizer.cpp) detects the
   13-byte literal `<|endoftext|>` and collapses it to one `eos_id` token, mirroring exactly how
   `"\n\n"` -> `para_id` already worked. `detokenize_join` renders it back to the literal string
   (preserves the round-trip invariant); gen_stage.cpp's sampling loops check for `eos_id` BEFORE
   pushing so it's never printed, and just end early instead of running to a fixed token budget --
   the actual behavioral payoff.

3. **ALL fixed marker ids made constexpr-verified, not runtime-scanned (user's explicit follow-up
   instruction, generalizing the EOS-placement fix to every marker):** `learn()` no longer stores
   `add_marker(...)`'s return value as the id -- it assigns the matching `casing::TOK_*` constant
   directly (add_marker's call still pushes the code into base_symbol/expansion, which genuinely
   need populating; only the *_id assignment stopped being "whatever the append returned").
   `deserialize()` no longer scans all n_base entries matching codes to DISCOVER marker positions --
   it directly checks `base_symbol[256..270)` against the expected fixed sequence and REJECTS the
   file (return false) if it doesn't match, then assigns the *_id fields from the constants. Rationale
   (user's framing): these ids are determined by the TOKENIZER CODE VERSION, not the corpus -- treating
   them as "read from the corpus" was already technically true-by-construction but architecturally
   misleading, and now the code says what it means. A `static_assert`-equivalent runtime assert
   (`t.n_base == 270`) in learn() catches any future accidental reordering immediately.

4. **Serialization magic bumped "S0TZ" -> "S0TE".** REQUIRED, not optional: TOK_EOS's new value (256)
   numerically COINCIDES with the pre-EOS scheme's TOK_CAP (also 256, before this session's
   renumbering shifted CAP to 257). Without the bump, deserialize()'s verify-fixed-position check
   would misread an old file's CAP marker as EOS. The bump makes old/new formats cleanly
   distinguishable at the magic-number level -- a stale tokenizer.tok fails to load outright (loud,
   correct) rather than silently misinterpreting itself.

5. **Configurator doc-boundary detection** (tools/configurator.cpp) now prefers `eos_id` when present,
   OR'd with the old `nl_run>=2` heuristic as a fallback for any corpus still using bare blank lines.
   No explicit "mode" needed: the new extraction format uses single newlines around the marker (not
   `\n\n`), so the two signals never both fire on the same corpus -- whichever the corpus provides is
   what's used.

**NOT changed:** the raw NUL byte (0x00) stays its own ordinary token, distinct from eos_id -- a
stray NUL in messy/binary-contaminated corpus data is not the same event as an intentional document
boundary and must not be silently conflated with one (discussed and explicitly rejected as a design
option, even though "\0 = end" is the same intuition that motivated EOS's low id in the first place).

**BACKLOG (user, 2026-07-02): bit-masking for specialized tokens.** Now that the byte range [0,255]
maps 1:1 to base id with zero offset, `id < 256` is already a cheap, maskable "is this a raw byte"
test. Future direction: extend this so id-RANGE (or a few reserved high bits) cheaply classifies a
token's KIND (raw byte / fixed marker / learned piece) without a branch table -- potentially useful
for faster bulk classification of a token stream (SIMD-friendly masking instead of per-token
comparisons), and for tooling that wants to filter/mask specific token classes. TBD, not scoped yet.

**Follow-up bug found + fixed (same day, via the project's own dogfood test):** `detokenize_join` had
NO case at all for `eos_id` initially (round-trip fell through to a garbled default -- a bare space).
After adding the case, the dogfood round-trip (tests/tok_roundtrip_fuzz.cpp, tokenizes the project's
OWN source) still failed: several source files literally contain the string `<|endoftext|>` in
comments (this very memory's origin session's code), and the decode was dropping the pending implicit
space before the literal. Root cause: EOS is emitted like WORD content on encode (via
`tile_ws(..., inter_content=true)`, which can leave `dps=true` pending for one implicit space), but
the decode case wasn't checking `dps` before rendering -- unlike the general word path and
`odquote_id`, which both do. Fix: `if (dps) out += ' ';` before the literal, mirroring `odquote_id`'s
pattern exactly. See [[regression-test-on-reproducible-bug]] -- a pinned unit test was added for this
exact case (`tests/tok_lib_tests.cpp`, "round-trips the literal <|endoftext|> EOS marker"), not just
the fix, per the user's standing instruction.

**DRY refactor:** the doc-boundary scan (eos_id-preferred, `\n\n`-fallback) was inline in
`tools/configurator.cpp` only, so nothing could test it against a real tokenizer without invoking the
full configurator binary. Extracted to `sub0::tok::scan_doc_boundaries()` (tokenizer.hpp/.cpp, engine-
free) -- configurator.cpp now calls it instead of duplicating the loop.

**Mechanism tests added (user, 2026-07-02: "we should check the output actually emits the EOS token
and not [infer from output length]... test case to cover that the training corpus mechanism works...
ensure the model actually sees the EOS and the windowing doesn't just cut it off"):**
- `tests/tok_lib_tests.cpp` "training pipeline: EOS is a reachable, in-window target (not cut off)" --
  builds a REAL mini corpus with 2 documents via the real tokenizer + `scan_doc_boundaries` +
  `sample_window` (window.hpp), and asserts the actual token id at the sampled target position equals
  `t.eos_id` -- not just that the index arithmetic could reach a boundary (that was already covered
  generically, with synthetic indices, in engine_tests.cpp's older window-sampler tests).
- `tests/engine_tests.cpp` "gen's EOS stop condition: sample_token returns eos_id when the model
  favors it" -- uses the exact two calls gen_stage.cpp makes (`sub0::eos_token_id()`,
  `sub0::sample_token()`) with a crafted logits array so EOS dominates, and asserts the returned id
  literally equals eos_id across 20 trials. Proves the id-level mechanism, not a length inference.

**Follow-up gap found + fixed (2026-07-08, `a016d17`):** the EOS-stop guard above only ever reached
`gen_stage.cpp`'s decode loops. `preview_at()` (training previews + `report.txt`'s sample battery) and
`gen_self_stats()` (autotemp's coherence tuner), both in `train_stage.cpp`, independently reimplement
the same forward-loop-plus-`sample_token` pattern rather than calling through `gen_stage.cpp`, and
neither had picked up the guard -- so `report.txt` kept showing `<|endoftext|>`-chained rambling long
after `gen` itself was fixed. See [[tinystories-d448-plateau-result]] for the full user-hypothesis ->
root-cause -> live-verification writeup.

**STATUS:** built + tested clean on ts_gpu (tinystories, 2.2GB corpus refetched via the new
get_tinystories.py, 675M tokens) -- sub0_frontend_tests 1243/1243 assertions, sub0_tests 17520251/
17520252 (the one failure, "CUDA backward matches the CPU gradient with short padded windows", is
PRE-EXISTING and unrelated: cuda_tests.cpp/backend_cuda.cu were not touched this session, last commit
528c32a predates this work). A 300-step smoke-scale train+gen cycle on tinystories showed generations
of wildly varying length across seeds (consistent with EOS firing sometimes), but the real proof is
the two mechanism tests above, not output-length inference. Also re-fetching data/fineweb_edu.txt +
will derive data/fineweb_smoke.txt with EOS markers (old versions deleted, per explicit user
instruction to replace/erase them) -- background HF fetch, ~46GB/14 shards, hours; native/smoke_gpu
builds are blocked until it finishes (their corpus files were deleted to force a clean re-extract, not
an append). NOT YET committed to git.
