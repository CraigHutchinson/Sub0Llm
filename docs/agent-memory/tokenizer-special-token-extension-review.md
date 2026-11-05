---
name: tokenizer-special-token-extension-review
description: docs/TOKENIZER_REVIEW.md extended with prior-art-grounded design for the next special-token family (conversational turn markers, bracket glue) + verified 3 specific code claims (TOK_NEWLINE used not dead, byte_base always-identity, EOS_LITERAL correct-not-a-workaround) + found the Tokenizer struct redundantly re-stores 14 compile-time constants as runtime fields -- research/design only, no code changed
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Context:** while d512/d768 TinyStories training ran in the background (see
[[capability-roadmap-discrete-experts]] for the roadmap that motivated this), user asked for prior-art
research + a full proposed special-token set + a review of parsing/id-layout tradeoffs, then two
follow-up questions via IDE selection (the `EOS_LITERAL` literal-string check; the `Tokenizer` struct's
apparent lack of constexpr discipline). All folded into `docs/TOKENIZER_REVIEW.md` §5 (new), which
already had a review structure (§1-4, mostly DONE) from 9 days prior that this extends, not replaces.

**Prior art researched (WebSearch, cited in the doc):** ChatML (`<|im_start|>{role}\n...<|im_end|>`,
only 2 structural tokens, role name is plain text not a dedicated id), Llama 3 (`<|begin_of_text|>`/
`<|start_header_id|>`/`<|end_header_id|>`/`<|eot_id|>`, same "role as text" pattern), Mistral
(`[INST]`/`[/INST]`, no system role), GGUF's `tokenizer.ggml.*` metadata (special-token ROLES tracked
as separate scalar KV keys + a per-token `token_type` tag -- metadata-driven, not id-range-encoded),
tiktoken's ID layout (cl100k_base: special tokens start at 100257; o200k_base: 199998 -- regular vocab
contiguous low, specials a separate tracked block above, never bit-flagged into the id itself).
**Two findings that settled real open questions:** (1) nobody mints a dedicated id per conversational
role -- a handful of structural markers + role-as-plain-text is universal; (2) every real scheme keeps
ids contiguous with specials tracked via metadata/small-range, never bit-flags -- this independently
validates the project's own existing `*_id`-field approach and argues against the "significant bit
assignment" idea from the original message.

**Proposed new token set (design only, not implemented):** 4 new markers for Stage 1 --
`TOK_TURN_USER`/`_ASSISTANT`/`_SYSTEM` (turn-start, matched from literal corpus text the SAME way
`TOK_EOS` matches `<|endoftext|>`) + `TOK_TURN_END` (distinct from `TOK_EOS` -- needed for future
loss-masking to "only backprop on assistant-authored tokens," a real requirement flagged not just
symmetry). Plus a bracket-glue extension (`(`/`)`/`[`/`]`/`{`/`}` treated like `is_interior_connector`
already treats `_`/`-`/`'`) motivated by real measured numbers, not speculative: hand-traced `f(x)`
costs 7 tokens for 4 chars under the current JOIN-tax rule, and this session's own test output showed
`std::vector<int>` at 22 tokens -- matches [[tokenizer-dogfood-fuzz-and-source-pathologies]]'s 9-day-old
24-token measurement almost exactly, confirming this pathology hasn't been addressed yet. Angle brackets
deliberately excluded (too overloaded: comparisons/generics/HTML). Backtick explicitly NOT needed --
already folded to `'` by `normalize_text` upstream.

**3 specific code claims verified against actual `src/tokenizer.cpp`/`casing.hpp`, not assumed:**
1. **`TOK_NEWLINE` is ACTIVELY USED, contradicting the original guess.** `encode_join`'s `tile_ws`
   (`tokenizer.cpp:477-478`) intercepts every lone `'\n'` before it could ever fall through to a raw
   byte. There IS a dead-in-practice raw-`'\n'`-decode fallback in `detokenize_join` (`:592-594`, via
   the generic `is_ws_byte` check) but it's unreachable from this encoder's own output -- defensive
   coverage for a hand-built/foreign id stream, not a bug.
2. **`byte_base` genuinely IS always the identity `byte_base[b]==b`, confirming the original claim.**
   `learn()` sets it unconditionally for all 256 bytes (not gated by actual corpus usage --
   `Scan::byte_used` is a separate reporting-only tracker); `deserialize()` independently re-derives
   the same identity and REJECTS any file where it doesn't hold. It's not even part of the serialized
   format. Recommended removing it now (nothing needs a non-identity mapping today), while documenting
   the one plausible future reason it existed (a corpus-compacted base alphabet -- only bytes the
   corpus actually contains get a base id, saving vocab slots on narrow-byte-range corpora) so a future
   "why did this array exist" doesn't have to re-derive the reasoning.
3. **`EOS_LITERAL` is correct and necessary, NOT a workaround for a bug (2nd IDE-selection question).**
   The corpus is plain UTF-8 text with no out-of-band channel -- literal-string-match-then-substitute
   is the ONLY way to signal "insert a special token here," and it's exactly what real tiktoken does
   for its own `<|endoftext|>` (not something that "auto-tokenizes" from the BPE vocab). The real gap:
   it's currently a one-off hardcoded check that won't scale to 4 more literal markers without
   duplicating the block -- generalize via the token-spec table (see below).

**Biggest finding, from the 3rd IDE-selection question (the `Tokenizer` struct's constexpr
discipline):** all 14 `*_id` fields (`eos_id`, `cap_id`, `join_id`, ... `tab4_id`) are runtime `int`
fields that ALWAYS hold their corresponding `casing::TOK_*` compile-time constant, for every corpus,
unconditionally -- proven by reading both `learn()` (direct assignment from the constant) and
`deserialize()` (which REJECTS any file not matching this exact scheme, then also just assigns the
constant directly, never actually reading a differing value off disk). This is the SAME redundancy
class as `byte_base`, generalized: the struct conflates corpus-derived state (genuinely runtime:
vocab/merges/pieces/attested) with fixed-scheme state (already `constexpr` in `casing.hpp`, pointlessly
re-stored as instance fields) -- inconsistent with this project's own stated engine philosophy
([[compile-time-decisions-preferred]]). Recommended deleting all 14 fields, referencing
`casing::TOK_*` directly everywhere (zero-call-site-change since `casing.hpp` is already
`using namespace`'d in `tokenizer.cpp`). This is not just tidiness -- it's the concrete throughput
angle the user was pointing at: a `switch`/dispatch table on a REAL compile-time constant is a much
more reliable jump-table-optimization candidate for the compiler than the same dispatch on a
runtime-loaded struct field, so this finding and the token-spec-table recommendation reinforce each
other and should land in the same pass.

**Enum recommendation, with a correction to the original ask:** plain `enum class` does NOT enforce
"no duplicate values" -- C++ only diagnoses duplicate enumerator NAMES, not values
(`enum class X { A=256, B=256 };` compiles fine). The mechanism that actually delivers the safety
property is AUTO-INCREMENT (give only the first enumerator an explicit value, let the rest increment)
-- a typo'd duplicate becomes structurally impossible since there's nothing to type. Recommended
**unscoped** `enum TokenId : int { ... }`, not `enum class`: grepped every `TOK_*` usage site
repo-wide, only `engine_core.cpp` compares against these outside `casing.hpp`/`tokenizer.cpp`, always
against a plain `int` -- an unscoped enum implicitly converts, so this is a zero-call-site-change swap,
while `enum class` would need casts everywhere for no extra safety (auto-increment is what buys the
safety, not the scoping).

**ID layout: recommended AGAINST bit-flagging/high-range special tokens, WITH a concrete alternative.**
Ids are direct array indices into the embedding table + lm_head logits (the actual hot path, unlike
tokenizer dispatch which runs once per generation call) -- bit-flagging would either break dense
indexing or add a translation-table indirection to the genuinely latency-critical path, to speed up a
path that isn't one. This is precisely the "trade one convenience for a degradation elsewhere" the
original message worried about, made concrete. The current contiguous layout ALREADY gives O(1)
classification for free (`id<256`=byte, `256<=id<n_base`=marker, `id>=n_base`=piece -- `n_base` already
exists, already 270). What IS real and worth fixing: `detokenize_join`'s 10-deep sequential
`if/else-if` marker dispatch (`tokenizer.cpp:578-591`) that every ordinary content token (the vast
majority) must fail through -- this is exactly TOKENIZER_REVIEW.md's pre-existing item #2 ("DRY the
encode/decode FSM via a declarative token-spec table"), whose own priority list already said "schedule
when adding the next special-token family" -- i.e. now. Recommended landing the enum conversion, the
`*_id` field removal, and this dispatch table together as one pass, since the new markers should be
born into the new scheme rather than bolted onto the old one first.

**GGUF separation: confirmed already correct, not a gap.** `include/sub0/gguf.hpp` has zero references
to any `sub0::tok`/`casing::TOK_*` type -- already fully decoupled. Recommended explicitly keeping it
that way (convert at the boundary if/when [[gguf-import-feasibility-review]]'s weight-transplant work
resumes, don't fold the two representations together) since GGGUF's own metadata-driven model is
architecturally close enough to invite an unwanted unification later.

**Status: research + design only, zero code changed.** `docs/TOKENIZER_REVIEW.md` §5 (7 subsections)
is the durable record. Explicitly NOT implemented yet: implementation needs a full configure+build+test
cycle, and would bump `tokenizer.tok`'s `"S0TE"` magic (invalidating every existing trained model's
tokenizer, same class of break as the original EOS-adding bump) -- flagged as needing confirmation
before it happens, not assumed. Also currently blocked from using the primary build dir (`ts_cuda`)
since the d512 TinyStories run is live there; `sub0_frontend_tests` is engine-free so a different idle
build dir could be used if implementation starts before that training completes.
