# Tokenizer Code-Quality Review

Scope: the `sub0::tok` library — `include/sub0/casing.hpp` (reversible text transform +
truecasing), `include/sub0/tokenizer.hpp` (the `Tokenizer`, `Scan`, `LearnOptions` types) and
`src/tokenizer.cpp` (`learn`, `encode`/`detokenize`, the JOIN FSMs, (de)serialize). Goal: an
implementation that operates optimally, with SIMD/CUDA opportunities flagged for later. The
contract throughout is the round-trip `detokenize(encode(x)) == normalize_text(x)`, now guarded
by `tests/tok_lib_tests.cpp` (pinned) + `tests/tok_roundtrip_fuzz.cpp` (dogfood + 8000 fuzz).

This is a review + plan; the zero-risk annotations (SIMD/CUDA `TODO`s) are already in the code.
Ordered by value/effort.

## 1. Remove the legacy (non-JOIN) tokenizer path  *(major refactor; highest cleanup value)*

**Phase A — library + tests + configurator: ✅ DONE.** `learn` always mints the full 256-byte base +
markers; `encode`/`detokenize` collapsed to the single JOIN path; `deserialize` no longer scheme-detects;
`Tokenizer::join_scheme`, `LearnOptions::join_scheme` and the now-unused `sym_to_base` are gone. Tests
dropped the `.join_scheme = true` opt (JOIN is the only scheme) and the field asserts became `join_id >= 0`.
Verified: 224 engine-free assertions incl the 8000-case round-trip fuzz, and the configurator round-trips.

**Phase B — build/registry plumbing: ✅ DONE.** Retired the now-vestigial flags: `SUB0_JOIN_TOKENIZER` +
`SUB0_JOIN_FLAG` + the `--join` CLI, the `JOIN_TOKENIZER` constexpr, `registry.hpp`'s `join_tokenizer`
field / `j` tag / `compatible()` arg, and `train_stage`'s pass-through. The configurator's corpus.tok
emit also lost its last `if(join)/else` legacy fork (and the now-dead `seq_key`/`word_index`/`recon`).
Model dirs are now `…v<V>r_<sha>` (no `j`). `frontend_tests` updated (the `r` dir-name + the shorter
`compatible()`); full ctest 110/110.

The original rationale (kept for context): the JOIN scheme is the default and **all old models were
discarded** (no backwards compat); the legacy "space-as-a-byte-token" scheme forked every core routine
behind a `join_scheme` flag, doubling the surface that must stay correct.

**Extent (what the legacy path touches):**
- `tokenizer.cpp`: the `else` branch in `learn` (corpus-derived partial alphabet, no markers),
  the `else` loop in `encode`, the `else` in `detokenize` (`casing::detokenize`), and the
  scheme-detection in `deserialize`.
- `tokenizer.hpp`: `LearnOptions::join_scheme`, `Tokenizer::join_scheme`, and `sym_to_base`
  (only the legacy `encode` uses it).
- `casing.hpp`: `detokenize` (the legacy inverse) becomes unused by the library — keep only if a
  test/tool still needs the marker-stream inverse, else drop.
- Ripple (build/tooling): `tools/configurator.cpp` `--join` flag + its legacy emission/round-trip
  branch; `CMakeLists.txt` `SUB0_JOIN_TOKENIZER` option + `--join ${flag}`; `registry.hpp`
  `join_tokenizer` field/tag/`compatible()` arg; `train_stage.cpp` passing `JOIN_TOKENIZER`.

**Plan:** make JOIN the *only* scheme in the library first (delete the branches; `learn` always
mints the full 256-byte base + 13 markers), update `tok_lib_tests` (its "current scheme" cases
currently exercise the legacy `learn(corpus)` with no opts), and confirm green with the
engine-free `sub0_frontend_tests` (fast, no GPU). Then retire the build/registry plumbing
(`SUB0_JOIN_TOKENIZER`, the `j` tag, `--join`) in a second commit, since that needs a full
configure+build. Net: ~one parallel code path and several conditionals deleted, no behaviour
change for current (JOIN) models.

## 2. DRY the encode/decode FSM  *(correctness robustness)*

`encode_join` and `detokenize_join` are hand-mirrored: the encoder tracks the decoder's
pending-space (`dps`), and any divergence is a silent round-trip bug (exactly the class the
fuzz harness exists to catch — e.g. the NASA's UP/apostrophe fix). The fuzz net makes this safe
today, but the two could share a single declarative **token spec** — for each special token: its
marker id, the literal bytes it emits, and its effect on `(pending_space, in_spell, recase)`.
The decoder becomes a table walk; the encoder picks the token whose spec realizes the next gap.
Lower priority than #1 (it's a structural nicety, well-fenced by tests), but it would make new
specials (the SPACE_N family, future bracket regions) additions to a table rather than edits to
two mirrored switch statements.

## 3. Performance — the corpus-tokenization hot path  *(SIMD/CUDA, flagged in-code)*

The configurator runs `normalize_text → truecase_tokenize → encode (→ bpe_encode_word)` over the
**entire corpus** (GBs) at configure time — this is the throughput-critical path, not per-prompt
encode. Annotated `TODO`s are in place:

- **`bpe_encode_word` memoization (biggest win, algorithmic).** It re-encodes every word
  occurrence from scratch at O(N²·merges); corpora are Zipfian, so a `word-bytes → ids` cache
  amortizes the per-unique-word cost to ~zero. The unique-word table already exists in `Scan`
  during `learn` — the emission path just doesn't reuse it. `TODO(perf, hot)` at the function.
- **`normalize_text` SIMD.** A per-byte scan for two rare lead bytes (0xE2, 0x60); a vectorized
  "find next 0xE2/0x60, bulk-copy the run" makes it near-memcpy on the >99% pass-through bytes.
  `TODO(simd, hot)`.
- **`truecase_tokenize` SIMD + CUDA.** Alpha-run/case classification is vectorizable (16/32 bytes
  at once via is_alpha/is_upper masks). The whole pipeline is embarrassingly parallel across
  newline-aligned chunks (already threaded), so a **CUDA tokenizer (one block per chunk)** is a
  natural later lever — the merge table + attested set are read-only and the work is regular.
  `TODO(simd/cuda, hot)`.

Runtime (per-prompt) `encode` is not hot; leave it simple. Keep `learn`'s incremental
heap-based BPE as-is (already optimized — see the configurator-ingest memory).

## 4. Smaller cleanups (do alongside #1)

- `Tokenizer` carries many parallel `*_id` marker fields; once legacy is gone, consider a small
  `markers` struct or an `enum`-indexed array so adding a special is one line, not four
  (field + learn mint + deserialize detect + decode case).
- `seq_key` / the per-word `std::string` key churn in `Scan` — fine for `learn`, but if `Scan`
  ever feeds a memoization cache (#3), share one keying helper.
- Marker codes are already named constants (`TOK_*`) — good; keep that discipline as the set
  grows (SPACE8/TAB8, bracket regions).

## Priority

1. **#1 legacy removal** — deletes a whole parallel path; engine-free tests cover it; do first.
2. **#3 `bpe_encode_word` memoization** — the real configure-time speedup; measurable via the
   configurator timing already printed.
3. **#3 SIMD** of `normalize_text` / `truecase_tokenize` — once memoization lands and BPE is no
   longer dominant, these byte scans become the next ceiling.
4. **#2 FSM table** — structural; schedule when adding the next special-token family.
5. **CUDA tokenizer** — large, later; aligns with the CUDA-first-for-iteration-time goal.

---

## 5. Special-token extension: prior art, the full set, and the layout question

Triggered by [docs/ROADMAP.md](ROADMAP.md) Stage 1 (conversational data needs turn/role
representation) and a direct request to survey what a "full set" of special tokens should look
like before adding to the current 14. This is the "#2 FSM table... schedule when adding the next
special-token family" trigger condition above — that item and this one are the same piece of work.
**Research and design only below; no code changed.** The tokenizer.tok binary format
(`serialize`/`deserialize`, the `"S0TE"` magic + the `n_base==270` layout assert) is a
compatibility boundary for every existing trained model, so any real layout change needs its own
confirmed pass, not a drive-by edit.

### 5.1 Prior art surveyed

| Scheme | Turn/role tokens | Notable design choice |
|---|---|---|
| GPT-2/3 (`tiktoken`) | `<|endoftext|>` only, no turn structure | Regular BPE vocab occupies a contiguous low range; special tokens are a small block **above** it (cl100k_base: special @ 100257+; o200k_base: @ 199998+) — matched by literal string in the encoder, not learned by BPE. |
| ChatML (Qwen, OpenHermes, ...) | `<|im_start|>{role}\n ... <|im_end|>` | Only **2** structural tokens (start/end); the role name (`system`/`user`/`assistant`) is ordinary text between them, not a per-role token id. |
| Llama 3 | `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` | 4 structural tokens; same pattern — role name is plain text inside the header, not its own id. |
| Mistral | `[INST]` / `[/INST]` | No system role at all (prepended into the first user turn); minimal by omission. |
| GGUF (`tokenizer.ggml.*`) | N/A (format, not a scheme) | Special-token *roles* (bos/eos/pad/unk/sep/cls) are tracked as separate scalar metadata keys pointing at ordinary vocab ids, plus a per-token `token_type` (NORMAL/CONTROL/BYTE/...) tag — role membership is **metadata-driven**, not id-range-encoded. |

Sources: [llama.cpp chat template wiki](https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template), [Llama-3-8B-Instruct template discussion](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/14), [ChatML/Llama3/Mistral comparison](https://kindatechnical.com/prompt-engineering/prompt-templates-chat-formats-across-models.html), [tiktoken ID layout](https://deepwiki.com/openai/tiktoken), [GGUF tokenizer metadata](https://deepwiki.com/ggml-org/llama.cpp/3.3-context-management-and-inference).

**Two findings that directly settle open questions below:**
- Nobody mints a dedicated token id *per role*. A small number of **structural** markers
  (turn-start/turn-end) plus the role name as ordinary text is the universal pattern. sub0's
  vocabulary already tokenizes "user"/"system"/"assistant" fine as plain words — no new mechanism
  needed there.
- Every scheme keeps the regular vocabulary contiguous and special tokens as a small, separately
  *tracked* set (by metadata field or reserved range), never bit-flagged into the id itself. This
  matches — and validates — the project's own existing `*_id` field-per-marker approach.

### 5.2 The full token set this project actually needs (proposal)

Organized by what's already decided vs. what's new, so this reads as "extend," not "redesign":

**Existing 14 (`TOK_EOS`..`TOK_TAB4`, ids 256-269) — keep as-is.** Doc-boundary, case markers, JOIN
spacing, directional quotes, spell-group delimiters, run-length whitespace. All independently
verified live during this review (see 5.4).

**New, for Stage 1 (conversational) — the concrete near-term need:**
> **Superseded — see §5.8.** The 4-per-role-marker proposal below directly contradicted 5.1's own
> finding ("nobody mints a dedicated token id per role") one paragraph up — caught during
> implementation planning, not here. The design that actually shipped is **2** structural markers
> (`TOK_TURN_START`/`TOK_TURN_END`, role name as ordinary text), verified against Qwen/ChatML AND
> Gemma by name. Left in place below as the record of the reasoning trajectory, not as current design.
- `TOK_TURN_USER`, `TOK_TURN_ASSISTANT`, `TOK_TURN_SYSTEM` — turn-start markers, one per role,
  matched from literal corpus text the same way `TOK_EOS` matches `<|endoftext|>` (e.g. a
  dataset-conversion step embeds `<|user|>`/`<|assistant|>`/`<|system|>` between turns, mirroring
  how `get_fineweb.py`/`get_tinystories.py` already embed `<|endoftext|>` between documents).
- `TOK_TURN_END` — **not optional, not just symmetry with START.** ChatML/Llama 3 both have an
  explicit end marker (`<|im_end|>`/`<|eot_id|>`) for a real reason beyond structure: instruction
  tuning needs to mask the training loss to *only* the assistant's own tokens, and that needs a
  clean, unambiguous span boundary. Reusing `TOK_EOS` for this would conflate "conversation over"
  with "this turn ended, another follows" — keep them distinct.
- This is 4 new markers, not one per role-pair or anything combinatorial — small, deliberate.

**New, motivated by the existing dogfood/fuzz measurements, not speculative** (see
[[tokenizer-dogfood-fuzz-and-source-pathologies]] and 5.3 below): bracket/paren handling for
code-bearing text, since Stage 4's whole premise (agentic C++ coding) makes this load-bearing, not
optional polish.

**Explicitly NOT adding:** per-role dedicated ids beyond the 3 turn-starts (no prior art does
this); a bit-flagged/high-range id space for specials (5.6 explains why); backtick handling (already
folded to `'` by `normalize_text` before the tokenizer ever sees it — nothing left to special-case).

### 5.3 Bracket/paren extension — why, and why it's different from the quote mechanism

`"` needs `TOK_ODQUOTE`/`TOK_CDQUOTE` because it's **symmetric** — the same byte for open and
close, disambiguated only by surrounding-space context. `(` `)` `[` `]` `{` `}` are **already
distinct bytes** (unambiguously open vs. close) — they don't need that trick. Their actual cost is
the JOIN tax on the zero-gap case: every glued punctuation boundary pays a `JOIN` to cancel the
decoder's pending-space state (see `encode_join`'s `tile_ws`, `src/tokenizer.cpp:468-492`). Traced
through by hand: `f(x)` costs **7 tokens for 4 characters** (`f`, JOIN, `(`, JOIN, `x`, JOIN, `)`).
This is not hypothetical — it's directly visible in this session's own `sub0_frontend_tests` output
tokenizing `"std::vector<int>"` at 22 tokens (`{st} <J> {d} <J> {:} <J> {:} <J> <[...]> <J> {<} <J>
{i} <J> {nt} <J> {>}`), matching the 9-day-old dogfood memory's finding almost exactly (was 24 tok
then — the gap hasn't materially closed, this specific pathology is still open).

Recommend treating `(` `[` `{` `)` `]` `}` the same way `is_interior_connector` already treats `_`/
`-`/`'` for snake_case: extend the "glued, no JOIN needed" rule to a byte-adjacency case rather than
only a word-interior one — a smaller, more surgical change than inventing new marker tokens for
them, and it directly targets the measured pathology instead of a family of tokens with no measured
cost. Angle brackets (`<`/`>`) are deliberately **excluded** from this — they're heavily overloaded
in real text (comparison operators, generics, HTML/XML) so a blanket "always glued" rule is far less
safe than for the parenthesis/bracket/brace family, which is unambiguous. Worth a second look once
real C++/code corpus measurements exist (Stage 4), not now.

### 5.4 Verifying the specific claims from this session

**`TOK_NEWLINE` — confirmed ACTIVELY USED, not dead code.** `encode_join`'s `tile_ws`
(`src/tokenizer.cpp:477-478`) explicitly intercepts every lone `'\n'` and emits `t.newline_id`
*before* it would ever fall through to a verbatim byte — `'\n'` is never `emit_byte()`'d by this
encoder. `detokenize_join` (`:592-594`) does still have a raw-`'\n'`-byte decode branch via the
generic `is_ws_byte` fallback, but per the above it's **unreachable from this encoder's own
output** — a defensive path only relevant to a hand-constructed or foreign id stream (this project's
fuzz harness works the other direction, encode→decode, so it wouldn't exercise this either). Not a
bug, not dead in the sense of "delete it" — but worth a one-line comment noting it's
decode-side-only defensive coverage, so a future reader doesn't go looking for an encoder path that
produces it.

**`byte_base` — confirmed genuinely redundant in the current scheme.** `learn()`
(`:250-252`) unconditionally sets `byte_base[b] = b` for all 256 bytes — every single one, not
conditioned on whether the corpus actually used that byte (`Scan::byte_used` is a *separate*,
reporting-only tracker). `deserialize()` (`:717-724`) independently re-derives the same identity and
**rejects** any file where a base id doesn't equal its byte value (`if (code != i) return false`).
So `t.byte_base[b] == b` holds in every reachable state, always — it is not itself part of the
serialized format (only `base_symbol` is written; `byte_base` is reconstructed from the
already-verified identity). Two honest options, not a clear-cut call:
  1. **Remove it, use the byte value directly.** Simplest, matches "don't design for a hypothetical
     requirement" ([[only-add-arguments-we-need]]) — nothing today needs a non-identity byte→id
     mapping.
  2. **Keep it, but as the documented seam for a real future feature**: a corpus-compacted base
     alphabet (only bytes the corpus actually contains get a base id — `Scan::byte_used` already
     collects exactly the data this would need) would save up to a few hundred vocab slots on a
     corpus with a narrow byte range (e.g. pure-ASCII prose) — genuinely useful for the tiny/fast-
     iteration corpora this project explicitly favors. This is currently unbuilt and un-asked-for,
     so per this project's own stated preference it shouldn't be built speculatively — but it's the
     one plausible reason the indirection existed in the first place, worth naming so a future
     "why does this array exist" doesn't have to re-derive it.
  Recommendation: **(1) now** — simplify to the identity, drop the array — with this paragraph as
  the record of what a compacted-alphabet feature would need to resurrect, since that's a genuine
  the-corpus-decides-the-vocab idea consistent with `autosize()`'s own philosophy, just not
  something to build until a real corpus (not TinyStories) makes the savings worth it.

**Enum for token ids — the "no duplicate values" goal needs a specific mechanism, plain `enum
class` doesn't give it for free.** C++ does not diagnose two enumerators sharing a value — only
duplicate *names* are a compile error; `enum class X { A = 256, B = 256 };` compiles silently. The
mechanism that actually delivers "no duplicates, enforced by the compiler" is **auto-increment**:
give only the first enumerator an explicit value and let the rest increment —
`enum TokenId : int { TOK_EOS = 256, TOK_CAP, TOK_UP, ..., TOK_TAB4, TOK_MARKER_COUNT };` — a typo'd
manual value becomes structurally impossible since there are no manual values to typo, and
`static_assert(TOK_MARKER_COUNT - TOK_EOS == 14)` (already effectively present as the `n_base==270`
assert in `learn()`) catches any accidental insertion/reorder. **Unscoped, not `enum class`**:
checked every call site (`grep` across the repo) — only `engine_core.cpp`'s `code ==
casing::TOK_CAP`-style comparisons live outside `casing.hpp`/`tokenizer.cpp`, and those compare
against a plain `int`. An unscoped enum implicitly converts to `int`, so this is a
**zero-call-site-change** swap; `enum class` would need an explicit cast at every comparison for no
extra safety benefit here (the auto-increment is what buys the safety, not the scoping). Recommend
this alongside the marker-family additions in 5.2, not as a separate pass — the new markers should
be born into the enum, not bolted onto the old `constexpr int` list first.

### 5.5 The `Tokenizer` struct redundantly re-stores compile-time constants as runtime fields

A sharper version of the `byte_base` finding, and it generalizes to most of the struct's `*_id`
fields, not just that one array. `cap_id`, `up_id`, `join_id`, `newline_id`, `para_id`, `odquote_id`,
`cdquote_id`, `spell_start_id`, `spell_end_id`, `space2_id`, `space4_id`, `tab2_id`, `tab4_id`,
`eos_id` (`include/sub0/tokenizer.hpp:100-108`) are **runtime `int` fields holding values that are
always, unconditionally, the corresponding `casing::TOK_*` compile-time constant** — never anything
else, for any corpus, ever. Proof is in the code that sets them: `learn()` assigns each directly
from the matching `TOK_*` (`src/tokenizer.cpp:253-266`, explicitly commented "these ids are
determined by the tokenizer CODE version, the same for every corpus"), and `deserialize()`
*rejects the file* if the loaded layout doesn't match this exact scheme, then likewise assigns the
compile-time constants directly (`:739-742`) — it never actually reads a differing value from disk
into these fields. So `t.eos_id == casing::TOK_EOS` is a true invariant, not a per-instance fact.

This is the same category of issue as `byte_base` (5.4), generalized: the `Tokenizer` struct
conflates two genuinely different kinds of state that this project's own engine design (constexpr
config baked at build time, only genuinely corpus/runtime-variable facts loaded at runtime — see
[[compile-time-decisions-preferred]]) already treats as separate elsewhere:
- **Corpus-derived, must stay runtime**: `vocab`, `merges`/`piece_logp`/`piece_index`, `expansion`,
  `attested` — these genuinely differ per corpus and are correctly loaded from `tokenizer.tok`.
- **Fixed scheme, already `constexpr` in `casing.hpp`, and redundantly re-stored here**: all 14
  `*_id` fields above, plus `byte_base`.

**Recommendation: delete all 14 `*_id` fields from `Tokenizer`; every call site references
`casing::TOK_EOS` / `casing::TOK_JOIN` / etc. directly** (already in scope everywhere via
`using namespace sub0::casing;` at the top of `tokenizer.cpp`). This is not just cleanup —
it's the concrete throughput-relevant fix the "high-throughput as per our engine design" framing is
asking for: a comparison against a real compile-time constant is a candidate for the compiler to
fold into a jump table (a `switch` on `id` with `case casing::TOK_JOIN:` etc. is trivially provable
as a dense, bounded case set at compile time); a comparison against `t.join_id` — a value the
compiler can't assume is invariant just from the type, only from reading and trusting scattered
initialization code — is a much harder case for the same optimization to fire reliably. In other
words: this finding and the item-#2 token-spec-table / switch-dispatch recommendation (5.6 below)
are mutually reinforcing — do the enum conversion (5.4) and this field removal *together*,
and the `switch`-based dispatch table becomes a genuinely compile-time-provable jump table, not just
"probably a jump table if the optimizer is in a good mood." Also removes an entire, currently
carefully-guarded desync risk: `deserialize()` spends real code re-verifying that a loaded file's
layout matches before trusting it enough to copy the constants in — with no redundant field to
desync, that verification still matters (rejecting a foreign/corrupt file is still correct), but
there is no longer a *second* copy of the truth that verification exists to protect.

### 5.6 The contiguous-vs-bit-flagged id question

**Recommendation: stay contiguous. Do not bit-flag or range-separate special tokens from content
ids.** Reasoning, concrete not just cautious:

- Token ids are used as **direct array indices** into the embedding table and the output logits row
  (`op_embed`, the lm_head GEMM) — this is the actual hot path (every training step, every
  generated token), unlike tokenizer dispatch which runs once per generation call. A bit-flagged or
  high-reserved-range id (e.g. `id | 0x8000_0000`, or "specials live at 60000+") either breaks dense
  indexing outright or forces a translation table between "public id" and "dense embedding row" —
  adding a real indirection to the one path in this whole subsystem that's actually latency-critical,
  to speed up a path (marker dispatch) that measurably is not. That's the exact "trade one
  convenience for a degradation elsewhere" the original question flagged, made concrete.
- The current layout **already gives O(1) special-token classification for free**, no scheme change
  needed: `id < 256` = raw byte, `256 <= id < t.n_base` = marker (`t.n_base` already exists and is
  exactly 270), `id >= t.n_base` = learned piece. Three range comparisons, zero indirection, fully
  compatible with dense embedding indexing. Prior art agrees: tiktoken's own "contiguous BPE range,
  then a small special block above it" is this exact same idea, just expressed as one boundary
  instead of two.
- What actually IS worth fixing — and is real, not a range-encoding problem — is `detokenize_join`'s
  dispatch: 10 sequential `if (id == t.join_id) ... else if (id == t.newline_id) ...` equality
  checks (`src/tokenizer.cpp:578-591`) that every ordinary content token (the overwhelming majority)
  must fail through before reaching the real content-handling code. This is exactly **item #2 above
  (the declarative token-spec table)** — once markers are `TOK_EOS..TOK_TAB4 (+ the 4 new turn
  markers)`, a small `id - 256`-indexed dispatch table (or a `switch`, which a decent compiler turns
  into a jump table on a dense case range) replaces the linear chain with true O(1) dispatch, with
  zero change to the id layout. This also directly generalizes the `EOS_LITERAL` mechanism (see the
  next paragraph) instead of hand-duplicating it once per new turn marker.

**Closing the loop on `EOS_LITERAL`** (the ide-selected code): correct and necessary as discussed
above, but currently a one-off hardcoded literal-string check inside `encode_join`. Adding
`TOK_TURN_USER`/`_ASSISTANT`/`_SYSTEM`/`_END` the same way would mean 4 more near-duplicate blocks —
not DRY. The token-spec table item #2 is already scoped to fix exactly this: a small
`{literal_string, marker_id}` table checked at the same point `EOS_LITERAL` is today, so a new
literal marker is a table row, not a new bespoke code block.

### 5.7 sub0's own tokenizer vs. the GGUF general tokenizer — confirmed already separate

Checked `include/sub0/gguf.hpp`: it reads GGUF's own vocab array structurally and has **zero
references** to `casing::TOK_*`, `sub0::tok::Tokenizer`, or any sub0-specific type. The two are
already fully decoupled — this is not a change to make, it's a design already in place, worth
stating explicitly since GGUF's own metadata-driven special-token model (5.1) is architecturally
close enough to sub0's `*_id`-field approach that a future maintainer might be tempted to unify
them. Recommend keeping them separate: sub0's tokenizer is BPE/Unigram + the JOIN spacing scheme
tuned for this project's own corpora; GGUF's is whatever the source model shipped with (often
byte-level BPE with a completely different marker set). Any future weight-transplant work
([[gguf-import-feasibility-review]]) should convert explicitly at the boundary, not fold one
representation into the other.

### 5.8 Implemented design (Stage 2 / WS5): 2 turn markers + reserved headroom, and the quote re-measurement

**What shipped, superseding 5.2's 4-marker proposal.** `TOK_TURN_START`/`TOK_TURN_END` only (ids
270/271) — role name flows through as ordinary text right after `TOK_TURN_START`, matching 5.1's own
finding. Literal strings adopted verbatim from ChatML (`<|im_start|>`, `<|im_end|>`), not invented, so
corpora already shipping in ChatML (SmolTalk/UltraChat/OASST) need no reformatting. Verified against
the two most recent model families by name before locking this in:
- **Qwen (ChatML, unchanged through the Qwen3.x line)**: exactly 2 structural tokens, role as text.
- **Gemma (unchanged through Gemma 1-3, referenced into Gemma 4)**: exactly 2 structural tokens
  (`<start_of_turn>`/`<end_of_turn>`), role as text, and **no system-role marker at all** — the
  system prompt folds into the first user turn. Both families reinforce 5.1; neither is an exception.

`TOK_TURN_END` stays distinct from `TOK_EOS` (5.2's reasoning holds, unchanged): matches ChatML/
Llama-3, and future instruction-tuning loss-masking needs a clean assistant-span boundary.

**Reserved headroom, new since 5.2** — `TOK_MARKER_COUNT` rounds up to 32 (14 original + 2 turn + 16
`TOK_RESERVED_*`), not 16. Since this enum extension is *already* a forced re-tokenization (every
learned piece id shifts, because pieces start at `n_base`), reserving slots now means a future marker
family (Stage 2/3 reasoning delimiters, tool-call structure — see `docs/ROADMAP.md`) is a rename of an
existing enumerator, not another insertion — this is the last forced re-tokenization, not the first of
a series. A reserved id gets a `base_symbol`/`expansion` entry (required by the format) but no
encode-side literal match and no decode-side effect; `detokenize_join` explicitly no-ops any
in-range-but-unassigned marker id rather than falling through to the generic content path (which would
have truncated the id to a byte via `static_cast<unsigned char>`, corrupting output — a trained model
CAN sample a reserved id, since it's a real embedding/output row, so this is reachable at gen time).

**The directional-quote re-measurement (5.2's open item, resolved).** Measured OPEN/CLOSE fire vs.
fallback on 100MB of real TinyStories dialogue (not FineWeb prose, so this specifically stress-tests
the conversational-data motivation for this whole pass): **89.07% fire rate**, close to but slightly
below §3's original ~90% FineWeb figure — dialogue is not dramatically worse than prose here.
Classified the ~11% fallback by cause (replicating the encoder's exact state machine, not a
loose text-shape heuristic — a first pass that used shape alone over-counted, since most
similarly-shaped quotes actually DID fire correctly):
- **55389 of 63068 fallbacks (87.8%)** — a quote right after a single `'\n'` (a new line/paragraph
  starting with dialogue). The plain OPEN check requires the preceding whitespace to literally be a
  space (`stream[i] == ' '`), which a newline never is.
- **5440 (8.6%)** — British/logical-style quoting, punctuation immediately after a closing quote
  (`"Hello", she said`) — CLOSE's `after_glue` check requires whitespace right after the quote.
- **2239 (3.6%)** — other (double-newline/paragraph-preceded quotes, fully-spaced quotes, etc.).

**Fixed the dominant cause** (`src/tokenizer.cpp`'s quote-handling block in `encode_join`): a
line-initial opening quote now emits `TOK_NEWLINE` (preserving the exact newline byte on round-trip —
critical, since silently reconstructing it as a plain space would flatten line structure) immediately
followed by `TOK_ODQUOTE`, with **no decoder change needed at all** — `TOK_ODQUOTE`'s decode already
only emits a leading space `if (dps)`, and `dps` is already `false` right after a `NEWLINE` token, so
the existing decode path reconstructs this correctly for free. Re-measured post-fix on the same
sample: **98.67% fire rate** (fallback residue is now almost entirely the British-style case above).
**Left the British-style CLOSE case as documented, pinned, known behavior** — a real but 20x smaller
effect (0.94% of all quotes vs. the fixed case's 9.6%), and a correct general fix needs a broader
"trailing punctuation byte set" design (comma/period/question/exclamation/semicolon, not one
character) that wasn't scoped as part of this quick-win pass.

### 5.9 WS5b implemented: bracket-glue markers

**Design resolved a real tension between 5.3's original recommendation and the plan's revised
instruction.** 5.3 (above) recommended modeling brackets on `is_interior_connector` (no new markers,
just widen the "glued, no JOIN" rule to a byte-adjacency case). Tracing through actual token costs by
hand showed this can't work as stated: a bracket byte alone can't tell the decoder whether the
original text had a space there or not (`"the (x)"` vs `"the(x)"` must round-trip distinctly, exactly
the same ambiguity quotes have) — so *some* marker is required, the same conclusion the plan's revised
"model on ODQUOTE/CDQUOTE" instruction reached, just via direct derivation rather than analogy.

**What shipped**: 6 new markers (`TOK_GLUE_OPAREN`/`_CPAREN`, `_OBRACKET`/`_CBRACKET`, `_OBRACE`/
`_CBRACE`), each firing only on the genuinely wasteful case — a bracket glued directly to what
precedes it (`gap==0 && dps`, mirroring `TOK_CDQUOTE`'s trigger shape). Unlike quotes, these don't
disambiguate direction (the byte already does that) — they exist purely to collapse the JOIN tax
`f(x)` pays (7 tokens: `f,JOIN,(,JOIN,x,JOIN,)`). An open-bracket marker ALSO clears `dps` afterward
(mirrors `TOK_ODQUOTE`/`TOK_SPELL_START`'s after-effect — content right inside a bracket typically
glues too), so one marker absorbs the JOIN on both sides at once: `f(x)` -> `f,GLUE_OPAREN,x,
GLUE_CPAREN` = 4 tokens. A close-bracket marker leaves `dps` true (closing brackets are typically
followed by a space, `") the"`). A bracket preceded by a real space (`"f (x)"`) was already free before
this change and is untouched — falls through to the ordinary byte path, same token count as before,
just not further optimized (matches the plan's own scoping: fix the measured dominant pathology, not
every spacing variant). Angle brackets stay excluded (5.3's reasoning, unchanged — too overloaded).

**Consumed 6 of the 16 reserved marker slots from 5.8 — a RENAME, not an insertion**, exactly the
scenario the reservation existed for: `TOK_MARKER_COUNT`/`n_base` are unchanged (still 32/288), so
this needed a `kSchemeVersion` bump (1 -> 2, bundled with the quote fix above, both transition-rule-
only changes) but NOT another `tokenizer.tok` magic-number bump.

**A real bug, caught by the existing fuzz/dogfood net, not by hand-tracing**: the first implementation
inverted the open/close `dps` assignment (`dps = !is_close_bracket(...)` instead of
`dps = is_close_bracket(...)`), which broke round-trips for any bracket immediately followed by
*another* bracket of a different family (e.g. `"{code}[index]"` — a `}` immediately followed by a
`[`). Caught immediately by `sub0_frontend_tests`' full-source dogfood test and both 4000-iteration
fuzz suites going red (not by the pinned worked-example cases, which happened to avoid this exact
adjacency) — direct validation of why this project keeps the generative fuzz net alongside curated
examples, per this file's own opening framing.

### 5.10 WS6: SIMD — one real win, one measured non-result (both evidence-based, not assumed)

Targeted `normalize_text`'s rare-lead-byte scan and `truecase_tokenize`'s alpha-run classification,
per §3's original framing. This project's existing SIMD convention is `#pragma omp simd`
auto-vectorization hints over plain loops (see `src/backend_cpu.cpp`), not hand-written intrinsics
— followed here rather than introducing a new pattern. One real, important discovery along the way:
`sub0_frontend` (the static lib `casing.hpp`/`tokenizer.cpp` actually compile as, used by both the
configurator and — via `sub0_core` — the runtime engine) never received an OpenMP flag; only
`sub0_core` links `SUB0_OPENMP_TARGET` (`cmake/OpenMP.cmake`). Every `#pragma omp simd` in this
header was **silently inert** until `-fopenmp-simd` (simd-pragma recognition only, no threading
runtime — safe and free to add regardless of the full OpenMP probe's outcome) was added to
`sub0_frontend`'s compile options in `CMakeLists.txt`.

**`normalize_text` — real, measured win, ~20% faster.** Rewrote as a two-pass: a branchless
`#pragma omp simd` classify pass flags the two rare lead bytes (`0xE2`, backtick) across the whole
text, then a second pass bulk-copies each pass-through run with `std::string::append` instead of one
`push_back` per byte, only re-examining flagged positions with the original scalar replacement logic.
Verified via a same-process **interleaved** A/B (this laptop has real thermal confounds on separate
sequential process runs — see [[thermal-confounds-ab-wallclock-testing]]) against the exact pre-WS6
scalar reference: consistently **16-28% faster across 10 rounds** (mean ≈20%), byte-identical output
in every round.

**`truecase_tokenize` — investigated, measured, deliberately left UNCHANGED.** Decomposed the cost
first (500MB real prose): the boundary-scan (finding alpha-run start/end) is only ~30% of this
function's total cost — the dominant cost is per-word case CLASSIFICATION (`emit_word`: a lowercase
copy, an `attested` hash lookup), which is inherently not a SIMD target (branchy, allocates, hashes).
Two rewrites were built and measured, not just reasoned about:
1. A whole-text `is_alpha`/`is_upper` mask precomputed up front, mirroring `normalize_text`'s
   pattern — measured **slower** (the inert-pragma finding above, plus two extra n-sized
   allocations with no compensating benefit).
2. Bulk-appending each run via `resize`+store instead of per-byte `push_back`, gated behind a
   length threshold (push_back wins for the length-1 runs — mostly single spaces — that dominate
   real prose), tried again after fixing the `-fopenmp-simd` gap above — an **interleaved** A/B
   still showed the "improved" version within ±3% of the original across 10 rounds: noise, not a
   real win.

Reverted to the exact original scalar implementation, with the investigation recorded in the
function's own comment so it isn't blindly re-attempted later. **The methodology point matters as
much as the result**: the first (wrong) measurement pass ran old-vs-new as *separate sequential
process invocations* and showed an apparent ~13% *regression* — reversed once measured
*interleaved, same-process* instead. A d512 GPU training run was also active in the background
throughout this investigation, adding to the load variance sequential runs are vulnerable to.
Don't trust a single sequential before/after number on this machine for anything close to this
scale of effect — interleave.

## Next steps

Status: **Stage 1 (WS1-WS4), Stage 2 (WS5 turn markers/reserved headroom/magic bump/quote fix, WS5b
bracket-glue), WS6 (SIMD — one real win, one measured non-result) and WS7 (the `sub0llm-tokenizer
export` interchange tool) are all implemented and test-verified.** The original plan
(`C:\Users\craig\.claude\plans\smooth-noodling-kurzweil.md`) is now fully executed; `sub0llm-tokenizer`
absorbing `vocab`/`encode`/`decode`/`roundtrip` off the engine (per
`docs/WORKFLOW_ARCHITECTURE.md`'s own staged sequencing) is future work, not scoped here.
