# Tokenizer v2 — Review + Breaking-Change Ideas

Forward-looking notes for a **next** tokenizer scheme, where breaking the on-disk format is
acceptable (a scheme bump already forces a full retokenize + fresh models — see §7). This is a
review of the current setup plus a catalogue of ideas, **not a commitment**. What began as an
observation about **open/close (range) pairs** has, through measurement (§4a–§4b and the built
modality-calibration scan), resolved into a sharper thesis: **spacing should be a corpus-derived
per-character default, not a uniform "single space" rule patched by per-delimiter glue tokens** — and
dual open/close tokens should be kept *only* where a character is measured-bimodal. The decisive,
data-settled conclusions are collected in **§0**; the reasoning and measurements follow in §1–§8.

The lossless contract is unchanged and non-negotiable for every idea here:
`decode(encode(x)) == normalize_text(x)`, guarded by the round-trip fuzz + dogfood net.

---

## 0. Design decisions the data settles

These are the **solid wins** — each backed by the cross-corpus modality scan (fineweb + cosmopedia +
minipile + tinystories + gsm8k, §4b) — stated as concrete v2 decisions. The rest of the doc is the
supporting detail.

**D1 — Spacing is a per-character DEFAULT, not a uniform rule.** Today the default between content is
always "one space," and every deviation (glue) costs a `JOIN`. Measured, that `JOIN` budget is
**~10.9% of prose tokens** and **99.7% of it is one thing: gluing sentence punctuation to the previous
word.** So: give each codepoint a default `(lead, trail) ∈ {space, glue}`. Letters/digits default
`(space, space)` (today's behaviour). Sentence punctuation `.,;:!?` defaults `(glue, space)` — measured
84–99% unimodal `GS`. This **removes essentially the entire JOIN budget (~10% of prose sequence length)
with no identity fork** — the single biggest lever, and it *preserves unity* (the char keeps one id).
Deviations still cost a generic modifier (`JOIN` to drop a default space; an explicit space to add
one), but become rare.

> **IMPLEMENTED (schemeV4, this branch).** A unified per-byte `(lead, trail)` glue table
> (`glue_default(byte)` in casing.hpp): the boundary between two tokens defaults to glue iff
> `prev.trail_glue || cur.lead_glue`, deviations lossless via `TOK_JOIN` (force glue) or a literal
> space byte (force a space). `. , ; : ! ? %` are `lead=glue`; `$` is `trail=glue`; everything else is
> space-both. Mechanically tiny: `encode_join`/`detokenize_join` set `dps = !trail_glue` after a
> content byte and suppress the pending space before a `lead_glue` byte — one `glue_default()` read on
> both sides (DRY, cannot drift). No id/marker change — only the transition rule.
> **Measured (4 MB held-out, in-domain 4k vocab):** tinystories **1,285,466 → 1,147,084 tokens
> (−10.8%)**, JOIN 10.80% → 0.04%; **fineweb 1,672,762 → 1,585,273 (−5.2%)**, JOIN 8.98% → 3.92% (the
> residue is code/URL/bracket glue — D3 / point-3 territory). Lossless (round-trip fuzz + full frontend
> suite green); encode throughput neutral (base 32.2 vs v2 32.8 ms/4 MB, within noise). Hardcoded set
> for now; **D2** makes it corpus-derived, and the `glue_default()` accessor is the single point a
> per-piece default (point-3 learned symbol tokens) would extend.

**Point 3 — gluing corpus-specific (learned) tokens, not just inbuilt bytes. IMPLEMENTED (schemeV4).**
The Unigram learner now also mints *symbol* pieces: `Scan::add_words` collects a maximal plain-symbol
run (`is_symbol_piece_byte` — non-word, non-ws, excluding quotes/brackets which keep their own markers)
as a learnable unit, so `://` `->` `==` `!=` `&&` `::` `//` `--` become single pieces where they recur.
`encode_join` emits a symbol piece when the WHOLE maximal run matches one learned piece (a partial run
stays byte-by-byte — still lossless). Each piece carries **per-piece glue** (`piece_lead_glue`/
`piece_trail_glue` inherit the lead-glue of the first byte and trail-glue of the last), so decode (keyed
by piece id) agrees with encode (keyed by boundary bytes) — a no-op for word pieces. Corpus-adaptive:
prose (fineweb) learns few (`://` `--` `...` `||`), code (minipile) learns many (`==` `//` `**` `</` `++`
`::` `->`). Measured code win: `x == y` 5→3 tokens, `i != 0 && j <= n` 14→10, `a::b::c` 9→7. Lossless.

**Point 4 — compound & numeric expressions (measured on fineweb, real learned vocab).**
- **Hyphenated ALPHA compounds are already optimal — no action.** `well-known` (4 tok), `state-of-the-art`
  (9), `non-commercial` (4) each encode as a **single word unit with 0 JOIN**: `is_interior_connector`
  (`'` `_` `-`) already binds them when flanked by word bytes on both sides, so BPE merges across the
  separator instead of paying a JOIN per gap. This is settled; the cost is elsewhere.
- **The real cost is NUMBERS / DATES / ALPHANUMERICS.** `is_word_byte` excludes digits, so a digit run
  never forms a unit and glues byte-by-byte with **one JOIN per gap**: `2026` → `[2][J][0][J][2][J][6]`
  (7 tok), `2026-07-29` → 19 tok / **9 JOIN**, `covid-19`/`H2O`/`mp3` split at the digit boundary,
  `1,000,000` → 15 tok / 6 JOIN. On a 4 MB fineweb held-out this is most of the residual JOIN budget.
- **Prototype (digits as word bytes → digit runs become units):** held-out `1,585,273 → 1,547,205 tok
  (−2.4%)`, **JOIN 3.92% → 1.47%** (≈⅔ of the residual eliminated), all round-trips lossless.
  `2026-07-29` → 8 tok / 0 JOIN, `the year 2026` → 6 tok / 0 JOIN.
- **BUT digit tokenization is LOAD-BEARING for the arithmetic op-curriculum.** `op_curriculum` +
  `node_frame`'s compute callback **parse numbers out of the token stream digit-by-digit** and assert
  every result token is a single digit byte (`t >= '0' && t <= '9'`). Merging digits into pieces broke
  both the dataset assertions AND the operand parser (a recomputed operand came back `"1"` instead of
  `"89"`). So digits-as-bytes is a **deliberate choice for digit-level arithmetic supervision**, not an
  oversight (AGENTS.md §10: enumerate consumers before changing a shared semantic).
- **Refinement that keeps both wins:** make digit runs *units* (kills inter-digit JOINs) but bar the
  Unigram from minting **all-digit pieces** (single digit bytes stay mandatory → digits always
  Viterbi-segment to individual bytes → arithmetic intact). Measured: **same JOIN win (3.92% → 1.45%)**,
  token −0.66% (forgoes digit-piece compression, correctly). Round-trip lossless.
- **Spacing modality (measured, per the D1/D2 method): a number is its own TYPED unit.** A digit run's
  boundary with a non-digit is space/punct-bounded in **90.06%** of fineweb occurrences; only **9.94%**
  are letter-fused (`mp3`, `covid19`, `3rd`, `Foo123`). So the dominant modality is *implicit space*, with
  letter-fusion the ~10% deviation. `word_unit_end` therefore **splits at a direct digit↔letter
  transition** so a number stays a clean, self-contained span: `123 + 456` → 0 JOIN (spaced by default),
  `Foo123`/`3rd` → 1 JOIN (fusion is the deviation), while a connector still binds across the class
  boundary so `covid-19`/`2026-07-29` stay whole. This complements the digit-piece bar (a number is
  always a pure, consistent digit span) and is what the model *should* see for numeric generalisation.
- **IMPLEMENTED (v2 schemeV4):** `is_word_byte` += digits, `word_unit_end` class-transition split,
  Unigram bars all-digit pieces. Held-out **1,585,273 → 1,575,871 tok (−0.59%)**, **JOIN 3.92% → 1.58%**
  (the +0.13% vs the merge-everything variant is exactly the letter-fusion cost — trivial, buys clean
  numbers). Round-trip fuzz + full frontend suite green EXCEPT the 2 `op_curriculum` assertions.
- **Deferred (spike):** `op_curriculum` + `node_frame` parse operands out of the stream and depend on the
  old digit spacing/JOIN framing. Per the owner, that curriculum is a spike/POC, so the tokenizer-level
  improvement lands now and the arithmetic-frame number parser is reworked to the typed-unit framing as a
  later, separate increment (AGENTS.md §10: consumer enumerated, comment in `op_curriculum.hpp` updated).
- **Still deferred:** alpha compounds need nothing; `/` compounds (`and/or`) and thousands separators
  (`1,000,000`'s commas) are smaller and locale/domain-dependent → Point 3 (corpus-derived symbol
  pieces), not a hard-coded rule.

**D2 — The per-byte default table is CORPUS-DERIVED, not a scheme constant. IMPLEMENTED (schemeV4).**
`.`/`:`/`?` are unimodal closers
in clean prose but **bimodal in web/code** (decimals, URLs, `a.b`). A single baked default is wrong for
one domain. The configurator measures each codepoint's dominant modality and bakes a per-target default
table into `tokenizer.tok` — the mechanism already exists: `sub0llm-tokenizer calibrate` /
`sub0::modality` (mergeable ledger + contradiction flagging), so a disagreeing corpus is surfaced, not
averaged away. **As built:** `sub0::modality` rides the `Scan` (accumulated in `add_words`, folded in
`merge_words`, cached with the scan — `WCACHE_VERSION` 2); `learn()` derives a `std::array<GlueDefault,
256>` where the hardcoded `casing::glue_default` is the FLOOR and a byte with decisive, unimodal evidence
(≥500 samples, second combo <25%) overrides to its dominant `(lead, trail)`. `Tokenizer::glue_lead/
glue_trail` is the single lookup both encode and decode consult; the table is a gracefully-degrading
trailing section of `tokenizer.tok`. Verified end-to-end: a code corpus bakes `= / > : -` glue-both
(prose leaves them spaced), byte-identical across a scan-cache hit.

**D3 — Keep a dual open/close token ONLY for a measured-BIMODAL character. RESOLVED for a code-capable
scheme.** The prose-only plan below (drop `{}`, gate `()[]`) was premised on a prose target. Since v2 is
**code-capable** (owner's call), `{}`/`()`/`[]` all fire on code and STAY — no marker-enum/format change.
`"` stays a dual regardless. The keep/drop criterion, for the record:
- **`"` → KEEP the dual** (`ODQUOTE`/`CDQUOTE`): universally bimodal (~50/50 open/close in every
  corpus). The one clear survivor of the glue-marker family.
- **`{}` → KEPT** (code-capable; the prose-only plan would have dropped them, ~0.19% even on code).
- **`()` `[]` → KEPT** (code-capable). Measured firing on fineweb prose is NOT zero (CPAREN ~5.6k/4 MB),
  so the design's "0 fires on prose, gate for code" was corrected by measurement — they earn keep on
  prose too.

**D4 — The tokenizer NEVER matches pairs.** Local glue only; the model learns closure from the true
distribution (asymmetric `[5,6)`, nested, isolated included). No balance checking, ever (§2a). "Pair"
stays a naming/table convenience (`partner()`/`is_open()` as data), not a semantic claim.

**D5 — Encode + decode ride one declarative marker table** (`kMarkerSpecs`, already implemented on
`feature/tokenizer-throughput-2`) so the reduced marker set is described once, not mirrored in two
switches (§3).

**Net effect on a prose model:** the glue/quote marker family shrinks from 8 ids to **1 dual (`"`)**;
punctuation glue moves from ~10% of the stream (JOIN tokens) into free per-char defaults; sequences are
~10% shorter; `(`/`"` regain single embeddings. A code target additionally opts `()`/`[]` back in.
**Still open** (need a call before v2): whether v2 targets code at all (gates `()[]`, `<>`, backtick,
single-quote); the exact "add a space the default lacks" modifier; the per-char-default × truecase/
word-piece round-trip design; and the contiguous pair/enum reorg (§5). None of these blocks D1–D3,
which are the decisive wins.

---

## 1. Current setup (as of mainline `6893624`)

**Id space.** `0..255` = raw bytes (id == byte value, no offset); `256..287` = 32 markers
(`TOK_EOS .. TOK_MARKER_COUNT-1`, auto-incrementing `enum TokenId` in `include/sub0/casing.hpp`);
`>= 288` = learned Unigram pieces. `n_base == TOK_MARKER_COUNT == 288`.

**Pipeline.** `normalize_text` (fold typographic glyphs → ASCII, reused thread_local scratch) →
`truecase_tokenize` (case markers + attestation) → `encode_join` (the spacing FSM + per-word
Viterbi, now memoised by a call-local word cache). Decode is `detokenize_join`, a hand-mirrored FSM.

**Marker handling today.** `detokenize_join` is a 22-case `switch (id)` that, per marker, emits
literal bytes and mutates `(pending_space, in_spell, recase)`; `encode_join` re-asserts the same
per-marker facts to pick which marker to emit. The two are kept in sync only by the fuzz net.

> **Foundation already staged.** The `feature/tokenizer-throughput-2` branch replaces both switches
> with a single declarative `kMarkerSpecs` table `{literal, lead_space, dps, recase, in_spell}` that
> decode walks and encode reads via `emit_marker()`. **Everything in this doc assumes that table
> lands first** — it is the substrate the pair abstraction below extends. Merge it before starting v2.

**Versioning / blast radius.** `kSchemeVersion` (currently `3`) versions the transition rules and
rides the `fingerprint()` hash; the serialize magic (`"S0TF"`) rejects incompatible files outright.
Any change to the marker enum shifts `n_base`, hence every learned piece id — so it is inherently a
full-retokenize, all-models-invalid event (§7).

---

## 2. The pattern: almost everything is an open/close pair

Catalogue of the current marker set by shape:

| Pair | Bytes | Kind | Open spec `(lead_space, dps_after)` | Close spec |
|---|---|---|---|---|
| `TOK_ODQUOTE` / `TOK_CDQUOTE` | `"` / `"` (**same byte**) | byte, directional | `(true, false)` | `(false, true)` |
| `TOK_GLUE_OPAREN` / `_CPAREN` | `(` / `)` | byte, distinct | `(false, false)` | `(false, true)` |
| `TOK_GLUE_OBRACKET` / `_CBRACKET` | `[` / `]` | byte, distinct | `(false, false)` | `(false, true)` |
| `TOK_GLUE_OBRACE` / `_CBRACE` | `{` / `}` | byte, distinct | `(false, false)` | `(false, true)` |
| `TOK_SPELL_START` / `_END` | — | structural span | wraps N sub-tokens (no bytes) | — |
| `TOK_TURN_START` / `_END` | `<|im_start|>` / `<|im_end|>` | literal-tag span | emits literal tag | — |
| `TOK_UNCOMBINE` / `_END`, `TOK_COMBINE` / `_END` | — | control-op span (reserved) | interceptor, not text | — |

Two facts fall out of the table:

- **The quote pair and the bracket pairs are the same shape.** `TOK_CDQUOTE` and any `_CPAREN`-style
  close are *byte-identical in spacing* (`lead_space=false, dps_after=true`). The open sides differ
  only in one column: a quote opens with a space in front (` "x`), a bracket opens glued (`f(x`) —
  i.e. `lead_space` true vs false. So the user's intuition is exactly right: quotes are a
  bracket-family pair with one flag flipped. `kMarkerSpecs` already stores that flag.
- **One real axis separates them: directional ambiguity.** `(`/`)` are *distinct bytes*, so the
  byte alone says open vs close. `"`/`"` are the *same byte*, so the encoder must infer direction
  from context (space-before → open, glue-before → close). This is the only thing the quote path
  needs that the bracket path does not, and it must survive any unification.

---

## 2a. Pairing is a naming convenience — not encode-time matching (the decisive constraint)

The most important refinement, and it corrects the framing of §2: **the tokenizer neither checks nor
needs a delimiter to have a partner.** Today's markers fire on *local spacing* alone — an open-bracket
marker on a byte glued to what precedes it, `TOK_CDQUOTE` on `x" ` — with a plain byte as the
fallback when the glue pattern does not hold. No balance, no lookahead, no matching. This is exactly
why the scheme already handles every "isolated delimiter" case: a lone `"` in code, an unterminated
bracket, a stray `)` — as the user notes, isolation applies to the *whole* set, not just `<>` — all
simply fall to the bare-byte path and round-trip. Nothing special-cases them.

So the honest primitive is not a *pair* but a **glued-delimiter variant**: a `(byte, glue-direction)`
→ marker (a `"` glued-after = open, glued-before = close; a bracket byte glued on the side that earns
the saving). The "open/close pair" is an *emergent* relationship between two such variants — real and
useful for **naming and table structure** (§3, `partner()`), but never asserted about the input.

**Why this must stay true — the `[5,6)` argument.** Real notation does not respect same-family
matching. A half-open interval `[5,6)` pairs `[` with `)` *across* families; `array<5,6>` pairs `<`
with `>`; `a > b`, `i <= n` use `>`/`<` with no partner at all; `f(g(x)` nests and dangles. Crucially,
**`[5,6)` already round-trips correctly today** — the encoder emits an open-`[` glue marker and a
close-`)` glue marker, mismatched families and all, because it only reproduces bytes, it never claims
they pair. The asymmetry the user raises therefore does **not** overthrow the design; it overthrows
only a *hypothetical matcher*, and it is the proof that we must not build one: computing balance needs
a parser the tokenizer deliberately is not, and "correct" balance is ill-defined for mixed/half-open
notation anyway. Pairing belongs to the **model** — to learn statistically from the true distribution,
including the asymmetric, nested, and isolated cases — not to the tokenizer to enforce.

This **validates the current assumption.** Keep delimiters as independent, locally-decided glue
variants. The v2 "pair" work (§3) is then strictly a re-expression at the id/table level — share the
row shape, make `partner()`/`is_open()` data — carrying **zero new matching logic** and making no
claim that the model is handed verified ranges. It is handed *consistent, distinct ids per glue
variant*, and learns whatever closure structure the data actually contains (which is the real value
of "learning range closures consistently": consistency of the *token*, not verification of the span).

---

## 3. Proposal: a first-class "delimiter pair" table

Promote the pair from *two independent marker rows* to *one row that owns both ids*. Sketch:

```cpp
struct DelimiterPair {
    int          open_id, close_id;     // the two marker ids
    std::string_view open_bytes, close_bytes;   // "" for structural spans (SPELL/TURN)
    bool         symmetric_byte;        // open_bytes == close_bytes  => encoder infers direction
    MarkerSpec   open_spec, close_spec; // the (lead_space, dps, recase, in_spell) rows, as today
};
constexpr std::array<DelimiterPair, N> kPairs = { ... };
```

Wins this unlocks (all built on the existing `kMarkerSpecs` walk):

- **Adding a byte-delimiter pair is one row** — id allocation, decode bytes, and both spacing specs
  in a single place, instead of two enum entries + two `kMarkerSpecs` rows + an `encode_join` clause.
- **One generic encode helper** replaces `glue_marker_for` + the bespoke quote block: given a
  delimiter byte at a boundary, look up its pair, resolve role (open/close — from the byte for
  distinct pairs, from `(space-before / glue-before)` context for `symmetric_byte` pairs), check the
  trigger, `emit_marker` the resolved id. Quotes stop being a special case; they become the one pair
  with `symmetric_byte = true`.
- **Partner/role queries become data, not `switch`es:** `partner(id)`, `is_open(id)`,
  `is_close(id)` read the table (or arithmetic — see §5), which the decode interceptor
  (`sub0/decode.hpp`) and any future balanced-span validation want.
- **Naming discipline for free.** A consistent `TOK_<NAME>_OPEN` / `TOK_<NAME>_CLOSE` (byte
  delimiters) and `TOK_<NAME>_START` / `TOK_<NAME>_END` (structural spans) convention, checked by
  the table shape, instead of the current mix (`OD/CD`, `O*/C*`, `START/END`).

Keep the byte-vs-structural split as a column: byte pairs decode to literal bytes and encode from
byte+context; structural pairs (`SPELL`, `TURN`, `COMBINE`) carry no bytes and encode from structural
conditions. They share the *pair relationship* (partner, open/close, balanced), not the byte path.

---

## 4. Candidate new pairs — with honest per-candidate assessment

The user asks to "complete the set" with `<>` and other tick marks. They are **not** equal in value;
each needs its own justification because a mis-fired glue marker either breaks round-trip or wastes a
vocab row. Order below is by confidence. But **§4a asks the prior question** — whether this whole
glue-marker family earns its keep — which reframes "add `<>`?" into "keep any of them?".

- **`<...>` chevrons — DO NOT add as a glue variant.** The discriminator is *not* "usually
  unpaired" (per §2a we never match anyway) — it is that a `<`/`>` byte's *glue role is not locally
  decidable*, whereas a `(` byte's is. A glued `(` is *always* a bracket; a glued `<` in `a<b`,
  `i<=n`, `x->y`, or `template<T>` is a comparison/arrow/operator, not a delimiter. So the local-glue
  primitive itself — the thing that works for `()[]{}` — misfires on `<>`: it would collapse operator
  bytes into delimiter markers, hurting fidelity/efficiency on exactly the code/math corpora where it
  looks tempting. This is a property of the *byte*, orthogonal to pairing. Most likely verdict:
  **not worth it as a byte-glue pair**; the genuinely tag-like uses (`<think>`, HTML) are better as
  explicit literal-tag *structural* spans (like `TURN`), where the whole tag is recognised, not a
  bare `<` glued by spacing.
- **Backtick `` ` `` code spans — blocked by normalization, revisit for code corpora.**
  `normalize_text` currently folds `` ` `` → `'` (apostrophe). A backtick code-span pair
  (`` `code` ``) would require *removing that fold* (itself a breaking normalize change) and then
  spans arbitrary content, not a single glued byte — closer to a `SPELL`-style structural wrapper
  than a bracket. Real value only on code/markdown-heavy corpora. Bundle the decision with whether v2
  targets code at all.
- **Single quote `'...'` strings — high ambiguity, likely no.** `'` is already load-bearing as the
  apostrophe (contractions `don't`, possessives `Lily's`) and as an interior connector that binds a
  word unit. A single-quote *delimiter* pair fights that logic directly and is ambiguous in prose.
  Skip unless a code corpus makes `'string'` common enough to measure a win against the apostrophe
  cost.
- **The existing three bracket families are the whole safe set.** `()`, `[]`, `{}` are unambiguous
  distinct bytes with a clear paired meaning; they are already pairs and just need to be *expressed*
  as pairs (§3), not extended.

**Net:** the valuable v2 change is the *unification* (quotes join the bracket family as one
`symmetric_byte` pair; structural spans share the pair relationship), **not** a land-grab of new
punctuation. Add new pairs only behind a measured win on a representative corpus.

---

## 4a. The deeper question: token-count reducers vs representation unity

A reframe that changes the whole evaluation: these markers are **token-count reducers**, not range
representers — each collapses a `[JOIN, byte]` pair into one id. That ranges *emerge* from the
open/close ids is a bonus, not the purpose (§2a). This cuts both ways on `<>`: "ranges aren't the
point" removes the range argument for chevrons, but a **stronger, separate objection** now applies to
the whole family, including the brackets already shipped.

**The unity principle we already hold — and where these markers break it.** A load-bearing goal
elsewhere: a token has *one* id regardless of surface variation, with the variation factored into a
*separate* modifier. `once` and `Once` are the **same** word id + `TOK_CAP` — the word keeps one
embedding (sentence-start and mid-sentence share it) and the marker carries the case context. Most
punctuation obeys this too: a comma is **one** id whether written ` , `, `, `, or ` ,` — the
inconsistent spacing rides the implicit-space / `JOIN` channel, not `,`-variants.

The bracket/quote glue markers do the **opposite**. `f(` makes `(` a `TOK_GLUE_OPAREN` — a *different
id* from a bare `(` — so `(` has **no single embedding**; it is split across "spaced `(`" and "glued
`(`". That is precisely the weakening flagged for `<` (range-or-operator), *self-inflicted* on
`()[]{}"` today, and an **internal inconsistency**: `,` keeps unity, `(` does not, for no reason
beyond "brackets glue often, so the saving looked worth it."

**No free lunch.** You cannot give `(` one id AND spend zero extra tokens on its glue — the 1-token
saving *is* the identity fork. The generic mechanisms don't escape it: plain `JOIN` keeps unity but
pays the extra token (the tax the markers removed); a `SPELL`-style run-group keeps unity but only
wins on runs long enough to amortise its two wrappers (`f(x)` as a spaceless group is *more* tokens).
So the current markers are a deliberate trade: **~1 token per glued delimiter, paid in 8 vocab rows
(not free at this scale — reserved-headroom note) and in `(`/`"` losing a unified embedding.**

**So the real v2 question is not "add `<>`" but "should this family exist?"** Three points to measure,
not argue:

1. **Drop the glue markers.** Every delimiter is its byte + the generic `JOIN`/implicit-space channel,
   exactly like `,`: maximal unity, 8 vocab rows returned, fully consistent — at one extra token per
   glued bracket/quote. Quotes lose directional *fusion* but still round-trip (bare `"` + spacing).
2. **Keep them (status quo).** Token-efficient on bracket-dense text; forks 8 identities.
3. **A generic space-contraction set** (factor glue like `CAP` factors case). Explored honestly it
   collapses into `JOIN` (a "glue-left" modifier is 2 tokens = no saving) or `SPELL` (run grouping),
   so it is a re-application of what we already have, not a new mechanism — likely folds into option 1.

**Decide with a smoke check.** On (a) this project's own C++ (bracket/quote-dense — the adversarial
case) and (b) a small prose sample, measure: how often each glue marker fires, **total tokens saved**
vs option 1, and how many distinct-context occurrences each forked byte has (how split `(`'s embedding
actually is). Low-single-digit % saving + heavy fork → option 1 is the better v2 default. A real
code-corpus saving → keep them, knowing the price. Same data-driven method the scheme uses everywhere.

**Smoke-check results (measured 2026-07-29).** A throwaway harness learned an in-domain ~4k-piece
vocab and encoded (a) this project's own C++ (~1.1 MB → 520 k tokens) and (b) a TinyStories prose
slice (4 MB → 1.28 M tokens). Fire counts are spacing-driven, so robust to vocab granularity; the %
is of the encoded stream.

| Tokens saved | Code (C++) | Prose |
|---|---:|---:|
| whole glue/quote family | **4.75 %** | **1.80 %** |
| — brackets `()[]{}` | 4.43 % | **0 %** (prose has none) |
| — quotes `"` | 0.32 % | 1.80 % |

The value is **very uneven, and splits cleanly by domain**:

- **Brackets are a code-only feature.** Zero fires on prose — on a prose corpus the 6 bracket rows are
  pure dead vocab. In code they save ~4.4%, but lopsidedly: parens `()` are 3.6% of it, `[]` 0.6%, and
  **braces `{}` only 0.19%** — `{` fired 389× glued vs **2609× bare** (C++ usually *spaces* braces:
  `) {`, `{` on its own line), so `TOK_GLUE_OBRACE/CBRACE` barely earn their two rows even here.
- **The fork is real and measurable.** In code `(` splits 7985 glued : 5209 bare (~60/40 — *both*
  embeddings heavily trained), `)` 10964 : 2281; `"` splits three ways (open 1084 / close 583 / bare
  1974). So the identity-fork cost of §4a is not hypothetical — `(` and `"` genuinely carry two/three
  distinct trained rows.
- **Quotes are a prose feature.** In prose `"` is almost always directional (23 067 markers : 306
  bare) and that *is* the entire 1.8% prose saving; in code quotes are mostly bare and save only 0.32%.

**Reading → a defensible v2 default:** keep the **quote** pair (the whole prose win) and **`()`/`[]`**
(measured code wins), **drop `{}`** (marginal ~0.19%, forks two rows for almost nothing), and treat
brackets as a **code-target opt-in** rather than an always-on cost on prose models. "Keep everything"
should be a *decision*, not the default — the fork and the dead-on-prose rows are a real price for a
low-single-digit, domain-specific saving.

---

## 4b. The bigger lever the smoke check surfaced: per-char default spacing (unimodal vs bimodal)

Extending the smoke check to the **JOIN budget** reframes the whole feature. `TOK_JOIN` is **10.9% of
the prose stream and 21.5% of code** — and in prose **99.7% of those JOINs do one thing: glue a
sentence punctuation to the preceding word.** Measured: `,` is preceded by a JOIN 45 162× (100% of its
occurrences), `.` 83 575× (100%), and `!`/`?`/`:`/`;` the same. The scheme makes a *space between
words* free (implicit) but still charges a whole token for the *far more predictable* "no space before
a period."

**So the user's per-char-default idea is not niche — it is the single biggest token lever on the
table.** Give each byte a baked default `(lead, trail) ∈ {space, glue}`: letters/digits default
`(space, space)` (today's behaviour); sentence punctuation `.,;:!?` defaults `(glue, space)`. Then
`word.` needs **no JOIN** (the `.` glues by default) and the rare ` .` pays a modifier. On prose that
removes ~10.85% of the stream (essentially the entire JOIN budget); on code a large share too
(`, . ; :` are 95–100% glued-before there as well). Crucially it **preserves unity** — `.` keeps one
id, only its default changes — the exact opposite of the glue *markers*, which fork.

**Make the default corpus-derived, not a scheme constant.** `.`'s best default is domain-dependent:
prose wants `(glue, space)` (`end. Next`), code/math wants `(glue, glue)` (`a.b`, `3.14`). Rather than
bake one, have the configurator **measure each byte's dominant spacing and store a 256-entry default
table in `tokenizer.tok`** — the same data-driven method the scheme already uses for sizing. A prose
model and a code model then get different, optimal defaults for free, and the decoder just consults
the table.

**This gives the principled keep/drop criterion for the dual tokens — spacing modality:**

| Mechanism | Justified by | Wins on | Cost |
|---|---|---|---|
| Per-char default spacing | a **dominant** spacing (unimodal) | huge (~10% prose) | a per-byte default table; **no fork** |
| Dual open/close token | **two common** spacings (bimodal) | quotes (prose), `()` (code) | one id fork per char |

- **Unimodal char** (one spacing dominates) → baked default captures it free; the rare form pays a
  generic modifier. One id, no fork. Sentence punctuation is ~100% unimodal; `{`/`}` are ~85% unimodal
  (bare) — their glue markers should become *defaults*, not forks.
- **Bimodal char** (two spacings both common) → no single default suffices, so two free tokens are
  genuinely justified — this is exactly where the user's "duality… may still make sense to keep" holds.
  Measured: prose `"` is 11 580 open : 11 487 close (near-perfect 50/50) → the `ODQUOTE`/`CDQUOTE` dual
  earns its keep; `(` in code is 7 985 : 5 209 (~60/40) → borderline, defensible for a code target;
  `{` is 389 : 2 609 → unimodal, drop.

**Cross-corpus modality scan (all ASCII punctuation, post-normalize).** A raw-adjacency scanner
(`out/modality_scan.cpp`, throwaway) classified every punctuation byte's neighbours into SS/SG/GS/GG
(space/glue × before/after) over 250 MB slices of Cosmopedia (clean textbook prose) and MiniPile
(diverse web + code), plus gsm8k/TinyStories. `SG` = opener shape (` "x`), `GS` = closer (`x" `),
`GG` = interior (`a,b`), `SS` = isolated. Dominant combo per byte:

| byte | Cosmopedia | MiniPile | reading |
|---|---|---|---|
| `,` | 98.7% GS | 90.5% GS | **unimodal closer everywhere** → default `(glue, space)`, no fork |
| `;` `!` | 99% / 94% GS | 91% GS | same — robust sentence punctuation |
| `.` | **92.8% GS (unimodal)** | **58.8% GS / 39.5% GG (BIMODAL)** | **domain-dependent** — decimals/URLs/`a.b` in web |
| `:` `?` | GS closer (unimodal) | bimodal (URLs, `key:`) | flips with domain, like `.` |
| `"` | 46.8% SG : 43.7% GS | 33/32/32 three-way | **bimodal in every corpus** → dual token justified |
| `(` `)` | bimodal | bimodal | opener/interior split |
| `{}[]=<>/\_*` | GG-interior, sparse | GG-interior, common | code operators; near-absent in pure prose |

Three conclusions fall out, each backed by the numbers:

1. **Sentence punctuation is robustly unimodal `GS` corpus-wide** (`,` 90–99%, `;`/`!` similar) — this
   *is* the ~10% JOIN budget from above, and a baked `(glue, space)` default captures it everywhere,
   no fork. The clear, universal, biggest win.
2. **`.`/`:`/`?` are domain-dependent** — unimodal closers in clean prose, bimodal in web/code — which
   is the decisive argument that the default table must be **corpus-derived (configurator-measured)**,
   not a scheme constant. `modality_scan` is exactly the pass a v2 configurator would run to populate it.
3. **`"` is bimodal in every corpus** — the one unambiguous "keep the dual token" case; brackets/
   operators are `GG`/bimodal and code-specific, reinforcing §4a (a code-target feature, not a prose
   cost). *(Note: `modality_scan` currently also tallies control bytes like 0x0C as punctuation — a
   cosmetic artifact, ignore those rows.)*

**Now a real, collatable tool (branch `feature/modality-calibration`).** The throwaway scan is
promoted to an engine-free calibration facility — `include/sub0/modality.hpp` (`ModalityStats`:
merges additively like `tok::Scan`, serialises to a ledger à la `data/tokenizer_calibration.txt`, and
`find_contradictions()` flags a corpus whose dominant modality for a char disagrees with the
accumulated ledger) — exposed as `sub0llm-tokenizer calibrate <corpus…> [--load] [-o] [--max-mb]`. The
scan is UTF-8-exhaustive (ASCII *and* non-ASCII symbols `£ € © ° § ± ′ …`, excluding letter blocks +
C0/C1 controls). This is the "unifying facts collation" substrate: a disagreeing future corpus
surfaces a mismatch rather than being averaged away, so the next scheme version is derived from a
decisive multi-corpus picture, not one corpus's bias.

**Collated over 5 corpora incl. fineweb (2.15 GB: fineweb + cosmopedia + minipile + tinystories +
gsm8k, 3210 codepoints):** sentence punctuation stays unimodal `GS` (`.` 84.6%, `,` 95.8%, `;` 92.4%,
`:` 73.5%); `"` (43.9 SG / 41.5 GS), `(`, `!`, `?`, `$` are bimodal; non-ASCII symbols land cleanly
(e.g. **`£` 83.5% SG-opener** — currency prefix). The contradiction log is the payoff: MiniPile
(web/code) systematically flips `" ( ) [ ] = ! #` to `GG-interior` (code glues both sides), gsm8k
flips math operators `+ < >`, and tinystories flips `! ?` — i.e. the ledger's mismatches *are* the
prose-vs-web-vs-math boundary, made explicit and per-character.

**v2 implication.** The headline win is **per-char (corpus-derived) default spacing** for punctuation —
unity-preserving, ~10% of prose, and it subsumes most of what the unimodal glue markers were doing.
Keep **dual** tokens only where modality is genuinely bimodal (`"`, arguably `()` for code); let
unimodal delimiters (`{}`, `.,;:!?`) ride a baked default. Both are per-char and measurable — the same
smoke check decides each. (Cost to weigh: a per-byte default interacts with the implicit-space/JOIN
FSM and needs careful round-trip design, and the default table is new serialized state — a v2-scope,
scheme-bumping change.)

---

## 5. Enum / table organisation ideas

- **Lay pairs out contiguously, open then close**, so `close_id == open_id + 1`. Then
  `is_open(id) = ((id - kPairBase) % 2 == 0)` and `partner(id) = id ^ 1` (relative to the pair
  region) — pair relationships become arithmetic, no lookup. This is a clean-break reorg (ids shift),
  which is exactly why it belongs in a v2, not a patch.
- **Group by taxonomy**: `[ byte-delimiter pairs | structural-span pairs | singletons (JOIN,
  NEWLINE, PARA, SPACE*, TAB*, CAP, UP, EOS) | reserved headroom ]`. Keeps the "range pairs" as one
  contiguous, table-describable block.
- **Fold `SPELL`/`TURN`/`COMBINE`/`UNCOMBINE` into the same pair table** as structural pairs
  (`open_bytes=""`), so the *one* concept "this is an open/close pair" is expressed once. Their
  encode conditions stay bespoke (they are structural, not byte-triggered), but `partner()`/balance
  checks and the reserved-headroom accounting come for free.
- **Single source for the count assert.** Today `static_assert(TOK_MARKER_COUNT - TOK_EOS == 32)`
  guards drift by hand; with a pair table, derive marker count from `kPairs.size()*2 + singletons`
  so adding a pair updates the count structurally.

---

## 6. What the unification does NOT change

- The **directional inference for `symmetric_byte` pairs stays** — it is genuine context analysis
  (space-before vs glue-before), just relocated behind the pair lookup, not deleted.
- The **spacing specs are unchanged** — v2 reuses the exact `(lead_space, dps_after)` columns that
  make `detokenize(encode(x)) == normalize_text(x)` hold today; the table is a re-expression, byte-
  identical in behaviour, not a re-tuning. (A re-tuning, e.g. changing when a quote opens, is a
  separate decision with its own measurement, and should not be bundled into the structural refactor.)
- **Reserved headroom stays inert** — reserved ids remain default rows (no bytes, no state change),
  the model-can-sample-one contract.

---

## 7. Breaking-change discipline (why this is a v2, and how to land it)

Per AGENTS.md §3/§10, a marker reorg is the highest-blast-radius change in the engine:

- **Everything derived from `n_base` moves.** Learned piece ids start at `n_base`, so reordering or
  adding markers shifts every piece id → `tokenizer.tok`, `corpus.tok`, `model.bin`, `.ckpt`, and the
  model-dir names all change meaning. There is no partial migration; it is a clean break.
- **Bundle ALL breaking marker changes into ONE v2 bump.** Do not ship incremental breaking marker
  edits — each one throws away in-progress training. The pair reorg (§5), any new pairs (§4), and any
  spacing re-tuning land together, once.
- **Fail loud, not silent.** Bump `kSchemeVersion` **and** the serialize magic (`"S0TF"` → `"S0TG"`)
  so a v1 file meeting v2 code is rejected at load, not silently mis-decoded (the discipline the
  `"S0TZ"→"S0TE"→"S0TF"` history already follows).
- **Gate on byte-identical where the refactor claims to be behaviour-preserving.** The §3 unification
  must reproduce v1 token streams exactly (fuzz + differential + the assertion-count gate); only the
  deliberately-new pairs (§4) may change output, each behind its own measured justification.

---

## 8. Sequencing

0. **Calibrate the corpus — DONE.** The keep/drop question is no longer open: `sub0::modality` /
   `sub0llm-tokenizer calibrate` (branch `feature/modality-calibration`) has measured the family across
   fineweb + cosmopedia + minipile + tinystories + gsm8k. The decisions in §0 (D1–D3) fall directly out
   of that ledger. A v2 configurator re-runs this pass per target corpus to bake the default table.
1. **Merge the substrate branches** — `feature/tokenizer-throughput-2` (`kMarkerSpecs`, D5) and
   `feature/modality-calibration` (the ledger, D2). Nothing below is worth doing without them.
2. **Non-breaking pre-work on mainline:** (a) express the *pair relationship* over the existing ids
   (`partner()`/`is_open()` as data, D4) so encode/decode is pair-driven before the ids move; (b) wire
   the modality ledger into `sub0llm-configure` so it accumulates per ingested corpus (like
   `tokenizer_calibration.txt`), giving the default table its data for free.
3. **The v2 clean break (one commit series):** implement **D1** (per-char default spacing, the headline
   win) + **D2** (corpus-derived default table in `tokenizer.tok`) + **D3** (drop `{}`, keep only the
   `"` dual, `()`/`[]` gated on a code target); contiguous marker/enum reorg (§5); bump scheme version
   + magic; retokenize + retrain. Byte-identical gate on anything claimed behaviour-preserving; the
   measured modality ledger is the gate on every default chosen.

## Non-goals / open questions

- **Semantic pair-matching / balance checking is an explicit non-goal (§2a).** The tokenizer stays
  local + reversible; the model learns closure from the true distribution (asymmetric `[5,6)`, nested,
  and isolated cases included). Any future idea that proposes emitting a marker *because* a delimiter
  is balanced is out of scope by construction.
- Whether v2 should target **code corpora** at all decides `<>`, backtick, and single-quote entirely.
- Whether to **collapse `SPELL` and the bracket-open glue** (both mean "content-glues-inside") into a
  single mechanism, or keep them distinct for clarity — needs a design pass once pairs are first-class.
- Interaction of a chevron/tag pair with the planned reasoning delimiters (`<think>` etc.) in the
  reserved headroom — likely they should be *the same* literal-tag-span mechanism, not two.
