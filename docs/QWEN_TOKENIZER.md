# WP5a — the real Qwen tokenizer

Status: **DONE.** `sub0::qwen_tok` encodes and decodes exactly like the real
Qwen3.8-Flash-Next tokenizer, gated case-by-case against the real thing.

**3041 assertions across 9 test cases, all passing** (`sub0_frontend_tests "[qwen_tok]"`), of which
1936 are fixture rows produced BY the real tokenizer: 269 encode+decode cases, 269 pre-tokenization
cases, 1389 NFC cases, 9 decode-only cases.

---

## 1. Why this exists

WP4 transplanted real Qwen weights into this engine and ran a real forward pass. The next goal is
**interactive inference** — type a prompt, get text back. Three things are missing; this is the first.

This project has its own from-scratch BPE tokenizer (`include/sub0/tokenizer.hpp` +
`include/sub0/casing.hpp`), trained on this project's own corpora, with its own marker set and its own
vocabulary. **The transplanted Qwen weights have no relationship to those token ids.** Feeding a sub0
token id to the real model is meaningless — the embedding row it selects is arbitrary. Talking to the
real model requires the real model's own tokenizer.

So WP5a adds a **second, separate** tokenizer. Nothing in `tokenizer.hpp`/`casing.hpp` is touched, and
the default build's test suite is byte-for-byte unchanged (AGENTS.md §4 — verified: 235 test cases /
118,416 assertions with and without this change).

| file | what |
|---|---|
| `include/sub0/qwen_tokenizer.hpp` | public API: `pretokenize()`, `class Tokenizer` |
| `src/qwen_tokenizer.cpp` | the transliterated regex, the BPE merge loop, the file readers |
| `include/sub0/qwen_unicode.hpp` | UTF-8 stepping, the four character classes, NFC |
| `include/sub0/qwen_unicode_tables.hpp` | generated data only |
| `scripts/qwen_tokenizer_tables.py` | regenerates + verifies that table header |
| `scripts/qwen_tokenizer_fixture.py` | regenerates `tests/fixtures/qwen_tokenizer/` |
| `tests/qwen_tokenizer_tests.cpp` | the gate |

Engine-free, in `sub0_frontend`, like `gguf.hpp` and `transplant.hpp`: no generated config, no engine,
unit-testable without a compiled model, and usable by a future CLI tool without linking the engine.

**Out of scope, deliberately** (AGENTS.md §8 — land the stage that is wired up): no chat-template /
prompt-formatting layer, and no `gen_stage.cpp` wiring. The interactive generation loop needs the full
48-layer model, which does not exist yet.

## 2. What the reference actually is

Verified this pass by reading the real files and probing the real library — not recalled.

```
text ──► split out the 33 added/special tokens, matched LITERALLY on the RAW bytes
     ──► NFC-normalize each remaining span
     ──► split into chunks with the pre-tokenization regex (§3)
     ──► map every byte of a chunk through the GPT-2 bytes_to_unicode alphabet
     ──► greedy BPE by merge rank, per chunk; merges never cross a chunk boundary
     ──► vocabulary lookup
```

No unknown token (`unk_token: None` — byte-level BPE makes every input representable), no BOS/EOS
insertion (the post-processor is a plain ByteLevel), no prefix space (`add_prefix_space: false`).
`vocab_size` 248,044 base + 33 added = 248,077; the model's own `VOCAB` axis is 248,320, so rows
248,077..248,319 are checkpoint padding that `decode` has to survive.

### 2.1 The oracle is NOT `AutoTokenizer` — a real trap, caught before it baked in

The obvious gate, `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.8-Flash-Next")`, is
**wrong for this model**, and it is wrong in a way that produces plausible output.

`tokenizer_config.json` declares `tokenizer_class: "Qwen2Tokenizer"`. In transformers 5.x,
`Qwen2Tokenizer` does not load `tokenizer.json`'s pipeline at all — it **rebuilds** the pipeline from
`vocab.json` + `merges.txt` using its own module-level constant:

```python
# transformers/models/qwen2/tokenization_qwen2.py
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

That regex has **no `\p{M}`**. The model's own `tokenizer.json` and its own
`tokenizer_config.json:pretokenize_regex` field both carry the `\p{M}` version, and transformers' own
`model_type → tokenizer` map sends this model's `qwen4_exp` to `Qwen3_5Tokenizer`, whose constant
**does** have `\p{M}`. Three independent sources say `\p{M}`; only the stale `tokenizer_class` string
points at the older class, and `AutoTokenizer` obeys that string.

Measured differences in real token ids — any script whose marks have **no precomposed form**, so NFC
leaves them standing for the regex to see:

| input | `AutoTokenizer` (no `\p{M}`) | the model's own `tokenizer.json` |
|---|---|---|
| `กัน` (Thai) | `[24400, 64020]` | `[148783]` |
| `क्ष` (Devanagari) | `[62516, 157451]` | `[151858]` |
| `ന്റ` (Malayalam) | `[148770, 70105, 150509]` | `[149694, 150509]` |
| `أَهْلاً` (vocalised Arabic) | `[67321, 162072, 57392, 78178, 148536]` | `[67321, 162072, 57392, 155440]` |

Without `\p{M}` a mark ends a letter run, so merges that should span "letter + mark" never fire. Every
Indic, Thai, Hebrew-with-points and vocalised-Arabic prompt would be tokenized wrongly — and nothing
downstream would look broken, because the ids are all valid. (Latin accents mostly hide the bug:
`"a" + U+0300` NFC-composes to `à` before the regex ever runs, so both pipelines agree there.) This is
exactly the failure AGENTS.md §5 exists to prevent.

**The oracle used here** is `tokenizers.Tokenizer.from_file(<the model's tokenizer.json>)` — the
model's own exported artifact — cross-checked case-by-case against `Qwen3_5Tokenizer`. All 269 cases
agreed on both encode and decode.

### 2.2 Which files, and why

`vocab.json` + `merges.txt` + `tokenizer_config.json`, not the unified `tokenizer.json`:

* `merges.txt` needs **no JSON at all** and makes "rank == line number" explicit and auditable.
* `vocab.json` is the simplest possible simdjson shape — a flat `string → int` object — matching this
  project's existing single-pass ondemand convention (`src/run_config.cpp`).
* `tokenizer_config.json` is the authoritative source for `added_tokens_decoder` and eos/pad.
* It avoids depending on `tokenizer.json`'s schema, which is a moving target across `tokenizers`
  releases (merges have shipped as both a list of strings and a list of pairs).

Verified before choosing: `vocab.json` is byte-identical to `tokenizer.json`'s `model.vocab`, and
`merges.txt` to `model.merges` (247,587 rules, and this file has **no** `#version:` header line — the
first line is a real merge rule). The loader still skips a leading `#version:` line, matching the
reference, and does so by that literal prefix rather than by "starts with `#`" — `#` is itself a
byte-alphabet character, so `"# #"` is a legitimate merge rule.

### 2.3 Decode convention

`decode` keeps every token's text, including special tokens. That is transformers' own default and the
only behaviour under which `decode(encode(x)) == NFC(x)`. (`tokenizers`' raw `Tokenizer.decode`
defaults to `skip_special_tokens=True`, which is a display policy, not a decode. No flag was added for
it — AGENTS.md §8; a caller that wants to hide `<|im_end|>` can filter its own id list.)

## 3. The Unicode-regex problem, and how it was actually solved

The real pre-tokenization regex, verbatim:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

**It cannot be handed to `std::regex`.** No standard C++ implementation supports Unicode property
escapes (`\p{L}`, `\p{N}`, `\p{M}`), and none supports the negative lookahead `(?!\S)`. Approximating
the classes with ASCII tests silently mis-tokenizes every non-English input.

The solution taken: **hand-transliterate it**, one function per alternative, in the alternation's own
order, reproducing leftmost-first-with-backtracking explicitly. Each clause returns the exact extent
its own backtracking would settle on (see the per-clause comments in `src/qwen_tokenizer.cpp`); the
two non-obvious ones:

* `\s*[\r\n]+` — the greedy `\s*` gives back exactly as much as it must, so the match ends immediately
  after the **last** CR/LF inside the whitespace run.
* `\s+(?!\S)` — the lookahead is satisfiable only at end-of-input or before whitespace, so this is
  "the whole run if it ends the input, otherwise the run minus its final character, and no match at
  all for a single-character run mid-text". This is what makes `"  a"` split `[" ", " a"]`.

An ICU dependency was not taken: the only thing this project needs Unicode for is these four
predicates plus NFC, and ICU would dwarf the engine.

### 3.1 How the character classes were verified — not asserted

The classes are emitted from Python `unicodedata` 16.0.0 and then **read back out of the reference's
own regex engine** and compared, for **every one of the 1,112,064 non-surrogate codepoints**. Three
probes, each constructed so the other clauses must fail:

| probe | holds iff |
|---|---|
| `pre_tokenize("a" + cp)` is 1 chunk | `cp ∈ L ∪ M` (clause 2 continues a letter run) |
| `pre_tokenize(cp + cp)` is 2 chunks | `cp ∈ N` (clause 3 is a bare `\p{N}` — one per chunk) |
| `pre_tokenize(cp+cp+".")` is not 1 chunk | `cp ∈ \s`, for `cp ∉ L∪M∪N` (clause 4's `+` excludes whitespace, so a whitespace codepoint cannot glue itself to the `.`) |

Result: **0 disagreements** for `\p{L}|\p{M}`, **0** for `\p{N}`, **0** for `\s`.

`L` and `M` cannot be separated this way — and provably need not be. `M` appears positively only
inside `[\p{L}\p{M}]+`, and the only place `L` and `M` differ is the optional prefix class
`[^\r\n\p{L}\p{N}]`, which admits a Mark. But a Mark taken as the prefix and a Mark taken as the first
character of the greedy `+` reach the *same* match extent, so the two are interchangeable throughout.
The split is therefore taken from `unicodedata` alone, and only checked for consistency (every Mark
must lie inside the reference's own `L|M` set — 0 outside). The C++ still implements the clause
literally, so that equivalence never has to be trusted.

Two traps this found, both real:

* **`\s` is the full Unicode `White_Space` property, not the ASCII six.** U+00A0, U+2000–U+200A,
  U+3000 all split like spaces. An ASCII-only `\s` changes the chunking of any text with typographic
  spaces.
* **`(?i:'s)` uses Unicode case folding, so it also matches `'` + U+017F LATIN SMALL LETTER LONG S.**
  Confirmed by sweeping every codepoint that case-folds to any of `s t r e v m l d` and testing each
  against the reference: U+017F is the only non-ASCII one. (U+212A KELVIN SIGN folds to `k`, for which
  there is no alternative, so it correctly does *not* split.)

### 3.2 NFC — needed, and verified

Normalization **is** applied, confirmed three ways rather than assumed: `tokenizer.json` declares
`"normalizer": {"type": "NFC"}`; both transformers tokenizer classes hardcode `normalizers.NFC()`; and
behaviourally `encode("café") == encode("café")`.

NFC is implemented from the UAX #15 algorithm — full canonical decomposition, canonical ordering,
canonical composition — over generated tables (934 combining classes, 2081 full decompositions, 961
composition pairs, Hangul handled arithmetically). Verified against the reference normalizer:

* `NFC(cp)` matches for **all 1,112,064** codepoints;
* NFD→NFC round-trips for all **2081** canonically-decomposable codepoints;
* all **961** composition pairs compose identically;
* all **11,172** algorithmic Hangul syllables compose identically;
* composition **exclusions** agree exactly — **0** spurious compositions (this is the one a naive
  "invert the decomposition table" implementation gets wrong).

### 3.3 The one measured divergence, stated honestly

The installed `tokenizers` 0.23.1 carries a Unicode table **older than 16.0.0**. Two consequences,
both measured:

1. **21 codepoints** have canonical data in Unicode 16.0.0 that the reference build has none for at
   all (Todhri, Tulu-Tigalari, Dives Akuru, Gurung Khema, Kirat Rai — scripts added in 15.1/16.0):
   `U+105C9 U+105E4 U+11383 U+11385 U+1138E U+11391 U+113C5 U+113C7 U+113C8 U+11938 U+16121..U+16128
   U+16D68..U+16D6A`. The generator classifies these automatically (a codepoint the reference cannot
   even *decompose* is one it does not know) and reports them separately from real disagreements.
2. **Sequences of two or more combining marks** can canonically order differently, because the
   reference's combining-class table lacks values for marks added later. Exhaustively measured:
   * **1-mark sequences: 8,406 tested (9 bases × 934 marks), 0 divergent.**
   * 2-mark sequences: 872,356 tested, 71,212 divergent (8.16%).

   Ordering of the long-established marks (U+0300–U+036F and friends) agrees; the divergence is
   entirely driven by marks whose class the reference does not know.

**The choice made: implement Unicode 16.0.0.** It is the standards-correct answer, it is what the
reference itself will produce once its crate updates, and the divergence class does not occur in real
text (single-mark sequences — every accented letter, every Vietnamese tone, every Devanagari
matra — agree exactly). The fixture therefore contains realistic combining-mark cases and not
adversarial multi-mark astral sequences, and this section is the record of what was deliberately not
gated.

## 4. Implementation notes worth knowing

* **BPE starts from raw bytes.** The byte alphabet's only job is to give each byte value a distinct
  vocabulary entry; `byte_id_[256]` holds the resulting byte → id table, so the merge loop never
  materialises the alphabet string. Doing it the other way round — encode to the alphabet, then read
  *those* bytes — is a real, silent bug that was hit and fixed during this work: a space came out as
  two tokens instead of one, because the UTF-8 of `Ġ` got tokenized instead of the space.
* **Merges are keyed on `(left id, right id)`, not on strings.** Every merge rule's result is itself a
  vocabulary entry, so every symbol always has an id — no string hashing in the merge loop.
* **The merge loop is a linked list + a min-heap**, mirroring the reference's own shape. A stale heap
  entry is detected by comparing the *rank*: ranks are unique per pair, and a position's pair can
  never repeat because every merge strictly extends the left symbol. This keeps a pathological chunk
  (a megabyte of one repeated symbol) at O(n log n) rather than the O(n²) a rescan loop would cost.
* **Added tokens are matched on the raw bytes, before normalization**, leftmost-longest — the
  reference's own order (its added-vocabulary split runs first, and all 33 are `normalized: false`).
* **Vocabulary storage is one blob + offsets**, twice (token text, decoded bytes), rather than ~248k
  separate `std::string`s.
* **Scratch is reused** across calls (`static thread_local`) in the per-chunk paths, per AGENTS.md §1.

## 5. The gate

`tests/qwen_tokenizer_tests.cpp`, four fixture files, four independently-failing layers — so a break
says *where* it broke, not just that some id moved:

| fixture | rows | needs the 10 MB vocabulary? |
|---|---|---|
| `nfc_cases.tsv` | 1389 (incl. every codepoint whose NFC ≠ itself) | no |
| `pretokenize_cases.tsv` | 269 (chunk boundaries only) | no |
| `encode_cases.tsv` | 269 (ids **and** the reference's own `decode(ids)`) | yes |
| `decode_cases.tsv` | 9 (a character cut in half; the model's padding rows) | yes |

Coverage: empty/whitespace runs (spaces 1–8, tabs, LF, CR, CRLF, mixed), ASCII words with and without
a leading space, contractions including every case variant and the `'ſ` fold, digit sequences
(per-digit, confirmed), CJK / Japanese / Korean (syllables *and* jamo) / Cyrillic / Greek / Hebrew /
Arabic / Thai / Devanagari, accented Latin in both NFC and NFD spellings, Vietnamese two-mark
stacks, emoji including ZWJ sequences / flags / skin tones, astral-plane codepoints, Unicode
whitespace (NBSP, U+3000, U+2028/9, U+1680, U+205F, U+202F), soft hyphen, ZWSP, BOM, NUL, code and
markup, all **33** added tokens each alone / embedded in text / surrounded by whitespace, plus partial
and nested marker cases, and prose paragraphs. Round-trip is asserted on every one.

Alongside the fixtures are **mutation-style checks** (the `transplant_tests.cpp` pattern) — each names
the specific wrong-but-plausible implementation it rules out: `\p{N}+` instead of a bare `\p{N}`; a
dropped `(?!\S)`; an ASCII-only `\s`; an ASCII-only `tolower` in clause 1; BPE run over un-split text;
a skipped normalizer; "decompose but never recompose"; only pair (not singleton) decompositions;
building the composition table without applying exclusions; and a forgotten added-token split.

### Running it

The NFC and pre-tokenization layers need nothing but this repo. The encode/decode layers additionally
need the real `vocab.json` / `merges.txt` / `tokenizer_config.json` (~10 MB) — too large to commit
under this project's own convention (models and corpora are fetched, not versioned), so they are
located at runtime and **skip with a WARNING** if absent, exactly like the other qwen4 fixture tests.

```sh
huggingface-cli download Qwen/Qwen3.8-Flash-Next vocab.json merges.txt tokenizer_config.json \
    --local-dir data/qwen_tokenizer
# ...or point at an existing copy:
export SUB0_QWEN_TOKENIZER_DIR=<dir containing those three files>

cmake --build out/build/native --target sub0_frontend_tests
./out/build/native/tests/sub0_frontend_tests "[qwen_tok]"
```

Regenerating the generated artifacts (both refuse to emit if the reference disagrees):

```sh
python scripts/qwen_tokenizer_tables.py  <snapshot-dir> --verify full   # ~25 min, the exhaustive sweep
python scripts/qwen_tokenizer_fixture.py <snapshot-dir>
```

## 6. Known limits

* **Malformed UTF-8 input is outside the reference-gated behaviour.** The reference is fed a Rust
  `str`, i.e. valid UTF-8 by construction, so there is nothing to compare against. This
  implementation defines it anyway: `decode_utf8` consumes a bad byte on its own, so each stray byte
  becomes its own pre-token chunk and then its own byte-level token, and `nfc()` re-encodes it as a
  codepoint (not byte-identical for bytes ≥ 0x80). Callers that must preserve arbitrary bytes should
  not normalize.
* **Decode of ids ≥ 248,077 yields nothing** for those ids — matching the reference, and required
  because the model's `VOCAB` axis is 248,320.
* The 2-mark canonical-ordering divergence in §3.3.

## 7. What is next for interactive inference

1. ~~the real tokenizer~~ — this work package.
2. a chat-template / prompt-formatting layer (`chat_template.jinja` ships with the model).
3. the generation loop against the full 48-layer model, wired into `gen_stage.cpp`.
