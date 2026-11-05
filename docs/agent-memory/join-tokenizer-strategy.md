---
name: join-tokenizer-strategy
description: "JOIN/implicit-space tokenizer — collapse common + specialize rare; ws/punct distribution MEASURED on FineWeb (5-token active set: JOIN/NEWLINE/PARA/OPEN+CLOSE_DQUOTE); word-N + impl pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Design strategy for the next tokenizer (supersedes the space-as-token scheme; depends on incremental BPE landing first so re-tokenization is cheap to iterate). Principle: **collapse the common case to implicit/free, specialize the rare case with explicit tokens, and DERIVE which to specialize from the corpus distribution** (same data-driven methodology as the sizing analysis).

Core encoding: single space between tokens is IMPLICIT (the 40.7% common case → free); a **JOIN** sentinel suppresses it (no-space boundaries: punct-attach + sub-word splits, ~16.2%) and doubles as the **composition operator** (`sun⌒day`) for OOV/morphology. Net 24.4% fewer tokens (see [[model-sizing-levers-from-corpus-stats]]).

Special-token inventory (specialize the rare/structured), all lossless-back to bytes:
- **RLE whitespace**: NEWLINE, PARA (\n\n), and run-length space tokens SPACE2/3/4/8 + TAB(s) for indentation — one token per indent instead of N spaces (user's idea; widths chosen from the measured run-length histogram).
- **Directional quotes**: bundle the common quote spacing into OPEN (` "` = space+quote+glue-after) vs CLOSE (`" ` = glue-before+quote+space) tokens. Disambiguates open/close (the model gets PAIRED tokens to attend start↔end of a quoted region) AND saves the JOIN. Classified at encode time by spacing context; both decode to `"` (lossless). Rare/mismatched cases fall back to bare quote + JOIN.
- Existing case markers <|cap|>/<|up|> stay (already "specialize the rare").

Decode = single-space-between-content default; JOIN clears it; whitespace/bundled tokens set spacing explicitly + clear it; markers set case of next content. The precise decode FSM + the lossless round-trip (decode(encode(x))==normalize_text(x)) is the correctness GATE (configurator already round-trip-checks per chunk).

Other aspects to plan for: digit-runs as word-units (BPE-mergeable) not space-separated standalones; URLs/code = JOIN-heavy rare-structured (accept, or specialize later); BPE stays within word-units (composition is model-learned, not a cross-JOIN merge — maybe a knob later); normalization already folds typographic glyphs (contract is to normalize_text, lossy on glyph identity); under-training risk for rare special tokens (only add tokens above a measured frequency); encoder greedy longest-match for ws runs (deterministic); versioned format + retrain (clean break). Validate by tokens/doc reduction AND an ablation that the model actually uses the structure.

Methodology step before implementing: extend the corpus analysis to MEASURE the whitespace-run-length histogram + punctuation-spacing-context distribution, and pick the specialized tokens from the head of that distribution (don't guess).

**MEASURED — DONE (committed 4456300, `docs/TOKENIZER_WS_PUNCT_STATS.md`).** 534MB FineWeb: single space = **97.2%** of ws runs (implicit confirmed); word→punct GLUE = **92%** (JOIN essential, ~24M/GB); newline 2.6% (NEWLINE), `\n\n` 0.13% (PARA); `"` **48.8% open / 41.1% close** (OPEN/CLOSE_DQUOTE decisive); `'` **81.6% glue/glue = contractions** (already in word-unit → SQUOTE deferred); multi-space **0.004%**, tabs **0%** → **SPACE_N/TAB NOT minted for prose (corpus-conditional**, mint per-corpus only above a floor). **Active set for clean prose = 5 new tokens** (JOIN, NEWLINE, PARA, OPEN_DQUOTE, CLOSE_DQUOTE) + existing CAP/UP — far below the speculative ~18-24. Scratch measure tool = `scratchpad/ws_punct_stats.py` (regex histogram, not committed; keeper → C++ configurator). **Still pending: word sub-token N histogram (JOIN-vs-SPELL §4 threshold) — needs BPE, measure IN the configurator.**

**LIBRARY IMPLEMENTED + TESTED (committed c282908)** — gated behind `LearnOptions::join_scheme` (default OFF, so production pipeline + existing models/corpus.tok stay valid). In `sub0::tok`: complete 256-byte base + always-on CAP/UP/JOIN/NEWLINE/PARA markers (ids 256-260, n_base=261); encode = implicit single space (free) + JOIN for glue AND intra-word sub-token splits + NEWLINE/PARA + verbatim-whitespace fallback; decode = token-level FSM (pending_space + case_mode, UP carried across a word's JOINed sub-tokens); deserialize infers scheme from TOK_JOIN presence. New ISOLATED test target `sub0_tok_tests` (links only sub0_tok + Catch2, no engine — builds even while training locks the engine DLLs); 47 assertions, both schemes. Test-corpus gotcha: legacy scheme only includes CAP/UP if the corpus USED them — give test corpora real capitals/all-caps (join scheme fixes this via always-on markers).

**SCHEME COMPLETE — directional quotes + SPELL (committed 51da4cf).** encode/decode rewritten around an explicit decoder-state mirror (`dps` pending-space + `in_spell`), n_base 261→265 (added OPEN_DQUOTE/CLOSE_DQUOTE/SPELL_START/SPELL_END). Directional DOUBLE quotes only (` "x`→OPEN, `x" `→CLOSE; bare fallback otherwise); single-quote directional DEFERRED per data (82% contractions). SPELL: word of N BPE sub-tokens → N=1 bare, N=2 `sub JOIN sub`, N≥3 `SPELL_START sub.. SPELL_END`; UP carried across the word's sub-tokens (JOIN or SPELL). Configurator reports the word-N histogram (tinystories: N1 92.9% / N2 4.4% / N≥3 2.7% SPELL → SPELL targets only rare long/OOV). tokens 10.56M→**7.50M (−29.0%)**, bytes/tok 2.951, lossless. sub0_tok_tests now 67 assertions. **Remaining: a retrain under the full scheme (the −4.2%..−8.7% bits/byte A/B was the impl-space-only n_base-261 version); FineWeb run; registry scheme tag.**

**Full design lives in `docs/TOKENIZER_DESIGN.md`** (committed). Added concepts:
- **Complete base alphabet (correctness fix)**: today base = corpus-USED bytes only, so a user prompt with an unseen byte is SILENTLY DROPPED in encode(). Must be the full 256-byte set (total tokenizer, byte-level fallback). Clean break (shifts base ids).
- **SPELL_START/END encapsulation**: a spaceless-region delimiter pair. Word-encoding rule by sub-token count N: N=1 bare, N=2 JOIN (cheaper), N≥3 SPELL-encapsulate (2 delimiters ≤ N-1 JOINs + signals spelled-out). One mechanism covers OOV/misspelled words, acronyms, CamelCase/function-names (`DoThisFooBar`). JOIN stays for sun+day (encapsulating 2 known tokens is worse).
- **Misspelling robustness (future)**: SPELL-region = the denoising unit, train corrupted→correct (`quix`→`quick`); fits the diffusion paradigm. See [[ch32-char-composition-oov-falsified]] (prior OOV attempt).
- User input/out-of-corpus must always be encodable → drives the complete base alphabet.
