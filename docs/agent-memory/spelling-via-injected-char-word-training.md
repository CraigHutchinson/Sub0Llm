---
name: spelling-via-injected-char-word-training
description: "backlog SPIKE — can a model, trained with the tokenizer's word<->character-spelling mapping injected into the training set and exercised via a thinking/reasoning harness, learn to access the char tokens underlying its own word tokens? Word-token->chars direction; TANDEM with [[scratch-tokens-context-translation-layer]] (the inverse, chars->word, direction)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Reframed 2026-07-10 as an explicit SPIKE, and confirmed to work IN TANDEM with
[[scratch-tokens-context-translation-layer]]** — these are the two directions of the SAME
underlying char<->word capability, not two independent backlog ideas:
- **This memory (spelling)**: word-token -> chars. Can the model learn to DECOMPOSE/inspect one of
  its own word tokens down to its character spelling, on demand?
- **The other memory (scratch tokens)**: chars/fragments -> word. Can a word the model just
  CONSTRUCTED from character/fragment tokens be COLLAPSED into a single reusable token for the
  rest of the context, once established?
The scratch-token direction (efficiency payoff) plausibly NEEDS this spike's capability to exist
first: minting a context-local scratch token for a freshly-spelled word, and later correctly
expanding/recognizing it, both depend on the model having a robust char<->word mapping in the first
place — this spike is arguably the prerequisite research question for the other memory's efficiency
mechanism, not just a thematically related idea.

**The spike question itself (user's framing, 2026-07-10):** not "does injecting spelling data into
training help spell-correction" (the original framing below) but specifically — **how, and
whether, a model can be trained so that, WITHIN a thinking/reasoning harness (a chain-of-thought /
scratchpad mechanism, not the base forward pass), it can access the character tokens that make up
one of the tokenizer's word tokens.** I.e. give the model a tool-like or scratchpad-like ability to
"expand" a word token into its spelling mid-reasoning when a task needs it (spelling questions,
letter-counting, rhyme, acronym construction), rather than trying to bake spelling awareness
directly into the base next-token prediction geometry (which the caveat below shows actively hurts
prediction).

**Original framing, still relevant background:** the tokenizer already knows each word token's
character spelling (its BPE sub-token / byte expansion). **Inject that word<->character
relationship into the TRAINING set** — some token stream that relates a word token to its
spelled-out character tokens — so the model *learns the association*. The goal is not that the
model memorises real spelling, but that it can **"get at" the characters** of a word token when
needed (acronyms, wordplay, rhyme, explicit spelling).

**Payoff the user highlighted:** if the model can ingest chars -> emit the word token after
training, then feeding a **misspelt** word should map to the **most likely** word token — i.e.
spell correction falls out of the same learned char<->word relation. Also unlocks acronym
generation (the open problem: a word token is opaque to its own letters unless reached via a
spell/reasoning loop).

**Open design question (user's, unresolved):** what the injected training token stream should LOOK
like is unclear — e.g. interleaved `WORD = SPELL_START c h a r s SPELL_END` exemplars, a separate
"spelling" curriculum mixed into the corpus, a dedicated spell<->word objective, or contrastive
char-token<->word-token pairs. The 2026-07-10 reframing narrows this somewhat: the exemplars likely
need to look like REASONING TRACES (a scratchpad turn that spells a word out), not just bare
word<->spelling pairs in the base LM data, tying this to the roadmap's Stage 2 reasoning work
([[capability-roadmap-discrete-experts]]) rather than being purely a tokenizer/encoding-level change.

**Honest caveat (do not ignore):** [[ch32-char-composition-oov-falsified]] found spelling !=
prediction geometry under a weight-tied head — even a PERFECT spelling autoencoder HURT rare-word
prediction (cliff 4.72->7.68x). So naively forcing the LM-head embedding to encode spelling can
damage the prediction manifold. The injection must teach the relation WITHOUT collapsing the two
geometries — likely a *separate* readout/path or an auxiliary objective, not the tied head, and the
reasoning-harness framing above (spelling as an on-demand SIDE capability invoked in a scratchpad,
not baked into the base prediction path) is one plausible way to sidestep this exact failure mode,
not just a stylistic choice. Relates to [[char-level-tokenization-lever]] (chars are the lever for
form) and the SPELL token (TOKENIZER_DESIGN §4). Status: BACKLOG SPIKE, after the encoding-level
wins ([[tokenizer-dogfood-fuzz-and-source-pathologies]]) and realistically gated on Stage 2
reasoning-harness groundwork existing at all.
