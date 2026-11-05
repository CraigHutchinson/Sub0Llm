---
name: capability-roadmap-discrete-experts
description: docs/ROADMAP.md created -- data/capability roadmap from tiny fluency toward conversational/reasoning/tool-use/agentic-C++-coding via a family of discrete, purpose-sized models (NOT classic MoE); corpus catalog by capability axis; stage 1 (conversational) is blocked on a turn/role marker-token design gap
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Context:** while the d512/d768 TinyStories sweep trained in the background, user asked to research
denser training resources, consider conversational datasets as the next step, and lay out a roadmap
toward "agentic coding for C++" as a long-term stretch goal -- plus an architecture idea: small
*discrete* dedicated expert models (not classic per-token-gated MoE) sized per task, with a
conversational front-end that can converse + look things up rather than needing encyclopedic knowledge
baked into weights. Wrote `docs/ROADMAP.md`, linked from README's new "## Roadmap" section.

**Thesis captured:** a dense model's params pay for two things that compress very differently --
world knowledge (wants scale/coverage) and behavior (grammar/coherence/reasoning/tool-use/turn-taking;
compresses well, mostly a data-density question). TinyStories already proved the behavior half of this
inside the project (few-M-param fluency from a behavior-dense synthetic corpus). FineWeb-Edu is
knowledge-dense but behavior-sparse -- right for the "capacity vs. a real broad corpus" question the
Chinchilla sweep answers, wrong for what comes next.

**Corpus catalog by axis (see the doc for full detail + license caveats):**
- A. Foundational fluency (done): TinyStories.
- B. Broader knowledge (parked): FineWeb-Edu (in use); Cosmopedia (HF synthetic textbook-style,
  what SmolLM2 trained on -- same SmolLM2 that anchors `kAspectMax` in `autosize()`) flagged as the
  strong "understanding-dense not just knowledge-dense" alternative/supplement worth a real look later.
- C. Conversational (recommended NEXT step): Dolly-15k (small, real, permissive -- recommended
  starting point, same TinyStories-sized fast-iteration role) -> Alpaca/Alpaca-GPT4 -> OASST1/2 (real
  multi-turn) -> UltraChat-200k (later scale-up).
- D. Reasoning: TinyGSM (direct TinyStories-for-math sibling, same "does the trick work for reasoning"
  precedent) / GSM8K / MetaMathQA / SlimOrca.
- E. Tool-use: Glaive-function-calling-v2 (most directly reusable) / Gorilla-APIBench / ToolBench.
- F. Code / the stretch goal: The Stack (C++ filterable) / CodeSearchNet (C++ coverage THIN, mostly
  Python/Java/JS/Go/PHP/Ruby) / competitive-programming corpora (genuinely C++-heavy in practice,
  pairs with a compile-and-run signal). **Named gap: public C++-agentic-coding data is scarce
  industry-wide** -- likely-necessary bootstrap is the project's OWN source tree + dev history
  (commit messages, PR-style diffs, docs/) as a small, genuinely-C++, exactly-matched self-hosting
  corpus.

**Architecture: discrete purpose-sized experts, explicitly NOT classic MoE.** Classic MoE = one model,
per-token gated routing across shared FFN sub-layers, all experts co-resident, trained end-to-end.
What's proposed = fully SEPARATE, independently-sized, independently-trained models dispatched at the
query/subtask level -- closer to a system of specialists than a gated network. Maps directly onto
EXISTING infra rather than needing new kernel work: the Chinchilla sweep +
[[model-identity-date-tag-and-comparison-table]]'s `models --metrics` table already characterizes
capability-vs-size across a model family; extending it across different corpora/tasks and picking a
right-sized model per task is a small step, not a new architecture. No shared-weight gating GEMM needed
in `backend_cuda.cu` -- dispatch happens outside any single model's forward pass.

**New components this implies, none built, explicitly future work:** a router/dispatcher (can start
as a keyword classifier, no need to be learned initially); a conversational orchestrator (smallest
general-purpose model in the family, talks + decomposes + delegates rather than doing everything);
and -- the real bottleneck, bigger than it looks -- **a task-success eval harness**. Every existing
measurement (`report`'s NELBO/bits-per-byte, `autotemp`'s coherence grid) scores language-modeling
quality only, not "was the tool call's arguments correct" or "did the generated code compile and pass
tests." Stages 2-4 below are each blocked on this before they're meaningfully evaluable, not just on
sourcing data.

**Staged progression (mirrors the project's own small-corpus-first methodology):**
- Stage 0 DONE: base LM (TinyStories sweep + parked FineWeb-Edu), the reusable sweep/comparison/
  autotemp substrate every later stage builds on.
- Stage 1 NEXT: conversational competence, Dolly-15k-scale first. **Concrete blocking engineering gap
  identified, not yet scoped as a plan**: the pipeline is one flat token stream with only
  document-boundary markers ([[eos-token-and-fixed-marker-scheme]]) -- no turn/role (user vs assistant)
  representation exists yet. Needs new fixed marker tokens in the SAME family as the existing
  TOK_EOS/doc-boundary scheme (not a new tokenizer mechanism) before any conversational corpus is
  ingestible at all. This is the literal next engineering task once the current sweep is done.
- Stage 2: reasoning density (GSM8K/TinyGSM-style), needs the eval harness.
- Stage 3: tool use (Glaive-scale), needs a minimal tool-call execution harness (even a stub).
- Stage 4 (stretch/final): agentic C++ coding -- curated/self-authored C++ corpus, a compile-and-test
  execution harness (biggest new infra investment on the list), plausibly the discrete-experts split
  actually implemented (dedicated coding specialist orchestrated by the Stage-1 front-end).

**Honest caveats flagged in the doc itself:** every license claim above is from training knowledge, not
verified -- re-check before actual use. Hardware budget is one 8GB laptop GPU; stretch goal is
explicitly far out, not a near-term compute target. Eval methodology, not data availability, is the
real bottleneck for stages 2-4.

**Not yet done:** no commit (repo convention -- only commit when explicitly asked); this is planning/
documentation only, no code changes.
