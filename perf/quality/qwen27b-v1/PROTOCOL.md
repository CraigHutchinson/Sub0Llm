# Dense 27B quality qualification v1 — frozen before candidate scores

Tracking #63. This is a new fixture for the bounded qwen35 dense backend. The earlier MoE
`ppl_blend_v2` depends on corpora absent from this machine and its token IDs belong to another tokenizer;
its historical perplexity values cannot serve as this model's baseline. Its deciding G-PPL limit is
retained: the upper 95% confidence bound of paired candidate-minus-reference NLL must be <= 0.03
nats/token. No numerical tolerance is relaxed.

## Frozen workload

`fixture.json` pins input and next-token target IDs before candidate model scores are collected.
Four domains have 1,024 scored targets each: literature, scientific prose, civics and software text.
Each uses its first 1,025 independent-tokenizer IDs, split into eight windows. A window has 128 input
positions and 128 next-token targets; the final target is predicted after the last supported input.
State resets between windows; no implicit BOS, sampling, unused-token filtering or cross-window state.
Every source target in the selected prefix is scored exactly once. Total: 32 windows / 4,096 targets.

The original text excerpts and source hashes are preserved beside this document. The three books'
primary pages declare public domain in the USA; see `sources.json` for URLs, authors/edition provenance
and deterministic excerpt positions. Software text is frozen from this repository at c21f0da.
The native tokenizer agrees exactly with pinned llama.cpp on all four complete text excerpts, not only
on the selected prefixes. This is an English prose/software qualification set, not a claim about all
languages, tasks or maximum-context model capabilities.

## Reference and mandatory gates

- Independent reference: llama.cpp cf659f240b232e15700b1721bbb43c1d973c52e9 plus the committed
  diagnostic float-activation patch, CPU repacking OFF, F32 K/V and flash attention disabled.
  This is the quality/math oracle; the clean optimized reference remains the speed comparator.
- All 4,096 candidate targets are paired with that reference on identical raw input/target IDs.
  All windows and domains must be present; omissions, duplicates, mismatched IDs, nonfinite values,
  stale artifacts or failed processes refuse a verdict. No early passing verdict is allowed.
- Global and per-domain token-paired 95% CIs use the existing mean +/- 1.96 standard errors contract.
  Additionally require conservative window-mean CIs (t critical 2.039513 for 32 windows, 2.364624
  for eight per domain), and a domain-mean global CI (t critical 3.182446 for four domains).
  Every upper bound must be <= +0.03. If any lower bound is > +0.03, fail; otherwise an upper
  bound above the limit is inconclusive and requires more predeclared text, never a weaker threshold.
- Preserve full logits for all 128 positions of the first window in every domain. Check candidate
  and retained original scalar mode against the independent reference at the unchanged relative L2
  limit 0.001 and full vocabulary 248320. This exercises all recurrent/full-attention transitions
  through the exact supported context boundary. Original scalar checks need not rescore the entire
  quality corpus: the independent reference is the deciding quality baseline.
- Verify the score implementation independently against analytic cases and recompute selected
  captured NLLs in Python from binary logits. Shared score serialization is not itself an oracle.
- Every model process must stay below 8 GiB peak RSS. Record exact model/scorer/library identities,
  commands, token traces, output hashes and process metrics. Raw qualification elapsed times are
  diagnostics, not controlled throughput claims.

## Promotion and continued scope

A stage with an explicit generated `native-all` default is evaluated before touching preserved live
artifacts. AUTO stays baseline until qualification passes. Then resolve AUTO for the qualified model
build class and supported ISA, preserve an explicit baseline override, and compare final artifact
identities with the qualified binaries. Re-test appropriately if executable identity changes.

Actual chat reset, overflow refusal, replay, EOS/length completion and context-dependent replies remain
mandatory. Run a staged controlled performance comparison for the final promoted artifact. Keep the
original artifact for rollback. Qualification here does not finish #29 engine integration, batched
prefill, reference performance parity or the measured hardware-limit campaign.
