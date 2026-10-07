# Sub0Llm performance report

Generated 2026-10-07T09:00:29Z -- label `O12-O9-v2`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | q8fast vs base, 9631 tokens | +0.0056 nats/token (95% CI -0.0041..+0.0153), ppl x1.0056, top-1 agree 88.1% | PASS |
| `G-PPL` | no_actsuper vs base, 9631 tokens | -0.0019 nats/token (95% CI -0.0124..+0.0086), ppl x0.9981, top-1 agree 87.7% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v2, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s (scoring run) | Speed rounds, tok/s |
|---|---:|---:|---:|---:|---|
| base | 11.2956 | 2.4244 | 0.5191 | 7.83 | 6.93, 7.28 |
| q8fast | 11.3589 | 2.4300 | 0.5177 | 8.29 | 7.45, 7.16 |
| no_actsuper | 11.2744 | 2.4225 | 0.5205 | 6.83 | 6.01, 5.88 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
