# Sub0Llm performance report

Generated 2026-09-29T13:56:24Z -- label `O12`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | q8fast vs base, 2418 tokens | +0.0103 nats/token (95% CI -0.0096..+0.0302), ppl x1.0104, top-1 agree 86.8% | inconclusive |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| base | 14.7035 | 2.6881 | 0.4876 | 6.19 |
| q8fast | 14.8559 | 2.6984 | 0.4888 | 6.64 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
