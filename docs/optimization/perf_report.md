# Sub0Llm performance report

Generated 2026-09-25T11:09:49Z -- label `ppl-harness-v1`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | native vs bf16, 2418 tokens | -0.0124 nats/token (95% CI -0.0354..+0.0106), ppl x0.9877, top-1 agree 84.5% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| bf16 | 15.0338 | 2.7103 | 0.4835 | 4.00 |
| native | 14.8488 | 2.6979 | 0.4864 | 5.49 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
