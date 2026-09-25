# Sub0Llm performance report

Generated 2026-09-25T12:31:55Z -- label `O8-tp-r3`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | split vs base, 300 tokens | +0.0000 nats/token (95% CI +0.0000..+0.0000), ppl x1.0000, top-1 agree 100.0% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| base | 25.7381 | 3.2480 | 0.3633 | 3.80 |
| split | 25.7381 | 3.2480 | 0.3633 | 5.79 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
