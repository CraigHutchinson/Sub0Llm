# Sub0Llm performance report

Generated 2026-10-07T12:42:56Z -- label `O13-regime2`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | cache17 vs reactive, 2000 tokens | +0.0000 nats/token (95% CI +0.0000..+0.0000), ppl x1.0000, top-1 agree 100.0% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v2, decode path, under a RAM ballast leaving 10 GiB for experts)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s (scoring run) | Speed rounds, tok/s |
|---|---:|---:|---:|---:|---|
| reactive | 11.9443 | 2.4803 | 0.4985 | 7.08 | 6.76 |
| cache17 | 11.9443 | 2.4803 | 0.4985 | 7.74 | 7.69 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
