# Ch32 Phase 0 — M1 (OOV-cliff): results

**Status: M1 DONE. The word-level OOV cliff is real and large (8.08×) — P1 is justified.**

M1 is the first Phase-0 metric (BUILD_PLAN §Phase 0): does a word-level diffusion model fall
off a cliff on rare/unseen words? It measures this **without retraining** by splitting the
per-masked-token NELBO of a held-out stream by whether the target is a RARE type (the rarest
fraction of the vocabulary by train-frequency — a proxy for OOV-at-test).

Code: [`sub0diff/eval/oov_cliff.hpp`](../../include/sub0diff/eval/oov_cliff.hpp) +
[`src/oov_cliff.cpp`](../../src/oov_cliff.cpp). Test: `[oov_cliff]` in
[`test_diffusion.cpp`](../../tests/test_diffusion.cpp). CLI: `ch29 --oov_cliff [--oov_rare_frac F]`.

## Result

Word-level TinyStories, 3000 steps on GPU (held-out NELBO 2.17, recall 61.2%), masked at
t=0.5, rarest **50%** of types treated as the OOV proxy:

| bucket | NLL (nats) | masked tokens |
|--------|-----------:|--------------:|
| **common** types | **1.911** | 255 358 |
| **rare** types   | **15.436** | 596 |
| **cliff ratio**  | **8.08×** | — |

**The model predicts rare word types ~8× worse than common ones.** NLL_rare = 15.4 nats is
*above* the uniform baseline (−log(1/12238) ≈ 9.4 nats), i.e. the model doesn't merely fail to
know rare words — it **actively mis-predicts** them (their embedding rows are barely trained).
The rare bucket is small in token count (596) precisely because rare types are rare, but the
per-token cost is enormous. This is a large, real cliff — exactly the failure mode the P1
char-composition codec is designed to remove.

## Method notes

- `rare_type_mask(train_ids, vocab, frac)`: counts train-frequency per token id, marks the
  rarest `frac` of types (ties by id; freq-0 types — the OOV-est — sort first).
- `evaluate_oov_cliff`: masks held-out windows, denoises (batched B=32 forward, GPU-friendly —
  same machinery as the recall sweep), accumulates CE = `logsumexp(logits) − logit[target]` per
  masked token, bucketed by `is_rare[target]`.
- Cost is a single sweep on the model's device (GPU when training on cuda); negligible.

## What this gates

P1 **1c** (wire the codec around the word Denoiser + lookup/composed gated blend) must drive
**this ratio toward ~1** while in-vocab NLL (≈ the common bucket, 1.91) does **not** regress.
Re-run `--oov_cliff` on the 1c model and compare. If the ratio doesn't move, the codec isn't
buying OOV robustness and the design is falsified (BUILD_PLAN §Phase 1 kill-test).

Pairs with [`P1_RESULTS.md`](P1_RESULTS.md) (the codec itself, validated in isolation). The two
remaining Phase-0 metrics are **M2** (topic-drift, for P2) and **M3** (correlation-decay, already
done — see [`M3_RESULTS.md`](M3_RESULTS.md)).
