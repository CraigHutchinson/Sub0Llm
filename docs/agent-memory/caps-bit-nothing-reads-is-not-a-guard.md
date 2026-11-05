---
name: caps-bit-nothing-reads-is-not-a-guard
description: "STANDING: a capability bit no consumer reads is not a safety mechanism -- and a guard above a bypassable seam is not a guard either. Two rounds, 2026-07-30: supports_decode was advertised falsely AND unread; then the raw externs beneath it had no check at all and the parity test bypassed them. Put refusals at the lowest callable seam; smoke-test on hardware before spending GPU-hours"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T19:16:11.993Z
---

`Sub0DeviceCaps` (device_backend.hpp) exists so consumers degrade around a MISSING capability instead of
being driven into an unimplemented path, and `docs/BACKENDS.md` states consumers "are DESIGNED to degrade
around zeros". On 2026-07-30 both halves of that were false for `supports_decode`:

* the CUDA backend advertised `supports_decode = 1` unconditionally, while its single-token decode path
  (`forward_one_device`) did not implement depth attention;
* `gpu_decode_try_enable()` (decode.hpp) gated only on `HAS_CUDA` + `sub0_dev_init()` success and never
  read the bit -- so correcting the advertisement alone would have changed nothing.

**Why:** a caps bit is documentation until a consumer branches on it. Two independent things must hold --
the backend tells the truth AND the consumer asks. Verifying only one gives false confidence, and the
combination is invisible to a green unit suite (nothing in 171 passing cases touched this path).

**How to apply:**
1. When adding a capability bit, land the CONSUMER's branch in the same change, and pin both with a test
   asserting the bit's value and the consumer's behaviour. Fixed + pinned in `ec1d8cf`.
2. When *removing* a capability (a build flavour that can no longer do X), grep for every reader of the
   bit before assuming the degradation path works.
3. Audit the remaining bits the same way -- `supports_interception` is the obvious next suspect.
4. **Before spending GPU-hours on a new mechanism, run the SHORT version of the entire pipeline the long
   run depends on**: train a few steps, then `report` (device eval + context curve + sample battery).
   Here that took ~10 minutes and caught a failure that would have hit ~1.6h into a ~5.7h run -- because
   `report`'s sample battery runs at EVERY eval, so this was never merely a `gen`-only problem.

## Sequel, 2026-07-30 (`f9f0019`): fixing the bit was still not enough -- a LOWER seam bypassed it

Correcting the advertisement and landing the consumer branch (`ec1d8cf`) fixed the *consumers*. It did not
fix the path underneath them. `sub0_cuda_forward_one` / `sub0_cuda_kv_reset` -- the raw externs -- had **no
depth-attention check at all**, so anything calling them DIRECTLY skipped `gpu_decode_try_enable()` and the
caps bit entirely and decoded without the depth mix.

Two things hid it:

* backend_cuda.cu's header comment asserted the guard existed ("hard-fails at RUNTIME rather than silently
  skipping the mix -- see that function"), naming a `forward_one_device` that **does not exist under that
  name**. The comment was the only implementation.
* the `[cuda][.slow]` decode-parity test calls the raw extern, so on a depth build it compared a
  depth-mixed full forward against a non-mixed decode and **passed at 98% top-1**. Top-1 agreement cannot
  see a missing V-rewrite. This is the concrete instance of
  [[depth-attention-stage0-done]]'s "a gradient check CANNOT validate this op -- presence test required".

**Why:** a guard placed above a bypassable seam is not a guard, it is a convention. The test that should
have caught it was itself routed around the guard, so green meant nothing.

**How to apply (adds to the list above):**
5. Put the refusal at the LOWEST seam that can be called -- the extern boundary -- not only in the polite
   wrapper. Then a bypass is impossible rather than merely discouraged.
6. When a comment claims a runtime check, grep for the function it names. Twice now the described guard
   was absent; the comment reads exactly the same either way.
7. A parity metric must be sensitive to the thing being tested. Top-1 agreement was the wrong instrument;
   assert the REFUSAL (a presence test) where the capability is absent.

Two sibling defects in the same pass, same shape -- a check that existed but could not fire:
`sub0-cuda-selftest` discarded every return code and hardcoded T=256, so six of its eight checks silently
no-oped on a short-`SEQ_LEN` build while it exited 0; and the no-clamp VRAM guard's chunk hint was gated on
`USE_DEPTH_ATTN`, suppressing it on the `DEPTH_ATTN_STRIDE=0` LoopSplit arms that lose ~1354 MiB to the
clamp. Both found only by RUNNING on hardware at two configs, never by compiling.

Prompted by the user's instruction to add a task to run arm D "once the feature is smoke tested" -- which
was correct, and which I had already skipped by launching the run straight after a 60-step throughput
probe. A probe that never reaches an eval does not exercise eval. See
[[depth-attention-arm-d-running]], [[regression-test-on-reproducible-bug]],
[[nanbeige-review-findings]] (2 silent defects that also survived full green suites).
