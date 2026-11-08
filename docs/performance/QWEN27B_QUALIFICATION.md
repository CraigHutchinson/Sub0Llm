# Dense 27B full-context and quality qualification

Tracking #63. The measured candidate from #62 remains opt-in until this qualification is complete.
The frozen [v1 protocol](../../perf/quality/qwen27b-v1/PROTOCOL.md) preserves the existing G-PPL
+0.03 nats/token upper-95%-CI limit and adds conservative domain/window checks. It covers 4,096
teacher-forced targets across 32 full 128-position windows. Missing/inconclusive data cannot pass.
The old MoE fixture is unavailable here and has a different tokenizer; its historical PPL is not reused.

## Foundation validation

- Pinned independent llama.cpp and native tokenizers agree on all four complete source texts:
  3,157 literature, 2,641 science, 2,586 civics and 8,245 software tokens. Selected prefixes are frozen
  before candidate corpus scores; source text, IDs, hashes and attribution are committed together.
- The score writer uses f64 log-sum-exp on unfiltered raw logits and retains actual target IDs.
  Independent Python analytic checks cover full-vocabulary normalization, target selection, additive
  shift invariance and invalid inputs. Cache/judge tests cover missing windows, wrong IDs, nonfinite
  data, changed output hashes, memory limits and an inconclusive cluster CI despite a token-level pass.
- The new float-reference scorer reproduces the established sixteen-position reference exactly
  (relative L2 zero). The staged candidate passes the unchanged 0.001 gate (worst 0.0000791262).
  Python recomputation from captured binary logits matches recorded NLLs within 1e-10.
- Eight CTest groups and 18 script tests pass. This validates the rig, not yet the 4,096-target gate.

## Stage and run

The stage on this machine is `out/build/qwen-qual/`. `sub0llm-configure` now accepts
`--dense-projection-kernel auto|baseline|scalar|native|native-all`, emitting a scheduling-only generated
macro. AUTO remains baseline while qualification is pending. Explicit `native-all` stages the final
candidate default; unrelated generated model/layout constants remain unchanged. Older generated
headers without the macro keep baseline behavior. The runtime selector is an explicit experiment override.

The local stage was generated with:

```sh
out/build/pr5_check/sub0llm-configure --profile qwen3.8-27b --compute 0 \
  --dense-projection-kernel native-all -o out/build/qwen-qual/generated/sub0_config.hpp
```

Compile `tools/sub0llm-dense-token.cpp` with that generated include directory, `-std=gnu++26 -O3
-march=native -fopenmp`, the project's frontend/simdjson libraries and CLI11 include path. The reference
uses `tools/reference/score_llama_sequence.cpp`, the pinned diagnostic patch and Release/native CPU
libraries with CPU repacking OFF. Its F32 K/V, flash-disabled and single-position evaluation settings
are explicit in the source. Staged reference libraries use origin-relative paths and are preserved beside
the executable; no live temporary-directory library dependency is needed.

The ignored `out/build/qwen-qual/run-manifest.json` records exact commands, libraries, model hash,
fixture/protocol pins and artifact locations. Capture uses the shared Sub0Math `sub0perf` lock and
Linux per-process RSS helper. A constant shell `exec` redirects logs while preserving the measured model
PID; actual log hashes are kept separately from the launcher's empty output stream. These elapsed times
are quality diagnostics, not performance comparisons.

```sh
python3 scripts/qwen_quality.py capture out/build/qwen-qual/run-manifest.json
python3 scripts/qwen_quality.py judge out/build/qwen-qual/run-manifest.json \
  --out out/quality/qwen27b-v1-result.json
```

Capture resumes only verified, complete, artifact-matched windows. Raw reference scores and nominated
full logits are cached by exact inputs and scorer/library/model identities, independently of candidate
identity, so future kernels can reuse the oracle. `--max-jobs N` is for bounded capture/debugging only;
judgement always requires every mandatory window and original-baseline check. Progress and logs live in
`out/quality/qwen27b-v1-cache/`. On failure, partial files remain untrusted until a successful re-capture.

## Remaining before promotion

Complete all corpus and full-logit gates, original scalar checks at all 128 positions in four domains,
independent captured-score spot checks and the 8 GiB process RSS gate. Validate actual chat reset,
overflow, replay and EOS/length behavior. Only then resolve AUTO for the qualified build/ISA class,
verify final artifact identities and controlled throughput, preserve rollback, and promote the live file.
This does not finish batched prefill, reference performance parity, engine integration or hardware-limit work.
