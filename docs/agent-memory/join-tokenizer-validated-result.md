---
name: join-tokenizer-validated-result
description: "JOIN/implicit-space tokenizer VALIDATED on tinystories: -27.6% tokens, -8.7% bits/byte vs legacy (same d160 model); wired end-to-end behind SUB0_JOIN_TOKENIZER"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

The JOIN tokenizer is now wired end-to-end (build flag `SUB0_JOIN_TOKENIZER` → `sub0-configure --join` → `corpus.tok` via `sub0::tok::encode`; engine deserializes + uses the FSM) and **validated as a real improvement**, committed `08346f2` (wiring) on top of `c282908` (library). See [[join-tokenizer-strategy]].

**tinystories A/B (same model d160 L5 H4 seq256 v2048 RoPE, ~8 epochs to plateau both):**
- corpus.tok tokens: 10,562,743 → **7,642,079 (−27.6%)**; bytes/token 2.095 → 2.895 (+38%); vocab unchanged 2048 (261 base + 1787 merges); lossless round-trip both ways.
- val_nelbo: legacy **1.4607** (per-token) vs JOIN **1.8426** — JOIN higher per-TOKEN (denser tokens) but that's the wrong metric.
- **bits/byte = NELBO / bytes_per_token / ln2** (scheme-independent), legacy **1.006**:
  - **GPU batch-256 (MATCHED to baseline — trust this one): JOIN val 1.9336 → 0.964 bits/byte = −4.2%.** Plateaued step 1342 (~12 ep); both legacy+JOIN plateau ~1320 optimizer steps.
  - CPU batch-96 (rougher, more updates): JOIN val 1.8426 → 0.918 bits/byte = −8.7%. Lower because batch-96 did 2430 steps; tells us JOIN keeps improving with more updates/smaller batch, so −4.2% is a conservative floor.
- Mechanism: not spending ~40% of tokens on trivially-predictable single spaces frees model capacity for content; also +38% effective context at fixed seq_len.
- JOIN generation sample (both CPU + GPU) = coherent TinyStories prose with correct spacing/caps/quotes/paragraph breaks (PARA tokens reconstruct blank lines) → the decode FSM works in real generation, not just unit tests.

**CUDA build CONFIRMED working** from a vcvars64 shell (VS 18 Community: `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`). Drive non-interactively: `cmd /c '"<vcvars64.bat>" && cmake -S . -B out/build/joingpu ... -DSUB0_COMPUTE=GPU -DSUB0_JOIN_TOKENIZER=ON ... && cmake --build out/build/joingpu --target sub0llm'`. nvcc needs cl.exe on PATH (clang stays the C++ compiler via -DCMAKE_CXX_COMPILER=clang++). Training/running the exe needs no vcvars (CUDA runtime DLLs already on PATH: CUDA v13.2\bin).

**NOW DEFAULT ON + registry resolved (committed 9a4730f, 66fb514).** `SUB0_JOIN_TOKENIZER=ON` is the default build. Registry: configurator bakes `constexpr bool JOIN_TOKENIZER`; model_dir tags `j` (like RoPE `r`) → `..._v2048rj_<sha>`; `compatible()` + ModelMeta + meta.txt include it, so JOIN vs legacy models no longer collide and cross-scheme loads flag incompatible. Renamed `SUB0_CORPUS_TOK`→`SUB0_CORPUS_PRETOK` / `--corpus-tok`→`--corpus-pretok`. **Round-trip bug fixed (66fb514):** UP bled past an interior apostrophe ("NASA's"→"NASA'S") because word_unit_end keeps "nasa's" one unit; UP now ends at the first non-alpha (mirrors casing::detokenize). Full 1GB FineWeb round-trips OK (was the FAIL that blocked default-on). Configurator round-trip now prints first divergence (expected-vs-actual). `case_mode` magic ints → `enum class Recase`.

**No backwards compatibility needed** (user: discard old models — all 11 legacy models deleted; models dir clean). So the legacy space-as-token scheme code path (the `join_scheme` else-branches in learn/encode/detokenize) can be REMOVED in the planned code-quality review; no tokenizer.bin/model.bin format versioning required.

**Next JOIN work:** directional quotes (§3) + SPELL N≥3 (§4) encapsulation (today N−1 JOINs per multi-token word); word-N histogram to size the SPELL threshold; consider a JOIN run on FineWeb (the measured ws/punct stats were FineWeb).
