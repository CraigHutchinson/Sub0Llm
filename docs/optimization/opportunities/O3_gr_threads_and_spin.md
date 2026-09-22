# O3 — Gated Residual through the GEMV primitive, and spinning OpenMP workers in decode

**Status:** merged 2026-09-22. Decode **0.313 → 0.213 s/token** at the recommended flags. Both changes are
scheduling or bit-exact only: L2 0.23252 and every neutral fingerprint unchanged.

## 1. Gated Residual's projections (bit-exact)

After O2, Gated Residual was 25% of decode (79 ms), still single-threaded. `gr::mix`'s down (10240 → 320)
and up (320 → 10240) projections are ~13 MB of bf16 per call at 96 calls per token (~1.26 GB/token),
bytes the post-O1 estimate had left out. Routed through `gemv::axpy<Threads>` (neither loop skipped
zeros, so rounding is unchanged): **GR 79 → 44 ms, decode 0.314 → ~0.277 s/token** (profiled, 2 runs,
background 12.8%, so indicative). `gate` (10240 → 4) is too narrow to be worth threading.

## 2. Spinning workers: `--decode-omp-spin 1`

Decode forks about 600 short OpenMP regions per token, one per threaded GEMV. With libomp's default
blocktime a worker can go to sleep between regions and then pays an OS wake-up. The default runs were
also erratic: GR went 43 → 94 ms between runs of the same binary. `kmp_set_blocktime(INT_MAX)` is set in
`kv_reset`, on the thread that masters every decode team (teams inherit their master's blocktime), and is
guarded to libomp builds (`KMP_VERSION_MAJOR`).

Interleaved sandboxed A/B through `run_perf_suite.py`, 3 runs each, 3 tokens, recommended flags,
background 3.4%:

| arm | median s/token | spread | runs |
|---|---:|---:|---|
| no spin | 0.267 | 28.5% | 0.267, 0.340, 0.264 |
| **spin** | **0.213** | **14.1%** | 0.205, 0.213, 0.235 |

**−20% decode, and half the run-to-run spread.** The cost is idle workers occupying their P-cores for the
duration of a generation, which is why it is an opt-in axis (default off) rather than always on.

## 3. Harness fix found on the way

The first A/B attempt failed to compile the spin arm: the real-axes build dir's `sub0llm-configure`
binary predated the new flag. Because `run()` now fails loudly, this surfaced as an error instead of a
silent measurement of the old binary. `configure()` now rebuilds the configurator before every configure.

## 4. Recommended real-axes decode flags

`--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 10 --decode-omp-spin 1`

## 5. Next

~0.21 s/token against a ~117 ms bandwidth floor (~9.3 GB/token at 79 GB/s, now counting GR): under 2x
left. The mixer is the largest remaining gap. Split it into GDN and QSA first, then GEMV versus the
serial recurrence and attention.
