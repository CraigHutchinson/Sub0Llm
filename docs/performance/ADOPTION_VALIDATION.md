# Adoption validation — 2026-11-07

Issue #48; baseline source commit `c5a638f6012733ce81356d3cdd2a73ecea911884`.

- Host: `CrogBuntu1`; Linux-7.0.0-42-generic-x86_64-with-glibc2.43.
- Compiler: Ubuntu clang version 21.1.8 (6ubuntu1).
- Build: `cmake --preset perf`, Release, generic ISA (`SUB0_NATIVE=OFF`), no model/configurator.
- Binary SHA256: `9de3a5eeed2930122b221fbd5ff8bd7511b2d5cfb232aa9fe99d69f2abecb190`.
- Shared upstream tooling: 287 unit tests passed; CTest fixture/tooling/provenance: 3/3 passed.
- Local C++ authored with the cpp skill and reviewed with cpp-review; no open findings.
- Exact upstream snapshot preserved, including existing final blank lines in five upstream tests.
- Full gate: all four cases, three measured samples each, CPU 31, seed 1, power {"state": "ac", "saver": false}.
- Initial baseline: `perf/baselines/crogbuntu1.json`, created from this full clean-tree record without overrides or regression acceptance.
- Calibration start/end ns by round: 4500.1926605504605/4501.5503035559395, 4499.9748154581/4500.92217898833, 4500.63486238532/4501.80398832685.
- Busy rounds: 0. Counters unavailable; no counter-based claim.

Raw unscaled samples (ns per projection; calibration-scaled medians/spread in baseline):

| Case | Samples |
|---|---|
| Q8_0 32x256, hand-written fixed-scale | 4501.242, 4497.885, 4505.996 |
| Q8_0 32x256, denseq | 4240.258, 4249.679, 4240.524 |
| Q8_0 32x5120, hand-written fixed-scale | 93302.083, 93179.364, 93193.000 |
| Q8_0 32x5120, denseq | 90505.000, 90502.000, 90493.333 |

These are warm-cache synthetic Q8_0 row slices. They establish an initial host reference, not an
optimization win, 27B speed/quality, DRAM bandwidth or RSS result. The local gate retains the clean
record and executable under the git common directory; CI uploads its own independent diagnostic samples.

The actual pre-push hook chains the pre-existing Git LFS hook. Project-board linkage is unavailable:
the active GitHub token lacks `read:project`; issue/PR tracking remains available.
