# Roofline analysis of the fused MoE kernel — and a correction to this thread's governing assumption

**Measured 2026-09-21**, derived from `profile_post_b35.md`'s phase split plus the per-format scaffold.
Prompted by a direct observation: wall-clock deltas and relative GB/s between formats are not
controlling factors — **achieved bandwidth against the bus ceiling, and achieved IOPS/FLOPS against the
ISA-width ceiling, are.** If bandwidth is not saturated, compute must be the limit; if neither is
saturated, the kernel is bound by something else and no amount of tuning either will help.

## The numbers

MoE is 68.7% of decode (975 ms/token). Per token it performs 1440 plane-GEMVs × 1,638,400 elements =
**2.36 G int8 MACs**, touching **780 MB** of encoded sidecar bytes (arithmetic intensity 3.02 MAC/byte).

| Roof | Achieved | Ceiling | Utilisation |
|---|---:|---:|---:|
| DRAM bandwidth | 0.80 GB/s | ~30 GB/s (measured, `BACKBONE_PRECISION.md` §2c) | **2.7%** |
| AVX2 integer SIMD (`vpmaddwd`, 16 MAC/instr, 2 ports, 3.5 GHz) | 2.42 GMAC/s | ~112 GMAC/s | **2.2%** |
| *Scalar* integer MAC (1/instr, 2 ports) | 2.42 GMAC/s | ~7 GMAC/s | **34.6%** |

**Neither roof is touched. The kernel runs at roughly one third of *scalar* issue rate** — i.e. at the
order of magnitude of scalar execution, with the SIMD roof and the memory roof each ~40x away.

Using the right unit width is the whole point of that third row. Measuring against a scalar FPU/ALU
ceiling would have said "35% utilised, not much left" — which is exactly the wrong conclusion. Against
the ceiling the hardware actually offers for this data type, there is ~45x of headroom.

## What this corrects

**This thread has operated since B27 on the premise that decode is DRAM-bandwidth-bound.** That premise
was correct when measured: pre-B35, the MoE path materialised 18.75 MiB of f32 planes per resolve and
read them back once, ~39 MB of traffic per resolve. It drove real wins (B29, B31, B35 all reduced bytes
moved) and it was used to explain away four negatives (B28's prefetch, B33's FP8, B34's SIMD, B39's
compile-time shaping) as "compute-side changes on a bandwidth-bound workload."

**B35 invalidated that premise and nobody re-checked it.** By eliminating the f32 materialisation it cut
per-resolve traffic 25x. The phase that is now 68.7% of decode sits at **2.7% of memory bandwidth**. It
is not bandwidth-bound. It is not SIMD-compute-bound either. It is **unpack-bound**: per-group scalar
work — dependent table lookups (`IQ2XXS_GRID`, `SIGNS64`), per-group scale computation, `memcpy`
shuffling, and IQ2_XXS's redundant second pass — none of which vectorises, and all of which sits
between the `dot_group` calls that do.

So `OPTIMIZATION_PROCESS.md` §5a's table is true as history but **must not be read as a standing rule**.
"O3/O4 levers lose" was a property of the pre-B35 constraint, not a law. The constraint has moved down
the stack, and ISA-level work is now aimed at the right layer for the first time in this thread.

## What it says to do

1. **The fused MoE unpack path is the target, and the goal is SIMD throughput, not byte reduction.**
   At 2.2% of the integer-SIMD roof inside a phase that is 68.7% of decode, this is not a
   single-digit-percent opportunity. Byte-reduction levers (the family that produced every win so far)
   now have almost nothing left to take: 2.7% of the memory roof.
2. **O1's framing needs widening.** Removing IQ2_XXS's redundant sign pass is correct and still worth
   doing, but it is one instance of the general problem. The brief should be read as "get the unpack
   path to SIMD throughput," with the sign pass as attempt 1.
3. **Re-check the roofline after any change over ~20%**, exactly as §5 requires for the phase profile.
   A kernel that moves from unpack-bound toward the SIMD roof will change which lever is next, and a
   kernel that reaches the memory roof is genuinely finished.

## Tooling available on this host

| Tool | Path | Use |
|---|---|---|
| **VTune 2026.4** | `C:\Program Files (x86)\Intel\oneAPI\vtune\2026.4\bin64\vtune.exe` | Real core utilisation, port pressure, and the memory-bound vs core-bound classification that turns the table above from arithmetic into measurement. Project precedent: run unelevated, use the standalone-benchmark + `-target-pid` pattern (`[[cpu-profiling-tooling-backlog]]`), which previously found a real MoE defect that code reading had missed. |
| **llvm-mca** | `C:\Program Files\LLVM\bin\llvm-mca.exe` | Static throughput/port analysis of a specific instruction sequence. Ideal here: feed it the unpacker's inner loop and get uops, port pressure and an IPC estimate **without running anything**, to compare unpacker variants before committing to one. |

Both were previously unused in this thread; every conclusion to date came from wall-clock plus hand
arithmetic. The arithmetic above is what motivates using them — it says where to look, but only a real
core-utilisation measurement will say *why* the unpack path stalls (dependent-load latency vs port
contention vs issue-width).
