// cuda_selftest.cpp — tiny clang-built host driver for the CUDA backend self-test.
//
// Exists to exercise the multi-compiler boundary: this is compiled by clang and links
// against sub0_backend_cuda (built by nvcc), calling across the extern "C" seam. If it
// builds, links and returns 0 at runtime, the clang <-> nvcc pipeline + CUDA device
// execution are all working on this host.
//
// Two rules keep that claim honest, both learned from this tool silently lying:
//
//   1. EVERY call's return code is checked. This driver used to discard all of them and
//      `return 0` unconditionally, so a failing check was indistinguishable from a passing
//      one. `track()` folds each result into the exit code.
//   2. EVERY sequence length is clamped to what the backend was BUILT for, via
//      sub0_cuda_seq_len(). The benchmark shapes below are the fineweb training config
//      (batch 184, T 256); on a build configured with a shorter SEQ_LEN the backend's
//      `T > SEQ_LEN` guards rejected them, so on e.g. a SEQ_LEN=64 build SIX of the eight
//      checks did nothing at all and the tool still exited 0.
//
// The backend deliberately REJECTS rather than clamps an over-long T: these are measurement
// entry points, and silently profiling a different shape than the caller asked for would make
// every number they report untrustworthy. Adapting is therefore the caller's job -- this file's.

#include <cstdio>
#include <cstring>
#include <initializer_list>

extern "C" int sub0_cuda_selftest();
extern "C" int sub0_cuda_seq_len();
extern "C" int sub0_cuda_benchmark(int batch, int T, int iters);
extern "C" int sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms);
extern "C" int sub0_cuda_attn_check(int batch, int T, int iters, double* out_maxreldiff, double* out_speedup);
extern "C" int sub0_cuda_attn_bwd_check(int batch, int T, int iters, double* out_maxreldiff, double* out_speedup);
extern "C" int sub0_cuda_forward_one_check(int T, int iters, double* out_maxreldiff, double* out_toks);

namespace {

// `want`, clamped to the context length this build was compiled for (see the header comment).
int clamp_T(int want) {
    const int built = sub0_cuda_seq_len();
    return built < want ? built : want;
}

// Fold one check's result into the process exit code, naming the failure rather than swallowing it.
// Returns rc so callers can chain. Keeps going after a failure: a full run's worth of evidence is
// more useful than the first error, and every check here is independent.
int track(const char* what, int rc, int& failures) {
    if (rc != 0) {
        std::fprintf(stderr, "cuda selftest: FAILED %s (rc=%d)\n", what, rc);
        ++failures;
    }
    return rc;
}

}  // namespace

// Any argument -> "bench only": skip the host-reference grad/AdamW parity check (a slow plain-C++
// reference pass, impractical at large model dims) so this is a fast kernel driver at the training
// config -- suitable for running under Nsight Compute (ncu) for an occupancy profile.
//   sub0-cuda-selftest        = correctness gate (parity + benchmarks)
//   sub0-cuda-selftest bench  = benchmarks only (FP32 vs TF32 vs graph, + the train-step tok/s curve)
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: stream each benchmark line as it finishes
    int failures = 0;
    const int T = clamp_T(256);                 // the fineweb training window
    if (T < 256)
        std::printf("cuda selftest: build SEQ_LEN=%d -- benchmark shapes clamped from the fineweb T=256\n", T);

    // `ncu` mode: run ONLY one d448 training-config forward, so an ncu capture profiles exactly the
    // training kernels (no gen-scale / parity noise) -- the first ~130 kernels are one forward.
    if (argc > 1 && std::strcmp(argv[1], "ncu") == 0) {
        track("benchmark(ncu)", sub0_cuda_benchmark(184, T, 1), failures);
        return failures;
    }
    // `attn` mode: naive-vs-tiled flash-attention forward parity + speedup at the training config
    // (fast: attention kernels only, no full step). The device correctness/regression guard.
    if (argc > 1 && std::strcmp(argv[1], "attn") == 0) {
        track("attn_check", sub0_cuda_attn_check(184, T, 50, nullptr, nullptr), failures);
        track("attn_bwd_check", sub0_cuda_attn_bwd_check(184, T, 30, nullptr, nullptr), failures);
        return failures;
    }
    // `decode` mode: GPU forward_one (KV-cache) parity vs the full forward + decode tok/s.
    if (argc > 1 && std::strcmp(argv[1], "decode") == 0)
        return track("forward_one_check", sub0_cuda_forward_one_check(T, 300, nullptr, nullptr), failures);

    if (argc <= 1) {
        const int rc = sub0_cuda_selftest();
        if (rc != 0) return rc;                 // the seam itself is broken: nothing below is meaningful
    }
    track("attn_check", sub0_cuda_attn_check(184, T, 50, nullptr, nullptr), failures);        // flash-attn forward parity + speedup
    track("attn_bwd_check", sub0_cuda_attn_bwd_check(184, T, 30, nullptr, nullptr), failures);// flash-attn backward parity + speedup
    // Generation-scale (small M): launch-bound -> graph wins. A short context by intent, not by clamp,
    // so it keeps its own T rather than following the training window above.
    const int short_T = clamp_T(64);
    track("benchmark(gen-scale)", sub0_cuda_benchmark(1, short_T, 2000), failures);
    track("benchmark(train-scale)", sub0_cuda_benchmark(64, short_T, 100), failures);   // large M: compute-bound
    track("benchmark(fineweb config)", sub0_cuda_benchmark(184, T, 40), failures);    // batch 184, T 256: FP32 vs TF32
    // Training-step throughput vs batch at the training T: tok/s should climb with batch until the
    // GEMMs saturate the device -- the curve the GPU autotuner uses to pick the batch.
    for (int b : {64, 128, 184})
        track("train_benchmark", sub0_cuda_train_benchmark(b, T, 20, nullptr), failures);
    if (failures != 0)
        std::fprintf(stderr, "cuda selftest: %d check(s) FAILED\n", failures);
    return failures;
}
