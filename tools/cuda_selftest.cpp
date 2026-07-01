// cuda_selftest.cpp — tiny clang-built host driver for the CUDA backend self-test.
//
// Exists to exercise the multi-compiler boundary: this is compiled by clang and links
// against sub0_backend_cuda (built by nvcc), calling across the extern "C" seam. If it
// builds, links and returns 0 at runtime, the clang <-> nvcc pipeline + CUDA device
// execution are all working on this host.

#include <cstdio>
#include <cstring>
#include <initializer_list>

extern "C" int sub0_cuda_selftest();
extern "C" int sub0_cuda_benchmark(int batch, int T, int iters);
extern "C" int sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms);

// Any argument -> "bench only": skip the host-reference grad/AdamW parity check (a slow plain-C++
// reference pass, impractical at large model dims) so this is a fast kernel driver at the training
// config -- suitable for running under Nsight Compute (ncu) for an occupancy profile.
//   sub0-cuda-selftest        = correctness gate (parity + benchmarks)
//   sub0-cuda-selftest bench  = benchmarks only (FP32 vs TF32 vs graph, + the train-step tok/s curve)
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: stream each benchmark line as it finishes
    // `ncu` mode: run ONLY one d448 training-config forward, so an ncu capture profiles exactly the
    // training kernels (no gen-scale / parity noise) -- the first ~130 kernels are one forward.
    if (argc > 1 && std::strcmp(argv[1], "ncu") == 0) {
        sub0_cuda_benchmark(184, 256, 1);
        return 0;
    }
    if (argc <= 1) {
        const int rc = sub0_cuda_selftest();
        if (rc != 0) return rc;
    }
    sub0_cuda_benchmark(1, 64, 2000);    // generation-scale (small M): launch-bound -> graph wins
    sub0_cuda_benchmark(64, 64, 100);    // training-scale (large M): compute-bound
    sub0_cuda_benchmark(184, 256, 40);   // the actual fineweb training config (batch 184, T 256): FP32 vs TF32
    // Training-step throughput vs batch at the training T=256: tok/s should climb with batch until the
    // GEMMs saturate the device -- the curve the GPU autotuner uses to pick the batch.
    for (int b : {64, 128, 184})
        sub0_cuda_train_benchmark(b, 256, 20, nullptr);
    return 0;
}
