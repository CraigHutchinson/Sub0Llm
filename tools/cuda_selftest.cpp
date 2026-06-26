// cuda_selftest.cpp — tiny clang-built host driver for the CUDA backend self-test.
//
// Exists to exercise the multi-compiler boundary: this is compiled by clang and links
// against sub0_backend_cuda (built by nvcc), calling across the extern "C" seam. If it
// builds, links and returns 0 at runtime, the clang <-> nvcc pipeline + CUDA device
// execution are all working on this host.

#include <initializer_list>

extern "C" int sub0_cuda_selftest();
extern "C" int sub0_cuda_benchmark(int batch, int T, int iters);
extern "C" int sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms);

int main() {
    const int rc = sub0_cuda_selftest();
    if (rc != 0) return rc;
    sub0_cuda_benchmark(64, 64, 100);    // training-scale (large M): expect compute-bound
    sub0_cuda_benchmark(1, 64, 2000);    // generation-scale (small M): expect launch-bound -> graph wins
    // Training-step throughput vs batch (the dimension knob): tok/s should climb with batch until
    // the GEMMs saturate the device -- the curve the GPU autotuner will use to pick the batch.
    for (int b : {32, 64, 128, 256, 384})
        sub0_cuda_train_benchmark(b, 64, 50, nullptr);
    return 0;
}
