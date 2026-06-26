// cuda_selftest.cpp — tiny clang-built host driver for the CUDA backend self-test.
//
// Exists to exercise the multi-compiler boundary: this is compiled by clang and links
// against sub0_backend_cuda (built by nvcc), calling across the extern "C" seam. If it
// builds, links and returns 0 at runtime, the clang <-> nvcc pipeline + CUDA device
// execution are all working on this host.

extern "C" int sub0_cuda_selftest();
extern "C" int sub0_cuda_benchmark(int batch, int T, int iters);

int main() {
    const int rc = sub0_cuda_selftest();
    if (rc != 0) return rc;
    sub0_cuda_benchmark(64, 64, 100);   // FP32 vs TF32 forward timing
    return 0;
}
