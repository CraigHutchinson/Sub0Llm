// cuda_selftest.cpp — tiny clang-built host driver for the CUDA backend self-test.
//
// Exists to exercise the multi-compiler boundary: this is compiled by clang and links
// against sub0_backend_cuda (built by nvcc), calling across the extern "C" seam. If it
// builds, links and returns 0 at runtime, the clang <-> nvcc pipeline + CUDA device
// execution are all working on this host.

extern "C" int sub0_cuda_selftest();

int main() { return sub0_cuda_selftest(); }
