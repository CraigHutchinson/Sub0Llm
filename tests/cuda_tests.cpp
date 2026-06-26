// cuda_tests.cpp — GPU backend smoke test. Built and linked only when SUB0_BUILD_CUDA is
// ON (see tests/CMakeLists.txt), since it requires nvcc-built code and a CUDA device. It
// drives the backend's self-test across the extern "C" seam: H2D -> kernel -> D2H over a
// parameter-blob-sized buffer must round-trip exactly on the device.

#include <catch2/catch_test_macros.hpp>

extern "C" int sub0_cuda_selftest();

TEST_CASE("CUDA backend self-test runs on the device", "[cuda]") {
    REQUIRE(sub0_cuda_selftest() == 0);
}
