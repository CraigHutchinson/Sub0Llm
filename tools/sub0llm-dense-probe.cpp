// Exercise a mapped real GGUF projection without expanding its weight matrix.
#include "sub0/dense_quant.hpp"
#include "sub0/file_map.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s MODEL.gguf TENSOR\n", argv[0]); return 2; }
    sub0::FileMap map;
    std::string error;
    if (!map.open(argv[1], error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    const sub0::gguf::Reader reader({map.data(), map.size()});
    if (!reader.ok()) { std::fprintf(stderr, "invalid GGUF\n"); return 1; }
    const auto* tensor = reader.find_tensor(argv[2]);
    if (!tensor || tensor->dims.size() != 2 || tensor->dims[0] == 0 || tensor->dims[1] == 0) {
        std::fprintf(stderr, "expected a nonempty matrix tensor\n"); return 1;
    }
    std::vector<float> x(tensor->dims[0]), y(tensor->dims[1]), scratch;
    scratch.reserve(256);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.f;
    const auto raw = reader.tensor_bytes(*tensor);
    if (!sub0::denseq::project(tensor->type_raw, raw, x, y, scratch)) {
        std::fprintf(stderr, "unsupported projection geometry or truncated data\n"); return 1;
    }
    // Check output ordering and accumulation against independently dequantized whole rows.
    const auto spec = sub0::gguf::block_spec(tensor->type_raw);
    const auto decode = sub0::gguf::dequant_fn(tensor->type_raw);
    const std::size_t row_bytes = x.size() / spec.elems * spec.bytes;
    std::vector<float> row;
    for (std::size_t r : {std::size_t{0}, y.size() / 2, y.size() - 1}) {
        if (!decode(raw.subspan(r * row_bytes, row_bytes), x.size(), row)) return 1;
        float ref = 0.f;
        for (std::size_t j = 0; j < x.size(); ++j) ref += x[j] * row[j];
        if (ref != y[r]) { std::fprintf(stderr, "row %zu mismatch: %g vs %g\n", r, ref, y[r]); return 1; }
    }
    double checksum = 0;
    for (float value : y) { if (!std::isfinite(value)) return 1; checksum += value; }
    std::printf("%s: type=%u in=%zu out=%zu encoded=%zu scratch_capacity=%zu checksum=%.9g; 3 row checks passed\n",
                argv[2], tensor->type_raw, x.size(), y.size(), raw.size(), scratch.capacity(), checksum);
}
