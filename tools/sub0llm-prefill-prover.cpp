// Preliminary prover for batched prefill (#74, docs/design/PREFILL_BATCHING.md).
//
// Question answered: if prompt tokens are projected in a batch (each weight block decoded once and reused for every token in a
// tile) instead of one token at a time, how much faster is the projection work on the REAL model weights, and is every output
// bit-identical to the serial path?
//
// For the largest weight tensor of each quant type in a GGUF it runs, with the same row-split OpenMP workers the proving-ground
// tool uses: (a) serial: one denseq::project per token, (b) batched: denseq::project_batch at several token tiles. It checks bitwise
// equality of every output, reports medians, and projects the whole-model projection speedup by weighting each format's measured
// time per MiB with that format's share of the model's weight bytes. It measures projections only (not GDN recurrence, attention
// scores or norms) and uses synthetic activations, so it proves the mechanism and sizes the opportunity; it is not a model result.
#include "sub0/dense_quant.hpp"
#include "sub0/file_map.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using sub0::denseq::Iq1Kernel;

struct Case {
    const sub0::gguf::TensorInfo* tensor = nullptr;
    std::span<const std::uint8_t> raw;
    std::size_t in = 0, out = 0;
    double mib = 0;
};

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// One row range per worker, exactly as Layer::project_impl splits an output.
template <class Body>
bool for_workers(int workers, std::size_t rows, Body body) {
    std::vector<char> ok(static_cast<std::size_t>(workers), 1);
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers) schedule(static)
#endif
    for (int w = 0; w < workers; ++w) {
        const std::size_t begin = rows * static_cast<std::size_t>(w) / static_cast<std::size_t>(workers);
        const std::size_t end = rows * (static_cast<std::size_t>(w) + 1) / static_cast<std::size_t>(workers);
        ok[static_cast<std::size_t>(w)] = body(w, begin, end) ? 1 : 0;
    }
    return std::all_of(ok.begin(), ok.end(), [](char c) { return c != 0; });
}

bool run_serial(const Case& c, const std::vector<float>& x, std::vector<float>& y, std::size_t tokens, int workers,
                std::vector<std::vector<float>>& scratch, Iq1Kernel kernel) {
    const auto spec = sub0::gguf::block_spec(c.tensor->type_raw);
    const std::size_t row_bytes = c.in / spec.elems * spec.bytes;
    for (std::size_t t = 0; t < tokens; ++t) {
        const bool ok = for_workers(workers, c.out, [&](int w, std::size_t begin, std::size_t end) {
            return sub0::denseq::project(c.tensor->type_raw, c.raw.subspan(begin * row_bytes, (end - begin) * row_bytes),
                                         {x.data() + t * c.in, c.in}, {y.data() + t * c.out + begin, end - begin},
                                         scratch[static_cast<std::size_t>(w)], kernel);
        });
        if (!ok) return false;
    }
    return true;
}

template <std::size_t Tile>
bool run_batch(const Case& c, const std::vector<float>& x, std::vector<float>& y, std::size_t tokens, int workers,
               std::vector<std::vector<float>>& scratch, Iq1Kernel kernel) {
    const auto spec = sub0::gguf::block_spec(c.tensor->type_raw);
    const std::size_t row_bytes = c.in / spec.elems * spec.bytes;
    return for_workers(workers, c.out, [&](int w, std::size_t begin, std::size_t end) {
        return sub0::denseq::project_batch<Tile>(c.tensor->type_raw, c.raw.subspan(begin * row_bytes, (end - begin) * row_bytes),
                                                 x.data(), c.in, tokens, y.data() + begin, end - begin, c.out,
                                                 scratch[static_cast<std::size_t>(w)], kernel);
    });
}

struct Variant {
    std::string name;
    std::vector<double> times;
    std::size_t mismatches = 0;
};

}  // namespace

int main(int argc, char** argv) try {
    CLI::App app{"Preliminary prover for batched prefill: serial versus batched projection on real model tensors"};
    std::string model, kernel_name = "native-all", filter, skip = "output.weight";
    std::size_t tokens = 13;
    int threads = 8, rounds = 5;
    app.add_option("model", model, "GGUF model")->required();
    app.add_option("--tokens", tokens, "Prompt tokens per batch")->capture_default_str();
    app.add_option("--threads", threads, "Row-split workers (OpenMP)")->capture_default_str()->check(CLI::Range(1, 64));
    app.add_option("--rounds", rounds, "Timed rounds (median reported)")->capture_default_str()->check(CLI::Range(1, 100));
    app.add_option("--kernel", kernel_name, "baseline|scalar|native|native-all")->capture_default_str()
        ->check(CLI::IsMember({"baseline", "scalar", "native", "native-all"}));
    app.add_option("--filter", filter, "Only tensors whose name contains this text");
    app.add_option("--skip", skip, "Tensors whose name contains this text are excluded from the model-level estimate (default: the LM head, which prefill evaluates for the last token only)")->capture_default_str();
    CLI11_PARSE(app, argc, argv);
    const Iq1Kernel kernel = kernel_name == "native-all" ? Iq1Kernel::NativeAll : kernel_name == "native" ? Iq1Kernel::Native
                           : kernel_name == "scalar" ? Iq1Kernel::Scalar : Iq1Kernel::Baseline;

    sub0::FileMap file;
    std::string error;
    if (!file.open(model, error)) { std::fprintf(stderr, "prover: %s\n", error.c_str()); return 2; }
    const sub0::gguf::Reader reader({file.data(), file.size()});
    if (!reader.ok()) { std::fprintf(stderr, "prover: cannot parse the GGUF header\n"); return 2; }

    // Per quant type: the largest 2D weight tensor (the embedding is a row gather, not a projection), and every type's total bytes.
    std::map<std::uint32_t, Case> chosen;
    std::map<std::uint32_t, double> total_mib;
    for (const auto& t : reader.tensors()) {
        const auto spec = sub0::gguf::block_spec(t.type_raw);
        if (t.dims.size() != 2 || spec.elems <= 1 || t.name.starts_with("token_embd") || t.dims[0] % spec.elems != 0) continue;
        const auto raw = reader.tensor_bytes(t);
        const double mib = static_cast<double>(raw.size()) / (1024.0 * 1024.0);
        if (skip.empty() || t.name.find(skip) == std::string::npos) total_mib[t.type_raw] += mib;
        if (!filter.empty() && t.name.find(filter) == std::string::npos) continue;
        auto& slot = chosen[t.type_raw];
        if (!slot.tensor || mib > slot.mib) slot = Case{&t, raw, static_cast<std::size_t>(t.dims[0]), static_cast<std::size_t>(t.dims[1]), mib};
    }

    std::printf("model %s\nbatch of %zu tokens, %d row workers, kernel %s, %d rounds (median), synthetic activations\n",
                model.c_str(), tokens, threads, kernel_name.c_str(), rounds);
    std::printf("%-8s %-30s %9s %10s | %-24s %-24s %-24s | %s\n", "format", "tensor", "in x out", "MiB", "tile 8 ms (speedup)",
                "tile 16 ms (speedup)", "tile 32 ms (speedup)", "serial ms | identical");

    std::vector<std::vector<float>> scratch(static_cast<std::size_t>(threads));
    for (auto& s : scratch) s.reserve(256);
    std::uint32_t seed = 0x1234567u;
    double all_mib = 0, serial_time_per_mib_weighted = 0, best_time_per_mib_weighted = 0;
    bool all_identical = true;
    for (const auto& [type, c] : chosen) {
        std::vector<float> x(tokens * c.in);
        for (auto& v : x) { seed = seed * 1664525u + 1013904223u; v = static_cast<float>(static_cast<std::int32_t>(seed >> 8) % 2001 - 1000) / 1000.f; }
        std::vector<float> y_serial(tokens * c.out), y_test(tokens * c.out);
        const std::string fmt = sub0::gguf::find_format(type)->name;

        if (!run_serial(c, x, y_serial, tokens, threads, scratch, kernel)) { std::printf("%-8s serial path rejected the tensor\n", fmt.c_str()); continue; }
        std::vector<Variant> variants{{"8", {}, 0}, {"16", {}, 0}, {"32", {}, 0}};
        std::vector<double> serial_times;
        auto check = [&](Variant& v) {
            for (std::size_t i = 0; i < y_serial.size(); ++i)
                if (std::memcmp(&y_serial[i], &y_test[i], sizeof(float)) != 0) ++v.mismatches;
        };
        // Warm-up, then interleaved rounds so drift touches all variants alike.
        (void)run_batch<8>(c, x, y_test, tokens, threads, scratch, kernel);
        for (int r = 0; r < rounds; ++r) {
            auto t0 = Clock::now();
            (void)run_serial(c, x, y_serial, tokens, threads, scratch, kernel);
            serial_times.push_back(ms_since(t0));
            t0 = Clock::now(); const bool ok8 = run_batch<8>(c, x, y_test, tokens, threads, scratch, kernel);
            variants[0].times.push_back(ms_since(t0)); if (ok8 && r == 0) check(variants[0]);
            t0 = Clock::now(); const bool ok16 = run_batch<16>(c, x, y_test, tokens, threads, scratch, kernel);
            variants[1].times.push_back(ms_since(t0)); if (ok16 && r == 0) check(variants[1]);
            t0 = Clock::now(); const bool ok32 = run_batch<32>(c, x, y_test, tokens, threads, scratch, kernel);
            variants[2].times.push_back(ms_since(t0)); if (ok32 && r == 0) check(variants[2]);
        }
        const double serial_ms = median(serial_times);
        std::string cols;
        double best_ms = serial_ms;
        bool identical = true;
        for (auto& v : variants) {
            const double ms = median(v.times);
            best_ms = std::min(best_ms, ms);
            identical = identical && v.mismatches == 0;
            char buf[64];
            std::snprintf(buf, sizeof buf, "%8.2f (%5.2fx)%s", ms, serial_ms / ms, v.mismatches ? "!" : " ");
            cols += std::string(buf) + "   ";
        }
        all_identical = all_identical && identical;
        std::printf("%-8s %-30s %4zux%-5zu %10.1f | %s| %8.2f | %s\n", fmt.c_str(), c.tensor->name.c_str(), c.in, c.out, c.mib,
                    cols.c_str(), serial_ms, identical ? "yes" : "NO");
        all_mib += total_mib[type];
        serial_time_per_mib_weighted += total_mib[type] * (serial_ms / c.mib);
        best_time_per_mib_weighted += total_mib[type] * (best_ms / c.mib);
    }

    // Projection-time estimate for the whole model: each format's measured time per MiB times that format's bytes in the model.
    std::printf("\nthe estimate covers %.0f MiB of 2D weight tensors (excluding '%s').\n", all_mib, skip.c_str());
    if (all_mib > 0 && best_time_per_mib_weighted > 0)
        std::printf("projected projection-time speedup for a %zu-token prompt (best tile per format): %.2fx  (serial %.0f ms -> batched %.0f ms)\n"
                    "projection work only: GDN recurrence, attention scores and norms are not included, so the model-level gain is smaller.\n",
                    tokens, serial_time_per_mib_weighted / best_time_per_mib_weighted, serial_time_per_mib_weighted, best_time_per_mib_weighted);
    std::printf("bit-identical to the serial path on every tested tensor: %s\n", all_identical ? "YES" : "NO");
    return all_identical ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "prover: %s\n", e.what());
    return 2;
}
