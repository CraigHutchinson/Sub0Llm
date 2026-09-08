// I05/I21: real projection shapes, synthetic f32 values; no encoded-expert or TTFT claim.
#include "../../../tools/intel_probe/runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

namespace {
using Clock = std::chrono::steady_clock;
using Allocation = std::unique_ptr<float, sub0::intel_probe::UsmDeleter>;
constexpr int trials = 5;
/// Row-major activation [m,k] times engine-oriented weights [k,n].
struct Shape { int m, k, n; };
float activation(int m, int k) { return float((m * 17 + k * 7) % 16 - 8) / 16; }
float weight(int k, int n) { return float((k * 13 + n * 3) % 17 - 8) / 16; }

template<class Operation>
double elapsed_ms(Operation&& operation) {
    const auto start = Clock::now();
    operation();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void measure(Shape shape, sycl::queue& queue, const dnnl::engine& engine, dnnl::stream& stream) {
    const auto [m, k, n] = shape;
    const std::size_t ak = std::size_t(m) * k, kn = std::size_t(k) * n, mn = std::size_t(m) * n;
    std::vector<float> a(ak), b(kn), custom_result(mn), library_result(mn);
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col) a[std::size_t(row) * k + col] = activation(row, col);
    for (int row = 0; row < k; ++row)
        for (int col = 0; col < n; ++col) b[std::size_t(row) * n + col] = weight(row, col);
    Allocation da(sycl::malloc_device<float>(ak, queue), {&queue});
    Allocation db(sycl::malloc_device<float>(kn, queue), {&queue});
    Allocation dc(sycl::malloc_device<float>(mn, queue), {&queue});
    if (!da || !db || !dc) throw std::runtime_error("bounded device allocation failed");
    auto* ap = da.get(); auto* bp = db.get(); auto* cp = dc.get();
    queue.wait_and_throw();
    const double upload = elapsed_ms([&] {
        queue.memcpy(ap, a.data(), ak * sizeof(float));
        queue.memcpy(bp, b.data(), kn * sizeof(float)).wait_and_throw();
    });
    auto source = dnnl::sycl_interop::make_memory(
        {{m, k}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, engine,
        dnnl::sycl_interop::memory_kind::usm, ap);
    auto weights = dnnl::sycl_interop::make_memory(
        {{k, n}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, engine,
        dnnl::sycl_interop::memory_kind::usm, bp);
    auto destination = dnnl::sycl_interop::make_memory(
        {{m, n}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab}, engine,
        dnnl::sycl_interop::memory_kind::usm, cp);
    dnnl::primitive_attr attributes;
    attributes.set_fpmath_mode(dnnl::fpmath_mode::strict);
    const auto setup_start = Clock::now();
    auto pd = dnnl::matmul::primitive_desc(engine, source.get_desc(), weights.get_desc(), destination.get_desc(), attributes);
    auto primitive = dnnl::matmul(pd);
    const double primitive_setup = std::chrono::duration<double, std::milli>(Clock::now() - setup_start).count();
    const std::unordered_map<int, dnnl::memory> arguments{
        {DNNL_ARG_SRC, source}, {DNNL_ARG_WEIGHTS, weights}, {DNNL_ARG_DST, destination}};
    auto custom = [&] {
        queue.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> at) {
            float sum = 0;
            for (int inner = 0; inner < k; ++inner)
                sum += ap[at[0] * k + inner] * bp[std::size_t(inner) * n + at[1]];
            cp[at[0] * n + at[1]] = sum;
        }).wait_and_throw();
    };
    auto library = [&] { primitive.execute(stream, arguments); stream.wait(); queue.wait_and_throw(); };
    const double custom_first = elapsed_ms(custom);
    queue.memcpy(custom_result.data(), cp, mn * sizeof(float)).wait_and_throw();
    const double library_first = elapsed_ms(library);
    queue.memcpy(library_result.data(), cp, mn * sizeof(float)).wait_and_throw();
    // All output elements cross-check; 64 independent double dots catch shared layout mistakes.
    for (std::size_t i = 0; i < mn; ++i)
        if (!std::isfinite(custom_result[i]) || !std::isfinite(library_result[i])
            || std::abs(custom_result[i] - library_result[i]) > 1e-4f)
            throw std::runtime_error("dense implementation cross-check failed");
    for (int sample = 0; sample < 64; ++sample) {
        const int row = (sample * 19) % m, col = (sample * 97) % n;
        double reference = 0;
        for (int inner = 0; inner < k; ++inner)
            reference += double(activation(row, inner)) * weight(inner, col);
        if (std::abs(custom_result[std::size_t(row) * n + col] - reference) > 1e-4)
            throw std::runtime_error("independent double dot failed");
    }
    std::array<double, trials> custom_ms{}, library_ms{};
    for (int trial = 0; trial < trials; ++trial) {
        if (trial % 2 == 0) { custom_ms[trial] = elapsed_ms(custom); library_ms[trial] = elapsed_ms(library); }
        else { library_ms[trial] = elapsed_ms(library); custom_ms[trial] = elapsed_ms(custom); }
    }
    const double download = elapsed_ms([&] { queue.memcpy(library_result.data(), cp, mn * sizeof(float)).wait_and_throw(); });
    std::cout << "setup," << m << ',' << k << ',' << n << ',' << upload << ',' << download << ','
              << primitive_setup << ',' << custom_first << ',' << library_first << ','
              << (ak + kn + mn) * sizeof(float) << '\n';
    for (int trial = 0; trial < trials; ++trial)
        std::cout << "sample," << m << ',' << k << ',' << n << ',' << trial << ','
                  << custom_ms[trial] << ',' << library_ms[trial] << '\n';
}
}
int main() {
    try {
        const auto device = sub0::intel_probe::select_device();
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (auto error : errors) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});
        auto engine = dnnl::sycl_interop::make_engine(device, queue.get_context());
        auto stream = dnnl::sycl_interop::make_stream(engine, queue);
        std::cout << "device=" << device.get_info<sycl::info::device::name>()
                  << "\nbackend=level_zero\nprecision=f32_strict\n"
                  << "setup_columns=m,k,n,upload_ms,download_ms,primitive_setup_ms,custom_first_ms,onednn_first_ms,device_bytes\n"
                  << "sample_columns=m,k,n,trial,custom_ms,onednn_ms\n";
        for (auto shape : {Shape{1,96,384}, {1,2560,640}, {32,2560,640}, {128,2560,640},
                           {1,640,2560}, {32,640,2560}, {128,640,2560},
                           {1,2560,6144}, {32,2560,6144}, {128,2560,6144}})
            measure(shape, queue, engine, stream);
        std::cout << "status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "status=fail\nerror=" << error.what() << '\n';
        return 1;
    }
}
