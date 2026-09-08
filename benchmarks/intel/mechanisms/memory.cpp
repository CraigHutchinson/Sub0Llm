// I19: bounded allocation/access comparison. No system-USM or model-residency claim.
#include "../../../tools/intel_probe/runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Allocation = std::unique_ptr<float, sub0::intel_probe::UsmDeleter>;
constexpr std::size_t count = 1024 * 1024;
constexpr std::size_t bytes = count * sizeof(float);
constexpr int trials = 5;
/// Allocation mode affects legal host access and first-touch behavior, not the kernel.
enum class InputMode { host, shared, device };
void measure(InputMode mode, sycl::queue& queue) {
    const char* name = mode == InputMode::host ? "host" : mode == InputMode::shared ? "shared" : "device";
    float* allocation = mode == InputMode::host ? sycl::malloc_host<float>(count, queue)
        : mode == InputMode::shared ? sycl::malloc_shared<float>(count, queue)
        : sycl::malloc_device<float>(count, queue);
    Allocation input(allocation, {&queue});
    Allocation output(sycl::malloc_device<float>(count, queue), {&queue});
    if (!input || !output) throw std::runtime_error("bounded USM allocation failed");
    std::vector<float> seed(count), check(count);
    for (std::size_t i = 0; i < count; ++i) seed[i] = float(i % 257);
    const auto* source = input.get(); auto* destination = output.get();
    const auto start = Clock::now();
    if (mode == InputMode::device) queue.memcpy(input.get(), seed.data(), bytes).wait_and_throw();
    else std::copy(seed.begin(), seed.end(), input.get());
    const double preparation = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    auto execute = [&] {
        queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            destination[i] = source[i] * 2 + 1;
        }).wait_and_throw();
    };
    const auto first = Clock::now(); execute();
    const double first_ms = std::chrono::duration<double, std::milli>(Clock::now() - first).count();
    queue.memcpy(check.data(), destination, bytes).wait_and_throw();
    for (std::size_t i = 0; i < count; ++i)
        if (check[i] != seed[i] * 2 + 1) throw std::runtime_error("USM access mismatch");
    std::array<double, trials> samples{};
    for (auto& sample : samples) {
        const auto begin = Clock::now(); execute();
        sample = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    }
    // Explicit completed CPU handoff, including upload for device-only storage.
    for (auto& value : seed) value += 3;
    const auto handoff = Clock::now();
    if (mode == InputMode::device) queue.memcpy(input.get(), seed.data(), bytes).wait_and_throw();
    else std::copy(seed.begin(), seed.end(), input.get());
    execute();
    const double handoff_ms = std::chrono::duration<double, std::milli>(Clock::now() - handoff).count();
    queue.memcpy(check.data(), destination, bytes).wait_and_throw();
    for (std::size_t i = 0; i < count; ++i)
        if (check[i] != seed[i] * 2 + 1) throw std::runtime_error("CPU handoff mismatch");
    std::cout << "allocation," << name << ',' << bytes << ',' << preparation << ',' << first_ms << ',' << handoff_ms << '\n';
    for (int trial = 0; trial < trials; ++trial) std::cout << "sample," << name << ',' << trial << ',' << samples[trial] << '\n';
}
}
int main() {
    try {
        const auto device = sub0::intel_probe::select_device();
        if (!device.has(sycl::aspect::usm_host_allocations) || !device.has(sycl::aspect::usm_shared_allocations)
            || !device.has(sycl::aspect::usm_device_allocations)) throw std::runtime_error("Required USM allocation mode unavailable");
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (auto error : errors) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});
        std::cout << "backend=level_zero\ndevice_id=0x7d67\n"
                  << "allocation_columns=mode,bytes,prepare_ms,first_ms,cpu_handoff_ms\n"
                  << "sample_columns=mode,trial,kernel_completion_ms\n";
        // Reversed second round exposes order sensitivity; no confidence/p95 claim from five samples.
        for (auto mode : {InputMode::host, InputMode::shared, InputMode::device,
                          InputMode::device, InputMode::shared, InputMode::host}) measure(mode, queue);
        std::cout << "status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "status=fail\nerror=" << error.what() << '\n'; return 1;
    }
}
