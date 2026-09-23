// Bounded I19 write-only scratch probe. Each process executes one allocation/hint arm.
#include "../../../tools/intel_probe/runtime.hpp"

#include <chrono>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {
using Clock = std::chrono::steady_clock;
/// Releases one probe allocation after outstanding queue work has completed.
struct ScratchDeleter {
    sycl::queue* queue; // non-owning; main's queue outlives both allocations
    void operator()(std::uint32_t* pointer) const noexcept {
        try { queue->wait(); sycl::free(pointer, *queue); }
        catch (...) { std::terminate(); }
    }
};
using Allocation = std::unique_ptr<std::uint32_t, ScratchDeleter>;
constexpr int warm_trials = 7;
constexpr std::size_t max_elements = 1U << 20;

enum class Arm { shared, device, shared_prefetch, shared_advice };

[[nodiscard]] std::optional<Arm> parse_arm(std::string_view value) noexcept {
    if (value == "shared") return Arm::shared;
    if (value == "device") return Arm::device;
    if (value == "shared_prefetch") return Arm::shared_prefetch;
    if (value == "shared_advice") return Arm::shared_advice;
    return std::nullopt;
}

[[nodiscard]] std::optional<std::size_t> parse_count(std::string_view value) noexcept {
    std::size_t count = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), count);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()
        || count == 0 || count > max_elements) return std::nullopt;
    return count;
}

[[nodiscard]] std::optional<int> parse_advice(std::string_view value) noexcept {
    int advice = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), advice);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return std::nullopt;
    return advice;
}

[[nodiscard]] double elapsed_ms(Clock::time_point begin) noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

[[nodiscard]] std::uint32_t expected(std::size_t index, std::uint32_t seed) noexcept {
    return static_cast<std::uint32_t>(index) ^ (seed * 0x9e3779b9U);
}

[[nodiscard]] bool verify(const std::uint32_t* values, std::size_t count,
                          std::uint32_t seed) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] != expected(i, seed)) return false;
    }
    return true;
}

struct Sample {
    double hint_ms;
    double kernel_ms;
    double readback_ms;
    double validation_ms;
    double validation_inclusive_ms;
};

/// Runs one fully overwritten scratch pass with host verification after GPU completion.
[[nodiscard]] Sample run_sample(sycl::queue& queue, Arm arm, std::uint32_t* scratch,
                                std::uint32_t* check, std::size_t count, std::uint32_t seed,
                                std::optional<int> advice) {
    const auto total_begin = Clock::now();
    const std::size_t bytes = count * sizeof(std::uint32_t);
    double hint_ms = 0.0;
    if (arm == Arm::shared_prefetch || arm == Arm::shared_advice) {
        const auto hint_begin = Clock::now();
        try {
            if (arm == Arm::shared_prefetch) queue.prefetch(scratch, bytes).wait_and_throw();
            else queue.mem_advise(scratch, bytes, *advice).wait_and_throw();
        } catch (const sycl::exception& error) {
            throw std::runtime_error(std::string("hint_unsupported: ") + error.what());
        }
        hint_ms = elapsed_ms(hint_begin);
    }

    // The kernel overwrites every element; no prior contents or CPU/GPU concurrent access is assumed.
    const auto kernel_begin = Clock::now();
    queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> item) {
        const std::size_t i = item[0];
        scratch[i] = static_cast<std::uint32_t>(i) ^ (seed * 0x9e3779b9U);
    }).wait_and_throw();
    const double kernel_ms = elapsed_ms(kernel_begin);

    const auto readback_begin = Clock::now();
    queue.memcpy(check, scratch, bytes).wait_and_throw();
    const double readback_ms = elapsed_ms(readback_begin);
    const auto validation_begin = Clock::now();
    if (!verify(check, count, seed)) throw std::runtime_error("Every-element scratch mismatch");
    const double validation_ms = elapsed_ms(validation_begin);
    return {hint_ms, kernel_ms, readback_ms, validation_ms, elapsed_ms(total_begin)};
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 5) {
            throw std::runtime_error("Usage: scratch_hint.exe <shared|device|shared_prefetch|shared_advice> <elements> [--advice-code <integer>]");
        }
        const auto arm = parse_arm(argv[1]);
        const auto count = parse_count(argv[2]);
        if (!arm || !count) throw std::runtime_error("Invalid arm or elements (1..1048576)");
        std::optional<int> advice;
        if (argc == 5) {
            if (std::string_view(argv[3]) != "--advice-code") throw std::runtime_error("Expected --advice-code");
            advice = parse_advice(argv[4]);
            if (!advice) throw std::runtime_error("Invalid advice code");
        }
        if (advice && *arm != Arm::shared_advice) throw std::runtime_error("Advice code requires shared_advice arm");
        std::cout << "schema=sub0.intel.scratch-hint-raw.v1\nbackend=level_zero\n"
                  << "execution_scope=selected\nselected_arm=" << argv[1] << '\n'
                  << "elements=" << *count << "\nbytes=" << *count * sizeof(std::uint32_t) << '\n'
                  << "timing_clock=std::chrono::steady_clock\n"
                  << "sample_columns=phase,arm,bytes,trial,hint_ms,kernel_ms,readback_ms,validation_ms,validation_inclusive_ms\n";
        if (*arm == Arm::shared_advice && !advice) {
            std::cout << "status=unsupported\nreason=no_verified_device_defined_advice_code\n";
            return 2;
        }
        const auto device = sub0::intel_probe::select_device();
        if (!device.has(sycl::aspect::usm_host_allocations)
            || (*arm != Arm::device && !device.has(sycl::aspect::usm_shared_allocations))
            || (*arm == Arm::device && !device.has(sycl::aspect::usm_device_allocations))) {
            std::cout << "status=unsupported\nreason=required_usm_aspect_absent\n";
            return 2;
        }
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (const auto& error : errors) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});
        Allocation scratch(*arm == Arm::device ? sycl::malloc_device<std::uint32_t>(*count, queue)
                                                : sycl::malloc_shared<std::uint32_t>(*count, queue), {&queue});
        Allocation check(sycl::malloc_host<std::uint32_t>(*count, queue), {&queue});
        if (!scratch || !check) throw std::runtime_error("Bounded USM allocation failed");
        std::cout << "allocation=" << (*arm == Arm::device ? "device" : "shared") << '\n'
                  << "advice_code=" << (advice ? argv[4] : "none") << '\n';
        for (int trial = -1; trial < warm_trials; ++trial) {
            const auto sample = run_sample(queue, *arm, scratch.get(), check.get(), *count,
                                           static_cast<std::uint32_t>(trial + 2), advice);
            std::cout << (trial < 0 ? "cold" : "warm") << ',' << argv[1] << ','
                      << *count * sizeof(std::uint32_t) << ',' << trial << ','
                      << sample.hint_ms << ',' << sample.kernel_ms << ',' << sample.readback_ms << ','
                      << sample.validation_ms << ',' << sample.validation_inclusive_ms << '\n';
        }
        std::cout << "verified_elements_per_run=" << *count << "\nstatus=pass\n";
        return 0;
    } catch (const std::exception& error) {
        const std::string_view message(error.what());
        if (message.substr(0, 18) == "hint_unsupported: ") {
            std::cout << "status=unsupported\nreason=" << message.substr(18) << '\n';
            return 2;
        }
        std::cerr << "status=fail\nerror=" << error.what() << '\n';
        return 1;
    }
}
