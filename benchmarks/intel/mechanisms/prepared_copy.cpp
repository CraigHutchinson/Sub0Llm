// Bounded Windows mapped-file staging spike. This is not a production memory path.
#include "../../../tools/intel_probe/runtime.hpp"

#include <Windows.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using Allocation = std::unique_ptr<float, sub0::intel_probe::UsmDeleter>;
constexpr int warm_trials = 7;

struct HandleCloser {
    void operator()(void* handle) const noexcept {
        if (handle != nullptr && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    }
};
using Handle = std::unique_ptr<void, HandleCloser>;
struct ViewCloser {
    void operator()(const float* pointer) const noexcept {
        if (pointer != nullptr) UnmapViewOfFile(pointer);
    }
};
using View = std::unique_ptr<const float, ViewCloser>;

struct Mapping {
    Handle file;
    Handle mapping;
    View view;
};

[[nodiscard]] double milliseconds(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

[[nodiscard]] Mapping open_read_only(const char* path, std::size_t bytes) {
    Mapping result;
    result.file.reset(CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr));
    if (result.file.get() == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open fixture");
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(result.file.get(), &length) || length.QuadPart != static_cast<LONGLONG>(bytes))
        throw std::runtime_error("Wrong fixture extent");
    result.mapping.reset(CreateFileMappingA(result.file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr));
    if (!result.mapping) throw std::runtime_error("Cannot create fixture mapping");
    result.view.reset(static_cast<const float*>(MapViewOfFile(result.mapping.get(), FILE_MAP_READ, 0, 0, bytes)));
    if (!result.view) throw std::runtime_error("Cannot map fixture");
    return result;
}

enum class Mode { ordinary, prepared, host_staging };

#if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE) && SYCL_EXT_ONEAPI_COPY_OPTIMIZE >= 1
/// Owns one successful preparation until an explicit release or scope exit.
class PreparedRange {
public:
    PreparedRange(const void* pointer, std::size_t bytes, sycl::queue& queue)
        : pointer_(pointer), queue_(&queue) {
        sycl::ext::oneapi::experimental::prepare_for_device_copy(pointer_, bytes, *queue_);
        active_ = true;
    }
    PreparedRange(const PreparedRange&) = delete;
    PreparedRange& operator=(const PreparedRange&) = delete;
    /** Releases after draining the queue when stack unwinding bypasses explicit release.
     * @note An unrecoverable drain or release failure terminates the probe.
     */
    ~PreparedRange() noexcept {
        if (!active_) return;
        try {
            queue_->wait();
            sycl::ext::oneapi::experimental::release_from_device_copy(pointer_, *queue_);
        }
        catch (...) { std::terminate(); }
    }
    void release() {
        sycl::ext::oneapi::experimental::release_from_device_copy(pointer_, *queue_);
        active_ = false;
    }

private:
    const void* pointer_ = nullptr; // non-owning; mapped view outlives this guard
    sycl::queue* queue_ = nullptr; // non-owning; queue outlives this guard
    bool active_ = false;
};
#endif

[[nodiscard]] const char* mode_name(Mode mode) {
    switch (mode) {
    case Mode::ordinary: return "ordinary";
    case Mode::prepared: return "prepared";
    case Mode::host_staging: return "host_usm_staging";
    }
    throw std::runtime_error("Invalid mode");
}

void verify(const float* actual, const float* expected, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float wanted = expected[i] * 2.0f + 1.0f;
        if (actual[i] != wanted) throw std::runtime_error("Every-element correctness mismatch");
    }
}

struct Sample {
    double staging_ms;
    double transfer_ms;
    double kernel_ms;
    double readback_ms;
    double validation_ms;
};

void measure(Mode mode, sycl::queue& queue, const Mapping& first, const Mapping& changed,
             std::size_t count) {
    const std::size_t bytes = count * sizeof(float);
    Allocation device_input(sycl::malloc_device<float>(count, queue), {&queue});
    Allocation device_output(sycl::malloc_device<float>(count, queue), {&queue});
    Allocation host_staging(mode == Mode::host_staging ? sycl::malloc_host<float>(count, queue) : nullptr,
                            {&queue});
    Allocation check(sycl::malloc_host<float>(count, queue), {&queue});
    if (!device_input || !device_output || !check || (mode == Mode::host_staging && !host_staging))
        throw std::runtime_error("Bounded USM allocation failed");

#if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE) && SYCL_EXT_ONEAPI_COPY_OPTIMIZE >= 1
    std::optional<PreparedRange> first_prepared;
    std::optional<PreparedRange> changed_prepared;
    double setup_first_ms = 0.0;
    double setup_changed_ms = 0.0;
    if (mode == Mode::prepared) {
        const auto first_begin = Clock::now();
        first_prepared.emplace(first.view.get(), bytes, queue);
        setup_first_ms = milliseconds(first_begin);
        const auto changed_begin = Clock::now();
        changed_prepared.emplace(changed.view.get(), bytes, queue);
        setup_changed_ms = milliseconds(changed_begin);
    }
#else
    if (mode == Mode::prepared) throw std::runtime_error("Prepared mode compiled without extension support");
    constexpr double setup_first_ms = 0.0;
    constexpr double setup_changed_ms = 0.0;
#endif

    const auto submit = [&](const float* mapped_source) {
        const float* copy_source = mapped_source;
        double staging_ms = 0.0;
        if (mode == Mode::host_staging) {
            const auto staging_begin = Clock::now();
            std::copy_n(mapped_source, count, host_staging.get());
            staging_ms = milliseconds(staging_begin);
            copy_source = host_staging.get();
        }
        const auto transfer_begin = Clock::now();
        queue.memcpy(device_input.get(), copy_source, bytes).wait_and_throw();
        const double transfer_ms = milliseconds(transfer_begin);
        const float* input = device_input.get();
        float* output = device_output.get();
        const auto kernel_begin = Clock::now();
        queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            output[i] = input[i] * 2.0f + 1.0f;
        }).wait_and_throw();
        const double kernel_ms = milliseconds(kernel_begin);
        const auto readback_begin = Clock::now();
        queue.memcpy(check.get(), device_output.get(), bytes).wait_and_throw();
        const double readback_ms = milliseconds(readback_begin);
        const auto validation_begin = Clock::now();
        verify(check.get(), mapped_source, count);
        const double validation_ms = milliseconds(validation_begin);
        return Sample{staging_ms, transfer_ms, kernel_ms, readback_ms, validation_ms};
    };

    const auto cold_begin = Clock::now();
    const auto cold_parts = submit(first.view.get());
    const double cold_validation_inclusive_ms = milliseconds(cold_begin);
    std::cout << "cold," << mode_name(mode) << ',' << bytes << ',' << setup_first_ms << ','
              << setup_changed_ms << ',' << cold_parts.staging_ms << ',' << cold_parts.transfer_ms << ','
              << cold_parts.kernel_ms << ',' << cold_parts.readback_ms << ','
              << cold_parts.validation_ms << ',' << cold_validation_inclusive_ms << '\n';

    for (int trial = 0; trial < warm_trials; ++trial) {
        const auto begin = Clock::now();
        const auto parts = submit((trial % 2 == 0) ? changed.view.get() : first.view.get());
        const double validation_inclusive_ms = milliseconds(begin);
        std::cout << "warm," << mode_name(mode) << ',' << bytes << ',' << trial << ','
                  << parts.staging_ms << ',' << parts.transfer_ms << ',' << parts.kernel_ms << ','
                  << parts.readback_ms << ',' << parts.validation_ms << ','
                  << validation_inclusive_ms << '\n';
    }

    queue.wait_and_throw();
#if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE) && SYCL_EXT_ONEAPI_COPY_OPTIMIZE >= 1
    double release_changed_ms = 0.0;
    double release_first_ms = 0.0;
    if (mode == Mode::prepared) {
        const auto changed_begin = Clock::now();
        changed_prepared->release();
        release_changed_ms = milliseconds(changed_begin);
        const auto first_begin = Clock::now();
        first_prepared->release();
        release_first_ms = milliseconds(first_begin);
    }
    std::cout << "release," << mode_name(mode) << ',' << bytes << ',' << release_first_ms << ','
              << release_changed_ms << '\n';
#else
    std::cout << "release," << mode_name(mode) << ',' << bytes << ",0,0\n";
#endif
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: prepared_copy.exe <fixture-a> <fixture-b> <elements>");
        const std::size_t count = std::stoull(argv[3]);
        if (count == 0 || count > 1024 * 1024) throw std::runtime_error("Elements must be in [1,1048576]");
        const std::size_t bytes = count * sizeof(float);
        const auto first = open_read_only(argv[1], bytes);
        const auto changed = open_read_only(argv[2], bytes);
        const auto device = sub0::intel_probe::select_device();
        if (!device.has(sycl::aspect::usm_device_allocations)
            || !device.has(sycl::aspect::usm_host_allocations))
            throw std::runtime_error("Required host/device USM unavailable");
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (auto error : errors) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});
        std::cout << "schema=sub0.intel.prepared-copy.v1\nbackend=level_zero\n"
                  << "cold_columns=phase,mode,bytes,setup_first_ms,setup_changed_ms,staging_ms,transfer_ms,kernel_ms,readback_ms,validation_ms,validation_inclusive_ms\n"
                  << "warm_columns=phase,mode,bytes,trial,staging_ms,transfer_ms,kernel_ms,readback_ms,validation_ms,validation_inclusive_ms\n"
                  << "release_columns=phase,mode,bytes,release_first_ms,release_changed_ms\n";
#if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE) && SYCL_EXT_ONEAPI_COPY_OPTIMIZE >= 1
        std::cout << "copy_optimize_macro=" << SYCL_EXT_ONEAPI_COPY_OPTIMIZE
                  << "\nprepared_status=supported\n";
        for (const auto mode : {Mode::ordinary, Mode::prepared, Mode::host_staging})
            measure(mode, queue, first, changed, count);
#else
        std::cout << "copy_optimize_macro=unavailable\nprepared_status=unsupported\n"
                  << "prepared_reason=SYCL_EXT_ONEAPI_COPY_OPTIMIZE >= 1 is required\n";
        for (const auto mode : {Mode::ordinary, Mode::host_staging})
            measure(mode, queue, first, changed, count);
#endif
        std::cout << "verified_elements_per_run=" << count << "\nstatus=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "status=fail\nerror=" << error.what() << '\n';
        return 1;
    }
}
