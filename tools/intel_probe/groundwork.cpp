// I00/S0/S1: bounded correctness probe; deliberately not a performance benchmark.
#include "runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <sycl/sycl.hpp>
#define NOMINMAX
#include <windows.h>
#ifdef SUB0_PROBE_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

namespace {
#ifndef SUB0_PROBE_ELEMENTS
#define SUB0_PROBE_ELEMENTS 65536
#endif
constexpr std::size_t count = SUB0_PROBE_ELEMENTS;
static_assert(count == 257 || count == 65536);
constexpr std::size_t bytes = count * sizeof(float);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

/// Releases a Windows file or mapping handle owned by this probe.
struct CloseHandleDeleter {
    void operator()(void* handle) const noexcept { CloseHandle(handle); }
};
/// Unmaps an owned view after all device work using its staged copy has completed.
struct UnmapDeleter {
    void operator()(void* view) const noexcept { UnmapViewOfFile(view); }
};

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Usage: groundwork.exe <generated float fixture for this build>");
        const auto device = sub0::intel_probe::select_device();
        std::cout << "device=" << device.get_info<sycl::info::device::name>() << '\n'
                  << "backend=level_zero\ndevice_id=0x7d67\n"
                  << "driver=" << device.get_info<sycl::info::device::driver_version>() << '\n'
                  << "reported_global_bytes=" << device.get_info<sycl::info::device::global_mem_size>() << '\n'
                  << "reported_max_allocation_bytes=" << device.get_info<sycl::info::device::max_mem_alloc_size>() << '\n'
                  << "compute_units=" << device.get_info<sycl::info::device::max_compute_units>() << '\n'
                  << "local_mem_bytes=" << device.get_info<sycl::info::device::local_mem_size>() << '\n';
        for (auto [name, aspect] : {std::pair{"fp16", sycl::aspect::fp16},
             {"fp64", sycl::aspect::fp64}, {"usm_host", sycl::aspect::usm_host_allocations},
             {"usm_shared", sycl::aspect::usm_shared_allocations},
             {"usm_device", sycl::aspect::usm_device_allocations},
             {"usm_system", sycl::aspect::usm_system_allocations}})
            std::cout << name << '=' << device.has(aspect) << '\n';
        std::cout << "subgroup_sizes=";
        for (auto width : device.get_info<sycl::info::device::sub_group_sizes>()) std::cout << width << ',';
        std::cout << '\n';
        try {
            const auto combinations = device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
            std::cout << "reported_matrix_combinations=" << combinations.size() << '\n';
        } catch (const sycl::exception& error) {
            std::cout << "reported_matrix_combinations=unknown\nmatrix_query_error=" << error.what() << '\n';
        }
        require(device.has(sycl::aspect::usm_shared_allocations)
                && device.has(sycl::aspect::usm_device_allocations), "Required USM mode unavailable");
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (auto error : errors) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});

        const auto raw_file = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, nullptr,
                                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(raw_file != INVALID_HANDLE_VALUE, "Cannot open generated fixture");
        std::unique_ptr<void, CloseHandleDeleter> file(raw_file);
        LARGE_INTEGER length{};
        require(GetFileSizeEx(file.get(), &length) && length.QuadPart == bytes, "Wrong fixture extent");
        std::unique_ptr<void, CloseHandleDeleter> mapping(
            CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr));
        require(mapping != nullptr, "Cannot create read-only file mapping");
        std::unique_ptr<void, UnmapDeleter> view(MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, bytes));
        require(view != nullptr, "Cannot map fixture");
        std::unique_ptr<float, sub0::intel_probe::UsmDeleter> shared(sycl::malloc_shared<float>(count, queue), {&queue});
        std::unique_ptr<float, sub0::intel_probe::UsmDeleter> staged(sycl::malloc_device<float>(count, queue), {&queue});
        require(shared && staged, "Bounded allocation failed");
        auto* host = shared.get();
        auto* gpu = staged.get();
        std::memcpy(host, view.get(), bytes);
        // The file contains i % 257 - 128; exact integer-valued floats avoid tolerance ambiguity.
        for (std::size_t i = 0; i < count; ++i)
            require(host[i] == float(int(i % 257) - 128), "Fixture contents invalid");
        for (int pass = 0; pass < 2; ++pass) {
            queue.memcpy(gpu, host, bytes);
            queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) { gpu[i] = gpu[i] * 2 + 1; });
            queue.memcpy(host, gpu, bytes).wait_and_throw();
        }
        for (std::size_t i = 0; i < count; ++i)
            require(host[i] == 4 * float(int(i % 257) - 128) + 3, "Device round-trip mismatch");
        std::cout << "mapped_staged_roundtrip=pass\nverified_elements=" << count
                  << "\nexplicit_usm_bytes=" << 2 * bytes
                  << "\ndirect_mapped_access=not_tested\n";
#ifdef SUB0_PROBE_DNNL
        // Same queue/context and borrowed USM: custom kernel -> library -> custom kernel.
        const auto* version = dnnl_version();
        std::cout << "onednn_version=" << version->major << '.' << version->minor << '.' << version->patch << '\n';
        auto engine = dnnl::sycl_interop::make_engine(device, queue.get_context());
        auto stream = dnnl::sycl_interop::make_stream(engine, queue);
        auto desc = dnnl::memory::desc({static_cast<std::int64_t>(count)},
            dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);
        auto source = dnnl::sycl_interop::make_memory(desc, engine,
            dnnl::sycl_interop::memory_kind::usm, gpu);
        auto destination = dnnl::sycl_interop::make_memory(desc, engine,
            dnnl::sycl_interop::memory_kind::usm, host);
        auto pd = dnnl::eltwise_forward::primitive_desc(engine, dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_relu, desc, desc, 0.0f, 0.0f);
        auto primitive = dnnl::eltwise_forward(pd);
        primitive.execute(stream, {{DNNL_ARG_SRC, source}, {DNNL_ARG_DST, destination}});
        stream.wait();
        queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) { host[i] += 7; }).wait_and_throw();
        for (std::size_t i = 0; i < count; ++i)
            require(host[i] == std::max(0.0f, 4 * float(int(i % 257) - 128) + 3) + 7,
                    "oneDNN interop mismatch");
        std::cout << "onednn_shared_context_relu=pass\n";
#else
        std::cout << "onednn_shared_context_relu=not_built\n";
#endif
        std::cout << "status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "status=fail\nerror=" << error.what() << '\n';
        return 1;
    }
}
