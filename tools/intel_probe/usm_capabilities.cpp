// I19/S1 capability inventory. This diagnostic makes no performance claims.
#include "runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sycl/sycl.hpp>

#if defined(SUB0_ENABLE_LEVEL_ZERO_INVENTORY)
#if !__has_include(<level_zero/ze_api.h>)
#error SUB0_ENABLE_LEVEL_ZERO_INVENTORY requires level_zero/ze_api.h
#endif
#include <level_zero/ze_api.h>
#define SUB0_HAS_LEVEL_ZERO_HEADERS 1
#else
#define SUB0_HAS_LEVEL_ZERO_HEADERS 0
#endif

namespace {

void print_aspect(const sycl::device& device, std::string_view name, sycl::aspect aspect) {
    std::cout << name << '=' << device.has(aspect) << '\n';
}

#if SUB0_HAS_LEVEL_ZERO_HEADERS
void print_level_zero_extensions(const sycl::device& selected_device) {
    auto result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    std::cout << "level_zero_init_result=" << static_cast<unsigned>(result) << '\n';
    if (result != ZE_RESULT_SUCCESS) throw std::runtime_error("Level Zero initialization failed");

    std::uint32_t driver_count = 0;
    result = zeDriverGet(&driver_count, nullptr);
    std::cout << "level_zero_driver_count_result=" << static_cast<unsigned>(result) << '\n';
    if (result != ZE_RESULT_SUCCESS) throw std::runtime_error("Level Zero driver count failed");
    if (driver_count == 0) throw std::runtime_error("Level Zero reported no GPU drivers");

    constexpr std::uint32_t driver_capacity = 16;
    ze_driver_handle_t drivers[driver_capacity]{};
    auto requested_drivers = driver_count < driver_capacity ? driver_count : driver_capacity;
    result = zeDriverGet(&requested_drivers, drivers);
    std::cout << "level_zero_driver_inventory_total=" << driver_count << '\n'
              << "level_zero_driver_inventory_recorded=" << requested_drivers << '\n'
              << "level_zero_driver_inventory_truncated=" << (driver_count > driver_capacity) << '\n'
              << "level_zero_driver_get_result=" << static_cast<unsigned>(result) << '\n';
    if (result != ZE_RESULT_SUCCESS) throw std::runtime_error("Level Zero driver enumeration failed");
    if (driver_count > driver_capacity)
        throw std::runtime_error("Level Zero driver inventory exceeded probe capacity");

    const auto native_device =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(selected_device);
    ze_driver_handle_t selected_driver = nullptr;
    for (std::uint32_t driver_index = 0; driver_index < requested_drivers; ++driver_index) {
        std::uint32_t device_count = 0;
        result = zeDeviceGet(drivers[driver_index], &device_count, nullptr);
        if (result != ZE_RESULT_SUCCESS)
            throw std::runtime_error("Level Zero device count failed during driver association");

        constexpr std::uint32_t device_capacity = 32;
        ze_device_handle_t devices[device_capacity]{};
        auto requested_devices = device_count < device_capacity ? device_count : device_capacity;
        if (device_count > device_capacity)
            throw std::runtime_error("Level Zero device inventory exceeded probe capacity");
        result = zeDeviceGet(drivers[driver_index], &requested_devices, devices);
        if (result != ZE_RESULT_SUCCESS)
            throw std::runtime_error("Level Zero device enumeration failed during driver association");
        for (std::uint32_t device_index = 0; device_index < requested_devices; ++device_index) {
            if (devices[device_index] == native_device) {
                selected_driver = drivers[driver_index];
                break;
            }
        }
        if (selected_driver != nullptr) break;
    }
    std::cout << "level_zero_selected_driver_matched=" << (selected_driver != nullptr) << '\n';
    if (selected_driver == nullptr)
        throw std::runtime_error("Selected SYCL device was not associated with a Level Zero driver");

    ze_api_version_t api_version{};
    result = zeDriverGetApiVersion(selected_driver, &api_version);
    std::cout << "level_zero_api_version_result=" << static_cast<unsigned>(result) << '\n';
    if (result != ZE_RESULT_SUCCESS)
        throw std::runtime_error("Level Zero API-version query failed for selected driver");
    std::cout << "level_zero_api_version=" << ZE_MAJOR_VERSION(api_version) << '.'
              << ZE_MINOR_VERSION(api_version) << '\n';

    std::uint32_t extension_count = 0;
    result = zeDriverGetExtensionProperties(selected_driver, &extension_count, nullptr);
    std::cout << "level_zero_extension_count_result=" << static_cast<unsigned>(result) << '\n';
    if (result != ZE_RESULT_SUCCESS)
        throw std::runtime_error("Level Zero extension count failed for selected driver");

    constexpr std::uint32_t capacity = 256;
    ze_driver_extension_properties_t extensions[capacity]{};
    auto requested_extensions = extension_count < capacity ? extension_count : capacity;
    if (extension_count > capacity)
        throw std::runtime_error("Level Zero extension inventory exceeded probe capacity");
    result = zeDriverGetExtensionProperties(selected_driver, &requested_extensions, extensions);
    std::cout << "level_zero_extension_inventory_total=" << extension_count << '\n'
              << "level_zero_extension_inventory_recorded=" << requested_extensions << '\n'
              << "level_zero_extension_inventory_truncated=" << (extension_count > capacity) << '\n'
              << "level_zero_extension_get_result=" << static_cast<unsigned>(result) << '\n';
    if (result != ZE_RESULT_SUCCESS)
        throw std::runtime_error("Level Zero extension enumeration failed for selected driver");

    for (std::uint32_t index = 0; index < requested_extensions; ++index) {
        const auto version = extensions[index].version;
        std::cout << "level_zero_extension=" << extensions[index].name
                  << ",version=" << ZE_MAJOR_VERSION(version) << '.' << ZE_MINOR_VERSION(version) << '\n';
    }
}
#endif

} // namespace

int main() {
    try {
        const auto device = sub0::intel_probe::select_device();
        std::cout << "device=" << device.get_info<sycl::info::device::name>() << '\n'
                  << "backend=level_zero\n"
                  << "device_id=0x7d67\n"
                  << "driver=" << device.get_info<sycl::info::device::driver_version>() << '\n'
                  << "level_zero_inventory_built=" << SUB0_HAS_LEVEL_ZERO_HEADERS << '\n';
        print_aspect(device, "usm_host", sycl::aspect::usm_host_allocations);
        print_aspect(device, "usm_shared", sycl::aspect::usm_shared_allocations);
        print_aspect(device, "usm_device", sycl::aspect::usm_device_allocations);
        print_aspect(device, "usm_system", sycl::aspect::usm_system_allocations);
        print_aspect(device, "usm_atomic_host", sycl::aspect::usm_atomic_host_allocations);
        print_aspect(device, "usm_atomic_shared", sycl::aspect::usm_atomic_shared_allocations);
#if SUB0_HAS_LEVEL_ZERO_HEADERS
        print_level_zero_extensions(device);
#else
        std::cout << "level_zero_extension_inventory=not_built_development_files_not_supplied\n";
#endif

#ifdef SUB0_PROBE_PREPARED_COPY_API
#ifdef SYCL_EXT_ONEAPI_COPY_OPTIMIZE
        sycl::queue queue(device, sycl::property::queue::in_order{});
        constexpr std::size_t bytes = 4096;
        alignas(64) std::byte host_range[bytes]{};
        sycl::ext::oneapi::experimental::prepare_for_device_copy(host_range, bytes, queue);
        sycl::ext::oneapi::experimental::release_from_device_copy(host_range, queue);
        std::cout << "prepared_copy_api_runtime=pass\n";
#else
#error SUB0_PROBE_PREPARED_COPY_API requires SYCL_EXT_ONEAPI_COPY_OPTIMIZE
#endif
#else
        std::cout << "prepared_copy_api_runtime=not_built\n";
#ifdef SYCL_EXT_ONEAPI_COPY_OPTIMIZE
        std::cout << "prepared_copy_api_header=declared\n";
#else
        std::cout << "prepared_copy_api_header=not_declared\n";
#endif
#endif
        std::cout << "status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "status=fail\nerror=" << error.what() << '\n';
        return 1;
    }
}
