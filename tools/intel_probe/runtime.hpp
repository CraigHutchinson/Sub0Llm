#pragma once

#include <exception>
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace sub0::intel_probe {
/// Frees USM only after completion; an unrecoverable drain error terminates the probe.
struct UsmDeleter {
    sycl::queue* queue; // non-owning; queue outlives every allocation
    void operator()(float* pointer) const noexcept {
        try { queue->wait(); sycl::free(pointer, *queue); }
        catch (...) { std::terminate(); }
    }
};

/// Selects only the measured PCI device through Level Zero; throws rather than falling back.
inline sycl::device select_device() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& device : platform.get_devices(sycl::info::device_type::gpu)) {
            if (device.get_info<sycl::info::device::vendor_id>() == 0x8086
                && device.has(sycl::aspect::ext_intel_device_id)
                && device.get_info<sycl::ext::intel::info::device::device_id>() == 0x7d67)
                return device;
        }
    }
    throw std::runtime_error("Required Intel 8086:7D67 Level Zero GPU missing; no fallback");
}
}
