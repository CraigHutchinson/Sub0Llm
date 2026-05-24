#pragma once
#include <cstdint>
#include <format>
#include <string>
#include <utility>   // std::unreachable

namespace sub0llm {

enum class DeviceType : std::uint8_t {
    CPU,
    CUDA,
    OpenVINO,
};

// Lightweight value type identifying a compute device.
struct Device {
    DeviceType type  = DeviceType::CPU;
    int        index = 0;  // 0 = first GPU, etc.

    // ── Named constructors ────────────────────────────────────────────────────
    [[nodiscard]] static constexpr Device cpu() noexcept {
        return {DeviceType::CPU, 0};
    }
    [[nodiscard]] static constexpr Device cuda(int idx = 0) noexcept {
        return {DeviceType::CUDA, idx};
    }
    [[nodiscard]] static constexpr Device openvino(int idx = 0) noexcept {
        return {DeviceType::OpenVINO, idx};
    }

    // ── Queries ───────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr bool is_cpu()      const noexcept { return type == DeviceType::CPU;      }
    [[nodiscard]] constexpr bool is_cuda()     const noexcept { return type == DeviceType::CUDA;     }
    [[nodiscard]] constexpr bool is_openvino() const noexcept { return type == DeviceType::OpenVINO; }

    [[nodiscard]] std::string str() const {
        switch (type) {
            case DeviceType::CPU:      return "cpu";
            case DeviceType::CUDA:     return std::format("cuda:{}", index);
            case DeviceType::OpenVINO: return std::format("openvino:{}", index);
        }
        std::unreachable();
    }

    // ── Comparison ────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr bool operator==(const Device&) const noexcept = default;
};

} // namespace sub0llm
