#include "sub0llm/nn/checkpoint.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace sub0llm {

void save_checkpoint(const std::vector<autograd::Variable>& params,
                     const std::string& dir, int64_t step)
{
    std::filesystem::create_directories(dir);

    const std::string filename = std::format("step_{:09d}.ckpt", step);
    const std::string path     = (std::filesystem::path(dir) / filename).string();

    // Build JSON header: format_version + step + list of shapes
    nlohmann::json header;
    header["format_version"] = kCheckpointFormatVersion;
    header["step"]           = step;
    nlohmann::json shapes = nlohmann::json::array();
    for (const auto& p : params) {
        nlohmann::json shape_arr = nlohmann::json::array();
        for (std::size_t d = 0; d < p.data().ndim(); ++d)
            shape_arr.push_back(p.data().shape(d));
        shapes.push_back(shape_arr);
    }
    header["shapes"] = shapes;

    const std::string header_str = header.dump();
    const uint32_t    header_len = static_cast<uint32_t>(header_str.size());

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::format("save_checkpoint: cannot open '{}' for writing", path));

    // 4-byte header length prefix
    f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    f.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));

    // Raw float32 data for each parameter. CUDA params are brought to host first (the checkpoint
    // is device-agnostic on disk; a GPU-trained model resumes on CPU or GPU identically).
    for (const auto& p : params) {
        const Tensor host = p.data().device().is_cpu() ? p.data() : p.data().to(Device::cpu());
        const auto sp     = host.data_as<float>();
        const auto n      = static_cast<std::streamsize>(sp.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(sp.data()), n);
    }
}

int64_t load_checkpoint(const std::vector<autograd::Variable>& params,
                         const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(
            std::format("load_checkpoint: file not found: '{}'", path));

    // Read header length
    uint32_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!f)
        throw std::runtime_error(
            std::format("load_checkpoint: truncated header-length in '{}'", path));

    // Read and parse JSON header
    std::string header_str(header_len, '\0');
    f.read(header_str.data(), static_cast<std::streamsize>(header_len));
    if (!f)
        throw std::runtime_error(
            std::format("load_checkpoint: truncated JSON header in '{}'", path));

    auto header = nlohmann::json::parse(header_str);

    // Reject incompatible checkpoints. Pre-versioned files (no "format_version")
    // are legacy v1 and used the interleaved RoPE convention, which would silently
    // produce wrong outputs under the current half-split convention.
    const int version = header.value("format_version", 1);
    if (version != kCheckpointFormatVersion)
        throw std::runtime_error(std::format(
            "load_checkpoint: incompatible checkpoint '{}' — format_version {} but this "
            "build expects {}. Checkpoints from before the RoPE convention change "
            "(half-split/NeoX) cannot be restored; retrain or convert the weights.",
            path, version, kCheckpointFormatVersion));

    const int64_t step = header["step"].get<int64_t>();
    const auto& shapes = header["shapes"];

    if (shapes.size() != params.size())
        throw std::runtime_error(
            std::format("load_checkpoint: param count mismatch: checkpoint has {} params, "
                        "model has {}",
                        shapes.size(), params.size()));

    // Read float data back into each parameter
    for (std::size_t i = 0; i < params.size(); ++i) {
        // Validate element count
        int64_t expected_numel = 1;
        for (const auto& d : shapes[i])
            expected_numel *= d.get<int64_t>();
        const int64_t actual_numel = params[i].data().numel();
        if (expected_numel != actual_numel)
            throw std::runtime_error(
                std::format("load_checkpoint: param {} numel mismatch: checkpoint={}, model={}",
                            i, expected_numel, actual_numel));

        // Read into a host buffer, then move to the param's device (H2D for CUDA params).
        auto&        var  = const_cast<autograd::Variable&>(params[i]);
        const Device dev  = var.data().device();
        Tensor       host = dev.is_cpu() ? var.data()
                                         : Tensor(var.data().shape(), DType::Float32, Device::cpu());
        auto sp = host.data_as<float>();
        f.read(reinterpret_cast<char*>(sp.data()),
               static_cast<std::streamsize>(sp.size() * sizeof(float)));
        if (!f)
            throw std::runtime_error(
                std::format("load_checkpoint: truncated data for param {} in '{}'", i, path));
        if (!dev.is_cpu())
            var.data() = host.to(dev);
    }

    return step;
}

std::string latest_checkpoint_path(const std::string& dir)
{
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
        return "";

    std::string best_path;
    int64_t     best_step = -1;
    // Pattern: step_XXXXXXXXX.ckpt
    const std::regex step_re(R"(step_(\d{9})\.ckpt)");

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string stem = entry.path().filename().string();
        std::smatch m;
        if (std::regex_match(stem, m, step_re)) {
            const int64_t s = std::stoll(m[1].str());
            if (s > best_step) {
                best_step = s;
                best_path = entry.path().string();
            }
        }
    }

    return best_path;
}

int64_t latest_checkpoint_step(const std::string& dir)
{
    const std::string p = latest_checkpoint_path(dir);
    if (p.empty()) return -1;

    // Extract step from filename stem
    const std::string stem = std::filesystem::path(p).filename().string();
    const std::regex  step_re(R"(step_(\d{9})\.ckpt)");
    std::smatch m;
    if (std::regex_match(stem, m, step_re))
        return std::stoll(m[1].str());
    return -1;
}

} // namespace sub0llm
