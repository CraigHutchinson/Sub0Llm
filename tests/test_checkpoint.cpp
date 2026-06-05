#include "sub0llm/nn/checkpoint.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string unique_tmp_dir(std::string_view tag)
{
    // Include PID for uniqueness across parallel test runs
    const auto base = std::filesystem::temp_directory_path()
                    / std::format("sub0llm_ckpt_{}_{}", tag,
                                  std::filesystem::path(__FILE__).stem().string());
    return base.string();
}

static void remove_dir(const std::string& dir)
{
    std::filesystem::remove_all(dir);
}

static Variable make_param(std::vector<int64_t> shape, float fill_value)
{
    Tensor t(shape, DType::Float32);
    for (float& v : t.data_as<float>()) v = fill_value;
    return Variable(std::move(t), /*requires_grad=*/true);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("save and load checkpoint round-trips param values", "[checkpoint]")
{
    const std::string dir = unique_tmp_dir("roundtrip");
    remove_dir(dir);

    // Two parameters with known values
    std::vector<Variable> params;
    params.push_back(make_param({4, 3}, 1.5f));
    params.push_back(make_param({7},    -2.0f));

    save_checkpoint(params, dir, /*step=*/100);

    // Corrupt the in-memory data
    for (float& v : params[0].data().data_as<float>()) v = 0.f;
    for (float& v : params[1].data().data_as<float>()) v = 0.f;

    // Reload
    const std::string path = latest_checkpoint_path(dir);
    REQUIRE(!path.empty());
    const int64_t loaded_step = load_checkpoint(params, path);

    REQUIRE(loaded_step == 100);

    for (const float v : params[0].data().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(1.5f, 1e-6f));
    for (const float v : params[1].data().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(-2.0f, 1e-6f));

    remove_dir(dir);
}

TEST_CASE("load throws on missing file", "[checkpoint]")
{
    const std::string nonexistent = "/tmp/sub0llm_no_such_file_xyz_999.ckpt";
    std::vector<Variable> params;
    params.push_back(make_param({2, 2}, 0.f));

    REQUIRE_THROWS_AS(load_checkpoint(params, nonexistent), std::runtime_error);
}

TEST_CASE("load throws on param count mismatch", "[checkpoint]")
{
    const std::string dir = unique_tmp_dir("mismatch");
    remove_dir(dir);

    // Save with 2 params
    std::vector<Variable> saved;
    saved.push_back(make_param({3, 3}, 1.f));
    saved.push_back(make_param({5},    2.f));
    save_checkpoint(saved, dir, 0);

    const std::string path = latest_checkpoint_path(dir);
    REQUIRE(!path.empty());

    // Try to load with 1 param
    std::vector<Variable> wrong;
    wrong.push_back(make_param({3, 3}, 0.f));

    REQUIRE_THROWS_AS(load_checkpoint(wrong, path), std::runtime_error);

    remove_dir(dir);
}

TEST_CASE("load rejects legacy checkpoint without format_version", "[checkpoint]")
{
    const std::string dir = unique_tmp_dir("legacy");
    remove_dir(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (std::filesystem::path(dir) / "step_000000000.ckpt").string();

    // Hand-craft a v1-style checkpoint: JSON header with NO "format_version"
    // (one param of shape [2]) followed by two float32 values.
    const std::string header = R"({"step":0,"shapes":[[2]]})";
    {
        std::ofstream f(path, std::ios::binary);
        const uint32_t hlen = static_cast<uint32_t>(header.size());
        f.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        const float vals[2] = {1.0f, 2.0f};
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    std::vector<Variable> params;
    params.push_back(make_param({2}, 0.f));
    REQUIRE_THROWS_AS(load_checkpoint(params, path), std::runtime_error);

    remove_dir(dir);
}

TEST_CASE("latest_checkpoint_path returns empty string when directory is empty", "[checkpoint]")
{
    const std::string dir = unique_tmp_dir("empty");
    std::filesystem::create_directories(dir);

    REQUIRE(latest_checkpoint_path(dir).empty());

    remove_dir(dir);
}

TEST_CASE("latest_checkpoint_path returns highest step among multiple checkpoints", "[checkpoint]")
{
    const std::string dir = unique_tmp_dir("multi");
    remove_dir(dir);

    std::vector<Variable> params;
    params.push_back(make_param({2}, 1.f));

    save_checkpoint(params, dir, 10);
    save_checkpoint(params, dir, 500);
    save_checkpoint(params, dir, 250);

    const std::string best = latest_checkpoint_path(dir);
    REQUIRE(!best.empty());

    const int64_t s = latest_checkpoint_step(dir);
    REQUIRE(s == 500);

    // The returned path should contain the highest step filename
    REQUIRE(best.find("step_000000500") != std::string::npos);

    remove_dir(dir);
}

TEST_CASE("save creates directory if it does not exist", "[checkpoint]")
{
    const std::string dir = unique_tmp_dir("autocreate") + "/nested/subdir";
    remove_dir(unique_tmp_dir("autocreate"));

    REQUIRE(!std::filesystem::exists(dir));

    std::vector<Variable> params;
    params.push_back(make_param({4}, 3.14f));

    REQUIRE_NOTHROW(save_checkpoint(params, dir, 0));
    REQUIRE(std::filesystem::exists(dir));

    remove_dir(unique_tmp_dir("autocreate"));
}
