// param_arena_tests.cpp -- O14: the parameter arena as a read-only view of the model file.
//
// A mapped arena is only legal where nothing writes the arena after load, so the positive cases need a
// FORWARD-ONLY engine build (Gated Residual / MoE / QSA). This file is its own executable for that reason:
// the default sub0llm_tests config can train, where adopt_param_file_view() must refuse, and its assertion
// counts must stay exactly what they were. In a trainable build the positive cases skip and the refusal
// case runs; configure the build with e.g. `--num-experts 4 --experts-per-tok 2` (and optionally
// `--param-arena mapped` to also drive load_model's own adoption) to run them.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr bool kForwardOnly = sub0::USE_GATED_RESIDUAL || sub0::USE_MOE || sub0::USE_QSA;
constexpr std::uint64_t kHeaderBytes = 48;   // sizeof(ModelHeader), pinned by model_file.hpp's static_assert

std::vector<std::uint8_t> arena_bytes() {
    const auto* p = static_cast<const std::uint8_t*>(sub0::param_store_view());
    return {p, p + sub0::param_store_bytes()};
}

std::vector<float> logits_of(const std::vector<int>& ids) {
    sub0::graph_reset();
    const sub0::Node* logits = sub0::forward(ids.data(), static_cast<int>(ids.size()));
    return {logits->data.begin(), logits->data.end()};
}

std::vector<int> window(int T) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> ids(static_cast<std::size_t>(T));
    for (int& id : ids) id = tok(rng);
    return ids;
}

struct TempModel {
    explicit TempModel(const char* name = "sub0_param_arena_model.bin")
        : path((std::filesystem::temp_directory_path() / name).string()) {}
    std::string path;
    ~TempModel() {
        (void)sub0::param_store_ptr();   // a mapped view holds the file open; a writer un-maps it first
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

}  // namespace

TEST_CASE("a build that can train refuses a file-view arena and stays on the heap", "[param_arena]") {
    if (kForwardOnly) SKIP("forward-only build: adoption is legal here");
    sub0::build_model();
    TempModel m;
    REQUIRE(sub0::save_model(m.path.c_str()));
    REQUIRE_FALSE(sub0::adopt_param_file_view(m.path.c_str(), kHeaderBytes));
    REQUIRE_FALSE(sub0::param_arena_mapped());
    REQUIRE(sub0::load_model(m.path.c_str()));   // the flag never changes whether a load works
    REQUIRE_FALSE(sub0::param_arena_mapped());
}

TEST_CASE("a file-view arena is byte- and output-identical to the heap arena", "[param_arena]") {
    if (!kForwardOnly) SKIP("needs a forward-only build (Gated Residual / MoE / QSA)");
    sub0::build_model();   // nodes exist BEFORE the arena moves: adoption must re-derive their pointers
    const std::vector<int> ids = window(8);
    const std::vector<std::uint8_t> bytes0 = arena_bytes();
    const std::vector<float> out0 = logits_of(ids);
    TempModel m;
    REQUIRE(sub0::save_model(m.path.c_str()));

    // Scribble over the heap arena so a view that silently kept the heap bytes could not pass.
    std::memset(sub0::param_store_ptr(), 0x5A, sub0::param_store_bytes());
    REQUIRE(arena_bytes() != bytes0);

    REQUIRE(sub0::adopt_param_file_view(m.path.c_str(), kHeaderBytes));
    REQUIRE(sub0::param_arena_mapped());
    REQUIRE(arena_bytes() == bytes0);
    REQUIRE(logits_of(ids) == out0);   // the captured node pointers follow the arena, bit for bit

    SECTION("a writer turns the view back into a writable heap arena with the same bytes") {
        void* w = sub0::param_store_ptr();
        REQUIRE_FALSE(sub0::param_arena_mapped());
        REQUIRE(arena_bytes() == bytes0);
        static_cast<std::uint8_t*>(w)[0] ^= 0xFF;   // writable now: would fault on a mapped view
        static_cast<std::uint8_t*>(w)[0] ^= 0xFF;
        REQUIRE(logits_of(ids) == out0);
    }
    SECTION("build_model after a view re-randomizes on the heap, not through the view") {
        sub0::build_model();
        REQUIRE_FALSE(sub0::param_arena_mapped());
        REQUIRE(arena_bytes() == bytes0);   // build_model's init is deterministic
    }
    SECTION("saving from a view does not un-map it, and writes the same file") {
        TempModel again("sub0_param_arena_resave.bin");
        REQUIRE(sub0::save_model(again.path.c_str()));
        REQUIRE(sub0::param_arena_mapped());
        REQUIRE(std::filesystem::file_size(again.path) == std::filesystem::file_size(m.path));
    }
}

TEST_CASE("load_model follows PARAM_ARENA_MAPPED and loads the same bytes either way", "[param_arena]") {
    if (!kForwardOnly) SKIP("needs a forward-only build (Gated Residual / MoE / QSA)");
    sub0::build_model();
    const std::vector<int> ids = window(8);
    const std::vector<std::uint8_t> bytes0 = arena_bytes();
    const std::vector<float> out0 = logits_of(ids);
    TempModel m;
    REQUIRE(sub0::save_model(m.path.c_str()));
    std::memset(sub0::param_store_ptr(), 0x5A, sub0::param_store_bytes());

    REQUIRE(sub0::load_model(m.path.c_str()));
    REQUIRE(sub0::param_arena_mapped() == PARAM_ARENA_MAPPED);
    REQUIRE(arena_bytes() == bytes0);
    REQUIRE(logits_of(ids) == out0);
}
