// backbone_quant_tests.cpp -- unit tests for sub0::bbq, the S0B1 native-quant BACKBONE sidecar
// (include/sub0/backbone_quant.hpp, docs/BACKBONE_NATIVE_QUANT.md S9/S10, O5 phase 2b-1).
//
// Engine-free, like moe_quant_tests.cpp/transplant_tests.cpp: backbone_quant.hpp deliberately does not
// include sub0_config.hpp, so none of this needs a compiled model.
//
// WHAT EACH CASE EXISTS TO RULE OUT, rather than merely exercise:
//   * role_pattern/role_name: a role silently missing from either switch (Role::Count falls through to
//     nullptr/"?" at runtime rather than a compile error, since C++ has no exhaustiveness check on a
//     switch over an enum class without -Werror=switch here) -- checked for every enumerator but Count.
//   * the SPARSE index: unlike moeq's dense (layer, expert, which) grid, this format's per-layer roles
//     may be absent at some layers (a QSA role on a GDN layer) -- Store::find must return nullptr for a
//     real gap, not misattribute a neighbouring entry, and a duplicate (role, layer) pair must be refused
//     at open() rather than silently picked by insertion order.
//   * dequantize_role_to_f32 reproducing gguf::to_f32 exactly -- the whole point of "raw bytes, an
//     independent decode of the SAME bytes must agree, or the round-trip gate the tool runs is measuring
//     a coincidence.
//   * Store refusing a truncated/foreign/version-mismatched/model_param_floats-mismatched file rather
//     than reading garbage -- AGENTS.md S3's "an old reader meeting a new file must not silently misread"
//     taken literally: this IS the old-vs-new case, since nothing has ever read an S0B1 file before.

#include "sub0/backbone_quant.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace sub0;

namespace {

std::string temp_path(const char* stem) {
    return (std::filesystem::temp_directory_path() / stem).string();
}

// Deterministic pseudo-random Q8_0 blocks -- same construction as moe_quant_tests.cpp's own make_q8_0,
// with every block's f16 scale patched to a small positive value so gguf::to_f32 never sees inf/NaN.
std::vector<std::uint8_t> make_q8_0(std::uint64_t n_elements, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const std::uint64_t blocks = (n_elements + 31) / 32;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(blocks) * 34);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        std::uint8_t* blk = raw.data() + b * 34;
        const std::uint16_t d_bits = static_cast<std::uint16_t>(0x3000u + (rng() & 0x0FFFu));
        std::memcpy(blk, &d_bits, 2);
        for (int i = 0; i < 32; ++i) blk[2 + i] = static_cast<std::uint8_t>(rng() & 0xFFu);
    }
    return raw;
}

constexpr int kIn = 32, kOut = 4;   // one Q8_0 block per row -- small, real block-aligned geometry

// A hand-built S0B1 file: a mix of per-layer and model-level roles, WITH a deliberate gap (one per-layer
// role missing at one layer, the same shape a QSA role on a GDN layer produces in the real file) so the
// sparse-index tests below have a real case to exercise, not just a fully-dense table that happens to
// behave like a dense one.
struct Built {
    std::string path;
    // (role, layer) -> the raw bytes written for that descriptor, so a test can compare against them.
    std::vector<std::tuple<bbq::Role, int, std::vector<std::uint8_t>>> entries;
};

Built build_sidecar(const std::string& path, std::uint64_t model_param_floats) {
    Built out;
    out.path = path;
    struct Plan { bbq::Role role; int layer; };
    const std::vector<Plan> plan = {
        {bbq::Role::TokEmb, -1},
        {bbq::Role::GrAttnDown, 0},
        {bbq::Role::GrAttnDown, 1},
        // Role::GrAttnUp is deliberately ABSENT at layer 1 -- the sparse-gap case.
        {bbq::Role::GrAttnUp, 0},
        {bbq::Role::QsaKProj, 2},
    };
    std::vector<bbq::Desc> descs(plan.size());
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        std::vector<std::uint8_t> bytes = make_q8_0(static_cast<std::uint64_t>(kIn) * kOut,
                                                     static_cast<std::uint32_t>(2000 + i));
        descs[i] = bbq::Desc{static_cast<std::int32_t>(plan[i].role), plan[i].layer,
                             static_cast<std::uint32_t>(gguf::TensorType::Q8_0), kIn, kOut, 0, cursor,
                             bytes.size()};
        cursor += bytes.size();
        out.entries.emplace_back(plan[i].role, plan[i].layer, std::move(bytes));
    }
    bbq::Header h;
    h.n_layers = 3;
    h.n_tensors = descs.size();
    h.data_off = sizeof(bbq::Header) + descs.size() * sizeof(bbq::Desc);
    h.data_bytes = cursor;
    h.model_param_floats = model_param_floats;
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    os.write(reinterpret_cast<const char*>(&h), sizeof h);
    os.write(reinterpret_cast<const char*>(descs.data()),
             static_cast<std::streamsize>(descs.size() * sizeof(bbq::Desc)));
    for (const auto& [role, layer, bytes] : out.entries)
        os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    os.close();
    return out;
}

}  // namespace

TEST_CASE("bbq: the on-disk structs are the size the format says", "[backbonequantsidecar]") {
    REQUIRE(sizeof(bbq::Header) == 48);
    REQUIRE(sizeof(bbq::Desc) == 40);
    REQUIRE(alignof(bbq::Header) == 8);
    REQUIRE(alignof(bbq::Desc) == 8);
}

TEST_CASE("bbq: every role has a name and a GGUF source pattern except Count", "[backbonequantsidecar]") {
    for (int r = 0; r < bbq::kRoleCount; ++r) {
        const auto role = static_cast<bbq::Role>(r);
        INFO("role index " << r);
        REQUIRE(bbq::role_pattern(role) != nullptr);
        REQUIRE(std::string(bbq::role_name(role)) != "?");
    }
    // Non-per-layer roles are exactly the model-level tensors named in this header's own comment.
    REQUIRE_FALSE(bbq::role_per_layer(bbq::Role::TokEmb));
    REQUIRE_FALSE(bbq::role_per_layer(bbq::Role::LmHead));
    REQUIRE_FALSE(bbq::role_per_layer(bbq::Role::GrExitDown));
    REQUIRE_FALSE(bbq::role_per_layer(bbq::Role::GrExitUp));
    REQUIRE(bbq::role_per_layer(bbq::Role::GrAttnDown));
    REQUIRE(bbq::role_per_layer(bbq::Role::QsaKProj));
}

TEST_CASE("bbq: role_pattern reads from transplant::recipe_for, not a second hand-copied table",
          "[backbonequantsidecar]") {
    using transplant::Dest;
    using transplant::recipe_for;
    CHECK(std::string(bbq::role_pattern(bbq::Role::TokEmb)) == recipe_for(Dest::TokEmb).src);
    CHECK(std::string(bbq::role_pattern(bbq::Role::LmHead)) == recipe_for(Dest::LmHead).src);
    CHECK(std::string(bbq::role_pattern(bbq::Role::GrAttnDown)) == recipe_for(Dest::GrAttnDown).src);
    CHECK(std::string(bbq::role_pattern(bbq::Role::MoeSharedDown)) == recipe_for(Dest::MoeSharedDown).src);
    // QsaQGateProj deliberately reads Dest::QsaQProj's source -- Dest::QsaGateProj shares the identical
    // pattern (both are PerHeadHalf slices of the one GGUF tensor), so either is a valid key.
    CHECK(std::string(bbq::role_pattern(bbq::Role::QsaQGateProj)) == recipe_for(Dest::QsaQProj).src);
    CHECK(std::string(bbq::role_pattern(bbq::Role::QsaQGateProj)) == recipe_for(Dest::QsaGateProj).src);
    // O5 phase 2b-2b: the three recovered GDN roles.
    CHECK(std::string(bbq::role_pattern(bbq::Role::GdnInProjQkv)) == recipe_for(Dest::GdnInProjQkv).src);
    CHECK(std::string(bbq::role_pattern(bbq::Role::GdnInProjZ)) == recipe_for(Dest::GdnInProjZ).src);
    CHECK(std::string(bbq::role_pattern(bbq::Role::GdnOutProj)) == recipe_for(Dest::GdnOutProj).src);
}

TEST_CASE("bbq: no role's transplant Dest carries a Fold -- every included tensor is storable verbatim",
          "[backbonequantsidecar]") {
    // The header comment's own claim, checked mechanically rather than left as prose: none of the roles
    // this format actually stores need a value transform (RMSNorm 1+w / ssm_a's -exp(A_log)) applied.
    // The three GDN roles DO carry a VPerm (that is the whole subject of this pass), but VPerm and Fold
    // are independent transforms (transplant.hpp's own two enums) -- this case is only about Fold.
    using transplant::Dest;
    using transplant::fold_for;
    using transplant::Fold;
    const Dest dests[] = {Dest::TokEmb,        Dest::LmHead,        Dest::GrAttnDown,   Dest::GrAttnUp,
                          Dest::GrFfnDown,     Dest::GrFfnUp,       Dest::GrExitDown,   Dest::GrExitUp,
                          Dest::MoeSharedGate, Dest::MoeSharedUp,   Dest::MoeSharedDown,
                          Dest::QsaQProj,      Dest::QsaGateProj,   Dest::QsaKProj,     Dest::QsaVProj,
                          Dest::QsaOProj,      Dest::GdnInProjQkv,  Dest::GdnInProjZ,   Dest::GdnOutProj};
    for (Dest d : dests) CHECK(fold_for(d) == Fold::None);
}

// O5 phase 2b-2b: gdn_row_permutation/gdn_out_gather_index must agree with transplant::ungroup_v_heads --
// the EXISTING, already-vetted reference this design re-derives from (AGENTS.md S5), not merely with
// themselves. Both helpers are cross-checked here against ungroup_v_heads run on identity-tagged synthetic
// data, at a small multi-k-head fixture (num_k_heads=2, num_v_heads=4, so rep=2 and the permutation is
// genuinely non-trivial -- the same "num_k_heads==1 is the identity, so it proves nothing" trap
// transplant.hpp's own header comment names for ungroup_v_heads itself).
TEST_CASE("bbq: gdn_row_permutation matches transplant::ungroup_v_heads's own Cols-axis mapping",
          "[backbonequantsidecar]") {
    constexpr int num_k_heads = 2, num_v_heads = 4, head_k_dim = 2;
    constexpr std::uint32_t head_v_dim = 3;
    constexpr std::uint32_t base = 2 * num_k_heads * head_k_dim;   // GdnInProjQkv's own "after Q|K" base
    constexpr std::uint32_t value_dim = num_v_heads * head_v_dim;
    constexpr std::uint32_t out_f = base + value_dim;

    // Identity-tagged "row vector": raw_ids[i] = i, laid out as a [1, out_f] array so ungroup_v_heads's
    // own Cols-axis permutation (a per-COLUMN reorder of a single row) is exactly the row-index mapping
    // gdn_row_permutation computes -- no reinterpretation needed to compare them.
    std::vector<float> raw_ids(out_f);
    for (std::uint32_t i = 0; i < out_f; ++i) raw_ids[i] = static_cast<float>(i);
    std::vector<float> dst_ids(out_f, -1.f);
    const transplant::VPerm vp{transplant::VAxis::Cols, /*after_keys=*/true, /*wide=*/true};
    transplant::ungroup_v_heads(raw_ids.data(), 1, static_cast<int>(out_f), vp, num_k_heads, num_v_heads,
                                head_k_dim, static_cast<int>(head_v_dim), dst_ids.data());

    std::vector<std::uint32_t> row_source(out_f, 0);
    bbq::gdn_row_permutation(out_f, base, num_k_heads, num_v_heads, head_v_dim, row_source);

    for (std::uint32_t d = 0; d < out_f; ++d) {
        INFO("destination index " << d);
        CHECK(row_source[d] == static_cast<std::uint32_t>(dst_ids[d]));
    }
    // Outside the V block: identity, both by ungroup_v_heads's own "copy through unchanged" contract and
    // by row_source's own definition.
    for (std::uint32_t d = 0; d < base; ++d) CHECK(row_source[d] == d);
    // Inside the V block: a genuine permutation, not the identity -- num_k_heads=2/num_v_heads=4 means
    // rep=2, so head 1 (dst_head=1) maps to src_head = 0*2+1 = ... actually verified structurally below:
    // every value in [base, out_f) appears in row_source exactly once (it is a bijection of that range).
    std::vector<bool> seen(value_dim, false);
    for (std::uint32_t d = base; d < out_f; ++d) {
        const std::uint32_t s = row_source[d];
        REQUIRE(s >= base);
        REQUIRE(s < out_f);
        REQUIRE_FALSE(seen[s - base]);
        seen[s - base] = true;
    }
    // And it is NOT the identity at this fixture (rep > 1) -- a real cross-check, not a vacuous one.
    bool any_moved = false;
    for (std::uint32_t d = base; d < out_f; ++d) any_moved |= (row_source[d] != d);
    CHECK(any_moved);
}

TEST_CASE("bbq: gdn_out_gather_index is the functional inverse of ungroup_v_heads's own Rows-axis mapping "
          "-- the direction that matters (AGENTS.md: getting it backwards runs and is wrong)",
          "[backbonequantsidecar]") {
    constexpr int num_k_heads = 2, num_v_heads = 4;
    constexpr std::uint32_t head_v_dim = 3;
    constexpr std::uint32_t value_dim = num_v_heads * head_v_dim;

    // dst_ids[d_idx] = the RAW/GGUF row index that supplies HF row d_idx -- ungroup_v_heads's own Rows-
    // axis semantics (one column, so dst[d_idx] = src[s_idx] directly).
    std::vector<float> raw_ids(value_dim);
    for (std::uint32_t i = 0; i < value_dim; ++i) raw_ids[i] = static_cast<float>(i);
    std::vector<float> dst_ids(value_dim, -1.f);
    const transplant::VPerm vp{transplant::VAxis::Rows, /*after_keys=*/false, /*wide=*/true};
    transplant::ungroup_v_heads(raw_ids.data(), static_cast<int>(value_dim), 1, vp, num_k_heads, num_v_heads,
                                /*head_k_dim=*/1, static_cast<int>(head_v_dim), dst_ids.data());

    // The functional inverse: inv[s_idx] = d_idx such that dst_ids[d_idx] == s_idx. ungroup_v_heads's own
    // mapping is a bijection over [0, value_dim) (a head permutation), so every s_idx has exactly one d_idx.
    std::vector<std::uint32_t> inv(value_dim, 0);
    std::vector<bool> covered(value_dim, false);
    for (std::uint32_t d = 0; d < value_dim; ++d) {
        const auto s = static_cast<std::uint32_t>(dst_ids[d]);
        REQUIRE(s < value_dim);
        REQUIRE_FALSE(covered[s]);
        covered[s] = true;
        inv[s] = d;
    }

    std::vector<std::uint32_t> idx(value_dim, 0);
    bbq::gdn_out_gather_index(num_k_heads, num_v_heads, head_v_dim, idx);
    for (std::uint32_t i = 0; i < value_dim; ++i) {
        INFO("raw GGUF position " << i);
        CHECK(idx[i] == inv[i]);
    }
    bool any_moved = false;
    for (std::uint32_t i = 0; i < value_dim; ++i) any_moved |= (idx[i] != i);
    CHECK(any_moved);
}

TEST_CASE("bbq: a written sidecar reads back, decodes bit-identically, and Store::find resolves the "
          "sparse (role, layer) index correctly, including a real gap",
          "[backbonequantsidecar]") {
    const std::string path = temp_path("sub0_bbq_roundtrip.bin");
    const Built built = build_sidecar(path, 999);
    {
    bbq::Store store;
    std::string err;
    REQUIRE(store.open(path, err, 999));
    REQUIRE(err.empty());
    REQUIRE(store.loaded());
    REQUIRE(store.header().n_tensors == built.entries.size());

    for (const auto& [role, layer, bytes] : built.entries) {
        const bbq::Desc* d = store.find(role, layer);
        REQUIRE(d != nullptr);
        REQUIRE(d->in_f == kIn);
        REQUIRE(d->out_f == kOut);
        const auto raw = store.raw(*d);
        REQUIRE(raw.size() == bytes.size());
        REQUIRE(std::memcmp(raw.data(), bytes.data(), bytes.size()) == 0);

        std::vector<float> decoded, reference;
        REQUIRE(bbq::dequantize_role_to_f32(*d, raw, decoded));
        gguf::TensorInfo t;
        t.type_raw = d->type_raw;
        t.dims = {static_cast<std::uint64_t>(d->in_f) * d->out_f};
        REQUIRE(gguf::to_f32(t, std::span<const std::uint8_t>(bytes), reference));
        REQUIRE(decoded.size() == reference.size());
        REQUIRE(std::memcmp(decoded.data(), reference.data(), decoded.size() * sizeof(float)) == 0);
        std::array<float, kIn> gathered{};
        REQUIRE(bbq::dequantize_row(*d, raw, 2, gathered));
        REQUIRE(std::memcmp(gathered.data(), reference.data() + 2 * kIn,
                            gathered.size() * sizeof(float)) == 0);
        REQUIRE_FALSE(bbq::dequantize_row(*d, raw, kOut, gathered));
    }

    // The deliberate gap: GrAttnUp exists at layer 0 but NOT layer 1 in this fixture.
    REQUIRE(store.find(bbq::Role::GrAttnUp, 0) != nullptr);
    REQUIRE(store.find(bbq::Role::GrAttnUp, 1) == nullptr);
    // A role never written at all.
    REQUIRE(store.find(bbq::Role::LmHead, -1) == nullptr);
    REQUIRE(store.find(bbq::Role::QsaKProj, 0) == nullptr);   // written only at layer 2
    REQUIRE(store.find(bbq::Role::QsaKProj, 2) != nullptr);
    }
    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("bbq: Store refuses a truncated, foreign, version-mismatched or paired-wrong file rather than "
          "reading garbage", "[backbonequantsidecar]") {
    const std::string path = temp_path("sub0_bbq_bad.bin");
    bbq::Store store;
    std::string err;

    {   // not an S0B1 file at all
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        const char junk[64] = {'N', 'O', 'P', 'E'};
        os.write(junk, sizeof junk);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());

    {   // right magic, header truncated
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        os.write("S0B1", 4);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));

    {   // valid header, descriptor table truncated
        const Built b = build_sidecar(path, 42);
        std::filesystem::resize_file(path, sizeof(bbq::Header) + 4);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));

    {   // valid header + table, payload missing
        const Built b = build_sidecar(path, 42);
        std::filesystem::resize_file(path, sizeof(bbq::Header) + b.entries.size() * sizeof(bbq::Desc) + 4);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));

    {   // version bump: an old reader must refuse a file from a hypothetical newer writer, not
        // silently reinterpret its bytes under the old layout (AGENTS.md S3).
        build_sidecar(path, 42);
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        std::uint32_t version = 2;
        f.seekp(4);
        f.write(reinterpret_cast<const char*>(&version), sizeof version);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));

    {   // model_param_floats mismatch: this sidecar belongs to a DIFFERENT .bin than the caller expects.
        build_sidecar(path, 42);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err, /*expect_model_param_floats=*/43));
    err.clear();
    REQUIRE(store.open(path, err, /*expect_model_param_floats=*/42));   // the matching value succeeds
    err.clear();
    REQUIRE(store.open(path, err));   // 0 (the default) means "do not check" -- also succeeds

    REQUIRE_FALSE(store.open(temp_path("sub0_bbq_does_not_exist.bin"), err));
    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("bbq: Store refuses a descriptor with an out-of-range role or a duplicate (role, layer) pair",
          "[backbonequantsidecar]") {
    const std::string path = temp_path("sub0_bbq_dup.bin");

    {   // out-of-range role
        std::vector<bbq::Desc> descs(1);
        std::vector<std::uint8_t> bytes = make_q8_0(kIn, 1);
        descs[0] = bbq::Desc{bbq::kRoleCount, -1, static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                             kIn, 1, 0, 0, bytes.size()};
        bbq::Header h;
        h.n_layers = 1;
        h.n_tensors = 1;
        h.data_off = sizeof(bbq::Header) + sizeof(bbq::Desc);
        h.data_bytes = bytes.size();
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        os.write(reinterpret_cast<const char*>(&h), sizeof h);
        os.write(reinterpret_cast<const char*>(descs.data()), sizeof(bbq::Desc));
        os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    bbq::Store store;
    std::string err;
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());

    {   // duplicate (role, layer)
        std::vector<bbq::Desc> descs(2);
        std::vector<std::uint8_t> a = make_q8_0(kIn, 1), b = make_q8_0(kIn, 2);
        descs[0] = bbq::Desc{static_cast<std::int32_t>(bbq::Role::TokEmb), -1,
                             static_cast<std::uint32_t>(gguf::TensorType::Q8_0), kIn, 1, 0, 0, a.size()};
        descs[1] = bbq::Desc{static_cast<std::int32_t>(bbq::Role::TokEmb), -1,
                             static_cast<std::uint32_t>(gguf::TensorType::Q8_0), kIn, 1, 0, a.size(),
                             b.size()};
        bbq::Header h;
        h.n_layers = 1;
        h.n_tensors = 2;
        h.data_off = sizeof(bbq::Header) + 2 * sizeof(bbq::Desc);
        h.data_bytes = a.size() + b.size();
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        os.write(reinterpret_cast<const char*>(&h), sizeof h);
        os.write(reinterpret_cast<const char*>(descs.data()), 2 * sizeof(bbq::Desc));
        os.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.size()));
        os.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());

    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("bbq: role_key distinguishes model-level (-1) from every real layer", "[backbonequantsidecar]") {
    const auto k1 = bbq::role_key(bbq::Role::TokEmb, -1);
    const auto k2 = bbq::role_key(bbq::Role::TokEmb, 0);
    const auto k3 = bbq::role_key(bbq::Role::GrAttnDown, -1);
    CHECK(k1 != k2);
    CHECK(k1 != k3);
    CHECK(k2 != k3);
}

TEST_CASE("bbq: Q4_K row gather matches independent GGUF dequantization", "[backbonequantsidecar]") {
    constexpr std::uint32_t width = 256, rows = 2;
    std::vector<std::uint8_t> raw(rows * 144);
    std::mt19937 rng(91);
    for (auto& byte : raw) byte = static_cast<std::uint8_t>(rng());
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint16_t one = 0x3c00;
        std::memcpy(raw.data() + row * 144, &one, sizeof one);
        std::memcpy(raw.data() + row * 144 + 2, &one, sizeof one);
    }
    const bbq::Desc d{static_cast<std::int32_t>(bbq::Role::TokEmb), -1,
                      static_cast<std::uint32_t>(gguf::TensorType::Q4_K), width, rows,
                      0, 0, raw.size()};
    gguf::TensorInfo t;
    t.type_raw = d.type_raw;
    t.dims = {static_cast<std::uint64_t>(width) * rows};
    std::vector<float> reference;
    REQUIRE(gguf::to_f32(t, raw, reference));
    std::array<float, width> gathered{};
    REQUIRE(bbq::dequantize_row(d, raw, 1, gathered));
    for (std::uint32_t col = 0; col < width; ++col)
        CHECK(gathered[col] == Catch::Approx(reference[width + col]).epsilon(1e-6));
}

TEST_CASE("bbq: exact model and sidecar contents are bound together", "[backbonequantsidecar]") {
    const std::string model = temp_path("sub0_bbq_pair_model.bin");
    const std::string sidecar = temp_path("sub0_bbq_pair_sidecar.bbq");
    auto put = [](const std::string& path, const char* data) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(data, 4);
    };
    put(model, "abcd");
    put(sidecar, "1234");
    std::string err;
    REQUIRE_FALSE(bbq::verify_pair_identity(model, sidecar, err));
    REQUIRE(bbq::write_pair_identity(model, sidecar, err));
    REQUIRE(bbq::verify_pair_identity(model, sidecar, err));

    put(model, "abce");   // same size and layout, different weights
    REQUIRE_FALSE(bbq::verify_pair_identity(model, sidecar, err));
    put(model, "abcd");
    REQUIRE(bbq::verify_pair_identity(model, sidecar, err));
    put(sidecar, "1235");
    REQUIRE_FALSE(bbq::verify_pair_identity(model, sidecar, err));

    REQUIRE(std::filesystem::remove(model));
    REQUIRE(std::filesystem::remove(sidecar));
    REQUIRE(std::filesystem::remove(sidecar + ".pair"));
}

TEST_CASE("bbq: Store rejects forged table sizes, overlapping payloads and invalid planes",
          "[backbonequantsidecar]") {
    const std::string path = temp_path("sub0_bbq_geometry.bin");
    bbq::Store store;
    std::string err;
    auto patch_header = [&](auto change) {
        build_sidecar(path, 42);
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        bbq::Header h;
        f.read(reinterpret_cast<char*>(&h), sizeof h);
        change(h);
        f.seekp(0);
        f.write(reinterpret_cast<const char*>(&h), sizeof h);
    };
    patch_header([](bbq::Header& h) { h.n_tensors = std::numeric_limits<std::uint64_t>::max(); });
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());
    patch_header([](bbq::Header& h) { h.data_off = sizeof(bbq::Header); });
    err.clear();
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());

    build_sidecar(path, 42);
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        bbq::Desc d;
        f.seekg(sizeof(bbq::Header));
        f.read(reinterpret_cast<char*>(&d), sizeof d);
        d.bytes = 1;
        f.seekp(sizeof(bbq::Header));
        f.write(reinterpret_cast<const char*>(&d), sizeof d);
    }
    err.clear();
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());
    REQUIRE(std::filesystem::remove(path));
}
