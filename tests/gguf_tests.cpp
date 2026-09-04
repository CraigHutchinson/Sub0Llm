// gguf_tests.cpp -- unit tests for sub0::gguf::Reader + check_llama_mha_compat, engine-free (no
// generated config, no real GGUF file needed): every fixture below is a handcrafted in-memory byte
// buffer built by GgufBuilder, mirroring exactly what a real GGUF writer would emit for the cases
// that matter to the import-compatibility question (plain MHA vs GQA, plain vs gated/SwiGLU FFN,
// quantized vs float tensors, and basic KV-metadata round-tripping).

#include "sub0/gguf.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <span>

using namespace sub0::gguf;

namespace {

// Builds a valid GGUF v3 byte buffer incrementally. Mirrors the Reader's own wire format 1:1 so a
// bug shared by both builder and reader wouldn't be caught -- kept deliberately separate/dumb (no
// helper functions reused from gguf.hpp) so it acts as an independent cross-check.
class GgufBuilder {
public:
    GgufBuilder() {
        push("GGUF", 4);
        push_pod<std::uint32_t>(3);          // version
        tensor_count_pos_ = buf_.size(); push_pod<std::uint64_t>(0);   // patched in finish()
        kv_count_pos_     = buf_.size(); push_pod<std::uint64_t>(0);   // patched in finish()
    }

    void add_string_kv(const std::string& key, const std::string& value) {
        push_string(key); push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::String));
        push_string(value); ++kv_count_;
    }
    void add_u32_kv(const std::string& key, std::uint32_t value) {
        push_string(key); push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::UInt32));
        push_pod(value); ++kv_count_;
    }
    void add_f32_kv(const std::string& key, float value) {
        push_string(key); push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::Float32));
        push_pod(value); ++kv_count_;
    }
    void add_string_array_kv(const std::string& key, const std::vector<std::string>& values) {
        push_string(key); push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::Array));
        push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::String));
        push_pod<std::uint64_t>(values.size());
        for (const auto& v : values) push_string(v);
        ++kv_count_;
    }
    void add_u32_array_kv(const std::string& key, const std::vector<std::uint32_t>& values) {
        push_string(key); push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::Array));
        push_pod<std::uint32_t>(static_cast<std::uint32_t>(ValueType::UInt32));
        push_pod<std::uint64_t>(values.size());
        for (auto v : values) push_pod(v);
        ++kv_count_;
    }

    // dims in GGML order (fastest-varying first); type_raw 0=F32, 1=F16, anything else = "quantized".
    void add_tensor(const std::string& name, std::vector<std::uint64_t> dims,
                    std::uint32_t type_raw = 0, std::uint64_t offset = 0) {
        push_string(name);
        push_pod<std::uint32_t>(static_cast<std::uint32_t>(dims.size()));
        for (auto d : dims) push_pod(d);
        push_pod(type_raw);
        push_pod(offset);
        ++tensor_count_;
    }

    std::vector<std::uint8_t> finish() {
        std::memcpy(buf_.data() + tensor_count_pos_, &tensor_count_, sizeof(tensor_count_));
        std::memcpy(buf_.data() + kv_count_pos_, &kv_count_, sizeof(kv_count_));
        return buf_;
    }

    // Truncates the finished buffer to `n` bytes -- for the "reject a cut-off file" tests.
    static std::vector<std::uint8_t> truncate(std::vector<std::uint8_t> full, std::size_t n) {
        full.resize(std::min(n, full.size()));
        return full;
    }

    // Pad up to the next multiple of `align` (default 32, matching the Reader's default when no
    // general.alignment KV is set) -- mirrors where the Reader computes the data section to start.
    // Call once, after the last add_tensor(), before add_data(). NOT part of finish()'s job: a file
    // with no tensor data (the metadata-only tests above) has no data section to align to.
    void pad_to_data_section(std::uint64_t align = 32) {
        const std::uint64_t rem = buf_.size() % align;
        if (rem != 0) buf_.insert(buf_.end(), align - rem, std::uint8_t{0});
    }
    // Appends raw tensor data bytes. Caller is responsible for pad_to_data_section() first and for
    // using offsets (in add_tensor's `offset` param) consistent with the append order here.
    void add_data(std::span<const std::uint8_t> bytes) { buf_.insert(buf_.end(), bytes.begin(), bytes.end()); }

private:
    template <class T> void push_pod(T v) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        buf_.insert(buf_.end(), p, p + sizeof(T));
    }
    void push(const char* s, std::size_t n) { buf_.insert(buf_.end(), s, s + n); }
    void push_string(const std::string& s) {
        push_pod<std::uint64_t>(s.size());
        buf_.insert(buf_.end(), s.begin(), s.end());
    }

    std::vector<std::uint8_t> buf_;
    std::size_t tensor_count_pos_ = 0, kv_count_pos_ = 0;
    std::uint64_t tensor_count_ = 0, kv_count_ = 0;
};

// A minimal but complete plain-MHA, 2-layer Llama-style tensor set (matches llama.cpp's naming),
// with `gated`/`biased`/`gqa`/`quantized` knobs so one builder covers every compat-report scenario.
std::vector<std::uint8_t> build_llama_fixture(bool gated, bool biased, bool gqa, bool quantized) {
    GgufBuilder b;
    b.add_string_kv("general.architecture", "llama");
    b.add_u32_kv("llama.block_count", 2);
    b.add_u32_kv("llama.embedding_length", 8);
    b.add_string_array_kv("tokenizer.ggml.tokens", {"<unk>", "<s>", "</s>", "hi", "there"});

    const std::uint32_t ttype = quantized ? 2u /* arbitrary non-F32/F16 id, e.g. Q4_0 */ : 0u /* F32 */;
    const std::uint64_t d = 8, kv_d = gqa ? 4 : 8;   // gqa: attn_k/v rows < attn_q rows

    b.add_tensor("token_embd.weight", {d, 16}, ttype);
    for (int l = 0; l < 2; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        b.add_tensor(p + "attn_norm.weight", {d}, ttype);
        b.add_tensor(p + "attn_q.weight", {d, d}, ttype);
        b.add_tensor(p + "attn_k.weight", {d, kv_d}, ttype);
        b.add_tensor(p + "attn_v.weight", {d, kv_d}, ttype);
        b.add_tensor(p + "attn_output.weight", {d, d}, ttype);
        if (biased) {
            b.add_tensor(p + "attn_q.bias", {d}, ttype);
            b.add_tensor(p + "ffn_down.bias", {d}, ttype);
        }
        b.add_tensor(p + "ffn_norm.weight", {d}, ttype);
        if (gated) {
            b.add_tensor(p + "ffn_gate.weight", {d, 32}, ttype);
            b.add_tensor(p + "ffn_up.weight", {d, 32}, ttype);
            b.add_tensor(p + "ffn_down.weight", {32, d}, ttype);
        } else {
            b.add_tensor(p + "ffn_up.weight", {d, 32}, ttype);      // plain 2-matrix FFN still has one "up" matrix
            b.add_tensor(p + "ffn_down.weight", {32, d}, ttype);
        }
    }
    b.add_tensor("output_norm.weight", {d}, ttype);
    b.add_tensor("output.weight", {d, 16}, ttype);
    return b.finish();
}

}  // namespace

TEST_CASE("gguf: rejects bad magic", "[gguf]") {
    std::vector<std::uint8_t> buf = {'B', 'A', 'D', '!', 3, 0, 0, 0};
    Reader r(buf);
    CHECK_FALSE(r.ok());
    CHECK(r.error() == Reader::Err::BadMagic);
}

TEST_CASE("gguf: rejects unsupported version", "[gguf]") {
    GgufBuilder b;   // constructor already wrote magic + version(3) + zeroed counts
    auto buf = b.finish();
    buf[4] = 99; buf[5] = 0; buf[6] = 0; buf[7] = 0;   // stomp version to 99
    Reader r(buf);
    CHECK_FALSE(r.ok());
    CHECK(r.error() == Reader::Err::UnsupportedVersion);
}

TEST_CASE("gguf: rejects a truncated file at every cut point", "[gguf]") {
    auto full = build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/false, /*quantized=*/false);
    // Cutting anywhere strictly before the end must fail cleanly (Truncated), never read out of bounds
    // (ASan/UBSan-clean is the real assertion here; the explicit CHECKs are a coarse cross-check).
    for (std::size_t cut : {std::size_t{0}, std::size_t{4}, std::size_t{8}, full.size() / 3, full.size() / 2, full.size() - 1}) {
        const std::vector<std::uint8_t> gguf_buf = GgufBuilder::truncate(full, cut);
        Reader r(gguf_buf);
        CHECK_FALSE(r.ok());
    }
    Reader whole(full);
    CHECK(whole.ok());
}

TEST_CASE("gguf: parses scalar and string KV metadata", "[gguf]") {
    GgufBuilder b;
    b.add_string_kv("general.architecture", "llama");
    b.add_u32_kv("llama.block_count", 12);
    b.add_f32_kv("llama.rope.freq_base", 10000.0f);
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());

    const Value* arch = r.find("general.architecture");
    REQUIRE(arch != nullptr);
    CHECK(arch->kind == Value::Kind::String);
    CHECK(arch->s == "llama");

    const Value* blocks = r.find("llama.block_count");
    REQUIRE(blocks != nullptr);
    CHECK(blocks->kind == Value::Kind::UInt);
    CHECK(blocks->u == 12);

    const Value* theta = r.find("llama.rope.freq_base");
    REQUIRE(theta != nullptr);
    CHECK(theta->kind == Value::Kind::Float);
    CHECK(theta->f == 10000.0);

    CHECK(r.find("does.not.exist") == nullptr);
}

TEST_CASE("gguf: parses a string-array KV (the tokenizer vocab shape)", "[gguf]") {
    GgufBuilder b;
    b.add_string_array_kv("tokenizer.ggml.tokens", {"<unk>", "hello", "world"});
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const Value* toks = r.find("tokenizer.ggml.tokens");
    REQUIRE(toks != nullptr);
    CHECK(toks->kind == Value::Kind::StringArray);
    REQUIRE(toks->sa.size() == 3);
    CHECK(toks->sa[0] == "<unk>");
    CHECK(toks->sa[2] == "world");
}

TEST_CASE("gguf: parses a non-string array KV without materializing it", "[gguf]") {
    GgufBuilder b;
    b.add_u32_array_kv("tokenizer.ggml.token_type", {1, 1, 2, 1});
    // A KV after the array must still parse correctly -- proves the array payload was skipped by
    // the right byte count, not under/over-consumed.
    b.add_string_kv("general.architecture", "llama");
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const Value* tt = r.find("tokenizer.ggml.token_type");
    REQUIRE(tt != nullptr);
    CHECK(tt->kind == Value::Kind::OtherArray);
    CHECK(tt->array_len == 4);
    CHECK(r.find("general.architecture")->s == "llama");
}

TEST_CASE("gguf: parses the tensor table (name, dims, type, offset)", "[gguf]") {
    GgufBuilder b;
    b.add_tensor("token_embd.weight", {8, 16}, /*type=*/0, /*offset=*/0);
    b.add_tensor("blk.0.attn_q.weight", {8, 8}, /*type=*/1, /*offset=*/512);
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    REQUIRE(r.tensors().size() == 2);

    const TensorInfo* emb = r.find_tensor("token_embd.weight");
    REQUIRE(emb != nullptr);
    CHECK(emb->dims == std::vector<std::uint64_t>{8, 16});
    CHECK(emb->is_f32());
    CHECK(emb->element_count() == 128);

    const TensorInfo* q = r.find_tensor("blk.0.attn_q.weight");
    REQUIRE(q != nullptr);
    CHECK(q->is_f16());
    CHECK(q->offset == 512);

    CHECK(r.find_tensor("blk.99.does_not_exist") == nullptr);
}

TEST_CASE("gguf compat: a plain 2-layer MHA Llama-style fixture matches cleanly", "[gguf]") {
    const std::vector<std::uint8_t> gguf_buf = build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/false, /*quantized=*/false);
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK(rep.embedding_present);
    CHECK(rep.output_norm_present);
    CHECK(rep.lm_head_present);
    CHECK(rep.matches_plain_mha);
    CHECK_FALSE(rep.has_gated_ffn);
    CHECK_FALSE(rep.has_attn_bias);
    CHECK_FALSE(rep.any_quantized);
    CHECK(rep.n_layers_checked == 2);
    CHECK(rep.missing.empty());
}

TEST_CASE("gguf compat: GQA (fewer KV rows than Q rows) is detected", "[gguf]") {
    const std::vector<std::uint8_t> gguf_buf = build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/true, /*quantized=*/false);
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK_FALSE(rep.matches_plain_mha);
}

TEST_CASE("gguf compat: gated (SwiGLU) FFN tensors are detected", "[gguf]") {
    const std::vector<std::uint8_t> gguf_buf = build_llama_fixture(/*gated=*/true, /*biased=*/false, /*gqa=*/false, /*quantized=*/false);
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK(rep.has_gated_ffn);
}

TEST_CASE("gguf compat: attention/FFN biases are detected", "[gguf]") {
    const std::vector<std::uint8_t> gguf_buf = build_llama_fixture(/*gated=*/false, /*biased=*/true, /*gqa=*/false, /*quantized=*/false);
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK(rep.has_attn_bias);
    CHECK(rep.has_ffn_bias);
}

TEST_CASE("gguf compat: quantized tensor types are flagged, not silently accepted", "[gguf]") {
    const std::vector<std::uint8_t> gguf_buf = build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/false, /*quantized=*/true);
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK(rep.any_quantized);
}

TEST_CASE("gguf compat: a missing layer is reported, not silently ignored", "[gguf]") {
    GgufBuilder b;
    b.add_tensor("token_embd.weight", {8, 16});
    b.add_tensor("blk.0.attn_q.weight", {8, 8});
    b.add_tensor("blk.0.attn_k.weight", {8, 8});
    b.add_tensor("blk.0.attn_v.weight", {8, 8});
    b.add_tensor("blk.0.attn_output.weight", {8, 8});
    b.add_tensor("blk.0.attn_norm.weight", {8});
    b.add_tensor("blk.0.ffn_norm.weight", {8});
    b.add_tensor("blk.0.ffn_down.weight", {32, 8});
    // Layer 1 deliberately absent -- ask for 2 layers explicitly so the checker doesn't just stop early.
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r, /*expect_layers=*/2);
    CHECK_FALSE(rep.missing.empty());
}

// --- tensor data access + dequantization (step 2 of the import spike) -----------------------------

TEST_CASE("gguf: f16_to_f32 matches known IEEE-754 binary16 bit patterns", "[gguf]") {
    CHECK(f16_to_f32(0x3C00) == 1.0f);
    CHECK(f16_to_f32(0xC000) == -2.0f);
    CHECK(f16_to_f32(0x0000) == 0.0f);
    CHECK(f16_to_f32(0x4200) == 3.0f);
    CHECK(f16_to_f32(0x7BFF) == 65504.0f);              // largest finite half
    CHECK(f16_to_f32(0x0001) == 5.9604644775390625e-08f); // smallest subnormal
    CHECK(f16_to_f32(0x03FF) == Catch::Approx(6.097555161e-05).epsilon(1e-6)); // largest subnormal
    CHECK(std::isinf(f16_to_f32(0x7C00)));
    CHECK(f16_to_f32(0x7C00) > 0);
    CHECK(std::isinf(f16_to_f32(0xFC00)));
    CHECK(f16_to_f32(0xFC00) < 0);
}

TEST_CASE("gguf: data_offset aligns past the tensor table to a 32-byte boundary", "[gguf]") {
    GgufBuilder b;
    b.add_string_kv("general.architecture", "llama");   // a few KVs so the table isn't already aligned
    b.add_tensor("token_embd.weight", {4, 4}, /*type=*/0, /*offset=*/0);
    const auto buf = b.finish();
    Reader r(buf);
    REQUIRE(r.ok());
    CHECK(r.data_offset() % 32 == 0);
    CHECK(r.data_offset() >= buf.size());   // padding never moves it BEFORE the table's own end
}

TEST_CASE("gguf: tensor_bytes round-trips exact F32 data through to_f32", "[gguf]") {
    GgufBuilder b;
    const std::vector<float> src = {1.0f, -2.5f, 0.0f, 3.25f};
    b.add_tensor("w", {static_cast<std::uint64_t>(src.size())}, /*type=F32*/ 0, /*offset=*/0);
    b.pad_to_data_section();
    std::span<const std::uint8_t> raw_src(reinterpret_cast<const std::uint8_t*>(src.data()), src.size() * sizeof(float));
    b.add_data(raw_src);

    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    CHECK(r.tensor_byte_size(*t) == src.size() * sizeof(float));

    const auto raw = r.tensor_bytes(*t);
    REQUIRE(raw.size() == src.size() * sizeof(float));
    std::vector<float> decoded;
    REQUIRE(to_f32(*t, raw, decoded));
    REQUIRE(decoded.size() == src.size());
    for (std::size_t i = 0; i < src.size(); ++i) CHECK(decoded[i] == src[i]);
}

TEST_CASE("gguf: dequantize_q8_0 decodes a hand-built block exactly", "[gguf]") {
    // One block_q8_0: a little-endian f16 scale (2.0, bits 0x4000) then 32 int8 quants 0..31.
    // Expected value[i] = qs[i] * 2.0.
    std::vector<std::uint8_t> raw(34, 0);
    raw[0] = 0x00; raw[1] = 0x40;                        // f16 2.0, little-endian
    for (int i = 0; i < 32; ++i) raw[2 + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
    std::vector<float> out;
    REQUIRE(dequantize_q8_0(raw, 32, out));
    REQUIRE(out.size() == 32);
    for (int i = 0; i < 32; ++i) CHECK(out[static_cast<std::size_t>(i)] == static_cast<float>(i) * 2.0f);
}

TEST_CASE("gguf: dequantize_q8_0 handles a partial final block (n not a multiple of 32)", "[gguf]") {
    // Two blocks' worth of bytes (68), but only 40 elements requested -- the second block is only
    // half-consumed. Scale 1.0 (bits 0x3C00) in both blocks for simplicity; quants = index within block.
    std::vector<std::uint8_t> raw(68, 0);
    for (int b = 0; b < 2; ++b) {
        raw[static_cast<std::size_t>(b) * 34] = 0x00; raw[static_cast<std::size_t>(b) * 34 + 1] = 0x3C;
        for (int i = 0; i < 32; ++i) raw[static_cast<std::size_t>(b) * 34 + 2 + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
    }
    std::vector<float> out;
    REQUIRE(dequantize_q8_0(raw, 40, out));
    REQUIRE(out.size() == 40);
    for (int i = 0; i < 32; ++i) CHECK(out[static_cast<std::size_t>(i)] == static_cast<float>(i));
    for (int i = 0; i < 8; ++i) CHECK(out[32 + static_cast<std::size_t>(i)] == static_cast<float>(i));  // second block, first 8
}

TEST_CASE("gguf: dequantize_q8_0 rejects a raw buffer shorter than the declared element count needs", "[gguf]") {
    std::vector<std::uint8_t> raw(33, 0);   // one byte short of a full 34-byte block
    std::vector<float> out;
    CHECK_FALSE(dequantize_q8_0(raw, 32, out));
}

TEST_CASE("gguf: a real Q8_0 tensor round-trips through Reader::tensor_bytes + to_f32", "[gguf]") {
    // Two blocks (64 elements) of Q8_0 data for a [64] tensor, embedded via the same
    // pad_to_data_section()/add_data() path a real writer's alignment would produce.
    GgufBuilder b;
    b.add_tensor("w", {64}, /*type=Q8_0*/ 8, /*offset=*/0);
    b.pad_to_data_section();
    std::vector<std::uint8_t> data(68, 0);
    data[0] = 0x00; data[1] = 0x3C;             // block 0 scale 1.0
    data[34] = 0x00; data[35] = 0x40;           // block 1 scale 2.0
    for (int i = 0; i < 32; ++i) data[2 + static_cast<std::size_t>(i)]      = static_cast<std::uint8_t>(i);
    for (int i = 0; i < 32; ++i) data[36 + static_cast<std::size_t>(i)]     = static_cast<std::uint8_t>(i);
    b.add_data(data);

    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    CHECK(t->is_q8_0());
    const auto raw = r.tensor_bytes(*t);
    REQUIRE(raw.size() == 68);
    std::vector<float> decoded;
    REQUIRE(to_f32(*t, raw, decoded));
    REQUIRE(decoded.size() == 64);
    for (int i = 0; i < 32; ++i) CHECK(decoded[static_cast<std::size_t>(i)] == static_cast<float>(i) * 1.0f);
    for (int i = 0; i < 32; ++i) CHECK(decoded[32 + static_cast<std::size_t>(i)] == static_cast<float>(i) * 2.0f);
}

TEST_CASE("gguf: tensor_bytes returns empty on a truncated/out-of-bounds data section", "[gguf]") {
    GgufBuilder b;
    b.add_tensor("w", {64}, /*type=F32*/ 0, /*offset=*/0);   // needs 256 bytes, none appended
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    CHECK(r.tensor_bytes(*t).empty());
}

// --- WP4c: the seven formats the REAL Qwen3.8-Flash-Next UD-IQ1_S file uses ------------------------
//
// Every case below hand-builds one block, then states the expected floats as an INDEPENDENT
// derivation -- worked from the format's own bit layout in the test, not by calling the decoder's own
// helpers. That is the point: a test that re-uses k_scale_min() to predict what k_scale_min() will do
// proves only self-consistency (the same blind spot
// [[independent-reimplementation-catches-identity-swap-bugs]] names).
//
// These synthetic blocks prove the BIT LAYOUT. They do not prove real-world fidelity -- that came from
// decoding real tensors out of the actual 49.99GB shard (AGENTS.md S9: "synthetic fixtures prove the
// parser logic; they don't prove real-world byte-format fidelity"). See docs/WP4_SCOPE.md S3a-ter for
// those measured per-format distributions.

namespace {

// f16 bit patterns used repeatedly below, spelled out so no test depends on a helper to make its input.
constexpr std::uint16_t kF16One  = 0x3C00;   // 1.0
constexpr std::uint16_t kF16Half = 0x3800;   // 0.5
constexpr std::uint16_t kF16Zero = 0x0000;   // 0.0

void put_u16(std::vector<std::uint8_t>& v, std::size_t at, std::uint16_t x) {
    v[at] = static_cast<std::uint8_t>(x & 0xFF);
    v[at + 1] = static_cast<std::uint8_t>(x >> 8);
}
void put_u32(std::vector<std::uint8_t>& v, std::size_t at, std::uint32_t x) {
    for (int i = 0; i < 4; ++i) v[at + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((x >> (8 * i)) & 0xFF);
}

}  // namespace

TEST_CASE("gguf: block_spec sizes every real-file format, and tensor_byte_size follows it", "[gguf]") {
    // The nine type ids actually present in the real UD-IQ1_S file. Each figure was cross-checked
    // against that file by differencing consecutive tensors' data offsets (docs/WP4_SCOPE.md S3a-ter).
    struct Case { std::uint32_t type; std::uint64_t elems, bytes; };
    const Case cases[] = {
        {0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {8, 32, 34}, {12, 256, 144},
        {13, 256, 176}, {14, 256, 210}, {16, 256, 66}, {19, 256, 50}, {20, 32, 18},
    };
    for (const Case& c : cases) {
        const BlockSpec b = block_spec(c.type);
        INFO("type " << c.type);
        CHECK(b.elems == c.elems);
        CHECK(b.bytes == c.bytes);
    }
    // Anything else still reports "unknown" -- Q4_0 (2) and Q2_K (10) are real ggml formats this
    // reader deliberately does not implement, so they must stay refusals rather than mis-sized reads.
    CHECK(block_spec(2).elems == 0);
    CHECK(block_spec(10).elems == 0);

    GgufBuilder b;
    b.add_tensor("q4k", {512}, 12);      // 2 super-blocks
    b.add_tensor("iq1s", {256}, 19);
    b.add_tensor("bf16", {7}, 30);
    b.add_tensor("iq4nl", {64}, 20);
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    CHECK(r.tensor_byte_size(*r.find_tensor("q4k")) == 288);
    CHECK(r.tensor_byte_size(*r.find_tensor("iq1s")) == 50);
    CHECK(r.tensor_byte_size(*r.find_tensor("bf16")) == 14);
    CHECK(r.tensor_byte_size(*r.find_tensor("iq4nl")) == 36);
}

TEST_CASE("gguf: bf16_to_f32 widens, and is NOT the f16 path", "[gguf]") {
    CHECK(bf16_to_f32(0x3F80) == 1.0f);
    CHECK(bf16_to_f32(0xC000) == -2.0f);
    CHECK(bf16_to_f32(0x0000) == 0.0f);
    CHECK(bf16_to_f32(0x4049) == Catch::Approx(3.140625f));   // pi, truncated to bf16
    CHECK(bf16_to_f32(0x7F80) > 0);
    CHECK(std::isinf(bf16_to_f32(0x7F80)));
    // The two 16-bit formats share no bit pattern's meaning: 0x3C00 is 1.0 as an f16 and 0.0078125 as
    // a bf16. A decoder that routed a bf16 tensor through f16_to_f32 would silently rescale by 128x.
    CHECK(f16_to_f32(0x3C00) == 1.0f);
    CHECK(bf16_to_f32(0x3C00) == 0.0078125f);
}

TEST_CASE("gguf: a BF16 tensor round-trips through Reader::tensor_bytes + to_f32", "[gguf]") {
    GgufBuilder b;
    b.add_tensor("w", {4}, /*type=BF16*/ 30, 0);
    b.pad_to_data_section();
    std::vector<std::uint8_t> data(8, 0);
    put_u16(data, 0, 0x3F80); put_u16(data, 2, 0xC000);
    put_u16(data, 4, 0x0000); put_u16(data, 6, 0x3F00);
    b.add_data(data);
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    CHECK(t->is_bf16());
    CHECK(t->is_plain_float());   // bf16 is storage, not quantization
    std::vector<float> out;
    REQUIRE(to_f32(*t, r.tensor_bytes(*t), out));
    REQUIRE(out.size() == 4);
    CHECK(out[0] == 1.0f);
    CHECK(out[1] == -2.0f);
    CHECK(out[2] == 0.0f);
    CHECK(out[3] == 0.5f);
}

TEST_CASE("gguf: dequantize_q4_k decodes a hand-built super-block exactly", "[gguf]") {
    // d = 1.0, dmin = 0.5. The 12 packed scale bytes are chosen so BOTH branches of the 6-bit
    // scale/min unpack are exercised and both give values readable straight off the bytes:
    //   scales[0..3]  = 1,2,3,4    -> sub-blocks 0..3 scale = 1,2,3,4 (j < 4 branch: q[j] & 63)
    //   scales[4..7]  = 8,9,10,11  -> sub-blocks 0..3 min   = 8,9,10,11 (j < 4: q[j+4] & 63)
    //   scales[8..11] = 5,6,7,8    -> sub-blocks 4..7 scale = 5,6,7,8, min = 0,0,0,0
    // The j >= 4 branch is (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4) for the scale and
    // (q[j+4] >> 4) | ((q[j-0] >> 6) << 4) for the min; every byte above has its top two bits clear
    // and a value below 16, so those reduce to scales[j+4] and 0 -- deliberately, so the EXPECTED
    // values below are stated as plain numbers rather than re-derived by the same packing code.
    std::vector<std::uint8_t> raw(144, 0);
    put_u16(raw, 0, kF16One);    // d
    put_u16(raw, 2, kF16Half);   // dmin
    const std::uint8_t scale_bytes[12] = {1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 7, 8};
    for (int i = 0; i < 12; ++i) raw[4 + static_cast<std::size_t>(i)] = scale_bytes[i];
    // qs[l] packs element l's low nibble and element l+32's high nibble (within each 64-group).
    for (int i = 0; i < 128; ++i)
        raw[16 + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((i % 16) | ((15 - (i % 16)) << 4));

    std::vector<float> out;
    REQUIRE(dequantize_q4_k(raw, 256, out));
    REQUIRE(out.size() == 256);

    const float sc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float mn[8] = {8, 9, 10, 11, 0, 0, 0, 0};
    for (int group = 0; group < 4; ++group) {          // four 64-element groups
        const int sub_lo = group * 2, sub_hi = sub_lo + 1;
        for (int l = 0; l < 32; ++l) {
            const int byte = group * 32 + l;
            const float lo = static_cast<float>(byte % 16);
            const float hi = static_cast<float>(15 - (byte % 16));
            INFO("group " << group << " l " << l);
            CHECK(out[static_cast<std::size_t>(group * 64 + l)] ==
                  Catch::Approx(1.0f * sc[sub_lo] * lo - 0.5f * mn[sub_lo]));
            CHECK(out[static_cast<std::size_t>(group * 64 + 32 + l)] ==
                  Catch::Approx(1.0f * sc[sub_hi] * hi - 0.5f * mn[sub_hi]));
        }
    }
    // Presence check: the min term must actually be subtracted. Sub-blocks 0..3 have nonzero mins, so
    // element 0 (quant 0, scale 1, min 8) must be -4, NOT 0 -- a decoder that dropped `- m1` would
    // give exactly 0 here and still pass every shape and range assertion.
    CHECK(out[0] == Catch::Approx(-4.0f));
}

TEST_CASE("gguf: dequantize_q5_k adds the fifth (high) bit from qh", "[gguf]") {
    // Same construction as Q4_K, but with all mins zero (dmin = 0) so the ONLY thing under test is the
    // high-bit contribution: with qh all-ones every element gains +16 before scaling.
    auto build = [](std::uint8_t qh_fill) {
        std::vector<std::uint8_t> raw(176, 0);
        put_u16(raw, 0, kF16One);
        put_u16(raw, 2, kF16Zero);
        const std::uint8_t scale_bytes[12] = {1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8};
        for (int i = 0; i < 12; ++i) raw[4 + static_cast<std::size_t>(i)] = scale_bytes[i];
        for (int i = 0; i < 32; ++i) raw[16 + static_cast<std::size_t>(i)] = qh_fill;
        for (int i = 0; i < 128; ++i)
            raw[48 + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((i % 16) | ((15 - (i % 16)) << 4));
        return raw;
    };
    const float sc[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<float> lo_only, all_high;
    REQUIRE(dequantize_q5_k(build(0x00), 256, lo_only));
    REQUIRE(dequantize_q5_k(build(0xFF), 256, all_high));
    REQUIRE(lo_only.size() == 256);

    for (int group = 0; group < 4; ++group) {
        for (int l = 0; l < 32; ++l) {
            const int byte = group * 32 + l;
            const float q_lo = static_cast<float>(byte % 16);
            const float q_hi = static_cast<float>(15 - (byte % 16));
            const std::size_t i_lo = static_cast<std::size_t>(group * 64 + l);
            const std::size_t i_hi = static_cast<std::size_t>(group * 64 + 32 + l);
            INFO("group " << group << " l " << l);
            CHECK(lo_only[i_lo] == Catch::Approx(sc[group * 2] * q_lo));
            CHECK(lo_only[i_hi] == Catch::Approx(sc[group * 2 + 1] * q_hi));
            CHECK(all_high[i_lo] == Catch::Approx(sc[group * 2] * (q_lo + 16.f)));
            CHECK(all_high[i_hi] == Catch::Approx(sc[group * 2 + 1] * (q_hi + 16.f)));
        }
    }
    // The qh bit PAIR advances per 64-element group (u1 <<= 2 each iteration). Setting only bits 0/1
    // must therefore light up group 0 alone -- a decoder that failed to advance the mask would apply
    // the same +16 to every group, which the whole-tensor comparisons above cannot distinguish from
    // correct when qh is uniform.
    std::vector<float> group0_only;
    REQUIRE(dequantize_q5_k(build(0x03), 256, group0_only));
    CHECK(group0_only[0] == Catch::Approx(lo_only[0] + 1.f * 16.f));
    CHECK(group0_only[64] == Catch::Approx(lo_only[64]));    // group 1 untouched
    CHECK(group0_only[128] == Catch::Approx(lo_only[128]));  // group 2 untouched
    CHECK(group0_only[192] == Catch::Approx(lo_only[192]));  // group 3 untouched
}

TEST_CASE("gguf: dequantize_q6_k is zero-centered (q - 32) with int8 scales", "[gguf]") {
    // d = 1.0, every 16-element scale = 1, ql/qh all zero. Every quant is then 0, and (0 - 32) * 1
    // means the WHOLE super-block must decode to exactly -32. Q6_K has no min term at all, so a
    // decoder that copied Q4_K's affine form would produce zeros here instead.
    std::vector<std::uint8_t> raw(210, 0);
    for (int i = 0; i < 16; ++i) raw[192 + static_cast<std::size_t>(i)] = 1;
    put_u16(raw, 208, kF16One);
    std::vector<float> out;
    REQUIRE(dequantize_q6_k(raw, 256, out));
    REQUIRE(out.size() == 256);
    for (std::size_t i = 0; i < out.size(); ++i) { INFO("i " << i); CHECK(out[i] == -32.0f); }

    // Now light up ql[0]'s low nibble to 5 and qh[0]'s bits [1:0] to 3 -> q = 5 | (3 << 4) = 53,
    // so element 0 = (53 - 32) * 1 = 21. Element 32 reads ql[32]'s low nibble with qh[0] bits [3:2],
    // element 64 reads ql[0]'s HIGH nibble with bits [5:4], element 96 ql[32]'s high with [7:6] --
    // the interleaved strip order, which a linearized rewrite would scramble.
    raw[0] = 0x05;
    raw[128] = 0x03;
    REQUIRE(dequantize_q6_k(raw, 256, out));
    CHECK(out[0] == 21.0f);
    CHECK(out[32] == -32.0f);
    CHECK(out[64] == -32.0f);
    CHECK(out[96] == -32.0f);
    raw[128] = 0x30;                      // bits [5:4] = 3 -> feeds element 64 (ql[0] >> 4 == 0)
    REQUIRE(dequantize_q6_k(raw, 256, out));
    CHECK(out[0] == 5.0f - 32.0f);
    CHECK(out[64] == 48.0f - 32.0f);
    // The scale index also strides: sc[is + 0/2/4/6] with is = l/16, so element 16 uses scales[1].
    raw[193] = 2;
    REQUIRE(dequantize_q6_k(raw, 256, out));
    CHECK(out[16] == -64.0f);             // 2 * (0 - 32)
    CHECK(out[0] == 5.0f - 32.0f);        // still scales[0]
}

TEST_CASE("gguf: dequantize_iq4_nl indexes the 16-entry non-linear codebook", "[gguf]") {
    // The codebook is DATA, so pin two of its entries literally here rather than only using the
    // symbol -- a silently reordered table would otherwise be invisible to every test in this file.
    CHECK(KVALUES_IQ4NL[0] == -127);
    CHECK(KVALUES_IQ4NL[15] == 113);
    CHECK(KVALUES_IQ4NL[8] == 1);

    std::vector<std::uint8_t> raw(18, 0);
    put_u16(raw, 0, kF16Half);   // d = 0.5
    for (int j = 0; j < 16; ++j)
        raw[2 + static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(j | ((15 - j) << 4));
    std::vector<float> out;
    REQUIRE(dequantize_iq4_nl(raw, 32, out));
    REQUIRE(out.size() == 32);
    // Byte j holds element j (low nibble) and element j + 16 (high nibble) -- NOT elements 2j/2j+1.
    for (int j = 0; j < 16; ++j) {
        INFO("j " << j);
        CHECK(out[static_cast<std::size_t>(j)] == 0.5f * static_cast<float>(KVALUES_IQ4NL[j]));
        CHECK(out[static_cast<std::size_t>(j + 16)] == 0.5f * static_cast<float>(KVALUES_IQ4NL[15 - j]));
    }
    // Concretely, in plain numbers, so the split above is not asserted only against itself:
    CHECK(out[0] == 0.5f * -127.0f);
    CHECK(out[16] == 0.5f * 113.0f);
}

TEST_CASE("gguf: dequantize_iq2_xxs expands grid magnitudes with the sign codebook", "[gguf]") {
    // The first grid entry is 0x0808080808080808 -- eight magnitudes of 8. Sign index 0 in
    // ksigns_iq2xs is 0 (no negation). So with d = 1.0 and the scale nibble 0, the block's first 8
    // elements must be db * 8 where db = 1.0 * (0.5 + 0) * 0.25 = 0.125, i.e. exactly 1.0.
    CHECK(IQ2XXS_GRID[0] == 0x0808080808080808ull);
    CHECK(KSIGNS_IQ2XS[0] == 0);
    CHECK(KSIGNS_IQ2XS[1] == 129);   // 0b10000001: negate elements 0 and 7

    std::vector<std::uint8_t> raw(66, 0);
    put_u16(raw, 0, kF16One);
    // Group 0 of 8: aux[0]'s four bytes are the four grid indices; aux[1] carries four 7-bit sign
    // indices at [6:0], [13:7], [20:14], [27:21] and the 4-bit scale at [31:28].
    put_u32(raw, 2, 0x00000000u);                       // grid indices 0,0,0,0
    put_u32(raw, 6, 0x00000000u);                       // all sign index 0, scale nibble 0
    std::vector<float> out;
    REQUIRE(dequantize_iq2_xxs(raw, 256, out));
    REQUIRE(out.size() == 256);
    for (int j = 0; j < 32; ++j) { INFO("j " << j); CHECK(out[static_cast<std::size_t>(j)] == 1.0f); }

    // Sign index 1 on the FIRST 8-element group only: elements 0 and 7 flip, 1..6 do not, and the
    // second group (elements 8..15) is untouched -- which a decoder that applied one sign pattern to
    // the whole 32-element sub-block would get wrong while still producing plausible values.
    put_u32(raw, 6, 0x00000001u);
    REQUIRE(dequantize_iq2_xxs(raw, 256, out));
    CHECK(out[0] == -1.0f);
    CHECK(out[1] == 1.0f);
    CHECK(out[6] == 1.0f);
    CHECK(out[7] == -1.0f);
    CHECK(out[8] == 1.0f);

    // The 4-bit scale nibble: db = d * (0.5 + s) * 0.25, so s = 3 gives db = 0.875 and every
    // magnitude-8 element becomes 7.0.
    put_u32(raw, 6, 0x30000000u);
    REQUIRE(dequantize_iq2_xxs(raw, 256, out));
    CHECK(out[0] == Catch::Approx(7.0f));
}

TEST_CASE("gguf: dequantize_iq1_s decodes the ternary grid with its per-block delta", "[gguf]") {
    // Grid entry 0 is all-0xff == eight -1s; entry 1 is 0xffffffffffffff01, i.e. +1 in element 0 and
    // -1 in elements 1..7 (little-endian byte order -- byte j IS element j).
    CHECK(IQ1S_GRID[0] == 0xffffffffffffffffull);
    CHECK(IQ1S_GRID[1] == 0xffffffffffffff01ull);
    CHECK(IQ1S_DELTA == 0.125f);

    std::vector<std::uint8_t> raw(50, 0);
    put_u16(raw, 0, kF16One);              // d = 1.0
    // qs: 32 low bytes of the grid index. Leave them 0 -> grid entry 0 (all -1) everywhere.
    // qh[0]: scale field [14:12] = 0 -> dl = d * (2*0 + 1) = 1; sign bit 15 clear -> delta = +0.125.
    put_u16(raw, 34, 0x0000);
    std::vector<float> out;
    REQUIRE(dequantize_iq1_s(raw, 256, out));
    REQUIRE(out.size() == 256);
    for (int j = 0; j < 32; ++j) { INFO("j " << j); CHECK(out[static_cast<std::size_t>(j)] == Catch::Approx(-0.875f)); }

    // Flip the delta sign bit: -1 + (-0.125) = -1.125. A decoder that dropped the delta entirely
    // would report exactly -1.0 in both cases and pass a range check.
    put_u16(raw, 34, 0x8000);
    REQUIRE(dequantize_iq1_s(raw, 256, out));
    CHECK(out[0] == Catch::Approx(-1.125f));

    // Scale field = 3 -> dl = 2*3 + 1 = 7, so -1 + 0.125 scaled by 7 = -6.125.
    put_u16(raw, 34, 0x3000);
    REQUIRE(dequantize_iq1_s(raw, 256, out));
    CHECK(out[0] == Catch::Approx(-6.125f));

    // The grid index's HIGH three bits come from qh[ib] >> (3*l) for the l-th group of 8. Setting
    // qs[0] = 1 with those bits clear selects grid entry 1: +1 then seven -1s.
    put_u16(raw, 34, 0x0000);
    raw[2] = 1;
    REQUIRE(dequantize_iq1_s(raw, 256, out));
    CHECK(out[0] == Catch::Approx(1.125f));
    CHECK(out[1] == Catch::Approx(-0.875f));
    // ...and the high bits genuinely shift the index by 256 per unit: qh bits [2:0] = 1 with qs[0] = 1
    // selects entry 257, which differs from entry 1. Compared over the whole 8-element group, not on
    // element 0 alone -- entries 1 and 257 happen to share their first byte, so an element-0 check
    // would pass for a decoder that ignored the high bits entirely.
    const std::vector<float> low_index_group(out.begin(), out.begin() + 8);
    put_u16(raw, 34, 0x0001);
    REQUIRE(dequantize_iq1_s(raw, 256, out));
    REQUIRE(IQ1S_GRID[257] != IQ1S_GRID[1]);
    bool group_differs = false;
    for (int j = 0; j < 8; ++j) group_differs |= (out[static_cast<std::size_t>(j)] != low_index_group[static_cast<std::size_t>(j)]);
    CHECK(group_differs);
}

TEST_CASE("gguf: every new format refuses a short buffer instead of reading past it", "[gguf]") {
    std::vector<float> out;
    CHECK_FALSE(dequantize_q4_k(std::vector<std::uint8_t>(143, 0), 256, out));
    CHECK_FALSE(dequantize_q5_k(std::vector<std::uint8_t>(175, 0), 256, out));
    CHECK_FALSE(dequantize_q6_k(std::vector<std::uint8_t>(209, 0), 256, out));
    CHECK_FALSE(dequantize_iq2_xxs(std::vector<std::uint8_t>(65, 0), 256, out));
    CHECK_FALSE(dequantize_iq1_s(std::vector<std::uint8_t>(49, 0), 256, out));
    CHECK_FALSE(dequantize_iq4_nl(std::vector<std::uint8_t>(17, 0), 32, out));
}

TEST_CASE("gguf: a partial final block decodes only the elements that exist", "[gguf]") {
    // The real file's tensors are all whole multiples of their block size, but the reader must not
    // depend on that (dequantize_q8_0 already did not) -- and, more importantly, must never write
    // past `out`'s end when they are not.
    std::vector<std::uint8_t> raw(50, 0);
    put_u16(raw, 0, kF16One);
    std::vector<float> out;
    REQUIRE(dequantize_iq1_s(raw, 5, out));
    CHECK(out.size() == 5);

    std::vector<std::uint8_t> nl(18, 0);
    put_u16(nl, 0, kF16One);
    REQUIRE(dequantize_iq4_nl(nl, 20, out));
    CHECK(out.size() == 20);

    std::vector<std::uint8_t> q6(210, 0);
    put_u16(q6, 208, kF16One);
    REQUIRE(dequantize_q6_k(q6, 100, out));
    CHECK(out.size() == 100);
}

TEST_CASE("gguf: to_f32 refuses an unsupported quantized type rather than misreading it", "[gguf]") {
    GgufBuilder b;
    b.add_tensor("w", {32}, /*type=*/2 /* Q4_0, not implemented */, /*offset=*/0);
    b.pad_to_data_section();
    std::vector<std::uint8_t> junk(64, 0xAB);
    b.add_data(junk);
    const std::vector<std::uint8_t> gguf_buf = b.finish();
    Reader r(gguf_buf);
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    // tensor_byte_size returns 0 for an unrecognized type -> tensor_bytes is empty -> to_f32 on the
    // (empty) span also correctly fails, so a caller checking either return value is safe.
    CHECK(r.tensor_byte_size(*t) == 0);
    std::vector<float> out;
    CHECK_FALSE(to_f32(*t, r.tensor_bytes(*t), out));
}
