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
        Reader r(GgufBuilder::truncate(full, cut));
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
    Reader r(b.finish());
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
    Reader r(b.finish());
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
    Reader r(b.finish());
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
    Reader r(b.finish());
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
    Reader r(build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/false, /*quantized=*/false));
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
    Reader r(build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/true, /*quantized=*/false));
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK_FALSE(rep.matches_plain_mha);
}

TEST_CASE("gguf compat: gated (SwiGLU) FFN tensors are detected", "[gguf]") {
    Reader r(build_llama_fixture(/*gated=*/true, /*biased=*/false, /*gqa=*/false, /*quantized=*/false));
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK(rep.has_gated_ffn);
}

TEST_CASE("gguf compat: attention/FFN biases are detected", "[gguf]") {
    Reader r(build_llama_fixture(/*gated=*/false, /*biased=*/true, /*gqa=*/false, /*quantized=*/false));
    REQUIRE(r.ok());
    auto rep = check_llama_mha_compat(r);
    CHECK(rep.has_attn_bias);
    CHECK(rep.has_ffn_bias);
}

TEST_CASE("gguf compat: quantized tensor types are flagged, not silently accepted", "[gguf]") {
    Reader r(build_llama_fixture(/*gated=*/false, /*biased=*/false, /*gqa=*/false, /*quantized=*/true));
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
    Reader r(b.finish());
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

    Reader r(b.finish());
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

    Reader r(b.finish());
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
    Reader r(b.finish());
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    CHECK(r.tensor_bytes(*t).empty());
}

TEST_CASE("gguf: to_f32 refuses an unsupported quantized type rather than misreading it", "[gguf]") {
    GgufBuilder b;
    b.add_tensor("w", {32}, /*type=*/2 /* Q4_0, not implemented */, /*offset=*/0);
    b.pad_to_data_section();
    std::vector<std::uint8_t> junk(64, 0xAB);
    b.add_data(junk);
    Reader r(b.finish());
    REQUIRE(r.ok());
    const TensorInfo* t = r.find_tensor("w");
    REQUIRE(t != nullptr);
    // tensor_byte_size returns 0 for an unrecognized type -> tensor_bytes is empty -> to_f32 on the
    // (empty) span also correctly fails, so a caller checking either return value is safe.
    CHECK(r.tensor_byte_size(*t) == 0);
    std::vector<float> out;
    CHECK_FALSE(to_f32(*t, r.tensor_bytes(*t), out));
}
