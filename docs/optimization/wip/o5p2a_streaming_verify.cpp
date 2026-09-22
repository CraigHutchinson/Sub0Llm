// Verify the planned streaming AVX2 kernels (superblock-cached scale decode + vectorized nibble/high-bit
// unpack) are BIT-IDENTICAL to the existing group()-based gemv_rows<Plane,true> path, on both synthetic
// and real bytes, before porting into the real header. Not committed -- scratchpad only.
#include "sub0/backbone_quant_dot.hpp"
#include "sub0/gguf.hpp"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using namespace sub0;
namespace fs = std::filesystem;

namespace stream {

using bbqd::detail::dot32_avx2;
using bbqd::detail::hsum256_epi32;

inline std::int32_t dot32_avx2_reg(__m256i vw, const std::int8_t* x) {
    const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
    const __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw, 1));
    const __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
    const __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
    const __m256i p_lo = _mm256_madd_epi16(w_lo, x_lo);
    const __m256i p_hi = _mm256_madd_epi16(w_hi, x_hi);
    return hsum256_epi32(_mm256_add_epi32(p_lo, p_hi));
}

inline std::int32_t dot16_avx2_reg(__m128i vw, const std::int8_t* x) {
    const __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
    const __m128i w_lo = _mm_cvtepi8_epi16(vw);
    const __m128i x_lo = _mm_cvtepi8_epi16(vx);
    const __m128i w_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vw, 8));
    const __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    const __m128i p = _mm_add_epi32(_mm_madd_epi16(w_lo, x_lo), _mm_madd_epi16(w_hi, x_hi));
    __m128i s = _mm_add_epi32(p, _mm_shuffle_epi32(p, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

inline __m256i nibble_lo(__m256i bytes) { return _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F)); }
inline __m256i nibble_hi(__m256i bytes) {
    return _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));
}
inline __m256i qh_bit_to_hi4(__m256i qh, int bit_idx) {
    const __m256i bitmask = _mm256_set1_epi8(static_cast<char>(1 << bit_idx));
    const __m256i is_set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bitmask), bitmask);
    return _mm256_and_si256(is_set, _mm256_set1_epi8(16));
}
inline __m256i qh_2bits_to_hi4(__m256i qh, int shift) {
    const __m256i masked = _mm256_and_si256(qh, _mm256_set1_epi8(static_cast<char>(3 << shift)));
    switch (shift) {
        case 0: return _mm256_slli_epi16(masked, 4);
        case 2: return _mm256_slli_epi16(masked, 2);
        case 4: return masked;
        default: return _mm256_srli_epi16(masked, 2);
    }
}

struct KScaleTable {
    float d = 0.f, dmin = 0.f;
    std::array<float, 8> sc{}, m{};
    void load(const std::uint8_t* blk) {
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        d = gguf::f16_to_f32(d_bits);
        dmin = gguf::f16_to_f32(dmin_bits);
        const std::uint8_t* scales = blk + 4;
        for (int is = 0; is < 8; ++is) {
            std::uint8_t s = 0, mm = 0;
            gguf::k_scale_min(is, scales, s, mm);
            sc[static_cast<std::size_t>(is)] = d * static_cast<float>(s);
            m[static_cast<std::size_t>(is)] = -dmin * static_cast<float>(mm);
        }
    }
};

float dot_row_q4_k(const std::uint8_t* plane, std::uint64_t row_base, int row_elems, const bbqd::ActBlocks& x) {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 144;
        KScaleTable t; t.load(blk);
        const std::uint8_t* ql = blk + 16;
        const int g0 = s * 8;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const int gi_lo = g0 + sub_lo, gi_hi = g0 + sub_hi;
            const __m256i wlo = nibble_lo(raw), whi = nibble_hi(raw);
            const std::int32_t isum_lo = dot32_avx2_reg(wlo, x.qs.data() + static_cast<std::size_t>(gi_lo) * bbqd::GROUP);
            const std::int32_t isum_hi = dot32_avx2_reg(whi, x.qs.data() + static_cast<std::size_t>(gi_hi) * bbqd::GROUP);
            acc += x.scale[static_cast<std::size_t>(gi_lo)] *
                   (t.sc[static_cast<std::size_t>(sub_lo)] * static_cast<float>(isum_lo) +
                    t.m[static_cast<std::size_t>(sub_lo)] * static_cast<float>(x.gsum[static_cast<std::size_t>(gi_lo)]));
            acc += x.scale[static_cast<std::size_t>(gi_hi)] *
                   (t.sc[static_cast<std::size_t>(sub_hi)] * static_cast<float>(isum_hi) +
                    t.m[static_cast<std::size_t>(sub_hi)] * static_cast<float>(x.gsum[static_cast<std::size_t>(gi_hi)]));
        }
    }
    return acc;
}

float dot_row_q5_k(const std::uint8_t* plane, std::uint64_t row_base, int row_elems, const bbqd::ActBlocks& x) {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 176;
        KScaleTable t; t.load(blk);
        const std::uint8_t* qh_base = blk + 16;
        const std::uint8_t* ql = blk + 48;
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));
        const int g0 = s * 8;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const int gi_lo = g0 + sub_lo, gi_hi = g0 + sub_hi;
            const __m256i nlo = nibble_lo(raw), nhi = nibble_hi(raw);
            const __m256i hlo = qh_bit_to_hi4(qhv, sub_lo), hhi = qh_bit_to_hi4(qhv, sub_hi);
            const __m256i wlo = _mm256_or_si256(nlo, hlo), whi = _mm256_or_si256(nhi, hhi);
            const std::int32_t isum_lo = dot32_avx2_reg(wlo, x.qs.data() + static_cast<std::size_t>(gi_lo) * bbqd::GROUP);
            const std::int32_t isum_hi = dot32_avx2_reg(whi, x.qs.data() + static_cast<std::size_t>(gi_hi) * bbqd::GROUP);
            acc += x.scale[static_cast<std::size_t>(gi_lo)] *
                   (t.sc[static_cast<std::size_t>(sub_lo)] * static_cast<float>(isum_lo) +
                    t.m[static_cast<std::size_t>(sub_lo)] * static_cast<float>(x.gsum[static_cast<std::size_t>(gi_lo)]));
            acc += x.scale[static_cast<std::size_t>(gi_hi)] *
                   (t.sc[static_cast<std::size_t>(sub_hi)] * static_cast<float>(isum_hi) +
                    t.m[static_cast<std::size_t>(sub_hi)] * static_cast<float>(x.gsum[static_cast<std::size_t>(gi_hi)]));
        }
    }
    return acc;
}

float dot_row_q6_k(const std::uint8_t* plane, std::uint64_t row_base, int row_elems, const bbqd::ActBlocks& x) {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 210;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = gguf::f16_to_f32(d_bits);
        const int g0 = s * 8;
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            const std::uint8_t* qh_base = blk + 128 + half * 32;
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));
            for (int strip = 0; strip < 4; ++strip) {
                const int gi = g0 + half * 4 + strip;
                const bool hi_nibble = (strip == 2 || strip == 3);
                const int ql_off = (strip == 1 || strip == 3) ? 32 : 0;
                const std::uint8_t* qlc = blk + half * 64 + ql_off;
                const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
                const __m256i nib = hi_nibble ? nibble_hi(raw) : nibble_lo(raw);
                const __m256i hi2 = qh_2bits_to_hi4(qhv, strip * 2);
                const __m256i wfull = _mm256_or_si256(nib, hi2);   // raw 6-bit code, 0..63
                const __m128i wlo = _mm256_castsi256_si128(wfull);
                const __m128i whi = _mm256_extracti128_si256(wfull, 1);
                const std::int32_t isum_lo = dot16_avx2_reg(wlo, x.qs.data() + static_cast<std::size_t>(gi) * bbqd::GROUP);
                const std::int32_t isum_hi = dot16_avx2_reg(whi, x.qs.data() + static_cast<std::size_t>(gi) * bbqd::GROUP + 16);
                const float scale_lo = d * static_cast<float>(sc[2 * strip + 0]);
                const float scale_hi = d * static_cast<float>(sc[2 * strip + 1]);
                const float bias_lo = -32.f * scale_lo, bias_hi = -32.f * scale_hi;
                // gsum_lo/hi -- the per-16 activation sum. Not cached yet in this scratch verification;
                // computed inline via sum_n_portable, matching the OLD path's own (also uncached) call.
                std::int32_t gsum_lo = 0, gsum_hi = 0;
                for (int l = 0; l < 16; ++l) gsum_lo += x.qs[static_cast<std::size_t>(gi) * bbqd::GROUP + static_cast<std::size_t>(l)];
                for (int l = 0; l < 16; ++l) gsum_hi += x.qs[static_cast<std::size_t>(gi) * bbqd::GROUP + 16 + static_cast<std::size_t>(l)];
                acc += x.scale[static_cast<std::size_t>(gi)] *
                       (scale_lo * static_cast<float>(isum_lo) + bias_lo * static_cast<float>(gsum_lo) +
                        scale_hi * static_cast<float>(isum_hi) + bias_hi * static_cast<float>(gsum_hi));
            }
        }
    }
    return acc;
}

}  // namespace stream

std::vector<std::uint8_t> make_blocks(gguf::TensorType type, std::uint64_t n_elements, std::uint32_t seed) {
    const gguf::BlockSpec spec = gguf::block_spec(static_cast<std::uint32_t>(type));
    std::mt19937 rng(seed);
    const std::uint64_t blocks = n_elements / spec.elems;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(blocks * spec.bytes));
    for (auto& b : raw) b = static_cast<std::uint8_t>(rng() & 0xFFu);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        std::uint8_t* blk = raw.data() + b * spec.bytes;
        auto patch = [&](std::size_t off) {
            const auto d_bits = static_cast<std::uint16_t>(0x3000u + (rng() & 0x0FFFu));
            std::memcpy(blk + off, &d_bits, sizeof d_bits);
        };
        switch (type) {
            case gguf::TensorType::Q4_K:
            case gguf::TensorType::Q5_K: patch(0); patch(2); break;
            case gguf::TensorType::Q6_K: patch(208); break;
            default: break;
        }
    }
    return raw;
}

int main() {
    // --- synthetic bytes, multiple superblocks ---
    {
        constexpr int kN = 2560;  // 10 superblocks
        constexpr int kRows = 4;
        std::mt19937 rng(99);
        std::normal_distribution<float> normal(0.f, 2.f);
        std::vector<float> x(kN);
        for (float& v : x) v = normal(rng);
        bbqd::ActBlocks xq;
        xq.quantize(x.data(), kN);

        for (auto type : {gguf::TensorType::Q4_K, gguf::TensorType::Q5_K, gguf::TensorType::Q6_K}) {
            const auto raw_t = static_cast<std::uint32_t>(type);
            const auto raw = make_blocks(type, static_cast<std::uint64_t>(kRows) * kN, 4242u + raw_t);
            for (int r = 0; r < kRows; ++r) {
                float old_avx2 = 0.f;
                bbqd::gemv_plane<true>(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, &old_avx2, r, r + 1);
                float new_stream = 0.f;
                const std::uint64_t row_base = static_cast<std::uint64_t>(r) * kN;
                if (type == gguf::TensorType::Q4_K) new_stream = stream::dot_row_q4_k(raw.data(), row_base, kN, xq);
                else if (type == gguf::TensorType::Q5_K) new_stream = stream::dot_row_q5_k(raw.data(), row_base, kN, xq);
                else new_stream = stream::dot_row_q6_k(raw.data(), row_base, kN, xq);
                const bool eq = (old_avx2 == new_stream);
                std::printf("%s row %d: old_avx2=%.6f new_stream=%.6f %s\n",
                            type == gguf::TensorType::Q4_K ? "Q4_K" : type == gguf::TensorType::Q5_K ? "Q5_K" : "Q6_K",
                            r, old_avx2, new_stream, eq ? "OK" : "MISMATCH");
            }
        }
    }

    // --- real bytes, output.weight (Q4_K), timing too ---
    {
        const std::string dir = "D:/ModelWeights/Qwen3.8-Flash-Next-GGUF/UD-IQ1_S";
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(dir)) if (e.path().extension() == ".gguf") files.push_back(e.path());
        std::sort(files.begin(), files.end());
        for (auto& p : files) {
            std::ifstream f(p, std::ios::binary);
            std::vector<std::uint8_t> head(64ull * 1024 * 1024);
            f.read((char*)head.data(), head.size());
            head.resize((size_t)f.gcount());
            gguf::Reader r(head);
            if (!r.ok()) continue;
            for (auto& t : r.tensors()) {
                if (t.type_raw != (std::uint32_t)gguf::TensorType::Q4_K || t.name != "output.weight") continue;
                const int row_elems = (int)t.dims[0];
                const int n_rows = 1024;
                const gguf::BlockSpec spec = gguf::block_spec(t.type_raw);
                const std::uint64_t byte_len = ((std::uint64_t)n_rows * row_elems + spec.elems - 1) / spec.elems * spec.bytes;
                std::ifstream f2(p, std::ios::binary);
                f2.seekg((std::streamoff)(r.data_offset() + t.offset));
                std::vector<std::uint8_t> raw(byte_len);
                f2.read((char*)raw.data(), byte_len);

                std::vector<float> x(row_elems);
                for (int i = 0; i < row_elems; ++i) x[i] = 0.01f * ((i % 17) - 8);
                bbqd::ActBlocks xq;
                xq.quantize(x.data(), row_elems);

                int mism = 0;
                for (int rr = 0; rr < 32; ++rr) {
                    float old_avx2 = 0.f;
                    bbqd::gemv_plane<true>(t.type_raw, std::span<const std::uint8_t>(raw), n_rows, row_elems, xq, &old_avx2, rr, rr + 1);
                    const float new_stream = stream::dot_row_q4_k(raw.data(), (std::uint64_t)rr * row_elems, row_elems, xq);
                    if (old_avx2 != new_stream) { mism++; if (mism < 5) std::printf("row %d MISMATCH old=%.6f new=%.6f\n", rr, old_avx2, new_stream); }
                }
                std::printf("real output.weight: %d/32 mismatches\n", mism);

                // timing
                std::vector<float> out(n_rows);
                auto t0 = std::chrono::steady_clock::now();
                for (int rep = 0; rep < 5; ++rep)
                    for (int rr = 0; rr < n_rows; ++rr) out[rr] = stream::dot_row_q4_k(raw.data(), (std::uint64_t)rr * row_elems, row_elems, xq);
                auto t1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double>(t1 - t0).count() / 5 * 1e6;
                std::printf("streaming Q4_K: %.1f us/call (%.2f GB/s)\n", us, byte_len / (us * 1e-6) / 1e9);
            }
        }
    }
    return 0;
}
