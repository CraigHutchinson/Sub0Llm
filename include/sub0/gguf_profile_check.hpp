// sub0/gguf_profile_check.hpp -- does a GGUF file's own metadata agree with a profile's dims?
//
// The GGUF header is the authoritative record of the weights actually on disk, so comparing it with a
// profile catches the "right profile, wrong file" and "profile drifted from the model" mistakes before a
// transplant. Keys are llama.cpp's `<arch>.*` names; <arch> comes from general.architecture (confirmed
// "qwen35" for unsloth/Qwen3.8-27B-GGUF, whose header was read 2026-11-06). Header-only.

#pragma once

#include "sub0/gguf.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace sub0::profile {

struct GgufMismatch {
    std::string key;                 // profile option name (no dashes)
    double profile = 0, gguf = 0;
};

struct GgufCheck {
    bool readable = false;           // the header parsed
    std::string arch;                // general.architecture ("" if absent)
    int compared = 0;                // how many profile keys had a GGUF counterpart present
    std::vector<GgufMismatch> mismatches;
};

// Reads only the first `max_bytes` of the file (metadata + tensor table, a few MiB; never the weights).
// Returns an empty buffer if the file cannot be opened.
[[nodiscard]] inline std::vector<std::uint8_t> read_gguf_prefix(const std::string& path,
                                                                std::uint64_t max_bytes = 64ull << 20) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(max_bytes));
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    buf.resize(static_cast<std::size_t>(f.gcount()));
    return buf;
}

// Compares `profile` (option name -> number) with the reader's metadata. A profile key is only checked
// when the GGUF carries the counterpart, so a missing GGUF key is "not compared", never a mismatch.
// Per-layer array values (OtherArray) are skipped for the same reason.
[[nodiscard]] inline GgufCheck check_against_gguf(const gguf::Reader& r, const std::map<std::string, double>& profile) {
    GgufCheck out;
    if (!r.ok()) return out;
    out.readable = true;
    if (const auto* a = r.find("general.architecture"); a && a->kind == gguf::Value::Kind::String) out.arch = a->s;
    if (out.arch.empty()) return out;

    auto number = [&](const std::string& key, double& v) {
        const auto* x = r.find(key);
        if (!x) return false;
        switch (x->kind) {
            case gguf::Value::Kind::Int:   v = static_cast<double>(x->i); return true;
            case gguf::Value::Kind::UInt:  v = static_cast<double>(x->u); return true;
            case gguf::Value::Kind::Float: v = x->f; return true;
            default: return false;
        }
    };
    auto compare = [&](const char* profile_key, double gguf_value) {
        const auto it = profile.find(profile_key);
        if (it == profile.end()) return;
        ++out.compared;
        if (std::abs(it->second - gguf_value) > 1e-9 * std::max(1.0, std::abs(gguf_value)))
            out.mismatches.push_back({profile_key, it->second, gguf_value});
    };
    const std::string p = out.arch + ".";
    double v = 0, w = 0;
    struct Row { const char* suffix; const char* key; };
    for (const Row& row : {Row{"embedding_length", "dmodel"}, Row{"block_count", "layers"},
                           Row{"attention.head_count", "heads"}, Row{"attention.head_count_kv", "kv-heads"},
                           Row{"attention.key_length", "head-dim"}, Row{"rope.freq_base", "rope-theta"},
                           Row{"rope.dimension_count", "rotary-dim"}, Row{"full_attention_interval", "gdn-full-attn-stride"},
                           Row{"ssm.group_count", "gdn-key-heads"}, Row{"ssm.time_step_rank", "gdn-value-heads"},
                           Row{"ssm.state_size", "gdn-key-head-dim"}, Row{"expert_count", "num-experts"},
                           Row{"expert_used_count", "experts-per-tok"}})
        if (number(p + row.suffix, v)) compare(row.key, v);
    // D_FF is the routed-expert width on a MoE model, else the dense FFN width.
    if (number(p + "expert_feed_forward_length", v)) compare("d-ff", v);
    else if (number(p + "feed_forward_length", v)) compare("d-ff", v);
    // The GDN value head dim is not stored directly: inner_size / time_step_rank (6144 / 48 = 128 on the 27B).
    if (number(p + "ssm.inner_size", v) && number(p + "ssm.time_step_rank", w) && w > 0)
        compare("gdn-value-head-dim", v / w);
    if (const auto* t = r.find("tokenizer.ggml.tokens"); t && t->array_len > 0)
        compare("vocab-exact", static_cast<double>(t->array_len));
    return out;
}

}  // namespace sub0::profile
