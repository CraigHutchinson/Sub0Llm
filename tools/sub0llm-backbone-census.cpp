// sub0llm-backbone-census.cpp -- O5 phase 1: header/tensor-table-only census of every NON-expert,
// NON-n-gram/PLE tensor in the real Qwen3.8-Flash-Next UD-IQ1_S GGUF shards -- name, GGML type, shape
// (ne0 = contraction dim, GGUF's fastest-varying axis), byte size, and which engine consumer reads it.
//
// SCOPE. This is the "backbone": everything sub0llm-transplant dequantizes to the S0L5 f32/bf16/fp8
// blob today (docs/BACKBONE_PRECISION.md). Routed-expert tensors ("_exps" in the name) already stay
// native (the .moeq sidecar, B35's moe_quant_dot.hpp) and are explicitly OUT of this census. The
// n-gram/PLE table ("ple_" prefix) is out of tools/sub0llm-transplant.cpp's own scope (docs/WP4_SCOPE.md
// S5) and stays out here too.
//
// HEADER/TENSOR-TABLE ONLY (AGENTS.md S9 -- real file, never a fixture; but a bulk load of a ~40 GiB
// artifact is not what this deliverable needs). Reads the first 64 MiB of each shard -- exactly
// backbone_dequant_bench.cpp's own precedent -- which is enough for every real shard's KV+tensor table
// here (confirmed: this tool aborts loudly, not silently, if a shard's table does not fit that window).
// No tensor PAYLOAD is ever read.
//
// CONSUMER CLASSIFICATION comes from include/sub0/transplant.hpp's own GGUF-name table (the single
// definition of which sub0llm tensor consumes which GGUF tensor, AGENTS.md S3/S5) -- transcribed here
// as a name-suffix match, not guessed. A name this table cannot classify is reported as "UNCLASSIFIED"
// rather than silently grouped with something else.
//
// Usage: sub0llm-backbone-census --gguf <dir with the model's .gguf shards>

#include "sub0/gguf.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sub0::gguf::BlockSpec;
using sub0::gguf::TensorInfo;
using sub0::gguf::TensorType;

namespace {

const char* type_name(std::uint32_t t) {
    switch (static_cast<TensorType>(t)) {
        case TensorType::F32:     return "F32";
        case TensorType::F16:     return "F16";
        case TensorType::BF16:    return "BF16";
        case TensorType::Q8_0:    return "Q8_0";
        case TensorType::Q4_K:    return "Q4_K";
        case TensorType::Q5_K:    return "Q5_K";
        case TensorType::Q6_K:    return "Q6_K";
        case TensorType::IQ2_XXS: return "IQ2_XXS";
        case TensorType::IQ1_S:   return "IQ1_S";
        case TensorType::IQ4_NL:  return "IQ4_NL";
        default:                  return "?";
    }
}

// The GGUF name -> engine consumer table, transcribed from include/sub0/transplant.hpp's dest_gguf()
// (AGENTS.md S5: re-derived from the project's own single source, not guessed). "%d" in a
// transplant.hpp pattern is the per-layer blk index; matched here as a suffix after "blk.N.".
struct Rule { std::string suffix; const char* consumer; bool per_layer; };
const std::vector<Rule>& rules() {
    static const std::vector<Rule> r = {
        {"token_embd.weight",            "Embed: TokEmb",                    false},
        {"output.weight",                "LmHead (untied)",                  false},
        {"output_hc_norm.weight",        "GatedResidual: GrExitNorm",        false},
        {"output_hc_down.weight",        "GatedResidual: GrExitDown",        false},
        {"output_hc_up.weight",          "GatedResidual: GrExitUp",          false},
        {"attn_qkv.weight",              "GDN: InProjQkv",                   true},
        {"attn_gate.weight",             "GDN: InProjZ",                     true},
        {"ssm_beta.weight",              "GDN: InProjB",                     true},
        {"ssm_alpha.weight",             "GDN: InProjA",                     true},
        {"ssm_conv1d.weight",            "GDN: Conv (depthwise)",            true},
        {"ssm_a",                        "GDN: ALog",                        true},
        {"ssm_dt.bias",                  "GDN: DtBias",                      true},
        {"ssm_norm.weight",              "GDN: Norm",                        true},
        {"ssm_out.weight",               "GDN: OutProj",                     true},
        {"hc_attn_norm.weight",          "GatedResidual: GrAttnNorm",        true},
        {"hc_attn_down.weight",          "GatedResidual: GrAttnDown",        true},
        {"hc_attn_up.weight",            "GatedResidual: GrAttnUp",          true},
        {"hc_attn_inject.weight",        "GatedResidual: GrAttnInject",      true},
        {"hc_ffn_norm.weight",           "GatedResidual: GrFfnNorm",         true},
        {"hc_ffn_down.weight",           "GatedResidual: GrFfnDown",         true},
        {"hc_ffn_up.weight",             "GatedResidual: GrFfnUp",           true},
        {"hc_ffn_inject.weight",         "GatedResidual: GrFfnInject",       true},
        {"ffn_gate_inp.weight",          "MoE: Router (dense, every token)", true},
        {"ffn_gate_shexp.weight",        "MoE: SharedGate",                  true},
        {"ffn_up_shexp.weight",          "MoE: SharedUp",                    true},
        {"ffn_down_shexp.weight",        "MoE: SharedDown",                  true},
        {"ffn_gate_inp_shexp.weight",    "MoE: SharedGateProj",              true},
        {"attn_q.weight",                "QSA: QProj + GateProj (PerHeadHalf, ne0=hidden)", true},
        {"attn_k.weight",                "QSA: KProj",                       true},
        {"attn_v.weight",                "QSA: VProj",                       true},
        {"attn_output.weight",           "QSA: OProj",                       true},
        {"attn_q_norm.weight",           "QSA: QNorm",                       true},
        {"attn_k_norm.weight",           "QSA: KNorm",                       true},
        {"indexer.q_proj.weight",        "QSA: IdxQkProj (q half)",          true},
        {"indexer.k_proj.weight",        "QSA: IdxQkProj (k half)",          true},
        {"indexer.q_norm.weight",        "QSA: IdxQNorm",                    true},
        {"indexer.k_norm.weight",        "QSA: IdxKNorm",                    true},
        {"attn_norm.weight",             "Norm (pre-block)",                 true},
        {"ffn_norm.weight",              "Norm (pre-FFN)",                   true},
        {"output_norm.weight",           "Norm (final, LnF)",                false},
    };
    return r;
}

std::string classify(const std::string& name) {
    std::string suffix = name;
    if (name.rfind("blk.", 0) == 0) {
        const auto dot2 = name.find('.', 4);
        if (dot2 != std::string::npos) suffix = name.substr(dot2 + 1);
    }
    for (const Rule& r : rules())
        if (suffix == r.suffix) return r.consumer;
    return "UNCLASSIFIED (" + suffix + ")";
}

struct Shard {
    fs::path path;
    std::uint64_t data_offset = 0;
    std::vector<TensorInfo> tensors;
};

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_dir = argv[++i];
    }
    if (gguf_dir.empty()) {
        std::fprintf(stderr, "usage: sub0llm-backbone-census --gguf <dir>\n");
        return 2;
    }

    std::vector<Shard> shards;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(gguf_dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::fprintf(stderr, "error: no .gguf under %s\n", gguf_dir.c_str()); return 2; }

    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot open %s\n", p.string().c_str()); return 2; }
        std::vector<std::uint8_t> head(64ull * 1024 * 1024);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        const auto got = static_cast<std::size_t>(f.gcount());
        head.resize(got);
        sub0::gguf::Reader r(head);
        if (!r.ok()) {
            std::fprintf(stderr, "error: %s not readable (err %d) -- header/table may exceed the 64MiB "
                                 "probe window\n", p.filename().string().c_str(), static_cast<int>(r.error()));
            return 2;
        }
        Shard sh;
        sh.path = p;
        sh.data_offset = r.data_offset();
        sh.tensors.assign(r.tensors().begin(), r.tensors().end());
        std::printf("shard %-52s tensors %5zu\n", p.filename().string().c_str(), sh.tensors.size());
        shards.push_back(std::move(sh));
    }

    // --- the census: every non-expert, non-PLE tensor ------------------------------------------------
    struct Row {
        std::string name, consumer;
        std::uint32_t type_raw = 0;
        std::vector<std::uint64_t> dims;
        std::uint64_t elements = 0, native_bytes = 0;
        bool misaligned = false;
    };
    std::vector<Row> rows;
    std::uint64_t total_native = 0, total_bf16 = 0, total_elems = 0;
    std::map<std::uint32_t, std::uint64_t> per_type_bytes, per_type_elems, per_type_count;

    for (const Shard& sh : shards)
        for (const TensorInfo& t : sh.tensors) {
            if (t.name.find("_exps") != std::string::npos) continue;    // routed experts: out of scope
            // n-gram/PLE table: the real file's own name is "per_layer_token_embd.weight" (found by
            // running this census against the real shards -- an earlier guess at "ple_*" matched nothing
            // and let a 51.2-BILLION-element tensor leak into the backbone total, a ~27 GiB error this
            // census exists to catch, not repeat). docs/WP4_SCOPE.md S5: out of transplant's scope.
            if (t.name.find("per_layer_token_embd") != std::string::npos) continue;
            // The n-gram/PLE mechanism's own per-layer sub-tensors (found here as "blk.1.ple_*" --
            // conv1d/key/value/norm_*), same WP4_SCOPE S5 exclusion, just not at name position 0.
            if (t.name.find("ple_") != std::string::npos) continue;
            Row row;
            row.name = t.name;
            row.consumer = classify(t.name);
            row.type_raw = t.type_raw;
            row.dims = t.dims;
            row.elements = t.element_count();
            const BlockSpec b = sub0::gguf::block_spec(t.type_raw);
            row.native_bytes = (b.elems == 0) ? 0
                : ((row.elements + b.elems - 1) / b.elems) * b.bytes;
            if (b.elems > 1 && !row.dims.empty() && (row.dims[0] % b.elems) != 0) row.misaligned = true;
            rows.push_back(row);

            total_elems += row.elements;
            total_native += row.native_bytes;
            total_bf16 += row.elements * 2;
            per_type_bytes[t.type_raw] += row.native_bytes;
            per_type_elems[t.type_raw] += row.elements;
            per_type_count[t.type_raw] += 1;
        }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.name < b.name; });

    std::printf("\n=== backbone census (non-expert, non-PLE tensors) ===\n");
    std::printf("%-38s %-8s %-16s %12s %12s %-6s %s\n", "name", "type", "shape(ne0..)", "elements",
                "nativeBytes", "MISAL", "consumer");
    for (const Row& row : rows) {
        std::string shape;
        for (std::size_t i = 0; i < row.dims.size(); ++i) {
            if (i) shape += "x";
            shape += std::to_string(row.dims[i]);
        }
        std::printf("%-38s %-8s %-16s %12llu %12llu %-6s %s\n", row.name.c_str(), type_name(row.type_raw),
                    shape.c_str(), static_cast<unsigned long long>(row.elements),
                    static_cast<unsigned long long>(row.native_bytes), row.misaligned ? "YES" : "-",
                    row.consumer.c_str());
    }

    std::printf("\n=== per-format totals ===\n");
    std::printf("%-8s %10s %14s %14s\n", "type", "count", "elements", "nativeBytes");
    for (auto& [t, cnt] : per_type_count)
        std::printf("%-8s %10llu %14llu %14llu (%.2f MiB)\n", type_name(t),
                    static_cast<unsigned long long>(cnt),
                    static_cast<unsigned long long>(per_type_elems[t]),
                    static_cast<unsigned long long>(per_type_bytes[t]),
                    static_cast<double>(per_type_bytes[t]) / (1024.0 * 1024.0));

    std::printf("\n=== grand totals ===\n");
    std::printf("tensors:          %zu\n", rows.size());
    std::printf("elements:         %llu\n", static_cast<unsigned long long>(total_elems));
    std::printf("native bytes:     %llu (%.2f GiB)\n", static_cast<unsigned long long>(total_native),
                static_cast<double>(total_native) / (1024.0 * 1024.0 * 1024.0));
    std::printf("bf16 bytes:       %llu (%.2f GiB)  [today's resident backbone format]\n",
                static_cast<unsigned long long>(total_bf16),
                static_cast<double>(total_bf16) / (1024.0 * 1024.0 * 1024.0));
    std::printf("f32 bytes:        %llu (%.2f GiB)\n", static_cast<unsigned long long>(total_elems) * 4,
                static_cast<double>(total_elems) * 4 / (1024.0 * 1024.0 * 1024.0));
    std::printf("native/bf16 ratio: %.3f (bytes/element native = %.4f)\n",
                static_cast<double>(total_native) / static_cast<double>(total_bf16),
                static_cast<double>(total_native) / static_cast<double>(total_elems));

    int unclassified = 0, misaligned = 0;
    for (const Row& row : rows) {
        if (row.consumer.rfind("UNCLASSIFIED", 0) == 0) ++unclassified;
        if (row.misaligned) ++misaligned;
    }
    std::printf("\nunclassified tensors: %d\n", unclassified);
    std::printf("misaligned tensors (ne0 not a multiple of the format's block size): %d\n", misaligned);
    return 0;
}
