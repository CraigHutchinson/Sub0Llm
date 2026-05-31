#pragma once

#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sub0llm::nn {

struct GGUFTensorInfo {
    std::string       name;
    std::vector<int64_t> shape;   // logical shape (GGUF dims reversed)
    int32_t           ggml_type;
    uint64_t          offset;     // from data section base
    int64_t           numel;
};

struct GGUFModelConfig {
    std::string arch;
    int64_t     vocab_size    = 0;
    int64_t     embed_dim     = 0;
    int64_t     n_layers      = 0;
    std::size_t n_heads       = 0;
    std::size_t n_kv_heads    = 0;
    int64_t     d_ff          = 0;
    int64_t     context_len   = 0;
    float       rope_base     = 10000.0f;
    // True when the GGUF has a separate output.weight (e.g. Qwen2).
    // load_gguf_model loads it in place of token_embd.weight so that
    // ModernGPT's tied-embedding forward() uses the correct output projection.
    bool        has_separate_lm_head = false;
};

struct GGUFVocab {
    std::vector<std::string> tokens;  // token_id → token_string
    std::vector<std::string> merges;  // "A B" format, in priority order
    std::string              model;   // "llama", "gpt2", etc.
};

// Loads a GGUF file: parses header, metadata (architecture config + vocab),
// and tensor info.  Tensor data is read lazily on demand via load_tensor().
class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);

    [[nodiscard]] const GGUFModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const GGUFVocab&       vocab()  const noexcept { return vocab_;  }

    // Dequantize tensor to float32 and return as flat vector.
    [[nodiscard]] std::vector<float> load_tensor(const std::string& name) const;
    [[nodiscard]] bool has_tensor(const std::string& name) const;

    [[nodiscard]] const std::unordered_map<std::string, GGUFTensorInfo>& tensors() const noexcept {
        return tensors_;
    }

private:
    std::string                                      path_;
    GGUFModelConfig                                  config_;
    GGUFVocab                                        vocab_;
    std::unordered_map<std::string, GGUFTensorInfo>  tensors_;
    uint64_t                                         data_section_offset_ = 0;

    void parse_header();

    [[nodiscard]] static std::vector<float> dequant_f32(const float*    src, int64_t n);
    [[nodiscard]] static std::vector<float> dequant_f16(const uint16_t* src, int64_t n);
    [[nodiscard]] static std::vector<float> dequant_q8_0(const uint8_t* src, int64_t n);
};

// Build a ModernGPT from a GGUFReader and load all weights.
[[nodiscard]] ModernGPT load_gguf_model(const GGUFReader& reader);

// Convenience: open file, parse, and return model.
[[nodiscard]] ModernGPT load_gguf_model(const std::string& path);

} // namespace sub0llm::nn
