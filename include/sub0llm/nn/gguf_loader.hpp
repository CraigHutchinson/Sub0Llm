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
    // RMSNorm epsilon (<arch>.attention.layer_norm_rms_epsilon). Default matches
    // RMSNorm's own default; Gemma uses 1e-6, LLaMA 1e-5.
    float       norm_eps      = 1e-6f;
    // Sliding-window size (<arch>.attention.sliding_window). 0 = full attention.
    // Gemma-family models interleave local (windowed) and global (full) layers.
    int64_t     sliding_window = 0;
    // Explicit per-head dimension (Qwen3-4B+: head_dim=128, D/H=80).
    // 0 = derive from embed_dim / n_heads (standard LLaMA/Qwen2 behaviour).
    int64_t     head_dim      = 0;
    // True when the GGUF has a separate output.weight (e.g. Qwen2).
    // load_gguf_model loads it in place of token_embd.weight so that
    // ModernGPT's tied-embedding forward() uses the correct output projection.
    bool        has_separate_lm_head = false;
};

struct GGUFVocab {
    std::vector<std::string> tokens;  // token_id → token_string
    std::vector<std::string> merges;  // "A B" format, in priority order (BPE only)
    std::string              model;   // "llama"/"gpt2"/"gemma4"/…
    // SentencePiece (SPM) extras — present for "gemma*"/"llama" tokenizers; the SPM
    // tokenizer merges adjacent symbols by score, not by a merge list.
    std::vector<float>       scores;       // per-token score (log-prob); SPM merge key
    std::vector<int32_t>     token_types;  // 1=normal 2=unknown 3=control 4=user 6=byte
    int32_t                  bos_id = -1;
    int32_t                  eos_id = -1;
    bool                     add_bos = false;
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

    // ggml_type of a tensor (8 = Q8_0). Throws if not present.
    [[nodiscard]] int32_t tensor_type(const std::string& name) const;

    // Read a Q8_0 tensor's RAW blocks (no dequant) — for quantize-on-load. The GGUF
    // Q8_0 block layout is byte-identical to backend::cpu::BlockQ8_0. Throws if the
    // tensor isn't Q8_0.
    [[nodiscard]] std::vector<backend::cpu::BlockQ8_0> load_tensor_q8(const std::string& name) const;

    [[nodiscard]] const std::unordered_map<std::string, GGUFTensorInfo>& tensors() const noexcept {
        return tensors_;
    }

    // All metadata key/value pairs, stringified, in file order (arrays shown as a
    // "[array ...]" placeholder). For inspection / debugging multimodal GGUFs.
    [[nodiscard]] const std::vector<std::pair<std::string, std::string>>& metadata() const noexcept {
        return metadata_;
    }

private:
    std::string                                      path_;
    GGUFModelConfig                                  config_;
    GGUFVocab                                        vocab_;
    std::unordered_map<std::string, GGUFTensorInfo>  tensors_;
    std::vector<std::pair<std::string, std::string>> metadata_;
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

// Quantize-on-load: build a Q8-resident inference model directly from a Q8_0 GGUF
// WITHOUT ever materializing f32 weights (peak RSS stays at the Q8 footprint). The
// big weights are copied raw into the model's int8 buffers; only the small norms are
// kept as f32. The returned model is inference-only (forward_one); its f32 weights
// are elided, so the autograd/training path is unavailable. Requires Q8_0 weights.
[[nodiscard]] ModernGPT load_gguf_model_q8(const GGUFReader& reader);

} // namespace sub0llm::nn
