#pragma once

// static_spec.hpp (Ch27) — the contract for a compile-time model specification.
//
// `sub0llm-specialize` emits a struct (e.g. sub0llm::spec::Qwen3_0_6B_Q8_0) whose
// members are all `static constexpr`. The StaticSpec concept names that contract
// so a monomorphized forward path can be written `template<StaticSpec S>` and the
// compiler can fold every shape, unroll every bounded loop, and size every buffer
// as a std::array. SpecDerived adds the constants a forward pass needs that are
// functions of the primitives (q/kv projection widths, GQA group size, …).

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace sub0llm::nn {

// A type S is a StaticSpec if it exposes the architectural axes as compile-time
// constants of the expected types, plus the per-layer attention-mode predicate.
template<class S>
concept StaticSpec = requires {
    requires std::same_as<std::remove_cv_t<decltype(S::vocab_size)>,  int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::embed_dim)>,   int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::n_heads)>,     std::size_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::n_kv_heads)>,  std::size_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::head_dim)>,    std::size_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::n_layers)>,    int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::d_ff)>,        int64_t>;
    requires std::same_as<std::remove_cv_t<decltype(S::rope_base)>,   float>;
    requires std::same_as<std::remove_cv_t<decltype(S::norm_eps)>,    float>;
    requires std::same_as<std::remove_cv_t<decltype(S::embed_scale)>, float>;
    { S::use_qk_norm }     -> std::convertible_to<bool>;
    { S::tied_embeddings } -> std::convertible_to<bool>;
    { S::is_global_layer(int64_t{}) } -> std::same_as<bool>;
};

// Constants derived from a spec — the things a forward pass actually indexes with.
template<StaticSpec S>
struct SpecDerived {
    // GQA: how many query heads share one KV head.
    static constexpr std::size_t kv_group = S::n_heads / S::n_kv_heads;

    // Projection widths (heads × head_dim). For Qwen3-0.6B these are NOT embed_dim:
    // q_proj = 16×128 = 2048, kv_proj = 8×128 = 1024 = embed_dim.
    static constexpr int64_t q_proj_dim  =
        static_cast<int64_t>(S::n_heads)    * static_cast<int64_t>(S::head_dim);
    static constexpr int64_t kv_proj_dim  =
        static_cast<int64_t>(S::n_kv_heads) * static_cast<int64_t>(S::head_dim);

    // 1/sqrt(head_dim) attention scale, folded at compile time via a constexpr
    // Newton-iteration sqrt (head_dim is a power-of-two-ish small int).
    static constexpr float inv_sqrt_head_dim = [] {
        // Newton's method constexpr sqrt — head_dim is small and exact-ish.
        float x = static_cast<float>(S::head_dim), g = x;
        for (int i = 0; i < 30; ++i) g = 0.5f * (g + x / g);
        return 1.0f / g;
    }();

    static constexpr int64_t n_params_per_kv_cache_token =
        2 * static_cast<int64_t>(S::n_kv_heads) * static_cast<int64_t>(S::head_dim) * S::n_layers;
};

} // namespace sub0llm::nn
