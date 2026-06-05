// ch27_specialized_build — Specialized Model Compilation (demo driver).
//
// This binary is *monomorphized* to one model: it includes the generated spec
// header produced by `sub0llm-specialize`, so every dimension below is a
// compile-time constant. Nothing here reads a config at runtime — the shapes ARE
// the program. The demo prints the baked-in spec and the constants a forward pass
// derives from it, and proves (via static_assert) that they are constant
// expressions the optimizer can fold.
//
// Regenerate the spec for a different model:
//   sub0llm-specialize --model models/Your-Model.gguf \
//                      --out-dir chapters/ch27_specialized_build/generated
// then point SUB0LLM_CH27_SPEC at the new header (see CMakeLists.txt).

#include "sub0llm/nn/static_spec.hpp"

#include SUB0LLM_CH27_SPEC_HEADER   // generated header path (CMake-injected)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

namespace {
using Spec = SUB0LLM_CH27_SPEC_TYPE;   // generated spec struct (CMake-injected)
using D    = sub0llm::nn::SpecDerived<Spec>;

static_assert(sub0llm::nn::StaticSpec<Spec>,
              "generated spec must satisfy the StaticSpec concept");

// Everything the optimizer can fold — verified at compile time.
static_assert(D::kv_group == Spec::n_heads / Spec::n_kv_heads);
static_assert(D::q_proj_dim  == static_cast<int64_t>(Spec::n_heads) * static_cast<int64_t>(Spec::head_dim));
static_assert(D::kv_proj_dim == static_cast<int64_t>(Spec::n_kv_heads) * static_cast<int64_t>(Spec::head_dim));
static_assert(D::inv_sqrt_head_dim > 0.0f);

// ── monomorphized FFN kernel — the SPECIALIZED half of the microbench ─────────
//
// Identical math to ffn_dynamic (microbench_dynamic.cpp), but D and F are
// constexpr from the spec, so the compiler knows the exact trip counts, can
// unroll/vectorize against fixed bounds, and (via __restrict) assume no aliasing.
// noinline keeps the call-site cost symmetric with the dynamic version so the
// benchmark isolates the kernel body, not inlining differences.
// constexpr trip count N: the compiler knows whether N is a multiple of the
// vector width, so the remainder loop folds away entirely when it is.
template<int64_t N>
inline float dot8_static(const float* __restrict a, const float* __restrict b) {
    float s[8] = {0};
    int64_t k = 0;
    for (; k + 8 <= N; k += 8)
        for (int j = 0; j < 8; ++j) s[j] += a[k + j] * b[k + j];
    float acc = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; k < N; ++k) acc += a[k] * b[k];
    return acc;
}

template<sub0llm::nn::StaticSpec S>
[[gnu::noinline]]
void ffn_static(const float* __restrict x,
                const float* __restrict Wg, const float* __restrict Wu,
                const float* __restrict Wd,
                float* __restrict out, float* __restrict g) {
    constexpr int64_t Dd = S::embed_dim;
    constexpr int64_t F  = S::d_ff;
    for (int64_t o = 0; o < F; ++o) {
        const float a = dot8_static<Dd>(Wg + o * Dd, x);
        const float b = dot8_static<Dd>(Wu + o * Dd, x);
        const float s = a / (1.0f + std::exp(-a));
        g[o] = s * b;
    }
    for (int64_t o = 0; o < Dd; ++o)
        out[o] = dot8_static<F>(Wd + o * F, g);
}

} // namespace

// Defined in microbench_dynamic.cpp (separate TU, runtime bounds, noinline).
void ffn_dynamic(const float* x, const float* Wg, const float* Wu, const float* Wd,
                 float* out, float* g, int64_t D, int64_t F);

namespace {

// Deterministic fill (no <random> dependence; reproducible across runs).
void fill(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (float& x : v) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;        // xorshift32
        x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.1f;
    }
}

int run_bench(int reps) {
    constexpr int64_t Dd = Spec::embed_dim;
    constexpr int64_t F  = Spec::d_ff;

    std::vector<float> x(Dd), Wg(F * Dd), Wu(F * Dd), Wd(Dd * F), out(Dd), g(F);
    fill(x, 1); fill(Wg, 2); fill(Wu, 3); fill(Wd, 4);

    // Launder the dims through volatile so the dynamic call cannot be specialized
    // (even under LTO): the optimizer must treat D_rt/F_rt as unknown runtime ints.
    volatile int64_t vD = Dd, vF = F;
    const int64_t D_rt = vD, F_rt = vF;

    const double flops_per_call = 6.0 * static_cast<double>(Dd) * static_cast<double>(F);

    auto time_it = [&](auto&& call) {
        for (int w = 0; w < 50; ++w) call();          // warm-up
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) call();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    };

    double sink = 0.0;
    const double t_static = time_it([&] {
        ffn_static<Spec>(x.data(), Wg.data(), Wu.data(), Wd.data(), out.data(), g.data());
        sink += out[0];
    });
    const double t_dynamic = time_it([&] {
        ffn_dynamic(x.data(), Wg.data(), Wu.data(), Wd.data(), out.data(), g.data(), D_rt, F_rt);
        sink += out[0];
    });

    const double gfs_static  = flops_per_call * reps / t_static  / 1e9;
    const double gfs_dynamic = flops_per_call * reps / t_dynamic / 1e9;

    std::printf("Ch27 microbench — SwiGLU FFN, D=%lld F=%lld, %d reps\n",
                static_cast<long long>(Dd), static_cast<long long>(F), reps);
    std::printf("  (compile-time shapes vs runtime shapes; identical math, both -march=native)\n\n");
    std::printf("  specialized (constexpr dims): %8.2f ms   %7.2f GFLOP/s\n",
                t_static * 1e3, gfs_static);
    std::printf("  dynamic     (runtime dims):   %8.2f ms   %7.2f GFLOP/s\n",
                t_dynamic * 1e3, gfs_dynamic);
    std::printf("  speedup (dynamic / specialized): %.2fx\n", t_dynamic / t_static);
    std::printf("  [checksum %.6f]\n", sink);
    return 0;
}

int run_spec_demo();   // defined below

} // namespace

int main(int argc, char** argv) {
    int reps = 20000;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--bench") {
            if (i + 1 < argc && argv[i + 1][0] != '-') reps = std::atoi(argv[++i]);
            return run_bench(reps);
        }
    }
    return run_spec_demo();
}

namespace {
int run_spec_demo() {
    std::printf("Chapter 27 — Specialized Model Compilation\n");
    std::printf("==========================================\n\n");
    std::printf("This binary is monomorphized to: %s  (arch=%s, family=%s)\n\n",
                SUB0LLM_CH27_SPEC_NAME, Spec::arch, Spec::family);

    std::printf("Compile-time dimensions (all constexpr — no runtime config):\n");
    std::printf("  vocab_size   = %lld\n",  static_cast<long long>(Spec::vocab_size));
    std::printf("  embed_dim    = %lld\n",  static_cast<long long>(Spec::embed_dim));
    std::printf("  n_heads      = %zu\n",   Spec::n_heads);
    std::printf("  n_kv_heads   = %zu\n",   Spec::n_kv_heads);
    std::printf("  head_dim     = %zu\n",   Spec::head_dim);
    std::printf("  n_layers     = %lld\n",  static_cast<long long>(Spec::n_layers));
    std::printf("  d_ff         = %lld\n",  static_cast<long long>(Spec::d_ff));
    std::printf("  rope_base    = %g\n",    static_cast<double>(Spec::rope_base));
    std::printf("  norm_eps     = %g\n",    static_cast<double>(Spec::norm_eps));
    std::printf("  embed_scale  = %g\n\n",  static_cast<double>(Spec::embed_scale));

    std::printf("Derived constants (folded at compile time):\n");
    std::printf("  kv_group (GQA fan-out)   = %zu\n",  D::kv_group);
    std::printf("  q projection width       = %lld\n", static_cast<long long>(D::q_proj_dim));
    std::printf("  kv projection width      = %lld\n", static_cast<long long>(D::kv_proj_dim));
    std::printf("  1/sqrt(head_dim)         = %g\n",   static_cast<double>(D::inv_sqrt_head_dim));
    std::printf("  KV-cache floats / token  = %lld\n\n",
                static_cast<long long>(D::n_params_per_kv_cache_token));

    std::printf("Features baked into the build:\n");
    std::printf("  qk_norm=%d  tied_emb=%d  (1+w)norm=%d  qkv_bias=%d  activation=%d\n",
                Spec::use_qk_norm, Spec::tied_embeddings, Spec::norm_plus_one,
                Spec::attn_qkv_bias, Spec::activation);
    if (Spec::sliding_window > 0)
        std::printf("  sliding_window=%lld  global every %d layers\n",
                    static_cast<long long>(Spec::sliding_window), Spec::local_global_stride);
    else
        std::printf("  full attention (no sliding window)\n");

    std::printf("\nBecause these are constants, the monomorphized forward (P2) can size\n"
                "buffers as std::array, unroll the %zu-head / %lld-layer loops, and let\n"
                "-march=native vectorize against fixed shapes.\n",
                Spec::n_heads, static_cast<long long>(Spec::n_layers));
    return 0;
}

} // namespace
