// Runtime-shaped FFN kernel — the BASELINE half of the Ch27 microbench.
//
// Mathematically identical to ffn_static<S> in main.cpp, but the dimensions D and
// F arrive as *runtime* arguments. It lives in its own translation unit and is
// marked noinline so that neither the optimizer nor LTO can discover the actual
// bounds and specialize them — this is the "generic engine" leg of the A/B, where
// trip counts and strides are unknown at compile time.
//
// SwiGLU FFN: out = W_down · ( SiLU(W_gate · x) ⊙ (W_up · x) )
//   x:   (D,)            W_gate, W_up: (F, D)   W_down: (D, F)
//   g:   (F,) scratch    out: (D,)

#include <cmath>
#include <cstdint>

// 8 independent accumulators per dot product: breaks the serial FP dependency
// chain so the compiler can emit SIMD without -ffast-math reassociation. The
// remainder loop handles a runtime-unknown D%8 / F%8 — the cost the specialized
// kernel avoids when the bound is a compile-time multiple of the vector width.
namespace {
inline float dot8(const float* a, const float* b, int64_t n) {
    float s[8] = {0};
    int64_t k = 0;
    for (; k + 8 <= n; k += 8)
        for (int j = 0; j < 8; ++j) s[j] += a[k + j] * b[k + j];
    float acc = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; k < n; ++k) acc += a[k] * b[k];
    return acc;
}
} // namespace

[[gnu::noinline]]
void ffn_dynamic(const float* x,
                 const float* Wg, const float* Wu, const float* Wd,
                 float* out, float* g,
                 int64_t D, int64_t F) {
    for (int64_t o = 0; o < F; ++o) {
        const float a = dot8(Wg + o * D, x, D);
        const float b = dot8(Wu + o * D, x, D);
        const float s = a / (1.0f + std::exp(-a));   // SiLU
        g[o] = s * b;
    }
    for (int64_t o = 0; o < D; ++o)
        out[o] = dot8(Wd + o * F, g, F);
}
