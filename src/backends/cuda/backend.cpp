#include "backend.hpp"
#include <stdexcept>

#ifdef SUB0LLM_CUDA
#  include <cuda_runtime.h>
#  include "kernels.cuh"
#endif

namespace sub0llm::backend::cuda {

// ── Memory helpers ────────────────────────────────────────────────────────────

std::shared_ptr<Storage> alloc(std::size_t byte_size, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    void* raw = nullptr;
    const cudaError_t err = cudaMalloc(&raw, byte_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::format("cudaMalloc({} bytes) failed: {}", byte_size, cudaGetErrorString(err)));
    }
    auto storage = std::make_shared<Storage>();
    storage->data = std::shared_ptr<std::byte[]>(
        static_cast<std::byte*>(raw),
        [](std::byte* p) { cudaFree(p); });
    storage->byte_capacity = byte_size;
    storage->device        = Device::cuda(device_index);
    return storage;
#else
    (void)byte_size; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in (rebuild with -DSUB0LLM_ENABLE_CUDA=ON)");
#endif
}

void memcpy_h2d(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemcpy H→D failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memcpy_d2h(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemcpy D→H failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memcpy_d2d(void* dst, const void* src, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemcpy D→D failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)src; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

void memset_zero(void* dst, std::size_t bytes, int device_index) {
#ifdef SUB0LLM_CUDA
    cudaSetDevice(device_index);
    if (const cudaError_t e = cudaMemset(dst, 0, bytes); e != cudaSuccess)
        throw std::runtime_error(std::format("cudaMemset failed: {}", cudaGetErrorString(e)));
#else
    (void)dst; (void)bytes; (void)device_index;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

// ── Ops ───────────────────────────────────────────────────────────────────────

Tensor add(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_CUDA
    Tensor out(a.shape(), a.dtype(), a.device());
    const std::size_t n = static_cast<std::size_t>(a.numel());
    kernels::launch_add_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), n);
    return out;
#else
    (void)a; (void)b;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

Tensor mul(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_CUDA
    Tensor out(a.shape(), a.dtype(), a.device());
    const std::size_t n = static_cast<std::size_t>(a.numel());
    kernels::launch_mul_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), n);
    return out;
#else
    (void)a; (void)b;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

Tensor relu(const Tensor& a) {
#ifdef SUB0LLM_CUDA
    Tensor out(a.shape(), a.dtype(), a.device());
    const std::size_t n = static_cast<std::size_t>(a.numel());
    kernels::launch_relu_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), n);
    return out;
#else
    (void)a;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

Tensor matmul(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_CUDA
    const auto M = static_cast<std::size_t>(a.shape(0));
    const auto K = static_cast<std::size_t>(a.shape(1));
    const auto N = static_cast<std::size_t>(b.shape(1));
    Tensor out(Tensor::Shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());
    kernels::launch_matmul_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()), M, N, K);
    return out;
#else
    (void)a; (void)b;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

double matmul_q8_0_bench(const cpu::BlockQ8_0* Wq, const cpu::BlockQ8_0* Xq, float* Y,
                         int M, int K, int T, int reps, int variant) {
#ifdef SUB0LLM_CUDA
    const auto launch = (variant == 1) ? kernels::launch_matmul_q8_0_mma
                                       : kernels::launch_matmul_q8_0;
    const std::size_t nb = static_cast<std::size_t>(K) / cpu::QK8_0;
    const std::size_t wbytes = static_cast<std::size_t>(M) * nb * sizeof(cpu::BlockQ8_0);
    const std::size_t xbytes = static_cast<std::size_t>(T) * nb * sizeof(cpu::BlockQ8_0);
    const std::size_t ybytes = static_cast<std::size_t>(M) * static_cast<std::size_t>(T) * sizeof(float);

    cpu::BlockQ8_0 *dW = nullptr, *dX = nullptr;  float* dY = nullptr;
    auto ck = [](cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(e)));
    };
    ck(cudaMalloc(&dW, wbytes), "cudaMalloc W");
    ck(cudaMalloc(&dX, xbytes), "cudaMalloc X");
    ck(cudaMalloc(&dY, ybytes), "cudaMalloc Y");
    ck(cudaMemcpy(dW, Wq, wbytes, cudaMemcpyHostToDevice), "H2D W");
    ck(cudaMemcpy(dX, Xq, xbytes, cudaMemcpyHostToDevice), "H2D X");

    for (int w = 0; w < 3; ++w) launch(dW, dX, dY, M, K, T);  // warm-up
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t t0, t1;  cudaEventCreate(&t0);  cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) launch(dW, dX, dY, M, K, T);
    cudaEventRecord(t1);
    ck(cudaEventSynchronize(t1), "kernel sync");
    float ms = 0.0f;  cudaEventElapsedTime(&ms, t0, t1);

    ck(cudaMemcpy(Y, dY, ybytes, cudaMemcpyDeviceToHost), "D2H Y");
    cudaEventDestroy(t0);  cudaEventDestroy(t1);
    cudaFree(dW);  cudaFree(dX);  cudaFree(dY);
    return static_cast<double>(ms) / 1000.0;
#else
    (void)Wq; (void)Xq; (void)Y; (void)M; (void)K; (void)T; (void)reps; (void)variant;
    throw std::runtime_error("CUDA backend not compiled in");
#endif
}

// ── Layer sub-kernel validation wrappers ────────────────────────────────────────────────
#ifdef SUB0LLM_CUDA
namespace {
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(e)));
}
// Round-trip a host buffer to the device, run `fn(d...)`, copy the named output back.
struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(std::size_t bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete; DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
    DevBuf& operator=(DevBuf&&) = delete;
};
} // namespace

void rmsnorm_dev(const float* x, const float* w, float* y, int n, float eps) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    DevBuf dx(bytes), dy(bytes);
    DevBuf dw(w ? bytes : sizeof(float));
    ck(cudaMemcpy(dx.p, x, bytes, cudaMemcpyHostToDevice), "H2D x");
    if (w) ck(cudaMemcpy(dw.p, w, bytes, cudaMemcpyHostToDevice), "H2D w");
    kernels::launch_rmsnorm(static_cast<float*>(dx.p), w ? static_cast<float*>(dw.p) : nullptr,
                            static_cast<float*>(dy.p), n, eps);
    ck(cudaDeviceSynchronize(), "rmsnorm sync");
    ck(cudaMemcpy(y, dy.p, bytes, cudaMemcpyDeviceToHost), "D2H y");
}

void rope_neox_dev(const float* x_in, float* x_out, int dh, int pos, float base, const float* ff) {
    const std::size_t bytes  = static_cast<std::size_t>(dh) * sizeof(float);
    const std::size_t fbytes = static_cast<std::size_t>(dh / 2) * sizeof(float);
    DevBuf din(bytes), dout(bytes);
    DevBuf dff(ff ? fbytes : sizeof(float));
    ck(cudaMemcpy(din.p, x_in, bytes, cudaMemcpyHostToDevice), "H2D x_in");
    if (ff) ck(cudaMemcpy(dff.p, ff, fbytes, cudaMemcpyHostToDevice), "H2D ff");
    kernels::launch_rope_neox(static_cast<float*>(din.p), static_cast<float*>(dout.p), dh, pos,
                              base, ff ? static_cast<float*>(dff.p) : nullptr);
    ck(cudaDeviceSynchronize(), "rope sync");
    ck(cudaMemcpy(x_out, dout.p, bytes, cudaMemcpyDeviceToHost), "D2H x_out");
}

void geglu_dev(const float* gate, const float* up, float* out, int n) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    DevBuf dg(bytes), du(bytes), dout(bytes);
    ck(cudaMemcpy(dg.p, gate, bytes, cudaMemcpyHostToDevice), "H2D gate");
    ck(cudaMemcpy(du.p, up,   bytes, cudaMemcpyHostToDevice), "H2D up");
    kernels::launch_geglu(static_cast<float*>(dg.p), static_cast<float*>(du.p),
                          static_cast<float*>(dout.p), n);
    ck(cudaDeviceSynchronize(), "geglu sync");
    ck(cudaMemcpy(out, dout.p, bytes, cudaMemcpyDeviceToHost), "D2H out");
}

void flash_attn_decode_dev(const float* q, const float* K, const float* V, float* o,
                           int dh, int kvlen) {
    const std::size_t qb  = static_cast<std::size_t>(dh) * sizeof(float);
    const std::size_t kvb = static_cast<std::size_t>(kvlen) * static_cast<std::size_t>(dh) * sizeof(float);
    DevBuf dq(qb), dk(kvb), dv(kvb), dout(qb);
    ck(cudaMemcpy(dq.p, q, qb,  cudaMemcpyHostToDevice), "H2D q");
    ck(cudaMemcpy(dk.p, K, kvb, cudaMemcpyHostToDevice), "H2D K");
    ck(cudaMemcpy(dv.p, V, kvb, cudaMemcpyHostToDevice), "H2D V");
    kernels::launch_flash_attn_decode(static_cast<float*>(dq.p), static_cast<float*>(dk.p),
                                      static_cast<float*>(dv.p), static_cast<float*>(dout.p),
                                      dh, kvlen);
    ck(cudaDeviceSynchronize(), "flash sync");
    ck(cudaMemcpy(o, dout.p, qb, cudaMemcpyDeviceToHost), "D2H o");
}

void quantize_q8_dev(const float* x, cpu::BlockQ8_0* y, int n) {
    const int nb = n / static_cast<int>(cpu::QK8_0);
    const std::size_t xbytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t ybytes = static_cast<std::size_t>(nb) * sizeof(cpu::BlockQ8_0);
    DevBuf dx(xbytes), dy(ybytes);
    ck(cudaMemcpy(dx.p, x, xbytes, cudaMemcpyHostToDevice), "H2D x");
    kernels::launch_quantize_q8(static_cast<float*>(dx.p),
                                static_cast<cpu::BlockQ8_0*>(dy.p), nb);
    ck(cudaDeviceSynchronize(), "quantize sync");
    ck(cudaMemcpy(y, dy.p, ybytes, cudaMemcpyDeviceToHost), "D2H y");
}

void gemma_layer_decode_dev(const GpuLayerDesc& L, const float* x, int pos,
                            float* kcache, float* vcache, int max_pos, float* out) {
    const int D = L.D, dff = L.d_ff, dh = L.dh, nH = L.n_head, nKV = L.n_kv_head;
    const int qM = nH * dh, kvM = nKV * dh, group = nH / nKV;
    const int nbD = D / 32, nbQM = qM / 32, nbFF = dff / 32;
    const float eps = L.eps, base = L.rope_base;
    const auto FB = [](int n) { return static_cast<std::size_t>(n) * sizeof(float); };
    const auto QB = [](int blocks) { return static_cast<std::size_t>(blocks) * sizeof(cpu::BlockQ8_0); };

    // One pool owns every allocation (freed on scope exit, exceptions included).
    std::vector<DevBuf> pool;
    auto dev = [&](std::size_t bytes) -> void* { pool.emplace_back(bytes); return pool.back().p; };
    auto upF = [&](const float* h, int n) -> float* {
        float* d = static_cast<float*>(dev(FB(n)));
        ck(cudaMemcpy(d, h, FB(n), cudaMemcpyHostToDevice), "H2D f32");  return d; };
    auto upQ = [&](const cpu::BlockQ8_0* h, int blocks) -> cpu::BlockQ8_0* {
        auto* d = static_cast<cpu::BlockQ8_0*>(dev(QB(blocks)));
        ck(cudaMemcpy(d, h, QB(blocks), cudaMemcpyHostToDevice), "H2D q8");  return d; };
    auto sF = [&](int n) -> float* { return static_cast<float*>(dev(FB(n))); };
    auto sQ = [&](int blocks) -> cpu::BlockQ8_0* { return static_cast<cpu::BlockQ8_0*>(dev(QB(blocks))); };

    // ── upload weights (Q8, out×in) + norms (f32) ──────────────────────────────────────────
    auto* wqD   = upQ(L.wq,   qM  * nbD);
    auto* wkD   = upQ(L.wk,   kvM * nbD);
    auto* wvD   = L.has_wv ? upQ(L.wv, kvM * nbD) : nullptr;
    auto* woD   = upQ(L.wo,   D   * nbQM);
    auto* gateD = upQ(L.gate, dff * nbD);
    auto* upD   = upQ(L.up,   dff * nbD);
    auto* downD = upQ(L.down, D   * nbFF);
    auto* anD   = upF(L.attn_norm,      D);
    auto* panD  = upF(L.post_attn_norm, D);
    auto* fnD   = upF(L.ffn_norm,       D);
    auto* pfnD  = upF(L.post_ffw_norm,  D);
    auto* qnD   = upF(L.q_norm,  dh);
    auto* knD   = upF(L.k_norm,  dh);
    float* ffD  = L.rope_freqs ? upF(L.rope_freqs, dh / 2) : nullptr;

    // KV cache (host → device); positions [0,pos) already hold this layer's RoPE'd K / normed V.
    const int kvElems = nKV * max_pos * dh;
    float* kcD = upF(kcache, kvElems);
    float* vcD = upF(vcache, kvElems);

    // ── scratch ────────────────────────────────────────────────────────────────────────────
    float* dx   = upF(x, D);                 // layer input
    float* h    = sF(D);
    auto*  hq   = sQ(nbD);
    float* q    = sF(qM);
    float* kcur = sF(kvM);
    float* vcur = sF(kvM);
    float* ac   = sF(qM);                    // attn_concat
    auto*  acq  = sQ(nbQM);
    float* ao   = sF(D);
    float* aout = sF(D);                     // attn_out (x + ao)
    float* gbuf = sF(dff);
    float* ubuf = sF(dff);
    auto*  gq   = sQ(nbFF);
    float* ffo  = sF(D);
    float* dout = sF(D);

    auto d2d = [&](float* dst, const float* src, int n) {
        ck(cudaMemcpy(dst, src, FB(n), cudaMemcpyDeviceToDevice), "D2D"); };

    // ── attention ──────────────────────────────────────────────────────────────────────────
    kernels::launch_rmsnorm(dx, anD, h, D, eps);
    kernels::launch_quantize_q8(h, hq, nbD);
    kernels::launch_matmul_q8_0(wqD, hq, q,    qM,  D, 1);
    kernels::launch_matmul_q8_0(wkD, hq, kcur, kvM, D, 1);
    if (L.has_wv) kernels::launch_matmul_q8_0(wvD, hq, vcur, kvM, D, 1);
    else          d2d(vcur, kcur, kvM);               // V = raw K projection (pre-norm)

    for (int hd = 0; hd < nH; ++hd) {                 // q-head rmsnorm + RoPE (in place)
        kernels::launch_rmsnorm(q + hd * dh, qnD, q + hd * dh, dh, eps);
        kernels::launch_rope_neox(q + hd * dh, q + hd * dh, dh, pos, base, ffD);
    }
    for (int g = 0; g < nKV; ++g) {                   // kv-head: K rmsnorm+RoPE, V rmsnorm, store
        kernels::launch_rmsnorm(kcur + g * dh, knD, kcur + g * dh, dh, eps);
        kernels::launch_rope_neox(kcur + g * dh, kcur + g * dh, dh, pos, base, ffD);
        kernels::launch_rmsnorm(vcur + g * dh, nullptr, vcur + g * dh, dh, eps);
        d2d(kcD + (static_cast<long long>(g) * max_pos + pos) * dh, kcur + g * dh, dh);
        d2d(vcD + (static_cast<long long>(g) * max_pos + pos) * dh, vcur + g * dh, dh);
    }

    const int kv_lo = L.window > 0 ? std::max(0, pos - L.window + 1) : 0;
    const int kvlen = pos - kv_lo + 1;
    for (int hd = 0; hd < nH; ++hd) {                 // per-q-head windowed attention
        const long long base_off = (static_cast<long long>(hd / group) * max_pos + kv_lo) * dh;
        kernels::launch_flash_attn_decode(q + hd * dh, kcD + base_off, vcD + base_off,
                                          ac + hd * dh, dh, kvlen);
    }
    kernels::launch_quantize_q8(ac, acq, nbQM);
    kernels::launch_matmul_q8_0(woD, acq, ao, D, qM, 1);
    kernels::launch_rmsnorm(ao, panD, ao, D, eps);
    kernels::launch_add_f32(dx, ao, aout, static_cast<std::size_t>(D));   // attn_out = x + ao

    // ── FFN (GeGLU) ─────────────────────────────────────────────────────────────────────────
    kernels::launch_rmsnorm(aout, fnD, h, D, eps);
    kernels::launch_quantize_q8(h, hq, nbD);
    kernels::launch_matmul_q8_0(gateD, hq, gbuf, dff, D, 1);
    kernels::launch_matmul_q8_0(upD,   hq, ubuf, dff, D, 1);
    kernels::launch_geglu(gbuf, ubuf, gbuf, dff);
    kernels::launch_quantize_q8(gbuf, gq, nbFF);
    kernels::launch_matmul_q8_0(downD, gq, ffo, D, dff, 1);
    kernels::launch_rmsnorm(ffo, pfnD, ffo, D, eps);
    kernels::launch_add_scale(aout, ffo, dout, L.out_scale, D);           // (attn_out + ff)·out_scale

    ck(cudaDeviceSynchronize(), "gemma layer sync");
    ck(cudaMemcpy(out, dout, FB(D), cudaMemcpyDeviceToHost), "D2H out");
    ck(cudaMemcpy(kcache, kcD, FB(kvElems), cudaMemcpyDeviceToHost), "D2H kcache");
    ck(cudaMemcpy(vcache, vcD, FB(kvElems), cudaMemcpyDeviceToHost), "D2H vcache");
}
#else
void rmsnorm_dev(const float*, const float*, float*, int, float) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void rope_neox_dev(const float*, float*, int, int, float, const float*) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void geglu_dev(const float*, const float*, float*, int) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void flash_attn_decode_dev(const float*, const float*, const float*, float*, int, int) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void quantize_q8_dev(const float*, cpu::BlockQ8_0*, int) {
    throw std::runtime_error("CUDA backend not compiled in");
}
void gemma_layer_decode_dev(const GpuLayerDesc&, const float*, int, float*, float*, int, float*) {
    throw std::runtime_error("CUDA backend not compiled in");
}
#endif

} // namespace sub0llm::backend::cuda
