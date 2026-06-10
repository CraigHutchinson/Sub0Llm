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
    auto ckb = [](cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(e)));
    };
    if (variant == 2) {   // aligned-repacked decode GEMV (T==1): coalesced int4 weight reads
        const std::size_t nb = static_cast<std::size_t>(K) / cpu::QK8_0;
        std::vector<signed char>    qs(static_cast<std::size_t>(M) * static_cast<std::size_t>(K));
        std::vector<unsigned short> sc(static_cast<std::size_t>(M) * nb);
        for (std::size_t m = 0; m < static_cast<std::size_t>(M); ++m)
            for (std::size_t b = 0; b < nb; ++b) {
                const cpu::BlockQ8_0& blk = Wq[m * nb + b];
                sc[m * nb + b] = blk.d;
                for (std::size_t j = 0; j < 32; ++j) qs[m * static_cast<std::size_t>(K) + b * 32 + j] = blk.qs[j];
            }
        signed char* dWqs = nullptr;  unsigned short* dWsc = nullptr;
        cpu::BlockQ8_0* dX = nullptr;  float* dY = nullptr;
        const std::size_t xbytes = static_cast<std::size_t>(T) * nb * sizeof(cpu::BlockQ8_0);
        const std::size_t ybytes = static_cast<std::size_t>(M) * static_cast<std::size_t>(T) * sizeof(float);
        ckb(cudaMalloc(&dWqs, qs.size()), "malloc qs");
        ckb(cudaMalloc(&dWsc, sc.size() * sizeof(unsigned short)), "malloc sc");
        ckb(cudaMalloc(&dX, xbytes), "malloc X");  ckb(cudaMalloc(&dY, ybytes), "malloc Y");
        ckb(cudaMemcpy(dWqs, qs.data(), qs.size(), cudaMemcpyHostToDevice), "H2D qs");
        ckb(cudaMemcpy(dWsc, sc.data(), sc.size() * sizeof(unsigned short), cudaMemcpyHostToDevice), "H2D sc");
        ckb(cudaMemcpy(dX, Xq, xbytes, cudaMemcpyHostToDevice), "H2D X");
        for (int w = 0; w < 3; ++w) kernels::launch_matmul_q8_0_gemv_aligned(dWqs, dWsc, dX, dY, M, K);
        ckb(cudaDeviceSynchronize(), "warmup");
        cudaEvent_t t0, t1;  cudaEventCreate(&t0);  cudaEventCreate(&t1);  cudaEventRecord(t0);
        for (int r = 0; r < reps; ++r) kernels::launch_matmul_q8_0_gemv_aligned(dWqs, dWsc, dX, dY, M, K);
        cudaEventRecord(t1);  ckb(cudaEventSynchronize(t1), "sync");
        float ms = 0.0f;  cudaEventElapsedTime(&ms, t0, t1);
        ckb(cudaMemcpy(Y, dY, ybytes, cudaMemcpyDeviceToHost), "D2H");
        cudaEventDestroy(t0);  cudaEventDestroy(t1);
        cudaFree(dWqs);  cudaFree(dWsc);  cudaFree(dX);  cudaFree(dY);
        return static_cast<double>(ms) / 1000.0;
    }
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

// All DEVICE pointers + shape for one Gemma layer (resident weights/norms).
struct DevLayer {
    int D, d_ff, dh, n_head, n_kv_head, window;
    float eps, rope_base, out_scale;  bool has_wv;
    const cpu::BlockQ8_0 *wq, *wk, *wv, *wo, *gate, *up, *down;
    const float *attn_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *q_norm, *k_norm, *rope_freqs;
};
// Reusable device scratch (sized for the widest layer; shared across sequential layers).
struct DevScratch {
    float *h, *q, *kcur, *vcur, *ac, *ao, *aout, *gbuf, *ubuf, *ffo;
    cpu::BlockQ8_0 *hq, *acq, *gq;
};

// Run ONE Gemma layer entirely on device: x_dev (D) -> out_dev (D); the layer's KV cache
// (kcD/vcD, laid out [kv_head][max_pos][dh]) is updated at `pos`. Mirrors forward_one's body.
// No host transfers, no sync — the caller owns those. Shared by the validation entry point and
// the resident GemmaGpuLayers path so the kernel sequence is defined once.
void run_gpu_layer(const DevLayer& L, const float* x_dev, float* out_dev,
                   bool kv_q8, void* kcD, void* vcD, int max_pos, int pos, const DevScratch& s) {
    const int D = L.D, dff = L.d_ff, dh = L.dh, nH = L.n_head, nKV = L.n_kv_head;
    const int qM = nH * dh, kvM = nKV * dh, group = nH / nKV;
    const int nbD = D / 32, nbQM = qM / 32, nbFF = dff / 32;
    const float eps = L.eps, base = L.rope_base;
    auto d2d = [](float* dst, const float* src, int n) {
        ck(cudaMemcpy(dst, src, static_cast<std::size_t>(n) * sizeof(float),
                      cudaMemcpyDeviceToDevice), "D2D"); };

    kernels::launch_rmsnorm(x_dev, L.attn_norm, s.h, D, eps);
    kernels::launch_quantize_q8(s.h, s.hq, nbD);
    kernels::launch_matmul_q8_0(L.wq, s.hq, s.q,    qM,  D, 1);
    kernels::launch_matmul_q8_0(L.wk, s.hq, s.kcur, kvM, D, 1);
    if (L.has_wv) kernels::launch_matmul_q8_0(L.wv, s.hq, s.vcur, kvM, D, 1);
    else          d2d(s.vcur, s.kcur, kvM);                 // V = raw K projection (pre-norm)

    // Per-head q/k norm+RoPE, v norm, KV store — batched (one launch each over all heads).
    kernels::launch_rmsnorm_heads(s.q, L.q_norm, s.q, nH, dh, eps);
    kernels::launch_rope_heads(s.q, s.q, nH, dh, pos, base, L.rope_freqs);
    kernels::launch_rmsnorm_heads(s.kcur, L.k_norm, s.kcur, nKV, dh, eps);
    kernels::launch_rope_heads(s.kcur, s.kcur, nKV, dh, pos, base, L.rope_freqs);
    kernels::launch_rmsnorm_heads(s.vcur, nullptr, s.vcur, nKV, dh, eps);

    const int kv_lo = L.window > 0 ? std::max(0, pos - L.window + 1) : 0;
    const int kvlen = pos - kv_lo + 1;
    if (kv_q8) {
        auto* kc = static_cast<cpu::BlockQ8_0*>(kcD);
        auto* vc = static_cast<cpu::BlockQ8_0*>(vcD);
        kernels::launch_store_kv_q8(s.kcur, s.vcur, kc, vc, nKV, dh, max_pos, pos);
        kernels::launch_flash_attn_decode_heads_q8(s.q, kc, vc, s.ac, nH, dh, kvlen, kv_lo, group, max_pos);
    } else {
        auto* kc = static_cast<float*>(kcD);
        auto* vc = static_cast<float*>(vcD);
        kernels::launch_store_kv(s.kcur, s.vcur, kc, vc, nKV, dh, max_pos, pos);
        kernels::launch_flash_attn_decode_heads(s.q, kc, vc, s.ac, nH, dh, kvlen, kv_lo, group, max_pos);
    }
    kernels::launch_quantize_q8(s.ac, s.acq, nbQM);
    kernels::launch_matmul_q8_0(L.wo, s.acq, s.ao, D, qM, 1);
    kernels::launch_rmsnorm(s.ao, L.post_attn_norm, s.ao, D, eps);
    kernels::launch_add_f32(x_dev, s.ao, s.aout, static_cast<std::size_t>(D));

    kernels::launch_rmsnorm(s.aout, L.ffn_norm, s.h, D, eps);
    kernels::launch_quantize_q8(s.h, s.hq, nbD);
    kernels::launch_matmul_q8_0(L.gate, s.hq, s.gbuf, dff, D, 1);
    kernels::launch_matmul_q8_0(L.up,   s.hq, s.ubuf, dff, D, 1);
    kernels::launch_geglu(s.gbuf, s.ubuf, s.gbuf, dff);
    kernels::launch_quantize_q8(s.gbuf, s.gq, nbFF);
    kernels::launch_matmul_q8_0(L.down, s.gq, s.ffo, D, dff, 1);
    kernels::launch_rmsnorm(s.ffo, L.post_ffw_norm, s.ffo, D, eps);
    kernels::launch_add_scale(s.aout, s.ffo, out_dev, L.out_scale, D);
}
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

namespace {
// Upload one layer's host weights/norms (GpuLayerDesc) to device, returning a DevLayer of device
// pointers + the DevBufs that own them. Shared by the validation entry point and GemmaGpuLayers.
DevLayer upload_layer(const GpuLayerDesc& L, std::vector<DevBuf>& pool) {
    const int D = L.D, dff = L.d_ff, dh = L.dh, qM = L.n_head * L.dh, kvM = L.n_kv_head * L.dh;
    const int nbD = D / 32, nbQM = qM / 32, nbFF = dff / 32;
    const auto QB = [](int b) { return static_cast<std::size_t>(b) * sizeof(cpu::BlockQ8_0); };
    const auto FB = [](int n) { return static_cast<std::size_t>(n) * sizeof(float); };
    auto upQ = [&](const cpu::BlockQ8_0* h, int blocks) -> const cpu::BlockQ8_0* {
        pool.emplace_back(QB(blocks)); void* d = pool.back().p;
        ck(cudaMemcpy(d, h, QB(blocks), cudaMemcpyHostToDevice), "H2D q8");
        return static_cast<const cpu::BlockQ8_0*>(d); };
    auto upF = [&](const float* h, int n) -> const float* {
        if (!h) return nullptr;
        pool.emplace_back(FB(n)); void* d = pool.back().p;
        ck(cudaMemcpy(d, h, FB(n), cudaMemcpyHostToDevice), "H2D f32");
        return static_cast<const float*>(d); };
    DevLayer d{};
    d.D = D; d.d_ff = dff; d.dh = dh; d.n_head = L.n_head; d.n_kv_head = L.n_kv_head;
    d.window = L.window; d.eps = L.eps; d.rope_base = L.rope_base; d.out_scale = L.out_scale;
    d.has_wv = L.has_wv;
    d.wq = upQ(L.wq, qM * nbD);  d.wk = upQ(L.wk, kvM * nbD);
    d.wv = L.has_wv ? upQ(L.wv, kvM * nbD) : nullptr;  d.wo = upQ(L.wo, D * nbQM);
    d.gate = upQ(L.gate, dff * nbD);  d.up = upQ(L.up, dff * nbD);  d.down = upQ(L.down, D * nbFF);
    d.attn_norm = upF(L.attn_norm, D); d.post_attn_norm = upF(L.post_attn_norm, D);
    d.ffn_norm = upF(L.ffn_norm, D); d.post_ffw_norm = upF(L.post_ffw_norm, D);
    d.q_norm = upF(L.q_norm, dh); d.k_norm = upF(L.k_norm, dh);
    d.rope_freqs = L.rope_freqs ? upF(L.rope_freqs, dh / 2) : nullptr;
    return d;
}
// Allocate reusable scratch sized for a layer (or the widest of several layers).
DevScratch alloc_scratch(int D, int dff, int qM, int kvM, std::vector<DevBuf>& pool) {
    const auto FB = [](int n) { return static_cast<std::size_t>(n) * sizeof(float); };
    const auto QB = [](int b) { return static_cast<std::size_t>(b) * sizeof(cpu::BlockQ8_0); };
    auto sF = [&](int n) { pool.emplace_back(FB(n)); return static_cast<float*>(pool.back().p); };
    auto sQ = [&](int b) { pool.emplace_back(QB(b)); return static_cast<cpu::BlockQ8_0*>(pool.back().p); };
    DevScratch s{};
    s.h = sF(D); s.q = sF(qM); s.kcur = sF(kvM); s.vcur = sF(kvM); s.ac = sF(qM);
    s.ao = sF(D); s.aout = sF(D); s.gbuf = sF(dff); s.ubuf = sF(dff); s.ffo = sF(D);
    s.hq = sQ(D / 32); s.acq = sQ(qM / 32); s.gq = sQ(dff / 32);
    return s;
}
} // namespace

void gemma_layer_decode_dev(const GpuLayerDesc& L, const float* x, int pos,
                            float* kcache, float* vcache, int max_pos, float* out) {
    const int D = L.D, dh = L.dh, qM = L.n_head * dh, kvM = L.n_kv_head * dh;
    const int kvElems = L.n_kv_head * max_pos * dh;
    const auto FB = [](int n) { return static_cast<std::size_t>(n) * sizeof(float); };

    std::vector<DevBuf> pool;
    const DevLayer dl = upload_layer(L, pool);
    const DevScratch s = alloc_scratch(D, L.d_ff, qM, kvM, pool);
    auto up = [&](const float* h, int n) {
        pool.emplace_back(FB(n)); void* d = pool.back().p;
        ck(cudaMemcpy(d, h, FB(n), cudaMemcpyHostToDevice), "H2D"); return static_cast<float*>(d); };
    float* dx   = up(x, D);
    float* kcD  = up(kcache, kvElems);
    float* vcD  = up(vcache, kvElems);
    pool.emplace_back(FB(D));  float* dout = static_cast<float*>(pool.back().p);

    run_gpu_layer(dl, dx, dout, /*kv_q8=*/false, kcD, vcD, max_pos, pos, s);

    ck(cudaDeviceSynchronize(), "gemma layer sync");
    ck(cudaMemcpy(out, dout, FB(D), cudaMemcpyDeviceToHost), "D2H out");
    ck(cudaMemcpy(kcache, kcD, FB(kvElems), cudaMemcpyDeviceToHost), "D2H kcache");
    ck(cudaMemcpy(vcache, vcD, FB(kvElems), cudaMemcpyDeviceToHost), "D2H vcache");
}

// ── Persistent device-resident layers ───────────────────────────────────────────────────────
struct GemmaGpuLayers::Impl {
    std::vector<DevBuf> pool;               // owns all weight/KV/scratch allocations
    std::vector<DevLayer> layers;           // device pointers per layer
    std::vector<void*> kcD, vcD;            // KV cache per layer (f32* or BlockQ8_0* per kv_q8)
    DevScratch scratch{};                   // shared across layers (sized for the widest)
    float* curAct = nullptr;                // ping-pong activation buffers
    float* nxtAct = nullptr;
    int D = 0, max_pos = 0;  bool kv_q8 = false;
};

GemmaGpuLayers::GemmaGpuLayers(const std::vector<GpuLayerDesc>& layers, int max_pos, bool kv_q8)
    : impl_(std::make_unique<Impl>()) {
    impl_->max_pos = max_pos;
    impl_->kv_q8 = kv_q8;
    impl_->D = layers.empty() ? 0 : layers.front().D;
    int maxD = 0, maxFF = 0, maxQM = 0, maxKVM = 0, maxKV = 0;
    for (const auto& L : layers) {
        maxD   = std::max(maxD,   L.D);
        maxFF  = std::max(maxFF,  L.d_ff);
        maxQM  = std::max(maxQM,  L.n_head * L.dh);
        maxKVM = std::max(maxKVM, L.n_kv_head * L.dh);
        maxKV  = std::max(maxKV,  L.n_kv_head * max_pos * L.dh);
    }
    const auto FB = [](int n) { return static_cast<std::size_t>(n) * sizeof(float); };
    const auto KVB = [](int blocks) { return static_cast<std::size_t>(blocks) * sizeof(cpu::BlockQ8_0); };
    for (const auto& L : layers) {
        impl_->layers.push_back(upload_layer(L, impl_->pool));
        const int kvf32 = L.n_kv_head * max_pos * L.dh;          // f32 KV elements per layer
        const int kvblk = L.n_kv_head * max_pos * (L.dh / 32);   // q8 KV: dh/32 blocks per (head,pos)
        const std::size_t kvbytes = kv_q8 ? KVB(kvblk) : FB(kvf32);
        impl_->pool.emplace_back(kvbytes);  impl_->kcD.push_back(impl_->pool.back().p);
        impl_->pool.emplace_back(kvbytes);  impl_->vcD.push_back(impl_->pool.back().p);
    }
    (void)maxKV;
    impl_->scratch = alloc_scratch(maxD, maxFF, maxQM, maxKVM, impl_->pool);
    impl_->pool.emplace_back(FB(maxD)); impl_->curAct = static_cast<float*>(impl_->pool.back().p);
    impl_->pool.emplace_back(FB(maxD)); impl_->nxtAct = static_cast<float*>(impl_->pool.back().p);
}

GemmaGpuLayers::~GemmaGpuLayers() = default;

void GemmaGpuLayers::decode(const float* x, int pos, float* out) {
    Impl& m = *impl_;
    const std::size_t Db = static_cast<std::size_t>(m.D) * sizeof(float);
    ck(cudaMemcpy(m.curAct, x, Db, cudaMemcpyHostToDevice), "H2D x");
    float* cur = m.curAct; float* nxt = m.nxtAct;
    for (std::size_t l = 0; l < m.layers.size(); ++l) {
        run_gpu_layer(m.layers[l], cur, nxt, m.kv_q8, m.kcD[l], m.vcD[l], m.max_pos, pos, m.scratch);
        std::swap(cur, nxt);                 // layer output becomes next layer's input (on device)
    }
    ck(cudaDeviceSynchronize(), "gpu layers sync");
    ck(cudaMemcpy(out, cur, Db, cudaMemcpyDeviceToHost), "D2H out");
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
struct GemmaGpuLayers::Impl {};
GemmaGpuLayers::GemmaGpuLayers(const std::vector<GpuLayerDesc>&, int, bool) {
    throw std::runtime_error("CUDA backend not compiled in");
}
GemmaGpuLayers::~GemmaGpuLayers() = default;
void GemmaGpuLayers::decode(const float*, int, float*) {
    throw std::runtime_error("CUDA backend not compiled in");
}
#endif

} // namespace sub0llm::backend::cuda
