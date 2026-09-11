// decode.cpp -- the CPU backend's incremental single-token (KV-cache) inference path.
//
// Split out of backend.cpp unchanged, so the decode path is a small translation unit of its own
// rather than a slice of a file that also carries the whole training/backward/optimizer
// machinery. What lives here is exactly what one generated token executes: the per-position row
// kernels, the three decode-persistent caches (KV, Gated DeltaNet recurrent state, QSA indexer
// keys), Model::forward_one, and the KV-cache entry points the engine API forwards to.
//
// The decode caches are deliberately a SEPARATE reset lifetime from anything training owns:
// kv_reset() below is the only thing that clears them, and Model::forward() (backend.cpp) never
// touches them at all. Everything shared with the batched path -- the arenas, the Worker pool,
// `Model`'s own node layout, the binding tables, the expert resolver -- comes from internal.hpp,
// so there is exactly one copy of each.
#include "internal.hpp"

#include "sub0/gated_residual_math.hpp"  // Gated Residual row math (gr::hc_norm/mix/gate/combine/tile)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// OpenMP gate, the same one backend.cpp carries and for the same reason -- see its copy for the full
// argument. This code USED to live in that file and was covered by that guard, so repeating it here is
// keeping the guard, not adding one. It matters more here than there, in fact: tied_head_row's
// `#pragma omp simd reduction(+ : s)` is what PERMITS the compiler to reassociate that float sum, so a
// build that silently lost OpenMP for this translation unit would not merely decode more slowly, it
// would decode to different last bits than the batched forward() that still had it -- breaking the
// forward/forward_one parity check with no compile-time signal at all.
#if defined(_OPENMP)
#include <omp.h>
#elif defined(SUB0_REQUIRE_OPENMP)
#error "OpenMP required but _OPENMP is undefined: this translation unit was compiled without OpenMP, so decode's `#pragma omp simd` reductions would change accumulation order relative to the batched forward path. Reconfigure with OpenMP available, or pass -DSUB0_REQUIRE_OPENMP=OFF to build single-threaded on purpose."
#else
// Intentional single-threaded fallback (configured with -DSUB0_REQUIRE_OPENMP=OFF), mirroring
// backend.cpp's own. The per-expert parallel region below then compiles to a plain loop on thread 0,
// which is exactly the pre-B20 behaviour -- and still bit-for-bit identical, since the answer never
// depended on how the selected experts were distributed (see moe_math.hpp's forward_row_via_run).
static inline int omp_get_thread_num()  { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_max_threads() { return 1; }
#endif

namespace sub0 {
namespace {
// --- Incremental single-token inference (KV-cache) --------------------------
// forward() recomputes the WHOLE context every token, so autoregressive gen is O(n*T^2). forward_one
// keeps a per-layer K/V cache and, given the new token + its position, runs the network over just
// that ONE row -- appending its K/V and attending the single new query against the cache. Per token
// this is O(T) attention + O(1) rows through the projections/FFN/head, instead of O(T) rows through
// everything. The row helpers below replicate the dense op_* math EXACTLY (same accumulation order),
// so forward_one's logits match forward()'s last row to fast-math tolerance (gated by a unit test).
// Dense weights only (the sparse-ternary linear path is not mirrored here); a ternary build keeps
// the full-forward gen path. Positions must stay < SEQ_LEN (gen falls back past the window).
struct KVCache {
    // Indexed by EXECUTION, not layer: under LoopSplit a repeated middle layer runs several times per
    // token and each execution attends over its own K/V history, so it needs its own slot. Equal to
    // N_LAYERS when looping is off.
    std::vector<float> k, v;                                    // [LOOP_EXEC_COUNT][SEQ_LEN][D_KV], flat
    void reset() {
        const size_t n = static_cast<size_t>(LOOP_EXEC_COUNT) * SEQ_LEN * D_KV;
        if (k.size() != n) { k.assign(n, 0.f); v.assign(n, 0.f); }
    }
    // Rows are D_KV wide, not D_MODEL: under GQA the cache holds N_KV_HEADS heads, which is the whole
    // point (the cache shrinks by N_HEADS/N_KV_HEADS). Identical to D_MODEL when not using GQA.
    float* krow(int l, int pos) { return k.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_KV; }
    float* vrow(int l, int pos) { return v.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_KV; }
};
thread_local KVCache g_kv;                                      // gen is single-threaded; lazily sized

// Gated DeltaNet's decode-persistent state (Stage 1; docs/GATED_DELTANET.md S2's "decode" bullet) --
// structurally NOT a bigger/smaller KVCache: it does not grow with position, it is READ AND WRITTEN IN
// PLACE by every forward_one call (an accumulator, not a per-position row store), and per-execution
// (LOOP_EXEC_COUNT slots) for the identical LoopSplit reason KVCache is. Because it is an accumulator,
// reset() must UNCONDITIONALLY re-zero it every time (unlike KVCache's reset(), which only assigns on a
// SIZE change -- safe there because every row it will ever READ was already WRITTEN earlier in the SAME
// generation; a GDN state left over from a PREVIOUS generation would otherwise silently leak into a new
// one, since nothing about starting a new generation naturally overwrites an accumulator's stale value).
struct GdnCache {
    std::vector<float> state, conv_hist;   // [LOOP_EXEC_COUNT][state/conv_hist floats-per-exec], flat
    void reset() {
        state.assign(static_cast<size_t>(LOOP_EXEC_COUNT) * gdn::state_floats(GDN_DIMS), 0.f);
        conv_hist.assign(static_cast<size_t>(LOOP_EXEC_COUNT) * gdn::conv_hist_floats(GDN_DIMS), 0.f);
    }
    float* state_of(int e) { return state.data() + static_cast<size_t>(e) * gdn::state_floats(GDN_DIMS); }
    float* conv_of(int e)  { return conv_hist.data() + static_cast<size_t>(e) * gdn::conv_hist_floats(GDN_DIMS); }
};
thread_local GdnCache g_gdn_cache;   // only ever populated/consulted when USE_GATED_DELTANET

// QSA's decode-persistent indexer key cache (Stage 1; docs/QSA.md S6's own note that this is QSA's ONLY
// cross-call state). Structurally a KVCache, not a GdnCache: it IS a per-position row store that grows
// with position (the indexer must be able to pool a block out of ANY earlier token's raw key), and it is
// per-EXECUTION for the identical LoopSplit reason KVCache is. Rows are the RAW, unnormed, unrotated
// indexer keys -- k_layernorm and RoPE happen later, on the POOLED block key, at the block's own start
// position (docs/QSA.md S1a). reset() follows KVCache's own assign-on-size-change rule, safe for the same
// reason: every row ever READ was WRITTEN earlier in the SAME generation.
//
// It ALSO carries the POOLED block-key cache (docs/QSA.md S11). A block's pooled+k_layernorm'd+RoPE'd
// key is a function of the block alone -- its fixed start position and the compress_ratio raw keys in
// it -- never of the querying position, so recomputing it on every decode step was O(T^2/ratio) work
// across a generation for an O(T/ratio) quantity. `n_cached[e]` is how many leading blocks of slot e
// are valid; qsa::indexer_select_row extends it by the (at most one) block that completed this step.
// UNLIKE raw_k, `n_cached` MUST be zeroed on every reset, not only when the size changes: a stale
// non-zero count would make a fresh generation trust the PREVIOUS generation's block keys. (raw_k keeps
// the assign-on-size-change rule, which is still safe for the same reason as before -- every row read
// was written earlier in the same generation -- and block_k needs no clearing at all, since only its
// first n_cached[e] entries are ever read.)
struct QsaCache {
    std::vector<float> raw_k;   // [LOOP_EXEC_COUNT][SEQ_LEN][QSA_IDX_HEAD_DIM_BUF], flat
    std::vector<float> block_k; // [LOOP_EXEC_COUNT][block_key_cache_floats(QSA_DIMS_BUF, SEQ_LEN)], flat
    std::vector<int>   n_cached;// [LOOP_EXEC_COUNT] -- valid leading block count per execution slot
    static constexpr size_t BLOCK_K_PER_EXEC = qsa::block_key_cache_floats(QSA_DIMS_BUF, SEQ_LEN);
    void reset() {
        const size_t n = static_cast<size_t>(LOOP_EXEC_COUNT) * SEQ_LEN * QSA_IDX_HEAD_DIM_BUF;
        if (raw_k.size() != n) raw_k.assign(n, 0.f);
        const size_t bn = static_cast<size_t>(LOOP_EXEC_COUNT) * BLOCK_K_PER_EXEC;
        if (block_k.size() != bn) block_k.assign(bn, 0.f);
        n_cached.assign(static_cast<size_t>(LOOP_EXEC_COUNT), 0);   // unconditional -- see above
    }
    float* base(int e) {
        return raw_k.data() + static_cast<size_t>(e) * SEQ_LEN * QSA_IDX_HEAD_DIM_BUF;
    }
    float* block_base(int e) { return block_k.data() + static_cast<size_t>(e) * BLOCK_K_PER_EXEC; }
    int*   n_cached_of(int e) { return n_cached.data() + e; }
};
thread_local QsaCache g_qsa_cache;   // only ever populated/consulted when USE_QSA

// --- decode's per-expert parallelism (B20 part 2) --------------------------------------------------
//
// THE PROBLEM, MEASURED. VTune on a live 48-layer decode run reported `Total Thread Count: 1` for a
// whole 60s window on a 24-core host (docs/INDEPENDENT_REVIEW_BACKLOG.md B20). Every `#pragma omp
// parallel` this backend had was in TRAINING code, parallelising the BATCH dimension -- which is the
// right axis when there is a batch, and there is no batch at all in a single decoded token. Meanwhile a
// layer's EXPERTS_PER_TOK (10) selected experts are wholly independent work being done one at a time.
//
// WHERE THE PER-THREAD STATE LIVES, AND WHY IT IS NOT A `Worker`. The obvious move -- let several
// Workers join in, exactly as train_batch does -- is not available here, and the reason is a real
// number rather than a preference: a Worker owns an ACT_CAP activation arena, which at the real Qwen4
// axes measures 7.02 GiB (the gen tool's own [mem] line moves 18.36 -> 25.39 GiB the first time one is
// touched; before internal.hpp's ACT_GRAD_FLOATS elided the dead activation-GRADIENT arena beside it,
// that step was 18.36 -> 32.41). Ten of those is more memory than this machine has, to hold an arena
// that decode never uses at all -- forward_one runs one row through stack buffers and allocates no
// arena slot and no node (see internal.hpp's note on why arena_alloc/mk_node stayed private to
// backend.cpp).
//
// So a decode thread brings the small part instead: its own single-slot resolve pool (18.75 MiB) and
// its own pair of expert_ffn_row accumulators. That is ~25 MiB per thread against a Worker's 7 GiB,
// for state whose only requirement is "not shared while several resolves are live".
//
// The alternative considered and rejected: making the SHARED 8-slot pool safe under concurrent resolve.
// It would need per-slot reservation AND pinning of a returned plane against a later round-robin
// overwrite -- refcounting on the hottest path in the engine -- to buy a hit rate that is provably zero
// in decode (see MOE_DECODE_SLOTS in internal.hpp for why). The batched path, which does have a real
// hit rate, keeps that pool untouched.
struct MoeDecodeThread {
    // B31: the fused, no-transpose pool (moeq::ExpertCacheSource) -- paired with
    // moe::expert_ffn_row_source below, replacing the old dequant-then-transpose ExpertCache/
    // expert_ffn_row pairing on this hot path only (op_moe's batched path is untouched).
    MoeDecodeExpertCacheSource cache{};
    // expert_ffn_row's two d_ff-wide accumulators. Never zero-length (D_FF >= 1 always), so this stays
    // a valid array bound in a MoE-off build where nothing ever reads it.
    std::array<float, 2 * static_cast<std::size_t>(D_FF)> ffn{};
};
// Lazily heap-allocated per participating thread, once, and reused for the process lifetime (AGENTS.md
// S1): only the threads a run actually uses allocate, and none of it sits in the DLL's static image
// (g_param_data's own SizeOfImage argument). NOT thread_local -- a multi-MB thread_local in a DLL
// overruns Windows' static-TLS block, the same hazard that made g_workers a pool rather than TLS.
std::array<std::unique_ptr<MoeDecodeThread>, MOE_DECODE_THREADS> g_moe_decode{};

// B36 (docs/INDEPENDENT_REVIEW_BACKLOG.md B25/B36): staging buffers for pipelined I/O's explicit reads,
// one gate/up/down triple PER SELECTED EXPERT (indexed by k, the router's own top-k selection order --
// NOT by decode thread: MOE_DECODE_THREADS is pinned to 1 (B29), so a single thread processes every k
// serially, but ParallelExperts::prefetch below still issues ALL n experts' reads up front, before that
// thread starts consuming the first one -- so up to EXPERTS_PER_TOK experts' bytes may be in flight or
// resolved-but-not-yet-consumed at once, which is what MaxInFlight/MOE_IO_MAX_INFLIGHT already sizes
// for). A plain global (not per-thread, not thread_local) for the same reason g_moe_decode is a pool
// rather than TLS -- gen is single-threaded at the forward_one call level (this file's own header
// comment), so nothing else can be using this stage concurrently.
struct MoeIoStage {
    std::array<std::vector<std::uint8_t>, static_cast<std::size_t>(MOE_IO_MAX_INFLIGHT)> buf{};
    void allocate(std::size_t max_bytes) {
        for (auto& b : buf) if (b.size() < max_bytes) b.resize(max_bytes);
    }
};
MoeIoStage g_moe_io_stage{};

// Runs one decode row's selected experts across MOE_DECODE_THREADS threads. Passed to
// moe::forward_row_via_run, which computes each expert into its OWN output buffer and does the weighted
// sum afterwards in the original selection order -- so this runner cannot change the answer, only who
// computes what. See that function's comment for the float-associativity argument in full.
//
// `ffn`/`g` (the caller's single shared scratch pair) are deliberately IGNORED here: they would be a
// data race across the team. Each thread uses its own pair out of g_moe_decode instead.
//
// B36: `layer_index` and `prefetch()` implement moe_math.hpp's own prefetch hook (forward_row_via_run_ex,
// detected via `requires`) -- called single-threaded, before ANY of this struct's own parallel region
// exists, with the row's EXPERTS_PER_TOK selected expert ids (already known from the router's own
// top-k, before any resolve). Under MOE_IO_PIPELINED it issues every selected expert's THREE plane reads
// as one explicit, batched overlapped-I/O submission into g_moe_io_stage, replacing the reactive
// mmap-fault-on-first-touch the default (reactive) build still triggers inside `resolve()` itself. Off
// that build config (MOE_IO_PIPELINED == false, the default) the body compiles away entirely and this
// struct's behaviour, including its generated code, is unchanged from pre-B36 `main`.
struct ParallelExperts {
    int layer_index = 0;

    void prefetch(const int* idx, int n) const {
        if constexpr (USE_MOE_QUANT && MOE_IO_PIPELINED) {
            g_moe_io_stage.allocate(static_cast<std::size_t>(g_moe_quant.max_desc_bytes()));
            std::array<moeio::Request, static_cast<std::size_t>(MOE_IO_MAX_INFLIGHT)> reqs{};
            int nr = 0;
            for (int k = 0; k < n; ++k) {
                for (int w = 0; w < moeq::PerExpert; ++w) {
                    const moeq::Desc& d = g_moe_quant.desc(layer_index, idx[k], w);
                    const int slot = k * moeq::PerExpert + w;
                    reqs[static_cast<std::size_t>(nr++)] = moeio::Request{
                        g_moe_quant.header().data_off + d.off, static_cast<std::uint32_t>(d.bytes),
                        g_moe_io_stage.buf[static_cast<std::size_t>(slot)].data()};
                }
            }
            std::string err;
            if (!g_moe_decode_io.submit(std::span<const moeio::Request>(reqs.data(), static_cast<std::size_t>(nr)),
                                        err)) {
                std::println(stderr, "fatal: B36 pipelined-I/O prefetch submit failed at layer {}: {}",
                             layer_index, err);
                std::abort();
            }
        }
    }

    template <class Body>
    void operator()(int n, float* /*ffn*/, float* /*g*/, Body&& body) const {
        #pragma omp parallel num_threads(MOE_DECODE_THREADS)
        {
            const int t = omp_get_thread_num() % MOE_DECODE_THREADS;
            if (!g_moe_decode[static_cast<std::size_t>(t)]) {
                g_moe_decode[static_cast<std::size_t>(t)] = std::make_unique<MoeDecodeThread>();
                if constexpr (USE_MOE_QUANT) g_moe_decode[static_cast<std::size_t>(t)]->cache.allocate();
            }
            // FTZ/DAZ is per-thread MXCSR state, and a thread that skipped it would compute DIFFERENT
            // floats from thread 0 the moment an intermediate went subnormal -- not merely slower ones.
            // Setting it every entry is a two-instruction write, far cheaper than tracking whether this
            // OpenMP worker has been seen before.
            set_flush_denormals();
            MoeDecodeThread& S = *g_moe_decode[static_cast<std::size_t>(t)];
            // `static` schedule with n == EXPERTS_PER_TOK == MOE_DECODE_THREADS gives each thread
            // exactly one expert, which is the whole point; it degrades correctly if a build's
            // EXPERTS_PER_TOK exceeds DEFAULT_THREADS.
            #pragma omp for schedule(static)
            for (int k = 0; k < n; ++k) body(k, S.ffn.data(), S.ffn.data() + D_FF);
        }
    }
};

// y[out] = x[in] . W[in,out]  (+ bias); dense, same order as op_linear's non-ternary path.
// B24: weights are read through Node::pdata (ParamCPtr), which is `const float*` under the F32 default
// and a widening bf16 proxy otherwise -- so this loop's source is unchanged and its F32 codegen is too.
// Note `__restrict` is dropped from the weight row: it is meaningless on a proxy, and the alias it was
// promising about (weights never alias `y`) is still true by construction.
static inline void linear_row(const float* __restrict x, const Node* W, const Node* bias,
                              float* __restrict y, int in, int out) {
    for (int o = 0; o < out; ++o) y[o] = 0.f;
    const ParamCPtr Wf = W->pdata;
    for (int p = 0; p < in; ++p) {
        const float xp = x[p];
        if (xp == 0.f) continue;
        const auto Wr = Wf + static_cast<size_t>(p) * out;
        for (int o = 0; o < out; ++o) y[o] += xp * Wr[o];
    }
    if (bias) for (int o = 0; o < out; ++o) y[o] += bias->pdata[o];
}
// Tied-embedding head, single-row (generation) form: y[v] = dot(x[:], table[v,:]), no bias -- see
// op_tied_head's comment for why this is a dot-product pattern rather than linear_row's axpy one.
static inline void tied_head_row(const float* __restrict x, const Node* table,
                                 float* __restrict y, int C, int V) {
    const ParamCPtr Tb = table->pdata;
    for (int v = 0; v < V; ++v) {
        const auto tv = Tb + static_cast<size_t>(v) * C;
        float s = 0.f;
        #pragma omp simd reduction(+ : s)
        for (int c = 0; c < C; ++c) s += x[c] * tv[c];
        y[v] = s;
    }
}
static inline void rmsnorm_row(const float* __restrict x, const Node* gamma, float* __restrict y, int C) {
    float ms = 0.f; for (int j = 0; j < C; ++j) ms += x[j] * x[j]; ms /= C;
    const float r = 1.f / std::sqrt(ms + 1e-5f);
    const ParamCPtr G = gamma->pdata;
    for (int j = 0; j < C; ++j) y[j] = x[j] * r * G[j];
}
static inline void qknorm_row(float* __restrict x, const Node* gamma, int H, int C) {   // in place, mirrors rope_row
    const int d = C / H;
    const ParamCPtr G = gamma->pdata;
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        float ms = 0.f; for (int j = 0; j < d; ++j) ms += x[off + j] * x[off + j]; ms /= d;
        const float r = 1.f / std::sqrt(ms + 1e-5f);
        for (int j = 0; j < d; ++j) x[off + j] *= r * G[j];
    }
}
static inline void rope_row(float* __restrict x, int pos, int H, int C) {   // rotate Q/K in place (t = pos)
    const int d = C / H;
    constexpr int rd = ROTARY_DIM, half = ROTARY_HALF;   // partial rotary -- mirrors op_rope; the
                                                          // un-rotated remainder needs no work in place
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        for (int m = 0; m < half; ++m) {
            const float ang = (static_cast<float>(pos) * ROPE_POS_SCALE) * std::pow(ROPE_THETA, -2.f * m / rd);
            const float cs = std::cos(ang), sn = std::sin(ang);
            const float x0 = x[off + 2 * m], x1 = x[off + 2 * m + 1];
            x[off + 2 * m]     = x0 * cs - x1 * sn;
            x[off + 2 * m + 1] = x0 * sn + x1 * cs;
        }
    }
}
// Depth attention for ONE position: forward_one's counterpart to op_depth_attn, which it must agree
// with numerically (the forward_one-vs-forward parity test is the gate). `dk`/`dv` are this token's
// depth cache, `n` its live entry count, and `v` is REWRITTEN IN PLACE with the mixture -- via a temp,
// since v is also entry `n` of the mix. Cheap because the softmax is over depth alone: a single
// position needs no sequence history, so decode needs no extra cache beyond the current token's rows.
static inline void depth_mix_row(const float* __restrict q, const float* __restrict k,
                                 float* __restrict v, const float* dk, const float* dv, int n) {
    constexpr int d = D_HEAD;
    const float scale = 1.f / std::sqrt((float)d);
    float mixed[D_KV];
    for (int hd = 0; hd < N_KV_HEADS; ++hd) {
        const int off = hd * d;
        float qb[D_HEAD] = {};
        for (int g = 0; g < GQA_GROUP; ++g) {
            const float* __restrict qh = q + (size_t)(hd * GQA_GROUP + g) * d;
            for (int a = 0; a < d; ++a) qb[a] += qh[a];
        }
        for (int a = 0; a < d; ++a) qb[a] *= 1.f / GQA_GROUP;
        float lg[DEPTH_CACHE_MAX + 1];
        float mx = -1e30f;
        for (int dd = 0; dd <= n; ++dd) {
            const float* __restrict kd = (dd < n) ? dk + (size_t)dd * D_KV + off : k + off;
            float s = 0.f;
            for (int a = 0; a < d; ++a) s += qb[a] * kd[a];
            s *= scale; lg[dd] = s; mx = std::max(mx, s);
        }
        float Z = 0.f;
        for (int dd = 0; dd <= n; ++dd) { lg[dd] = FAST_MATH ? fast_exp(lg[dd] - mx) : std::exp(lg[dd] - mx); Z += lg[dd]; }
        for (int a = 0; a < d; ++a) mixed[off + a] = 0.f;
        for (int dd = 0; dd <= n; ++dd) {
            const float p = lg[dd] / Z;
            const float* __restrict vd = (dd < n) ? dv + (size_t)dd * D_KV + off : v + off;
            for (int a = 0; a < d; ++a) mixed[off + a] += p * vd[a];
        }
    }
    for (int j = 0; j < D_KV; ++j) v[j] = mixed[j];
}
static inline float gelu_row(float v) {
    if constexpr (FAST_MATH) return gelu_fast(v);
    else return 0.5f * v * (1.f + std::erf(v * 0.70710678f));
}
static inline float silu_row(float v) {
    if constexpr (FAST_MATH) return silu_fast(v);
    else return v / (1.f + std::exp(-v));
}
}  // anonymous namespace

// Declared in internal.hpp's Model, where this function's contract comment lives.
const float* Model::forward_one(int id, int pos) {
    static thread_local std::array<float, VOCAB> logits;
    // Sentinel-pair detection needs the PREVIOUS fed token: decode feeds positions strictly
    // sequentially (prefill 0..n-1, then one per generated token), so "previous" is simply the last
    // (id,pos) this thread fed -- valid only when pos follows it directly. No explicit reset needed:
    // a NEW generation starts at pos 0, and 0 == prev_pos+1 is unsatisfiable for any real prev_pos
    // (>= 0), so a previous generation's tail can never leak in as this one's sigil.
    static thread_local int prev_id = -1, prev_pos = -2;
    const int prev = (pos == prev_pos + 1) ? prev_id : -1;
    prev_id = id; prev_pos = pos;
    // `d` is D_HEAD, NOT C / H: WP4b blocker A made the head width its own axis, so the two differ
    // the moment --head-dim is set (and the decode path's per-head strides are all in D_HEAD).
    // Spelling it C / H here was a real bug -- the forward-vs-forward_one parity check caught it at
    // 1.71 relative error on the first --head-dim build, exactly the AGENTS.md S10 class of defect
    // (a derived width re-spelled locally instead of read from the one source of truth).
    constexpr int C = D_MODEL, H = N_HEADS, d = D_HEAD;
    const float scale = 1.f / std::sqrt(static_cast<float>(d));
    // Gated Residual (Stage 1): `h` is HC_WIDE wide when GR is on (== D_MODEL, i.e. today's exact
    // size, when off -- layout.hpp's own collapse). Every OTHER per-layer temporary below (a, qn,
    // kn, vn, att, proj, f1, g1) stays exactly D_MODEL/D_FF-wide -- GR only ever widens the
    // persistent residual itself, never the sub-block's own internal working set (S2).
    // qn/att are D_Q = N_HEADS*D_HEAD wide and kn/vn are D_KV wide -- NOT D_MODEL, since WP4b
    // blocker A made D_HEAD its own axis. All four are sized to the widest of the three so one
    // constant covers them (D_KV <= D_Q by construction, see layout.hpp's static_assert); at the
    // derived head width this is exactly C, so these arrays are unchanged for every existing build.
    constexpr int ATT_W = sub0::D_Q > C ? sub0::D_Q : C;
    float h[HC_WIDE], a[C], qn[ATT_W], kn[ATT_W], vn[ATT_W], att[ATT_W], proj[C], f1[D_FF];
    [[maybe_unused]] float g1[D_FF];   // gate branch, gated FFN only
    [[maybe_unused]] float packed_copy[C];             // periodic-reinject spike, see below
    // Gated Residual scratch/output buffers -- sized 1 (never zero-length) when off, same idiom as
    // every other never-degenerate buffer in this function. gr_normed is shared by gr_read_row's
    // mix AND gate steps (both read-only consumers of the SAME hc_norm(wide) result) -- a cheaper,
    // equally correct alternative to op_gr_mix/op_gr_gate's own independent-recompute form
    // (docs/GATED_RESIDUAL.md S4c), available here because this is plain imperative code, not two
    // separate Node-graph ops that would each need their own self-contained scratch.
    [[maybe_unused]] float gr_normed[HC_WIDE];
    [[maybe_unused]] float gr_mixscr[GR_MIX_SCRATCH1 ? GR_MIX_SCRATCH1 : 1];
    [[maybe_unused]] float gr_mixed[C];
    [[maybe_unused]] float gr_inj[HC_COUNT_BUF];
    // Mixture of Experts (Stage 1, docs/MOE.md S4b) decode-path scratch -- MOE_SCRATCH1 is never
    // zero (see its own comment), unlike GR_MIX_SCRATCH1 above, so no ternary-guard is needed here.
    // (The NUM_EXPERTS-wide pointer arrays that used to sit here are gone: WP4e's moe_resolve()
    // hands moe::forward_row_via one expert at a time, so nothing needs a table of all 512.)
    [[maybe_unused]] float moe_scratch[MOE_SCRATCH1];
    // QSA (Stage 1, docs/QSA.md S4b) decode-path scratch. Every bound uses QSA_DIMS_BUF, whose widths
    // are never zero even when QSA is off, so these stay valid array bounds in a QSA-off build
    // (where nothing ever reads them) -- the same never-degenerate idiom GR_MIX_SCRATCH1 needed.
    // Sized for the WHOLE window (SEQ_LEN), not the current position: forward_one's kv window grows
    // with `pos`, and AGENTS.md S1 forbids sizing per call.
    [[maybe_unused]] float qsa_idx_q[QSA_DIMS_BUF.idx_q_width()];
    [[maybe_unused]] float qsa_gate_row[QSA_DIMS_BUF.q_width()];
    [[maybe_unused]] float qsa_mask[SEQ_LEN];
    [[maybe_unused]] float qsa_sel_scr[qsa::select_scratch_floats(QSA_DIMS_BUF, SEQ_LEN)];
    [[maybe_unused]] float qsa_att_scr[qsa::attn_scratch_floats(QSA_DIMS_BUF, SEQ_LEN)];
    // GR READ (mix+gate, into `out_a`) / WRITE (combine, in place on `wide`) row-helpers, T==1 --
    // the exact decode-path counterpart of forward()'s gr_read/gr_write lambdas (docs/GATED_RESIDUAL.md
    // S2). Safe to write `wide` in place in gr_write_row: combine()'s per-(stream,channel) output
    // depends only on that SAME index's own input (plus mixer_out, a disjoint buffer), never on any
    // other index, so there is no read-after-write hazard.
    auto gr_read_row = [&](const float* wide, Node* norm_w, Node* down_w, Node* up_w, Node* inject_w,
                            Node* ln, float* out_a) {
        if constexpr (USE_GATED_RESIDUAL) {
            gr::hc_norm<USE_SIMD_REDUCE>(GR_DIMS, 1, wide, norm_w->pdata, gr_normed);
            gr::mix(GR_DIMS, 1, gr_normed, down_w->pdata, up_w->pdata, gr_mixed, gr_mixscr);
            gr::gate(GR_DIMS, 1, gr_normed, inject_w->pdata, gr_inj);
            // WP4b blocker D: the mixer reads mixed_input DIRECTLY -- no Ln1/Ln2 exists under GR
            // (`ln` is nullptr), and gr::hc_norm above already applied the real model's own
            // pre-block norm at its own 1e-6 eps. Mirrors forward()'s gr_read exactly; the
            // forward-vs-forward_one parity test is what gates that they stay mirrored.
            (void)ln;
            std::copy_n(gr_mixed, C, out_a);
        } else {
            rmsnorm_row(wide, ln, out_a, C);
        }
    };
    auto gr_write_row = [&](float* wide, const float* mixer_out) {
        if constexpr (USE_GATED_RESIDUAL) gr::combine(GR_DIMS, 1, wide, mixer_out, gr_inj, wide);
        else                              for (int j = 0; j < C; ++j) wide[j] += mixer_out[j];
    };
    // This token's depth cache: rows only, since the depth softmax never crosses positions (see
    // depth_mix_row). Scoped to one forward_one call -- unlike g_kv, nothing here spans tokens.
    [[maybe_unused]] float dep_k[(DEPTH_CACHE_MAX ? DEPTH_CACHE_MAX : 1) * D_KV];
    [[maybe_unused]] float dep_v[(DEPTH_CACHE_MAX ? DEPTH_CACHE_MAX : 1) * D_KV];
    [[maybe_unused]] int   dep_n = 0;
    bool do_reinject = false;

    if (g_sentinel_binds && prev == g_sentinel_binds->sigil && g_sentinel_binds->bound(id)) {
        encode_slot(tok_emb->pdata, C, g_sentinel_binds->fragments(id), g_sentinel_binds->encoding,
                    h, g_sentinel_binds->enc_w);
    } else if (g_scratch_binds && is_scratch_slot(id) && g_scratch_binds->bound(id)) {
        encode_slot(tok_emb->pdata, C, g_scratch_binds->fragments(id), g_scratch_binds->encoding, h,
                    g_scratch_binds->enc_w);
        if (g_scratch_reinject_stride > 0) {   // save the layer-0 packed vector for periodic re-injection
            for (int j = 0; j < C; ++j) packed_copy[j] = h[j];
            do_reinject = true;
        }
    } else if (is_persistent_slot(id, VOCAB)) {
        // Same unconditional guard as op_embed's forward branch (backend.cpp) -- see its
        // comment. Decode never runs backward, so this path only needs the forward compose (enc_w,
        // never enc_w_grad).
        const SlotEncoding enc = g_persistent_binds ? g_persistent_binds->encoding : SlotEncoding::MeanPool;
        encode_slot(tok_emb->pdata, C, persistent_fragments(g_persistent_binds, id), enc, h,
                    g_persistent_binds ? g_persistent_binds->enc_w : nullptr);
    } else {
        const auto emb = tok_emb->pdata + static_cast<size_t>(id) * C;
        for (int j = 0; j < C; ++j) h[j] = emb[j];
    }
    if constexpr (POS_ENCODING == PosEncoding::Absolute) {
        const auto pe = pos_emb->pdata + static_cast<size_t>(pos) * C;
        for (int j = 0; j < C; ++j) h[j] += pe[j];
    }
    // N-gram embeddings, decode path: mirrors forward()'s block exactly (same hashing, same
    // block-matmul-via-row-slice composition), but the context comes from a small rolling history
    // of the last NGRAM_MAX_N-1 FED ids instead of indexing back into a batched window -- this
    // engine's analogue of the reference's `NgramCache.update_ngram_context`. `prev < 0` (computed
    // above, BEFORE it was overwritten) is the same "fresh generation or non-sequential jump"
    // signal forward_one's sentinel-pair detection already relies on, so history resets exactly
    // when that context would otherwise be wrong.
    if constexpr (NGRAM_EMBED) {
        static thread_local int hist[NGRAM_MAX_SHIFT_BUF];
        static thread_local int hist_len = 0;
        if (prev < 0) hist_len = 0;
        // Same guard as forward()'s ngram_tok: a persistent-slot id (>= VOCAB) is not a real
        // recurring vocabulary token and its raw integer value is unbounded, so it hashes as "no
        // signal" (id 0) -- see forward()'s comment for the full reasoning and the differential
        // test (persistent_slots_engine_tests.cpp) that caught this without the guard.
        const int id_tok = (id < VOCAB) ? id : 0;
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) {
            const int order     = NGRAM_ORDERS[static_cast<std::size_t>(e)];
            const int vocab_dim = NGRAM_VOCAB_DIMS[static_cast<std::size_t>(e)];
            std::int64_t acc = id_tok;
            for (int k = 2; k <= order; ++k) {
                const int shift = k - 1;
                const int ptok  = (shift - 1 < hist_len) ? hist[shift - 1] : 0;
                acc += static_cast<std::int64_t>(ptok) * NGRAM_VOCAB_MODS[static_cast<std::size_t>(e)][static_cast<std::size_t>(k - 2)];
            }
            const int tid = static_cast<int>(((acc % vocab_dim) + vocab_dim) % vocab_dim);
            const auto row = ngram_tab[static_cast<std::size_t>(e)]->pdata
                             + static_cast<std::size_t>(tid) * NGRAM_EMB_DIM;
            const ParamCPtr Wb = ngram_wblock[static_cast<std::size_t>(e)].pdata;
            for (int p = 0; p < NGRAM_EMB_DIM; ++p) {
                const float xp = row[p];
                const auto Wr = Wb + static_cast<std::size_t>(p) * C;
                for (int o = 0; o < C; ++o) h[o] += xp * Wr[o];
            }
        }
        for (int s = NGRAM_MAX_SHIFT_BUF - 1; s > 0; --s) hist[s] = hist[s - 1];
        hist[0] = id_tok;   // history stores the GUARDED value, so it stays consistent later too
        if (hist_len < NGRAM_MAX_SHIFT_BUF) ++hist_len;
    }
    // Gated Residual's model-level ENTRY tile (docs/GATED_RESIDUAL.md S1c): everything above (embed,
    // absolute pos, n-gram injection) wrote into h's first D_MODEL elements exactly as before --
    // this duplicates that across the other HC_COUNT-1 streams. Safe in place (out_wide aliases the
    // same buffer tile() reads from): stream 0's write is `h[j] = h[j]` (a no-op), and every later
    // stream's write target lies entirely outside the [0,D_MODEL) range tile() ever reads from.
    if constexpr (USE_GATED_RESIDUAL) gr::tile(GR_DIMS, 1, h, h);
    // `e` is the EXECUTION index (the KV-cache slot); `li` is which layer's weights run there.
    // They differ only under LoopSplit -- see LAYER_EXEC_ORDER and KVCache's own comments.
    for (int e = 0; e < LOOP_EXEC_COUNT; ++e) {
        const int l = LAYER_EXEC_ORDER[static_cast<std::size_t>(e)];
        Layer& L = layers[static_cast<std::size_t>(l)];
        // `h` is mutated IN PLACE by gr_write_row below (unlike forward()'s Node graph, where a new
        // Node replaces `h`), so gr_read_row's read and gr_write_row's later write on the SAME `h`
        // need no separate "before" snapshot -- nothing between them mutates it.
        gr_read_row(h, L.gr_attn_norm, L.gr_attn_down, L.gr_attn_up, L.gr_attn_inject, L.ln1, a);
        // The softmax-attention mixer sublayer, factored into a lambda (rather than duplicated
        // verbatim in both branches below) so the GDN_SCHEDULE dispatch reads as a single small
        // if/else rather than two copies of this block drifting apart over time.
        auto do_attention_mixer = [&] {
            // Wq is [C, D_Q] -- the OUT width is D_Q = N_HEADS*D_HEAD, not D_MODEL, since WP4b
            // blocker A. Identical at the derived head width.
            linear_row(a, L.Wq, nullptr, qn, C, sub0::D_Q);
            // K/V project to D_KV (N_KV_HEADS heads), Q to D_Q (N_HEADS heads). qknorm_row/rope_row
            // both derive their per-head width as (width / heads), so passing the KV pair yields
            // the same D_HEAD they always did -- the rotation and norm are per-head, so nothing
            // else changes.
            linear_row(a, L.Wk, nullptr, kn, C, D_KV);
            linear_row(a, L.Wv, nullptr, vn, C, D_KV);
            if constexpr (USE_QK_NORM) { qknorm_row(qn, L.q_norm, H, sub0::D_Q); qknorm_row(kn, L.k_norm, N_KV_HEADS, D_KV); }
            if constexpr (POS_ENCODING == PosEncoding::Rope) { rope_row(qn, pos, H, sub0::D_Q); rope_row(kn, pos, N_KV_HEADS, D_KV); }
            // Depth attention, before the KV-cache append: the sequence cache must hold the MIXED
            // V, because that is what op_attn receives in the batched forward. Mirrors forward()'s
            // block.
            if constexpr (USE_DEPTH_ATTN) {
                depth_mix_row(qn, kn, vn, dep_k, dep_v, dep_n);
                if (DEPTH_SCHEDULE.own[static_cast<std::size_t>(e)] >= 0) {
                    for (int j = 0; j < D_KV; ++j) {
                        dep_k[(size_t)dep_n * D_KV + j] = kn[j];
                        dep_v[(size_t)dep_n * D_KV + j] = vn[j];
                    }
                    ++dep_n;
                }
            }
            float* kc = g_kv.krow(e, pos); float* vc = g_kv.vrow(e, pos);    // append this token's K/V
            for (int j = 0; j < D_KV; ++j) { kc[j] = kn[j]; vc[j] = vn[j]; }
            for (int hd = 0; hd < H; ++hd) {                                 // attend query pos over j<=pos
                const int off    = hd * d;
                const int off_kv = (hd / GQA_GROUP) * d;    // shared KV head for this query group
                std::array<float, SEQ_LEN> sc{};
                float mx = -1e30f;
                for (int j = 0; j <= pos; ++j) {
                    const float* kj = g_kv.krow(e, j) + off_kv;
                    float s = 0.f; for (int aa = 0; aa < d; ++aa) s += qn[off + aa] * kj[aa];
                    s *= scale; sc[j] = s; mx = std::max(mx, s);
                }
                float Z = 0.f;
                for (int j = 0; j <= pos; ++j) { sc[j] = FAST_MATH ? fast_exp(sc[j] - mx) : std::exp(sc[j] - mx); Z += sc[j]; }
                for (int aa = 0; aa < d; ++aa) att[off + aa] = 0.f;
                for (int j = 0; j <= pos; ++j) {
                    const float p = sc[j] / Z; const float* vj = g_kv.vrow(e, j) + off_kv;
                    for (int aa = 0; aa < d; ++aa) att[off + aa] += p * vj[aa];
                }
            }
            linear_row(att, L.Wo, nullptr, proj, sub0::D_Q, C);   // Wo is [D_Q, D_MODEL] (blocker A)
            gr_write_row(h, proj);                                           // residual (write step)
        };
        // The QSA decode counterpart of the same sublayer -- the T==1 form of op_qsa's own batched
        // call, built from the SAME qsa_math.hpp row helpers op_qsa's qsa::forward() loops over, so
        // the two provably compute identical arithmetic (the forward-vs-forward_one parity test is
        // what gates that -- docs/QSA.md S9). Reuses g_kv for K/V exactly as the softmax path does
        // (kv_width() == D_KV), and g_qsa_cache for the indexer's own raw keys, which g_kv cannot
        // hold (a different width and a different -- unnormed, unrotated -- content).
        [[maybe_unused]] auto do_qsa_mixer = [&] {
            float* raw_k_base = g_qsa_cache.base(e);
            // Row stride is ROTARY_DIM (the cos/sin tables are [SEQ_LEN][ROTARY_DIM]) -- see
            // QsaRopeTables (internal.hpp); it was D_HEAD before --rotary-dim became an axis.
            const float* cos_pos = g_qsa_rope.cos.data() + static_cast<size_t>(pos) * ROTARY_DIM;
            const float* sin_pos = g_qsa_rope.sin.data() + static_cast<size_t>(pos) * ROTARY_DIM;
            qsa::indexer_project_row<USE_SIMD_REDUCE>(QSA_DIMS, a, L.qsa_idx_qk->pdata,
                                      L.qsa_idx_qnorm->pdata, cos_pos, sin_pos, qsa::RMS_EPS,
                                      qsa_idx_q,
                                      raw_k_base + static_cast<size_t>(pos) * QSA_INDEXER_HEAD_DIM);
            qsa::attn_project_row<USE_SIMD_REDUCE>(QSA_DIMS, a, L.qsa_q->pdata, L.qsa_gate->pdata,
                                   L.qsa_k->pdata, L.qsa_v->pdata,
                                   L.qsa_qnorm->pdata, L.qsa_knorm->pdata,
                                   cos_pos, sin_pos, qsa::RMS_EPS,
                                   qn, qsa_gate_row, g_kv.krow(e, pos), g_kv.vrow(e, pos));
            // The pooled block keys persist across decode steps in this execution's own QsaCache
            // slot, exactly as the raw keys above already do -- the decode counterpart of the
            // batched path's in-scratch cache, filled by the SAME primitive (docs/QSA.md S11).
            qsa::indexer_select_row<USE_SIMD_REDUCE>(QSA_DIMS, qsa_idx_q, raw_k_base, pos + 1,
                                     L.qsa_idx_knorm->pdata, g_qsa_rope.cos.data(),
                                     g_qsa_rope.sin.data(), qsa::RMS_EPS,
                                     g_qsa_cache.block_base(e), g_qsa_cache.n_cached_of(e),
                                     qsa_mask, qsa_sel_scr);
            qsa::attn_row<USE_SIMD_REDUCE>(QSA_DIMS, qn, qsa_gate_row, g_kv.krow(e, 0), g_kv.vrow(e, 0), pos + 1,
                           qsa_mask, L.qsa_o->pdata, proj, qsa_att_scr);
            gr_write_row(h, proj);                                           // residual (write step)
        };
        // Which of the two full-attention forms this build uses, decided ONCE at compile time. This
        // is the seam the MoE stage's own SIGSEGV came through (docs/MOE.md S9: a replacement wired
        // into only ONE of two call sites), so it is expressed as a single named lambda that BOTH
        // the GDN-on and GDN-off dispatch paths below call, rather than duplicated in each.
        auto do_full_attn_mixer = [&] {
            if constexpr (USE_QSA) do_qsa_mixer();
            else                   do_attention_mixer();
        };
        // GDN_SCHEDULE.full_attn[l] mirrors Model::forward()'s own dispatch exactly (per-LAYER, not
        // per-execution) -- see that function's comment. `if constexpr` keeps a GDN-off build
        // exactly the original single call, no branch at all.
        if constexpr (USE_GATED_DELTANET) {
            if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
                do_full_attn_mixer();
            } else {
                // Gated DeltaNet decode: T=1, persistent per-execution state/conv history
                // (GdnCache, reset by kv_reset() at the start of each generation). op_gdn's whole
                // output (in_proj*, conv, recurrence, gated-norm, out_proj) IS the mixer sublayer,
                // so it goes straight into the residual the same way `proj` does above -- no
                // separate Wo, no RoPE, no QK-norm, no depth-attention (see Model::forward()'s
                // matching comment for why those are out of scope for a GDN layer).
                // Same dt_bias/a_log ARGUMENT-ORDER fix as op_gdn's batched forward (backend.cpp) (this
                // decode path had the identical swap, independently) -- see that call site's comment.
                float gdn_scratch[GDN_SCRATCH1];
                gdn::forward<USE_SIMD_REDUCE>(GDN_DIMS, 1, a,
                             L.gdn_in_qkv->pdata, L.gdn_in_z->pdata,
                             L.gdn_in_b->pdata, L.gdn_in_a->pdata,
                             L.gdn_conv->pdata, L.gdn_dt_bias->pdata,
                             L.gdn_a_log->pdata, L.gdn_norm->pdata,
                             L.gdn_out_proj->pdata,
                             g_gdn_cache.state_of(e), g_gdn_cache.conv_of(e), proj, gdn_scratch);
                gr_write_row(h, proj);                                       // residual (write step)
            }
        } else {
            do_full_attn_mixer();
        }
        gr_read_row(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject, L.ln2, a);
        if constexpr (USE_MOE) {
            // WP4e: the same moe_resolve() op_moe uses, so the decode path and the batched path
            // resolve experts identically -- which is what keeps forward/forward_one parity (the
            // check docs/WP4_SCOPE.md S6 records as having caught a real bug in every one of WP1-3)
            // a check of the MODEL rather than of two different residency strategies.
            //
            // B20 part 2: ...and the same moe_math.hpp body, too. The only difference from the batched
            // path is WHICH runner walks the selected experts (ParallelExperts here, the serial default
            // there) and therefore which pool each resolve writes into -- the arithmetic, and the order
            // the ten contributions are summed in, are identical by construction. `resolve` reads
            // omp_get_thread_num() rather than closing over one pool, because it is called from inside
            // the runner's own parallel region, once per thread.
            // B31: fused resolve+FFN -- under MOE_QUANT_EXPERTS this dequantizes straight into
            // SOURCE order (no transpose) and immediately runs expert_ffn_row_source over it, cutting
            // the resolve's DRAM round trips from ~3 to 1 per plane-set (see moeq::ExpertCacheSource's
            // and moe::expert_ffn_row_source's own comments for the full mechanism and the bit-exactness
            // argument). The f32-resident (non-quantized) build keeps the original resolve+expert_ffn_row
            // pairing unchanged -- there is no transpose to fuse away there at all.
            // B36: under MOE_IO_PIPELINED, ParallelExperts::prefetch() (moe_math.hpp's own hook, run
            // once before run_experts) has already issued this row's EXPERTS_PER_TOK*PerExpert explicit
            // reads into g_moe_io_stage before this lambda is ever called -- resolve_from_bytes waits
            // for and consumes THIS iteration's (k's) three planes from that stage instead of
            // ExpertCacheSource::resolve()'s own mmap-backed path. Off that build config (the default),
            // this whole branch compiles away and `resolve()` runs exactly as it did pre-B36.
            // B34/B38: the explicit <USE_SIMD_REDUCE> template arg selects the shared-expert gate-logit
            // reduction strategy inside forward_row_via_run_ex; see docs/INDEPENDENT_REVIEW_BACKLOG.md
            // B38 and include/sub0/simd_reduce.hpp. expert_ffn_row_source below (this lambda's own call)
            // is NEVER gated by this -- it always uses simd::dot_seq regardless, to preserve B31's
            // forward()/forward_one() bit-exactness invariant.
            moe::forward_row_via_run_ex<USE_SIMD_REDUCE>(
                MOE_DIMS, a, L.moe_router->pdata,
                [&](int k, int e, float* out_ptr, float* ffn, float* g) {
                    const int t = omp_get_thread_num() % MOE_DECODE_THREADS;
                    MoeDecodeThread& S = *g_moe_decode[static_cast<std::size_t>(t)];
                    if constexpr (USE_MOE_QUANT) {
                        MoeDecodeExpertCacheSource::Resolved r;
                        if constexpr (MOE_IO_PIPELINED) {
                            std::string err;
                            for (int w = 0; w < moeq::PerExpert; ++w) {
                                if (!g_moe_decode_io.wait(k * moeq::PerExpert + w, err)) {
                                    std::println(stderr,
                                                 "fatal: B36 pipelined-I/O read failed (layer {} expert "
                                                 "{} plane {}): {}", l, e, w, err);
                                    std::abort();
                                }
                            }
                            const moeq::Desc d0 = g_moe_quant.desc(l, e, moeq::Gate);
                            const moeq::Desc d1 = g_moe_quant.desc(l, e, moeq::Up);
                            const moeq::Desc d2 = g_moe_quant.desc(l, e, moeq::Down);
                            const int base = k * moeq::PerExpert;
                            r = S.cache.resolve_from_bytes(
                                g_moe_quant, l, e,
                                std::span<const std::uint8_t>(g_moe_io_stage.buf[static_cast<std::size_t>(base + moeq::Gate)].data(), d0.bytes),
                                std::span<const std::uint8_t>(g_moe_io_stage.buf[static_cast<std::size_t>(base + moeq::Up)].data(), d1.bytes),
                                std::span<const std::uint8_t>(g_moe_io_stage.buf[static_cast<std::size_t>(base + moeq::Down)].data(), d2.bytes));
                        } else {
                            r = S.cache.resolve(g_moe_quant, l, e);
                        }
                        if (r.gate == nullptr) {
                            std::println(stderr,
                                         "fatal: could not dequantize routed expert {} of layer {} from "
                                         "the S0Q1 sidecar (unsupported GGML type or corrupt payload)",
                                         e, l);
                            std::abort();
                        }
                        moe::expert_ffn_row_source(MOE_DIMS, a, r.gate, r.up, r.down, out_ptr, ffn, g);
                    } else {
                        const moe::ExpertWeights w = moe_resolve(L, l, e, S.cache);
                        moe::expert_ffn_row(MOE_DIMS, a, w.gate, w.up, w.down, out_ptr, ffn, g);
                    }
                },
                ParallelExperts{l},
                L.moe_shared_gate->pdata, L.moe_shared_up->pdata,
                L.moe_shared_down->pdata, L.moe_shared_gate_proj->pdata,
                proj, moe_scratch);
        } else if constexpr (USE_GATED_FFN) {
            linear_row(a, L.Wg, nullptr, g1, C, D_FF);
            linear_row(a, L.W1, nullptr, f1, C, D_FF);
            for (int j = 0; j < D_FF; ++j) f1[j] = silu_row(g1[j]) * f1[j];
            linear_row(f1, L.W2, nullptr, proj, D_FF, C);
        } else {
            linear_row(a, L.W1, L.b1, f1, C, D_FF);
            for (int j = 0; j < D_FF; ++j) f1[j] = gelu_row(f1[j]);
            linear_row(f1, L.W2, L.b2, proj, D_FF, C);
        }
        gr_write_row(h, proj);                                              // residual (write step)
        // Periodic packed-content re-injection spike (Nanbeige-inspired, see core.hpp's
        // set_scratch_reinject doc comment): every `stride` layers, add the SAME layer-0 packed
        // vector back into this position's hidden state -- tests whether reinforcing the signal
        // partway through the stack counters the dilution a single upfront injection leaves (axis 9).
        // SCALE-ADAPTIVE: the residual stream's own RMS norm grows across depth (standard pre-norm
        // behavior -- it's why ln_f exists before the head at all), while `packed_copy` sits at fixed
        // ordinary-embedding-row scale (scratch_slots.hpp's own amplitude convention) -- a first,
        // fixed-scale version of this spike was measurably a near no-op even at full strength every
        // layer, because the addition was dwarfed by h's already-larger accumulated norm by mid-stack.
        // Rescaling packed_copy to match h's CURRENT norm before applying `scale` (now a fraction of
        // h's own magnitude, not an absolute embedding-scale constant) is what makes this a real test.
        //
        // NOT Gated-Residual-aware: this spike only ever touches h's first D_MODEL elements (stream 0
        // when GR is wide), same deliberate-scope-gap shape as GDN+depth-attention's own "neither doc
        // discusses the combination" (docs/GATED_DELTANET.md) -- this spike and GR have not been
        // designed to interact, and this pass does not attempt it.
        if (do_reinject && ((l + 1) % g_scratch_reinject_stride == 0)) {
            float h_ms = 0.f; for (int j = 0; j < C; ++j) h_ms += h[j] * h[j]; h_ms /= C;
            float pk_ms = 0.f; for (int j = 0; j < C; ++j) pk_ms += packed_copy[j] * packed_copy[j]; pk_ms /= C;
            const float norm_scale = g_scratch_reinject_scale * std::sqrt((h_ms + 1e-8f) / (pk_ms + 1e-8f));
            for (int j = 0; j < C; ++j) h[j] += norm_scale * packed_copy[j];
        }
    }
    // Gated Residual's model-level EXIT collapse (docs/GATED_RESIDUAL.md S1c), mirroring forward()'s
    // own exit-collapse placement: use_combine=False, so just the mix half (no gate/combine) --
    // last_hidden captures the FULLY-COLLAPSED, D_MODEL-wide representation, same as the GR-off path.
    if constexpr (USE_GATED_RESIDUAL) {
        gr::hc_norm<USE_SIMD_REDUCE>(GR_DIMS, 1, h, gr_top_norm->pdata, gr_normed);
        gr::mix(GR_DIMS, 1, gr_normed, gr_top_down->pdata, gr_top_up->pdata, gr_mixed, gr_mixscr);
        for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = gr_mixed[j];
        // No final norm under GR -- the exit collapse's own hc_norm is it, so mixed_input feeds the
        // head directly. Mirrors forward()'s own branch exactly (the forward-vs-forward_one parity
        // check is what gates that the two stay mirrored).
        for (int j = 0; j < C; ++j) a[j] = gr_mixed[j];
    } else {
        for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = h[j];   // diagnostic capture
        rmsnorm_row(h, ln_f, a, C);
    }
    if constexpr (USE_TIED_EMBEDDINGS) tied_head_row(a, tok_emb, logits.data(), C, VOCAB);
    else                               linear_row(a, lm_head, lm_bias, logits.data(), C, VOCAB);
    return logits.data();
}

namespace cpu_detail {

// Incremental single-token inference (KV-cache). kv_reset() clears/sizes the cache at the start of a
// generation; forward_one(id, pos) returns the logits [VOCAB] for the next token. See KVCache above.
// Also resets GdnCache's decode-persistent recurrent state -- `if constexpr` so this costs nothing
// (not even the vector-emptiness check) on a build with GDN off, per AGENTS.md S4.
void kv_reset() {
    g_kv.reset();
    if constexpr (USE_GATED_DELTANET) g_gdn_cache.reset();
    if constexpr (USE_QSA) g_qsa_cache.reset();   // the indexer's own raw-key store -- docs/QSA.md S6
}
const float* forward_one(int id, int pos) { ensure_thread_built(); return g_model.forward_one(id, pos); }
const float* last_hidden_ptr() { return g_model.last_hidden.data(); }   // see Model::last_hidden's comment

// KV-trace memoization primitives (spike, see core.hpp's declarations for the full design comment).
// Caller-responsibility contract matches forward_one's own: valid only after kv_reset() has sized g_kv
// on this thread (no redundant guard here, same discipline as g_kv's other two entry points above).
const float* kv_krow_ptr(int layer, int pos) { return g_kv.krow(layer, pos); }
const float* kv_vrow_ptr(int layer, int pos) { return g_kv.vrow(layer, pos); }
// KV-cache rows are D_KV wide with N_KV_HEADS heads (== D_MODEL/N_HEADS when not using GQA); the
// per-head rotation width is D_HEAD either way.
void kv_rope_rotate(float* row, int pos) { rope_row(row, pos, N_KV_HEADS, D_KV); }
void kv_splice_row(int layer, int pos, const float* k_canonical, const float* v) {
    float k[D_KV];
    for (int j = 0; j < D_KV; ++j) k[j] = k_canonical[j];
    rope_row(k, pos, N_KV_HEADS, D_KV);          // rotate the de-rotated canonical row to its splice position
    float* kc = g_kv.krow(layer, pos);
    float* vc = g_kv.vrow(layer, pos);
    for (int j = 0; j < D_KV; ++j) { kc[j] = k[j]; vc[j] = v[j]; }      // V is position-invariant, no rotation
}

}  // namespace cpu_detail
}  // namespace sub0
