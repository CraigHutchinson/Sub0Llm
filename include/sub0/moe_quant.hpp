// sub0/moe_quant.hpp -- the `S0Q1` quantized-resident routed-expert sidecar: its on-disk format, its
// reader, and the fixed-capacity dequantize-on-demand cache the engine resolves experts through.
//
// WHY THIS EXISTS (docs/WP4_SCOPE.md WP4e, rescoped 2026-09-04). WP4c's transplant dequantizes EVERY
// tensor to f32 and writes one flat PARAM_LAYOUT-ordered blob. That is correct for its own purpose -- a
// quantization-noise-free correctness gate at 4-layer scale, where 43 GiB happens to fit -- and it does
// not scale: the same shape at the real 48 layers is ~500 GB of f32, which defeats the entire reason a
// 1-2 bit tier was downloaded. The mass is all in one place: the 512 ROUTED experts are 97.3% of every
// layer's floats (WP4c's own measured table). So they, and ONLY they, stay resident in their native
// GGUF bytes here, and the engine dequantizes the `experts_per_tok` (10) actually selected per token
// into a small reused scratch buffer, consumes them, and discards them -- which is what llama.cpp
// itself does, and is the same design decision as choosing a quantized file format at all.
//
// WHAT STAYS EXACTLY AS WP4c HAS IT, and why each is not a gap:
//   * the MoeRouter -- `blk.N.ffn_gate_inp.weight` is F32 in the real file (docs/WP4_SCOPE.md S3a-bis),
//     and it is DENSE-read over all 512 experts for every token, so it must be resident and there is no
//     dequantization to skip. It has no quantized form to keep.
//   * the SHARED expert (gate/up/down + its gate projection) -- always-on for every token, one copy per
//     layer, 3.9 MiB of f32. Nothing is saved by making it on-demand and it would only add a second
//     code path through moe::expert_ffn_row.
//   * embeddings, GDN, Gated Residual, QSA -- unchanged, still in the f32 S0L5 blob.
//
// --- THE TWO-FILE RELATIONSHIP, stated explicitly ------------------------------------------------
//
// A MOE_QUANT_EXPERTS build's weights are TWO files, not one:
//
//     <model>.bin        the existing S0L5 header + flat PARAM_LAYOUT-ordered f32 blob -- but with NO
//                        MoeGate/MoeUp/MoeDown tensors in the layout at all (make_param_layout() emits
//                        none under MOE_QUANT_EXPERTS, exactly as blocker D stopped emitting Ln1/Ln2)
//     <model>.bin.moeq   THIS format: every routed expert's own gate/up/down, in its own native GGUF
//                        encoding, byte-for-byte as the source shard held it
//
// The `.moeq` path is DERIVED from the model path rather than being a second CLI argument (AGENTS.md
// S8: no surface nothing needs). The pairing is checked, not assumed: `model_param_floats` below pins
// which S0L5 blob this sidecar belongs to, so a sidecar from a different build cannot be silently
// loaded next to the wrong weights -- the same job ModelHeader::param_floats already does for the blob.
//
// This is deliberately NOT folded into the S0L5 file as a trailing section (AGENTS.md S3 -- binary
// formats here are the highest-blast-radius change category). S0L5 is `header + float[PARAM_FLOATS] +
// three u64 trailers`, read by engine_core.cpp's load_model into ONE contiguous arena; a variable-length
// mixed-format section in the middle of that would change how every existing checkpoint is read. A
// separate file with its own magic changes nothing about S0L5 at all, and a build that does not use it
// never opens it.
//
// --- WHY THE BYTES ARE THE SOURCE'S, VERBATIM ----------------------------------------------------
//
// The sidecar stores each expert's slice of `blk.N.ffn_{gate,up,down}_exps.weight` EXACTLY as the GGUF
// shard holds it -- same encoding, same bytes, copied not re-encoded. Two consequences, both load-
// bearing:
//   1. Per-layer mixed quantization (docs/WP4_SCOPE.md S3a-bis) is carried, not flattened: each Desc
//      records its OWN `type_raw`, so layer 0's IQ1_S gate and layer 1's IQ2_XXS gate coexist with no
//      per-role assumption anywhere. Re-encoding to one format would have been a second, unvalidated
//      quantizer and would have changed the values.
//   2. `dequantize_expert()` below performs the SAME two operations, on the SAME bytes, in the SAME
//      order as tools/sub0llm-transplant.cpp's own f32 path: gguf::to_f32 on the slice, then
//      transplant::transpose_out_in. That is what makes WP4e's gate -- bitwise-identical engine output
//      between the all-f32-resident path and this one -- a structural property rather than a hope.
//
// Engine-free (gguf.hpp + transplant.hpp + file_map.hpp only, no sub0_config.hpp, no layout.hpp), like
// all of those, so the format and the cache are unit-testable without compiling a model at the real
// axes.

#pragma once

#include "sub0/file_map.hpp"
#include "sub0/gguf.hpp"
#include "sub0/transplant.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sub0::moeq {

// --- on-disk format -------------------------------------------------------------------------------

// Fixed-size, written directly to disk. Pinned by static_assert below for the same reason
// ModelHeader's own size is (AGENTS.md S3): a field change that moves it must be a build error, not a
// file that reads as garbage.
struct Header {
    char           magic[4] = {'S', '0', 'Q', '1'};
    std::uint32_t  version  = 1;
    std::int32_t   n_layers = 0, num_experts = 0, d_model = 0, d_ff = 0;
    std::uint64_t  n_tensors = 0;             // n_layers * num_experts * 3
    std::uint64_t  data_off  = 0;             // byte offset from file start to the payload
    std::uint64_t  data_bytes = 0;
    // The PARAM_FLOATS of the S0L5 blob this sidecar belongs beside. Not decoration: it is what makes
    // "these two files are one model" checkable instead of assumed.
    std::uint64_t  model_param_floats = 0;
};
static_assert(sizeof(Header) == 56, "the S0Q1 header's on-disk size must not change");
static_assert(alignof(Header) == 8);

// One routed expert tensor. `in_f`/`out_f` are the GGUF DECLARED extents (ne[0], ne[1]) of the source
// plane, i.e. this destination's [rows=in_f, cols=out_f] after the transpose -- the same naming
// tools/sub0llm-transplant.cpp's own ne_in()/ne_out() use, and named the same way for the same reason
// (dims[0] is the input width is the most inversion-prone line in either file).
struct Desc {
    std::uint32_t type_raw = 0;      // raw GGML type id -- PER TENSOR, never per role (S3a-bis)
    std::uint32_t in_f = 0;
    std::uint32_t out_f = 0;
    std::uint32_t reserved = 0;
    std::uint64_t off = 0;           // byte offset from Header::data_off
    std::uint64_t bytes = 0;         // encoded length
};
static_assert(sizeof(Desc) == 32, "the S0Q1 descriptor's on-disk size must not change");

// Which of an expert's three tensors. The order matches make_param_layout()'s own per-expert triple
// (MoeGate, MoeUp, MoeDown), so a walk of this table and a walk of PARAM_LAYOUT stay in step.
enum Which : int { Gate = 0, Up = 1, Down = 2, PerExpert = 3 };

inline constexpr std::uint64_t desc_index(int n_experts, int layer, int expert, int which) {
    return ((static_cast<std::uint64_t>(layer) * n_experts) + static_cast<std::uint64_t>(expert))
               * PerExpert + static_cast<std::uint64_t>(which);
}

// The encoded byte range of expert `e`'s own [out, in] plane inside a 3-D `[in, out, n_experts]` GGUF
// expert-stack tensor. Shared by the writer (which copies these bytes) and by any consumer that wants
// to check them, so the two cannot disagree.
//
// Every quantized format here is block-structured, so a per-expert slice is only addressable at all if
// it lands on a block boundary. It does at the real axes (2560*640 = 1,638,400 elements per expert:
// 6,400 whole 256-element blocks, or 51,200 whole 32-element ones) -- but that is CHECKED, not assumed:
// a misaligned slice would decode a neighbouring expert's values with no other symptom. Returns
// {0, 0} when the type is unknown or the slice is not block-aligned.
struct ByteRange { std::uint64_t off = 0, bytes = 0; };
inline ByteRange expert_byte_range(std::uint32_t type_raw, std::uint64_t elems_per_expert, int expert) {
    const gguf::BlockSpec b = gguf::block_spec(type_raw);
    if (b.elems == 0 || elems_per_expert == 0) return {};
    if (elems_per_expert % b.elems != 0) return {};
    const std::uint64_t blocks = elems_per_expert / b.elems;
    return {static_cast<std::uint64_t>(expert) * blocks * b.bytes, blocks * b.bytes};
}

// --- the one decode, shared by every consumer -------------------------------------------------------

// Raw encoded bytes -> this project's own [rows=in_f, cols=out_f] f32 plane.
//
// It is TWO steps and they are exactly tools/sub0llm-transplant.cpp's own, in the same order, because
// producing the identical floats is the whole point (see this file's header comment). `raw_scratch` is
// a caller-owned reused buffer holding the intermediate SOURCE-order f32 -- caller-owned rather than
// local because this runs per selected expert per token (AGENTS.md S1); `dst` is `in_f * out_f` floats.
inline bool dequantize_expert(const Desc& d, std::span<const std::uint8_t> raw, float* dst,
                               std::vector<float>& raw_scratch) {
    gguf::TensorInfo t;
    t.type_raw = d.type_raw;
    t.dims = {static_cast<std::uint64_t>(d.in_f) * d.out_f};
    if (!gguf::to_f32(t, raw, raw_scratch)) return false;
    transplant::transpose_out_in(raw_scratch.data(), static_cast<int>(d.out_f),
                                  static_cast<int>(d.in_f), dst);
    return true;
}

// Raw encoded bytes -> GGUF SOURCE-order f32 plane, i.e. WITHOUT the transpose step dequantize_expert
// above applies (B31, docs/INDEPENDENT_REVIEW_BACKLOG.md). `dst` is a caller-owned std::vector sized
// exactly `in_f*out_f` floats by the caller once (ExpertCacheSource::allocate below) -- gguf::to_f32's
// own decoders all call resize()/assign() on their `out` vector, so handing them the PERSISTENT slot
// plane directly (rather than a scratch buffer later copied or transposed elsewhere) means the dequant
// write is the ONLY DRAM touch this step makes: no separate raw_scratch materialization, no transpose
// read-back, no transpose write. Paired with moe::expert_ffn_row_source (moe_math.hpp), which reads
// exactly this layout and is proven (see that function's own comment) to produce bit-identical output to
// expert_ffn_row on dequantize_expert's transposed planes.
inline bool dequantize_expert_source(const Desc& d, std::span<const std::uint8_t> raw,
                                      std::vector<float>& dst) {
    gguf::TensorInfo t;
    t.type_raw = d.type_raw;
    t.dims = {static_cast<std::uint64_t>(d.in_f) * d.out_f};
    return gguf::to_f32(t, raw, dst);
}

// --- reader ------------------------------------------------------------------------------------------

// Holds the sidecar's descriptor table and a read-only MAPPING of its encoded payload, in its native
// form. Opened once at load time, never written; every resolve reads a byte range out of the mapping.
//
// WHY MAPPED AND NOT READ (docs/WP4_SCOPE.md WP5b; WP4e's own "deliberately still open" list named this
// as the change the full model would need). The payload used to be read into an owned buffer, which at
// the 4-layer sub-stack is 3.17 GiB and invisible. At the real 48 layers it is ~38 GiB, and an eager
// read of that plus the 18.3 GiB f32 backbone plus a 14 GiB activation arena exceeds this machine's
// free RAM before a single token is embedded. A mapping changes the arithmetic rather than shaving it:
//   * a forward pass dequantizes only the experts it ROUTES to (EXPERTS_PER_TOK of NUM_EXPERTS per
//     token per layer), so the vast majority of those 38 GiB is never touched and never faulted in;
//   * the pages that ARE touched are file-backed and evictable, so they count against the working set
//     the OS can reclaim under pressure, not against committed private bytes that it cannot.
// Nothing above `raw()` can tell: the accessor still hands back a `span<const uint8_t>` over bytes that
// are addressable for the life of the Store, which is the only property `dequantize_expert` and the
// resolve pool ever relied on. That claim is kept true rather than merely asserted --
// tests/moe_quant_tests.cpp reproduces the OLD eager read independently and requires bit-identical
// expert planes AND bit-identical moe::expert_ffn_row output from the two.
//
// The descriptor table is COPIED out of the mapping rather than pointed at: it is small (32 bytes per
// routed-expert plane -- 2.25 MiB even at 48 layers), and copying it sidesteps any question about
// reading a `Desc` through a pointer into mapped bytes whose alignment is the file's business.
class Store {
public:
    // Returns false and fills `err` on any problem; never throws, never partially initializes.
    bool open(const std::string& path, std::string& err) {
        map_.close();
        descs_.clear();
        h_ = Header{};
        data_ = nullptr;

        FileMap m;
        if (!m.open(path, err)) return false;
        const std::uint8_t* p = m.data();
        const std::uint64_t file_bytes = m.size();

        if (file_bytes < sizeof(Header)) { err = path + ": truncated header"; return false; }
        std::memcpy(&h_, p, sizeof h_);
        if (std::memcmp(h_.magic, "S0Q1", 4) != 0) { err = path + ": not an S0Q1 sidecar"; return false; }
        if (h_.version != 1) { err = path + ": unsupported S0Q1 version"; return false; }
        if (h_.n_tensors != static_cast<std::uint64_t>(h_.n_layers) * h_.num_experts * PerExpert) {
            err = path + ": tensor count does not match n_layers * num_experts * 3";
            return false;
        }
        const std::uint64_t table_bytes = h_.n_tensors * sizeof(Desc);
        if (file_bytes < sizeof(Header) + table_bytes) {
            err = path + ": truncated descriptor table";
            return false;
        }
        descs_.resize(static_cast<std::size_t>(h_.n_tensors));
        std::memcpy(descs_.data(), p + sizeof(Header), static_cast<std::size_t>(table_bytes));

        // The payload must lie wholly inside the file. This is the check the eager read got for free
        // from a short read; with a mapping it has to be explicit, because addressing past the end of a
        // view is an access violation rather than a `gcount()` shortfall.
        if (h_.data_off > file_bytes || h_.data_bytes > file_bytes - h_.data_off) {
            err = path + ": the payload runs past the end of the file";
            return false;
        }
        // And every descriptor must lie inside the payload. Checked once here so no resolve has to.
        for (const Desc& d : descs_)
            if (d.off > h_.data_bytes || d.bytes > h_.data_bytes - d.off) {
                err = path + ": a descriptor's byte range runs past the payload";
                return false;
            }
        map_ = std::move(m);
        data_ = map_.data() + h_.data_off;
        return true;
    }

    const Header& header() const { return h_; }
    bool loaded() const { return data_ != nullptr; }

    const Desc& desc(int layer, int expert, int which) const {
        return descs_[static_cast<std::size_t>(desc_index(h_.num_experts, layer, expert, which))];
    }
    std::span<const std::uint8_t> raw(const Desc& d) const {
        return {data_ + d.off, static_cast<std::size_t>(d.bytes)};
    }

    // The sidecar's encoded payload size. Named for what it measures -- how many bytes of routed
    // expert this model has -- which is unchanged by the payload now being mapped rather than read;
    // what changed is how much of it a given run actually faults in, which only a real RSS measurement
    // can say (docs/QWEN4_MEMORY_ORCHESTRATION.md's "a prediction is not a measurement").
    std::uint64_t resident_bytes() const { return h_.data_bytes; }

private:
    Header                    h_{};
    std::vector<Desc>         descs_;
    FileMap                   map_;
    const std::uint8_t*       data_ = nullptr;   // map_.data() + h_.data_off, or null
};

// --- the fixed-capacity resolve pool ------------------------------------------------------------------
//
// `Slots` concurrently-live dequantized experts, sized at COMPILE TIME by the caller (AGENTS.md S1/S2),
// against 512 * n_layers that exist. One slot holds a whole expert -- gate, up and down together --
// because moe::expert_ffn_row needs all three at once and consumes them in a single call.
//
// HOW BIG DOES IT ACTUALLY NEED TO BE? ONE. expert_ffn_row is called, finishes, and its weights are dead;
// nothing holds a resolved pointer across the next resolve. Every slot beyond the first is therefore a
// pure CACHE, not a correctness requirement -- which is why replacement policy cannot affect the answer,
// and why this class can be a plain round-robin with no invalidation: the weights are immutable for the
// life of a forward-only run (docs/WP4_SCOPE.md S5 -- all four mechanisms abort in backward_node), so a
// cached expert can never be stale.
//
// The cache earns its slots on real routing, not in principle: a single op_moe call runs T rows through
// the SAME layer, each picking 10 of 512, so an expert selected by two rows of one batch is dequantized
// once instead of twice. It is keyed by (layer, expert) rather than expert alone so it survives across
// layers without a flush.
template <int Slots, std::size_t SlotFloats>
class ExpertCache {
public:
    static_assert(Slots >= 1, "at least one slot -- the resolve path has nowhere to put an expert otherwise");
    static constexpr std::size_t kFloats = SlotFloats;

    struct Resolved { const float* gate; const float* up; const float* down; };

    // Heap, allocated once: at the real axes one slot is 3 * 2560 * 640 floats = 18.75 MiB, which is
    // neither a stack array nor something that may sit in the DLL's static image (backend_cpu.cpp's own
    // g_param_data comment: zero-init BSS this size pushes SizeOfImage past what Windows will map).
    void allocate() {
        if (!pool_) pool_ = std::make_unique<float[]>(static_cast<std::size_t>(Slots) * PerExpert * SlotFloats);
        if (raw_scratch_.size() < SlotFloats) raw_scratch_.resize(SlotFloats);
    }

    // Dequantizes expert (layer, expert) if it is not already in a slot, and returns pointers to its
    // three planes. Returns {nullptr,...} only if a decode fails, which means an unsupported GGML type
    // or a corrupt payload -- the caller must treat that as fatal, not as a miss.
    Resolved resolve(const Store& store, int layer, int expert) {
        const std::uint64_t key = (static_cast<std::uint64_t>(layer) << 32)
                                  | static_cast<std::uint32_t>(expert);
        for (int s = 0; s < Slots; ++s)
            if (live_[static_cast<std::size_t>(s)] && key_[static_cast<std::size_t>(s)] == key) {
                ++hits_;
                return at(s);
            }
        const int s = next_;
        next_ = (next_ + 1) % Slots;
        live_[static_cast<std::size_t>(s)] = false;
        for (int w = 0; w < PerExpert; ++w) {
            const Desc& d = store.desc(layer, expert, w);
            if (!dequantize_expert(d, store.raw(d), plane(s, w), raw_scratch_))
                return {nullptr, nullptr, nullptr};
        }
        key_[static_cast<std::size_t>(s)] = key;
        live_[static_cast<std::size_t>(s)] = true;
        ++misses_;
        return at(s);
    }

    std::uint64_t hits() const { return hits_; }
    std::uint64_t misses() const { return misses_; }
    static constexpr std::uint64_t pool_bytes() {
        return static_cast<std::uint64_t>(Slots) * PerExpert * SlotFloats * sizeof(float);
    }

private:
    float* plane(int slot, int which) {
        return pool_.get() + (static_cast<std::size_t>(slot) * PerExpert + static_cast<std::size_t>(which))
                                 * SlotFloats;
    }
    Resolved at(int s) { return {plane(s, Gate), plane(s, Up), plane(s, Down)}; }

    std::unique_ptr<float[]>            pool_;
    std::vector<float>                  raw_scratch_;   // source-order f32, reused (AGENTS.md S1)
    std::array<std::uint64_t, Slots>    key_{};
    std::array<bool, Slots>             live_{};
    int                                  next_ = 0;
    std::uint64_t                        hits_ = 0, misses_ = 0;
};

// --- fused, no-transpose resolve pool (B31) -------------------------------------------------------
//
// ExpertCache above resolves an expert in THREE touches of DRAM per plane: gguf::to_f32 writes a
// scratch buffer in GGUF source order, then transplant::transpose_out_in reads that scratch buffer AND
// writes the slot's persistent (transposed) plane -- a real, measured cost (docs/
// INDEPENDENT_REVIEW_BACKLOG.md B30's own byte accounting: ~3 full round trips over each expert's
// ~19.66 MB plane-set before moe::expert_ffn_row's own read). This class instead dequantizes DIRECTLY
// into each slot's OWN persistent plane, in GGUF's native SOURCE order, with no transpose at all --
// dequantize_expert_source above hands gguf::to_f32 the slot's own std::vector as its `out` parameter,
// so the dequant write IS the plane's only DRAM touch. Pairs with moe::expert_ffn_row_source
// (moe_math.hpp), which consumes exactly this layout and is proven bit-identical to expert_ffn_row's
// own output (see that function's own comment for the summation-order argument).
//
// Otherwise structurally identical to ExpertCache (same round-robin/key/live bookkeeping, same "only
// one slot is a correctness requirement, the rest are a pure cache" reasoning) -- kept as a SEPARATE
// class rather than a mode flag on ExpertCache so the batched path (op_moe, 8 real cross-row cache hits)
// and every existing test/tool that depends on ExpertCache's current (transposed) contract stay
// completely untouched (AGENTS.md S10 -- this is a new consumer of dequantize_expert_source, not a
// changed contract on an existing one). Currently wired only to decode's own single-slot pool
// (src/backends/cpu/decode.cpp's MoeDecodeThread) -- see that file for why decode's hit rate against
// ANY pool bigger than one slot is provably zero.
template <int Slots, std::size_t SlotFloats>
class ExpertCacheSource {
public:
    static_assert(Slots >= 1, "at least one slot -- the resolve path has nowhere to put an expert otherwise");
    static constexpr std::size_t kFloats = SlotFloats;

    struct Resolved { const float* gate; const float* up; const float* down; };

    // Each plane is its own std::vector, sized ONCE here and never resized again (dequantize_expert_source
    // always writes exactly SlotFloats elements, so every later gguf::to_f32 call inside resolve() is a
    // same-size resize -- no reallocation, no new DRAM traffic beyond the dequant write itself).
    void allocate() {
        if (plane_[0][0].size() != SlotFloats)
            for (auto& slot : plane_) for (auto& p : slot) p.assign(SlotFloats, 0.f);
    }

    Resolved resolve(const Store& store, int layer, int expert) {
        const std::uint64_t key = (static_cast<std::uint64_t>(layer) << 32)
                                  | static_cast<std::uint32_t>(expert);
        for (int s = 0; s < Slots; ++s)
            if (live_[static_cast<std::size_t>(s)] && key_[static_cast<std::size_t>(s)] == key) {
                ++hits_;
                return at(s);
            }
        const int s = next_;
        next_ = (next_ + 1) % Slots;
        live_[static_cast<std::size_t>(s)] = false;
        for (int w = 0; w < PerExpert; ++w) {
            const Desc& d = store.desc(layer, expert, w);
            if (!dequantize_expert_source(d, store.raw(d), plane_[static_cast<std::size_t>(s)][static_cast<std::size_t>(w)]))
                return {nullptr, nullptr, nullptr};
        }
        key_[static_cast<std::size_t>(s)] = key;
        live_[static_cast<std::size_t>(s)] = true;
        ++misses_;
        return at(s);
    }

    std::uint64_t hits() const { return hits_; }
    std::uint64_t misses() const { return misses_; }
    static constexpr std::uint64_t pool_bytes() {
        return static_cast<std::uint64_t>(Slots) * PerExpert * SlotFloats * sizeof(float);
    }

private:
    Resolved at(int s) {
        auto& slot = plane_[static_cast<std::size_t>(s)];
        return {slot[Gate].data(), slot[Up].data(), slot[Down].data()};
    }

    std::array<std::array<std::vector<float>, PerExpert>, Slots> plane_{};
    std::array<std::uint64_t, Slots>    key_{};
    std::array<bool, Slots>             live_{};
    int                                  next_ = 0;
    std::uint64_t                        hits_ = 0, misses_ = 0;
};

}  // namespace sub0::moeq
