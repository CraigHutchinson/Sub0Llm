// sub0/hidden_dump.hpp -- "S0HD": the per-layer hidden-state container WP4f's llama.cpp comparison
// diffs (docs/WP4_SCOPE.md S6, WP4f).
//
// WHY A FORMAT AT ALL, AND WHY THIS ONE. WP4f's premise is "same tokens in, compare hidden states" --
// and the two sides of that comparison are produced by two DIFFERENT programs, one of which is not this
// repo and is not written in C++ by us. So the container has three jobs and no others:
//
//   1. Carry NAMED tensors with their shapes, because the comparison is per-layer and a positional
//      convention between two independently-written producers is exactly how a per-layer table ends up
//      silently comparing layer 1 against layer 2.
//   2. Carry the INPUT TOKEN IDS. "Same tokens in" is the whole premise of the stage, and a file that
//      states its own input lets sub0llm-hidden-diff VERIFY the premise instead of assuming it. This
//      matters more than it sounds: this engine's tokenizer is emphatically not Qwen's, so the ids are
//      hand-picked on both sides and a transcription slip is a live failure mode.
//   3. Be trivially writable by something that is not this header -- a 30-line Python/C snippet on the
//      llama.cpp side must be able to emit a valid file. Hence: no alignment padding, no compression,
//      no nested structure, fixed little-endian widths, sequential records.
//
// It is deliberately NOT the S0L5 model format nor an extension of it (AGENTS.md S3: checkpoint-format
// changes are this project's highest-blast-radius category, and this is a throwaway diagnostic artifact
// with none of a checkpoint's compatibility obligations).
//
// LAYOUT (little-endian throughout; f32 is IEEE-754 binary32):
//
//   offset  0 : char  magic[4]  = "S0HD"
//   offset  4 : u32   version   = 1
//   offset  8 : u32   n_tensors
//   offset 12 : u32   n_tokens
//   offset 16 : i32   tokens[n_tokens]        -- the exact input array this pass was run on
//   then n_tensors records, back to back, no padding between or within:
//               u32   name_len                -- bytes, NOT NUL-terminated
//               u8    name[name_len]
//               u32   rows                    -- token positions (T)
//               u32   cols                    -- feature width
//               f32   data[rows * cols]       -- row-major
//
// Duplicate names are permitted by the format and resolved by the reader as "first occurrence wins,
// later ones reported": a LoopSplit build re-executes the same layer index, so it would legitimately
// emit two `blk.3.out` tensors. The real Qwen4 axes have LoopSplit off, so this does not arise there --
// but the reader must not silently pick the wrong one if it ever does.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace sub0::hidden {

inline constexpr char     kMagic[4]  = {'S', '0', 'H', 'D'};
inline constexpr std::uint32_t kVersion = 1;

struct Tensor {
    std::string        name;
    std::uint32_t      rows = 0, cols = 0;
    std::vector<float> data;                       // rows * cols, row-major
    [[nodiscard]] std::size_t n() const { return static_cast<std::size_t>(rows) * cols; }
};

struct Dump {
    std::vector<int>    tokens;
    std::vector<Tensor> tensors;
    // First tensor whose name matches, or nullptr. `dup` (optional) receives how many further tensors
    // carried the same name, so a caller can report the ambiguity rather than resolve it silently.
    [[nodiscard]] const Tensor* find(const std::string& name, int* dup = nullptr) const {
        const Tensor* first = nullptr;
        int extra = 0;
        for (const Tensor& t : tensors) {
            if (t.name != name) continue;
            if (first) ++extra; else first = &t;
        }
        if (dup) *dup = extra;
        return first;
    }
};

// --- writing ---------------------------------------------------------------
// Streaming writer: the producer is a forward pass emitting tensors one at a time through a callback
// (sub0::forward_capture), so buffering the whole set before writing would double a multi-hundred-MB
// footprint on a machine that (docs/WP4_SCOPE.md WP4d) already runs with a ~3.7 GiB margin. The tensor
// count is not known until the pass ends, so it is BACK-PATCHED at close(): the header is written with
// a placeholder and rewritten in place. That means an aborted run leaves a file whose count is 0 --
// deliberately, since a truncated dump must not read as a short-but-valid one.
class Writer {
public:
    [[nodiscard]] bool open(const std::string& path, const int* tokens, int n_tokens) {
        os_.open(path, std::ios::binary | std::ios::trunc);
        if (!os_) return false;
        os_.write(kMagic, 4);
        put_u32(kVersion);
        put_u32(0);                                        // n_tensors, back-patched by close()
        put_u32(static_cast<std::uint32_t>(n_tokens));
        for (int i = 0; i < n_tokens; ++i) put_i32(tokens[i]);
        return static_cast<bool>(os_);
    }
    void add(const char* name, int rows, int cols, const float* data) {
        const auto len = static_cast<std::uint32_t>(std::strlen(name));
        put_u32(len);
        os_.write(name, static_cast<std::streamsize>(len));
        put_u32(static_cast<std::uint32_t>(rows));
        put_u32(static_cast<std::uint32_t>(cols));
        os_.write(reinterpret_cast<const char*>(data),
                  static_cast<std::streamsize>(static_cast<std::size_t>(rows) * cols * sizeof(float)));
        ++count_;
    }
    [[nodiscard]] bool close() {
        if (!os_) return false;
        os_.seekp(8, std::ios::beg);
        put_u32(count_);
        os_.flush();
        const bool ok = static_cast<bool>(os_);
        os_.close();
        return ok;
    }
    [[nodiscard]] std::uint32_t count() const { return count_; }

private:
    void put_u32(std::uint32_t v) { os_.write(reinterpret_cast<const char*>(&v), 4); }
    void put_i32(std::int32_t v)  { os_.write(reinterpret_cast<const char*>(&v), 4); }
    std::ofstream os_;
    std::uint32_t count_ = 0;
};

// --- reading ---------------------------------------------------------------
// Every failure sets `err` and returns false rather than throwing or asserting: this reads a file
// produced by a DIFFERENT program, so a malformed one is an expected outcome to report, not a bug.
[[nodiscard]] inline bool read(const std::string& path, Dump& out, std::string& err) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { err = "cannot open '" + path + "'"; return false; }
    const auto u32 = [&](std::uint32_t& v) { is.read(reinterpret_cast<char*>(&v), 4); return static_cast<bool>(is); };
    char magic[4]{};
    is.read(magic, 4);
    if (!is || std::memcmp(magic, kMagic, 4) != 0) {
        err = "'" + path + "' is not an S0HD dump (bad magic)";
        return false;
    }
    std::uint32_t version = 0, n_tensors = 0, n_tokens = 0;
    if (!u32(version) || !u32(n_tensors) || !u32(n_tokens)) { err = "truncated header"; return false; }
    if (version != kVersion) {
        err = "'" + path + "' is S0HD version " + std::to_string(version) + ", expected " +
              std::to_string(kVersion);
        return false;
    }
    if (n_tensors == 0) {
        err = "'" + path + "' declares 0 tensors -- an aborted/truncated dump (see Writer::close)";
        return false;
    }
    out.tokens.resize(n_tokens);
    for (std::uint32_t i = 0; i < n_tokens; ++i) {
        std::int32_t v = 0;
        is.read(reinterpret_cast<char*>(&v), 4);
        if (!is) { err = "truncated token array"; return false; }
        out.tokens[i] = v;
    }
    out.tensors.resize(n_tensors);
    for (std::uint32_t i = 0; i < n_tensors; ++i) {
        Tensor& t = out.tensors[i];
        std::uint32_t len = 0;
        if (!u32(len)) { err = "truncated at tensor " + std::to_string(i); return false; }
        if (len == 0 || len > 4096) { err = "implausible name length at tensor " + std::to_string(i); return false; }
        t.name.resize(len);
        is.read(t.name.data(), len);
        if (!u32(t.rows) || !u32(t.cols)) { err = "truncated shape for '" + t.name + "'"; return false; }
        t.data.resize(t.n());
        is.read(reinterpret_cast<char*>(t.data.data()),
                static_cast<std::streamsize>(t.data.size() * sizeof(float)));
        if (!is) { err = "truncated data for '" + t.name + "'"; return false; }
    }
    return true;
}

}  // namespace sub0::hidden
