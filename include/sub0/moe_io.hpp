// sub0/moe_io.hpp -- explicit, batched, overlapped I/O for the S0Q1 sidecar's payload.
//
// WHY THIS EXISTS (docs/INDEPENDENT_REVIEW_BACKLOG.md B25, following on from B20/B21/B23; retargeted
// onto B31's fused no-transpose resolve and merged as a real toggle by B36). Decode's resolve
// (src/backends/cpu/decode.cpp) dequantizes a layer's EXPERTS_PER_TOK selected experts, but each
// plane's FIRST touch of its own bytes is a reactive mmap page fault (`moeq::Store::raw()`, backed by
// `sub0::FileMap`) -- serviced one fault at a time by whatever the OS's own fault-handling path can
// deliver. B21 measured that this leaves concurrent resolves nowhere near as concurrent on the disk as
// their thread count would suggest. B23 separately established the sidecar's own disk I/O is
// *structurally* unavoidable on SOME machine states (working set exceeds available RAM by
// construction), so the achievable goal is overlapping that unavoidable I/O with useful compute, not
// eliminating it -- exactly what this class is for. (B29 later pinned decode to ONE resolve thread --
// fewer threads measured faster once B27 established this host's own I/O is not the bottleneck at all
// -- so the overlap this class buys is now a single thread's compute overlapping WITH ITS OWN
// still-in-flight reads for later experts, not several threads' reads overlapping each other; see
// decode.cpp's own ParallelExperts::prefetch for how that plays out.)
//
// THE MECHANISM: the caller knows a whole layer's selected expert ids (the router's own top-k output)
// before touching a single byte of any of them. So instead of faulting each plane in reactively as the
// resolve loop reaches it, ONE thread issues every plane read (experts_per_tok * 3 = ~30 requests at
// the real axes) as an EXPLICIT, BATCHED `ReadFile` with `FILE_FLAG_OVERLAPPED`, all before any
// dequant starts -- reaching real queue depth ~30 immediately, rather than however many faults happen
// to be concurrent. Compute is then PIPELINED against the remaining in-flight reads: `wait()` blocks
// only on the ONE request its caller actually needs, but services (and remembers) ANY other request's
// completion that arrives while it waits -- the standard multi-consumer I/O-completion-port pattern --
// so an expert whose three planes land first can start its dequant+FFN compute immediately, while the
// other in-flight reads continue completing in the background. This is a superset of a strict
// wait-for-everything-then-compute design: batched submission still reaches full queue depth, and
// nothing here forces a thread to wait longer than the data it personally needs.
//
// SCOPE, deliberately narrow (AGENTS.md S8): this changes HOW the resolve path's bytes arrive, never
// what they mean. `moeq::dequantize_expert_source`'s own decode is untouched -- this class only ever
// hands back the SAME bytes `moeq::Store::raw()` would have, via a different code path, into a
// caller-owned buffer instead of a caller-owned pointer into a mapping. Read-only, no writes, no
// flush, matching file_map.hpp's own stated scope for the identical reason.
//
// PLATFORM: Windows (`FILE_FLAG_OVERLAPPED` + one shared I/O completion port) is the real, measured
// path -- see the B25/B36 backlog entries for the before/after numbers. A POSIX build falls back to a
// synchronous `pread` per request, issued lazily inside `wait()`: functionally correct (same bytes,
// same order), but NOT concurrent -- this project's development and target machine for this pass is
// Windows, and a real POSIX overlapped path (io_uring) is intentionally not implemented here.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace sub0::moeio {

// One explicit read request: an absolute file offset, a byte count, and a caller-owned destination
// buffer that must stay valid (and unaliased by any other in-flight request) until `wait()` for this
// request's tag returns.
struct Request {
    std::uint64_t  abs_off = 0;
    std::uint32_t  bytes   = 0;
    std::uint8_t*  dst     = nullptr;
};

// `MaxInFlight`: the largest `reqs.size()` any single `submit()` call will pass, baked at compile time
// (AGENTS.md S1) rather than sized per call -- the caller (decode.cpp) knows this is always
// EXPERTS_PER_TOK * moeq::PerExpert for this pass.
template <int MaxInFlight>
class PlaneIo {
public:
    static_assert(MaxInFlight >= 1, "at least one in-flight request -- a MoE-off build never opens this");

    PlaneIo() = default;
    ~PlaneIo() { close(); }
    PlaneIo(const PlaneIo&) = delete;
    PlaneIo& operator=(const PlaneIo&) = delete;

    bool open(const std::string& path, std::string& err) {
        close();
#if defined(_WIN32)
        file_ = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) { err = "moeio: cannot open " + path; return false; }
        // Deliberately buffered (no FILE_FLAG_NO_BUFFERING): measured WORSE on this project's own
        // target machine (flat ~1.44 GB/s regardless of queue depth) -- see the B25 backlog entry.
        port_ = ::CreateIoCompletionPort(file_, nullptr, 1, 0);
        if (port_ == nullptr) { err = "moeio: CreateIoCompletionPort failed for " + path; close(); return false; }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { err = "moeio: cannot open " + path; return false; }
#endif
        return true;
    }

    void close() {
#if defined(_WIN32)
        if (port_ != nullptr) { ::CloseHandle(port_); port_ = nullptr; }
        if (file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_); file_ = INVALID_HANDLE_VALUE; }
#else
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
    }

    bool ready() const {
#if defined(_WIN32)
        return file_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }

    // Issues every request's read. Must be called from ONE thread, and every request in `reqs` must
    // complete (via `wait()`, by tag == its index in `reqs`) before the NEXT `submit()` -- true by
    // construction at the one call site, whose `#pragma omp parallel` region provides the join barrier
    // between one layer's resolve and the next's `prefetch()`. Returns false (fatal, matching
    // `moe_resolve`'s own "a resolve failure is fatal, not a miss" contract) only on a real submission
    // error.
    bool submit(std::span<const Request> reqs, std::string& err) {
        const int n = static_cast<int>(reqs.size());
#if defined(_WIN32)
        for (int i = 0; i < n; ++i) {
            Op& op = ops_[static_cast<std::size_t>(i)];
            op = Op{};
            const std::uint64_t off = reqs[static_cast<std::size_t>(i)].abs_off;
            op.ov.Offset     = static_cast<DWORD>(off & 0xffffffffu);
            op.ov.OffsetHigh = static_cast<DWORD>(off >> 32);
            done_[static_cast<std::size_t>(i)].store(false, std::memory_order_relaxed);
            const BOOL ok = ::ReadFile(file_, reqs[static_cast<std::size_t>(i)].dst,
                                       reqs[static_cast<std::size_t>(i)].bytes, nullptr, &op.ov);
            // A synchronous TRUE (or a FALSE with ERROR_IO_PENDING) both still post a completion
            // packet to the port -- we never call GetOverlappedResult, only GetQueuedCompletionStatus
            // in wait() below, so both outcomes are handled by the exact same path.
            if (!ok && ::GetLastError() != ERROR_IO_PENDING) {
                err = "moeio: ReadFile failed for request " + std::to_string(i);
                return false;
            }
        }
#else
        for (int i = 0; i < n; ++i) {
            pending_[static_cast<std::size_t>(i)] = reqs[static_cast<std::size_t>(i)];
            done_[static_cast<std::size_t>(i)].store(false, std::memory_order_relaxed);
        }
#endif
        return true;
    }

    // Blocks until request `tag`'s read has completed. On Windows, services (and remembers) whichever
    // OTHER request's completion the OS happens to deliver first while waiting -- several callers may
    // `wait()` on different tags concurrently, each cooperatively pumping the shared port; this is the
    // pipelining property the whole class exists for (see the file header comment). Returns false on a
    // real I/O error for THIS tag.
    bool wait(int tag, std::string& err) {
#if defined(_WIN32)
        while (!done_[static_cast<std::size_t>(tag)].load(std::memory_order_acquire)) {
            DWORD bytes = 0; ULONG_PTR key = 0; LPOVERLAPPED lpo = nullptr;
            const BOOL ok = ::GetQueuedCompletionStatus(port_, &bytes, &key, &lpo, INFINITE);
            if (lpo == nullptr) {
                err = "moeio: GetQueuedCompletionStatus returned no OVERLAPPED (port error)";
                return false;
            }
            Op* op = reinterpret_cast<Op*>(lpo);   // OVERLAPPED is Op's first member -- same address
            const int idx = static_cast<int>(op - ops_.data());
            op->ok = (ok != 0);
            done_[static_cast<std::size_t>(idx)].store(true, std::memory_order_release);
        }
        if (!ops_[static_cast<std::size_t>(tag)].ok) {
            err = "moeio: read failed for request " + std::to_string(tag);
            return false;
        }
        return true;
#else
        if (!done_[static_cast<std::size_t>(tag)].load(std::memory_order_acquire)) {
            const Request& r = pending_[static_cast<std::size_t>(tag)];
            const long got = static_cast<long>(::pread(fd_, r.dst, r.bytes, static_cast<off_t>(r.abs_off)));
            done_[static_cast<std::size_t>(tag)].store(true, std::memory_order_release);
            if (got != static_cast<long>(r.bytes)) { err = "moeio: pread short or failed"; return false; }
        }
        return true;
#endif
    }

private:
#if defined(_WIN32)
    struct Op { OVERLAPPED ov{}; bool ok = true; };
    std::array<Op, static_cast<std::size_t>(MaxInFlight)>       ops_{};
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE port_ = nullptr;
#else
    std::array<Request, static_cast<std::size_t>(MaxInFlight)>  pending_{};
    int fd_ = -1;
#endif
    std::array<std::atomic<bool>, static_cast<std::size_t>(MaxInFlight)> done_{};
};

}  // namespace sub0::moeio
