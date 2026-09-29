#pragma once

#include "sub0/moe_io.hpp"
#include <sub0mempage/local_file_backend.hpp>
#include <sub0mempage/transfer_set.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace sub0::moeio {

/// Diagnostic name for a MemPage status (the pinned MemPage revision provides none of its own).
[[nodiscard]] constexpr std::string_view status_name(sub0mempage::Status status) noexcept {
    using enum sub0mempage::Status;
    switch (status) {
    case ok:               return "ok";
    case pending:          return "pending";
    case not_resident:     return "not_resident";
    case pool_exhausted:   return "pool_exhausted";
    case batch_too_large:  return "batch_too_large";
    case queue_exhausted:  return "queue_exhausted";
    case ticket_exhausted: return "ticket_exhausted";
    case out_of_range:     return "out_of_range";
    case empty_range:      return "empty_range";
    case invalid_argument: return "invalid_argument";
    case busy:             return "busy";
    case short_read:       return "short_read";
    case io_error:         return "io_error";
    case cancelled:        return "cancelled";
    case timeout:          return "timeout";
    case declined:         return "declined";
    }
    return "unknown";
}

/** Moves one selected-expert batch into registered raw plane buffers using MemPage.
 *  One backend worker per registered destination: each worker performs one blocking positional read at
 *  a time, so the worker count is the I/O queue depth, matching PlaneIo's whole-batch IOCP submission.
 *  The caller owns the buffers and keeps them alive until close/destruction finishes. Claims remain
 *  held after wait: retire_batch is called only after all compute readers have joined. open, close,
 *  submit and retire_batch require exclusive caller access; wait may run concurrently on live tags.
 *  Administrative open/close may allocate/block. Submission and retirement never allocate or do I/O.
 */
class MemPagePlaneIo {
public:
    /// Constructs a closed reader; no worker or storage is allocated until open.
    MemPagePlaneIo() = default;
    MemPagePlaneIo(const MemPagePlaneIo&) = delete;
    MemPagePlaneIo& operator=(const MemPagePlaneIo&) = delete;
    /// Drains writers before releasing registrations; callers must first join all compute readers.
    ~MemPagePlaneIo();

    /** Registers one immutable file and non-overlapping caller destinations (one per request tag).
     *  Any old session is closed first. Failure leaves the reader closed. Source identity/lifetime is
     *  the caller's responsibility; replacing or modifying the file during a session is unsupported.
     */
    [[nodiscard]] sub0mempage::Status open(const std::filesystem::path& path,
                                          std::span<const std::span<std::byte>> destinations);
    /** Administrative drain, including a partially submitted failed batch; leaves the reader closed.
     *  A worker-join or mutex failure terminates: freeing live destinations after failed drain is unsafe.
     */
    void close() noexcept;
    /** Submits a prevalidated batch with each request writing to its registered tag's destination.
     *  A failed partial submission retains accepted claims until close or successful retire_batch.
     *  No output from a failed batch is readable. An unretired batch returns busy without writing.
     */
    [[nodiscard]] sub0mempage::Status submit(std::span<const Request> requests) noexcept;
    /// Exact completion/error for a submitted tag; timeout preserves ownership and may be retried.
    [[nodiscard]] sub0mempage::Status wait(int tag, sub0mempage::Deadline deadline = std::nullopt) const noexcept;
    /** Releases a terminal batch after the caller has joined its last compute reader.
     *  Returns busy without releasing any claim if a writer is pending. No I/O wait is hidden here.
     */
    [[nodiscard]] sub0mempage::Status retire_batch() noexcept;

private:
    // Reverse destruction order: claims, registrations, then the worker backend. close drains first.
    std::unique_ptr<sub0mempage::LocalFileBackend> backend_;
    std::vector<std::unique_ptr<sub0mempage::TransferSet>> transfers_;
    std::vector<sub0mempage::Claim> claims_;
    std::vector<std::span<std::byte>> destinations_; // non-owning; registered caller buffers
    std::uint64_t source_bytes_ = 0;
    std::size_t submitted_ = 0;
    sub0mempage::Status batch_status_ = sub0mempage::Status::ok;
};

} // namespace sub0::moeio
