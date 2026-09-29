#include "sub0/moe_io_mempage.hpp"

#include <cstdint>
#include <system_error>

namespace sub0::moeio {
using sub0mempage::Status;

MemPagePlaneIo::~MemPagePlaneIo() { close(); }

Status MemPagePlaneIo::open(const std::filesystem::path& path,
                           std::span<const std::span<std::byte>> destinations) {
    close();
    if (destinations.empty() || destinations.size() >= UINT32_MAX) return Status::invalid_argument;
    for (std::size_t i = 0; i < destinations.size(); ++i) {
        const auto start = reinterpret_cast<std::uintptr_t>(destinations[i].data());
        if (destinations[i].empty() || start == 0 || destinations[i].size() > UINTPTR_MAX - start)
            return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(destinations[j].data());
            if (start < other + destinations[j].size() && other < start + destinations[i].size())
                return Status::invalid_argument;
        }
    }
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error) return Status::io_error;
    if (bytes == 0) return Status::empty_range;
    // Build a temporary session so exceptions or registration failures leave this instance closed.
    MemPagePlaneIo next;
    auto backend = sub0mempage::LocalFileBackend::create({
        .workers = 2, .queue_capacity = static_cast<std::uint32_t>(destinations.size()), .max_sources = 1});
    if (!backend) return backend.error();
    next.backend_ = std::move(*backend);
    const auto source = static_cast<sub0mempage::SourceId>(1);
    if (const auto status = next.backend_->register_file(source, path); status != Status::ok) return status;
    next.source_bytes_ = bytes;
    next.destinations_.assign(destinations.begin(), destinations.end());
    next.claims_.resize(destinations.size());
    next.transfers_.reserve(destinations.size());
    for (auto destination : destinations) {
        auto transfer = sub0mempage::TransferSet::create(
            {.source = source, .source_bytes = bytes, .destination = destination, .max_claims = 1},
            sub0mempage::FillBackendRef(*next.backend_));
        if (!transfer) return transfer.error();
        next.transfers_.push_back(std::move(*transfer));
    }
    backend_ = std::move(next.backend_);
    transfers_ = std::move(next.transfers_);
    claims_ = std::move(next.claims_);
    destinations_ = std::move(next.destinations_);
    source_bytes_ = next.source_bytes_;
    return Status::ok;
}

void MemPagePlaneIo::close() noexcept {
    // shutdown delivers cancellation for queued requests and joins all active writers.
    if (backend_) backend_->shutdown();
    claims_.clear();
    transfers_.clear();
    destinations_.clear();
    backend_.reset();
    submitted_ = 0;
    source_bytes_ = 0;
    batch_status_ = Status::ok;
}

Status MemPagePlaneIo::submit(std::span<const Request> requests) noexcept {
    if (!backend_) return Status::invalid_argument;
    if (submitted_ != 0) return Status::busy;
    if (requests.empty()) return Status::empty_range;
    if (requests.size() > transfers_.size()) return Status::batch_too_large;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        if (request.dst != reinterpret_cast<std::uint8_t*>(destinations_[i].data()))
            return Status::invalid_argument;
        if (request.bytes == 0) return Status::empty_range;
        if (request.abs_off > source_bytes_ || request.bytes > source_bytes_ - request.abs_off ||
            request.bytes > destinations_[i].size()) return Status::out_of_range;
    }
    batch_status_ = Status::ok;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        auto claim = transfers_[i]->submit({requests[i].abs_off, requests[i].bytes}, 0);
        if (!claim) {
            batch_status_ = claim.error();
            return batch_status_;
        }
        claims_[i] = std::move(*claim);
        ++submitted_;
    }
    return Status::ok;
}

Status MemPagePlaneIo::wait(int tag, sub0mempage::Deadline deadline) const noexcept {
    if (batch_status_ != Status::ok) return batch_status_;
    if (tag < 0 || static_cast<std::size_t>(tag) >= submitted_) return Status::out_of_range;
    return claims_[static_cast<std::size_t>(tag)].wait(deadline);
}

Status MemPagePlaneIo::retire_batch() noexcept {
    for (std::size_t i = 0; i < submitted_; ++i)
        if (claims_[i].status() == Status::pending) return Status::busy;
    for (std::size_t i = 0; i < submitted_; ++i) claims_[i].reset();
    submitted_ = 0;
    batch_status_ = Status::ok;
    return Status::ok;
}

} // namespace sub0::moeio
