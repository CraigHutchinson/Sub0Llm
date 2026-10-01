#pragma once

// OS memory residency: pin a region in RAM, and ask which of its pages are resident right now.
//
// An owned cache or buffer is only worth having if it stays in RAM. Unpinned private memory is written
// to the pagefile under memory pressure, so a "hit" on it becomes a disk read (measured 2026-10-01, see
// docs/STORAGE_STACK_PLAN.md "Residency intent and guards"). These helpers are the one place the
// per-platform mechanics live, shared by the routed-expert cache's storage and by the engine's
// end-of-run residency report.

#include <cstddef>
#include <cstdint>

namespace sub0::residency {

/// Page counts over one region. `locked` counts resident pages the OS cannot page out.
struct Report {
    std::uint64_t pages = 0;
    std::uint64_t resident = 0;
    std::uint64_t locked = 0;
    /// Every page resident and locked: nothing of the region can have reached the pagefile.
    [[nodiscard]] bool complete() const noexcept { return pages != 0 && locked == pages; }
    /// Share of the region's pages resident, 0..1.
    [[nodiscard]] double resident_fraction() const noexcept {
        return pages ? static_cast<double>(resident) / static_cast<double>(pages) : 0.0;
    }
};

/** Locks `bytes` at `data` into RAM. Windows first grows the working-set quota by `bytes`
 *  (VirtualLock is bounded by it); Linux first raises the soft RLIMIT_MEMLOCK to the hard limit.
 *  @return false if the OS refused; the region is then unchanged and pageable.
 *  @note Faults in every page, so it takes time proportional to `bytes`.
 */
[[nodiscard]] bool pin(void* data, std::size_t bytes) noexcept;

/// Gives back the working-set quota pin() took (Windows); the lock itself goes with the memory.
void release_quota(std::size_t bytes) noexcept;

/** Asks the OS which pages of the region are resident, and which are locked, right now.
 *  Windows: QueryWorkingSetEx, Valid and Locked per page. Linux: mincore, with VmLck as a
 *  process-wide lower bound for `locked_hint`. macOS: mincore; it exposes no lock state, so
 *  `locked_hint` is trusted.
 *  @param locked_hint Whether the caller pinned this region (used where the OS cannot say per page).
 *  @note Administrative: walks every page in fixed-size chunks; allocates nothing.
 */
[[nodiscard]] Report query(const void* data, std::size_t bytes, bool locked_hint) noexcept;

} // namespace sub0::residency
