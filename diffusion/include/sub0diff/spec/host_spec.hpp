#pragma once

// host_spec.hpp (Ch28) — compile-time HOST hardware facts + a runtime `-verify`.
//
// The sub0llm-specialize philosophy (Ch27) moves a model's *shapes* to compile time.
// A diffusion inference binary is built for one machine, so the same idea applies to
// facts about that machine that "should never change": logical/physical core counts,
// the performance- vs efficiency-core split (heterogeneous CPUs), total RAM, cache-line
// size, and the widest SIMD the CPU supports. `sub0diff-specialize` detects these on the
// build host and emits a `HostSpec` struct of `static constexpr` members. The specialized
// binary then sizes thread pools / tiling against constants instead of querying the OS at
// startup — and exposes a `--verify` mode that re-detects the live machine and asserts the
// compiled-in assumptions still hold (warn on benign drift, abort on a SIGILL-class mismatch).

#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace sub0diff {

// ── runtime detection ───────────────────────────────────────────────────────────
// Facts about the machine the process is currently running on. Filled by detect_host().
struct HostFacts {
    unsigned logical_cores   = 0;   // OS-visible hardware threads (SMT counted)
    unsigned physical_cores  = 0;   // distinct physical cores
    unsigned perf_cores      = 0;   // "P" cores (highest efficiency class); == physical if homogeneous
    unsigned eff_cores       = 0;   // "E" cores (lower efficiency classes); 0 if homogeneous
    std::uint64_t ram_bytes  = 0;   // total physical RAM
    unsigned cacheline_bytes = 0;   // L1 line size (0 = unknown)
    unsigned simd_width_bits = 0;   // widest supported vector: 512 / 256 / 128 (0 = unknown)
    std::string cpu_brand;          // CPUID brand string (best-effort)
};

// Detect the running host. Platform-specific (Windows / POSIX); never throws — unknown
// fields are left 0 rather than failing.
[[nodiscard]] HostFacts detect_host() noexcept;

// ── compile-time contract ─────────────────────────────────────────────────────────
// A type H is a HostSpec if it exposes the host axes as compile-time constants.
template<class H>
concept HostSpec = requires {
    requires std::same_as<std::remove_cv_t<decltype(H::logical_cores)>,   unsigned>;
    requires std::same_as<std::remove_cv_t<decltype(H::physical_cores)>,  unsigned>;
    requires std::same_as<std::remove_cv_t<decltype(H::perf_cores)>,      unsigned>;
    requires std::same_as<std::remove_cv_t<decltype(H::eff_cores)>,       unsigned>;
    requires std::same_as<std::remove_cv_t<decltype(H::ram_bytes)>,       std::uint64_t>;
    requires std::same_as<std::remove_cv_t<decltype(H::simd_width_bits)>, unsigned>;
};

// ── verification ────────────────────────────────────────────────────────────────
enum class Severity { Ok, Warn, Fatal };

struct VerifyIssue {
    Severity    severity;
    std::string field;
    std::string detail;   // "compiled=8 live=6"
};

struct VerifyReport {
    std::vector<VerifyIssue> issues;
    [[nodiscard]] bool ok()      const noexcept { return !has(Severity::Fatal); }
    [[nodiscard]] bool perfect() const noexcept { return issues.empty(); }
    [[nodiscard]] bool has(Severity s) const noexcept {
        for (const auto& i : issues) if (i.severity == s) return true;
        return false;
    }
};

// Compare a compiled-in HostSpec against the live machine. Classification:
//   • SIMD width: live < compiled is FATAL (instructions would SIGILL); live >= compiled is Ok.
//   • core counts: any mismatch is WARN (thread-pool sizing is a perf assumption, not safety);
//     live having FEWER physical cores than compiled is FATAL (we'd oversubscribe a smaller box).
//   • RAM: live < compiled is FATAL (model/buffers may not fit); more RAM is Ok.
template<HostSpec H>
[[nodiscard]] VerifyReport verify_host(const HostFacts& live) {
    VerifyReport r;
    auto cmp_uint = [&](const char* field, std::uint64_t compiled, std::uint64_t got,
                        Severity if_less, Severity if_more) {
        if (got == compiled) return;
        r.issues.push_back({got < compiled ? if_less : if_more, field,
                            "compiled=" + std::to_string(compiled) +
                            " live=" + std::to_string(got)});
    };

    cmp_uint("simd_width_bits", H::simd_width_bits, live.simd_width_bits,
             Severity::Fatal, Severity::Warn);
    cmp_uint("physical_cores", H::physical_cores, live.physical_cores,
             Severity::Fatal, Severity::Warn);
    cmp_uint("logical_cores", H::logical_cores, live.logical_cores,
             Severity::Warn, Severity::Warn);
    cmp_uint("perf_cores", H::perf_cores, live.perf_cores, Severity::Warn, Severity::Warn);
    cmp_uint("eff_cores", H::eff_cores, live.eff_cores, Severity::Warn, Severity::Warn);
    cmp_uint("ram_bytes", H::ram_bytes, live.ram_bytes, Severity::Fatal, Severity::Ok);

    return r;
}

} // namespace sub0diff
