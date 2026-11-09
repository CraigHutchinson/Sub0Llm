// machine_compute_roofs_tests.cpp -- sub0/measure/compute_roofs.hpp and sub0/measure/ceilings.hpp (issue #78,
// increment 3). Engine-free: standard library, pthreads through std::thread, and sched_setaffinity on Linux. Each
// kernel case runs on one or two threads with a few thousand iterations, so every case stays well under 3 seconds.
// A kernel whose ISA is not compiled into this build is a SKIP with its reason, never a pass. The planner cases read a
// fabricated 2-CCD topology (the same shape as machine_roofs_tests.cpp) through readMachineFacts.

#include <catch2/catch_test_macros.hpp>

#include "sub0/measure/ceilings.hpp"
#include "sub0/measure/compute_roofs.hpp"
#include "sub0/measure/machine_facts.hpp"
#include "sub0/measure/memory_roofs.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace sub0::measure;

namespace {

namespace fs = std::filesystem;

struct ScratchDir {
    fs::path path;
    ScratchDir() {
        std::random_device rd;
        path = fs::temp_directory_path() /
               ("sub0_machine_compute_test_" + std::to_string(rd()) + "_" + std::to_string(rd()));
        fs::create_directories(path);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void writeFile(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream os(p, std::ios::binary);
    os << content;
}

// Topology and L3 files only: 2 CCDs of 8 cores, SMT2, logical N and N+16 are one core, CCD0 = cores 0-7,
// CCD1 = cores 8-15.
void buildTwoCcdTopology(const fs::path& root) {
    for (int c = 0; c < 32; ++c) {
        const std::string cpu = "sys/devices/system/cpu/cpu" + std::to_string(c) + "/";
        const int core = c % 16;
        const std::string pair = std::to_string(core) + "," + std::to_string(core + 16);
        const std::string ccd = core < 8 ? "0-7,16-23" : "8-15,24-31";
        writeFile(root / (cpu + "topology/core_id"), std::to_string(core) + "\n");
        writeFile(root / (cpu + "topology/physical_package_id"), "0\n");
        writeFile(root / (cpu + "topology/thread_siblings_list"), pair + "\n");
        writeFile(root / (cpu + "cache/index0/level"), "3\n");
        writeFile(root / (cpu + "cache/index0/type"), "Unified\n");
        writeFile(root / (cpu + "cache/index0/size"), "32768K\n");
        writeFile(root / (cpu + "cache/index0/shared_cpu_list"), ccd + "\n");
    }
}

std::vector<int> range(int lo, int hi) {
    std::vector<int> v;
    for (int i = lo; i <= hi; ++i) v.push_back(i);
    return v;
}

const Placement* find(const std::vector<Placement>& ps, const std::string& name) {
    for (const Placement& p : ps)
        if (p.name == name) return &p;
    return nullptr;
}

// A short compute run: 2 ms passes, one repeat, two thread counts.
ComputeConfig tinyComputeConfig() {
    ComputeConfig cfg;
    cfg.budgetSeconds = 3.0;
    cfg.repeats = 1;
    cfg.threadCounts = {1, 2};
    cfg.passSeconds = 0.002;
    return cfg;
}

// Brace and bracket balance outside of strings: a cheap well-formedness check for the writers.
bool balanced(const std::string& s) {
    int braces = 0;
    int brackets = 0;
    bool inString = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (inString) {
            if (c == '\\') ++i;
            else if (c == '"') inString = false;
            continue;
        }
        if (c == '"') inString = true;
        else if (c == '{') ++braces;
        else if (c == '}') --braces;
        else if (c == '[') ++brackets;
        else if (c == ']') --brackets;
        if (braces < 0 || brackets < 0) return false;
    }
    return braces == 0 && brackets == 0 && !inString;
}

// Verified kernel run on `cpus`: every repeat's lanes match the exact expected values.
void expectVerified(ComputeKernel k, const std::vector<int>& cpus) {
    ComputeRequest req;
    req.kernel = k;
    req.cpus = cpus;
    req.iterations = 2000;
    req.repeats = 2;
    const ComputeMeasurement m = measureComputeKernel(req);
    INFO(computeKernelName(k) << " on " << cpus.size() << " thread(s): " << m.verifyFailure);
    CHECK(m.verified);
    CHECK(m.repeats == 2);
    CHECK(m.iterations == 2000);
    REQUIRE(m.medianRate.value.has_value());
    CHECK(*m.medianRate.value > 0.0);
    CHECK(*m.minRate.value <= *m.maxRate.value);
}

// The corrupted accumulator (test hook) must make the point invalid, with a reason.
void expectCorruptionCaught(ComputeKernel k) {
    ComputeRequest req;
    req.kernel = k;
    req.cpus = {0};
    req.iterations = 2000;
    req.repeats = 1;
    req.corruptOutput = true;
    const ComputeMeasurement m = measureComputeKernel(req);
    INFO(computeKernelName(k));
    CHECK_FALSE(m.verified);
    CHECK_FALSE(m.verifyFailure.empty());
    CHECK_FALSE(m.medianRate.value.has_value());
    CHECK_FALSE(m.medianRate.reason.empty());
}

}  // namespace

TEST_CASE("compute_roofs: the ISA table matches the compiled macros and the documented roof factors",
          "[machine_compute_roofs]") {
    const std::vector<ComputeIsa> table = computeIsaTable();
    REQUIRE(table.size() == 3);
    for (const ComputeIsa& row : table) {
        CHECK(row.compiled == (row.isa != "none"));
        CHECK((row.compiled || !row.reason.empty()));
        CHECK((!row.compiled || row.reason.empty()));
    }
    CHECK(table[0].kernel == "fma_f32");
    CHECK(table[0].opsPerCycleCore == 32.0);
    CHECK(table[0].opsPerInstruction == 2 * fmaLanes());
    CHECK(table[1].kernel == "dot_i8");
    CHECK(table[1].opsPerCycleCore == 128.0);
    CHECK(table[1].opsPerInstruction == 128);   // 2 x 4 x 16 lanes
    CHECK(table[2].kernel == "dot_bf16");
    CHECK(table[2].opsPerCycleCore == 64.0);
    CHECK(table[2].opsPerInstruction == 64);    // 2 x 2 x 16 lanes
}

TEST_CASE("compute_roofs: physical cores count SMT siblings once", "[machine_compute_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);
    CHECK(physicalCoresOf(range(0, 15), m) == 16);
    CHECK(physicalCoresOf(range(0, 31), m) == 16);   // both SMT siblings of every core
    CHECK(physicalCoresOf({0, 16}, m) == 1);         // one core, two threads
    CHECK(physicalCoresOf({}, m) == 0);
}

TEST_CASE("compute_roofs: fma_f32 verifies exactly on one and two threads", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::fmaF32)) SKIP(computeKernelReason(ComputeKernel::fmaF32));
    expectVerified(ComputeKernel::fmaF32, {0});
    expectVerified(ComputeKernel::fmaF32, {0, 1});
}

TEST_CASE("compute_roofs: dot_i8 verifies exactly on one and two threads", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::dotI8)) SKIP(computeKernelReason(ComputeKernel::dotI8));
    expectVerified(ComputeKernel::dotI8, {0});
    expectVerified(ComputeKernel::dotI8, {0, 1});
}

TEST_CASE("compute_roofs: dot_bf16 verifies exactly on one and two threads", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::dotBf16)) SKIP(computeKernelReason(ComputeKernel::dotBf16));
    expectVerified(ComputeKernel::dotBf16, {0});
    expectVerified(ComputeKernel::dotBf16, {0, 1});
}

TEST_CASE("compute_roofs: a corrupted fma accumulator makes the point invalid", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::fmaF32)) SKIP(computeKernelReason(ComputeKernel::fmaF32));
    expectCorruptionCaught(ComputeKernel::fmaF32);
}

TEST_CASE("compute_roofs: a corrupted dot_i8 accumulator makes the point invalid", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::dotI8)) SKIP(computeKernelReason(ComputeKernel::dotI8));
    expectCorruptionCaught(ComputeKernel::dotI8);
}

TEST_CASE("compute_roofs: a corrupted dot_bf16 accumulator makes the point invalid", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::dotBf16)) SKIP(computeKernelReason(ComputeKernel::dotBf16));
    expectCorruptionCaught(ComputeKernel::dotBf16);
}

TEST_CASE("compute_roofs: iteration counts past each kernel's exact range are refused", "[machine_compute_roofs]") {
    // Refused before any thread starts, so this runs whatever the ISA: the cap is the same arithmetic either way.
    ComputeRequest i8;
    i8.kernel = ComputeKernel::dotI8;
    i8.cpus = {0};
    i8.iterations = (std::uint64_t{1} << 28) + 1;   // 4 x iterations would pass 2^30
    i8.repeats = 1;
    CHECK_FALSE(measureComputeKernel(i8).verified);

    ComputeRequest bf;
    bf.kernel = ComputeKernel::dotBf16;
    bf.cpus = {0};
    bf.iterations = (std::uint64_t{1} << 23) + 1;   // 2 x iterations would pass 2^24
    bf.repeats = 1;
    CHECK_FALSE(measureComputeKernel(bf).verified);
}

TEST_CASE("compute_roofs: calibration gives a pass of about the requested length", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    if (!computeKernelCompiled(ComputeKernel::fmaF32)) SKIP(computeKernelReason(ComputeKernel::fmaF32));
    ComputeRequest req;
    req.kernel = ComputeKernel::fmaF32;
    req.cpus = {0};
    req.iterations = 0;          // calibrate
    req.passSeconds = 0.01;
    req.repeats = 1;
    const ComputeMeasurement m = measureComputeKernel(req);
    INFO(m.verifyFailure);
    CHECK(m.verified);
    CHECK(m.iterations > 0);
}

TEST_CASE("compute_roofs: the effective clock is verified and plausible", "[machine_compute_roofs]") {
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    const ClockPoint c = measureEffectiveClock({0}, 0.002, 1, std::chrono::steady_clock::time_point::max());
    INFO(c.verifyFailure);
    CHECK(c.verified);
    REQUIRE(c.medianGHz.value.has_value());
    CHECK(*c.medianGHz.value > 0.1);
    CHECK(*c.medianGHz.value < 10.0);
}

TEST_CASE("compute_roofs: the planner reuses the placements and shrinks the run to fit", "[machine_compute_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const std::vector<Placement> ps = planPlacements(readMachineFacts(dir.path));
    REQUIRE(ps.size() == 3);
    const Placement* pack = find(ps, "pack");
    REQUIRE(pack != nullptr);
    CHECK(cpusFor(*pack, 8) == range(0, 7));    // the same placement the memory roofs use

    ComputeConfig cfg;
    cfg.repeats = 5;
    cfg.threadCounts = {1, 2, 4, 8, 12, 16};
    cfg.passSeconds = 0.05;

    const ComputePlan roomy = planComputeRun(ps, [&] {
        ComputeConfig c = cfg;
        c.budgetSeconds = 1.0e6;
        return c;
    }());
    CHECK(roomy.repeats == 5);
    CHECK(roomy.note.empty());
    REQUIRE(roomy.threadCounts.size() == 3);
    CHECK(roomy.threadCounts[0] == std::vector<int>{1, 2, 4, 8, 12, 16});   // pack
    CHECK(roomy.threadCounts[2] == std::vector<int>{1, 2, 4, 8, 12, 16, 32});   // smt keeps its full count

    ComputeConfig tight = cfg;
    tight.budgetSeconds = 2.0;
    const ComputePlan shrunk = planComputeRun(ps, tight);
    CHECK(shrunk.repeats < 5);
    CHECK_FALSE(shrunk.note.empty());
    CHECK(shrunk.estimateSeconds <= 2.0 + 1e-9);
    std::size_t counts = 0;
    for (const auto& c : shrunk.threadCounts) counts += c.size();
    CHECK(counts < 19);   // the 19 full lists were thinned
}

TEST_CASE("compute_roofs: a full tiny run verifies every compiled kernel and writes deterministic JSON",
          "[machine_compute_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);
    if (!kRoofsSupported) SKIP("compute roofs need Linux");
    const ComputeRoofs r = measureComputeRoofs(m, tinyComputeConfig());
    REQUIRE(r.supported);
    REQUIRE(r.isas.size() == 3);
    // pack 1,2; spread 1,2; smt 1,2 and its full 32 CPUs: 7 thread counts, a clock and three kernels each.
    CHECK(r.clocks.size() == 7);
    CHECK(r.points.size() == 7 * 3);
    for (const ComputePoint& p : r.points) {
        if (!p.skipReason.empty()) continue;
        INFO(p.kernel << " " << p.placement << " " << p.threads << ": " << p.measurement.verifyFailure);
        CHECK(p.measurement.verified);
        CHECK(p.physicalCores >= 1);
        CHECK(p.physicalCores <= p.threads);
    }
    CHECK(r.totalSeconds < tinyComputeConfig().budgetSeconds + 1.0);

    const std::string a = toJson(r);
    CHECK(a.rfind("{\"schema\":\"sub0llm-machine-1\",\"roofs\":{\"compute\":{", 0) == 0);
    CHECK(a.find("\"compute\"") != std::string::npos);
    CHECK(a.back() == '}');
    CHECK(balanced(a));
    CHECK(a == toJson(r));

    // The combined document carries the facts object exactly as toJson(facts) writes it; the memory roofs are absent.
    const std::string combined = toRoofsJson(m, nullptr, &r);
    CHECK(balanced(combined));
    CHECK(combined.find("\"facts\":" + detail::roofs::factsObject(m) + ",\"roofs\":{\"compute\":{") !=
          std::string::npos);
    CHECK(combined.find("\"roofs\":{\"memory\"") == std::string::npos);

    const std::string summary = toSummary(r);
    CHECK(summary.find("rate (median) by kernel, placement and threads") != std::string::npos);
    CHECK(summary.find("effective GHz") != std::string::npos);
    CHECK(summary.find("best fma_f32") != std::string::npos);
}

TEST_CASE("ceilings: decode and prefill arithmetic matches hand-computed values", "[machine_ceilings]") {
    // 16.8e9 bytes per token at 33 GB/s: 33e9 / 16.8e9 = 1.9643 tokens per second.
    CHECK(std::fabs(decodeCeilingTokPerS(16.8e9, 33.0) - 1.9643) < 1e-4);
    // 54 GFLOP per token at 2000 GFLOP/s: 2000e9 / 54e9 = 37.037 tokens per second.
    CHECK(std::fabs(prefillCeilingTokPerS(54e9, 2000.0) - 37.037) < 1e-3);
    // A measured value at 40% of its ceiling.
    CHECK(std::fabs(ratioToCeiling(0.4, 1.0) - 0.4) < 1e-12);
    CHECK(std::fabs(ratioToCeiling(37.0, 37.037) - 0.99900) < 1e-4);
    static_assert(decodeCeilingTokPerS(1.0e9, 2.0) == 2.0);
    static_assert(prefillCeilingTokPerS(1.0e9, 3.0) == 3.0);
    static_assert(ratioToCeiling(1.0, 2.0) == 0.5);
}

TEST_CASE("ceilings: zero, negative and NaN inputs give 0 (no ceiling known)", "[machine_ceilings]") {
    CHECK(decodeCeilingTokPerS(0.0, 33.0) == 0.0);
    CHECK(decodeCeilingTokPerS(16.8e9, 0.0) == 0.0);
    CHECK(decodeCeilingTokPerS(-1.0, 33.0) == 0.0);
    CHECK(decodeCeilingTokPerS(16.8e9, -33.0) == 0.0);
    CHECK(decodeCeilingTokPerS(std::nan(""), 33.0) == 0.0);
    CHECK(prefillCeilingTokPerS(0.0, 2000.0) == 0.0);
    CHECK(prefillCeilingTokPerS(54e9, -1.0) == 0.0);
    CHECK(ratioToCeiling(1.0, 0.0) == 0.0);
    CHECK(ratioToCeiling(1.0, -2.0) == 0.0);
    CHECK(ratioToCeiling(-1.0, 2.0) == 0.0);
    CHECK(ratioToCeiling(std::nan(""), 2.0) == 0.0);
}
