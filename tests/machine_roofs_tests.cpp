// machine_roofs_tests.cpp -- sub0/measure/memory_roofs.hpp (issue #78, increment 2). Engine-free: the roofs use
// the standard library, pthreads through std::thread, and sched_setaffinity on Linux. Each case runs on small
// buffers and one or two threads, so every case stays well under 3 seconds. The planner cases read a fabricated
// 2-CCD topology (the same shape as machine_facts_tests.cpp) through readMachineFacts, so they exercise the
// real facts path.

#include <catch2/catch_test_macros.hpp>

#include "sub0/measure/machine_facts.hpp"
#include "sub0/measure/memory_roofs.hpp"

#include <chrono>
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
               ("sub0_machine_roofs_test_" + std::to_string(rd()) + "_" + std::to_string(rd()));
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
// CCD1 = cores 8-15. Enough for readMachineFacts to give onePerCore, smtGroups, l3Groups and the L3 size.
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

// Small roofs run: 256 KiB arrays, one repeat, two thread counts, short sweeps.
RoofConfig tinyConfig() {
    RoofConfig cfg;
    cfg.budgetSeconds = 2.0;
    cfg.repeats = 1;
    cfg.threadCounts = {1, 2};
    cfg.arrayBytesOverride = 256u << 10;
    cfg.sweepMaxBytes = 256u << 10;
    cfg.latencyMaxBytes = 64u << 10;
    return cfg;
}

}  // namespace

TEST_CASE("machine_roofs: pack fills one CCD first, one thread per core", "[machine_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);
    const std::vector<Placement> ps = planPlacements(m);
    REQUIRE(ps.size() == 3);
    const Placement* pack = find(ps, "pack");
    REQUIRE(pack != nullptr);
    CHECK(cpusFor(*pack, 8) == range(0, 7));              // the whole of CCD0 before CCD1
    CHECK(cpusFor(*pack, 16) == range(0, 15));            // 16 one-per-core CPUs, no SMT sibling yet
    CHECK(cpusFor(*pack, 17).empty());                    // out of range is empty, not a clip
    CHECK(cpusFor(*pack, 0).empty());
}

TEST_CASE("machine_roofs: spread alternates CCDs, so two threads use one core on each", "[machine_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const std::vector<Placement> ps = planPlacements(readMachineFacts(dir.path));
    const Placement* spread = find(ps, "spread");
    REQUIRE(spread != nullptr);
    CHECK(cpusFor(*spread, 4) == std::vector<int>{0, 8, 1, 9});
    CHECK(cpusFor(*spread, 2) == std::vector<int>{0, 8});
    CHECK(spread->cpus.size() == 16);
}

TEST_CASE("machine_roofs: smt appends the sibling threads after the one-per-core CPUs", "[machine_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const std::vector<Placement> ps = planPlacements(readMachineFacts(dir.path));
    const Placement* smt = find(ps, "smt");
    REQUIRE(smt != nullptr);
    REQUIRE(smt->cpus.size() == 32);
    CHECK(std::vector<int>(smt->cpus.begin(), smt->cpus.begin() + 16) == range(0, 15));
    CHECK(std::vector<int>(smt->cpus.begin() + 16, smt->cpus.end()) == range(16, 31));
    CHECK(threadCountsFor(*smt) == std::vector<int>{1, 2, 4, 8, 12, 16, 32});
}

TEST_CASE("machine_roofs: thread counts are clipped to the placement and the full count is kept for smt",
          "[machine_roofs]") {
    const Placement small{"pack", {0, 1, 2, 3}};
    CHECK(threadCountsFor(small) == std::vector<int>{1, 2, 4});
    const Placement smallSmt{"smt", {0, 1, 2, 3, 4, 5}};
    CHECK(threadCountsFor(smallSmt) == std::vector<int>{1, 2, 4, 6});
    const Placement none{"pack", {}};
    CHECK(threadCountsFor(none).empty());
}

TEST_CASE("machine_roofs: an unknown topology gives no placements and an unsupported roofs result",
          "[machine_roofs]") {
    ScratchDir dir;   // empty root: no topology
    const MachineFacts empty = readMachineFacts(dir.path);
    CHECK(planPlacements(empty).empty());
    const MemoryRoofs r = measureMemoryRoofs(empty, tinyConfig());
    CHECK_FALSE(r.supported);
    CHECK_FALSE(r.unsupportedReason.empty());
    CHECK(toSummary(r).find("unsupported") != std::string::npos);
}

TEST_CASE("machine_roofs: the sweep sizes run 16 KiB to 1 GiB in factors of four", "[machine_roofs]") {
    const std::vector<std::size_t> s = sweepSizes(std::size_t{1} << 30);
    REQUIRE(s.size() == 9);
    CHECK(s.front() == 16u << 10);
    CHECK(s.back() == std::size_t{1} << 30);
    for (std::size_t i = 1; i < s.size(); ++i) CHECK(s[i] == 4 * s[i - 1]);
}

TEST_CASE("machine_roofs: the Sattolo permutation is one cycle that visits every element once",
          "[machine_roofs]") {
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{1000}, std::size_t{4096}}) {
        const std::vector<std::uint64_t> p = buildSingleCyclePermutation(n);
        REQUIRE(p.size() == n);
        CHECK(isPermutation(p));
        CHECK(isSingleCycle(p));
        // Walking from 0 visits each element exactly once before returning.
        std::vector<bool> seen(n, false);
        std::uint64_t idx = 0;
        for (std::size_t step = 0; step < n; ++step) {
            REQUIRE_FALSE(seen[static_cast<std::size_t>(idx)]);
            seen[static_cast<std::size_t>(idx)] = true;
            idx = p[static_cast<std::size_t>(idx)];
        }
        CHECK(idx == 0);
    }
    // The same seed gives the same chain (deterministic), and a broken chain is caught.
    CHECK(buildSingleCyclePermutation(500) == buildSingleCyclePermutation(500));
    std::vector<std::uint64_t> identity(8);
    for (std::size_t i = 0; i < identity.size(); ++i) identity[i] = i;
    CHECK(isPermutation(identity));
    CHECK_FALSE(isSingleCycle(identity));
    std::vector<std::uint64_t> two_cycles = {1, 0, 3, 2};
    CHECK(isPermutation(two_cycles));
    CHECK_FALSE(isSingleCycle(two_cycles));
}

TEST_CASE("machine_roofs: read, write, copy and triad pass their checks on a tiny buffer", "[machine_roofs]") {
    if (!kRoofsSupported) SKIP("memory roofs need Linux");
    RoofArrays arr(256u << 10);
    REQUIRE(arr.ok());
    for (const RoofKernel k : {RoofKernel::read, RoofKernel::write, RoofKernel::copy, RoofKernel::triad}) {
        for (const std::vector<int>& cpus : {std::vector<int>{0}, std::vector<int>{0, 1}}) {
            BandwidthRequest req;
            req.kernel = k;
            req.cpus = cpus;
            req.elements = arr.elements();
            req.passes = 2;
            req.repeats = 2;
            const Measurement m = measureBandwidth(arr, req);
            INFO("kernel " << kernelName(k) << " threads " << cpus.size() << ": " << m.verifyFailure);
            CHECK(m.verified);
            CHECK(m.repeats == 2);
            REQUIRE(m.medianGbPerSec.value.has_value());
            CHECK(*m.medianGbPerSec.value > 0.0);
            CHECK(*m.minGbPerSec.value <= *m.maxGbPerSec.value);
        }
    }
}

TEST_CASE("machine_roofs: a corrupted output makes the point invalid, not just flagged", "[machine_roofs]") {
    if (!kRoofsSupported) SKIP("memory roofs need Linux");
    RoofArrays arr(256u << 10);
    REQUIRE(arr.ok());
    for (const RoofKernel k : {RoofKernel::read, RoofKernel::write, RoofKernel::copy, RoofKernel::triad}) {
        BandwidthRequest req;
        req.kernel = k;
        req.cpus = {0};
        req.elements = arr.elements();
        req.repeats = 1;
        req.corruptOutput = true;   // TEST HOOK: damages element 0 so the check must fail
        const Measurement m = measureBandwidth(arr, req);
        INFO("kernel " << kernelName(k));
        CHECK_FALSE(m.verified);
        CHECK_FALSE(m.verifyFailure.empty());
        CHECK_FALSE(m.medianGbPerSec.value.has_value());
        CHECK_FALSE(m.medianGbPerSec.reason.empty());
    }
}

TEST_CASE("machine_roofs: the latency chase reports a verified ns-per-load", "[machine_roofs]") {
    if (!kRoofsSupported) SKIP("memory roofs need Linux");
    const LatencyPoint lp = measureLatency(64u << 10, 0, 1, std::chrono::steady_clock::time_point::max());
    CHECK(lp.verified);
    REQUIRE(lp.medianNsPerLoad.value.has_value());
    CHECK(*lp.medianNsPerLoad.value > 0.0);
}

TEST_CASE("machine_roofs: the planner shrinks repeats and thins thread counts to fit the budget",
          "[machine_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const std::vector<Placement> ps = planPlacements(readMachineFacts(dir.path));
    RoofConfig cfg;
    cfg.repeats = 5;
    cfg.threadCounts = {1, 2, 4, 8, 12, 16};
    cfg.assumedGbPerSec = 40.0;
    const std::size_t oneGiB = std::size_t{1} << 30;

    const RoofPlan roomy = planRoofRun(ps, cfg, oneGiB, 1.0e6);
    CHECK(roomy.repeats == 5);
    CHECK(roomy.note.empty());

    // At 40 GB/s one repeat of all 19 points (fill plus pass) is about 7 s, so 4 s needs thinning as well.
    const RoofPlan tight = planRoofRun(ps, cfg, oneGiB, 4.0);
    CHECK(tight.repeats == 1);
    CHECK_FALSE(tight.note.empty());
    CHECK(tight.estimateSeconds <= 4.0 + 1e-9);

    const RoofPlan starved = planRoofRun(ps, cfg, oneGiB, 0.01);
    CHECK(starved.repeats == 1);
    std::size_t counts = 0;
    for (const auto& c : starved.threadCounts) counts += c.size();
    CHECK(counts < 19);   // the 19 full lists (pack 6, spread 6, smt 7) were thinned
}

TEST_CASE("machine_roofs: toJson is deterministic, schema-tagged and carries roofs", "[machine_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);
    if (!kRoofsSupported) SKIP("memory roofs need Linux");
    const MemoryRoofs r = measureMemoryRoofs(m, tinyConfig());
    REQUIRE(r.supported);
    REQUIRE_FALSE(r.kernels.empty());

    const std::string a = toJson(r);
    CHECK(a.rfind("{\"schema\":\"sub0llm-machine-1\",\"roofs\":{\"memory\":{", 0) == 0);
    CHECK(a.find("\"roofs\"") != std::string::npos);
    CHECK(a.back() == '}');
    CHECK(a == toJson(r));

    // The combined document carries the facts object exactly as toJson(facts) writes it.
    const std::string prefix = "{\"schema\":\"sub0llm-machine-1\",\"facts\":";
    const std::string factsDoc = toJson(m);
    const std::string factsBody = factsDoc.substr(prefix.size(), factsDoc.size() - prefix.size() - 1);
    const std::string combined = toJson(m, r);
    CHECK(combined.find("\"facts\":" + factsBody + ",\"roofs\":{\"memory\":") != std::string::npos);
    CHECK(combined.find("\"kernels\":[") != std::string::npos);

    const std::string summary = toSummary(r);
    CHECK(summary.find("read GB/s (median) by placement and threads") != std::string::npos);
    CHECK(summary.find("best write") != std::string::npos);
}

TEST_CASE("machine_roofs: a full tiny run measures the kernels, sweep and latency inside its budget",
          "[machine_roofs]") {
    ScratchDir dir;
    buildTwoCcdTopology(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);
    if (!kRoofsSupported) SKIP("memory roofs need Linux");
    const RoofConfig cfg = tinyConfig();
    const MemoryRoofs r = measureMemoryRoofs(m, cfg);
    REQUIRE(r.supported);
    CHECK(r.arrayBytes == 256u << 10);
    CHECK(r.plannedRepeats == 1);
    // pack 1,2; spread 1,2; smt 1,2 and its full 32 CPUs; each with 4 kernels
    CHECK(r.kernels.size() == (2 + 2 + 3) * 4);
    for (const KernelPoint& k : r.kernels) {
        if (!k.skipReason.empty()) continue;
        INFO(k.placement << " " << k.kernel << " " << k.threads << ": " << k.measurement.verifyFailure);
        CHECK(k.measurement.verified);
    }
    CHECK(r.sweep.size() == 3 * 2);       // 16 KiB, 64 KiB, 256 KiB, single and one-per-core
    CHECK(r.latency.size() == 2);         // 16 KiB and 64 KiB
    CHECK(r.totalSeconds < cfg.budgetSeconds + 1.0);
}
