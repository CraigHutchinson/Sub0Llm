// machine_facts_tests.cpp -- sub0/measure/machine_facts.hpp (issue #78, increment 1). Engine-free: the reader
// is standard library only, so these cases link Catch2 and nothing else. The fixtures are fabricated /proc
// and /sys trees written into a scratch directory unique to each test (see blend_schedule_tests.cpp for why
// a fixed shared path is a real flake under `ctest -j`).

#include <catch2/catch_test_macros.hpp>

#include "sub0/measure/machine_facts.hpp"

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
               ("sub0_machine_facts_test_" + std::to_string(rd()) + "_" + std::to_string(rd()));
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

constexpr const char* kModel = "Test \"CPU\" 9999X2 16-Core";

// A 2-CCD machine: 16 physical cores, SMT2, 32 logical CPUs. Logical N and N+16 are the two threads of core
// N%16. Cores 0-7 (CPUs 0-7 and 16-23) share one 32 MiB L3 (CCD0); cores 8-15 (CPUs 8-15 and 24-31) share
// the other (CCD1). L1 and L2 are private to each core. Plus the memory, cpufreq, THP, kernel and GPU files
// a real Linux box exposes.
void buildTwoCcdMachine(const fs::path& root) {
    std::string cpuinfo;
    for (int c = 0; c < 32; ++c) {
        cpuinfo += "processor\t: " + std::to_string(c) + "\n";
        cpuinfo += "vendor_id\t: AuthenticAMD\n";
        cpuinfo += std::string("model name\t: ") + kModel + "\n";
        cpuinfo += "flags\t\t: fpu sse4_2 avx avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_vnni "
                   "avx512_bf16\n\n";
    }
    writeFile(root / "proc/cpuinfo", cpuinfo);
    writeFile(root / "proc/meminfo",
              "MemTotal:       65536000 kB\n"
              "MemAvailable:   40000000 kB\n"
              "SwapTotal:       2097152 kB\n");
    writeFile(root / "proc/sys/kernel/osrelease", "6.8.0-test\n");
    writeFile(root / "proc/sys/kernel/hostname", "testhost\n");

    for (int c = 0; c < 32; ++c) {
        const std::string cpu = "sys/devices/system/cpu/cpu" + std::to_string(c) + "/";
        const int core = c % 16;
        const std::string pair = std::to_string(core) + "," + std::to_string(core + 16);
        const std::string ccd = core < 8 ? "0-7,16-23" : "8-15,24-31";
        writeFile(root / (cpu + "topology/core_id"), std::to_string(core) + "\n");
        writeFile(root / (cpu + "topology/physical_package_id"), "0\n");
        writeFile(root / (cpu + "topology/thread_siblings_list"), pair + "\n");

        const std::string idx = cpu + "cache/index";
        writeFile(root / (idx + "0/level"), "1\n");
        writeFile(root / (idx + "0/type"), "Data\n");
        writeFile(root / (idx + "0/size"), "32K\n");
        writeFile(root / (idx + "0/shared_cpu_list"), pair + "\n");
        writeFile(root / (idx + "1/level"), "1\n");
        writeFile(root / (idx + "1/type"), "Instruction\n");
        writeFile(root / (idx + "1/size"), "32K\n");
        writeFile(root / (idx + "1/shared_cpu_list"), pair + "\n");
        writeFile(root / (idx + "2/level"), "2\n");
        writeFile(root / (idx + "2/type"), "Unified\n");
        writeFile(root / (idx + "2/size"), "1024K\n");
        writeFile(root / (idx + "2/shared_cpu_list"), pair + "\n");
        writeFile(root / (idx + "3/level"), "3\n");
        writeFile(root / (idx + "3/type"), "Unified\n");
        writeFile(root / (idx + "3/size"), "32768K\n");
        writeFile(root / (idx + "3/shared_cpu_list"), ccd + "\n");
    }

    writeFile(root / "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "performance\n");
    writeFile(root / "sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference", "balance_performance\n");
    writeFile(root / "sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "5000000\n");
    writeFile(root / "sys/devices/system/cpu/cpufreq/boost", "1\n");
    writeFile(root / "sys/devices/system/cpu/amd_pstate/status", "active\n");
    writeFile(root / "sys/kernel/mm/transparent_hugepage/enabled", "always [madvise] never\n");
    writeFile(root / "sys/devices/system/node/node0/cpulist", "0-31\n");

    const std::string dev = "sys/class/drm/card0/device/";
    writeFile(root / (dev + "vendor"), "0x1002\n");
    writeFile(root / (dev + "device"), "0x164e\n");
    writeFile(root / (dev + "uevent"), "DRIVER=amdgpu\nPCI_ID=1002:164E\n");
    writeFile(root / (dev + "mem_info_vram_total"), "1073741824\n");
    writeFile(root / (dev + "mem_info_gtt_total"), "15900000000\n");
    // A connector entry: a card-like directory with no device/vendor. It must be ignored.
    writeFile(root / "sys/class/drm/card0-DP-1/status", "disconnected\n");
}

std::vector<int> range(int lo, int hi) {
    std::vector<int> v;
    for (int i = lo; i <= hi; ++i) v.push_back(i);
    return v;
}

}  // namespace

TEST_CASE("machine_facts: a 2-CCD SMT2 machine reports topology, L3 groups, caches, ISA, memory and GPU",
          "[machine_facts]") {
    ScratchDir dir;
    buildTwoCcdMachine(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);

    CHECK(m.hostname.value.value_or("") == "testhost");
    CHECK(m.kernelRelease.value.value_or("") == "6.8.0-test");
    CHECK(m.cpuModel.value.value_or("") == kModel);
    CHECK(m.vendor.value.value_or("") == "AuthenticAMD");

    CHECK(m.logicalCpus.value.value_or(-1) == 32);
    CHECK(m.physicalCores.value.value_or(-1) == 16);
    CHECK(m.smtWays.value.value_or(-1) == 2);
    REQUIRE(m.cpus.value.has_value());
    CHECK(m.cpus.value->size() == 32);
    REQUIRE(m.onePerCore.value.has_value());
    CHECK(*m.onePerCore.value == range(0, 15));

    // SMT pairs: one group per physical core, holding that core's two logical CPUs.
    REQUIRE(m.smtGroups.value.has_value());
    REQUIRE(m.smtGroups.value->size() == 16);
    for (int core = 0; core < 16; ++core) {
        const std::vector<int> pair = {core, core + 16};
        CHECK((*m.smtGroups.value)[static_cast<std::size_t>(core)] == pair);
    }

    // Packages: this fabricated machine has one package, so one group holding every CPU.
    REQUIRE(m.packageCpus.value.has_value());
    REQUIRE(m.packageCpus.value->size() == 1);
    CHECK((*m.packageCpus.value)[0] == range(0, 31));

    REQUIRE(m.l3Groups.value.has_value());
    REQUIRE(m.l3Groups.value->size() == 2);
    const std::vector<int> ccd0 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    const std::vector<int> ccd1 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    CHECK((*m.l3Groups.value)[0] == ccd0);
    CHECK((*m.l3Groups.value)[1] == ccd1);

    // L1 and L2 are per core: each core's two SMT threads share one cache, so 16 instances of each.
    REQUIRE(m.caches.size() == 4);
    CHECK(m.caches[0].name == "L1d");
    CHECK(m.caches[0].sizeBytes.value.value_or(0) == 32u * 1024u);
    CHECK(m.caches[0].instances.value.value_or(-1) == 16);
    CHECK(m.caches[1].name == "L1i");
    CHECK(m.caches[2].name == "L2");
    CHECK(m.caches[2].sizeBytes.value.value_or(0) == 1024u * 1024u);
    CHECK(m.caches[2].instances.value.value_or(-1) == 16);
    CHECK(m.caches[3].name == "L3");
    CHECK(m.caches[3].sizeBytes.value.value_or(0) == 32u * 1024u * 1024u);
    CHECK(m.caches[3].instances.value.value_or(-1) == 2);

    REQUIRE(m.isaFlags.value.has_value());
    CHECK(*m.isaFlags.value == std::vector<std::string>{"avx", "avx2", "fma", "f16c", "avx512f", "avx512bw",
                                                        "avx512dq", "avx512vl", "avx512_vnni", "avx512_bf16",
                                                        "sse4_2"});

    CHECK(m.memTotalBytes.value.value_or(0) == 65536000ull * 1024ull);
    CHECK(m.memAvailableBytes.value.value_or(0) == 40000000ull * 1024ull);
    CHECK(m.swapTotalBytes.value.value_or(0) == 2097152ull * 1024ull);
    CHECK(m.numaNodes.value.value_or(-1) == 1);
    CHECK_FALSE(m.memoryInstalledBytes.value.has_value());
    CHECK(m.memoryInstalledBytes.reason.find("root") != std::string::npos);

    CHECK(m.cpuGovernor.value.value_or("") == "performance");
    CHECK(m.energyPerformancePreference.value.value_or("") == "balance_performance");
    CHECK(m.maxFreqMhz.value.value_or(0.0) == 5000.0);
    CHECK(m.boostEnabled.value.value_or(false));
    CHECK(m.amdPstateStatus.value.value_or("") == "active");
    CHECK(m.transparentHugepages.value.value_or("") == "madvise");

    REQUIRE(m.gpus.value.has_value());
    REQUIRE(m.gpus.value->size() == 1);
    const GpuDevice& g = (*m.gpus.value)[0];
    CHECK(g.card == "card0");
    CHECK(g.vendorId == "1002");
    CHECK(g.deviceId == "164e");
    CHECK(g.driver == "amdgpu");
    CHECK(g.vramTotalBytes.value.value_or(0) == 1073741824ull);
    CHECK(g.gttTotalBytes.value.value_or(0) == 15900000000ull);

    const std::string summary = toSummary(m);
    CHECK(summary.find("32 logical CPUs, 16 physical cores, SMT 2 way") != std::string::npos);
    CHECK(summary.find("2 (each group shares one L3)") != std::string::npos);
    CHECK(summary.find("SMT 2 way, 1 package(s)") != std::string::npos);
}

TEST_CASE("machine_facts: an empty or absent root reads without throwing and reports unknowns",
          "[machine_facts]") {
    ScratchDir dir;   // exists, holds nothing
    MachineFacts empty;
    REQUIRE_NOTHROW(empty = readMachineFacts(dir.path));
    CHECK_FALSE(empty.logicalCpus.value.has_value());
    CHECK_FALSE(empty.cpuModel.value.has_value());
    CHECK_FALSE(empty.cpuModel.reason.empty());
    CHECK_FALSE(empty.physicalCores.value.has_value());
    CHECK_FALSE(empty.l3Groups.value.has_value());
    CHECK_FALSE(empty.smtGroups.value.has_value());
    CHECK_FALSE(empty.packageCpus.value.has_value());
    CHECK_FALSE(empty.memTotalBytes.value.has_value());
    CHECK_FALSE(empty.gpus.value.has_value());
    CHECK_FALSE(empty.memoryInstalledBytes.value.has_value());
    REQUIRE(empty.caches.size() == 4);
    CHECK_FALSE(empty.caches[3].sizeBytes.value.has_value());
    CHECK_FALSE(empty.caches[3].sizeBytes.reason.empty());
    CHECK_FALSE(toSummary(empty).empty());

    MachineFacts absent;
    REQUIRE_NOTHROW(absent = readMachineFacts(dir.path / "does-not-exist"));
    CHECK_FALSE(absent.logicalCpus.value.has_value());
    CHECK_FALSE(toJson(absent).empty());
}

TEST_CASE("machine_facts: toJson is schema-tagged, escaped and byte-identical across calls", "[machine_facts]") {
    ScratchDir dir;
    buildTwoCcdMachine(dir.path);
    const MachineFacts m = readMachineFacts(dir.path);

    const std::string a = toJson(m);
    CHECK(a == toJson(m));
    CHECK(a == toJson(readMachineFacts(dir.path)));

    CHECK(a.rfind("{\"schema\":\"sub0llm-machine-1\",\"facts\":{", 0) == 0);
    CHECK(a.find("\"physicalCores\":{\"value\":16,\"reason\":\"\"}") != std::string::npos);
    CHECK(a.find("\"smtGroups\":{\"value\":[[0,16],[1,17]") != std::string::npos);
    CHECK(a.find("\"packageCpus\":{\"value\":[[0,1,") != std::string::npos);
    CHECK(a.find("coreSiblings") == std::string::npos);
    CHECK(a.find("\"memoryInstalledBytes\"") == std::string::npos);   // key is "installedBytes"
    CHECK(a.find("\"installedBytes\":{\"value\":null,\"reason\":\"needs root") != std::string::npos);
    CHECK(a.find("Test \\\"CPU\\\" 9999X2 16-Core") != std::string::npos);   // quotes are escaped
    CHECK(a.back() == '}');
}

TEST_CASE("machine_facts: the real root reads without throwing", "[machine_facts]") {
    MachineFacts real;
    REQUIRE_NOTHROW(real = readMachineFacts());
#if defined(__linux__)
    CHECK(real.logicalCpus.value.value_or(0) > 0);
#endif
    CHECK_FALSE(toSummary(real).empty());
    CHECK_FALSE(toJson(real).empty());
}
