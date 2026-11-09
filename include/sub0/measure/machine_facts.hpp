// sub0/measure/machine_facts.hpp -- hardware FACTS for `sub0llm machine` (issue #78, increment 1).
//
// WHY THIS EXISTS. A roofline or a measured kernel number means nothing without the machine it was taken
// on: which logical CPUs share an L3 (the CCD/CCX grouping that decides whether a memory-bound kernel
// scales across cores), how many physical cores sit behind the SMT siblings, what the cores can execute
// (AVX-512, VNNI, AMX), and how much memory the OS is allowed to use. This header reads those facts and
// nothing else. Timing kernels are a later increment. Design: docs/design/MEASURE_SELECT_VERIFY.md.
//
// NON-ROOT SOURCES ONLY. Everything comes from /proc and /sys files an unprivileged process can read.
// Installed DIMM size, speed and channel count need root (dmidecode), so they are reported as unknown with
// that reason. They are never estimated from the OS-visible total.
//
// NOTHING THROWS, NOTHING GUESSES. A file that is missing, unreadable or malformed becomes an empty Fact
// with a reason. Reasons name the path RELATIVE to the root, so one machine reports the same reasons
// whichever root it was read from.
//
// ROOT. Every read is made under `root`, so a test can point the reader at a fabricated /proc and /sys
// tree. The default "/" is the real machine.
//
// PLATFORMS. Linux-first. On a platform without these files every fact comes back unknown with a reason;
// nothing fails to compile.

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace sub0::measure {

/// One fact the machine may not expose. `value` is empty when it could not be read; `reason` then says
/// why. A known value carries an empty reason.
template <typename T>
struct Fact {
    std::optional<T> value;
    std::string reason;
};

/// One logical CPU's place in the topology. `physicalCore` is a dense index (0..physicalCores-1), so two
/// logical CPUs with the same value are SMT siblings of one physical core.
struct LogicalCpu {
    int id = -1;
    int package = -1;      // physical_package_id, -1 when the kernel did not report it
    int coreId = -1;       // core_id within the package, -1 when the kernel did not report it
    int physicalCore = -1;
};

/// One cache level summarised across the machine. `instances` counts distinct shared_cpu_list values at
/// this level, i.e. how many separate caches of that kind exist (two L3s on a two-CCD part).
struct CacheLevel {
    std::string name;   // "L1d", "L1i", "L2", "L3"
    Fact<std::uint64_t> sizeBytes;
    Fact<int> instances;
};

/// One GPU found under sys/class/drm/cardN/device.
struct GpuDevice {
    std::string card;       // "card0"
    std::string vendorId;   // PCI vendor id, lowercase hex without "0x", e.g. "1002"
    std::string deviceId;   // PCI device id, lowercase hex without "0x", e.g. "164e"
    std::string driver;     // device/uevent DRIVER=, empty when absent
    Fact<std::uint64_t> vramTotalBytes;   // device/mem_info_vram_total
    Fact<std::uint64_t> gttTotalBytes;    // device/mem_info_gtt_total
};

/// ISA extensions the report checks for in the cpuinfo `flags` line, in the order the report lists them.
inline constexpr std::string_view kIsaFlags[] = {
    "avx",       "avx2",       "fma",         "f16c",      "avx512f",   "avx512bw",   "avx512dq",
    "avx512vl",  "avx512_vnni", "avx512_bf16", "avx512_vbmi", "avx_vnni", "amx_tile",  "amx_int8",
    "amx_bf16",  "sse4_2",
};

struct MachineFacts {
    Fact<std::string> hostname;
    Fact<std::string> kernelRelease;

    Fact<std::string> cpuModel;
    Fact<std::string> vendor;
    Fact<int> logicalCpus;
    Fact<int> physicalCores;
    Fact<int> smtWays;                                  // most logical CPUs on one physical core
    Fact<std::vector<LogicalCpu>> cpus;                 // one entry per logical CPU, ascending id
    Fact<std::vector<int>> onePerCore;                  // lowest logical CPU of each physical core
    Fact<std::vector<std::vector<int>>> coreSiblings;   // logical CPUs per physical package (sysfs core_siblings)
    Fact<std::vector<std::vector<int>>> l3Groups;       // logical CPUs sharing one L3: the CCD/CCX grouping
    std::vector<CacheLevel> caches;                     // L1d, L1i, L2, L3, in that order
    Fact<std::vector<std::string>> isaFlags;            // entries of kIsaFlags this CPU reports, in list order

    Fact<std::uint64_t> memTotalBytes;                  // MemTotal: what the OS can see
    Fact<std::uint64_t> memAvailableBytes;
    Fact<std::uint64_t> swapTotalBytes;
    Fact<int> numaNodes;
    Fact<std::uint64_t> memoryInstalledBytes;           // needs root (dmidecode): always unknown here

    Fact<std::string> cpuGovernor;                      // cpu0 scaling_governor
    Fact<std::string> energyPerformancePreference;      // cpu0 energy_performance_preference
    Fact<double> maxFreqMhz;                            // cpu0 cpuinfo_max_freq
    Fact<bool> boostEnabled;
    Fact<std::string> amdPstateStatus;
    Fact<std::string> transparentHugepages;             // the bracketed mode, e.g. "madvise"

    Fact<std::vector<GpuDevice>> gpus;
};

/// Reads every fact under `root` (default: the real machine). Never throws; unreadable items are unknown.
[[nodiscard]] MachineFacts readMachineFacts(const std::filesystem::path& root = "/");

/// Deterministic JSON: fixed key order, `{"schema":"sub0llm-machine-1","facts":{...}}`. Each fact is
/// `{"value": <value or null>, "reason": "<why unknown, empty when known>"}`.
[[nodiscard]] std::string toJson(const MachineFacts& facts);

/// A short human-readable report, a few lines per section.
[[nodiscard]] std::string toSummary(const MachineFacts& facts);

namespace detail {

namespace fs = std::filesystem;

template <typename T>
Fact<T> known(T value) {
    return Fact<T>{std::move(value), std::string{}};
}

template <typename T>
Fact<T> unknown(std::string reason) {
    return Fact<T>{std::nullopt, std::move(reason)};
}

inline std::string_view trimView(std::string_view s) {
    constexpr std::string_view ws = " \t\r\n\f\v";
    const std::size_t first = s.find_first_not_of(ws);
    if (first == std::string_view::npos) return {};
    const std::size_t last = s.find_last_not_of(ws);
    return s.substr(first, last - first + 1);
}

inline fs::path at(const fs::path& root, std::string_view rel) {
    return root / fs::path(rel);
}

inline std::optional<std::string> readFile(const fs::path& path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) return std::nullopt;
    return text;
}

inline std::optional<std::string> readTextOpt(const fs::path& root, std::string_view rel) {
    const auto text = readFile(at(root, rel));
    if (!text) return std::nullopt;
    return std::string(trimView(*text));
}

inline Fact<std::string> readTextFact(const fs::path& root, std::string_view rel) {
    const auto text = readTextOpt(root, rel);
    if (!text) return unknown<std::string>(std::string(rel) + " not readable");
    return known(*text);
}

inline bool parseInt(std::string_view s, int& out) {
    s = trimView(s);
    if (s.empty()) return false;
    const char* last = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), last, out);
    return ec == std::errc{} && ptr == last;
}

inline std::optional<std::uint64_t> parseUint(std::string_view s) {
    s = trimView(s);
    if (s.empty()) return std::nullopt;
    const char* last = s.data() + s.size();
    std::uint64_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), last, v);
    if (ec != std::errc{} || ptr != last) return std::nullopt;
    return v;
}

inline Fact<std::uint64_t> readUintFact(const fs::path& root, std::string_view rel) {
    const auto text = readFile(at(root, rel));
    if (!text) return unknown<std::uint64_t>(std::string(rel) + " not readable");
    if (const auto v = parseUint(*text)) return known(*v);
    return unknown<std::uint64_t>(std::string(rel) + " is not a number");
}

inline std::optional<int> readIntOpt(const fs::path& root, std::string_view rel) {
    const auto text = readFile(at(root, rel));
    int v = 0;
    if (!text || !parseInt(*text, v)) return std::nullopt;
    return v;
}

inline Fact<bool> readBoolFact(const fs::path& root, std::string_view first, std::string_view second) {
    for (const std::string_view rel : {first, second}) {
        const auto text = readTextOpt(root, rel);
        if (!text) continue;
        if (*text == "1") return known(true);
        if (*text == "0") return known(false);
        return unknown<bool>(std::string(rel) + " is neither 0 nor 1");
    }
    return unknown<bool>(std::string(first) + " and " + std::string(second) + " not readable");
}

/// Parses sysfs cpu lists such as "0-7,16-23". Empty on malformed input.
inline std::vector<int> parseCpuList(std::string_view s) {
    std::vector<int> out;
    s = trimView(s);
    while (!s.empty()) {
        const std::size_t comma = s.find(',');
        const std::string_view item = trimView(s.substr(0, comma));
        s = comma == std::string_view::npos ? std::string_view{} : s.substr(comma + 1);
        if (item.empty()) return {};
        const std::size_t dash = item.find('-');
        int lo = 0;
        int hi = 0;
        if (!parseInt(item.substr(0, dash), lo)) return {};
        hi = lo;
        if (dash != std::string_view::npos && !parseInt(item.substr(dash + 1), hi)) return {};
        if (hi < lo) return {};
        for (int i = lo; i <= hi; ++i) out.push_back(i);
    }
    return out;
}

/// Parses sysfs cache sizes such as "32K", "1M", "512".
inline std::optional<std::uint64_t> parseSize(std::string_view s) {
    s = trimView(s);
    if (s.empty()) return std::nullopt;
    std::uint64_t mult = 1;
    switch (s.back()) {
        case 'K': case 'k': mult = 1ull << 10; s.remove_suffix(1); break;
        case 'M': case 'm': mult = 1ull << 20; s.remove_suffix(1); break;
        case 'G': case 'g': mult = 1ull << 30; s.remove_suffix(1); break;
        default: break;
    }
    const auto v = parseUint(s);
    if (!v) return std::nullopt;
    return *v * mult;
}

template <typename Fn>
void forEachLine(std::string_view text, Fn&& fn) {
    while (!text.empty()) {
        const std::size_t eol = text.find('\n');
        fn(text.substr(0, eol));
        text = eol == std::string_view::npos ? std::string_view{} : text.substr(eol + 1);
    }
}

/// Value of the first "key : value" line whose key is exactly `key` (proc/cpuinfo and proc/meminfo style).
inline std::optional<std::string_view> keyedField(std::string_view text, std::string_view key) {
    std::optional<std::string_view> found;
    forEachLine(text, [&](std::string_view line) {
        if (found) return;
        const std::size_t colon = line.find(':');
        if (colon != std::string_view::npos && trimView(line.substr(0, colon)) == key)
            found = trimView(line.substr(colon + 1));
    });
    return found;
}

inline int countKeyedLines(std::string_view text, std::string_view key) {
    int n = 0;
    forEachLine(text, [&](std::string_view line) {
        const std::size_t colon = line.find(':');
        if (colon != std::string_view::npos && trimView(line.substr(0, colon)) == key) ++n;
    });
    return n;
}

inline Fact<std::string> keyedTextFact(const std::optional<std::string>& text, std::string_view file,
                                       std::string_view key) {
    if (!text) return unknown<std::string>(std::string(file) + " not readable");
    const auto field = keyedField(*text, key);
    if (!field) return unknown<std::string>(std::string(key) + " not in " + std::string(file));
    return known(std::string(*field));
}

/// Whether `word` appears as a whole space-separated token of `list`.
inline bool hasWord(std::string_view list, std::string_view word) {
    while (!list.empty()) {
        list = trimView(list);
        const std::size_t sp = list.find(' ');
        if (list.substr(0, sp) == word) return true;
        if (sp == std::string_view::npos) break;
        list = list.substr(sp + 1);
    }
    return false;
}

inline Fact<std::uint64_t> meminfoBytes(const std::optional<std::string>& text, std::string_view key) {
    if (!text) return unknown<std::uint64_t>("proc/meminfo not readable");
    const auto field = keyedField(*text, key);
    if (!field) return unknown<std::uint64_t>(std::string(key) + " not in proc/meminfo");
    const auto kib = parseUint(field->substr(0, field->find(' ')));
    if (!kib) return unknown<std::uint64_t>(std::string(key) + " is not a number in proc/meminfo");
    return known(*kib * 1024u);
}

/// The number after `prefix` when the name is exactly `prefix` followed by digits ("cpu12" -> 12).
inline std::optional<int> numberAfter(std::string_view name, std::string_view prefix) {
    if (name.size() <= prefix.size() || name.substr(0, prefix.size()) != prefix) return std::nullopt;
    int v = 0;
    if (!parseInt(name.substr(prefix.size()), v) || v < 0) return std::nullopt;
    return v;
}

/// Entry names directly under `dir`, or nullopt when the directory cannot be read.
inline std::optional<std::vector<std::string>> listDir(const fs::path& dir) {
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    if (ec) return std::nullopt;
    std::vector<std::string> names;
    const fs::directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) break;
        names.push_back(it->path().filename().string());
    }
    return names;
}

inline std::vector<int> listCpuIds(const fs::path& root) {
    std::vector<int> ids;
    if (const auto names = listDir(at(root, "sys/devices/system/cpu"))) {
        for (const std::string& name : *names)
            if (const auto n = numberAfter(name, "cpu")) ids.push_back(*n);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

struct CacheRow {
    int level = -1;
    std::string type;
    std::optional<std::uint64_t> sizeBytes;
    std::vector<int> shared;   // sorted logical CPUs sharing this cache
};

inline std::vector<CacheRow> readCacheRows(const fs::path& root, int cpu) {
    std::vector<CacheRow> rows;
    for (int idx = 0; idx < 16; ++idx) {
        const std::string rel = "sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" +
                                std::to_string(idx) + "/";
        const auto level = readIntOpt(root, rel + "level");
        if (!level) break;
        CacheRow row;
        row.level = *level;
        row.type = readTextOpt(root, rel + "type").value_or("");
        if (const auto size = readTextOpt(root, rel + "size")) row.sizeBytes = parseSize(*size);
        row.shared = parseCpuList(readTextOpt(root, rel + "shared_cpu_list").value_or(""));
        std::sort(row.shared.begin(), row.shared.end());
        rows.push_back(std::move(row));
    }
    return rows;
}

/// Index into the caches vector: 0 L1d, 1 L1i, 2 L2, 3 L3; -1 for anything else.
inline int cacheSlot(const CacheRow& r) {
    if (r.level == 1 && r.type == "Data") return 0;
    if (r.level == 1 && r.type == "Instruction") return 1;
    if (r.level == 2) return 2;
    if (r.level == 3) return 3;
    return -1;
}

constexpr std::string_view kCacheNames[] = {"L1d", "L1i", "L2", "L3"};

inline Fact<std::string> thpFact(const fs::path& root) {
    const auto text = readTextOpt(root, "sys/kernel/mm/transparent_hugepage/enabled");
    if (!text) return unknown<std::string>("sys/kernel/mm/transparent_hugepage/enabled not readable");
    const std::size_t open = text->find('[');
    const std::size_t close = text->find(']');
    if (open != std::string::npos && close != std::string::npos && close > open)
        return known(text->substr(open + 1, close - open - 1));
    return known(*text);
}

/// "0x1002" -> "1002"; lowercase, no prefix.
inline std::string hexId(std::string_view s) {
    s = trimView(s);
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s.remove_prefix(2);
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

inline std::string driverFromUevent(const std::optional<std::string>& uevent) {
    if (!uevent) return {};
    std::string driver;
    constexpr std::string_view key = "DRIVER=";
    forEachLine(*uevent, [&](std::string_view line) {
        if (driver.empty() && line.substr(0, key.size()) == key)
            driver = std::string(trimView(line.substr(key.size())));
    });
    return driver;
}

// ---- JSON writer (hand-written, dependency-free, fixed key order) ----------------------------------------

inline void appendInt(std::string& o, std::int64_t v) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    o.append(buf, r.ptr);
}

inline void appendUint(std::string& o, std::uint64_t v) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    o.append(buf, r.ptr);
}

/// Locale-independent fixed-point formatting.
inline void appendFixed(std::string& o, double v, int digits) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed, digits);
    o.append(buf, r.ptr);
}

inline void jsonEscapeTo(std::string& o, std::string_view s) {
    constexpr char hex[] = "0123456789abcdef";
    o += '"';
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) {
                    o += "\\u00";
                    o += hex[c >> 4];
                    o += hex[c & 0xF];
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    o += '"';
}

template <typename Fn>
void jsonArray(std::string& o, std::size_t n, Fn&& elem) {
    o += '[';
    for (std::size_t i = 0; i < n; ++i) {
        if (i) o += ',';
        elem(i);
    }
    o += ']';
}

// Forward declarations: the overloads are defined after JsonObject, which the object-valued ones need.
inline void jsonValue(std::string& o, std::string_view v);
inline void jsonValue(std::string& o, int v);
inline void jsonValue(std::string& o, std::uint64_t v);
inline void jsonValue(std::string& o, double v);
inline void jsonValue(std::string& o, bool v);
inline void jsonValue(std::string& o, const std::vector<int>& v);
inline void jsonValue(std::string& o, const std::vector<std::vector<int>>& v);
inline void jsonValue(std::string& o, const std::vector<std::string>& v);
inline void jsonValue(std::string& o, const LogicalCpu& v);
inline void jsonValue(std::string& o, const std::vector<LogicalCpu>& v);
inline void jsonValue(std::string& o, const GpuDevice& v);
inline void jsonValue(std::string& o, const std::vector<GpuDevice>& v);

template <typename T>
void jsonFact(std::string& o, const Fact<T>& f) {
    o += "{\"value\":";
    if (f.value) jsonValue(o, *f.value);
    else o += "null";
    o += ",\"reason\":";
    jsonEscapeTo(o, f.reason);
    o += '}';
}

/// Writes one JSON object; the closing brace is emitted by the destructor, so scopes nest naturally.
class JsonObject {
public:
    explicit JsonObject(std::string& out) : out_(out) { out_ += '{'; }
    JsonObject(JsonObject& parent, std::string_view name) : out_(parent.out_) {
        parent.key(name);
        out_ += '{';
    }
    ~JsonObject() { out_ += '}'; }
    JsonObject(const JsonObject&) = delete;
    JsonObject& operator=(const JsonObject&) = delete;

    void key(std::string_view name) {
        if (!first_) out_ += ',';
        first_ = false;
        jsonEscapeTo(out_, name);
        out_ += ':';
    }
    template <typename T>
    void value(std::string_view name, const T& v) {
        key(name);
        jsonValue(out_, v);
    }
    template <typename T>
    void fact(std::string_view name, const Fact<T>& f) {
        key(name);
        jsonFact(out_, f);
    }

private:
    std::string& out_;
    bool first_ = true;
};

inline void jsonValue(std::string& o, std::string_view v) { jsonEscapeTo(o, v); }
inline void jsonValue(std::string& o, int v) { appendInt(o, v); }
inline void jsonValue(std::string& o, std::uint64_t v) { appendUint(o, v); }
inline void jsonValue(std::string& o, double v) { appendFixed(o, v, 2); }
inline void jsonValue(std::string& o, bool v) { o += v ? "true" : "false"; }

inline void jsonValue(std::string& o, const std::vector<int>& v) {
    jsonArray(o, v.size(), [&](std::size_t i) { appendInt(o, v[i]); });
}

inline void jsonValue(std::string& o, const std::vector<std::vector<int>>& v) {
    jsonArray(o, v.size(), [&](std::size_t i) { jsonValue(o, v[i]); });
}

inline void jsonValue(std::string& o, const std::vector<std::string>& v) {
    jsonArray(o, v.size(), [&](std::size_t i) { jsonEscapeTo(o, v[i]); });
}

inline void jsonValue(std::string& o, const LogicalCpu& v) {
    JsonObject obj(o);
    obj.value("id", v.id);
    obj.value("package", v.package);
    obj.value("coreId", v.coreId);
    obj.value("physicalCore", v.physicalCore);
}

inline void jsonValue(std::string& o, const std::vector<LogicalCpu>& v) {
    jsonArray(o, v.size(), [&](std::size_t i) { jsonValue(o, v[i]); });
}

inline void jsonValue(std::string& o, const GpuDevice& v) {
    JsonObject obj(o);
    obj.value("card", std::string_view(v.card));
    obj.value("vendorId", std::string_view(v.vendorId));
    obj.value("deviceId", std::string_view(v.deviceId));
    obj.value("driver", std::string_view(v.driver));
    obj.fact("vramTotalBytes", v.vramTotalBytes);
    obj.fact("gttTotalBytes", v.gttTotalBytes);
}

inline void jsonValue(std::string& o, const std::vector<GpuDevice>& v) {
    jsonArray(o, v.size(), [&](std::size_t i) { jsonValue(o, v[i]); });
}

// ---- human-readable summary ----------------------------------------------------------------------------

inline void appendCpuRanges(std::string& o, const std::vector<int>& ids) {
    std::size_t i = 0;
    while (i < ids.size()) {
        std::size_t j = i;
        while (j + 1 < ids.size() && ids[j + 1] == ids[j] + 1) ++j;
        if (i) o += ',';
        appendInt(o, ids[i]);
        if (j > i) {
            o += '-';
            appendInt(o, ids[j]);
        }
        i = j + 1;
    }
}

template <typename T, typename Fmt>
void putFact(std::string& o, const Fact<T>& f, Fmt&& fmt) {
    if (f.value) {
        fmt(o, *f.value);
    } else {
        o += "unknown (";
        o += f.reason;
        o += ')';
    }
}

inline void sText(std::string& o, const std::string& v) { o += v; }
inline void sInt(std::string& o, const int& v) { appendInt(o, v); }
inline void sMhz(std::string& o, const double& v) {
    appendFixed(o, v, 0);
    o += " MHz";
}
inline void sOnOff(std::string& o, const bool& v) { o += v ? "on" : "off"; }
inline void sBytes(std::string& o, const std::uint64_t& b) {
    constexpr std::uint64_t kK = 1ull << 10;
    constexpr std::uint64_t kM = kK << 10;
    constexpr std::uint64_t kG = kM << 10;
    if (b >= kG) {
        appendFixed(o, static_cast<double>(b) / static_cast<double>(kG), 1);
        o += " GiB";
    } else if (b >= kM) {
        appendFixed(o, static_cast<double>(b) / static_cast<double>(kM), 1);
        o += " MiB";
    } else if (b >= kK) {
        appendUint(o, b / kK);
        o += " KiB";
    } else {
        appendUint(o, b);
        o += " B";
    }
}

}  // namespace detail

[[nodiscard]] inline MachineFacts readMachineFacts(const std::filesystem::path& root) {
    using namespace detail;
    MachineFacts f;

    f.hostname = readTextFact(root, "proc/sys/kernel/hostname");
    f.kernelRelease = readTextFact(root, "proc/sys/kernel/osrelease");

    // ---- /proc/cpuinfo: model, vendor, ISA flags, and the logical-CPU count fallback ----
    const auto cpuinfo = readFile(at(root, "proc/cpuinfo"));
    f.cpuModel = keyedTextFact(cpuinfo, "proc/cpuinfo", "model name");
    f.vendor = keyedTextFact(cpuinfo, "proc/cpuinfo", "vendor_id");
    if (!cpuinfo) {
        f.isaFlags = unknown<std::vector<std::string>>("proc/cpuinfo not readable");
    } else if (const auto flags = keyedField(*cpuinfo, "flags")) {
        std::vector<std::string> present;
        for (const std::string_view isa : kIsaFlags)
            if (hasWord(*flags, isa)) present.emplace_back(isa);
        f.isaFlags = known(std::move(present));
    } else {
        f.isaFlags = unknown<std::vector<std::string>>("no flags line in proc/cpuinfo");
    }

    // ---- logical CPUs and topology ----
    const std::vector<int> cpuIds = listCpuIds(root);
    if (!cpuIds.empty()) {
        f.logicalCpus = known(static_cast<int>(cpuIds.size()));
    } else if (cpuinfo && countKeyedLines(*cpuinfo, "processor") > 0) {
        f.logicalCpus = known(countKeyedLines(*cpuinfo, "processor"));
    } else {
        f.logicalCpus = unknown<int>("no sys/devices/system/cpu/cpuN and no processor line in proc/cpuinfo");
    }

    // Physical core identity: (package, core_id), falling back to the lowest sibling in thread_siblings_list.
    std::string topoWhy;
    if (cpuIds.empty()) topoWhy = "no sys/devices/system/cpu/cpuN directories";
    std::vector<LogicalCpu> cpus;
    std::map<std::pair<int, int>, int> coreIndex;
    int unresolved = 0;
    for (const int id : cpuIds) {
        const std::string base = "sys/devices/system/cpu/cpu" + std::to_string(id) + "/topology/";
        const auto pkg = readIntOpt(root, base + "physical_package_id");
        const auto core = readIntOpt(root, base + "core_id");
        std::pair<int, int> key{-1, -1};
        if (pkg && core) {
            key = {*pkg, *core};
        } else {
            const std::vector<int> sib = parseCpuList(readTextOpt(root, base + "thread_siblings_list").value_or(""));
            if (sib.empty()) {
                ++unresolved;
                continue;
            }
            key = {-1, *std::min_element(sib.begin(), sib.end())};
        }
        LogicalCpu c;
        c.id = id;
        c.package = pkg.value_or(-1);
        c.coreId = core.value_or(-1);
        c.physicalCore = coreIndex.try_emplace(key, static_cast<int>(coreIndex.size())).first->second;
        cpus.push_back(c);
    }
    if (topoWhy.empty() && unresolved > 0) {
        topoWhy = std::to_string(unresolved) + " of " + std::to_string(cpuIds.size()) +
                  " CPUs have no core_id/physical_package_id or thread_siblings_list";
    }

    if (topoWhy.empty()) {
        f.cpus = known(cpus);
        f.physicalCores = known(static_cast<int>(coreIndex.size()));

        std::vector<int> threads(coreIndex.size(), 0);
        std::vector<bool> seen(coreIndex.size(), false);
        std::vector<int> one;
        for (const LogicalCpu& c : cpus) {
            ++threads[static_cast<std::size_t>(c.physicalCore)];
            if (!seen[static_cast<std::size_t>(c.physicalCore)]) {
                seen[static_cast<std::size_t>(c.physicalCore)] = true;
                one.push_back(c.id);
            }
        }
        f.smtWays = known(threads.empty() ? 0 : *std::max_element(threads.begin(), threads.end()));
        f.onePerCore = known(one);

        std::map<int, std::vector<int>> byPackage;
        bool allPackages = true;
        for (const LogicalCpu& c : cpus) {
            if (c.package < 0) allPackages = false;
            else byPackage[c.package].push_back(c.id);
        }
        if (allPackages) {
            std::vector<std::vector<int>> groups;
            for (const auto& [pkgId, ids] : byPackage) groups.push_back(ids);
            f.coreSiblings = known(groups);
        } else {
            f.coreSiblings = unknown<std::vector<std::vector<int>>>("physical_package_id not readable for some CPUs");
        }
    } else {
        f.cpus = unknown<std::vector<LogicalCpu>>(topoWhy);
        f.physicalCores = unknown<int>(topoWhy);
        f.smtWays = unknown<int>(topoWhy);
        f.onePerCore = unknown<std::vector<int>>(topoWhy);
        f.coreSiblings = unknown<std::vector<std::vector<int>>>(topoWhy);
    }

    // ---- caches: per level size, instance count, and the L3 sharing groups ----
    struct Seen {
        bool any = false;
        std::optional<std::uint64_t> size;
        std::set<std::vector<int>> shared;
    };
    std::vector<Seen> seenByLevel(4);
    for (const int id : cpuIds) {
        for (const CacheRow& r : readCacheRows(root, id)) {
            const int slot = cacheSlot(r);
            if (slot < 0) continue;
            Seen& s = seenByLevel[static_cast<std::size_t>(slot)];
            s.any = true;
            if (!s.size) s.size = r.sizeBytes;
            if (!r.shared.empty()) s.shared.insert(r.shared);
        }
    }
    constexpr std::string_view noCache = "no cache index under sys/devices/system/cpu/cpuN/cache";
    for (std::size_t i = 0; i < 4; ++i) {
        const Seen& s = seenByLevel[i];
        CacheLevel level;
        level.name = std::string(kCacheNames[i]);
        if (!s.any) {
            level.sizeBytes = unknown<std::uint64_t>(std::string(noCache));
            level.instances = unknown<int>(std::string(noCache));
        } else {
            level.sizeBytes = s.size ? known(*s.size) : unknown<std::uint64_t>("cache size not readable");
            level.instances = s.shared.empty() ? unknown<int>("no shared_cpu_list")
                                               : known(static_cast<int>(s.shared.size()));
        }
        f.caches.push_back(std::move(level));
    }
    const Seen& l3 = seenByLevel[3];
    if (!l3.any || l3.shared.empty()) {
        f.l3Groups = unknown<std::vector<std::vector<int>>>(std::string(noCache) + " for level 3");
    } else {
        f.l3Groups = known(std::vector<std::vector<int>>(l3.shared.begin(), l3.shared.end()));
    }

    // ---- memory ----
    const auto meminfo = readFile(at(root, "proc/meminfo"));
    f.memTotalBytes = meminfoBytes(meminfo, "MemTotal");
    f.memAvailableBytes = meminfoBytes(meminfo, "MemAvailable");
    f.swapTotalBytes = meminfoBytes(meminfo, "SwapTotal");
    if (const auto nodes = listDir(at(root, "sys/devices/system/node"))) {
        int count = 0;
        for (const std::string& name : *nodes)
            if (numberAfter(name, "node")) ++count;
        f.numaNodes = known(count);
    } else {
        f.numaNodes = unknown<int>("sys/devices/system/node not readable");
    }
    f.memoryInstalledBytes = unknown<std::uint64_t>(
        "needs root (dmidecode): installed size, speed and channel count are not readable without root");

    // ---- cpufreq, pstate, THP ----
    f.cpuGovernor = readTextFact(root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    f.energyPerformancePreference =
        readTextFact(root, "sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference");
    const Fact<std::uint64_t> maxKhz = readUintFact(root, "sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    f.maxFreqMhz = maxKhz.value ? known(static_cast<double>(*maxKhz.value) / 1000.0)
                                : unknown<double>(maxKhz.reason);
    f.boostEnabled = readBoolFact(root, "sys/devices/system/cpu/cpufreq/boost",
                                  "sys/devices/system/cpu/cpu0/cpufreq/boost");
    f.amdPstateStatus = readTextFact(root, "sys/devices/system/cpu/amd_pstate/status");
    f.transparentHugepages = thpFact(root);

    // ---- GPUs: every sys/class/drm/cardN with a device/vendor file ----
    if (const auto cards = listDir(at(root, "sys/class/drm"))) {
        std::vector<std::pair<int, std::string>> order;
        for (const std::string& name : *cards)
            if (const auto n = numberAfter(name, "card")) order.emplace_back(*n, name);
        std::sort(order.begin(), order.end());
        std::vector<GpuDevice> gpus;
        for (const auto& entry : order) {
            const std::string& name = entry.second;
            const std::string rel = "sys/class/drm/" + name + "/device/";
            const auto vendor = readTextOpt(root, rel + "vendor");
            if (!vendor) continue;   // connector entries such as card0-DP-1 carry no device/vendor
            GpuDevice g;
            g.card = name;
            g.vendorId = hexId(*vendor);
            g.deviceId = hexId(readTextOpt(root, rel + "device").value_or(""));
            g.driver = driverFromUevent(readTextOpt(root, rel + "uevent"));
            g.vramTotalBytes = readUintFact(root, rel + "mem_info_vram_total");
            g.gttTotalBytes = readUintFact(root, rel + "mem_info_gtt_total");
            gpus.push_back(std::move(g));
        }
        f.gpus = known(gpus);
    } else {
        f.gpus = unknown<std::vector<GpuDevice>>("sys/class/drm not readable");
    }

    return f;
}

[[nodiscard]] inline std::string toJson(const MachineFacts& f) {
    using namespace detail;
    std::string out;
    {
        JsonObject root(out);
        root.value("schema", std::string_view("sub0llm-machine-1"));
        JsonObject facts(root, "facts");
        {
            JsonObject host(facts, "host");
            host.fact("hostname", f.hostname);
            host.fact("kernelRelease", f.kernelRelease);
        }
        {
            JsonObject cpu(facts, "cpu");
            cpu.fact("model", f.cpuModel);
            cpu.fact("vendor", f.vendor);
            cpu.fact("logicalCpus", f.logicalCpus);
            cpu.fact("physicalCores", f.physicalCores);
            cpu.fact("smtWays", f.smtWays);
            cpu.fact("isaFlags", f.isaFlags);
            JsonObject topo(cpu, "topology");
            topo.fact("cpus", f.cpus);
            topo.fact("onePerCore", f.onePerCore);
            topo.fact("coreSiblings", f.coreSiblings);
            topo.fact("l3Groups", f.l3Groups);
        }
        facts.key("caches");
        jsonArray(out, f.caches.size(), [&](std::size_t i) {
            const CacheLevel& c = f.caches[i];
            JsonObject level(out);
            level.value("name", std::string_view(c.name));
            level.fact("sizeBytes", c.sizeBytes);
            level.fact("instances", c.instances);
        });
        {
            JsonObject mem(facts, "memory");
            mem.fact("totalBytes", f.memTotalBytes);
            mem.fact("availableBytes", f.memAvailableBytes);
            mem.fact("swapTotalBytes", f.swapTotalBytes);
            mem.fact("numaNodes", f.numaNodes);
            mem.fact("installedBytes", f.memoryInstalledBytes);
        }
        {
            JsonObject freq(facts, "cpufreq");
            freq.fact("governor", f.cpuGovernor);
            freq.fact("energyPerformancePreference", f.energyPerformancePreference);
            freq.fact("maxFreqMhz", f.maxFreqMhz);
            freq.fact("boost", f.boostEnabled);
            freq.fact("amdPstateStatus", f.amdPstateStatus);
            freq.fact("transparentHugepages", f.transparentHugepages);
        }
        facts.fact("gpus", f.gpus);
    }
    return out;
}

[[nodiscard]] inline std::string toSummary(const MachineFacts& f) {
    using namespace detail;
    std::string o = "sub0llm machine facts\n";
    const auto label = [&o](std::string_view name) {
        o += "  ";
        o += name;
        o.append(name.size() < 13 ? 13 - name.size() : 1, ' ');
    };

    label("host");
    putFact(o, f.hostname, sText);
    o += "  kernel ";
    putFact(o, f.kernelRelease, sText);
    o += '\n';

    label("cpu");
    putFact(o, f.cpuModel, sText);
    o += " (";
    putFact(o, f.vendor, sText);
    o += ")\n";

    label("topology");
    putFact(o, f.logicalCpus, sInt);
    o += " logical CPUs, ";
    putFact(o, f.physicalCores, sInt);
    o += " physical cores, SMT ";
    putFact(o, f.smtWays, sInt);
    o += " way\n";

    label("L3 groups");
    if (!f.l3Groups.value) {
        o += "unknown (" + f.l3Groups.reason + ")\n";
    } else {
        o += std::to_string(f.l3Groups.value->size()) + " (each group shares one L3)\n";
        std::size_t n = 0;
        for (const std::vector<int>& g : *f.l3Groups.value) {
            o.append(15, ' ');
            o += "group " + std::to_string(n++) + ": ";
            appendCpuRanges(o, g);
            o += " (" + std::to_string(g.size()) + " CPUs)\n";
        }
    }

    label("caches");
    for (std::size_t i = 0; i < f.caches.size(); ++i) {
        const CacheLevel& c = f.caches[i];
        if (i) o += ", ";
        o += c.name;
        o += ' ';
        putFact(o, c.sizeBytes, sBytes);
        o += " x";
        putFact(o, c.instances, sInt);
    }
    o += '\n';

    label("isa");
    if (!f.isaFlags.value) {
        o += "unknown (" + f.isaFlags.reason + ")\n";
    } else {
        o += "present:";
        for (const std::string& s : *f.isaFlags.value) {
            o += ' ';
            o += s;
        }
        o += " (of " + std::to_string(std::size(kIsaFlags)) + " checked)\n";
    }

    label("memory");
    o += "OS-visible ";
    putFact(o, f.memTotalBytes, sBytes);
    o += ", available ";
    putFact(o, f.memAvailableBytes, sBytes);
    o += ", swap ";
    putFact(o, f.swapTotalBytes, sBytes);
    o += ", NUMA nodes ";
    putFact(o, f.numaNodes, sInt);
    o += '\n';

    label("installed");
    putFact(o, f.memoryInstalledBytes, sBytes);
    o += '\n';

    label("cpufreq");
    o += "governor ";
    putFact(o, f.cpuGovernor, sText);
    o += ", EPP ";
    putFact(o, f.energyPerformancePreference, sText);
    o += ", max ";
    putFact(o, f.maxFreqMhz, sMhz);
    o += ", boost ";
    putFact(o, f.boostEnabled, sOnOff);
    o += '\n';

    label("pstate, THP");
    o += "amd_pstate ";
    putFact(o, f.amdPstateStatus, sText);
    o += ", THP ";
    putFact(o, f.transparentHugepages, sText);
    o += '\n';

    label("gpu");
    if (!f.gpus.value) {
        o += "unknown (" + f.gpus.reason + ")\n";
    } else if (f.gpus.value->empty()) {
        o += "none found\n";
    } else {
        bool first = true;
        for (const GpuDevice& g : *f.gpus.value) {
            if (!first) o.append(15, ' ');
            first = false;
            o += g.card + " " + g.vendorId + ":" + g.deviceId;
            o += g.driver.empty() ? std::string(" driver unknown") : " driver " + g.driver;
            o += ", VRAM ";
            putFact(o, g.vramTotalBytes, sBytes);
            o += ", GTT ";
            putFact(o, g.gttTotalBytes, sBytes);
            o += '\n';
        }
    }
    return o;
}

}  // namespace sub0::measure
