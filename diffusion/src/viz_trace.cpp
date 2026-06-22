#include "sub0diff/viz/trace_json.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace sub0diff::viz {

std::string serialize_trace_json(const GenerationTrace& trace,
                                 const std::function<std::string(std::int32_t)>& token_str,
                                 std::int32_t mask_id) {
    using nlohmann::json;
    auto tok = [&](std::int32_t id) { return id == mask_id ? std::string("·") : token_str(id); };
    auto r3  = [](float v) { return std::round(v * 1000.0f) / 1000.0f; };  // 3-dp for compactness

    json j;
    j["version"]   = 1;
    j["meta"] = {
        {"T", trace.T}, {"prompt", trace.prompt}, {"model", trace.model},
        {"commit_order", trace.commit_order}, {"temperature", trace.temperature},
        {"conf_threshold", trace.conf_threshold}, {"min_commit_frac", trace.min_commit_frac},
        {"remask_threshold", trace.remask_threshold}, {"N", trace.N}, {"c", trace.c}, {"w", trace.w},
        {"levels", trace.levels},
        {"n_frames", static_cast<std::int64_t>(trace.frames.size())},
    };
    j["frames"] = json::array();
    for (const auto& f : trace.frames) {
        json jf;
        jf["iter"] = f.iter;
        jf["ms"] = std::round(f.ms * 100.0) / 100.0;
        jf["committed_count"] = f.committed_count;
        json toks = json::array(), pred = json::array();
        for (auto id : f.tokens)    toks.push_back(tok(id));
        for (auto id : f.predicted) pred.push_back(tok(id));
        jf["tokens"]     = std::move(toks);
        jf["predicted"]  = std::move(pred);
        jf["masked"]     = f.masked;       // 0/1 arrays
        jf["committed"]  = f.committed;
        jf["remasked"]   = f.remasked;
        json conf = json::array(), ent = json::array();
        for (auto v : f.confidence) conf.push_back(r3(v));
        for (auto v : f.entropy)    ent.push_back(r3(v));
        jf["confidence"] = std::move(conf);
        jf["entropy"]    = std::move(ent);
        // Phase B: TRUE per-level activation magnitudes (one inner array per MERA level, order =
        // meta.levels). Absent/empty for flat — the viewer falls back to the derived settled-fraction.
        if (!f.level_rms.empty()) {
            json lv = json::array();
            for (const auto& level : f.level_rms) {
                json slots = json::array();
                for (auto v : level) slots.push_back(r3(v));
                lv.push_back(std::move(slots));
            }
            jf["level_rms"] = std::move(lv);
        }
        // Phase E: the GIST decoded to tokens — per top-level slot, the top-k nearest tokens (strings)
        // and their projection scores. Absent for flat models. Lets the viewer show the coarse "plan."
        if (!f.gist.empty()) {
            json gtoks = json::array(), gsc = json::array();
            for (const auto& slot : f.gist.tokens) {
                json st = json::array();
                for (auto id : slot) st.push_back(tok(id));
                gtoks.push_back(std::move(st));
            }
            for (const auto& slot : f.gist.scores) {
                json ss = json::array();
                for (auto v : slot) ss.push_back(r3(v));
                gsc.push_back(std::move(ss));
            }
            jf["gist_tokens"] = std::move(gtoks);
            jf["gist_scores"] = std::move(gsc);
        }
        j["frames"].push_back(std::move(jf));
    }
    return j.dump();
}

std::size_t write_trace_json(const GenerationTrace& trace, const std::string& path,
                             const std::function<std::string(std::int32_t)>& token_str,
                             std::int32_t mask_id) {
    std::ofstream os(path);
    if (!os) throw std::runtime_error("write_trace_json: cannot open " + path);
    os << serialize_trace_json(trace, token_str, mask_id);
    return trace.frames.size();
}

}  // namespace sub0diff::viz
