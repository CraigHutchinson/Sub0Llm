// blend_schedule.cpp -- simdjson::ondemand parser + validator for the blend-schedule JSON format
// (sub0/blend_schedule.hpp's ScheduleSpec). Single forward pass per object, matching this project's
// established JSON convention (see run_config.cpp's own header comment) -- this file is the first place
// in the codebase that needs to walk JSON ARRAYS/nested objects (config.json is flat), so the array/
// per-element-object loops here are new idioms for this project, not reuse of an existing helper.
//
// Deliberately does NOT call sub0::log itself: this file lives in sub0_core.dll, whose sub0::log::logger()
// is a SEPARATE header-only static instance from sub0_train.dll's (each DLL that includes the header-only
// logger gets its own copy) -- a message logged here would never reach train.log's file sink, which
// sub0_train.dll's own copy configures. Returning structured error/warning text and letting the caller
// (train_stage.cpp, where the correctly-configured logger lives) do the actual logging avoids that gap
// entirely, and matches read_config_json's own existing precedent of doing no internal logging.

#include "sub0/blend_schedule.hpp"

#include <simdjson.h>

#include <cmath>
#include <format>

namespace sub0 {

namespace {

bool read_double(simdjson::ondemand::value v, double& out) {
    double d;
    if (v.get_double().get(d)) return false;
    out = d;
    return true;
}
bool read_int(simdjson::ondemand::value v, int& out) {
    int64_t i;
    if (v.get_int64().get(i)) return false;
    out = static_cast<int>(i);
    return true;
}
bool read_string(simdjson::ondemand::value v, std::string& out) {
    std::string_view sv;
    if (v.get_string().get(sv)) return false;
    out.assign(sv);
    return true;
}

bool parse_source(simdjson::ondemand::value v, SourceSpec& s, std::string& error) {
    simdjson::ondemand::object obj;
    if (v.get_object().get(obj)) { error = "a 'sources' entry is not a JSON object"; return false; }
    for (auto field : obj) {
        std::string_view key;
        if (field.unescaped_key().get(key)) continue;
        if (key == "name") { if (!read_string(field.value(), s.name)) { error = "source 'name' must be a string"; return false; } continue; }
        if (key == "corpus") { if (!read_string(field.value(), s.corpus)) { error = "source 'corpus' must be a string"; return false; } continue; }
        if (key == "generator") { if (!read_string(field.value(), s.generator)) { error = "source 'generator' must be a string"; return false; } continue; }
        if (key == "gsm8k") { if (!read_string(field.value(), s.gsm8k_path)) { error = "source 'gsm8k' must be a string"; return false; } continue; }
        if (key == "params") {
            simdjson::ondemand::object params;
            if (field.value().get_object().get(params)) { error = "source 'params' must be an object"; return false; }
            for (auto p : params) {
                std::string_view pkey;
                if (p.unescaped_key().get(pkey)) continue;
                if (pkey == "n_oov")          { if (!read_int(p.value(), s.n_oov))          { error = "params.n_oov must be an integer"; return false; } }
                else if (pkey == "tasks_per_oov")  { if (!read_int(p.value(), s.tasks_per_oov))  { error = "params.tasks_per_oov must be an integer"; return false; } }
                else if (pkey == "contains_k")     { if (!read_int(p.value(), s.contains_k))     { error = "params.contains_k must be an integer"; return false; } }
                else if (pkey == "n_examples")     { if (!read_int(p.value(), s.n_examples))     { error = "params.n_examples must be an integer"; return false; } }
                else if (pkey == "chain_frac")     { if (!read_double(p.value(), s.chain_frac))  { error = "params.chain_frac must be a number"; return false; } }
                else if (pkey == "max_digits")     { if (!read_int(p.value(), s.max_digits))     { error = "params.max_digits must be an integer"; return false; } }
                else if (pkey == "tasks_per_word") { if (!read_int(p.value(), s.tasks_per_word)) { error = "params.tasks_per_word must be an integer"; return false; }
                }
                // An unrecognized param key is tolerated (forward-compat), matching config.json's own
                // stance on unrecognized top-level keys -- see registry.hpp's own comment on this.
            }
            continue;
        }
    }
    if (s.name.empty()) { error = "every source needs a non-empty 'name'"; return false; }
    const bool has_corpus = !s.corpus.empty(), has_gen = !s.generator.empty();
    if (has_corpus == has_gen) {
        error = std::format("source '{}' must set EXACTLY ONE of 'corpus' or 'generator' (has_corpus={}, "
                            "has_generator={})", s.name, has_corpus, has_gen);
        return false;
    }
    return true;
}

bool parse_stage(simdjson::ondemand::value v, ScheduleStage& st, std::string& error) {
    simdjson::ondemand::object obj;
    if (v.get_object().get(obj)) { error = "a 'schedule' entry is not a JSON object"; return false; }
    bool have_until_epoch = false, have_weights = false;
    for (auto field : obj) {
        std::string_view key;
        if (field.unescaped_key().get(key)) continue;
        if (key == "until_epoch") {
            simdjson::ondemand::json_type t;
            if (field.value().type().get(t)) { error = "stage 'until_epoch' has an unreadable type"; return false; }
            if (t == simdjson::ondemand::json_type::string) {
                std::string_view sv;
                if (field.value().get_string().get(sv)) { error = "stage 'until_epoch' string is unreadable"; return false; }
                if (sv != "end") { error = std::format("stage 'until_epoch' string must be \"end\", got \"{}\"", sv); return false; }
                st.until_epoch = std::numeric_limits<double>::infinity();
            } else if (t == simdjson::ondemand::json_type::number) {
                if (!read_double(field.value(), st.until_epoch)) { error = "stage 'until_epoch' number is unreadable"; return false; }
            } else {
                error = "stage 'until_epoch' must be a number or \"end\"";
                return false;
            }
            have_until_epoch = true;
            continue;
        }
        if (key == "weights") {
            simdjson::ondemand::object weights;
            if (field.value().get_object().get(weights)) { error = "stage 'weights' must be an object"; return false; }
            for (auto w : weights) {
                std::string_view name;
                if (w.unescaped_key().get(name)) continue;
                double rate = 0.0;
                if (!read_double(w.value(), rate)) { error = std::format("weights.{} must be a number", name); return false; }
                st.weights.emplace_back(std::string(name), rate);
            }
            have_weights = true;
            continue;
        }
    }
    if (!have_until_epoch) { error = "every schedule stage needs 'until_epoch'"; return false; }
    if (!have_weights || st.weights.empty()) { error = "every schedule stage needs a non-empty 'weights' object"; return false; }
    return true;
}

}  // namespace

bool parse_blend_schedule_json(const std::filesystem::path& path, ScheduleSpec& out, std::string& error,
                               std::vector<std::string>& warnings) {
    out = ScheduleSpec{};
    error.clear();
    warnings.clear();

    simdjson::padded_string json;
    if (simdjson::padded_string::load(path.string()).get(json)) {
        error = std::format("could not read '{}'", path.string());
        return false;
    }
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc)) { error = "not valid JSON"; return false; }
    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj)) { error = "top level must be a JSON object"; return false; }

    bool have_sources = false, have_schedule = false;
    std::string content_embed_str;
    for (auto field : obj) {
        std::string_view key;
        if (field.unescaped_key().get(key)) continue;
        if (key == "sources") {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr)) { error = "'sources' must be an array"; return false; }
            for (auto elem : arr) {
                SourceSpec s;
                if (!parse_source(elem.value(), s, error)) return false;
                out.sources.push_back(std::move(s));
            }
            have_sources = true;
            continue;
        }
        if (key == "schedule") {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr)) { error = "'schedule' must be an array"; return false; }
            for (auto elem : arr) {
                ScheduleStage st;
                if (!parse_stage(elem.value(), st, error)) return false;
                out.stages.push_back(std::move(st));
            }
            have_schedule = true;
            continue;
        }
        if (key == "content_embed") {
            if (!read_string(field.value(), content_embed_str)) { error = "'content_embed' must be a string"; return false; }
            continue;
        }
        if (key == "expected_plateau_epoch") {
            double v;
            if (!read_double(field.value(), v)) { error = "'expected_plateau_epoch' must be a number"; return false; }
            out.expected_plateau_epoch = v;
            continue;
        }
        // Unrecognized top-level key: tolerated (forward-compat), matching config.json's own stance.
    }
    if (!have_sources || out.sources.empty()) { error = "schedule needs a non-empty 'sources' array"; return false; }
    if (!have_schedule || out.stages.empty()) { error = "schedule needs a non-empty 'schedule' array"; return false; }

    // --- Cross-referential validation (needs the full parsed structure, not just per-element checks) ---

    // Unique names; EXACTLY one corpus-typed source (see this file's header + blend_schedule.hpp's own
    // SourceSpec doc comment on why a SECOND arbitrary file-backed corpus is out of scope for now).
    // Requiring at least one (not just at most one) matters: train_stage.cpp defines its whole epoch
    // clock (epoch_steps, frac_epoch, the val-NELBO split plateau-detection reads) against the base
    // corpus's own token count -- a schedule built entirely from generator sources would still run
    // against SOME corpus (whatever train_stage.cpp was invoked with), but "epoch" would silently stop
    // meaning what the schedule's own until_epoch stage boundaries assume it means.
    int corpus_sources = 0;
    for (std::size_t i = 0; i < out.sources.size(); ++i) {
        if (!out.sources[i].corpus.empty()) ++corpus_sources;
        for (std::size_t j = i + 1; j < out.sources.size(); ++j) {
            if (out.sources[i].name == out.sources[j].name) {
                error = std::format("duplicate source name '{}'", out.sources[i].name);
                return false;
            }
        }
    }
    if (corpus_sources > 1) {
        error = std::format("at most one 'corpus'-typed source is supported (found {}) -- a second "
                            "file-backed corpus needs its own tokenization/doc-index/on-demand plumbing "
                            "this build does not yet have; use additional 'generator' sources instead",
                            corpus_sources);
        return false;
    }
    if (corpus_sources == 0) {
        error = "schedule needs exactly one 'corpus'-typed source (the base corpus this build was "
               "configured against) -- 'epoch' is defined against its token count, so a schedule built "
               "entirely from 'generator' sources has no meaningful epoch clock";
        return false;
    }

    // Stages ascending by until_epoch; "end" (+infinity) must be exactly the last stage.
    for (std::size_t i = 1; i < out.stages.size(); ++i) {
        if (out.stages[i].until_epoch <= out.stages[i - 1].until_epoch) {
            error = std::format("schedule stages must have strictly ascending 'until_epoch' (stage {} "
                                "does not exceed stage {})", i, i - 1);
            return false;
        }
    }
    for (std::size_t i = 0; i < out.stages.size(); ++i) {
        const bool is_end = std::isinf(out.stages[i].until_epoch);
        if (is_end && i + 1 != out.stages.size()) {
            error = "the \"end\" stage (until_epoch=\"end\") must be the LAST entry in 'schedule'";
            return false;
        }
    }
    if (!std::isinf(out.stages.back().until_epoch)) {
        error = "the last schedule stage must use \"until_epoch\": \"end\" (the schedule must cover the "
               "whole run, not just a prefix)";
        return false;
    }

    // Every stage's weight keys must name a declared source, and every stage needs at least one positive
    // weight (an all-zero/negative stage would leave the scheduler with nothing to draw from).
    std::vector<bool> referenced(out.sources.size(), false);
    for (std::size_t si = 0; si < out.stages.size(); ++si) {
        bool any_positive = false;
        for (const auto& [name, rate] : out.stages[si].weights) {
            bool found = false;
            for (std::size_t i = 0; i < out.sources.size(); ++i) {
                if (out.sources[i].name == name) { found = true; if (rate > 0.0) { any_positive = true; referenced[i] = true; } break; }
            }
            if (!found) {
                error = std::format("schedule stage {} references undeclared source '{}'", si, name);
                return false;
            }
        }
        if (!any_positive) {
            error = std::format("schedule stage {} has no source with a positive weight -- nothing to "
                                "draw from", si);
            return false;
        }
    }
    for (std::size_t i = 0; i < out.sources.size(); ++i) {
        if (!referenced[i])
            warnings.push_back(std::format("source '{}' is declared but never given a positive weight in "
                                           "any schedule stage -- it will never be drawn from",
                                           out.sources[i].name));
    }

    if (!content_embed_str.empty() && content_embed_str != "off") {
        if (content_embed_str == "meanpool")      out.content_embed = ContentEmbedKind::MeanPool;
        else if (content_embed_str == "hash")     out.content_embed = ContentEmbedKind::Hash;
        else if (content_embed_str == "hrr")      out.content_embed = ContentEmbedKind::HRR;
        else {
            error = std::format("'content_embed' must be \"off\", \"meanpool\", \"hash\", or \"hrr\" "
                                "(got \"{}\")", content_embed_str);
            return false;
        }
    }
    return true;
}

}  // namespace sub0
