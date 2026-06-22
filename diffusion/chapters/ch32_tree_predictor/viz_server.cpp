// viz_server.cpp (Ch32 Viz, Phase C) — serve the diffusion "thought process" interactively.
//
// Trains a small MERA denoiser ONCE at startup, then runs an HTTP server so the web scrubber can drive
// generation live: POST a prompt + sampler/model knobs and get back a fresh GenerationTrace (the same
// JSON the offline ch32_viz_gen writes). The viewer's prompt box + param panel call this, and its A/B
// compare diffs two traces (same prompt+seed, different knobs) — the parameter-adjustment payoff.
//
// One process serves BOTH the static viewer (mounts the repo root, so /tools/viz/ and /ACRONYMS.json
// work) and the API — no separate Python static server needed. Generation is serialized behind a mutex
// (the model/autograd graph is not re-entrant); fine for an interactive single-user tool.
//
// Build: cmake --build build-cuda --target ch32_viz_server   (CUDA build trains fast at startup)
// Run:   ./build-cuda/diffusion/chapters/ch32_tree_predictor/ch32_viz_server.exe --port 8080
// Open:  http://localhost:8080/tools/viz/

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/mera_denoiser.hpp"
#include "sub0diff/nn/model_io.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/diffusion_loss.hpp"
#include "sub0diff/viz/trace.hpp"
#include "sub0diff/viz/trace_json.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/runtime.hpp"   // init_cpu_compute (FTZ+DAZ) — main + each httplib worker thread
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

#include <httplib.h>
#include <simdjson.h>   // direct on-demand (forward, single-pass) request-body parsing

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <format>
#include <functional>
#include <mutex>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using sub0llm::BPETokenizer;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
namespace dv = sub0diff::viz;

namespace {

std::vector<std::string> read_paragraphs(const std::string& path, std::int64_t limit) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::format("cannot open corpus: {}", path));
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        std::size_t b = 0;
        while (b < line.size() && std::isspace(static_cast<unsigned char>(line[b]))) ++b;
        if (b >= line.size()) continue;
        out.push_back(line.substr(b));
        if (limit > 0 && static_cast<std::int64_t>(out.size()) >= limit) break;
    }
    return out;
}

void train(dn::MeraDenoiser& model, std::span<const std::int32_t> stream, int steps, std::int64_t B,
           std::int64_t N, std::uint64_t seed, sub0llm::Device dev) {
    model.to(dev);
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 1e-3f);
    dt::BatchedDiffusionLossContext ctx(B, N);
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<std::size_t> off(0, stream.size() - static_cast<std::size_t>(N));
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        std::vector<std::size_t> offs(static_cast<std::size_t>(B));
        for (auto& o : offs) o = off(rng);
        for (auto* p : params)
            if (p->grad().numel() > 0)
                p->grad() = sub0llm::zeros(p->data().shape(), p->data().dtype(), p->data().device());
        auto res = dt::batched_diffusion_loss(model, stream, offs, rng, ctx, 0.02f, 1.0f);
        res.loss.backward();
        (void)sub0llm::nn::clip_grad_norm(params, 5.0f);
        opt.step();
        if ((s + 1) % 1000 == 0)
            std::println("  step {:>5}  nelbo={:.4f}  ({:.1f}s)", s + 1,
                         static_cast<double>(res.loss.data().to(sub0llm::Device::cpu()).template item<float>()),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    model.to(sub0llm::Device::cpu());
}

// Forward, single-pass request reader over simdjson on-demand: register a typed handler per expected
// key UP FRONT, then walk the top-level object ONCE in document order. No DOM / no random access —
// matches simdjson's streaming on-demand design. Unknown keys are skipped; absent keys keep defaults.
class JsonFields {
public:
    JsonFields& f64(std::string_view k, float& o) {
        return on(k, [&o](simdjson::ondemand::value v) { double d; if (!v.get(d)) o = static_cast<float>(d); });
    }
    JsonFields& i64(std::string_view k, std::int64_t& o) {
        return on(k, [&o](simdjson::ondemand::value v) { std::int64_t i; if (!v.get(i)) o = i; });
    }
    JsonFields& u64(std::string_view k, std::uint64_t& o) {
        return on(k, [&o](simdjson::ondemand::value v) { std::uint64_t u; if (!v.get(u)) o = u; });
    }
    JsonFields& str(std::string_view k, std::string& o) {
        return on(k, [&o](simdjson::ondemand::value v) { std::string_view s; if (!v.get(s)) o.assign(s); });
    }
    // Parse `body` and dispatch in one forward pass. false on a JSON / non-object error.
    [[nodiscard]] bool read(std::string_view body) {
        simdjson::ondemand::parser parser;
        simdjson::padded_string padded(body.data(), body.size());
        auto doc = parser.iterate(padded);
        if (doc.error()) return false;
        auto obj = doc.get_object();
        if (obj.error()) return false;
        for (auto field : obj) {
            auto key = field.unescaped_key();
            if (key.error()) continue;
            auto val = field.value();
            if (val.error()) continue;
            for (auto& [name, h] : handlers_)
                if (name == key.value_unsafe()) { h(std::move(val).value_unsafe()); break; }
        }
        return true;
    }

private:
    using Handler = std::function<void(simdjson::ondemand::value)>;
    JsonFields& on(std::string_view k, Handler h) { handlers_.emplace_back(k, std::move(h)); return *this; }
    std::vector<std::pair<std::string_view, Handler>> handlers_;
};

// Minimal JSON string escape for the small hand-built responses (/health, error). Trace serialization
// uses serialize_trace_json; request *reading* uses JsonFields above.
std::string jesc(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char ch : s) {
        switch (ch) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\t': o += "\\t";  break;
            case '\r': o += "\\r";  break;
            default:   o += ch;
        }
    }
    return o;
}

std::int64_t arg_i(int argc, char** argv, const std::string& k, std::int64_t d) {
    for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return std::stoll(argv[i + 1]);
    return d;
}
std::string arg_s(int argc, char** argv, const std::string& k, const std::string& d) {
    for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return argv[i + 1];
    return d;
}

// Level pyramid for a model: MERA exposes level_lens(); flat is a single level [N].
template <class Model>
std::vector<std::int64_t> levels_of(const Model& m, std::int64_t N) {
    if constexpr (requires { m.level_lens(N); }) return m.level_lens(N);
    else return {N};
}

// The HTTP server, templated on the model type so it serves a loaded/trained MERA or a flat Denoiser
// identically (both satisfy the refine_canvas sampler). `model` must be on CPU. Blocks in listen().
template <class Model>
int serve(Model& model, BPETokenizer& tok, std::int64_t Vr, const std::string& model_type,
          std::int64_t maxN, std::int64_t c, std::int64_t w,
          const std::string& host, int port, const std::string& webroot) {
    const std::int32_t mask_id = model.mask_id();
    auto decode = [&tok, Vr](std::int32_t id) {
        return (id >= 0 && id < Vr) ? std::string(tok.token_str(static_cast<BPETokenizer::TokenId>(id)))
                                    : std::string("?");
    };
    std::println("ready ({}). levels at max N={}: {}", model_type, maxN,
                 [&]{ std::string s; for (auto l : levels_of(model, maxN)) s += std::format("{} ", l); return s; }());

    std::mutex gen_mtx;  // serialize generation (model + autograd graph are not re-entrant)
    httplib::Server srv;

    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        std::string lv;
        for (auto l : levels_of(model, maxN)) { if (!lv.empty()) lv += ','; lv += std::format("{}", l); }
        res.set_content(std::format(
            R"({{"status":"ok","model":"{}","vocab":{},"max_seq_len":{},"coarsen":{},"window":{},"levels":[{}]}})",
            model_type, Vr, maxN, c, w, lv), "application/json");
    });

    // POST /v1/generate_trace {prompt, seq_len, temperature, conf_threshold, min_commit_frac,
    //   remask_threshold, entropy_bound, commit_order:"confidence"|"spread", seed} -> GenerationTrace JSON.
    srv.Post("/v1/generate_trace", [&](const httplib::Request& req, httplib::Response& res) {
        sub0llm::init_cpu_compute();   // FTZ+DAZ on THIS httplib worker thread (per-thread MXCSR state)
        try {
            // Defaults, then a single forward pass fills whatever keys the request actually sent.
            dn::SamplerConfig cfg;
            cfg.temperature = 0.9f;
            std::int64_t seq_len = maxN;
            std::uint64_t rseed = 7;
            std::string prompt_text, order = "spread";
            JsonFields fields;
            fields.f64("temperature", cfg.temperature)
                  .f64("conf_threshold", cfg.conf_threshold)
                  .f64("min_commit_frac", cfg.min_commit_frac)
                  .f64("remask_threshold", cfg.remask_threshold)
                  .f64("entropy_bound", cfg.entropy_bound)
                  .i64("seq_len", seq_len)
                  .u64("seed", rseed)
                  .str("prompt", prompt_text)
                  .str("commit_order", order);
            if (!fields.read(req.body.empty() ? "{}" : req.body))
                throw std::runtime_error("request body is not valid JSON");
            cfg.commit_order = (order == "spread") ? dn::CommitOrder::Spread : dn::CommitOrder::Confidence;

            // Snap N down to the nearest length the model accepts (MERA needs a valid pyramid; flat any
            // N), so a knob never 500s.
            auto valid = [&](std::int64_t n) {
                if (n <= 0) return false;
                if constexpr (requires { model.level_lens(n); }) {
                    try { (void)model.level_lens(n); return true; } catch (...) { return false; }
                } else return true;
            };
            std::int64_t N = std::min(maxN, seq_len);
            while (N > 0 && !valid(N)) --N;
            if (N <= 0) throw std::runtime_error("no valid seq_len <= requested");

            std::vector<std::int32_t> prompt;
            if (!prompt_text.empty()) {
                auto enc = tok.encode(prompt_text);
                for (auto t : enc) if (t >= 0 && t < Vr && static_cast<std::int64_t>(prompt.size()) < N) prompt.push_back(t);
            }

            dv::GenerationTrace trace;
            trace.T = N; trace.model = model_type; trace.N = N; trace.c = c; trace.w = w;
            trace.commit_order = (cfg.commit_order == dn::CommitOrder::Spread) ? "spread" : "confidence";
            trace.temperature = cfg.temperature; trace.conf_threshold = cfg.conf_threshold;
            trace.min_commit_frac = cfg.min_commit_frac; trace.remask_threshold = cfg.remask_threshold;
            trace.prompt = prompt_text;
            trace.levels = levels_of(model, N);

            std::string payload;
            {
                std::scoped_lock lk(gen_mtx);
                std::mt19937 rng(static_cast<std::uint32_t>(rseed * 131 + 5));
                auto canvas = dn::make_canvas(model, N, prompt);
                dn::refine_canvas(model, canvas, cfg, rng, {}, &trace);
                payload = dv::serialize_trace_json(trace, decode, mask_id);
            }
            res.set_content(payload, "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(std::format(R"({{"error":"{}"}})", jesc(e.what())), "application/json");
        }
    });

    // static viewer + glossary (mount repo root so /tools/viz/ and /ACRONYMS.json resolve)
    if (!srv.set_mount_point("/", webroot))
        std::println("WARN: could not mount static webroot '{}' (API still works)", webroot);
    srv.Get("/", [](const httplib::Request&, httplib::Response& res) { res.set_redirect("/tools/viz/"); });

    std::println("\nserving on http://{}:{}", host, port);
    std::println("  open:   http://{}:{}/tools/viz/", host, port);
    std::println("  api:    POST http://{}:{}/v1/generate_trace", host, port);
    std::println("  health: GET  http://{}:{}/health", host, port);
    if (!srv.listen(host, port)) { std::println("ERROR: failed to bind {}:{}", host, port); return 1; }
    return 0;
}

}  // namespace

int run_main(int argc, char** argv) {
    sub0llm::init_cpu_compute();   // FTZ+DAZ on the main thread
    const std::string model_dir = arg_s(argc, argv, "--model-dir", "");
    const std::string host    = arg_s(argc, argv, "--host", "127.0.0.1");
    const std::string webroot = arg_s(argc, argv, "--webroot", ".");
    const int port            = static_cast<int>(arg_i(argc, argv, "--port", 8080));
    std::println("== Ch32 Viz server ==");

    // ── A) load a trained model dir (the GPU trainer's output) — no retraining ──────────────────────
    if (!model_dir.empty()) {
        std::println("loading model dir {} ...", model_dir);
        dn::LoadedModel lm = dn::load_model_dir(model_dir);
        const std::int64_t Vr = static_cast<std::int64_t>(lm.tokenizer->vocab_size());
        std::println("loaded {} model (step {}, seq_len {}, vocab {})", lm.model_type, lm.step, lm.seq_len, Vr);
        if (lm.model_type == "mera")
            return serve(*lm.mera, *lm.tokenizer, Vr, "mera", lm.seq_len, lm.mera_coarsen, lm.mera_window, host, port, webroot);
        return serve(*lm.model, *lm.tokenizer, Vr, "flat", lm.seq_len, 0, 0, host, port, webroot);
    }

    // ── B) no model dir: train a small MERA inline (the original Phase-C convenience path) ──────────
    const std::string corpus  = arg_s(argc, argv, "--corpus", "data/tinystories_clean.txt");
    const std::int64_t plimit = arg_i(argc, argv, "--paragraphs", 600);
    const std::int64_t steps  = arg_i(argc, argv, "--steps", 2000);
    const std::int64_t D      = arg_i(argc, argv, "--embed_dim", 256);
    const std::int64_t maxN   = arg_i(argc, argv, "--seq_len", 128);   // also the trained max length
    const std::int64_t w      = arg_i(argc, argv, "--window", 64);
    const std::int64_t c      = arg_i(argc, argv, "--coarsen", 4);
    const std::int64_t Btr    = arg_i(argc, argv, "--batch", 8);
    const std::uint64_t seed  = static_cast<std::uint64_t>(arg_i(argc, argv, "--seed", 7));
    const std::string devs    = arg_s(argc, argv, "--device", "cuda");
    const sub0llm::Device train_dev = devs == "cpu" ? sub0llm::Device::cpu() : sub0llm::Device::cuda();

    std::println("(no --model-dir) training a small MERA inline...");
    auto paras = read_paragraphs(corpus, plimit);
    if (paras.size() < 20) throw std::runtime_error("need >=20 paragraphs");
    std::print("building word tokenizer over {} paragraphs... ", paras.size());
    BPETokenizer tok = BPETokenizer::word_level(paras);
    const std::int64_t Vr = static_cast<std::int64_t>(tok.vocab_size());
    std::println("done — {} word vocab", Vr);
    std::vector<std::int32_t> ids;
    for (const auto& p : paras) { auto v = tok.encode(p); ids.insert(ids.end(), v.begin(), v.end()); }
    if (static_cast<std::int64_t>(ids.size()) < maxN + Btr + 8) throw std::runtime_error("corpus too short");

    std::println("training MERA ({} steps, N={}, D={}, c={}, w={}, device={})...", steps, maxN, D, c, w, devs);
    dn::MeraDenoiser model(Vr, D, 8, 4, c, w, maxN, 0, seed);
    train(model, ids, static_cast<int>(steps), Btr, maxN, seed, train_dev);
    return serve(model, tok, Vr, "mera", maxN, c, w, host, port, webroot);
}

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        std::fflush(stdout);
        std::println(stderr, "ERROR: {}", e.what());
        return 1;
    }
}
