// sub0llm/http/server.hpp — minimal synchronous Beast HTTP server.
//
// Drop-in replacement for the cpp-httplib server interface.  Provides the same
// Request / Response / Handler API so existing route handlers need only their
// include and type-name changed.
//
// Thread model: one detached thread per accepted connection (same as the Ch32
// viz server).  The `threads` parameter of `listen()` is accepted for call-site
// compatibility but is not used to cap concurrency; the inference-engine mutex
// inside InferenceEngine naturally serialises generation.
//
// Usage:
//   sub0llm::http::Server svr;
//   svr.set_pre_routing_handler([](const auto&, auto& res) {
//       res.set_header("Access-Control-Allow-Origin", "*");
//   });
//   svr.Get("/health",  handler);
//   svr.Post("/v1/completions", handler);
//   svr.listen("0.0.0.0", 8080);   // blocks

#pragma once

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace sub0llm::http {

// ── Request / Response ────────────────────────────────────────────────────────

struct Request {
    std::string method;   // "GET", "POST", "OPTIONS", …
    std::string path;     // URL path, query string stripped
    std::string body;     // raw request body
};

struct Response {
    int         status       = 200;
    std::string content_type = "application/json";
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;

    void set_content(std::string b, std::string_view ct) {
        body         = std::move(b);
        content_type = std::string{ct};
    }
    void set_header(std::string_view name, std::string_view value) {
        headers.emplace_back(std::string{name}, std::string{value});
    }
};

// ── Server ────────────────────────────────────────────────────────────────────

class Server {
public:
    using Handler    = std::function<void(const Request&, Response&)>;
    using PreHandler = std::function<void(const Request&, Response&)>;

    void Get (std::string path, Handler h) { routes_.push_back({"GET",  std::move(path), std::move(h)}); }
    void Post(std::string path, Handler h) { routes_.push_back({"POST", std::move(path), std::move(h)}); }

    // Called before route dispatch on every request.  Use it to add CORS
    // headers or other cross-cutting concerns.  Void return — always continues.
    void set_pre_routing_handler(PreHandler h) { pre_handler_ = std::move(h); }

    // Blocking accept loop.  Returns false if the acceptor cannot bind.
    // `threads` is accepted for call-site compatibility; concurrency is
    // naturally bounded by the inference-engine mutex in practice.
    [[nodiscard]] bool listen(const std::string& host, int port, int /*threads*/ = 4) {
        namespace net   = boost::asio;
        using tcp       = net::ip::tcp;

        try {
            net::io_context ioc{1};
            tcp::acceptor   acc{ioc,
                {net::ip::make_address(host), static_cast<unsigned short>(port)}};
            acc.set_option(net::socket_base::reuse_address(true));

            for (;;) {
                boost::beast::error_code ec;
                tcp::socket sock{ioc};
                acc.accept(sock, ec);
                if (ec) break;

                std::thread([this, sock = std::move(sock)]() mutable {
                    handle_session(std::move(sock));
                }).detach();
            }
        } catch (...) {
            return false;
        }
        return true;
    }

private:
    struct Route { std::string method; std::string path; Handler handler; };

    std::vector<Route> routes_;
    PreHandler         pre_handler_;

    void handle_session(boost::asio::ip::tcp::socket sock) {
        namespace beast = boost::beast;
        namespace bhttp = beast::http;

        beast::flat_buffer buf;
        beast::error_code  ec;

        for (;;) {
            bhttp::request<bhttp::string_body> req;
            bhttp::read(sock, buf, req, ec);
            if (ec) break;

            const auto version    = req.version();
            const bool keep_alive = req.keep_alive();

            Request r;
            r.method = std::string{req.method_string()};
            r.path   = std::string{req.target()};
            if (auto q = r.path.find('?'); q != std::string::npos) r.path.resize(q);
            r.body = req.body();

            Response res;

            // Pre-routing (e.g. CORS headers) runs on every request.
            if (pre_handler_) pre_handler_(r, res);

            // CORS preflight: respond immediately with the headers set above.
            if (r.method == "OPTIONS") {
                write_response(sock, version, keep_alive, res, ec);
                if (ec || !keep_alive) break;
                continue;
            }

            // Route dispatch
            bool matched = false;
            for (const auto& route : routes_) {
                if (route.method == r.method && route.path == r.path) {
                    route.handler(r, res);
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                res.status = 404;
                res.set_content(R"({"error":{"code":404,"message":"not found","type":"invalid_request_error"}})",
                                "application/json");
            }

            write_response(sock, version, keep_alive, res, ec);
            if (ec || !keep_alive) break;
        }

        boost::beast::error_code shut_ec;
        sock.shutdown(boost::asio::ip::tcp::socket::shutdown_send, shut_ec);
    }

    static void write_response(boost::asio::ip::tcp::socket& sock,
                                unsigned version, bool keep_alive,
                                const Response& res,
                                boost::beast::error_code& ec)
    {
        namespace bhttp = boost::beast::http;

        bhttp::response<bhttp::string_body> bres{
            static_cast<bhttp::status>(res.status), version};
        bres.set(bhttp::field::server, "sub0llm/1.0");
        bres.set(bhttp::field::content_type, res.content_type);
        for (const auto& [name, val] : res.headers)
            bres.insert(name, val);
        bres.keep_alive(keep_alive);
        bres.body() = res.body;
        bres.prepare_payload();
        bhttp::write(sock, bres, ec);
    }
};

} // namespace sub0llm::http
