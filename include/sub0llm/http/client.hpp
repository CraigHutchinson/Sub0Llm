// sub0llm/http/client.hpp — lightweight synchronous HTTP client backed by Boost.Beast + ASIO.
//
// Drop-in replacement for the cpp-httplib client interface used in ch24 (Ollama synthetic data)
// and any future chapter that needs a local HTTP POST.  Only the subset of the httplib API that
// is actually used is implemented; the interface is intentionally minimal.
//
// Usage:
//   sub0llm::http::Client cli("127.0.0.1", 11434);
//   cli.set_connection_timeout(3);
//   cli.set_read_timeout(3);
//   auto res = cli.Post("/api/generate", body, "application/json");
//   if (!res || res.status != 200) { /* error */ }
//   std::string text = res.body;

#pragma once

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <chrono>
#include <string>
#include <string_view>

namespace sub0llm::http {

// ── Result ────────────────────────────────────────────────────────────────────

struct Result {
    int         status = 0;   // HTTP status code, or 0 on connection/protocol failure
    std::string body;

    // False when the request failed entirely (status == 0).
    explicit operator bool() const noexcept { return status > 0; }
};

// ── Client ────────────────────────────────────────────────────────────────────

class Client {
public:
    Client(std::string_view host, int port)
        : host_{host}, port_{std::to_string(port)} {}

    void set_connection_timeout(int seconds) noexcept { connect_timeout_s_ = seconds; }
    void set_read_timeout(int seconds) noexcept       { read_timeout_s_    = seconds; }

    // Synchronous HTTP POST.  Returns Result{0,""} on any network/protocol error.
    [[nodiscard]] Result Post(std::string_view path,
                              std::string_view body,
                              std::string_view content_type)
    {
        namespace beast = boost::beast;
        namespace bhttp = beast::http;
        namespace net   = boost::asio;
        using tcp       = net::ip::tcp;

        try {
            net::io_context ioc;
            tcp::resolver   resolver{ioc};
            beast::tcp_stream stream{ioc};

            auto const endpoints = resolver.resolve(host_, port_);

            stream.expires_after(std::chrono::seconds(connect_timeout_s_));
            stream.connect(endpoints);

            bhttp::request<bhttp::string_body> req{bhttp::verb::post, path, 11};
            req.set(bhttp::field::host,         host_);
            req.set(bhttp::field::user_agent,   "sub0llm/1.0");
            req.set(bhttp::field::content_type, content_type);
            req.body() = std::string{body};
            req.prepare_payload();

            stream.expires_after(std::chrono::seconds(read_timeout_s_));
            bhttp::write(stream, req);

            beast::flat_buffer buf;
            bhttp::response<bhttp::string_body> res;
            bhttp::read(stream, buf, res);

            beast::error_code ec;
            stream.socket().shutdown(tcp::socket::shutdown_both, ec);

            return {static_cast<int>(res.result_int()), std::move(res.body())};
        } catch (...) {
            return {0, {}};
        }
    }

private:
    std::string host_;
    std::string port_;
    int         connect_timeout_s_ = 30;
    int         read_timeout_s_    = 30;
};

} // namespace sub0llm::http
