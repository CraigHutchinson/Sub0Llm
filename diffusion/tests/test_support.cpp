// test_support.cpp — the reusable support units extracted from ch29 in cleanup 3a:
// the honest-resume sidecar (train_state) and the token-stream cache (token_cache).

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

#include "sub0diff/data/token_cache.hpp"
#include "sub0diff/train/train_state.hpp"

namespace ts = sub0diff::train::trainstate;
namespace tc = sub0diff::data::tokcache;
namespace fs = std::filesystem;

namespace {
fs::path tmp(const char* name) { return fs::temp_directory_path() / name; }
}

TEST_CASE("train_state round-trips and is gated on the matching step", "[support][trainstate]") {
    const fs::path p = tmp("ch29_ts_test.txt");
    ts::State s;
    s.step = 12345;
    s.curr_k = 14;
    s.curr_best = 2.0571f;
    s.curr_stalls = 1;
    s.curr_converged = false;
    s.best_nelbo = 3.402823e38f;
    s.evals_since_best = 7;
    ts::save(p, s);

    SECTION("matching step → have=true, fields restored") {
        const ts::State r = ts::load(p, 12345);
        REQUIRE(r.have);
        CHECK(r.step == 12345);
        CHECK(r.curr_k == 14);
        CHECK(r.curr_best == 2.0571f);
        CHECK(r.curr_stalls == 1);
        CHECK(r.curr_converged == false);
        CHECK(r.evals_since_best == 7);
    }
    SECTION("step mismatch → have=false (stale sidecar ignored)") {
        CHECK_FALSE(ts::load(p, 99999).have);
    }
    SECTION("absent file → have=false") {
        CHECK_FALSE(ts::load(tmp("ch29_ts_absent.txt"), 12345).have);
    }
    SECTION("malformed value → have=false") {
        std::ofstream(p) << "step=not_a_number\ncurriculum_k=3\n";
        CHECK_FALSE(ts::load(p, 12345).have);
    }
    fs::remove(p);
}

TEST_CASE("token_cache round-trips and validates its fingerprint", "[support][tokcache]") {
    const fs::path p = tmp("ch29_tok_test.bin");
    const std::vector<std::int32_t> train{1, 2, 3, 4, 5, 6, 7};
    const std::vector<std::int32_t> eval{8, 9, 10};
    const std::uint64_t corpus_size = 555, vocab = 512;
    const std::int64_t  paras = 100;
    tc::save(p, corpus_size, paras, vocab, train, eval);

    SECTION("matching fingerprint → streams restored exactly") {
        std::vector<std::int32_t> rt, re;
        REQUIRE(tc::load(p, corpus_size, paras, vocab, rt, re));
        CHECK(rt == train);
        CHECK(re == eval);
    }
    SECTION("any fingerprint mismatch → load fails (forces re-encode)") {
        std::vector<std::int32_t> rt, re;
        CHECK_FALSE(tc::load(p, corpus_size + 1, paras, vocab, rt, re));   // corpus changed
        CHECK_FALSE(tc::load(p, corpus_size, paras + 1, vocab, rt, re));   // paragraph cap changed
        CHECK_FALSE(tc::load(p, corpus_size, paras, vocab + 1, rt, re));   // vocab changed
    }
    SECTION("absent file → load fails") {
        std::vector<std::int32_t> rt, re;
        CHECK_FALSE(tc::load(tmp("ch29_tok_absent.bin"), corpus_size, paras, vocab, rt, re));
    }
    fs::remove(p);
}
