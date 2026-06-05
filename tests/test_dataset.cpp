#include <catch2/catch_test_macros.hpp>

#include "sub0llm/data/dataloader.hpp"
#include "sub0llm/data/dataset.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>

using namespace sub0llm;
using TokenId = TextDataset::TokenId;

// Helper: consecutive ids 0,1,2,...,n-1
static std::vector<TokenId> iota_tokens(std::size_t n) {
    std::vector<TokenId> v(n);
    std::iota(v.begin(), v.end(), TokenId{0});
    return v;
}

// ── TextDataset construction ──────────────────────────────────────────────────

TEST_CASE("TextDataset - empty when tokens too short", "[dataset]") {
    // Need at least context_length+1 tokens for one sample.
    TextDataset ds(iota_tokens(4), 5, 1);
    REQUIRE(ds.empty());
    REQUIRE(ds.size() == 0);
}

TEST_CASE("TextDataset - single sample", "[dataset]") {
    // Exactly context_length+1 tokens → exactly 1 sample.
    TextDataset ds(iota_tokens(5), 4, 1);
    REQUIRE(ds.size() == 1);
}

TEST_CASE("TextDataset - stride=1 sample count", "[dataset]") {
    // N tokens, context L, stride 1: (N-L-1)/1 + 1 = N-L samples
    constexpr std::size_t N = 20, L = 4;
    TextDataset ds(iota_tokens(N), L, 1);
    REQUIRE(ds.size() == N - L);
}

TEST_CASE("TextDataset - stride=context_length gives non-overlapping windows", "[dataset]") {
    // 20 tokens, L=4, stride=4: window size = L+1 = 5
    // samples = (20-5)/4 + 1 = 4  (a 5th window would need token[20] which doesn't exist)
    constexpr std::size_t N = 20, L = 4;
    TextDataset ds(iota_tokens(N), L, L);
    REQUIRE(ds.size() == (N - (L + 1)) / L + 1);
}

TEST_CASE("TextDataset - context_length=0 throws", "[dataset]") {
    REQUIRE_THROWS_AS(TextDataset(iota_tokens(10), 0, 1), std::runtime_error);
}

// ── TextDataset sample content ────────────────────────────────────────────────

TEST_CASE("TextDataset - input is target shifted left by 1", "[dataset]") {
    // Tokens: 0,1,2,3,4,5,6,7,8,9  L=4 stride=1
    TextDataset ds(iota_tokens(10), 4, 1);
    const auto s = ds[0];
    // input  = [0,1,2,3], target = [1,2,3,4]
    REQUIRE(s.input_ids  == std::vector<TokenId>{0, 1, 2, 3});
    REQUIRE(s.target_ids == std::vector<TokenId>{1, 2, 3, 4});
}

TEST_CASE("TextDataset - second sample with stride=1", "[dataset]") {
    TextDataset ds(iota_tokens(10), 4, 1);
    const auto s = ds[1];
    REQUIRE(s.input_ids  == std::vector<TokenId>{1, 2, 3, 4});
    REQUIRE(s.target_ids == std::vector<TokenId>{2, 3, 4, 5});
}

TEST_CASE("TextDataset - stride=2 sample offset", "[dataset]") {
    TextDataset ds(iota_tokens(10), 3, 2);
    // Sample 0: offset 0 → input=[0,1,2] target=[1,2,3]
    // Sample 1: offset 2 → input=[2,3,4] target=[3,4,5]
    const auto s0 = ds[0];
    const auto s1 = ds[1];
    REQUIRE(s0.input_ids == std::vector<TokenId>{0, 1, 2});
    REQUIRE(s1.input_ids == std::vector<TokenId>{2, 3, 4});
}

TEST_CASE("TextDataset - stride=0 is clamped to 1", "[dataset]") {
    // stride=0 would infinite-loop; constructor clamps it to 1.
    TextDataset ds0(iota_tokens(10), 4, 0);
    TextDataset ds1(iota_tokens(10), 4, 1);
    REQUIRE(ds0.size() == ds1.size());
    REQUIRE(ds0[0].input_ids == ds1[0].input_ids);
}

TEST_CASE("TextDataset - out-of-range throws", "[dataset]") {
    TextDataset ds(iota_tokens(6), 3, 1);
    REQUIRE_THROWS_AS(ds[ds.size()], std::out_of_range);
}

TEST_CASE("TextDataset - input_span and target_span match operator[]", "[dataset]") {
    TextDataset ds(iota_tokens(12), 4, 2);
    for (std::size_t i = 0; i < ds.size(); ++i) {
        const auto s   = ds[i];
        const auto ins = ds.input_span(i);
        const auto tgs = ds.target_span(i);
        REQUIRE(std::vector<TokenId>(ins.begin(), ins.end())  == s.input_ids);
        REQUIRE(std::vector<TokenId>(tgs.begin(), tgs.end()) == s.target_ids);
    }
}

// ── DataLoader construction ───────────────────────────────────────────────────

TEST_CASE("DataLoader - batch_size=0 throws", "[dataloader]") {
    TextDataset ds(iota_tokens(10), 4, 1);
    REQUIRE_THROWS_AS(DataLoader(ds, 0), std::invalid_argument);
}

TEST_CASE("DataLoader - num_batches = floor(dataset_size / batch_size)", "[dataloader]") {
    TextDataset ds(iota_tokens(20), 4, 1);  // 16 samples
    DataLoader loader(ds, 4);
    REQUIRE(loader.num_batches() == 4);
}

TEST_CASE("DataLoader - last incomplete batch is dropped", "[dataloader]") {
    TextDataset ds(iota_tokens(20), 4, 1);  // 16 samples
    DataLoader loader(ds, 5);               // 16/5 = 3 full batches
    REQUIRE(loader.num_batches() == 3);
}

// ── DataLoader iteration ──────────────────────────────────────────────────────

TEST_CASE("DataLoader - iterates all full batches", "[dataloader]") {
    TextDataset ds(iota_tokens(20), 4, 1);  // 16 samples
    DataLoader loader(ds, 4, false);
    loader.reset();

    std::size_t count = 0;
    while (auto b = loader.next()) {
        REQUIRE(b->batch_size     == 4);
        REQUIRE(b->context_length == 4);
        REQUIRE(b->input_ids.size()  == 16u);
        REQUIRE(b->target_ids.size() == 16u);
        ++count;
    }
    REQUIRE(count == loader.num_batches());
}

TEST_CASE("DataLoader - returns nullopt after epoch exhausted", "[dataloader]") {
    TextDataset ds(iota_tokens(10), 3, 1);
    DataLoader loader(ds, 3);
    loader.reset();
    while (loader.next()) {}
    REQUIRE(loader.next() == std::nullopt);
}

TEST_CASE("DataLoader - reset restarts iteration", "[dataloader]") {
    TextDataset ds(iota_tokens(10), 3, 1);
    DataLoader loader(ds, 2, false);
    loader.reset();
    const auto b1 = loader.next();

    loader.reset();
    const auto b2 = loader.next();

    REQUIRE(b1.has_value());
    REQUIRE(b2.has_value());
    // Without shuffle the two first batches are identical.
    REQUIRE(b1->input_ids == b2->input_ids);
}

TEST_CASE("DataLoader - shuffle produces different order across resets", "[dataloader]") {
    // Use enough samples that two resets are very unlikely to produce the same order.
    TextDataset ds(iota_tokens(100), 4, 1);  // 96 samples
    DataLoader loader(ds, 8, /*shuffle=*/true, /*seed=*/123);

    loader.reset();
    const auto b_epoch1 = loader.next();
    loader.reset();
    const auto b_epoch2 = loader.next();

    REQUIRE(b_epoch1.has_value());
    REQUIRE(b_epoch2.has_value());
    // Statistically near-certain to differ with 96 samples.
    REQUIRE(b_epoch1->input_ids != b_epoch2->input_ids);
}

TEST_CASE("DataLoader - batch covers correct token range without shuffle", "[dataloader]") {
    // tokens: 0..14, L=3, stride=1, no shuffle → sample 0 = [0,1,2] → [1,2,3]
    TextDataset ds(iota_tokens(15), 3, 1);
    DataLoader loader(ds, 1, false);
    loader.reset();

    const auto b = loader.next();
    REQUIRE(b.has_value());
    REQUIRE(b->input_ids  == std::vector<TokenId>{0, 1, 2});
    REQUIRE(b->target_ids == std::vector<TokenId>{1, 2, 3});
}

// ── MappedFile ────────────────────────────────────────────────────────────────

TEST_CASE("MappedFile - throws on missing file", "[dataset][io]") {
    REQUIRE_THROWS_AS(MappedFile("/tmp/sub0llm_does_not_exist_xyz.txt"),
                      std::runtime_error);
}

TEST_CASE("MappedFile - reads file contents correctly", "[dataset][io]") {
    const std::filesystem::path p = "/tmp/sub0llm_mmap_test.txt";
    {
        std::ofstream f(p);
        f << "hello mmap";
    }
    MappedFile mf(p);
    REQUIRE(mf.data() == "hello mmap");
    REQUIRE(mf.size() == 10u);
    std::filesystem::remove(p);
}

TEST_CASE("MappedFile - empty file has size 0 and empty data view", "[dataset][io]") {
    const std::filesystem::path p = "/tmp/sub0llm_mmap_empty.txt";
    { std::ofstream f(p); }  // create empty file
    MappedFile mf(p);
    REQUIRE(mf.size() == 0u);
    REQUIRE(mf.data().empty());
    std::filesystem::remove(p);
}
