// tests/mock_device_backend.cpp -- a TEST-ONLY implementation of the device seam whose "device" is
// the CPU engine. Selected by defining SUB0_BUILD_MOCK_DEVICE (see include/sub0/device_backend.hpp,
// which checks it BEFORE SUB0_BUILD_CUDA); linked only into the sub0_eval_seam_tests target.
//
// WHY THIS EXISTS
//
// The device seam previously had exactly one implementation, and it needed a GPU. That meant the
// consumer-side logic around it -- window batching, ids/targets pairing, group recombination -- had no
// test at all: a CPU-only machine skipped the whole area, and a GPU machine could only test it
// end-to-end against real kernels, where a plumbing bug and a kernel bug look identical.
//
// This backend answers sub0_dev_forward_loss by running the SAME sub0::forward + sub0::cross_entropy
// the CPU path uses, over the ids and targets it is handed. It therefore computes the right answer if
// and only if it was handed the right question. Any of these silently-wrong-number bugs makes the
// device and CPU routes disagree here, on any machine, in milliseconds:
//
//   * targets shifted (ids[t] paired with targets[t] instead of targets[t] = ids[t+1]);
//   * a window's start offset computed differently than the CPU path computes it;
//   * a ragged final group averaged as a plain mean of means instead of weighted by window count;
//   * a group's rows read past the windows actually filled.
//
// It is NOT a substitute for the real CUDA parity gate (cuda_tests.cpp's "forward_loss matches the CPU
// evaluate"), which is what proves the kernels agree. The two cover different halves and neither
// subsumes the other.

#include "sub0/core.hpp"
#include "sub0/device_backend.hpp"

#include <cmath>
#include <vector>

#if !defined(SUB0_BUILD_MOCK_DEVICE)
#error "mock_device_backend.cpp must be compiled with SUB0_BUILD_MOCK_DEVICE defined"
#endif

namespace {

bool             g_up = false;
std::vector<int> g_batches;   // batch size of every forward_loss call served since the last reset

}  // namespace

extern "C" int sub0_mock_init() { g_up = true; return 0; }

extern "C" void sub0_mock_shutdown() { g_up = false; }

// The real backends copy the host parameter blob to device memory. Here the engine already holds the
// only copy, so this only has to verify the caller passed the params it claims to be scoring -- a
// consumer that forgot to upload must still fail, or the test would pass for the wrong reason.
extern "C" int sub0_mock_upload_params(const float* host) {
    if (!g_up || host == nullptr) return 1;
    return host == sub0::params_ptr() ? 0 : 1;
}

// Contract, mirroring sub0_cuda_forward_loss exactly: the mean cross-entropy over `batch` windows of
// `T` tokens, where window b occupies rows [b*T, (b+1)*T) of ids/targets. `lengths` (optional) gives
// each window's real length; positions at or past it are padding and are not graded. Targets < 0
// (LOSS_IGNORE_INDEX) are excluded from both the loss and the per-window normalizer.
//
// `out_win_loss` (optional, [batch]) returns each window's own mean before the 1/batch reduction -- the
// quantity this function already forms internally, which is exactly the point: the CUDA kernel likewise
// derives it from a value it already has, so the mock can pin the same `mean(win) == batch mean`
// identity without a GPU. A window with nothing gradeable reports 0.0, matching its contribution here.
extern "C" int sub0_mock_forward_loss(const int* ids, const int* targets, int batch, int T,
                                      double* out_loss, const int* lengths, double* out_win_loss) {
    if (!g_up || !ids || !targets || !out_loss) return 1;
    if (batch < 1 || T < 2 || T > SEQ_LEN) return 1;
    g_batches.push_back(batch);

    double total = 0.0;
    std::vector<int> win(static_cast<std::size_t>(T));
    if (out_win_loss)
        for (int b = 0; b < batch; ++b) out_win_loss[b] = 0.0;
    for (int b = 0; b < batch; ++b) {
        const int* id_row = ids     + static_cast<std::size_t>(b) * T;
        const int* tg_row = targets + static_cast<std::size_t>(b) * T;
        const int  len    = lengths ? lengths[b] : T;
        if (len < 2) continue;                       // nothing gradeable in this window
        for (int t = 0; t < len; ++t) win[static_cast<std::size_t>(t)] = id_row[t];

        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(win.data(), len);
        // Per-window mean over its ACTIVE positions -- the same normalizer the CUDA CE kernel applies,
        // computed here from the logits directly rather than through sub0::cross_entropy, because that
        // helper has no ignore-index-aware denominator to borrow.
        double  wsum = 0.0;
        int     act  = 0;
        for (int t = 0; t < len; ++t) {
            const int tgt = tg_row[t];
            if (tgt < 0) continue;                   // LOSS_IGNORE_INDEX
            const float* row = logits->data.data() + static_cast<std::size_t>(t) * VOCAB;
            float mx = -1e30f;
            for (int j = 0; j < VOCAB; ++j) mx = std::max(mx, row[j]);
            double Z = 0.0;
            for (int j = 0; j < VOCAB; ++j) Z += std::exp(static_cast<double>(row[j] - mx));
            wsum += -(static_cast<double>(row[tgt] - mx) - std::log(Z));
            ++act;
        }
        if (act > 0) {
            total += wsum / act;
            if (out_win_loss) out_win_loss[b] = wsum / act;
        }
    }
    sub0::graph_reset();
    *out_loss = total / batch;
    return 0;
}

extern "C" int  sub0_mock_call_count()      { return static_cast<int>(g_batches.size()); }
extern "C" int  sub0_mock_batch_at(int i)   {
    return (i >= 0 && i < static_cast<int>(g_batches.size())) ? g_batches[static_cast<std::size_t>(i)] : -1;
}
extern "C" void sub0_mock_reset_log()       { g_batches.clear(); }
