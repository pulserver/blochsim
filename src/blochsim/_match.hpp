// Exhaustive best-atom matching on the card (_match.cu).
#pragma once

namespace blochsim_match {

// The largest contrast count the kernel is built for.
constexpr int MAX_CONTRASTS = 16;

// For each of ``voxels`` normalized signals ``(voxels, contrasts)`` (float, or
// interleaved complex when ``is_complex``), the index of the best of ``count``
// normalized real atoms ``(count, contrasts)`` by |<s, d>|, and that score.
// Queued on ``stream``. Returns a cudaError_t, or -1 when ``contrasts`` is
// above MAX_CONTRASTS.
int launch(const float* signals, const float* atoms, long long voxels, long long count, int contrasts,
           bool is_complex, long long* index, float* score, void* stream);

}  // namespace blochsim_match
