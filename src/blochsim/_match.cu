// Exhaustive dictionary matching in one pass on the card.
//
// Every signal is scored against every atom, |<s, d>| for normalized real
// atoms, and only the best atom and its score are kept: the voxels-by-atoms
// score matrix a matching in tensors writes, reads back and reduces is never
// stored. A thread holds VOXELS signals in registers; a block stages a tile of
// atoms in shared memory, where every thread of the block reads the same atom
// at once (a broadcast). So the work is the C multiply-adds per pair and
// little else, and a whole volume is matched in a launch per chunk instead of
// a launch per group and chunk.
//
// The signals are real or complex (interleaved); a complex signal scores
// hypot(<Re s, d>, <Im s, d>), as correlate() in estimators/_grouped.py does.
#include "_match.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int THREADS = 256;
constexpr int VOXELS = 4;  // signals per thread
constexpr int TILE = 256;  // atoms staged per pass

template <int C, bool COMPLEX>
__global__ void __launch_bounds__(THREADS) best_atom(const float* __restrict__ signals,
                                                     const float* __restrict__ atoms, long long voxels,
                                                     long long count, long long* __restrict__ index,
                                                     float* __restrict__ score) {
    constexpr int PARTS = COMPLEX ? 2 : 1;
    __shared__ float tile[TILE * C];

    const long long first = (static_cast<long long>(blockIdx.x) * THREADS * VOXELS) + threadIdx.x;
    float s[VOXELS][PARTS][C];
    float best[VOXELS];
    long long where[VOXELS];
#pragma unroll
    for (int v = 0; v < VOXELS; ++v) {
        const long long n = first + static_cast<long long>(v) * THREADS;
        const bool inside = n < voxels;
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int p = 0; p < PARTS; ++p) {
                s[v][p][c] = inside ? signals[(n * C + c) * PARTS + p] : 0.0f;
            }
        }
        best[v] = -1.0f;
        where[v] = 0;
    }

    for (long long start = 0; start < count; start += TILE) {
        const int here = static_cast<int>(count - start < TILE ? count - start : TILE);
        __syncthreads();
        for (int k = threadIdx.x; k < here * C; k += THREADS) {
            tile[k] = atoms[start * C + k];
        }
        __syncthreads();
        for (int a = 0; a < here; ++a) {
            const float* d = tile + a * C;
#pragma unroll
            for (int v = 0; v < VOXELS; ++v) {
                float value;
                if (COMPLEX) {
                    float re = 0.0f, im = 0.0f;
#pragma unroll
                    for (int c = 0; c < C; ++c) {
                        re = fmaf(s[v][0][c], d[c], re);
                        im = fmaf(s[v][PARTS - 1][c], d[c], im);
                    }
                    value = hypotf(re, im);
                } else {
                    float dot = 0.0f;
#pragma unroll
                    for (int c = 0; c < C; ++c) {
                        dot = fmaf(s[v][0][c], d[c], dot);
                    }
                    value = fabsf(dot);
                }
                // Strictly greater: of equal scores the first atom stays.
                if (value > best[v]) {
                    best[v] = value;
                    where[v] = start + a;
                }
            }
        }
    }

#pragma unroll
    for (int v = 0; v < VOXELS; ++v) {
        const long long n = first + static_cast<long long>(v) * THREADS;
        if (n < voxels) {
            index[n] = where[v];
            score[n] = best[v];
        }
    }
}

template <int C>
cudaError_t launch_for(const float* signals, const float* atoms, long long voxels, long long count, bool is_complex,
                       long long* index, float* score, cudaStream_t stream) {
    const long long per_block = static_cast<long long>(THREADS) * VOXELS;
    const long long blocks = (voxels + per_block - 1) / per_block;
    if (blocks > 2147483647LL) {
        return cudaErrorInvalidValue;
    }
    if (is_complex) {
        best_atom<C, true><<<static_cast<unsigned>(blocks), THREADS, 0, stream>>>(signals, atoms, voxels, count,
                                                                                  index, score);
    } else {
        best_atom<C, false><<<static_cast<unsigned>(blocks), THREADS, 0, stream>>>(signals, atoms, voxels, count,
                                                                                   index, score);
    }
    return cudaGetLastError();
}

}  // namespace

namespace blochsim_match {

int launch(const float* signals, const float* atoms, long long voxels, long long count, int contrasts,
           bool is_complex, long long* index, float* score, void* stream) {
    if (voxels <= 0) {
        return cudaSuccess;
    }
    const cudaStream_t queue = reinterpret_cast<cudaStream_t>(stream);
    switch (contrasts) {
#define BLOCHSIM_MATCH_CASE(C) \
    case C:                    \
        return launch_for<C>(signals, atoms, voxels, count, is_complex, index, score, queue);
        BLOCHSIM_MATCH_CASE(1)
        BLOCHSIM_MATCH_CASE(2)
        BLOCHSIM_MATCH_CASE(3)
        BLOCHSIM_MATCH_CASE(4)
        BLOCHSIM_MATCH_CASE(5)
        BLOCHSIM_MATCH_CASE(6)
        BLOCHSIM_MATCH_CASE(7)
        BLOCHSIM_MATCH_CASE(8)
        BLOCHSIM_MATCH_CASE(9)
        BLOCHSIM_MATCH_CASE(10)
        BLOCHSIM_MATCH_CASE(11)
        BLOCHSIM_MATCH_CASE(12)
        BLOCHSIM_MATCH_CASE(13)
        BLOCHSIM_MATCH_CASE(14)
        BLOCHSIM_MATCH_CASE(15)
        BLOCHSIM_MATCH_CASE(16)
#undef BLOCHSIM_MATCH_CASE
        default:
            return -1;  // more contrasts than the kernel is built for: the caller matches in tensors
    }
}

}  // namespace blochsim_match
