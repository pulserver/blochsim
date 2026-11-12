// EXPERIMENT ONLY: the real kernel and its JVP instantiated over a grid of
// (Y, warps per block, minimum blocks per SM), picked at run time from
// BLOCHSIM_REAL_SWEEP="Y,W,OCC" (forward) and BLOCHSIM_REAL_JVP_SWEEP (dual),
// where OCC is the resident-warp target per SM (0 = no bound).
#include <cstdio>
#include <cstdlib>
#include "_layout.hpp"

namespace blochsim_layout {
namespace {

template <class T, bool MAPS, int Y, bool ONE_TRAIN, int W, int MINB>
__global__ void __launch_bounds__(32 * W, MINB) sweep_kernel(layout_real::Params p) {
    layout_real::real_loop<T, MAPS, Y, ONE_TRAIN>(p);
}

template <class T, int Y, int W, int MINB>
int go(const layout_real::Params& p, cudaStream_t stream) {
    const int groups = 32 / p.width;
    const long long problems = static_cast<long long>(p.train_count) * p.atom_count;
    const long long per_block = static_cast<long long>(W) * groups * Y;
    const unsigned grid = static_cast<unsigned>((problems + per_block - 1) / per_block);
    if (!p.single_train || p.transmit || p.density || p.inverting) return 1;
    sweep_kernel<T, false, Y, true, W, MINB><<<grid, 32 * W, 0, stream>>>(p);
    return static_cast<int>(cudaGetLastError());
}

#define OCCS(T, Y, W)                                                     \
    if (y == Y && w == W && occ == 0) return go<T, Y, W, 1>(p, stream);   \
    if (y == Y && w == W && occ == 16) return go<T, Y, W, 16 / W>(p, stream); \
    if (y == Y && w == W && occ == 24) return go<T, Y, W, 24 / W>(p, stream); \
    if (y == Y && w == W && occ == 32) return go<T, Y, W, 32 / W>(p, stream);
#define WS(T, Y) OCCS(T, Y, 2) OCCS(T, Y, 4) OCCS(T, Y, 8)

template <class T>
int dispatch(const char* env, int y_default, bool capped, const layout_real::Params& p, cudaStream_t stream) {
    int y = y_default, w = WARPS, occ = capped ? 16 : 0;
    if (const char* s = std::getenv(env)) std::sscanf(s, "%d,%d,%d", &y, &w, &occ);
    if constexpr (num::is_dual<T>::value) {
        WS(T, 1) WS(T, 2) WS(T, 4)
    } else {
        WS(T, 2) WS(T, 4) WS(T, 8)
    }
    std::fprintf(stderr, "blochsim sweep: no variant %d,%d,%d\n", y, w, occ);
    return 1;
}

}  // namespace

int real_forward(const layout_real::Params& p, cudaStream_t stream) {
    return dispatch<float>("BLOCHSIM_REAL_SWEEP", 8, false, p, stream);
}

int real_jvp(const layout_real::Params& p, cudaStream_t stream) {
    return dispatch<num::Dual>("BLOCHSIM_REAL_JVP_SWEEP", 4, true, p, stream);
}

}  // namespace blochsim_layout
