// The real kernel and its Jacobian-vector product in their layouts.
#include "_layout.hpp"

namespace blochsim_layout {
namespace {

template <class T, bool MAPS, int Y, bool ONE_TRAIN>
__global__ void __launch_bounds__(32 * WARPS) real_kernel(layout_real::Params p) {
    layout_real::real_loop<T, MAPS, Y, ONE_TRAIN>(p);
}

// A dual number doubles what a thread holds; capped at a quarter of the
// register file the eight programs an SM then holds hide each other's
// latency better than four larger ones do.
template <class T, bool MAPS, int Y, bool ONE_TRAIN>
__global__ void __launch_bounds__(32 * WARPS, 8) real_kernel_capped(layout_real::Params p) {
    layout_real::real_loop<T, MAPS, Y, ONE_TRAIN>(p);
}

template <class T, int Y, bool CAPPED>
int launch_real(const layout_real::Params& p, cudaStream_t stream) {
    const int groups = 32 / p.width;
    const long long problems = static_cast<long long>(p.train_count) * p.atom_count;
    const long long per_block = static_cast<long long>(WARPS) * groups * Y;
    const unsigned grid = static_cast<unsigned>((problems + per_block - 1) / per_block);
    const bool maps = p.transmit || p.density || p.inverting;
#define BLOCHSIM_GO(MP, ONE)                                                                 \
    if constexpr (CAPPED) real_kernel_capped<T, MP, Y, ONE><<<grid, 32 * WARPS, 0, stream>>>(p); \
    else real_kernel<T, MP, Y, ONE><<<grid, 32 * WARPS, 0, stream>>>(p)
    if (p.single_train) {
        if (maps) { BLOCHSIM_GO(true, true); } else { BLOCHSIM_GO(false, true); }
    } else {
        if (maps) { BLOCHSIM_GO(true, false); } else { BLOCHSIM_GO(false, false); }
    }
#undef BLOCHSIM_GO
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

int real_forward(const layout_real::Params& p, cudaStream_t stream) { return launch_real<float, 8, false>(p, stream); }

int real_jvp(const layout_real::Params& p, cudaStream_t stream) { return launch_real<num::Dual, 4, true>(p, stream); }

}  // namespace blochsim_layout
