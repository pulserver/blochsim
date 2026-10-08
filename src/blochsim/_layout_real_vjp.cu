// The real adjoint and its derivative along a direction in their layouts.
#include "_layout.hpp"

namespace blochsim_layout {
namespace {

constexpr int SEGMENT = 4;

template <class T, bool MAPS, int Y, bool ONE_TRAIN>
__global__ void __launch_bounds__(32 * WARPS) real_adjoint_kernel(layout_real_vjp::Params p) {
    extern __shared__ unsigned char segment[];
    layout_real_vjp::real_vjp_loop<T, MAPS, Y, ONE_TRAIN, SEGMENT>(p, reinterpret_cast<T*>(segment));
}

template <class T, int Y>
int launch_real_adjoint(const layout_real_vjp::Params& p, cudaStream_t stream) {
    const int groups = 32 / p.width;
    const long long problems = p.problem_end - p.problem_base;
    const long long per_block = static_cast<long long>(WARPS) * groups * Y;
    const unsigned grid = static_cast<unsigned>((problems + per_block - 1) / per_block);
    const bool maps = p.transmit || p.density || p.inverting;
    const size_t shared = static_cast<size_t>(SEGMENT) * 3 * Y * 32 * WARPS * sizeof(T);
#define BLOCHSIM_GO(MP, ONE)                                                                                \
    cudaFuncSetAttribute(real_adjoint_kernel<T, MP, Y, ONE>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                         static_cast<int>(shared));                                                     \
    real_adjoint_kernel<T, MP, Y, ONE><<<grid, 32 * WARPS, shared, stream>>>(p)
    if (p.single_train) {
        if (maps) { BLOCHSIM_GO(true, true); } else { BLOCHSIM_GO(false, true); }
    } else {
        if (maps) { BLOCHSIM_GO(true, false); } else { BLOCHSIM_GO(false, false); }
    }
#undef BLOCHSIM_GO
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

int real_vjp(const layout_real_vjp::Params& p, cudaStream_t stream) { return launch_real_adjoint<float, 4>(p, stream); }

int real_vjp_jvp(const layout_real_vjp::Params& p, cudaStream_t stream) {
    return launch_real_adjoint<num::Dual, 2>(p, stream);
}

}  // namespace blochsim_layout
