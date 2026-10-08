// The EPG kernels written for their layouts rather than over tiles.
//
// A layout is what a kernel is compiled for: the pools it carries, how a
// pulse is formed, whether the tissue has per-voxel maps, how many problems a
// thread holds and whether there is one train. Every other switch is read at
// run time and steers whole blocks once per event, so one compile serves
// every combination of them. The forward kernels and their Jacobian-vector
// products are one source over a number type: a float, or a dual number that
// carries a direction beside each value (_layout_numbers.hpp).
#pragma once

#include <cuda_runtime.h>

#include "_kernels.hpp"
#include "_layout_complex.hpp"
#include "_layout_complex_vjp.hpp"
#include "_layout_pooled.hpp"
#include "_layout_real.hpp"
#include "_layout_real_vjp.hpp"

namespace blochsim_layout {

// Queue ``kernel`` over ``programs`` on ``stream`` and return its launch
// status, or -1 where it has no layout for these arguments and the tile
// kernel is to run instead.
int launch(int kernel, const bsk::Arguments& arguments, long long programs, cudaStream_t stream);

// The floats of the buffer the many-pool adjoint's layout takes for
// ``problems``, or -1 where the tile kernels run it. A launch the layout
// takes records nothing first: the adjoint keeps its own checkpoints.
long long pooled_adjoint_floats(long long problems, int event_count, int n, int m, int width, bool dual);

// One translation unit per kernel and pool layout, so a build compiles them
// side by side. ``rf`` and ``mode`` are epg::Rf and epg::Mode.
#define BLOCHSIM_LAYOUT_COMPLEX(X) \
    X(forward, 0) X(forward, 1) X(forward, 2) X(forward, 3) X(jvp, 0) X(jvp, 1) X(jvp, 2) X(jvp, 3)
#define BLOCHSIM_LAYOUT_COMPLEX_DECLARATION(kind, pools) \
    int complex_##kind##_##pools(const epg::Params& p, int rf, int mode, cudaStream_t stream);
BLOCHSIM_LAYOUT_COMPLEX(BLOCHSIM_LAYOUT_COMPLEX_DECLARATION)
#undef BLOCHSIM_LAYOUT_COMPLEX_DECLARATION

int real_forward(const layout_real::Params& p, cudaStream_t stream);
int real_jvp(const layout_real::Params& p, cudaStream_t stream);

// The adjoints: the forward sweep keeps a checkpoint every few events and
// the reverse sweep replays each stretch from it, so a launch is one sweep
// each way and the recording launch of the tile kernels' protocol does
// nothing.
#define BLOCHSIM_LAYOUT_ADJOINT(X) \
    X(vjp, 0) X(vjp, 1) X(vjp, 2) X(vjp, 3) X(vjp_jvp, 0) X(vjp_jvp, 1) X(vjp_jvp, 2) X(vjp_jvp, 3)
#define BLOCHSIM_LAYOUT_ADJOINT_DECLARATION(kind, pools) \
    int complex_##kind##_##pools(const epg_vjp::Params& v, int rf, cudaStream_t stream);
BLOCHSIM_LAYOUT_ADJOINT(BLOCHSIM_LAYOUT_ADJOINT_DECLARATION)
#undef BLOCHSIM_LAYOUT_ADJOINT_DECLARATION

int real_vjp(const layout_real_vjp::Params& p, cudaStream_t stream);
int real_vjp_jvp(const layout_real_vjp::Params& p, cudaStream_t stream);

// The many-pool kernels, a unit per pool count. The adjoint keeps every
// POOLED_SEGMENT-th state and replays the stretches between.
constexpr int POOLED_SEGMENT = 4;
#define BLOCHSIM_LAYOUT_POOLED(X) X(2) X(3) X(4) X(5) X(6) X(7) X(8)
#define BLOCHSIM_LAYOUT_POOLED_DECLARATION(pools)                                                   \
    int pooled_forward_##pools(const epg_pooled::Params& p, int rf, bool dual, cudaStream_t stream); \
    int pooled_adjoint_##pools(const epg_pooled::Params& p, const epg_pooled::Adjoint& g, int rf, bool dual, \
                               cudaStream_t stream);
BLOCHSIM_LAYOUT_POOLED(BLOCHSIM_LAYOUT_POOLED_DECLARATION)
#undef BLOCHSIM_LAYOUT_POOLED_DECLARATION

// Programs of two warps each.
constexpr int WARPS = 2;

}  // namespace blochsim_layout
