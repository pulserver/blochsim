// The real EPG event loop over a number type: the forward simulation at
// float, its Jacobian-vector product at num::Dual. One warp per program;
// lanes along the states, groups of lanes across problems, Y problems of a
// group in each thread's registers.
//
// Layout (compile time): MAPS (per-voxel transmit, density or inversion
// efficiency in registers), Y problems per thread, ONE_TRAIN. Run time:
// atom_stride, shimmed, diffusing, transmit, density, inverting.
#pragma once
#include <cuda_runtime.h>
#include "_layout_numbers.hpp"

namespace layout_real {

template <class T>
__device__ __forceinline__ float stored(T x) {
    if constexpr (num::is_dual<T>::value) return x.d;
    else return x;
}

struct Params {
    const float *t1, *t2, *m0, *b1, *inversion_efficiency, *diffusion, *duration, *flip;
    const float *d_t1, *d_t2, *d_m0, *d_b1, *d_inversion_efficiency, *d_diffusion, *d_duration, *d_flip;
    const int *kind, *output_index, *shim_index;
    const unsigned char* action;
    float *output_real, *output_imag;
    int atom_count, train_count, event_count, output_count, state_count, width;
    bool single_train, atom_stride, shimmed, diffusing, transmit, density, inverting;
};

template <class T>
__device__ __forceinline__ void rotate(T c, T s, T& plus, T& minus, T& z) {
    const T chs = 0.5f * (1.0f + c), shs = 0.5f * (1.0f - c), hs = 0.5f * s;
    const T p = chs * plus + shs * minus - s * z;
    const T m = shs * plus + chs * minus + s * z;
    z = hs * plus - hs * minus + c * z;
    plus = p;
    minus = m;
}

template <class T, bool MAPS, int Y, bool ONE_TRAIN>
__device__ __forceinline__ void real_loop(const Params& p) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int width = p.width;
    const int state = lane & (width - 1);
    const int group = lane / width;
    const int groups = 32 / width;
    const int first = ((blockIdx.x * (blockDim.x >> 5) + warp) * groups + group) * Y;
    const float order = static_cast<float>(state);
    const int total = p.train_count * p.atom_count;

    int problem[Y], atom[Y], train[Y];
    bool active[Y], live[Y];
    T r1[Y], r2[Y], m0[MAPS ? Y : 1], b1[MAPS ? Y : 1], inversion[MAPS ? Y : 1];
    // Along a direction the interval is rarely the last one again, so its
    // factors are formed every event and diffusion is held for them.
    constexpr bool HOLD = num::is_dual<T>::value;
    T diffusion[HOLD ? Y : 1];
    T plus[Y], minus[Y], z[Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        problem[y] = first + y;
        active[y] = problem[y] < total;
        live[y] = active[y] && state < p.state_count;
        atom[y] = problem[y] % p.atom_count;
        train[y] = problem[y] / p.atom_count;
        r1[y] = num::rate(active[y] ? num::load<T>(p.t1, p.d_t1, atom[y]) : T(1.0f));
        r2[y] = num::rate(active[y] ? num::load<T>(p.t2, p.d_t2, atom[y]) : T(1.0f));
        if constexpr (MAPS) {
            const int at = p.atom_stride ? atom[y] : 0;
            m0[y] = p.density ? (active[y] ? num::load<T>(p.m0, p.d_m0, at) : T(0.0f)) : T(1.0f);
            b1[y] = p.transmit ? (active[y] ? num::load<T>(p.b1, p.d_b1, at) : T(1.0f)) : T(1.0f);
            inversion[y] = p.inverting ? (active[y] ? num::load<T>(p.inversion_efficiency, p.d_inversion_efficiency, at) : T(1.0f)) : T(1.0f);
        }
        if constexpr (HOLD) {
            diffusion[y] = p.diffusing && active[y] ? num::load<T>(p.diffusion, p.d_diffusion, p.atom_stride ? atom[y] : 0) : T(0.0f);
        }
        plus[y] = minus[y] = 0.0f;
        z[y] = state == 0 ? 1.0f : 0.0f;
    }
    // A thread's problems are almost always voxels of one train; then an
    // event's duration and flip are one read for all of them.
    const bool uniform = ONE_TRAIN || (active[0] && train[0] == train[Y - 1]);
    const int base = ONE_TRAIN ? 0 : train[0] * p.event_count;
    const bool shared_pulse = uniform && !p.transmit;
    // The relaxation factors, with diffusion's damping of this thread's
    // order folded in, kept while the interval repeats.
    T e1c[Y], e2c[Y], recovery[Y];
    T last_dt = -1.0f;

    auto shift = [&]() {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T up = num::shfl_up(plus[y], 1, width);
            const T down = num::shfl_down(minus[y], 1, width);
            const T shifted_plus = (state > 0 && live[y]) ? up : T(0.0f);
            const T shifted_minus = (state + 1 < p.state_count && live[y]) ? down : T(0.0f);
            plus[y] = state == 0 ? -shifted_minus : shifted_plus;
            minus[y] = shifted_minus;
        }
    };
    auto factors = [&](int y, T dt) {
        T e1 = num::exp_(-r1[y] * dt), e2 = num::exp_(-r2[y] * dt);
        recovery[y] = 1.0f - e1;
        if (p.diffusing) {
            T held_diffusion;
            if constexpr (HOLD) {
                held_diffusion = diffusion[y];
            } else {
                const int at = p.atom_stride ? atom[y] : 0;
                held_diffusion = active[y] ? num::load<T>(p.diffusion, p.d_diffusion, at) : T(0.0f);
            }
            const T b = held_diffusion * dt;
            const float sq = order * order;
            e1 *= num::exp_(-b * sq);
            e2 *= num::exp_(-b * (sq + order + 0.3333333333333333f));
        }
        e1c[y] = e1;
        e2c[y] = e2;
    };

    float held[Y];
    int run_start = 0, run_len = 0;
    auto flush = [&]() {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            if (state < run_len && active[y]) {
                const long at_out = static_cast<long>(problem[y]) * p.output_count + run_start + state;
                p.output_real[at_out] = 0.0f;
                p.output_imag[at_out] = held[y];
            }
        }
        run_len = 0;
    };
#pragma unroll(num::is_dual<T>::value ? 1 : 2)
    for (int event = 0; event < p.event_count; ++event) {
        const T dt_shared = uniform ? num::load<T>(p.duration, p.d_duration, base + event) : T(0.0f);
        const unsigned char act = p.action[event];
        const int kind = p.kind[event];
        if (uniform) {
            if (!num::same(dt_shared, T(0.0f))) {
                if (!num::same(dt_shared, last_dt)) {
#pragma unroll
                    for (int y = 0; y < Y; ++y) factors(y, dt_shared);
                    last_dt = dt_shared;
                }
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    plus[y] *= e2c[y];
                    minus[y] *= e2c[y];
                    z[y] = z[y] * e1c[y] + (state == 0 ? recovery[y] : T(0.0f));
                }
            }
        } else {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                factors(y, active[y] ? num::load<T>(p.duration, p.d_duration, train[y] * p.event_count + event) : T(0.0f));
                plus[y] *= e2c[y];
                minus[y] *= e2c[y];
                z[y] = z[y] * e1c[y] + (state == 0 ? recovery[y] : T(0.0f));
            }
            last_dt = -1.0f;
        }
        if (act & 1) shift();
        if (kind == 1 && (act & 4)) {
#pragma unroll
            for (int y = 0; y < Y; ++y) z[y] = -(MAPS ? inversion[MAPS ? y : 0] : T(1.0f)) * z[y];
        } else if (kind == 1) {
            if (shared_pulse) {
                T s, c;
                num::sincos_(num::load<T>(p.flip, p.d_flip, base + event), s, c);
#pragma unroll
                for (int y = 0; y < Y; ++y) rotate(c, s, plus[y], minus[y], z[y]);
            } else {
                const T flip_u = uniform ? num::load<T>(p.flip, p.d_flip, base + event) : T(0.0f);
                const bool shim = p.shimmed && p.transmit;
                const int shim_row = shim ? p.shim_index[event] * p.atom_count : 0;
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    T alpha = uniform ? flip_u
                                      : (active[y] ? num::load<T>(p.flip, p.d_flip, train[y] * p.event_count + event) : T(0.0f));
                    T pulse_b1 = MAPS ? b1[MAPS ? y : 0] : T(1.0f);
                    if (shim) pulse_b1 = active[y] ? num::load<T>(p.b1, p.d_b1, shim_row + atom[y]) : T(1.0f);
                    T s, c;
                    num::sincos_(alpha * pulse_b1, s, c);
                    rotate(c, s, plus[y], minus[y], z[y]);
                }
            }
        }
        if ((act & 32) && kind == 2) {
            const int out = p.output_index[event];
            if (out >= 0) {
                if (run_len > 0 && (out != run_start + run_len || run_len == width)) flush();
                if (run_len == 0) run_start = out;
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    const T first_state = num::shfl(plus[y], 0, width);
                    if (state == run_len) held[y] = stored((MAPS ? m0[MAPS ? y : 0] : T(1.0f)) * first_state);
                }
                ++run_len;
            }
        }
        if (act & 18) shift();
        if (act & 8) {
#pragma unroll
            for (int y = 0; y < Y; ++y) plus[y] = minus[y] = 0.0f;
        }
    }
    if (run_len > 0) flush();
}

}  // namespace layout_real
