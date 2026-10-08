// The real EPG adjoint over a number type: the vector-Jacobian product at
// float, its derivative along a direction (the Hessian-vector product's
// sweep) at num::Dual.
//
// The forward sweep keeps the state every K events; the reverse sweep
// replays each stretch of K events from its checkpoint into shared memory
// and walks it back. One extra forward pass replaces a trajectory of every
// event through global memory.
#pragma once
#include <cuda_runtime.h>

#include "_layout_numbers.hpp"

namespace layout_real_vjp {

using num::exp_;
using num::same;

struct Params {
    const float *t1, *t2, *m0, *b1, *inversion_efficiency, *diffusion, *duration, *flip;
    const float *d_t1, *d_t2, *d_m0, *d_b1, *d_inversion_efficiency, *d_diffusion, *d_duration, *d_flip;
    const int *kind, *output_index, *shim_index;
    const unsigned char* action;
    const float* grad_output_imag;
    // Value and, for a dual sweep, direction of each gradient.
    float *grad_tissue, *grad_flip, *grad_duration;
    float *grad_tissue_t, *grad_flip_t, *grad_duration_t;
    float *trajectory, *trajectory_t;
    int problem_base, problem_end, atom_count, train_count, event_count, output_count, state_count, width;
    int shim_rows;
    bool single_train, atom_stride, shimmed, diffusing, transmit, density, inverting;
};

template <class T>
__device__ __forceinline__ void atomic_add(float* value, float* tangent, long long at, T x) {
    if constexpr (num::is_dual<T>::value) {
        atomicAdd(value + at, x.v);
        atomicAdd(tangent + at, x.d);
    } else {
        atomicAdd(value + at, x);
    }
}

template <class T>
__device__ __forceinline__ T shfl_xor(T x, int mask) {
    if constexpr (num::is_dual<T>::value) {
        return T{__shfl_xor_sync(0xffffffffu, x.v, mask), __shfl_xor_sync(0xffffffffu, x.d, mask)};
    } else {
        return __shfl_xor_sync(0xffffffffu, x, mask);
    }
}

// Sum over the lanes ``from`` apart and wider, up to a warp.
template <class T>
__device__ __forceinline__ T lanes_sum(T x, int from, int to) {
    for (int offset = from; offset < to; offset <<= 1) x += shfl_xor(x, offset);
    return x;
}

template <class T>
__device__ __forceinline__ void store(float* values, float* tangents, long long at, T x) {
    if constexpr (num::is_dual<T>::value) {
        values[at] = x.v;
        tangents[at] = x.d;
    } else {
        values[at] = x;
    }
}

template <class T>
__device__ __forceinline__ T fetch(const float* values, const float* tangents, long long at) {
    if constexpr (num::is_dual<T>::value) {
        return T{values[at], tangents[at]};
    } else {
        return values[at];
    }
}

template <class T, bool MAPS, int Y, bool ONE_TRAIN, int K>
__device__ __forceinline__ void real_vjp_loop(const Params& p, T* segment) {
    constexpr int PLANES = 3;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int width = p.width;
    const int state = lane & (width - 1);
    const int group = lane / width;
    const int groups = 32 / width;
    const int first = p.problem_base + ((blockIdx.x * (blockDim.x >> 5) + warp) * groups + group) * Y;
    const float order = static_cast<float>(state);
    const float transverse_weight = order * order + order + 0.3333333333333333f;
    const float longitudinal_weight = order * order;

    int problem[Y], atom[Y], train[Y];
    bool active[Y], live[Y];
    T t1[Y], t2[Y], r1[Y], r2[Y], m0[MAPS ? Y : 1], b1[MAPS ? Y : 1], inversion[MAPS ? Y : 1], diffusion[Y];
    T plus[Y], minus[Y], z[Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        problem[y] = first + y;
        active[y] = problem[y] < p.problem_end;
        live[y] = active[y] && state < p.state_count;
        atom[y] = problem[y] % p.atom_count;
        train[y] = problem[y] / p.atom_count;
        const int at = p.atom_stride ? atom[y] : 0;
        t1[y] = active[y] ? num::load<T>(p.t1, p.d_t1, atom[y]) : T(1.0f);
        t2[y] = active[y] ? num::load<T>(p.t2, p.d_t2, atom[y]) : T(1.0f);
        r1[y] = num::rate(t1[y]);
        r2[y] = num::rate(t2[y]);
        if constexpr (MAPS) {
            m0[y] = p.density ? (active[y] ? num::load<T>(p.m0, p.d_m0, at) : T(0.0f)) : T(1.0f);
            b1[y] = p.transmit ? (active[y] ? num::load<T>(p.b1, p.d_b1, at) : T(1.0f)) : T(1.0f);
            inversion[y] = p.inverting ? (active[y] ? num::load<T>(p.inversion_efficiency, p.d_inversion_efficiency, at)
                                                    : T(1.0f))
                                       : T(1.0f);
        }
        diffusion[y] = p.diffusing && active[y] ? num::load<T>(p.diffusion, p.d_diffusion, at) : T(0.0f);
        plus[y] = minus[y] = 0.0f;
        z[y] = state == 0 ? 1.0f : 0.0f;
    }
    const bool uniform = ONE_TRAIN || (active[0] && train[0] == train[Y - 1]);
    const int base = ONE_TRAIN ? 0 : train[0] * p.event_count;
    // Every problem of the warp in one train: an event's gradient is one
    // sum over the warp and one atomic.
    const int first_train = __shfl_sync(0xffffffffu, train[0], 0);
    const bool warp_train = ONE_TRAIN || __all_sync(0xffffffffu, uniform && train[0] == first_train);
    const bool shared_pulse = uniform && !p.transmit;

    // The interval's factors, kept while it repeats: with diffusion's
    // damping of this thread's order, and bare.
    T e1c[Y], e2c[Y], bare1[Y], bare2[Y], damp_z[Y], damp_t[Y];
    T last_dt = -1.0f;
    auto factors = [&](T dt_shared, int event) {
        if (uniform && same(dt_shared, last_dt)) return;
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T dt = uniform ? dt_shared
                                 : (active[y] ? num::load<T>(p.duration, p.d_duration, train[y] * p.event_count + event)
                                              : T(0.0f));
            bare1[y] = exp_(-r1[y] * dt);
            bare2[y] = exp_(-r2[y] * dt);
            damp_z[y] = 1.0f;
            damp_t[y] = 1.0f;
            if (p.diffusing) {
                const T b = diffusion[y] * dt;
                damp_z[y] = exp_(-b * longitudinal_weight);
                damp_t[y] = exp_(-b * transverse_weight);
            }
            e1c[y] = bare1[y] * damp_z[y];
            e2c[y] = bare2[y] * damp_t[y];
        }
        last_dt = uniform ? dt_shared : T(-1.0f);
    };
    auto event_dt = [&](int event) {
        return uniform ? num::load<T>(p.duration, p.d_duration, base + event) : T(0.0f);
    };
    auto shift = [&](T* pl, T* mi) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T up = num::shfl_up(pl[y], 1, width);
            const T down = num::shfl_down(mi[y], 1, width);
            const T shifted_plus = (state > 0 && live[y]) ? up : T(0.0f);
            const T shifted_minus = (state + 1 < p.state_count && live[y]) ? down : T(0.0f);
            pl[y] = state == 0 ? -shifted_minus : shifted_plus;
            mi[y] = shifted_minus;
        }
    };
    // Transpose of the shift: the order-zero refill sends plus's adjoint
    // back onto minus at order one.
    auto shift_adjoint = [&](T* pl, T* mi) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T carry = -num::shfl(pl[y], 0, width);
            const T down = num::shfl_down(pl[y], 1, width);
            const T up = num::shfl_up(mi[y], 1, width);
            T shifted_minus = (state > 0 && live[y]) ? up : T(0.0f);
            if (state == 1 && live[y]) shifted_minus += carry;
            pl[y] = (state + 1 < p.state_count && live[y]) ? down : T(0.0f);
            mi[y] = shifted_minus;
        }
    };
    auto flip_of = [&](int y, int event) {
        return uniform ? num::load<T>(p.flip, p.d_flip, base + event)
                       : (active[y] ? num::load<T>(p.flip, p.d_flip, train[y] * p.event_count + event) : T(0.0f));
    };
    auto pulse_b1 = [&](int y, int event) {
        T value = MAPS ? b1[MAPS ? y : 0] : T(1.0f);
        if (p.shimmed && p.transmit) {
            value = active[y] ? num::load<T>(p.b1, p.d_b1, p.shim_index[event] * p.atom_count + atom[y]) : T(1.0f);
        }
        return value;
    };
    // One event forward, from its entry state.
    auto forward = [&](int event) {
        const T dt_shared = event_dt(event);
        const unsigned char act = p.action[event];
        const int kind = p.kind[event];
        if (!(uniform && same(dt_shared, T(0.0f)))) {
            factors(dt_shared, event);
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                plus[y] *= e2c[y];
                minus[y] *= e2c[y];
                z[y] = z[y] * e1c[y] + (state == 0 ? 1.0f - bare1[y] : T(0.0f));
            }
        }
        if (act & 1) shift(plus, minus);
        if (kind == 1 && (act & 4)) {
#pragma unroll
            for (int y = 0; y < Y; ++y) z[y] = -(MAPS ? inversion[MAPS ? y : 0] : T(1.0f)) * z[y];
        } else if (kind == 1) {
            T s_u, c_u;
            if (shared_pulse) num::sincos_(num::load<T>(p.flip, p.d_flip, base + event), s_u, c_u);
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                T s = s_u, c = c_u;
                if (!shared_pulse) num::sincos_(flip_of(y, event) * pulse_b1(y, event), s, c);
                const T chs = 0.5f * (1.0f + c), shs = 0.5f * (1.0f - c), hs = 0.5f * s;
                const T pl = chs * plus[y] + shs * minus[y] - s * z[y];
                const T mi = shs * plus[y] + chs * minus[y] + s * z[y];
                z[y] = hs * plus[y] - hs * minus[y] + c * z[y];
                plus[y] = pl;
                minus[y] = mi;
            }
        }
        if (act & 18) shift(plus, minus);
        if (act & 8) {
#pragma unroll
            for (int y = 0; y < Y; ++y) plus[y] = minus[y] = 0.0f;
        }
    };
    const long long stride = static_cast<long long>(p.event_count) * PLANES * p.state_count;
    auto slot = [&](int y, int checkpoint, int plane) {
        return (problem[y] - p.problem_base) * stride + (static_cast<long long>(checkpoint) * PLANES + plane) * p.state_count +
               state;
    };
    const int checkpoints = (p.event_count + K - 1) / K;
    #pragma unroll 1
    for (int event = 0; event < p.event_count; ++event) {
        if (event % K == 0) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                if (live[y]) {
                    store(p.trajectory, p.trajectory_t, slot(y, event / K, 0), plus[y]);
                    store(p.trajectory, p.trajectory_t, slot(y, event / K, 1), minus[y]);
                    store(p.trajectory, p.trajectory_t, slot(y, event / K, 2), z[y]);
                }
            }
        }
        forward(event);
    }

    // The walk back. ``plus`` and friends now hold the state, and the bars
    // its cotangent.
    T plus_bar[Y], minus_bar[Y], z_bar[Y];
    T g_t1[Y], g_t2[Y], g_m0[Y], g_b1[Y], g_inversion[Y], g_diffusion[Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        plus_bar[y] = minus_bar[y] = z_bar[y] = 0.0f;
        g_t1[y] = g_t2[y] = g_m0[y] = g_b1[y] = g_inversion[y] = g_diffusion[y] = 0.0f;
    }
    // An event's gradient, summed over the problems that share its row.
    auto event_gradient = [&](float* value, float* tangent, int event, const T* lane_values) {
        if (warp_train) {
            T sum = lane_values[0];
#pragma unroll
            for (int y = 1; y < Y; ++y) sum += lane_values[y];
            sum = lanes_sum(sum, 1, 32);
            if (lane == 0 && active[0]) atomic_add(value, tangent, base + event, sum);
        } else {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                const T sum = lanes_sum(lane_values[y], 1, width);
                if (state == 0 && active[y]) atomic_add(value, tangent, train[y] * p.event_count + event, sum);
            }
        }
    };
    const int threads = blockDim.x;
    auto smem = [&](int k, int plane, int y) -> T& { return segment[((k * PLANES + plane) * Y + y) * threads + threadIdx.x]; };
    #pragma unroll 1
    for (int checkpoint = checkpoints - 1; checkpoint >= 0; --checkpoint) {
        const int start = checkpoint * K;
        const int stop = min(start + K, p.event_count);
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            plus[y] = live[y] ? fetch<T>(p.trajectory, p.trajectory_t, slot(y, checkpoint, 0)) : T(0.0f);
            minus[y] = live[y] ? fetch<T>(p.trajectory, p.trajectory_t, slot(y, checkpoint, 1)) : T(0.0f);
            z[y] = live[y] ? fetch<T>(p.trajectory, p.trajectory_t, slot(y, checkpoint, 2)) : T(0.0f);
        }
        #pragma unroll 1
        for (int event = start; event < stop; ++event) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                smem(event - start, 0, y) = plus[y];
                smem(event - start, 1, y) = minus[y];
                smem(event - start, 2, y) = z[y];
            }
            if (event + 1 < stop) forward(event);
        }
        #pragma unroll 1
        for (int event = stop - 1; event >= start; --event) {
            T entry_p[Y], entry_m[Y], entry_z[Y];
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                entry_p[y] = smem(event - start, 0, y);
                entry_m[y] = smem(event - start, 1, y);
                entry_z[y] = smem(event - start, 2, y);
            }
            const T dt_shared = event_dt(event);
            const unsigned char act = p.action[event];
            const int kind = p.kind[event];
            factors(dt_shared, event);
            // The stage the pulse and the sample see.
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                plus[y] = entry_p[y] * e2c[y];
                minus[y] = entry_m[y] * e2c[y];
                z[y] = entry_z[y] * e1c[y] + (state == 0 ? 1.0f - bare1[y] : T(0.0f));
            }
            if (act & 1) shift(plus, minus);
            if (act & 8) {
#pragma unroll
                for (int y = 0; y < Y; ++y) plus_bar[y] = minus_bar[y] = 0.0f;
            } else if (act & 18) {
                shift_adjoint(plus_bar, minus_bar);
            }
            if (kind == 1 && (act & 4)) {
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    g_inversion[y] -= z_bar[y] * z[y];
                    z_bar[y] = -(MAPS ? inversion[MAPS ? y : 0] : T(1.0f)) * z_bar[y];
                }
            } else if (kind == 1) {
                T flip_gradient[Y];
                // Several shims give each pulse's transmit gradient its shim's row.
                const int shim_row = p.shimmed ? p.shim_index[event] * p.atom_count : 0;
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    const T flip = flip_of(y, event), b1y = pulse_b1(y, event);
                    T s, c;
                    num::sincos_(flip * b1y, s, c);
                    const T chs = 0.5f * (1.0f + c), shs = 0.5f * (1.0f - c), hs = 0.5f * s;
                    // d/dalpha of each output row, against its cotangent.
                    const T row_p = hs * minus[y] - hs * plus[y] - c * z[y];
                    const T row_m = hs * plus[y] - hs * minus[y] + c * z[y];
                    const T row_z = 0.5f * c * plus[y] - 0.5f * c * minus[y] - s * z[y];
                    const T alpha_bar = plus_bar[y] * row_p + minus_bar[y] * row_m + z_bar[y] * row_z;
                    const T pb = chs * plus_bar[y] + shs * minus_bar[y] + hs * z_bar[y];
                    const T mb = shs * plus_bar[y] + chs * minus_bar[y] - hs * z_bar[y];
                    const T zb = -s * plus_bar[y] + s * minus_bar[y] + c * z_bar[y];
                    plus_bar[y] = pb;
                    minus_bar[y] = mb;
                    z_bar[y] = zb;
                    flip_gradient[y] = alpha_bar * b1y;
                    if (p.shimmed) {
                        const T sum = lanes_sum(alpha_bar * flip, 1, width);
                        if (state == 0 && active[y]) {
                            atomic_add(p.grad_tissue, p.grad_tissue_t, 3LL * p.atom_count + shim_row + atom[y], sum);
                        }
                    } else {
                        g_b1[y] += alpha_bar * flip;
                    }
                }
                event_gradient(p.grad_flip, p.grad_flip_t, event, flip_gradient);
            }
            if ((act & 32) && kind == 2) {
                const int out = p.output_index[event];
                if (out >= 0) {
#pragma unroll
                    for (int y = 0; y < Y; ++y) {
                        if (state == 0 && active[y]) {
                            const float seed = p.grad_output_imag[static_cast<long long>(problem[y]) * p.output_count + out];
                            g_m0[y] += seed * plus[y];
                            plus_bar[y] += seed * (MAPS ? m0[MAPS ? y : 0] : T(1.0f));
                        }
                    }
                }
            }
            if (act & 1) shift_adjoint(plus_bar, minus_bar);
            T duration_gradient[Y];
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                const T cot2 = plus_bar[y] * entry_p[y] + minus_bar[y] * entry_m[y];
                const T cot1 = z_bar[y] * entry_z[y];
                const T ge2 = cot2 * damp_t[y];
                const T ge1 = cot1 * damp_z[y] - (state == 0 ? z_bar[y] : T(0.0f));
                const T dt = uniform ? dt_shared
                                     : (active[y] ? num::load<T>(p.duration, p.d_duration, train[y] * p.event_count + event)
                                                  : T(0.0f));
                T spread = 0.0f;
                if (p.diffusing) {
                    spread = cot1 * bare1[y] * damp_z[y] * longitudinal_weight +
                             cot2 * bare2[y] * damp_t[y] * transverse_weight;
                    g_diffusion[y] -= spread * dt;
                }
                g_t1[y] += ge1 * bare1[y] * dt * num::div_(T(1000.0f), t1[y] * t1[y]);
                g_t2[y] += ge2 * bare2[y] * dt * num::div_(T(1000.0f), t2[y] * t2[y]);
                duration_gradient[y] = -ge1 * r1[y] * bare1[y] - ge2 * r2[y] * bare2[y] - spread * diffusion[y];
                plus_bar[y] *= e2c[y];
                minus_bar[y] *= e2c[y];
                z_bar[y] *= e1c[y];
            }
            event_gradient(p.grad_duration, p.grad_duration_t, event, duration_gradient);
        }
    }
    const int past_transmit = 2 * (p.shim_rows - 1);
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        const T t1_sum = lanes_sum(g_t1[y], 1, width), t2_sum = lanes_sum(g_t2[y], 1, width);
        const T m0_sum = lanes_sum(g_m0[y], 1, width), b1_sum = lanes_sum(g_b1[y], 1, width);
        const T inv_sum = lanes_sum(g_inversion[y], 1, width), diff_sum = lanes_sum(g_diffusion[y], 1, width);
        if (state == 0 && active[y]) {
            const long long n = p.atom_count;
            atomic_add(p.grad_tissue, p.grad_tissue_t, atom[y], t1_sum);
            atomic_add(p.grad_tissue, p.grad_tissue_t, n + atom[y], t2_sum);
            atomic_add(p.grad_tissue, p.grad_tissue_t, 2 * n + atom[y], m0_sum);
            if (!p.shimmed) atomic_add(p.grad_tissue, p.grad_tissue_t, 3 * n + atom[y], b1_sum);
            atomic_add(p.grad_tissue, p.grad_tissue_t, (6 + past_transmit) * n + atom[y], inv_sum);
            atomic_add(p.grad_tissue, p.grad_tissue_t, (7 + past_transmit) * n + atom[y], diff_sum);
        }
    }
}

}  // namespace layout_real_vjp
