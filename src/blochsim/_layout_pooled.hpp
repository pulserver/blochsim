// The many-pool EPG kernels over a number type: the forward simulation at
// float, its Jacobian-vector product at num::Dual, and the adjoint whose
// derivative along a direction is the same source at num::Dual. Every interval's exchange
// is read from the tissue's own table: a longitudinal operator over the N
// pools, what each recovers, and a complex transverse operator over the m
// exchanging pools. Lanes along the states, groups of lanes across problems,
// one problem's pools in each thread's registers.
#pragma once
#include "_layout_complex.hpp"
#include "_layout_complex_vjp.hpp"

namespace epg_pooled {

using namespace epg;

struct Params {
    const float *m0, *b1, *b1_phase, *b0, *efficiency, *diffusion, *velocity;
    const float *dm0, *db1, *db1_phase, *db0, *defficiency, *ddiffusion, *dvelocity;
    const float *duration, *flip, *phase, *saturation, *rf_frequency, *dduration, *dflip, *dphase;
    const float *table, *dtable, *profile, *lineshape, *pairs, *dpairs;
    const int *kind, *output_index, *shim_index, *pool_index, *profile_index, *pair_index;
    const unsigned char* action;
    float *output_real, *output_imag;
    long long base;
    int problems, atom_count, event_count, output_count, state_count, rows, width, m, blocks;
    int locations, profile_bins, lineshape_bins;
    float flow_scale, washout_scale, profile_step, lineshape_step;
    bool atom_stride, shimmed, directed_pairs, directed_table, off_axis, moving, diffusing, transmit, density,
        inverting;
};

template <class T>
struct Z {
    T r, i;
};
template <class T>
__device__ __forceinline__ Z<T> zmul(const Z<T>& a, const Z<T>& b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
template <class T>
__device__ __forceinline__ Z<T> zconj(const Z<T>& a) {
    return {a.r, -a.i};
}

template <class T, int N, int RF>
__device__ __forceinline__ void pooled_loop(const Params& p) {
    constexpr bool DUAL = num::is_dual<T>::value;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int width = p.width;
    const int state = lane & (width - 1);
    const int group = lane / width;
    const int groups = 32 / width;
    const long long index = (static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + warp) * groups + group;
    const bool active = index < p.problems;
    const long long problem = p.base + (active ? index : 0);
    const bool live = active && state < p.state_count;
    const int atom = static_cast<int>(problem % p.atom_count);
    const int train = static_cast<int>(problem / p.atom_count);
    const int event_base = train * p.event_count;
    const int voxel_at = p.atom_stride ? atom : 0;
    const int location = RF == PROFILE ? atom % p.locations : 0;
    const float order = static_cast<float>(state);
    const float squared = order * order, transverse_weight = squared + order + 0.3333333333333333f;
    const int m = p.m;
    const int row_width = N * N + N + 2 * m * m;
    const long long table_width = N + static_cast<long long>(p.rows) * row_width * p.blocks;
    const float* slot = p.table + atom * table_width;
    const float* directions = p.directed_table ? p.dtable + atom * table_width : nullptr;
    const long long sloped_at = static_cast<long long>(p.rows) * row_width;

    auto read = [&](const float* values, const float* tangents, long long at, bool on, float identity) -> T {
        if (!on) return T(identity);
        if constexpr (DUAL) {
            return T{__ldg(values + at), __ldg(tangents + at)};
        } else {
            return __ldg(values + at);
        }
    };
    // A table entry, moving with the table's direction and, where the table
    // carries a slope along the interval's length, along that.
    auto entry = [&](long long at, float along, bool sloped) -> T {
        const float value = __ldg(slot + at);
        if constexpr (DUAL) {
            float tangent = 0.0f;
            if (directions != nullptr) tangent += __ldg(directions + at);
            if (sloped && p.blocks > 1) tangent += __ldg(slot + sloped_at + at) * along;
            return T{value, tangent};
        } else {
            return value;
        }
    };
    const T density = read(p.m0, p.dm0, voxel_at, p.density, 1.0f);
    const T voxel_b1 = read(p.b1, p.db1, voxel_at, p.transmit, 1.0f);
    const T voxel_b1_phase = read(p.b1_phase, p.db1_phase, voxel_at, p.off_axis, 0.0f);
    const T voxel_b0 = read(p.b0, p.db0, voxel_at, p.off_axis, 0.0f);
    const T inversion = read(p.efficiency, p.defficiency, voxel_at, p.inverting, 1.0f);
    const T damping_rate = read(p.diffusion, p.ddiffusion, voxel_at, p.diffusing, 0.0f);
    const T moved = read(p.velocity, p.dvelocity, voxel_at, p.moving, 0.0f);
    const T flow_rate = p.flow_scale * moved;
    const T washout_rate = p.washout_scale * abs_(moved);

    T equilibrium[N];
    Z<T> plus[N], minus[N], z[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        equilibrium[i] = entry(i, 0.0f, false);
        plus[i] = minus[i] = {T(0.0f), T(0.0f)};
        z[i] = {state == 0 ? equilibrium[i] : T(0.0f), T(0.0f)};
    }
    auto shift = [&]() {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (i >= m) break;
            const T up_r = num::shfl_up(plus[i].r, 1, width), up_i = num::shfl_up(plus[i].i, 1, width);
            const T dn_r = num::shfl_down(minus[i].r, 1, width), dn_i = num::shfl_down(minus[i].i, 1, width);
            const bool keep_up = state > 0 && live, keep_down = state + 1 < p.state_count && live;
            const T pr = keep_up ? up_r : T(0.0f), pi = keep_up ? up_i : T(0.0f);
            const T mr = keep_down ? dn_r : T(0.0f), mi = keep_down ? dn_i : T(0.0f);
            plus[i] = {state == 0 ? mr : pr, state == 0 ? -mi : pi};
            minus[i] = {mr, mi};
        }
    };
    auto event_value = [&](const float* values, const float* tangents, int event) -> T {
        if constexpr (DUAL) {
            return T{__ldg(values + event_base + event), __ldg(tangents + event_base + event)};
        } else {
            return __ldg(values + event_base + event);
        }
    };

#pragma unroll 1
    for (int event = 0; event < p.event_count; ++event) {
        const T dt = event_value(p.duration, p.dduration, event);
        // What the interval does to every pool alike at this order.
        T wout = 1.0f, damp_z = 1.0f, damp_t = 1.0f;
        if (p.diffusing) {
            const T b = damping_rate * dt;
            damp_z = exp_(-squared * b);
            damp_t = exp_(-transverse_weight * b);
        }
        if (p.moving) wout = 1.0f - min_(washout_rate * dt, T(1.0f));
        Z<T> carried = {wout * damp_t, T(0.0f)}, spin = {wout * damp_z, T(0.0f)};
        if (p.off_axis || p.moving) {
            const T turn = p.moving ? flow_rate * dt : T(0.0f);
            T s, c;
            sincos_(-TWO_PI * voxel_b0 * dt - (order + 0.5f) * turn, s, c);
            carried = {wout * damp_t * c, wout * damp_t * s};
            if (p.moving) {
                sincos_(-order * turn, s, c);
                spin = {wout * damp_z * c, wout * damp_z * s};
            }
        }
        // The interval's exchange, from the table row its length reads.
        const int row = p.pool_index[event_base + event];
        const long long row_at = N + static_cast<long long>(row) * row_width;
        const float along = DUAL ? num::tangent(dt) : 0.0f;
        Z<T> mixed_plus[N], mixed_minus[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            mixed_plus[i] = mixed_minus[i] = {T(0.0f), T(0.0f)};
            if (i >= m) continue;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                if (j >= m) break;
                const long long at = row_at + N * N + N + 2 * (i * m + j);
                const Z<T> x = {entry(at, along, true), entry(at + 1, along, true)};
                const Z<T> a = zmul(x, plus[j]), b = zmul(zconj(x), minus[j]);
                mixed_plus[i] = {mixed_plus[i].r + a.r, mixed_plus[i].i + a.i};
                mixed_minus[i] = {mixed_minus[i].r + b.r, mixed_minus[i].i + b.i};
            }
        }
        Z<T> mixed_z[N];
#pragma unroll
        for (int i = 0; i < N; ++i) {
            mixed_z[i] = {T(0.0f), T(0.0f)};
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const T l = entry(row_at + i * N + j, along, true);
                mixed_z[i] = {mixed_z[i].r + l * z[j].r, mixed_z[i].i + l * z[j].i};
            }
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (i < m) {
                plus[i] = zmul(carried, mixed_plus[i]);
                minus[i] = zmul(zconj(carried), mixed_minus[i]);
            }
            z[i] = zmul(spin, mixed_z[i]);
            if (state == 0) z[i].r += equilibrium[i] - wout * entry(row_at + N * N + i, along, true);
        }

        const unsigned char act = p.action[event];
        const int kind = p.kind[event];
        if (act & 1) shift();
        if (kind == 1) {
            if (act & 4) {
                // Every exchanging pool is free water and inverts like it; a
                // semisolid one is saturated by the pulse's own term.
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i < m) z[i] = {-inversion * z[i].r, -inversion * z[i].i};
                }
            } else {
                T pulse_b1 = voxel_b1, pulse_b1_phase = voxel_b1_phase;
                if (p.shimmed) {
                    const long long transmit_at = static_cast<long long>(p.shim_index[event]) * p.atom_count + atom;
                    pulse_b1 = read(p.b1, p.db1, transmit_at, p.transmit, 1.0f);
                    pulse_b1_phase = read(p.b1_phase, p.db1_phase, transmit_at, true, 0.0f);
                }
                const T alpha = event_value(p.flip, p.dflip, event) * pulse_b1;
                const T phi = event_value(p.phase, p.dphase, event) + pulse_b1_phase;
                if (N > m) {
                    const T shape = lineshape_at(p.lineshape, p.rf_frequency[event] - voxel_b0, p.lineshape_bins,
                                                 p.lineshape_step);
                    const T absorbed = exp_(p.saturation[event] * alpha * alpha * shape);
                    z[N - 1] = {absorbed * z[N - 1].r, absorbed * z[N - 1].i};
                }
                T ts, tc;
                sincos_(-phi, ts, tc);
                const Z<T> turn = {tc, ts};
                Z<T> a, b;
                if constexpr (RF == DYNAMIC) {
                    const int pair_row = p.pair_index[event_base + event];
                    const long long cell = (static_cast<long long>(pair_row) * p.atom_count + atom) * 4;
                    const float* direction = p.directed_pairs ? p.dpairs : nullptr;
                    a = {num::load<T>(p.pairs, direction, cell), num::load<T>(p.pairs, direction, cell + 1)};
                    b = {num::load<T>(p.pairs, direction, cell + 2), num::load<T>(p.pairs, direction, cell + 3)};
                } else if constexpr (RF == PROFILE) {
                    const int table_row = p.profile_index[event] * p.locations + location;
                    const int last = p.profile_bins - 1;
                    const T scaled = min_(max_(div_(alpha, T(p.profile_step)), T(0.0f)), T(last + 0.0f));
                    const float lower = fminf(floorf(primal(scaled)), last - 1.0f);
                    T h10, h01, h11;
                    const T h00 = hermite_weights(scaled, lower, p.profile_step, h10, h01, h11);
                    const float* knot = p.profile + (table_row * p.profile_bins + static_cast<int>(lower)) * 8;
                    T pair[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        pair[c] = h00 * __ldg(knot + c) + h10 * __ldg(knot + 4 + c) + h01 * __ldg(knot + 8 + c) +
                                  h11 * __ldg(knot + 12 + c);
                    }
                    a = {pair[0], pair[1]};
                    b = {pair[2], pair[3]};
                } else {
                    T hs, hc;
                    sincos_(0.5f * alpha, hs, hc);
                    a = {hc, T(0.0f)};
                    b = {T(0.0f), -hs};
                }
                const Z<T> spun = zmul(b, turn);
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i < m) rotate_spinor(a.r, a.i, spun.r, spun.i, plus[i].r, plus[i].i, minus[i].r, minus[i].i,
                                             z[i].r, z[i].i);
                }
            }
        }
        if (kind == 2 && (act & 32)) {
            const int out = p.output_index[event];
            T recorded_r = 0.0f, recorded_i = 0.0f;
#pragma unroll
            for (int i = 0; i < N; ++i) {
                if (i < m) {
                    recorded_r += plus[i].r;
                    recorded_i += plus[i].i;
                }
            }
            if (out >= 0 && state == 0 && active) {
                T ts, tc;
                sincos_(-event_value(p.phase, p.dphase, event), ts, tc);
                const Z<T> signal = zmul(Z<T>{density * recorded_r, density * recorded_i}, Z<T>{tc, ts});
                const long long written = problem * p.output_count + out;
                if constexpr (DUAL) {
                    p.output_real[written] = signal.r.d;
                    p.output_imag[written] = signal.i.d;
                } else {
                    p.output_real[written] = signal.r;
                    p.output_imag[written] = signal.i;
                }
            }
        }
        if (act & 2) shift();
        if (act & 8) {
#pragma unroll
            for (int i = 0; i < N; ++i) plus[i] = minus[i] = {T(0.0f), T(0.0f)};
        } else if (act & 16) {
            shift();
        }
    }
}


// The adjoint's own buffers: the output's cotangent, the value and
// derivative planes of every gradient it writes, the checkpoints (``kept``
// floats a problem) and the programs' scratch after them.
struct Adjoint {
    const float *grad_real, *grad_imag;
    float *grad_tissue, *dgrad_tissue, *grad_duration, *dgrad_duration, *grad_flip, *dgrad_flip;
    float *grad_phase, *dgrad_phase, *grad_table, *dgrad_table, *grad_pairs, *dgrad_pairs;
    float *trajectory, *scratch;
    long long kept;
    int m0_row, b1_row, b1_phase_row, b0_row, efficiency_row, diffusion_row, velocity_row;
};

template <class T>
__device__ __forceinline__ void add_both(float* value, float* tangent, long long at, T x) {
    if constexpr (num::is_dual<T>::value) {
        atomicAdd(value + at, x.v);
        atomicAdd(tangent + at, x.d);
    } else {
        atomicAdd(value + at, x);
    }
}
// The sum over a group's lanes, in every lane of it.
template <class T>
__device__ __forceinline__ T group_sum(T x, int width) {
    for (int offset = 1; offset < width; offset <<= 1) {
        if constexpr (num::is_dual<T>::value) {
            x = x + T{__shfl_xor_sync(0xffffffffu, x.v, offset), __shfl_xor_sync(0xffffffffu, x.d, offset)};
        } else {
            x += __shfl_xor_sync(0xffffffffu, x, offset);
        }
    }
    return x;
}

// What an interval does to every pool alike at an order, as a function of the
// inputs it depends on: its length, B0, the diffusion and the velocity.
template <class U>
__device__ __forceinline__ void pool_factors(const Params& p, U dt, U b0, U damping_rate, U moved, float order,
                                             U& wout, Z<U>& carried, Z<U>& spin) {
    const float squared = order * order, weight = squared + order + 0.3333333333333333f;
    U damp_z = 1.0f, damp_t = 1.0f;
    wout = 1.0f;
    if (p.diffusing) {
        const U b = damping_rate * dt;
        damp_z = exp_(-squared * b);
        damp_t = exp_(-weight * b);
    }
    if (p.moving) wout = 1.0f - min_(p.washout_scale * abs_(moved) * dt, U(1.0f));
    carried = {wout * damp_t, U(0.0f)};
    spin = {wout * damp_z, U(0.0f)};
    if (p.off_axis || p.moving) {
        const U turn = p.moving ? p.flow_scale * moved * dt : U(0.0f);
        U s, c;
        sincos_(-TWO_PI * b0 * dt - (order + 0.5f) * turn, s, c);
        carried = {wout * damp_t * c, wout * damp_t * s};
        if (p.moving) {
            sincos_(-order * turn, s, c);
            spin = {wout * damp_z * c, wout * damp_z * s};
        }
    }
}

// The floats a problem keeps its checkpoints in, every K-th state with its
// tangent at num::Dual; and the scratch a program of ``threads`` works in --
// the states of the stretch it replays, and the per-lane sums of a table
// row's cotangent -- which is a warp's own and stays in its caches.
__host__ __device__ constexpr long long adjoint_kept_floats(int n, int width, int event_count, int k, bool dual) {
    return static_cast<long long>((event_count + k - 1) / k) * 6 * n * width * (dual ? 2 : 1);
}
__host__ __device__ constexpr long long adjoint_scratch_floats(int n, int m, int threads, int k, bool dual) {
    return static_cast<long long>(threads) * (k * 6 * n * (dual ? 2 : 1) + (n * n + n + 2 * m * m) * (dual ? 3 : 1));
}

template <class T, int N, int RF, int K>
__device__ __forceinline__ void pooled_adjoint_loop(const Params& p, const Adjoint& g, float* scratch) {
    constexpr bool DUAL = num::is_dual<T>::value;
    constexpr int PLANES = 6 * N;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int width = p.width;
    const int state = lane & (width - 1);
    const int group = lane / width;
    const int groups = 32 / width;
    const long long index = (static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + warp) * groups + group;
    const bool active = index < p.problems;
    const long long problem = p.base + (active ? index : 0);
    const bool live = active && state < p.state_count;
    const int atom = static_cast<int>(problem % p.atom_count);
    const int train = static_cast<int>(problem / p.atom_count);
    const int event_base = train * p.event_count;
    const int voxel_at = p.atom_stride ? atom : 0;
    const int location = RF == PROFILE ? atom % p.locations : 0;
    const float order = static_cast<float>(state);
    const int m = p.m;
    const int row_width = N * N + N + 2 * m * m;
    const long long table_width = N + static_cast<long long>(p.rows) * row_width * p.blocks;
    const float* slot = p.table + atom * table_width;
    const float* directions = p.directed_table ? p.dtable + atom * table_width : nullptr;
    const long long sloped_at = static_cast<long long>(p.rows) * row_width;
    const long long n_atoms = p.atom_count;
    const int threads = blockDim.x;

    auto read = [&](const float* values, const float* tangents, long long at, bool on, float identity) -> T {
        if (!on) return T(identity);
        if constexpr (DUAL) {
            return T{__ldg(values + at), __ldg(tangents + at)};
        } else {
            return __ldg(values + at);
        }
    };
    // A table entry, moving with the table's direction and, where the table
    // carries the next block, along the interval's length with it.
    auto entry = [&](long long at, float along, bool sloped) -> T {
        const float value = __ldg(slot + at);
        if constexpr (DUAL) {
            float tangent = 0.0f;
            if (directions != nullptr) tangent += __ldg(directions + at);
            if (sloped && p.blocks > 1) tangent += __ldg(slot + sloped_at + at) * along;
            return T{value, tangent};
        } else {
            return value;
        }
    };
    auto event_value = [&](const float* values, const float* tangents, int event) -> T {
        if constexpr (DUAL) {
            return T{__ldg(values + event_base + event), __ldg(tangents + event_base + event)};
        } else {
            return __ldg(values + event_base + event);
        }
    };
    const T density = read(p.m0, p.dm0, voxel_at, p.density, 1.0f);
    const T voxel_b1 = read(p.b1, p.db1, voxel_at, p.transmit, 1.0f);
    const T voxel_b1_phase = read(p.b1_phase, p.db1_phase, voxel_at, p.off_axis, 0.0f);
    const T voxel_b0 = read(p.b0, p.db0, voxel_at, p.off_axis, 0.0f);
    const T inversion = read(p.efficiency, p.defficiency, voxel_at, p.inverting, 1.0f);
    const T damping_rate = read(p.diffusion, p.ddiffusion, voxel_at, p.diffusing, 0.0f);
    const T moved = read(p.velocity, p.dvelocity, voxel_at, p.moving, 0.0f);

    T equilibrium[N];
    Z<T> plus[N], minus[N], z[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        equilibrium[i] = entry(i, 0.0f, false);
        plus[i] = minus[i] = {T(0.0f), T(0.0f)};
        z[i] = {state == 0 ? equilibrium[i] : T(0.0f), T(0.0f)};
    }
    auto shift = [&]() {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (i >= m) break;
            const T up_r = num::shfl_up(plus[i].r, 1, width), up_i = num::shfl_up(plus[i].i, 1, width);
            const T dn_r = num::shfl_down(minus[i].r, 1, width), dn_i = num::shfl_down(minus[i].i, 1, width);
            const bool keep_up = state > 0 && live, keep_down = state + 1 < p.state_count && live;
            const T pr = keep_up ? up_r : T(0.0f), pi = keep_up ? up_i : T(0.0f);
            const T mr = keep_down ? dn_r : T(0.0f), mi = keep_down ? dn_i : T(0.0f);
            plus[i] = {state == 0 ? mr : pr, state == 0 ? -mi : pi};
            minus[i] = {mr, mi};
        }
    };
    // The interval at the event's row; ``mixed_*`` are what the exchange
    // makes of the state before the factors every pool shares.
    auto relax = [&](T dt, int row, T& wout, Z<T>& carried, Z<T>& spin, Z<T>* mixed_plus, Z<T>* mixed_minus,
                     Z<T>* mixed_z) {
        pool_factors(p, dt, voxel_b0, damping_rate, moved, order, wout, carried, spin);
        const long long row_at = N + static_cast<long long>(row) * row_width;
        const float along = DUAL ? num::tangent(dt) : 0.0f;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            mixed_plus[i] = mixed_minus[i] = {T(0.0f), T(0.0f)};
            if (i >= m) continue;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                if (j >= m) break;
                const long long at = row_at + N * N + N + 2 * (i * m + j);
                const Z<T> x = {entry(at, along, true), entry(at + 1, along, true)};
                const Z<T> a = zmul(x, plus[j]), b = zmul(zconj(x), minus[j]);
                mixed_plus[i] = {mixed_plus[i].r + a.r, mixed_plus[i].i + a.i};
                mixed_minus[i] = {mixed_minus[i].r + b.r, mixed_minus[i].i + b.i};
            }
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            mixed_z[i] = {T(0.0f), T(0.0f)};
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const T l = entry(row_at + i * N + j, along, true);
                mixed_z[i] = {mixed_z[i].r + l * z[j].r, mixed_z[i].i + l * z[j].i};
            }
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (i < m) {
                plus[i] = zmul(carried, mixed_plus[i]);
                minus[i] = zmul(zconj(carried), mixed_minus[i]);
            }
            z[i] = zmul(spin, mixed_z[i]);
            if (state == 0) z[i].r += equilibrium[i] - wout * entry(row_at + N * N + i, along, true);
        }
    };
    // The pulse's flip, phase and transmit field.
    auto pulse_inputs = [&](int event, T& pulse_b1, T& alpha, T& phi, T& nominal, int& shim) {
        pulse_b1 = voxel_b1;
        T pulse_b1_phase = voxel_b1_phase;
        shim = 0;
        if (p.shimmed) {
            shim = p.shim_index[event];
            const long long transmit_at = static_cast<long long>(shim) * n_atoms + atom;
            pulse_b1 = read(p.b1, p.db1, transmit_at, p.transmit, 1.0f);
            pulse_b1_phase = read(p.b1_phase, p.db1_phase, transmit_at, true, 0.0f);
        }
        nominal = event_value(p.flip, p.dflip, event);
        alpha = nominal * pulse_b1;
        phi = event_value(p.phase, p.dphase, event) + pulse_b1_phase;
    };
    // The Cayley-Klein pair a flip forms, turned by the phase.
    auto pair_of = [&](int event, auto alpha, auto phi, auto& a, auto& b) {
        using U = typename std::decay<decltype(alpha)>::type;
        if constexpr (RF == PROFILE) {
            const int table_row = p.profile_index[event] * p.locations + location;
            const int last = p.profile_bins - 1;
            const U scaled = min_(max_(div_(alpha, U(p.profile_step)), U(0.0f)), U(last + 0.0f));
            const float lower = fminf(floorf(primal(scaled)), last - 1.0f);
            U h10, h01, h11;
            const U h00 = hermite_weights(scaled, lower, p.profile_step, h10, h01, h11);
            const float* knot = p.profile + (table_row * p.profile_bins + static_cast<int>(lower)) * 8;
            U pair[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                pair[c] = h00 * __ldg(knot + c) + h10 * __ldg(knot + 4 + c) + h01 * __ldg(knot + 8 + c) +
                          h11 * __ldg(knot + 12 + c);
            }
            a = {pair[0], pair[1]};
            b = {pair[2], pair[3]};
        } else {
            U hs, hc;
            sincos_(0.5f * alpha, hs, hc);
            a = {hc, U(0.0f)};
            b = {U(0.0f), -hs};
        }
        U ts, tc;
        sincos_(-phi, ts, tc);
        b = zmul(b, Z<U>{tc, ts});
    };
    auto dynamic_cell = [&](int event) {
        return (static_cast<long long>(p.pair_index[event_base + event]) * n_atoms + atom) * 4;
    };
    auto saturate = [&](int event, T alpha) {
        if (N > m) {
            const T shape = lineshape_at(p.lineshape, p.rf_frequency[event] - voxel_b0, p.lineshape_bins,
                                         p.lineshape_step);
            const T absorbed = exp_(p.saturation[event] * alpha * alpha * shape);
            z[N - 1] = {absorbed * z[N - 1].r, absorbed * z[N - 1].i};
        }
    };
    auto forward = [&](int event) {
        T wout;
        Z<T> carried, spin, mixed_plus[N], mixed_minus[N], mixed_z[N];
        relax(event_value(p.duration, p.dduration, event), p.pool_index[event_base + event], wout, carried, spin,
              mixed_plus, mixed_minus, mixed_z);
        const unsigned char act = p.action[event];
        if (act & 1) shift();
        if (p.kind[event] == 1) {
            if (act & 4) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i < m) z[i] = {-inversion * z[i].r, -inversion * z[i].i};
                }
            } else {
                T pulse_b1, alpha, phi, nominal;
                int shim;
                pulse_inputs(event, pulse_b1, alpha, phi, nominal, shim);
                saturate(event, alpha);
                Z<T> a, b;
                if constexpr (RF == DYNAMIC) {
                    const long long cell = dynamic_cell(event);
                    const float* direction = p.directed_pairs ? p.dpairs : nullptr;
                    a = {num::load<T>(p.pairs, direction, cell), num::load<T>(p.pairs, direction, cell + 1)};
                    b = {num::load<T>(p.pairs, direction, cell + 2), num::load<T>(p.pairs, direction, cell + 3)};
                    T ts, tc;
                    sincos_(-phi, ts, tc);
                    b = zmul(b, Z<T>{tc, ts});
                } else {
                    pair_of(event, alpha, phi, a, b);
                }
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i < m) rotate_spinor(a.r, a.i, b.r, b.i, plus[i].r, plus[i].i, minus[i].r, minus[i].i, z[i].r,
                                             z[i].i);
                }
            }
        }
        if (act & 2) shift();
        if (act & 8) {
#pragma unroll
            for (int i = 0; i < N; ++i) plus[i] = minus[i] = {T(0.0f), T(0.0f)};
        } else if (act & 16) {
            shift();
        }
    };

    // Checkpoints, in the problem's share of the trajectory buffer: the
    // values, then the tangents.
    const int checkpoints = (p.event_count + K - 1) / K;
    float* kept = g.trajectory + (problem - p.base) * g.kept;
    const long long tangents_at = static_cast<long long>(checkpoints) * PLANES * width;
    auto component = [&](int plane) -> T& {
        const int i = plane / 6, part = plane % 6;
        Z<T>& c = part < 2 ? plus[i] : (part < 4 ? minus[i] : z[i]);
        return (part & 1) ? c.i : c.r;
    };
    auto kept_at = [&](int checkpoint, int plane) {
        return (static_cast<long long>(checkpoint) * PLANES + plane) * width + state;
    };
    T* segment = reinterpret_cast<T*>(scratch);
    float* sums = scratch + K * PLANES * threads * (DUAL ? 2 : 1);
    auto stretch = [&](int k, int plane) -> T& { return segment[(k * PLANES + plane) * threads + threadIdx.x]; };

#pragma unroll 1
    for (int event = 0; event < p.event_count; ++event) {
        if (event % K == 0 && live) {
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) {
                const long long at = kept_at(event / K, plane);
                if constexpr (DUAL) {
                    kept[at] = component(plane).v;
                    kept[tangents_at + at] = component(plane).d;
                } else {
                    kept[at] = component(plane);
                }
            }
        }
        forward(event);
    }

    // ---- the walk back ----
    Z<T> bplus[N], bminus[N], bz[N];
#pragma unroll
    for (int i = 0; i < N; ++i) bplus[i] = bminus[i] = bz[i] = {T(0.0f), T(0.0f)};
    T g_m0 = 0.0f, g_b1 = 0.0f, g_b1_phase = 0.0f, g_b0 = 0.0f, g_efficiency = 0.0f, g_damping = 0.0f,
      g_velocity = 0.0f;
    T g_equilibrium[N];
#pragma unroll
    for (int i = 0; i < N; ++i) g_equilibrium[i] = 0.0f;
    auto shift_back = [&]() {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (i >= m) break;
            const T dn_r = num::shfl_down(bplus[i].r, 1, width), dn_i = num::shfl_down(bplus[i].i, 1, width);
            const T up_r = num::shfl_up(bminus[i].r, 1, width), up_i = num::shfl_up(bminus[i].i, 1, width);
            const T head_r = num::shfl_up(bplus[i].r, 1, width), head_i = num::shfl_up(bplus[i].i, 1, width);
            const bool keep_down = state + 1 < p.state_count && live, keep_up = state > 0 && live;
            bplus[i] = {keep_down ? dn_r : T(0.0f), keep_down ? dn_i : T(0.0f)};
            T mr = keep_up ? up_r : T(0.0f), mi = keep_up ? up_i : T(0.0f);
            if (state == 1 && live) {
                mr = mr + head_r;
                mi = mi - head_i;
            }
            bminus[i] = {mr, mi};
        }
    };
    auto event_gradient = [&](float* value, float* tangent, int event, T lane_value) {
        const T sum = group_sum(lane_value, width);
        if (state == 0 && active) add_both(value, tangent, event_base + event, sum);
    };

    // A table row's cotangent, summed per lane while the row repeats: the
    // value, and at num::Dual its tangent and what the slope block takes.
    const int entries = row_width;
    auto sum_value = [&](int k) -> float& { return sums[(DUAL ? 3 * k : k) * threads + threadIdx.x]; };
    auto sum_tangent = [&](int k) -> float& { return sums[(3 * k + 1) * threads + threadIdx.x]; };
    auto sum_sloped = [&](int k) -> float& { return sums[(3 * k + 2) * threads + threadIdx.x]; };
#pragma unroll 1
    for (int k = 0; k < entries; ++k) {
        sum_value(k) = 0.0f;
        if constexpr (DUAL) sum_tangent(k) = sum_sloped(k) = 0.0f;
    }
    float* grad_row = g.grad_table + problem * table_width;
    float* curve_row = g.dgrad_table + problem * table_width;
    int summed_row = -1;
    auto flush = [&]() {
        if (!__any_sync(0xffffffffu, summed_row >= 0)) return;
        const bool mine = summed_row >= 0 && state == 0 && active;
        const long long row_at = N + static_cast<long long>(summed_row < 0 ? 0 : summed_row) * row_width;
#pragma unroll 1
        for (int k = 0; k < entries; ++k) {
            const float v = group_sum(sum_value(k), width);
            sum_value(k) = 0.0f;
            if (mine) grad_row[row_at + k] += v;
            if constexpr (DUAL) {
                const float d = group_sum(sum_tangent(k), width);
                const float sd = group_sum(sum_sloped(k), width);
                sum_tangent(k) = sum_sloped(k) = 0.0f;
                if (mine) {
                    curve_row[row_at + k] += d;
                    if (p.blocks > 1) curve_row[sloped_at + row_at + k] += sd;
                }
            }
        }
    };

#pragma unroll 1
    for (int checkpoint = checkpoints - 1; checkpoint >= 0; --checkpoint) {
        const int start = checkpoint * K;
        const int stop = min(start + K, p.event_count);
#pragma unroll
        for (int plane = 0; plane < PLANES; ++plane) {
            const long long at = kept_at(checkpoint, plane);
            if constexpr (DUAL) {
                component(plane) = live ? T{kept[at], kept[tangents_at + at]} : T(0.0f);
            } else {
                component(plane) = live ? kept[at] : 0.0f;
            }
        }
#pragma unroll 1
        for (int event = start; event < stop; ++event) {
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) stretch(event - start, plane) = component(plane);
            if (event + 1 < stop) forward(event);
        }
#pragma unroll 1
        for (int event = stop - 1; event >= start; --event) {
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) component(plane) = stretch(event - start, plane);
            const unsigned char act = p.action[event];
            const int kind = p.kind[event];
            const T dt = event_value(p.duration, p.dduration, event);
            const int row = p.pool_index[event_base + event];
            T wout;
            Z<T> carried, spin, mixed_plus[N], mixed_minus[N], mixed_z[N];
            relax(dt, row, wout, carried, spin, mixed_plus, mixed_minus, mixed_z);
            if (act & 1) shift();

            // The trailing spoil or shift, then the shift before it.
            if (act & 8) {
#pragma unroll
                for (int i = 0; i < N; ++i) bplus[i] = bminus[i] = {T(0.0f), T(0.0f)};
            } else if (act & 16) {
                shift_back();
            }
            if (act & 2) shift_back();

            // A sample records the stage the event reached.
            if (kind == 2 && (act & 32) && p.output_index[event] >= 0) {
                T phase_gradient = 0.0f;
                if (state == 0 && active) {
                    const long long at = problem * p.output_count + p.output_index[event];
                    const Z<T> seed = {T(g.grad_real[at]), T(g.grad_imag[at])};
                    T ts, tc;
                    sincos_(-event_value(p.phase, p.dphase, event), ts, tc);
                    Z<T> recorded = {T(0.0f), T(0.0f)};
#pragma unroll
                    for (int i = 0; i < N; ++i) {
                        if (i < m) recorded = {recorded.r + plus[i].r, recorded.i + plus[i].i};
                    }
                    const Z<T> demodulated = zmul(recorded, Z<T>{tc, ts});
                    g_m0 += seed.r * demodulated.r + seed.i * demodulated.i;
                    phase_gradient = density * (seed.r * demodulated.i - seed.i * demodulated.r);
                    const Z<T> weighted = zmul(Z<T>{density * tc, -(density * ts)}, seed);
#pragma unroll
                    for (int i = 0; i < N; ++i) {
                        if (i < m) bplus[i] = {bplus[i].r + weighted.r, bplus[i].i + weighted.i};
                    }
                }
                event_gradient(g.grad_phase, g.dgrad_phase, event, phase_gradient);
            }

            if (kind == 1 && (act & 4)) {
                T taken = 0.0f;
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i < m) {
                        taken += bz[i].r * z[i].r + bz[i].i * z[i].i;
                        bz[i] = {-inversion * bz[i].r, -inversion * bz[i].i};
                    }
                }
                g_efficiency -= taken;
            } else if (kind == 1) {
                T pulse_b1, alpha, phi, nominal;
                int shim;
                pulse_inputs(event, pulse_b1, alpha, phi, nominal, shim);
                // Each product of a state with its cotangent, over the
                // exchanging pools, so the rotation's derivatives are taken
                // against nine numbers.
                Z<T> met[9];
#pragma unroll
                for (int k = 0; k < 9; ++k) met[k] = {T(0.0f), T(0.0f)};
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i >= m) continue;
                    const Z<T> S[3] = {plus[i], minus[i], z[i]};
                    const Z<T> L[3] = {bplus[i], bminus[i], bz[i]};
#pragma unroll
                    for (int r = 0; r < 3; ++r) {
#pragma unroll
                        for (int c = 0; c < 3; ++c) {
                            const Z<T> term = zmul(S[c], zconj(L[r]));
                            met[3 * r + c] = {met[3 * r + c].r + term.r, met[3 * r + c].i + term.i};
                        }
                    }
                }
                auto against = [&](const auto* dR) {
                    using U = typename std::decay<decltype(dR[0].r)>::type;
                    U sum = 0.0f;
#pragma unroll
                    for (int k = 0; k < 9; ++k) sum = sum + (dR[k].r * met[k].r - dR[k].i * met[k].i);
                    return sum;
                };
                T g_alpha = 0.0f, g_phi = 0.0f;
                Z<T> R[9];
                if constexpr (RF == DYNAMIC) {
                    using V = num::Multi<5, T>;
                    const long long cell = dynamic_cell(event);
                    const float* direction = p.directed_pairs ? p.dpairs : nullptr;
                    const Z<V> a = {V(num::load<T>(p.pairs, direction, cell), 1),
                                    V(num::load<T>(p.pairs, direction, cell + 1), 2)};
                    Z<V> b = {V(num::load<T>(p.pairs, direction, cell + 2), 3),
                              V(num::load<T>(p.pairs, direction, cell + 3), 4)};
                    V ts, tc;
                    sincos_(-V(phi, 0), ts, tc);
                    b = zmul(b, Z<V>{tc, ts});
                    epg_vjp::Cx<V> dR[9];
                    epg_vjp::spinor_rotation(epg_vjp::Cx<V>{a.r, a.i}, epg_vjp::Cx<V>{b.r, b.i}, dR);
                    const V got = against(dR);
                    g_phi = got.d[0];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const T total = group_sum(got.d[1 + c], width);
                        if (state == 0 && active) add_both(g.grad_pairs, g.dgrad_pairs, cell + c, total);
                    }
#pragma unroll
                    for (int k = 0; k < 9; ++k) R[k] = {dR[k].r.v, dR[k].i.v};
                } else {
                    using V = num::Multi<2, T>;
                    Z<V> a, b;
                    pair_of(event, V(alpha, 0), V(phi, 1), a, b);
                    epg_vjp::Cx<V> dR[9];
                    epg_vjp::spinor_rotation(epg_vjp::Cx<V>{a.r, a.i}, epg_vjp::Cx<V>{b.r, b.i}, dR);
                    const V got = against(dR);
                    g_alpha = got.d[0];
                    g_phi = got.d[1];
#pragma unroll
                    for (int k = 0; k < 9; ++k) R[k] = {dR[k].r.v, dR[k].i.v};
                }
                // The semisolid pool is saturated before the rotation, which
                // leaves it alone.
                if (N > m) {
                    using V = num::Multi<2, T>;
                    const V shape = lineshape_at(p.lineshape, V(p.rf_frequency[event]) - V(voxel_b0, 1),
                                                 p.lineshape_bins, p.lineshape_step);
                    const V absorbed = exp_(p.saturation[event] * V(alpha, 0) * V(alpha, 0) * shape);
                    const T taken = bz[N - 1].r * z[N - 1].r + bz[N - 1].i * z[N - 1].i;
                    g_alpha += absorbed.d[0] * taken;
                    g_b0 += absorbed.d[1] * taken;
                    bz[N - 1] = {absorbed.v * bz[N - 1].r, absorbed.v * bz[N - 1].i};
                }
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i >= m) continue;
                    const Z<T> L[3] = {bplus[i], bminus[i], bz[i]};
                    Z<T> back[3];
#pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        back[c] = {T(0.0f), T(0.0f)};
#pragma unroll
                        for (int r = 0; r < 3; ++r) {
                            const Z<T> term = zmul(zconj(R[3 * r + c]), L[r]);
                            back[c] = {back[c].r + term.r, back[c].i + term.i};
                        }
                    }
                    bplus[i] = back[0];
                    bminus[i] = back[1];
                    bz[i] = back[2];
                }
                event_gradient(g.grad_flip, g.dgrad_flip, event, g_alpha * pulse_b1);
                event_gradient(g.grad_phase, g.dgrad_phase, event, g_phi);
                if (p.shimmed) {
                    const T b1_total = group_sum(g_alpha * nominal, width);
                    const T phase_total = group_sum(g_phi, width);
                    if (state == 0 && active) {
                        add_both(g.grad_tissue, g.dgrad_tissue, (g.b1_row + shim) * n_atoms + atom, b1_total);
                        add_both(g.grad_tissue, g.dgrad_tissue, (g.b1_phase_row + shim) * n_atoms + atom,
                                 phase_total);
                    }
                } else {
                    g_b1 += g_alpha * nominal;
                    g_b1_phase += g_phi;
                }
            }
            if (act & 1) shift_back();

            // The interval, from the state it met: one pass over the row
            // takes the row's cotangent, sends the state's back through it
            // and rebuilds what the exchange made, which the shared factors'
            // derivatives are taken against.
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) component(plane) = stretch(event - start, plane);
            if (__any_sync(0xffffffffu, row != summed_row)) {
                flush();
                summed_row = row;
            }
            const long long row_at = N + static_cast<long long>(row) * row_width;
            const float along = DUAL ? num::tangent(dt) : 0.0f;
            const bool factored = p.off_axis || p.moving || p.diffusing;
            T table_duration = 0.0f;
            auto take = [&](int k, T cotangent) {
                if constexpr (DUAL) {
                    sum_value(k) += cotangent.v;
                    sum_tangent(k) += cotangent.d;
                    if (p.blocks > 1) {
                        sum_sloped(k) += cotangent.v * along;
                        const long long at = sloped_at + row_at + k;
                        float tangent = __ldg(slot + sloped_at + at) * along;
                        if (directions != nullptr) tangent += __ldg(directions + at);
                        table_duration = table_duration + cotangent * T{__ldg(slot + at), tangent};
                    }
                } else {
                    sum_value(k) += cotangent;
                    if (p.blocks > 1) table_duration += cotangent * __ldg(slot + sloped_at + row_at + k);
                }
            };
            Z<T> back_plus[N], back_minus[N], back_z[N];
#pragma unroll
            for (int j = 0; j < N; ++j) back_plus[j] = back_minus[j] = back_z[j] = {T(0.0f), T(0.0f)};
            Z<T> across_plus = {T(0.0f), T(0.0f)}, across_minus = {T(0.0f), T(0.0f)}, along_z = {T(0.0f), T(0.0f)};
            T restoring = 0.0f;
#pragma unroll
            for (int i = 0; i < N; ++i) {
                const Z<T> lp = zmul(zconj(carried), bplus[i]), lm = zmul(carried, bminus[i]);
                const Z<T> lz = zmul(zconj(spin), bz[i]);
                Z<T> mp = {T(0.0f), T(0.0f)}, mm = {T(0.0f), T(0.0f)}, mz = {T(0.0f), T(0.0f)};
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    if (i < m && j < m) {
                        const int k = N * N + N + 2 * (i * m + j);
                        const Z<T> x = {entry(row_at + k, along, true), entry(row_at + k + 1, along, true)};
                        if (factored) {
                            const Z<T> a = zmul(x, plus[j]), b = zmul(zconj(x), minus[j]);
                            mp = {mp.r + a.r, mp.i + a.i};
                            mm = {mm.r + b.r, mm.i + b.i};
                        }
                        const Z<T> a = zmul(zconj(x), lp), b = zmul(x, lm);
                        back_plus[j] = {back_plus[j].r + a.r, back_plus[j].i + a.i};
                        back_minus[j] = {back_minus[j].r + b.r, back_minus[j].i + b.i};
                        const Z<T> gp = zmul(lp, zconj(plus[j])), gm = zmul(zconj(lm), minus[j]);
                        take(k, gp.r + gm.r);
                        take(k + 1, gp.i + gm.i);
                    }
                    const T l = entry(row_at + i * N + j, along, true);
                    if (factored) mz = {mz.r + l * z[j].r, mz.i + l * z[j].i};
                    back_z[j] = {back_z[j].r + l * lz.r, back_z[j].i + l * lz.i};
                    take(i * N + j, lz.r * z[j].r + lz.i * z[j].i);
                }
                if (state == 0) {
                    g_equilibrium[i] += bz[i].r;
                    take(N * N + i, -(wout * bz[i].r));
                    if (factored) restoring += bz[i].r * entry(row_at + N * N + i, along, true);
                }
                if (factored) {
                    if (i < m) {
                        const Z<T> a = zmul(mp, zconj(bplus[i])), b = zmul(mm, zconj(bminus[i]));
                        across_plus = {across_plus.r + a.r, across_plus.i + a.i};
                        across_minus = {across_minus.r + b.r, across_minus.i + b.i};
                    }
                    const Z<T> c = zmul(mz, zconj(bz[i]));
                    along_z = {along_z.r + c.r, along_z.i + c.i};
                }
            }
            T duration_gradient = table_duration;
            if (factored) {
                using V = num::Multi<4, T>;
                V w;
                Z<V> c, sp;
                pool_factors(p, V(dt, 0), V(voxel_b0, 1), V(damping_rate, 2), V(moved, 3), order, w, c, sp);
                const V got = (c.r * across_plus.r - c.i * across_plus.i) +
                              (c.r * across_minus.r + c.i * across_minus.i) + (sp.r * along_z.r - sp.i * along_z.i) -
                              w * restoring;
                duration_gradient = duration_gradient + got.d[0];
                g_b0 += got.d[1];
                g_damping += got.d[2];
                g_velocity += got.d[3];
            }
            event_gradient(g.grad_duration, g.dgrad_duration, event, duration_gradient);
#pragma unroll
            for (int i = 0; i < N; ++i) {
                bplus[i] = back_plus[i];
                bminus[i] = back_minus[i];
                bz[i] = back_z[i];
            }
        }
    }
    flush();
    // The equilibrium is also where every pool starts.
#pragma unroll
    for (int i = 0; i < N; ++i) {
        if (state == 0) g_equilibrium[i] += bz[i].r;
        const T total = group_sum(g_equilibrium[i], width);
        if (state == 0 && active) {
            if constexpr (DUAL) {
                grad_row[i] += total.v;
                curve_row[i] += total.d;
            } else {
                grad_row[i] += total;
            }
        }
    }
    auto store_row = [&](int row, T lane_value) {
        const T total = group_sum(lane_value, width);
        if (state == 0 && active) add_both(g.grad_tissue, g.dgrad_tissue, row * n_atoms + atom, total);
    };
    store_row(g.m0_row, g_m0);
    if (!p.shimmed) {
        store_row(g.b1_row, g_b1);
        store_row(g.b1_phase_row, g_b1_phase);
    }
    store_row(g.b0_row, g_b0);
    store_row(g.efficiency_row, g_efficiency);
    store_row(g.diffusion_row, g_damping);
    store_row(g.velocity_row, g_velocity);
}

}  // namespace epg_pooled
