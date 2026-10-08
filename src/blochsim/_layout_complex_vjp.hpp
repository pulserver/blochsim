// The complex EPG adjoint over a number type: the vector-Jacobian product at
// float, its derivative along a direction at num::Dual.
//
// The forward sweep keeps the state every K events; the reverse sweep replays
// each stretch of K events from its checkpoint into shared memory and walks it
// back. An interval's operator repeats while its length does, so the walk
// back sums, per lane, the products of each state with the cotangent the
// operator met it with, and contracts that sum against the operator's
// derivatives -- taken along every tissue input at once by num::Multi -- only
// when the operator changes. Pulses and samples are explicit complex maps
// and contract against their own derivatives at once.
#pragma once
#include "_layout_complex.hpp"

namespace epg_vjp {

using namespace epg;
using num::Multi;

struct Params {
    epg::Params f;
    const float *grad_output_real, *grad_output_imag;
    // Value and, for a dual sweep, direction of each gradient.
    float *grad_tissue, *grad_flip, *grad_phase, *grad_duration, *grad_pair;
    float *grad_tissue_t, *grad_flip_t, *grad_phase_t, *grad_duration_t, *grad_pair_t;
    float *trajectory_r, *trajectory_i, *trajectory_tr, *trajectory_ti;
    int problem_base, problem_end, shim_rows;
    // How a three-pool interval is formed: an epg::Mode.
    int mode;
};

template <class T>
struct Cx {
    T r, i;
};
template <class T>
__device__ __forceinline__ Cx<T> cmul(const Cx<T>& a, const Cx<T>& b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
template <class T>
__device__ __forceinline__ Cx<T> cconj(const Cx<T>& a) {
    return {a.r, -a.i};
}
// Re(conj(a) b).
template <class T>
__device__ __forceinline__ T rdot(const Cx<T>& a, const Cx<T>& b) {
    return a.r * b.r + a.i * b.i;
}

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
template <class T>
__device__ __forceinline__ T lanes_sum(T x, int from, int to) {
    for (int offset = from; offset < to; offset <<= 1) x += shfl_xor(x, offset);
    return x;
}
template <class T>
__device__ __forceinline__ void put(float* values, float* tangents, long long at, T x) {
    if constexpr (num::is_dual<T>::value) {
        values[at] = x.v;
        tangents[at] = x.d;
    } else {
        values[at] = x;
    }
}
template <class T>
__device__ __forceinline__ T get(const float* values, const float* tangents, long long at) {
    if constexpr (num::is_dual<T>::value) {
        return T{values[at], tangents[at]};
    } else {
        return values[at];
    }
}

// What an interval reads of the launch besides the tissue and its length.
struct Geometry {
    float flow_scale, washout_scale;
    const float* pool_table;
    int atom_count;
};

// Tissue directions an interval's operator is differentiated along.
enum Direction { D_R1 = 0, D_R2, D_DIFFUSION, D_B0, D_VELOCITY, D_FB, D_XB, D_R1B, D_R2B, D_SHIFT, D_FC, D_XC, D_R1C };
template <int POOLS>
constexpr int DIRECTIONS = POOLS == 3 ? 13 : (POOLS == 2 ? 10 : (POOLS == 1 ? 8 : 5));

// One interval's operator for one problem at this lane's order: transverse
// entries (one, or a 2x2 for an exchanging pool), longitudinal ones (the
// exchange matrix times the flow and damping factor), and recoveries.
template <class T, int POOLS>
struct Interval {
    static constexpr int NT = POOLS >= 2 ? 4 : 1, NL = POOLS == 3 ? 9 : (POOLS ? 4 : 1);
    static constexpr int NG = POOLS == 3 ? 3 : (POOLS ? 2 : 1), N = POOLS == 3 ? 3 : (POOLS ? 2 : 1);
    Cx<T> t[NT], l[NL];
    T g[NG];
};

// What one problem is made of, as the interval reads it.
template <class T>
struct Inputs {
    T r1, r2, diffusion, b0, velocity, fb, xb, r1b, r2b, shift, fc, xc, r1c;
};

// Whether an interval keeps the three pools' eigenvalues within
// NARROW_SPREAD (4) of each other: the shifted roots sum to zero, so the sum
// of their squares is -2 * minors.
__device__ __forceinline__ bool three_pool_narrow(float r1_free, float r1_b, float r1_c, float exchange_b,
                                                  float exchange_c, float fraction_b, float fraction_c, float dt) {
    const float free = 1.0f - fraction_b - fraction_c;
    const float kab = exchange_b * fraction_b, kba = exchange_b * free;
    const float kac = exchange_c * fraction_c, kca = exchange_c * free;
    const float a00 = (-kab - kac - r1_free) * dt, a01 = kba * dt, a02 = kca * dt;
    const float a10 = kab * dt, a11 = (-kba - r1_b) * dt, a20 = kac * dt, a22 = (-kca - r1_c) * dt;
    const float third = (a00 + a11 + a22) * (1.0f / 3.0f);
    const float s00 = a00 - third, s11 = a11 - third, s22 = a22 - third;
    const float minors = s00 * s11 - a01 * a10 + s00 * s22 - a02 * a20 + s11 * s22;
    return -2.0f * minors < 16.0f;
}

// The switches are read at run time: an interval is formed only when it
// changes, and one body then serves all of them.
template <class T, int POOLS, int MODE>
__device__ __forceinline__ void interval(const Geometry& p, bool MOVING, bool DIFFUSING, bool OFF_AXIS, T dt,
                                         const Inputs<T>& in, float order, int row, int atom, Interval<T, POOLS>& out) {
    T wout = 1.0f, turn = 0.0f;
    if (MOVING) {
        wout = 1.0f - min_(abs_(in.velocity) * p.washout_scale * dt, T(1.0f));
        turn = in.velocity * p.flow_scale * dt;
    }
    T damp_z = 1.0f, damp_t = 1.0f;
    if (DIFFUSING) {
        const T b = in.diffusion * dt;
        const float sq = order * order;
        damp_z = exp_(-b * sq);
        damp_t = exp_(-b * (sq + order + 0.3333333333333333f));
    }
    T oc = 1.0f, os = 0.0f;
    if (MOVING || OFF_AXIS) {
        T phase = -(order + 0.5f) * turn;
        if (OFF_AXIS) phase = phase - TWO_PI * in.b0 * dt;
        sincos_(phase, os, oc);
    }
    if constexpr (POOLS < 2) {
        const T e2 = exp_(-in.r2 * dt) * wout * damp_t;
        out.t[0] = {e2 * oc, e2 * os};
    } else {
        T x[8];
        const T free = 1.0f - in.fb - (POOLS == 3 ? in.fc : T(0.0f));
        transverse_step(in.r2, in.r2b, in.xb, in.fb, free, in.shift, dt, wout, x);
        const T rr = damp_t * oc, ri = damp_t * os;
#pragma unroll
        for (int k = 0; k < 4; ++k) out.t[k] = {rr * x[2 * k] - ri * x[2 * k + 1], rr * x[2 * k + 1] + ri * x[2 * k]};
    }
    T e[Interval<T, POOLS>::NL];
    if constexpr (POOLS == 0) {
        e[0] = exp_(-in.r1 * dt) * wout;
        out.g[0] = 1.0f - e[0];
    } else if constexpr (POOLS == 3) {
        if constexpr (MODE == TABLE && !num::is_dual<T>::value) {
            const float* base = p.pool_table + static_cast<long long>(row) * (9 * p.atom_count) + atom;
#pragma unroll
            for (int k = 0; k < 9; ++k) e[k] = wout * __ldg(base + k * p.atom_count);
            const T fa = 1.0f - in.fb - in.fc;
            out.g[0] = fa - (e[0] * fa + e[1] * in.fb + e[2] * in.fc);
            out.g[1] = in.fb - (e[3] * fa + e[4] * in.fb + e[5] * in.fc);
            out.g[2] = in.fc - (e[6] * fa + e[7] * in.fb + e[8] * in.fc);
        } else if constexpr (MODE == NARROW) {
            three_pool_step<true>(in.r1, in.r1b, in.r1c, in.xb, in.xc, in.fb, in.fc, dt, wout, e, out.g);
        } else {
            // The float series holds the operator to float32 while the
            // eigenvalues stay within NARROW_SPREAD of each other; double is
            // for an interval that spreads them further.
            if (three_pool_narrow(primal(in.r1), primal(in.r1b), primal(in.r1c), primal(in.xb), primal(in.xc),
                                  primal(in.fb), primal(in.fc), primal(dt))) {
                three_pool_step<true>(in.r1, in.r1b, in.r1c, in.xb, in.xc, in.fb, in.fc, dt, wout, e, out.g);
            } else {
                three_pool_step<false>(in.r1, in.r1b, in.r1c, in.xb, in.xc, in.fb, in.fc, dt, wout, e, out.g);
            }
        }
    } else {
        two_pool_step(in.r1, in.r1b, in.xb, in.fb, dt, wout, e, out.g);
    }
    T fr = damp_z, fi = 0.0f;
    if (MOVING) {
        T ts, tc;
        sincos_(-order * turn, ts, tc);
        fr = damp_z * tc;
        fi = damp_z * ts;
    }
#pragma unroll
    for (int k = 0; k < Interval<T, POOLS>::NL; ++k) out.l[k] = {fr * e[k], fi * e[k]};
}

// One interval for the launch's switches.
template <int POOLS, int MODE, class U>
__device__ __forceinline__ void form_interval(const Geometry& p, int relax_code, U dt, const Inputs<U>& in,
                                              float order, int row, int atom, Interval<U, POOLS>& out) {
    interval<U, POOLS, MODE>(p, (relax_code & 4) != 0, (relax_code & 2) != 0, (relax_code & 1) != 0, dt, in, order, row,
                             atom, out);
}

// Products of state and cotangent an operator met, summed while it repeats.
template <class T, int POOLS>
struct Met {
    Cx<T> t[Interval<T, POOLS>::NT], l[Interval<T, POOLS>::NL];
    T g[Interval<T, POOLS>::NG];
};

// Re of every entry's derivative times what it met: the interval's share of
// a gradient.
template <class U, class T, int POOLS>
__device__ __forceinline__ U contract(const Interval<U, POOLS>& derivative, const Met<T, POOLS>& met) {
    U sum = 0.0f;
#pragma unroll
    for (int k = 0; k < Interval<T, POOLS>::NT; ++k) sum += derivative.t[k].r * met.t[k].r - derivative.t[k].i * met.t[k].i;
#pragma unroll
    for (int k = 0; k < Interval<T, POOLS>::NL; ++k) sum += derivative.l[k].r * met.l[k].r - derivative.l[k].i * met.l[k].i;
#pragma unroll
    for (int k = 0; k < Interval<T, POOLS>::NG; ++k) sum += derivative.g[k] * met.g[k];
    return sum;
}

// The derivative of each entry along direction ``k`` of a multi-valued interval.
template <int K, class T, int POOLS>
__device__ __forceinline__ Interval<T, POOLS> along(const Interval<Multi<K, T>, POOLS>& m, int k) {
    Interval<T, POOLS> out;
#pragma unroll
    for (int j = 0; j < Interval<T, POOLS>::NT; ++j) out.t[j] = {m.t[j].r.d[k], m.t[j].i.d[k]};
#pragma unroll
    for (int j = 0; j < Interval<T, POOLS>::NL; ++j) out.l[j] = {m.l[j].r.d[k], m.l[j].i.d[k]};
#pragma unroll
    for (int j = 0; j < Interval<T, POOLS>::NG; ++j) out.g[j] = m.g[j].d[k];
    return out;
}
template <int K, class T, int POOLS>
__device__ __forceinline__ Interval<T, POOLS> values(const Interval<Multi<K, T>, POOLS>& m) {
    Interval<T, POOLS> out;
#pragma unroll
    for (int j = 0; j < Interval<T, POOLS>::NT; ++j) out.t[j] = {m.t[j].r.v, m.t[j].i.v};
#pragma unroll
    for (int j = 0; j < Interval<T, POOLS>::NL; ++j) out.l[j] = {m.l[j].r.v, m.l[j].i.v};
#pragma unroll
    for (int j = 0; j < Interval<T, POOLS>::NG; ++j) out.g[j] = m.g[j].v;
    return out;
}

// A number with its direction dropped, and the direction alone.
template <class T>
__device__ __forceinline__ T flat(T x) {
    if constexpr (num::is_dual<T>::value) return T(x.v);
    else return x;
}
template <class T>
__device__ __forceinline__ float direction(T x) {
    if constexpr (num::is_dual<T>::value) return x.d;
    else return 0.0f;
}
template <class T>
__device__ __forceinline__ Inputs<T> flat(const Inputs<T>& in) {
    return {flat(in.r1), flat(in.r2), flat(in.diffusion), flat(in.b0), flat(in.velocity), flat(in.fb), flat(in.xb),
            flat(in.r1b), flat(in.r2b), flat(in.shift), flat(in.fc), flat(in.xc), flat(in.r1c)};
}
// An operator moved along its length by ``d``: its direction gains d times
// the slope's value. A float operator has no direction to move.
template <class T, int POOLS>
__device__ __forceinline__ Interval<T, POOLS> moved(const Interval<T, POOLS>& a, const Interval<T, POOLS>& slope, float d) {
    Interval<T, POOLS> out = a;
    if (d == 0.0f) return out;
    if constexpr (num::is_dual<T>::value) {
#pragma unroll
        for (int k = 0; k < Interval<T, POOLS>::NT; ++k) {
            out.t[k].r.d += d * slope.t[k].r.v;
            out.t[k].i.d += d * slope.t[k].i.v;
        }
#pragma unroll
        for (int k = 0; k < Interval<T, POOLS>::NL; ++k) {
            out.l[k].r.d += d * slope.l[k].r.v;
            out.l[k].i.d += d * slope.l[k].i.v;
        }
#pragma unroll
        for (int k = 0; k < Interval<T, POOLS>::NG; ++k) out.g[k].d += d * slope.g[k].v;
    }
    return out;
}

// What an interval contributes to each tissue direction's gradient.
template <class T, int POOLS>
struct Gradients {
    T g[DIRECTIONS<POOLS>];
};

// Out of line: an interval changes rarely, and one copy then serves every
// kernel of the layout instead of one inlined per kernel. Everything goes in
// and comes out by value, so nothing in the caller is addressed.
template <class T, int POOLS, int MODE>
__device__ __noinline__ Gradients<T, POOLS> contract_interval(Geometry g, int relax_code, T dt, Inputs<T> in, float order,
                                                              int row, int atom, Met<T, POOLS> met) {
    // Directions taken per formation of the operator. The multi-valued
    // operator is held on the stack, which the driver backs for every thread
    // a card can hold: where its entries are wide -- three pools, or a dual
    // -- one direction at a time keeps that stack small, and is no slower.
    constexpr int KD = DIRECTIONS<POOLS>;
    constexpr int CHUNK = num::is_dual<T>::value || POOLS == 3 ? 1 : 4;
    Gradients<T, POOLS> out;
#pragma unroll
    for (int k = 0; k < KD; ++k) out.g[k] = 0.0f;
#pragma unroll 1
    for (int c = 0; c * CHUNK < KD; ++c) {
        using V = Multi<CHUNK, T>;
        auto seed = [&](T value, int direction) {
            const int local = direction - c * CHUNK;
            return V(value, direction < KD && local >= 0 && local < CHUNK ? local : -1);
        };
        Inputs<V> seeded;
        seeded.r1 = seed(in.r1, D_R1);
        seeded.r2 = seed(in.r2, D_R2);
        seeded.diffusion = seed(in.diffusion, D_DIFFUSION);
        seeded.b0 = seed(in.b0, D_B0);
        seeded.velocity = seed(in.velocity, D_VELOCITY);
        seeded.fb = seed(in.fb, D_FB);
        seeded.xb = seed(in.xb, D_XB);
        seeded.r1b = seed(in.r1b, D_R1B);
        seeded.r2b = seed(in.r2b, D_R2B);
        seeded.shift = seed(in.shift, D_SHIFT);
        seeded.fc = seed(in.fc, D_FC);
        seeded.xc = seed(in.xc, D_XC);
        seeded.r1c = seed(in.r1c, D_R1C);
        Interval<V, POOLS> m;
        form_interval<POOLS, MODE>(g, relax_code, V(dt, -1), seeded, order, row, atom, m);
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            const T got = contract(along(m, k), met);
#pragma unroll
            for (int j = 0; j < KD; ++j) {
                if (j == c * CHUNK + k) out.g[j] = got;
            }
        }
    }
    return out;
}

// An interval's operator and its derivative along its own length.
template <class T, int POOLS>
struct Opened {
    Interval<T, POOLS> value, slope;
};
template <class T, int POOLS, int MODE>
__device__ __noinline__ Opened<T, POOLS> open_interval_at(Geometry g, int relax_code, T dt, Inputs<T> in, float order,
                                                          int row, int atom) {
    using V = Multi<1, T>;
    Inputs<V> seeded;
    seeded.r1 = V(in.r1, -1);
    seeded.r2 = V(in.r2, -1);
    seeded.diffusion = V(in.diffusion, -1);
    seeded.b0 = V(in.b0, -1);
    seeded.velocity = V(in.velocity, -1);
    seeded.fb = V(in.fb, -1);
    seeded.xb = V(in.xb, -1);
    seeded.r1b = V(in.r1b, -1);
    seeded.r2b = V(in.r2b, -1);
    seeded.shift = V(in.shift, -1);
    seeded.fc = V(in.fc, -1);
    seeded.xc = V(in.xc, -1);
    seeded.r1c = V(in.r1c, -1);
    Interval<V, POOLS> m;
    form_interval<POOLS, MODE>(g, relax_code, V(dt, 0), seeded, order, row, atom, m);
    return {values(m), along(m, 0)};
}

// An interval's operator alone, for the forward sweep.
template <class T, int POOLS, int MODE>
__device__ __noinline__ Interval<T, POOLS> interval_at(Geometry g, int relax_code, T dt, Inputs<T> in, float order, int row,
                                                       int atom) {
    Interval<T, POOLS> out;
    form_interval<POOLS, MODE>(g, relax_code, dt, in, order, row, atom, out);
    return out;
}

// The three out-of-line interval functions with the three-pool mode read at
// run time: only an interval's change pays for the choice.
template <class T, int POOLS>
__device__ __forceinline__ Interval<T, POOLS> interval_in(int mode, Geometry g, int relax_code, T dt, const Inputs<T>& in,
                                                          float order, int row, int atom) {
    if constexpr (POOLS == 3) {
        if (mode == TABLE) return interval_at<T, POOLS, TABLE>(g, relax_code, dt, in, order, row, atom);
        if (mode == ROOTS) return interval_at<T, POOLS, ROOTS>(g, relax_code, dt, in, order, row, atom);
    }
    return interval_at<T, POOLS, NARROW>(g, relax_code, dt, in, order, row, atom);
}
template <class T, int POOLS>
__device__ __forceinline__ Opened<T, POOLS> open_in(int mode, Geometry g, int relax_code, T dt, const Inputs<T>& in,
                                                    float order, int row, int atom) {
    if constexpr (POOLS == 3) {
        if (mode == TABLE) return open_interval_at<T, POOLS, TABLE>(g, relax_code, dt, in, order, row, atom);
        if (mode == ROOTS) return open_interval_at<T, POOLS, ROOTS>(g, relax_code, dt, in, order, row, atom);
    }
    return open_interval_at<T, POOLS, NARROW>(g, relax_code, dt, in, order, row, atom);
}
template <class T, int POOLS>
__device__ __forceinline__ Gradients<T, POOLS> contract_in(int mode, Geometry g, int relax_code, T dt, const Inputs<T>& in,
                                                           float order, int row, int atom, const Met<T, POOLS>& met) {
    if constexpr (POOLS == 3) {
        if (mode == TABLE) return contract_interval<T, POOLS, TABLE>(g, relax_code, dt, in, order, row, atom, met);
        if (mode == ROOTS) return contract_interval<T, POOLS, ROOTS>(g, relax_code, dt, in, order, row, atom, met);
    }
    return contract_interval<T, POOLS, NARROW>(g, relax_code, dt, in, order, row, atom, met);
}

// The pulse as a complex 3x3 on (F+, F-, Z): a hard pulse of flip ``alpha``
// at phase (c, s), or the rotation of a Cayley-Klein pair (a, b).
template <class T>
__device__ __forceinline__ void hard_rotation(T alpha, T cphi, T sphi, Cx<T>* R) {
    T s, c;
    sincos_(alpha, s, c);
    const T chs = 0.5f * (1.0f + c), shs = 0.5f * (1.0f - c), hs = 0.5f * s;
    const T c2 = cphi * cphi - sphi * sphi, s2 = 2.0f * sphi * cphi;
    R[0] = {chs, T(0.0f)};
    R[1] = {shs * c2, shs * s2};
    R[2] = {s * sphi, -(s * cphi)};
    R[3] = {shs * c2, -(shs * s2)};
    R[4] = {chs, T(0.0f)};
    R[5] = {s * sphi, s * cphi};
    R[6] = {-(hs * sphi), -(hs * cphi)};
    R[7] = {-(hs * sphi), hs * cphi};
    R[8] = {c, T(0.0f)};
}
template <class T>
__device__ __forceinline__ void spinor_rotation(Cx<T> a, Cx<T> b, Cx<T>* R) {
    const Cx<T> aa = cmul(a, a), bb = cmul(b, b), ab = cmul(a, b);
    const Cx<T> cross = cmul(cconj(a), b);
    R[0] = cconj(aa);
    R[1] = {-bb.r, bb.i};
    R[2] = {-2.0f * ab.r, 2.0f * ab.i};
    R[3] = {-bb.r, -bb.i};
    R[4] = aa;
    R[5] = {-2.0f * ab.r, -2.0f * ab.i};
    R[6] = cross;
    R[7] = cconj(cross);
    R[8] = {a.r * a.r + a.i * a.i - b.r * b.r - b.i * b.i, T(0.0f)};
}

template <class T, int POOLS, int RF, int Y, int K, bool ONE_TRAIN>
__device__ __forceinline__ void complex_vjp_loop(const Params& v, T* segment) {
    const epg::Params& p = v.f;
    const int mode = v.mode;
    constexpr bool one_train = ONE_TRAIN;
    constexpr int NS = POOLS >= 2 ? 2 : 1;          // transverse pools
    constexpr int NZ = Interval<T, POOLS>::N;       // longitudinal pools
    constexpr int PLANES = 2 * NS + NZ;              // complex planes a state is
    constexpr int KD = DIRECTIONS<POOLS>;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int width = p.width;
    const int state = lane & (width - 1);
    const int group = lane / width;
    const int groups = 32 / width;
    const int first = v.problem_base + ((blockIdx.x * (blockDim.x >> 5) + warp) * groups + group) * Y;
    const float order = static_cast<float>(state);

    int problem[Y], atom[Y], train[Y], location[Y];
    bool active[Y], live[Y];
    Inputs<T> in[Y];
    T t1v[Y], t2v[Y], t1b[Y], t2b[Y], t1c[Y];
    T m0[Y], b1[Y], b1c[Y], b1s[Y], inversion[Y];
    // The state: per transverse pool F+ and F-, per longitudinal pool Z.
    Cx<T> plus[NS][Y], minus[NS][Y], z[NZ][Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        problem[y] = first + y;
        active[y] = problem[y] < v.problem_end;
        live[y] = active[y] && state < p.state_count;
        atom[y] = problem[y] % p.atom_count;
        train[y] = problem[y] / p.atom_count;
        location[y] = RF == PROFILE ? atom[y] % p.locations : 0;
        const int at = p.atom_stride ? atom[y] : 0;
        auto read = [&](const float* values, const float* directions, int where, float otherwise) {
            return active[y] ? num::load<T>(values, directions, where) : T(otherwise);
        };
        t1v[y] = read(p.t1, p.d_t1, atom[y], 1.0f);
        t2v[y] = read(p.t2, p.d_t2, atom[y], 1.0f);
        in[y].r1 = num::rate(t1v[y]);
        in[y].r2 = num::rate(t2v[y]);
        m0[y] = p.density ? read(p.m0, p.d_m0, at, 0.0f) : T(1.0f);
        b1[y] = p.transmit ? read(p.b1, p.d_b1, at, 1.0f) : T(1.0f);
        const T b1_phase = p.off_axis ? read(p.b1_phase, p.d_b1_phase, at, 0.0f) : T(0.0f);
        sincos_(b1_phase, b1s[y], b1c[y]);
        in[y].b0 = p.off_axis ? read(p.b0, p.d_b0, at, 0.0f) : T(0.0f);
        inversion[y] = p.inverting ? read(p.inversion_efficiency, p.d_inversion_efficiency, at, 1.0f) : T(1.0f);
        in[y].diffusion = p.diffusing ? read(p.diffusion, p.d_diffusion, at, 0.0f) : T(0.0f);
        in[y].velocity = p.moving ? read(p.velocity, p.d_velocity, at, 0.0f) : T(0.0f);
        in[y].fb = in[y].xb = in[y].r1b = in[y].r2b = in[y].shift = in[y].fc = in[y].xc = in[y].r1c = 0.0f;
        t1b[y] = t2b[y] = t1c[y] = 1.0f;
        if constexpr (POOLS == 1) {
            in[y].fb = read(p.bound_fraction, p.d_bound_fraction, at, 0.0f);
            in[y].xb = read(p.bound_exchange, p.d_bound_exchange, at, 0.0f);
            t1b[y] = read(p.t1_bound, p.d_t1_bound, at, 1.0f);
        }
        if constexpr (POOLS >= 2) {
            in[y].fb = read(p.pool_b_fraction, p.d_pool_b_fraction, at, 0.0f);
            in[y].xb = read(p.pool_b_exchange, p.d_pool_b_exchange, at, 0.0f);
            t1b[y] = read(p.t1_pool_b, p.d_t1_pool_b, at, 1.0f);
            t2b[y] = read(p.t2_pool_b, p.d_t2_pool_b, at, 1.0f);
            in[y].shift = read(p.pool_b_shift, p.d_pool_b_shift, at, 0.0f);
        }
        if constexpr (POOLS == 3) {
            in[y].fc = read(p.bound_fraction, p.d_bound_fraction, at, 0.0f);
            in[y].xc = read(p.bound_exchange, p.d_bound_exchange, at, 0.0f);
            t1c[y] = read(p.t1_bound, p.d_t1_bound, at, 1.0f);
        }
        in[y].r1b = num::rate(t1b[y]);
        in[y].r2b = num::rate(t2b[y]);
        in[y].r1c = num::rate(t1c[y]);
    }
    const bool uniform = one_train || (active[0] && train[0] == train[Y - 1]);
    const int base = one_train ? 0 : train[0] * p.event_count;
    const int first_train = __shfl_sync(0xffffffffu, train[0], 0);
    const bool warp_train = one_train || __all_sync(0xffffffffu, uniform && train[0] == first_train);
    const int relax_code = (p.moving ? 4 : 0) | (p.diffusing ? 2 : 0) | (p.off_axis ? 1 : 0);
    const Geometry geometry{p.flow_scale, p.washout_scale, p.pool_table, p.atom_count};

    auto initial = [&]() {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
#pragma unroll
            for (int s = 0; s < NS; ++s) plus[s][y] = minus[s][y] = {T(0.0f), T(0.0f)};
            const T free = 1.0f - in[y].fb - in[y].fc;
            z[0][y] = {state == 0 ? free : T(0.0f), T(0.0f)};
            if constexpr (NZ >= 2) z[1][y] = {state == 0 ? in[y].fb : T(0.0f), T(0.0f)};
            if constexpr (NZ >= 3) z[2][y] = {state == 0 ? in[y].fc : T(0.0f), T(0.0f)};
        }
    };
    auto event_dt = [&](int y, int event) {
        if (uniform) return num::load<T>(p.duration, p.d_duration, base + event);
        return active[y] ? num::load<T>(p.duration, p.d_duration, train[y] * p.event_count + event) : T(0.0f);
    };
    auto event_row = [&](int y, int event) {
        if (mode != TABLE) return 0;
        return p.duration_row[(uniform || !active[y] ? base : train[y] * p.event_count) + event];
    };
    // The forward step's interval, kept while it repeats.
    Interval<T, POOLS> ahead[Y], ahead_slope[Y];
    T ahead_dt = -1.0f;
    int ahead_row = -1;
    auto apply = [&](const Interval<T, POOLS>* op) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const Interval<T, POOLS>& o = op[y];
            if constexpr (NS == 1) {
                plus[0][y] = cmul(o.t[0], plus[0][y]);
                minus[0][y] = cmul(cconj(o.t[0]), minus[0][y]);
            } else {
                const Cx<T> p0 = plus[0][y], p1 = plus[1][y], m0v = minus[0][y], m1 = minus[1][y];
                const Cx<T> a = cmul(o.t[0], p0), b = cmul(o.t[1], p1), c = cmul(o.t[2], p0), d = cmul(o.t[3], p1);
                plus[0][y] = {a.r + b.r, a.i + b.i};
                plus[1][y] = {c.r + d.r, c.i + d.i};
                const Cx<T> e = cmul(cconj(o.t[0]), m0v), f = cmul(cconj(o.t[1]), m1);
                const Cx<T> g = cmul(cconj(o.t[2]), m0v), h = cmul(cconj(o.t[3]), m1);
                minus[0][y] = {e.r + f.r, e.i + f.i};
                minus[1][y] = {g.r + h.r, g.i + h.i};
            }
            Cx<T> next[NZ];
#pragma unroll
            for (int i = 0; i < NZ; ++i) {
                next[i] = {state == 0 ? o.g[i] : T(0.0f), T(0.0f)};
#pragma unroll
                for (int j = 0; j < NZ; ++j) {
                    const Cx<T> term = cmul(o.l[i * NZ + j], z[j][y]);
                    next[i] = {next[i].r + term.r, next[i].i + term.i};
                }
            }
#pragma unroll
            for (int i = 0; i < NZ; ++i) z[i][y] = next[i];
        }
    };
    // A train repeats its interval's length, whatever direction each event
    // moves it in: the operator is kept at the length's value and each event
    // moves it along its own direction through the slope.
    auto relax_forward = [&](int event) {
        const T dt0 = event_dt(0, event);
        const int row0 = event_row(0, event);
        if (uniform && same(dt0, T(0.0f))) return;
        if (!uniform) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                ahead[y] = interval_in<T, POOLS>(mode, geometry, relax_code, event_dt(y, event), in[y], order,
                                                       event_row(y, event), atom[y]);
            }
            ahead_dt = -1.0f;
            apply(ahead);
            return;
        }
        if (!same(flat(dt0), ahead_dt) || row0 != ahead_row) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                if constexpr (num::is_dual<T>::value) {
                    const Opened<T, POOLS> opened =
                        open_in<T, POOLS>(mode, geometry, relax_code, flat(dt0), in[y], order, row0, atom[y]);
                    ahead[y] = opened.value;
                    ahead_slope[y] = opened.slope;
                } else {
                    ahead[y] = interval_in<T, POOLS>(mode, geometry, relax_code, dt0, in[y], order, row0, atom[y]);
                }
            }
            ahead_dt = flat(dt0);
            ahead_row = row0;
        }
        if constexpr (num::is_dual<T>::value) {
            const float d = direction(dt0);
            if (d != 0.0f) {
                Interval<T, POOLS> now[Y];
#pragma unroll
                for (int y = 0; y < Y; ++y) now[y] = moved(ahead[y], ahead_slope[y], d);
                apply(now);
                return;
            }
        }
        apply(ahead);
    };
    auto shift = [&](int s) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T up_r = num::shfl_up(plus[s][y].r, 1, width), up_i = num::shfl_up(plus[s][y].i, 1, width);
            const T dn_r = num::shfl_down(minus[s][y].r, 1, width), dn_i = num::shfl_down(minus[s][y].i, 1, width);
            const bool keep_up = state > 0 && live[y], keep_down = state + 1 < p.state_count && live[y];
            const T pr = keep_up ? up_r : T(0.0f), pi = keep_up ? up_i : T(0.0f);
            const T mr = keep_down ? dn_r : T(0.0f), mi = keep_down ? dn_i : T(0.0f);
            plus[s][y] = {state == 0 ? mr : pr, state == 0 ? -mi : pi};
            minus[s][y] = {mr, mi};
        }
    };
    // The flip, phase and transmit of a pulse for problem y.
    auto pulse_terms = [&](int y, int event, T& alpha, T& cphi, T& sphi, T& transmit) {
        const int e = uniform ? base + event : (active[y] ? train[y] * p.event_count + event : event);
        const T flip = num::load<T>(p.flip, p.d_flip, e);
        T ce, se;
        event_phase(p, e, ce, se);
        T tb1 = b1[y], tc = b1c[y], ts = b1s[y];
        if (p.shimmed) {
            const int cell = p.shim_index[event] * p.atom_count + atom[y];
            tb1 = p.transmit ? (active[y] ? num::load<T>(p.b1, p.d_b1, cell) : T(1.0f)) : T(1.0f);
            if (p.off_axis) sincos_(active[y] ? num::load<T>(p.b1_phase, p.d_b1_phase, cell) : T(0.0f), ts, tc);
        }
        alpha = flip * tb1;
        transmit = tb1;
        cphi = ce * tc - se * ts;
        sphi = se * tc + ce * ts;
        return flip;
    };
    auto pair_of = [&](int y, int event, Cx<T>& a, Cx<T>& b) {
        const int row = p.pair_index[(active[y] ? train[y] * p.event_count : 0) + event];
        const long long cell = (static_cast<long long>(row) * p.atom_count + atom[y]) * 4;
        a = {num::load<T>(p.pairs, p.pair_direction, cell), num::load<T>(p.pairs, p.pair_direction, cell + 1)};
        b = {num::load<T>(p.pairs, p.pair_direction, cell + 2), num::load<T>(p.pairs, p.pair_direction, cell + 3)};
        return cell;
    };
    // The rotation a pulse performs on problem y, from its flip and phase.
    auto rotation = [&](int y, int event, auto alpha, auto cphi, auto sphi, auto* R) {
        using U = typename std::decay<decltype(alpha)>::type;
        if constexpr (RF == HARD) {
            hard_rotation(alpha, cphi, sphi, R);
        } else if constexpr (RF == PROFILE) {
            const int row = p.profile_index[event] * p.locations + location[y];
            const int last = p.profile_bins - 1;
            const U scaled = min_(max_(div_(alpha, U(p.profile_step)), U(0.0f)), U(last + 0.0f));
            const float lower = fminf(floorf(primal(scaled)), last - 1.0f);
            U h10, h01, h11;
            const U h00 = hermite_weights(scaled, lower, p.profile_step, h10, h01, h11);
            const float* knot = p.profile + (row * p.profile_bins + static_cast<int>(lower)) * 8;
            U pair[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                pair[c] = h00 * __ldg(knot + c) + h10 * __ldg(knot + 4 + c) + h01 * __ldg(knot + 8 + c) +
                          h11 * __ldg(knot + 12 + c);
            }
            const Cx<U> turn = {cphi, -sphi};
            spinor_rotation(Cx<U>{pair[0], pair[1]}, cmul(Cx<U>{pair[2], pair[3]}, turn), R);
        }
    };
    auto absorbed_of = [&](int y, int event, auto alpha, auto offset_b0) {
        using U = typename std::decay<decltype(alpha)>::type;
        const float saturation = p.saturation[event];
        const U shape = lineshape_at(p.lineshape, p.rf_frequency[event] - offset_b0, p.lineshape_bins, p.lineshape_step);
        return exp_(saturation * alpha * alpha * shape);
    };
    auto rotate = [&](int y, const Cx<T>* R, int s, int zi) {
        const Cx<T> fp = plus[s][y], fm = minus[s][y], zz = z[zi][y];
        Cx<T> out[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const Cx<T> a = cmul(R[3 * i], fp), b = cmul(R[3 * i + 1], fm), c = cmul(R[3 * i + 2], zz);
            out[i] = {a.r + b.r + c.r, a.i + b.i + c.i};
        }
        plus[s][y] = out[0];
        minus[s][y] = out[1];
        z[zi][y] = out[2];
    };
    auto pulse_forward = [&](int event) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            T alpha, cphi, sphi, transmit;
            pulse_terms(y, event, alpha, cphi, sphi, transmit);
            Cx<T> R[9];
            if constexpr (RF == DYNAMIC) {
                Cx<T> a, b;
                pair_of(y, event, a, b);
                spinor_rotation(a, cmul(b, Cx<T>{cphi, -sphi}), R);
            } else {
                rotation(y, event, alpha, cphi, sphi, R);
            }
            rotate(y, R, 0, 0);
            if constexpr (POOLS >= 2) rotate(y, R, 1, 1);
            if constexpr (POOLS == 1 || POOLS == 3) {
                const T absorbed = absorbed_of(y, event, alpha, p.off_axis ? in[y].b0 : T(0.0f));
                Cx<T>& zz = z[POOLS == 1 ? 1 : 2][y];
                zz = {absorbed * zz.r, absorbed * zz.i};
            }
        }
    };
    auto forward = [&](int event) {
        const unsigned char act = p.action[event];
        const int kind = p.kind[event];
        relax_forward(event);
        if (act & 1) {
#pragma unroll
            for (int s = 0; s < NS; ++s) shift(s);
        }
        if (kind == 1 && (act & 4)) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                const T eff = inversion[y];
                z[0][y] = {-eff * z[0][y].r, -eff * z[0][y].i};
                if constexpr (POOLS >= 2) z[1][y] = {-eff * z[1][y].r, -eff * z[1][y].i};
            }
        } else if (kind == 1) {
            pulse_forward(event);
        }
        if (act & 18) {
#pragma unroll
            for (int s = 0; s < NS; ++s) shift(s);
        }
        if (act & 8) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
#pragma unroll
                for (int s = 0; s < NS; ++s) plus[s][y] = minus[s][y] = {T(0.0f), T(0.0f)};
            }
        }
    };
    // Checkpoints: a complex plane per transverse and longitudinal component.
    const long long stride = static_cast<long long>(p.event_count) * PLANES * p.state_count;
    auto slot = [&](int y, int checkpoint, int plane) {
        return (problem[y] - v.problem_base) * stride +
               (static_cast<long long>(checkpoint) * PLANES + plane) * p.state_count + state;
    };
    auto plane_of = [&](int y, int plane) -> Cx<T>& {
        if (plane < NS) return plus[plane][y];
        if (plane < 2 * NS) return minus[plane - NS][y];
        return z[plane - 2 * NS][y];
    };
    const int checkpoints = (p.event_count + K - 1) / K;
    initial();
#pragma unroll 1
    for (int event = 0; event < p.event_count; ++event) {
        if (event % K == 0) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                if (live[y]) {
#pragma unroll
                    for (int plane = 0; plane < PLANES; ++plane) {
                        const Cx<T> value = plane_of(y, plane);
                        put(v.trajectory_r, v.trajectory_tr, slot(y, event / K, plane), value.r);
                        put(v.trajectory_i, v.trajectory_ti, slot(y, event / K, plane), value.i);
                    }
                }
            }
        }
        forward(event);
    }

    // ---- the walk back ----
    // Cotangents of the state, laid out as the state is.
    Cx<T> bplus[NS][Y], bminus[NS][Y], bz[NZ][Y];
    // Per-lane gradients of each problem's tissue: the interval's directions,
    // then m0, b1, the transmit phase and the inversion.
    T grad[KD][Y], g_m0[Y], g_b1[Y], g_b1_phase[Y], g_inversion[Y], g_b0_pulse[Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
#pragma unroll
        for (int s = 0; s < NS; ++s) bplus[s][y] = bminus[s][y] = {T(0.0f), T(0.0f)};
#pragma unroll
        for (int i = 0; i < NZ; ++i) bz[i][y] = {T(0.0f), T(0.0f)};
#pragma unroll
        for (int k = 0; k < KD; ++k) grad[k][y] = 0.0f;
        g_m0[y] = g_b1[y] = g_b1_phase[y] = g_inversion[y] = g_b0_pulse[y] = 0.0f;
    }
    // The interval the walk back is in, its derivative along its own length,
    // and what it has met since it began.
    Interval<T, POOLS> back[Y], back_dt[Y], back_curve[Y];
    // What the interval met, weighted by each event's direction along its
    // length: the mixed second derivatives contract against it.
    Met<float, POOLS> met_along[Y];
    Met<T, POOLS> met[Y];
    T back_length[Y];
    int back_row[Y];
    bool back_open = false;
    auto clear_met = [&]() {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
#pragma unroll
            for (int k = 0; k < Interval<T, POOLS>::NT; ++k) {
                met[y].t[k] = {T(0.0f), T(0.0f)};
                met_along[y].t[k] = {0.0f, 0.0f};
            }
#pragma unroll
            for (int k = 0; k < Interval<T, POOLS>::NL; ++k) {
                met[y].l[k] = {T(0.0f), T(0.0f)};
                met_along[y].l[k] = {0.0f, 0.0f};
            }
#pragma unroll
            for (int k = 0; k < Interval<T, POOLS>::NG; ++k) {
                met[y].g[k] = 0.0f;
                met_along[y].g[k] = 0.0f;
            }
        }
    };
    auto close_interval = [&]() {
        if (!back_open) return;
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const Gradients<T, POOLS> got = contract_in<T, POOLS>(mode, geometry, relax_code, back_length[y], in[y],
                                                                              order, back_row[y], atom[y], met[y]);
#pragma unroll
            for (int k = 0; k < KD; ++k) grad[k][y] += got.g[k];
            if constexpr (num::is_dual<T>::value) {
                // d/dlength of each tissue derivative, against the met the
                // events' own length directions weighted.
                Met<T, POOLS> along_met;
#pragma unroll
                for (int k = 0; k < Interval<T, POOLS>::NT; ++k) along_met.t[k] = {T(met_along[y].t[k].r), T(met_along[y].t[k].i)};
#pragma unroll
                for (int k = 0; k < Interval<T, POOLS>::NL; ++k) along_met.l[k] = {T(met_along[y].l[k].r), T(met_along[y].l[k].i)};
#pragma unroll
                for (int k = 0; k < Interval<T, POOLS>::NG; ++k) along_met.g[k] = T(met_along[y].g[k]);
                const Gradients<T, POOLS> mixed = contract_in<T, POOLS>(mode, geometry, relax_code, T(back_length[y].v, 1.0f), flat(in[y]), order, back_row[y], atom[y], along_met);
#pragma unroll
                for (int k = 0; k < KD; ++k) grad[k][y].d += mixed.g[k].d;
            }
        }
        clear_met();
    };
    auto open_interval = [&](int event) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T dt = event_dt(y, event);
            back_length[y] = uniform ? flat(dt) : dt;
            back_row[y] = event_row(y, event);
            const Opened<T, POOLS> opened =
                open_in<T, POOLS>(mode, geometry, relax_code, back_length[y], in[y], order, back_row[y], atom[y]);
            back[y] = opened.value;
            back_dt[y] = opened.slope;
            if constexpr (num::is_dual<T>::value) {
                if (uniform) {
                    // The slope's own derivative along the length.
                    const Opened<T, POOLS> curved = open_in<T, POOLS>(mode, geometry, relax_code, T(back_length[y].v, 1.0f), flat(in[y]), order, back_row[y], atom[y]);
                    // Held as values, which is what moving the slope reads.
#pragma unroll
                    for (int k = 0; k < Interval<T, POOLS>::NT; ++k) {
                        back_curve[y].t[k] = {T(curved.slope.t[k].r.d), T(curved.slope.t[k].i.d)};
                    }
#pragma unroll
                    for (int k = 0; k < Interval<T, POOLS>::NL; ++k) {
                        back_curve[y].l[k] = {T(curved.slope.l[k].r.d), T(curved.slope.l[k].i.d)};
                    }
#pragma unroll
                    for (int k = 0; k < Interval<T, POOLS>::NG; ++k) back_curve[y].g[k] = T(curved.slope.g[k].d);
                }
            }
        }
        back_open = true;
    };
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
    auto shift_adjoint = [&](int s) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T carry_r = num::shfl(bplus[s][y].r, 0, width), carry_i = -num::shfl(bplus[s][y].i, 0, width);
            const T dn_r = num::shfl_down(bplus[s][y].r, 1, width), dn_i = num::shfl_down(bplus[s][y].i, 1, width);
            const T up_r = num::shfl_up(bminus[s][y].r, 1, width), up_i = num::shfl_up(bminus[s][y].i, 1, width);
            const bool forward_ok = state + 1 < p.state_count && live[y], back_ok = state > 0 && live[y];
            Cx<T> m = {back_ok ? up_r : T(0.0f), back_ok ? up_i : T(0.0f)};
            if (state == 1 && live[y]) m = {m.r + carry_r, m.i + carry_i};
            bplus[s][y] = {forward_ok ? dn_r : T(0.0f), forward_ok ? dn_i : T(0.0f)};
            bminus[s][y] = m;
        }
    };
    clear_met();
    const int threads = blockDim.x;
    auto smem = [&](int k, int plane, int y, int part) -> T& {
        return segment[(((k * PLANES + plane) * Y + y) * 2 + part) * threads + threadIdx.x];
    };
    const long long n_atoms = p.atom_count;
    const int past_transmit = 2 * (v.shim_rows - 1);

#pragma unroll 1
    for (int checkpoint = checkpoints - 1; checkpoint >= 0; --checkpoint) {
        const int start = checkpoint * K;
        const int stop = min(start + K, p.event_count);
#pragma unroll
        for (int y = 0; y < Y; ++y) {
#pragma unroll
            for (int plane = 0; plane < PLANES; ++plane) {
                plane_of(y, plane) = live[y] ? Cx<T>{get<T>(v.trajectory_r, v.trajectory_tr, slot(y, checkpoint, plane)),
                                                    get<T>(v.trajectory_i, v.trajectory_ti, slot(y, checkpoint, plane))}
                                             : Cx<T>{T(0.0f), T(0.0f)};
            }
        }
#pragma unroll 1
        for (int event = start; event < stop; ++event) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
#pragma unroll
                for (int plane = 0; plane < PLANES; ++plane) {
                    smem(event - start, plane, y, 0) = plane_of(y, plane).r;
                    smem(event - start, plane, y, 1) = plane_of(y, plane).i;
                }
            }
            if (event + 1 < stop) forward(event);
        }
#pragma unroll 1
        for (int event = stop - 1; event >= start; --event) {
            const unsigned char act = p.action[event];
            const int kind = p.kind[event];
            // The entry state, kept for the interval's contraction.
            Cx<T> entry[PLANES][Y];
#pragma unroll
            for (int y = 0; y < Y; ++y) {
#pragma unroll
                for (int plane = 0; plane < PLANES; ++plane) {
                    entry[plane][y] = {smem(event - start, plane, y, 0), smem(event - start, plane, y, 1)};
                    plane_of(y, plane) = entry[plane][y];
                }
            }
            // The interval this event relaxes over, and the stage its pulse
            // or sample sees.
            {
                const T dt0 = event_dt(0, event);
                const int row0 = event_row(0, event);
                bool changed = !back_open || !uniform || row0 != back_row[0];
                if (!changed) changed = !same(flat(dt0), back_length[0]);
                if (changed) {
                    close_interval();
                    open_interval(event);
                }
            }
            // This event's operator and slope: the kept ones moved along its
            // length's direction.
            const float along = uniform ? direction(event_dt(0, event)) : 0.0f;
            constexpr bool MOVES = num::is_dual<T>::value;
            Interval<T, POOLS> moved_op[MOVES ? Y : 1], moved_slope[MOVES ? Y : 1];
            if constexpr (MOVES) {
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    moved_op[y] = moved(back[y], back_dt[y], along);
                    moved_slope[y] = moved(back_dt[y], back_curve[y], along);
                }
            }
            const Interval<T, POOLS>* op = MOVES ? moved_op : back;
            const Interval<T, POOLS>* slope = MOVES ? moved_slope : back_dt;
            apply(op);
            if (act & 1) {
#pragma unroll
                for (int s = 0; s < NS; ++s) shift(s);
            }
            if (act & 8) {
#pragma unroll
                for (int y = 0; y < Y; ++y) {
#pragma unroll
                    for (int s = 0; s < NS; ++s) bplus[s][y] = bminus[s][y] = {T(0.0f), T(0.0f)};
                }
            } else if (act & 18) {
#pragma unroll
                for (int s = 0; s < NS; ++s) shift_adjoint(s);
            }
            if (kind == 1 && (act & 4)) {
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    const T eff = inversion[y];
                    g_inversion[y] -= rdot(bz[0][y], z[0][y]);
                    bz[0][y] = {-eff * bz[0][y].r, -eff * bz[0][y].i};
                    if constexpr (POOLS >= 2) {
                        g_inversion[y] -= rdot(bz[1][y], z[1][y]);
                        bz[1][y] = {-eff * bz[1][y].r, -eff * bz[1][y].i};
                    }
                }
            } else if (kind == 1) {
                T flip_gradient[Y], phase_gradient[Y];
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    T alpha, cphi, sphi, transmit;
                    const T flip = pulse_terms(y, event, alpha, cphi, sphi, transmit);
                    T g_alpha = 0.0f, g_phi = 0.0f;
                    Cx<T> R[9];
                    // Re(conj(cotangent) dR S) along the pulse's own inputs.
                    auto against = [&](const auto* dR, int s, int zi) {
                        using U = typename std::decay<decltype(dR[0].r)>::type;
                        const Cx<T> S[3] = {plus[s][y], minus[s][y], z[zi][y]};
                        const Cx<T> L[3] = {bplus[s][y], bminus[s][y], bz[zi][y]};
                        U sum = 0.0f;
#pragma unroll
                        for (int i = 0; i < 3; ++i) {
#pragma unroll
                            for (int j = 0; j < 3; ++j) {
                                const Cx<U> term = {dR[3 * i + j].r * S[j].r - dR[3 * i + j].i * S[j].i,
                                                    dR[3 * i + j].r * S[j].i + dR[3 * i + j].i * S[j].r};
                                sum += L[i].r * term.r + L[i].i * term.i;
                            }
                        }
                        return sum;
                    };
                    if constexpr (RF == DYNAMIC) {
                        using V = Multi<5, T>;
                        Cx<T> a, b;
                        const long long cell = pair_of(y, event, a, b);
                        // The phase turns b by e^{-i phi}: d/dphi of (cos, -sin) is (-sin, -cos).
                        Cx<V> turn = {V(cphi, -1), V(-sphi, -1)};
                        turn.r.d[0] = -sphi;
                        turn.i.d[0] = -cphi;
                        Cx<V> mb[9];
                        spinor_rotation(Cx<V>{V(a.r, 1), V(a.i, 2)}, cmul(Cx<V>{V(b.r, 3), V(b.i, 4)}, turn), mb);
                        V got = against(mb, 0, 0);
                        if constexpr (POOLS >= 2) got += against(mb, 1, 1);
                        g_phi = got.d[0];
                        T pair_sums[4] = {got.d[1], got.d[2], got.d[3], got.d[4]};
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const T sum = lanes_sum(pair_sums[c], 1, width);
                            if (state == 0 && active[y]) atomic_add(v.grad_pair, v.grad_pair_t, cell + c, sum);
                        }
                        Cx<T> RR[9];
                        spinor_rotation(a, cmul(b, Cx<T>{cphi, -sphi}), RR);
#pragma unroll
                        for (int k = 0; k < 9; ++k) R[k] = RR[k];
                    } else {
                        using V = Multi<2, T>;
                        Cx<V> mr[9];
                        V va = V(alpha, 0), vc = V(cphi, 1), vs = V(sphi, 1);
                        vc.d[1] = -sphi;
                        vs.d[1] = cphi;
                        rotation(y, event, va, vc, vs, mr);
                        V got = against(mr, 0, 0);
                        if constexpr (POOLS >= 2) got += against(mr, 1, 1);
                        g_alpha = got.d[0];
                        g_phi = got.d[1];
#pragma unroll
                        for (int k = 0; k < 9; ++k) R[k] = {mr[k].r.v, mr[k].i.v};
                    }
                    if constexpr (POOLS == 1 || POOLS == 3) {
                        constexpr int ZS = POOLS == 1 ? 1 : 2;
                        using V = Multi<2, T>;
                        const V absorbed = absorbed_of(y, event, V(alpha, 0), p.off_axis ? V(in[y].b0, 1) : V(T(0.0f), -1));
                        const T met_z = rdot(bz[ZS][y], z[ZS][y]);
                        g_alpha += absorbed.d[0] * met_z;
                        g_b0_pulse[y] += absorbed.d[1] * met_z;
                        bz[ZS][y] = {absorbed.v * bz[ZS][y].r, absorbed.v * bz[ZS][y].i};
                    }
                    // The cotangent through the rotation: R^H.
#pragma unroll
                    for (int s = 0; s < NS; ++s) {
                        const int zi = s;
                        const Cx<T> L[3] = {bplus[s][y], bminus[s][y], bz[zi][y]};
                        Cx<T> back_out[3];
#pragma unroll
                        for (int j = 0; j < 3; ++j) {
                            back_out[j] = {T(0.0f), T(0.0f)};
#pragma unroll
                            for (int i = 0; i < 3; ++i) {
                                const Cx<T> term = cmul(cconj(R[3 * i + j]), L[i]);
                                back_out[j] = {back_out[j].r + term.r, back_out[j].i + term.i};
                            }
                        }
                        bplus[s][y] = back_out[0];
                        bminus[s][y] = back_out[1];
                        bz[zi][y] = back_out[2];
                    }
                    flip_gradient[y] = g_alpha * transmit;
                    phase_gradient[y] = g_phi;
                    if (p.shimmed) {
                        const long long cell = static_cast<long long>(p.shim_index[event]) * p.atom_count + atom[y];
                        const T b1_sum = lanes_sum(g_alpha * flip, 1, width);
                        const T phase_sum = lanes_sum(g_phi, 1, width);
                        if (state == 0 && active[y]) {
                            atomic_add(v.grad_tissue, v.grad_tissue_t, 3 * n_atoms + cell, b1_sum);
                            atomic_add(v.grad_tissue, v.grad_tissue_t, (3 + v.shim_rows) * n_atoms + cell, phase_sum);
                        }
                    } else {
                        g_b1[y] += g_alpha * flip;
                        g_b1_phase[y] += g_phi;
                    }
                }
                event_gradient(v.grad_flip, v.grad_flip_t, event, flip_gradient);
                event_gradient(v.grad_phase, v.grad_phase_t, event, phase_gradient);
            }
            if ((act & 32) && kind == 2) {
                const int out = p.output_index[event];
                if (out >= 0) {
                    T phase_gradient[Y];
#pragma unroll
                    for (int y = 0; y < Y; ++y) {
                        const int e = uniform ? base + event : (active[y] ? train[y] * p.event_count + event : event);
                        T ac, as;
                        event_phase(p, e, ac, as);
                        phase_gradient[y] = 0.0f;
                        if (state == 0 && active[y]) {
                            const long long at = static_cast<long long>(problem[y]) * p.output_count + out;
                            const Cx<T> seed = {T(v.grad_output_real[at]), T(v.grad_output_imag[at])};
                            Cx<T> read = plus[0][y];
                            if constexpr (POOLS >= 2) read = {read.r + plus[1][y].r, read.i + plus[1][y].i};
                            const Cx<T> demodulated = cmul(read, Cx<T>{ac, -as});
                            g_m0[y] += rdot(seed, demodulated);
                            phase_gradient[y] = m0[y] * rdot(seed, Cx<T>{demodulated.i, -demodulated.r});
                            const Cx<T> back_seed = cmul(Cx<T>{m0[y] * ac, m0[y] * as}, seed);
                            bplus[0][y] = {bplus[0][y].r + back_seed.r, bplus[0][y].i + back_seed.i};
                            if constexpr (POOLS >= 2) {
                                bplus[1][y] = {bplus[1][y].r + back_seed.r, bplus[1][y].i + back_seed.i};
                            }
                        }
                    }
                    event_gradient(v.grad_phase, v.grad_phase_t, event, phase_gradient);
                }
            }
            if (act & 1) {
#pragma unroll
                for (int s = 0; s < NS; ++s) shift_adjoint(s);
            }
            // The interval: what it met, its length's gradient, and the
            // cotangent through it.
            T duration_gradient[Y];
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                const Interval<T, POOLS>& o = op[y];
                Met<T, POOLS> now;
                if constexpr (NS == 1) {
                    const Cx<T> sp = entry[0][y], sm = entry[1][y];
                    now.t[0] = {sp.r * bplus[0][y].r + sp.i * bplus[0][y].i + sm.r * bminus[0][y].r + sm.i * bminus[0][y].i,
                                sp.i * bplus[0][y].r - sp.r * bplus[0][y].i - sm.i * bminus[0][y].r + sm.r * bminus[0][y].i};
                } else {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const Cx<T> sp = entry[j][y], sm = entry[2 + j][y];
                            const Cx<T> lp = bplus[i][y], lm = bminus[i][y];
                            now.t[2 * i + j] = {sp.r * lp.r + sp.i * lp.i + sm.r * lm.r + sm.i * lm.i,
                                                sp.i * lp.r - sp.r * lp.i - sm.i * lm.r + sm.r * lm.i};
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < NZ; ++i) {
#pragma unroll
                    for (int j = 0; j < NZ; ++j) {
                        const Cx<T> s = entry[2 * NS + j][y], l = bz[i][y];
                        now.l[i * NZ + j] = {s.r * l.r + s.i * l.i, s.i * l.r - s.r * l.i};
                    }
                    now.g[i] = state == 0 ? bz[i][y].r : T(0.0f);
                }
                duration_gradient[y] = contract(slope[y], now);
                if constexpr (num::is_dual<T>::value) {
#pragma unroll
                    for (int k = 0; k < Interval<T, POOLS>::NT; ++k) {
                        met_along[y].t[k] = {met_along[y].t[k].r + along * now.t[k].r.v, met_along[y].t[k].i + along * now.t[k].i.v};
                    }
#pragma unroll
                    for (int k = 0; k < Interval<T, POOLS>::NL; ++k) {
                        met_along[y].l[k] = {met_along[y].l[k].r + along * now.l[k].r.v, met_along[y].l[k].i + along * now.l[k].i.v};
                    }
#pragma unroll
                    for (int k = 0; k < Interval<T, POOLS>::NG; ++k) met_along[y].g[k] += along * now.g[k].v;
                }
#pragma unroll
                for (int k = 0; k < Interval<T, POOLS>::NT; ++k) {
                    met[y].t[k] = {met[y].t[k].r + now.t[k].r, met[y].t[k].i + now.t[k].i};
                }
#pragma unroll
                for (int k = 0; k < Interval<T, POOLS>::NL; ++k) {
                    met[y].l[k] = {met[y].l[k].r + now.l[k].r, met[y].l[k].i + now.l[k].i};
                }
#pragma unroll
                for (int k = 0; k < Interval<T, POOLS>::NG; ++k) met[y].g[k] += now.g[k];
                // Cotangent through the operator: its conjugate transpose.
                if constexpr (NS == 1) {
                    bplus[0][y] = cmul(cconj(o.t[0]), bplus[0][y]);
                    bminus[0][y] = cmul(o.t[0], bminus[0][y]);
                } else {
                    const Cx<T> lp0 = bplus[0][y], lp1 = bplus[1][y], lm0 = bminus[0][y], lm1 = bminus[1][y];
                    const Cx<T> a = cmul(cconj(o.t[0]), lp0), b = cmul(cconj(o.t[2]), lp1);
                    const Cx<T> c = cmul(cconj(o.t[1]), lp0), d = cmul(cconj(o.t[3]), lp1);
                    bplus[0][y] = {a.r + b.r, a.i + b.i};
                    bplus[1][y] = {c.r + d.r, c.i + d.i};
                    const Cx<T> e = cmul(o.t[0], lm0), f = cmul(o.t[2], lm1), g = cmul(o.t[1], lm0), h = cmul(o.t[3], lm1);
                    bminus[0][y] = {e.r + f.r, e.i + f.i};
                    bminus[1][y] = {g.r + h.r, g.i + h.i};
                }
                Cx<T> next[NZ];
#pragma unroll
                for (int j = 0; j < NZ; ++j) {
                    next[j] = {T(0.0f), T(0.0f)};
#pragma unroll
                    for (int i = 0; i < NZ; ++i) {
                        const Cx<T> term = cmul(cconj(o.l[i * NZ + j]), bz[i][y]);
                        next[j] = {next[j].r + term.r, next[j].i + term.i};
                    }
                }
#pragma unroll
                for (int j = 0; j < NZ; ++j) bz[j][y] = next[j];
            }
            event_gradient(v.grad_duration, v.grad_duration_t, event, duration_gradient);
        }
    }
    close_interval();

    // The fractions also set where each pool starts.
    T g_fb[Y], g_fc[Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        g_fb[y] = g_fc[y] = 0.0f;
        if (state == 0) {
            if constexpr (NZ >= 2) g_fb[y] = bz[1][y].r - bz[0][y].r;
            if constexpr (NZ >= 3) g_fc[y] = bz[2][y].r - bz[0][y].r;
        }
    }
    auto store_row = [&](int y, int row, T lane_value) {
        const T sum = lanes_sum(lane_value, 1, width);
        if (state == 0 && active[y]) atomic_add(v.grad_tissue, v.grad_tissue_t, row * n_atoms + atom[y], sum);
    };
    auto from_rate = [&](T gradient, T rate) { return gradient * (-rate * rate * (1.0f / 1000.0f)); };
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        store_row(y, 0, from_rate(grad[D_R1][y], in[y].r1));
        store_row(y, 1, from_rate(grad[D_R2][y], in[y].r2));
        store_row(y, 2, g_m0[y]);
        if (!p.shimmed) {
            store_row(y, 3, g_b1[y]);
            store_row(y, 4, g_b1_phase[y]);
        }
        store_row(y, 5 + past_transmit, grad[D_B0][y] + g_b0_pulse[y]);
        store_row(y, 6 + past_transmit, g_inversion[y]);
        store_row(y, 7 + past_transmit, grad[D_DIFFUSION][y]);
        store_row(y, 8 + past_transmit, grad[D_VELOCITY][y]);
        if constexpr (POOLS == 1) {
            store_row(y, 9 + past_transmit, grad[D_FB][y] + g_fb[y]);
            store_row(y, 10 + past_transmit, grad[D_XB][y]);
            store_row(y, 11 + past_transmit, from_rate(grad[D_R1B][y], in[y].r1b));
        }
        if constexpr (POOLS >= 2) {
            store_row(y, 12 + past_transmit, grad[D_FB][y] + g_fb[y]);
            store_row(y, 13 + past_transmit, grad[D_XB][y]);
            store_row(y, 14 + past_transmit, from_rate(grad[D_R1B][y], in[y].r1b));
            store_row(y, 15 + past_transmit, from_rate(grad[D_R2B][y], in[y].r2b));
            store_row(y, 16 + past_transmit, grad[D_SHIFT][y]);
        }
        if constexpr (POOLS == 3) {
            store_row(y, 9 + past_transmit, grad[D_FC][y] + g_fc[y]);
            store_row(y, 10 + past_transmit, grad[D_XC][y]);
            store_row(y, 11 + past_transmit, from_rate(grad[D_R1C][y], in[y].r1c));
        }
    }
}

}  // namespace epg_vjp
