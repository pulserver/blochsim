// The complex EPG event loop over a number type ``T``: the forward simulation
// at ``float``, its Jacobian-vector product at ``num::Dual``.
//
// Layout (compile time): POOLS (0 free water alone, 1 beside a semisolid
// pool, 2 beside an exchanging pool b, 3 beside both), MODE (how a three-pool
// interval is formed: series, roots or a table), RF (hard pulse, slice-
// profile table or a pair per pulse per voxel), MAPS (per-voxel maps in
// registers), Y problems per thread, ONE_TRAIN. Run time: atom_stride,
// transmit, density, inverting, diffusing, off_axis, moving, shimmed.
// One warp per program; lanes along the states, groups of lanes across
// problems, Y problems of a group in each thread's registers.
#pragma once

#include <cuda_runtime.h>

#include <type_traits>

#include "_layout_numbers.hpp"

namespace epg {

using num::abs_;
using num::div_;
using num::exp_;
using num::fma_;
using num::max_;
using num::min_;
using num::same;
using num::sincos_;
using num::sqrt_;
using num::primal;
using num::value;

constexpr float TWO_PI = 6.283185307179586f;

enum Rf { HARD = 0, PROFILE = 1, DYNAMIC = 2 };
enum Mode { NARROW = 0, ROOTS = 1, TABLE = 2 };

// What a launch hands the loop: the forward kernel's arguments, and for a
// Jacobian-vector product the direction along every input that has one.
struct Params {
    const float *t1, *t2, *m0, *b1, *b1_phase, *b0, *inversion_efficiency, *diffusion, *velocity;
    const float *bound_fraction, *bound_exchange, *t1_bound, *pool_b_fraction, *pool_b_exchange;
    const float *t1_pool_b, *t2_pool_b, *pool_b_shift;
    const float *duration, *flip, *phase, *phase_cos, *phase_sin, *profile;
    const float *saturation, *rf_frequency, *lineshape, *pairs, *pool_table;
    const int *kind, *output_index, *shim_index, *profile_index, *pair_index, *duration_row;
    const unsigned char* action;
    // Directions, read only by a dual loop.
    const float *d_t1, *d_t2, *d_m0, *d_b1, *d_b1_phase, *d_b0, *d_inversion_efficiency;
    const float *d_diffusion, *d_velocity, *d_bound_fraction, *d_bound_exchange, *d_t1_bound;
    const float *d_pool_b_fraction, *d_pool_b_exchange, *d_t1_pool_b, *d_t2_pool_b, *d_pool_b_shift;
    const float *d_duration, *d_flip, *d_phase, *pair_direction;
    float *output_real, *output_imag;
    int atom_count, train_count, event_count, output_count, state_count, width;
    int locations, profile_bins, lineshape_bins;
    float flow_scale, washout_scale, profile_step, lineshape_step;
    bool single_train, atom_stride, shimmed, off_axis, moving, diffusing, transmit, density,
        inverting;
};

template <class T>
constexpr bool DUAL = num::is_dual<T>::value;

template <class T>
__device__ __forceinline__ T read(const float* values, const float* directions, long long at) {
    return num::load<T>(values, directions, at);
}

// An event's phase as its cosine and sine: read where the launch took them,
// formed where a direction moves the phase.
template <class T>
__device__ __forceinline__ void event_phase(const Params& p, int at, T& c, T& s) {
    if constexpr (DUAL<T>) {
        sincos_(T{__ldg(p.phase + at), __ldg(p.d_phase + at)}, s, c);
    } else if (p.phase_cos != nullptr) {
        c = p.phase_cos[at];
        s = p.phase_sin[at];
    } else {
        sincos_(__ldg(p.phase + at), s, c);
    }
}

// What a readout stores: the value, or along a direction its derivative.
template <class T>
__device__ __forceinline__ float stored(T x) {
    if constexpr (DUAL<T>) {
        return x.d;
    } else {
        return x;
    }
}

// One pool through a hard pulse, its phase folded into the rotation's entries.
template <class T>
__device__ __forceinline__ void rotate_flip_phase(T cosine, T sine, T cos_phi, T sin_phi, T cos_2phi,
                                                  T sin_2phi, T& fp_r, T& fp_i, T& fm_r, T& fm_i,
                                                  T& z_r, T& z_i) {
    const T chs = 0.5f * (1.0f + cosine), shs = 0.5f * (1.0f - cosine), hs = 0.5f * sine;
    const T m2r = fma_(cos_2phi, fm_r, -(sin_2phi * fm_i));
    const T m2i = fma_(sin_2phi, fm_r, cos_2phi * fm_i);
    const T p2r = fma_(cos_2phi, fp_r, sin_2phi * fp_i);
    const T p2i = fma_(cos_2phi, fp_i, -(sin_2phi * fp_r));
    const T za = fma_(sin_phi, z_r, cos_phi * z_i);
    const T zb = fma_(sin_phi, z_i, -(cos_phi * z_r));
    const T zc = fma_(sin_phi, z_r, -(cos_phi * z_i));
    const T zd = fma_(cos_phi, z_r, sin_phi * z_i);
    const T pr = fma_(sine, za, fma_(shs, m2r, chs * fp_r));
    const T pi = fma_(sine, zb, fma_(shs, m2i, chs * fp_i));
    const T mr = fma_(sine, zc, fma_(chs, fm_r, shs * p2r));
    const T mi = fma_(sine, zd, fma_(chs, fm_i, shs * p2i));
    const T ptr = fma_(sin_phi, fp_r, -(cos_phi * fp_i));
    const T mtr = fma_(sin_phi, fm_r, cos_phi * fm_i);
    const T pti = fma_(cos_phi, fp_r, sin_phi * fp_i);
    const T mti = fma_(cos_phi, fm_r, -(sin_phi * fm_i));
    const T zr = fma_(cosine, z_r, fma_(-hs, mtr, -hs * ptr));
    const T zi = fma_(cosine, z_i, fma_(hs, mti, -hs * pti));
    fp_r = pr; fp_i = pi; fm_r = mr; fm_i = mi; z_r = zr; z_i = zi;
}

// The rotation named by its Cayley-Klein pair.
template <class T>
__device__ __forceinline__ void rotate_spinor(T ar, T ai, T br, T bi, T& fp_r, T& fp_i, T& fm_r, T& fm_i,
                                              T& z_r, T& z_i) {
    const T aa_r = ar * ar - ai * ai, aa_i = 2.0f * ar * ai;
    const T bb_r = br * br - bi * bi, bb_i = 2.0f * br * bi;
    const T ab_r = ar * br - ai * bi, ab_i = ar * bi + ai * br;
    const T t00_r = aa_r, t00_i = -aa_i, t01_r = -bb_r, t01_i = bb_i;
    const T t02_r = -2.0f * ab_r, t02_i = 2.0f * ab_i;
    const T t10_r = -bb_r, t10_i = -bb_i, t11_r = aa_r, t11_i = aa_i;
    const T t12_r = -2.0f * ab_r, t12_i = -2.0f * ab_i;
    const T cross_r = ar * br + ai * bi, cross_i = ar * bi - ai * br;
    const T t20_r = cross_r, t20_i = cross_i, t21_r = cross_r, t21_i = -cross_i;
    const T t22 = ar * ar + ai * ai - br * br - bi * bi;
    const T pr = t00_r * fp_r - t00_i * fp_i + t01_r * fm_r - t01_i * fm_i + t02_r * z_r - t02_i * z_i;
    const T pi = t00_r * fp_i + t00_i * fp_r + t01_r * fm_i + t01_i * fm_r + t02_r * z_i + t02_i * z_r;
    const T mr = t10_r * fp_r - t10_i * fp_i + t11_r * fm_r - t11_i * fm_i + t12_r * z_r - t12_i * z_i;
    const T mi = t10_r * fp_i + t10_i * fp_r + t11_r * fm_i + t11_i * fm_r + t12_r * z_i + t12_i * z_r;
    const T zr = t20_r * fp_r - t20_i * fp_i + t21_r * fm_r - t21_i * fm_i + t22 * z_r;
    const T zi = t20_r * fp_i + t20_i * fp_r + t21_r * fm_i + t21_i * fm_r + t22 * z_i;
    fp_r = pr; fp_i = pi; fm_r = mr; fm_i = mi; z_r = zr; z_i = zi;
}

// expm((K - diag(R1)) dt) for free water beside one second pool, times the
// attenuation, as e[0..3]; what each pool recovers as grow[0..1]. Where the
// root is on its series branch it carries no direction of its own.
template <class T>
__device__ __forceinline__ void two_pool_step(T r1_free, T r1_bound, T exchange, T bound, T dt,
                                              T attenuation, T* e, T* grow) {
    const T free = 1.0f - bound;
    const T kab = exchange * bound, kba = exchange * free;
    const T l11 = (-kab - r1_free) * dt, l12 = kba * dt, l21 = kab * dt;
    const T l22 = (-kba - r1_bound) * dt;
    const T half_trace = 0.5f * (l11 + l22), half_gap = 0.5f * (l11 - l22);
    const T square = half_gap * half_gap + l12 * l21;
    const bool turning = primal(square) > 1e-12f;
    const T root = turning ? sqrt_(square) : T(num::sqrt_approx(fmaxf(primal(square), 0.0f)));
    const T upper = exp_(half_trace + root), lower = exp_(half_trace - root);
    const T cosine = 0.5f * (upper + lower);
    const T scale = turning ? div_(0.5f * (upper - lower), root)
                            : exp_(half_trace) *
                                  (1.0f + square * (1.0f / 6.0f) + square * square * (1.0f / 120.0f));
    e[0] = attenuation * (cosine + scale * half_gap);
    e[1] = attenuation * scale * l12;
    e[2] = attenuation * scale * l21;
    e[3] = attenuation * (cosine - scale * half_gap);
    grow[0] = free - (e[0] * free + e[1] * bound);
    grow[1] = bound - (e[2] * free + e[3] * bound);
}

// The principal square root of a complex number; along a direction
// dz / (2 w), and none at the origin. The larger part is taken by a root
// and the smaller as im / (2 x): (|z| + re) / 2 alone cancels to nothing
// just above the negative real axis.
__device__ __forceinline__ void complex_sqrt(float re, float im, float& rr, float& ri) {
    const float magnitude = num::sqrt_approx(re * re + im * im);
    if (re >= 0.0f) {
        rr = num::sqrt_approx(0.5f * (magnitude + re));
        ri = rr > 0.0f ? __fdividef(im, 2.0f * rr) : 0.0f;
    } else {
        const float root_imag = num::sqrt_approx(0.5f * (magnitude - re));
        rr = __fdividef(fabsf(im), 2.0f * root_imag);
        ri = im < 0.0f ? -root_imag : root_imag;
    }
}
__device__ __forceinline__ void complex_sqrt(num::Dual re, num::Dual im, num::Dual& rr, num::Dual& ri) {
    float vr, vi;
    complex_sqrt(re.v, im.v, vr, vi);
    const float guard = 2.0f * (vr * vr + vi * vi);
    float tr = 0.0f, ti = 0.0f;
    if (guard > 0.0f) {
        tr = __fdividef(re.d * vr + im.d * vi, guard);
        ti = __fdividef(im.d * vr - re.d * vi, guard);
    }
    rr = {vr, tr};
    ri = {vi, ti};
}

template <int K, class S>
__device__ __forceinline__ void complex_sqrt(const num::Multi<K, S>& re, const num::Multi<K, S>& im, num::Multi<K, S>& rr,
                                             num::Multi<K, S>& ri) {
    S vr, vi;
    complex_sqrt(re.v, im.v, vr, vi);
    const S guard = 2.0f * (vr * vr + vi * vi);
    const bool live = primal(guard) > 0.0f;
    rr.v = vr;
    ri.v = vi;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        rr.d[k] = live ? div_(re.d[k] * vr + im.d[k] * vi, guard) : S(0.0f);
        ri.d[k] = live ? div_(im.d[k] * vr - re.d[k] * vi, guard) : S(0.0f);
    }
}

template <class T>
__device__ __forceinline__ void complex_exp(T re, T im, T& er, T& ei) {
    const T scale = exp_(re);
    T s, c;
    sincos_(im, s, c);
    er = scale * c;
    ei = scale * s;
}

// expm((K - diag(R2) - 2 pi i diag(0, df)) dt) for two exchanging pools'
// transverse states, times the attenuation: four complex entries, x[0..7].
template <class T>
__device__ __forceinline__ void transverse_step(T r2_free, T r2_bound, T exchange, T bound, T free,
                                                T shift_hz, T dt, T attenuation, T* x) {
    const T kab = exchange * bound, kba = exchange * free;
    const T l11 = (-kab - r2_free) * dt, l12 = kba * dt, l21 = kab * dt;
    const T l22 = (-kba - r2_bound) * dt;
    const T l22_imag = -TWO_PI * shift_hz * dt;
    const T trace_r = 0.5f * (l11 + l22), trace_i = 0.5f * l22_imag;
    const T gap_r = 0.5f * (l11 - l22), gap_i = -0.5f * l22_imag;
    const T square_r = gap_r * gap_r - gap_i * gap_i + l12 * l21;
    const T square_i = 2.0f * gap_r * gap_i;
    T root_r, root_i, upper_r, upper_i, lower_r, lower_i;
    complex_sqrt(square_r, square_i, root_r, root_i);
    complex_exp(trace_r + root_r, trace_i + root_i, upper_r, upper_i);
    complex_exp(trace_r - root_r, trace_i - root_i, lower_r, lower_i);
    const T cos_r = 0.5f * (upper_r + lower_r), cos_i = 0.5f * (upper_i + lower_i);
    T scale_r, scale_i;
    if (primal(square_r) * primal(square_r) + primal(square_i) * primal(square_i) > 1e-24f) {
        const T half_r = 0.5f * (upper_r - lower_r), half_i = 0.5f * (upper_i - lower_i);
        const T inverse = div_(T(1.0f), root_r * root_r + root_i * root_i);
        scale_r = (half_r * root_r + half_i * root_i) * inverse;
        scale_i = (half_i * root_r - half_r * root_i) * inverse;
    } else {
        T plain_r, plain_i;
        complex_exp(trace_r, trace_i, plain_r, plain_i);
        const T square2_r = square_r * square_r - square_i * square_i;
        const T square2_i = 2.0f * square_r * square_i;
        const T poly_r = 1.0f + square_r * (1.0f / 6.0f) + square2_r * (1.0f / 120.0f);
        const T poly_i = square_i * (1.0f / 6.0f) + square2_i * (1.0f / 120.0f);
        scale_r = plain_r * poly_r - plain_i * poly_i;
        scale_i = plain_r * poly_i + plain_i * poly_r;
    }
    const T off_r = scale_r * gap_r - scale_i * gap_i;
    const T off_i = scale_r * gap_i + scale_i * gap_r;
    x[0] = attenuation * (cos_r + off_r);
    x[1] = attenuation * (cos_i + off_i);
    x[2] = attenuation * scale_r * l12;
    x[3] = attenuation * scale_i * l12;
    x[4] = attenuation * scale_r * l21;
    x[5] = attenuation * scale_i * l21;
    x[6] = attenuation * (cos_r - off_r);
    x[7] = attenuation * (cos_i - off_i);
}

// [a, b] exp from exponentials already taken; a series near coalescence.
template <class W>
__device__ __forceinline__ W exp_difference(W lower, W upper, W exp_lower, W exp_upper) {
    const W half = 0.5 * (upper - lower);
    if (fabs(primal(half)) < 1e-4) {
        const W square = half * half;
        return exp_lower * (1.0 + half + 0.5 * square) * (1.0 + square / 6.0);
    }
    return (exp_upper - exp_lower) / (upper - lower);
}

__host__ __device__ constexpr double inverse_factorial(int k) {
    double f = 1.0;
    for (int i = 2; i <= k; ++i) f *= i;
    return 1.0 / f;
}

// Terms K.. of the exponential's series reduced modulo the characteristic
// polynomial, each weight a constant.
template <class W, int K, int TERMS>
__device__ __forceinline__ void series_terms(W determinant, W minors, W& flat, W& linear, W& square,
                                             W& sum_flat, W& sum_linear, W& sum_square) {
    if constexpr (K < TERMS) {
        const W next_flat = square * determinant;
        const W next_linear = flat - square * minors;
        const W next_square = linear;
        flat = next_flat;
        linear = next_linear;
        square = next_square;
        using V = decltype(primal(determinant));
        constexpr double weight = inverse_factorial(K);
        sum_flat = sum_flat + V(weight) * flat;
        sum_linear = sum_linear + V(weight) * linear;
        sum_square = sum_square + V(weight) * square;
        series_terms<W, K + 1, TERMS>(determinant, minors, flat, linear, square, sum_flat, sum_linear,
                                      sum_square);
    }
}

// The precision a three-pool interval is formed in.
template <class T, bool NARROW>
struct WorkOf {
    using type = typename std::conditional<NARROW, float, double>::type;
};
template <bool NARROW>
struct WorkOf<num::Dual, NARROW> {
    using type = typename std::conditional<NARROW, num::Dual, num::Dual64>::type;
};
template <int K, class S, bool NARROW>
struct WorkOf<num::Multi<K, S>, NARROW> {
    using type = num::Multi<K, typename WorkOf<S, NARROW>::type>;
};

// expm((K - diag(R1)) dt) for free water (a), pool b and the semisolid pool
// (c), which exchange with a and not with each other, times the attenuation:
// e[0..8] row-major, and the recoveries grow[0..2]. NARROW: the series alone,
// in float; otherwise in double, the series where the roots are close and the
// Newton form at the three roots where they are not.
template <bool NARROW, class T>
__device__ __forceinline__ void three_pool_step(T r1_free, T r1_b, T r1_c, T exchange_b, T exchange_c,
                                                T fraction_b, T fraction_c, T dt, T attenuation, T* e,
                                                T* grow) {
    using W = typename WorkOf<T, NARROW>::type;
    using V = decltype(primal(W()));
    constexpr int TERMS = NARROW ? 24 : 16;
    const W step = W(dt);
    const W free = W(1.0f - fraction_b - fraction_c);
    const W pool_b = W(fraction_b), pool_c = W(fraction_c);
    const W kab = W(exchange_b) * pool_b, kba = W(exchange_b) * free;
    const W kac = W(exchange_c) * pool_c, kca = W(exchange_c) * free;
    const W a00 = (-kab - kac - W(r1_free)) * step, a01 = kba * step, a02 = kca * step;
    const W a10 = kab * step, a11 = (-kba - W(r1_b)) * step;
    const W a20 = kac * step, a22 = (-kca - W(r1_c)) * step;
    W third;
    if constexpr (NARROW) third = (a00 + a11 + a22) * V(1.0 / 3.0);
    else third = (a00 + a11 + a22) / V(3.0);
    const W s00 = a00 - third, s11 = a11 - third, s22 = a22 - third;
    const W minors = s00 * s11 - a01 * a10 + s00 * s22 - a02 * a20 + s11 * s22;
    const W determinant = s00 * s11 * s22 - a01 * (a10 * s22) + a02 * (-s11 * a20);
    W c[9];
    bool close = true;
    if constexpr (!NARROW) close = -2.0 * primal(minors) < 1.0;
    if (close) {
        W flat = V(1), linear = V(0), square = V(0), sum_flat = V(1), sum_linear = V(0), sum_square = V(0);
        series_terms<W, 1, TERMS>(determinant, minors, flat, linear, square, sum_flat, sum_linear,
                                  sum_square);
        const W q00 = s00 * s00 + a01 * a10 + a02 * a20, q01 = s00 * a01 + a01 * s11;
        const W q02 = s00 * a02 + a02 * s22, q10 = a10 * s00 + s11 * a10;
        const W q11 = a10 * a01 + s11 * s11, q12 = a10 * a02;
        const W q20 = a20 * s00 + s22 * a20, q21 = a20 * a01, q22 = a20 * a02 + s22 * s22;
        const W lift = exp_(third);
        c[0] = lift * (sum_flat + sum_linear * s00 + sum_square * q00);
        c[1] = lift * (sum_linear * a01 + sum_square * q01);
        c[2] = lift * (sum_linear * a02 + sum_square * q02);
        c[3] = lift * (sum_linear * a10 + sum_square * q10);
        c[4] = lift * (sum_flat + sum_linear * s11 + sum_square * q11);
        c[5] = lift * (sum_square * q12);
        c[6] = lift * (sum_linear * a20 + sum_square * q20);
        c[7] = lift * (sum_square * q21);
        c[8] = lift * (sum_flat + sum_linear * s22 + sum_square * q22);
    } else {
        if constexpr (!NARROW) {
            const W radius = sqrt_(max_(-minors * (1.0 / 3.0), W(1e-300)));
            const double limit = 1.0 - 1e-16;
            const W argument = min_(max_(0.5 * determinant / (radius * radius * radius), W(-limit)), W(limit));
            const W angle = num::acos_(argument) / 3.0;
            const double turn = 2.09439510239319549231;
            const W root_a = 2.0 * radius * num::cos_(angle) + third;
            const W root_b = 2.0 * radius * num::cos_(angle - turn) + third;
            const W root_c = 2.0 * radius * num::cos_(angle - 2.0 * turn) + third;
            const W low = min_(min_(root_a, root_b), root_c);
            const W high = max_(max_(root_a, root_b), root_c);
            const W middle = max_(min_(root_a, root_b), min_(max_(root_a, root_b), root_c));
            const W leading = exp_(low), centre = exp_(middle), trailing = exp_(high);
            const W first = exp_difference(low, middle, leading, centre);
            const W span = high - low;
            const W second =
                (exp_difference(middle, high, centre, trailing) - first) / (primal(span) > 0.0 ? span : W(1.0));
            const W m00 = a00 - low, m11 = a11 - low, m22 = a22 - low;
            const W n00 = a00 - middle, n11 = a11 - middle, n22 = a22 - middle;
            const W p00 = m00 * n00 + a01 * a10 + a02 * a20, p01 = m00 * a01 + a01 * n11;
            const W p02 = m00 * a02 + a02 * n22, p10 = a10 * n00 + m11 * a10;
            const W p11 = a10 * a01 + m11 * n11, p12 = a10 * a02;
            const W p20 = a20 * n00 + m22 * a20, p21 = a20 * a01;
            const W p22 = a20 * a02 + m22 * n22;
            c[0] = leading + first * m00 + second * p00;
            c[1] = first * a01 + second * p01;
            c[2] = first * a02 + second * p02;
            c[3] = first * a10 + second * p10;
            c[4] = leading + first * m11 + second * p11;
            c[5] = second * p12;
            c[6] = first * a20 + second * p20;
            c[7] = second * p21;
            c[8] = leading + first * m22 + second * p22;
        }
    }
    const W damp = W(attenuation);
    W d[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) d[k] = damp * c[k];
    grow[0] = T(free - (d[0] * free + d[1] * pool_b + d[2] * pool_c));
    grow[1] = T(pool_b - (d[3] * free + d[4] * pool_b + d[5] * pool_c));
    grow[2] = T(pool_c - (d[6] * free + d[7] * pool_b + d[8] * pool_c));
#pragma unroll
    for (int k = 0; k < 9; ++k) e[k] = T(d[k]);
}

// One interval's three-pool operator read from the table, undamped there.
// Along a direction the row carries the tissue's share and the interval's
// own is the generator times the operator.
template <class T>
__device__ __forceinline__ void three_pool_from_table(const float* table, int row, int atom, int voxels,
                                                      T dt, T attenuation, T r1_free, T r1_b, T r1_c,
                                                      T exchange_b, T exchange_c, T free, T pool_b,
                                                      T pool_c, T* e, T* grow) {
    if constexpr (DUAL<T>) {
        const float* base = table + static_cast<long long>(row) * (18 * voxels) + atom;
        float c[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) c[k] = __ldg(base + k * voxels);
        const float xb = exchange_b.v, xc = exchange_c.v, fb = pool_b.v, fc = pool_c.v;
        const float fa = 1.0f - fb - fc;
        const float a00 = -xb * fb - xc * fc - r1_free.v, a01 = xb * fa, a02 = xc * fa;
        const float a10 = xb * fb, a11 = -xb * fa - r1_b.v, a20 = xc * fc, a22 = -xc * fa - r1_c.v;
        const float g[9] = {a00 * c[0] + a01 * c[3] + a02 * c[6], a00 * c[1] + a01 * c[4] + a02 * c[7],
                            a00 * c[2] + a01 * c[5] + a02 * c[8], a10 * c[0] + a11 * c[3],
                            a10 * c[1] + a11 * c[4],             a10 * c[2] + a11 * c[5],
                            a20 * c[0] + a22 * c[6],             a20 * c[1] + a22 * c[7],
                            a20 * c[2] + a22 * c[8]};
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            e[k] = attenuation * T{c[k], __ldg(base + (9 + k) * voxels) + dt.d * g[k]};
        }
    } else {
        const float* base = table + static_cast<long long>(row) * (9 * voxels) + atom;
#pragma unroll
        for (int k = 0; k < 9; ++k) e[k] = attenuation * __ldg(base + k * voxels);
    }
    grow[0] = free - (e[0] * free + e[1] * pool_b + e[2] * pool_c);
    grow[1] = pool_b - (e[3] * free + e[4] * pool_b + e[5] * pool_c);
    grow[2] = pool_c - (e[6] * free + e[7] * pool_b + e[8] * pool_c);
}

// Cubic Hermite through a table of knots (value, slope) spaced ``step``,
// clamped at the far end.
template <class T>
__device__ __forceinline__ T hermite_weights(T scaled, float lower, float step, T& h10, T& h01, T& h11) {
    const T u = scaled - lower, u2 = u * u, u3 = u2 * u;
    h10 = (u3 - 2.0f * u2 + u) * step;
    h01 = -2.0f * u3 + 3.0f * u2;
    h11 = (u3 - u2) * step;
    return 2.0f * u3 - 3.0f * u2 + 1.0f;
}

// How well the semisolid pool absorbs a pulse this far off its resonance.
template <class T>
__device__ __forceinline__ T lineshape_at(const float* lineshape, T offset_hz, int bins, float step) {
    const int last = bins - 1;
    const T scaled = min_(div_(abs_(offset_hz), T(step)), T(last + 0.0f));
    const float lower = fminf(floorf(primal(scaled)), last - 1.0f);
    T h10, h01, h11;
    const T h00 = hermite_weights(scaled, lower, step, h10, h01, h11);
    const float* base = lineshape + static_cast<int>(lower) * 2;
    return h00 * __ldg(base) + h10 * __ldg(base + 1) + h01 * __ldg(base + 2) + h11 * __ldg(base + 3);
}

// Relaxation over one interval for one combination of its switches.
template <class T, int Y, bool SINGLE, bool MOVING, bool DIFFUSING, bool OFF_AXIS, bool MAPS>
__device__ __forceinline__ void relax(const Params& p, int event, T dt_shared, const bool* active,
                                      const int* train, const int* atom, float order, int state,
                                      const T* r1, const T* r2, const T* b0, T* fpr, T* fpi, T* fmr,
                                      T* fmi, T* zr, T* zi) {
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        T dt = dt_shared;
        if constexpr (!SINGLE) {
            dt = active[y] ? read<T>(p.duration, p.d_duration, train[y] * p.event_count + event) : T(0.0f);
        }
        const int at = p.atom_stride ? atom[y] : 0;
        T wout = 1.0f, velocity = 0.0f;
        if constexpr (MOVING) {
            velocity = active[y] ? read<T>(p.velocity, p.d_velocity, at) : T(0.0f);
            wout = 1.0f - min_(abs_(velocity) * p.washout_scale * dt, T(1.0f));
        }
        T e1 = exp_(-r1[y] * dt) * wout;
        T e2 = exp_(-r2[y] * dt) * wout;
        const T recovery = 1.0f - e1;
        if constexpr (DIFFUSING) {
            const T b = (active[y] ? read<T>(p.diffusion, p.d_diffusion, at) : T(0.0f)) * dt;
            const float sq = order * order;
            e1 *= exp_(-b * sq);
            e2 *= exp_(-b * (sq + order + 0.3333333333333333f));
        }
        T b0y = 0.0f;
        if constexpr (OFF_AXIS) b0y = MAPS ? b0[MAPS ? y : 0] : T(0.0f);
        if constexpr (MOVING || OFF_AXIS) {
            T turn = 0.0f;
            if constexpr (MOVING) turn = velocity * p.flow_scale * dt;
            T os, oc;
            sincos_(-2.0f * 3.141592653589793f * b0y * dt - (order + 0.5f) * turn, os, oc);
            T old = fpr[y];
            fpr[y] = e2 * (old * oc - fpi[y] * os);
            fpi[y] = e2 * (old * os + fpi[y] * oc);
            old = fmr[y];
            fmr[y] = e2 * (old * oc + fmi[y] * os);
            fmi[y] = e2 * (-old * os + fmi[y] * oc);
            if constexpr (MOVING) {
                T ts, tc;
                sincos_(-order * turn, ts, tc);
                old = zr[y];
                zr[y] = e1 * (old * tc - zi[y] * ts);
                zi[y] = e1 * (old * ts + zi[y] * tc);
            } else {
                zr[y] = e1 * zr[y];
                zi[y] = e1 * zi[y];
            }
        } else {
            fpr[y] = e2 * fpr[y];
            fpi[y] = e2 * fpi[y];
            fmr[y] = e2 * fmr[y];
            fmi[y] = e2 * fmi[y];
            zr[y] = e1 * zr[y];
            zi[y] = e1 * zi[y];
        }
        if (state == 0) zr[y] += recovery;
    }
}

// Off-resonance alone over one train: the interval repeats, and with it the
// two factors and the precession, which are then taken once per repeat.
template <class T, int Y>
__device__ __forceinline__ void relax_off_axis_repeat(T dt, T& last_dt, int state, const T* r1, const T* r2,
                                                      const T* b0, T* e1c, T* e2c, T* occ, T* osc, T* fpr,
                                                      T* fpi, T* fmr, T* fmi, T* zr, T* zi) {
    if (!same(dt, last_dt)) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            e1c[y] = exp_(-r1[y] * dt);
            e2c[y] = exp_(-r2[y] * dt);
            sincos_(-2.0f * 3.141592653589793f * b0[y] * dt, osc[y], occ[y]);
        }
        last_dt = dt;
    }
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        const T e2 = e2c[y], oc = occ[y], os = osc[y];
        T old = fpr[y];
        fpr[y] = e2 * (old * oc - fpi[y] * os);
        fpi[y] = e2 * (old * os + fpi[y] * oc);
        old = fmr[y];
        fmr[y] = e2 * (old * oc + fmi[y] * os);
        fmi[y] = e2 * (-old * os + fmi[y] * oc);
        zr[y] = e1c[y] * zr[y];
        zi[y] = e1c[y] * zi[y];
        if (state == 0) zr[y] += 1.0f - e1c[y];
    }
}

// What a thread's problems are made of beyond free water, read once.
template <class T, int POOLS, int Y>
struct PoolTissue {
    static constexpr int YL = POOLS >= 1 ? Y : 1, YB = POOLS >= 2 ? Y : 1, YC = POOLS == 3 ? Y : 1;
    T fraction_b[YL], exchange_b[YL], r1_b[YL], free[YL];
    T r2_b[YB], shift_b[YB];
    T fraction_c[YC], exchange_c[YC], r1_c[YC];
};

// One interval's operators for each problem a thread holds, with the
// damping, precession and flow of the thread's own order folded in; kept
// while the interval repeats. Transverse: a complex factor (POOLS 1) or a
// complex 2x2 (POOLS 2, 3). Longitudinal: the exchange matrix, its
// recoveries, and the complex factor flow and damping put on it.
template <class T, int POOLS, int Y>
struct PoolOperators {
    static constexpr int NT = POOLS >= 2 ? 8 : 2, NE = POOLS == 3 ? 9 : 4, NG = POOLS == 3 ? 3 : 2;
    T t[Y][NT], e[Y][NE], grow[Y][NG], factor_r[Y], factor_i[Y];
};

// One problem's operators over one interval at this lane's order, out of
// line: an interval is formed only when it changes, and one copy serves every
// kernel of the layout. The switches are read at run time for the same
// reason. Everything goes in and comes out by value.
template <class T, int POOLS>
struct OneOperator {
    static constexpr int NT = POOLS >= 2 ? 8 : 2, NE = POOLS == 3 ? 9 : 4, NG = POOLS == 3 ? 3 : 2;
    T t[NT], e[NE], grow[NG], factor_r, factor_i, damp_z, wout;
};
template <class T>
struct OneTissue {
    T r1, r2, b0, velocity, diffusion, fraction_b, exchange_b, r1_b, free, r2_b, shift_b, fraction_c, exchange_c, r1_c;
};
struct OperatorGeometry {
    float flow_scale, washout_scale;
    const float* pool_table;
    int atom_count;
};

template <class T, int POOLS, int MODE>
__device__ __noinline__ OneOperator<T, POOLS> one_operator(OperatorGeometry g, int relax_code, T dt, int row, int atom,
                                                           float order, OneTissue<T> in, bool three_elsewhere) {
    const bool moving = (relax_code & 4) != 0, diffusing = (relax_code & 2) != 0, off_axis = (relax_code & 1) != 0;
    OneOperator<T, POOLS> out;
    T wout = 1.0f, turn = 0.0f;
    if (moving) {
        wout = 1.0f - min_(abs_(in.velocity) * g.washout_scale * dt, T(1.0f));
        turn = in.velocity * g.flow_scale * dt;
    }
    T damp_z = 1.0f, damp_t = 1.0f;
    if (diffusing) {
        const T b = in.diffusion * dt;
        const float sq = order * order;
        damp_z = exp_(-b * sq);
        damp_t = exp_(-b * (sq + order + 0.3333333333333333f));
    }
    T oc = 1.0f, os = 0.0f;
    if (moving || off_axis) sincos_(-TWO_PI * in.b0 * dt - (order + 0.5f) * turn, os, oc);
    if constexpr (POOLS == 1) {
        const T e2 = exp_(-in.r2 * dt) * wout * damp_t;
        out.t[0] = e2 * oc;
        out.t[1] = e2 * os;
    } else {
        T x[8];
        transverse_step(in.r2, in.r2_b, in.exchange_b, in.fraction_b, in.free, in.shift_b, dt, wout, x);
        const T rr = damp_t * oc, ri = damp_t * os;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            out.t[2 * k] = rr * x[2 * k] - ri * x[2 * k + 1];
            out.t[2 * k + 1] = rr * x[2 * k + 1] + ri * x[2 * k];
        }
    }
    if constexpr (POOLS == 3) {
        if constexpr (MODE == TABLE) {
            three_pool_from_table(g.pool_table, row, atom, g.atom_count, dt, wout, in.r1, in.r1_b, in.r1_c, in.exchange_b,
                                  in.exchange_c, in.free, in.fraction_b, in.fraction_c, out.e, out.grow);
        } else if (!three_elsewhere) {
            three_pool_step<MODE == NARROW>(in.r1, in.r1_b, in.r1_c, in.exchange_b, in.exchange_c, in.fraction_b,
                                            in.fraction_c, dt, wout, out.e, out.grow);
        }
    } else {
        two_pool_step(in.r1, in.r1_b, in.exchange_b, in.fraction_b, dt, wout, out.e, out.grow);
    }
    out.damp_z = damp_z;
    out.wout = wout;
    out.factor_r = damp_z;
    out.factor_i = 0.0f;
    if (moving) {
        T ts, tc;
        sincos_(-order * turn, ts, tc);
        out.factor_r = damp_z * tc;
        out.factor_i = damp_z * ts;
    }
    return out;
}

template <class T, int POOLS, int MODE, int Y, bool MAPS>
__device__ __forceinline__ void pool_operators(const Params& p, int relax_code, const T* dts, const int* rows,
                                               const bool* active, const int* atom, int state, float order,
                                               const T* r1, const T* r2, const T* b0,
                                               const PoolTissue<T, POOLS, Y>& tissue,
                                               PoolOperators<T, POOLS, Y>& ops) {
    // A row of at least Y states spreads the three-pool step over its lanes
    // where it is formed in double; in float the shuffles cost what they save.
    const bool SPREAD = POOLS == 3 && MODE == ROOTS && Y > 1 && p.width >= Y;
    const bool moving = (relax_code & 4) != 0;
    const OperatorGeometry g{p.flow_scale, p.washout_scale, p.pool_table, p.atom_count};
    T attenuations[Y], damps[Y];
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        const int at = p.atom_stride ? atom[y] : 0;
        OneTissue<T> in;
        in.r1 = r1[y];
        in.r2 = r2[y];
        in.b0 = MAPS && p.off_axis ? b0[MAPS ? y : 0] : T(0.0f);
        in.velocity = p.moving && active[y] ? read<T>(p.velocity, p.d_velocity, at) : T(0.0f);
        in.diffusion = p.diffusing && active[y] ? read<T>(p.diffusion, p.d_diffusion, at) : T(0.0f);
        in.fraction_b = tissue.fraction_b[y];
        in.exchange_b = tissue.exchange_b[y];
        in.r1_b = tissue.r1_b[y];
        in.free = tissue.free[y];
        if constexpr (POOLS >= 2) {
            in.r2_b = tissue.r2_b[y];
            in.shift_b = tissue.shift_b[y];
        } else {
            in.r2_b = in.shift_b = 0.0f;
        }
        if constexpr (POOLS == 3) {
            in.fraction_c = tissue.fraction_c[y];
            in.exchange_c = tissue.exchange_c[y];
            in.r1_c = tissue.r1_c[y];
        } else {
            in.fraction_c = in.exchange_c = in.r1_c = 0.0f;
        }
        const OneOperator<T, POOLS> one = one_operator<T, POOLS, MODE>(g, relax_code, dts[y], rows[y], atom[y], order, in, SPREAD);
#pragma unroll
        for (int k = 0; k < OneOperator<T, POOLS>::NT; ++k) ops.t[y][k] = one.t[k];
#pragma unroll
        for (int k = 0; k < OneOperator<T, POOLS>::NE; ++k) ops.e[y][k] = one.e[k];
#pragma unroll
        for (int k = 0; k < OneOperator<T, POOLS>::NG; ++k) ops.grow[y][k] = one.grow[k];
        ops.factor_r[y] = one.factor_r;
        ops.factor_i[y] = one.factor_i;
        damps[y] = one.damp_z;
        attenuations[y] = one.wout;
    }
    if constexpr (POOLS == 3 && MODE == ROOTS && Y > 1) {
        if (SPREAD) {
            // The group's lanes share their problems, so each forms one
            // problem's three-pool operator and the others read it: a
            // Y-th of the work, which in double is most of the interval's.
            const int mine = state % Y;
            auto pick = [&](const T* values) {
                T picked = values[0];
#pragma unroll
                for (int y = 1; y < Y; ++y) picked = mine == y ? values[y] : picked;
                return picked;
            };
            const T e_r1 = pick(r1), e_r1b = pick(tissue.r1_b), e_r1c = pick(tissue.r1_c);
            const T e_xb = pick(tissue.exchange_b), e_xc = pick(tissue.exchange_c);
            const T e_fb = pick(tissue.fraction_b), e_fc = pick(tissue.fraction_c);
            const T e_dt = pick(dts), e_wout = pick(attenuations);
            T e[9], grow[3];
            three_pool_step<MODE == NARROW>(e_r1, e_r1b, e_r1c, e_xb, e_xc, e_fb, e_fc, e_dt, e_wout, e, grow);
            const unsigned mask = __activemask();
            const int first_lane = (threadIdx.x & 31) - state;
#pragma unroll
            for (int y = 0; y < Y; ++y) {
#pragma unroll
                for (int k = 0; k < 9; ++k) ops.e[y][k] = num::shfl_from(e[k], first_lane + y, mask);
#pragma unroll
                for (int k = 0; k < 3; ++k) ops.grow[y][k] = num::shfl_from(grow[k], first_lane + y, mask);
            }
        }
    }
    if (!moving) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
#pragma unroll
            for (int k = 0; k < PoolOperators<T, POOLS, Y>::NE; ++k) ops.e[y][k] *= damps[y];
        }
    }
}

template <class T, int POOLS, int Y, bool MOVING>
__device__ __forceinline__ void apply_pools(const PoolOperators<T, POOLS, Y>& ops, int state, T* fpr, T* fpi,
                                            T* fmr, T* fmi, T* zr, T* zi, T* bpr, T* bpi, T* bmr, T* bmi,
                                            T* lr, T* li, T* cr, T* ci) {
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        const T* t = ops.t[y];
        if constexpr (POOLS == 1) {
            const T pr = t[0] * fpr[y] - t[1] * fpi[y], pi = t[0] * fpi[y] + t[1] * fpr[y];
            const T mr = t[0] * fmr[y] + t[1] * fmi[y], mi = t[0] * fmi[y] - t[1] * fmr[y];
            fpr[y] = pr; fpi[y] = pi; fmr[y] = mr; fmi[y] = mi;
        } else {
            const T pr = t[0] * fpr[y] - t[1] * fpi[y] + t[2] * bpr[y] - t[3] * bpi[y];
            const T pi = t[0] * fpi[y] + t[1] * fpr[y] + t[2] * bpi[y] + t[3] * bpr[y];
            const T qr = t[4] * fpr[y] - t[5] * fpi[y] + t[6] * bpr[y] - t[7] * bpi[y];
            const T qi = t[4] * fpi[y] + t[5] * fpr[y] + t[6] * bpi[y] + t[7] * bpr[y];
            const T mr = t[0] * fmr[y] + t[1] * fmi[y] + t[2] * bmr[y] + t[3] * bmi[y];
            const T mi = t[0] * fmi[y] - t[1] * fmr[y] + t[2] * bmi[y] - t[3] * bmr[y];
            const T nr = t[4] * fmr[y] + t[5] * fmi[y] + t[6] * bmr[y] + t[7] * bmi[y];
            const T ni = t[4] * fmi[y] - t[5] * fmr[y] + t[6] * bmi[y] - t[7] * bmr[y];
            fpr[y] = pr; fpi[y] = pi; bpr[y] = qr; bpi[y] = qi;
            fmr[y] = mr; fmi[y] = mi; bmr[y] = nr; bmi[y] = ni;
        }
        const T* e = ops.e[y];
        T ar, ai, br, bi, cr2 = 0.0f, ci2 = 0.0f;
        if constexpr (POOLS == 3) {
            ar = e[0] * zr[y] + e[1] * lr[y] + e[2] * cr[y];
            ai = e[0] * zi[y] + e[1] * li[y] + e[2] * ci[y];
            br = e[3] * zr[y] + e[4] * lr[y] + e[5] * cr[y];
            bi = e[3] * zi[y] + e[4] * li[y] + e[5] * ci[y];
            cr2 = e[6] * zr[y] + e[7] * lr[y] + e[8] * cr[y];
            ci2 = e[6] * zi[y] + e[7] * li[y] + e[8] * ci[y];
        } else {
            ar = e[0] * zr[y] + e[1] * lr[y];
            ai = e[0] * zi[y] + e[1] * li[y];
            br = e[2] * zr[y] + e[3] * lr[y];
            bi = e[2] * zi[y] + e[3] * li[y];
        }
        if constexpr (MOVING) {
            const T fr = ops.factor_r[y], fi = ops.factor_i[y];
            zr[y] = fr * ar - fi * ai; zi[y] = fr * ai + fi * ar;
            lr[y] = fr * br - fi * bi; li[y] = fr * bi + fi * br;
            if constexpr (POOLS == 3) {
                cr[y] = fr * cr2 - fi * ci2;
                ci[y] = fr * ci2 + fi * cr2;
            }
        } else {
            zr[y] = ar; zi[y] = ai; lr[y] = br; li[y] = bi;
            if constexpr (POOLS == 3) {
                cr[y] = cr2;
                ci[y] = ci2;
            }
        }
        if (state == 0) {
            zr[y] += ops.grow[y][0];
            lr[y] += ops.grow[y][1];
            if constexpr (POOLS == 3) cr[y] += ops.grow[y][2];
        }
    }
}

// One pulse for one combination of its switches: the thread's train row or
// a row per problem, one shim or several, and whether every problem a thread
// holds sees the same rotation.
template <class T, int Y, int RF, bool MAPS, bool SINGLE, bool SHIMMED, bool SHARED, int POOLS>
__device__ __forceinline__ void pulse(const Params& p, int event, int base, const bool* active, const int* train,
                                      const int* atom, const int* location, const T* b1, const T* b1c,
                                      const T* b1s, const T* b0, T* fpr, T* fpi, T* fmr, T* fmi, T* zr, T* zi,
                                      T* bpr, T* bpi, T* bmr, T* bmi, T* lr, T* li, T* cr, T* ci) {
    constexpr bool SATURATES = POOLS == 1 || POOLS == 3;
    T flip_u = 0.0f, ce_u = 0.0f, se_u = 0.0f;
    if constexpr (SINGLE) {
        flip_u = read<T>(p.flip, p.d_flip, base + event);
        event_phase(p, base + event, ce_u, se_u);
    }
    int profile_row = 0;
    if constexpr (RF == PROFILE) profile_row = p.profile_index[event] * p.locations;
    int shim_row = 0;
    if constexpr (SHIMMED) shim_row = p.shim_index[event] * p.atom_count;
    float saturation = 0.0f, rf_frequency = 0.0f;
    T absorbed_shared = 1.0f, shape_shared = 0.0f;
    if constexpr (SATURATES) {
        saturation = p.saturation[event];
        rf_frequency = p.rf_frequency[event];
        // With no off-resonance map every voxel sits at the pulse's own
        // offset, and the line shape is read once.
        if (!MAPS || !p.off_axis) {
            shape_shared = lineshape_at(p.lineshape, T(rf_frequency), p.lineshape_bins, p.lineshape_step);
        }
    }
    T sine = 0.0f, cosine = 1.0f, cphi = 1.0f, sphi = 0.0f, c2 = 1.0f, s2 = 0.0f;
    if constexpr (SHARED) {
        cphi = ce_u;
        sphi = se_u;
        if constexpr (RF == HARD) sincos_(flip_u, sine, cosine);
        c2 = fma_(cphi, cphi, -(sphi * sphi));
        s2 = 2.0f * sphi * cphi;
        if constexpr (SATURATES) absorbed_shared = exp_(saturation * flip_u * flip_u * shape_shared);
    }
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        T alpha;
        if constexpr (SHARED) {
            alpha = flip_u;
        } else {
            T flip_v = flip_u, ce = ce_u, se = se_u;
            if constexpr (!SINGLE) {
                const int e = active[y] ? train[y] * p.event_count + event : event;
                flip_v = read<T>(p.flip, p.d_flip, e);
                event_phase(p, e, ce, se);
            }
            T tb1 = 1.0f, tc = 1.0f, ts = 0.0f;
            if constexpr (MAPS) {
                tb1 = b1[y];
                tc = b1c[y];
                ts = b1s[y];
            }
            if constexpr (SHIMMED) {
                const int cell = shim_row + atom[y];
                tb1 = p.transmit ? (active[y] ? read<T>(p.b1, p.d_b1, cell) : T(1.0f)) : T(1.0f);
                if (p.off_axis) sincos_(active[y] ? read<T>(p.b1_phase, p.d_b1_phase, cell) : T(0.0f), ts, tc);
            }
            alpha = flip_v * tb1;
            cphi = fma_(ce, tc, -(se * ts));
            sphi = fma_(se, tc, ce * ts);
            if constexpr (RF == HARD) sincos_(alpha, sine, cosine);
            c2 = fma_(cphi, cphi, -(sphi * sphi));
            s2 = 2.0f * sphi * cphi;
        }
        if constexpr (RF == HARD) {
            rotate_flip_phase(cosine, sine, cphi, sphi, c2, s2, fpr[y], fpi[y], fmr[y], fmi[y], zr[y], zi[y]);
            if constexpr (POOLS >= 2) {
                rotate_flip_phase(cosine, sine, cphi, sphi, c2, s2, bpr[y], bpi[y], bmr[y], bmi[y], lr[y], li[y]);
            }
        } else {
            T pair[4];
            if constexpr (RF == PROFILE) {
                const int row = profile_row + location[y];
                const int last = p.profile_bins - 1;
                const T scaled = min_(max_(div_(alpha, T(p.profile_step)), T(0.0f)), T(last + 0.0f));
                const float lower = fminf(floorf(primal(scaled)), last - 1.0f);
                T h10, h01, h11;
                const T h00 = hermite_weights(scaled, lower, p.profile_step, h10, h01, h11);
                const float* base_row = p.profile + (row * p.profile_bins + static_cast<int>(lower)) * 8;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    pair[c] = h00 * __ldg(base_row + c) + h10 * __ldg(base_row + 4 + c) +
                              h01 * __ldg(base_row + 8 + c) + h11 * __ldg(base_row + 12 + c);
                }
            } else {
                // Integrated per pulse per voxel, so the read is the pair.
                const int row = p.pair_index[(active[y] ? train[y] * p.event_count : 0) + event];
                const long long cell = static_cast<long long>(row) * p.atom_count + atom[y];
                const float4 entry = __ldg(reinterpret_cast<const float4*>(p.pairs) + cell);
                if constexpr (DUAL<T>) {
                    const float4 direction = __ldg(reinterpret_cast<const float4*>(p.pair_direction) + cell);
                    pair[0] = T{entry.x, direction.x};
                    pair[1] = T{entry.y, direction.y};
                    pair[2] = T{entry.z, direction.z};
                    pair[3] = T{entry.w, direction.w};
                } else {
                    pair[0] = entry.x;
                    pair[1] = entry.y;
                    pair[2] = entry.z;
                    pair[3] = entry.w;
                }
            }
            const T turn_r = cphi, turn_i = -sphi;
            const T sbr = pair[2] * turn_r - pair[3] * turn_i;
            const T sbi = pair[2] * turn_i + pair[3] * turn_r;
            rotate_spinor(pair[0], pair[1], sbr, sbi, fpr[y], fpi[y], fmr[y], fmi[y], zr[y], zi[y]);
            if constexpr (POOLS >= 2) {
                rotate_spinor(pair[0], pair[1], sbr, sbi, bpr[y], bpi[y], bmr[y], bmi[y], lr[y], li[y]);
            }
        }
        if constexpr (SATURATES) {
            // The semisolid pool absorbs the power the pulse deposits at the
            // bare flip the transmit field gives the voxel.
            T absorbed = absorbed_shared;
            if constexpr (!SHARED) {
                T shape = shape_shared;
                if constexpr (MAPS) {
                    if (p.off_axis) {
                        shape = lineshape_at(p.lineshape, rf_frequency - b0[y], p.lineshape_bins, p.lineshape_step);
                    }
                }
                absorbed = exp_(saturation * alpha * alpha * shape);
            }
            if constexpr (POOLS == 1) {
                lr[y] *= absorbed;
                li[y] *= absorbed;
            } else {
                cr[y] *= absorbed;
                ci[y] *= absorbed;
            }
        }
    }
}
#define EPG_PULSE_ARGS                                                                               \
    p, event, base, active, train, atom, location, b1, b1c, b1s, b0, fpr, fpi, fmr, fmi, zr, zi, bpr, bpi, \
        bmr, bmi, lr, li, cr, ci
#define EPG_RELAX_ARGS p, event, dt_shared, active, train, atom, order, state, r1, r2, b0, fpr, fpi, fmr, fmi, zr, zi
#define EPG_POOL_ARGS p, dts, rows, active, atom, state, order, r1, r2, b0, tissue, ops

template <class T, int POOLS, int MODE, int RF, bool MAPS, int Y, bool ONE_TRAIN>
__device__ __forceinline__ void complex_loop(const Params& p) {
    constexpr int YL = POOLS >= 1 ? Y : 1, YB = POOLS >= 2 ? Y : 1, YC = POOLS == 3 ? Y : 1;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int width = p.width;
    const int state = lane & (width - 1);
    const int group = lane / width;
    const int groups = 32 / width;
    const int first = ((blockIdx.x * (blockDim.x >> 5) + warp) * groups + group) * Y;
    const float order = static_cast<float>(state);
    const int total = p.train_count * p.atom_count;

    int problem[Y], atom[Y], train[Y], location[Y];
    bool active[Y], live[Y];
    T r1[Y], r2[Y];
    // A mapped tissue holds its maps, and the transmit phase's rotation, in
    // registers from the start; a uniform one holds none of them.
    T m0[MAPS ? Y : 1], b1[MAPS ? Y : 1], b1c[MAPS ? Y : 1], b1s[MAPS ? Y : 1];
    T b0[MAPS ? Y : 1], inversion[MAPS ? Y : 1];
    T fpr[Y], fpi[Y], fmr[Y], fmi[Y], zr[Y], zi[Y];
    // Pool b's transverse and longitudinal states, or the semisolid pool's
    // longitudinal ones where it is the only second pool; then the semisolid
    // pool's beside pool b.
    T bpr[YB], bpi[YB], bmr[YB], bmi[YB], lr[YL], li[YL], cr[YC], ci[YC];
    PoolTissue<T, POOLS, Y> tissue;
#pragma unroll
    for (int y = 0; y < Y; ++y) {
        problem[y] = first + y;
        active[y] = problem[y] < total;
        live[y] = active[y] && state < p.state_count;
        atom[y] = problem[y] % p.atom_count;
        train[y] = problem[y] / p.atom_count;
        // A voxel's place along the slice, which picks its row of a table.
        location[y] = RF == PROFILE ? atom[y] % p.locations : 0;
        r1[y] = num::rate(active[y] ? read<T>(p.t1, p.d_t1, atom[y]) : T(1.0f));
        r2[y] = num::rate(active[y] ? read<T>(p.t2, p.d_t2, atom[y]) : T(1.0f));
        const int at = p.atom_stride ? atom[y] : 0;
        if constexpr (MAPS) {
            m0[y] = p.density ? (active[y] ? read<T>(p.m0, p.d_m0, at) : T(0.0f)) : T(1.0f);
            b1[y] = p.transmit ? (active[y] ? read<T>(p.b1, p.d_b1, at) : T(1.0f)) : T(1.0f);
            const T b1_phase = p.off_axis ? (active[y] ? read<T>(p.b1_phase, p.d_b1_phase, at) : T(0.0f)) : T(0.0f);
            if constexpr (DUAL<T>) {
                sincos_(b1_phase, b1s[y], b1c[y]);
            } else {
                sincosf(b1_phase, &b1s[y], &b1c[y]);
            }
            b0[y] = p.off_axis ? (active[y] ? read<T>(p.b0, p.d_b0, at) : T(0.0f)) : T(0.0f);
            inversion[y] = p.inverting ? (active[y] ? read<T>(p.inversion_efficiency, p.d_inversion_efficiency, at)
                                                    : T(1.0f))
                                       : T(1.0f);
        }
        T free = 1.0f;
        if constexpr (POOLS >= 1) {
            const float* fraction = POOLS == 1 ? p.bound_fraction : p.pool_b_fraction;
            const float* d_fraction = POOLS == 1 ? p.d_bound_fraction : p.d_pool_b_fraction;
            const float* exchange = POOLS == 1 ? p.bound_exchange : p.pool_b_exchange;
            const float* d_exchange = POOLS == 1 ? p.d_bound_exchange : p.d_pool_b_exchange;
            const float* t1_second = POOLS == 1 ? p.t1_bound : p.t1_pool_b;
            const float* d_t1_second = POOLS == 1 ? p.d_t1_bound : p.d_t1_pool_b;
            tissue.fraction_b[y] = active[y] ? read<T>(fraction, d_fraction, at) : T(0.0f);
            tissue.exchange_b[y] = active[y] ? read<T>(exchange, d_exchange, at) : T(0.0f);
            tissue.r1_b[y] = num::rate(active[y] ? read<T>(t1_second, d_t1_second, at) : T(1.0f));
            free = 1.0f - tissue.fraction_b[y];
            if constexpr (POOLS >= 2) {
                tissue.r2_b[y] = num::rate(active[y] ? read<T>(p.t2_pool_b, p.d_t2_pool_b, at) : T(1.0f));
                tissue.shift_b[y] = active[y] ? read<T>(p.pool_b_shift, p.d_pool_b_shift, at) : T(0.0f);
            }
            if constexpr (POOLS == 3) {
                tissue.fraction_c[y] = active[y] ? read<T>(p.bound_fraction, p.d_bound_fraction, at) : T(0.0f);
                tissue.exchange_c[y] = active[y] ? read<T>(p.bound_exchange, p.d_bound_exchange, at) : T(0.0f);
                tissue.r1_c[y] = num::rate(active[y] ? read<T>(p.t1_bound, p.d_t1_bound, at) : T(1.0f));
                free = 1.0f - tissue.fraction_b[y] - tissue.fraction_c[y];
            }
            tissue.free[y] = free;
            lr[y] = state == 0 ? tissue.fraction_b[y] : T(0.0f);
            li[y] = 0.0f;
        }
        if constexpr (POOLS >= 2) bpr[y] = bpi[y] = bmr[y] = bmi[y] = 0.0f;
        if constexpr (POOLS == 3) {
            cr[y] = state == 0 ? tissue.fraction_c[y] : T(0.0f);
            ci[y] = 0.0f;
        }
        fpr[y] = fpi[y] = fmr[y] = fmi[y] = zi[y] = 0.0f;
        zr[y] = state == 0 ? free : T(0.0f);
    }
    // A thread's problems are consecutive, so they are almost always voxels
    // of one train, and then an event's duration, flip and phase are one
    // read for all of them: ``base`` is that train's row.
    const bool uniform = ONE_TRAIN || (active[0] && train[0] == train[Y - 1]);
    const int base = ONE_TRAIN ? 0 : train[0] * p.event_count;
    // The pulse's flip and phase are then the same for every problem a
    // thread holds unless a transmit field or its phase moves them, and the
    // trigonometry is taken once.
    const bool shared_pulse = RF != DYNAMIC && uniform && !p.transmit && !p.off_axis && !p.shimmed;
    const bool plain_relax = !p.off_axis && !p.moving && !p.diffusing;
    const int relax_code = (p.moving ? 4 : 0) | (p.diffusing ? 2 : 0) | (p.off_axis ? 1 : 0);
    T e1c[Y], e2c[Y];
    T occ[MAPS ? Y : 1], osc[MAPS ? Y : 1];
    T last_dt = -1.0f;
    int last_row = -1;
    PoolOperators<T, POOLS ? POOLS : 1, POOLS ? Y : 1> ops;

    auto shift_pair = [&](T* pr_, T* pi_, T* mr_, T* mi_) {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            const T up_r = num::shfl_up(pr_[y], 1, width);
            const T up_i = num::shfl_up(pi_[y], 1, width);
            const T down_r = num::shfl_down(mr_[y], 1, width);
            const T down_i = num::shfl_down(mi_[y], 1, width);
            const bool keep_up = state > 0 && live[y];
            const bool keep_down = state + 1 < p.state_count && live[y];
            const T pr = keep_up ? up_r : T(0.0f), pi = keep_up ? up_i : T(0.0f);
            const T mr = keep_down ? down_r : T(0.0f), mi = keep_down ? down_i : T(0.0f);
            pr_[y] = state == 0 ? mr : pr;
            pi_[y] = state == 0 ? -mi : pi;
            mr_[y] = mr;
            mi_[y] = mi;
        }
    };
    auto shift = [&]() {
        shift_pair(fpr, fpi, fmr, fmi);
        if constexpr (POOLS >= 2) shift_pair(bpr, bpi, bmr, bmi);
    };

    float held_r[Y], held_i[Y];
    int run_start = 0, run_len = 0;
    auto flush = [&]() {
#pragma unroll
        for (int y = 0; y < Y; ++y) {
            if (state < run_len && active[y]) {
                const long at_out = static_cast<long>(problem[y]) * p.output_count + run_start + state;
                p.output_real[at_out] = held_r[y];
                p.output_imag[at_out] = held_i[y];
            }
        }
        run_len = 0;
    };
#pragma unroll((MAPS || POOLS || DUAL<T>) ? 1 : 2)
    for (int event = 0; event < p.event_count; ++event) {
        const T dt_shared = uniform ? read<T>(p.duration, p.d_duration, base + event) : T(0.0f);
        const unsigned char act = p.action[event];
        const int kind = p.kind[event];
        // Relaxation over the interval; an interval of no length, along no
        // direction, leaves every state exactly where it is. Each switch
        // branches around a whole block once per event, so the common case
        // runs only its own terms.
        if (!(uniform && same(dt_shared, T(0.0f)))) {
            if constexpr (POOLS > 0) {
                // The operators are a property of the interval, so one train
                // whose interval repeats forms them once per repeat.
                int row_shared = 0;
                if constexpr (MODE == TABLE) row_shared = uniform ? p.duration_row[base + event] : 0;
                const bool fresh =
                    !uniform || !same(dt_shared, last_dt) || (MODE == TABLE && row_shared != last_row);
                if (fresh) {
                    T dts[Y];
                    int rows[Y];
#pragma unroll
                    for (int y = 0; y < Y; ++y) {
                        const int e = active[y] ? train[y] * p.event_count + event : event;
                        dts[y] = uniform ? dt_shared : (active[y] ? read<T>(p.duration, p.d_duration, e) : T(0.0f));
                        rows[y] = 0;
                        if constexpr (MODE == TABLE) rows[y] = uniform ? row_shared : p.duration_row[e];
                    }
                    pool_operators<T, POOLS, MODE, Y, MAPS>(p, relax_code, dts, rows, active, atom, state, order, r1, r2,
                                                            b0, tissue, ops);
                    last_dt = uniform ? dt_shared : T(-1.0f);
                    last_row = row_shared;
                }
                if (p.moving) {
                    apply_pools<T, POOLS, Y, true>(ops, state, fpr, fpi, fmr, fmi, zr, zi, bpr, bpi, bmr, bmi, lr,
                                                   li, cr, ci);
                } else {
                    apply_pools<T, POOLS, Y, false>(ops, state, fpr, fpi, fmr, fmi, zr, zi, bpr, bpi, bmr, bmi, lr,
                                                    li, cr, ci);
                }
            } else if (plain_relax) {
                // No off-resonance, flow or diffusion: two factors per
                // problem, reused while the interval repeats.
                if (!uniform || !same(dt_shared, last_dt)) {
#pragma unroll
                    for (int y = 0; y < Y; ++y) {
                        const T dt =
                            uniform ? dt_shared
                                    : (active[y] ? read<T>(p.duration, p.d_duration, train[y] * p.event_count + event)
                                                 : T(0.0f));
                        e1c[y] = exp_(-r1[y] * dt);
                        e2c[y] = exp_(-r2[y] * dt);
                    }
                    last_dt = uniform ? dt_shared : T(-1.0f);
                }
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    fpr[y] = e2c[y] * fpr[y];
                    fpi[y] = e2c[y] * fpi[y];
                    fmr[y] = e2c[y] * fmr[y];
                    fmi[y] = e2c[y] * fmi[y];
                    zr[y] = e1c[y] * zr[y];
                    zi[y] = e1c[y] * zi[y];
                    if (state == 0) zr[y] += 1.0f - e1c[y];
                }
            } else {
                // One variant per combination of the interval's switches,
                // chosen once per event: each runs only its own terms.
                if (uniform) {
                    switch (relax_code) {
                        case 1:
                            if constexpr (MAPS) {
                                relax_off_axis_repeat<T, Y>(dt_shared, last_dt, state, r1, r2, b0, e1c, e2c, occ,
                                                            osc, fpr, fpi, fmr, fmi, zr, zi);
                            } else {
                                relax<T, Y, true, false, false, true, MAPS>(EPG_RELAX_ARGS);
                            }
                            break;
                        case 2: relax<T, Y, true, false, true, false, MAPS>(EPG_RELAX_ARGS); break;
                        case 3: relax<T, Y, true, false, true, true, MAPS>(EPG_RELAX_ARGS); break;
                        case 4: relax<T, Y, true, true, false, false, MAPS>(EPG_RELAX_ARGS); break;
                        case 5: relax<T, Y, true, true, false, true, MAPS>(EPG_RELAX_ARGS); break;
                        case 6: relax<T, Y, true, true, true, false, MAPS>(EPG_RELAX_ARGS); break;
                        default: relax<T, Y, true, true, true, true, MAPS>(EPG_RELAX_ARGS); break;
                    }
                } else {
                    switch (relax_code) {
                        case 1: relax<T, Y, false, false, false, true, MAPS>(EPG_RELAX_ARGS); break;
                        case 2: relax<T, Y, false, false, true, false, MAPS>(EPG_RELAX_ARGS); break;
                        case 3: relax<T, Y, false, false, true, true, MAPS>(EPG_RELAX_ARGS); break;
                        case 4: relax<T, Y, false, true, false, false, MAPS>(EPG_RELAX_ARGS); break;
                        case 5: relax<T, Y, false, true, false, true, MAPS>(EPG_RELAX_ARGS); break;
                        case 6: relax<T, Y, false, true, true, false, MAPS>(EPG_RELAX_ARGS); break;
                        default: relax<T, Y, false, true, true, true, MAPS>(EPG_RELAX_ARGS); break;
                    }
                }
            }
        }
        if (act & 1) shift();
        if (kind == 1 && (act & 4)) {
            // Pool b is free water and turns over like any other; a semisolid
            // pool is saturated instead.
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                const T eff = MAPS ? inversion[MAPS ? y : 0] : T(1.0f);
                zr[y] = -eff * zr[y];
                zi[y] = -eff * zi[y];
                if constexpr (POOLS >= 2) {
                    lr[y] = -eff * lr[y];
                    li[y] = -eff * li[y];
                }
            }
        }
        if (kind == 1 && !(act & 4)) {
            if (shared_pulse) {
                if constexpr (RF != DYNAMIC) pulse<T, Y, RF, MAPS, true, false, true, POOLS>(EPG_PULSE_ARGS);
            } else if (p.shimmed) {
                if (uniform) pulse<T, Y, RF, MAPS, true, true, false, POOLS>(EPG_PULSE_ARGS);
                else pulse<T, Y, RF, MAPS, false, true, false, POOLS>(EPG_PULSE_ARGS);
            } else {
                if (uniform) pulse<T, Y, RF, MAPS, true, false, false, POOLS>(EPG_PULSE_ARGS);
                else pulse<T, Y, RF, MAPS, false, false, false, POOLS>(EPG_PULSE_ARGS);
            }
        }
        if ((act & 32) && kind == 2) {
            const int out = p.output_index[event];
            if (out >= 0) {
                // A readout is kept by the lane whose state is its slot in the
                // run, and a run of consecutive outputs goes out at once.
                if (run_len > 0 && (out != run_start + run_len || run_len == width)) flush();
                if (run_len == 0) run_start = out;
                T ac, as;
                if (uniform) event_phase(p, base + event, ac, as);
#pragma unroll
                for (int y = 0; y < Y; ++y) {
                    if (!uniform) event_phase(p, active[y] ? train[y] * p.event_count + event : event, ac, as);
                    // A coil sees the whole voxel: the sum over the pools.
                    T read_r = fpr[y], read_i = fpi[y];
                    if constexpr (POOLS >= 2) {
                        read_r += bpr[y];
                        read_i += bpi[y];
                    }
                    const T r0 = num::shfl(read_r, 0, width);
                    const T i0 = num::shfl(read_i, 0, width);
                    const T m0y = MAPS ? m0[MAPS ? y : 0] : T(1.0f);
                    if (state == run_len) {
                        held_r[y] = stored(m0y * (r0 * ac + i0 * as));
                        held_i[y] = stored(m0y * (i0 * ac - r0 * as));
                    }
                }
                ++run_len;
            }
        }
        if (act & 18) shift();
        if (act & 8) {
#pragma unroll
            for (int y = 0; y < Y; ++y) {
                fpr[y] = fpi[y] = fmr[y] = fmi[y] = 0.0f;
                if constexpr (POOLS >= 2) bpr[y] = bpi[y] = bmr[y] = bmi[y] = 0.0f;
            }
        }
    }
    if (run_len > 0) flush();
}

}  // namespace epg
