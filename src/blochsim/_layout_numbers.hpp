// The arithmetic the kernels are written in, for a plain number and for a
// dual number: a value and its derivative along one direction. A kernel
// written once over ``T`` is the forward simulation at ``float`` and its
// Jacobian-vector product at ``Dual``.
#pragma once

#include <cuda_runtime.h>

#include <type_traits>

namespace num {

template <class F>
struct DualT {
    F v, d;
    DualT() = default;
    __host__ __device__ constexpr DualT(F value) : v(value), d(F(0)) {}
    __host__ __device__ constexpr DualT(F value, F tangent) : v(value), d(tangent) {}
    template <class G>
    __host__ __device__ explicit constexpr DualT(const DualT<G>& other) : v(F(other.v)), d(F(other.d)) {}
};
using Dual = DualT<float>;
using Dual64 = DualT<double>;

template <class T>
struct is_dual {
    static constexpr bool value = false;
};
template <class F>
struct is_dual<DualT<F>> {
    static constexpr bool value = true;
};

__device__ __forceinline__ float value(float x) { return x; }
__device__ __forceinline__ double value(double x) { return x; }
template <class F>
__device__ __forceinline__ F value(DualT<F> x) {
    return x.v;
}
__device__ __forceinline__ float tangent(float) { return 0.0f; }
template <class F>
__device__ __forceinline__ F tangent(DualT<F> x) {
    return x.d;
}

// Equal as a cache key: the value and, for a dual, the direction too.
__device__ __forceinline__ bool same(float a, float b) { return a == b; }
__device__ __forceinline__ bool same(double a, double b) { return a == b; }
template <class F>
__device__ __forceinline__ bool same(DualT<F> a, DualT<F> b) {
    return a.v == b.v && a.d == b.d;
}

template <class F>
__device__ __forceinline__ DualT<F> operator+(DualT<F> a, DualT<F> b) {
    return {a.v + b.v, a.d + b.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator-(DualT<F> a, DualT<F> b) {
    return {a.v - b.v, a.d - b.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator-(DualT<F> a) {
    return {-a.v, -a.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator*(DualT<F> a, DualT<F> b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator/(DualT<F> a, DualT<F> b) {
    const F q = a.v / b.v;
    return {q, (a.d - q * b.d) / b.v};
}
template <class F>
__device__ __forceinline__ DualT<F> operator+(DualT<F> a, F b) {
    return {a.v + b, a.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator+(F a, DualT<F> b) {
    return {a + b.v, b.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator-(DualT<F> a, F b) {
    return {a.v - b, a.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator-(F a, DualT<F> b) {
    return {a - b.v, -b.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator*(DualT<F> a, F b) {
    return {a.v * b, a.d * b};
}
template <class F>
__device__ __forceinline__ DualT<F> operator*(F a, DualT<F> b) {
    return {a * b.v, a * b.d};
}
template <class F>
__device__ __forceinline__ DualT<F> operator/(DualT<F> a, F b) {
    return {a.v / b, a.d / b};
}
template <class F>
__device__ __forceinline__ DualT<F> operator/(F a, DualT<F> b) {
    const F q = a / b.v;
    return {q, -q * b.d / b.v};
}
template <class F>
__device__ __forceinline__ DualT<F>& operator+=(DualT<F>& a, DualT<F> b) {
    return a = a + b;
}
template <class F>
__device__ __forceinline__ DualT<F>& operator-=(DualT<F>& a, DualT<F> b) {
    return a = a - b;
}
template <class F>
__device__ __forceinline__ DualT<F>& operator*=(DualT<F>& a, DualT<F> b) {
    return a = a * b;
}
template <class F>
__device__ __forceinline__ DualT<F>& operator*=(DualT<F>& a, F b) {
    return a = a * b;
}
template <class F>
__device__ __forceinline__ bool operator<(DualT<F> a, F b) {
    return a.v < b;
}
template <class F>
__device__ __forceinline__ bool operator>(DualT<F> a, F b) {
    return a.v > b;
}
template <class F>
__device__ __forceinline__ bool operator<(DualT<F> a, DualT<F> b) {
    return a.v < b.v;
}
template <class F>
__device__ __forceinline__ bool operator>(DualT<F> a, DualT<F> b) {
    return a.v > b.v;
}

__device__ __forceinline__ float fma_(float a, float b, float c) { return fmaf(a, b, c); }
template <class F>
__device__ __forceinline__ DualT<F> fma_(DualT<F> a, DualT<F> b, DualT<F> c) {
    return {fma(a.v, b.v, c.v), fma(a.d, b.v, fma(a.v, b.d, c.d))};
}

__device__ __forceinline__ float exp_(float x) { return __expf(x); }
__device__ __forceinline__ double exp_(double x) { return exp(x); }
template <class F>
__device__ __forceinline__ DualT<F> exp_(DualT<F> x) {
    const F e = exp_(x.v);
    return {e, e * x.d};
}

// Division the way Triton's float32 ``/`` divides: approximate.
__device__ __forceinline__ float div_(float a, float b) { return __fdividef(a, b); }
__device__ __forceinline__ double div_(double a, double b) { return a / b; }
template <class F>
__device__ __forceinline__ DualT<F> div_(DualT<F> a, DualT<F> b) {
    const F q = div_(a.v, b.v);
    return {q, div_(a.d - q * b.d, b.v)};
}

__device__ __forceinline__ float sqrt_approx(float x) {
    float r;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float sqrt_(float x) { return sqrt_approx(x); }
__device__ __forceinline__ double sqrt_(double x) { return sqrt(x); }
template <class F>
__device__ __forceinline__ DualT<F> sqrt_(DualT<F> x) {
    const F r = sqrt_(x.v);
    return {r, div_(F(0.5) * x.d, r)};
}

__device__ __forceinline__ float min_(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float max_(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double min_(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ double max_(double a, double b) { return fmax(a, b); }
template <class F>
__device__ __forceinline__ DualT<F> min_(DualT<F> a, DualT<F> b) {
    return b.v < a.v ? b : a;
}
template <class F>
__device__ __forceinline__ DualT<F> max_(DualT<F> a, DualT<F> b) {
    return b.v > a.v ? b : a;
}
__device__ __forceinline__ float abs_(float a) { return fabsf(a); }
__device__ __forceinline__ double abs_(double a) { return fabs(a); }
// |x| carries no derivative at the origin.
template <class F>
__device__ __forceinline__ DualT<F> abs_(DualT<F> a) {
    if (a.v == F(0)) return {a.v, F(0)};
    return a.v < F(0) ? -a : a;
}

__device__ __forceinline__ double acos_(double x) { return acos(x); }
__device__ __forceinline__ Dual64 acos_(Dual64 x) {
    return {acos(x.v), -x.d / sqrt(1.0 - x.v * x.v)};
}
__device__ __forceinline__ double cos_(double x) { return cos(x); }
__device__ __forceinline__ double sin_(double x) { return sin(x); }
__device__ __forceinline__ Dual64 sin_(Dual64 x) {
    double s, c;
    sincos(x.v, &s, &c);
    return {s, c * x.d};
}
__device__ __forceinline__ Dual64 cos_(Dual64 x) {
    double s, c;
    sincos(x.v, &s, &c);
    return {c, -s * x.d};
}

// Triton's _sincos: one Cody-Waite reduction by a quarter turn, the Cephes
// polynomials either side of zero.
__device__ __forceinline__ void sincos_cw(float x, float& s, float& c) {
    const float quarter = rintf(x * 0.6366197723675814f);
    float r = fmaf(-quarter, 1.5703125f, x);
    r = fmaf(-quarter, 4.837512969970703125e-4f, r);
    r = fmaf(-quarter, 7.54978995489188216e-8f, r);
    const float r2 = r * r;
    float sine = fmaf(-1.9515295891e-4f, r2, 8.3321608736e-3f);
    sine = fmaf(sine, r2, -1.6666654611e-1f);
    sine = fmaf(r * r2, sine, r);
    float cosine = fmaf(2.443315711809948e-5f, r2, -1.388731625493765e-3f);
    cosine = fmaf(cosine, r2, 4.166664568298827e-2f);
    cosine = fmaf(r2 * r2, cosine, fmaf(-0.5f, r2, 1.0f));
    const int q = static_cast<int>(quarter) & 3;
    s = q == 0 ? sine : (q == 1 ? cosine : (q == 2 ? -sine : -cosine));
    c = q == 0 ? cosine : (q == 1 ? -sine : (q == 2 ? -cosine : sine));
}
__device__ __forceinline__ void sincos_(float x, float& s, float& c) { sincos_cw(x, s, c); }
__device__ __forceinline__ void sincos_(Dual x, Dual& s, Dual& c) {
    float sv, cv;
    sincos_cw(x.v, sv, cv);
    s = {sv, cv * x.d};
    c = {cv, -sv * x.d};
}

__device__ __forceinline__ float shfl_up(float x, int delta, int width) {
    return __shfl_up_sync(0xffffffffu, x, delta, width);
}
__device__ __forceinline__ Dual shfl_up(Dual x, int delta, int width) {
    return {__shfl_up_sync(0xffffffffu, x.v, delta, width), __shfl_up_sync(0xffffffffu, x.d, delta, width)};
}
__device__ __forceinline__ float shfl_down(float x, int delta, int width) {
    return __shfl_down_sync(0xffffffffu, x, delta, width);
}
__device__ __forceinline__ Dual shfl_down(Dual x, int delta, int width) {
    return {__shfl_down_sync(0xffffffffu, x.v, delta, width),
            __shfl_down_sync(0xffffffffu, x.d, delta, width)};
}
__device__ __forceinline__ float shfl(float x, int lane, int width) {
    return __shfl_sync(0xffffffffu, x, lane, width);
}
__device__ __forceinline__ Dual shfl(Dual x, int lane, int width) {
    return {__shfl_sync(0xffffffffu, x.v, lane, width), __shfl_sync(0xffffffffu, x.d, lane, width)};
}

// A value from an absolute lane, among the lanes ``mask`` names.
__device__ __forceinline__ float shfl_from(float x, int lane, unsigned mask) {
    return __shfl_sync(mask, x, lane);
}
__device__ __forceinline__ Dual shfl_from(Dual x, int lane, unsigned mask) {
    return {__shfl_sync(mask, x.v, lane), __shfl_sync(mask, x.d, lane)};
}

// A value read with its direction where the type carries one; no direction
// given is a direction of zero.
template <class T>
__device__ __forceinline__ T load(const float* values, const float* tangents, long long at) {
    if constexpr (is_dual<T>::value) {
        return T{__ldg(values + at), tangents != nullptr ? __ldg(tangents + at) : 0.0f};
    } else {
        return __ldg(values + at);
    }
}

// A rate in 1/s from a time constant in ms, divided exactly as the launch
// formed it.
__device__ __forceinline__ float rate(float ms) { return 1000.0f / ms; }
__device__ __forceinline__ Dual rate(Dual ms) {
    const float r = 1000.0f / ms.v;
    return {r, -1000.0f * ms.d / (ms.v * ms.v)};
}


// The number a value finally is, through any nesting of duals: what a branch
// decides on.
__device__ __forceinline__ float primal(float x) { return x; }
__device__ __forceinline__ double primal(double x) { return x; }
template <class F>
__device__ __forceinline__ auto primal(DualT<F> x) {
    return primal(x.v);
}

// A value with its derivatives along K directions at once: what a gradient
// contracts against when many inputs reach one interval's operator.
template <int K, class S>
struct Multi {
    S v;
    S d[K];
    Multi() = default;
    __device__ __forceinline__ Multi(float value) : v(S(value)) {
#pragma unroll
        for (int k = 0; k < K; ++k) d[k] = S(0.0f);
    }
    __device__ __forceinline__ Multi(double value) : v(S(value)) {
#pragma unroll
        for (int k = 0; k < K; ++k) d[k] = S(0.0f);
    }
    __device__ __forceinline__ Multi(S value, int direction) : v(value) {
#pragma unroll
        for (int k = 0; k < K; ++k) d[k] = S(k == direction ? 1.0f : 0.0f);
    }
    template <int J, class R>
    __device__ __forceinline__ explicit Multi(const Multi<J, R>& other) : v(S(other.v)) {
        static_assert(J == K, "a conversion keeps the directions");
#pragma unroll
        for (int k = 0; k < K; ++k) d[k] = S(other.d[k]);
    }
};
template <int K, class S>
struct is_dual<Multi<K, S>> {
    static constexpr bool value = true;
};
template <int K, class S>
__device__ __forceinline__ auto primal(const Multi<K, S>& x) {
    return primal(x.v);
}
template <int K, class S>
__device__ __forceinline__ S value(const Multi<K, S>& x) {
    return x.v;
}

#define NUM_MULTI_BINARY(OP, VALUE, TANGENT)                                                  \
    template <int K, class S>                                                                 \
    __device__ __forceinline__ Multi<K, S> operator OP(const Multi<K, S>& a, const Multi<K, S>& b) { \
        Multi<K, S> r;                                                                        \
        r.v = VALUE;                                                                          \
        _Pragma("unroll") for (int k = 0; k < K; ++k) r.d[k] = TANGENT;                       \
        return r;                                                                             \
    }
NUM_MULTI_BINARY(+, a.v + b.v, a.d[k] + b.d[k])
NUM_MULTI_BINARY(-, a.v - b.v, a.d[k] - b.d[k])
NUM_MULTI_BINARY(*, a.v * b.v, a.d[k] * b.v + a.v * b.d[k])
#undef NUM_MULTI_BINARY
template <int K, class S>
__device__ __forceinline__ Multi<K, S> operator/(const Multi<K, S>& a, const Multi<K, S>& b) {
    Multi<K, S> r;
    r.v = a.v / b.v;
#pragma unroll
    for (int k = 0; k < K; ++k) r.d[k] = (a.d[k] - r.v * b.d[k]) / b.v;
    return r;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> operator-(const Multi<K, S>& a) {
    Multi<K, S> r;
    r.v = -a.v;
#pragma unroll
    for (int k = 0; k < K; ++k) r.d[k] = -a.d[k];
    return r;
}
// With a constant, of the innermost kind or of the direction's own.
template <class S>
using primal_of = decltype(primal(S()));
#define NUM_MULTI_SCALAR(OP, LEFT_V, LEFT_D, RIGHT_V, RIGHT_D)                                \
    template <int K, class S>                                                                 \
    __device__ __forceinline__ Multi<K, S> operator OP(const Multi<K, S>& a, primal_of<S> b) { \
        Multi<K, S> r;                                                                        \
        r.v = LEFT_V;                                                                         \
        _Pragma("unroll") for (int k = 0; k < K; ++k) r.d[k] = LEFT_D;                        \
        return r;                                                                             \
    }                                                                                         \
    template <int K, class S>                                                                 \
    __device__ __forceinline__ Multi<K, S> operator OP(primal_of<S> b, const Multi<K, S>& a) { \
        Multi<K, S> r;                                                                        \
        r.v = RIGHT_V;                                                                        \
        _Pragma("unroll") for (int k = 0; k < K; ++k) r.d[k] = RIGHT_D;                       \
        return r;                                                                             \
    }
NUM_MULTI_SCALAR(+, a.v + b, a.d[k], b + a.v, a.d[k])
NUM_MULTI_SCALAR(-, a.v - b, a.d[k], b - a.v, -a.d[k])
NUM_MULTI_SCALAR(*, a.v * b, a.d[k] * b, b * a.v, b * a.d[k])
#undef NUM_MULTI_SCALAR
// With a value of the direction's own kind, where that kind is itself a dual.
#define NUM_MULTI_SAME(OP, LEFT_V, LEFT_D, RIGHT_V, RIGHT_D)                                       \
    template <int K, class S, typename std::enable_if<is_dual<S>::value, int>::type = 0>          \
    __device__ __forceinline__ Multi<K, S> operator OP(const Multi<K, S>& a, const S& b) {        \
        Multi<K, S> r;                                                                            \
        r.v = LEFT_V;                                                                             \
        _Pragma("unroll") for (int k = 0; k < K; ++k) r.d[k] = LEFT_D;                            \
        return r;                                                                                 \
    }                                                                                             \
    template <int K, class S, typename std::enable_if<is_dual<S>::value, int>::type = 0>          \
    __device__ __forceinline__ Multi<K, S> operator OP(const S& b, const Multi<K, S>& a) {        \
        Multi<K, S> r;                                                                            \
        r.v = RIGHT_V;                                                                            \
        _Pragma("unroll") for (int k = 0; k < K; ++k) r.d[k] = RIGHT_D;                           \
        return r;                                                                                 \
    }
NUM_MULTI_SAME(+, a.v + b, a.d[k], b + a.v, a.d[k])
NUM_MULTI_SAME(-, a.v - b, a.d[k], b - a.v, -a.d[k])
NUM_MULTI_SAME(*, a.v * b, a.d[k] * b, b * a.v, b * a.d[k])
#undef NUM_MULTI_SAME
template <int K, class S>
__device__ __forceinline__ Multi<K, S> operator/(const Multi<K, S>& a, primal_of<S> b) {
    Multi<K, S> r;
    r.v = a.v / b;
#pragma unroll
    for (int k = 0; k < K; ++k) r.d[k] = a.d[k] / b;
    return r;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S>& operator+=(Multi<K, S>& a, const Multi<K, S>& b) {
    return a = a + b;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S>& operator-=(Multi<K, S>& a, const Multi<K, S>& b) {
    return a = a - b;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S>& operator*=(Multi<K, S>& a, const Multi<K, S>& b) {
    return a = a * b;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S>& operator*=(Multi<K, S>& a, primal_of<S> b) {
    return a = a * b;
}
template <int K, class S>
__device__ __forceinline__ bool operator<(const Multi<K, S>& a, primal_of<S> b) {
    return primal(a) < b;
}
template <int K, class S>
__device__ __forceinline__ bool operator>(const Multi<K, S>& a, primal_of<S> b) {
    return primal(a) > b;
}
template <int K, class S>
__device__ __forceinline__ bool same(const Multi<K, S>& a, const Multi<K, S>& b) {
    bool equal = same(a.v, b.v);
#pragma unroll
    for (int k = 0; k < K; ++k) equal = equal && same(a.d[k], b.d[k]);
    return equal;
}

// A function of one argument along every direction: f(v) and f'(v) d.
template <int K, class S, class Value, class Slope>
__device__ __forceinline__ Multi<K, S> chain(const Multi<K, S>& x, Value value, Slope slope) {
    Multi<K, S> r;
    r.v = value;
#pragma unroll
    for (int k = 0; k < K; ++k) r.d[k] = slope * x.d[k];
    return r;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> fma_(const Multi<K, S>& a, const Multi<K, S>& b, const Multi<K, S>& c) {
    return a * b + c;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> exp_(const Multi<K, S>& x) {
    const S e = exp_(x.v);
    return chain(x, e, e);
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> div_(const Multi<K, S>& a, const Multi<K, S>& b) {
    Multi<K, S> r;
    r.v = div_(a.v, b.v);
#pragma unroll
    for (int k = 0; k < K; ++k) r.d[k] = div_(a.d[k] - r.v * b.d[k], b.v);
    return r;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> sqrt_(const Multi<K, S>& x) {
    const S r = sqrt_(x.v);
    return chain(x, r, div_(S(0.5f), r));
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> min_(const Multi<K, S>& a, const Multi<K, S>& b) {
    return primal(b) < primal(a) ? b : a;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> max_(const Multi<K, S>& a, const Multi<K, S>& b) {
    return primal(b) > primal(a) ? b : a;
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> abs_(const Multi<K, S>& a) {
    if (primal(a) == 0) return Multi<K, S>(a.v, -1);
    return primal(a) < 0 ? -a : a;
}
template <int K, class S>
__device__ __forceinline__ void sincos_(const Multi<K, S>& x, Multi<K, S>& s, Multi<K, S>& c) {
    S sv, cv;
    sincos_(x.v, sv, cv);
    s = chain(x, sv, cv);
    c = chain(x, cv, -sv);
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> acos_(const Multi<K, S>& x) {
    return chain(x, acos_(x.v), -1.0 / sqrt_(1.0 - x.v * x.v));
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> cos_(const Multi<K, S>& x) {
    return chain(x, cos_(x.v), -sin_(x.v));
}
template <int K, class S>
__device__ __forceinline__ Multi<K, S> rate(const Multi<K, S>& ms) {
    const S r = rate(ms.v);
    return chain(ms, r, -r * r * (1.0f / 1000.0f));
}

}  // namespace num
