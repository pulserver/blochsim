// The tile model the GPU kernels are written in, for a CUDA block and for the
// host.
//
// A kernel program holds a tile of at most two axes: ``x`` runs along the
// threads of one row of the block and ``y`` across its rows. A third, ``z``,
// stands in for ``x`` where a tile is square in a short axis -- an operator
// over pools -- and is held as an array in each thread. A value carries, in
// its type, the axes it varies along (``AX``: bit 0 for x, bit 1 for y, bit 2
// for z), so
// a value of one row's problem is not mistaken for a value of every state of
// it: a sum along x of a value that does not vary along x is the value itself,
// and a store or an atomic of it is made once rather than once per thread.
// Values that vary along neither are plain C++ scalars, uniform over the block.
//
// Under ``BLOCHSIM_SIMT`` (the CUDA build) a tile is one element per thread,
// and the operations across a row are warp shuffles or shared memory. Without
// it, a tile is the whole array, so the same kernel source runs on the host one
// program at a time: that is the build the tests run without a card.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(BLOCHSIM_SIMT)
#define BSK_HD __device__ __forceinline__
#define BSK_CONSTEXPR __host__ __device__ constexpr
#else
#define BSK_CONSTEXPR constexpr
#include <algorithm>
#include <vector>
#define BSK_HD inline
#endif

namespace bsk {

// ---------------------------------------------------------------------------
// The launch a program belongs to.
// ---------------------------------------------------------------------------

#if defined(BLOCHSIM_SIMT)

extern __shared__ unsigned long long shared_words[];

// The length of z, set once by a kernel's entry.
__shared__ int z_width;

BSK_HD int width_x() { return static_cast<int>(blockDim.x); }
BSK_HD int width_y() { return static_cast<int>(blockDim.y); }
BSK_HD int width_z() { return z_width; }

__device__ __forceinline__ void enter(int nz) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        z_width = nz;
    }
    __syncthreads();
}
BSK_HD std::int64_t program_id(int axis) {
    return axis == 0 ? static_cast<std::int64_t>(blockIdx.x)
                     : static_cast<std::int64_t>(blockIdx.y);
}

#else

struct HostProgram {
    int nx = 1;
    int ny = 1;
    int nz = 1;
    std::int64_t pid[2] = {0, 0};
};

inline thread_local HostProgram program;

inline int width_x() { return program.nx; }
inline int width_y() { return program.ny; }
inline int width_z() { return program.nz; }
inline std::int64_t program_id(int axis) { return program.pid[axis]; }

#endif

// ---------------------------------------------------------------------------
// Tiles.
// ---------------------------------------------------------------------------

// The longest z a tile holds.
constexpr int MAX_Z = 8;

template <class T, int AX>
struct V;

template <class A>
struct tile_traits {
    static constexpr int axes = 0;
    using element = A;
};

template <class T, int AX>
struct tile_traits<V<T, AX>> {
    static constexpr int axes = AX;
    using element = T;
};

template <class A>
using element_t = typename tile_traits<std::decay_t<A>>::element;

template <class A>
constexpr int axes_of = tile_traits<std::decay_t<A>>::axes;

template <class... A>
constexpr bool any_tile = ((axes_of<A> != 0) || ...);

#if defined(BLOCHSIM_SIMT)

template <class T, int AX>
struct V {
    static_assert(AX > 0 && AX < 8 && AX != 5 && AX != 7, "a tile varies along x or z, y, or both");
    static constexpr int lanes = (AX & 4) ? MAX_Z : 1;
    T v[lanes];

    V() = default;

    template <class U, std::enable_if_t<std::is_arithmetic_v<U> || std::is_pointer_v<U>, int> = 0>
    BSK_HD V(U scalar) {
        for (int z = 0; z < lanes; ++z) v[z] = static_cast<T>(scalar);
    }

    template <class U, int BX, std::enable_if_t<(BX | AX) == AX, int> = 0>
    BSK_HD V(const V<U, BX>& other) {
        for (int z = 0; z < lanes; ++z) v[z] = static_cast<T>(other.v[(BX & 4) ? z : 0]);
    }
};

// The element a thread holds at ``z``.
template <class A>
BSK_HD decltype(auto) element(const A& a, int z = 0) {
    if constexpr (axes_of<A> == 0) {
        return a;
    } else if constexpr ((axes_of<A> & 4) != 0) {
        return (a.v[z]);
    } else {
        return (a.v[0]);
    }
}

// Lanes of z past its length are left unset, so nothing reads memory for them.
template <class F, class... A>
BSK_HD auto zip(F f, const A&... a) {
    constexpr int AX = (0 | ... | axes_of<A>);
    if constexpr (AX == 0) {
        return f(a...);
    } else {
        using R = decltype(f(element(a)...));
        V<R, AX> out;
        if constexpr ((AX & 4) != 0) {
            const int nz = width_z();
            for (int z = 0; z < MAX_Z; ++z) {
                if (z < nz) {
                    out.v[z] = f(element(a, z)...);
                }
            }
        } else {
            out.v[0] = f(element(a)...);
        }
        return out;
    }
}

#else

template <int AX>
inline std::size_t count() {
    return static_cast<std::size_t>((AX & 1) ? program.nx : 1)
        * static_cast<std::size_t>((AX & 2) ? program.ny : 1)
        * static_cast<std::size_t>((AX & 4) ? program.nz : 1);
}

// The extent of a loop over the axes in AX.
template <int AX>
inline int extent_x() { return (AX & 1) ? program.nx : 1; }
template <int AX>
inline int extent_y() { return (AX & 2) ? program.ny : 1; }
template <int AX>
inline int extent_z() { return (AX & 4) ? program.nz : 1; }

template <class T, int AX>
struct V {
    static_assert(AX > 0 && AX < 8 && AX != 5 && AX != 7, "a tile varies along x or z, y, or both");
    // std::vector<bool> holds no addressable elements.
    using Stored = std::conditional_t<std::is_same_v<T, bool>, unsigned char, T>;
    std::vector<Stored> v;

    V() : v(count<AX>()) {}

    template <class U, std::enable_if_t<std::is_arithmetic_v<U> || std::is_pointer_v<U>, int> = 0>
    V(U scalar) : v(count<AX>(), static_cast<Stored>(static_cast<T>(scalar))) {}

    template <class U, int BX, std::enable_if_t<(BX | AX) == AX, int> = 0>
    V(const V<U, BX>& other) : v(count<AX>()) {
        for (int y = 0; y < extent_y<AX>(); ++y) {
            for (int x = 0; x < extent_x<AX>(); ++x) {
                for (int z = 0; z < extent_z<AX>(); ++z) {
                    v[index(y, x, z)] = static_cast<Stored>(static_cast<T>(other.at(y, x, z)));
                }
            }
        }
    }

    std::size_t index(int y, int x, int z) const {
        const std::size_t row = static_cast<std::size_t>(extent_x<AX>()) * extent_z<AX>();
        return static_cast<std::size_t>((AX & 2) ? y : 0) * row
            + static_cast<std::size_t>((AX & 1) ? x : 0) * extent_z<AX>()
            + static_cast<std::size_t>((AX & 4) ? z : 0);
    }
    const Stored& at(int y, int x, int z = 0) const { return v[index(y, x, z)]; }
    Stored& at(int y, int x, int z = 0) { return v[index(y, x, z)]; }
};

template <class A>
inline decltype(auto) element(const A& a, int y, int x, int z = 0) {
    if constexpr (axes_of<A> == 0) {
        return a;
    } else if constexpr (std::is_same_v<element_t<A>, bool>) {
        return static_cast<bool>(a.at(y, x, z));
    } else {
        return a.at(y, x, z);
    }
}

template <class F, class... A>
inline auto zip(F f, const A&... a) {
    constexpr int AX = (0 | ... | axes_of<A>);
    if constexpr (AX == 0) {
        return f(a...);
    } else {
        using R = decltype(f(element(a, 0, 0)...));
        V<R, AX> out;
        for (int y = 0; y < extent_y<AX>(); ++y) {
            for (int x = 0; x < extent_x<AX>(); ++x) {
                for (int z = 0; z < extent_z<AX>(); ++z) {
                    out.at(y, x, z) = f(element(a, y, x, z)...);
                }
            }
        }
        return out;
    }
}

#endif

// ---------------------------------------------------------------------------
// Element-wise arithmetic. Between two scalars the language's own operators
// apply; these take over as soon as one side is a tile.
// ---------------------------------------------------------------------------

#define BSK_BINARY(op)                                                         \
    template <class A, class B, std::enable_if_t<any_tile<A, B>, int> = 0>     \
    BSK_HD auto operator op(const A& a, const B& b) {                          \
        return zip([](auto x, auto y) { return x op y; }, a, b);               \
    }

BSK_BINARY(+)
BSK_BINARY(-)
BSK_BINARY(*)
BSK_BINARY(<)
BSK_BINARY(<=)
BSK_BINARY(>)
BSK_BINARY(>=)
BSK_BINARY(==)
BSK_BINARY(!=)
BSK_BINARY(<<)
BSK_BINARY(>>)
#undef BSK_BINARY

template <class T, int AX>
BSK_HD auto operator-(const V<T, AX>& a) {
    return zip([](auto x) { return -x; }, a);
}

// Python's ``/`` divides integers into a float; ``//`` and ``%`` keep them.
template <class A, class B>
BSK_HD auto truediv(const A& a, const B& b) {
    return zip(
        [](auto x, auto y) {
            if constexpr (std::is_integral_v<decltype(x)> && std::is_integral_v<decltype(y)>) {
                return static_cast<float>(x) / static_cast<float>(y);
            } else {
                return x / y;
            }
        },
        a, b);
}

template <class A, class B>
BSK_HD auto floordiv(const A& a, const B& b) {
    return zip(
        [](auto x, auto y) {
            if constexpr (std::is_integral_v<decltype(x)> && std::is_integral_v<decltype(y)>) {
                return x / y;
            } else {
                return floor(x / y);
            }
        },
        a, b);
}

template <class A, class B>
BSK_HD auto mod(const A& a, const B& b) {
    return zip(
        [](auto x, auto y) {
            if constexpr (std::is_integral_v<decltype(x)> && std::is_integral_v<decltype(y)>) {
                return x % y;
            } else {
                return fmod(x, y);
            }
        },
        a, b);
}

// ``&``, ``|``, ``^`` and ``~`` are logical on masks and bitwise on integers,
// and a mask stays a mask.
template <class X, class Y>
BSK_HD auto bit_and(X x, Y y) {
    if constexpr (std::is_same_v<X, bool> && std::is_same_v<Y, bool>) {
        return static_cast<bool>(x && y);
    } else {
        return x & y;
    }
}
template <class X, class Y>
BSK_HD auto bit_or(X x, Y y) {
    if constexpr (std::is_same_v<X, bool> && std::is_same_v<Y, bool>) {
        return static_cast<bool>(x || y);
    } else {
        return x | y;
    }
}
template <class X, class Y>
BSK_HD auto bit_xor(X x, Y y) {
    if constexpr (std::is_same_v<X, bool> && std::is_same_v<Y, bool>) {
        return static_cast<bool>(x != y);
    } else {
        return x ^ y;
    }
}
template <class X>
BSK_HD auto bit_not(X x) {
    if constexpr (std::is_same_v<X, bool>) {
        return !x;
    } else {
        return ~x;
    }
}

template <class A, class B>
BSK_HD auto band(const A& a, const B& b) {
    return zip([](auto x, auto y) { return bit_and(x, y); }, a, b);
}
template <class A, class B>
BSK_HD auto bor(const A& a, const B& b) {
    return zip([](auto x, auto y) { return bit_or(x, y); }, a, b);
}
template <class A, class B>
BSK_HD auto bxor(const A& a, const B& b) {
    return zip([](auto x, auto y) { return bit_xor(x, y); }, a, b);
}
template <class A>
BSK_HD auto bnot(const A& a) {
    return zip([](auto x) { return bit_not(x); }, a);
}

template <class T, class A>
BSK_HD auto cast(const A& a) {
    return zip([](auto x) { return static_cast<T>(x); }, a);
}

template <class C, class A, class B>
BSK_HD auto where(const C& c, const A& a, const B& b) {
    return zip(
        [](auto test, auto yes, auto no) {
            using R = decltype(yes + no);
            return test ? static_cast<R>(yes) : static_cast<R>(no);
        },
        c, a, b);
}

// A branch condition. Every thread of a block takes the same branch, so a
// tile reaching one holds the same value everywhere it is read.
template <class A>
BSK_HD bool truth(const A& a) {
    if constexpr (axes_of<A> == 0) {
        return static_cast<bool>(a);
    } else {
        return static_cast<bool>(a.v[0]);
    }
}

template <class A, class B>
BSK_HD auto select(bool test, const A& a, const B& b) {
    return where(test, a, b);
}

// ---------------------------------------------------------------------------
// Element-wise functions, in the precision of their argument.
// ---------------------------------------------------------------------------

BSK_HD float s_exp(float x) { return expf(x); }
BSK_HD double s_exp(double x) { return ::exp(x); }
BSK_HD float s_cos(float x) { return cosf(x); }
BSK_HD double s_cos(double x) { return ::cos(x); }
BSK_HD float s_sin(float x) { return sinf(x); }
BSK_HD double s_sin(double x) { return ::sin(x); }
BSK_HD float s_sqrt(float x) { return sqrtf(x); }
BSK_HD double s_sqrt(double x) { return ::sqrt(x); }
BSK_HD float s_floor(float x) { return floorf(x); }
BSK_HD double s_floor(double x) { return ::floor(x); }
BSK_HD float s_rint(float x) { return rintf(x); }
BSK_HD double s_rint(double x) { return ::rint(x); }
BSK_HD float s_acos(float x) { return acosf(x); }
BSK_HD double s_acos(double x) { return ::acos(x); }
BSK_HD float s_fma(float x, float y, float z) { return fmaf(x, y, z); }
BSK_HD double s_fma(double x, double y, double z) { return ::fma(x, y, z); }
template <class X>
BSK_HD X s_abs(X x) {
    if constexpr (std::is_floating_point_v<X>) {
        return x < X(0) ? -x : (x == X(0) ? X(0) : x);
    } else {
        return x < X(0) ? -x : x;
    }
}
// ``tl.minimum`` and ``tl.maximum`` return the number where one side is NaN.
template <class X, class Y>
BSK_HD auto s_min(X x, Y y) {
    using R = decltype(x + y);
    if constexpr (std::is_floating_point_v<R>) {
        return static_cast<R>(fmin(static_cast<R>(x), static_cast<R>(y)));
    } else {
        return static_cast<R>(x) < static_cast<R>(y) ? static_cast<R>(x) : static_cast<R>(y);
    }
}
template <class X, class Y>
BSK_HD auto s_max(X x, Y y) {
    using R = decltype(x + y);
    if constexpr (std::is_floating_point_v<R>) {
        return static_cast<R>(fmax(static_cast<R>(x), static_cast<R>(y)));
    } else {
        return static_cast<R>(x) > static_cast<R>(y) ? static_cast<R>(x) : static_cast<R>(y);
    }
}

#define BSK_UNARY(name)                                                        \
    template <class A>                                                         \
    BSK_HD auto name(const A& a) {                                             \
        return zip([](auto x) { return s_##name(x); }, a);                     \
    }
BSK_UNARY(exp)
BSK_UNARY(cos)
BSK_UNARY(sin)
BSK_UNARY(sqrt)
BSK_UNARY(floor)
BSK_UNARY(rint)
BSK_UNARY(acos)
BSK_UNARY(abs)
#undef BSK_UNARY

template <class A, class B>
BSK_HD auto minimum(const A& a, const B& b) {
    return zip([](auto x, auto y) { return s_min(x, y); }, a, b);
}
template <class A, class B>
BSK_HD auto maximum(const A& a, const B& b) {
    return zip([](auto x, auto y) { return s_max(x, y); }, a, b);
}
template <class A, class B, class C>
BSK_HD auto fma(const A& a, const B& b, const C& c) {
    return zip(
        [](auto x, auto y, auto z) {
            using R = decltype(x * y + z);
            return s_fma(static_cast<R>(x), static_cast<R>(y), static_cast<R>(z));
        },
        a, b, c);
}

// ---------------------------------------------------------------------------
// Indices.
// ---------------------------------------------------------------------------

BSK_HD V<std::int32_t, 1> arange_x() {
#if defined(BLOCHSIM_SIMT)
    V<std::int32_t, 1> out;
    out.v[0] = static_cast<std::int32_t>(threadIdx.x);
    return out;
#else
    V<std::int32_t, 1> out;
    for (int x = 0; x < program.nx; ++x) out.v[static_cast<std::size_t>(x)] = x;
    return out;
#endif
}

BSK_HD V<std::int32_t, 2> arange_y() {
#if defined(BLOCHSIM_SIMT)
    V<std::int32_t, 2> out;
    out.v[0] = static_cast<std::int32_t>(threadIdx.y);
    return out;
#else
    V<std::int32_t, 2> out;
    for (int y = 0; y < program.ny; ++y) out.v[static_cast<std::size_t>(y)] = y;
    return out;
#endif
}

BSK_HD V<std::int32_t, 4> arange_z() {
    V<std::int32_t, 4> out;
#if defined(BLOCHSIM_SIMT)
    for (int z = 0; z < MAX_Z; ++z) out.v[z] = z;
#else
    for (int z = 0; z < program.nz; ++z) out.v[static_cast<std::size_t>(z)] = z;
#endif
    return out;
}

template <class T, int AX>
BSK_HD auto full(T value) {
    if constexpr (AX == 0) {
        return value;
    } else {
        return V<T, AX>(value);
    }
}

template <class A>
BSK_HD auto zeros_like(const A& a) {
    using T = element_t<A>;
    if constexpr (axes_of<A> == 0) {
        return T(0);
    } else {
        return V<T, axes_of<A>>(T(0));
    }
}

// ---------------------------------------------------------------------------
// Memory.
// ---------------------------------------------------------------------------

template <class P>
BSK_HD auto ld(const P& pointer) {
    return zip([](auto p) { return *p; }, pointer);
}

template <class P, class M, class O>
BSK_HD auto ld(const P& pointer, const M& mask, const O& other) {
    return zip(
        [](auto p, auto m, auto o) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(p)>>;
            return m ? *p : static_cast<T>(o);
        },
        pointer, mask, other);
}

#if defined(BLOCHSIM_SIMT)

// An address that does not vary along an axis is held by every thread along
// it; only the first of them writes, with the value that thread holds.
template <int AX>
BSK_HD bool writes() {
    return ((AX & 1) || threadIdx.x == 0) && ((AX & 2) || threadIdx.y == 0);
}

template <int AX, class F>
BSK_HD void each_written(F f) {
    if (!writes<AX>()) {
        return;
    }
    if constexpr ((AX & 4) != 0) {
        const int nz = width_z();
        for (int z = 0; z < MAX_Z; ++z) {
            if (z < nz) {
                f(z);
            }
        }
    } else {
        f(0);
    }
}

template <class P, class T, class M>
BSK_HD void st(const P& pointer, const T& value, const M& mask) {
    constexpr int AX = axes_of<P> | axes_of<M>;
    each_written<AX>([&](int z) {
        if (element(mask, z)) {
            auto p = element(pointer, z);
            *p = static_cast<std::remove_pointer_t<decltype(p)>>(element(value, z));
        }
    });
}

template <class P, class T, class M>
BSK_HD void atomic_add(const P& pointer, const T& value, const M& mask) {
    constexpr int AX = axes_of<P> | axes_of<M>;
    each_written<AX>([&](int z) {
        if (element(mask, z)) {
            auto p = element(pointer, z);
            atomicAdd(p, static_cast<std::remove_pointer_t<decltype(p)>>(element(value, z)));
        }
    });
}

#else

template <class P, class T, class M>
inline void st(const P& pointer, const T& value, const M& mask) {
    constexpr int AX = axes_of<P> | axes_of<M>;
    for (int y = 0; y < extent_y<AX>(); ++y) {
        for (int x = 0; x < extent_x<AX>(); ++x) {
            for (int z = 0; z < extent_z<AX>(); ++z) {
                if (element(mask, y, x, z)) {
                    auto p = element(pointer, y, x, z);
                    *p = static_cast<std::remove_pointer_t<decltype(p)>>(element(value, y, x, z));
                }
            }
        }
    }
}

template <class P, class T, class M>
inline void atomic_add(const P& pointer, const T& value, const M& mask) {
    constexpr int AX = axes_of<P> | axes_of<M>;
    for (int y = 0; y < extent_y<AX>(); ++y) {
        for (int x = 0; x < extent_x<AX>(); ++x) {
            for (int z = 0; z < extent_z<AX>(); ++z) {
                if (element(mask, y, x, z)) {
                    auto p = element(pointer, y, x, z);
                    *p += static_cast<std::remove_pointer_t<decltype(p)>>(element(value, y, x, z));
                }
            }
        }
    }
}

#endif

template <class P, class T>
BSK_HD void st(const P& pointer, const T& value) {
    st(pointer, value, true);
}

template <class P, class T>
BSK_HD void atomic_add(const P& pointer, const T& value) {
    atomic_add(pointer, value, true);
}

// ---------------------------------------------------------------------------
// Across a row or a column of the block.
// ---------------------------------------------------------------------------

#if defined(BLOCHSIM_SIMT)

template <class T>
BSK_HD T shuffle_xor(T value, int offset, int width) {
    if constexpr (std::is_same_v<T, bool>) {
        return __shfl_xor_sync(0xffffffffu, static_cast<int>(value), offset, width) != 0;
    } else {
        return __shfl_xor_sync(0xffffffffu, value, offset, width);
    }
}

template <class T>
BSK_HD T shuffle(T value, int lane, int width) {
    if constexpr (std::is_same_v<T, bool>) {
        return __shfl_sync(0xffffffffu, static_cast<int>(value), lane, width) != 0;
    } else {
        return __shfl_sync(0xffffffffu, value, lane, width);
    }
}

// Every thread of the block runs these together: a kernel's control flow
// depends on nothing a single thread holds.
template <class T, class Op>
BSK_HD T reduce_row(T value, Op op) {
    const int nx = width_x();
    const int width = nx < 32 ? nx : 32;
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        value = op(value, shuffle_xor(value, offset, width));
    }
    if (nx <= 32) {
        return value;
    }
    T* words = reinterpret_cast<T*>(shared_words);
    const int warps = nx >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        words[threadIdx.y * warps + (threadIdx.x >> 5)] = value;
    }
    __syncthreads();
    T total = words[threadIdx.y * warps];
    for (int w = 1; w < warps; ++w) {
        total = op(total, words[threadIdx.y * warps + w]);
    }
    return total;
}

template <class T, class Op>
BSK_HD T reduce_column(T value, Op op) {
    T* words = reinterpret_cast<T*>(shared_words);
    __syncthreads();
    words[threadIdx.y * blockDim.x + threadIdx.x] = value;
    __syncthreads();
    T total = words[threadIdx.x];
    for (unsigned y = 1; y < blockDim.y; ++y) {
        total = op(total, words[y * blockDim.x + threadIdx.x]);
    }
    return total;
}

template <class T>
BSK_HD T gather_row(T value, int lane) {
    const int nx = width_x();
    if (nx <= 32) {
        return shuffle(value, lane, nx);
    }
    T* words = reinterpret_cast<T*>(shared_words);
    __syncthreads();
    words[threadIdx.y * nx + threadIdx.x] = value;
    __syncthreads();
    return words[threadIdx.y * nx + lane];
}

#endif

struct Add {
    template <class X>
    BSK_HD X operator()(X a, X b) const { return a + b; }
};
struct Max {
    template <class X>
    BSK_HD X operator()(X a, X b) const { return s_max(a, b); }
};

// A value with an axis taken out of AX, held where the reduction left it.
template <int AX, class T>
BSK_HD auto reduced(const T* lanes) {
    if constexpr (AX == 0) {
        return lanes[0];
    } else {
        V<T, AX> out;
        for (int z = 0; z < V<T, AX>::lanes; ++z) out.v[z] = lanes[z];
        return out;
    }
}

template <class Op, class T, int AX>
BSK_HD auto reduce_x(const V<T, AX>& a, Op op) {
    if constexpr ((AX & 1) == 0) {
        return a;
    } else {
        constexpr int RX = AX & ~1;
#if defined(BLOCHSIM_SIMT)
        T total = reduce_row(a.v[0], op);
        return reduced<RX>(&total);
#else
        if constexpr (RX == 0) {
            T total = a.at(0, 0);
            for (int x = 1; x < program.nx; ++x) total = op(total, a.at(0, x));
            return total;
        } else {
            V<T, RX> out;
            for (int y = 0; y < program.ny; ++y) {
                T total = a.at(y, 0);
                for (int x = 1; x < program.nx; ++x) total = op(total, a.at(y, x));
                out.at(y, 0) = total;
            }
            return out;
        }
#endif
    }
}

template <class Op, class T, int AX>
BSK_HD auto reduce_y(const V<T, AX>& a, Op op) {
    if constexpr ((AX & 2) == 0) {
        return a;
    } else {
        constexpr int RX = AX & ~2;
#if defined(BLOCHSIM_SIMT)
        T totals[V<T, AX>::lanes];
        for (int z = 0; z < V<T, AX>::lanes; ++z) totals[z] = reduce_column(a.v[z], op);
        return reduced<RX>(totals);
#else
        if constexpr (RX == 0) {
            T total = a.at(0, 0);
            for (int y = 1; y < program.ny; ++y) total = op(total, a.at(y, 0));
            return total;
        } else {
            V<T, RX> out;
            for (int x = 0; x < extent_x<RX>(); ++x) {
                for (int z = 0; z < extent_z<RX>(); ++z) {
                    T total = a.at(0, x, z);
                    for (int y = 1; y < program.ny; ++y) total = op(total, a.at(y, x, z));
                    out.at(0, x, z) = total;
                }
            }
            return out;
        }
#endif
    }
}

template <class Op, class T, int AX>
BSK_HD auto reduce_z(const V<T, AX>& a, Op op) {
    if constexpr ((AX & 4) == 0) {
        return a;
    } else {
        constexpr int RX = AX & ~4;
#if defined(BLOCHSIM_SIMT)
        T total = a.v[0];
        const int nz = width_z();
        for (int z = 1; z < MAX_Z; ++z) {
            if (z < nz) {
                total = op(total, a.v[z]);
            }
        }
        return reduced<RX>(&total);
#else
        if constexpr (RX == 0) {
            T total = a.at(0, 0, 0);
            for (int z = 1; z < program.nz; ++z) total = op(total, a.at(0, 0, z));
            return total;
        } else {
            V<T, RX> out;
            for (int y = 0; y < program.ny; ++y) {
                T total = a.at(y, 0, 0);
                for (int z = 1; z < program.nz; ++z) total = op(total, a.at(y, 0, z));
                out.at(y, 0) = total;
            }
            return out;
        }
#endif
    }
}

// ``tl.sum(a, axis=1)`` of a two-axis tile: along x, or along z where the
// tile's second axis is z.
template <class A>
BSK_HD auto sum_x(const A& a) {
    if constexpr (axes_of<A> == 0) {
        return a;
    } else if constexpr ((axes_of<A> & 4) != 0) {
        return reduce_z(a, Add{});
    } else {
        return reduce_x(a, Add{});
    }
}
template <class A>
BSK_HD auto sum_y(const A& a) {
    if constexpr (axes_of<A> == 0) {
        return a;
    } else {
        return reduce_y(a, Add{});
    }
}
template <class A>
BSK_HD auto sum_all(const A& a) {
    return sum_y(sum_x(a));
}
template <class A>
BSK_HD auto max_all(const A& a) {
    static_assert((axes_of<A> & 4) == 0, "a maximum along z is not taken");
    if constexpr (axes_of<A> == 0) {
        return a;
    } else {
        auto row = reduce_x(a, Max{});
        if constexpr (axes_of<decltype(row)> == 0) {
            return row;
        } else {
            return reduce_y(row, Max{});
        }
    }
}

// ``values`` read at ``index`` along x, row by row: ``tl.gather(values, index, 1)``.
template <class T, int AX, class I>
BSK_HD auto gather_x(const V<T, AX>& values, const I& index) {
    static_assert(AX & 1, "a gather along x reads a value that varies along x");
#if defined(BLOCHSIM_SIMT)
    V<T, AX | axes_of<I>> out;
    out.v[0] = gather_row(values.v[0], static_cast<int>(element(index)));
    return out;
#else
    constexpr int RX = AX | axes_of<I>;
    V<T, RX> out;
    for (int y = 0; y < ((RX & 2) ? program.ny : 1); ++y) {
        for (int x = 0; x < program.nx; ++x) {
            out.at(y, x) = values.at(y, static_cast<int>(element(index, y, x)));
        }
    }
    return out;
#endif
}

// ---------------------------------------------------------------------------
// Square operators along y and z against tiles along y and x.
// ---------------------------------------------------------------------------

// ``operator @ planes`` over y, or ``operator.T @ planes``: the operator's rows
// along y and its columns along z, the planes' rows along y.
template <class O, class P>
BSK_HD auto times(const O& op_in, const P& planes_in, bool transposed) {
    using T = decltype(element_t<O>() * element_t<P>());
    const V<T, 6> op(op_in);
    const V<T, 3> planes(planes_in);
    V<T, 3> out(T(0));
#if defined(BLOCHSIM_SIMT)
    // The planes, then the operator, through shared memory: a thread reads the
    // column of planes beneath its state and the operator's entries it needs.
    T* words = reinterpret_cast<T*>(shared_words);
    const int nx = width_x();
    const int ny = width_y();
    const int nz = width_z();
    T* matrix = words + nx * ny;
    __syncthreads();
    words[threadIdx.y * nx + threadIdx.x] = planes.v[0];
    if (threadIdx.x == 0) {
        for (int z = 0; z < MAX_Z; ++z) {
            if (z < nz) {
                matrix[threadIdx.y * nz + z] = op.v[z];
            }
        }
    }
    __syncthreads();
    T total = T(0);
    for (int k = 0; k < ny; ++k) {
        const T entry = transposed ? matrix[k * nz + threadIdx.y] : matrix[threadIdx.y * nz + k];
        total += entry * words[k * nx + threadIdx.x];
    }
    out.v[0] = total;
#else
    for (int i = 0; i < program.ny; ++i) {
        for (int x = 0; x < program.nx; ++x) {
            T total = T(0);
            for (int k = 0; k < program.ny; ++k) {
                const T entry = transposed ? op.at(k, 0, i) : op.at(i, 0, k);
                total += entry * planes.at(k, x);
            }
            out.at(i, x) = total;
        }
    }
#endif
    return out;
}

// ``sum_x left[i, x] right[j, x]``: rows ``i`` along y, columns ``j`` along z.
template <class L, class R>
BSK_HD auto outer(const L& left_in, const R& right_in) {
    using T = decltype(element_t<L>() * element_t<R>());
    const V<T, 3> left(left_in);
    const V<T, 3> right(right_in);
    V<T, 6> out(T(0));
#if defined(BLOCHSIM_SIMT)
    T* words = reinterpret_cast<T*>(shared_words);
    const int nx = width_x();
    const int ny = width_y();
    T* rows = words + nx * ny;
    __syncthreads();
    rows[threadIdx.y * nx + threadIdx.x] = right.v[0];
    __syncthreads();
    const int nz = width_z();
    for (int z = 0; z < MAX_Z; ++z) {
        if (z < nz) {
            out.v[z] = reduce_row(left.v[0] * rows[z * nx + threadIdx.x], Add{});
        }
    }
#else
    for (int i = 0; i < program.ny; ++i) {
        for (int j = 0; j < program.nz; ++j) {
            T total = T(0);
            for (int x = 0; x < program.nx; ++x) {
                total += left.at(i, x) * right.at(j, x);
            }
            out.at(i, 0, j) = total;
        }
    }
#endif
    return out;
}

// ---------------------------------------------------------------------------
// Tuples, which is how a helper returns more than one value.
// ---------------------------------------------------------------------------

template <class... T>
struct tup;

template <>
struct tup<> {};

template <class H, class... T>
struct tup<H, T...> {
    H head;
    tup<T...> tail;

    tup() = default;

    // From a tuple of narrower elements, element by element; a shorter one
    // fills the front.
    template <class UH, class... U>
    BSK_HD tup(const tup<UH, U...>& other) : head(other.head), tail(other.tail) {}

    BSK_HD tup(const tup<>&) : head(), tail() {}
};

template <class H, class... T>
BSK_HD tup<H, T...> make_tup(H head, T... tail) {
    tup<H, T...> out;
    out.head = head;
    if constexpr (sizeof...(T) > 0) {
        out.tail = make_tup(tail...);
    }
    return out;
}

template <std::size_t I, class H, class... T>
BSK_HD auto& get(tup<H, T...>& t) {
    if constexpr (I == 0) {
        return t.head;
    } else {
        return get<I - 1>(t.tail);
    }
}

template <std::size_t I, class H, class... T>
BSK_HD const auto& get(const tup<H, T...>& t) {
    if constexpr (I == 0) {
        return t.head;
    } else {
        return get<I - 1>(t.tail);
    }
}

template <class... T>
struct tile_traits<tup<T...>> {
    static constexpr int axes = (0 | ... | tile_traits<T>::axes);
    using element = void;
};

// ``f`` called with each of ``Start``, ``Start + Step``, ... below ``Stop`` as
// a compile-time constant, for a loop whose counter indexes a tuple.
template <std::int64_t Start, std::int64_t Stop, std::int64_t Step, class F>
BSK_HD void static_for(F&& f) {
    if constexpr (Start < Stop) {
        f(std::integral_constant<std::int64_t, Start>{});
        static_for<Start + Step, Stop, Step>(f);
    }
}

// The axes any of the arguments varies along.
template <class... A>
constexpr int joint_axes = (0 | ... | axes_of<A>);

// The axes a helper's local takes: those of the arguments it was called with
// and those it makes itself, keeping x or z -- never both -- as traced.
BSK_CONSTEXPR int fit_axes(int axes, int traced) {
    return (axes & 5) == 5 ? (axes & ~5) | ((traced & 4) ? 4 : 1) : axes;
}

// A value of element type T varying along AX: a scalar where AX is zero.
template <class T, int AX>
struct tile_type {
    using type = V<T, AX>;
};

template <class T>
struct tile_type<T, 0> {
    using type = T;
};

template <class T, int AX>
using tile_t = typename tile_type<T, AX>::type;

}  // namespace bsk

namespace bsk {

// A value carried into a wider type of the same shape: a tile into a wider
// element type or more axes, a tuple element by element.
template <class R, class A>
BSK_HD R convert(const A& a);

template <class R>
struct converter {
    template <class A>
    BSK_HD static R run(const A& a) {
        return R(a);
    }
};

template <class... T>
struct converter<tup<T...>> {
    template <class... U>
    BSK_HD static tup<T...> run(const tup<U...>& a) {
        tup<T...> out;
        fill<0>(out, a);
        return out;
    }

    template <std::size_t I, class O, class A>
    BSK_HD static void fill(O& out, const A& a) {
        if constexpr (I < sizeof...(T)) {
            get<I>(out) = convert<std::decay_t<decltype(get<I>(out))>>(get<I>(a));
            fill<I + 1>(out, a);
        }
    }
};

template <class R, class A>
BSK_HD R convert(const A& a) {
    return converter<R>::run(a);
}

}  // namespace bsk
