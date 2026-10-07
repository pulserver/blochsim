// A kernel with some of its arguments fixed at compile time.
//
// The kernels take every feature switch as an argument and branch on it, so
// one kernel serves every combination and is compiled for the worst of them:
// the registers and code of every term it might evaluate. Called with those
// switches as constants, the same kernel is compiled for one combination
// alone -- every function it calls is inlined, so a constant switch folds and
// the terms it turns off are never generated. _specializations.json lists the
// combinations compiled this way; the launcher runs one where a launch's
// switches match it exactly and the kernel as compiled for all of them where
// none does.
#pragma once

#include "_kernels.hpp"

namespace bsk {

constexpr bool same_name(const char* a, const char* b) {
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

constexpr int kernel_index(const char* name) {
    int index = 0;
    for (const auto& info : KERNELS) {
        if (same_name(info.name, name)) {
            return index;
        }
        ++index;
    }
    return -1;
}

// Where ``param`` sits in the comma-separated parameter list of ``kernel``.
constexpr int param_index(int kernel, const char* param) {
    const char* p = KERNELS[kernel].params;
    int index = 0;
    while (*p != '\0') {
        const char* q = param;
        const char* s = p;
        while (*q != '\0' && *s == *q) {
            ++q;
            ++s;
        }
        if (*q == '\0' && (*s == ',' || *s == '\0')) {
            return index;
        }
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        if (*p == ',') {
            ++p;
        }
        ++index;
    }
    return -1;
}

// ``Call::run`` with the arguments at ``Fixed``'s even entries replaced by the
// integers that follow them; the copy and the replaced reads fold away. The
// indices are worked out where a constant expression is host code -- an alias
// at namespace scope -- and ``Call`` is a type, so no device function's
// address is taken there.
template <long long... Fixed>
constexpr bool every_index_named() {
    constexpr long long pairs[sizeof...(Fixed) + 1] = {Fixed..., 0};
    for (int k = 0; k + 1 < static_cast<int>(sizeof...(Fixed)); k += 2) {
        if (pairs[k] < 0) {
            return false;
        }
    }
    return true;
}

template <class Call, long long... Fixed>
struct Fixing {
    static_assert(sizeof...(Fixed) % 2 == 0, "fixed arguments come as index, value pairs");
    static_assert(every_index_named<Fixed...>(), "a fixed argument names no parameter");

    BSK_HD static void run(const Arg* in) {
        constexpr long long pairs[sizeof...(Fixed) + 1] = {Fixed..., 0};
        Arg a[MAX_ARGUMENTS];
#pragma unroll
        for (int i = 0; i < MAX_ARGUMENTS; ++i) {
            a[i] = in[i];
        }
#pragma unroll
        for (int k = 0; k + 1 < static_cast<int>(sizeof...(Fixed)); k += 2) {
            a[pairs[k]].i = pairs[k + 1];
        }
        Call::run(a);
    }
};

}  // namespace bsk
