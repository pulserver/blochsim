// A kernel's place in the table and an argument's place in its parameter
// list, by name, as constant expressions: the layouts (_layout.hpp) read
// their arguments with them.
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

}  // namespace bsk
