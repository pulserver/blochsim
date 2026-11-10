// The EPG operators of the real subspaces, over a number type: at float they
// are the forward simulation's (_epg_cpu.cpp), at double the train design's
// (_trains_cpu.cpp).
//
// When every pulse of a train turns about one axis and nothing carries a
// state off it -- no off-resonance, no transmit phase -- the complex states
// split into two sets of real ones that never mix:
//
//   across the axis:  F+ = e^{i phi} i a,  F- = e^{-i phi} i b,  Z = c
//   along the axis:   F+ = e^{i phi} a,    F- = e^{-i phi} b,    Z = -i c
//
// with phi the pulses' phase. The first holds what a pulse of the train
// excites out of the longitudinal magnetization, recovery included; the
// second holds the magnetization an excitation a quarter turn from the train
// leaves along its axis, which is the CPMG arrangement. A pulse turns either
// set through the same real 3x3 in its flip angle, relaxation scales both,
// and the shift's conjugate coupling F+0 = conj(F-0) is a0 = -b0 across the
// axis and a0 = b0 along it.
#ifndef BLOCHSIM_REAL_SUBSPACE_HPP
#define BLOCHSIM_REAL_SUBSPACE_HPP

#include <cstddef>

// MSVC rejects the GNU spelling of both of these outright, and the kernel is
// built by whichever compiler the platform ships.
#if defined(_MSC_VER)
#define BLOCHSIM_ALWAYS_INLINE __forceinline
#define BLOCHSIM_RESTRICT __restrict
#else
#define BLOCHSIM_ALWAYS_INLINE __attribute__((always_inline)) inline
#define BLOCHSIM_RESTRICT __restrict__
#endif

namespace blochsim {
namespace real_subspace {

// The sign the shift couples the two zeroth orders with.
template <typename Real>
constexpr Real ACROSS_AXIS = Real(-1);
template <typename Real>
constexpr Real ALONG_AXIS = Real(1);

// One dephasing order: F+ moves up, F- moves down, and the zeroth F+ is the
// conjugate of the zeroth F-, which is ``coupling`` times it.
//
// No states is impossible -- every entry point rejects a state count below
// one -- but not provable here, and the loops below count from ``states - 1``.
template <typename Real>
BLOCHSIM_ALWAYS_INLINE void shift(
    Real* const BLOCHSIM_RESTRICT plus,
    Real* const BLOCHSIM_RESTRICT minus,
    const std::size_t states,
    const Real coupling
) {
    if (states == 0) {
        return;
    }
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = Real(0);
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = coupling * minus[0];
}

// A pulse of flip angle alpha, given as its cosine and sine.
template <typename Real>
BLOCHSIM_ALWAYS_INLINE void rotate(
    Real* const BLOCHSIM_RESTRICT plus,
    Real* const BLOCHSIM_RESTRICT minus,
    Real* const BLOCHSIM_RESTRICT longitudinal,
    const std::size_t states,
    const Real cosine,
    const Real sine
) {
    const Real cosine_half_sq = Real(0.5) * (Real(1) + cosine);
    const Real sine_half_sq = Real(0.5) * (Real(1) - cosine);
    const Real half_sine = Real(0.5) * sine;
    for (std::size_t state = 0; state < states; ++state) {
        const Real p = plus[state];
        const Real m = minus[state];
        const Real z = longitudinal[state];
        plus[state] = cosine_half_sq * p + sine_half_sq * m - sine * z;
        minus[state] = sine_half_sq * p + cosine_half_sq * m + sine * z;
        longitudinal[state] = half_sine * p - half_sine * m + cosine * z;
    }
}

}  // namespace real_subspace
}  // namespace blochsim

#endif  // BLOCHSIM_REAL_SUBSPACE_HPP
