// The kernels for two or more exchanging pools, forward and adjoint, which
// carry the pools along the tile's y and z axes. Written over the tiles of
// _tile.hpp and included by _kernels.hpp inside ``pools``.

// The lineshape, its slope and its curvature, from the same cubic.
//
// The table covers the magnitude, so the slope changes sign with the offset
// and the curvature does not: an even function's second derivative is even.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _lineshape_at_curve(const T0& lineshape, const T1& offset_hz, const T2& bins, const T3& step) {
    auto last = (bins - 1);
    auto magnitude = bsk::truediv(bsk::abs(offset_hz), step);
    auto scaled = bsk::minimum(magnitude, (last + 0.0f));
    auto lower = bsk::minimum(bsk::floor(scaled), (last - 1.0f));
    auto u = (scaled - lower);
    auto u2 = (u * u);
    auto u3 = (u2 * u);
    auto base = (bsk::cast<std::int64_t>(lower) * 2);
    auto near = bsk::ld((lineshape + base));
    auto near_slope = bsk::ld(((lineshape + base) + 1));
    auto far = bsk::ld(((lineshape + base) + 2));
    auto far_slope = bsk::ld(((lineshape + base) + 3));
    auto value = (((((((2.0f * u3) - (3.0f * u2)) + 1.0f) * near) + ((((u3 - (2.0f * u2)) + u) * step) * near_slope)) + (((-2.0f * u3) + (3.0f * u2)) * far)) + (((u3 - u2) * step) * far_slope));
    auto direction = bsk::where((offset_hz < 0.0f), -1.0f, 1.0f);
    auto slope = (direction * (((bsk::truediv((((6.0f * u2) - (6.0f * u)) * near), step) + ((((3.0f * u2) - (4.0f * u)) + 1.0f) * near_slope)) + bsk::truediv((((-6.0f * u2) + (6.0f * u)) * far), step)) + (((3.0f * u2) - (2.0f * u)) * far_slope)));
    auto curve = (((bsk::truediv((((12.0f * u) - 6.0f) * near), (step * step)) + bsk::truediv((((6.0f * u) - 4.0f) * near_slope), step)) + bsk::truediv((((-12.0f * u) + 6.0f) * far), (step * step))) + bsk::truediv((((6.0f * u) - 2.0f) * far_slope), step));
    auto beyond = (magnitude > last);
    return bsk::make_tup(value, bsk::where(beyond, 0.0f, slope), bsk::where(beyond, 0.0f, curve));
}

// The lineshape and its derivative in the *signed* offset.
//
// The table covers the magnitude, so the slope changes sign with the offset;
// past the last knot the read is constant and the slope is zero.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _lineshape_at_slope(const T0& lineshape, const T1& offset_hz, const T2& bins, const T3& step) {
    auto last = (bins - 1);
    auto magnitude = bsk::truediv(bsk::abs(offset_hz), step);
    auto scaled = bsk::minimum(magnitude, (last + 0.0f));
    auto lower = bsk::minimum(bsk::floor(scaled), (last - 1.0f));
    auto u = (scaled - lower);
    auto u2 = (u * u);
    auto u3 = (u2 * u);
    auto base = (bsk::cast<std::int64_t>(lower) * 2);
    auto near = bsk::ld((lineshape + base));
    auto near_slope = bsk::ld(((lineshape + base) + 1));
    auto far = bsk::ld(((lineshape + base) + 2));
    auto far_slope = bsk::ld(((lineshape + base) + 3));
    auto value = (((((((2.0f * u3) - (3.0f * u2)) + 1.0f) * near) + ((((u3 - (2.0f * u2)) + u) * step) * near_slope)) + (((-2.0f * u3) + (3.0f * u2)) * far)) + (((u3 - u2) * step) * far_slope));
    auto direction = bsk::where((offset_hz < 0.0f), -1.0f, 1.0f);
    auto slope = (direction * (((bsk::truediv((((6.0f * u2) - (6.0f * u)) * near), step) + ((((3.0f * u2) - (4.0f * u)) + 1.0f) * near_slope)) + bsk::truediv((((-6.0f * u2) + (6.0f * u)) * far), step)) + (((3.0f * u2) - (2.0f * u)) * far_slope)));
    return bsk::make_tup(value, bsk::where((magnitude > last), 0.0f, slope));
}

// Two real duals multiplied.
template <class T0, class T1, class T2>
BSK_HD auto _rmul(const T0& x, const T1& y, const T2& following) {
    using Ret = bsk::tup<float, float>;
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup((bsk::get<0>(x) * bsk::get<0>(y)), ((bsk::get<1>(x) * bsk::get<0>(y)) + (bsk::get<0>(x) * bsk::get<1>(y)))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup((bsk::get<0>(x) * bsk::get<0>(y)), 0.0f));
    }
}

// The semisolid pool's saturation by a pulse, and the lineshape it read.
//
// Returns ``exp(saturation * alpha^2 * G(offset))``, ``G`` and its slope in
// the offset, each a real dual.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
BSK_HD auto _absorption(const T0& lineshape, const T1& rf_frequency, const T2& saturation, const T3& event, const T4& alpha, const T5& b0, const T6& lineshape_bins, const T7& lineshape_step, const T8& following) {
    float shape{};
    bsk::tup<float, float> shape_dual{};
    float slope{};
    bsk::tup<float, float> slope_dual{};
    auto offset = (bsk::ld((rf_frequency + event)) - bsk::get<0>(b0));
    auto deposited = bsk::ld((saturation + event));
    if (bsk::truth(following)) {
        auto t0_ = _lineshape_at_curve(lineshape, offset, lineshape_bins, lineshape_step);
        shape = bsk::get<0>(t0_);
        slope = bsk::get<1>(t0_);
        auto curve = bsk::get<2>(t0_);
        // The lineshape is read at the pulse's offset from the voxel, so a
        // step in the voxel's own off-resonance moves the read the other way.
        shape_dual = bsk::make_tup(shape, (slope * (-bsk::get<1>(b0))));
        slope_dual = bsk::make_tup(slope, (curve * (-bsk::get<1>(b0))));
    } else {
        auto t1_ = _lineshape_at_slope(lineshape, offset, lineshape_bins, lineshape_step);
        shape = bsk::get<0>(t1_);
        slope = bsk::get<1>(t1_);
        shape_dual = bsk::make_tup(shape, 0.0f);
        slope_dual = bsk::make_tup(slope, 0.0f);
    }
    auto exponent = _rmul(bsk::make_tup(deposited, 0.0f), _rmul(_rmul(alpha, alpha, following), shape_dual, following), following);
    auto absorbed = bsk::exp(bsk::get<0>(exponent));
    return bsk::make_tup(bsk::make_tup(absorbed, (absorbed * bsk::get<1>(exponent))), shape_dual, slope_dual, deposited);
}

// Two complex duals multiplied.
template <class T0, class T1, class T2>
BSK_HD auto _cmul(const T0& x, const T1& y, const T2& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 3)>>;
    auto real = ((bsk::get<0>(x) * bsk::get<0>(y)) - (bsk::get<1>(x) * bsk::get<1>(y)));
    auto imag = ((bsk::get<0>(x) * bsk::get<1>(y)) + (bsk::get<1>(x) * bsk::get<0>(y)));
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, ((((bsk::get<2>(x) * bsk::get<0>(y)) - (bsk::get<3>(x) * bsk::get<1>(y))) + (bsk::get<0>(x) * bsk::get<2>(y))) - (bsk::get<1>(x) * bsk::get<3>(y))), ((((bsk::get<2>(x) * bsk::get<1>(y)) + (bsk::get<3>(x) * bsk::get<0>(y))) + (bsk::get<0>(x) * bsk::get<3>(y))) + (bsk::get<1>(x) * bsk::get<2>(y)))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, 0.0f, 0.0f));
    }
}

// A real dual times a complex one.
template <class T0, class T1, class T2>
BSK_HD auto _cscale(const T0& r, const T1& x, const T2& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 1)>>;
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup((bsk::get<0>(r) * bsk::get<0>(x)), (bsk::get<0>(r) * bsk::get<1>(x)), ((bsk::get<1>(r) * bsk::get<0>(x)) + (bsk::get<0>(r) * bsk::get<2>(x))), ((bsk::get<1>(r) * bsk::get<1>(x)) + (bsk::get<0>(r) * bsk::get<3>(x)))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup((bsk::get<0>(r) * bsk::get<0>(x)), (bsk::get<0>(r) * bsk::get<1>(x)), 0.0f, 0.0f));
    }
}

// The rotation a pulse performs at this voxel, read rather than read off.
//
// A tabulated pair covers a shape's every pulse because a static array
// reaches the rotation through one complex scalar; this one is integrated per
// pulse per voxel, so there is nothing to interpolate and the read is four
// floats. The row runs per train and per event, as the flip does.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
BSK_HD auto _dynamic_pair_at(const T0& pairs, const T1& pair_index, const T2& event_base, const T3& event, const T4& atom, const T5& atom_count, const T6& mask) {
    auto row = bsk::cast<std::int64_t>(bsk::ld(((pair_index + event_base) + event)));
    auto entry = (pairs + (((row * atom_count) + atom) * 4));
    return bsk::make_tup(bsk::ld((entry + 0), mask, 1.0f), bsk::ld((entry + 1), mask, 0.0f), bsk::ld((entry + 2), mask, 0.0f), bsk::ld((entry + 3), mask, 0.0f));
}

// ``exp(i * angle)`` for a real dual angle.
template <class T0, class T1>
BSK_HD auto _dual_polar(const T0& angle_value, const T1& angle_tangent) {
    auto cosine = bsk::cos(angle_value);
    auto sine = bsk::sin(angle_value);
    return bsk::make_tup(cosine, sine, ((-sine) * angle_tangent), (cosine * angle_tangent));
}

template <class T0, class T1, class T2, class T3>
BSK_HD auto _complex_mul(const T0& a_real, const T1& a_imag, const T2& b_real, const T3& b_imag) {
    return bsk::make_tup(((a_real * b_real) - (a_imag * b_imag)), ((a_real * b_imag) + (a_imag * b_real)));
}

// Product of two dual complex numbers.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
BSK_HD auto _dual_mul(const T0& a_vr, const T1& a_vi, const T2& a_tr, const T3& a_ti, const T4& b_vr, const T5& b_vi, const T6& b_tr, const T7& b_ti) {
    auto t0_ = _complex_mul(a_vr, a_vi, b_vr, b_vi);
    auto value_real = bsk::get<0>(t0_);
    auto value_imag = bsk::get<1>(t0_);
    auto t1_ = _complex_mul(a_tr, a_ti, b_vr, b_vi);
    auto left_real = bsk::get<0>(t1_);
    auto left_imag = bsk::get<1>(t1_);
    auto t2_ = _complex_mul(a_vr, a_vi, b_tr, b_ti);
    auto right_real = bsk::get<0>(t2_);
    auto right_imag = bsk::get<1>(t2_);
    return bsk::make_tup(value_real, value_imag, (left_real + right_real), (left_imag + right_imag));
}

// Two dual complex numbers multiplied.
template <class T0, class T1>
BSK_HD auto _dual_product(const T0& x, const T1& y) {
    return _dual_mul(bsk::get<0>(x), bsk::get<1>(x), bsk::get<2>(x), bsk::get<3>(x), bsk::get<0>(y), bsk::get<1>(y), bsk::get<2>(y), bsk::get<3>(y));
}

// The rotation and the direction along it, with the phase applied.
//
// Shaped exactly as :func:`_profiled_pair_dual` returns, so the spinor
// operator and its adjoint read one from the other without knowing which
// they were handed. A pass that follows no direction holds the rotation
// still, and ``directed`` keeps the read for one out of the kernel.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
BSK_HD auto _dynamic_pair_dual_at(const T0& pairs, const T1& pair_direction, const T2& pair_index, const T3& event_base, const T4& event, const T5& atom, const T6& atom_count, const T7& mask, const T8& phi_value, const T9& phi_tangent, const T10& directed) {
    bsk::tup<float, float, float, float> moved{};
    auto held = _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, mask);
    auto still = (bsk::get<0>(held) * 0.0f);
    moved = bsk::make_tup(still, still, still, still);
    if (bsk::truth(directed)) {
        moved = _dynamic_pair_at(pair_direction, pair_index, event_base, event, atom, atom_count, mask);
    }
    auto a = bsk::make_tup(bsk::get<0>(held), bsk::get<1>(held), bsk::get<0>(moved), bsk::get<1>(moved));
    auto b = bsk::make_tup(bsk::get<2>(held), bsk::get<3>(held), bsk::get<2>(moved), bsk::get<3>(moved));
    auto turn = _dual_polar((-phi_value), (-phi_tangent));
    return bsk::make_tup(a, _dual_product(b, turn));
}

// Table entries at ``offset``, moving with the table's direction and slope.
//
// An event reads the row of its own interval length; a pass following a
// direction in that length moves every entry along the row's slope, which
// sits ``sloped_at`` further into the table.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
BSK_HD auto _entries(const T0& slot, const T1& directions, const T2& offset, const T3& mask, const T4& along, const T5& sloped_at, const T6& directed, const T7& sloped, const T8& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8> | 0, 6)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8> | 0, 6)>>;
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8> | 0, 6)> tangent{};
    auto value = bsk::ld((slot + offset), mask, 0.0f);
    if (bsk::truth(following)) {
        tangent = (value * 0.0f);
        if (bsk::truth(directed)) {
            tangent = (tangent + bsk::ld((directions + offset), mask, 0.0f));
        }
        if (bsk::truth(sloped)) {
            tangent = (tangent + (bsk::ld(((slot + sloped_at) + offset), mask, 0.0f) * along));
        }
        return bsk::convert<Ret>(bsk::make_tup(value, tangent));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(value, 0.0f));
    }
}

// ``exp(i angle)`` for a real dual angle.
template <class T0, class T1>
BSK_HD auto _polar(const T0& angle, const T1& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1> | 0, 1)>>;
    auto cosine = bsk::cos(bsk::get<0>(angle));
    auto sine = bsk::sin(bsk::get<0>(angle));
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup(cosine, sine, ((-sine) * bsk::get<1>(angle)), (cosine * bsk::get<1>(angle))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(cosine, sine, 0.0f, 0.0f));
    }
}

// What an interval does to every pool alike, per dephasing order.
//
// Returns the fraction washout leaves, the transverse and longitudinal
// factors before washout, and the per-order damping weights.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _factors(const T0& dt, const T1& damping_rate, const T2& b0, const T3& flow_rate, const T4& washout_rate, const T5& order, const T6& off_axis, const T7& moving, const T8& diffusing, const T9& following) {
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>> damp_t{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>> damp_z{};
    bsk::tup<float, float> turn{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>> unit_t{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 1)>> unit_z{};
    bsk::tup<float, float> wout{};
    auto squared = (order * order);
    auto transverse_weight = ((squared + order) + 0.3333333333333333f);
    damp_z = bsk::make_tup(((order * 0.0f) + 1.0f), 0.0f);
    damp_t = bsk::make_tup(((order * 0.0f) + 1.0f), 0.0f);
    if (bsk::truth(diffusing)) {
        auto b_factor = _rmul(damping_rate, dt, following);
        auto z = bsk::exp(((-squared) * bsk::get<0>(b_factor)));
        auto t = bsk::exp(((-transverse_weight) * bsk::get<0>(b_factor)));
        if (bsk::truth(following)) {
            damp_z = bsk::make_tup(z, (z * ((-squared) * bsk::get<1>(b_factor))));
            damp_t = bsk::make_tup(t, (t * ((-transverse_weight) * bsk::get<1>(b_factor))));
        } else {
            damp_z = bsk::make_tup(z, 0.0f);
            damp_t = bsk::make_tup(t, 0.0f);
        }
    }
    wout = bsk::make_tup(1.0f, 0.0f);
    if (bsk::truth(moving)) {
        auto fraction = (bsk::get<0>(washout_rate) * bsk::get<0>(dt));
        auto left = (1.0f - bsk::minimum(fraction, 1.0f));
        if (bsk::truth(following)) {
            wout = bsk::make_tup(left, bsk::where((fraction < 1.0f), (-((bsk::get<1>(washout_rate) * bsk::get<0>(dt)) + (bsk::get<0>(washout_rate) * bsk::get<1>(dt)))), 0.0f));
        } else {
            wout = bsk::make_tup(left, 0.0f);
        }
    }
    unit_t = bsk::make_tup(bsk::get<0>(damp_t), (bsk::get<0>(damp_t) * 0.0f), bsk::get<1>(damp_t), 0.0f);
    unit_z = bsk::make_tup(bsk::get<0>(damp_z), (bsk::get<0>(damp_z) * 0.0f), bsk::get<1>(damp_z), 0.0f);
    if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
        auto angle = _rmul(bsk::make_tup(-6.283185307179586f, 0.0f), _rmul(b0, dt, following), following);
        turn = bsk::make_tup(0.0f, 0.0f);
        if (bsk::truth(moving)) {
            turn = _rmul(flow_rate, dt, following);
        }
        auto half = (-(order + 0.5f));
        auto theta = bsk::make_tup((bsk::get<0>(angle) + (half * bsk::get<0>(turn))), (bsk::get<1>(angle) + (half * bsk::get<1>(turn))));
        unit_t = _cscale(damp_t, _polar(theta, following), following);
        if (bsk::truth(moving)) {
            unit_z = _cscale(damp_z, _polar(bsk::make_tup(((-order) * bsk::get<0>(turn)), ((-order) * bsk::get<1>(turn))), following), following);
        }
    }
    if (bsk::truth(following)) {
        // Every tangent a tile, so the operator products can take them.
        unit_t = bsk::make_tup(bsk::get<0>(unit_t), bsk::get<1>(unit_t), (bsk::get<2>(unit_t) + (order * 0.0f)), (bsk::get<3>(unit_t) + (order * 0.0f)));
        unit_z = bsk::make_tup(bsk::get<0>(unit_z), bsk::get<1>(unit_z), (bsk::get<2>(unit_z) + (order * 0.0f)), (bsk::get<3>(unit_z) + (order * 0.0f)));
    }
    return bsk::make_tup(wout, unit_t, unit_z, squared, transverse_weight);
}

// A hard pulse as its Cayley-Klein pair, with the pair's slope in the flip.
//
// ``a = cos(alpha / 2)`` and ``b = -i sin(alpha / 2) exp(-i phi)``, the
// rotation ``_rotate_flip_phase`` performs.
template <class T0, class T1, class T2>
BSK_HD auto _hard_pair(const T0& alpha, const T1& phi, const T2& following) {
    bsk::tup<float, float, float, float> a{};
    bsk::tup<float, float, float, float> b{};
    bsk::tup<float, float, float, float> slope_a{};
    bsk::tup<float, float, float, float> slope_b{};
    auto half = (0.5f * bsk::get<0>(alpha));
    auto cosine = bsk::cos(half);
    auto sine = bsk::sin(half);
    auto nothing = (cosine * 0.0f);
    auto turn = _polar(bsk::make_tup((-bsk::get<0>(phi)), (-bsk::get<1>(phi))), following);
    if (bsk::truth(following)) {
        a = bsk::make_tup(cosine, nothing, ((-0.5f * sine) * bsk::get<1>(alpha)), nothing);
        b = bsk::make_tup(nothing, (-sine), nothing, ((-0.5f * cosine) * bsk::get<1>(alpha)));
        slope_a = bsk::make_tup((-0.5f * sine), nothing, ((-0.25f * cosine) * bsk::get<1>(alpha)), nothing);
        slope_b = bsk::make_tup(nothing, (-0.5f * cosine), nothing, ((0.25f * sine) * bsk::get<1>(alpha)));
    } else {
        a = bsk::make_tup(cosine, nothing, 0.0f, 0.0f);
        b = bsk::make_tup(nothing, (-sine), 0.0f, 0.0f);
        slope_a = bsk::make_tup((-0.5f * sine), nothing, 0.0f, 0.0f);
        slope_b = bsk::make_tup(nothing, (-0.5f * cosine), 0.0f, 0.0f);
    }
    return bsk::make_tup(a, _cmul(b, turn, following), slope_a, _cmul(slope_b, turn, following), turn);
}

// One interval's longitudinal operator, its recovery and transverse operator.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
BSK_HD auto _operators(const T0& slot, const T1& directions, const T2& row_offset, const T3& along, const T4& slope_offset, const T5& pool, const T6& column, const T7& n, const T8& m, const T9& directed, const T10& sloped, const T11& following) {
    auto longitudinal = _entries(slot, directions, ((row_offset + (pool * n)) + column), bsk::band((pool < n), (column < n)), along, slope_offset, directed, sloped, following);
    auto restored = _entries(slot, directions, ((row_offset + (n * n)) + pool), (pool < n), along, slope_offset, directed, sloped, following);
    auto across = (((row_offset + (n * n)) + n) + (2 * ((pool * m) + column)));
    auto carried = bsk::band((pool < m), (column < m));
    auto real = _entries(slot, directions, across, carried, along, slope_offset, directed, sloped, following);
    auto imag = _entries(slot, directions, (across + 1), carried, along, slope_offset, directed, sloped, following);
    return bsk::make_tup(longitudinal, restored, bsk::make_tup(bsk::get<0>(real), bsk::get<0>(imag), bsk::get<1>(real), bsk::get<1>(imag)));
}

// The Cayley-Klein pair the transition table holds at this flip angle.
//
// Cubic Hermite between the two knots bracketing ``theta``, clamped at both
// ends: a cubic run off its grid leaves the unit circle. Each knot is eight
// floats -- the pair then its slope, real before imaginary -- so the two a
// read needs are sixteen contiguous ones.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _profile_pair(const T0& profile, const T1& row, const T2& theta, const T3& bins, const T4& step) {
    auto last = (bins - 1);
    auto scaled = bsk::minimum(bsk::maximum(bsk::truediv(theta, step), 0.0f), (last + 0.0f));
    auto lower = bsk::minimum(bsk::floor(scaled), (last - 1.0f));
    auto u = (scaled - lower);
    auto u2 = (u * u);
    auto u3 = (u2 * u);
    auto h00 = (((2.0f * u3) - (3.0f * u2)) + 1.0f);
    auto h10 = (((u3 - (2.0f * u2)) + u) * step);
    auto h01 = (((-2.0f) * u3) + (3.0f * u2));
    auto h11 = ((u3 - u2) * step);
    auto base = (((row * bins) + bsk::cast<std::int64_t>(lower)) * 8);
    auto component = [&](int c) {
        auto near = bsk::ld(((profile + base) + c));
        auto near_slope = bsk::ld((((profile + base) + 4) + c));
        auto far = bsk::ld((((profile + base) + 8) + c));
        auto far_slope = bsk::ld((((profile + base) + 12) + c));
        return ((((h00 * near) + (h10 * near_slope)) + (h01 * far)) + (h11 * far_slope));
    };
    return bsk::make_tup(component(0), component(1), component(2), component(3));
}

// The pair and its derivative in the flip angle, from the same cubic.
//
// The derivative of a Hermite segment is another polynomial in the same four
// knot values, so reading both costs one extra combination rather than a
// second table. Returned interleaved: each component's value then its slope,
// in the order ``a`` real, ``a`` imaginary, ``b`` real, ``b`` imaginary.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _profile_pair_slope(const T0& profile, const T1& row, const T2& theta, const T3& bins, const T4& step) {
    auto last = (bins - 1);
    auto scaled = bsk::minimum(bsk::maximum(bsk::truediv(theta, step), 0.0f), (last + 0.0f));
    auto lower = bsk::minimum(bsk::floor(scaled), (last - 1.0f));
    auto u = (scaled - lower);
    auto u2 = (u * u);
    auto u3 = (u2 * u);
    auto h00 = (((2.0f * u3) - (3.0f * u2)) + 1.0f);
    auto h10 = (((u3 - (2.0f * u2)) + u) * step);
    auto h01 = (((-2.0f) * u3) + (3.0f * u2));
    auto h11 = ((u3 - u2) * step);
    // d/dtheta is d/du over the knot spacing.
    auto g00 = bsk::truediv(((6.0f * u2) - (6.0f * u)), step);
    auto g10 = (((3.0f * u2) - (4.0f * u)) + 1.0f);
    auto g01 = bsk::truediv(((6.0f * u) - (6.0f * u2)), step);
    auto g11 = ((3.0f * u2) - (2.0f * u));
    auto base = (((row * bins) + bsk::cast<std::int64_t>(lower)) * 8);
    auto near = [&](int c) { return bsk::ld(((profile + base) + c)); };
    auto near_slope = [&](int c) { return bsk::ld((((profile + base) + 4) + c)); };
    auto far = [&](int c) { return bsk::ld((((profile + base) + 8) + c)); };
    auto far_slope = [&](int c) { return bsk::ld((((profile + base) + 12) + c)); };
    auto value = [&](int c) {
        return ((((h00 * near(c)) + (h10 * near_slope(c))) + (h01 * far(c))) + (h11 * far_slope(c)));
    };
    auto slope = [&](int c) {
        return ((((g00 * near(c)) + (g10 * near_slope(c))) + (g01 * far(c))) + (g11 * far_slope(c)));
    };
    return bsk::make_tup(value(0), slope(0), value(1), slope(1), value(2), slope(2), value(3), slope(3));
}

// A buffer entry and the direction along it, or its identity when absent.
template <class T0, class T1, class T2, class T3, class T4, class T5>
BSK_HD auto _read(const T0& values, const T1& directions, const T2& at, const T3& live, const T4& identity, const T5& following) {
    using Ret = bsk::tup<float, float>;
    if (bsk::truth(live)) {
        auto value = bsk::ld((values + at));
        if (bsk::truth(following)) {
            return bsk::convert<Ret>(bsk::make_tup(value, bsk::ld((directions + at))));
        } else {
            return bsk::convert<Ret>(bsk::make_tup(value, 0.0f));
        }
    } else {
        return bsk::convert<Ret>(bsk::make_tup(identity, 0.0f));
    }
}

// ``operator @ planes`` over the pools, or its transpose's.
template <class T0, class T1, class T2>
BSK_HD auto _times(const T0& operator_, const T1& planes, const T2& transposed) {
    return bsk::times(operator_, planes, bsk::truth(transposed));
}

// A complex dual operator applied to complex dual pool tiles.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _apply(const T0& operator_, const T1& planes, const T2& conjugate, const T3& transposed, const T4& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 3)>>;
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 6)> ei{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 6)> eti{};
    auto er = bsk::get<0>(operator_);
    ei = bsk::get<1>(operator_);
    if (bsk::truth(conjugate)) {
        ei = (-ei);
    }
    auto real = (_times(er, bsk::get<0>(planes), transposed) - _times(ei, bsk::get<1>(planes), transposed));
    auto imag = (_times(er, bsk::get<1>(planes), transposed) + _times(ei, bsk::get<0>(planes), transposed));
    if (bsk::truth(following)) {
        auto etr = bsk::get<2>(operator_);
        eti = bsk::get<3>(operator_);
        if (bsk::truth(conjugate)) {
            eti = (-eti);
        }
        auto tangent_real = (((_times(etr, bsk::get<0>(planes), transposed) - _times(eti, bsk::get<1>(planes), transposed)) + _times(er, bsk::get<2>(planes), transposed)) - _times(ei, bsk::get<3>(planes), transposed));
        auto tangent_imag = (((_times(etr, bsk::get<1>(planes), transposed) + _times(eti, bsk::get<0>(planes), transposed)) + _times(er, bsk::get<3>(planes), transposed)) + _times(ei, bsk::get<2>(planes), transposed));
        return bsk::convert<Ret>(bsk::make_tup(real, imag, tangent_real, tangent_imag));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, 0.0f, 0.0f));
    }
}

// A real dual operator applied to complex dual pool tiles.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _apply_real(const T0& operator_, const T1& planes, const T2& transposed, const T3& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3> | 0, 3)>>;
    auto real = _times(bsk::get<0>(operator_), bsk::get<0>(planes), transposed);
    auto imag = _times(bsk::get<0>(operator_), bsk::get<1>(planes), transposed);
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, (_times(bsk::get<1>(operator_), bsk::get<0>(planes), transposed) + _times(bsk::get<0>(operator_), bsk::get<2>(planes), transposed)), (_times(bsk::get<1>(operator_), bsk::get<1>(planes), transposed) + _times(bsk::get<0>(operator_), bsk::get<3>(planes), transposed))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, 0.0f, 0.0f));
    }
}

template <class T0>
BSK_HD auto _conj(const T0& x) {
    return bsk::make_tup(bsk::get<0>(x), (-bsk::get<1>(x)), bsk::get<2>(x), (-bsk::get<3>(x)));
}

// One interval's relaxation and exchange over every order.
//
// Returns the three states it leaves and the operator products before the
// per-order factors, which the adjoint reuses.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
BSK_HD auto _relax(const T0& plus, const T1& minus, const T2& longitudinal, const T3& transverse_op, const T4& longitudinal_op, const T5& restored, const T6& equilibrium, const T7& wout, const T8& carried, const T9& spin, const T10& state, const T11& following) {
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> | 0, 3)>> out_z{};
    auto mixed_plus = _apply(transverse_op, plus, false, false, following);
    auto mixed_minus = _apply(transverse_op, minus, true, false, following);
    auto mixed_z = _apply_real(longitudinal_op, longitudinal, false, following);
    auto out_plus = _cmul(carried, mixed_plus, following);
    auto out_minus = _cmul(_conj(carried), mixed_minus, following);
    out_z = _cmul(spin, mixed_z, following);
    // Inflowing spins arrive at equilibrium, so washout scales what the pools
    // held and not what they recover towards.
    auto origin = (state == 0);
    auto grown = (bsk::get<0>(equilibrium) - (bsk::get<0>(wout) * bsk::get<0>(restored)));
    if (bsk::truth(following)) {
        auto grown_tangent = (bsk::get<1>(equilibrium) - ((bsk::get<1>(wout) * bsk::get<0>(restored)) + (bsk::get<0>(wout) * bsk::get<1>(restored))));
        out_z = bsk::make_tup((bsk::get<0>(out_z) + bsk::where(origin, grown, 0.0f)), bsk::get<1>(out_z), (bsk::get<2>(out_z) + bsk::where(origin, grown_tangent, 0.0f)), bsk::get<3>(out_z));
    } else {
        out_z = bsk::make_tup((bsk::get<0>(out_z) + bsk::where(origin, grown, 0.0f)), bsk::get<1>(out_z), 0.0f, 0.0f);
    }
    return bsk::make_tup(out_plus, out_minus, out_z, mixed_plus, mixed_minus, mixed_z);
}

// The rotation named by its Cayley-Klein pair, applied to the states.
//
// T = [ conj(a)^2   -conj(b)^2   -2 conj(a b) ]
//     [ -b^2         a^2         -2 a b       ]
//     [ conj(a) b    a conj(b)   |a|^2-|b|^2  ]
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _rotate_spinor(const T0& ar, const T1& ai, const T2& br, const T3& bi, const T4& fp_r, const T5& fp_i, const T6& fm_r, const T7& fm_i, const T8& z_r, const T9& z_i) {
    auto aa_r = ((ar * ar) - (ai * ai));
    auto aa_i = ((2.0f * ar) * ai);
    auto bb_r = ((br * br) - (bi * bi));
    auto bb_i = ((2.0f * br) * bi);
    auto ab_r = ((ar * br) - (ai * bi));
    auto ab_i = ((ar * bi) + (ai * br));
    auto t0_ = bsk::make_tup(aa_r, (-aa_i));
    auto t00_r = bsk::get<0>(t0_);
    auto t00_i = bsk::get<1>(t0_);
    auto t1_ = bsk::make_tup((-bb_r), bb_i);
    auto t01_r = bsk::get<0>(t1_);
    auto t01_i = bsk::get<1>(t1_);
    auto t2_ = bsk::make_tup((-2.0f * ab_r), (2.0f * ab_i));
    auto t02_r = bsk::get<0>(t2_);
    auto t02_i = bsk::get<1>(t2_);
    auto t3_ = bsk::make_tup((-bb_r), (-bb_i));
    auto t10_r = bsk::get<0>(t3_);
    auto t10_i = bsk::get<1>(t3_);
    auto t4_ = bsk::make_tup(aa_r, aa_i);
    auto t11_r = bsk::get<0>(t4_);
    auto t11_i = bsk::get<1>(t4_);
    auto t5_ = bsk::make_tup((-2.0f * ab_r), (-2.0f * ab_i));
    auto t12_r = bsk::get<0>(t5_);
    auto t12_i = bsk::get<1>(t5_);
    auto cross_r = ((ar * br) + (ai * bi));
    auto cross_i = ((ar * bi) - (ai * br));
    auto t6_ = bsk::make_tup(cross_r, cross_i);
    auto t20_r = bsk::get<0>(t6_);
    auto t20_i = bsk::get<1>(t6_);
    auto t7_ = bsk::make_tup(cross_r, (-cross_i));
    auto t21_r = bsk::get<0>(t7_);
    auto t21_i = bsk::get<1>(t7_);
    auto t22 = ((((ar * ar) + (ai * ai)) - (br * br)) - (bi * bi));
    auto out_pr = ((((((t00_r * fp_r) - (t00_i * fp_i)) + (t01_r * fm_r)) - (t01_i * fm_i)) + (t02_r * z_r)) - (t02_i * z_i));
    auto out_pi = ((((((t00_r * fp_i) + (t00_i * fp_r)) + (t01_r * fm_i)) + (t01_i * fm_r)) + (t02_r * z_i)) + (t02_i * z_r));
    auto out_mr = ((((((t10_r * fp_r) - (t10_i * fp_i)) + (t11_r * fm_r)) - (t11_i * fm_i)) + (t12_r * z_r)) - (t12_i * z_i));
    auto out_mi = ((((((t10_r * fp_i) + (t10_i * fp_r)) + (t11_r * fm_i)) + (t11_i * fm_r)) + (t12_r * z_i)) + (t12_i * z_r));
    auto out_zr = (((((t20_r * fp_r) - (t20_i * fp_i)) + (t21_r * fm_r)) - (t21_i * fm_i)) + (t22 * z_r));
    auto out_zi = (((((t20_r * fp_i) + (t20_i * fp_r)) + (t21_r * fm_i)) + (t21_i * fm_r)) + (t22 * z_i));
    return bsk::make_tup(out_pr, out_pi, out_mr, out_mi, out_zr, out_zi);
}

// One row of the rotation applied to the states, values only.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
BSK_HD auto _dual_row(const T0& first, const T1& second, const T2& third, const T3& fp_r, const T4& fp_i, const T5& fm_r, const T6& fm_i, const T7& z_r, const T8& z_i) {
    auto real = ((((((bsk::get<0>(first) * fp_r) - (bsk::get<1>(first) * fp_i)) + (bsk::get<0>(second) * fm_r)) - (bsk::get<1>(second) * fm_i)) + (bsk::get<0>(third) * z_r)) - (bsk::get<1>(third) * z_i));
    auto imag = ((((((bsk::get<0>(first) * fp_i) + (bsk::get<1>(first) * fp_r)) + (bsk::get<0>(second) * fm_i)) + (bsk::get<1>(second) * fm_r)) + (bsk::get<0>(third) * z_i)) + (bsk::get<1>(third) * z_r));
    return bsk::make_tup(real, imag);
}

// The rotation's nine coefficients and their tangents.
//
// Every entry is a product of two factors drawn from the pair and its
// conjugate, so five products carry all nine: ``a^2``, ``b^2``, ``a b``,
// ``a conj(b)`` and the norm difference.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
BSK_HD auto _spinor_coefficients(const T0& ar, const T1& ai, const T2& br, const T3& bi, const T4& dar, const T5& dai, const T6& dbr, const T7& dbi) {
    auto aa_r = ((ar * ar) - (ai * ai));
    auto aa_i = ((2.0f * ar) * ai);
    auto daa_r = (2.0f * ((ar * dar) - (ai * dai)));
    auto daa_i = (2.0f * ((dar * ai) + (ar * dai)));
    auto bb_r = ((br * br) - (bi * bi));
    auto bb_i = ((2.0f * br) * bi);
    auto dbb_r = (2.0f * ((br * dbr) - (bi * dbi)));
    auto dbb_i = (2.0f * ((dbr * bi) + (br * dbi)));
    auto ab_r = ((ar * br) - (ai * bi));
    auto ab_i = ((ar * bi) + (ai * br));
    auto dab_r = ((((dar * br) + (ar * dbr)) - (dai * bi)) - (ai * dbi));
    auto dab_i = ((((dar * bi) + (ar * dbi)) + (dai * br)) + (ai * dbr));
    auto cross_r = ((ar * br) + (ai * bi));
    auto cross_i = ((ar * bi) - (ai * br));
    auto dcross_r = ((((dar * br) + (ar * dbr)) + (dai * bi)) + (ai * dbi));
    auto dcross_i = ((((dar * bi) + (ar * dbi)) - (dai * br)) - (ai * dbr));
    auto t22 = ((((ar * ar) + (ai * ai)) - (br * br)) - (bi * bi));
    auto dt22 = (2.0f * ((((ar * dar) + (ai * dai)) - (br * dbr)) - (bi * dbi)));
    return bsk::make_tup(bsk::make_tup(aa_r, (-aa_i), daa_r, (-daa_i)), bsk::make_tup((-bb_r), bb_i, (-dbb_r), dbb_i), bsk::make_tup((-2.0f * ab_r), (2.0f * ab_i), (-2.0f * dab_r), (2.0f * dab_i)), bsk::make_tup((-bb_r), (-bb_i), (-dbb_r), (-dbb_i)), bsk::make_tup(aa_r, aa_i, daa_r, daa_i), bsk::make_tup((-2.0f * ab_r), (-2.0f * ab_i), (-2.0f * dab_r), (-2.0f * dab_i)), bsk::make_tup(cross_r, cross_i, dcross_r, dcross_i), bsk::make_tup(cross_r, (-cross_i), dcross_r, (-dcross_i)), bsk::make_tup(t22, (0.0f * t22), dt22, (0.0f * dt22)));
}

// The same row built from the coefficients' tangents instead.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
BSK_HD auto _tangent_row(const T0& first, const T1& second, const T2& third, const T3& fp_r, const T4& fp_i, const T5& fm_r, const T6& fm_i, const T7& z_r, const T8& z_i) {
    auto real = ((((((bsk::get<2>(first) * fp_r) - (bsk::get<3>(first) * fp_i)) + (bsk::get<2>(second) * fm_r)) - (bsk::get<3>(second) * fm_i)) + (bsk::get<2>(third) * z_r)) - (bsk::get<3>(third) * z_i));
    auto imag = ((((((bsk::get<2>(first) * fp_i) + (bsk::get<3>(first) * fp_r)) + (bsk::get<2>(second) * fm_i)) + (bsk::get<3>(second) * fm_r)) + (bsk::get<2>(third) * z_i)) + (bsk::get<3>(third) * z_r));
    return bsk::make_tup(real, imag);
}

// The spinor rotation carrying a forward-mode tangent.
//
// Both the states and the pair naming the rotation move, so the tangent is
// ``T dx + dT x``.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19>
BSK_HD auto _rotate_spinor_dual(const T0& ar, const T1& ai, const T2& br, const T3& bi, const T4& dar, const T5& dai, const T6& dbr, const T7& dbi, const T8& fp_r, const T9& fp_i, const T10& fm_r, const T11& fm_i, const T12& z_r, const T13& z_i, const T14& dfp_r, const T15& dfp_i, const T16& dfm_r, const T17& dfm_i, const T18& dz_r, const T19& dz_i) {
    auto t0_ = _spinor_coefficients(ar, ai, br, bi, dar, dai, dbr, dbi);
    auto t00 = bsk::get<0>(t0_);
    auto t01 = bsk::get<1>(t0_);
    auto t02 = bsk::get<2>(t0_);
    auto t10 = bsk::get<3>(t0_);
    auto t11 = bsk::get<4>(t0_);
    auto t12 = bsk::get<5>(t0_);
    auto t20 = bsk::get<6>(t0_);
    auto t21 = bsk::get<7>(t0_);
    auto t22 = bsk::get<8>(t0_);
    auto t1_ = _dual_row(t00, t01, t02, fp_r, fp_i, fm_r, fm_i, z_r, z_i);
    auto out_pr = bsk::get<0>(t1_);
    auto out_pi = bsk::get<1>(t1_);
    auto t2_ = _dual_row(t10, t11, t12, fp_r, fp_i, fm_r, fm_i, z_r, z_i);
    auto out_mr = bsk::get<0>(t2_);
    auto out_mi = bsk::get<1>(t2_);
    auto t3_ = _dual_row(t20, t21, t22, fp_r, fp_i, fm_r, fm_i, z_r, z_i);
    auto out_zr = bsk::get<0>(t3_);
    auto out_zi = bsk::get<1>(t3_);
    auto t4_ = _dual_row(t00, t01, t02, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i);
    auto dpr = bsk::get<0>(t4_);
    auto dpi = bsk::get<1>(t4_);
    auto t5_ = _dual_row(t10, t11, t12, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i);
    auto dmr = bsk::get<0>(t5_);
    auto dmi = bsk::get<1>(t5_);
    auto t6_ = _dual_row(t20, t21, t22, dfp_r, dfp_i, dfm_r, dfm_i, dz_r, dz_i);
    auto dzr = bsk::get<0>(t6_);
    auto dzi = bsk::get<1>(t6_);
    auto t7_ = _tangent_row(t00, t01, t02, fp_r, fp_i, fm_r, fm_i, z_r, z_i);
    auto tpr = bsk::get<0>(t7_);
    auto tpi = bsk::get<1>(t7_);
    auto t8_ = _tangent_row(t10, t11, t12, fp_r, fp_i, fm_r, fm_i, z_r, z_i);
    auto tmr = bsk::get<0>(t8_);
    auto tmi = bsk::get<1>(t8_);
    auto t9_ = _tangent_row(t20, t21, t22, fp_r, fp_i, fm_r, fm_i, z_r, z_i);
    auto tzr = bsk::get<0>(t9_);
    auto tzi = bsk::get<1>(t9_);
    return bsk::make_tup(out_pr, out_pi, out_mr, out_mi, out_zr, out_zi, (dpr + tpr), (dpi + tpi), (dmr + tmr), (dmi + tmi), (dzr + tzr), (dzi + tzi));
}

// ``values`` moved one order down: ``result[k] = values[k + 1]``.
//
// The top order has no neighbour to read, so it reads itself and the caller
// masks it away.
template <class T0, class T1>
BSK_HD auto _down(const T0& values, const T1& state) {
    return bsk::gather_x(values, bsk::minimum(state + 1, bsk::width_x() - 1));
}

// ``values`` moved one configuration order up: ``result[k] = values[k - 1]``.
//
// Order zero is left to the caller, which fills it from the sequence's own
// boundary condition rather than from a neighbour.
template <class T0, class T1>
BSK_HD auto _up(const T0& values, const T1& state) {
    return bsk::gather_x(values, bsk::maximum(state - 1, 0));
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
BSK_HD auto _shift(const T0& fplus_real, const T1& fplus_imag, const T2& fminus_real, const T3& fminus_imag, const T4& state, const T5& state_mask, const T6& state_count) {
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6> | 0, 3)> plus_imag{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6> | 0, 3)> plus_real{};
    auto keep_up = bsk::band((state > 0), state_mask);
    auto keep_down = bsk::band(((state + 1) < state_count), state_mask);
    plus_real = bsk::where(keep_up, _up(fplus_real, state), 0.0f);
    plus_imag = bsk::where(keep_up, _up(fplus_imag, state), 0.0f);
    auto minus_real = bsk::where(keep_down, _down(fminus_real, state), 0.0f);
    auto minus_imag = bsk::where(keep_down, _down(fminus_imag, state), 0.0f);
    plus_real = bsk::where((state == 0), minus_real, plus_real);
    plus_imag = bsk::where((state == 0), (-minus_imag), plus_imag);
    return bsk::make_tup(plus_real, plus_imag, minus_real, minus_imag);
}

// Which row of the stacked tables this pulse reads.
//
// Its own shape's block of ``locations`` rows, then the voxel's place along
// the slice.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _table_row(const T0& profile_index, const T1& event, const T2& location, const T3& locations) {
    return ((bsk::cast<std::int64_t>(bsk::ld((profile_index + event))) * locations) + location);
}

// The sum of a tile over the entries ``mask`` keeps.
template <class T0, class T1>
BSK_HD auto _total(const T0& value, const T1& mask) {
    return bsk::sum_y(bsk::sum_x(bsk::where(mask, value, 0.0f)));
}

BSK_HD void _pooled_kernel(float* m0, float* b1, float* b1_phase, float* b0, float* efficiency, float* diffusion, float* velocity, float* dm0, float* db1, float* db1_phase, float* db0, float* defficiency, float* ddiffusion, float* dvelocity, float* duration, std::int32_t* kind, float* flip, float* phase, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* saturation, float* rf_frequency, float* dduration, float* dflip, float* dphase, float* table, float* dtable, std::int32_t* pool_index, float* profile, std::int32_t* profile_index, float* lineshape, float* pairs, std::int32_t* pair_index, float* dpairs, float* output_real, float* output_imag, float* trajectory, std::int64_t base, std::int64_t atom_count, std::int64_t event_count, std::int64_t output_count, std::int64_t state_count, std::int64_t rows, float flow_scale, float washout_scale, float profile_step, float lineshape_step, std::int64_t locations, std::int64_t profile_bins, std::int64_t lineshape_bins, std::int64_t n, std::int64_t m, std::int64_t blocks, std::int64_t planes, std::int64_t atom_stride, std::int64_t shimmed, std::int64_t profiled, std::int64_t dynamic, std::int64_t directed_pairs, std::int64_t directed_table, std::int64_t following, std::int64_t keep, std::int64_t off_axis, std::int64_t moving, std::int64_t diffusing, std::int64_t transmit, std::int64_t density, std::int64_t inverting, std::int64_t P, std::int64_t S) {
    bsk::tup<float, float, float, float> a{};
    bsk::tup<float, float, float, float> b{};
    bsk::V<float, 3> dfmi{};
    bsk::V<float, 3> dfmr{};
    bsk::V<float, 3> dfpi{};
    bsk::V<float, 3> dfpr{};
    bsk::V<float, 3> dzi{};
    bsk::V<float, 3> dzr{};
    bsk::V<float, 3> fmi{};
    bsk::V<float, 3> fmr{};
    bsk::V<float, 3> fpi{};
    bsk::V<float, 3> fpr{};
    bsk::tup<float, float> pulse_b1{};
    bsk::tup<float, float> pulse_b1_phase{};
    bsk::tup<float, float, float, float, float, float, float, float> read{};
    bsk::tup<float, float, float, float> spun{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> turned{};
    bsk::tup<float, float> washout_rate{};
    bsk::V<float, 3> zi{};
    bsk::V<float, 3> zr{};
    auto problem = (bsk::cast<std::int64_t>(bsk::program_id(0)) + base);
    auto atom = bsk::mod(problem, atom_count);
    auto train = bsk::floordiv(problem, atom_count);
    auto event_base = (train * event_count);
    auto voxel_at = (atom * atom_stride);
    auto location = bsk::mod(atom, locations);
    auto live = (problem >= 0);
    auto pool = bsk::arange_y();
    auto column = bsk::arange_z();
    auto state = bsk::arange_x();
    auto state_mask = (state < state_count);
    auto order = bsk::cast<float>(state);
    auto exchanging = (pool < m);
    auto semisolid = (pool == (n - 1));
    auto row_width = (((n * n) + n) + ((2 * m) * m));
    auto width = (n + ((rows * row_width) * blocks));
    auto slot = (table + (atom * width));
    auto directions = (dtable + (atom * width));
    auto density_of = _read(m0, dm0, voxel_at, density, 1.0f, following);
    auto voxel_b1 = _read(b1, db1, voxel_at, transmit, 1.0f, following);
    auto voxel_b1_phase = _read(b1_phase, db1_phase, voxel_at, off_axis, 0.0f, following);
    auto voxel_b0 = _read(b0, db0, voxel_at, off_axis, 0.0f, following);
    auto inversion = _read(efficiency, defficiency, voxel_at, inverting, 1.0f, following);
    auto damping_rate = _read(diffusion, ddiffusion, voxel_at, diffusing, 0.0f, following);
    auto moved = _read(velocity, dvelocity, voxel_at, moving, 0.0f, following);
    auto flow_rate = bsk::make_tup((flow_scale * bsk::get<0>(moved)), (flow_scale * bsk::get<1>(moved)));
    washout_rate = bsk::make_tup(0.0f, 0.0f);
    if (bsk::truth(moving)) {
        auto heading = (bsk::where((bsk::get<0>(moved) > 0.0f), 1.0f, 0.0f) - bsk::where((bsk::get<0>(moved) < 0.0f), 1.0f, 0.0f));
        washout_rate = bsk::make_tup((washout_scale * bsk::abs(bsk::get<0>(moved))), ((washout_scale * heading) * bsk::get<1>(moved)));
    }
    auto equilibrium = _entries(slot, directions, pool, (pool < n), 0.0f, 0, directed_table, false, following);
    auto zero = bsk::full<float, 3>(0);
    fpr = zero;
    fpi = zero;
    fmr = zero;
    fmi = zero;
    zr = bsk::where((state == 0), bsk::get<0>(equilibrium), 0.0f);
    zi = zero;
    dfpr = zero;
    dfpi = zero;
    dfmr = zero;
    dfmi = zero;
    dzr = zero;
    dzi = zero;
    if (bsk::truth(following)) {
        dzr = bsk::where((state == 0), bsk::get<1>(equilibrium), 0.0f);
    }
    auto tile = ((pool * S) + state);
    for (std::int64_t event = 0; event < event_count; event += 1) {
        if (bsk::truth(keep)) {
            auto at = (trajectory + ((((problem - base) * event_count) + event) * ((planes * P) * S)));
            bsk::st(((at + ((0 * P) * S)) + tile), fpr);
            bsk::st(((at + ((1 * P) * S)) + tile), fpi);
            bsk::st(((at + ((2 * P) * S)) + tile), fmr);
            bsk::st(((at + ((3 * P) * S)) + tile), fmi);
            bsk::st(((at + ((4 * P) * S)) + tile), zr);
            bsk::st(((at + ((5 * P) * S)) + tile), zi);
            if (bsk::truth(following)) {
                bsk::st(((at + ((6 * P) * S)) + tile), dfpr);
                bsk::st(((at + ((7 * P) * S)) + tile), dfpi);
                bsk::st(((at + ((8 * P) * S)) + tile), dfmr);
                bsk::st(((at + ((9 * P) * S)) + tile), dfmi);
                bsk::st(((at + ((10 * P) * S)) + tile), dzr);
                bsk::st(((at + ((11 * P) * S)) + tile), dzi);
            }
        }
        auto dt = _read((duration + event_base), (dduration + event_base), event, true, 0.0f, following);
        auto t0_ = _factors(dt, damping_rate, voxel_b0, flow_rate, washout_rate, order, off_axis, moving, diffusing, following);
        auto wout = bsk::get<0>(t0_);
        auto unit_t = bsk::get<1>(t0_);
        auto unit_z = bsk::get<2>(t0_);
        auto _squared = bsk::get<3>(t0_);
        auto _weight = bsk::get<4>(t0_);
        auto carried = _cscale(wout, unit_t, following);
        auto spin = _cscale(wout, unit_z, following);
        auto row = bsk::cast<std::int64_t>(bsk::ld(((pool_index + event_base) + event)));
        auto t1_ = _operators(slot, directions, (n + (row * row_width)), bsk::get<1>(dt), (rows * row_width), pool, column, n, m, directed_table, (blocks > 1), following);
        auto longitudinal_op = bsk::get<0>(t1_);
        auto restored = bsk::get<1>(t1_);
        auto transverse_op = bsk::get<2>(t1_);
        auto t2_ = _relax(bsk::make_tup(fpr, fpi, dfpr, dfpi), bsk::make_tup(fmr, fmi, dfmr, dfmi), bsk::make_tup(zr, zi, dzr, dzi), transverse_op, longitudinal_op, restored, equilibrium, wout, carried, spin, state, following);
        auto plus = bsk::get<0>(t2_);
        auto minus = bsk::get<1>(t2_);
        auto longitudinal = bsk::get<2>(t2_);
        auto _mp = bsk::get<3>(t2_);
        auto _mm = bsk::get<4>(t2_);
        auto _mz = bsk::get<5>(t2_);
        fpr = bsk::get<0>(plus);
        fpi = bsk::get<1>(plus);
        fmr = bsk::get<0>(minus);
        fmi = bsk::get<1>(minus);
        zr = bsk::get<0>(longitudinal);
        zi = bsk::get<1>(longitudinal);
        if (bsk::truth(following)) {
            dfpr = bsk::get<2>(plus);
            dfpi = bsk::get<3>(plus);
            dfmr = bsk::get<2>(minus);
            dfmi = bsk::get<3>(minus);
            dzr = bsk::get<2>(longitudinal);
            dzi = bsk::get<3>(longitudinal);
        }
        auto event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        auto event_kind = bsk::cast<std::int32_t>(bsk::ld((kind + event)));
        if (bsk::truth((bsk::band(event_action, 1) != 0))) {
            auto t3_ = _shift(fpr, fpi, fmr, fmi, state, state_mask, state_count);
            fpr = bsk::get<0>(t3_);
            fpi = bsk::get<1>(t3_);
            fmr = bsk::get<2>(t3_);
            fmi = bsk::get<3>(t3_);
            if (bsk::truth(following)) {
                auto t4_ = _shift(dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count);
                dfpr = bsk::get<0>(t4_);
                dfpi = bsk::get<1>(t4_);
                dfmr = bsk::get<2>(t4_);
                dfmi = bsk::get<3>(t4_);
            }
        }
        if (bsk::truth((event_kind == 1))) {
            if (bsk::truth((bsk::band(event_action, 4) != 0))) {
                // Every exchanging pool is free water and inverts like it; a
                // semisolid one is saturated by the pulse's own term.
                if (bsk::truth(following)) {
                    dzr = bsk::where(exchanging, (-((bsk::get<1>(inversion) * zr) + (bsk::get<0>(inversion) * dzr))), dzr);
                    dzi = bsk::where(exchanging, (-((bsk::get<1>(inversion) * zi) + (bsk::get<0>(inversion) * dzi))), dzi);
                }
                zr = bsk::where(exchanging, ((-bsk::get<0>(inversion)) * zr), zr);
                zi = bsk::where(exchanging, ((-bsk::get<0>(inversion)) * zi), zi);
            } else {
                pulse_b1 = voxel_b1;
                pulse_b1_phase = voxel_b1_phase;
                if (bsk::truth(shimmed)) {
                    auto transmit_at = ((bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count) + atom);
                    pulse_b1 = _read(b1, db1, transmit_at, transmit, 1.0f, following);
                    pulse_b1_phase = _read(b1_phase, db1_phase, transmit_at, true, 0.0f, following);
                }
                auto nominal = _read((flip + event_base), (dflip + event_base), event, true, 0.0f, following);
                auto played = _read((phase + event_base), (dphase + event_base), event, true, 0.0f, following);
                auto alpha = _rmul(nominal, pulse_b1, following);
                auto phi = bsk::make_tup((bsk::get<0>(played) + bsk::get<0>(pulse_b1_phase)), (bsk::get<1>(played) + bsk::get<1>(pulse_b1_phase)));
                if (bsk::truth((n > m))) {
                    auto t5_ = _absorption(lineshape, rf_frequency, saturation, event, alpha, voxel_b0, lineshape_bins, lineshape_step, following);
                    auto absorbed = bsk::get<0>(t5_);
                    auto _shape = bsk::get<1>(t5_);
                    auto _slope = bsk::get<2>(t5_);
                    auto _deposited = bsk::get<3>(t5_);
                    if (bsk::truth(following)) {
                        dzr = bsk::where(semisolid, ((bsk::get<1>(absorbed) * zr) + (bsk::get<0>(absorbed) * dzr)), dzr);
                        dzi = bsk::where(semisolid, ((bsk::get<1>(absorbed) * zi) + (bsk::get<0>(absorbed) * dzi)), dzi);
                    }
                    zr = bsk::where(semisolid, (bsk::get<0>(absorbed) * zr), zr);
                    zi = bsk::where(semisolid, (bsk::get<0>(absorbed) * zi), zi);
                }
                if (bsk::truth(dynamic)) {
                    if (bsk::truth(following)) {
                        auto t6_ = _dynamic_pair_dual_at(pairs, dpairs, pair_index, event_base, event, atom, atom_count, live, bsk::get<0>(phi), bsk::get<1>(phi), directed_pairs);
                        a = bsk::get<0>(t6_);
                        spun = bsk::get<1>(t6_);
                    } else {
                        auto held = _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, live);
                        a = bsk::make_tup(bsk::get<0>(held), bsk::get<1>(held), 0.0f, 0.0f);
                        spun = _cmul(bsk::make_tup(bsk::get<2>(held), bsk::get<3>(held), 0.0f, 0.0f), _polar(bsk::make_tup((-bsk::get<0>(phi)), 0.0f), following), following);
                    }
                } else if (bsk::truth(profiled)) {
                    auto at_row = _table_row(profile_index, event, location, locations);
                    auto turn = _polar(bsk::make_tup((-bsk::get<0>(phi)), (-bsk::get<1>(phi))), following);
                    if (bsk::truth(following)) {
                        read = _profile_pair_slope(profile, at_row, bsk::get<0>(alpha), profile_bins, profile_step);
                        a = bsk::make_tup(bsk::get<0>(read), bsk::get<2>(read), (bsk::get<1>(read) * bsk::get<1>(alpha)), (bsk::get<3>(read) * bsk::get<1>(alpha)));
                        b = bsk::make_tup(bsk::get<4>(read), bsk::get<6>(read), (bsk::get<5>(read) * bsk::get<1>(alpha)), (bsk::get<7>(read) * bsk::get<1>(alpha)));
                    } else {
                        read = _profile_pair(profile, at_row, bsk::get<0>(alpha), profile_bins, profile_step);
                        a = bsk::make_tup(bsk::get<0>(read), bsk::get<1>(read), 0.0f, 0.0f);
                        b = bsk::make_tup(bsk::get<2>(read), bsk::get<3>(read), 0.0f, 0.0f);
                    }
                    spun = _cmul(b, turn, following);
                } else {
                    auto t7_ = _hard_pair(alpha, phi, following);
                    a = bsk::get<0>(t7_);
                    spun = bsk::get<1>(t7_);
                    auto _sa = bsk::get<2>(t7_);
                    auto _sb = bsk::get<3>(t7_);
                    auto _turn = bsk::get<4>(t7_);
                }
                if (bsk::truth(following)) {
                    turned = _rotate_spinor_dual(bsk::get<0>(a), bsk::get<1>(a), bsk::get<0>(spun), bsk::get<1>(spun), bsk::get<2>(a), bsk::get<3>(a), bsk::get<2>(spun), bsk::get<3>(spun), fpr, fpi, fmr, fmi, zr, zi, dfpr, dfpi, dfmr, dfmi, dzr, dzi);
                    dfpr = bsk::where(exchanging, bsk::get<6>(turned), dfpr);
                    dfpi = bsk::where(exchanging, bsk::get<7>(turned), dfpi);
                    dfmr = bsk::where(exchanging, bsk::get<8>(turned), dfmr);
                    dfmi = bsk::where(exchanging, bsk::get<9>(turned), dfmi);
                    dzr = bsk::where(exchanging, bsk::get<10>(turned), dzr);
                    dzi = bsk::where(exchanging, bsk::get<11>(turned), dzi);
                } else {
                    turned = _rotate_spinor(bsk::get<0>(a), bsk::get<1>(a), bsk::get<0>(spun), bsk::get<1>(spun), fpr, fpi, fmr, fmi, zr, zi);
                }
                fpr = bsk::where(exchanging, bsk::get<0>(turned), fpr);
                fpi = bsk::where(exchanging, bsk::get<1>(turned), fpi);
                fmr = bsk::where(exchanging, bsk::get<2>(turned), fmr);
                fmi = bsk::where(exchanging, bsk::get<3>(turned), fmi);
                zr = bsk::where(exchanging, bsk::get<4>(turned), zr);
                zi = bsk::where(exchanging, bsk::get<5>(turned), zi);
            }
        }
        if (bsk::truth((!bsk::truth(keep)))) {
            if (bsk::truth(bsk::band((event_kind == 2), (bsk::band(event_action, 32) != 0)))) {
                auto origin = (state == 0);
                auto recorded = bsk::make_tup(_total(fpr, origin), _total(fpi, origin), _total(dfpr, origin), _total(dfpi, origin));
                auto read_phase = _read((phase + event_base), (dphase + event_base), event, true, 0.0f, following);
                auto signal_ = _cscale(density_of, _cmul(recorded, _polar(bsk::make_tup((-bsk::get<0>(read_phase)), (-bsk::get<1>(read_phase))), following), following), following);
                auto out_ = bsk::ld((output_index + event));
                auto written = ((problem * output_count) + out_);
                if (bsk::truth(following)) {
                    bsk::st((output_real + written), bsk::get<2>(signal_), (out_ >= 0));
                    bsk::st((output_imag + written), bsk::get<3>(signal_), (out_ >= 0));
                } else {
                    bsk::st((output_real + written), bsk::get<0>(signal_), (out_ >= 0));
                    bsk::st((output_imag + written), bsk::get<1>(signal_), (out_ >= 0));
                }
            }
        }
        if (bsk::truth((bsk::band(event_action, 2) != 0))) {
            auto t8_ = _shift(fpr, fpi, fmr, fmi, state, state_mask, state_count);
            fpr = bsk::get<0>(t8_);
            fpi = bsk::get<1>(t8_);
            fmr = bsk::get<2>(t8_);
            fmi = bsk::get<3>(t8_);
            if (bsk::truth(following)) {
                auto t9_ = _shift(dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count);
                dfpr = bsk::get<0>(t9_);
                dfpi = bsk::get<1>(t9_);
                dfmr = bsk::get<2>(t9_);
                dfmi = bsk::get<3>(t9_);
            }
        }
        if (bsk::truth((bsk::band(event_action, 8) != 0))) {
            fpr = zero;
            fpi = zero;
            fmr = zero;
            fmi = zero;
            if (bsk::truth(following)) {
                dfpr = zero;
                dfpi = zero;
                dfmr = zero;
                dfmi = zero;
            }
        } else if (bsk::truth((bsk::band(event_action, 16) != 0))) {
            auto t10_ = _shift(fpr, fpi, fmr, fmi, state, state_mask, state_count);
            fpr = bsk::get<0>(t10_);
            fpi = bsk::get<1>(t10_);
            fmr = bsk::get<2>(t10_);
            fmi = bsk::get<3>(t10_);
            if (bsk::truth(following)) {
                auto t11_ = _shift(dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count);
                dfpr = bsk::get<0>(t11_);
                dfpi = bsk::get<1>(t11_);
                dfmr = bsk::get<2>(t11_);
                dfmi = bsk::get<3>(t11_);
            }
        }
    }
}

template <class T0, class T1>
BSK_HD auto _cadd(const T0& x, const T1& y) {
    return bsk::make_tup((bsk::get<0>(x) + bsk::get<0>(y)), (bsk::get<1>(x) + bsk::get<1>(y)), (bsk::get<2>(x) + bsk::get<2>(y)), (bsk::get<3>(x) + bsk::get<3>(y)));
}

// ``sum_k left[i, k] right[j, k]``: the operator a pair of tiles makes.
template <class T0, class T1>
BSK_HD auto _outer(const T0& left, const T1& right) {
    return bsk::outer(left, right);
}

// ``sum_k left[i, k] right[j, k]`` of two complex dual tiles.
template <class T0, class T1, class T2>
BSK_HD auto _couter(const T0& left, const T1& right, const T2& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 4, 6)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 4, 6)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 4, 6)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 4, 6)>>;
    auto real = (_outer(bsk::get<0>(left), bsk::get<0>(right)) - _outer(bsk::get<1>(left), bsk::get<1>(right)));
    auto imag = (_outer(bsk::get<0>(left), bsk::get<1>(right)) + _outer(bsk::get<1>(left), bsk::get<0>(right)));
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, (((_outer(bsk::get<2>(left), bsk::get<0>(right)) - _outer(bsk::get<3>(left), bsk::get<1>(right))) + _outer(bsk::get<0>(left), bsk::get<2>(right))) - _outer(bsk::get<1>(left), bsk::get<3>(right))), (((_outer(bsk::get<2>(left), bsk::get<1>(right)) + _outer(bsk::get<3>(left), bsk::get<0>(right))) + _outer(bsk::get<0>(left), bsk::get<3>(right))) + _outer(bsk::get<1>(left), bsk::get<2>(right)))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(real, imag, 0.0f, 0.0f));
    }
}

// The pair, its slope and its curvature in the flip angle.
//
// The second-order pass differentiates the read twice, and a Hermite segment
// is a cubic, so all three come from the same four knot values. Returned in
// threes per component: value, slope, curvature.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _profile_pair_curve(const T0& profile, const T1& row, const T2& theta, const T3& bins, const T4& step) {
    auto last = (bins - 1);
    auto scaled = bsk::minimum(bsk::maximum(bsk::truediv(theta, step), 0.0f), (last + 0.0f));
    auto lower = bsk::minimum(bsk::floor(scaled), (last - 1.0f));
    auto u = (scaled - lower);
    auto u2 = (u * u);
    auto u3 = (u2 * u);
    auto h00 = (((2.0f * u3) - (3.0f * u2)) + 1.0f);
    auto h10 = (((u3 - (2.0f * u2)) + u) * step);
    auto h01 = (((-2.0f) * u3) + (3.0f * u2));
    auto h11 = ((u3 - u2) * step);
    auto g00 = bsk::truediv(((6.0f * u2) - (6.0f * u)), step);
    auto g10 = (((3.0f * u2) - (4.0f * u)) + 1.0f);
    auto g01 = bsk::truediv(((6.0f * u) - (6.0f * u2)), step);
    auto g11 = ((3.0f * u2) - (2.0f * u));
    auto c00 = bsk::truediv(((12.0f * u) - 6.0f), (step * step));
    auto c10 = bsk::truediv(((6.0f * u) - 4.0f), step);
    auto c01 = bsk::truediv((6.0f - (12.0f * u)), (step * step));
    auto c11 = bsk::truediv(((6.0f * u) - 2.0f), step);
    auto base = (((row * bins) + bsk::cast<std::int64_t>(lower)) * 8);
    auto near = [&](int c) { return bsk::ld(((profile + base) + c)); };
    auto near_slope = [&](int c) { return bsk::ld((((profile + base) + 4) + c)); };
    auto far = [&](int c) { return bsk::ld((((profile + base) + 8) + c)); };
    auto far_slope = [&](int c) { return bsk::ld((((profile + base) + 12) + c)); };
    auto value = [&](int c) {
        return ((((h00 * near(c)) + (h10 * near_slope(c))) + (h01 * far(c))) + (h11 * far_slope(c)));
    };
    auto slope = [&](int c) {
        return ((((g00 * near(c)) + (g10 * near_slope(c))) + (g01 * far(c))) + (g11 * far_slope(c)));
    };
    auto curve = [&](int c) {
        return ((((c00 * near(c)) + (c10 * near_slope(c))) + (c01 * far(c))) + (c11 * far_slope(c)));
    };
    return bsk::make_tup(value(0), slope(0), curve(0), value(1), slope(1), curve(1),
                         value(2), slope(2), curve(2), value(3), slope(3), curve(3));
}

// The pair a shaped pulse turns through, and its slope, as duals.
//
// The flip angle carries the tangent into the table, so the pair's tangent is
// the stored slope and the slope's own tangent is the segment's curvature.
// The RF phase turns the axis once the pair is out, and so reaches ``b``.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
BSK_HD auto _profiled_pair_dual(const T0& profile, const T1& row, const T2& alpha_value, const T3& alpha_tangent, const T4& phi_value, const T5& phi_tangent, const T6& bins, const T7& step) {
    auto read = _profile_pair_curve(profile, row, alpha_value, bins, step);
    auto a = bsk::make_tup(bsk::get<0>(read), bsk::get<3>(read), (bsk::get<1>(read) * alpha_tangent), (bsk::get<4>(read) * alpha_tangent));
    auto slope_a = bsk::make_tup(bsk::get<1>(read), bsk::get<4>(read), (bsk::get<2>(read) * alpha_tangent), (bsk::get<5>(read) * alpha_tangent));
    auto b = bsk::make_tup(bsk::get<6>(read), bsk::get<9>(read), (bsk::get<7>(read) * alpha_tangent), (bsk::get<10>(read) * alpha_tangent));
    auto slope_b = bsk::make_tup(bsk::get<7>(read), bsk::get<10>(read), (bsk::get<8>(read) * alpha_tangent), (bsk::get<11>(read) * alpha_tangent));
    auto turn = _dual_polar((-phi_value), (-phi_tangent));
    return bsk::make_tup(a, _dual_product(b, turn), slope_a, _dual_product(slope_b, turn));
}

// ``Re(conj(x) y)`` as a real dual.
template <class T0, class T1, class T2>
BSK_HD auto _re_dot(const T0& x, const T1& y, const T2& following) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2> | 0, 3)>>;
    auto value = ((bsk::get<0>(x) * bsk::get<0>(y)) + (bsk::get<1>(x) * bsk::get<1>(y)));
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup(value, ((((bsk::get<2>(x) * bsk::get<0>(y)) + (bsk::get<3>(x) * bsk::get<1>(y))) + (bsk::get<0>(x) * bsk::get<2>(y))) + (bsk::get<1>(x) * bsk::get<3>(y)))));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(value, 0.0f));
    }
}

// A real dual tile summed to a real dual scalar.
template <class T0, class T1, class T2>
BSK_HD auto _rtotal(const T0& x, const T1& mask, const T2& following) {
    using Ret = bsk::tup<float, float>;
    if (bsk::truth(following)) {
        return bsk::convert<Ret>(bsk::make_tup(_total(bsk::get<0>(x), mask), _total(bsk::get<1>(x), mask)));
    } else {
        return bsk::convert<Ret>(bsk::make_tup(_total(bsk::get<0>(x), mask), 0.0f));
    }
}

// Order zero of ``values``, spread across every order.
template <class T0, class T1>
BSK_HD auto _first(const T0& values, const T1& state) {
    return bsk::gather_x(values, state * 0);
}

// Transpose of ``_shift``.
//
// The conjugate refill at order zero sends the incoming plus adjoint back
// onto minus, conjugated, at the index the minus shift moves it to.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
BSK_HD auto _shift_adjoint(const T0& plus_bar_real, const T1& plus_bar_imag, const T2& minus_bar_real, const T3& minus_bar_imag, const T4& state, const T5& state_mask, const T6& state_count) {
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6> | 0, 3)> shifted_mi{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6> | 0, 3)> shifted_mr{};
    auto carry_real = bsk::where(state_mask, _first(plus_bar_real, state), 0.0f);
    auto carry_imag = (-bsk::where(state_mask, _first(plus_bar_imag, state), 0.0f));
    auto forward = bsk::band(((state + 1) < state_count), state_mask);
    auto backward = bsk::band((state > 0), state_mask);
    auto shifted_pr = bsk::where(forward, _down(plus_bar_real, state), 0.0f);
    auto shifted_pi = bsk::where(forward, _down(plus_bar_imag, state), 0.0f);
    shifted_mr = bsk::where(backward, _up(minus_bar_real, state), 0.0f);
    shifted_mi = bsk::where(backward, _up(minus_bar_imag, state), 0.0f);
    shifted_mr = bsk::where((state == 1), (shifted_mr + carry_real), shifted_mr);
    shifted_mi = bsk::where((state == 1), (shifted_mi + carry_imag), shifted_mi);
    return bsk::make_tup(shifted_pr, shifted_pi, shifted_mr, shifted_mi);
}

// The spinor rotation's adjoint, carrying no forward direction.
//
// Returns the cotangent on the Cayley-Klein pair and the three state
// cotangents sent back through the conjugate transpose. Every entry of the
// matrix is a product of two factors drawn from the pair and its conjugate,
// so the pair's two Wirtinger halves are linear in the outer product of the
// seed with the state the rotation acted on -- a closed form rather than a
// differentiated matrix.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15>
BSK_HD auto _spinor_adjoint(const T0& ar, const T1& ai, const T2& br, const T3& bi, const T4& spr, const T5& spi, const T6& smr, const T7& smi, const T8& rzr, const T9& rzi, const T10& pbr, const T11& pbi, const T12& mbr, const T13& mbi, const T14& zbr, const T15& zbi) {
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> | 0, 3)>> n0{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> | 0, 3)>> n1{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> | 0, 3)>> n2{};
    auto aa_r = ((ar * ar) - (ai * ai));
    auto aa_i = ((2.0f * ar) * ai);
    auto bb_r = ((br * br) - (bi * bi));
    auto bb_i = ((2.0f * br) * bi);
    auto ab_r = ((ar * br) - (ai * bi));
    auto ab_i = ((ar * bi) + (ai * br));
    auto cross_r = ((ar * br) + (ai * bi));
    auto cross_i = ((ar * bi) - (ai * br));
    auto t0_ = bsk::make_tup(aa_r, (-aa_i));
    auto t00_r = bsk::get<0>(t0_);
    auto t00_i = bsk::get<1>(t0_);
    auto t1_ = bsk::make_tup((-bb_r), bb_i);
    auto t01_r = bsk::get<0>(t1_);
    auto t01_i = bsk::get<1>(t1_);
    auto t2_ = bsk::make_tup((-2.0f * ab_r), (2.0f * ab_i));
    auto t02_r = bsk::get<0>(t2_);
    auto t02_i = bsk::get<1>(t2_);
    auto t3_ = bsk::make_tup((-bb_r), (-bb_i));
    auto t10_r = bsk::get<0>(t3_);
    auto t10_i = bsk::get<1>(t3_);
    auto t4_ = bsk::make_tup(aa_r, aa_i);
    auto t11_r = bsk::get<0>(t4_);
    auto t11_i = bsk::get<1>(t4_);
    auto t5_ = bsk::make_tup((-2.0f * ab_r), (-2.0f * ab_i));
    auto t12_r = bsk::get<0>(t5_);
    auto t12_i = bsk::get<1>(t5_);
    auto t6_ = bsk::make_tup(cross_r, cross_i);
    auto t20_r = bsk::get<0>(t6_);
    auto t20_i = bsk::get<1>(t6_);
    auto t7_ = bsk::make_tup(cross_r, (-cross_i));
    auto t21_r = bsk::get<0>(t7_);
    auto t21_i = bsk::get<1>(t7_);
    auto t22 = ((((ar * ar) + (ai * ai)) - (br * br)) - (bi * bi));
    // ``m[i][j] = conj(seed_i) * state_j``: the outer product the pair's
    // derivative is linear in.
    auto m00 = _complex_mul(pbr, (-pbi), spr, spi);
    auto m01 = _complex_mul(pbr, (-pbi), smr, smi);
    auto m02 = _complex_mul(pbr, (-pbi), rzr, rzi);
    auto m10 = _complex_mul(mbr, (-mbi), spr, spi);
    auto m11 = _complex_mul(mbr, (-mbi), smr, smi);
    auto m12 = _complex_mul(mbr, (-mbi), rzr, rzi);
    auto m20 = _complex_mul(zbr, (-zbi), spr, spi);
    auto m21 = _complex_mul(zbr, (-zbi), smr, smi);
    auto m22 = _complex_mul(zbr, (-zbi), rzr, rzi);
    auto hca = _complex_mul(ar, ai, bsk::get<0>(m11), bsk::get<1>(m11));
    auto hcb = _complex_mul(br, bi, bsk::get<0>(m12), bsk::get<1>(m12));
    auto hcc = _complex_mul(br, (-bi), bsk::get<0>(m21), bsk::get<1>(m21));
    auto hcd = _complex_mul(ar, (-ai), bsk::get<0>(m22), bsk::get<1>(m22));
    auto holding_conj_a_r = ((((2.0f * bsk::get<0>(hca)) - (2.0f * bsk::get<0>(hcb))) + bsk::get<0>(hcc)) + bsk::get<0>(hcd));
    auto holding_conj_a_i = ((((2.0f * bsk::get<1>(hca)) - (2.0f * bsk::get<1>(hcb))) + bsk::get<1>(hcc)) + bsk::get<1>(hcd));
    auto ha = _complex_mul(ar, (-ai), bsk::get<0>(m00), bsk::get<1>(m00));
    auto hb = _complex_mul(br, (-bi), bsk::get<0>(m02), bsk::get<1>(m02));
    auto hc = _complex_mul(br, bi, bsk::get<0>(m20), bsk::get<1>(m20));
    auto hd = _complex_mul(ar, ai, bsk::get<0>(m22), bsk::get<1>(m22));
    auto holding_a_r = ((((2.0f * bsk::get<0>(ha)) - (2.0f * bsk::get<0>(hb))) + bsk::get<0>(hc)) + bsk::get<0>(hd));
    auto holding_a_i = ((((2.0f * bsk::get<1>(ha)) - (2.0f * bsk::get<1>(hb))) + bsk::get<1>(hc)) + bsk::get<1>(hd));
    auto ka = _complex_mul(br, bi, bsk::get<0>(m10), bsk::get<1>(m10));
    auto kb = _complex_mul(ar, ai, bsk::get<0>(m12), bsk::get<1>(m12));
    auto kc = _complex_mul(ar, (-ai), bsk::get<0>(m20), bsk::get<1>(m20));
    auto kd = _complex_mul(br, (-bi), bsk::get<0>(m22), bsk::get<1>(m22));
    auto holding_conj_b_r = ((((-2.0f * bsk::get<0>(ka)) - (2.0f * bsk::get<0>(kb))) + bsk::get<0>(kc)) - bsk::get<0>(kd));
    auto holding_conj_b_i = ((((-2.0f * bsk::get<1>(ka)) - (2.0f * bsk::get<1>(kb))) + bsk::get<1>(kc)) - bsk::get<1>(kd));
    auto la = _complex_mul(br, (-bi), bsk::get<0>(m01), bsk::get<1>(m01));
    auto lb = _complex_mul(ar, (-ai), bsk::get<0>(m02), bsk::get<1>(m02));
    auto lc = _complex_mul(ar, ai, bsk::get<0>(m21), bsk::get<1>(m21));
    auto ld_ = _complex_mul(br, bi, bsk::get<0>(m22), bsk::get<1>(m22));
    auto holding_b_r = ((((-2.0f * bsk::get<0>(la)) - (2.0f * bsk::get<0>(lb))) + bsk::get<0>(lc)) - bsk::get<0>(ld_));
    auto holding_b_i = ((((-2.0f * bsk::get<1>(la)) - (2.0f * bsk::get<1>(lb))) + bsk::get<1>(lc)) - bsk::get<1>(ld_));
    auto grad_a_r = (holding_conj_a_r + holding_a_r);
    auto grad_a_i = ((-holding_conj_a_i) + holding_a_i);
    auto grad_b_r = (holding_conj_b_r + holding_b_r);
    auto grad_b_i = ((-holding_conj_b_i) + holding_b_i);
    n0 = _complex_mul(t00_r, (-t00_i), pbr, pbi);
    n1 = _complex_mul(t10_r, (-t10_i), mbr, mbi);
    n2 = _complex_mul(t20_r, (-t20_i), zbr, zbi);
    auto t8_ = bsk::make_tup(((bsk::get<0>(n0) + bsk::get<0>(n1)) + bsk::get<0>(n2)), ((bsk::get<1>(n0) + bsk::get<1>(n1)) + bsk::get<1>(n2)));
    auto next_pr = bsk::get<0>(t8_);
    auto next_pi = bsk::get<1>(t8_);
    n0 = _complex_mul(t01_r, (-t01_i), pbr, pbi);
    n1 = _complex_mul(t11_r, (-t11_i), mbr, mbi);
    n2 = _complex_mul(t21_r, (-t21_i), zbr, zbi);
    auto t9_ = bsk::make_tup(((bsk::get<0>(n0) + bsk::get<0>(n1)) + bsk::get<0>(n2)), ((bsk::get<1>(n0) + bsk::get<1>(n1)) + bsk::get<1>(n2)));
    auto next_mr = bsk::get<0>(t9_);
    auto next_mi = bsk::get<1>(t9_);
    n0 = _complex_mul(t02_r, (-t02_i), pbr, pbi);
    n1 = _complex_mul(t12_r, (-t12_i), mbr, mbi);
    auto next_zr = ((bsk::get<0>(n0) + bsk::get<0>(n1)) + (t22 * zbr));
    auto next_zi = ((bsk::get<1>(n0) + bsk::get<1>(n1)) + (t22 * zbi));
    return bsk::make_tup(grad_a_r, grad_a_i, grad_b_r, grad_b_i, next_pr, next_pi, next_mr, next_mi, next_zr, next_zi);
}

// A dual complex number's conjugate, both halves.
template <class T0>
BSK_HD auto _dual_conj(const T0& z) {
    return bsk::make_tup(bsk::get<0>(z), (-bsk::get<1>(z)), bsk::get<2>(z), (-bsk::get<3>(z)));
}

// Four dual complex numbers added.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _dual_sum(const T0& first, const T1& second, const T2& third, const T3& fourth) {
    return bsk::make_tup((((bsk::get<0>(first) + bsk::get<0>(second)) + bsk::get<0>(third)) + bsk::get<0>(fourth)), (((bsk::get<1>(first) + bsk::get<1>(second)) + bsk::get<1>(third)) + bsk::get<1>(fourth)), (((bsk::get<2>(first) + bsk::get<2>(second)) + bsk::get<2>(third)) + bsk::get<2>(fourth)), (((bsk::get<3>(first) + bsk::get<3>(second)) + bsk::get<3>(third)) + bsk::get<3>(fourth)));
}

// A dual complex number scaled by a real constant.
template <class T0, class T1>
BSK_HD auto _dual_weigh(const T0& z, const T1& factor) {
    return bsk::make_tup((factor * bsk::get<0>(z)), (factor * bsk::get<1>(z)), (factor * bsk::get<2>(z)), (factor * bsk::get<3>(z)));
}

// The spinor rotation's adjoint, on dual numbers.
//
// Returns the cotangent on the Cayley-Klein pair and the three state
// cotangents sent back through the conjugate transpose. Every entry of the
// matrix is a product of two factors drawn from the pair and its conjugate,
// so the pair's two Wirtinger halves are linear in the outer product of the
// seed with the state the rotation acted on -- which is why this is a closed
// form rather than a differentiated matrix.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
BSK_HD auto _spinor_adjoint_dual(const T0& a, const T1& b, const T2& sp, const T3& sm, const T4& rz, const T5& pb, const T6& mb, const T7& zb) {
    auto t0_ = _spinor_coefficients(bsk::get<0>(a), bsk::get<1>(a), bsk::get<0>(b), bsk::get<1>(b), bsk::get<2>(a), bsk::get<3>(a), bsk::get<2>(b), bsk::get<3>(b));
    auto t00 = bsk::get<0>(t0_);
    auto t01 = bsk::get<1>(t0_);
    auto t02 = bsk::get<2>(t0_);
    auto t10 = bsk::get<3>(t0_);
    auto t11 = bsk::get<4>(t0_);
    auto t12 = bsk::get<5>(t0_);
    auto t20 = bsk::get<6>(t0_);
    auto t21 = bsk::get<7>(t0_);
    auto t22 = bsk::get<8>(t0_);
    auto conj_pb = _dual_conj(pb);
    auto conj_mb = _dual_conj(mb);
    auto conj_zb = _dual_conj(zb);
    auto m00 = _dual_product(conj_pb, sp);
    auto m01 = _dual_product(conj_pb, sm);
    auto m02 = _dual_product(conj_pb, rz);
    auto m10 = _dual_product(conj_mb, sp);
    auto m11 = _dual_product(conj_mb, sm);
    auto m12 = _dual_product(conj_mb, rz);
    auto m20 = _dual_product(conj_zb, sp);
    auto m21 = _dual_product(conj_zb, sm);
    auto m22 = _dual_product(conj_zb, rz);
    auto conj_a = _dual_conj(a);
    auto conj_b = _dual_conj(b);
    auto holding_conj_a = _dual_sum(_dual_weigh(_dual_product(a, m11), 2.0f), _dual_weigh(_dual_product(b, m12), -2.0f), _dual_product(conj_b, m21), _dual_product(conj_a, m22));
    auto holding_a = _dual_sum(_dual_weigh(_dual_product(conj_a, m00), 2.0f), _dual_weigh(_dual_product(conj_b, m02), -2.0f), _dual_product(b, m20), _dual_product(a, m22));
    auto holding_conj_b = _dual_sum(_dual_weigh(_dual_product(b, m10), -2.0f), _dual_weigh(_dual_product(a, m12), -2.0f), _dual_product(conj_a, m20), _dual_weigh(_dual_product(conj_b, m22), -1.0f));
    auto holding_b = _dual_sum(_dual_weigh(_dual_product(conj_b, m01), -2.0f), _dual_weigh(_dual_product(conj_a, m02), -2.0f), _dual_product(a, m21), _dual_weigh(_dual_product(b, m22), -1.0f));
    auto zero = _dual_weigh(m00, 0.0f);
    auto grad_a = _dual_sum(_dual_conj(holding_conj_a), holding_a, zero, zero);
    auto grad_b = _dual_sum(_dual_conj(holding_conj_b), holding_b, zero, zero);
    auto next_pb = _dual_sum(_dual_product(_dual_conj(t00), pb), _dual_product(_dual_conj(t10), mb), _dual_product(_dual_conj(t20), zb), zero);
    auto next_mb = _dual_sum(_dual_product(_dual_conj(t01), pb), _dual_product(_dual_conj(t11), mb), _dual_product(_dual_conj(t21), zb), zero);
    auto next_zb = _dual_sum(_dual_product(_dual_conj(t02), pb), _dual_product(_dual_conj(t12), mb), _dual_product(_dual_conj(t22), zb), zero);
    return bsk::make_tup(grad_a, grad_b, next_pb, next_mb, next_zb);
}

BSK_HD void _pooled_adjoint_kernel(float* m0, float* b1, float* b1_phase, float* b0, float* efficiency, float* diffusion, float* velocity, float* dm0, float* db1, float* db1_phase, float* db0, float* defficiency, float* ddiffusion, float* dvelocity, float* duration, std::int32_t* kind, float* flip, float* phase, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* saturation, float* rf_frequency, float* dduration, float* dflip, float* dphase, float* table, float* dtable, std::int32_t* pool_index, float* profile, std::int32_t* profile_index, float* lineshape, float* pairs, std::int32_t* pair_index, float* dpairs, float* grad_real, float* grad_imag, float* grad_tissue, float* dgrad_tissue, float* grad_duration, float* dgrad_duration, float* grad_flip, float* dgrad_flip, float* grad_phase, float* dgrad_phase, float* grad_table, float* dgrad_table, float* grad_pairs, float* dgrad_pairs, float* trajectory, std::int64_t base, std::int64_t atom_count, std::int64_t event_count, std::int64_t output_count, std::int64_t state_count, std::int64_t rows, float flow_scale, float washout_scale, float profile_step, float lineshape_step, std::int64_t locations, std::int64_t profile_bins, std::int64_t lineshape_bins, std::int64_t m0_row, std::int64_t b1_row, std::int64_t b1_phase_row, std::int64_t b0_row, std::int64_t efficiency_row, std::int64_t diffusion_row, std::int64_t velocity_row, std::int64_t n, std::int64_t m, std::int64_t blocks, std::int64_t planes, std::int64_t atom_stride, std::int64_t shimmed, std::int64_t profiled, std::int64_t dynamic, std::int64_t directed_pairs, std::int64_t directed_table, std::int64_t following, std::int64_t off_axis, std::int64_t moving, std::int64_t diffusing, std::int64_t transmit, std::int64_t density, std::int64_t inverting, std::int64_t P, std::int64_t S) {
    bsk::tup<float, float, float, float> a{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_z{};
    float curve_b0{};
    float curve_damping{};
    bsk::V<float, 2> curve_eq{};
    float curve_flow{};
    float curve_washout{};
    bsk::V<float, 3> dmbi{};
    bsk::V<float, 3> dmbr{};
    bsk::V<float, 3> dpbi{};
    bsk::V<float, 3> dpbr{};
    bsk::V<float, 3> dsmi{};
    bsk::V<float, 3> dsmr{};
    bsk::V<float, 3> dspi{};
    bsk::V<float, 3> dspr{};
    bsk::V<float, 3> dzbi{};
    bsk::V<float, 3> dzbr{};
    bsk::tup<float, float> fraction_grad{};
    bsk::tup<float, float, float, float> grad_a{};
    bsk::tup<float, float> grad_alpha{};
    bsk::tup<float, float> grad_angle{};
    bsk::tup<float, float, float, float> grad_b{};
    float grad_b0{};
    float grad_b1{};
    float grad_b1_phase{};
    bsk::tup<float, float> grad_b_factor{};
    float grad_damping{};
    float grad_efficiency{};
    bsk::V<float, 2> grad_eq{};
    float grad_flow{};
    bsk::V<float, 2> grad_m0{};
    bsk::tup<float, float> grad_turn{};
    float grad_washout{};
    bsk::tup<float, float> grad_wout{};
    float heading{};
    std::int32_t held{};
    bsk::tup<bsk::V<float, 6>, bsk::V<float, 6>> longitudinal_grad{};
    bsk::V<float, 3> mbi{};
    bsk::V<float, 3> mbr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> minus_bar{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> minus_in{};
    bsk::V<float, 3> pbi{};
    bsk::V<float, 3> pbr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> plus_bar{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> plus_in{};
    bsk::tup<float, float> pulse_b1{};
    bsk::tup<float, float> pulse_b1_phase{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, float, float> seed{};
    bsk::tup<float, float, float, float> slope_a{};
    bsk::tup<float, float, float, float> slope_b{};
    bsk::V<float, 3> smi{};
    bsk::V<float, 3> smr{};
    bsk::V<float, 3> spi{};
    bsk::V<float, 3> spr{};
    bsk::tup<float, float, float, float> spun{};
    bsk::tup<float, float> table_duration{};
    bsk::tup<float, float> taken{};
    std::int32_t transmit_row{};
    bsk::tup<float, float> washout_rate{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> z_bar{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> z_in{};
    bsk::V<float, 3> zbi{};
    bsk::V<float, 3> zbr{};
    auto problem = (bsk::cast<std::int64_t>(bsk::program_id(0)) + base);
    auto atom = bsk::mod(problem, atom_count);
    auto train = bsk::floordiv(problem, atom_count);
    auto event_base = (train * event_count);
    auto voxel_at = (atom * atom_stride);
    auto location = bsk::mod(atom, locations);
    auto live = (problem >= 0);
    auto pool = bsk::arange_y();
    auto column = bsk::arange_z();
    auto state = bsk::arange_x();
    auto state_mask = (state < state_count);
    auto order = bsk::cast<float>(state);
    auto origin = (state == 0);
    auto exchanging = (pool < m);
    auto semisolid = (pool == (n - 1));
    auto held_rows = (pool < n);
    auto square = bsk::band((pool < n), (column < n));
    auto across_mask = bsk::band((pool < m), (column < m));
    auto live_exchanging = bsk::band(exchanging, state_mask);
    auto row_width = (((n * n) + n) + ((2 * m) * m));
    auto width = (n + ((rows * row_width) * blocks));
    auto slot = (table + (atom * width));
    auto directions = (dtable + (atom * width));
    auto slot_grad = (grad_table + (problem * width));
    auto slot_curve = (dgrad_table + (problem * width));
    auto density_of = _read(m0, dm0, voxel_at, density, 1.0f, following);
    auto voxel_b1 = _read(b1, db1, voxel_at, transmit, 1.0f, following);
    auto voxel_b1_phase = _read(b1_phase, db1_phase, voxel_at, off_axis, 0.0f, following);
    auto voxel_b0 = _read(b0, db0, voxel_at, off_axis, 0.0f, following);
    auto inversion = _read(efficiency, defficiency, voxel_at, inverting, 1.0f, following);
    auto damping_rate = _read(diffusion, ddiffusion, voxel_at, diffusing, 0.0f, following);
    auto moved = _read(velocity, dvelocity, voxel_at, moving, 0.0f, following);
    auto flow_rate = bsk::make_tup((flow_scale * bsk::get<0>(moved)), (flow_scale * bsk::get<1>(moved)));
    washout_rate = bsk::make_tup(0.0f, 0.0f);
    heading = 0.0f;
    if (bsk::truth(moving)) {
        heading = (bsk::where((bsk::get<0>(moved) > 0.0f), 1.0f, 0.0f) - bsk::where((bsk::get<0>(moved) < 0.0f), 1.0f, 0.0f));
        washout_rate = bsk::make_tup((washout_scale * bsk::abs(bsk::get<0>(moved))), ((washout_scale * heading) * bsk::get<1>(moved)));
    }
    auto equilibrium = _entries(slot, directions, pool, held_rows, 0.0f, 0, directed_table, false, following);
    auto zero = bsk::full<float, 3>(0);
    pbr = zero;
    pbi = zero;
    mbr = zero;
    mbi = zero;
    zbr = zero;
    zbi = zero;
    dpbr = zero;
    dpbi = zero;
    dmbr = zero;
    dmbi = zero;
    dzbr = zero;
    dzbi = zero;
    grad_eq = (bsk::get<0>(equilibrium) * 0.0f);
    curve_eq = (bsk::get<0>(equilibrium) * 0.0f);
    grad_m0 = 0.0f;
    grad_b1 = 0.0f;
    grad_b1_phase = 0.0f;
    grad_b0 = 0.0f;
    grad_efficiency = 0.0f;
    grad_damping = 0.0f;
    grad_flow = 0.0f;
    grad_washout = 0.0f;
    // Only what every interval adds to is carried in the derivative plane;
    // what a pulse or a readout adds is stored as the branch reaches it.
    curve_b0 = 0.0f;
    curve_damping = 0.0f;
    curve_flow = 0.0f;
    curve_washout = 0.0f;
    // Transmit gradients are summed per shim: the running pair is flushed to
    // its row whenever the walk back reaches a pulse on a different one.
    held = 0;
    auto tile = ((pool * S) + state);
    for (std::int64_t step = 0; step < event_count; step += 1) {
        auto event = ((event_count - 1) - step);
        auto at = (trajectory + ((((problem - base) * event_count) + event) * ((planes * P) * S)));
        if (bsk::truth(following)) {
            plus_in = bsk::make_tup(bsk::ld(((at + ((0 * P) * S)) + tile)), bsk::ld(((at + ((1 * P) * S)) + tile)), bsk::ld(((at + ((6 * P) * S)) + tile)), bsk::ld(((at + ((7 * P) * S)) + tile)));
            minus_in = bsk::make_tup(bsk::ld(((at + ((2 * P) * S)) + tile)), bsk::ld(((at + ((3 * P) * S)) + tile)), bsk::ld(((at + ((8 * P) * S)) + tile)), bsk::ld(((at + ((9 * P) * S)) + tile)));
            z_in = bsk::make_tup(bsk::ld(((at + ((4 * P) * S)) + tile)), bsk::ld(((at + ((5 * P) * S)) + tile)), bsk::ld(((at + ((10 * P) * S)) + tile)), bsk::ld(((at + ((11 * P) * S)) + tile)));
        } else {
            plus_in = bsk::make_tup(bsk::ld((at + tile)), bsk::ld(((at + (P * S)) + tile)), 0.0f, 0.0f);
            minus_in = bsk::make_tup(bsk::ld(((at + ((2 * P) * S)) + tile)), bsk::ld(((at + ((3 * P) * S)) + tile)), 0.0f, 0.0f);
            z_in = bsk::make_tup(bsk::ld(((at + ((4 * P) * S)) + tile)), bsk::ld(((at + ((5 * P) * S)) + tile)), 0.0f, 0.0f);
        }
        auto dt = _read((duration + event_base), (dduration + event_base), event, true, 0.0f, following);
        auto t0_ = _factors(dt, damping_rate, voxel_b0, flow_rate, washout_rate, order, off_axis, moving, diffusing, following);
        auto wout = bsk::get<0>(t0_);
        auto unit_t = bsk::get<1>(t0_);
        auto unit_z = bsk::get<2>(t0_);
        auto squared = bsk::get<3>(t0_);
        auto weight = bsk::get<4>(t0_);
        auto carried = _cscale(wout, unit_t, following);
        auto spin = _cscale(wout, unit_z, following);
        auto row = bsk::cast<std::int64_t>(bsk::ld(((pool_index + event_base) + event)));
        auto row_offset = (n + (row * row_width));
        auto t1_ = _operators(slot, directions, row_offset, bsk::get<1>(dt), (rows * row_width), pool, column, n, m, directed_table, (blocks > 1), following);
        auto longitudinal_op = bsk::get<0>(t1_);
        auto restored = bsk::get<1>(t1_);
        auto transverse_op = bsk::get<2>(t1_);
        // Replay the interval to recover the states the event acted on.
        auto t2_ = _relax(plus_in, minus_in, z_in, transverse_op, longitudinal_op, restored, equilibrium, wout, carried, spin, state, following);
        auto relaxed_plus = bsk::get<0>(t2_);
        auto relaxed_minus = bsk::get<1>(t2_);
        auto relaxed_z = bsk::get<2>(t2_);
        auto mixed_plus = bsk::get<3>(t2_);
        auto mixed_minus = bsk::get<4>(t2_);
        auto mixed_z = bsk::get<5>(t2_);
        spr = bsk::get<0>(relaxed_plus);
        spi = bsk::get<1>(relaxed_plus);
        smr = bsk::get<0>(relaxed_minus);
        smi = bsk::get<1>(relaxed_minus);
        dspr = zero;
        dspi = zero;
        dsmr = zero;
        dsmi = zero;
        if (bsk::truth(following)) {
            dspr = bsk::get<2>(relaxed_plus);
            dspi = bsk::get<3>(relaxed_plus);
            dsmr = bsk::get<2>(relaxed_minus);
            dsmi = bsk::get<3>(relaxed_minus);
        }
        auto event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        auto event_kind = bsk::cast<std::int32_t>(bsk::ld((kind + event)));
        if (bsk::truth((bsk::band(event_action, 1) != 0))) {
            auto t3_ = _shift(spr, spi, smr, smi, state, state_mask, state_count);
            spr = bsk::get<0>(t3_);
            spi = bsk::get<1>(t3_);
            smr = bsk::get<2>(t3_);
            smi = bsk::get<3>(t3_);
            if (bsk::truth(following)) {
                auto t4_ = _shift(dspr, dspi, dsmr, dsmi, state, state_mask, state_count);
                dspr = bsk::get<0>(t4_);
                dspi = bsk::get<1>(t4_);
                dsmr = bsk::get<2>(t4_);
                dsmi = bsk::get<3>(t4_);
            }
        }
        // The trailing shift or spoil.
        if (bsk::truth((bsk::band(event_action, 8) != 0))) {
            pbr = zero;
            pbi = zero;
            mbr = zero;
            mbi = zero;
            if (bsk::truth(following)) {
                dpbr = zero;
                dpbi = zero;
                dmbr = zero;
                dmbi = zero;
            }
        } else if (bsk::truth((bsk::band(event_action, 16) != 0))) {
            auto t5_ = _shift_adjoint(pbr, pbi, mbr, mbi, state, state_mask, state_count);
            pbr = bsk::get<0>(t5_);
            pbi = bsk::get<1>(t5_);
            mbr = bsk::get<2>(t5_);
            mbi = bsk::get<3>(t5_);
            if (bsk::truth(following)) {
                auto t6_ = _shift_adjoint(dpbr, dpbi, dmbr, dmbi, state, state_mask, state_count);
                dpbr = bsk::get<0>(t6_);
                dpbi = bsk::get<1>(t6_);
                dmbr = bsk::get<2>(t6_);
                dmbi = bsk::get<3>(t6_);
            }
        }
        if (bsk::truth((bsk::band(event_action, 2) != 0))) {
            auto t7_ = _shift_adjoint(pbr, pbi, mbr, mbi, state, state_mask, state_count);
            pbr = bsk::get<0>(t7_);
            pbi = bsk::get<1>(t7_);
            mbr = bsk::get<2>(t7_);
            mbi = bsk::get<3>(t7_);
            if (bsk::truth(following)) {
                auto t8_ = _shift_adjoint(dpbr, dpbi, dmbr, dmbi, state, state_mask, state_count);
                dpbr = bsk::get<0>(t8_);
                dpbi = bsk::get<1>(t8_);
                dmbr = bsk::get<2>(t8_);
                dmbi = bsk::get<3>(t8_);
            }
        }
        // The readout. It carries no pulse, so what it recorded is the state
        // the pre-shift left.
        auto out_ = bsk::ld((output_index + event));
        if (bsk::truth(bsk::band(bsk::band((event_kind == 2), (bsk::band(event_action, 32) != 0)), (out_ >= 0)))) {
            auto index_ = ((problem * output_count) + out_);
            seed = bsk::make_tup(bsk::ld((grad_real + index_)), bsk::ld((grad_imag + index_)), 0.0f, 0.0f);
            auto read_phase = _read((phase + event_base), (dphase + event_base), event, true, 0.0f, following);
            auto demodulation = _polar(bsk::make_tup((-bsk::get<0>(read_phase)), (-bsk::get<1>(read_phase))), following);
            auto recorded = bsk::make_tup(_total(spr, origin), _total(spi, origin), _total(dspr, origin), _total(dspi, origin));
            auto density_grad = _re_dot(seed, _cmul(recorded, demodulation, following), following);
            auto turned = bsk::make_tup(bsk::get<1>(demodulation), (-bsk::get<0>(demodulation)), bsk::get<3>(demodulation), (-bsk::get<2>(demodulation)));
            auto phase_grad = _re_dot(seed, _cscale(density_of, _cmul(recorded, turned, following), following), following);
            grad_m0 = (grad_m0 + bsk::get<0>(density_grad));
            bsk::atomic_add(((grad_phase + event_base) + event), bsk::get<0>(phase_grad));
            auto weighted = _cmul(_conj(_cscale(density_of, demodulation, following)), seed, following);
            auto put = bsk::band(origin, exchanging);
            pbr = (pbr + bsk::where(put, bsk::get<0>(weighted), 0.0f));
            pbi = (pbi + bsk::where(put, bsk::get<1>(weighted), 0.0f));
            if (bsk::truth(following)) {
                bsk::atomic_add(((dgrad_tissue + (m0_row * atom_count)) + atom), bsk::get<1>(density_grad));
                bsk::atomic_add(((dgrad_phase + event_base) + event), bsk::get<1>(phase_grad));
                dpbr = (dpbr + bsk::where(put, bsk::get<2>(weighted), 0.0f));
                dpbi = (dpbi + bsk::where(put, bsk::get<3>(weighted), 0.0f));
            }
        }
        // The pulse.
        if (bsk::truth((event_kind == 1))) {
            if (bsk::truth((bsk::band(event_action, 4) != 0))) {
                auto bar = bsk::make_tup(zbr, zbi, dzbr, dzbi);
                taken = _rtotal(_re_dot(bar, relaxed_z, following), live_exchanging, following);
                grad_efficiency = (grad_efficiency - bsk::get<0>(taken));
                if (bsk::truth(following)) {
                    bsk::atomic_add(((dgrad_tissue + (efficiency_row * atom_count)) + atom), (-bsk::get<1>(taken)));
                    dzbr = bsk::where(exchanging, (-((bsk::get<1>(inversion) * zbr) + (bsk::get<0>(inversion) * dzbr))), dzbr);
                    dzbi = bsk::where(exchanging, (-((bsk::get<1>(inversion) * zbi) + (bsk::get<0>(inversion) * dzbi))), dzbi);
                }
                zbr = bsk::where(exchanging, ((-bsk::get<0>(inversion)) * zbr), zbr);
                zbi = bsk::where(exchanging, ((-bsk::get<0>(inversion)) * zbi), zbi);
            } else {
                pulse_b1 = voxel_b1;
                pulse_b1_phase = voxel_b1_phase;
                transmit_row = 0;
                if (bsk::truth(shimmed)) {
                    auto shim = bsk::cast<std::int32_t>(bsk::ld((shim_index + event)));
                    auto changed = (shim != held);
                    auto b1_at = ((bsk::cast<std::int64_t>((b1_row + held)) * atom_count) + atom);
                    auto b1_phase_at = ((bsk::cast<std::int64_t>((b1_phase_row + held)) * atom_count) + atom);
                    bsk::atomic_add((grad_tissue + b1_at), grad_b1, changed);
                    bsk::atomic_add((grad_tissue + b1_phase_at), grad_b1_phase, changed);
                    grad_b1 = bsk::where(changed, 0.0f, grad_b1);
                    grad_b1_phase = bsk::where(changed, 0.0f, grad_b1_phase);
                    held = shim;
                    transmit_row = shim;
                    auto transmit_at = ((bsk::cast<std::int64_t>(shim) * atom_count) + atom);
                    pulse_b1 = _read(b1, db1, transmit_at, transmit, 1.0f, following);
                    pulse_b1_phase = _read(b1_phase, db1_phase, transmit_at, true, 0.0f, following);
                }
                auto nominal = _read((flip + event_base), (dflip + event_base), event, true, 0.0f, following);
                auto played = _read((phase + event_base), (dphase + event_base), event, true, 0.0f, following);
                auto alpha = _rmul(nominal, pulse_b1, following);
                auto phi = bsk::make_tup((bsk::get<0>(played) + bsk::get<0>(pulse_b1_phase)), (bsk::get<1>(played) + bsk::get<1>(pulse_b1_phase)));
                auto turn = _polar(bsk::make_tup((-bsk::get<0>(phi)), (-bsk::get<1>(phi))), following);
                if (bsk::truth(dynamic)) {
                    auto t9_ = _dynamic_pair_dual_at(pairs, dpairs, pair_index, event_base, event, atom, atom_count, live, bsk::get<0>(phi), bsk::get<1>(phi), directed_pairs);
                    a = bsk::get<0>(t9_);
                    spun = bsk::get<1>(t9_);
                    slope_a = a;
                    slope_b = spun;
                } else if (bsk::truth(profiled)) {
                    auto t10_ = _profiled_pair_dual(profile, _table_row(profile_index, event, location, locations), bsk::get<0>(alpha), bsk::get<1>(alpha), bsk::get<0>(phi), bsk::get<1>(phi), profile_bins, profile_step);
                    a = bsk::get<0>(t10_);
                    spun = bsk::get<1>(t10_);
                    slope_a = bsk::get<2>(t10_);
                    slope_b = bsk::get<3>(t10_);
                } else {
                    auto t11_ = _hard_pair(alpha, phi, following);
                    a = bsk::get<0>(t11_);
                    spun = bsk::get<1>(t11_);
                    slope_a = bsk::get<2>(t11_);
                    slope_b = bsk::get<3>(t11_);
                    auto _turn = bsk::get<4>(t11_);
                }
                if (bsk::truth(following)) {
                    auto t12_ = _spinor_adjoint_dual(a, spun, bsk::make_tup(spr, spi, dspr, dspi), bsk::make_tup(smr, smi, dsmr, dsmi), relaxed_z, bsk::make_tup(pbr, pbi, dpbr, dpbi), bsk::make_tup(mbr, mbi, dmbr, dmbi), bsk::make_tup(zbr, zbi, dzbr, dzbi));
                    auto pair_a = bsk::get<0>(t12_);
                    auto pair_b = bsk::get<1>(t12_);
                    auto back_p = bsk::get<2>(t12_);
                    auto back_m = bsk::get<3>(t12_);
                    back_z = bsk::get<4>(t12_);
                    grad_a = bsk::make_tup(_total(bsk::get<0>(pair_a), live_exchanging), _total(bsk::get<1>(pair_a), live_exchanging), _total(bsk::get<2>(pair_a), live_exchanging), _total(bsk::get<3>(pair_a), live_exchanging));
                    grad_b = bsk::make_tup(_total(bsk::get<0>(pair_b), live_exchanging), _total(bsk::get<1>(pair_b), live_exchanging), _total(bsk::get<2>(pair_b), live_exchanging), _total(bsk::get<3>(pair_b), live_exchanging));
                    dpbr = bsk::where(exchanging, bsk::get<2>(back_p), dpbr);
                    dpbi = bsk::where(exchanging, bsk::get<3>(back_p), dpbi);
                    dmbr = bsk::where(exchanging, bsk::get<2>(back_m), dmbr);
                    dmbi = bsk::where(exchanging, bsk::get<3>(back_m), dmbi);
                    dzbr = bsk::where(exchanging, bsk::get<2>(back_z), dzbr);
                    dzbi = bsk::where(exchanging, bsk::get<3>(back_z), dzbi);
                    pbr = bsk::where(exchanging, bsk::get<0>(back_p), pbr);
                    pbi = bsk::where(exchanging, bsk::get<1>(back_p), pbi);
                    mbr = bsk::where(exchanging, bsk::get<0>(back_m), mbr);
                    mbi = bsk::where(exchanging, bsk::get<1>(back_m), mbi);
                    zbr = bsk::where(exchanging, bsk::get<0>(back_z), zbr);
                    zbi = bsk::where(exchanging, bsk::get<1>(back_z), zbi);
                } else {
                    auto back = _spinor_adjoint(bsk::get<0>(a), bsk::get<1>(a), bsk::get<0>(spun), bsk::get<1>(spun), spr, spi, smr, smi, bsk::get<0>(relaxed_z), bsk::get<1>(relaxed_z), pbr, pbi, mbr, mbi, zbr, zbi);
                    grad_a = bsk::make_tup(_total(bsk::get<0>(back), live_exchanging), _total(bsk::get<1>(back), live_exchanging), 0.0f, 0.0f);
                    grad_b = bsk::make_tup(_total(bsk::get<2>(back), live_exchanging), _total(bsk::get<3>(back), live_exchanging), 0.0f, 0.0f);
                    pbr = bsk::where(exchanging, bsk::get<4>(back), pbr);
                    pbi = bsk::where(exchanging, bsk::get<5>(back), pbi);
                    mbr = bsk::where(exchanging, bsk::get<6>(back), mbr);
                    mbi = bsk::where(exchanging, bsk::get<7>(back), mbi);
                    zbr = bsk::where(exchanging, bsk::get<8>(back), zbr);
                    zbi = bsk::where(exchanging, bsk::get<9>(back), zbi);
                }
                // The RF phase turns the axis once the pair is out, so it
                // reaches ``b`` alone -- under every mode.
                auto grad_phi = _re_dot(grad_b, bsk::make_tup(bsk::get<1>(spun), (-bsk::get<0>(spun)), bsk::get<3>(spun), (-bsk::get<2>(spun))), following);
                grad_alpha = bsk::make_tup(0.0f, 0.0f);
                if (bsk::truth(dynamic)) {
                    // The flip is inside the pair rather than read against it,
                    // so the cotangent goes out on the pair; ``b`` was turned
                    // by the phase after the pair came out, so it turns back.
                    auto unturned = _cmul(grad_b, _conj(turn), following);
                    auto entry = (((bsk::cast<std::int64_t>(bsk::ld(((pair_index + event_base) + event))) * atom_count) + atom) * 4);
                    bsk::atomic_add(((grad_pairs + entry) + 0), bsk::get<0>(grad_a));
                    bsk::atomic_add(((grad_pairs + entry) + 1), bsk::get<1>(grad_a));
                    bsk::atomic_add(((grad_pairs + entry) + 2), bsk::get<0>(unturned));
                    bsk::atomic_add(((grad_pairs + entry) + 3), bsk::get<1>(unturned));
                    if (bsk::truth(following)) {
                        bsk::atomic_add(((dgrad_pairs + entry) + 0), bsk::get<2>(grad_a));
                        bsk::atomic_add(((dgrad_pairs + entry) + 1), bsk::get<3>(grad_a));
                        bsk::atomic_add(((dgrad_pairs + entry) + 2), bsk::get<2>(unturned));
                        bsk::atomic_add(((dgrad_pairs + entry) + 3), bsk::get<3>(unturned));
                    }
                } else {
                    auto along_a = _re_dot(grad_a, slope_a, following);
                    auto along_b = _re_dot(grad_b, slope_b, following);
                    grad_alpha = bsk::make_tup((bsk::get<0>(along_a) + bsk::get<0>(along_b)), (bsk::get<1>(along_a) + bsk::get<1>(along_b)));
                }
                if (bsk::truth((n > m))) {
                    // The pulse scales every order of the semisolid pool by one
                    // real number, so its cotangent is one sum over the states.
                    auto t13_ = _absorption(lineshape, rf_frequency, saturation, event, alpha, voxel_b0, lineshape_bins, lineshape_step, following);
                    auto absorbed = bsk::get<0>(t13_);
                    auto shape = bsk::get<1>(t13_);
                    auto slope = bsk::get<2>(t13_);
                    auto deposited = bsk::get<3>(t13_);
                    taken = _rtotal(_re_dot(bsk::make_tup(zbr, zbi, dzbr, dzbi), relaxed_z, following), bsk::band(semisolid, state_mask), following);
                    if (bsk::truth(following)) {
                        dzbr = bsk::where(semisolid, ((bsk::get<1>(absorbed) * zbr) + (bsk::get<0>(absorbed) * dzbr)), dzbr);
                        dzbi = bsk::where(semisolid, ((bsk::get<1>(absorbed) * zbi) + (bsk::get<0>(absorbed) * dzbi)), dzbi);
                    }
                    zbr = bsk::where(semisolid, (bsk::get<0>(absorbed) * zbr), zbr);
                    zbi = bsk::where(semisolid, (bsk::get<0>(absorbed) * zbi), zbi);
                    auto exponent = _rmul(taken, absorbed, following);
                    auto swing = _rmul(bsk::make_tup((2.0f * deposited), 0.0f), _rmul(_rmul(exponent, alpha, following), shape, following), following);
                    grad_alpha = bsk::make_tup((bsk::get<0>(grad_alpha) + bsk::get<0>(swing)), (bsk::get<1>(grad_alpha) + bsk::get<1>(swing)));
                    auto shifted = _rmul(bsk::make_tup(deposited, 0.0f), _rmul(_rmul(_rmul(exponent, alpha, following), alpha, following), slope, following), following);
                    grad_b0 = (grad_b0 - bsk::get<0>(shifted));
                    if (bsk::truth(following)) {
                        bsk::atomic_add(((dgrad_tissue + (b0_row * atom_count)) + atom), (-bsk::get<1>(shifted)));
                    }
                }
                auto flip_grad = _rmul(grad_alpha, pulse_b1, following);
                auto b1_grad = _rmul(grad_alpha, nominal, following);
                bsk::atomic_add(((grad_flip + event_base) + event), bsk::get<0>(flip_grad));
                bsk::atomic_add(((grad_phase + event_base) + event), bsk::get<0>(grad_phi));
                grad_b1 = (grad_b1 + bsk::get<0>(b1_grad));
                grad_b1_phase = (grad_b1_phase + bsk::get<0>(grad_phi));
                if (bsk::truth(following)) {
                    bsk::atomic_add(((dgrad_flip + event_base) + event), bsk::get<1>(flip_grad));
                    bsk::atomic_add(((dgrad_phase + event_base) + event), bsk::get<1>(grad_phi));
                    bsk::atomic_add(((dgrad_tissue + (bsk::cast<std::int64_t>((b1_row + transmit_row)) * atom_count)) + atom), bsk::get<1>(b1_grad));
                    bsk::atomic_add(((dgrad_tissue + (bsk::cast<std::int64_t>((b1_phase_row + transmit_row)) * atom_count)) + atom), bsk::get<1>(grad_phi));
                }
            }
        }
        if (bsk::truth((bsk::band(event_action, 1) != 0))) {
            auto t14_ = _shift_adjoint(pbr, pbi, mbr, mbi, state, state_mask, state_count);
            pbr = bsk::get<0>(t14_);
            pbi = bsk::get<1>(t14_);
            mbr = bsk::get<2>(t14_);
            mbi = bsk::get<3>(t14_);
            if (bsk::truth(following)) {
                auto t15_ = _shift_adjoint(dpbr, dpbi, dmbr, dmbi, state, state_mask, state_count);
                dpbr = bsk::get<0>(t15_);
                dpbi = bsk::get<1>(t15_);
                dmbr = bsk::get<2>(t15_);
                dmbi = bsk::get<3>(t15_);
            }
        }
        // The interval. Order zero also carries the recovery, which is the
        // equilibrium less what washout leaves of the operator applied to it.
        plus_bar = bsk::make_tup(pbr, pbi, dpbr, dpbi);
        minus_bar = bsk::make_tup(mbr, mbi, dmbr, dmbi);
        z_bar = bsk::make_tup(zbr, zbi, dzbr, dzbi);
        if (bsk::truth((!bsk::truth(following)))) {
            plus_bar = bsk::make_tup(pbr, pbi, 0.0f, 0.0f);
            minus_bar = bsk::make_tup(mbr, mbi, 0.0f, 0.0f);
            z_bar = bsk::make_tup(zbr, zbi, 0.0f, 0.0f);
        }
        seed = bsk::make_tup(bsk::sum_x(bsk::where(origin, zbr, 0.0f)), bsk::sum_x(bsk::where(origin, dzbr, 0.0f)));
        grad_eq = (grad_eq + bsk::get<0>(seed));
        auto restored_grad = bsk::make_tup((-(bsk::get<0>(wout) * bsk::get<0>(seed))), (-((bsk::get<1>(wout) * bsk::get<0>(seed)) + (bsk::get<0>(wout) * bsk::get<1>(seed)))));
        grad_wout = bsk::make_tup((-_total((bsk::get<0>(seed) * bsk::get<0>(restored)), held_rows)), (-_total(((bsk::get<1>(seed) * bsk::get<0>(restored)) + (bsk::get<0>(seed) * bsk::get<1>(restored))), held_rows)));
        if (bsk::truth(following)) {
            curve_eq = (curve_eq + bsk::get<1>(seed));
        }
        auto out_plus = _cmul(_conj(plus_bar), _cmul(carried, mixed_plus, following), following);
        auto out_minus = _cmul(_conj(minus_bar), _cmul(_conj(carried), mixed_minus, following), following);
        auto out_z = _cmul(_conj(z_bar), _cmul(spin, mixed_z, following), following);
        // The damping is homogeneous of degree one in every state it acts on,
        // so its gradient times the damping itself is the cotangent taken
        // against the states the interval leaves; the turns are the same
        // derivatives with an imaginary weight.
        auto transverse_scaled = bsk::sum_y((bsk::get<0>(out_plus) + bsk::get<0>(out_minus)));
        auto transverse_angle = bsk::sum_y((bsk::get<1>(out_minus) - bsk::get<1>(out_plus)));
        auto longitudinal_scaled = bsk::sum_y(bsk::get<0>(out_z));
        auto longitudinal_angle = bsk::sum_y((-bsk::get<1>(out_z)));
        auto wout_plus = _re_dot(plus_bar, _cmul(unit_t, mixed_plus, following), following);
        auto wout_minus = _re_dot(minus_bar, _cmul(_conj(unit_t), mixed_minus, following), following);
        auto wout_z = _re_dot(z_bar, _cmul(unit_z, mixed_z, following), following);
        grad_wout = bsk::make_tup((bsk::get<0>(grad_wout) + _total(((bsk::get<0>(wout_plus) + bsk::get<0>(wout_minus)) + bsk::get<0>(wout_z)), state_mask)), bsk::get<1>(grad_wout));
        auto half = (order + 0.5f);
        grad_angle = bsk::make_tup(_total(transverse_angle, state_mask), 0.0f);
        grad_b_factor = bsk::make_tup((-_total(((weight * transverse_scaled) + (squared * longitudinal_scaled)), state_mask)), 0.0f);
        grad_turn = bsk::make_tup((-_total(((half * transverse_angle) + (order * longitudinal_angle)), state_mask)), 0.0f);
        if (bsk::truth(following)) {
            grad_wout = bsk::make_tup(bsk::get<0>(grad_wout), (bsk::get<1>(grad_wout) + _total(((bsk::get<1>(wout_plus) + bsk::get<1>(wout_minus)) + bsk::get<1>(wout_z)), state_mask)));
            auto transverse_scaled_t = bsk::sum_y((bsk::get<2>(out_plus) + bsk::get<2>(out_minus)));
            auto transverse_angle_t = bsk::sum_y((bsk::get<3>(out_minus) - bsk::get<3>(out_plus)));
            auto longitudinal_scaled_t = bsk::sum_y(bsk::get<2>(out_z));
            auto longitudinal_angle_t = bsk::sum_y((-bsk::get<3>(out_z)));
            grad_angle = bsk::make_tup(bsk::get<0>(grad_angle), _total(transverse_angle_t, state_mask));
            grad_b_factor = bsk::make_tup(bsk::get<0>(grad_b_factor), (-_total(((weight * transverse_scaled_t) + (squared * longitudinal_scaled_t)), state_mask)));
            grad_turn = bsk::make_tup(bsk::get<0>(grad_turn), (-_total(((half * transverse_angle_t) + (order * longitudinal_angle_t)), state_mask)));
        }
        // The operators' own entries. ``F-`` follows the conjugate of the
        // transverse operator, so its cotangent lands on the entry itself.
        auto released = _conj(carried);
        auto transverse_grad = _cadd(_couter(_cmul(plus_bar, released, following), _conj(plus_in), following), _couter(_cmul(_conj(minus_bar), released, following), minus_in, following));
        auto weighed = _cmul(_conj(z_bar), spin, following);
        longitudinal_grad = bsk::make_tup((_outer(bsk::get<0>(weighed), bsk::get<0>(z_in)) - _outer(bsk::get<1>(weighed), bsk::get<1>(z_in))), 0.0f);
        if (bsk::truth(following)) {
            longitudinal_grad = bsk::make_tup(bsk::get<0>(longitudinal_grad), (((_outer(bsk::get<2>(weighed), bsk::get<0>(z_in)) - _outer(bsk::get<3>(weighed), bsk::get<1>(z_in))) + _outer(bsk::get<0>(weighed), bsk::get<2>(z_in))) - _outer(bsk::get<1>(weighed), bsk::get<3>(z_in))));
        }
        // The cotangents back through the interval.
        auto back_plus = _cmul(released, _apply(transverse_op, plus_bar, true, true, following), following);
        auto back_minus = _cmul(carried, _apply(transverse_op, minus_bar, false, true, following), following);
        back_z = _apply_real(longitudinal_op, _cmul(_conj(spin), z_bar, following), true, following);
        pbr = bsk::get<0>(back_plus);
        pbi = bsk::get<1>(back_plus);
        mbr = bsk::get<0>(back_minus);
        mbi = bsk::get<1>(back_minus);
        zbr = bsk::get<0>(back_z);
        zbi = bsk::get<1>(back_z);
        if (bsk::truth(following)) {
            dpbr = bsk::get<2>(back_plus);
            dpbi = bsk::get<3>(back_plus);
            dmbr = bsk::get<2>(back_minus);
            dmbi = bsk::get<3>(back_minus);
            dzbr = bsk::get<2>(back_z);
            dzbi = bsk::get<3>(back_z);
        }
        // The row this event read is shared with every event of its length;
        // its cotangent is summed into it, and reaches the event's own length
        // through the row's slope.
        auto longitudinal_at = ((row_offset + (pool * n)) + column);
        auto restored_at = ((row_offset + (n * n)) + pool);
        auto across = (((row_offset + (n * n)) + n) + (2 * ((pool * m) + column)));
        bsk::atomic_add((slot_grad + longitudinal_at), bsk::get<0>(longitudinal_grad), square);
        bsk::atomic_add((slot_grad + restored_at), bsk::get<0>(restored_grad), held_rows);
        bsk::atomic_add((slot_grad + across), bsk::get<0>(transverse_grad), across_mask);
        bsk::atomic_add(((slot_grad + across) + 1), bsk::get<1>(transverse_grad), across_mask);
        if (bsk::truth(following)) {
            bsk::atomic_add((slot_curve + longitudinal_at), bsk::get<1>(longitudinal_grad), square);
            bsk::atomic_add((slot_curve + restored_at), bsk::get<1>(restored_grad), held_rows);
            bsk::atomic_add((slot_curve + across), bsk::get<2>(transverse_grad), across_mask);
            bsk::atomic_add(((slot_curve + across) + 1), bsk::get<3>(transverse_grad), across_mask);
        }
        table_duration = bsk::make_tup(0.0f, 0.0f);
        if (bsk::truth((blocks > 1))) {
            auto further = (rows * row_width);
            auto slope_z = _entries(slot, directions, (further + longitudinal_at), square, bsk::get<1>(dt), further, directed_table, true, following);
            auto slope_restored = _entries(slot, directions, (further + restored_at), held_rows, bsk::get<1>(dt), further, directed_table, true, following);
            auto slope_real = _entries(slot, directions, (further + across), across_mask, bsk::get<1>(dt), further, directed_table, true, following);
            auto slope_imag = _entries(slot, directions, ((further + across) + 1), across_mask, bsk::get<1>(dt), further, directed_table, true, following);
            table_duration = bsk::make_tup(((_total((bsk::get<0>(longitudinal_grad) * bsk::get<0>(slope_z)), square) + _total((bsk::get<0>(restored_grad) * bsk::get<0>(slope_restored)), held_rows)) + _total(((bsk::get<0>(transverse_grad) * bsk::get<0>(slope_real)) + (bsk::get<1>(transverse_grad) * bsk::get<0>(slope_imag))), across_mask)), 0.0f);
            if (bsk::truth(following)) {
                table_duration = bsk::make_tup(bsk::get<0>(table_duration), ((_total(((bsk::get<1>(longitudinal_grad) * bsk::get<0>(slope_z)) + (bsk::get<0>(longitudinal_grad) * bsk::get<1>(slope_z))), square) + _total(((bsk::get<1>(restored_grad) * bsk::get<0>(slope_restored)) + (bsk::get<0>(restored_grad) * bsk::get<1>(slope_restored))), held_rows)) + _total(((((bsk::get<2>(transverse_grad) * bsk::get<0>(slope_real)) + (bsk::get<0>(transverse_grad) * bsk::get<1>(slope_real))) + (bsk::get<3>(transverse_grad) * bsk::get<0>(slope_imag))) + (bsk::get<1>(transverse_grad) * bsk::get<1>(slope_imag))), across_mask)));
                // At the row's own length the slope reaches the output only
                // through a direction in that length, so only the tangent
                // plane takes this.
                bsk::atomic_add(((slot_curve + further) + longitudinal_at), (bsk::get<0>(longitudinal_grad) * bsk::get<1>(dt)), square);
                bsk::atomic_add(((slot_curve + further) + restored_at), (bsk::get<0>(restored_grad) * bsk::get<1>(dt)), held_rows);
                bsk::atomic_add(((slot_curve + further) + across), (bsk::get<0>(transverse_grad) * bsk::get<1>(dt)), across_mask);
                bsk::atomic_add((((slot_curve + further) + across) + 1), (bsk::get<1>(transverse_grad) * bsk::get<1>(dt)), across_mask);
            }
        }
        // Washout scales every factor the interval applies and the recovery it
        // leaves; past the clamp nothing depends on the rate.
        fraction_grad = bsk::make_tup(0.0f, 0.0f);
        if (bsk::truth(moving)) {
            auto inside = ((bsk::get<0>(washout_rate) * bsk::get<0>(dt)) < 1.0f);
            fraction_grad = bsk::make_tup(bsk::where(inside, (-bsk::get<0>(grad_wout)), 0.0f), bsk::where(inside, (-bsk::get<1>(grad_wout)), 0.0f));
        }
        auto angle_rate = _rmul(bsk::make_tup(-6.283185307179586f, 0.0f), dt, following);
        auto b0_grad = _rmul(grad_angle, angle_rate, following);
        auto damping_grad = _rmul(grad_b_factor, dt, following);
        auto flow_grad = _rmul(grad_turn, dt, following);
        auto washout_grad = _rmul(fraction_grad, dt, following);
        auto duration_grad = _rmul(grad_angle, _rmul(bsk::make_tup(-6.283185307179586f, 0.0f), voxel_b0, following), following);
        auto through_damping = _rmul(grad_b_factor, damping_rate, following);
        auto through_flow = _rmul(grad_turn, flow_rate, following);
        auto through_washout = _rmul(fraction_grad, washout_rate, following);
        grad_b0 = (grad_b0 + bsk::get<0>(b0_grad));
        grad_damping = (grad_damping + bsk::get<0>(damping_grad));
        grad_flow = (grad_flow + bsk::get<0>(flow_grad));
        grad_washout = (grad_washout + bsk::get<0>(washout_grad));
        bsk::atomic_add(((grad_duration + event_base) + event), ((((bsk::get<0>(duration_grad) + bsk::get<0>(through_damping)) + bsk::get<0>(through_flow)) + bsk::get<0>(through_washout)) + bsk::get<0>(table_duration)));
        if (bsk::truth(following)) {
            curve_b0 = (curve_b0 + bsk::get<1>(b0_grad));
            curve_damping = (curve_damping + bsk::get<1>(damping_grad));
            curve_flow = (curve_flow + bsk::get<1>(flow_grad));
            curve_washout = (curve_washout + bsk::get<1>(washout_grad));
            bsk::atomic_add(((dgrad_duration + event_base) + event), ((((bsk::get<1>(duration_grad) + bsk::get<1>(through_damping)) + bsk::get<1>(through_flow)) + bsk::get<1>(through_washout)) + bsk::get<1>(table_duration)));
        }
    }
    // The equilibrium is also where every pool starts, which the walk back
    // reaches last.
    grad_eq = (grad_eq + bsk::sum_x(bsk::where(origin, zbr, 0.0f)));
    bsk::atomic_add((slot_grad + pool), grad_eq, held_rows);
    bsk::atomic_add(((grad_tissue + (m0_row * atom_count)) + atom), grad_m0);
    bsk::atomic_add(((grad_tissue + (bsk::cast<std::int64_t>((b1_row + held)) * atom_count)) + atom), grad_b1);
    bsk::atomic_add(((grad_tissue + (bsk::cast<std::int64_t>((b1_phase_row + held)) * atom_count)) + atom), grad_b1_phase);
    bsk::atomic_add(((grad_tissue + (b0_row * atom_count)) + atom), grad_b0);
    bsk::atomic_add(((grad_tissue + (efficiency_row * atom_count)) + atom), grad_efficiency);
    bsk::atomic_add(((grad_tissue + (diffusion_row * atom_count)) + atom), grad_damping);
    // One buffer drives two rates, so the velocity gradient is the sum of what
    // each geometry carries back.
    bsk::atomic_add(((grad_tissue + (velocity_row * atom_count)) + atom), ((flow_scale * grad_flow) + ((heading * washout_scale) * grad_washout)));
    if (bsk::truth(following)) {
        curve_eq = (curve_eq + bsk::sum_x(bsk::where(origin, dzbr, 0.0f)));
        bsk::atomic_add((slot_curve + pool), curve_eq, held_rows);
        bsk::atomic_add(((dgrad_tissue + (b0_row * atom_count)) + atom), curve_b0);
        bsk::atomic_add(((dgrad_tissue + (diffusion_row * atom_count)) + atom), curve_damping);
        bsk::atomic_add(((dgrad_tissue + (velocity_row * atom_count)) + atom), ((flow_scale * curve_flow) + ((heading * washout_scale) * curve_washout)));
    }
}
