// The EPG kernels for the card: forward, forward-mode, adjoint and
// forward-over-reverse, complex and real, and the three-pool tables. Written
// over the tiles of _tile.hpp and included by _kernels.hpp inside ``epg``.

// Diffusion damping and its directional derivative, per state order.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _damping_jvp(const T0& rate, const T1& rate_tangent, const T2& dt, const T3& dt_tangent, const T4& order) {
    auto b_factor = (rate * dt);
    auto b_tangent = ((rate_tangent * dt) + (rate * dt_tangent));
    auto squared = (order * order);
    auto transverse_weight = ((squared + order) + 0.3333333333333333f);
    auto damp_z = bsk::exp(((-b_factor) * squared));
    auto damp_t = bsk::exp(((-b_factor) * transverse_weight));
    return bsk::make_tup(damp_z, (damp_z * ((-b_tangent) * squared)), damp_t, (damp_t * ((-b_tangent) * transverse_weight)));
}

// Two dual complex numbers added.
template <class T0, class T1>
BSK_HD auto _dual_add(const T0& x, const T1& y) {
    return bsk::make_tup((bsk::get<0>(x) + bsk::get<0>(y)), (bsk::get<1>(x) + bsk::get<1>(y)), (bsk::get<2>(x) + bsk::get<2>(y)), (bsk::get<3>(x) + bsk::get<3>(y)));
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

// ``conj(entry * spin)`` against a cotangent, entry and spin both dual.
//
// One row of a real mixing operator carried through the per-order turn, which
// is what a longitudinal cotangent walks back through.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _dual_back(const T0& entry, const T1& d_entry, const T2& spin_vr, const T3& spin_vi, const T4& spin_tr, const T5& spin_ti, const T6& br, const T7& bi, const T8& tr, const T9& ti) {
    return _dual_mul((entry * spin_vr), (-(entry * spin_vi)), ((d_entry * spin_vr) + (entry * spin_tr)), (-((d_entry * spin_vi) + (entry * spin_ti))), br, bi, tr, ti);
}

// A dual complex number's conjugate, both halves.
template <class T0>
BSK_HD auto _dual_conj(const T0& z) {
    return bsk::make_tup(bsk::get<0>(z), (-bsk::get<1>(z)), bsk::get<2>(z), (-bsk::get<3>(z)));
}

// ``exp(i * angle)`` for a real dual angle.
template <class T0, class T1>
BSK_HD auto _dual_polar(const T0& angle_value, const T1& angle_tangent) {
    auto cosine = bsk::cos(angle_value);
    auto sine = bsk::sin(angle_value);
    return bsk::make_tup(cosine, sine, ((-sine) * angle_tangent), (cosine * angle_tangent));
}

// Two dual complex numbers multiplied.
template <class T0, class T1>
BSK_HD auto _dual_product(const T0& x, const T1& y) {
    return _dual_mul(bsk::get<0>(x), bsk::get<1>(x), bsk::get<2>(x), bsk::get<3>(x), bsk::get<0>(y), bsk::get<1>(y), bsk::get<2>(y), bsk::get<3>(y));
}

// ``real_part(conj(a) * b)``, the contraction an adjoint asks for.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
BSK_HD auto _dual_real_conj_mul(const T0& a_vr, const T1& a_vi, const T2& a_tr, const T3& a_ti, const T4& b_vr, const T5& b_vi, const T6& b_tr, const T7& b_ti) {
    auto value = ((a_vr * b_vr) + (a_vi * b_vi));
    auto tangent = ((((a_tr * b_vr) + (a_ti * b_vi)) + (a_vr * b_tr)) + (a_vi * b_ti));
    return bsk::make_tup(value, tangent);
}

// A real dual number times a complex one.
template <class T0, class T1, class T2, class T3, class T4, class T5>
BSK_HD auto _dual_scale(const T0& scale_value, const T1& scale_tangent, const T2& vr, const T3& vi, const T4& tr, const T5& ti) {
    return bsk::make_tup((scale_value * vr), (scale_value * vi), ((scale_tangent * vr) + (scale_value * tr)), ((scale_tangent * vi) + (scale_value * ti)));
}

// One dual complex number less another.
template <class T0, class T1>
BSK_HD auto _dual_subtract(const T0& x, const T1& y) {
    return bsk::make_tup((bsk::get<0>(x) - bsk::get<0>(y)), (bsk::get<1>(x) - bsk::get<1>(y)), (bsk::get<2>(x) - bsk::get<2>(y)), (bsk::get<3>(x) - bsk::get<3>(y)));
}

template <class T0, class T1, class T2, class T3>
BSK_HD auto _dual_times_i(const T0& vr, const T1& vi, const T2& tr, const T3& ti) {
    return bsk::make_tup((-vi), vr, (-ti), tr);
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

// The rotation and the direction along it, with the phase applied.
//
// Shaped exactly as :func:`_profiled_pair_dual` returns, so the spinor
// operator and its adjoint read one from the other without knowing which
// they were handed. A pass that follows no direction holds the rotation
// still, and ``directed`` keeps the read for one out of the kernel.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
BSK_HD auto _dynamic_pair_dual_at(const T0& pairs, const T1& pair_direction, const T2& pair_index, const T3& event_base, const T4& event, const T5& atom, const T6& atom_count, const T7& mask, const T8& phi_value, const T9& phi_tangent, const T10& directed) {
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> | 0, 2)>> moved{};
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

// One event's entry of a buffer carrying a row per train.
//
// ``duration``, ``flip`` and ``phase`` are indexed by the train and the event
// and never by the atom, so where there is one train the address is the same
// for every lane of the program and the value can be read once rather than
// once per element of the tile.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _event_value(const T0& values, const T1& event_base, const T2& event, const T3& active_atom, const T4& single_train) {
    using Ret = bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 2)>;
    if (bsk::truth(single_train)) {
        // Spread over the program's problems, which a jitted helper has to do
        // for itself: both arms of the branch have to hand back the one shape.
        return bsk::convert<Ret>((bsk::ld((values + event)) + bsk::zeros_like(bsk::cast<float>(event_base))));
    }
    return bsk::convert<Ret>(bsk::ld(((values + event_base) + event), active_atom, 0.0f));
}

// Phase each dephasing order turns through over one interval.
//
// ``rate`` already carries the sequence's gradient geometry, so it is the
// winding per unit order per second: a longitudinal state at order l turns
// through ``l * rate * dt``. The transverse states sit half an order further
// along the gradient, which is where the extra half turn comes from. Order
// zero is left alone while longitudinal, so the recovery term is unaffected.
template <class T0, class T1, class T2>
BSK_HD auto _flow(const T0& rate, const T1& dt, const T2& order) {
    auto turn = (rate * dt);
    return bsk::make_tup(((-order) * turn), ((-(order + 0.5f)) * turn));
}

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

// Seven of the nine rotation coefficients; the rest follow by symmetry.
//
// ``t11`` repeats ``t00`` and ``t10`` is the conjugate of ``t01``, so the
// caller derives those. Feeding ``(cos, sin)`` gives the rotation itself and
// ``(sin, cos)`` rearranged gives its derivative in the flip angle, which is
// why this is one routine rather than two.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19>
BSK_HD auto _rotation_block(const T0& a_value, const T1& a_tangent, const T2& b_value, const T3& b_tangent, const T4& c_value, const T5& c_tangent, const T6& d_value, const T7& d_tangent, const T8& p1r, const T9& p1i, const T10& p1tr, const T11& p1ti, const T12& p2r, const T13& p2i, const T14& p2tr, const T15& p2ti, const T16& pcr, const T17& pci, const T18& pctr, const T19& pcti) {
    auto t00 = bsk::make_tup(a_value, (0.0f * a_value), a_tangent, (0.0f * a_tangent));
    auto t01 = _dual_scale(b_value, b_tangent, p2r, p2i, p2tr, p2ti);
    auto t02 = _dual_mul((0.0f * c_value), (-c_value), (0.0f * c_tangent), (-c_tangent), p1r, p1i, p1tr, p1ti);
    auto t12 = _dual_mul((0.0f * c_value), c_value, (0.0f * c_tangent), c_tangent, pcr, pci, pctr, pcti);
    auto t20 = _dual_mul((0.0f * c_value), (-0.5f * c_value), (0.0f * c_tangent), (-0.5f * c_tangent), pcr, pci, pctr, pcti);
    auto t21 = _dual_mul((0.0f * c_value), (0.5f * c_value), (0.0f * c_tangent), (0.5f * c_tangent), p1r, p1i, p1tr, p1ti);
    auto t22 = bsk::make_tup(d_value, (0.0f * d_value), d_tangent, (0.0f * d_tangent));
    return bsk::make_tup(t00, t01, t02, t12, t20, t21, t22);
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

// Send the cotangent on one pulse's rotation to its row.
//
// Summed over the dephasing orders first: the pair multiplies every one of
// them, so what reaches the row is the sum. The value plane is the adjoint
// and the tangent plane its own derivative, which is the split every other
// gradient here takes.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
BSK_HD auto _store_pair_cotangent(const T0& grad_value, const T1& grad_tangent, const T2& pair_index, const T3& event_base, const T4& event, const T5& atom, const T6& atom_count, const T7& turning, const T8& mask, const T9& state_mask, const T10& grad_a, const T11& grad_b) {
    auto row = bsk::cast<std::int64_t>(bsk::ld(((pair_index + event_base) + event)));
    auto entry = (((row * atom_count) + atom) * 4);
    // The block is padded to a power of two, and the orders past the last one
    // carry whatever the sweep left there -- so the sum is taken over the
    // orders that exist rather than over the block.
    auto keep = bsk::band(turning, state_mask);
    bsk::atomic_add(((grad_value + entry) + 0), bsk::sum_x(bsk::where(keep, bsk::get<0>(grad_a), 0.0f)), mask);
    bsk::atomic_add(((grad_value + entry) + 1), bsk::sum_x(bsk::where(keep, bsk::get<1>(grad_a), 0.0f)), mask);
    bsk::atomic_add(((grad_value + entry) + 2), bsk::sum_x(bsk::where(keep, bsk::get<0>(grad_b), 0.0f)), mask);
    bsk::atomic_add(((grad_value + entry) + 3), bsk::sum_x(bsk::where(keep, bsk::get<1>(grad_b), 0.0f)), mask);
    bsk::atomic_add(((grad_tangent + entry) + 0), bsk::sum_x(bsk::where(keep, bsk::get<2>(grad_a), 0.0f)), mask);
    bsk::atomic_add(((grad_tangent + entry) + 1), bsk::sum_x(bsk::where(keep, bsk::get<3>(grad_a), 0.0f)), mask);
    bsk::atomic_add(((grad_tangent + entry) + 2), bsk::sum_x(bsk::where(keep, bsk::get<2>(grad_b), 0.0f)), mask);
    bsk::atomic_add(((grad_tangent + entry) + 3), bsk::sum_x(bsk::where(keep, bsk::get<3>(grad_b), 0.0f)), mask);
}

// Which row of the stacked tables this pulse reads.
//
// Its own shape's block of ``locations`` rows, then the voxel's place along
// the slice.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _table_row(const T0& profile_index, const T1& event, const T2& location, const T3& locations) {
    return ((bsk::cast<std::int64_t>(bsk::ld((profile_index + event))) * locations) + location);
}

// The bare three-pool operator, assembled from its shared pieces.
//
// Both branches are formed and one is chosen: a ``where`` evaluates each
// side, so the divisor each of them carries is guarded whether or not it
// is the side taken.
//
// In double, and before any attenuation -- what a reverse sweep reads,
// and what :func:`_three_pool_weigh_jvp` turns into an interval's step.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24, class T25, class T26, class T27, class T28, class T29, class T30, class T31, class T32, class T33, class T34, class T35, class T36, class T37, class T38, class T39, class T40, class T41, class T42, class T43, class T44, class T45, class T46, class T47, class T48, class T49, class T50, class T51, class T52, class T53, class T54, class T55, class T56, class T57, class T58, class T59, class T60, class T61, class T62, class T63, class T64, class T65, class T66, class T67, class T68, class T69, class T70, class T71, class T72, class T73, class T74, class T75, class T76, class T77, class T78, class T79, class T80, class T81, class T82, class T83>
BSK_HD auto _three_pool_assemble_jvp(const T0& free, const T1& d_free, const T2& pool_b, const T3& d_pool_b, const T4& pool_c, const T5& d_pool_c, const T6& a00, const T7& d_a00, const T8& a01, const T9& d_a01, const T10& a02, const T11& d_a02, const T12& a10, const T13& d_a10, const T14& a11, const T15& d_a11, const T16& a20, const T17& d_a20, const T18& a22, const T19& d_a22, const T20& s00, const T21& d_s00, const T22& s11, const T23& d_s11, const T24& s22, const T25& d_s22, const T26& minors, const T27& d_minors, const T28& sum_flat, const T29& sum_linear, const T30& sum_square, const T31& d_sum_flat, const T32& d_sum_linear, const T33& d_sum_square, const T34& lift, const T35& d_lift, const T36& low, const T37& middle, const T38& d_low, const T39& d_middle, const T40& leading, const T41& d_leading, const T42& first, const T43& d_first, const T44& second, const T45& d_second, const T46& determinant, const T47& d_determinant, const T48& high, const T49& d_high, const T50& radius, const T51& d_radius, const T52& cube, const T53& raw, const T54& d_raw, const T55& argument, const T56& inside_limit, const T57& angle, const T58& d_angle, const T59& centre, const T60& d_centre, const T61& trailing, const T62& d_trailing, const T63& guarded, const T64& d_guarded, const T65& q00, const T66& d_q00, const T67& q01, const T68& d_q01, const T69& q02, const T70& d_q02, const T71& q10, const T72& d_q10, const T73& q11, const T74& d_q11, const T75& q12, const T76& d_q12, const T77& q20, const T78& d_q20, const T79& q21, const T80& d_q21, const T81& q22, const T82& d_q22, const T83& narrow) {
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_00{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_01{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_02{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_10{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_11{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_12{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_20{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_21{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> def_22{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_00{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_01{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_02{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_10{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_11{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_12{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_20{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_21{};
    bsk::tile_t<double, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83> | 0, 3)> dif_22{};
    auto c00 = (lift * ((sum_flat + (sum_linear * s00)) + (sum_square * q00)));
    auto d_c00 = ((d_lift * ((sum_flat + (sum_linear * s00)) + (sum_square * q00))) + (lift * ((((d_sum_flat + (d_sum_linear * s00)) + (sum_linear * d_s00)) + (d_sum_square * q00)) + (sum_square * d_q00))));
    auto c01 = (lift * ((sum_linear * a01) + (sum_square * q01)));
    auto d_c01 = ((d_lift * ((sum_linear * a01) + (sum_square * q01))) + (lift * ((((d_sum_linear * a01) + (sum_linear * d_a01)) + (d_sum_square * q01)) + (sum_square * d_q01))));
    auto c02 = (lift * ((sum_linear * a02) + (sum_square * q02)));
    auto d_c02 = ((d_lift * ((sum_linear * a02) + (sum_square * q02))) + (lift * ((((d_sum_linear * a02) + (sum_linear * d_a02)) + (d_sum_square * q02)) + (sum_square * d_q02))));
    auto c10 = (lift * ((sum_linear * a10) + (sum_square * q10)));
    auto d_c10 = ((d_lift * ((sum_linear * a10) + (sum_square * q10))) + (lift * ((((d_sum_linear * a10) + (sum_linear * d_a10)) + (d_sum_square * q10)) + (sum_square * d_q10))));
    auto c11 = (lift * ((sum_flat + (sum_linear * s11)) + (sum_square * q11)));
    auto d_c11 = ((d_lift * ((sum_flat + (sum_linear * s11)) + (sum_square * q11))) + (lift * ((((d_sum_flat + (d_sum_linear * s11)) + (sum_linear * d_s11)) + (d_sum_square * q11)) + (sum_square * d_q11))));
    auto c12 = (lift * (sum_square * q12));
    auto d_c12 = ((d_lift * (sum_square * q12)) + (lift * ((d_sum_square * q12) + (sum_square * d_q12))));
    auto c20 = (lift * ((sum_linear * a20) + (sum_square * q20)));
    auto d_c20 = ((d_lift * ((sum_linear * a20) + (sum_square * q20))) + (lift * ((((d_sum_linear * a20) + (sum_linear * d_a20)) + (d_sum_square * q20)) + (sum_square * d_q20))));
    auto c21 = (lift * (sum_square * q21));
    auto d_c21 = ((d_lift * (sum_square * q21)) + (lift * ((d_sum_square * q21) + (sum_square * d_q21))));
    auto c22 = (lift * ((sum_flat + (sum_linear * s22)) + (sum_square * q22)));
    auto d_c22 = ((d_lift * ((sum_flat + (sum_linear * s22)) + (sum_square * q22))) + (lift * ((((d_sum_flat + (d_sum_linear * s22)) + (sum_linear * d_s22)) + (d_sum_square * q22)) + (sum_square * d_q22))));
    // --- the Newton form's two factors, for the eigenvalue branch ---
    auto m00 = (a00 - low);
    auto d_m00 = (d_a00 - d_low);
    auto m11 = (a11 - low);
    auto d_m11 = (d_a11 - d_low);
    auto m22 = (a22 - low);
    auto d_m22 = (d_a22 - d_low);
    auto n00 = (a00 - middle);
    auto d_n00 = (d_a00 - d_middle);
    auto n11 = (a11 - middle);
    auto d_n11 = (d_a11 - d_middle);
    auto n22 = (a22 - middle);
    auto d_n22 = (d_a22 - d_middle);
    auto p00 = (((m00 * n00) + (a01 * a10)) + (a02 * a20));
    auto d_p00 = ((((((d_m00 * n00) + (m00 * d_n00)) + (d_a01 * a10)) + (a01 * d_a10)) + (d_a02 * a20)) + (a02 * d_a20));
    auto p01 = (a01 * (m00 + n11));
    auto d_p01 = ((d_a01 * (m00 + n11)) + (a01 * (d_m00 + d_n11)));
    auto p02 = (a02 * (m00 + n22));
    auto d_p02 = ((d_a02 * (m00 + n22)) + (a02 * (d_m00 + d_n22)));
    auto p10 = (a10 * (n00 + m11));
    auto d_p10 = ((d_a10 * (n00 + m11)) + (a10 * (d_n00 + d_m11)));
    auto p11 = ((a10 * a01) + (m11 * n11));
    auto d_p11 = ((((d_a10 * a01) + (a10 * d_a01)) + (d_m11 * n11)) + (m11 * d_n11));
    auto p12 = (a10 * a02);
    auto d_p12 = ((d_a10 * a02) + (a10 * d_a02));
    auto p20 = (a20 * (n00 + m22));
    auto d_p20 = ((d_a20 * (n00 + m22)) + (a20 * (d_n00 + d_m22)));
    auto p21 = (a20 * a01);
    auto d_p21 = ((d_a20 * a01) + (a20 * d_a01));
    auto p22 = ((a20 * a02) + (m22 * n22));
    auto d_p22 = ((((d_a20 * a02) + (a20 * d_a02)) + (d_m22 * n22)) + (m22 * d_n22));
    auto e00 = ((leading + (first * m00)) + (second * p00));
    auto d_e00 = ((((d_leading + (d_first * m00)) + (first * d_m00)) + (d_second * p00)) + (second * d_p00));
    auto e01 = ((first * a01) + (second * p01));
    auto d_e01 = ((((d_first * a01) + (first * d_a01)) + (d_second * p01)) + (second * d_p01));
    auto e02 = ((first * a02) + (second * p02));
    auto d_e02 = ((((d_first * a02) + (first * d_a02)) + (d_second * p02)) + (second * d_p02));
    auto e10 = ((first * a10) + (second * p10));
    auto d_e10 = ((((d_first * a10) + (first * d_a10)) + (d_second * p10)) + (second * d_p10));
    auto e11 = ((leading + (first * m11)) + (second * p11));
    auto d_e11 = ((((d_leading + (d_first * m11)) + (first * d_m11)) + (d_second * p11)) + (second * d_p11));
    auto e12 = (second * p12);
    auto d_e12 = ((d_second * p12) + (second * d_p12));
    auto e20 = ((first * a20) + (second * p20));
    auto d_e20 = ((((d_first * a20) + (first * d_a20)) + (d_second * p20)) + (second * d_p20));
    auto e21 = (second * p21);
    auto d_e21 = ((d_second * p21) + (second * d_p21));
    auto e22 = ((leading + (first * m22)) + (second * p22));
    auto d_e22 = ((((d_leading + (d_first * m22)) + (first * d_m22)) + (d_second * p22)) + (second * d_p22));
    if (bsk::truth(narrow)) {
        // The caller has bounded the spread, so the roots are unreachable and
        // everything that leads to them goes with this select.
        auto t0_ = bsk::make_tup(c00, d_c00);
        def_00 = bsk::get<0>(t0_);
        dif_00 = bsk::get<1>(t0_);
        auto t1_ = bsk::make_tup(c01, d_c01);
        def_01 = bsk::get<0>(t1_);
        dif_01 = bsk::get<1>(t1_);
        auto t2_ = bsk::make_tup(c02, d_c02);
        def_02 = bsk::get<0>(t2_);
        dif_02 = bsk::get<1>(t2_);
        auto t3_ = bsk::make_tup(c10, d_c10);
        def_10 = bsk::get<0>(t3_);
        dif_10 = bsk::get<1>(t3_);
        auto t4_ = bsk::make_tup(c11, d_c11);
        def_11 = bsk::get<0>(t4_);
        dif_11 = bsk::get<1>(t4_);
        auto t5_ = bsk::make_tup(c12, d_c12);
        def_12 = bsk::get<0>(t5_);
        dif_12 = bsk::get<1>(t5_);
        auto t6_ = bsk::make_tup(c20, d_c20);
        def_20 = bsk::get<0>(t6_);
        dif_20 = bsk::get<1>(t6_);
        auto t7_ = bsk::make_tup(c21, d_c21);
        def_21 = bsk::get<0>(t7_);
        dif_21 = bsk::get<1>(t7_);
        auto t8_ = bsk::make_tup(c22, d_c22);
        def_22 = bsk::get<0>(t8_);
        dif_22 = bsk::get<1>(t8_);
    } else {
        auto close = ((-2.0f * minors) < 1.0f);
        def_00 = bsk::where(close, c00, e00);
        dif_00 = bsk::where(close, d_c00, d_e00);
        def_01 = bsk::where(close, c01, e01);
        dif_01 = bsk::where(close, d_c01, d_e01);
        def_02 = bsk::where(close, c02, e02);
        dif_02 = bsk::where(close, d_c02, d_e02);
        def_10 = bsk::where(close, c10, e10);
        dif_10 = bsk::where(close, d_c10, d_e10);
        def_11 = bsk::where(close, c11, e11);
        dif_11 = bsk::where(close, d_c11, d_e11);
        def_12 = bsk::where(close, c12, e12);
        dif_12 = bsk::where(close, d_c12, d_e12);
        def_20 = bsk::where(close, c20, e20);
        dif_20 = bsk::where(close, d_c20, d_e20);
        def_21 = bsk::where(close, c21, e21);
        dif_21 = bsk::where(close, d_c21, d_e21);
        def_22 = bsk::where(close, c22, e22);
        dif_22 = bsk::where(close, d_c22, d_e22);
    }
    return bsk::make_tup(def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22);
}

// Read one interval's three-pool operator and a direction through it.
//
// The row carries the tissue's share of the direction; the interval's own is
// ``A1 C d_dt``, formed here because ``d_dt`` belongs to the event and the
// row is shared. Returns the nine entries and three recoveries with their
// tangents, in the order :func:`_three_pool_step_jvp` returns them.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16>
BSK_HD auto _three_pool_from_table_jvp(const T0& table, const T1& row, const T2& atom, const T3& voxel_count, const T4& mask, const T5& r1_free, const T6& r1_pool_b, const T7& r1_bound, const T8& exchange_b, const T9& exchange_c, const T10& fraction_b, const T11& d_fraction_b, const T12& fraction_c, const T13& d_fraction_c, const T14& d_dt, const T15& attenuation, const T16& d_attenuation) {
    auto free = ((1.0f - fraction_b) - fraction_c);
    auto d_free = ((-d_fraction_b) - d_fraction_c);
    auto a00 = ((((-exchange_b) * fraction_b) - (exchange_c * fraction_c)) - r1_free);
    auto a01 = (exchange_b * free);
    auto a02 = (exchange_c * free);
    auto a10 = (exchange_b * fraction_b);
    auto a11 = (((-exchange_b) * free) - r1_pool_b);
    auto a20 = (exchange_c * fraction_c);
    auto a22 = (((-exchange_c) * free) - r1_bound);
    auto base = ((table + (row * (18 * voxel_count))) + atom);
    auto c00 = bsk::ld((base + (0 * voxel_count)), mask, 0.0f);
    auto c01 = bsk::ld((base + (1 * voxel_count)), mask, 0.0f);
    auto c02 = bsk::ld((base + (2 * voxel_count)), mask, 0.0f);
    auto c10 = bsk::ld((base + (3 * voxel_count)), mask, 0.0f);
    auto c11 = bsk::ld((base + (4 * voxel_count)), mask, 0.0f);
    auto c12 = bsk::ld((base + (5 * voxel_count)), mask, 0.0f);
    auto c20 = bsk::ld((base + (6 * voxel_count)), mask, 0.0f);
    auto c21 = bsk::ld((base + (7 * voxel_count)), mask, 0.0f);
    auto c22 = bsk::ld((base + (8 * voxel_count)), mask, 0.0f);
    // The row's tangent, plus what the event's own interval direction adds.
    auto t00 = (bsk::ld((base + (9 * voxel_count)), mask, 0.0f) + (d_dt * (((a00 * c00) + (a01 * c10)) + (a02 * c20))));
    auto t01 = (bsk::ld((base + (10 * voxel_count)), mask, 0.0f) + (d_dt * (((a00 * c01) + (a01 * c11)) + (a02 * c21))));
    auto t02 = (bsk::ld((base + (11 * voxel_count)), mask, 0.0f) + (d_dt * (((a00 * c02) + (a01 * c12)) + (a02 * c22))));
    auto t10 = (bsk::ld((base + (12 * voxel_count)), mask, 0.0f) + (d_dt * ((a10 * c00) + (a11 * c10))));
    auto t11 = (bsk::ld((base + (13 * voxel_count)), mask, 0.0f) + (d_dt * ((a10 * c01) + (a11 * c11))));
    auto t12 = (bsk::ld((base + (14 * voxel_count)), mask, 0.0f) + (d_dt * ((a10 * c02) + (a11 * c12))));
    auto t20 = (bsk::ld((base + (15 * voxel_count)), mask, 0.0f) + (d_dt * ((a20 * c00) + (a22 * c20))));
    auto t21 = (bsk::ld((base + (16 * voxel_count)), mask, 0.0f) + (d_dt * ((a20 * c01) + (a22 * c21))));
    auto t22 = (bsk::ld((base + (17 * voxel_count)), mask, 0.0f) + (d_dt * ((a20 * c02) + (a22 * c22))));
    auto e00 = (attenuation * c00);
    auto e01 = (attenuation * c01);
    auto e02 = (attenuation * c02);
    auto e10 = (attenuation * c10);
    auto e11 = (attenuation * c11);
    auto e12 = (attenuation * c12);
    auto e20 = (attenuation * c20);
    auto e21 = (attenuation * c21);
    auto e22 = (attenuation * c22);
    auto f00 = ((d_attenuation * c00) + (attenuation * t00));
    auto f01 = ((d_attenuation * c01) + (attenuation * t01));
    auto f02 = ((d_attenuation * c02) + (attenuation * t02));
    auto f10 = ((d_attenuation * c10) + (attenuation * t10));
    auto f11 = ((d_attenuation * c11) + (attenuation * t11));
    auto f12 = ((d_attenuation * c12) + (attenuation * t12));
    auto f20 = ((d_attenuation * c20) + (attenuation * t20));
    auto f21 = ((d_attenuation * c21) + (attenuation * t21));
    auto f22 = ((d_attenuation * c22) + (attenuation * t22));
    // The equilibrium the recoveries are taken against moves with the
    // fractions, so it carries a direction of its own.
    auto grow_free = (free - (((e00 * free) + (e01 * fraction_b)) + (e02 * fraction_c)));
    auto grow_pool_b = (fraction_b - (((e10 * free) + (e11 * fraction_b)) + (e12 * fraction_c)));
    auto grow_bound = (fraction_c - (((e20 * free) + (e21 * fraction_b)) + (e22 * fraction_c)));
    auto d_grow_free = (d_free - ((((((f00 * free) + (f01 * fraction_b)) + (f02 * fraction_c)) + (e00 * d_free)) + (e01 * d_fraction_b)) + (e02 * d_fraction_c)));
    auto d_grow_pool_b = (d_fraction_b - ((((((f10 * free) + (f11 * fraction_b)) + (f12 * fraction_c)) + (e10 * d_free)) + (e11 * d_fraction_b)) + (e12 * d_fraction_c)));
    auto d_grow_bound = (d_fraction_c - ((((((f20 * free) + (f21 * fraction_b)) + (f22 * fraction_c)) + (e20 * d_free)) + (e21 * d_fraction_b)) + (e22 * d_fraction_c)));
    return bsk::make_tup(e00, e01, e02, e10, e11, e12, e20, e21, e22, grow_free, grow_pool_b, grow_bound, f00, f01, f02, f10, f11, f12, f20, f21, f22, d_grow_free, d_grow_pool_b, d_grow_bound);
}

// Nine cotangents against nine entries, less what the recoveries take.
//
// A recovery is ``m - E m`` with ``m`` the equilibrium
// ``(free, fraction_b, fraction_c)``, so it differentiates through the same
// nine entries with the equilibrium contracted out of them.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23>
BSK_HD auto _three_pool_contract(const T0& x00, const T1& x01, const T2& x02, const T3& x10, const T4& x11, const T5& x12, const T6& x20, const T7& x21, const T8& x22, const T9& e11, const T10& e12, const T11& e13, const T12& e21, const T13& e22, const T14& e23, const T15& e31, const T16& e32, const T17& e33, const T18& rec_free, const T19& rec_pool_b, const T20& rec_bound, const T21& free, const T22& fraction_b, const T23& fraction_c) {
    return ((((((((((((e11 * x00) + (e12 * x01)) + (e13 * x02)) + (e21 * x10)) + (e22 * x11)) + (e23 * x12)) + (e31 * x20)) + (e32 * x21)) + (e33 * x22)) - (rec_free * (((x00 * free) + (x01 * fraction_b)) + (x02 * fraction_c)))) - (rec_pool_b * (((x10 * free) + (x11 * fraction_b)) + (x12 * fraction_c)))) - (rec_bound * (((x20 * free) + (x21 * fraction_b)) + (x22 * fraction_c))));
}

// The interval and the attenuation, from a dual pair's cotangents.
//
// ``dE/d(dt)`` is ``A1 E``, and the direction that quantity carries follows
// from the same generator: with ``C_dot == C_row + A1 C d_dt``, the
// derivative in the interval is ``A1_dot C + A1 C_row + A1 A1 C d_dt``. So
// three products of the generator against the tabulated row serve what the
// eigenvalues would otherwise be re-formed for, and these two quantities are
// the only ones that stay per event.
//
// Returns the interval and attenuation gradients, value then tangent, in the
// order :func:`_three_pool_step_adjoint_jvp` returns them.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24, class T25, class T26, class T27, class T28, class T29, class T30, class T31, class T32, class T33, class T34, class T35, class T36, class T37, class T38, class T39, class T40, class T41, class T42, class T43, class T44, class T45>
BSK_HD auto _three_pool_interval_adjoint_jvp(const T0& table, const T1& row, const T2& atom, const T3& voxel_count, const T4& mask, const T5& r1_free, const T6& d_r1_free, const T7& r1_pool_b, const T8& d_r1_pool_b, const T9& r1_bound, const T10& d_r1_bound, const T11& exchange_b, const T12& d_exchange_b, const T13& exchange_c, const T14& d_exchange_c, const T15& fraction_b, const T16& d_fraction_b, const T17& fraction_c, const T18& d_fraction_c, const T19& d_dt, const T20& attenuation, const T21& d_attenuation, const T22& b11, const T23& b12, const T24& b13, const T25& b21, const T26& b22, const T27& b23, const T28& b31, const T29& b32, const T30& b33, const T31& bfree, const T32& bpool_b, const T33& bbound, const T34& t11, const T35& t12, const T36& t13, const T37& t21, const T38& t22, const T39& t23, const T40& t31, const T41& t32, const T42& t33, const T43& tfree, const T44& tpool_b, const T45& tbound) {
    auto free = ((1.0f - fraction_b) - fraction_c);
    auto d_free = ((-d_fraction_b) - d_fraction_c);
    auto a00 = ((((-exchange_b) * fraction_b) - (exchange_c * fraction_c)) - r1_free);
    auto a01 = (exchange_b * free);
    auto a02 = (exchange_c * free);
    auto a10 = (exchange_b * fraction_b);
    auto a11 = (((-exchange_b) * free) - r1_pool_b);
    auto a20 = (exchange_c * fraction_c);
    auto a22 = (((-exchange_c) * free) - r1_bound);
    auto da00 = ((((((-d_exchange_b) * fraction_b) - (exchange_b * d_fraction_b)) - (d_exchange_c * fraction_c)) - (exchange_c * d_fraction_c)) - d_r1_free);
    auto da01 = ((d_exchange_b * free) + (exchange_b * d_free));
    auto da02 = ((d_exchange_c * free) + (exchange_c * d_free));
    auto da10 = ((d_exchange_b * fraction_b) + (exchange_b * d_fraction_b));
    auto da11 = ((((-d_exchange_b) * free) - (exchange_b * d_free)) - d_r1_pool_b);
    auto da20 = ((d_exchange_c * fraction_c) + (exchange_c * d_fraction_c));
    auto da22 = ((((-d_exchange_c) * free) - (exchange_c * d_free)) - d_r1_bound);
    auto base = ((table + (row * (18 * voxel_count))) + atom);
    auto c00 = bsk::ld((base + (0 * voxel_count)), mask, 0.0f);
    auto c01 = bsk::ld((base + (1 * voxel_count)), mask, 0.0f);
    auto c02 = bsk::ld((base + (2 * voxel_count)), mask, 0.0f);
    auto c10 = bsk::ld((base + (3 * voxel_count)), mask, 0.0f);
    auto c11 = bsk::ld((base + (4 * voxel_count)), mask, 0.0f);
    auto c12 = bsk::ld((base + (5 * voxel_count)), mask, 0.0f);
    auto c20 = bsk::ld((base + (6 * voxel_count)), mask, 0.0f);
    auto c21 = bsk::ld((base + (7 * voxel_count)), mask, 0.0f);
    auto c22 = bsk::ld((base + (8 * voxel_count)), mask, 0.0f);
    auto r00 = bsk::ld((base + (9 * voxel_count)), mask, 0.0f);
    auto r01 = bsk::ld((base + (10 * voxel_count)), mask, 0.0f);
    auto r02 = bsk::ld((base + (11 * voxel_count)), mask, 0.0f);
    auto r10 = bsk::ld((base + (12 * voxel_count)), mask, 0.0f);
    auto r11 = bsk::ld((base + (13 * voxel_count)), mask, 0.0f);
    auto r12 = bsk::ld((base + (14 * voxel_count)), mask, 0.0f);
    auto r20 = bsk::ld((base + (15 * voxel_count)), mask, 0.0f);
    auto r21 = bsk::ld((base + (16 * voxel_count)), mask, 0.0f);
    auto r22 = bsk::ld((base + (17 * voxel_count)), mask, 0.0f);
    // P = A1 C, Q = A1_dot C + A1 C_row, S = A1 P.
    auto p00 = (((a00 * c00) + (a01 * c10)) + (a02 * c20));
    auto p01 = (((a00 * c01) + (a01 * c11)) + (a02 * c21));
    auto p02 = (((a00 * c02) + (a01 * c12)) + (a02 * c22));
    auto p10 = ((a10 * c00) + (a11 * c10));
    auto p11 = ((a10 * c01) + (a11 * c11));
    auto p12 = ((a10 * c02) + (a11 * c12));
    auto p20 = ((a20 * c00) + (a22 * c20));
    auto p21 = ((a20 * c01) + (a22 * c21));
    auto p22 = ((a20 * c02) + (a22 * c22));
    auto q00 = ((((((da00 * c00) + (da01 * c10)) + (da02 * c20)) + (a00 * r00)) + (a01 * r10)) + (a02 * r20));
    auto q01 = ((((((da00 * c01) + (da01 * c11)) + (da02 * c21)) + (a00 * r01)) + (a01 * r11)) + (a02 * r21));
    auto q02 = ((((((da00 * c02) + (da01 * c12)) + (da02 * c22)) + (a00 * r02)) + (a01 * r12)) + (a02 * r22));
    auto q10 = ((((da10 * c00) + (da11 * c10)) + (a10 * r00)) + (a11 * r10));
    auto q11 = ((((da10 * c01) + (da11 * c11)) + (a10 * r01)) + (a11 * r11));
    auto q12 = ((((da10 * c02) + (da11 * c12)) + (a10 * r02)) + (a11 * r12));
    auto q20 = ((((da20 * c00) + (da22 * c20)) + (a20 * r00)) + (a22 * r20));
    auto q21 = ((((da20 * c01) + (da22 * c21)) + (a20 * r01)) + (a22 * r21));
    auto q22 = ((((da20 * c02) + (da22 * c22)) + (a20 * r02)) + (a22 * r22));
    auto s00 = (((a00 * p00) + (a01 * p10)) + (a02 * p20));
    auto s01 = (((a00 * p01) + (a01 * p11)) + (a02 * p21));
    auto s02 = (((a00 * p02) + (a01 * p12)) + (a02 * p22));
    auto s10 = ((a10 * p00) + (a11 * p10));
    auto s11 = ((a10 * p01) + (a11 * p11));
    auto s12 = ((a10 * p02) + (a11 * p12));
    auto s20 = ((a20 * p00) + (a22 * p20));
    auto s21 = ((a20 * p01) + (a22 * p21));
    auto s22 = ((a20 * p02) + (a22 * p22));
    // The direction the tabulated operator carries, and the interval's own
    // share of it.
    auto d00 = (r00 + (p00 * d_dt));
    auto d01 = (r01 + (p01 * d_dt));
    auto d02 = (r02 + (p02 * d_dt));
    auto d10 = (r10 + (p10 * d_dt));
    auto d11 = (r11 + (p11 * d_dt));
    auto d12 = (r12 + (p12 * d_dt));
    auto d20 = (r20 + (p20 * d_dt));
    auto d21 = (r21 + (p21 * d_dt));
    auto d22 = (r22 + (p22 * d_dt));
    // dE/d(dt) with the attenuation held, and the direction that carries.
    auto g00 = (attenuation * p00);
    auto g01 = (attenuation * p01);
    auto g02 = (attenuation * p02);
    auto g10 = (attenuation * p10);
    auto g11 = (attenuation * p11);
    auto g12 = (attenuation * p12);
    auto g20 = (attenuation * p20);
    auto g21 = (attenuation * p21);
    auto g22 = (attenuation * p22);
    auto w00 = ((d_attenuation * p00) + (attenuation * (q00 + (s00 * d_dt))));
    auto w01 = ((d_attenuation * p01) + (attenuation * (q01 + (s01 * d_dt))));
    auto w02 = ((d_attenuation * p02) + (attenuation * (q02 + (s02 * d_dt))));
    auto w10 = ((d_attenuation * p10) + (attenuation * (q10 + (s10 * d_dt))));
    auto w11 = ((d_attenuation * p11) + (attenuation * (q11 + (s11 * d_dt))));
    auto w12 = ((d_attenuation * p12) + (attenuation * (q12 + (s12 * d_dt))));
    auto w20 = ((d_attenuation * p20) + (attenuation * (q20 + (s20 * d_dt))));
    auto w21 = ((d_attenuation * p21) + (attenuation * (q21 + (s21 * d_dt))));
    auto w22 = ((d_attenuation * p22) + (attenuation * (q22 + (s22 * d_dt))));
    // This kernel carries every quantity as a dual pair, so the tangent
    // returned beside a gradient is that gradient's own directional
    // derivative -- not the gradient with respect to the direction.
    auto grad_dt_v = _three_pool_contract(g00, g01, g02, g10, g11, g12, g20, g21, g22, b11, b12, b13, b21, b22, b23, b31, b32, b33, bfree, bpool_b, bbound, free, fraction_b, fraction_c);
    auto grad_dt_t = ((_three_pool_contract(g00, g01, g02, g10, g11, g12, g20, g21, g22, t11, t12, t13, t21, t22, t23, t31, t32, t33, tfree, tpool_b, tbound, free, fraction_b, fraction_c) + _three_pool_contract(w00, w01, w02, w10, w11, w12, w20, w21, w22, b11, b12, b13, b21, b22, b23, b31, b32, b33, bfree, bpool_b, bbound, free, fraction_b, fraction_c)) - (((bfree * (((g00 * d_free) + (g01 * d_fraction_b)) + (g02 * d_fraction_c))) + (bpool_b * (((g10 * d_free) + (g11 * d_fraction_b)) + (g12 * d_fraction_c)))) + (bbound * (((g20 * d_free) + (g21 * d_fraction_b)) + (g22 * d_fraction_c)))));
    auto grad_att_v = _three_pool_contract(c00, c01, c02, c10, c11, c12, c20, c21, c22, b11, b12, b13, b21, b22, b23, b31, b32, b33, bfree, bpool_b, bbound, free, fraction_b, fraction_c);
    auto grad_att_t = ((_three_pool_contract(c00, c01, c02, c10, c11, c12, c20, c21, c22, t11, t12, t13, t21, t22, t23, t31, t32, t33, tfree, tpool_b, tbound, free, fraction_b, fraction_c) + _three_pool_contract(d00, d01, d02, d10, d11, d12, d20, d21, d22, b11, b12, b13, b21, b22, b23, b31, b32, b33, bfree, bpool_b, bbound, free, fraction_b, fraction_c)) - (((bfree * (((c00 * d_free) + (c01 * d_fraction_b)) + (c02 * d_fraction_c))) + (bpool_b * (((c10 * d_free) + (c11 * d_fraction_b)) + (c12 * d_fraction_c)))) + (bbound * (((c20 * d_free) + (c21 * d_fraction_b)) + (c22 * d_fraction_c)))));
    return bsk::make_tup(grad_dt_v, grad_att_v, grad_dt_t, grad_att_t);
}

// ``[a, b] exp`` and its directional derivative.
//
// The derivative of a divided difference is the next one along,
// ``d/da [a,b] = [a,a,b]``, which near the coalescence is again a series in
// the gap's square rather than a quotient that vanishes over a vanishing
// denominator.
template <class T0, class T1, class T2, class T3, class T4, class T5>
BSK_HD auto _exp_difference_jvp(const T0& lower, const T1& d_lower, const T2& upper, const T3& d_upper, const T4& low_exp, const T5& high_exp) {
    auto half = (0.5f * (upper - lower));
    auto d_half = (0.5f * (d_upper - d_lower));
    auto near = (bsk::abs(half) < 0.0001f);
    auto square = (half * half);
    auto d_square = ((2.0f * half) * d_half);
    // exp(mid) * sinh(half)/half, both factors expanded about zero.
    auto lift = (low_exp * ((1.0f + half) + (0.5f * square)));
    auto d_lift = (low_exp * (((d_lower * ((1.0f + half) + (0.5f * square))) + d_half) + (half * d_half)));
    auto sinch = ((1.0f + bsk::truediv(square, 6.0f)) + bsk::truediv((square * square), 120.0f));
    auto d_sinch = (bsk::truediv(d_square, 6.0f) + bsk::truediv(((2.0f * square) * d_square), 120.0f));
    auto series = (lift * sinch);
    auto d_series = ((d_lift * sinch) + (lift * d_sinch));
    auto gap = bsk::where(near, 1.0f, (upper - lower));
    auto d_gap = bsk::where(near, 0.0f, (d_upper - d_lower));
    auto quotient = bsk::truediv((high_exp - low_exp), gap);
    auto d_quotient = bsk::truediv((((high_exp * d_upper) - (low_exp * d_lower)) - (quotient * d_gap)), gap);
    return bsk::make_tup(bsk::where(near, series, quotient), bsk::where(near, d_series, d_quotient));
}

// The three-pool operator's shared front half, as duals.
//
// The generator, its two invariants, the series coefficients and the three
// roots with the divided differences between them -- everything both the
// operator and its reverse sweep are assembled from, computed once in double
// so the two cannot drift apart.
template <class Work, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16>
BSK_HD auto _three_pool_pieces_jvp_in_precision(const T0& r1_free, const T1& d_r1_free, const T2& r1_pool_b, const T3& d_r1_pool_b, const T4& r1_bound, const T5& d_r1_bound, const T6& exchange_b, const T7& d_exchange_b, const T8& exchange_c, const T9& d_exchange_c, const T10& fraction_b, const T11& d_fraction_b, const T12& fraction_c, const T13& d_fraction_c, const T14& dt, const T15& d_dt, const T16& narrow) {
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> d_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> d_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> d_square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> d_sum_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> d_sum_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> d_sum_square{};
    float factorial{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> sum_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> sum_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16> | 0, 3)> sum_square{};
    auto step = bsk::cast<Work>(dt);
    auto d_step = bsk::cast<Work>(d_dt);
    auto free = bsk::cast<Work>(((Work(1.0) - fraction_b) - fraction_c));
    auto d_free = bsk::cast<Work>(((-d_fraction_b) - d_fraction_c));
    auto pool_b = bsk::cast<Work>(fraction_b);
    auto d_pool_b = bsk::cast<Work>(d_fraction_b);
    auto pool_c = bsk::cast<Work>(fraction_c);
    auto d_pool_c = bsk::cast<Work>(d_fraction_c);
    auto rate_b = bsk::cast<Work>(exchange_b);
    auto d_rate_b = bsk::cast<Work>(d_exchange_b);
    auto rate_c = bsk::cast<Work>(exchange_c);
    auto d_rate_c = bsk::cast<Work>(d_exchange_c);
    auto kab = (rate_b * pool_b);
    auto d_kab = ((d_rate_b * pool_b) + (rate_b * d_pool_b));
    auto kba = (rate_b * free);
    auto d_kba = ((d_rate_b * free) + (rate_b * d_free));
    auto kac = (rate_c * pool_c);
    auto d_kac = ((d_rate_c * pool_c) + (rate_c * d_pool_c));
    auto kca = (rate_c * free);
    auto d_kca = ((d_rate_c * free) + (rate_c * d_free));
    auto row_a = (((-kab) - kac) - bsk::cast<Work>(r1_free));
    auto d_row_a = (((-d_kab) - d_kac) - bsk::cast<Work>(d_r1_free));
    auto row_b = ((-kba) - bsk::cast<Work>(r1_pool_b));
    auto d_row_b = ((-d_kba) - bsk::cast<Work>(d_r1_pool_b));
    auto row_c = ((-kca) - bsk::cast<Work>(r1_bound));
    auto d_row_c = ((-d_kca) - bsk::cast<Work>(d_r1_bound));
    auto a00 = (row_a * step);
    auto d_a00 = ((d_row_a * step) + (row_a * d_step));
    auto a01 = (kba * step);
    auto d_a01 = ((d_kba * step) + (kba * d_step));
    auto a02 = (kca * step);
    auto d_a02 = ((d_kca * step) + (kca * d_step));
    auto a10 = (kab * step);
    auto d_a10 = ((d_kab * step) + (kab * d_step));
    auto a11 = (row_b * step);
    auto d_a11 = ((d_row_b * step) + (row_b * d_step));
    auto a20 = (kac * step);
    auto d_a20 = ((d_kac * step) + (kac * d_step));
    auto a22 = (row_c * step);
    auto d_a22 = ((d_row_c * step) + (row_c * d_step));
    auto third = bsk::truediv(((a00 + a11) + a22), Work(3.0));
    auto d_third = bsk::truediv(((d_a00 + d_a11) + d_a22), Work(3.0));
    auto s00 = (a00 - third);
    auto d_s00 = (d_a00 - d_third);
    auto s11 = (a11 - third);
    auto d_s11 = (d_a11 - d_third);
    auto s22 = (a22 - third);
    auto d_s22 = (d_a22 - d_third);
    auto minors = (((((s00 * s11) - (a01 * a10)) + (s00 * s22)) - (a02 * a20)) + (s11 * s22));
    auto d_minors = ((((((((((d_s00 * s11) + (s00 * d_s11)) - (d_a01 * a10)) - (a01 * d_a10)) + (d_s00 * s22)) + (s00 * d_s22)) - (d_a02 * a20)) - (a02 * d_a20)) + (d_s11 * s22)) + (s11 * d_s22));
    auto determinant = ((((s00 * s11) * s22) - (a01 * (a10 * s22))) + (a02 * ((-s11) * a20)));
    auto d_determinant = ((((((((((d_s00 * s11) * s22) + ((s00 * d_s11) * s22)) + ((s00 * s11) * d_s22)) - ((d_a01 * a10) * s22)) - ((a01 * d_a10) * s22)) - ((a01 * a10) * d_s22)) - ((d_a02 * s11) * a20)) - ((a02 * d_s11) * a20)) - ((a02 * s11) * d_a20));
    // --- close together: the series reduced modulo x^3 + minors x - det ---
    flat = (Work(1.0) + (Work(0.0) * third));
    linear = (Work(0.0) * third);
    square = (Work(0.0) * third);
    d_flat = (Work(0.0) * third);
    d_linear = (Work(0.0) * third);
    d_square = (Work(0.0) * third);
    sum_flat = flat;
    sum_linear = linear;
    sum_square = square;
    d_sum_flat = d_flat;
    d_sum_linear = d_linear;
    d_sum_square = d_square;
    factorial = Work(1.0);
    #pragma unroll
    for (bsk::index_t order = 1; order < 16; order += 1) {
        auto next_flat = (square * determinant);
        auto d_next_flat = ((d_square * determinant) + (square * d_determinant));
        auto next_linear = (flat - (square * minors));
        auto d_next_linear = ((d_flat - (d_square * minors)) - (square * d_minors));
        auto next_square = linear;
        auto d_next_square = d_linear;
        flat = next_flat;
        linear = next_linear;
        square = next_square;
        d_flat = d_next_flat;
        d_linear = d_next_linear;
        d_square = d_next_square;
        factorial = (factorial * order);
        auto weight = bsk::truediv(Work(1.0), factorial);
        sum_flat = (sum_flat + (weight * flat));
        sum_linear = (sum_linear + (weight * linear));
        sum_square = (sum_square + (weight * square));
        d_sum_flat = (d_sum_flat + (weight * d_flat));
        d_sum_linear = (d_sum_linear + (weight * d_linear));
        d_sum_square = (d_sum_square + (weight * d_square));
    }
    auto lift = bsk::exp(third);
    auto d_lift = (lift * d_third);
    // --- far apart: the Newton form at the three roots ---
    auto inside = ((-minors) * Work(0.3333333333333333));
    auto d_inside = ((-d_minors) * Work(0.3333333333333333));
    auto radius = bsk::sqrt(bsk::maximum(inside, Work(1e-300)));
    auto d_radius = bsk::where((inside > Work(0.0)), bsk::truediv((Work(0.5) * d_inside), radius), Work(0.0));
    auto cube = ((radius * radius) * radius);
    auto raw = bsk::truediv((Work(0.5) * determinant), cube);
    auto d_raw = bsk::truediv(((Work(0.5) * d_determinant) - ((((raw * Work(3.0)) * radius) * radius) * d_radius)), cube);
    auto inside_limit = bsk::band((raw > Work(-0.9999999999999999)), (raw < Work(0.9999999999999999)));
    auto argument = bsk::minimum(bsk::maximum(raw, Work(-0.9999999999999999)), Work(0.9999999999999999));
    auto d_argument = bsk::where(inside_limit, d_raw, Work(0.0));
    auto angle = bsk::truediv(bsk::acos(argument), Work(3.0));
    auto d_angle = bsk::truediv((-d_argument), (Work(3.0) * bsk::sqrt(bsk::maximum((Work(1.0) - (argument * argument)), Work(1e-300)))));
    auto root_a = (((Work(2.0) * radius) * bsk::cos(angle)) + third);
    auto d_root_a = ((((Work(2.0) * d_radius) * bsk::cos(angle)) - (((Work(2.0) * radius) * bsk::sin(angle)) * d_angle)) + d_third);
    auto root_b = (((Work(2.0) * radius) * bsk::cos((angle - Work(2.0943951023931957)))) + third);
    auto d_root_b = ((((Work(2.0) * d_radius) * bsk::cos((angle - Work(2.0943951023931957)))) - (((Work(2.0) * radius) * bsk::sin((angle - Work(2.0943951023931957)))) * d_angle)) + d_third);
    auto root_c = (((Work(2.0) * radius) * bsk::cos((angle - Work(4.188790204786391)))) + third);
    auto d_root_c = ((((Work(2.0) * d_radius) * bsk::cos((angle - Work(4.188790204786391)))) - (((Work(2.0) * radius) * bsk::sin((angle - Work(4.188790204786391)))) * d_angle)) + d_third);
    // Sorting is a permutation, so the tangents follow their own values.
    auto low = bsk::minimum(bsk::minimum(root_a, root_b), root_c);
    auto high = bsk::maximum(bsk::maximum(root_a, root_b), root_c);
    auto middle = bsk::maximum(bsk::minimum(root_a, root_b), bsk::minimum(bsk::maximum(root_a, root_b), root_c));
    auto d_low = bsk::where((root_a == low), d_root_a, bsk::where((root_b == low), d_root_b, d_root_c));
    auto d_high = bsk::where((root_a == high), d_root_a, bsk::where((root_b == high), d_root_b, d_root_c));
    auto d_middle = bsk::where((root_a == middle), d_root_a, bsk::where((root_b == middle), d_root_b, d_root_c));
    auto leading = bsk::exp(low);
    auto d_leading = (leading * d_low);
    auto centre = bsk::exp(middle);
    auto d_centre = (centre * d_middle);
    auto trailing = bsk::exp(high);
    auto d_trailing = (trailing * d_high);
    auto t0_ = _exp_difference_jvp(low, d_low, middle, d_middle, leading, centre);
    auto first = bsk::get<0>(t0_);
    auto d_first = bsk::get<1>(t0_);
    auto t1_ = _exp_difference_jvp(middle, d_middle, high, d_high, centre, trailing);
    auto upper = bsk::get<0>(t1_);
    auto d_upper = bsk::get<1>(t1_);
    auto span = (high - low);
    auto d_span = (d_high - d_low);
    auto guarded = bsk::where((span > Work(0.0)), span, Work(1.0));
    auto d_guarded = bsk::where((span > Work(0.0)), d_span, Work(0.0));
    auto second = bsk::truediv((upper - first), guarded);
    auto d_second = bsk::truediv(((d_upper - d_first) - (second * d_guarded)), guarded);
    // --- the shifted generator squared, for the series branch ---
    auto q00 = (((s00 * s00) + (a01 * a10)) + (a02 * a20));
    auto d_q00 = ((((((Work(2.0) * s00) * d_s00) + (d_a01 * a10)) + (a01 * d_a10)) + (d_a02 * a20)) + (a02 * d_a20));
    auto q01 = (a01 * (s00 + s11));
    auto d_q01 = ((d_a01 * (s00 + s11)) + (a01 * (d_s00 + d_s11)));
    auto q02 = (a02 * (s00 + s22));
    auto d_q02 = ((d_a02 * (s00 + s22)) + (a02 * (d_s00 + d_s22)));
    auto q10 = (a10 * (s00 + s11));
    auto d_q10 = ((d_a10 * (s00 + s11)) + (a10 * (d_s00 + d_s11)));
    auto q11 = ((a10 * a01) + (s11 * s11));
    auto d_q11 = (((d_a10 * a01) + (a10 * d_a01)) + ((Work(2.0) * s11) * d_s11));
    auto q12 = (a10 * a02);
    auto d_q12 = ((d_a10 * a02) + (a10 * d_a02));
    auto q20 = (a20 * (s00 + s22));
    auto d_q20 = ((d_a20 * (s00 + s22)) + (a20 * (d_s00 + d_s22)));
    auto q21 = (a20 * a01);
    auto d_q21 = ((d_a20 * a01) + (a20 * d_a01));
    auto q22 = ((a20 * a02) + (s22 * s22));
    auto d_q22 = (((d_a20 * a02) + (a20 * d_a02)) + ((Work(2.0) * s22) * d_s22));
    return bsk::make_tup(free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, a00, d_a00, a01, d_a01, a02, d_a02, a10, d_a10, a11, d_a11, a20, d_a20, a22, d_a22, s00, d_s00, s11, d_s11, s22, d_s22, minors, d_minors, sum_flat, sum_linear, sum_square, d_sum_flat, d_sum_linear, d_sum_square, lift, d_lift, low, middle, d_low, d_middle, leading, d_leading, first, d_first, second, d_second, determinant, d_determinant, high, d_high, radius, d_radius, cube, raw, d_raw, argument, inside_limit, angle, d_angle, centre, d_centre, trailing, d_trailing, guarded, d_guarded, q00, d_q00, q01, d_q01, q02, d_q02, q10, d_q10, q11, d_q11, q12, d_q12, q20, d_q20, q21, d_q21, q22, d_q22);
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16>
BSK_HD auto _three_pool_pieces_jvp(const T0& r1_free, const T1& d_r1_free, const T2& r1_pool_b, const T3& d_r1_pool_b, const T4& r1_bound, const T5& d_r1_bound, const T6& exchange_b, const T7& d_exchange_b, const T8& exchange_c, const T9& d_exchange_c, const T10& fraction_b, const T11& d_fraction_b, const T12& fraction_c, const T13& d_fraction_c, const T14& dt, const T15& d_dt, const T16& narrow) {
    using R = decltype(_three_pool_pieces_jvp_in_precision<double>(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, narrow));
    if (bsk::truth(narrow)) {
        return bsk::convert<R>(_three_pool_pieces_jvp_in_precision<float>(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, narrow));
    }
    return _three_pool_pieces_jvp_in_precision<double>(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, narrow);
}

// The reverse of :func:`_exp_difference`, onto both points, on a direction.
//
// Near the coalescence the slope comes from the same series the value does,
// because the difference quotient's own derivative is a cancellation divided
// by a small number twice over.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _exp_difference_adjoint_jvp(const T0& lower, const T1& d_lower, const T2& upper, const T3& d_upper, const T4& exp_lower, const T5& d_exp_lower, const T6& exp_upper, const T7& d_exp_upper, const T8& seed, const T9& d_seed) {
    auto half = (0.5f * (upper - lower));
    auto d_half = (0.5f * (d_upper - d_lower));
    auto near = (bsk::abs(half) < 0.0001f);
    auto poly = ((1.0f + half) + ((0.5f * half) * half));
    auto d_poly = (d_half + (half * d_half));
    auto even = (1.0f + bsk::truediv((half * half), 6.0f));
    auto d_even = bsk::truediv((half * d_half), 3.0f);
    auto slope = (((1.0f + half) * even) + ((poly * half) * 0.3333333333333333f));
    auto d_slope = (((d_half * even) + ((1.0f + half) * d_even)) + (((d_poly * half) + (poly * d_half)) * 0.3333333333333333f));
    auto series = ((exp_lower * poly) * even);
    auto d_series = (((d_exp_lower * poly) * even) + (exp_lower * ((d_poly * even) + (poly * d_even))));
    auto swing = ((0.5f * exp_lower) * slope);
    auto d_swing = (0.5f * ((d_exp_lower * slope) + (exp_lower * d_slope)));
    auto gap = bsk::where(near, 1.0f, (upper - lower));
    auto d_gap = bsk::where(near, 0.0f, (d_upper - d_lower));
    auto value = bsk::truediv((exp_upper - exp_lower), gap);
    auto d_value = bsk::truediv(((d_exp_upper - d_exp_lower) - (value * d_gap)), gap);
    auto far_lower = bsk::truediv((value - exp_lower), gap);
    auto d_far_lower = bsk::truediv(((d_value - d_exp_lower) - (far_lower * d_gap)), gap);
    auto far_upper = bsk::truediv((exp_upper - value), gap);
    auto d_far_upper = bsk::truediv(((d_exp_upper - d_value) - (far_upper * d_gap)), gap);
    auto to_lower = bsk::where(near, (series - swing), far_lower);
    auto d_to_lower = bsk::where(near, (d_series - d_swing), d_far_lower);
    auto to_upper = bsk::where(near, swing, far_upper);
    auto d_to_upper = bsk::where(near, d_swing, d_far_upper);
    return bsk::make_tup((seed * to_lower), ((d_seed * to_lower) + (seed * d_to_lower)), (seed * to_upper), ((d_seed * to_upper) + (seed * d_to_upper)));
}

// The reverse sweep of :func:`_three_pool_step`, carried on a direction.
//
// Reads the pieces and the bare operator the replay already formed, so an
// interval's transcendentals are taken once for the pass rather than once
// for each direction through it, and in the same double.
//
// Both branches are swept, each by the algebra its own forward used, and the
// choice between them is made on the cotangents rather than on the way in --
// a ``where`` evaluates both sides, so each side's divisors are guarded.
//
// The series branch is a polynomial in the two invariants alone, so its
// reverse is reached by carrying the recurrence's sensitivity to those two
// forward beside it, which needs no history of the sixteen terms.
//
// Returned as the gradients w.r.t. ``(r1_free, r1_pool_b, r1_bound,
// exchange_b, exchange_c, fraction_b, fraction_c, dt, attenuation)`` and
// then their nine tangents.
template <class Work, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24, class T25, class T26, class T27, class T28, class T29, class T30, class T31, class T32, class T33, class T34, class T35, class T36, class T37, class T38, class T39, class T40, class T41, class T42, class T43, class T44, class T45, class T46, class T47, class T48, class T49, class T50, class T51, class T52, class T53, class T54, class T55, class T56, class T57, class T58, class T59, class T60, class T61, class T62, class T63, class T64, class T65, class T66, class T67, class T68, class T69, class T70, class T71, class T72, class T73, class T74, class T75, class T76, class T77, class T78, class T79, class T80, class T81, class T82, class T83, class T84, class T85, class T86, class T87, class T88, class T89, class T90, class T91, class T92, class T93, class T94, class T95, class T96, class T97, class T98, class T99, class T100, class T101, class T102, class T103, class T104, class T105, class T106, class T107, class T108, class T109, class T110, class T111, class T112, class T113, class T114, class T115, class T116, class T117, class T118, class T119, class T120, class T121, class T122, class T123, class T124, class T125, class T126, class T127, class T128, class T129, class T130, class T131, class T132, class T133, class T134, class T135, class T136, class T137, class T138, class T139, class T140, class T141, class T142, class T143>
BSK_HD auto _three_pool_step_adjoint_jvp_in_precision(const T0& r1_free, const T1& d_r1_free, const T2& r1_pool_b, const T3& d_r1_pool_b, const T4& r1_bound, const T5& d_r1_bound, const T6& exchange_b, const T7& d_exchange_b, const T8& exchange_c, const T9& d_exchange_c, const T10& fraction_b, const T11& d_fraction_b, const T12& fraction_c, const T13& d_fraction_c, const T14& dt, const T15& d_dt, const T16& attenuation, const T17& d_attenuation, const T18& bar_e00, const T19& d_bar_e00, const T20& bar_e01, const T21& d_bar_e01, const T22& bar_e02, const T23& d_bar_e02, const T24& bar_e10, const T25& d_bar_e10, const T26& bar_e11, const T27& d_bar_e11, const T28& bar_e12, const T29& d_bar_e12, const T30& bar_e20, const T31& d_bar_e20, const T32& bar_e21, const T33& d_bar_e21, const T34& bar_e22, const T35& d_bar_e22, const T36& bar_grow_free, const T37& d_bar_grow_free, const T38& bar_grow_pool_b, const T39& d_bar_grow_pool_b, const T40& bar_grow_bound, const T41& d_bar_grow_bound, const T42& free, const T43& d_free, const T44& pool_b, const T45& d_pool_b, const T46& pool_c, const T47& d_pool_c, const T48& a00, const T49& d_a00, const T50& a01, const T51& d_a01, const T52& a02, const T53& d_a02, const T54& a10, const T55& d_a10, const T56& a11, const T57& d_a11, const T58& a20, const T59& d_a20, const T60& a22, const T61& d_a22, const T62& s00, const T63& d_s00, const T64& s11, const T65& d_s11, const T66& s22, const T67& d_s22, const T68& minors, const T69& d_minors, const T70& sum_flat, const T71& sum_linear, const T72& sum_square, const T73& d_sum_flat, const T74& d_sum_linear, const T75& d_sum_square, const T76& lift, const T77& d_lift, const T78& low, const T79& middle, const T80& d_low, const T81& d_middle, const T82& leading, const T83& d_leading, const T84& first, const T85& d_first, const T86& second, const T87& d_second, const T88& determinant, const T89& d_determinant, const T90& high, const T91& d_high, const T92& radius, const T93& d_radius, const T94& cube, const T95& raw, const T96& d_raw, const T97& argument, const T98& inside_limit, const T99& angle, const T100& d_angle, const T101& centre, const T102& d_centre, const T103& trailing, const T104& d_trailing, const T105& guarded, const T106& d_guarded, const T107& q00, const T108& d_q00, const T109& q01, const T110& d_q01, const T111& q02, const T112& d_q02, const T113& q10, const T114& d_q10, const T115& q11, const T116& d_q11, const T117& q12, const T118& d_q12, const T119& q20, const T120& d_q20, const T121& q21, const T122& d_q21, const T123& q22, const T124& d_q22, const T125& def_00, const T126& dif_00, const T127& def_01, const T128& dif_01, const T129& def_02, const T130& dif_02, const T131& def_10, const T132& dif_10, const T133& def_11, const T134& dif_11, const T135& def_12, const T136& dif_12, const T137& def_20, const T138& dif_20, const T139& def_21, const T140& dif_21, const T141& def_22, const T142& dif_22, const T143& narrow) {
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a00{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a01{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a02{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a10{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a11{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a20{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_a22{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_determinant{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_first{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_high{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_low{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_middle{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_minors{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_radius{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> bar_third{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a00{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a01{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a02{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a10{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a11{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a20{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_a22{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_determinant{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_first{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_high{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_low{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_middle{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_minors{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_radius{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_bar_third{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_fu{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_fv{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_lu{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_lv{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_slope_u_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_slope_u_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_slope_u_square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_slope_v_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_slope_v_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_slope_v_square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_su{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_sv{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> d_turn_series{};
    float factorial{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> fu{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> fv{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> lu{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> lv{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> slope_u_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> slope_u_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> slope_u_square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> slope_v_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> slope_v_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> slope_v_square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> su{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> sv{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63, T64, T65, T66, T67, T68, T69, T70, T71, T72, T73, T74, T75, T76, T77, T78, T79, T80, T81, T82, T83, T84, T85, T86, T87, T88, T89, T90, T91, T92, T93, T94, T95, T96, T97, T98, T99, T100, T101, T102, T103, T104, T105, T106, T107, T108, T109, T110, T111, T112, T113, T114, T115, T116, T117, T118, T119, T120, T121, T122, T123, T124, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T138, T139, T140, T141, T142, T143> | 0, 2)> turn_series{};
    // --- the recovery and the attenuation, which both branches share ---
    auto damp = bsk::cast<Work>(attenuation);
    auto d_damp = bsk::cast<Work>(d_attenuation);
    auto r0 = bsk::cast<Work>(bar_grow_free);
    auto d_r0 = bsk::cast<Work>(d_bar_grow_free);
    auto r1 = bsk::cast<Work>(bar_grow_pool_b);
    auto d_r1 = bsk::cast<Work>(d_bar_grow_pool_b);
    auto r2 = bsk::cast<Work>(bar_grow_bound);
    auto d_r2 = bsk::cast<Work>(d_bar_grow_bound);
    auto y00 = (bsk::cast<Work>(bar_e00) - (r0 * free));
    auto d_y00 = ((bsk::cast<Work>(d_bar_e00) - (d_r0 * free)) - (r0 * d_free));
    auto y01 = (bsk::cast<Work>(bar_e01) - (r0 * pool_b));
    auto d_y01 = ((bsk::cast<Work>(d_bar_e01) - (d_r0 * pool_b)) - (r0 * d_pool_b));
    auto y02 = (bsk::cast<Work>(bar_e02) - (r0 * pool_c));
    auto d_y02 = ((bsk::cast<Work>(d_bar_e02) - (d_r0 * pool_c)) - (r0 * d_pool_c));
    auto y10 = (bsk::cast<Work>(bar_e10) - (r1 * free));
    auto d_y10 = ((bsk::cast<Work>(d_bar_e10) - (d_r1 * free)) - (r1 * d_free));
    auto y11 = (bsk::cast<Work>(bar_e11) - (r1 * pool_b));
    auto d_y11 = ((bsk::cast<Work>(d_bar_e11) - (d_r1 * pool_b)) - (r1 * d_pool_b));
    auto y12 = (bsk::cast<Work>(bar_e12) - (r1 * pool_c));
    auto d_y12 = ((bsk::cast<Work>(d_bar_e12) - (d_r1 * pool_c)) - (r1 * d_pool_c));
    auto y20 = (bsk::cast<Work>(bar_e20) - (r2 * free));
    auto d_y20 = ((bsk::cast<Work>(d_bar_e20) - (d_r2 * free)) - (r2 * d_free));
    auto y21 = (bsk::cast<Work>(bar_e21) - (r2 * pool_b));
    auto d_y21 = ((bsk::cast<Work>(d_bar_e21) - (d_r2 * pool_b)) - (r2 * d_pool_b));
    auto y22 = (bsk::cast<Work>(bar_e22) - (r2 * pool_c));
    auto d_y22 = ((bsk::cast<Work>(d_bar_e22) - (d_r2 * pool_c)) - (r2 * d_pool_c));
    // The bare operator is read rather than the attenuation divided back out
    // of the weighed one -- a washed-out interval leaves nothing to divide by.
    auto bar_damp = (((((((((y00 * def_00) + (y01 * def_01)) + (y02 * def_02)) + (y10 * def_10)) + (y11 * def_11)) + (y12 * def_12)) + (y20 * def_20)) + (y21 * def_21)) + (y22 * def_22));
    auto d_bar_damp = ((((((((((((((((((d_y00 * def_00) + (y00 * dif_00)) + (d_y01 * def_01)) + (y01 * dif_01)) + (d_y02 * def_02)) + (y02 * dif_02)) + (d_y10 * def_10)) + (y10 * dif_10)) + (d_y11 * def_11)) + (y11 * dif_11)) + (d_y12 * def_12)) + (y12 * dif_12)) + (d_y20 * def_20)) + (y20 * dif_20)) + (d_y21 * def_21)) + (y21 * dif_21)) + (d_y22 * def_22)) + (y22 * dif_22));
    auto column_free = (((r0 * def_00) + (r1 * def_10)) + (r2 * def_20));
    auto d_column_free = ((((((d_r0 * def_00) + (r0 * dif_00)) + (d_r1 * def_10)) + (r1 * dif_10)) + (d_r2 * def_20)) + (r2 * dif_20));
    auto column_pool_b = (((r0 * def_01) + (r1 * def_11)) + (r2 * def_21));
    auto d_column_pool_b = ((((((d_r0 * def_01) + (r0 * dif_01)) + (d_r1 * def_11)) + (r1 * dif_11)) + (d_r2 * def_21)) + (r2 * dif_21));
    auto column_bound = (((r0 * def_02) + (r1 * def_12)) + (r2 * def_22));
    auto d_column_bound = ((((((d_r0 * def_02) + (r0 * dif_02)) + (d_r1 * def_12)) + (r1 * dif_12)) + (d_r2 * def_22)) + (r2 * dif_22));
    auto bar_free = (r0 - (damp * column_free));
    auto d_bar_free = ((d_r0 - (d_damp * column_free)) - (damp * d_column_free));
    auto bar_pool_b = (r1 - (damp * column_pool_b));
    auto d_bar_pool_b = ((d_r1 - (d_damp * column_pool_b)) - (damp * d_column_pool_b));
    auto bar_pool_c = (r2 - (damp * column_bound));
    auto d_bar_pool_c = ((d_r2 - (d_damp * column_bound)) - (damp * d_column_bound));
    auto o00 = (damp * y00);
    auto d_o00 = ((d_damp * y00) + (damp * d_y00));
    auto o01 = (damp * y01);
    auto d_o01 = ((d_damp * y01) + (damp * d_y01));
    auto o02 = (damp * y02);
    auto d_o02 = ((d_damp * y02) + (damp * d_y02));
    auto o10 = (damp * y10);
    auto d_o10 = ((d_damp * y10) + (damp * d_y10));
    auto o11 = (damp * y11);
    auto d_o11 = ((d_damp * y11) + (damp * d_y11));
    auto o12 = (damp * y12);
    auto d_o12 = ((d_damp * y12) + (damp * d_y12));
    auto o20 = (damp * y20);
    auto d_o20 = ((d_damp * y20) + (damp * d_y20));
    auto o21 = (damp * y21);
    auto d_o21 = ((d_damp * y21) + (damp * d_y21));
    auto o22 = (damp * y22);
    auto d_o22 = ((d_damp * y22) + (damp * d_y22));
    // --- close together: the series in the two invariants, run backwards ---
    auto scale00 = (o00 * lift);
    auto d_scale00 = ((d_o00 * lift) + (o00 * d_lift));
    auto scale01 = (o01 * lift);
    auto d_scale01 = ((d_o01 * lift) + (o01 * d_lift));
    auto scale02 = (o02 * lift);
    auto d_scale02 = ((d_o02 * lift) + (o02 * d_lift));
    auto scale10 = (o10 * lift);
    auto d_scale10 = ((d_o10 * lift) + (o10 * d_lift));
    auto scale11 = (o11 * lift);
    auto d_scale11 = ((d_o11 * lift) + (o11 * d_lift));
    auto scale12 = (o12 * lift);
    auto d_scale12 = ((d_o12 * lift) + (o12 * d_lift));
    auto scale20 = (o20 * lift);
    auto d_scale20 = ((d_o20 * lift) + (o20 * d_lift));
    auto scale21 = (o21 * lift);
    auto d_scale21 = ((d_o21 * lift) + (o21 * d_lift));
    auto scale22 = (o22 * lift);
    auto d_scale22 = ((d_o22 * lift) + (o22 * d_lift));
    auto bar_flat = ((scale00 + scale11) + scale22);
    auto d_bar_flat = ((d_scale00 + d_scale11) + d_scale22);
    auto bar_linear = (((((((scale00 * s00) + (scale01 * a01)) + (scale02 * a02)) + (scale10 * a10)) + (scale11 * s11)) + (scale20 * a20)) + (scale22 * s22));
    auto d_bar_linear = ((((((((((((((d_scale00 * s00) + (scale00 * d_s00)) + (d_scale01 * a01)) + (scale01 * d_a01)) + (d_scale02 * a02)) + (scale02 * d_a02)) + (d_scale10 * a10)) + (scale10 * d_a10)) + (d_scale11 * s11)) + (scale11 * d_s11)) + (d_scale20 * a20)) + (scale20 * d_a20)) + (d_scale22 * s22)) + (scale22 * d_s22));
    auto bar_square = (((((((((scale00 * q00) + (scale01 * q01)) + (scale02 * q02)) + (scale10 * q10)) + (scale11 * q11)) + (scale12 * q12)) + (scale20 * q20)) + (scale21 * q21)) + (scale22 * q22));
    auto d_bar_square = ((((((((((((((((((d_scale00 * q00) + (scale00 * d_q00)) + (d_scale01 * q01)) + (scale01 * d_q01)) + (d_scale02 * q02)) + (scale02 * d_q02)) + (d_scale10 * q10)) + (scale10 * d_q10)) + (d_scale11 * q11)) + (scale11 * d_q11)) + (d_scale12 * q12)) + (scale12 * d_q12)) + (d_scale20 * q20)) + (scale20 * d_q20)) + (d_scale21 * q21)) + (scale21 * d_q21)) + (d_scale22 * q22)) + (scale22 * d_q22));
    // ``lift`` multiplies the whole bracket, so the shift it carries picks up
    // the bracket back again -- which is what the three sums contract to.
    turn_series = (((sum_flat * bar_flat) + (sum_linear * bar_linear)) + (sum_square * bar_square));
    d_turn_series = ((((((d_sum_flat * bar_flat) + (sum_flat * d_bar_flat)) + (d_sum_linear * bar_linear)) + (sum_linear * d_bar_linear)) + (d_sum_square * bar_square)) + (sum_square * d_bar_square));
    auto g00 = (sum_square * scale00);
    auto d_g00 = ((d_sum_square * scale00) + (sum_square * d_scale00));
    auto g01 = (sum_square * scale01);
    auto d_g01 = ((d_sum_square * scale01) + (sum_square * d_scale01));
    auto g02 = (sum_square * scale02);
    auto d_g02 = ((d_sum_square * scale02) + (sum_square * d_scale02));
    auto g10 = (sum_square * scale10);
    auto d_g10 = ((d_sum_square * scale10) + (sum_square * d_scale10));
    auto g11 = (sum_square * scale11);
    auto d_g11 = ((d_sum_square * scale11) + (sum_square * d_scale11));
    auto g12 = (sum_square * scale12);
    auto d_g12 = ((d_sum_square * scale12) + (sum_square * d_scale12));
    auto g20 = (sum_square * scale20);
    auto d_g20 = ((d_sum_square * scale20) + (sum_square * d_scale20));
    auto g21 = (sum_square * scale21);
    auto d_g21 = ((d_sum_square * scale21) + (sum_square * d_scale21));
    auto g22 = (sum_square * scale22);
    auto d_g22 = ((d_sum_square * scale22) + (sum_square * d_scale22));
    // The square's reverse, ``g @ shifted^T + shifted^T @ g``.
    auto v00 = (((((((g00 * s00) + (g01 * a01)) + (g02 * a02)) + (s00 * g00)) + (a10 * g10)) + (a20 * g20)) + (sum_linear * scale00));
    auto d_v00 = ((((((((((((((d_g00 * s00) + (g00 * d_s00)) + (d_g01 * a01)) + (g01 * d_a01)) + (d_g02 * a02)) + (g02 * d_a02)) + (d_s00 * g00)) + (s00 * d_g00)) + (d_a10 * g10)) + (a10 * d_g10)) + (d_a20 * g20)) + (a20 * d_g20)) + (d_sum_linear * scale00)) + (sum_linear * d_scale00));
    auto v01 = ((((((g00 * a10) + (g01 * s11)) + (s00 * g01)) + (a10 * g11)) + (a20 * g21)) + (sum_linear * scale01));
    auto d_v01 = ((((((((((((d_g00 * a10) + (g00 * d_a10)) + (d_g01 * s11)) + (g01 * d_s11)) + (d_s00 * g01)) + (s00 * d_g01)) + (d_a10 * g11)) + (a10 * d_g11)) + (d_a20 * g21)) + (a20 * d_g21)) + (d_sum_linear * scale01)) + (sum_linear * d_scale01));
    auto v02 = ((((((g00 * a20) + (g02 * s22)) + (s00 * g02)) + (a10 * g12)) + (a20 * g22)) + (sum_linear * scale02));
    auto d_v02 = ((((((((((((d_g00 * a20) + (g00 * d_a20)) + (d_g02 * s22)) + (g02 * d_s22)) + (d_s00 * g02)) + (s00 * d_g02)) + (d_a10 * g12)) + (a10 * d_g12)) + (d_a20 * g22)) + (a20 * d_g22)) + (d_sum_linear * scale02)) + (sum_linear * d_scale02));
    auto v10 = ((((((g10 * s00) + (g11 * a01)) + (g12 * a02)) + (a01 * g00)) + (s11 * g10)) + (sum_linear * scale10));
    auto d_v10 = ((((((((((((d_g10 * s00) + (g10 * d_s00)) + (d_g11 * a01)) + (g11 * d_a01)) + (d_g12 * a02)) + (g12 * d_a02)) + (d_a01 * g00)) + (a01 * d_g00)) + (d_s11 * g10)) + (s11 * d_g10)) + (d_sum_linear * scale10)) + (sum_linear * d_scale10));
    auto v11 = (((((g10 * a10) + (g11 * s11)) + (a01 * g01)) + (s11 * g11)) + (sum_linear * scale11));
    auto d_v11 = ((((((((((d_g10 * a10) + (g10 * d_a10)) + (d_g11 * s11)) + (g11 * d_s11)) + (d_a01 * g01)) + (a01 * d_g01)) + (d_s11 * g11)) + (s11 * d_g11)) + (d_sum_linear * scale11)) + (sum_linear * d_scale11));
    auto v20 = ((((((g20 * s00) + (g21 * a01)) + (g22 * a02)) + (a02 * g00)) + (s22 * g20)) + (sum_linear * scale20));
    auto d_v20 = ((((((((((((d_g20 * s00) + (g20 * d_s00)) + (d_g21 * a01)) + (g21 * d_a01)) + (d_g22 * a02)) + (g22 * d_a02)) + (d_a02 * g00)) + (a02 * d_g00)) + (d_s22 * g20)) + (s22 * d_g20)) + (d_sum_linear * scale20)) + (sum_linear * d_scale20));
    auto v22 = (((((g20 * a20) + (g22 * s22)) + (a02 * g02)) + (s22 * g22)) + (sum_linear * scale22));
    auto d_v22 = ((((((((((d_g20 * a20) + (g20 * d_a20)) + (d_g22 * s22)) + (g22 * d_s22)) + (d_a02 * g02)) + (a02 * d_g02)) + (d_s22 * g22)) + (s22 * d_g22)) + (d_sum_linear * scale22)) + (sum_linear * d_scale22));
    turn_series = (turn_series - ((v00 + v11) + v22));
    d_turn_series = (d_turn_series - ((d_v00 + d_v11) + d_v22));
    // The recurrence's own sensitivity to the two invariants, carried forward
    // beside it: two numbers reach the whole series, so their derivatives are
    // cheaper to push forward than the sixteen terms are to keep.
    flat = (Work(1.0) + (Work(0.0) * a00));
    linear = (Work(0.0) * a00);
    square = (Work(0.0) * a00);
    d_flat = (Work(0.0) * a00);
    d_linear = (Work(0.0) * a00);
    d_square = (Work(0.0) * a00);
    fu = (Work(0.0) * a00);
    lu = (Work(0.0) * a00);
    su = (Work(0.0) * a00);
    d_fu = (Work(0.0) * a00);
    d_lu = (Work(0.0) * a00);
    d_su = (Work(0.0) * a00);
    fv = (Work(0.0) * a00);
    lv = (Work(0.0) * a00);
    sv = (Work(0.0) * a00);
    d_fv = (Work(0.0) * a00);
    d_lv = (Work(0.0) * a00);
    d_sv = (Work(0.0) * a00);
    slope_u_flat = (Work(0.0) * a00);
    slope_u_linear = (Work(0.0) * a00);
    slope_u_square = (Work(0.0) * a00);
    d_slope_u_flat = (Work(0.0) * a00);
    d_slope_u_linear = (Work(0.0) * a00);
    d_slope_u_square = (Work(0.0) * a00);
    slope_v_flat = (Work(0.0) * a00);
    slope_v_linear = (Work(0.0) * a00);
    slope_v_square = (Work(0.0) * a00);
    d_slope_v_flat = (Work(0.0) * a00);
    d_slope_v_linear = (Work(0.0) * a00);
    d_slope_v_square = (Work(0.0) * a00);
    factorial = Work(1.0);
    #pragma unroll
    for (bsk::index_t order = 1; order < 16; order += 1) {
        auto next_flat = (square * determinant);
        auto d_next_flat = ((d_square * determinant) + (square * d_determinant));
        auto next_linear = (flat - (square * minors));
        auto d_next_linear = ((d_flat - (d_square * minors)) - (square * d_minors));
        auto next_square = linear;
        auto d_next_square = d_linear;
        auto next_fu = (su * determinant);
        auto d_next_fu = ((d_su * determinant) + (su * d_determinant));
        auto next_lu = ((fu - (su * minors)) - square);
        auto d_next_lu = (((d_fu - (d_su * minors)) - (su * d_minors)) - d_square);
        auto next_su = lu;
        auto d_next_su = d_lu;
        auto next_fv = ((sv * determinant) + square);
        auto d_next_fv = (((d_sv * determinant) + (sv * d_determinant)) + d_square);
        auto next_lv = (fv - (sv * minors));
        auto d_next_lv = ((d_fv - (d_sv * minors)) - (sv * d_minors));
        auto next_sv = lv;
        auto d_next_sv = d_lv;
        flat = next_flat;
        linear = next_linear;
        square = next_square;
        d_flat = d_next_flat;
        d_linear = d_next_linear;
        d_square = d_next_square;
        fu = next_fu;
        lu = next_lu;
        su = next_su;
        d_fu = d_next_fu;
        d_lu = d_next_lu;
        d_su = d_next_su;
        fv = next_fv;
        lv = next_lv;
        sv = next_sv;
        d_fv = d_next_fv;
        d_lv = d_next_lv;
        d_sv = d_next_sv;
        factorial = (factorial * order);
        auto weight = bsk::truediv(Work(1.0), factorial);
        slope_u_flat = (slope_u_flat + (weight * fu));
        slope_u_linear = (slope_u_linear + (weight * lu));
        slope_u_square = (slope_u_square + (weight * su));
        d_slope_u_flat = (d_slope_u_flat + (weight * d_fu));
        d_slope_u_linear = (d_slope_u_linear + (weight * d_lu));
        d_slope_u_square = (d_slope_u_square + (weight * d_su));
        slope_v_flat = (slope_v_flat + (weight * fv));
        slope_v_linear = (slope_v_linear + (weight * lv));
        slope_v_square = (slope_v_square + (weight * sv));
        d_slope_v_flat = (d_slope_v_flat + (weight * d_fv));
        d_slope_v_linear = (d_slope_v_linear + (weight * d_lv));
        d_slope_v_square = (d_slope_v_square + (weight * d_sv));
    }
    auto minors_series = (((bar_flat * slope_u_flat) + (bar_linear * slope_u_linear)) + (bar_square * slope_u_square));
    auto d_minors_series = ((((((d_bar_flat * slope_u_flat) + (bar_flat * d_slope_u_flat)) + (d_bar_linear * slope_u_linear)) + (bar_linear * d_slope_u_linear)) + (d_bar_square * slope_u_square)) + (bar_square * d_slope_u_square));
    auto determinant_series = (((bar_flat * slope_v_flat) + (bar_linear * slope_v_linear)) + (bar_square * slope_v_square));
    auto d_determinant_series = ((((((d_bar_flat * slope_v_flat) + (bar_flat * d_slope_v_flat)) + (d_bar_linear * slope_v_linear)) + (bar_linear * d_slope_v_linear)) + (d_bar_square * slope_v_square)) + (bar_square * d_slope_v_square));
    // --- far apart: back through the Newton form and the three roots ---
    auto m00 = (a00 - low);
    auto d_m00 = (d_a00 - d_low);
    auto m11 = (a11 - low);
    auto d_m11 = (d_a11 - d_low);
    auto m22 = (a22 - low);
    auto d_m22 = (d_a22 - d_low);
    auto n00 = (a00 - middle);
    auto d_n00 = (d_a00 - d_middle);
    auto n11 = (a11 - middle);
    auto d_n11 = (d_a11 - d_middle);
    auto n22 = (a22 - middle);
    auto d_n22 = (d_a22 - d_middle);
    auto p00 = (((m00 * n00) + (a01 * a10)) + (a02 * a20));
    auto d_p00 = ((((((d_m00 * n00) + (m00 * d_n00)) + (d_a01 * a10)) + (a01 * d_a10)) + (d_a02 * a20)) + (a02 * d_a20));
    auto p01 = (a01 * (m00 + n11));
    auto d_p01 = ((d_a01 * (m00 + n11)) + (a01 * (d_m00 + d_n11)));
    auto p02 = (a02 * (m00 + n22));
    auto d_p02 = ((d_a02 * (m00 + n22)) + (a02 * (d_m00 + d_n22)));
    auto p10 = (a10 * (m11 + n00));
    auto d_p10 = ((d_a10 * (m11 + n00)) + (a10 * (d_m11 + d_n00)));
    auto p11 = ((m11 * n11) + (a01 * a10));
    auto d_p11 = ((((d_m11 * n11) + (m11 * d_n11)) + (d_a01 * a10)) + (a01 * d_a10));
    auto p12 = (a10 * a02);
    auto d_p12 = ((d_a10 * a02) + (a10 * d_a02));
    auto p20 = (a20 * (m22 + n00));
    auto d_p20 = ((d_a20 * (m22 + n00)) + (a20 * (d_m22 + d_n00)));
    auto p21 = (a20 * a01);
    auto d_p21 = ((d_a20 * a01) + (a20 * d_a01));
    auto p22 = ((m22 * n22) + (a02 * a20));
    auto d_p22 = ((((d_m22 * n22) + (m22 * d_n22)) + (d_a02 * a20)) + (a02 * d_a20));
    auto bar_leading = ((o00 + o11) + o22);
    auto d_bar_leading = ((d_o00 + d_o11) + d_o22);
    bar_first = (((((((o00 * m00) + (o01 * a01)) + (o02 * a02)) + (o10 * a10)) + (o11 * m11)) + (o20 * a20)) + (o22 * m22));
    d_bar_first = ((((((((((((((d_o00 * m00) + (o00 * d_m00)) + (d_o01 * a01)) + (o01 * d_a01)) + (d_o02 * a02)) + (o02 * d_a02)) + (d_o10 * a10)) + (o10 * d_a10)) + (d_o11 * m11)) + (o11 * d_m11)) + (d_o20 * a20)) + (o20 * d_a20)) + (d_o22 * m22)) + (o22 * d_m22));
    auto bar_second = (((((((((o00 * p00) + (o01 * p01)) + (o02 * p02)) + (o10 * p10)) + (o11 * p11)) + (o12 * p12)) + (o20 * p20)) + (o21 * p21)) + (o22 * p22));
    auto d_bar_second = ((((((((((((((((((d_o00 * p00) + (o00 * d_p00)) + (d_o01 * p01)) + (o01 * d_p01)) + (d_o02 * p02)) + (o02 * d_p02)) + (d_o10 * p10)) + (o10 * d_p10)) + (d_o11 * p11)) + (o11 * d_p11)) + (d_o12 * p12)) + (o12 * d_p12)) + (d_o20 * p20)) + (o20 * d_p20)) + (d_o21 * p21)) + (o21 * d_p21)) + (d_o22 * p22)) + (o22 * d_p22));
    auto z00 = (second * o00);
    auto d_z00 = ((d_second * o00) + (second * d_o00));
    auto z01 = (second * o01);
    auto d_z01 = ((d_second * o01) + (second * d_o01));
    auto z02 = (second * o02);
    auto d_z02 = ((d_second * o02) + (second * d_o02));
    auto z10 = (second * o10);
    auto d_z10 = ((d_second * o10) + (second * d_o10));
    auto z11 = (second * o11);
    auto d_z11 = ((d_second * o11) + (second * d_o11));
    auto z12 = (second * o12);
    auto d_z12 = ((d_second * o12) + (second * d_o12));
    auto z20 = (second * o20);
    auto d_z20 = ((d_second * o20) + (second * d_o20));
    auto z21 = (second * o21);
    auto d_z21 = ((d_second * o21) + (second * d_o21));
    auto z22 = (second * o22);
    auto d_z22 = ((d_second * o22) + (second * d_o22));
    // ``z @ n^T``, the product's reverse onto the first factor.
    auto u00 = (((z00 * n00) + (z01 * a01)) + (z02 * a02));
    auto d_u00 = ((((((d_z00 * n00) + (z00 * d_n00)) + (d_z01 * a01)) + (z01 * d_a01)) + (d_z02 * a02)) + (z02 * d_a02));
    auto u01 = ((z00 * a10) + (z01 * n11));
    auto d_u01 = ((((d_z00 * a10) + (z00 * d_a10)) + (d_z01 * n11)) + (z01 * d_n11));
    auto u02 = ((z00 * a20) + (z02 * n22));
    auto d_u02 = ((((d_z00 * a20) + (z00 * d_a20)) + (d_z02 * n22)) + (z02 * d_n22));
    auto u10 = (((z10 * n00) + (z11 * a01)) + (z12 * a02));
    auto d_u10 = ((((((d_z10 * n00) + (z10 * d_n00)) + (d_z11 * a01)) + (z11 * d_a01)) + (d_z12 * a02)) + (z12 * d_a02));
    auto u11 = ((z10 * a10) + (z11 * n11));
    auto d_u11 = ((((d_z10 * a10) + (z10 * d_a10)) + (d_z11 * n11)) + (z11 * d_n11));
    auto u20 = (((z20 * n00) + (z21 * a01)) + (z22 * a02));
    auto d_u20 = ((((((d_z20 * n00) + (z20 * d_n00)) + (d_z21 * a01)) + (z21 * d_a01)) + (d_z22 * a02)) + (z22 * d_a02));
    auto u22 = ((z20 * a20) + (z22 * n22));
    auto d_u22 = ((((d_z20 * a20) + (z20 * d_a20)) + (d_z22 * n22)) + (z22 * d_n22));
    // ``m^T @ z``, onto the second.
    auto w00 = (((m00 * z00) + (a10 * z10)) + (a20 * z20));
    auto d_w00 = ((((((d_m00 * z00) + (m00 * d_z00)) + (d_a10 * z10)) + (a10 * d_z10)) + (d_a20 * z20)) + (a20 * d_z20));
    auto w01 = (((m00 * z01) + (a10 * z11)) + (a20 * z21));
    auto d_w01 = ((((((d_m00 * z01) + (m00 * d_z01)) + (d_a10 * z11)) + (a10 * d_z11)) + (d_a20 * z21)) + (a20 * d_z21));
    auto w02 = (((m00 * z02) + (a10 * z12)) + (a20 * z22));
    auto d_w02 = ((((((d_m00 * z02) + (m00 * d_z02)) + (d_a10 * z12)) + (a10 * d_z12)) + (d_a20 * z22)) + (a20 * d_z22));
    auto w10 = ((a01 * z00) + (m11 * z10));
    auto d_w10 = ((((d_a01 * z00) + (a01 * d_z00)) + (d_m11 * z10)) + (m11 * d_z10));
    auto w11 = ((a01 * z01) + (m11 * z11));
    auto d_w11 = ((((d_a01 * z01) + (a01 * d_z01)) + (d_m11 * z11)) + (m11 * d_z11));
    auto w20 = ((a02 * z00) + (m22 * z20));
    auto d_w20 = ((((d_a02 * z00) + (a02 * d_z00)) + (d_m22 * z20)) + (m22 * d_z20));
    auto w22 = ((a02 * z02) + (m22 * z22));
    auto d_w22 = ((((d_a02 * z02) + (a02 * d_z02)) + (d_m22 * z22)) + (m22 * d_z22));
    bar_low = (((bar_leading * leading) - (first * ((o00 + o11) + o22))) - ((u00 + u11) + u22));
    d_bar_low = (((((d_bar_leading * leading) + (bar_leading * d_leading)) - (d_first * ((o00 + o11) + o22))) - (first * ((d_o00 + d_o11) + d_o22))) - ((d_u00 + d_u11) + d_u22));
    bar_middle = (-((w00 + w11) + w22));
    d_bar_middle = (-((d_w00 + d_w11) + d_w22));
    bar_high = (Work(0.0) * a00);
    d_bar_high = (Work(0.0) * a00);
    auto span = (high - low);
    auto positive = (span > Work(0.0));
    auto bar_upper = bsk::truediv(bar_second, guarded);
    auto d_bar_upper = bsk::truediv((d_bar_second - (bar_upper * d_guarded)), guarded);
    bar_first = (bar_first - bar_upper);
    d_bar_first = (d_bar_first - d_bar_upper);
    auto bar_span = bsk::where(positive, ((-bar_upper) * second), Work(0.0));
    auto d_bar_span = bsk::where(positive, (((-d_bar_upper) * second) - (bar_upper * d_second)), Work(0.0));
    bar_high = (bar_high + bar_span);
    d_bar_high = (d_bar_high + d_bar_span);
    bar_low = (bar_low - bar_span);
    d_bar_low = (d_bar_low - d_bar_span);
    auto t0_ = _exp_difference_adjoint_jvp(low, d_low, middle, d_middle, leading, d_leading, centre, d_centre, bar_first, d_bar_first);
    auto from_first_low = bsk::get<0>(t0_);
    auto d_from_first_low = bsk::get<1>(t0_);
    auto from_first_middle = bsk::get<2>(t0_);
    auto d_from_first_middle = bsk::get<3>(t0_);
    auto t1_ = _exp_difference_adjoint_jvp(middle, d_middle, high, d_high, centre, d_centre, trailing, d_trailing, bar_upper, d_bar_upper);
    auto from_upper_middle = bsk::get<0>(t1_);
    auto d_from_upper_middle = bsk::get<1>(t1_);
    auto from_upper_high = bsk::get<2>(t1_);
    auto d_from_upper_high = bsk::get<3>(t1_);
    bar_low = (bar_low + from_first_low);
    d_bar_low = (d_bar_low + d_from_first_low);
    bar_middle = ((bar_middle + from_first_middle) + from_upper_middle);
    d_bar_middle = ((d_bar_middle + d_from_first_middle) + d_from_upper_middle);
    bar_high = (bar_high + from_upper_high);
    d_bar_high = (d_bar_high + d_from_upper_high);
    // The three roots come off one angle a third of a turn apart, and the
    // cosine puts them in a fixed order: the last turn is the lowest, the
    // first the highest, whatever the angle is.
    auto swing_low = (angle - Work(4.188790204786391));
    auto swing_middle = (angle - Work(2.0943951023931957));
    auto cos_low = bsk::cos(swing_low);
    auto cos_middle = bsk::cos(swing_middle);
    auto cos_high = bsk::cos(angle);
    auto sin_low = bsk::sin(swing_low);
    auto sin_middle = bsk::sin(swing_middle);
    auto sin_high = bsk::sin(angle);
    bar_radius = (Work(2.0) * (((cos_low * bar_low) + (cos_middle * bar_middle)) + (cos_high * bar_high)));
    d_bar_radius = (Work(2.0) * ((((cos_low * d_bar_low) + (cos_middle * d_bar_middle)) + (cos_high * d_bar_high)) - (d_angle * (((sin_low * bar_low) + (sin_middle * bar_middle)) + (sin_high * bar_high)))));
    auto swept = (((sin_low * bar_low) + (sin_middle * bar_middle)) + (sin_high * bar_high));
    auto d_swept = ((((sin_low * d_bar_low) + (sin_middle * d_bar_middle)) + (sin_high * d_bar_high)) + (d_angle * (((cos_low * bar_low) + (cos_middle * bar_middle)) + (cos_high * bar_high))));
    auto bar_angle = ((Work(-2.0) * radius) * swept);
    auto d_bar_angle = (Work(-2.0) * ((d_radius * swept) + (radius * d_swept)));
    auto turn_roots = ((bar_low + bar_middle) + bar_high);
    auto d_turn_roots = ((d_bar_low + d_bar_middle) + d_bar_high);
    // ``acos`` is clamped, and where it is the angle no longer moves with the
    // cubic's argument -- which is what keeps a double root differentiable.
    auto d_argument = bsk::where(inside_limit, d_raw, Work(0.0));
    auto inner = (Work(1.0) - (argument * argument));
    auto d_inner = ((Work(-2.0) * argument) * d_argument);
    auto stem = bsk::sqrt(bsk::maximum(inner, Work(1e-300)));
    auto tilt = bsk::truediv(Work(-1.0), (Work(3.0) * stem));
    auto d_tilt = bsk::truediv(d_inner, (((Work(6.0) * stem) * stem) * stem));
    auto bar_raw = bsk::where(inside_limit, (bar_angle * tilt), Work(0.0));
    auto d_bar_raw = bsk::where(inside_limit, ((d_bar_angle * tilt) + (bar_angle * d_tilt)), Work(0.0));
    auto safe_radius = bsk::where((radius > Work(1e-30)), radius, Work(1.0));
    auto d_safe_radius = bsk::where((radius > Work(1e-30)), d_radius, Work(0.0));
    auto safe_cube = ((safe_radius * safe_radius) * safe_radius);
    auto d_safe_cube = (((Work(3.0) * safe_radius) * safe_radius) * d_safe_radius);
    auto determinant_roots = bsk::truediv((Work(0.5) * bar_raw), safe_cube);
    auto d_determinant_roots = bsk::truediv(((Work(0.5) * d_bar_raw) - (determinant_roots * d_safe_cube)), safe_cube);
    auto pull = bsk::where(inside_limit, bsk::truediv(((Work(-3.0) * raw) * bar_raw), safe_radius), Work(0.0));
    auto d_pull = bsk::where(inside_limit, bsk::truediv(((Work(-3.0) * ((d_raw * bar_raw) + (raw * d_bar_raw))) - (pull * d_safe_radius)), safe_radius), Work(0.0));
    bar_radius = (bar_radius + pull);
    d_bar_radius = (d_bar_radius + d_pull);
    auto minors_roots = bsk::truediv((-bar_radius), (Work(6.0) * safe_radius));
    auto d_minors_roots = bsk::truediv(((-d_bar_radius) - ((minors_roots * Work(6.0)) * d_safe_radius)), (Work(6.0) * safe_radius));
    // --- the branch chosen on the cotangents, not on the way in ---
    if (bsk::truth(narrow)) {
        auto t2_ = bsk::make_tup(v00, d_v00);
        bar_a00 = bsk::get<0>(t2_);
        d_bar_a00 = bsk::get<1>(t2_);
        auto t3_ = bsk::make_tup(v01, d_v01);
        bar_a01 = bsk::get<0>(t3_);
        d_bar_a01 = bsk::get<1>(t3_);
        auto t4_ = bsk::make_tup(v02, d_v02);
        bar_a02 = bsk::get<0>(t4_);
        d_bar_a02 = bsk::get<1>(t4_);
        auto t5_ = bsk::make_tup(v10, d_v10);
        bar_a10 = bsk::get<0>(t5_);
        d_bar_a10 = bsk::get<1>(t5_);
        auto t6_ = bsk::make_tup(v11, d_v11);
        bar_a11 = bsk::get<0>(t6_);
        d_bar_a11 = bsk::get<1>(t6_);
        auto t7_ = bsk::make_tup(v20, d_v20);
        bar_a20 = bsk::get<0>(t7_);
        d_bar_a20 = bsk::get<1>(t7_);
        auto t8_ = bsk::make_tup(v22, d_v22);
        bar_a22 = bsk::get<0>(t8_);
        d_bar_a22 = bsk::get<1>(t8_);
        auto t9_ = bsk::make_tup(turn_series, d_turn_series);
        bar_third = bsk::get<0>(t9_);
        d_bar_third = bsk::get<1>(t9_);
        auto t10_ = bsk::make_tup(minors_series, d_minors_series);
        bar_minors = bsk::get<0>(t10_);
        d_bar_minors = bsk::get<1>(t10_);
        auto t11_ = bsk::make_tup(determinant_series, d_determinant_series);
        bar_determinant = bsk::get<0>(t11_);
        d_bar_determinant = bsk::get<1>(t11_);
    } else {
        auto close = ((Work(-2.0) * minors) < Work(1.0));
        bar_a00 = bsk::where(close, v00, (((first * o00) + u00) + w00));
        d_bar_a00 = bsk::where(close, d_v00, ((((d_first * o00) + (first * d_o00)) + d_u00) + d_w00));
        bar_a01 = bsk::where(close, v01, (((first * o01) + u01) + w01));
        d_bar_a01 = bsk::where(close, d_v01, ((((d_first * o01) + (first * d_o01)) + d_u01) + d_w01));
        bar_a02 = bsk::where(close, v02, (((first * o02) + u02) + w02));
        d_bar_a02 = bsk::where(close, d_v02, ((((d_first * o02) + (first * d_o02)) + d_u02) + d_w02));
        bar_a10 = bsk::where(close, v10, (((first * o10) + u10) + w10));
        d_bar_a10 = bsk::where(close, d_v10, ((((d_first * o10) + (first * d_o10)) + d_u10) + d_w10));
        bar_a11 = bsk::where(close, v11, (((first * o11) + u11) + w11));
        d_bar_a11 = bsk::where(close, d_v11, ((((d_first * o11) + (first * d_o11)) + d_u11) + d_w11));
        bar_a20 = bsk::where(close, v20, (((first * o20) + u20) + w20));
        d_bar_a20 = bsk::where(close, d_v20, ((((d_first * o20) + (first * d_o20)) + d_u20) + d_w20));
        bar_a22 = bsk::where(close, v22, (((first * o22) + u22) + w22));
        d_bar_a22 = bsk::where(close, d_v22, ((((d_first * o22) + (first * d_o22)) + d_u22) + d_w22));
        bar_third = bsk::where(close, turn_series, turn_roots);
        d_bar_third = bsk::where(close, d_turn_series, d_turn_roots);
        bar_minors = bsk::where(close, minors_series, minors_roots);
        d_bar_minors = bsk::where(close, d_minors_series, d_minors_roots);
        bar_determinant = bsk::where(close, determinant_series, determinant_roots);
        d_bar_determinant = bsk::where(close, d_determinant_series, d_determinant_roots);
    }
    // --- the two invariants back onto the shifted generator ---
    auto cofactor00 = (s11 * s22);
    auto d_cofactor00 = ((d_s11 * s22) + (s11 * d_s22));
    auto cofactor11 = ((s00 * s22) - (a02 * a20));
    auto d_cofactor11 = ((((d_s00 * s22) + (s00 * d_s22)) - (d_a02 * a20)) - (a02 * d_a20));
    auto cofactor22 = ((s00 * s11) - (a01 * a10));
    auto d_cofactor22 = ((((d_s00 * s11) + (s00 * d_s11)) - (d_a01 * a10)) - (a01 * d_a10));
    auto shift00 = ((bar_minors * (s11 + s22)) + (bar_determinant * cofactor00));
    auto d_shift00 = ((((d_bar_minors * (s11 + s22)) + (bar_minors * (d_s11 + d_s22))) + (d_bar_determinant * cofactor00)) + (bar_determinant * d_cofactor00));
    auto shift11 = ((bar_minors * (s00 + s22)) + (bar_determinant * cofactor11));
    auto d_shift11 = ((((d_bar_minors * (s00 + s22)) + (bar_minors * (d_s00 + d_s22))) + (d_bar_determinant * cofactor11)) + (bar_determinant * d_cofactor11));
    auto shift22 = ((bar_minors * (s00 + s11)) + (bar_determinant * cofactor22));
    auto d_shift22 = ((((d_bar_minors * (s00 + s11)) + (bar_minors * (d_s00 + d_s11))) + (d_bar_determinant * cofactor22)) + (bar_determinant * d_cofactor22));
    auto shift01 = ((-a10) * (bar_minors + (bar_determinant * s22)));
    auto d_shift01 = (((-d_a10) * (bar_minors + (bar_determinant * s22))) - (a10 * ((d_bar_minors + (d_bar_determinant * s22)) + (bar_determinant * d_s22))));
    auto shift10 = ((-a01) * (bar_minors + (bar_determinant * s22)));
    auto d_shift10 = (((-d_a01) * (bar_minors + (bar_determinant * s22))) - (a01 * ((d_bar_minors + (d_bar_determinant * s22)) + (bar_determinant * d_s22))));
    auto shift02 = ((-a20) * (bar_minors + (bar_determinant * s11)));
    auto d_shift02 = (((-d_a20) * (bar_minors + (bar_determinant * s11))) - (a20 * ((d_bar_minors + (d_bar_determinant * s11)) + (bar_determinant * d_s11))));
    auto shift20 = ((-a02) * (bar_minors + (bar_determinant * s11)));
    auto d_shift20 = (((-d_a02) * (bar_minors + (bar_determinant * s11))) - (a02 * ((d_bar_minors + (d_bar_determinant * s11)) + (bar_determinant * d_s11))));
    bar_a00 = (bar_a00 + shift00);
    d_bar_a00 = (d_bar_a00 + d_shift00);
    bar_a01 = (bar_a01 + shift01);
    d_bar_a01 = (d_bar_a01 + d_shift01);
    bar_a02 = (bar_a02 + shift02);
    d_bar_a02 = (d_bar_a02 + d_shift02);
    bar_a10 = (bar_a10 + shift10);
    d_bar_a10 = (d_bar_a10 + d_shift10);
    bar_a11 = (bar_a11 + shift11);
    d_bar_a11 = (d_bar_a11 + d_shift11);
    bar_a20 = (bar_a20 + shift20);
    d_bar_a20 = (d_bar_a20 + d_shift20);
    bar_a22 = (bar_a22 + shift22);
    d_bar_a22 = (d_bar_a22 + d_shift22);
    bar_third = (bar_third - ((shift00 + shift11) + shift22));
    d_bar_third = (d_bar_third - ((d_shift00 + d_shift11) + d_shift22));
    bar_a00 = (bar_a00 + bsk::truediv(bar_third, Work(3.0)));
    d_bar_a00 = (d_bar_a00 + bsk::truediv(d_bar_third, Work(3.0)));
    bar_a11 = (bar_a11 + bsk::truediv(bar_third, Work(3.0)));
    d_bar_a11 = (d_bar_a11 + bsk::truediv(d_bar_third, Work(3.0)));
    bar_a22 = (bar_a22 + bsk::truediv(bar_third, Work(3.0)));
    d_bar_a22 = (d_bar_a22 + bsk::truediv(d_bar_third, Work(3.0)));
    // --- the generator back onto the rates, the fractions and the interval ---
    auto step = bsk::cast<Work>(dt);
    auto d_step = bsk::cast<Work>(d_dt);
    auto rate_b = bsk::cast<Work>(exchange_b);
    auto d_rate_b = bsk::cast<Work>(d_exchange_b);
    auto rate_c = bsk::cast<Work>(exchange_c);
    auto d_rate_c = bsk::cast<Work>(d_exchange_c);
    auto kab = (rate_b * pool_b);
    auto d_kab = ((d_rate_b * pool_b) + (rate_b * d_pool_b));
    auto kba = (rate_b * free);
    auto d_kba = ((d_rate_b * free) + (rate_b * d_free));
    auto kac = (rate_c * pool_c);
    auto d_kac = ((d_rate_c * pool_c) + (rate_c * d_pool_c));
    auto kca = (rate_c * free);
    auto d_kca = ((d_rate_c * free) + (rate_c * d_free));
    auto row_a = (((-kab) - kac) - bsk::cast<Work>(r1_free));
    auto d_row_a = (((-d_kab) - d_kac) - bsk::cast<Work>(d_r1_free));
    auto row_b = ((-kba) - bsk::cast<Work>(r1_pool_b));
    auto d_row_b = ((-d_kba) - bsk::cast<Work>(d_r1_pool_b));
    auto row_c = ((-kca) - bsk::cast<Work>(r1_bound));
    auto d_row_c = ((-d_kca) - bsk::cast<Work>(d_r1_bound));
    auto bar_step = (((((((row_a * bar_a00) + (kba * bar_a01)) + (kca * bar_a02)) + (kab * bar_a10)) + (row_b * bar_a11)) + (kac * bar_a20)) + (row_c * bar_a22));
    auto d_bar_step = ((((((((((((((d_row_a * bar_a00) + (row_a * d_bar_a00)) + (d_kba * bar_a01)) + (kba * d_bar_a01)) + (d_kca * bar_a02)) + (kca * d_bar_a02)) + (d_kab * bar_a10)) + (kab * d_bar_a10)) + (d_row_b * bar_a11)) + (row_b * d_bar_a11)) + (d_kac * bar_a20)) + (kac * d_bar_a20)) + (d_row_c * bar_a22)) + (row_c * d_bar_a22));
    auto bar_kab = (step * (bar_a10 - bar_a00));
    auto d_bar_kab = ((d_step * (bar_a10 - bar_a00)) + (step * (d_bar_a10 - d_bar_a00)));
    auto bar_kba = (step * (bar_a01 - bar_a11));
    auto d_bar_kba = ((d_step * (bar_a01 - bar_a11)) + (step * (d_bar_a01 - d_bar_a11)));
    auto bar_kac = (step * (bar_a20 - bar_a00));
    auto d_bar_kac = ((d_step * (bar_a20 - bar_a00)) + (step * (d_bar_a20 - d_bar_a00)));
    auto bar_kca = (step * (bar_a02 - bar_a22));
    auto d_bar_kca = ((d_step * (bar_a02 - bar_a22)) + (step * (d_bar_a02 - d_bar_a22)));
    auto whole_free = ((bar_free + (rate_b * bar_kba)) + (rate_c * bar_kca));
    auto d_whole_free = ((((d_bar_free + (d_rate_b * bar_kba)) + (rate_b * d_bar_kba)) + (d_rate_c * bar_kca)) + (rate_c * d_bar_kca));
    auto whole_pool_b = (bar_pool_b + (rate_b * bar_kab));
    auto d_whole_pool_b = ((d_bar_pool_b + (d_rate_b * bar_kab)) + (rate_b * d_bar_kab));
    auto whole_pool_c = (bar_pool_c + (rate_c * bar_kac));
    auto d_whole_pool_c = ((d_bar_pool_c + (d_rate_c * bar_kac)) + (rate_c * d_bar_kac));
    return bsk::make_tup(bsk::cast<float>(((-step) * bar_a00)), bsk::cast<float>(((-step) * bar_a11)), bsk::cast<float>(((-step) * bar_a22)), bsk::cast<float>(((pool_b * bar_kab) + (free * bar_kba))), bsk::cast<float>(((pool_c * bar_kac) + (free * bar_kca))), bsk::cast<float>((whole_pool_b - whole_free)), bsk::cast<float>((whole_pool_c - whole_free)), bsk::cast<float>(bar_step), bsk::cast<float>(bar_damp), bsk::cast<float>((((-d_step) * bar_a00) - (step * d_bar_a00))), bsk::cast<float>((((-d_step) * bar_a11) - (step * d_bar_a11))), bsk::cast<float>((((-d_step) * bar_a22) - (step * d_bar_a22))), bsk::cast<float>(((((d_pool_b * bar_kab) + (pool_b * d_bar_kab)) + (d_free * bar_kba)) + (free * d_bar_kba))), bsk::cast<float>(((((d_pool_c * bar_kac) + (pool_c * d_bar_kac)) + (d_free * bar_kca)) + (free * d_bar_kca))), bsk::cast<float>((d_whole_pool_b - d_whole_free)), bsk::cast<float>((d_whole_pool_c - d_whole_free)), bsk::cast<float>(d_bar_step), bsk::cast<float>(d_bar_damp));
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24, class T25, class T26, class T27, class T28, class T29, class T30, class T31, class T32, class T33, class T34, class T35, class T36, class T37, class T38, class T39, class T40, class T41, class T42, class T43, class T44, class T45, class T46, class T47, class T48, class T49, class T50, class T51, class T52, class T53, class T54, class T55, class T56, class T57, class T58, class T59, class T60, class T61, class T62, class T63, class T64, class T65, class T66, class T67, class T68, class T69, class T70, class T71, class T72, class T73, class T74, class T75, class T76, class T77, class T78, class T79, class T80, class T81, class T82, class T83, class T84, class T85, class T86, class T87, class T88, class T89, class T90, class T91, class T92, class T93, class T94, class T95, class T96, class T97, class T98, class T99, class T100, class T101, class T102, class T103, class T104, class T105, class T106, class T107, class T108, class T109, class T110, class T111, class T112, class T113, class T114, class T115, class T116, class T117, class T118, class T119, class T120, class T121, class T122, class T123, class T124, class T125, class T126, class T127, class T128, class T129, class T130, class T131, class T132, class T133, class T134, class T135, class T136, class T137, class T138, class T139, class T140, class T141, class T142, class T143>
BSK_HD auto _three_pool_step_adjoint_jvp(const T0& r1_free, const T1& d_r1_free, const T2& r1_pool_b, const T3& d_r1_pool_b, const T4& r1_bound, const T5& d_r1_bound, const T6& exchange_b, const T7& d_exchange_b, const T8& exchange_c, const T9& d_exchange_c, const T10& fraction_b, const T11& d_fraction_b, const T12& fraction_c, const T13& d_fraction_c, const T14& dt, const T15& d_dt, const T16& attenuation, const T17& d_attenuation, const T18& bar_e00, const T19& d_bar_e00, const T20& bar_e01, const T21& d_bar_e01, const T22& bar_e02, const T23& d_bar_e02, const T24& bar_e10, const T25& d_bar_e10, const T26& bar_e11, const T27& d_bar_e11, const T28& bar_e12, const T29& d_bar_e12, const T30& bar_e20, const T31& d_bar_e20, const T32& bar_e21, const T33& d_bar_e21, const T34& bar_e22, const T35& d_bar_e22, const T36& bar_grow_free, const T37& d_bar_grow_free, const T38& bar_grow_pool_b, const T39& d_bar_grow_pool_b, const T40& bar_grow_bound, const T41& d_bar_grow_bound, const T42& free, const T43& d_free, const T44& pool_b, const T45& d_pool_b, const T46& pool_c, const T47& d_pool_c, const T48& a00, const T49& d_a00, const T50& a01, const T51& d_a01, const T52& a02, const T53& d_a02, const T54& a10, const T55& d_a10, const T56& a11, const T57& d_a11, const T58& a20, const T59& d_a20, const T60& a22, const T61& d_a22, const T62& s00, const T63& d_s00, const T64& s11, const T65& d_s11, const T66& s22, const T67& d_s22, const T68& minors, const T69& d_minors, const T70& sum_flat, const T71& sum_linear, const T72& sum_square, const T73& d_sum_flat, const T74& d_sum_linear, const T75& d_sum_square, const T76& lift, const T77& d_lift, const T78& low, const T79& middle, const T80& d_low, const T81& d_middle, const T82& leading, const T83& d_leading, const T84& first, const T85& d_first, const T86& second, const T87& d_second, const T88& determinant, const T89& d_determinant, const T90& high, const T91& d_high, const T92& radius, const T93& d_radius, const T94& cube, const T95& raw, const T96& d_raw, const T97& argument, const T98& inside_limit, const T99& angle, const T100& d_angle, const T101& centre, const T102& d_centre, const T103& trailing, const T104& d_trailing, const T105& guarded, const T106& d_guarded, const T107& q00, const T108& d_q00, const T109& q01, const T110& d_q01, const T111& q02, const T112& d_q02, const T113& q10, const T114& d_q10, const T115& q11, const T116& d_q11, const T117& q12, const T118& d_q12, const T119& q20, const T120& d_q20, const T121& q21, const T122& d_q21, const T123& q22, const T124& d_q22, const T125& def_00, const T126& dif_00, const T127& def_01, const T128& dif_01, const T129& def_02, const T130& dif_02, const T131& def_10, const T132& dif_10, const T133& def_11, const T134& dif_11, const T135& def_12, const T136& dif_12, const T137& def_20, const T138& dif_20, const T139& def_21, const T140& dif_21, const T141& def_22, const T142& dif_22, const T143& narrow) {
    using R = decltype(_three_pool_step_adjoint_jvp_in_precision<double>(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, attenuation, d_attenuation, bar_e00, d_bar_e00, bar_e01, d_bar_e01, bar_e02, d_bar_e02, bar_e10, d_bar_e10, bar_e11, d_bar_e11, bar_e12, d_bar_e12, bar_e20, d_bar_e20, bar_e21, d_bar_e21, bar_e22, d_bar_e22, bar_grow_free, d_bar_grow_free, bar_grow_pool_b, d_bar_grow_pool_b, bar_grow_bound, d_bar_grow_bound, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, a00, d_a00, a01, d_a01, a02, d_a02, a10, d_a10, a11, d_a11, a20, d_a20, a22, d_a22, s00, d_s00, s11, d_s11, s22, d_s22, minors, d_minors, sum_flat, sum_linear, sum_square, d_sum_flat, d_sum_linear, d_sum_square, lift, d_lift, low, middle, d_low, d_middle, leading, d_leading, first, d_first, second, d_second, determinant, d_determinant, high, d_high, radius, d_radius, cube, raw, d_raw, argument, inside_limit, angle, d_angle, centre, d_centre, trailing, d_trailing, guarded, d_guarded, q00, d_q00, q01, d_q01, q02, d_q02, q10, d_q10, q11, d_q11, q12, d_q12, q20, d_q20, q21, d_q21, q22, d_q22, def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, narrow));
    if (bsk::truth(narrow)) {
        return bsk::convert<R>(_three_pool_step_adjoint_jvp_in_precision<float>(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, attenuation, d_attenuation, bar_e00, d_bar_e00, bar_e01, d_bar_e01, bar_e02, d_bar_e02, bar_e10, d_bar_e10, bar_e11, d_bar_e11, bar_e12, d_bar_e12, bar_e20, d_bar_e20, bar_e21, d_bar_e21, bar_e22, d_bar_e22, bar_grow_free, d_bar_grow_free, bar_grow_pool_b, d_bar_grow_pool_b, bar_grow_bound, d_bar_grow_bound, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, a00, d_a00, a01, d_a01, a02, d_a02, a10, d_a10, a11, d_a11, a20, d_a20, a22, d_a22, s00, d_s00, s11, d_s11, s22, d_s22, minors, d_minors, sum_flat, sum_linear, sum_square, d_sum_flat, d_sum_linear, d_sum_square, lift, d_lift, low, middle, d_low, d_middle, leading, d_leading, first, d_first, second, d_second, determinant, d_determinant, high, d_high, radius, d_radius, cube, raw, d_raw, argument, inside_limit, angle, d_angle, centre, d_centre, trailing, d_trailing, guarded, d_guarded, q00, d_q00, q01, d_q01, q02, d_q02, q10, d_q10, q11, d_q11, q12, d_q12, q20, d_q20, q21, d_q21, q22, d_q22, def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, narrow));
    }
    return _three_pool_step_adjoint_jvp_in_precision<double>(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, attenuation, d_attenuation, bar_e00, d_bar_e00, bar_e01, d_bar_e01, bar_e02, d_bar_e02, bar_e10, d_bar_e10, bar_e11, d_bar_e11, bar_e12, d_bar_e12, bar_e20, d_bar_e20, bar_e21, d_bar_e21, bar_e22, d_bar_e22, bar_grow_free, d_bar_grow_free, bar_grow_pool_b, d_bar_grow_pool_b, bar_grow_bound, d_bar_grow_bound, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, a00, d_a00, a01, d_a01, a02, d_a02, a10, d_a10, a11, d_a11, a20, d_a20, a22, d_a22, s00, d_s00, s11, d_s11, s22, d_s22, minors, d_minors, sum_flat, sum_linear, sum_square, d_sum_flat, d_sum_linear, d_sum_square, lift, d_lift, low, middle, d_low, d_middle, leading, d_leading, first, d_first, second, d_second, determinant, d_determinant, high, d_high, radius, d_radius, cube, raw, d_raw, argument, inside_limit, angle, d_angle, centre, d_centre, trailing, d_trailing, guarded, d_guarded, q00, d_q00, q01, d_q01, q02, d_q02, q10, d_q10, q11, d_q11, q12, d_q12, q20, d_q20, q21, d_q21, q22, d_q22, def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, narrow);
}

// An interval's step, from the bare operator and what survives it.
//
// The recovery is ``(I - E) m0`` rather than a solve, which is what the
// equilibrium being a fixed point of the generator buys.
template <class Work, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24, class T25, class T26>
BSK_HD auto _three_pool_weigh_jvp_in_precision(const T0& def_00, const T1& dif_00, const T2& def_01, const T3& dif_01, const T4& def_02, const T5& dif_02, const T6& def_10, const T7& dif_10, const T8& def_11, const T9& dif_11, const T10& def_12, const T11& dif_12, const T12& def_20, const T13& dif_20, const T14& def_21, const T15& dif_21, const T16& def_22, const T17& dif_22, const T18& free, const T19& d_free, const T20& pool_b, const T21& d_pool_b, const T22& pool_c, const T23& d_pool_c, const T24& attenuation, const T25& d_attenuation, const T26& narrow) {
    auto damp = bsk::cast<Work>(attenuation);
    auto d_damp = bsk::cast<Work>(d_attenuation);
    auto w00 = (damp * def_00);
    auto dw00 = ((d_damp * def_00) + (damp * dif_00));
    auto w01 = (damp * def_01);
    auto dw01 = ((d_damp * def_01) + (damp * dif_01));
    auto w02 = (damp * def_02);
    auto dw02 = ((d_damp * def_02) + (damp * dif_02));
    auto w10 = (damp * def_10);
    auto dw10 = ((d_damp * def_10) + (damp * dif_10));
    auto w11 = (damp * def_11);
    auto dw11 = ((d_damp * def_11) + (damp * dif_11));
    auto w12 = (damp * def_12);
    auto dw12 = ((d_damp * def_12) + (damp * dif_12));
    auto w20 = (damp * def_20);
    auto dw20 = ((d_damp * def_20) + (damp * dif_20));
    auto w21 = (damp * def_21);
    auto dw21 = ((d_damp * def_21) + (damp * dif_21));
    auto w22 = (damp * def_22);
    auto dw22 = ((d_damp * def_22) + (damp * dif_22));
    auto grow_free = (free - (((w00 * free) + (w01 * pool_b)) + (w02 * pool_c)));
    auto d_grow_free = (d_free - ((((((dw00 * free) + (w00 * d_free)) + (dw01 * pool_b)) + (w01 * d_pool_b)) + (dw02 * pool_c)) + (w02 * d_pool_c)));
    auto grow_pool_b = (pool_b - (((w10 * free) + (w11 * pool_b)) + (w12 * pool_c)));
    auto d_grow_pool_b = (d_pool_b - ((((((dw10 * free) + (w10 * d_free)) + (dw11 * pool_b)) + (w11 * d_pool_b)) + (dw12 * pool_c)) + (w12 * d_pool_c)));
    auto grow_bound = (pool_c - (((w20 * free) + (w21 * pool_b)) + (w22 * pool_c)));
    auto d_grow_bound = (d_pool_c - ((((((dw20 * free) + (w20 * d_free)) + (dw21 * pool_b)) + (w21 * d_pool_b)) + (dw22 * pool_c)) + (w22 * d_pool_c)));
    return bsk::make_tup(w00, w01, w02, w10, w11, w12, w20, w21, w22, grow_free, grow_pool_b, grow_bound, dw00, dw01, dw02, dw10, dw11, dw12, dw20, dw21, dw22, d_grow_free, d_grow_pool_b, d_grow_bound);
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24, class T25, class T26>
BSK_HD auto _three_pool_weigh_jvp(const T0& def_00, const T1& dif_00, const T2& def_01, const T3& dif_01, const T4& def_02, const T5& dif_02, const T6& def_10, const T7& dif_10, const T8& def_11, const T9& dif_11, const T10& def_12, const T11& dif_12, const T12& def_20, const T13& dif_20, const T14& def_21, const T15& dif_21, const T16& def_22, const T17& dif_22, const T18& free, const T19& d_free, const T20& pool_b, const T21& d_pool_b, const T22& pool_c, const T23& d_pool_c, const T24& attenuation, const T25& d_attenuation, const T26& narrow) {
    using R = decltype(_three_pool_weigh_jvp_in_precision<double>(def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, attenuation, d_attenuation, narrow));
    if (bsk::truth(narrow)) {
        return bsk::convert<R>(_three_pool_weigh_jvp_in_precision<float>(def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, attenuation, d_attenuation, narrow));
    }
    return _three_pool_weigh_jvp_in_precision<double>(def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, attenuation, d_attenuation, narrow);
}

// The three-pool longitudinal step and its directional derivative.
//
// The same closed form :func:`_three_pool_step` evaluates, carried
// alongside a tangent and in the same double precision. Returns the
// nine entries and three recoveries, then their twelve tangents.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18>
BSK_HD auto _three_pool_step_jvp(const T0& r1_free, const T1& d_r1_free, const T2& r1_pool_b, const T3& d_r1_pool_b, const T4& r1_bound, const T5& d_r1_bound, const T6& exchange_b, const T7& d_exchange_b, const T8& exchange_c, const T9& d_exchange_c, const T10& fraction_b, const T11& d_fraction_b, const T12& fraction_c, const T13& d_fraction_c, const T14& dt, const T15& d_dt, const T16& attenuation, const T17& d_attenuation, const T18& narrow) {
    auto t0_ = _three_pool_pieces_jvp(r1_free, d_r1_free, r1_pool_b, d_r1_pool_b, r1_bound, d_r1_bound, exchange_b, d_exchange_b, exchange_c, d_exchange_c, fraction_b, d_fraction_b, fraction_c, d_fraction_c, dt, d_dt, narrow);
    auto free = bsk::get<0>(t0_);
    auto d_free = bsk::get<1>(t0_);
    auto pool_b = bsk::get<2>(t0_);
    auto d_pool_b = bsk::get<3>(t0_);
    auto pool_c = bsk::get<4>(t0_);
    auto d_pool_c = bsk::get<5>(t0_);
    auto a00 = bsk::get<6>(t0_);
    auto d_a00 = bsk::get<7>(t0_);
    auto a01 = bsk::get<8>(t0_);
    auto d_a01 = bsk::get<9>(t0_);
    auto a02 = bsk::get<10>(t0_);
    auto d_a02 = bsk::get<11>(t0_);
    auto a10 = bsk::get<12>(t0_);
    auto d_a10 = bsk::get<13>(t0_);
    auto a11 = bsk::get<14>(t0_);
    auto d_a11 = bsk::get<15>(t0_);
    auto a20 = bsk::get<16>(t0_);
    auto d_a20 = bsk::get<17>(t0_);
    auto a22 = bsk::get<18>(t0_);
    auto d_a22 = bsk::get<19>(t0_);
    auto s00 = bsk::get<20>(t0_);
    auto d_s00 = bsk::get<21>(t0_);
    auto s11 = bsk::get<22>(t0_);
    auto d_s11 = bsk::get<23>(t0_);
    auto s22 = bsk::get<24>(t0_);
    auto d_s22 = bsk::get<25>(t0_);
    auto minors = bsk::get<26>(t0_);
    auto d_minors = bsk::get<27>(t0_);
    auto sum_flat = bsk::get<28>(t0_);
    auto sum_linear = bsk::get<29>(t0_);
    auto sum_square = bsk::get<30>(t0_);
    auto d_sum_flat = bsk::get<31>(t0_);
    auto d_sum_linear = bsk::get<32>(t0_);
    auto d_sum_square = bsk::get<33>(t0_);
    auto lift = bsk::get<34>(t0_);
    auto d_lift = bsk::get<35>(t0_);
    auto low = bsk::get<36>(t0_);
    auto middle = bsk::get<37>(t0_);
    auto d_low = bsk::get<38>(t0_);
    auto d_middle = bsk::get<39>(t0_);
    auto leading = bsk::get<40>(t0_);
    auto d_leading = bsk::get<41>(t0_);
    auto first = bsk::get<42>(t0_);
    auto d_first = bsk::get<43>(t0_);
    auto second = bsk::get<44>(t0_);
    auto d_second = bsk::get<45>(t0_);
    auto determinant = bsk::get<46>(t0_);
    auto d_determinant = bsk::get<47>(t0_);
    auto high = bsk::get<48>(t0_);
    auto d_high = bsk::get<49>(t0_);
    auto radius = bsk::get<50>(t0_);
    auto d_radius = bsk::get<51>(t0_);
    auto cube = bsk::get<52>(t0_);
    auto raw = bsk::get<53>(t0_);
    auto d_raw = bsk::get<54>(t0_);
    auto argument = bsk::get<55>(t0_);
    auto inside_limit = bsk::get<56>(t0_);
    auto angle = bsk::get<57>(t0_);
    auto d_angle = bsk::get<58>(t0_);
    auto centre = bsk::get<59>(t0_);
    auto d_centre = bsk::get<60>(t0_);
    auto trailing = bsk::get<61>(t0_);
    auto d_trailing = bsk::get<62>(t0_);
    auto guarded = bsk::get<63>(t0_);
    auto d_guarded = bsk::get<64>(t0_);
    auto q00 = bsk::get<65>(t0_);
    auto d_q00 = bsk::get<66>(t0_);
    auto q01 = bsk::get<67>(t0_);
    auto d_q01 = bsk::get<68>(t0_);
    auto q02 = bsk::get<69>(t0_);
    auto d_q02 = bsk::get<70>(t0_);
    auto q10 = bsk::get<71>(t0_);
    auto d_q10 = bsk::get<72>(t0_);
    auto q11 = bsk::get<73>(t0_);
    auto d_q11 = bsk::get<74>(t0_);
    auto q12 = bsk::get<75>(t0_);
    auto d_q12 = bsk::get<76>(t0_);
    auto q20 = bsk::get<77>(t0_);
    auto d_q20 = bsk::get<78>(t0_);
    auto q21 = bsk::get<79>(t0_);
    auto d_q21 = bsk::get<80>(t0_);
    auto q22 = bsk::get<81>(t0_);
    auto d_q22 = bsk::get<82>(t0_);
    auto t1_ = _three_pool_assemble_jvp(free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, a00, d_a00, a01, d_a01, a02, d_a02, a10, d_a10, a11, d_a11, a20, d_a20, a22, d_a22, s00, d_s00, s11, d_s11, s22, d_s22, minors, d_minors, sum_flat, sum_linear, sum_square, d_sum_flat, d_sum_linear, d_sum_square, lift, d_lift, low, middle, d_low, d_middle, leading, d_leading, first, d_first, second, d_second, determinant, d_determinant, high, d_high, radius, d_radius, cube, raw, d_raw, argument, inside_limit, angle, d_angle, centre, d_centre, trailing, d_trailing, guarded, d_guarded, q00, d_q00, q01, d_q01, q02, d_q02, q10, d_q10, q11, d_q11, q12, d_q12, q20, d_q20, q21, d_q21, q22, d_q22, narrow);
    auto def_00 = bsk::get<0>(t1_);
    auto dif_00 = bsk::get<1>(t1_);
    auto def_01 = bsk::get<2>(t1_);
    auto dif_01 = bsk::get<3>(t1_);
    auto def_02 = bsk::get<4>(t1_);
    auto dif_02 = bsk::get<5>(t1_);
    auto def_10 = bsk::get<6>(t1_);
    auto dif_10 = bsk::get<7>(t1_);
    auto def_11 = bsk::get<8>(t1_);
    auto dif_11 = bsk::get<9>(t1_);
    auto def_12 = bsk::get<10>(t1_);
    auto dif_12 = bsk::get<11>(t1_);
    auto def_20 = bsk::get<12>(t1_);
    auto dif_20 = bsk::get<13>(t1_);
    auto def_21 = bsk::get<14>(t1_);
    auto dif_21 = bsk::get<15>(t1_);
    auto def_22 = bsk::get<16>(t1_);
    auto dif_22 = bsk::get<17>(t1_);
    auto t2_ = _three_pool_weigh_jvp(def_00, dif_00, def_01, dif_01, def_02, dif_02, def_10, dif_10, def_11, dif_11, def_12, dif_12, def_20, dif_20, def_21, dif_21, def_22, dif_22, free, d_free, pool_b, d_pool_b, pool_c, d_pool_c, attenuation, d_attenuation, narrow);
    auto w00 = bsk::get<0>(t2_);
    auto w01 = bsk::get<1>(t2_);
    auto w02 = bsk::get<2>(t2_);
    auto w10 = bsk::get<3>(t2_);
    auto w11 = bsk::get<4>(t2_);
    auto w12 = bsk::get<5>(t2_);
    auto w20 = bsk::get<6>(t2_);
    auto w21 = bsk::get<7>(t2_);
    auto w22 = bsk::get<8>(t2_);
    auto grow_free = bsk::get<9>(t2_);
    auto grow_pool_b = bsk::get<10>(t2_);
    auto grow_bound = bsk::get<11>(t2_);
    auto dw00 = bsk::get<12>(t2_);
    auto dw01 = bsk::get<13>(t2_);
    auto dw02 = bsk::get<14>(t2_);
    auto dw10 = bsk::get<15>(t2_);
    auto dw11 = bsk::get<16>(t2_);
    auto dw12 = bsk::get<17>(t2_);
    auto dw20 = bsk::get<18>(t2_);
    auto dw21 = bsk::get<19>(t2_);
    auto dw22 = bsk::get<20>(t2_);
    auto d_grow_free = bsk::get<21>(t2_);
    auto d_grow_pool_b = bsk::get<22>(t2_);
    auto d_grow_bound = bsk::get<23>(t2_);
    return bsk::make_tup(bsk::cast<float>(w00), bsk::cast<float>(w01), bsk::cast<float>(w02), bsk::cast<float>(w10), bsk::cast<float>(w11), bsk::cast<float>(w12), bsk::cast<float>(w20), bsk::cast<float>(w21), bsk::cast<float>(w22), bsk::cast<float>(grow_free), bsk::cast<float>(grow_pool_b), bsk::cast<float>(grow_bound), bsk::cast<float>(dw00), bsk::cast<float>(dw01), bsk::cast<float>(dw02), bsk::cast<float>(dw10), bsk::cast<float>(dw11), bsk::cast<float>(dw12), bsk::cast<float>(dw20), bsk::cast<float>(dw21), bsk::cast<float>(dw22), bsk::cast<float>(d_grow_free), bsk::cast<float>(d_grow_pool_b), bsk::cast<float>(d_grow_bound));
}

// The reverse sweep of :func:`_two_pool_step`, carried on a direction.
//
// Recomputes the forward rather than carrying it across the event: the whole
// thing is a handful of transcendentals once per interval, against a state
// loop that runs per dephasing order.
//
// Where the discriminant is small the value is still formed from the two
// eigenvalues -- a sum, which loses nothing -- but the derivative is taken
// from the series, because ``d cosh(d)/d(d^2)`` reached through
// ``(e^{t+d} - e^{t-d})/2d`` is a cancellation divided by a small number.
//
// Returned as the gradients w.r.t. ``(r1_free, r1_bound, exchange, bound,
// dt, attenuation)`` then their six tangents.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23>
BSK_HD auto _two_pool_step_adjoint_jvp(const T0& r1_free, const T1& d_r1_free, const T2& r1_bound, const T3& d_r1_bound, const T4& exchange, const T5& d_exchange, const T6& bound, const T7& d_bound, const T8& dt, const T9& d_dt, const T10& attenuation, const T11& d_attenuation, const T12& bar_e11, const T13& d_bar_e11, const T14& bar_e12, const T15& d_bar_e12, const T16& bar_e21, const T17& d_bar_e21, const T18& bar_e22, const T19& d_bar_e22, const T20& bar_free, const T21& d_bar_free, const T22& bar_bound, const T23& d_bar_bound) {
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> back_bound{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> back_free{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> bar_half_gap{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> bar_l12{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> bar_l21{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> d_back_bound{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> d_back_free{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> d_bar_half_gap{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> d_bar_l12{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 2)> d_bar_l21{};
    auto free = (1.0f - bound);
    auto d_free = (-d_bound);
    auto kab = (exchange * bound);
    auto d_kab = ((d_exchange * bound) + (exchange * d_bound));
    auto kba = (exchange * free);
    auto d_kba = ((d_exchange * free) + (exchange * d_free));
    auto l11 = (((-kab) - r1_free) * dt);
    auto d_l11 = ((((-d_kab) - d_r1_free) * dt) + (((-kab) - r1_free) * d_dt));
    auto l12 = (kba * dt);
    auto d_l12 = ((d_kba * dt) + (kba * d_dt));
    auto l21 = (kab * dt);
    auto d_l21 = ((d_kab * dt) + (kab * d_dt));
    auto l22 = (((-kba) - r1_bound) * dt);
    auto d_l22 = ((((-d_kba) - d_r1_bound) * dt) + (((-kba) - r1_bound) * d_dt));
    auto half_trace = (0.5f * (l11 + l22));
    auto d_half_trace = (0.5f * (d_l11 + d_l22));
    auto half_gap = (0.5f * (l11 - l22));
    auto d_half_gap = (0.5f * (d_l11 - d_l22));
    auto square = ((half_gap * half_gap) + (l12 * l21));
    auto d_square = ((((2.0f * half_gap) * d_half_gap) + (d_l12 * l21)) + (l12 * d_l21));
    auto turning = (square > 1e-12f);
    auto root = bsk::sqrt(bsk::maximum(square, 0.0f));
    auto guarded = bsk::where(turning, root, 1.0f);
    auto d_root = bsk::where(turning, bsk::truediv((0.5f * d_square), guarded), 0.0f);
    auto upper = bsk::exp((half_trace + root));
    auto d_upper = (upper * (d_half_trace + d_root));
    auto lower = bsk::exp((half_trace - root));
    auto d_lower = (lower * (d_half_trace - d_root));
    auto cosine = (0.5f * (upper + lower));
    auto d_cosine = (0.5f * (d_upper + d_lower));
    auto plain = bsk::exp(half_trace);
    auto d_plain = (plain * d_half_trace);
    auto poly = ((1.0f + bsk::truediv(square, 6.0f)) + bsk::truediv((square * square), 120.0f));
    auto d_poly = (bsk::truediv(d_square, 6.0f) + bsk::truediv((square * d_square), 60.0f));
    auto scale = bsk::where(turning, bsk::truediv((0.5f * (upper - lower)), guarded), (plain * poly));
    auto d_scale = bsk::where(turning, (bsk::truediv((0.5f * (d_upper - d_lower)), guarded) - bsk::truediv(((0.5f * (upper - lower)) * d_root), (guarded * guarded))), ((d_plain * poly) + (plain * d_poly)));
    auto bare11 = (cosine + (scale * half_gap));
    auto d_bare11 = ((d_cosine + (d_scale * half_gap)) + (scale * d_half_gap));
    auto bare12 = (scale * l12);
    auto d_bare12 = ((d_scale * l12) + (scale * d_l12));
    auto bare21 = (scale * l21);
    auto d_bare21 = ((d_scale * l21) + (scale * d_l21));
    auto bare22 = (cosine - (scale * half_gap));
    auto d_bare22 = ((d_cosine - (d_scale * half_gap)) - (scale * d_half_gap));
    // The recovery reaches the operator's four entries and the two fractions.
    auto carried11 = (bar_e11 - (bar_free * free));
    auto d_carried11 = (d_bar_e11 - ((d_bar_free * free) + (bar_free * d_free)));
    auto carried12 = (bar_e12 - (bar_free * bound));
    auto d_carried12 = (d_bar_e12 - ((d_bar_free * bound) + (bar_free * d_bound)));
    auto carried21 = (bar_e21 - (bar_bound * free));
    auto d_carried21 = (d_bar_e21 - ((d_bar_bound * free) + (bar_bound * d_free)));
    auto carried22 = (bar_e22 - (bar_bound * bound));
    auto d_carried22 = (d_bar_e22 - ((d_bar_bound * bound) + (bar_bound * d_bound)));
    auto e11 = (attenuation * bare11);
    auto d_e11 = ((d_attenuation * bare11) + (attenuation * d_bare11));
    auto e12 = (attenuation * bare12);
    auto d_e12 = ((d_attenuation * bare12) + (attenuation * d_bare12));
    auto e21 = (attenuation * bare21);
    auto d_e21 = ((d_attenuation * bare21) + (attenuation * d_bare21));
    auto e22 = (attenuation * bare22);
    auto d_e22 = ((d_attenuation * bare22) + (attenuation * d_bare22));
    back_free = ((bar_free * (1.0f - e11)) - (bar_bound * e21));
    d_back_free = (((d_bar_free * (1.0f - e11)) - (bar_free * d_e11)) - ((d_bar_bound * e21) + (bar_bound * d_e21)));
    back_bound = ((bar_bound * (1.0f - e22)) - (bar_free * e12));
    d_back_bound = (((d_bar_bound * (1.0f - e22)) - (bar_bound * d_e22)) - ((d_bar_free * e12) + (bar_free * d_e12)));
    auto back_attenuation = ((((carried11 * bare11) + (carried12 * bare12)) + (carried21 * bare21)) + (carried22 * bare22));
    auto d_back_attenuation = ((((((((d_carried11 * bare11) + (carried11 * d_bare11)) + (d_carried12 * bare12)) + (carried12 * d_bare12)) + (d_carried21 * bare21)) + (carried21 * d_bare21)) + (d_carried22 * bare22)) + (carried22 * d_bare22));
    auto scaled11 = (attenuation * carried11);
    auto d_scaled11 = ((d_attenuation * carried11) + (attenuation * d_carried11));
    auto scaled12 = (attenuation * carried12);
    auto d_scaled12 = ((d_attenuation * carried12) + (attenuation * d_carried12));
    auto scaled21 = (attenuation * carried21);
    auto d_scaled21 = ((d_attenuation * carried21) + (attenuation * d_carried21));
    auto scaled22 = (attenuation * carried22);
    auto d_scaled22 = ((d_attenuation * carried22) + (attenuation * d_carried22));
    auto bar_cosine = (scaled11 + scaled22);
    auto d_bar_cosine = (d_scaled11 + d_scaled22);
    auto gap = (scaled11 - scaled22);
    auto d_gap = (d_scaled11 - d_scaled22);
    auto bar_scale = (((gap * half_gap) + (scaled12 * l12)) + (scaled21 * l21));
    auto d_bar_scale = ((((((d_gap * half_gap) + (gap * d_half_gap)) + (d_scaled12 * l12)) + (scaled12 * d_l12)) + (d_scaled21 * l21)) + (scaled21 * d_l21));
    bar_half_gap = (scale * gap);
    d_bar_half_gap = ((d_scale * gap) + (scale * d_gap));
    bar_l12 = (scale * scaled12);
    d_bar_l12 = ((d_scale * scaled12) + (scale * d_scaled12));
    bar_l21 = (scale * scaled21);
    d_bar_l21 = ((d_scale * scaled21) + (scale * d_scaled21));
    auto series_trace = ((bar_cosine * cosine) + (bar_scale * scale));
    auto d_series_trace = ((((d_bar_cosine * cosine) + (bar_cosine * d_cosine)) + (d_bar_scale * scale)) + (bar_scale * d_scale));
    auto cosine_poly = (0.5f + bsk::truediv(square, 12.0f));
    auto d_cosine_poly = bsk::truediv(d_square, 12.0f);
    auto scale_poly = (0.16666666666666666f + bsk::truediv(square, 60.0f));
    auto d_scale_poly = bsk::truediv(d_square, 60.0f);
    auto series_square = (plain * ((bar_cosine * cosine_poly) + (bar_scale * scale_poly)));
    auto d_series_square = ((d_plain * ((bar_cosine * cosine_poly) + (bar_scale * scale_poly))) + (plain * ((((d_bar_cosine * cosine_poly) + (bar_cosine * d_cosine_poly)) + (d_bar_scale * scale_poly)) + (bar_scale * d_scale_poly))));
    auto inverse = bsk::where(turning, bsk::truediv(1.0f, guarded), 0.0f);
    auto d_inverse = bsk::where(turning, bsk::truediv((-d_root), (guarded * guarded)), 0.0f);
    auto bar_upper = (0.5f * (bar_cosine + (bar_scale * inverse)));
    auto d_bar_upper = (0.5f * ((d_bar_cosine + (d_bar_scale * inverse)) + (bar_scale * d_inverse)));
    auto bar_lower = (0.5f * (bar_cosine - (bar_scale * inverse)));
    auto d_bar_lower = (0.5f * ((d_bar_cosine - (d_bar_scale * inverse)) - (bar_scale * d_inverse)));
    auto root_trace = ((bar_upper * upper) + (bar_lower * lower));
    auto d_root_trace = ((((d_bar_upper * upper) + (bar_upper * d_upper)) + (d_bar_lower * lower)) + (bar_lower * d_lower));
    auto bar_root = (((bar_upper * upper) - (bar_lower * lower)) - ((bar_scale * scale) * inverse));
    auto d_bar_root = (((((d_bar_upper * upper) + (bar_upper * d_upper)) - (d_bar_lower * lower)) - (bar_lower * d_lower)) - ((((d_bar_scale * scale) * inverse) + ((bar_scale * d_scale) * inverse)) + ((bar_scale * scale) * d_inverse)));
    auto root_square = ((0.5f * bar_root) * inverse);
    auto d_root_square = (0.5f * ((d_bar_root * inverse) + (bar_root * d_inverse)));
    auto bar_half_trace = bsk::where(turning, root_trace, series_trace);
    auto d_bar_half_trace = bsk::where(turning, d_root_trace, d_series_trace);
    auto bar_square = bsk::where(turning, root_square, series_square);
    auto d_bar_square = bsk::where(turning, d_root_square, d_series_square);
    bar_half_gap = (bar_half_gap + ((2.0f * bar_square) * half_gap));
    d_bar_half_gap = (d_bar_half_gap + (2.0f * ((d_bar_square * half_gap) + (bar_square * d_half_gap))));
    bar_l12 = (bar_l12 + (bar_square * l21));
    d_bar_l12 = (d_bar_l12 + ((d_bar_square * l21) + (bar_square * d_l21)));
    bar_l21 = (bar_l21 + (bar_square * l12));
    d_bar_l21 = (d_bar_l21 + ((d_bar_square * l12) + (bar_square * d_l12)));
    auto bar_l11 = (0.5f * (bar_half_trace + bar_half_gap));
    auto d_bar_l11 = (0.5f * (d_bar_half_trace + d_bar_half_gap));
    auto bar_l22 = (0.5f * (bar_half_trace - bar_half_gap));
    auto d_bar_l22 = (0.5f * (d_bar_half_trace - d_bar_half_gap));
    auto bar_kab = ((bar_l21 - bar_l11) * dt);
    auto d_bar_kab = (((d_bar_l21 - d_bar_l11) * dt) + ((bar_l21 - bar_l11) * d_dt));
    auto bar_kba = ((bar_l12 - bar_l22) * dt);
    auto d_bar_kba = (((d_bar_l12 - d_bar_l22) * dt) + ((bar_l12 - bar_l22) * d_dt));
    auto back_dt = ((((bar_l11 * ((-kab) - r1_free)) + (bar_l12 * kba)) + (bar_l21 * kab)) + (bar_l22 * ((-kba) - r1_bound)));
    auto d_back_dt = ((((((((d_bar_l11 * ((-kab) - r1_free)) + (bar_l11 * ((-d_kab) - d_r1_free))) + (d_bar_l12 * kba)) + (bar_l12 * d_kba)) + (d_bar_l21 * kab)) + (bar_l21 * d_kab)) + (d_bar_l22 * ((-kba) - r1_bound))) + (bar_l22 * ((-d_kba) - d_r1_bound)));
    back_bound = (back_bound + (bar_kab * exchange));
    d_back_bound = (d_back_bound + ((d_bar_kab * exchange) + (bar_kab * d_exchange)));
    back_free = (back_free + (bar_kba * exchange));
    d_back_free = (d_back_free + ((d_bar_kba * exchange) + (bar_kba * d_exchange)));
    return bsk::make_tup(((-bar_l11) * dt), ((-bar_l22) * dt), ((bar_kab * bound) + (bar_kba * free)), (back_bound - back_free), back_dt, back_attenuation, (-((d_bar_l11 * dt) + (bar_l11 * d_dt))), (-((d_bar_l22 * dt) + (bar_l22 * d_dt))), ((((d_bar_kab * bound) + (bar_kab * d_bound)) + (d_bar_kba * free)) + (bar_kba * d_free)), (d_back_bound - d_back_free), d_back_dt, d_back_attenuation);
}

// The two-pool operator and its directional derivative.
//
// The same closed form :func:`_two_pool_step` evaluates, carried alongside a
// tangent. Returned as the six outputs then their six tangents.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
BSK_HD auto _two_pool_step_jvp(const T0& r1_free, const T1& d_r1_free, const T2& r1_bound, const T3& d_r1_bound, const T4& exchange, const T5& d_exchange, const T6& bound, const T7& d_bound, const T8& dt, const T9& d_dt, const T10& attenuation, const T11& d_attenuation) {
    auto free = (1.0f - bound);
    auto d_free = (-d_bound);
    auto kab = (exchange * bound);
    auto d_kab = ((d_exchange * bound) + (exchange * d_bound));
    auto kba = (exchange * free);
    auto d_kba = ((d_exchange * free) + (exchange * d_free));
    auto l11 = (((-kab) - r1_free) * dt);
    auto d_l11 = ((((-d_kab) - d_r1_free) * dt) + (((-kab) - r1_free) * d_dt));
    auto l12 = (kba * dt);
    auto d_l12 = ((d_kba * dt) + (kba * d_dt));
    auto l21 = (kab * dt);
    auto d_l21 = ((d_kab * dt) + (kab * d_dt));
    auto l22 = (((-kba) - r1_bound) * dt);
    auto d_l22 = ((((-d_kba) - d_r1_bound) * dt) + (((-kba) - r1_bound) * d_dt));
    auto half_trace = (0.5f * (l11 + l22));
    auto d_half_trace = (0.5f * (d_l11 + d_l22));
    auto half_gap = (0.5f * (l11 - l22));
    auto d_half_gap = (0.5f * (d_l11 - d_l22));
    auto square = ((half_gap * half_gap) + (l12 * l21));
    auto d_square = ((((2.0f * half_gap) * d_half_gap) + (d_l12 * l21)) + (l12 * d_l21));
    auto turning = (square > 1e-12f);
    auto root = bsk::sqrt(bsk::maximum(square, 0.0f));
    auto guarded = bsk::where(turning, root, 1.0f);
    auto d_root = bsk::where(turning, bsk::truediv((0.5f * d_square), guarded), 0.0f);
    auto upper = bsk::exp((half_trace + root));
    auto d_upper = (upper * (d_half_trace + d_root));
    auto lower = bsk::exp((half_trace - root));
    auto d_lower = (lower * (d_half_trace - d_root));
    auto cosine = (0.5f * (upper + lower));
    auto d_cosine = (0.5f * (d_upper + d_lower));
    // sinh(d)/d by series where the root has no derivative of its own.
    auto plain = bsk::exp(half_trace);
    auto d_plain = (plain * d_half_trace);
    auto poly = ((1.0f + bsk::truediv(square, 6.0f)) + bsk::truediv((square * square), 120.0f));
    auto d_poly = (bsk::truediv(d_square, 6.0f) + bsk::truediv((square * d_square), 60.0f));
    auto scale = bsk::where(turning, bsk::truediv((0.5f * (upper - lower)), guarded), (plain * poly));
    auto d_scale = bsk::where(turning, (bsk::truediv((0.5f * (d_upper - d_lower)), guarded) - bsk::truediv(((0.5f * (upper - lower)) * d_root), (guarded * guarded))), ((d_plain * poly) + (plain * d_poly)));
    auto e11 = (attenuation * (cosine + (scale * half_gap)));
    auto d_e11 = ((d_attenuation * (cosine + (scale * half_gap))) + (attenuation * ((d_cosine + (d_scale * half_gap)) + (scale * d_half_gap))));
    auto e12 = ((attenuation * scale) * l12);
    auto d_e12 = ((((d_attenuation * scale) * l12) + ((attenuation * d_scale) * l12)) + ((attenuation * scale) * d_l12));
    auto e21 = ((attenuation * scale) * l21);
    auto d_e21 = ((((d_attenuation * scale) * l21) + ((attenuation * d_scale) * l21)) + ((attenuation * scale) * d_l21));
    auto e22 = (attenuation * (cosine - (scale * half_gap)));
    auto d_e22 = ((d_attenuation * (cosine - (scale * half_gap))) + (attenuation * ((d_cosine - (d_scale * half_gap)) - (scale * d_half_gap))));
    auto grow_free = (free - ((e11 * free) + (e12 * bound)));
    auto d_grow_free = (d_free - ((((d_e11 * free) + (e11 * d_free)) + (d_e12 * bound)) + (e12 * d_bound)));
    auto grow_bound = (bound - ((e21 * free) + (e22 * bound)));
    auto d_grow_bound = (d_bound - ((((d_e21 * free) + (e21 * d_free)) + (d_e22 * bound)) + (e22 * d_bound)));
    return bsk::make_tup(e11, e12, e21, e22, grow_free, grow_bound, d_e11, d_e12, d_e21, d_e22, d_grow_free, d_grow_bound);
}

// ``e^z`` for ``z`` carried as a pair of floats.
template <class T0, class T1>
BSK_HD auto _complex_exp(const T0& real, const T1& imag) {
    auto scale = bsk::exp(real);
    return bsk::make_tup((scale * bsk::cos(imag)), (scale * bsk::sin(imag)));
}

// ``e^z`` and its directional derivative, which is ``e^z`` times it.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _complex_exp_jvp(const T0& real, const T1& imag, const T2& d_real, const T3& d_imag) {
    auto t0_ = _complex_exp(real, imag);
    auto value_real = bsk::get<0>(t0_);
    auto value_imag = bsk::get<1>(t0_);
    return bsk::make_tup(value_real, value_imag, ((value_real * d_real) - (value_imag * d_imag)), ((value_real * d_imag) + (value_imag * d_real)));
}

// A square root of a complex number carried as a pair of floats.
//
// Which of the two it is does not matter here: the only thing that reads it
// is even in it, so the branch cut the principal root carries is unreachable.
template <class T0, class T1>
BSK_HD auto _complex_sqrt(const T0& real, const T1& imag) {
    auto magnitude = bsk::sqrt(((real * real) + (imag * imag)));
    auto root_real = bsk::sqrt(bsk::maximum((0.5f * (magnitude + real)), 0.0f));
    auto root_imag = bsk::sqrt(bsk::maximum((0.5f * (magnitude - real)), 0.0f));
    return bsk::make_tup(root_real, bsk::where((imag < 0.0f), (-root_imag), root_imag));
}

// A complex square root and its directional derivative.
//
// The derivative divides by twice the root, so a caller keeps the origin --
// where the root has none -- on its series branch.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _complex_sqrt_jvp(const T0& real, const T1& imag, const T2& d_real, const T3& d_imag) {
    auto t0_ = _complex_sqrt(real, imag);
    auto root_real = bsk::get<0>(t0_);
    auto root_imag = bsk::get<1>(t0_);
    auto guard = (2.0f * ((root_real * root_real) + (root_imag * root_imag)));
    auto live = (guard > 0.0f);
    auto guarded = bsk::where(live, guard, 1.0f);
    // dz / (2 w) == dz * conj(2 w) / |2 w|^2
    auto tangent_real = bsk::where(live, bsk::truediv(((d_real * root_real) + (d_imag * root_imag)), guarded), 0.0f);
    auto tangent_imag = bsk::where(live, bsk::truediv(((d_imag * root_real) - (d_real * root_imag)), guarded), 0.0f);
    return bsk::make_tup(root_real, root_imag, tangent_real, tangent_imag);
}

// ``1/z`` for a dual complex number, and the tangent that goes with it.
template <class T0>
BSK_HD auto _dual_reciprocal(const T0& z) {
    auto norm = ((bsk::get<0>(z) * bsk::get<0>(z)) + (bsk::get<1>(z) * bsk::get<1>(z)));
    auto guard = bsk::where((norm > 0.0f), norm, 1.0f);
    auto value_real = bsk::truediv(bsk::get<0>(z), guard);
    auto value_imag = bsk::truediv((-bsk::get<1>(z)), guard);
    auto t0_ = _complex_mul(value_real, value_imag, value_real, value_imag);
    auto square_real = bsk::get<0>(t0_);
    auto square_imag = bsk::get<1>(t0_);
    auto t1_ = _complex_mul(square_real, square_imag, bsk::get<2>(z), bsk::get<3>(z));
    auto tangent_real = bsk::get<0>(t1_);
    auto tangent_imag = bsk::get<1>(t1_);
    return bsk::make_tup(value_real, value_imag, (-tangent_real), (-tangent_imag));
}

// The reverse sweep of :func:`_two_pool_transverse_step_jvp`.
//
// Every step from the four generator entries to the four operator entries is
// holomorphic, so the sweep is the longitudinal one with complex numbers in
// place of real ones and no conjugates along the way. That holds because the
// cotangents arrive as row covectors -- ``bar_e`` is the number with ``dL =
// Re(bar_e de)`` -- and only where a complex intermediate meets one of the
// real inputs is a real part taken.
//
// Takes the four cotangents as dual complex quadruples and returns the seven
// real gradients, each as a value and a tangent.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19>
BSK_HD auto _two_pool_transverse_adjoint_jvp(const T0& r2_free, const T1& d_r2_free, const T2& r2_bound, const T3& d_r2_bound, const T4& exchange, const T5& d_exchange, const T6& bound, const T7& d_bound, const T8& free, const T9& d_free, const T10& shift_hz, const T11& d_shift_hz, const T12& dt, const T13& d_dt, const T14& attenuation, const T15& d_attenuation, const T16& bar_e11, const T17& bar_e12, const T18& bar_e21, const T19& bar_e22) {
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>> bar_half_gap{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>> bar_l12{};
    bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19> | 0, 2)>> bar_l21{};
    auto zero = (0.0f * dt);
    auto kab = (exchange * bound);
    auto d_kab = ((d_exchange * bound) + (exchange * d_bound));
    auto kba = (exchange * free);
    auto d_kba = ((d_exchange * free) + (exchange * d_free));
    auto turn = -6.283185307179586f;
    auto l11 = bsk::make_tup((((-kab) - r2_free) * dt), zero, ((((-d_kab) - d_r2_free) * dt) + (((-kab) - r2_free) * d_dt)), zero);
    auto l12 = bsk::make_tup((kba * dt), zero, ((d_kba * dt) + (kba * d_dt)), zero);
    auto l21 = bsk::make_tup((kab * dt), zero, ((d_kab * dt) + (kab * d_dt)), zero);
    auto l22 = bsk::make_tup((((-kba) - r2_bound) * dt), (turn * (shift_hz * dt)), ((((-d_kba) - d_r2_bound) * dt) + (((-kba) - r2_bound) * d_dt)), (turn * ((d_shift_hz * dt) + (shift_hz * d_dt))));
    auto half_trace = _dual_weigh(_dual_add(l11, l22), 0.5f);
    auto half_gap = _dual_weigh(_dual_subtract(l11, l22), 0.5f);
    auto square = _dual_add(_dual_product(half_gap, half_gap), _dual_product(l12, l21));
    auto delta = _complex_sqrt_jvp(bsk::get<0>(square), bsk::get<1>(square), bsk::get<2>(square), bsk::get<3>(square));
    auto upper = [&](const auto& s0_) { return _complex_exp_jvp(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(_dual_add(half_trace, delta));
    auto lower = [&](const auto& s0_) { return _complex_exp_jvp(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(_dual_subtract(half_trace, delta));
    auto plain = _complex_exp_jvp(bsk::get<0>(half_trace), bsk::get<1>(half_trace), bsk::get<2>(half_trace), bsk::get<3>(half_trace));
    auto cosine = _dual_weigh(_dual_add(upper, lower), 0.5f);
    auto turning = (((bsk::get<0>(square) * bsk::get<0>(square)) + (bsk::get<1>(square) * bsk::get<1>(square))) > 1e-24f);
    // Off the branch the reciprocal is taken at one instead, so a discriminant
    // at the origin never divides anything the series answer then discards.
    auto guarded = bsk::make_tup(bsk::where(turning, bsk::get<0>(delta), 1.0f), bsk::where(turning, bsk::get<1>(delta), 0.0f), bsk::where(turning, bsk::get<2>(delta), 0.0f), bsk::where(turning, bsk::get<3>(delta), 0.0f));
    auto inverse = _dual_reciprocal(guarded);
    auto divided = _dual_product(_dual_weigh(_dual_subtract(upper, lower), 0.5f), inverse);
    auto square2 = _dual_product(square, square);
    auto poly = bsk::make_tup(((1.0f + bsk::truediv(bsk::get<0>(square), 6.0f)) + bsk::truediv(bsk::get<0>(square2), 120.0f)), (bsk::truediv(bsk::get<1>(square), 6.0f) + bsk::truediv(bsk::get<1>(square2), 120.0f)), (bsk::truediv(bsk::get<2>(square), 6.0f) + bsk::truediv(bsk::get<2>(square2), 120.0f)), (bsk::truediv(bsk::get<3>(square), 6.0f) + bsk::truediv(bsk::get<3>(square2), 120.0f)));
    auto series = _dual_product(plain, poly);
    auto scale = bsk::make_tup(bsk::where(turning, bsk::get<0>(divided), bsk::get<0>(series)), bsk::where(turning, bsk::get<1>(divided), bsk::get<1>(series)), bsk::where(turning, bsk::get<2>(divided), bsk::get<2>(series)), bsk::where(turning, bsk::get<3>(divided), bsk::get<3>(series)));
    auto off = _dual_product(scale, half_gap);
    auto bare_11 = _dual_add(cosine, off);
    auto bare_12 = _dual_product(scale, l12);
    auto bare_21 = _dual_product(scale, l21);
    auto bare_22 = _dual_subtract(cosine, off);
    auto bar_attenuation = _dual_sum(_dual_product(bar_e11, bare_11), _dual_product(bar_e12, bare_12), _dual_product(bar_e21, bare_21), _dual_product(bar_e22, bare_22));
    auto scaled_11 = [&](const auto& s2_) { return _dual_scale(attenuation, d_attenuation, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_e11);
    auto scaled_12 = [&](const auto& s2_) { return _dual_scale(attenuation, d_attenuation, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_e12);
    auto scaled_21 = [&](const auto& s2_) { return _dual_scale(attenuation, d_attenuation, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_e21);
    auto scaled_22 = [&](const auto& s2_) { return _dual_scale(attenuation, d_attenuation, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_e22);
    auto diagonal = _dual_subtract(scaled_11, scaled_22);
    auto bar_cosine = _dual_add(scaled_11, scaled_22);
    auto bar_scale = _dual_add(_dual_product(diagonal, half_gap), _dual_add(_dual_product(scaled_12, l12), _dual_product(scaled_21, l21)));
    bar_half_gap = _dual_product(scale, diagonal);
    bar_l12 = _dual_product(scale, scaled_12);
    bar_l21 = _dual_product(scale, scaled_21);
    auto series_trace = _dual_add(_dual_product(bar_cosine, cosine), _dual_product(bar_scale, scale));
    auto series_square = _dual_product(plain, _dual_add(_dual_product(bar_cosine, bsk::make_tup((0.5f + bsk::truediv(bsk::get<0>(square), 12.0f)), bsk::truediv(bsk::get<1>(square), 12.0f), bsk::truediv(bsk::get<2>(square), 12.0f), bsk::truediv(bsk::get<3>(square), 12.0f))), _dual_product(bar_scale, bsk::make_tup((0.16666666666666666f + bsk::truediv(bsk::get<0>(square), 60.0f)), bsk::truediv(bsk::get<1>(square), 60.0f), bsk::truediv(bsk::get<2>(square), 60.0f), bsk::truediv(bsk::get<3>(square), 60.0f)))));
    auto bar_upper = _dual_weigh(_dual_add(bar_cosine, _dual_product(bar_scale, inverse)), 0.5f);
    auto bar_lower = _dual_weigh(_dual_subtract(bar_cosine, _dual_product(bar_scale, inverse)), 0.5f);
    auto split_trace = _dual_add(_dual_product(bar_upper, upper), _dual_product(bar_lower, lower));
    auto bar_delta = _dual_subtract(_dual_subtract(_dual_product(bar_upper, upper), _dual_product(bar_lower, lower)), _dual_product(_dual_product(bar_scale, scale), inverse));
    auto split_square = _dual_weigh(_dual_product(bar_delta, inverse), 0.5f);
    auto bar_half_trace = bsk::make_tup(bsk::where(turning, bsk::get<0>(split_trace), bsk::get<0>(series_trace)), bsk::where(turning, bsk::get<1>(split_trace), bsk::get<1>(series_trace)), bsk::where(turning, bsk::get<2>(split_trace), bsk::get<2>(series_trace)), bsk::where(turning, bsk::get<3>(split_trace), bsk::get<3>(series_trace)));
    auto bar_square = bsk::make_tup(bsk::where(turning, bsk::get<0>(split_square), bsk::get<0>(series_square)), bsk::where(turning, bsk::get<1>(split_square), bsk::get<1>(series_square)), bsk::where(turning, bsk::get<2>(split_square), bsk::get<2>(series_square)), bsk::where(turning, bsk::get<3>(split_square), bsk::get<3>(series_square)));
    bar_half_gap = _dual_add(bar_half_gap, _dual_weigh(_dual_product(bar_square, half_gap), 2.0f));
    bar_l12 = _dual_add(bar_l12, _dual_product(bar_square, l21));
    bar_l21 = _dual_add(bar_l21, _dual_product(bar_square, l12));
    auto bar_l11 = _dual_weigh(_dual_add(bar_half_trace, bar_half_gap), 0.5f);
    auto bar_l22 = _dual_weigh(_dual_subtract(bar_half_trace, bar_half_gap), 0.5f);
    auto bar_kab = [&](const auto& s2_) { return _dual_scale(dt, d_dt, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(_dual_subtract(bar_l21, bar_l11));
    auto bar_kba = [&](const auto& s2_) { return _dual_scale(dt, d_dt, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(_dual_subtract(bar_l12, bar_l22));
    auto slope_22 = bsk::make_tup(((-kba) - r2_bound), (turn * shift_hz), ((-d_kba) - d_r2_bound), (turn * d_shift_hz));
    auto bar_dt = _dual_sum([&](const auto& s2_) { return _dual_scale(((-kab) - r2_free), ((-d_kab) - d_r2_free), bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_l11), [&](const auto& s2_) { return _dual_scale(kba, d_kba, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_l12), [&](const auto& s2_) { return _dual_scale(kab, d_kab, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_l21), _dual_product(slope_22, bar_l22));
    auto r2_free_bar = [&](const auto& s2_) { return _dual_scale((-dt), (-d_dt), bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_l11);
    auto r2_bound_bar = [&](const auto& s2_) { return _dual_scale((-dt), (-d_dt), bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_l22);
    auto exchange_bar = _dual_add([&](const auto& s2_) { return _dual_scale(bound, d_bound, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_kab), [&](const auto& s2_) { return _dual_scale(free, d_free, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_kba));
    auto bound_bar = [&](const auto& s2_) { return _dual_scale(exchange, d_exchange, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_kab);
    auto free_bar = [&](const auto& s2_) { return _dual_scale(exchange, d_exchange, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(bar_kba);
    auto shift_bar = [&](const auto& s2_) { return _dual_scale((turn * dt), (turn * d_dt), bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }([&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(bar_l22));
    return bsk::make_tup(bsk::get<0>(r2_free_bar), bsk::get<2>(r2_free_bar), bsk::get<0>(r2_bound_bar), bsk::get<2>(r2_bound_bar), bsk::get<0>(exchange_bar), bsk::get<2>(exchange_bar), bsk::get<0>(bound_bar), bsk::get<2>(bound_bar), bsk::get<0>(free_bar), bsk::get<2>(free_bar), bsk::get<0>(shift_bar), bsk::get<2>(shift_bar), bsk::get<0>(bar_dt), bsk::get<2>(bar_dt), bsk::get<0>(bar_attenuation), bsk::get<2>(bar_attenuation));
}

// The transverse operator and its directional derivative.
//
// The same closed form :func:`_two_pool_transverse_step` evaluates, carried
// alongside a tangent. Returned as the four entries then their four tangents,
// each a pair of floats.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15>
BSK_HD auto _two_pool_transverse_step_jvp(const T0& r2_free, const T1& d_r2_free, const T2& r2_bound, const T3& d_r2_bound, const T4& exchange, const T5& d_exchange, const T6& bound, const T7& d_bound, const T8& free, const T9& d_free, const T10& shift_hz, const T11& d_shift_hz, const T12& dt, const T13& d_dt, const T14& attenuation, const T15& d_attenuation) {
    auto kab = (exchange * bound);
    auto d_kab = ((d_exchange * bound) + (exchange * d_bound));
    auto kba = (exchange * free);
    auto d_kba = ((d_exchange * free) + (exchange * d_free));
    auto l11 = (((-kab) - r2_free) * dt);
    auto d_l11 = ((((-d_kab) - d_r2_free) * dt) + (((-kab) - r2_free) * d_dt));
    auto l12 = (kba * dt);
    auto d_l12 = ((d_kba * dt) + (kba * d_dt));
    auto l21 = (kab * dt);
    auto d_l21 = ((d_kab * dt) + (kab * d_dt));
    auto l22 = (((-kba) - r2_bound) * dt);
    auto d_l22 = ((((-d_kba) - d_r2_bound) * dt) + (((-kba) - r2_bound) * d_dt));
    auto turn = -6.283185307179586f;
    auto l22_imag = ((turn * shift_hz) * dt);
    auto d_l22_imag = (turn * ((d_shift_hz * dt) + (shift_hz * d_dt)));
    auto trace_real = (0.5f * (l11 + l22));
    auto d_trace_real = (0.5f * (d_l11 + d_l22));
    auto trace_imag = (0.5f * l22_imag);
    auto d_trace_imag = (0.5f * d_l22_imag);
    auto gap_real = (0.5f * (l11 - l22));
    auto d_gap_real = (0.5f * (d_l11 - d_l22));
    auto gap_imag = (-0.5f * l22_imag);
    auto d_gap_imag = (-0.5f * d_l22_imag);
    auto square_real = (((gap_real * gap_real) - (gap_imag * gap_imag)) + (l12 * l21));
    auto d_square_real = (((((2.0f * gap_real) * d_gap_real) - ((2.0f * gap_imag) * d_gap_imag)) + (d_l12 * l21)) + (l12 * d_l21));
    auto square_imag = ((2.0f * gap_real) * gap_imag);
    auto d_square_imag = (2.0f * ((d_gap_real * gap_imag) + (gap_real * d_gap_imag)));
    auto t0_ = _complex_sqrt_jvp(square_real, square_imag, d_square_real, d_square_imag);
    auto root_real = bsk::get<0>(t0_);
    auto root_imag = bsk::get<1>(t0_);
    auto d_root_real = bsk::get<2>(t0_);
    auto d_root_imag = bsk::get<3>(t0_);
    auto t1_ = _complex_exp_jvp((trace_real + root_real), (trace_imag + root_imag), (d_trace_real + d_root_real), (d_trace_imag + d_root_imag));
    auto upper_real = bsk::get<0>(t1_);
    auto upper_imag = bsk::get<1>(t1_);
    auto d_upper_real = bsk::get<2>(t1_);
    auto d_upper_imag = bsk::get<3>(t1_);
    auto t2_ = _complex_exp_jvp((trace_real - root_real), (trace_imag - root_imag), (d_trace_real - d_root_real), (d_trace_imag - d_root_imag));
    auto lower_real = bsk::get<0>(t2_);
    auto lower_imag = bsk::get<1>(t2_);
    auto d_lower_real = bsk::get<2>(t2_);
    auto d_lower_imag = bsk::get<3>(t2_);
    auto cos_real = (0.5f * (upper_real + lower_real));
    auto cos_imag = (0.5f * (upper_imag + lower_imag));
    auto d_cos_real = (0.5f * (d_upper_real + d_lower_real));
    auto d_cos_imag = (0.5f * (d_upper_imag + d_lower_imag));
    auto turning = (((square_real * square_real) + (square_imag * square_imag)) > 1e-24f);
    auto half_real = (0.5f * (upper_real - lower_real));
    auto half_imag = (0.5f * (upper_imag - lower_imag));
    auto d_half_real = (0.5f * (d_upper_real - d_lower_real));
    auto d_half_imag = (0.5f * (d_upper_imag - d_lower_imag));
    auto norm = ((root_real * root_real) + (root_imag * root_imag));
    auto guard = bsk::where(turning, norm, 1.0f);
    auto d_norm = bsk::where(turning, (2.0f * ((root_real * d_root_real) + (root_imag * d_root_imag))), 0.0f);
    // (a / w) with w complex: a * conj(w) / |w|^2, differentiated as a quotient.
    auto top_real = ((half_real * root_real) + (half_imag * root_imag));
    auto top_imag = ((half_imag * root_real) - (half_real * root_imag));
    auto d_top_real = ((((d_half_real * root_real) + (half_real * d_root_real)) + (d_half_imag * root_imag)) + (half_imag * d_root_imag));
    auto d_top_imag = ((((d_half_imag * root_real) + (half_imag * d_root_real)) - (d_half_real * root_imag)) - (half_real * d_root_imag));
    auto divided_real = bsk::truediv(top_real, guard);
    auto divided_imag = bsk::truediv(top_imag, guard);
    auto d_divided_real = bsk::truediv((d_top_real - (divided_real * d_norm)), guard);
    auto d_divided_imag = bsk::truediv((d_top_imag - (divided_imag * d_norm)), guard);
    auto t3_ = _complex_exp_jvp(trace_real, trace_imag, d_trace_real, d_trace_imag);
    auto plain_real = bsk::get<0>(t3_);
    auto plain_imag = bsk::get<1>(t3_);
    auto d_plain_real = bsk::get<2>(t3_);
    auto d_plain_imag = bsk::get<3>(t3_);
    auto square2_real = ((square_real * square_real) - (square_imag * square_imag));
    auto square2_imag = ((2.0f * square_real) * square_imag);
    auto d_square2_real = (((2.0f * square_real) * d_square_real) - ((2.0f * square_imag) * d_square_imag));
    auto d_square2_imag = (2.0f * ((d_square_real * square_imag) + (square_real * d_square_imag)));
    auto poly_real = ((1.0f + bsk::truediv(square_real, 6.0f)) + bsk::truediv(square2_real, 120.0f));
    auto poly_imag = (bsk::truediv(square_imag, 6.0f) + bsk::truediv(square2_imag, 120.0f));
    auto d_poly_real = (bsk::truediv(d_square_real, 6.0f) + bsk::truediv(d_square2_real, 120.0f));
    auto d_poly_imag = (bsk::truediv(d_square_imag, 6.0f) + bsk::truediv(d_square2_imag, 120.0f));
    auto series_real = ((plain_real * poly_real) - (plain_imag * poly_imag));
    auto series_imag = ((plain_real * poly_imag) + (plain_imag * poly_real));
    auto d_series_real = ((((d_plain_real * poly_real) + (plain_real * d_poly_real)) - (d_plain_imag * poly_imag)) - (plain_imag * d_poly_imag));
    auto d_series_imag = ((((d_plain_real * poly_imag) + (plain_real * d_poly_imag)) + (d_plain_imag * poly_real)) + (plain_imag * d_poly_real));
    auto scale_real = bsk::where(turning, divided_real, series_real);
    auto scale_imag = bsk::where(turning, divided_imag, series_imag);
    auto d_scale_real = bsk::where(turning, d_divided_real, d_series_real);
    auto d_scale_imag = bsk::where(turning, d_divided_imag, d_series_imag);
    auto off_real = ((scale_real * gap_real) - (scale_imag * gap_imag));
    auto off_imag = ((scale_real * gap_imag) + (scale_imag * gap_real));
    auto d_off_real = ((((d_scale_real * gap_real) + (scale_real * d_gap_real)) - (d_scale_imag * gap_imag)) - (scale_imag * d_gap_imag));
    auto d_off_imag = ((((d_scale_real * gap_imag) + (scale_real * d_gap_imag)) + (d_scale_imag * gap_real)) + (scale_imag * d_gap_real));
    auto e11_real = (cos_real + off_real);
    auto e11_imag = (cos_imag + off_imag);
    auto d_e11_real = (d_cos_real + d_off_real);
    auto d_e11_imag = (d_cos_imag + d_off_imag);
    auto e22_real = (cos_real - off_real);
    auto e22_imag = (cos_imag - off_imag);
    auto d_e22_real = (d_cos_real - d_off_real);
    auto d_e22_imag = (d_cos_imag - d_off_imag);
    return bsk::make_tup((attenuation * e11_real), (attenuation * e11_imag), ((attenuation * scale_real) * l12), ((attenuation * scale_imag) * l12), ((attenuation * scale_real) * l21), ((attenuation * scale_imag) * l21), (attenuation * e22_real), (attenuation * e22_imag), ((d_attenuation * e11_real) + (attenuation * d_e11_real)), ((d_attenuation * e11_imag) + (attenuation * d_e11_imag)), (((d_attenuation * scale_real) * l12) + (attenuation * ((d_scale_real * l12) + (scale_real * d_l12)))), (((d_attenuation * scale_imag) * l12) + (attenuation * ((d_scale_imag * l12) + (scale_imag * d_l12)))), (((d_attenuation * scale_real) * l21) + (attenuation * ((d_scale_real * l21) + (scale_real * d_l21)))), (((d_attenuation * scale_imag) * l21) + (attenuation * ((d_scale_imag * l21) + (scale_imag * d_l21)))), ((d_attenuation * e22_real) + (attenuation * d_e22_real)), ((d_attenuation * e22_imag) + (attenuation * d_e22_imag)));
}

// The same fraction and its directional derivative.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _washout_jvp(const T0& rate, const T1& rate_tangent, const T2& dt, const T3& dt_tangent) {
    auto fraction = (rate * dt);
    auto live = (fraction < 1.0f);
    return bsk::make_tup(bsk::where(live, (1.0f - fraction), 0.0f), bsk::where(live, (-((rate_tangent * dt) + (rate * dt_tangent))), 0.0f));
}

BSK_HD void _epg_vjp_jvp_kernel(float* t1, float* t2, float* m0, float* b1, float* b1_phase, float* b0, float* inversion_efficiency, float* diffusion, float* velocity, float* bound_fraction, float* exchange_rate, float* t1_bound, float* pool_b_fraction, float* pool_b_exchange, float* t1_pool_b, float* t2_pool_b, float* pool_b_shift, float* duration, std::int32_t* kind, float* flip, float* phase, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* saturation, float* rf_frequency, float* profile, std::int32_t* profile_index, float* lineshape, float* pairs, std::int32_t* pair_index, float* pair_direction, float* grad_pair_value, float* grad_pair_tangent, float* dot_t1, float* dot_t2, float* dot_m0, float* dot_b1, float* dot_b1_phase, float* dot_b0, float* dot_inversion_efficiency, float* dot_diffusion, float* dot_velocity, float* dot_bound_fraction, float* dot_exchange_rate, float* dot_t1_bound, float* dot_pool_b_fraction, float* dot_pool_b_exchange, float* dot_t1_pool_b, float* dot_t2_pool_b, float* dot_pool_b_shift, float* dot_duration, float* dot_flip, float* dot_phase, std::int32_t* duration_row, float* pool_table, float* pool_bars, float* pool_durations, bsk::index_t row_count, float* grad_output_real, float* grad_output_imag, float* grad_tissue_value, float* grad_tissue_tangent, float* grad_flip_value, float* grad_flip_tangent, float* grad_phase_value, float* grad_phase_tangent, float* grad_duration_value, float* grad_duration_tangent, float* trajectory_vr, float* trajectory_vi, float* trajectory_tr, float* trajectory_ti, bsk::index_t problem_base, bsk::index_t problem_end, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, float flow_scale, float washout_scale, float profile_step, float lineshape_step, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shim_rows, bsk::index_t shimmed, bsk::index_t locations, bsk::index_t profiled, bsk::index_t profile_bins, bsk::index_t dynamic, bsk::index_t directed, bsk::index_t off_axis, bsk::index_t moving, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t broadened, bsk::index_t lineshape_bins, bsk::index_t pools, bsk::index_t narrow, bsk::index_t tabulated, bsk::index_t recording, bsk::index_t block_states, bsk::index_t problems) {
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> a11{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> a12{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> a21{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> a22{};
    bsk::V<float, 2> absorbed_tangent{};
    bsk::V<float, 2> absorbed_value{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> across{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> add1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> add2{};
    bsk::V<float, 3> alpha_b_t{};
    bsk::V<float, 3> alpha_b_v{};
    bsk::V<float, 3> alpha_t{};
    bsk::V<float, 2> alpha_tangent{};
    bsk::V<float, 3> alpha_v{};
    bsk::V<float, 2> alpha_value{};
    bsk::V<float, 3> angle_tangent{};
    bsk::V<float, 3> angle_value{};
    bsk::V<float, 3> ati{};
    bsk::V<float, 2> atom_b0{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_b1_phase{};
    bsk::V<float, 2> atom_bound{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_exchange{};
    bsk::V<float, 2> atom_flow{};
    bsk::V<float, 2> atom_free{};
    bsk::V<float, 2> atom_inv{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 2> atom_semisolid{};
    bsk::V<float, 2> atom_semisolid_exchange{};
    bsk::V<float, 2> atom_shift{};
    bsk::V<float, 2> atom_t1b{};
    bsk::V<float, 2> atom_t2b{};
    bsk::V<float, 2> atom_washout{};
    bsk::V<float, 3> atr{};
    bsk::V<float, 2> att_rate{};
    bsk::V<float, 2> att_span{};
    bsk::V<float, 2> attenuation_t{};
    bsk::V<float, 2> attenuation_v{};
    bsk::V<float, 3> avi{};
    bsk::V<float, 3> avr{};
    bsk::V<float, 2> back_att_t{};
    bsk::V<float, 2> back_att_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_bb{};
    bsk::V<float, 2> back_bound_t{};
    bsk::V<float, 2> back_bound_v{};
    bsk::V<float, 2> back_dt_t{};
    bsk::V<float, 2> back_dt_v{};
    bsk::V<float, 2> back_exch_t{};
    bsk::V<float, 2> back_exch_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_mb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_pb{};
    bsk::V<float, 2> back_r1_t{};
    bsk::V<float, 2> back_r1_v{};
    bsk::V<float, 2> back_r1b_t{};
    bsk::V<float, 2> back_r1b_v{};
    bsk::V<float, 2> back_r1c_t{};
    bsk::V<float, 2> back_r1c_v{};
    bsk::V<float, 2> back_semi_t{};
    bsk::V<float, 2> back_semi_v{};
    bsk::V<float, 2> back_sexch_t{};
    bsk::V<float, 2> back_sexch_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_ub{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_wb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> back_zb{};
    bsk::V<float, 3> bare1_tangent{};
    bsk::V<float, 3> bare1_value{};
    bsk::V<float, 3> bare2_tangent{};
    bsk::V<float, 3> bare2_value{};
    std::int32_t base_row{};
    bsk::V<float, 3> bbti{};
    bsk::V<float, 3> bbtr{};
    bsk::V<float, 3> bbvi{};
    bsk::V<float, 3> bbvr{};
    bsk::V<float, 3> bmti{};
    bsk::V<float, 3> bmtr{};
    bsk::V<float, 3> bmvi{};
    bsk::V<float, 3> bmvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> bound_bar{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> bound_part{};
    bsk::V<float, 3> bpti{};
    bsk::V<float, 3> bptr{};
    bsk::V<float, 3> bpvi{};
    bsk::V<float, 3> bpvr{};
    bsk::V<float, 3> bti{};
    bsk::V<float, 3> btr{};
    bsk::V<float, 3> bvi{};
    bsk::V<float, 3> bvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> carried{};
    bsk::V<float, 3> cbti{};
    bsk::V<float, 3> cbtr{};
    bsk::V<float, 3> cbvi{};
    bsk::V<float, 3> cbvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> conjugated{};
    bsk::V<float, 2> cos_tangent{};
    bsk::V<float, 2> cos_value{};
    bsk::V<float, 3> cot2_t{};
    bsk::V<float, 3> cot2_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> cross_in{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> cross_out{};
    bsk::V<float, 3> cti{};
    bsk::V<float, 3> ctr{};
    bsk::V<float, 3> cvi{};
    bsk::V<float, 3> cvr{};
    bsk::V<float, 2> d_b0{};
    bsk::V<float, 2> d_b1{};
    bsk::V<float, 2> d_b1_phase{};
    bsk::V<float, 2> d_boundf{};
    bsk::V<float, 2> d_damping{};
    bsk::V<float, 2> d_exchange{};
    bsk::V<float, 2> d_flow{};
    bsk::V<float, 2> d_free{};
    bsk::V<float, 2> d_grow_free{};
    bsk::V<float, 2> d_grow_pool_b{};
    bsk::V<float, 2> d_grow_semisolid{};
    bsk::V<float, 2> d_inv{};
    bsk::V<float, 2> d_m0{};
    bsk::V<float, 2> d_semisolid_exchange{};
    bsk::V<float, 2> d_semisolid_t1{};
    bsk::V<float, 2> d_semisolidf{};
    bsk::V<float, 2> d_shift{};
    bsk::V<float, 2> d_t11{};
    bsk::V<float, 2> d_t12{};
    bsk::V<float, 2> d_t13{};
    bsk::V<float, 2> d_t1b{};
    bsk::V<float, 2> d_t21{};
    bsk::V<float, 2> d_t22{};
    bsk::V<float, 2> d_t23{};
    bsk::V<float, 2> d_t2b{};
    bsk::V<float, 2> d_t31{};
    bsk::V<float, 2> d_t32{};
    bsk::V<float, 2> d_t33{};
    bsk::V<float, 2> d_turn{};
    bsk::V<float, 2> d_w11{};
    bsk::V<float, 2> d_w12{};
    bsk::V<float, 2> d_w13{};
    bsk::V<float, 2> d_w21{};
    bsk::V<float, 2> d_w22{};
    bsk::V<float, 2> d_w23{};
    bsk::V<float, 2> d_w31{};
    bsk::V<float, 2> d_w32{};
    bsk::V<float, 2> d_w33{};
    bsk::V<float, 2> d_washout{};
    bsk::V<float, 3> damp_pair_t{};
    bsk::V<float, 3> damp_pair_v{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_t_tangent{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 3> damp_z_tangent{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> damped{};
    bsk::V<float, 2> de11{};
    bsk::V<float, 2> de12{};
    bsk::V<float, 2> de21{};
    bsk::V<float, 2> de22{};
    bsk::V<float, 2> direction{};
    bool do_shift{};
    bsk::V<float, 2> drec_b{};
    bsk::V<float, 2> drec_f{};
    bsk::V<float, 2> dry1_tangent{};
    bsk::V<float, 2> dry1_value{};
    bsk::V<float, 2> dry2_tangent{};
    bsk::V<float, 2> dry2_value{};
    bsk::V<float, 2> dt_tangent{};
    bsk::V<float, 2> dt_value{};
    bsk::V<float, 3> dturn_t{};
    bsk::V<float, 3> dturn_z{};
    bsk::V<float, 3> duration_t{};
    bsk::V<float, 3> duration_v{};
    bsk::V<float, 3> e11_t{};
    bsk::V<float, 3> e11_v{};
    bsk::V<float, 3> e12_t{};
    bsk::V<float, 3> e12_v{};
    bsk::V<float, 3> e1_tangent{};
    bsk::V<float, 3> e1_value{};
    bsk::V<float, 3> e21_t{};
    bsk::V<float, 3> e21_v{};
    bsk::V<float, 3> e22_t{};
    bsk::V<float, 3> e22_v{};
    bsk::V<float, 3> e2_tangent{};
    bsk::V<float, 3> e2_value{};
    std::int64_t event{};
    std::int32_t event_action{};
    bsk::V<float, 2> event_dot_flip{};
    bsk::V<float, 2> event_dot_phase{};
    bsk::V<float, 2> event_flip{};
    std::int32_t event_kind{};
    bsk::V<float, 2> event_phase{};
    float event_saturation{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> free_bar{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> free_minus{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> free_part{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> free_plus{};
    bsk::V<float, 2> g_b0t{};
    bsk::V<float, 2> g_b0v{};
    bsk::V<float, 2> g_b1pt{};
    bsk::V<float, 2> g_b1pv{};
    bsk::V<float, 2> g_b1t{};
    bsk::V<float, 2> g_b1v{};
    bsk::V<float, 2> g_boundt{};
    bsk::V<float, 2> g_boundv{};
    bsk::V<float, 2> g_difft{};
    bsk::V<float, 2> g_diffv{};
    bsk::V<float, 2> g_excht{};
    bsk::V<float, 2> g_exchv{};
    bsk::V<float, 2> g_flowt{};
    bsk::V<float, 2> g_flowv{};
    bsk::V<float, 2> g_invt{};
    bsk::V<float, 2> g_invv{};
    bsk::V<float, 2> g_m0t{};
    bsk::V<float, 2> g_m0v{};
    bsk::V<float, 2> g_semit{};
    bsk::V<float, 2> g_semiv{};
    bsk::V<float, 2> g_sexcht{};
    bsk::V<float, 2> g_sexchv{};
    bsk::V<float, 2> g_shiftt{};
    bsk::V<float, 2> g_shiftv{};
    bsk::V<float, 2> g_t1bt{};
    bsk::V<float, 2> g_t1bv{};
    bsk::V<float, 2> g_t1ct{};
    bsk::V<float, 2> g_t1cv{};
    bsk::V<float, 3> g_t1t{};
    bsk::V<float, 3> g_t1v{};
    bsk::V<float, 2> g_t2bt{};
    bsk::V<float, 2> g_t2bv{};
    bsk::V<float, 3> g_t2t{};
    bsk::V<float, 3> g_t2v{};
    bsk::V<float, 2> g_washt{};
    bsk::V<float, 2> g_washv{};
    bsk::V<float, 2> grad_alpha_t{};
    bsk::V<float, 2> grad_alpha_v{};
    bsk::V<float, 2> grad_angle_t{};
    bsk::V<float, 2> grad_angle_v{};
    bsk::V<float, 2> grad_e1_t{};
    bsk::V<float, 2> grad_e1_v{};
    bsk::V<float, 2> grad_e2_t{};
    bsk::V<float, 2> grad_e2_v{};
    bsk::V<float, 2> grow_free{};
    bsk::V<float, 2> grow_pool_b{};
    bsk::V<float, 2> grow_semisolid{};
    bsk::V<float*, 2> held{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> held_bar{};
    bsk::V<float, 2> held_semisolid{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> held_state{};
    bool invert{};
    bool is_inversion{};
    bool is_rf{};
    bsk::V<float, 3> iti{};
    bsk::V<float, 3> itr{};
    bsk::V<float, 3> ivi{};
    bsk::V<float, 3> ivr{};
    bsk::V<float, 3> long_damp_t{};
    bsk::V<float, 3> long_damp_v{};
    bsk::V<float, 3> lti{};
    bsk::V<float, 3> ltr{};
    bsk::V<float, 3> lvi{};
    bsk::V<float, 3> lvr{};
    bsk::V<float, 3> mbti{};
    bsk::V<float, 3> mbtr{};
    bsk::V<float, 3> mbvi{};
    bsk::V<float, 3> mbvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> mixed_bound{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> mixed_free{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> mixed_semisolid{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> mo{};
    bsk::V<float, 3> mti{};
    bsk::V<float, 3> mtr{};
    bsk::V<float, 3> mvi{};
    bsk::V<float, 3> mvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> n0{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> n1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> n2{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> next_mb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> next_pb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> next_ub{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> next_wb{};
    bsk::V<float, 2> offset_value{};
    bsk::V<float, 2> one_att{};
    bsk::V<float, 3> other_t{};
    bsk::V<float, 3> other_v{};
    bsk::V<float, 3> oti{};
    bsk::V<float, 3> otr{};
    bsk::V<float, 3> ovi{};
    bsk::V<float, 3> ovr{};
    bsk::V<float, 2> p1i{};
    bsk::V<float, 2> p1r{};
    bsk::V<float, 2> p1ti{};
    bsk::V<float, 2> p1tr{};
    bsk::V<float, 2> p2i{};
    bsk::V<float, 2> p2r{};
    bsk::V<float, 2> p2ti{};
    bsk::V<float, 2> p2tr{};
    bsk::V<float, 3> part_t{};
    bsk::V<float, 3> part_v{};
    bsk::V<float, 3> pbti{};
    bsk::V<float, 3> pbtr{};
    bsk::V<float, 3> pbvi{};
    bsk::V<float, 3> pbvr{};
    bsk::V<float, 2> pe11{};
    bsk::V<float, 2> pe12{};
    bsk::V<float, 2> pe21{};
    bsk::V<float, 2> pe22{};
    bsk::V<float, 3> per_angle_t{};
    bsk::V<float, 3> per_angle_v{};
    bsk::V<float, 3> phi_b_t{};
    bsk::V<float, 3> phi_b_v{};
    bsk::V<float, 3> phi_t{};
    bsk::V<float, 2> phi_tangent{};
    bsk::V<float, 3> phi_v{};
    bsk::V<float, 2> phi_value{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> po{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> pool_minus{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> pool_plus{};
    bsk::V<std::int32_t, 2> pool_row{};
    bsk::V<float, 2> power_tangent{};
    bsk::V<float, 2> power_value{};
    bool pre_shift{};
    bsk::V<float, 2> prec_b{};
    bsk::V<float, 2> prec_f{};
    bsk::V<std::int32_t, 2> problem{};
    bsk::V<float, 3> pti{};
    bsk::V<float, 3> ptr{};
    bsk::V<float, 3> pvi{};
    bsk::V<float, 3> pvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> q0{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> q1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> q2{};
    bsk::V<float, 3> qi{};
    bsk::V<float, 3> qr{};
    bsk::V<float, 3> qti{};
    bsk::V<float, 3> qtr{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> r12{};
    bsk::V<float, 2> r1b_tangent{};
    bsk::V<float, 2> r1b_value{};
    bsk::V<float, 2> r1c_tangent{};
    bsk::V<float, 2> r1c_value{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> r21{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> r22{};
    bsk::V<float, 2> r2b_tangent{};
    bsk::V<float, 2> r2b_value{};
    bsk::V<float, 3> rbmti{};
    bsk::V<float, 3> rbmtr{};
    bsk::V<float, 3> rbmvi{};
    bsk::V<float, 3> rbmvr{};
    bsk::V<float, 3> rbpti{};
    bsk::V<float, 3> rbptr{};
    bsk::V<float, 3> rbpvi{};
    bsk::V<float, 3> rbpvr{};
    bsk::V<float, 3> rbti{};
    bsk::V<float, 3> rbtr{};
    bsk::V<float, 3> rbvi{};
    bsk::V<float, 3> rbvr{};
    bsk::V<float, 3> rcti{};
    bsk::V<float, 3> rctr{};
    bsk::V<float, 3> rcvi{};
    bsk::V<float, 3> rcvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> recorded{};
    bsk::V<float, 3> recovery_tangent{};
    bsk::V<float, 3> recovery_value{};
    bsk::V<float, 3> rmti{};
    bsk::V<float, 3> rmtr{};
    bsk::V<float, 3> rmvi{};
    bsk::V<float, 3> rmvr{};
    bool rotate{};
    std::int64_t row{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> row0{};
    bsk::V<float, 3> rpti{};
    bsk::V<float, 3> rptr{};
    bsk::V<float, 3> rpvi{};
    bsk::V<float, 3> rpvr{};
    bsk::V<float, 3> rzti{};
    bsk::V<float, 3> rztr{};
    bsk::V<float, 3> rzvi{};
    bsk::V<float, 3> rzvr{};
    bsk::V<float, 2> sat_alpha_t{};
    bsk::V<float, 2> sat_alpha_v{};
    bsk::V<float, 2> sat_b0_t{};
    bsk::V<float, 2> sat_b0_v{};
    bool saturating{};
    bsk::V<float, 3> sbmti{};
    bsk::V<float, 3> sbmtr{};
    bsk::V<float, 3> sbmvi{};
    bsk::V<float, 3> sbmvr{};
    bsk::V<float, 3> sbpti{};
    bsk::V<float, 3> sbptr{};
    bsk::V<float, 3> sbpvi{};
    bsk::V<float, 3> sbpvr{};
    bsk::V<float, 3> scale1_tangent{};
    bsk::V<float, 3> scale2_tangent{};
    bsk::V<float, 2> shape_slope{};
    bsk::V<float, 2> shape_tangent{};
    bsk::V<float, 2> shape_value{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> shaped_a{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> shaped_b{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_bb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_mb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_pb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_slope_a{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_slope_b{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_ub{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_wb{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> shaped_zb{};
    bsk::V<float, 2> sin_tangent{};
    bsk::V<float, 2> sin_value{};
    bsk::V<float, 2> slope1_t{};
    bsk::V<float, 2> slope1_v{};
    bsk::V<float, 2> slope1b_t{};
    bsk::V<float, 2> slope1b_v{};
    bsk::V<float, 2> slope1c_t{};
    bsk::V<float, 2> slope1c_v{};
    bsk::V<std::int32_t, 3> slot{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> spin{};
    bool spoil{};
    bsk::V<float, 2> spread_t{};
    bsk::V<float, 2> spread_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> spun_bound{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> spun_free{};
    bsk::V<float, 3> spun_mti{};
    bsk::V<float, 3> spun_mtr{};
    bsk::V<float, 3> spun_mvi{};
    bsk::V<float, 3> spun_mvr{};
    bsk::V<float, 3> spun_pti{};
    bsk::V<float, 3> spun_ptr{};
    bsk::V<float, 3> spun_pvi{};
    bsk::V<float, 3> spun_pvr{};
    bsk::V<float, 3> spun_zti{};
    bsk::V<float, 3> spun_ztr{};
    bsk::V<float, 3> spun_zvi{};
    bsk::V<float, 3> spun_zvr{};
    bsk::V<float, 3> sti{};
    bsk::V<float, 3> str_{};
    bsk::V<float, 3> svi{};
    bsk::V<float, 3> svr{};
    bsk::V<float, 3> szi{};
    bsk::V<float, 3> szr{};
    bsk::V<float, 3> szti{};
    bsk::V<float, 3> sztr{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> t00{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> t01{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> t02{};
    bsk::V<float, 2> t11{};
    bsk::V<float, 2> t12{};
    bsk::V<float, 2> t13{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> t20{};
    bsk::V<float, 2> t21{};
    bsk::V<float, 2> t22{};
    bsk::V<float, 2> t23{};
    bsk::V<float, 2> t31{};
    bsk::V<float, 2> t32{};
    bsk::V<float, 2> t33{};
    bsk::V<double, 2> three_a00{};
    bsk::V<double, 2> three_a01{};
    bsk::V<double, 2> three_a02{};
    bsk::V<double, 2> three_a10{};
    bsk::V<double, 2> three_a11{};
    bsk::V<double, 2> three_a20{};
    bsk::V<double, 2> three_a22{};
    bsk::V<double, 2> three_angle{};
    bsk::V<double, 2> three_argument{};
    bsk::V<double, 2> three_centre{};
    bsk::V<double, 2> three_cube{};
    bsk::V<double, 2> three_d_a00{};
    bsk::V<double, 2> three_d_a01{};
    bsk::V<double, 2> three_d_a02{};
    bsk::V<double, 2> three_d_a10{};
    bsk::V<double, 2> three_d_a11{};
    bsk::V<double, 2> three_d_a20{};
    bsk::V<double, 2> three_d_a22{};
    bsk::V<double, 2> three_d_angle{};
    bsk::V<double, 2> three_d_centre{};
    bsk::V<double, 2> three_d_determinant{};
    bsk::V<double, 2> three_d_first{};
    bsk::V<double, 2> three_d_free{};
    bsk::V<double, 2> three_d_guarded{};
    bsk::V<double, 2> three_d_high{};
    bsk::V<double, 2> three_d_leading{};
    bsk::V<double, 2> three_d_lift{};
    bsk::V<double, 2> three_d_low{};
    bsk::V<double, 2> three_d_middle{};
    bsk::V<double, 2> three_d_minors{};
    bsk::V<double, 2> three_d_pool_b{};
    bsk::V<double, 2> three_d_pool_c{};
    bsk::V<double, 2> three_d_q00{};
    bsk::V<double, 2> three_d_q01{};
    bsk::V<double, 2> three_d_q02{};
    bsk::V<double, 2> three_d_q10{};
    bsk::V<double, 2> three_d_q11{};
    bsk::V<double, 2> three_d_q12{};
    bsk::V<double, 2> three_d_q20{};
    bsk::V<double, 2> three_d_q21{};
    bsk::V<double, 2> three_d_q22{};
    bsk::V<double, 2> three_d_radius{};
    bsk::V<double, 2> three_d_raw{};
    bsk::V<double, 2> three_d_s00{};
    bsk::V<double, 2> three_d_s11{};
    bsk::V<double, 2> three_d_s22{};
    bsk::V<double, 2> three_d_second{};
    bsk::V<double, 2> three_d_sum_flat{};
    bsk::V<double, 2> three_d_sum_linear{};
    bsk::V<double, 2> three_d_sum_square{};
    bsk::V<double, 2> three_d_trailing{};
    bsk::V<double, 2> three_def_00{};
    bsk::V<double, 2> three_def_01{};
    bsk::V<double, 2> three_def_02{};
    bsk::V<double, 2> three_def_10{};
    bsk::V<double, 2> three_def_11{};
    bsk::V<double, 2> three_def_12{};
    bsk::V<double, 2> three_def_20{};
    bsk::V<double, 2> three_def_21{};
    bsk::V<double, 2> three_def_22{};
    bsk::V<double, 2> three_determinant{};
    bsk::V<double, 2> three_dif_00{};
    bsk::V<double, 2> three_dif_01{};
    bsk::V<double, 2> three_dif_02{};
    bsk::V<double, 2> three_dif_10{};
    bsk::V<double, 2> three_dif_11{};
    bsk::V<double, 2> three_dif_12{};
    bsk::V<double, 2> three_dif_20{};
    bsk::V<double, 2> three_dif_21{};
    bsk::V<double, 2> three_dif_22{};
    bsk::V<double, 2> three_first{};
    bsk::V<double, 2> three_free{};
    bsk::V<double, 2> three_guarded{};
    bsk::V<double, 2> three_high{};
    bsk::V<bool, 2> three_inside_limit{};
    bsk::V<double, 2> three_leading{};
    bsk::V<double, 2> three_lift{};
    bsk::V<double, 2> three_low{};
    bsk::V<double, 2> three_middle{};
    bsk::V<double, 2> three_minors{};
    bsk::V<double, 2> three_pool_b{};
    bsk::V<double, 2> three_pool_c{};
    bsk::V<double, 2> three_q00{};
    bsk::V<double, 2> three_q01{};
    bsk::V<double, 2> three_q02{};
    bsk::V<double, 2> three_q10{};
    bsk::V<double, 2> three_q11{};
    bsk::V<double, 2> three_q12{};
    bsk::V<double, 2> three_q20{};
    bsk::V<double, 2> three_q21{};
    bsk::V<double, 2> three_q22{};
    bsk::V<double, 2> three_radius{};
    bsk::V<double, 2> three_raw{};
    bsk::V<double, 2> three_s00{};
    bsk::V<double, 2> three_s11{};
    bsk::V<double, 2> three_s22{};
    bsk::V<double, 2> three_second{};
    bsk::V<double, 2> three_sum_flat{};
    bsk::V<double, 2> three_sum_linear{};
    bsk::V<double, 2> three_sum_square{};
    bsk::V<double, 2> three_trailing{};
    bsk::V<float, 3> turn_t{};
    bsk::V<float, 3> turn_z{};
    bsk::V<float, 3> turned_mti{};
    bsk::V<float, 3> turned_mtr{};
    bsk::V<float, 3> turned_mvi{};
    bsk::V<float, 3> turned_mvr{};
    bsk::V<float, 3> turned_pti{};
    bsk::V<float, 3> turned_ptr{};
    bsk::V<float, 3> turned_pvi{};
    bsk::V<float, 3> turned_pvr{};
    bsk::V<float, 3> turned_zti{};
    bsk::V<float, 3> turned_ztr{};
    bsk::V<float, 3> turned_zvi{};
    bsk::V<float, 3> turned_zvr{};
    bsk::V<float, 2> two_pool_dt_t{};
    bsk::V<float, 2> two_pool_dt_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> u1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> u2{};
    bsk::V<float, 3> ubti{};
    bsk::V<float, 3> ubtr{};
    bsk::V<float, 3> ubvi{};
    bsk::V<float, 3> ubvr{};
    bsk::V<float, 3> ui{};
    bsk::V<float, 3> ur{};
    bsk::V<float, 3> uti{};
    bsk::V<float, 3> utr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> w0{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> w1{};
    bsk::V<float, 2> w11{};
    bsk::V<float, 2> w12{};
    bsk::V<float, 2> w13{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> w2{};
    bsk::V<float, 2> w21{};
    bsk::V<float, 2> w22{};
    bsk::V<float, 2> w23{};
    bsk::V<float, 2> w31{};
    bsk::V<float, 2> w32{};
    bsk::V<float, 2> w33{};
    bsk::V<float, 2> wash_t{};
    bsk::V<float, 2> wash_v{};
    bsk::V<float, 3> wbti{};
    bsk::V<float, 3> wbtr{};
    bsk::V<float, 3> wbvi{};
    bsk::V<float, 3> wbvr{};
    bsk::V<float, 2> wound_t{};
    bsk::V<float, 2> wound_v{};
    bsk::V<float, 2> wout_tangent{};
    bsk::V<float, 2> wout_value{};
    bsk::V<float, 3> wti{};
    bsk::V<float, 3> wtr{};
    bsk::V<float, 3> wvi{};
    bsk::V<float, 3> wvr{};
    bsk::V<float, 3> xbmti{};
    bsk::V<float, 3> xbmtr{};
    bsk::V<float, 3> xbmvi{};
    bsk::V<float, 3> xbmvr{};
    bsk::V<float, 3> xbpti{};
    bsk::V<float, 3> xbptr{};
    bsk::V<float, 3> xbpvi{};
    bsk::V<float, 3> xbpvr{};
    bsk::V<float, 3> xbti{};
    bsk::V<float, 3> xbtr{};
    bsk::V<float, 3> xbvi{};
    bsk::V<float, 3> xbvr{};
    bsk::V<float, 3> xcti{};
    bsk::V<float, 3> xctr{};
    bsk::V<float, 3> xcvi{};
    bsk::V<float, 3> xcvr{};
    bsk::V<float, 3> yi{};
    bsk::V<float, 3> yr{};
    bsk::V<float, 3> yti{};
    bsk::V<float, 3> ytr{};
    bsk::V<float, 3> zangle_t{};
    bsk::V<float, 3> zangle_v{};
    bsk::V<float, 3> zbti{};
    bsk::V<float, 3> zbtr{};
    bsk::V<float, 3> zbvi{};
    bsk::V<float, 3> zbvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>, bsk::V<float, 3>> zo{};
    bsk::V<float, 3> zti{};
    bsk::V<float, 3> ztr{};
    bsk::V<float, 3> zvi{};
    bsk::V<float, 3> zvr{};
    problem = (problem_base + (bsk::program_id(0) * problems));
    problem = (problem + bsk::arange_y());
    auto state = bsk::arange_x();
    auto active_atom = (problem < problem_end);
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    // Voxels are spread over the slice voxel-major, so a voxel's place along
    // the slice is its index modulo the profile's width. One pulse shape holds
    // that many consecutive rows, and the event says which shape it drives.
    auto location = bsk::mod(atom, locations);
    auto local = (problem - problem_base);
    // A second pool rides along as planes of its own: it enters an event as its
    // own vector and the RF operator acts on it, so the reverse sweep cannot
    // replay it from the free pool's. A semisolid pool adds one plane, a
    // chemically exchanging one three, and the two together add four.
    auto record_stride = (bsk::select(bsk::truth((pools == 3)), 7, bsk::select(bsk::truth((pools == 2)), 6, bsk::select(bsk::truth((pools == 1)), 4, 3))) * state_count);
    auto trajectory = (((local * event_count) * record_stride) + state);
    auto minus_plane = state_count;
    auto long_plane = (2 * state_count);
    auto bound_plane = (3 * state_count);
    auto bplus_plane = (4 * state_count);
    auto bminus_plane = (5 * state_count);
    auto semisolid_plane = (6 * state_count);
    auto empty = bsk::full<float, 3>(0);
    pvr = empty;
    pvi = empty;
    ptr = empty;
    pti = empty;
    mvr = empty;
    mvi = empty;
    mtr = empty;
    mti = empty;
    bvr = empty;
    bvi = empty;
    btr = empty;
    bti = empty;
    bpvr = empty;
    bpvi = empty;
    bptr = empty;
    bpti = empty;
    bmvr = empty;
    bmvi = empty;
    bmtr = empty;
    bmti = empty;
    cvr = empty;
    cvi = empty;
    ctr = empty;
    cti = empty;
    atom_bound = 0.0f;
    d_boundf = 0.0f;
    atom_exchange = 0.0f;
    d_exchange = 0.0f;
    atom_t1b = 1.0f;
    d_t1b = 0.0f;
    r1b_value = 0.0f;
    r1b_tangent = 0.0f;
    atom_t2b = 1.0f;
    d_t2b = 0.0f;
    r2b_value = 0.0f;
    r2b_tangent = 0.0f;
    atom_shift = 0.0f;
    d_shift = 0.0f;
    atom_semisolid = 0.0f;
    d_semisolidf = 0.0f;
    atom_semisolid_exchange = 0.0f;
    d_semisolid_exchange = 0.0f;
    r1c_value = 0.0f;
    r1c_tangent = 0.0f;
    if (bsk::truth((pools == 1))) {
        atom_bound = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        d_boundf = bsk::ld((dot_bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((exchange_rate + scalar_atom), active_atom, 0.0f);
        d_exchange = bsk::ld((dot_exchange_rate + scalar_atom), active_atom, 0.0f);
        atom_t1b = bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f);
        d_t1b = bsk::ld((dot_t1_bound + scalar_atom), active_atom, 0.0f);
        r1b_value = bsk::truediv(1000.0f, atom_t1b);
        r1b_tangent = bsk::truediv((-1000.0f * d_t1b), (atom_t1b * atom_t1b));
    }
    if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
        atom_bound = bsk::ld((pool_b_fraction + scalar_atom), active_atom, 0.0f);
        d_boundf = bsk::ld((dot_pool_b_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((pool_b_exchange + scalar_atom), active_atom, 0.0f);
        d_exchange = bsk::ld((dot_pool_b_exchange + scalar_atom), active_atom, 0.0f);
        atom_t1b = bsk::ld((t1_pool_b + scalar_atom), active_atom, 1.0f);
        d_t1b = bsk::ld((dot_t1_pool_b + scalar_atom), active_atom, 0.0f);
        r1b_value = bsk::truediv(1000.0f, atom_t1b);
        r1b_tangent = bsk::truediv((-1000.0f * d_t1b), (atom_t1b * atom_t1b));
        atom_t2b = bsk::ld((t2_pool_b + scalar_atom), active_atom, 1.0f);
        d_t2b = bsk::ld((dot_t2_pool_b + scalar_atom), active_atom, 0.0f);
        r2b_value = bsk::truediv(1000.0f, atom_t2b);
        r2b_tangent = bsk::truediv((-1000.0f * d_t2b), (atom_t2b * atom_t2b));
        atom_shift = bsk::ld((pool_b_shift + scalar_atom), active_atom, 0.0f);
        d_shift = bsk::ld((dot_pool_b_shift + scalar_atom), active_atom, 0.0f);
    }
    if (bsk::truth((pools == 3))) {
        atom_semisolid = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        d_semisolidf = bsk::ld((dot_bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_semisolid_exchange = bsk::ld((exchange_rate + scalar_atom), active_atom, 0.0f);
        d_semisolid_exchange = bsk::ld((dot_exchange_rate + scalar_atom), active_atom, 0.0f);
        held_semisolid = bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f);
        d_semisolid_t1 = bsk::ld((dot_t1_bound + scalar_atom), active_atom, 0.0f);
        r1c_value = bsk::truediv(1000.0f, held_semisolid);
        r1c_tangent = bsk::truediv((-1000.0f * d_semisolid_t1), (held_semisolid * held_semisolid));
        cvr = (empty + bsk::where((state == 0), (atom_semisolid + 0.0f), 0.0f));
        ctr = (empty + bsk::where((state == 0), (d_semisolidf + 0.0f), 0.0f));
    }
    if (bsk::truth((pools > 0))) {
        atom_free = ((1.0f - atom_bound) - atom_semisolid);
        d_free = ((-d_boundf) - d_semisolidf);
        zvr = (empty + bsk::where((state == 0), atom_free, 0.0f));
        ztr = (empty + bsk::where((state == 0), d_free, 0.0f));
        bvr = (empty + bsk::where((state == 0), (atom_bound + 0.0f), 0.0f));
        btr = (empty + bsk::where((state == 0), (d_boundf + 0.0f), 0.0f));
    } else {
        atom_free = (1.0f + (0.0f * atom_bound));
        d_free = (0.0f * atom_bound);
        zvr = (empty + bsk::where((state == 0), 1.0f, 0.0f));
        ztr = empty;
    }
    zvi = empty;
    zti = empty;
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_b1_phase = 0.0f;
    atom_b0 = 0.0f;
    if (bsk::truth(off_axis)) {
        atom_b1_phase = bsk::ld((b1_phase + scalar_atom), active_atom, 0.0f);
        atom_b0 = bsk::ld((b0 + scalar_atom), active_atom, 0.0f);
    }
    atom_inv = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inv = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    auto d_t1 = bsk::ld((dot_t1 + atom), active_atom, 0.0f);
    auto d_t2 = bsk::ld((dot_t2 + atom), active_atom, 0.0f);
    d_m0 = 0.0f;
    if (bsk::truth(density)) {
        d_m0 = bsk::ld((dot_m0 + scalar_atom), active_atom, 0.0f);
    }
    d_b1 = 0.0f;
    if (bsk::truth(transmit)) {
        d_b1 = bsk::ld((dot_b1 + scalar_atom), active_atom, 0.0f);
    }
    d_b1_phase = 0.0f;
    d_b0 = 0.0f;
    if (bsk::truth(off_axis)) {
        d_b1_phase = bsk::ld((dot_b1_phase + scalar_atom), active_atom, 0.0f);
        d_b0 = bsk::ld((dot_b0 + scalar_atom), active_atom, 0.0f);
    }
    d_inv = 0.0f;
    if (bsk::truth(inverting)) {
        d_inv = bsk::ld((dot_inversion_efficiency + scalar_atom), active_atom, 0.0f);
    }
    atom_damping = 0.0f;
    d_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
        d_damping = bsk::ld((dot_diffusion + scalar_atom), active_atom, 0.0f);
    }
    atom_flow = 0.0f;
    d_flow = 0.0f;
    direction = 0.0f;
    atom_washout = 0.0f;
    d_washout = 0.0f;
    if (bsk::truth(moving)) {
        auto atom_velocity = bsk::ld((velocity + scalar_atom), active_atom, 0.0f);
        auto d_velocity = bsk::ld((dot_velocity + scalar_atom), active_atom, 0.0f);
        atom_flow = (atom_velocity * flow_scale);
        d_flow = (d_velocity * flow_scale);
        // |v| has no derivative at the origin, so a still voxel contributes
        // none.
        direction = (bsk::cast<float>((atom_velocity > 0.0f)) - bsk::cast<float>((atom_velocity < 0.0f)));
        atom_washout = (bsk::abs(atom_velocity) * washout_scale);
        d_washout = ((direction * d_velocity) * washout_scale);
    }
    auto order = bsk::cast<float>(state);
    auto longitudinal_weight = (order * order);
    auto transverse_weight = ((longitudinal_weight + order) + 0.3333333333333333f);
    auto r1_value = bsk::truediv(1000.0f, atom_t1);
    auto r1_tangent = bsk::truediv((-1000.0f * d_t1), (atom_t1 * atom_t1));
    auto r2_value = bsk::truediv(1000.0f, atom_t2);
    auto r2_tangent = bsk::truediv((-1000.0f * d_t2), (atom_t2 * atom_t2));
    auto event_base = (train * event_count);
    // The forward half records the trajectory the reverse half walks back,
    // and the two are launched separately: each compiles the sweep it is
    // asked for and no more.
    if (bsk::truth(recording)) {
        for (bsk::index_t event = 0; event < event_count; event += 1) {
            slot = (trajectory + (event * record_stride));
            bsk::st((trajectory_vr + slot), pvr, state_mask);
            bsk::st((trajectory_vi + slot), pvi, state_mask);
            bsk::st((trajectory_tr + slot), ptr, state_mask);
            bsk::st((trajectory_ti + slot), pti, state_mask);
            bsk::st(((trajectory_vr + slot) + minus_plane), mvr, state_mask);
            bsk::st(((trajectory_vi + slot) + minus_plane), mvi, state_mask);
            bsk::st(((trajectory_tr + slot) + minus_plane), mtr, state_mask);
            bsk::st(((trajectory_ti + slot) + minus_plane), mti, state_mask);
            bsk::st(((trajectory_vr + slot) + long_plane), zvr, state_mask);
            bsk::st(((trajectory_vi + slot) + long_plane), zvi, state_mask);
            bsk::st(((trajectory_tr + slot) + long_plane), ztr, state_mask);
            bsk::st(((trajectory_ti + slot) + long_plane), zti, state_mask);
            if (bsk::truth((pools > 0))) {
                bsk::st(((trajectory_vr + slot) + bound_plane), bvr, state_mask);
                bsk::st(((trajectory_vi + slot) + bound_plane), bvi, state_mask);
                bsk::st(((trajectory_tr + slot) + bound_plane), btr, state_mask);
                bsk::st(((trajectory_ti + slot) + bound_plane), bti, state_mask);
            }
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                bsk::st(((trajectory_vr + slot) + bplus_plane), bpvr, state_mask);
                bsk::st(((trajectory_vi + slot) + bplus_plane), bpvi, state_mask);
                bsk::st(((trajectory_tr + slot) + bplus_plane), bptr, state_mask);
                bsk::st(((trajectory_ti + slot) + bplus_plane), bpti, state_mask);
                bsk::st(((trajectory_vr + slot) + bminus_plane), bmvr, state_mask);
                bsk::st(((trajectory_vi + slot) + bminus_plane), bmvi, state_mask);
                bsk::st(((trajectory_tr + slot) + bminus_plane), bmtr, state_mask);
                bsk::st(((trajectory_ti + slot) + bminus_plane), bmti, state_mask);
            }
            if (bsk::truth((pools == 3))) {
                bsk::st(((trajectory_vr + slot) + semisolid_plane), cvr, state_mask);
                bsk::st(((trajectory_vi + slot) + semisolid_plane), cvi, state_mask);
                bsk::st(((trajectory_tr + slot) + semisolid_plane), ctr, state_mask);
                bsk::st(((trajectory_ti + slot) + semisolid_plane), cti, state_mask);
            }
            dt_value = _event_value(duration, event_base, event, active_atom, single_train);
            dt_tangent = _event_value(dot_duration, event_base, event, active_atom, single_train);
            wout_value = 1.0f;
            wout_tangent = 0.0f;
            if (bsk::truth(moving)) {
                auto t0_ = _washout_jvp(atom_washout, d_washout, dt_value, dt_tangent);
                wout_value = bsk::get<0>(t0_);
                wout_tangent = bsk::get<1>(t0_);
            }
            dry1_value = bsk::exp(((-r1_value) * dt_value));
            dry1_tangent = ((-dry1_value) * ((r1_value * dt_tangent) + (r1_tangent * dt_value)));
            dry2_value = bsk::exp(((-r2_value) * dt_value));
            dry2_tangent = ((-dry2_value) * ((r2_value * dt_tangent) + (r2_tangent * dt_value)));
            e1_value = (dry1_value * wout_value);
            e1_tangent = ((dry1_tangent * wout_value) + (dry1_value * wout_tangent));
            e2_value = (dry2_value * wout_value);
            e2_tangent = ((dry2_tangent * wout_value) + (dry2_value * wout_tangent));
            damp_z = 1.0f;
            damp_z_tangent = 0.0f;
            damp_t = 1.0f;
            damp_t_tangent = 0.0f;
            if (bsk::truth(diffusing)) {
                auto t1_ = _damping_jvp(atom_damping, d_damping, dt_value, dt_tangent, order);
                damp_z = bsk::get<0>(t1_);
                damp_z_tangent = bsk::get<1>(t1_);
                damp_t = bsk::get<2>(t1_);
                damp_t_tangent = bsk::get<3>(t1_);
            }
            // Order zero is undamped, so recovery keeps the bare longitudinal factor.
            auto t2_ = bsk::make_tup((1.0f - e1_value), (-e1_tangent));
            recovery_value = bsk::get<0>(t2_);
            recovery_tangent = bsk::get<1>(t2_);
            auto t3_ = bsk::make_tup(e1_value, e1_tangent);
            bare1_value = bsk::get<0>(t3_);
            bare1_tangent = bsk::get<1>(t3_);
            auto t4_ = bsk::make_tup(e2_value, e2_tangent);
            bare2_value = bsk::get<0>(t4_);
            bare2_tangent = bsk::get<1>(t4_);
            e1_tangent = ((e1_tangent * damp_z) + (bare1_value * damp_z_tangent));
            e1_value = (bare1_value * damp_z);
            e2_tangent = ((e2_tangent * damp_t) + (bare2_value * damp_t_tangent));
            e2_value = (bare2_value * damp_t);
            turn_t = 0.0f;
            dturn_t = 0.0f;
            auto t5_ = bsk::make_tup(1.0f, 0.0f, 0.0f, 0.0f);
            szr = bsk::get<0>(t5_);
            szi = bsk::get<1>(t5_);
            sztr = bsk::get<2>(t5_);
            szti = bsk::get<3>(t5_);
            if (bsk::truth(moving)) {
                auto t6_ = _flow(atom_flow, dt_value, order);
                turn_z = bsk::get<0>(t6_);
                turn_t = bsk::get<1>(t6_);
                d_turn = ((d_flow * dt_value) + (atom_flow * dt_tangent));
                dturn_z = ((-order) * d_turn);
                dturn_t = ((-(order + 0.5f)) * d_turn);
                auto t7_ = _dual_polar(turn_z, dturn_z);
                szr = bsk::get<0>(t7_);
                szi = bsk::get<1>(t7_);
                sztr = bsk::get<2>(t7_);
                szti = bsk::get<3>(t7_);
            }
            auto t8_ = bsk::make_tup(1.0f, 0.0f, 0.0f, 0.0f);
            qr = bsk::get<0>(t8_);
            qi = bsk::get<1>(t8_);
            qtr = bsk::get<2>(t8_);
            qti = bsk::get<3>(t8_);
            if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
                angle_value = ((-6.283185307179586f * (atom_b0 * dt_value)) + turn_t);
                angle_tangent = ((-6.283185307179586f * ((d_b0 * dt_value) + (atom_b0 * dt_tangent))) + dturn_t);
                auto t9_ = _dual_polar(angle_value, angle_tangent);
                qr = bsk::get<0>(t9_);
                qi = bsk::get<1>(t9_);
                qtr = bsk::get<2>(t9_);
                qti = bsk::get<3>(t9_);
            }
            auto t10_ = _dual_scale(e2_value, e2_tangent, qr, qi, qtr, qti);
            ovr = bsk::get<0>(t10_);
            ovi = bsk::get<1>(t10_);
            otr = bsk::get<2>(t10_);
            oti = bsk::get<3>(t10_);
            auto t11_ = _dual_scale(e1_value, e1_tangent, szr, szi, sztr, szti);
            lvr = bsk::get<0>(t11_);
            lvi = bsk::get<1>(t11_);
            ltr = bsk::get<2>(t11_);
            lti = bsk::get<3>(t11_);
            // The damping and the off-resonance turn both pools take; with an
            // exchanging one the relaxation itself sits inside the operator instead
            // of in the scalar the free pool alone multiplies by.
            carried = _dual_scale(damp_t, damp_t_tangent, qr, qi, qtr, qti);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                across = _two_pool_transverse_step_jvp(r2_value, r2_tangent, r2b_value, r2b_tangent, atom_exchange, d_exchange, atom_bound, d_boundf, atom_free, d_free, atom_shift, d_shift, dt_value, dt_tangent, wout_value, wout_tangent);
                a11 = bsk::make_tup(bsk::get<0>(across), bsk::get<1>(across), bsk::get<8>(across), bsk::get<9>(across));
                a12 = bsk::make_tup(bsk::get<2>(across), bsk::get<3>(across), bsk::get<10>(across), bsk::get<11>(across));
                a21 = bsk::make_tup(bsk::get<4>(across), bsk::get<5>(across), bsk::get<12>(across), bsk::get<13>(across));
                a22 = bsk::make_tup(bsk::get<6>(across), bsk::get<7>(across), bsk::get<14>(across), bsk::get<15>(across));
                free_plus = bsk::make_tup(pvr, pvi, ptr, pti);
                pool_plus = bsk::make_tup(bpvr, bpvi, bptr, bpti);
                free_minus = bsk::make_tup(mvr, mvi, mtr, mti);
                pool_minus = bsk::make_tup(bmvr, bmvi, bmtr, bmti);
                conjugated = _dual_conj(carried);
                // ``F-`` takes the conjugate of the operator entry by entry, not
                // its transpose: it is the conjugate state following the conjugate
                // map.
                auto t12_ = _dual_product(_dual_add(_dual_product(a11, free_plus), _dual_product(a12, pool_plus)), carried);
                pvr = bsk::get<0>(t12_);
                pvi = bsk::get<1>(t12_);
                ptr = bsk::get<2>(t12_);
                pti = bsk::get<3>(t12_);
                auto t13_ = _dual_product(_dual_add(_dual_product(a21, free_plus), _dual_product(a22, pool_plus)), carried);
                bpvr = bsk::get<0>(t13_);
                bpvi = bsk::get<1>(t13_);
                bptr = bsk::get<2>(t13_);
                bpti = bsk::get<3>(t13_);
                auto t14_ = _dual_product(_dual_add(_dual_product(_dual_conj(a11), free_minus), _dual_product(_dual_conj(a12), pool_minus)), conjugated);
                mvr = bsk::get<0>(t14_);
                mvi = bsk::get<1>(t14_);
                mtr = bsk::get<2>(t14_);
                mti = bsk::get<3>(t14_);
                auto t15_ = _dual_product(_dual_add(_dual_product(_dual_conj(a21), free_minus), _dual_product(_dual_conj(a22), pool_minus)), conjugated);
                bmvr = bsk::get<0>(t15_);
                bmvi = bsk::get<1>(t15_);
                bmtr = bsk::get<2>(t15_);
                bmti = bsk::get<3>(t15_);
            } else {
                auto t16_ = _dual_mul(ovr, ovi, otr, oti, pvr, pvi, ptr, pti);
                pvr = bsk::get<0>(t16_);
                pvi = bsk::get<1>(t16_);
                ptr = bsk::get<2>(t16_);
                pti = bsk::get<3>(t16_);
                auto t17_ = _dual_mul(ovr, (-ovi), otr, (-oti), mvr, mvi, mtr, mti);
                mvr = bsk::get<0>(t17_);
                mvi = bsk::get<1>(t17_);
                mtr = bsk::get<2>(t17_);
                mti = bsk::get<3>(t17_);
            }
            if (bsk::truth((pools == 3))) {
                // Three pools mix through a 3x3 formed in double, tangent and all:
                // a direction through an operator this ill-conditioned needs the
                // width as much as the value does.
                if (bsk::truth(tabulated)) {
                    auto t18_ = _three_pool_from_table_jvp(pool_table, bsk::ld(((duration_row + event_base) + event), active_atom, 0), atom, atom_count, active_atom, r1_value, r1b_value, r1c_value, atom_exchange, atom_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, dt_tangent, wout_value, wout_tangent);
                    t11 = bsk::get<0>(t18_);
                    t12 = bsk::get<1>(t18_);
                    t13 = bsk::get<2>(t18_);
                    t21 = bsk::get<3>(t18_);
                    t22 = bsk::get<4>(t18_);
                    t23 = bsk::get<5>(t18_);
                    t31 = bsk::get<6>(t18_);
                    t32 = bsk::get<7>(t18_);
                    t33 = bsk::get<8>(t18_);
                    grow_free = bsk::get<9>(t18_);
                    grow_pool_b = bsk::get<10>(t18_);
                    grow_semisolid = bsk::get<11>(t18_);
                    d_t11 = bsk::get<12>(t18_);
                    d_t12 = bsk::get<13>(t18_);
                    d_t13 = bsk::get<14>(t18_);
                    d_t21 = bsk::get<15>(t18_);
                    d_t22 = bsk::get<16>(t18_);
                    d_t23 = bsk::get<17>(t18_);
                    d_t31 = bsk::get<18>(t18_);
                    d_t32 = bsk::get<19>(t18_);
                    d_t33 = bsk::get<20>(t18_);
                    d_grow_free = bsk::get<21>(t18_);
                    d_grow_pool_b = bsk::get<22>(t18_);
                    d_grow_semisolid = bsk::get<23>(t18_);
                } else {
                    auto t19_ = _three_pool_step_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, r1c_value, r1c_tangent, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, dt_value, dt_tangent, wout_value, wout_tangent, narrow);
                    t11 = bsk::get<0>(t19_);
                    t12 = bsk::get<1>(t19_);
                    t13 = bsk::get<2>(t19_);
                    t21 = bsk::get<3>(t19_);
                    t22 = bsk::get<4>(t19_);
                    t23 = bsk::get<5>(t19_);
                    t31 = bsk::get<6>(t19_);
                    t32 = bsk::get<7>(t19_);
                    t33 = bsk::get<8>(t19_);
                    grow_free = bsk::get<9>(t19_);
                    grow_pool_b = bsk::get<10>(t19_);
                    grow_semisolid = bsk::get<11>(t19_);
                    d_t11 = bsk::get<12>(t19_);
                    d_t12 = bsk::get<13>(t19_);
                    d_t13 = bsk::get<14>(t19_);
                    d_t21 = bsk::get<15>(t19_);
                    d_t22 = bsk::get<16>(t19_);
                    d_t23 = bsk::get<17>(t19_);
                    d_t31 = bsk::get<18>(t19_);
                    d_t32 = bsk::get<19>(t19_);
                    d_t33 = bsk::get<20>(t19_);
                    d_grow_free = bsk::get<21>(t19_);
                    d_grow_pool_b = bsk::get<22>(t19_);
                    d_grow_semisolid = bsk::get<23>(t19_);
                }
                spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti);
                auto was_free = bsk::make_tup(zvr, zvi, ztr, zti);
                auto was_pool_b = bsk::make_tup(bvr, bvi, btr, bti);
                auto was_semisolid = bsk::make_tup(cvr, cvi, ctr, cti);
                mixed_free = _dual_add(_dual_add(_dual_scale(t11, d_t11, bsk::get<0>(was_free), bsk::get<1>(was_free), bsk::get<2>(was_free), bsk::get<3>(was_free)), _dual_scale(t12, d_t12, bsk::get<0>(was_pool_b), bsk::get<1>(was_pool_b), bsk::get<2>(was_pool_b), bsk::get<3>(was_pool_b))), _dual_scale(t13, d_t13, bsk::get<0>(was_semisolid), bsk::get<1>(was_semisolid), bsk::get<2>(was_semisolid), bsk::get<3>(was_semisolid)));
                auto mixed_pool_b = _dual_add(_dual_add(_dual_scale(t21, d_t21, bsk::get<0>(was_free), bsk::get<1>(was_free), bsk::get<2>(was_free), bsk::get<3>(was_free)), _dual_scale(t22, d_t22, bsk::get<0>(was_pool_b), bsk::get<1>(was_pool_b), bsk::get<2>(was_pool_b), bsk::get<3>(was_pool_b))), _dual_scale(t23, d_t23, bsk::get<0>(was_semisolid), bsk::get<1>(was_semisolid), bsk::get<2>(was_semisolid), bsk::get<3>(was_semisolid)));
                mixed_semisolid = _dual_add(_dual_add(_dual_scale(t31, d_t31, bsk::get<0>(was_free), bsk::get<1>(was_free), bsk::get<2>(was_free), bsk::get<3>(was_free)), _dual_scale(t32, d_t32, bsk::get<0>(was_pool_b), bsk::get<1>(was_pool_b), bsk::get<2>(was_pool_b), bsk::get<3>(was_pool_b))), _dual_scale(t33, d_t33, bsk::get<0>(was_semisolid), bsk::get<1>(was_semisolid), bsk::get<2>(was_semisolid), bsk::get<3>(was_semisolid)));
                auto t20_ = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(mixed_free), bsk::get<1>(mixed_free), bsk::get<2>(mixed_free), bsk::get<3>(mixed_free));
                zvr = bsk::get<0>(t20_);
                zvi = bsk::get<1>(t20_);
                ztr = bsk::get<2>(t20_);
                zti = bsk::get<3>(t20_);
                auto t21_ = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(mixed_pool_b), bsk::get<1>(mixed_pool_b), bsk::get<2>(mixed_pool_b), bsk::get<3>(mixed_pool_b));
                bvr = bsk::get<0>(t21_);
                bvi = bsk::get<1>(t21_);
                btr = bsk::get<2>(t21_);
                bti = bsk::get<3>(t21_);
                auto t22_ = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(mixed_semisolid), bsk::get<1>(mixed_semisolid), bsk::get<2>(mixed_semisolid), bsk::get<3>(mixed_semisolid));
                cvr = bsk::get<0>(t22_);
                cvi = bsk::get<1>(t22_);
                ctr = bsk::get<2>(t22_);
                cti = bsk::get<3>(t22_);
                zvr = (zvr + bsk::where((state == 0), grow_free, 0.0f));
                ztr = (ztr + bsk::where((state == 0), d_grow_free, 0.0f));
                bvr = (bvr + bsk::where((state == 0), grow_pool_b, 0.0f));
                btr = (btr + bsk::where((state == 0), d_grow_pool_b, 0.0f));
                cvr = (cvr + bsk::where((state == 0), grow_semisolid, 0.0f));
                ctr = (ctr + bsk::where((state == 0), d_grow_semisolid, 0.0f));
            } else if (bsk::truth((pools > 0))) {
                // The exchange operator is a property of the interval, not of a
                // dephasing order, so it is formed once and the per-order damping
                // and turn multiply it.
                auto t23_ = _two_pool_step_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, atom_exchange, d_exchange, atom_bound, d_boundf, dt_value, dt_tangent, wout_value, wout_tangent);
                pe11 = bsk::get<0>(t23_);
                pe12 = bsk::get<1>(t23_);
                pe21 = bsk::get<2>(t23_);
                pe22 = bsk::get<3>(t23_);
                prec_f = bsk::get<4>(t23_);
                prec_b = bsk::get<5>(t23_);
                de11 = bsk::get<6>(t23_);
                de12 = bsk::get<7>(t23_);
                de21 = bsk::get<8>(t23_);
                de22 = bsk::get<9>(t23_);
                drec_f = bsk::get<10>(t23_);
                drec_b = bsk::get<11>(t23_);
                spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti);
                free_part = _dual_scale(pe11, de11, zvr, zvi, ztr, zti);
                cross_in = _dual_scale(pe12, de12, bvr, bvi, btr, bti);
                cross_out = _dual_scale(pe21, de21, zvr, zvi, ztr, zti);
                bound_part = _dual_scale(pe22, de22, bvr, bvi, btr, bti);
                auto t24_ = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), (bsk::get<0>(free_part) + bsk::get<0>(cross_in)), (bsk::get<1>(free_part) + bsk::get<1>(cross_in)), (bsk::get<2>(free_part) + bsk::get<2>(cross_in)), (bsk::get<3>(free_part) + bsk::get<3>(cross_in)));
                zvr = bsk::get<0>(t24_);
                zvi = bsk::get<1>(t24_);
                ztr = bsk::get<2>(t24_);
                zti = bsk::get<3>(t24_);
                auto t25_ = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), (bsk::get<0>(cross_out) + bsk::get<0>(bound_part)), (bsk::get<1>(cross_out) + bsk::get<1>(bound_part)), (bsk::get<2>(cross_out) + bsk::get<2>(bound_part)), (bsk::get<3>(cross_out) + bsk::get<3>(bound_part)));
                bvr = bsk::get<0>(t25_);
                bvi = bsk::get<1>(t25_);
                btr = bsk::get<2>(t25_);
                bti = bsk::get<3>(t25_);
                zvr = (zvr + bsk::where((state == 0), prec_f, 0.0f));
                ztr = (ztr + bsk::where((state == 0), drec_f, 0.0f));
                bvr = (bvr + bsk::where((state == 0), prec_b, 0.0f));
                btr = (btr + bsk::where((state == 0), drec_b, 0.0f));
            } else {
                auto t26_ = _dual_mul(lvr, lvi, ltr, lti, zvr, zvi, ztr, zti);
                zvr = bsk::get<0>(t26_);
                zvi = bsk::get<1>(t26_);
                ztr = bsk::get<2>(t26_);
                zti = bsk::get<3>(t26_);
                zvr = (zvr + bsk::where((state == 0), recovery_value, 0.0f));
                ztr = (ztr + bsk::where((state == 0), recovery_tangent, 0.0f));
            }
            event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
            pre_shift = (bsk::band(event_action, 1) != 0);
            auto t27_ = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count);
            svr = bsk::get<0>(t27_);
            svi = bsk::get<1>(t27_);
            wvr = bsk::get<2>(t27_);
            wvi = bsk::get<3>(t27_);
            auto t28_ = _shift(ptr, pti, mtr, mti, state, state_mask, state_count);
            str_ = bsk::get<0>(t28_);
            sti = bsk::get<1>(t28_);
            wtr = bsk::get<2>(t28_);
            wti = bsk::get<3>(t28_);
            pvr = bsk::where(pre_shift, svr, pvr);
            pvi = bsk::where(pre_shift, svi, pvi);
            ptr = bsk::where(pre_shift, str_, ptr);
            pti = bsk::where(pre_shift, sti, pti);
            mvr = bsk::where(pre_shift, wvr, mvr);
            mvi = bsk::where(pre_shift, wvi, mvi);
            mtr = bsk::where(pre_shift, wtr, mtr);
            mti = bsk::where(pre_shift, wti, mti);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t29_ = _shift(bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count);
                svr = bsk::get<0>(t29_);
                svi = bsk::get<1>(t29_);
                wvr = bsk::get<2>(t29_);
                wvi = bsk::get<3>(t29_);
                auto t30_ = _shift(bptr, bpti, bmtr, bmti, state, state_mask, state_count);
                str_ = bsk::get<0>(t30_);
                sti = bsk::get<1>(t30_);
                wtr = bsk::get<2>(t30_);
                wti = bsk::get<3>(t30_);
                bpvr = bsk::where(pre_shift, svr, bpvr);
                bpvi = bsk::where(pre_shift, svi, bpvi);
                bptr = bsk::where(pre_shift, str_, bptr);
                bpti = bsk::where(pre_shift, sti, bpti);
                bmvr = bsk::where(pre_shift, wvr, bmvr);
                bmvi = bsk::where(pre_shift, wvi, bmvi);
                bmtr = bsk::where(pre_shift, wtr, bmtr);
                bmti = bsk::where(pre_shift, wti, bmti);
            }
            event_kind = bsk::ld((kind + event));
            is_rf = (event_kind == 1);
            is_inversion = (bsk::band(event_action, 4) != 0);
            invert = bsk::band(is_rf, is_inversion);
            auto t31_ = _dual_scale((-atom_inv), (-d_inv), zvr, zvi, ztr, zti);
            ivr = bsk::get<0>(t31_);
            ivi = bsk::get<1>(t31_);
            itr = bsk::get<2>(t31_);
            iti = bsk::get<3>(t31_);
            zvr = bsk::where(invert, ivr, zvr);
            zvi = bsk::where(invert, ivi, zvi);
            ztr = bsk::where(invert, itr, ztr);
            zti = bsk::where(invert, iti, zti);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                // A semisolid pool is saturated by an adiabatic sweep rather than
                // turned over; a chemically exchanging one is free water and
                // inverts like any other.
                auto t32_ = _dual_scale((-atom_inv), (-d_inv), bvr, bvi, btr, bti);
                ivr = bsk::get<0>(t32_);
                ivi = bsk::get<1>(t32_);
                itr = bsk::get<2>(t32_);
                iti = bsk::get<3>(t32_);
                bvr = bsk::where(invert, ivr, bvr);
                bvi = bsk::where(invert, ivi, bvi);
                btr = bsk::where(invert, itr, btr);
                bti = bsk::where(invert, iti, bti);
            }
            event_flip = _event_value(flip, event_base, event, active_atom, single_train);
            event_dot_flip = _event_value(dot_flip, event_base, event, active_atom, single_train);
            event_phase = _event_value(phase, event_base, event, active_atom, single_train);
            event_dot_phase = _event_value(dot_phase, event_base, event, active_atom, single_train);
            // One shim is the whole sequence's transmit field, loaded once above;
            // several give each pulse a row of its own.
            if (bsk::truth(shimmed)) {
                row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
                atom_b1 = 1.0f;
                if (bsk::truth(transmit)) {
                    atom_b1 = bsk::ld(((b1 + row) + atom), active_atom, 1.0f);
                }
                if (bsk::truth(off_axis)) {
                    atom_b1_phase = bsk::ld(((b1_phase + row) + atom), active_atom, 0.0f);
                }
                d_b1 = bsk::ld(((dot_b1 + row) + atom), active_atom, 0.0f);
                if (bsk::truth(off_axis)) {
                    d_b1_phase = bsk::ld(((dot_b1_phase + row) + atom), active_atom, 0.0f);
                }
            }
            alpha_value = (event_flip * atom_b1);
            alpha_tangent = ((event_dot_flip * atom_b1) + (event_flip * d_b1));
            phi_value = (event_phase + atom_b1_phase);
            phi_tangent = (event_dot_phase + d_b1_phase);
            if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
                // The semisolid pool absorbs the power the pulse deposits, so it
                // reads the bare flip the transmit field gives the voxel -- not the
                // slice-shaped rotation the free pool takes from the table.
                offset_value = (bsk::ld((rf_frequency + event)) - atom_b0);
                auto t33_ = _lineshape_at_slope(lineshape, offset_value, lineshape_bins, lineshape_step);
                shape_value = bsk::get<0>(t33_);
                shape_slope = bsk::get<1>(t33_);
                shape_tangent = (shape_slope * (-d_b0));
                event_saturation = bsk::ld((saturation + event));
                power_value = ((event_saturation * alpha_value) * alpha_value);
                power_tangent = (((event_saturation * 2.0f) * alpha_value) * alpha_tangent);
                absorbed_value = bsk::exp((power_value * shape_value));
                absorbed_tangent = (absorbed_value * ((power_tangent * shape_value) + (power_value * shape_tangent)));
                saturating = bsk::band(is_rf, bsk::bnot(is_inversion));
                if (bsk::truth((pools == 1))) {
                    auto sat_b = _dual_scale(absorbed_value, absorbed_tangent, bvr, bvi, btr, bti);
                    bvr = bsk::where(saturating, bsk::get<0>(sat_b), bvr);
                    bvi = bsk::where(saturating, bsk::get<1>(sat_b), bvi);
                    btr = bsk::where(saturating, bsk::get<2>(sat_b), btr);
                    bti = bsk::where(saturating, bsk::get<3>(sat_b), bti);
                } else {
                    auto sat_c = _dual_scale(absorbed_value, absorbed_tangent, cvr, cvi, ctr, cti);
                    cvr = bsk::where(saturating, bsk::get<0>(sat_c), cvr);
                    cvi = bsk::where(saturating, bsk::get<1>(sat_c), cvi);
                    ctr = bsk::where(saturating, bsk::get<2>(sat_c), ctr);
                    cti = bsk::where(saturating, bsk::get<3>(sat_c), cti);
                }
            }
            cos_value = bsk::cos(alpha_value);
            sin_value = bsk::sin(alpha_value);
            cos_tangent = ((-sin_value) * alpha_tangent);
            sin_tangent = (cos_value * alpha_tangent);
            auto t34_ = _dual_polar(phi_value, phi_tangent);
            p1r = bsk::get<0>(t34_);
            p1i = bsk::get<1>(t34_);
            p1tr = bsk::get<2>(t34_);
            p1ti = bsk::get<3>(t34_);
            auto t35_ = _dual_mul(p1r, p1i, p1tr, p1ti, p1r, p1i, p1tr, p1ti);
            p2r = bsk::get<0>(t35_);
            p2i = bsk::get<1>(t35_);
            p2tr = bsk::get<2>(t35_);
            p2ti = bsk::get<3>(t35_);
            auto t36_ = _rotation_block((0.5f * (1.0f + cos_value)), (0.5f * cos_tangent), (0.5f * (1.0f - cos_value)), (-0.5f * cos_tangent), sin_value, sin_tangent, cos_value, cos_tangent, p1r, p1i, p1tr, p1ti, p2r, p2i, p2tr, p2ti, p1r, (-p1i), p1tr, (-p1ti));
            t00 = bsk::get<0>(t36_);
            t01 = bsk::get<1>(t36_);
            t02 = bsk::get<2>(t36_);
            r12 = bsk::get<3>(t36_);
            t20 = bsk::get<4>(t36_);
            r21 = bsk::get<5>(t36_);
            r22 = bsk::get<6>(t36_);
            auto a0 = _dual_mul(bsk::get<0>(t00), bsk::get<1>(t00), bsk::get<2>(t00), bsk::get<3>(t00), pvr, pvi, ptr, pti);
            auto a1 = _dual_mul(bsk::get<0>(t01), bsk::get<1>(t01), bsk::get<2>(t01), bsk::get<3>(t01), mvr, mvi, mtr, mti);
            auto a2 = _dual_mul(bsk::get<0>(t02), bsk::get<1>(t02), bsk::get<2>(t02), bsk::get<3>(t02), zvr, zvi, ztr, zti);
            auto b0_ = _dual_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bsk::get<2>(t01), (-bsk::get<3>(t01)), pvr, pvi, ptr, pti);
            auto b1_ = _dual_mul(bsk::get<0>(t00), bsk::get<1>(t00), bsk::get<2>(t00), bsk::get<3>(t00), mvr, mvi, mtr, mti);
            auto b2 = _dual_mul(bsk::get<0>(r12), bsk::get<1>(r12), bsk::get<2>(r12), bsk::get<3>(r12), zvr, zvi, ztr, zti);
            auto c0 = _dual_mul(bsk::get<0>(t20), bsk::get<1>(t20), bsk::get<2>(t20), bsk::get<3>(t20), pvr, pvi, ptr, pti);
            auto c1 = _dual_mul(bsk::get<0>(r21), bsk::get<1>(r21), bsk::get<2>(r21), bsk::get<3>(r21), mvr, mvi, mtr, mti);
            auto c2 = _dual_mul(bsk::get<0>(r22), bsk::get<1>(r22), bsk::get<2>(r22), bsk::get<3>(r22), zvr, zvi, ztr, zti);
            turned_pvr = ((bsk::get<0>(a0) + bsk::get<0>(a1)) + bsk::get<0>(a2));
            turned_pvi = ((bsk::get<1>(a0) + bsk::get<1>(a1)) + bsk::get<1>(a2));
            turned_ptr = ((bsk::get<2>(a0) + bsk::get<2>(a1)) + bsk::get<2>(a2));
            turned_pti = ((bsk::get<3>(a0) + bsk::get<3>(a1)) + bsk::get<3>(a2));
            turned_mvr = ((bsk::get<0>(b0_) + bsk::get<0>(b1_)) + bsk::get<0>(b2));
            turned_mvi = ((bsk::get<1>(b0_) + bsk::get<1>(b1_)) + bsk::get<1>(b2));
            turned_mtr = ((bsk::get<2>(b0_) + bsk::get<2>(b1_)) + bsk::get<2>(b2));
            turned_mti = ((bsk::get<3>(b0_) + bsk::get<3>(b1_)) + bsk::get<3>(b2));
            turned_zvr = ((bsk::get<0>(c0) + bsk::get<0>(c1)) + bsk::get<0>(c2));
            turned_zvi = ((bsk::get<1>(c0) + bsk::get<1>(c1)) + bsk::get<1>(c2));
            turned_ztr = ((bsk::get<2>(c0) + bsk::get<2>(c1)) + bsk::get<2>(c2));
            turned_zti = ((bsk::get<3>(c0) + bsk::get<3>(c1)) + bsk::get<3>(c2));
            if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                if (bsk::truth(dynamic)) {
                    auto t37_ = _dynamic_pair_dual_at(pairs, pair_direction, pair_index, event_base, event, atom, atom_count, active_atom, phi_value, phi_tangent, directed);
                    shaped_a = bsk::get<0>(t37_);
                    shaped_b = bsk::get<1>(t37_);
                } else {
                    auto t38_ = _profiled_pair_dual(profile, _table_row(profile_index, event, location, locations), alpha_value, alpha_tangent, phi_value, phi_tangent, profile_bins, profile_step);
                    shaped_a = bsk::get<0>(t38_);
                    shaped_b = bsk::get<1>(t38_);
                }
                auto t39_ = _rotate_spinor_dual(bsk::get<0>(shaped_a), bsk::get<1>(shaped_a), bsk::get<0>(shaped_b), bsk::get<1>(shaped_b), bsk::get<2>(shaped_a), bsk::get<3>(shaped_a), bsk::get<2>(shaped_b), bsk::get<3>(shaped_b), pvr, pvi, mvr, mvi, zvr, zvi, ptr, pti, mtr, mti, ztr, zti);
                turned_pvr = bsk::get<0>(t39_);
                turned_pvi = bsk::get<1>(t39_);
                turned_mvr = bsk::get<2>(t39_);
                turned_mvi = bsk::get<3>(t39_);
                turned_zvr = bsk::get<4>(t39_);
                turned_zvi = bsk::get<5>(t39_);
                turned_ptr = bsk::get<6>(t39_);
                turned_pti = bsk::get<7>(t39_);
                turned_mtr = bsk::get<8>(t39_);
                turned_mti = bsk::get<9>(t39_);
                turned_ztr = bsk::get<10>(t39_);
                turned_zti = bsk::get<11>(t39_);
            }
            rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                // The same pulse, the same rotation. A chemical shift moves where a
                // pool precesses, not what a pulse does to it.
                auto e0 = _dual_mul(bsk::get<0>(t00), bsk::get<1>(t00), bsk::get<2>(t00), bsk::get<3>(t00), bpvr, bpvi, bptr, bpti);
                auto e1_ = _dual_mul(bsk::get<0>(t01), bsk::get<1>(t01), bsk::get<2>(t01), bsk::get<3>(t01), bmvr, bmvi, bmtr, bmti);
                auto e2_ = _dual_mul(bsk::get<0>(t02), bsk::get<1>(t02), bsk::get<2>(t02), bsk::get<3>(t02), bvr, bvi, btr, bti);
                auto f0 = _dual_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bsk::get<2>(t01), (-bsk::get<3>(t01)), bpvr, bpvi, bptr, bpti);
                auto f1 = _dual_mul(bsk::get<0>(t00), bsk::get<1>(t00), bsk::get<2>(t00), bsk::get<3>(t00), bmvr, bmvi, bmtr, bmti);
                auto f2 = _dual_mul(bsk::get<0>(r12), bsk::get<1>(r12), bsk::get<2>(r12), bsk::get<3>(r12), bvr, bvi, btr, bti);
                auto h0 = _dual_mul(bsk::get<0>(t20), bsk::get<1>(t20), bsk::get<2>(t20), bsk::get<3>(t20), bpvr, bpvi, bptr, bpti);
                auto h1 = _dual_mul(bsk::get<0>(r21), bsk::get<1>(r21), bsk::get<2>(r21), bsk::get<3>(r21), bmvr, bmvi, bmtr, bmti);
                auto h2 = _dual_mul(bsk::get<0>(r22), bsk::get<1>(r22), bsk::get<2>(r22), bsk::get<3>(r22), bvr, bvi, btr, bti);
                spun_pvr = ((bsk::get<0>(e0) + bsk::get<0>(e1_)) + bsk::get<0>(e2_));
                spun_pvi = ((bsk::get<1>(e0) + bsk::get<1>(e1_)) + bsk::get<1>(e2_));
                spun_ptr = ((bsk::get<2>(e0) + bsk::get<2>(e1_)) + bsk::get<2>(e2_));
                spun_pti = ((bsk::get<3>(e0) + bsk::get<3>(e1_)) + bsk::get<3>(e2_));
                spun_mvr = ((bsk::get<0>(f0) + bsk::get<0>(f1)) + bsk::get<0>(f2));
                spun_mvi = ((bsk::get<1>(f0) + bsk::get<1>(f1)) + bsk::get<1>(f2));
                spun_mtr = ((bsk::get<2>(f0) + bsk::get<2>(f1)) + bsk::get<2>(f2));
                spun_mti = ((bsk::get<3>(f0) + bsk::get<3>(f1)) + bsk::get<3>(f2));
                spun_zvr = ((bsk::get<0>(h0) + bsk::get<0>(h1)) + bsk::get<0>(h2));
                spun_zvi = ((bsk::get<1>(h0) + bsk::get<1>(h1)) + bsk::get<1>(h2));
                spun_ztr = ((bsk::get<2>(h0) + bsk::get<2>(h1)) + bsk::get<2>(h2));
                spun_zti = ((bsk::get<3>(h0) + bsk::get<3>(h1)) + bsk::get<3>(h2));
                if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                    auto t40_ = _rotate_spinor_dual(bsk::get<0>(shaped_a), bsk::get<1>(shaped_a), bsk::get<0>(shaped_b), bsk::get<1>(shaped_b), bsk::get<2>(shaped_a), bsk::get<3>(shaped_a), bsk::get<2>(shaped_b), bsk::get<3>(shaped_b), bpvr, bpvi, bmvr, bmvi, bvr, bvi, bptr, bpti, bmtr, bmti, btr, bti);
                    spun_pvr = bsk::get<0>(t40_);
                    spun_pvi = bsk::get<1>(t40_);
                    spun_mvr = bsk::get<2>(t40_);
                    spun_mvi = bsk::get<3>(t40_);
                    spun_zvr = bsk::get<4>(t40_);
                    spun_zvi = bsk::get<5>(t40_);
                    spun_ptr = bsk::get<6>(t40_);
                    spun_pti = bsk::get<7>(t40_);
                    spun_mtr = bsk::get<8>(t40_);
                    spun_mti = bsk::get<9>(t40_);
                    spun_ztr = bsk::get<10>(t40_);
                    spun_zti = bsk::get<11>(t40_);
                }
                bpvr = bsk::where(rotate, spun_pvr, bpvr);
                bpvi = bsk::where(rotate, spun_pvi, bpvi);
                bptr = bsk::where(rotate, spun_ptr, bptr);
                bpti = bsk::where(rotate, spun_pti, bpti);
                bmvr = bsk::where(rotate, spun_mvr, bmvr);
                bmvi = bsk::where(rotate, spun_mvi, bmvi);
                bmtr = bsk::where(rotate, spun_mtr, bmtr);
                bmti = bsk::where(rotate, spun_mti, bmti);
                bvr = bsk::where(rotate, spun_zvr, bvr);
                bvi = bsk::where(rotate, spun_zvi, bvi);
                btr = bsk::where(rotate, spun_ztr, btr);
                bti = bsk::where(rotate, spun_zti, bti);
            }
            pvr = bsk::where(rotate, turned_pvr, pvr);
            pvi = bsk::where(rotate, turned_pvi, pvi);
            ptr = bsk::where(rotate, turned_ptr, ptr);
            pti = bsk::where(rotate, turned_pti, pti);
            mvr = bsk::where(rotate, turned_mvr, mvr);
            mvi = bsk::where(rotate, turned_mvi, mvi);
            mtr = bsk::where(rotate, turned_mtr, mtr);
            mti = bsk::where(rotate, turned_mti, mti);
            zvr = bsk::where(rotate, turned_zvr, zvr);
            zvi = bsk::where(rotate, turned_zvi, zvi);
            ztr = bsk::where(rotate, turned_ztr, ztr);
            zti = bsk::where(rotate, turned_zti, zti);
            do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
            auto t41_ = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count);
            svr = bsk::get<0>(t41_);
            svi = bsk::get<1>(t41_);
            wvr = bsk::get<2>(t41_);
            wvi = bsk::get<3>(t41_);
            auto t42_ = _shift(ptr, pti, mtr, mti, state, state_mask, state_count);
            str_ = bsk::get<0>(t42_);
            sti = bsk::get<1>(t42_);
            wtr = bsk::get<2>(t42_);
            wti = bsk::get<3>(t42_);
            pvr = bsk::where(do_shift, svr, pvr);
            pvi = bsk::where(do_shift, svi, pvi);
            ptr = bsk::where(do_shift, str_, ptr);
            pti = bsk::where(do_shift, sti, pti);
            mvr = bsk::where(do_shift, wvr, mvr);
            mvi = bsk::where(do_shift, wvi, mvi);
            mtr = bsk::where(do_shift, wtr, mtr);
            mti = bsk::where(do_shift, wti, mti);
            spoil = (bsk::band(event_action, 8) != 0);
            pvr = bsk::where(spoil, 0.0f, pvr);
            pvi = bsk::where(spoil, 0.0f, pvi);
            ptr = bsk::where(spoil, 0.0f, ptr);
            pti = bsk::where(spoil, 0.0f, pti);
            mvr = bsk::where(spoil, 0.0f, mvr);
            mvi = bsk::where(spoil, 0.0f, mvi);
            mtr = bsk::where(spoil, 0.0f, mtr);
            mti = bsk::where(spoil, 0.0f, mti);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t43_ = _shift(bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count);
                svr = bsk::get<0>(t43_);
                svi = bsk::get<1>(t43_);
                wvr = bsk::get<2>(t43_);
                wvi = bsk::get<3>(t43_);
                auto t44_ = _shift(bptr, bpti, bmtr, bmti, state, state_mask, state_count);
                str_ = bsk::get<0>(t44_);
                sti = bsk::get<1>(t44_);
                wtr = bsk::get<2>(t44_);
                wti = bsk::get<3>(t44_);
                bpvr = bsk::where(spoil, 0.0f, bsk::where(do_shift, svr, bpvr));
                bpvi = bsk::where(spoil, 0.0f, bsk::where(do_shift, svi, bpvi));
                bptr = bsk::where(spoil, 0.0f, bsk::where(do_shift, str_, bptr));
                bpti = bsk::where(spoil, 0.0f, bsk::where(do_shift, sti, bpti));
                bmvr = bsk::where(spoil, 0.0f, bsk::where(do_shift, wvr, bmvr));
                bmvi = bsk::where(spoil, 0.0f, bsk::where(do_shift, wvi, bmvi));
                bmtr = bsk::where(spoil, 0.0f, bsk::where(do_shift, wtr, bmtr));
                bmti = bsk::where(spoil, 0.0f, bsk::where(do_shift, wti, bmti));
            }
        }
        return;
    }
    // ---- reverse ----
    pbvr = empty;
    pbvi = empty;
    pbtr = empty;
    pbti = empty;
    mbvr = empty;
    mbvi = empty;
    mbtr = empty;
    mbti = empty;
    zbvr = empty;
    zbvi = empty;
    zbtr = empty;
    zbti = empty;
    bbvr = empty;
    bbvi = empty;
    bbtr = empty;
    bbti = empty;
    ubvr = empty;
    ubvi = empty;
    ubtr = empty;
    ubti = empty;
    wbvr = empty;
    wbvi = empty;
    wbtr = empty;
    wbti = empty;
    cbvr = empty;
    cbvi = empty;
    cbtr = empty;
    cbti = empty;
    auto zero = bsk::full<float, 2>(0);
    g_boundv = zero;
    g_boundt = zero;
    g_exchv = zero;
    g_excht = zero;
    g_t1bv = zero;
    g_t1bt = zero;
    g_t2bv = zero;
    g_t2bt = zero;
    g_shiftv = zero;
    g_shiftt = zero;
    g_semiv = zero;
    g_semit = zero;
    g_sexchv = zero;
    g_sexcht = zero;
    g_t1cv = zero;
    g_t1ct = zero;
    g_diffv = zero;
    g_difft = zero;
    g_flowv = zero;
    g_flowt = zero;
    g_washv = zero;
    g_washt = zero;
    g_t1v = zero;
    g_t1t = zero;
    g_t2v = zero;
    g_t2t = zero;
    g_m0v = zero;
    g_m0t = zero;
    g_b1v = zero;
    g_b1t = zero;
    g_b1pv = zero;
    g_b1pt = zero;
    g_b0v = zero;
    g_b0t = zero;
    g_invv = zero;
    g_invt = zero;
    for (bsk::index_t reverse = 0; reverse < event_count; reverse += 1) {
        event = ((event_count - 1) - reverse);
        slot = (trajectory + (event * record_stride));
        auto xpvr = bsk::ld((trajectory_vr + slot), state_mask, 0.0f);
        auto xpvi = bsk::ld((trajectory_vi + slot), state_mask, 0.0f);
        auto xptr = bsk::ld((trajectory_tr + slot), state_mask, 0.0f);
        auto xpti = bsk::ld((trajectory_ti + slot), state_mask, 0.0f);
        auto xmvr = bsk::ld(((trajectory_vr + slot) + minus_plane), state_mask, 0.0f);
        auto xmvi = bsk::ld(((trajectory_vi + slot) + minus_plane), state_mask, 0.0f);
        auto xmtr = bsk::ld(((trajectory_tr + slot) + minus_plane), state_mask, 0.0f);
        auto xmti = bsk::ld(((trajectory_ti + slot) + minus_plane), state_mask, 0.0f);
        auto xzvr = bsk::ld(((trajectory_vr + slot) + long_plane), state_mask, 0.0f);
        auto xzvi = bsk::ld(((trajectory_vi + slot) + long_plane), state_mask, 0.0f);
        auto xztr = bsk::ld(((trajectory_tr + slot) + long_plane), state_mask, 0.0f);
        auto xzti = bsk::ld(((trajectory_ti + slot) + long_plane), state_mask, 0.0f);
        xbvr = empty;
        xbvi = empty;
        xbtr = empty;
        xbti = empty;
        xbpvr = empty;
        xbpvi = empty;
        xbptr = empty;
        xbpti = empty;
        xbmvr = empty;
        xbmvi = empty;
        xbmtr = empty;
        xbmti = empty;
        xcvr = empty;
        xcvi = empty;
        xctr = empty;
        xcti = empty;
        if (bsk::truth((pools > 0))) {
            xbvr = bsk::ld(((trajectory_vr + slot) + bound_plane), state_mask, 0.0f);
            xbvi = bsk::ld(((trajectory_vi + slot) + bound_plane), state_mask, 0.0f);
            xbtr = bsk::ld(((trajectory_tr + slot) + bound_plane), state_mask, 0.0f);
            xbti = bsk::ld(((trajectory_ti + slot) + bound_plane), state_mask, 0.0f);
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            xbpvr = bsk::ld(((trajectory_vr + slot) + bplus_plane), state_mask, 0.0f);
            xbpvi = bsk::ld(((trajectory_vi + slot) + bplus_plane), state_mask, 0.0f);
            xbptr = bsk::ld(((trajectory_tr + slot) + bplus_plane), state_mask, 0.0f);
            xbpti = bsk::ld(((trajectory_ti + slot) + bplus_plane), state_mask, 0.0f);
            xbmvr = bsk::ld(((trajectory_vr + slot) + bminus_plane), state_mask, 0.0f);
            xbmvi = bsk::ld(((trajectory_vi + slot) + bminus_plane), state_mask, 0.0f);
            xbmtr = bsk::ld(((trajectory_tr + slot) + bminus_plane), state_mask, 0.0f);
            xbmti = bsk::ld(((trajectory_ti + slot) + bminus_plane), state_mask, 0.0f);
        }
        if (bsk::truth((pools == 3))) {
            xcvr = bsk::ld(((trajectory_vr + slot) + semisolid_plane), state_mask, 0.0f);
            xcvi = bsk::ld(((trajectory_vi + slot) + semisolid_plane), state_mask, 0.0f);
            xctr = bsk::ld(((trajectory_tr + slot) + semisolid_plane), state_mask, 0.0f);
            xcti = bsk::ld(((trajectory_ti + slot) + semisolid_plane), state_mask, 0.0f);
        }
        event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        event_kind = bsk::ld((kind + event));
        dt_value = _event_value(duration, event_base, event, active_atom, single_train);
        dt_tangent = _event_value(dot_duration, event_base, event, active_atom, single_train);
        wout_value = 1.0f;
        wout_tangent = 0.0f;
        if (bsk::truth(moving)) {
            auto t45_ = _washout_jvp(atom_washout, d_washout, dt_value, dt_tangent);
            wout_value = bsk::get<0>(t45_);
            wout_tangent = bsk::get<1>(t45_);
        }
        dry1_value = bsk::exp(((-r1_value) * dt_value));
        dry1_tangent = ((-dry1_value) * ((r1_value * dt_tangent) + (r1_tangent * dt_value)));
        dry2_value = bsk::exp(((-r2_value) * dt_value));
        dry2_tangent = ((-dry2_value) * ((r2_value * dt_tangent) + (r2_tangent * dt_value)));
        e1_value = (dry1_value * wout_value);
        e1_tangent = ((dry1_tangent * wout_value) + (dry1_value * wout_tangent));
        e2_value = (dry2_value * wout_value);
        e2_tangent = ((dry2_tangent * wout_value) + (dry2_value * wout_tangent));
        damp_z = 1.0f;
        damp_z_tangent = 0.0f;
        damp_t = 1.0f;
        damp_t_tangent = 0.0f;
        if (bsk::truth(diffusing)) {
            auto t46_ = _damping_jvp(atom_damping, d_damping, dt_value, dt_tangent, order);
            damp_z = bsk::get<0>(t46_);
            damp_z_tangent = bsk::get<1>(t46_);
            damp_t = bsk::get<2>(t46_);
            damp_t_tangent = bsk::get<3>(t46_);
        }
        // Order zero is undamped, so recovery keeps the bare longitudinal factor.
        auto t47_ = bsk::make_tup((1.0f - e1_value), (-e1_tangent));
        recovery_value = bsk::get<0>(t47_);
        recovery_tangent = bsk::get<1>(t47_);
        auto t48_ = bsk::make_tup(e1_value, e1_tangent);
        bare1_value = bsk::get<0>(t48_);
        bare1_tangent = bsk::get<1>(t48_);
        auto t49_ = bsk::make_tup(e2_value, e2_tangent);
        bare2_value = bsk::get<0>(t49_);
        bare2_tangent = bsk::get<1>(t49_);
        e1_tangent = ((e1_tangent * damp_z) + (bare1_value * damp_z_tangent));
        e1_value = (bare1_value * damp_z);
        e2_tangent = ((e2_tangent * damp_t) + (bare2_value * damp_t_tangent));
        e2_value = (bare2_value * damp_t);
        turn_t = 0.0f;
        dturn_t = 0.0f;
        auto t50_ = bsk::make_tup(1.0f, 0.0f, 0.0f, 0.0f);
        szr = bsk::get<0>(t50_);
        szi = bsk::get<1>(t50_);
        sztr = bsk::get<2>(t50_);
        szti = bsk::get<3>(t50_);
        if (bsk::truth(moving)) {
            auto t51_ = _flow(atom_flow, dt_value, order);
            turn_z = bsk::get<0>(t51_);
            turn_t = bsk::get<1>(t51_);
            d_turn = ((d_flow * dt_value) + (atom_flow * dt_tangent));
            dturn_z = ((-order) * d_turn);
            dturn_t = ((-(order + 0.5f)) * d_turn);
            auto t52_ = _dual_polar(turn_z, dturn_z);
            szr = bsk::get<0>(t52_);
            szi = bsk::get<1>(t52_);
            sztr = bsk::get<2>(t52_);
            szti = bsk::get<3>(t52_);
        }
        auto t53_ = bsk::make_tup(1.0f, 0.0f, 0.0f, 0.0f);
        qr = bsk::get<0>(t53_);
        qi = bsk::get<1>(t53_);
        qtr = bsk::get<2>(t53_);
        qti = bsk::get<3>(t53_);
        if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
            angle_value = ((-6.283185307179586f * (atom_b0 * dt_value)) + turn_t);
            angle_tangent = ((-6.283185307179586f * ((d_b0 * dt_value) + (atom_b0 * dt_tangent))) + dturn_t);
            auto t54_ = _dual_polar(angle_value, angle_tangent);
            qr = bsk::get<0>(t54_);
            qi = bsk::get<1>(t54_);
            qtr = bsk::get<2>(t54_);
            qti = bsk::get<3>(t54_);
        }
        auto t55_ = _dual_scale(e2_value, e2_tangent, qr, qi, qtr, qti);
        ovr = bsk::get<0>(t55_);
        ovi = bsk::get<1>(t55_);
        otr = bsk::get<2>(t55_);
        oti = bsk::get<3>(t55_);
        auto t56_ = _dual_scale(e1_value, e1_tangent, szr, szi, sztr, szti);
        lvr = bsk::get<0>(t56_);
        lvi = bsk::get<1>(t56_);
        ltr = bsk::get<2>(t56_);
        lti = bsk::get<3>(t56_);
        // Replay the intra-event stages from the recorded entry state.
        carried = _dual_scale(damp_t, damp_t_tangent, qr, qi, qtr, qti);
        rbpvr = empty;
        rbpvi = empty;
        rbptr = empty;
        rbpti = empty;
        rbmvr = empty;
        rbmvi = empty;
        rbmtr = empty;
        rbmti = empty;
        a11 = bsk::make_tup(empty, empty, empty, empty);
        a12 = bsk::make_tup(empty, empty, empty, empty);
        a21 = bsk::make_tup(empty, empty, empty, empty);
        a22 = bsk::make_tup(empty, empty, empty, empty);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            across = _two_pool_transverse_step_jvp(r2_value, r2_tangent, r2b_value, r2b_tangent, atom_exchange, d_exchange, atom_bound, d_boundf, atom_free, d_free, atom_shift, d_shift, dt_value, dt_tangent, wout_value, wout_tangent);
            a11 = bsk::make_tup(bsk::get<0>(across), bsk::get<1>(across), bsk::get<8>(across), bsk::get<9>(across));
            a12 = bsk::make_tup(bsk::get<2>(across), bsk::get<3>(across), bsk::get<10>(across), bsk::get<11>(across));
            a21 = bsk::make_tup(bsk::get<4>(across), bsk::get<5>(across), bsk::get<12>(across), bsk::get<13>(across));
            a22 = bsk::make_tup(bsk::get<6>(across), bsk::get<7>(across), bsk::get<14>(across), bsk::get<15>(across));
            free_plus = bsk::make_tup(xpvr, xpvi, xptr, xpti);
            pool_plus = bsk::make_tup(xbpvr, xbpvi, xbptr, xbpti);
            free_minus = bsk::make_tup(xmvr, xmvi, xmtr, xmti);
            pool_minus = bsk::make_tup(xbmvr, xbmvi, xbmtr, xbmti);
            conjugated = _dual_conj(carried);
            auto t57_ = _dual_product(_dual_add(_dual_product(a11, free_plus), _dual_product(a12, pool_plus)), carried);
            rpvr = bsk::get<0>(t57_);
            rpvi = bsk::get<1>(t57_);
            rptr = bsk::get<2>(t57_);
            rpti = bsk::get<3>(t57_);
            auto t58_ = _dual_product(_dual_add(_dual_product(a21, free_plus), _dual_product(a22, pool_plus)), carried);
            rbpvr = bsk::get<0>(t58_);
            rbpvi = bsk::get<1>(t58_);
            rbptr = bsk::get<2>(t58_);
            rbpti = bsk::get<3>(t58_);
            auto t59_ = _dual_product(_dual_add(_dual_product(_dual_conj(a11), free_minus), _dual_product(_dual_conj(a12), pool_minus)), conjugated);
            rmvr = bsk::get<0>(t59_);
            rmvi = bsk::get<1>(t59_);
            rmtr = bsk::get<2>(t59_);
            rmti = bsk::get<3>(t59_);
            auto t60_ = _dual_product(_dual_add(_dual_product(_dual_conj(a21), free_minus), _dual_product(_dual_conj(a22), pool_minus)), conjugated);
            rbmvr = bsk::get<0>(t60_);
            rbmvi = bsk::get<1>(t60_);
            rbmtr = bsk::get<2>(t60_);
            rbmti = bsk::get<3>(t60_);
        } else {
            auto t61_ = _dual_mul(ovr, ovi, otr, oti, xpvr, xpvi, xptr, xpti);
            rpvr = bsk::get<0>(t61_);
            rpvi = bsk::get<1>(t61_);
            rptr = bsk::get<2>(t61_);
            rpti = bsk::get<3>(t61_);
            auto t62_ = _dual_mul(ovr, (-ovi), otr, (-oti), xmvr, xmvi, xmtr, xmti);
            rmvr = bsk::get<0>(t62_);
            rmvi = bsk::get<1>(t62_);
            rmtr = bsk::get<2>(t62_);
            rmti = bsk::get<3>(t62_);
        }
        rbvr = empty;
        rbvi = empty;
        rbtr = empty;
        rbti = empty;
        rcvr = empty;
        rcvi = empty;
        rctr = empty;
        rcti = empty;
        if (bsk::truth((pools == 3))) {
            if (bsk::truth(tabulated)) {
                // The walk back needs the operator and the direction
                // through it, which the row already holds -- and pooling
                // the cotangents took what the eigenvalues were formed
                // for, so nothing here reads them.
                pool_row = bsk::ld(((duration_row + event_base) + event), active_atom, 0);
                auto t63_ = _three_pool_from_table_jvp(pool_table, pool_row, atom, atom_count, active_atom, r1_value, r1b_value, r1c_value, atom_exchange, atom_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, dt_tangent, wout_value, wout_tangent);
                w11 = bsk::get<0>(t63_);
                w12 = bsk::get<1>(t63_);
                w13 = bsk::get<2>(t63_);
                w21 = bsk::get<3>(t63_);
                w22 = bsk::get<4>(t63_);
                w23 = bsk::get<5>(t63_);
                w31 = bsk::get<6>(t63_);
                w32 = bsk::get<7>(t63_);
                w33 = bsk::get<8>(t63_);
                grow_free = bsk::get<9>(t63_);
                grow_pool_b = bsk::get<10>(t63_);
                grow_semisolid = bsk::get<11>(t63_);
                d_w11 = bsk::get<12>(t63_);
                d_w12 = bsk::get<13>(t63_);
                d_w13 = bsk::get<14>(t63_);
                d_w21 = bsk::get<15>(t63_);
                d_w22 = bsk::get<16>(t63_);
                d_w23 = bsk::get<17>(t63_);
                d_w31 = bsk::get<18>(t63_);
                d_w32 = bsk::get<19>(t63_);
                d_w33 = bsk::get<20>(t63_);
                d_grow_free = bsk::get<21>(t63_);
                d_grow_pool_b = bsk::get<22>(t63_);
                d_grow_semisolid = bsk::get<23>(t63_);
            } else {
                auto t64_ = _three_pool_pieces_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, r1c_value, r1c_tangent, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, dt_value, dt_tangent, narrow);
                three_free = bsk::get<0>(t64_);
                three_d_free = bsk::get<1>(t64_);
                three_pool_b = bsk::get<2>(t64_);
                three_d_pool_b = bsk::get<3>(t64_);
                three_pool_c = bsk::get<4>(t64_);
                three_d_pool_c = bsk::get<5>(t64_);
                three_a00 = bsk::get<6>(t64_);
                three_d_a00 = bsk::get<7>(t64_);
                three_a01 = bsk::get<8>(t64_);
                three_d_a01 = bsk::get<9>(t64_);
                three_a02 = bsk::get<10>(t64_);
                three_d_a02 = bsk::get<11>(t64_);
                three_a10 = bsk::get<12>(t64_);
                three_d_a10 = bsk::get<13>(t64_);
                three_a11 = bsk::get<14>(t64_);
                three_d_a11 = bsk::get<15>(t64_);
                three_a20 = bsk::get<16>(t64_);
                three_d_a20 = bsk::get<17>(t64_);
                three_a22 = bsk::get<18>(t64_);
                three_d_a22 = bsk::get<19>(t64_);
                three_s00 = bsk::get<20>(t64_);
                three_d_s00 = bsk::get<21>(t64_);
                three_s11 = bsk::get<22>(t64_);
                three_d_s11 = bsk::get<23>(t64_);
                three_s22 = bsk::get<24>(t64_);
                three_d_s22 = bsk::get<25>(t64_);
                three_minors = bsk::get<26>(t64_);
                three_d_minors = bsk::get<27>(t64_);
                three_sum_flat = bsk::get<28>(t64_);
                three_sum_linear = bsk::get<29>(t64_);
                three_sum_square = bsk::get<30>(t64_);
                three_d_sum_flat = bsk::get<31>(t64_);
                three_d_sum_linear = bsk::get<32>(t64_);
                three_d_sum_square = bsk::get<33>(t64_);
                three_lift = bsk::get<34>(t64_);
                three_d_lift = bsk::get<35>(t64_);
                three_low = bsk::get<36>(t64_);
                three_middle = bsk::get<37>(t64_);
                three_d_low = bsk::get<38>(t64_);
                three_d_middle = bsk::get<39>(t64_);
                three_leading = bsk::get<40>(t64_);
                three_d_leading = bsk::get<41>(t64_);
                three_first = bsk::get<42>(t64_);
                three_d_first = bsk::get<43>(t64_);
                three_second = bsk::get<44>(t64_);
                three_d_second = bsk::get<45>(t64_);
                three_determinant = bsk::get<46>(t64_);
                three_d_determinant = bsk::get<47>(t64_);
                three_high = bsk::get<48>(t64_);
                three_d_high = bsk::get<49>(t64_);
                three_radius = bsk::get<50>(t64_);
                three_d_radius = bsk::get<51>(t64_);
                three_cube = bsk::get<52>(t64_);
                three_raw = bsk::get<53>(t64_);
                three_d_raw = bsk::get<54>(t64_);
                three_argument = bsk::get<55>(t64_);
                three_inside_limit = bsk::get<56>(t64_);
                three_angle = bsk::get<57>(t64_);
                three_d_angle = bsk::get<58>(t64_);
                three_centre = bsk::get<59>(t64_);
                three_d_centre = bsk::get<60>(t64_);
                three_trailing = bsk::get<61>(t64_);
                three_d_trailing = bsk::get<62>(t64_);
                three_guarded = bsk::get<63>(t64_);
                three_d_guarded = bsk::get<64>(t64_);
                three_q00 = bsk::get<65>(t64_);
                three_d_q00 = bsk::get<66>(t64_);
                three_q01 = bsk::get<67>(t64_);
                three_d_q01 = bsk::get<68>(t64_);
                three_q02 = bsk::get<69>(t64_);
                three_d_q02 = bsk::get<70>(t64_);
                three_q10 = bsk::get<71>(t64_);
                three_d_q10 = bsk::get<72>(t64_);
                three_q11 = bsk::get<73>(t64_);
                three_d_q11 = bsk::get<74>(t64_);
                three_q12 = bsk::get<75>(t64_);
                three_d_q12 = bsk::get<76>(t64_);
                three_q20 = bsk::get<77>(t64_);
                three_d_q20 = bsk::get<78>(t64_);
                three_q21 = bsk::get<79>(t64_);
                three_d_q21 = bsk::get<80>(t64_);
                three_q22 = bsk::get<81>(t64_);
                three_d_q22 = bsk::get<82>(t64_);
                auto t65_ = _three_pool_assemble_jvp(three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, narrow);
                three_def_00 = bsk::get<0>(t65_);
                three_dif_00 = bsk::get<1>(t65_);
                three_def_01 = bsk::get<2>(t65_);
                three_dif_01 = bsk::get<3>(t65_);
                three_def_02 = bsk::get<4>(t65_);
                three_dif_02 = bsk::get<5>(t65_);
                three_def_10 = bsk::get<6>(t65_);
                three_dif_10 = bsk::get<7>(t65_);
                three_def_11 = bsk::get<8>(t65_);
                three_dif_11 = bsk::get<9>(t65_);
                three_def_12 = bsk::get<10>(t65_);
                three_dif_12 = bsk::get<11>(t65_);
                three_def_20 = bsk::get<12>(t65_);
                three_dif_20 = bsk::get<13>(t65_);
                three_def_21 = bsk::get<14>(t65_);
                three_dif_21 = bsk::get<15>(t65_);
                three_def_22 = bsk::get<16>(t65_);
                three_dif_22 = bsk::get<17>(t65_);
                auto t66_ = _three_pool_weigh_jvp(three_def_00, three_dif_00, three_def_01, three_dif_01, three_def_02, three_dif_02, three_def_10, three_dif_10, three_def_11, three_dif_11, three_def_12, three_dif_12, three_def_20, three_dif_20, three_def_21, three_dif_21, three_def_22, three_dif_22, three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, wout_value, wout_tangent, narrow);
                w11 = bsk::get<0>(t66_);
                w12 = bsk::get<1>(t66_);
                w13 = bsk::get<2>(t66_);
                w21 = bsk::get<3>(t66_);
                w22 = bsk::get<4>(t66_);
                w23 = bsk::get<5>(t66_);
                w31 = bsk::get<6>(t66_);
                w32 = bsk::get<7>(t66_);
                w33 = bsk::get<8>(t66_);
                grow_free = bsk::get<9>(t66_);
                grow_pool_b = bsk::get<10>(t66_);
                grow_semisolid = bsk::get<11>(t66_);
                d_w11 = bsk::get<12>(t66_);
                d_w12 = bsk::get<13>(t66_);
                d_w13 = bsk::get<14>(t66_);
                d_w21 = bsk::get<15>(t66_);
                d_w22 = bsk::get<16>(t66_);
                d_w23 = bsk::get<17>(t66_);
                d_w31 = bsk::get<18>(t66_);
                d_w32 = bsk::get<19>(t66_);
                d_w33 = bsk::get<20>(t66_);
                d_grow_free = bsk::get<21>(t66_);
                d_grow_pool_b = bsk::get<22>(t66_);
                d_grow_semisolid = bsk::get<23>(t66_);
                // The operator is O(1) once formed, so the per-order loop below
                // takes it at the width the states are carried in.
                w11 = bsk::cast<float>(w11);
                w12 = bsk::cast<float>(w12);
                w13 = bsk::cast<float>(w13);
                w21 = bsk::cast<float>(w21);
                w22 = bsk::cast<float>(w22);
                w23 = bsk::cast<float>(w23);
                w31 = bsk::cast<float>(w31);
                w32 = bsk::cast<float>(w32);
                w33 = bsk::cast<float>(w33);
                grow_free = bsk::cast<float>(grow_free);
                grow_pool_b = bsk::cast<float>(grow_pool_b);
                grow_semisolid = bsk::cast<float>(grow_semisolid);
                d_w11 = bsk::cast<float>(d_w11);
                d_w12 = bsk::cast<float>(d_w12);
                d_w13 = bsk::cast<float>(d_w13);
                d_w21 = bsk::cast<float>(d_w21);
                d_w22 = bsk::cast<float>(d_w22);
                d_w23 = bsk::cast<float>(d_w23);
                d_w31 = bsk::cast<float>(d_w31);
                d_w32 = bsk::cast<float>(d_w32);
                d_w33 = bsk::cast<float>(d_w33);
                d_grow_free = bsk::cast<float>(d_grow_free);
                d_grow_pool_b = bsk::cast<float>(d_grow_pool_b);
                d_grow_semisolid = bsk::cast<float>(d_grow_semisolid);
            }
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti);
            mixed_free = _dual_add(_dual_add(_dual_scale(w11, d_w11, xzvr, xzvi, xztr, xzti), _dual_scale(w12, d_w12, xbvr, xbvi, xbtr, xbti)), _dual_scale(w13, d_w13, xcvr, xcvi, xctr, xcti));
            mixed_bound = _dual_add(_dual_add(_dual_scale(w21, d_w21, xzvr, xzvi, xztr, xzti), _dual_scale(w22, d_w22, xbvr, xbvi, xbtr, xbti)), _dual_scale(w23, d_w23, xcvr, xcvi, xctr, xcti));
            mixed_semisolid = _dual_add(_dual_add(_dual_scale(w31, d_w31, xzvr, xzvi, xztr, xzti), _dual_scale(w32, d_w32, xbvr, xbvi, xbtr, xbti)), _dual_scale(w33, d_w33, xcvr, xcvi, xctr, xcti));
            auto t67_ = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_free);
            rzvr = bsk::get<0>(t67_);
            rzvi = bsk::get<1>(t67_);
            rztr = bsk::get<2>(t67_);
            rzti = bsk::get<3>(t67_);
            auto t68_ = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_bound);
            rbvr = bsk::get<0>(t68_);
            rbvi = bsk::get<1>(t68_);
            rbtr = bsk::get<2>(t68_);
            rbti = bsk::get<3>(t68_);
            auto t69_ = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_semisolid);
            rcvr = bsk::get<0>(t69_);
            rcvi = bsk::get<1>(t69_);
            rctr = bsk::get<2>(t69_);
            rcti = bsk::get<3>(t69_);
            rzvr = (rzvr + bsk::where((state == 0), grow_free, 0.0f));
            rztr = (rztr + bsk::where((state == 0), d_grow_free, 0.0f));
            rbvr = (rbvr + bsk::where((state == 0), grow_pool_b, 0.0f));
            rbtr = (rbtr + bsk::where((state == 0), d_grow_pool_b, 0.0f));
            rcvr = (rcvr + bsk::where((state == 0), grow_semisolid, 0.0f));
            rctr = (rctr + bsk::where((state == 0), d_grow_semisolid, 0.0f));
        } else if (bsk::truth((pools > 0))) {
            auto t70_ = _two_pool_step_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, atom_exchange, d_exchange, atom_bound, d_boundf, dt_value, dt_tangent, wout_value, wout_tangent);
            pe11 = bsk::get<0>(t70_);
            pe12 = bsk::get<1>(t70_);
            pe21 = bsk::get<2>(t70_);
            pe22 = bsk::get<3>(t70_);
            prec_f = bsk::get<4>(t70_);
            prec_b = bsk::get<5>(t70_);
            de11 = bsk::get<6>(t70_);
            de12 = bsk::get<7>(t70_);
            de21 = bsk::get<8>(t70_);
            de22 = bsk::get<9>(t70_);
            drec_f = bsk::get<10>(t70_);
            drec_b = bsk::get<11>(t70_);
            spin = _dual_scale(damp_z, damp_z_tangent, szr, szi, sztr, szti);
            free_part = _dual_scale(pe11, de11, xzvr, xzvi, xztr, xzti);
            cross_in = _dual_scale(pe12, de12, xbvr, xbvi, xbtr, xbti);
            cross_out = _dual_scale(pe21, de21, xzvr, xzvi, xztr, xzti);
            bound_part = _dual_scale(pe22, de22, xbvr, xbvi, xbtr, xbti);
            mixed_free = bsk::make_tup((bsk::get<0>(free_part) + bsk::get<0>(cross_in)), (bsk::get<1>(free_part) + bsk::get<1>(cross_in)), (bsk::get<2>(free_part) + bsk::get<2>(cross_in)), (bsk::get<3>(free_part) + bsk::get<3>(cross_in)));
            mixed_bound = bsk::make_tup((bsk::get<0>(cross_out) + bsk::get<0>(bound_part)), (bsk::get<1>(cross_out) + bsk::get<1>(bound_part)), (bsk::get<2>(cross_out) + bsk::get<2>(bound_part)), (bsk::get<3>(cross_out) + bsk::get<3>(bound_part)));
            auto t71_ = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_free);
            rzvr = bsk::get<0>(t71_);
            rzvi = bsk::get<1>(t71_);
            rztr = bsk::get<2>(t71_);
            rzti = bsk::get<3>(t71_);
            auto t72_ = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_bound);
            rbvr = bsk::get<0>(t72_);
            rbvi = bsk::get<1>(t72_);
            rbtr = bsk::get<2>(t72_);
            rbti = bsk::get<3>(t72_);
            rzvr = (rzvr + bsk::where((state == 0), prec_f, 0.0f));
            rztr = (rztr + bsk::where((state == 0), drec_f, 0.0f));
            rbvr = (rbvr + bsk::where((state == 0), prec_b, 0.0f));
            rbtr = (rbtr + bsk::where((state == 0), drec_b, 0.0f));
        } else {
            auto t73_ = _dual_mul(lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti);
            rzvr = bsk::get<0>(t73_);
            rzvi = bsk::get<1>(t73_);
            rztr = bsk::get<2>(t73_);
            rzti = bsk::get<3>(t73_);
            rzvr = (rzvr + bsk::where((state == 0), recovery_value, 0.0f));
            rztr = (rztr + bsk::where((state == 0), recovery_tangent, 0.0f));
        }
        pre_shift = (bsk::band(event_action, 1) != 0);
        auto t74_ = _shift(rpvr, rpvi, rmvr, rmvi, state, state_mask, state_count);
        svr = bsk::get<0>(t74_);
        svi = bsk::get<1>(t74_);
        wvr = bsk::get<2>(t74_);
        wvi = bsk::get<3>(t74_);
        auto t75_ = _shift(rptr, rpti, rmtr, rmti, state, state_mask, state_count);
        str_ = bsk::get<0>(t75_);
        sti = bsk::get<1>(t75_);
        wtr = bsk::get<2>(t75_);
        wti = bsk::get<3>(t75_);
        auto spvr = bsk::where(pre_shift, svr, rpvr);
        auto spvi = bsk::where(pre_shift, svi, rpvi);
        auto sptr = bsk::where(pre_shift, str_, rptr);
        auto spti = bsk::where(pre_shift, sti, rpti);
        auto smvr = bsk::where(pre_shift, wvr, rmvr);
        auto smvi = bsk::where(pre_shift, wvi, rmvi);
        auto smtr = bsk::where(pre_shift, wtr, rmtr);
        auto smti = bsk::where(pre_shift, wti, rmti);
        sbpvr = rbpvr;
        sbpvi = rbpvi;
        sbptr = rbptr;
        sbpti = rbpti;
        sbmvr = rbmvr;
        sbmvi = rbmvi;
        sbmtr = rbmtr;
        sbmti = rbmti;
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t76_ = _shift(rbpvr, rbpvi, rbmvr, rbmvi, state, state_mask, state_count);
            svr = bsk::get<0>(t76_);
            svi = bsk::get<1>(t76_);
            wvr = bsk::get<2>(t76_);
            wvi = bsk::get<3>(t76_);
            auto t77_ = _shift(rbptr, rbpti, rbmtr, rbmti, state, state_mask, state_count);
            str_ = bsk::get<0>(t77_);
            sti = bsk::get<1>(t77_);
            wtr = bsk::get<2>(t77_);
            wti = bsk::get<3>(t77_);
            sbpvr = bsk::where(pre_shift, svr, rbpvr);
            sbpvi = bsk::where(pre_shift, svi, rbpvi);
            sbptr = bsk::where(pre_shift, str_, rbptr);
            sbpti = bsk::where(pre_shift, sti, rbpti);
            sbmvr = bsk::where(pre_shift, wvr, rbmvr);
            sbmvi = bsk::where(pre_shift, wvi, rbmvi);
            sbmtr = bsk::where(pre_shift, wtr, rbmtr);
            sbmti = bsk::where(pre_shift, wti, rbmti);
        }
        // Undo the trailing spoil or shift.
        do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        spoil = (bsk::band(event_action, 8) != 0);
        auto t78_ = _shift_adjoint(pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count);
        avr = bsk::get<0>(t78_);
        avi = bsk::get<1>(t78_);
        bvr = bsk::get<2>(t78_);
        bvi = bsk::get<3>(t78_);
        auto t79_ = _shift_adjoint(pbtr, pbti, mbtr, mbti, state, state_mask, state_count);
        atr = bsk::get<0>(t79_);
        ati = bsk::get<1>(t79_);
        btr = bsk::get<2>(t79_);
        bti = bsk::get<3>(t79_);
        auto trailing = bsk::band(do_shift, bsk::bnot(spoil));
        pbvr = bsk::where(spoil, 0.0f, bsk::where(trailing, avr, pbvr));
        pbvi = bsk::where(spoil, 0.0f, bsk::where(trailing, avi, pbvi));
        pbtr = bsk::where(spoil, 0.0f, bsk::where(trailing, atr, pbtr));
        pbti = bsk::where(spoil, 0.0f, bsk::where(trailing, ati, pbti));
        mbvr = bsk::where(spoil, 0.0f, bsk::where(trailing, bvr, mbvr));
        mbvi = bsk::where(spoil, 0.0f, bsk::where(trailing, bvi, mbvi));
        mbtr = bsk::where(spoil, 0.0f, bsk::where(trailing, btr, mbtr));
        mbti = bsk::where(spoil, 0.0f, bsk::where(trailing, bti, mbti));
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t80_ = _shift_adjoint(ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count);
            avr = bsk::get<0>(t80_);
            avi = bsk::get<1>(t80_);
            bvr = bsk::get<2>(t80_);
            bvi = bsk::get<3>(t80_);
            auto t81_ = _shift_adjoint(ubtr, ubti, wbtr, wbti, state, state_mask, state_count);
            atr = bsk::get<0>(t81_);
            ati = bsk::get<1>(t81_);
            btr = bsk::get<2>(t81_);
            bti = bsk::get<3>(t81_);
            ubvr = bsk::where(spoil, 0.0f, bsk::where(trailing, avr, ubvr));
            ubvi = bsk::where(spoil, 0.0f, bsk::where(trailing, avi, ubvi));
            ubtr = bsk::where(spoil, 0.0f, bsk::where(trailing, atr, ubtr));
            ubti = bsk::where(spoil, 0.0f, bsk::where(trailing, ati, ubti));
            wbvr = bsk::where(spoil, 0.0f, bsk::where(trailing, bvr, wbvr));
            wbvi = bsk::where(spoil, 0.0f, bsk::where(trailing, bvi, wbvi));
            wbtr = bsk::where(spoil, 0.0f, bsk::where(trailing, btr, wbtr));
            wbti = bsk::where(spoil, 0.0f, bsk::where(trailing, bti, wbti));
        }
        event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        event_dot_flip = _event_value(dot_flip, event_base, event, active_atom, single_train);
        event_phase = _event_value(phase, event_base, event, active_atom, single_train);
        event_dot_phase = _event_value(dot_phase, event_base, event, active_atom, single_train);
        // ---- recorded sample ----
        auto record = bsk::band((bsk::band(event_action, 32) != 0), (event_kind == 2));
        auto out_ = bsk::ld((output_index + event));
        auto seed_mask = bsk::band(bsk::band(active_atom, record), (out_ >= 0));
        auto seed_real = bsk::ld(((grad_output_real + (problem * output_count)) + out_), seed_mask, 0.0f);
        auto seed_imag = bsk::ld(((grad_output_imag + (problem * output_count)) + out_), seed_mask, 0.0f);
        auto t82_ = _dual_polar((-event_phase), (-event_dot_phase));
        auto dvr = bsk::get<0>(t82_);
        auto dvi = bsk::get<1>(t82_);
        auto dtr = bsk::get<2>(t82_);
        auto dti = bsk::get<3>(t82_);
        // A coil sees the whole voxel, so what it records is the sum over pools.
        recorded = bsk::make_tup(spvr, spvi, sptr, spti);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            recorded = _dual_add(recorded, bsk::make_tup(sbpvr, sbpvi, sbptr, sbpti));
        }
        // grad_m0 = Re(conj(seed) * recorded * demodulation)
        auto t83_ = [&](const auto& s0_) { return _dual_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), dvr, dvi, dtr, dti); }(recorded);
        auto wr = bsk::get<0>(t83_);
        auto wi = bsk::get<1>(t83_);
        auto wtr_ = bsk::get<2>(t83_);
        auto wti_ = bsk::get<3>(t83_);
        auto t84_ = _dual_real_conj_mul(seed_real, seed_imag, (0.0f * seed_real), (0.0f * seed_imag), wr, wi, wtr_, wti_);
        auto m0_value = bsk::get<0>(t84_);
        auto m0_tangent = bsk::get<1>(t84_);
        g_m0v = (g_m0v + bsk::sum_x(bsk::where((state == 0), m0_value, 0.0f)));
        g_m0t = (g_m0t + bsk::sum_x(bsk::where((state == 0), m0_tangent, 0.0f)));
        // grad_phase = Re(conj(seed) * m0 * recorded * (-i) * demodulation)
        auto t85_ = [&](const auto& s2_) { return _dual_scale(atom_m0, d_m0, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(recorded);
        yr = bsk::get<0>(t85_);
        yi = bsk::get<1>(t85_);
        ytr = bsk::get<2>(t85_);
        yti = bsk::get<3>(t85_);
        auto t86_ = _dual_times_i(yr, yi, ytr, yti);
        yr = bsk::get<0>(t86_);
        yi = bsk::get<1>(t86_);
        ytr = bsk::get<2>(t86_);
        yti = bsk::get<3>(t86_);
        auto t87_ = bsk::make_tup((-yr), (-yi), (-ytr), (-yti));
        yr = bsk::get<0>(t87_);
        yi = bsk::get<1>(t87_);
        ytr = bsk::get<2>(t87_);
        yti = bsk::get<3>(t87_);
        auto t88_ = _dual_mul(yr, yi, ytr, yti, dvr, dvi, dtr, dti);
        yr = bsk::get<0>(t88_);
        yi = bsk::get<1>(t88_);
        ytr = bsk::get<2>(t88_);
        yti = bsk::get<3>(t88_);
        auto t89_ = _dual_real_conj_mul(seed_real, seed_imag, (0.0f * seed_real), (0.0f * seed_imag), yr, yi, ytr, yti);
        auto phase_value = bsk::get<0>(t89_);
        auto phase_tangent = bsk::get<1>(t89_);
        bsk::atomic_add(((grad_phase_value + event_base) + event), bsk::sum_x(bsk::where((state == 0), phase_value, 0.0f)), seed_mask);
        bsk::atomic_add(((grad_phase_tangent + event_base) + event), bsk::sum_x(bsk::where((state == 0), phase_tangent, 0.0f)), seed_mask);
        // fplus_bar[0] += conj(m0 * demodulation) * seed
        auto t90_ = _dual_scale(atom_m0, d_m0, dvr, dvi, dtr, dti);
        auto kr = bsk::get<0>(t90_);
        auto ki = bsk::get<1>(t90_);
        auto ktr = bsk::get<2>(t90_);
        auto kti = bsk::get<3>(t90_);
        auto t91_ = _dual_mul(kr, (-ki), ktr, (-kti), seed_real, seed_imag, (0.0f * seed_real), (0.0f * seed_imag));
        auto sr = bsk::get<0>(t91_);
        auto si = bsk::get<1>(t91_);
        auto stg_r = bsk::get<2>(t91_);
        auto stg_i = bsk::get<3>(t91_);
        pbvr = (pbvr + bsk::where((state == 0), sr, 0.0f));
        pbvi = (pbvi + bsk::where((state == 0), si, 0.0f));
        pbtr = (pbtr + bsk::where((state == 0), stg_r, 0.0f));
        pbti = (pbti + bsk::where((state == 0), stg_i, 0.0f));
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            ubvr = (ubvr + bsk::where((state == 0), sr, 0.0f));
            ubvi = (ubvi + bsk::where((state == 0), si, 0.0f));
            ubtr = (ubtr + bsk::where((state == 0), stg_r, 0.0f));
            ubti = (ubti + bsk::where((state == 0), stg_i, 0.0f));
        }
        // ---- RF adjoint ----
        is_rf = (event_kind == 1);
        is_inversion = (bsk::band(event_action, 4) != 0);
        invert = bsk::band(is_rf, is_inversion);
        auto t92_ = _dual_real_conj_mul(zbvr, zbvi, zbtr, zbti, (-rzvr), (-rzvi), (-rztr), (-rzti));
        auto inv_value = bsk::get<0>(t92_);
        auto inv_tangent = bsk::get<1>(t92_);
        g_invv = (g_invv + bsk::sum_x(bsk::where(invert, inv_value, 0.0f)));
        g_invt = (g_invt + bsk::sum_x(bsk::where(invert, inv_tangent, 0.0f)));
        auto t93_ = _dual_scale((-atom_inv), (-d_inv), zbvr, zbvi, zbtr, zbti);
        ivr = bsk::get<0>(t93_);
        ivi = bsk::get<1>(t93_);
        itr = bsk::get<2>(t93_);
        iti = bsk::get<3>(t93_);
        zbvr = bsk::where(invert, ivr, zbvr);
        zbvi = bsk::where(invert, ivi, zbvi);
        zbtr = bsk::where(invert, itr, zbtr);
        zbti = bsk::where(invert, iti, zbti);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t94_ = _dual_real_conj_mul(bbvr, bbvi, bbtr, bbti, (-rbvr), (-rbvi), (-rbtr), (-rbti));
            auto pool_v = bsk::get<0>(t94_);
            auto pool_t = bsk::get<1>(t94_);
            g_invv = (g_invv + bsk::sum_x(bsk::where(invert, pool_v, 0.0f)));
            g_invt = (g_invt + bsk::sum_x(bsk::where(invert, pool_t, 0.0f)));
            auto t95_ = _dual_scale((-atom_inv), (-d_inv), bbvr, bbvi, bbtr, bbti);
            ivr = bsk::get<0>(t95_);
            ivi = bsk::get<1>(t95_);
            itr = bsk::get<2>(t95_);
            iti = bsk::get<3>(t95_);
            bbvr = bsk::where(invert, ivr, bbvr);
            bbvi = bsk::where(invert, ivi, bbvi);
            bbtr = bsk::where(invert, itr, bbtr);
            bbti = bsk::where(invert, iti, bbti);
        }
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse a row of its own.
        if (bsk::truth(shimmed)) {
            row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            atom_b1 = 1.0f;
            if (bsk::truth(transmit)) {
                atom_b1 = bsk::ld(((b1 + row) + atom), active_atom, 1.0f);
            }
            if (bsk::truth(off_axis)) {
                atom_b1_phase = bsk::ld(((b1_phase + row) + atom), active_atom, 0.0f);
            }
            d_b1 = bsk::ld(((dot_b1 + row) + atom), active_atom, 0.0f);
            if (bsk::truth(off_axis)) {
                d_b1_phase = bsk::ld(((dot_b1_phase + row) + atom), active_atom, 0.0f);
            }
        }
        alpha_value = (event_flip * atom_b1);
        alpha_tangent = ((event_dot_flip * atom_b1) + (event_flip * d_b1));
        phi_value = (event_phase + atom_b1_phase);
        phi_tangent = (event_dot_phase + d_b1_phase);
        sat_alpha_v = zero;
        sat_alpha_t = zero;
        sat_b0_v = zero;
        sat_b0_t = zero;
        if (bsk::truth(broadened)) {
            // The pulse scales every order of the bound pool by one real
            // number, so its cotangent is a single sum over the states it
            // multiplied. The lineshape's own slope is differentiated too,
            // which is what the curvature the reader returns is for.
            offset_value = (bsk::ld((rf_frequency + event)) - atom_b0);
            auto t96_ = _lineshape_at_curve(lineshape, offset_value, lineshape_bins, lineshape_step);
            shape_value = bsk::get<0>(t96_);
            shape_slope = bsk::get<1>(t96_);
            auto shape_curve = bsk::get<2>(t96_);
            shape_tangent = (shape_slope * (-d_b0));
            auto slope_tangent = (shape_curve * (-d_b0));
            event_saturation = bsk::ld((saturation + event));
            power_value = ((event_saturation * alpha_value) * alpha_value);
            power_tangent = (((event_saturation * 2.0f) * alpha_value) * alpha_tangent);
            absorbed_value = bsk::exp((power_value * shape_value));
            absorbed_tangent = (absorbed_value * ((power_tangent * shape_value) + (power_value * shape_tangent)));
            if (bsk::truth((pools == 1))) {
                held_bar = bsk::make_tup(bbvr, bbvi, bbtr, bbti);
                held_state = bsk::make_tup(rbvr, rbvi, rbtr, rbti);
            } else {
                held_bar = bsk::make_tup(cbvr, cbvi, cbtr, cbti);
                held_state = bsk::make_tup(rcvr, rcvi, rctr, rcti);
            }
            auto t97_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(held_bar, held_state);
            auto per_state_v = bsk::get<0>(t97_);
            auto per_state_t = bsk::get<1>(t97_);
            auto grad_absorbed_v = bsk::sum_x(per_state_v);
            auto grad_absorbed_t = bsk::sum_x(per_state_t);
            auto grad_exponent_v = (grad_absorbed_v * absorbed_value);
            auto grad_exponent_t = ((grad_absorbed_t * absorbed_value) + (grad_absorbed_v * absorbed_tangent));
            auto twice = (event_saturation * 2.0f);
            sat_alpha_v = (grad_exponent_v * ((twice * alpha_value) * shape_value));
            sat_alpha_t = ((grad_exponent_t * ((twice * alpha_value) * shape_value)) + ((grad_exponent_v * twice) * ((alpha_tangent * shape_value) + (alpha_value * shape_tangent))));
            // The lineshape is read at the pulse's offset from the voxel, so a
            // step in the voxel's own off-resonance moves the read the other
            // way.
            sat_b0_v = ((-grad_exponent_v) * (power_value * shape_slope));
            sat_b0_t = (-((grad_exponent_t * (power_value * shape_slope)) + (grad_exponent_v * ((power_tangent * shape_slope) + (power_value * slope_tangent)))));
            damped = [&](const auto& s2_) { return _dual_scale(absorbed_value, absorbed_tangent, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_)); }(held_bar);
            saturating = bsk::band(is_rf, bsk::bnot(is_inversion));
            if (bsk::truth((pools == 1))) {
                bbvr = bsk::where(saturating, bsk::get<0>(damped), bbvr);
                bbvi = bsk::where(saturating, bsk::get<1>(damped), bbvi);
                bbtr = bsk::where(saturating, bsk::get<2>(damped), bbtr);
                bbti = bsk::where(saturating, bsk::get<3>(damped), bbti);
            } else {
                cbvr = bsk::where(saturating, bsk::get<0>(damped), cbvr);
                cbvi = bsk::where(saturating, bsk::get<1>(damped), cbvi);
                cbtr = bsk::where(saturating, bsk::get<2>(damped), cbtr);
                cbti = bsk::where(saturating, bsk::get<3>(damped), cbti);
            }
        }
        cos_value = bsk::cos(alpha_value);
        sin_value = bsk::sin(alpha_value);
        cos_tangent = ((-sin_value) * alpha_tangent);
        sin_tangent = (cos_value * alpha_tangent);
        auto t98_ = _dual_polar(phi_value, phi_tangent);
        p1r = bsk::get<0>(t98_);
        p1i = bsk::get<1>(t98_);
        p1tr = bsk::get<2>(t98_);
        p1ti = bsk::get<3>(t98_);
        auto t99_ = _dual_mul(p1r, p1i, p1tr, p1ti, p1r, p1i, p1tr, p1ti);
        p2r = bsk::get<0>(t99_);
        p2i = bsk::get<1>(t99_);
        p2tr = bsk::get<2>(t99_);
        p2ti = bsk::get<3>(t99_);
        auto t100_ = _rotation_block((0.5f * (1.0f + cos_value)), (0.5f * cos_tangent), (0.5f * (1.0f - cos_value)), (-0.5f * cos_tangent), sin_value, sin_tangent, cos_value, cos_tangent, p1r, p1i, p1tr, p1ti, p2r, p2i, p2tr, p2ti, p1r, (-p1i), p1tr, (-p1ti));
        t00 = bsk::get<0>(t100_);
        t01 = bsk::get<1>(t100_);
        t02 = bsk::get<2>(t100_);
        r12 = bsk::get<3>(t100_);
        t20 = bsk::get<4>(t100_);
        r21 = bsk::get<5>(t100_);
        r22 = bsk::get<6>(t100_);
        // The flip angle reaches a shaped pulse's rotation through the slope
        // stored beside it, or not at all when the rotation is read per voxel,
        // so the operator's derivative in the flip is only built where the
        // pulse is a flip and a phase.
        alpha_v = empty;
        alpha_t = empty;
        phi_v = empty;
        phi_t = empty;
        alpha_b_v = empty;
        alpha_b_t = empty;
        phi_b_v = empty;
        phi_b_t = empty;
        if (bsk::truth((bsk::truth((!bsk::truth(profiled))) && bsk::truth((!bsk::truth(dynamic)))))) {
            auto t101_ = _rotation_block((-0.5f * sin_value), (-0.5f * sin_tangent), (0.5f * sin_value), (0.5f * sin_tangent), cos_value, cos_tangent, (-sin_value), (-sin_tangent), p1r, p1i, p1tr, p1ti, p2r, p2i, p2tr, p2ti, p1r, (-p1i), p1tr, (-p1ti));
            auto d00 = bsk::get<0>(t101_);
            auto d01 = bsk::get<1>(t101_);
            auto d02 = bsk::get<2>(t101_);
            auto d12 = bsk::get<3>(t101_);
            auto d20 = bsk::get<4>(t101_);
            auto d21 = bsk::get<5>(t101_);
            auto d22 = bsk::get<6>(t101_);
            // d/dalpha, contracted with the adjoint.
            row0 = _dual_mul(bsk::get<0>(d00), bsk::get<1>(d00), bsk::get<2>(d00), bsk::get<3>(d00), spvr, spvi, sptr, spti);
            add1 = _dual_mul(bsk::get<0>(d01), bsk::get<1>(d01), bsk::get<2>(d01), bsk::get<3>(d01), smvr, smvi, smtr, smti);
            add2 = _dual_mul(bsk::get<0>(d02), bsk::get<1>(d02), bsk::get<2>(d02), bsk::get<3>(d02), rzvr, rzvi, rztr, rzti);
            auto t102_ = _dual_real_conj_mul(pbvr, pbvi, pbtr, pbti, ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)), ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2)), ((bsk::get<2>(row0) + bsk::get<2>(add1)) + bsk::get<2>(add2)), ((bsk::get<3>(row0) + bsk::get<3>(add1)) + bsk::get<3>(add2)));
            alpha_v = bsk::get<0>(t102_);
            alpha_t = bsk::get<1>(t102_);
            row0 = _dual_mul(bsk::get<0>(d01), (-bsk::get<1>(d01)), bsk::get<2>(d01), (-bsk::get<3>(d01)), spvr, spvi, sptr, spti);
            add1 = _dual_mul(bsk::get<0>(d00), bsk::get<1>(d00), bsk::get<2>(d00), bsk::get<3>(d00), smvr, smvi, smtr, smti);
            add2 = _dual_mul(bsk::get<0>(d12), bsk::get<1>(d12), bsk::get<2>(d12), bsk::get<3>(d12), rzvr, rzvi, rztr, rzti);
            auto t103_ = _dual_real_conj_mul(mbvr, mbvi, mbtr, mbti, ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)), ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2)), ((bsk::get<2>(row0) + bsk::get<2>(add1)) + bsk::get<2>(add2)), ((bsk::get<3>(row0) + bsk::get<3>(add1)) + bsk::get<3>(add2)));
            part_v = bsk::get<0>(t103_);
            part_t = bsk::get<1>(t103_);
            alpha_v = (alpha_v + part_v);
            alpha_t = (alpha_t + part_t);
            row0 = _dual_mul(bsk::get<0>(d20), bsk::get<1>(d20), bsk::get<2>(d20), bsk::get<3>(d20), spvr, spvi, sptr, spti);
            add1 = _dual_mul(bsk::get<0>(d21), bsk::get<1>(d21), bsk::get<2>(d21), bsk::get<3>(d21), smvr, smvi, smtr, smti);
            add2 = _dual_mul(bsk::get<0>(d22), bsk::get<1>(d22), bsk::get<2>(d22), bsk::get<3>(d22), rzvr, rzvi, rztr, rzti);
            auto t104_ = _dual_real_conj_mul(zbvr, zbvi, zbtr, zbti, ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)), ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2)), ((bsk::get<2>(row0) + bsk::get<2>(add1)) + bsk::get<2>(add2)), ((bsk::get<3>(row0) + bsk::get<3>(add1)) + bsk::get<3>(add2)));
            part_v = bsk::get<0>(t104_);
            part_t = bsk::get<1>(t104_);
            alpha_v = (alpha_v + part_v);
            alpha_t = (alpha_t + part_t);
            // d/dphi, where only the phase factors carry the dependence.
            u1 = _dual_mul(bsk::get<0>(t01), bsk::get<1>(t01), bsk::get<2>(t01), bsk::get<3>(t01), smvr, smvi, smtr, smti);
            u2 = _dual_mul(bsk::get<0>(t02), bsk::get<1>(t02), bsk::get<2>(t02), bsk::get<3>(t02), rzvr, rzvi, rztr, rzti);
            auto t105_ = _dual_times_i(((2.0f * bsk::get<0>(u1)) + bsk::get<0>(u2)), ((2.0f * bsk::get<1>(u1)) + bsk::get<1>(u2)), ((2.0f * bsk::get<2>(u1)) + bsk::get<2>(u2)), ((2.0f * bsk::get<3>(u1)) + bsk::get<3>(u2)));
            ur = bsk::get<0>(t105_);
            ui = bsk::get<1>(t105_);
            utr = bsk::get<2>(t105_);
            uti = bsk::get<3>(t105_);
            auto t106_ = _dual_real_conj_mul(pbvr, pbvi, pbtr, pbti, ur, ui, utr, uti);
            phi_v = bsk::get<0>(t106_);
            phi_t = bsk::get<1>(t106_);
            u1 = _dual_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bsk::get<2>(t01), (-bsk::get<3>(t01)), spvr, spvi, sptr, spti);
            u2 = _dual_mul(bsk::get<0>(r12), bsk::get<1>(r12), bsk::get<2>(r12), bsk::get<3>(r12), rzvr, rzvi, rztr, rzti);
            auto t107_ = _dual_times_i(((-2.0f * bsk::get<0>(u1)) - bsk::get<0>(u2)), ((-2.0f * bsk::get<1>(u1)) - bsk::get<1>(u2)), ((-2.0f * bsk::get<2>(u1)) - bsk::get<2>(u2)), ((-2.0f * bsk::get<3>(u1)) - bsk::get<3>(u2)));
            ur = bsk::get<0>(t107_);
            ui = bsk::get<1>(t107_);
            utr = bsk::get<2>(t107_);
            uti = bsk::get<3>(t107_);
            auto t108_ = _dual_real_conj_mul(mbvr, mbvi, mbtr, mbti, ur, ui, utr, uti);
            part_v = bsk::get<0>(t108_);
            part_t = bsk::get<1>(t108_);
            phi_v = (phi_v + part_v);
            phi_t = (phi_t + part_t);
            u1 = _dual_mul(bsk::get<0>(t20), bsk::get<1>(t20), bsk::get<2>(t20), bsk::get<3>(t20), spvr, spvi, sptr, spti);
            u2 = _dual_mul(bsk::get<0>(r21), bsk::get<1>(r21), bsk::get<2>(r21), bsk::get<3>(r21), smvr, smvi, smtr, smti);
            auto t109_ = _dual_times_i((bsk::get<0>(u2) - bsk::get<0>(u1)), (bsk::get<1>(u2) - bsk::get<1>(u1)), (bsk::get<2>(u2) - bsk::get<2>(u1)), (bsk::get<3>(u2) - bsk::get<3>(u1)));
            ur = bsk::get<0>(t109_);
            ui = bsk::get<1>(t109_);
            utr = bsk::get<2>(t109_);
            uti = bsk::get<3>(t109_);
            auto t110_ = _dual_real_conj_mul(zbvr, zbvi, zbtr, zbti, ur, ui, utr, uti);
            part_v = bsk::get<0>(t110_);
            part_t = bsk::get<1>(t110_);
            phi_v = (phi_v + part_v);
            phi_t = (phi_t + part_t);
            // The same pulse turns the exchanging pool, so its cotangent adds to
            // the flip and phase the free pool already left.
            alpha_b_v = (0.0f * alpha_v);
            alpha_b_t = (0.0f * alpha_v);
            phi_b_v = (0.0f * alpha_v);
            phi_b_t = (0.0f * alpha_v);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                row0 = _dual_mul(bsk::get<0>(d00), bsk::get<1>(d00), bsk::get<2>(d00), bsk::get<3>(d00), sbpvr, sbpvi, sbptr, sbpti);
                add1 = _dual_mul(bsk::get<0>(d01), bsk::get<1>(d01), bsk::get<2>(d01), bsk::get<3>(d01), sbmvr, sbmvi, sbmtr, sbmti);
                add2 = _dual_mul(bsk::get<0>(d02), bsk::get<1>(d02), bsk::get<2>(d02), bsk::get<3>(d02), rbvr, rbvi, rbtr, rbti);
                auto t111_ = _dual_real_conj_mul(ubvr, ubvi, ubtr, ubti, ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)), ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2)), ((bsk::get<2>(row0) + bsk::get<2>(add1)) + bsk::get<2>(add2)), ((bsk::get<3>(row0) + bsk::get<3>(add1)) + bsk::get<3>(add2)));
                alpha_b_v = bsk::get<0>(t111_);
                alpha_b_t = bsk::get<1>(t111_);
                row0 = _dual_mul(bsk::get<0>(d01), (-bsk::get<1>(d01)), bsk::get<2>(d01), (-bsk::get<3>(d01)), sbpvr, sbpvi, sbptr, sbpti);
                add1 = _dual_mul(bsk::get<0>(d00), bsk::get<1>(d00), bsk::get<2>(d00), bsk::get<3>(d00), sbmvr, sbmvi, sbmtr, sbmti);
                add2 = _dual_mul(bsk::get<0>(d12), bsk::get<1>(d12), bsk::get<2>(d12), bsk::get<3>(d12), rbvr, rbvi, rbtr, rbti);
                auto t112_ = _dual_real_conj_mul(wbvr, wbvi, wbtr, wbti, ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)), ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2)), ((bsk::get<2>(row0) + bsk::get<2>(add1)) + bsk::get<2>(add2)), ((bsk::get<3>(row0) + bsk::get<3>(add1)) + bsk::get<3>(add2)));
                part_v = bsk::get<0>(t112_);
                part_t = bsk::get<1>(t112_);
                alpha_b_v = (alpha_b_v + part_v);
                alpha_b_t = (alpha_b_t + part_t);
                row0 = _dual_mul(bsk::get<0>(d20), bsk::get<1>(d20), bsk::get<2>(d20), bsk::get<3>(d20), sbpvr, sbpvi, sbptr, sbpti);
                add1 = _dual_mul(bsk::get<0>(d21), bsk::get<1>(d21), bsk::get<2>(d21), bsk::get<3>(d21), sbmvr, sbmvi, sbmtr, sbmti);
                add2 = _dual_mul(bsk::get<0>(d22), bsk::get<1>(d22), bsk::get<2>(d22), bsk::get<3>(d22), rbvr, rbvi, rbtr, rbti);
                auto t113_ = _dual_real_conj_mul(bbvr, bbvi, bbtr, bbti, ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)), ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2)), ((bsk::get<2>(row0) + bsk::get<2>(add1)) + bsk::get<2>(add2)), ((bsk::get<3>(row0) + bsk::get<3>(add1)) + bsk::get<3>(add2)));
                part_v = bsk::get<0>(t113_);
                part_t = bsk::get<1>(t113_);
                alpha_b_v = (alpha_b_v + part_v);
                alpha_b_t = (alpha_b_t + part_t);
                u1 = _dual_mul(bsk::get<0>(t01), bsk::get<1>(t01), bsk::get<2>(t01), bsk::get<3>(t01), sbmvr, sbmvi, sbmtr, sbmti);
                u2 = _dual_mul(bsk::get<0>(t02), bsk::get<1>(t02), bsk::get<2>(t02), bsk::get<3>(t02), rbvr, rbvi, rbtr, rbti);
                auto t114_ = _dual_times_i(((2.0f * bsk::get<0>(u1)) + bsk::get<0>(u2)), ((2.0f * bsk::get<1>(u1)) + bsk::get<1>(u2)), ((2.0f * bsk::get<2>(u1)) + bsk::get<2>(u2)), ((2.0f * bsk::get<3>(u1)) + bsk::get<3>(u2)));
                ur = bsk::get<0>(t114_);
                ui = bsk::get<1>(t114_);
                utr = bsk::get<2>(t114_);
                uti = bsk::get<3>(t114_);
                auto t115_ = _dual_real_conj_mul(ubvr, ubvi, ubtr, ubti, ur, ui, utr, uti);
                phi_b_v = bsk::get<0>(t115_);
                phi_b_t = bsk::get<1>(t115_);
                u1 = _dual_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bsk::get<2>(t01), (-bsk::get<3>(t01)), sbpvr, sbpvi, sbptr, sbpti);
                u2 = _dual_mul(bsk::get<0>(r12), bsk::get<1>(r12), bsk::get<2>(r12), bsk::get<3>(r12), rbvr, rbvi, rbtr, rbti);
                auto t116_ = _dual_times_i(((-2.0f * bsk::get<0>(u1)) - bsk::get<0>(u2)), ((-2.0f * bsk::get<1>(u1)) - bsk::get<1>(u2)), ((-2.0f * bsk::get<2>(u1)) - bsk::get<2>(u2)), ((-2.0f * bsk::get<3>(u1)) - bsk::get<3>(u2)));
                ur = bsk::get<0>(t116_);
                ui = bsk::get<1>(t116_);
                utr = bsk::get<2>(t116_);
                uti = bsk::get<3>(t116_);
                auto t117_ = _dual_real_conj_mul(wbvr, wbvi, wbtr, wbti, ur, ui, utr, uti);
                part_v = bsk::get<0>(t117_);
                part_t = bsk::get<1>(t117_);
                phi_b_v = (phi_b_v + part_v);
                phi_b_t = (phi_b_t + part_t);
                u1 = _dual_mul(bsk::get<0>(t20), bsk::get<1>(t20), bsk::get<2>(t20), bsk::get<3>(t20), sbpvr, sbpvi, sbptr, sbpti);
                u2 = _dual_mul(bsk::get<0>(r21), bsk::get<1>(r21), bsk::get<2>(r21), bsk::get<3>(r21), sbmvr, sbmvi, sbmtr, sbmti);
                auto t118_ = _dual_times_i((bsk::get<0>(u2) - bsk::get<0>(u1)), (bsk::get<1>(u2) - bsk::get<1>(u1)), (bsk::get<2>(u2) - bsk::get<2>(u1)), (bsk::get<3>(u2) - bsk::get<3>(u1)));
                ur = bsk::get<0>(t118_);
                ui = bsk::get<1>(t118_);
                utr = bsk::get<2>(t118_);
                uti = bsk::get<3>(t118_);
                auto t119_ = _dual_real_conj_mul(bbvr, bbvi, bbtr, bbti, ur, ui, utr, uti);
                part_v = bsk::get<0>(t119_);
                part_t = bsk::get<1>(t119_);
                phi_b_v = (phi_b_v + part_v);
                phi_b_t = (phi_b_t + part_t);
            }
        }
        if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
            shaped_slope_a = bsk::make_tup(empty, empty, empty, empty);
            shaped_slope_b = bsk::make_tup(empty, empty, empty, empty);
            if (bsk::truth(dynamic)) {
                auto t120_ = _dynamic_pair_dual_at(pairs, pair_direction, pair_index, event_base, event, atom, atom_count, active_atom, phi_value, phi_tangent, directed);
                shaped_a = bsk::get<0>(t120_);
                shaped_b = bsk::get<1>(t120_);
            } else {
                auto t121_ = _profiled_pair_dual(profile, _table_row(profile_index, event, location, locations), alpha_value, alpha_tangent, phi_value, phi_tangent, profile_bins, profile_step);
                shaped_a = bsk::get<0>(t121_);
                shaped_b = bsk::get<1>(t121_);
                shaped_slope_a = bsk::get<2>(t121_);
                shaped_slope_b = bsk::get<3>(t121_);
            }
            auto t122_ = _spinor_adjoint_dual(shaped_a, shaped_b, bsk::make_tup(spvr, spvi, sptr, spti), bsk::make_tup(smvr, smvi, smtr, smti), bsk::make_tup(rzvr, rzvi, rztr, rzti), bsk::make_tup(pbvr, pbvi, pbtr, pbti), bsk::make_tup(mbvr, mbvi, mbtr, mbti), bsk::make_tup(zbvr, zbvi, zbtr, zbti));
            auto grad_a = bsk::get<0>(t122_);
            auto grad_b = bsk::get<1>(t122_);
            shaped_pb = bsk::get<2>(t122_);
            shaped_mb = bsk::get<3>(t122_);
            shaped_zb = bsk::get<4>(t122_);
            if (bsk::truth(dynamic)) {
                // The flip is inside the pair rather than read against it, so
                // it has no gradient here: the cotangent goes out on the
                // rotation and whatever integrated it carries the rest. ``b``
                // was turned by the phase after the pair came out, so the
                // cotangent turns back the other way.
                alpha_v = empty;
                alpha_t = empty;
                auto back = _dual_product(grad_b, _dual_conj(_dual_polar((-phi_value), (-phi_tangent))));
                _store_pair_cotangent(grad_pair_value, grad_pair_tangent, pair_index, event_base, event, atom, atom_count, bsk::band(is_rf, bsk::bnot(is_inversion)), active_atom, state_mask, grad_a, back);
            } else {
                auto t123_ = _dual_real_conj_mul(bsk::get<0>(grad_a), bsk::get<1>(grad_a), bsk::get<2>(grad_a), bsk::get<3>(grad_a), bsk::get<0>(shaped_slope_a), bsk::get<1>(shaped_slope_a), bsk::get<2>(shaped_slope_a), bsk::get<3>(shaped_slope_a));
                alpha_v = bsk::get<0>(t123_);
                alpha_t = bsk::get<1>(t123_);
                auto t124_ = _dual_real_conj_mul(bsk::get<0>(grad_b), bsk::get<1>(grad_b), bsk::get<2>(grad_b), bsk::get<3>(grad_b), bsk::get<0>(shaped_slope_b), bsk::get<1>(shaped_slope_b), bsk::get<2>(shaped_slope_b), bsk::get<3>(shaped_slope_b));
                part_v = bsk::get<0>(t124_);
                part_t = bsk::get<1>(t124_);
                alpha_v = (alpha_v + part_v);
                alpha_t = (alpha_t + part_t);
            }
            // d(b e^{-i phi})/dphi is -i times it, and nothing else moves.
            auto t125_ = _dual_times_i(bsk::get<0>(shaped_b), bsk::get<1>(shaped_b), bsk::get<2>(shaped_b), bsk::get<3>(shaped_b));
            auto turn_r = bsk::get<0>(t125_);
            auto turn_i = bsk::get<1>(t125_);
            auto turn_tr = bsk::get<2>(t125_);
            auto turn_ti = bsk::get<3>(t125_);
            auto t126_ = _dual_real_conj_mul(bsk::get<0>(grad_b), bsk::get<1>(grad_b), bsk::get<2>(grad_b), bsk::get<3>(grad_b), (-turn_r), (-turn_i), (-turn_tr), (-turn_ti));
            phi_v = bsk::get<0>(t126_);
            phi_t = bsk::get<1>(t126_);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t127_ = _spinor_adjoint_dual(shaped_a, shaped_b, bsk::make_tup(sbpvr, sbpvi, sbptr, sbpti), bsk::make_tup(sbmvr, sbmvi, sbmtr, sbmti), bsk::make_tup(rbvr, rbvi, rbtr, rbti), bsk::make_tup(ubvr, ubvi, ubtr, ubti), bsk::make_tup(wbvr, wbvi, wbtr, wbti), bsk::make_tup(bbvr, bbvi, bbtr, bbti));
                auto pool_a = bsk::get<0>(t127_);
                auto pool_b_pair = bsk::get<1>(t127_);
                shaped_ub = bsk::get<2>(t127_);
                shaped_wb = bsk::get<3>(t127_);
                shaped_bb = bsk::get<4>(t127_);
                if (bsk::truth(dynamic)) {
                    // The same pulse turned this pool, so its cotangent lands
                    // on the same row.
                    auto pool_back = _dual_product(pool_b_pair, _dual_conj(_dual_polar((-phi_value), (-phi_tangent))));
                    _store_pair_cotangent(grad_pair_value, grad_pair_tangent, pair_index, event_base, event, atom, atom_count, bsk::band(is_rf, bsk::bnot(is_inversion)), active_atom, state_mask, pool_a, pool_back);
                } else {
                    auto t128_ = _dual_real_conj_mul(bsk::get<0>(pool_a), bsk::get<1>(pool_a), bsk::get<2>(pool_a), bsk::get<3>(pool_a), bsk::get<0>(shaped_slope_a), bsk::get<1>(shaped_slope_a), bsk::get<2>(shaped_slope_a), bsk::get<3>(shaped_slope_a));
                    alpha_b_v = bsk::get<0>(t128_);
                    alpha_b_t = bsk::get<1>(t128_);
                    auto t129_ = _dual_real_conj_mul(bsk::get<0>(pool_b_pair), bsk::get<1>(pool_b_pair), bsk::get<2>(pool_b_pair), bsk::get<3>(pool_b_pair), bsk::get<0>(shaped_slope_b), bsk::get<1>(shaped_slope_b), bsk::get<2>(shaped_slope_b), bsk::get<3>(shaped_slope_b));
                    part_v = bsk::get<0>(t129_);
                    part_t = bsk::get<1>(t129_);
                    alpha_b_v = (alpha_b_v + part_v);
                    alpha_b_t = (alpha_b_t + part_t);
                }
                auto t130_ = _dual_real_conj_mul(bsk::get<0>(pool_b_pair), bsk::get<1>(pool_b_pair), bsk::get<2>(pool_b_pair), bsk::get<3>(pool_b_pair), (-turn_r), (-turn_i), (-turn_tr), (-turn_ti));
                phi_b_v = bsk::get<0>(t130_);
                phi_b_t = bsk::get<1>(t130_);
            }
        }
        alpha_v = (alpha_v + alpha_b_v);
        alpha_t = (alpha_t + alpha_b_t);
        phi_v = (phi_v + phi_b_v);
        phi_t = (phi_t + phi_b_t);
        rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        grad_alpha_v = bsk::sum_x(bsk::where(rotate, alpha_v, 0.0f));
        grad_alpha_t = bsk::sum_x(bsk::where(rotate, alpha_t, 0.0f));
        auto grad_phi_v = bsk::sum_x(bsk::where(rotate, phi_v, 0.0f));
        auto grad_phi_t = bsk::sum_x(bsk::where(rotate, phi_t, 0.0f));
        if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
            auto turning = bsk::where(rotate, 1.0f, 0.0f);
            grad_alpha_v = (grad_alpha_v + (sat_alpha_v * turning));
            grad_alpha_t = (grad_alpha_t + (sat_alpha_t * turning));
            g_b0v = (g_b0v + (sat_b0_v * turning));
            g_b0t = (g_b0t + (sat_b0_t * turning));
        }
        // Conjugate transpose of the rotation.
        n0 = _dual_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), bsk::get<2>(t00), (-bsk::get<3>(t00)), pbvr, pbvi, pbtr, pbti);
        n1 = _dual_mul(bsk::get<0>(t01), bsk::get<1>(t01), bsk::get<2>(t01), bsk::get<3>(t01), mbvr, mbvi, mbtr, mbti);
        n2 = _dual_mul(bsk::get<0>(t20), (-bsk::get<1>(t20)), bsk::get<2>(t20), (-bsk::get<3>(t20)), zbvr, zbvi, zbtr, zbti);
        q0 = _dual_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bsk::get<2>(t01), (-bsk::get<3>(t01)), pbvr, pbvi, pbtr, pbti);
        q1 = _dual_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), bsk::get<2>(t00), (-bsk::get<3>(t00)), mbvr, mbvi, mbtr, mbti);
        q2 = _dual_mul(bsk::get<0>(r21), (-bsk::get<1>(r21)), bsk::get<2>(r21), (-bsk::get<3>(r21)), zbvr, zbvi, zbtr, zbti);
        w0 = _dual_mul(bsk::get<0>(t02), (-bsk::get<1>(t02)), bsk::get<2>(t02), (-bsk::get<3>(t02)), pbvr, pbvi, pbtr, pbti);
        w1 = _dual_mul(bsk::get<0>(r12), (-bsk::get<1>(r12)), bsk::get<2>(r12), (-bsk::get<3>(r12)), mbvr, mbvi, mbtr, mbti);
        w2 = _dual_mul(bsk::get<0>(r22), (-bsk::get<1>(r22)), bsk::get<2>(r22), (-bsk::get<3>(r22)), zbvr, zbvi, zbtr, zbti);
        back_pb = bsk::make_tup(((bsk::get<0>(n0) + bsk::get<0>(n1)) + bsk::get<0>(n2)), ((bsk::get<1>(n0) + bsk::get<1>(n1)) + bsk::get<1>(n2)), ((bsk::get<2>(n0) + bsk::get<2>(n1)) + bsk::get<2>(n2)), ((bsk::get<3>(n0) + bsk::get<3>(n1)) + bsk::get<3>(n2)));
        back_mb = bsk::make_tup(((bsk::get<0>(q0) + bsk::get<0>(q1)) + bsk::get<0>(q2)), ((bsk::get<1>(q0) + bsk::get<1>(q1)) + bsk::get<1>(q2)), ((bsk::get<2>(q0) + bsk::get<2>(q1)) + bsk::get<2>(q2)), ((bsk::get<3>(q0) + bsk::get<3>(q1)) + bsk::get<3>(q2)));
        back_zb = bsk::make_tup(((bsk::get<0>(w0) + bsk::get<0>(w1)) + bsk::get<0>(w2)), ((bsk::get<1>(w0) + bsk::get<1>(w1)) + bsk::get<1>(w2)), ((bsk::get<2>(w0) + bsk::get<2>(w1)) + bsk::get<2>(w2)), ((bsk::get<3>(w0) + bsk::get<3>(w1)) + bsk::get<3>(w2)));
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            n0 = _dual_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), bsk::get<2>(t00), (-bsk::get<3>(t00)), ubvr, ubvi, ubtr, ubti);
            n1 = _dual_mul(bsk::get<0>(t01), bsk::get<1>(t01), bsk::get<2>(t01), bsk::get<3>(t01), wbvr, wbvi, wbtr, wbti);
            n2 = _dual_mul(bsk::get<0>(t20), (-bsk::get<1>(t20)), bsk::get<2>(t20), (-bsk::get<3>(t20)), bbvr, bbvi, bbtr, bbti);
            q0 = _dual_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bsk::get<2>(t01), (-bsk::get<3>(t01)), ubvr, ubvi, ubtr, ubti);
            q1 = _dual_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), bsk::get<2>(t00), (-bsk::get<3>(t00)), wbvr, wbvi, wbtr, wbti);
            q2 = _dual_mul(bsk::get<0>(r21), (-bsk::get<1>(r21)), bsk::get<2>(r21), (-bsk::get<3>(r21)), bbvr, bbvi, bbtr, bbti);
            w0 = _dual_mul(bsk::get<0>(t02), (-bsk::get<1>(t02)), bsk::get<2>(t02), (-bsk::get<3>(t02)), ubvr, ubvi, ubtr, ubti);
            w1 = _dual_mul(bsk::get<0>(r12), (-bsk::get<1>(r12)), bsk::get<2>(r12), (-bsk::get<3>(r12)), wbvr, wbvi, wbtr, wbti);
            w2 = _dual_mul(bsk::get<0>(r22), (-bsk::get<1>(r22)), bsk::get<2>(r22), (-bsk::get<3>(r22)), bbvr, bbvi, bbtr, bbti);
            back_ub = bsk::make_tup(((bsk::get<0>(n0) + bsk::get<0>(n1)) + bsk::get<0>(n2)), ((bsk::get<1>(n0) + bsk::get<1>(n1)) + bsk::get<1>(n2)), ((bsk::get<2>(n0) + bsk::get<2>(n1)) + bsk::get<2>(n2)), ((bsk::get<3>(n0) + bsk::get<3>(n1)) + bsk::get<3>(n2)));
            back_wb = bsk::make_tup(((bsk::get<0>(q0) + bsk::get<0>(q1)) + bsk::get<0>(q2)), ((bsk::get<1>(q0) + bsk::get<1>(q1)) + bsk::get<1>(q2)), ((bsk::get<2>(q0) + bsk::get<2>(q1)) + bsk::get<2>(q2)), ((bsk::get<3>(q0) + bsk::get<3>(q1)) + bsk::get<3>(q2)));
            back_bb = bsk::make_tup(((bsk::get<0>(w0) + bsk::get<0>(w1)) + bsk::get<0>(w2)), ((bsk::get<1>(w0) + bsk::get<1>(w1)) + bsk::get<1>(w2)), ((bsk::get<2>(w0) + bsk::get<2>(w1)) + bsk::get<2>(w2)), ((bsk::get<3>(w0) + bsk::get<3>(w1)) + bsk::get<3>(w2)));
            if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                back_ub = shaped_ub;
                back_wb = shaped_wb;
                back_bb = shaped_bb;
            }
            ubvr = bsk::where(rotate, bsk::get<0>(back_ub), ubvr);
            ubvi = bsk::where(rotate, bsk::get<1>(back_ub), ubvi);
            ubtr = bsk::where(rotate, bsk::get<2>(back_ub), ubtr);
            ubti = bsk::where(rotate, bsk::get<3>(back_ub), ubti);
            wbvr = bsk::where(rotate, bsk::get<0>(back_wb), wbvr);
            wbvi = bsk::where(rotate, bsk::get<1>(back_wb), wbvi);
            wbtr = bsk::where(rotate, bsk::get<2>(back_wb), wbtr);
            wbti = bsk::where(rotate, bsk::get<3>(back_wb), wbti);
            bbvr = bsk::where(rotate, bsk::get<0>(back_bb), bbvr);
            bbvi = bsk::where(rotate, bsk::get<1>(back_bb), bbvi);
            bbtr = bsk::where(rotate, bsk::get<2>(back_bb), bbtr);
            bbti = bsk::where(rotate, bsk::get<3>(back_bb), bbti);
        }
        if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
            back_pb = shaped_pb;
            back_mb = shaped_mb;
            back_zb = shaped_zb;
        }
        pbvr = bsk::where(rotate, bsk::get<0>(back_pb), pbvr);
        pbvi = bsk::where(rotate, bsk::get<1>(back_pb), pbvi);
        pbtr = bsk::where(rotate, bsk::get<2>(back_pb), pbtr);
        pbti = bsk::where(rotate, bsk::get<3>(back_pb), pbti);
        mbvr = bsk::where(rotate, bsk::get<0>(back_mb), mbvr);
        mbvi = bsk::where(rotate, bsk::get<1>(back_mb), mbvi);
        mbtr = bsk::where(rotate, bsk::get<2>(back_mb), mbtr);
        mbti = bsk::where(rotate, bsk::get<3>(back_mb), mbti);
        zbvr = bsk::where(rotate, bsk::get<0>(back_zb), zbvr);
        zbvi = bsk::where(rotate, bsk::get<1>(back_zb), zbvi);
        zbtr = bsk::where(rotate, bsk::get<2>(back_zb), zbtr);
        zbti = bsk::where(rotate, bsk::get<3>(back_zb), zbti);
        auto writes_flip = bsk::band(active_atom, rotate);
        bsk::atomic_add(((grad_flip_value + event_base) + event), (grad_alpha_v * atom_b1), writes_flip);
        bsk::atomic_add(((grad_flip_tangent + event_base) + event), ((grad_alpha_t * atom_b1) + (grad_alpha_v * d_b1)), writes_flip);
        bsk::atomic_add(((grad_phase_value + event_base) + event), grad_phi_v, writes_flip);
        bsk::atomic_add(((grad_phase_tangent + event_base) + event), grad_phi_t, writes_flip);
        // A pulse's transmit gradient belongs to the shim it drives, so with
        // several it lands in that shim's row here rather than in a register
        // summed over the whole train. ``row`` is the offset of the row the
        // replay above read.
        if (bsk::truth(shimmed)) {
            bsk::atomic_add((((grad_tissue_value + (3 * atom_count)) + row) + atom), (grad_alpha_v * event_flip), writes_flip);
            bsk::atomic_add((((grad_tissue_tangent + (3 * atom_count)) + row) + atom), ((grad_alpha_t * event_flip) + (grad_alpha_v * event_dot_flip)), writes_flip);
            bsk::atomic_add((((grad_tissue_value + (((4 + shim_rows) - 1) * atom_count)) + row) + atom), grad_phi_v, writes_flip);
            bsk::atomic_add((((grad_tissue_tangent + (((4 + shim_rows) - 1) * atom_count)) + row) + atom), grad_phi_t, writes_flip);
        } else {
            g_b1v = (g_b1v + (grad_alpha_v * event_flip));
            g_b1t = (g_b1t + ((grad_alpha_t * event_flip) + (grad_alpha_v * event_dot_flip)));
            g_b1pv = (g_b1pv + grad_phi_v);
            g_b1pt = (g_b1pt + grad_phi_t);
        }
        auto t131_ = _shift_adjoint(pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count);
        avr = bsk::get<0>(t131_);
        avi = bsk::get<1>(t131_);
        bvr = bsk::get<2>(t131_);
        bvi = bsk::get<3>(t131_);
        auto t132_ = _shift_adjoint(pbtr, pbti, mbtr, mbti, state, state_mask, state_count);
        atr = bsk::get<0>(t132_);
        ati = bsk::get<1>(t132_);
        btr = bsk::get<2>(t132_);
        bti = bsk::get<3>(t132_);
        pbvr = bsk::where(pre_shift, avr, pbvr);
        pbvi = bsk::where(pre_shift, avi, pbvi);
        pbtr = bsk::where(pre_shift, atr, pbtr);
        pbti = bsk::where(pre_shift, ati, pbti);
        mbvr = bsk::where(pre_shift, bvr, mbvr);
        mbvi = bsk::where(pre_shift, bvi, mbvi);
        mbtr = bsk::where(pre_shift, btr, mbtr);
        mbti = bsk::where(pre_shift, bti, mbti);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t133_ = _shift_adjoint(ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count);
            avr = bsk::get<0>(t133_);
            avi = bsk::get<1>(t133_);
            bvr = bsk::get<2>(t133_);
            bvi = bsk::get<3>(t133_);
            auto t134_ = _shift_adjoint(ubtr, ubti, wbtr, wbti, state, state_mask, state_count);
            atr = bsk::get<0>(t134_);
            ati = bsk::get<1>(t134_);
            btr = bsk::get<2>(t134_);
            bti = bsk::get<3>(t134_);
            ubvr = bsk::where(pre_shift, avr, ubvr);
            ubvi = bsk::where(pre_shift, avi, ubvi);
            ubtr = bsk::where(pre_shift, atr, ubtr);
            ubti = bsk::where(pre_shift, ati, ubti);
            wbvr = bsk::where(pre_shift, bvr, wbvr);
            wbvi = bsk::where(pre_shift, bvi, wbvi);
            wbtr = bsk::where(pre_shift, btr, wbtr);
            wbti = bsk::where(pre_shift, bti, wbti);
        }
        // ---- relaxation and off-resonance adjoint ----
        grad_e2_v = zero;
        grad_e2_t = zero;
        attenuation_v = zero;
        attenuation_t = zero;
        two_pool_dt_v = zero;
        two_pool_dt_t = zero;
        // The damping is homogeneous of degree one in every transverse state it
        // acts on, so its gradient times the damping itself is the cotangent
        // taken against the states the interval leaves. With one pool that is
        // the same thing as the relaxation factor's own gradient, scaled.
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto plus_side = _dual_add(_dual_product(_dual_conj(bsk::make_tup(pbvr, pbvi, pbtr, pbti)), bsk::make_tup(rpvr, rpvi, rptr, rpti)), _dual_product(_dual_conj(bsk::make_tup(ubvr, ubvi, ubtr, ubti)), bsk::make_tup(rbpvr, rbpvi, rbptr, rbpti)));
            auto minus_side = _dual_add(_dual_product(_dual_conj(bsk::make_tup(mbvr, mbvi, mbtr, mbti)), bsk::make_tup(rmvr, rmvi, rmtr, rmti)), _dual_product(_dual_conj(bsk::make_tup(wbvr, wbvi, wbtr, wbti)), bsk::make_tup(rbmvr, rbmvi, rbmtr, rbmti)));
            damped = _dual_add(plus_side, minus_side);
            auto wound = [&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(_dual_subtract(plus_side, minus_side));
            cot2_v = bsk::get<0>(damped);
            cot2_t = bsk::get<2>(damped);
            per_angle_v = bsk::get<0>(wound);
            per_angle_t = bsk::get<2>(wound);
        } else {
            auto pq = _dual_mul(qr, qi, qtr, qti, xpvr, xpvi, xptr, xpti);
            auto mq = _dual_mul(qr, (-qi), qtr, (-qti), xmvr, xmvi, xmtr, xmti);
            auto t135_ = _dual_real_conj_mul(pbvr, pbvi, pbtr, pbti, bsk::get<0>(pq), bsk::get<1>(pq), bsk::get<2>(pq), bsk::get<3>(pq));
            auto e2_v = bsk::get<0>(t135_);
            auto e2_t = bsk::get<1>(t135_);
            auto t136_ = _dual_real_conj_mul(mbvr, mbvi, mbtr, mbti, bsk::get<0>(mq), bsk::get<1>(mq), bsk::get<2>(mq), bsk::get<3>(mq));
            part_v = bsk::get<0>(t136_);
            part_t = bsk::get<1>(t136_);
            auto bare_cot_v = (e2_v + part_v);
            auto bare_cot_t = (e2_t + part_t);
            grad_e2_v = bsk::sum_x((bare_cot_v * damp_t));
            grad_e2_t = bsk::sum_x(((bare_cot_v * damp_t_tangent) + (bare_cot_t * damp_t)));
            per_angle_v = empty;
            per_angle_t = empty;
            if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
                po = _dual_mul(ovr, ovi, otr, oti, xpvr, xpvi, xptr, xpti);
                po = _dual_times_i(bsk::get<0>(po), bsk::get<1>(po), bsk::get<2>(po), bsk::get<3>(po));
                mo = _dual_mul(ovr, (-ovi), otr, (-oti), xmvr, xmvi, xmtr, xmti);
                mo = _dual_times_i(bsk::get<0>(mo), bsk::get<1>(mo), bsk::get<2>(mo), bsk::get<3>(mo));
                auto t137_ = _dual_real_conj_mul(pbvr, pbvi, pbtr, pbti, bsk::get<0>(po), bsk::get<1>(po), bsk::get<2>(po), bsk::get<3>(po));
                auto angle_v = bsk::get<0>(t137_);
                auto angle_t = bsk::get<1>(t137_);
                auto t138_ = _dual_real_conj_mul(mbvr, mbvi, mbtr, mbti, bsk::get<0>(mo), bsk::get<1>(mo), bsk::get<2>(mo), bsk::get<3>(mo));
                part_v = bsk::get<0>(t138_);
                part_t = bsk::get<1>(t138_);
                per_angle_v = (angle_v - part_v);
                per_angle_t = (angle_t - part_t);
            }
            cot2_v = ((bare_cot_v * bare2_value) * damp_t);
            cot2_t = ((((bare_cot_t * bare2_value) * damp_t) + ((bare_cot_v * bare2_tangent) * damp_t)) + ((bare_cot_v * bare2_value) * damp_t_tangent));
        }
        // A turn of the transverse states and the off-resonance angle are the
        // same derivative; only the weight each order carries differs.
        grad_angle_v = zero;
        grad_angle_t = zero;
        if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
            grad_angle_v = bsk::sum_x(per_angle_v);
            grad_angle_t = bsk::sum_x(per_angle_t);
        }
        grad_e1_v = zero;
        grad_e1_t = zero;
        if (bsk::truth((pools == 3))) {
            // The nine entries of the mixing operator and the three recoveries,
            // summed over the orders that share them, then pushed back through
            // the closed form once for the whole interval and in double.
            free_bar = bsk::make_tup(zbvr, zbvi, zbtr, zbti);
            bound_bar = bsk::make_tup(bbvr, bbvi, bbtr, bbti);
            auto semi_bar = bsk::make_tup(cbvr, cbvi, cbtr, cbti);
            spun_free = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), xzvr, xzvi, xztr, xzti);
            spun_bound = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), xbvr, xbvi, xbtr, xbti);
            auto spun_semi = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), xcvr, xcvi, xctr, xcti);
            auto t139_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, spun_free);
            e11_v = bsk::get<0>(t139_);
            e11_t = bsk::get<1>(t139_);
            auto t140_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, spun_bound);
            e12_v = bsk::get<0>(t140_);
            e12_t = bsk::get<1>(t140_);
            auto t141_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, spun_semi);
            auto e13_v = bsk::get<0>(t141_);
            auto e13_t = bsk::get<1>(t141_);
            auto t142_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, spun_free);
            e21_v = bsk::get<0>(t142_);
            e21_t = bsk::get<1>(t142_);
            auto t143_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, spun_bound);
            e22_v = bsk::get<0>(t143_);
            e22_t = bsk::get<1>(t143_);
            auto t144_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, spun_semi);
            auto e23_v = bsk::get<0>(t144_);
            auto e23_t = bsk::get<1>(t144_);
            auto t145_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(semi_bar, spun_free);
            auto e31_v = bsk::get<0>(t145_);
            auto e31_t = bsk::get<1>(t145_);
            auto t146_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(semi_bar, spun_bound);
            auto e32_v = bsk::get<0>(t146_);
            auto e32_t = bsk::get<1>(t146_);
            auto t147_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(semi_bar, spun_semi);
            auto e33_v = bsk::get<0>(t147_);
            auto e33_t = bsk::get<1>(t147_);
            if (bsk::truth(tabulated)) {
                // Every gradient but the interval's own and the
                // attenuation's is linear in these cotangents, so the
                // events sharing a length pool them here and pay the
                // closed form once each after the walk back. The tangent
                // gradient carries a third term in the event's own
                // interval direction, which pools as the value cotangents
                // weighted by it.
                auto bar11_v = bsk::sum_x(e11_v);
                auto bar12_v = bsk::sum_x(e12_v);
                auto bar13_v = bsk::sum_x(e13_v);
                auto bar21_v = bsk::sum_x(e21_v);
                auto bar22_v = bsk::sum_x(e22_v);
                auto bar23_v = bsk::sum_x(e23_v);
                auto bar31_v = bsk::sum_x(e31_v);
                auto bar32_v = bsk::sum_x(e32_v);
                auto bar33_v = bsk::sum_x(e33_v);
                auto barfree_v = bsk::sum_x(bsk::where((state == 0), zbvr, 0.0f));
                auto barpool_v = bsk::sum_x(bsk::where((state == 0), bbvr, 0.0f));
                auto barbound_v = bsk::sum_x(bsk::where((state == 0), cbvr, 0.0f));
                auto bar11_t = bsk::sum_x(e11_t);
                auto bar12_t = bsk::sum_x(e12_t);
                auto bar13_t = bsk::sum_x(e13_t);
                auto bar21_t = bsk::sum_x(e21_t);
                auto bar22_t = bsk::sum_x(e22_t);
                auto bar23_t = bsk::sum_x(e23_t);
                auto bar31_t = bsk::sum_x(e31_t);
                auto bar32_t = bsk::sum_x(e32_t);
                auto bar33_t = bsk::sum_x(e33_t);
                auto barfree_t = bsk::sum_x(bsk::where((state == 0), zbtr, 0.0f));
                auto barpool_t = bsk::sum_x(bsk::where((state == 0), bbtr, 0.0f));
                auto barbound_t = bsk::sum_x(bsk::where((state == 0), cbtr, 0.0f));
                held = (pool_bars + (((local * row_count) + pool_row) * 36));
                bsk::st((held + 0), (bsk::ld((held + 0), active_atom, 0.0f) + bar11_v), active_atom);
                bsk::st((held + 1), (bsk::ld((held + 1), active_atom, 0.0f) + bar12_v), active_atom);
                bsk::st((held + 2), (bsk::ld((held + 2), active_atom, 0.0f) + bar13_v), active_atom);
                bsk::st((held + 3), (bsk::ld((held + 3), active_atom, 0.0f) + bar21_v), active_atom);
                bsk::st((held + 4), (bsk::ld((held + 4), active_atom, 0.0f) + bar22_v), active_atom);
                bsk::st((held + 5), (bsk::ld((held + 5), active_atom, 0.0f) + bar23_v), active_atom);
                bsk::st((held + 6), (bsk::ld((held + 6), active_atom, 0.0f) + bar31_v), active_atom);
                bsk::st((held + 7), (bsk::ld((held + 7), active_atom, 0.0f) + bar32_v), active_atom);
                bsk::st((held + 8), (bsk::ld((held + 8), active_atom, 0.0f) + bar33_v), active_atom);
                bsk::st((held + 9), (bsk::ld((held + 9), active_atom, 0.0f) + barfree_v), active_atom);
                bsk::st((held + 10), (bsk::ld((held + 10), active_atom, 0.0f) + barpool_v), active_atom);
                bsk::st((held + 11), (bsk::ld((held + 11), active_atom, 0.0f) + barbound_v), active_atom);
                bsk::st((held + 12), (bsk::ld((held + 12), active_atom, 0.0f) + bar11_t), active_atom);
                bsk::st((held + 13), (bsk::ld((held + 13), active_atom, 0.0f) + bar12_t), active_atom);
                bsk::st((held + 14), (bsk::ld((held + 14), active_atom, 0.0f) + bar13_t), active_atom);
                bsk::st((held + 15), (bsk::ld((held + 15), active_atom, 0.0f) + bar21_t), active_atom);
                bsk::st((held + 16), (bsk::ld((held + 16), active_atom, 0.0f) + bar22_t), active_atom);
                bsk::st((held + 17), (bsk::ld((held + 17), active_atom, 0.0f) + bar23_t), active_atom);
                bsk::st((held + 18), (bsk::ld((held + 18), active_atom, 0.0f) + bar31_t), active_atom);
                bsk::st((held + 19), (bsk::ld((held + 19), active_atom, 0.0f) + bar32_t), active_atom);
                bsk::st((held + 20), (bsk::ld((held + 20), active_atom, 0.0f) + bar33_t), active_atom);
                bsk::st((held + 21), (bsk::ld((held + 21), active_atom, 0.0f) + barfree_t), active_atom);
                bsk::st((held + 22), (bsk::ld((held + 22), active_atom, 0.0f) + barpool_t), active_atom);
                bsk::st((held + 23), (bsk::ld((held + 23), active_atom, 0.0f) + barbound_t), active_atom);
                bsk::st((held + 24), (bsk::ld((held + 24), active_atom, 0.0f) + (dt_tangent * bar11_v)), active_atom);
                bsk::st((held + 25), (bsk::ld((held + 25), active_atom, 0.0f) + (dt_tangent * bar12_v)), active_atom);
                bsk::st((held + 26), (bsk::ld((held + 26), active_atom, 0.0f) + (dt_tangent * bar13_v)), active_atom);
                bsk::st((held + 27), (bsk::ld((held + 27), active_atom, 0.0f) + (dt_tangent * bar21_v)), active_atom);
                bsk::st((held + 28), (bsk::ld((held + 28), active_atom, 0.0f) + (dt_tangent * bar22_v)), active_atom);
                bsk::st((held + 29), (bsk::ld((held + 29), active_atom, 0.0f) + (dt_tangent * bar23_v)), active_atom);
                bsk::st((held + 30), (bsk::ld((held + 30), active_atom, 0.0f) + (dt_tangent * bar31_v)), active_atom);
                bsk::st((held + 31), (bsk::ld((held + 31), active_atom, 0.0f) + (dt_tangent * bar32_v)), active_atom);
                bsk::st((held + 32), (bsk::ld((held + 32), active_atom, 0.0f) + (dt_tangent * bar33_v)), active_atom);
                bsk::st((held + 33), (bsk::ld((held + 33), active_atom, 0.0f) + (dt_tangent * barfree_v)), active_atom);
                bsk::st((held + 34), (bsk::ld((held + 34), active_atom, 0.0f) + (dt_tangent * barpool_v)), active_atom);
                bsk::st((held + 35), (bsk::ld((held + 35), active_atom, 0.0f) + (dt_tangent * barbound_v)), active_atom);
                auto t148_ = _three_pool_interval_adjoint_jvp(pool_table, pool_row, atom, atom_count, active_atom, r1_value, r1_tangent, r1b_value, r1b_tangent, r1c_value, r1c_tangent, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, dt_tangent, wout_value, wout_tangent, bar11_v, bar12_v, bar13_v, bar21_v, bar22_v, bar23_v, bar31_v, bar32_v, bar33_v, barfree_v, barpool_v, barbound_v, bar11_t, bar12_t, bar13_t, bar21_t, bar22_t, bar23_t, bar31_t, bar32_t, bar33_t, barfree_t, barpool_t, barbound_t);
                back_dt_v = bsk::get<0>(t148_);
                back_att_v = bsk::get<1>(t148_);
                back_dt_t = bsk::get<2>(t148_);
                back_att_t = bsk::get<3>(t148_);
                attenuation_v = back_att_v;
                attenuation_t = back_att_t;
                two_pool_dt_v = back_dt_v;
                two_pool_dt_t = back_dt_t;
            } else {
                auto t149_ = _three_pool_step_adjoint_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, r1c_value, r1c_tangent, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, dt_value, dt_tangent, wout_value, wout_tangent, bsk::sum_x(e11_v), bsk::sum_x(e11_t), bsk::sum_x(e12_v), bsk::sum_x(e12_t), bsk::sum_x(e13_v), bsk::sum_x(e13_t), bsk::sum_x(e21_v), bsk::sum_x(e21_t), bsk::sum_x(e22_v), bsk::sum_x(e22_t), bsk::sum_x(e23_v), bsk::sum_x(e23_t), bsk::sum_x(e31_v), bsk::sum_x(e31_t), bsk::sum_x(e32_v), bsk::sum_x(e32_t), bsk::sum_x(e33_v), bsk::sum_x(e33_t), bsk::sum_x(bsk::where((state == 0), zbvr, 0.0f)), bsk::sum_x(bsk::where((state == 0), zbtr, 0.0f)), bsk::sum_x(bsk::where((state == 0), bbvr, 0.0f)), bsk::sum_x(bsk::where((state == 0), bbtr, 0.0f)), bsk::sum_x(bsk::where((state == 0), cbvr, 0.0f)), bsk::sum_x(bsk::where((state == 0), cbtr, 0.0f)), three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, three_def_00, three_dif_00, three_def_01, three_dif_01, three_def_02, three_dif_02, three_def_10, three_dif_10, three_def_11, three_dif_11, three_def_12, three_dif_12, three_def_20, three_dif_20, three_def_21, three_dif_21, three_def_22, three_dif_22, narrow);
                back_r1_v = bsk::get<0>(t149_);
                back_r1b_v = bsk::get<1>(t149_);
                back_r1c_v = bsk::get<2>(t149_);
                back_exch_v = bsk::get<3>(t149_);
                back_sexch_v = bsk::get<4>(t149_);
                back_bound_v = bsk::get<5>(t149_);
                back_semi_v = bsk::get<6>(t149_);
                back_dt_v = bsk::get<7>(t149_);
                back_att_v = bsk::get<8>(t149_);
                back_r1_t = bsk::get<9>(t149_);
                back_r1b_t = bsk::get<10>(t149_);
                back_r1c_t = bsk::get<11>(t149_);
                back_exch_t = bsk::get<12>(t149_);
                back_sexch_t = bsk::get<13>(t149_);
                back_bound_t = bsk::get<14>(t149_);
                back_semi_t = bsk::get<15>(t149_);
                back_dt_t = bsk::get<16>(t149_);
                back_att_t = bsk::get<17>(t149_);
                slope1_v = bsk::truediv(-1000.0f, (atom_t1 * atom_t1));
                slope1_t = bsk::truediv((2000.0f * d_t1), ((atom_t1 * atom_t1) * atom_t1));
                slope1b_v = bsk::truediv(-1000.0f, (atom_t1b * atom_t1b));
                slope1b_t = bsk::truediv((2000.0f * d_t1b), ((atom_t1b * atom_t1b) * atom_t1b));
                slope1c_v = bsk::truediv(-1000.0f, (held_semisolid * held_semisolid));
                slope1c_t = bsk::truediv((2000.0f * d_semisolid_t1), ((held_semisolid * held_semisolid) * held_semisolid));
                g_t1v = (g_t1v + (back_r1_v * slope1_v));
                g_t1t = (g_t1t + ((back_r1_t * slope1_v) + (back_r1_v * slope1_t)));
                g_t1bv = (g_t1bv + (back_r1b_v * slope1b_v));
                g_t1bt = (g_t1bt + ((back_r1b_t * slope1b_v) + (back_r1b_v * slope1b_t)));
                g_t1cv = (g_t1cv + (back_r1c_v * slope1c_v));
                g_t1ct = (g_t1ct + ((back_r1c_t * slope1c_v) + (back_r1c_v * slope1c_t)));
                g_exchv = (g_exchv + back_exch_v);
                g_excht = (g_excht + back_exch_t);
                g_sexchv = (g_sexchv + back_sexch_v);
                g_sexcht = (g_sexcht + back_sexch_t);
                g_boundv = (g_boundv + back_bound_v);
                g_boundt = (g_boundt + back_bound_t);
                g_semiv = (g_semiv + back_semi_v);
                g_semit = (g_semit + back_semi_t);
                attenuation_v = back_att_v;
                attenuation_t = back_att_t;
                two_pool_dt_v = back_dt_v;
                two_pool_dt_t = back_dt_t;
            }
            auto turned_free = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_free);
            auto turned_bound = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_bound);
            auto turned_semi = [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_semisolid);
            auto t150_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, turned_free);
            damp_pair_v = bsk::get<0>(t150_);
            damp_pair_t = bsk::get<1>(t150_);
            auto t151_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, turned_bound);
            other_v = bsk::get<0>(t151_);
            other_t = bsk::get<1>(t151_);
            auto t152_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(semi_bar, turned_semi);
            auto stuck_v = bsk::get<0>(t152_);
            auto stuck_t = bsk::get<1>(t152_);
            long_damp_v = ((damp_pair_v + other_v) + stuck_v);
            long_damp_t = ((damp_pair_t + other_t) + stuck_t);
            auto t153_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, [&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(turned_free));
            zangle_v = bsk::get<0>(t153_);
            zangle_t = bsk::get<1>(t153_);
            auto t154_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, [&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(turned_bound));
            part_v = bsk::get<0>(t154_);
            part_t = bsk::get<1>(t154_);
            zangle_v = (zangle_v + part_v);
            zangle_t = (zangle_t + part_t);
            auto t155_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(semi_bar, [&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }(turned_semi));
            part_v = bsk::get<0>(t155_);
            part_t = bsk::get<1>(t155_);
            zangle_v = (zangle_v + part_v);
            zangle_t = (zangle_t + part_t);
            auto col_free = _dual_add(_dual_add([&](const auto& s2_, const auto& s3_) { return _dual_back(w11, d_w11, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, free_bar), [&](const auto& s2_, const auto& s3_) { return _dual_back(w21, d_w21, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, bound_bar)), [&](const auto& s2_, const auto& s3_) { return _dual_back(w31, d_w31, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, semi_bar));
            auto col_bound = _dual_add(_dual_add([&](const auto& s2_, const auto& s3_) { return _dual_back(w12, d_w12, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, free_bar), [&](const auto& s2_, const auto& s3_) { return _dual_back(w22, d_w22, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, bound_bar)), [&](const auto& s2_, const auto& s3_) { return _dual_back(w32, d_w32, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, semi_bar));
            auto col_semi = _dual_add(_dual_add([&](const auto& s2_, const auto& s3_) { return _dual_back(w13, d_w13, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, free_bar), [&](const auto& s2_, const auto& s3_) { return _dual_back(w23, d_w23, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, bound_bar)), [&](const auto& s2_, const auto& s3_) { return _dual_back(w33, d_w33, bsk::get<0>(s2_), bsk::get<1>(s2_), bsk::get<2>(s2_), bsk::get<3>(s2_), bsk::get<0>(s3_), bsk::get<1>(s3_), bsk::get<2>(s3_), bsk::get<3>(s3_)); }(spin, semi_bar));
            auto t156_ = col_free;
            zbvr = bsk::get<0>(t156_);
            zbvi = bsk::get<1>(t156_);
            zbtr = bsk::get<2>(t156_);
            zbti = bsk::get<3>(t156_);
            auto t157_ = col_bound;
            bbvr = bsk::get<0>(t157_);
            bbvi = bsk::get<1>(t157_);
            bbtr = bsk::get<2>(t157_);
            bbti = bsk::get<3>(t157_);
            auto t158_ = col_semi;
            cbvr = bsk::get<0>(t158_);
            cbvi = bsk::get<1>(t158_);
            cbtr = bsk::get<2>(t158_);
            cbti = bsk::get<3>(t158_);
        } else if (bsk::truth((pools > 0))) {
            // The four entries of the exchange operator and the two recoveries,
            // summed over the orders that share them, then pushed back through
            // the closed form once for the whole interval.
            free_bar = bsk::make_tup(zbvr, zbvi, zbtr, zbti);
            bound_bar = bsk::make_tup(bbvr, bbvi, bbtr, bbti);
            spun_free = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), xzvr, xzvi, xztr, xzti);
            spun_bound = _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), xbvr, xbvi, xbtr, xbti);
            auto t159_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, spun_free);
            e11_v = bsk::get<0>(t159_);
            e11_t = bsk::get<1>(t159_);
            auto t160_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, spun_bound);
            e12_v = bsk::get<0>(t160_);
            e12_t = bsk::get<1>(t160_);
            auto t161_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, spun_free);
            e21_v = bsk::get<0>(t161_);
            e21_t = bsk::get<1>(t161_);
            auto t162_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, spun_bound);
            e22_v = bsk::get<0>(t162_);
            e22_t = bsk::get<1>(t162_);
            auto bar_e11_v = bsk::sum_x(e11_v);
            auto bar_e11_t = bsk::sum_x(e11_t);
            auto bar_e12_v = bsk::sum_x(e12_v);
            auto bar_e12_t = bsk::sum_x(e12_t);
            auto bar_e21_v = bsk::sum_x(e21_v);
            auto bar_e21_t = bsk::sum_x(e21_t);
            auto bar_e22_v = bsk::sum_x(e22_v);
            auto bar_e22_t = bsk::sum_x(e22_t);
            auto rec_f_v = bsk::sum_x(bsk::where((state == 0), zbvr, 0.0f));
            auto rec_f_t = bsk::sum_x(bsk::where((state == 0), zbtr, 0.0f));
            auto rec_b_v = bsk::sum_x(bsk::where((state == 0), bbvr, 0.0f));
            auto rec_b_t = bsk::sum_x(bsk::where((state == 0), bbtr, 0.0f));
            auto t163_ = _two_pool_step_adjoint_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, atom_exchange, d_exchange, atom_bound, d_boundf, dt_value, dt_tangent, wout_value, wout_tangent, bar_e11_v, bar_e11_t, bar_e12_v, bar_e12_t, bar_e21_v, bar_e21_t, bar_e22_v, bar_e22_t, rec_f_v, rec_f_t, rec_b_v, rec_b_t);
            back_r1_v = bsk::get<0>(t163_);
            back_r1b_v = bsk::get<1>(t163_);
            back_exch_v = bsk::get<2>(t163_);
            back_bound_v = bsk::get<3>(t163_);
            back_dt_v = bsk::get<4>(t163_);
            back_att_v = bsk::get<5>(t163_);
            back_r1_t = bsk::get<6>(t163_);
            back_r1b_t = bsk::get<7>(t163_);
            back_exch_t = bsk::get<8>(t163_);
            back_bound_t = bsk::get<9>(t163_);
            back_dt_t = bsk::get<10>(t163_);
            back_att_t = bsk::get<11>(t163_);
            // r1 = 1000/t1, so a rate gradient reaches the time through the
            // square of it.
            slope1_v = bsk::truediv(-1000.0f, (atom_t1 * atom_t1));
            slope1_t = bsk::truediv((2000.0f * d_t1), ((atom_t1 * atom_t1) * atom_t1));
            slope1b_v = bsk::truediv(-1000.0f, (atom_t1b * atom_t1b));
            slope1b_t = bsk::truediv((2000.0f * d_t1b), ((atom_t1b * atom_t1b) * atom_t1b));
            g_t1v = (g_t1v + (back_r1_v * slope1_v));
            g_t1t = (g_t1t + ((back_r1_t * slope1_v) + (back_r1_v * slope1_t)));
            g_t1bv = (g_t1bv + (back_r1b_v * slope1b_v));
            g_t1bt = (g_t1bt + ((back_r1b_t * slope1b_v) + (back_r1b_v * slope1b_t)));
            g_exchv = (g_exchv + back_exch_v);
            g_excht = (g_excht + back_exch_t);
            g_boundv = (g_boundv + back_bound_v);
            g_boundt = (g_boundt + back_bound_t);
            attenuation_v = back_att_v;
            attenuation_t = back_att_t;
            two_pool_dt_v = back_dt_v;
            two_pool_dt_t = back_dt_t;
            // Both pools take the same per-order damping and turn, so each
            // collects the cotangent of the mixture that reached it.
            auto t164_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_free));
            damp_pair_v = bsk::get<0>(t164_);
            damp_pair_t = bsk::get<1>(t164_);
            auto t165_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, [&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_bound));
            other_v = bsk::get<0>(t165_);
            other_t = bsk::get<1>(t165_);
            long_damp_v = (damp_pair_v + other_v);
            long_damp_t = (damp_pair_t + other_t);
            auto spun_mix_free = [&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }([&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_free));
            auto spun_mix_bound = [&](const auto& s0_) { return _dual_times_i(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_)); }([&](const auto& s4_) { return _dual_mul(bsk::get<0>(spin), bsk::get<1>(spin), bsk::get<2>(spin), bsk::get<3>(spin), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(mixed_bound));
            auto t166_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(free_bar, spun_mix_free);
            zangle_v = bsk::get<0>(t166_);
            zangle_t = bsk::get<1>(t166_);
            auto t167_ = [&](const auto& s0_, const auto& s1_) { return _dual_real_conj_mul(bsk::get<0>(s0_), bsk::get<1>(s0_), bsk::get<2>(s0_), bsk::get<3>(s0_), bsk::get<0>(s1_), bsk::get<1>(s1_), bsk::get<2>(s1_), bsk::get<3>(s1_)); }(bound_bar, spun_mix_bound);
            part_v = bsk::get<0>(t167_);
            part_t = bsk::get<1>(t167_);
            zangle_v = (zangle_v + part_v);
            zangle_t = (zangle_t + part_t);
            auto back_z = [&](const auto& s4_) { return _dual_mul((pe11 * bsk::get<0>(spin)), (-(pe11 * bsk::get<1>(spin))), ((de11 * bsk::get<0>(spin)) + (pe11 * bsk::get<2>(spin))), (-((de11 * bsk::get<1>(spin)) + (pe11 * bsk::get<3>(spin)))), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(free_bar);
            auto cross_z = [&](const auto& s4_) { return _dual_mul((pe21 * bsk::get<0>(spin)), (-(pe21 * bsk::get<1>(spin))), ((de21 * bsk::get<0>(spin)) + (pe21 * bsk::get<2>(spin))), (-((de21 * bsk::get<1>(spin)) + (pe21 * bsk::get<3>(spin)))), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(bound_bar);
            auto back_b = [&](const auto& s4_) { return _dual_mul((pe12 * bsk::get<0>(spin)), (-(pe12 * bsk::get<1>(spin))), ((de12 * bsk::get<0>(spin)) + (pe12 * bsk::get<2>(spin))), (-((de12 * bsk::get<1>(spin)) + (pe12 * bsk::get<3>(spin)))), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(free_bar);
            auto cross_b = [&](const auto& s4_) { return _dual_mul((pe22 * bsk::get<0>(spin)), (-(pe22 * bsk::get<1>(spin))), ((de22 * bsk::get<0>(spin)) + (pe22 * bsk::get<2>(spin))), (-((de22 * bsk::get<1>(spin)) + (pe22 * bsk::get<3>(spin)))), bsk::get<0>(s4_), bsk::get<1>(s4_), bsk::get<2>(s4_), bsk::get<3>(s4_)); }(bound_bar);
            auto next_zbvr = (bsk::get<0>(back_z) + bsk::get<0>(cross_z));
            auto next_zbvi = (bsk::get<1>(back_z) + bsk::get<1>(cross_z));
            auto next_zbtr = (bsk::get<2>(back_z) + bsk::get<2>(cross_z));
            auto next_zbti = (bsk::get<3>(back_z) + bsk::get<3>(cross_z));
            bbvr = (bsk::get<0>(back_b) + bsk::get<0>(cross_b));
            bbvi = (bsk::get<1>(back_b) + bsk::get<1>(cross_b));
            bbtr = (bsk::get<2>(back_b) + bsk::get<2>(cross_b));
            bbti = (bsk::get<3>(back_b) + bsk::get<3>(cross_b));
            zbvr = next_zbvr;
            zbvi = next_zbvi;
            zbtr = next_zbtr;
            zbti = next_zbti;
        } else {
            auto spun = _dual_mul(szr, szi, sztr, szti, xzvr, xzvi, xztr, xzti);
            auto t168_ = _dual_real_conj_mul(zbvr, zbvi, zbtr, zbti, bsk::get<0>(spun), bsk::get<1>(spun), bsk::get<2>(spun), bsk::get<3>(spun));
            auto e1_v = bsk::get<0>(t168_);
            auto e1_t = bsk::get<1>(t168_);
            grad_e1_v = bsk::sum_x((e1_v * damp_z));
            grad_e1_t = bsk::sum_x(((e1_v * damp_z_tangent) + (e1_t * damp_z)));
            grad_e1_v = (grad_e1_v - bsk::sum_x(bsk::where((state == 0), zbvr, 0.0f)));
            grad_e1_t = (grad_e1_t - bsk::sum_x(bsk::where((state == 0), zbtr, 0.0f)));
            long_damp_v = ((e1_v * bare1_value) * damp_z);
            long_damp_t = ((((e1_t * bare1_value) * damp_z) + ((e1_v * bare1_tangent) * damp_z)) + ((e1_v * bare1_value) * damp_z_tangent));
            // The longitudinal states turn too, and by a whole order rather
            // than the transverse half-order more.
            zo = _dual_mul(lvr, lvi, ltr, lti, xzvr, xzvi, xztr, xzti);
            zo = _dual_times_i(bsk::get<0>(zo), bsk::get<1>(zo), bsk::get<2>(zo), bsk::get<3>(zo));
            auto t169_ = _dual_real_conj_mul(zbvr, zbvi, zbtr, zbti, bsk::get<0>(zo), bsk::get<1>(zo), bsk::get<2>(zo), bsk::get<3>(zo));
            zangle_v = bsk::get<0>(t169_);
            zangle_t = bsk::get<1>(t169_);
            auto t170_ = _dual_mul(lvr, (-lvi), ltr, (-lti), zbvr, zbvi, zbtr, zbti);
            zbvr = bsk::get<0>(t170_);
            zbvi = bsk::get<1>(t170_);
            zbtr = bsk::get<2>(t170_);
            zbti = bsk::get<3>(t170_);
        }
        next_pb = bsk::make_tup(pbvr, pbvi, pbtr, pbti);
        next_mb = bsk::make_tup(mbvr, mbvi, mbtr, mbti);
        next_ub = bsk::make_tup(ubvr, ubvi, ubtr, ubti);
        next_wb = bsk::make_tup(wbvr, wbvi, wbtr, wbti);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // The four entries of the transverse operator, summed over the
            // orders that share them, then pushed back through the closed form
            // once for the whole interval. ``F-`` follows the conjugate of the
            // operator, so its cotangent lands on the entry itself rather than
            // on the conjugate of it.
            auto ap = bsk::make_tup(pbvr, pbvi, pbtr, pbti);
            auto am = bsk::make_tup(mbvr, mbvi, mbtr, mbti);
            auto aub = bsk::make_tup(ubvr, ubvi, ubtr, ubti);
            auto awb = bsk::make_tup(wbvr, wbvi, wbtr, wbti);
            auto fp = bsk::make_tup(xpvr, xpvi, xptr, xpti);
            auto fm = bsk::make_tup(xmvr, xmvi, xmtr, xmti);
            auto bp = bsk::make_tup(xbpvr, xbpvi, xbptr, xbpti);
            auto bm = bsk::make_tup(xbmvr, xbmvi, xbmtr, xbmti);
            auto term11 = _dual_product(_dual_add(_dual_product(_dual_conj(ap), fp), _dual_product(am, _dual_conj(fm))), carried);
            auto term12 = _dual_product(_dual_add(_dual_product(_dual_conj(ap), bp), _dual_product(am, _dual_conj(bm))), carried);
            auto term21 = _dual_product(_dual_add(_dual_product(_dual_conj(aub), fp), _dual_product(awb, _dual_conj(fm))), carried);
            auto term22 = _dual_product(_dual_add(_dual_product(_dual_conj(aub), bp), _dual_product(awb, _dual_conj(bm))), carried);
            auto bar11 = bsk::make_tup(bsk::sum_x(bsk::get<0>(term11)), bsk::sum_x(bsk::get<1>(term11)), bsk::sum_x(bsk::get<2>(term11)), bsk::sum_x(bsk::get<3>(term11)));
            auto bar12 = bsk::make_tup(bsk::sum_x(bsk::get<0>(term12)), bsk::sum_x(bsk::get<1>(term12)), bsk::sum_x(bsk::get<2>(term12)), bsk::sum_x(bsk::get<3>(term12)));
            auto bar21 = bsk::make_tup(bsk::sum_x(bsk::get<0>(term21)), bsk::sum_x(bsk::get<1>(term21)), bsk::sum_x(bsk::get<2>(term21)), bsk::sum_x(bsk::get<3>(term21)));
            auto bar22 = bsk::make_tup(bsk::sum_x(bsk::get<0>(term22)), bsk::sum_x(bsk::get<1>(term22)), bsk::sum_x(bsk::get<2>(term22)), bsk::sum_x(bsk::get<3>(term22)));
            auto t171_ = _two_pool_transverse_adjoint_jvp(r2_value, r2_tangent, r2b_value, r2b_tangent, atom_exchange, d_exchange, atom_bound, d_boundf, atom_free, d_free, atom_shift, d_shift, dt_value, dt_tangent, wout_value, wout_tangent, bar11, bar12, bar21, bar22);
            auto back_r2_v = bsk::get<0>(t171_);
            auto back_r2_t = bsk::get<1>(t171_);
            auto back_r2b_v = bsk::get<2>(t171_);
            auto back_r2b_t = bsk::get<3>(t171_);
            auto back_xexch_v = bsk::get<4>(t171_);
            auto back_xexch_t = bsk::get<5>(t171_);
            auto back_xbound_v = bsk::get<6>(t171_);
            auto back_xbound_t = bsk::get<7>(t171_);
            auto back_xfree_v = bsk::get<8>(t171_);
            auto back_xfree_t = bsk::get<9>(t171_);
            auto back_shift_v = bsk::get<10>(t171_);
            auto back_shift_t = bsk::get<11>(t171_);
            auto back_xdt_v = bsk::get<12>(t171_);
            auto back_xdt_t = bsk::get<13>(t171_);
            auto back_xatt_v = bsk::get<14>(t171_);
            auto back_xatt_t = bsk::get<15>(t171_);
            auto slope2_v = bsk::truediv(-1000.0f, (atom_t2 * atom_t2));
            auto slope2_t = bsk::truediv((2000.0f * d_t2), ((atom_t2 * atom_t2) * atom_t2));
            auto slope2b_v = bsk::truediv(-1000.0f, (atom_t2b * atom_t2b));
            auto slope2b_t = bsk::truediv((2000.0f * d_t2b), ((atom_t2b * atom_t2b) * atom_t2b));
            g_t2v = (g_t2v + (back_r2_v * slope2_v));
            g_t2t = (g_t2t + ((back_r2_t * slope2_v) + (back_r2_v * slope2_t)));
            g_t2bv = (g_t2bv + (back_r2b_v * slope2b_v));
            g_t2bt = (g_t2bt + ((back_r2b_t * slope2b_v) + (back_r2b_v * slope2b_t)));
            g_exchv = (g_exchv + back_xexch_v);
            g_excht = (g_excht + back_xexch_t);
            // The free water is what both second pools leave, so a cotangent
            // on it reaches each of their fractions turned over.
            g_boundv = (g_boundv + (back_xbound_v - back_xfree_v));
            g_boundt = (g_boundt + (back_xbound_t - back_xfree_t));
            if (bsk::truth((pools == 3))) {
                g_semiv = (g_semiv - back_xfree_v);
                g_semit = (g_semit - back_xfree_t);
            }
            g_shiftv = (g_shiftv + back_shift_v);
            g_shiftt = (g_shiftt + back_shift_t);
            attenuation_v = (attenuation_v + back_xatt_v);
            attenuation_t = (attenuation_t + back_xatt_t);
            two_pool_dt_v = (two_pool_dt_v + back_xdt_v);
            two_pool_dt_t = (two_pool_dt_t + back_xdt_t);
            auto step11 = _dual_product(a11, carried);
            auto step12 = _dual_product(a12, carried);
            auto step21 = _dual_product(a21, carried);
            auto step22 = _dual_product(a22, carried);
            next_pb = _dual_add(_dual_product(_dual_conj(step11), ap), _dual_product(_dual_conj(step21), aub));
            next_ub = _dual_add(_dual_product(_dual_conj(step12), ap), _dual_product(_dual_conj(step22), aub));
            next_mb = _dual_add(_dual_product(step11, am), _dual_product(step21, awb));
            next_wb = _dual_add(_dual_product(step12, am), _dual_product(step22, awb));
        }
        // The rate and the interval multiply every order's b-weight, so both
        // take a weighted sum rather than one scalar. Order zero carries no
        // longitudinal weight, which keeps recovery out of this.
        spread_v = zero;
        spread_t = zero;
        if (bsk::truth(diffusing)) {
            auto weighted_v = ((long_damp_v * longitudinal_weight) + (cot2_v * transverse_weight));
            auto weighted_t = ((long_damp_t * longitudinal_weight) + (cot2_t * transverse_weight));
            spread_v = bsk::sum_x(weighted_v);
            spread_t = bsk::sum_x(weighted_t);
            g_diffv = (g_diffv + ((-spread_v) * dt_value));
            g_difft = (g_difft + (-((spread_v * dt_tangent) + (spread_t * dt_value))));
        }
        wound_v = zero;
        wound_t = zero;
        if (bsk::truth(moving)) {
            wound_v = bsk::sum_x(((per_angle_v * (order + 0.5f)) + (zangle_v * order)));
            wound_t = bsk::sum_x(((per_angle_t * (order + 0.5f)) + (zangle_t * order)));
            g_flowv = (g_flowv + ((-wound_v) * dt_value));
            g_flowt = (g_flowt + (-((wound_v * dt_tangent) + (wound_t * dt_value))));
        }
        // Washout scales both relaxation factors, so its gradient is the one
        // they already carry, taken against the factors before that scaling.
        // Past the clamp the interval has replaced the voxel outright and
        // nothing further depends on the rate.
        wash_v = zero;
        wash_t = zero;
        if (bsk::truth(moving)) {
            auto live = bsk::cast<float>(((atom_washout * dt_value) < 1.0f));
            wash_v = ((-live) * (((grad_e1_v * dry1_value) + (grad_e2_v * dry2_value)) + attenuation_v));
            wash_t = ((-live) * (((((grad_e1_v * dry1_tangent) + (grad_e1_t * dry1_value)) + (grad_e2_v * dry2_tangent)) + (grad_e2_t * dry2_value)) + attenuation_t));
            g_washv = (g_washv + (wash_v * dt_value));
            g_washt = (g_washt + ((wash_v * dt_tangent) + (wash_t * dt_value)));
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t172_ = next_pb;
            pbvr = bsk::get<0>(t172_);
            pbvi = bsk::get<1>(t172_);
            pbtr = bsk::get<2>(t172_);
            pbti = bsk::get<3>(t172_);
            auto t173_ = next_mb;
            mbvr = bsk::get<0>(t173_);
            mbvi = bsk::get<1>(t173_);
            mbtr = bsk::get<2>(t173_);
            mbti = bsk::get<3>(t173_);
            auto t174_ = next_ub;
            ubvr = bsk::get<0>(t174_);
            ubvi = bsk::get<1>(t174_);
            ubtr = bsk::get<2>(t174_);
            ubti = bsk::get<3>(t174_);
            auto t175_ = next_wb;
            wbvr = bsk::get<0>(t175_);
            wbvi = bsk::get<1>(t175_);
            wbtr = bsk::get<2>(t175_);
            wbti = bsk::get<3>(t175_);
        } else {
            auto t176_ = _dual_mul(ovr, (-ovi), otr, (-oti), pbvr, pbvi, pbtr, pbti);
            pbvr = bsk::get<0>(t176_);
            pbvi = bsk::get<1>(t176_);
            pbtr = bsk::get<2>(t176_);
            pbti = bsk::get<3>(t176_);
            auto t177_ = _dual_mul(ovr, ovi, otr, oti, mbvr, mbvi, mbtr, mbti);
            mbvr = bsk::get<0>(t177_);
            mbvi = bsk::get<1>(t177_);
            mbtr = bsk::get<2>(t177_);
            mbti = bsk::get<3>(t177_);
        }
        auto inverse1_value = bsk::truediv(1000.0f, (atom_t1 * atom_t1));
        auto inverse1_tangent = bsk::truediv((-2000.0f * d_t1), ((atom_t1 * atom_t1) * atom_t1));
        auto inverse2_value = bsk::truediv(1000.0f, (atom_t2 * atom_t2));
        auto inverse2_tangent = bsk::truediv((-2000.0f * d_t2), ((atom_t2 * atom_t2) * atom_t2));
        auto scale1_value = ((bare1_value * dt_value) * inverse1_value);
        scale1_tangent = ((bare1_tangent * dt_value) * inverse1_value);
        scale1_tangent = (scale1_tangent + ((bare1_value * dt_tangent) * inverse1_value));
        scale1_tangent = (scale1_tangent + ((bare1_value * dt_value) * inverse1_tangent));
        auto scale2_value = ((bare2_value * dt_value) * inverse2_value);
        scale2_tangent = ((bare2_tangent * dt_value) * inverse2_value);
        scale2_tangent = (scale2_tangent + ((bare2_value * dt_tangent) * inverse2_value));
        scale2_tangent = (scale2_tangent + ((bare2_value * dt_value) * inverse2_tangent));
        g_t1v = (g_t1v + (grad_e1_v * scale1_value));
        g_t1t = (g_t1t + ((grad_e1_v * scale1_tangent) + (grad_e1_t * scale1_value)));
        g_t2v = (g_t2v + (grad_e2_v * scale2_value));
        g_t2t = (g_t2t + ((grad_e2_v * scale2_tangent) + (grad_e2_t * scale2_value)));
        auto turn = -6.283185307179586f;
        g_b0v = (g_b0v + (grad_angle_v * (turn * dt_value)));
        g_b0t = (g_b0t + ((grad_angle_v * (turn * dt_tangent)) + (grad_angle_t * (turn * dt_value))));
        auto decay1_value = (r1_value * bare1_value);
        auto decay1_tangent = ((r1_value * bare1_tangent) + (r1_tangent * bare1_value));
        auto decay2_value = (r2_value * bare2_value);
        auto decay2_tangent = ((r2_value * bare2_tangent) + (r2_tangent * bare2_value));
        duration_v = (((-grad_e1_v) * decay1_value) - (grad_e2_v * decay2_value));
        duration_v = (duration_v + ((grad_angle_v * (turn * atom_b0)) + two_pool_dt_v));
        duration_t = (-((grad_e1_v * decay1_tangent) + (grad_e1_t * decay1_value)));
        duration_t = (duration_t - ((grad_e2_v * decay2_tangent) + (grad_e2_t * decay2_value)));
        duration_t = (duration_t + ((grad_angle_v * (turn * d_b0)) + (grad_angle_t * (turn * atom_b0))));
        duration_t = (duration_t + two_pool_dt_t);
        duration_v = (duration_v + (((-spread_v) * atom_damping) - (wound_v * atom_flow)));
        duration_t = (duration_t + (-((spread_v * d_damping) + (spread_t * atom_damping))));
        duration_t = (duration_t + (-((wound_v * d_flow) + (wound_t * atom_flow))));
        duration_v = (duration_v + (wash_v * atom_washout));
        duration_t = (duration_t + ((wash_v * d_washout) + (wash_t * atom_washout)));
        bsk::atomic_add(((grad_duration_value + event_base) + event), duration_v, active_atom);
        bsk::atomic_add(((grad_duration_tangent + event_base) + event), duration_t, active_atom);
    }
    if (bsk::truth((bsk::truth((pools == 3)) && bsk::truth(tabulated)))) {
        // One closed form per distinct length rather than one per event,
        // run twice. The walk back pooled the cotangents the eigenvalues
        // are pushed through and the closed form is linear in them, so the
        // pieces of the sum are the sum of the pieces. A gradient's own
        // direction depends on the interval as well, and a row is shared
        // by events whose interval directions differ -- so the second pass
        // takes that dependence alone, driven by the cotangents the walk
        // back weighted by each event's direction and read at a unit one.
        for (bsk::index_t row = 0; row < row_count; row += 1) {
            held = (pool_bars + (((local * row_count) + row) * 36));
            auto row_dt = (bsk::ld((pool_durations + row)) + zero);
            auto nil = (0.0f * row_dt);
            auto unit = (1.0f + nil);
            one_att = unit;
            att_rate = nil;
            att_span = nil;
            if (bsk::truth(moving)) {
                auto t178_ = _washout_jvp(atom_washout, d_washout, row_dt, nil);
                one_att = bsk::get<0>(t178_);
                att_rate = bsk::get<1>(t178_);
                auto t179_ = _washout_jvp(atom_washout, nil, row_dt, unit);
                auto _held_att = bsk::get<0>(t179_);
                att_span = bsk::get<1>(t179_);
            }
            auto t180_ = _three_pool_pieces_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, r1c_value, r1c_tangent, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, row_dt, nil, narrow);
            three_free = bsk::get<0>(t180_);
            three_d_free = bsk::get<1>(t180_);
            three_pool_b = bsk::get<2>(t180_);
            three_d_pool_b = bsk::get<3>(t180_);
            three_pool_c = bsk::get<4>(t180_);
            three_d_pool_c = bsk::get<5>(t180_);
            three_a00 = bsk::get<6>(t180_);
            three_d_a00 = bsk::get<7>(t180_);
            three_a01 = bsk::get<8>(t180_);
            three_d_a01 = bsk::get<9>(t180_);
            three_a02 = bsk::get<10>(t180_);
            three_d_a02 = bsk::get<11>(t180_);
            three_a10 = bsk::get<12>(t180_);
            three_d_a10 = bsk::get<13>(t180_);
            three_a11 = bsk::get<14>(t180_);
            three_d_a11 = bsk::get<15>(t180_);
            three_a20 = bsk::get<16>(t180_);
            three_d_a20 = bsk::get<17>(t180_);
            three_a22 = bsk::get<18>(t180_);
            three_d_a22 = bsk::get<19>(t180_);
            three_s00 = bsk::get<20>(t180_);
            three_d_s00 = bsk::get<21>(t180_);
            three_s11 = bsk::get<22>(t180_);
            three_d_s11 = bsk::get<23>(t180_);
            three_s22 = bsk::get<24>(t180_);
            three_d_s22 = bsk::get<25>(t180_);
            three_minors = bsk::get<26>(t180_);
            three_d_minors = bsk::get<27>(t180_);
            three_sum_flat = bsk::get<28>(t180_);
            three_sum_linear = bsk::get<29>(t180_);
            three_sum_square = bsk::get<30>(t180_);
            three_d_sum_flat = bsk::get<31>(t180_);
            three_d_sum_linear = bsk::get<32>(t180_);
            three_d_sum_square = bsk::get<33>(t180_);
            three_lift = bsk::get<34>(t180_);
            three_d_lift = bsk::get<35>(t180_);
            three_low = bsk::get<36>(t180_);
            three_middle = bsk::get<37>(t180_);
            three_d_low = bsk::get<38>(t180_);
            three_d_middle = bsk::get<39>(t180_);
            three_leading = bsk::get<40>(t180_);
            three_d_leading = bsk::get<41>(t180_);
            three_first = bsk::get<42>(t180_);
            three_d_first = bsk::get<43>(t180_);
            three_second = bsk::get<44>(t180_);
            three_d_second = bsk::get<45>(t180_);
            three_determinant = bsk::get<46>(t180_);
            three_d_determinant = bsk::get<47>(t180_);
            three_high = bsk::get<48>(t180_);
            three_d_high = bsk::get<49>(t180_);
            three_radius = bsk::get<50>(t180_);
            three_d_radius = bsk::get<51>(t180_);
            three_cube = bsk::get<52>(t180_);
            three_raw = bsk::get<53>(t180_);
            three_d_raw = bsk::get<54>(t180_);
            three_argument = bsk::get<55>(t180_);
            three_inside_limit = bsk::get<56>(t180_);
            three_angle = bsk::get<57>(t180_);
            three_d_angle = bsk::get<58>(t180_);
            three_centre = bsk::get<59>(t180_);
            three_d_centre = bsk::get<60>(t180_);
            three_trailing = bsk::get<61>(t180_);
            three_d_trailing = bsk::get<62>(t180_);
            three_guarded = bsk::get<63>(t180_);
            three_d_guarded = bsk::get<64>(t180_);
            three_q00 = bsk::get<65>(t180_);
            three_d_q00 = bsk::get<66>(t180_);
            three_q01 = bsk::get<67>(t180_);
            three_d_q01 = bsk::get<68>(t180_);
            three_q02 = bsk::get<69>(t180_);
            three_d_q02 = bsk::get<70>(t180_);
            three_q10 = bsk::get<71>(t180_);
            three_d_q10 = bsk::get<72>(t180_);
            three_q11 = bsk::get<73>(t180_);
            three_d_q11 = bsk::get<74>(t180_);
            three_q12 = bsk::get<75>(t180_);
            three_d_q12 = bsk::get<76>(t180_);
            three_q20 = bsk::get<77>(t180_);
            three_d_q20 = bsk::get<78>(t180_);
            three_q21 = bsk::get<79>(t180_);
            three_d_q21 = bsk::get<80>(t180_);
            three_q22 = bsk::get<81>(t180_);
            three_d_q22 = bsk::get<82>(t180_);
            auto t181_ = _three_pool_assemble_jvp(three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, narrow);
            three_def_00 = bsk::get<0>(t181_);
            three_dif_00 = bsk::get<1>(t181_);
            three_def_01 = bsk::get<2>(t181_);
            three_dif_01 = bsk::get<3>(t181_);
            three_def_02 = bsk::get<4>(t181_);
            three_dif_02 = bsk::get<5>(t181_);
            three_def_10 = bsk::get<6>(t181_);
            three_dif_10 = bsk::get<7>(t181_);
            three_def_11 = bsk::get<8>(t181_);
            three_dif_11 = bsk::get<9>(t181_);
            three_def_12 = bsk::get<10>(t181_);
            three_dif_12 = bsk::get<11>(t181_);
            three_def_20 = bsk::get<12>(t181_);
            three_dif_20 = bsk::get<13>(t181_);
            three_def_21 = bsk::get<14>(t181_);
            three_dif_21 = bsk::get<15>(t181_);
            three_def_22 = bsk::get<16>(t181_);
            three_dif_22 = bsk::get<17>(t181_);
            auto t182_ = _three_pool_step_adjoint_jvp(r1_value, r1_tangent, r1b_value, r1b_tangent, r1c_value, r1c_tangent, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_boundf, atom_semisolid, d_semisolidf, row_dt, nil, one_att, att_rate, bsk::ld((held + 0), active_atom, 0.0f), bsk::ld((held + 12), active_atom, 0.0f), bsk::ld((held + 1), active_atom, 0.0f), bsk::ld((held + 13), active_atom, 0.0f), bsk::ld((held + 2), active_atom, 0.0f), bsk::ld((held + 14), active_atom, 0.0f), bsk::ld((held + 3), active_atom, 0.0f), bsk::ld((held + 15), active_atom, 0.0f), bsk::ld((held + 4), active_atom, 0.0f), bsk::ld((held + 16), active_atom, 0.0f), bsk::ld((held + 5), active_atom, 0.0f), bsk::ld((held + 17), active_atom, 0.0f), bsk::ld((held + 6), active_atom, 0.0f), bsk::ld((held + 18), active_atom, 0.0f), bsk::ld((held + 7), active_atom, 0.0f), bsk::ld((held + 19), active_atom, 0.0f), bsk::ld((held + 8), active_atom, 0.0f), bsk::ld((held + 20), active_atom, 0.0f), bsk::ld((held + 9), active_atom, 0.0f), bsk::ld((held + 21), active_atom, 0.0f), bsk::ld((held + 10), active_atom, 0.0f), bsk::ld((held + 22), active_atom, 0.0f), bsk::ld((held + 11), active_atom, 0.0f), bsk::ld((held + 23), active_atom, 0.0f), three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, three_def_00, three_dif_00, three_def_01, three_dif_01, three_def_02, three_dif_02, three_def_10, three_dif_10, three_def_11, three_dif_11, three_def_12, three_dif_12, three_def_20, three_dif_20, three_def_21, three_dif_21, three_def_22, three_dif_22, narrow);
            back_r1_v = bsk::get<0>(t182_);
            back_r1b_v = bsk::get<1>(t182_);
            back_r1c_v = bsk::get<2>(t182_);
            back_exch_v = bsk::get<3>(t182_);
            back_sexch_v = bsk::get<4>(t182_);
            back_bound_v = bsk::get<5>(t182_);
            back_semi_v = bsk::get<6>(t182_);
            auto _row_dt_v = bsk::get<7>(t182_);
            auto _row_att_v = bsk::get<8>(t182_);
            back_r1_t = bsk::get<9>(t182_);
            back_r1b_t = bsk::get<10>(t182_);
            back_r1c_t = bsk::get<11>(t182_);
            back_exch_t = bsk::get<12>(t182_);
            back_sexch_t = bsk::get<13>(t182_);
            back_bound_t = bsk::get<14>(t182_);
            back_semi_t = bsk::get<15>(t182_);
            auto _row_dt_t = bsk::get<16>(t182_);
            auto _row_att_t = bsk::get<17>(t182_);
            auto t183_ = _three_pool_pieces_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, row_dt, unit, narrow);
            auto alt_free = bsk::get<0>(t183_);
            auto alt_d_free = bsk::get<1>(t183_);
            auto alt_pool_b = bsk::get<2>(t183_);
            auto alt_d_pool_b = bsk::get<3>(t183_);
            auto alt_pool_c = bsk::get<4>(t183_);
            auto alt_d_pool_c = bsk::get<5>(t183_);
            auto alt_a00 = bsk::get<6>(t183_);
            auto alt_d_a00 = bsk::get<7>(t183_);
            auto alt_a01 = bsk::get<8>(t183_);
            auto alt_d_a01 = bsk::get<9>(t183_);
            auto alt_a02 = bsk::get<10>(t183_);
            auto alt_d_a02 = bsk::get<11>(t183_);
            auto alt_a10 = bsk::get<12>(t183_);
            auto alt_d_a10 = bsk::get<13>(t183_);
            auto alt_a11 = bsk::get<14>(t183_);
            auto alt_d_a11 = bsk::get<15>(t183_);
            auto alt_a20 = bsk::get<16>(t183_);
            auto alt_d_a20 = bsk::get<17>(t183_);
            auto alt_a22 = bsk::get<18>(t183_);
            auto alt_d_a22 = bsk::get<19>(t183_);
            auto alt_s00 = bsk::get<20>(t183_);
            auto alt_d_s00 = bsk::get<21>(t183_);
            auto alt_s11 = bsk::get<22>(t183_);
            auto alt_d_s11 = bsk::get<23>(t183_);
            auto alt_s22 = bsk::get<24>(t183_);
            auto alt_d_s22 = bsk::get<25>(t183_);
            auto alt_minors = bsk::get<26>(t183_);
            auto alt_d_minors = bsk::get<27>(t183_);
            auto alt_sum_flat = bsk::get<28>(t183_);
            auto alt_sum_linear = bsk::get<29>(t183_);
            auto alt_sum_square = bsk::get<30>(t183_);
            auto alt_d_sum_flat = bsk::get<31>(t183_);
            auto alt_d_sum_linear = bsk::get<32>(t183_);
            auto alt_d_sum_square = bsk::get<33>(t183_);
            auto alt_lift = bsk::get<34>(t183_);
            auto alt_d_lift = bsk::get<35>(t183_);
            auto alt_low = bsk::get<36>(t183_);
            auto alt_middle = bsk::get<37>(t183_);
            auto alt_d_low = bsk::get<38>(t183_);
            auto alt_d_middle = bsk::get<39>(t183_);
            auto alt_leading = bsk::get<40>(t183_);
            auto alt_d_leading = bsk::get<41>(t183_);
            auto alt_first = bsk::get<42>(t183_);
            auto alt_d_first = bsk::get<43>(t183_);
            auto alt_second = bsk::get<44>(t183_);
            auto alt_d_second = bsk::get<45>(t183_);
            auto alt_determinant = bsk::get<46>(t183_);
            auto alt_d_determinant = bsk::get<47>(t183_);
            auto alt_high = bsk::get<48>(t183_);
            auto alt_d_high = bsk::get<49>(t183_);
            auto alt_radius = bsk::get<50>(t183_);
            auto alt_d_radius = bsk::get<51>(t183_);
            auto alt_cube = bsk::get<52>(t183_);
            auto alt_raw = bsk::get<53>(t183_);
            auto alt_d_raw = bsk::get<54>(t183_);
            auto alt_argument = bsk::get<55>(t183_);
            auto alt_inside_limit = bsk::get<56>(t183_);
            auto alt_angle = bsk::get<57>(t183_);
            auto alt_d_angle = bsk::get<58>(t183_);
            auto alt_centre = bsk::get<59>(t183_);
            auto alt_d_centre = bsk::get<60>(t183_);
            auto alt_trailing = bsk::get<61>(t183_);
            auto alt_d_trailing = bsk::get<62>(t183_);
            auto alt_guarded = bsk::get<63>(t183_);
            auto alt_d_guarded = bsk::get<64>(t183_);
            auto alt_q00 = bsk::get<65>(t183_);
            auto alt_d_q00 = bsk::get<66>(t183_);
            auto alt_q01 = bsk::get<67>(t183_);
            auto alt_d_q01 = bsk::get<68>(t183_);
            auto alt_q02 = bsk::get<69>(t183_);
            auto alt_d_q02 = bsk::get<70>(t183_);
            auto alt_q10 = bsk::get<71>(t183_);
            auto alt_d_q10 = bsk::get<72>(t183_);
            auto alt_q11 = bsk::get<73>(t183_);
            auto alt_d_q11 = bsk::get<74>(t183_);
            auto alt_q12 = bsk::get<75>(t183_);
            auto alt_d_q12 = bsk::get<76>(t183_);
            auto alt_q20 = bsk::get<77>(t183_);
            auto alt_d_q20 = bsk::get<78>(t183_);
            auto alt_q21 = bsk::get<79>(t183_);
            auto alt_d_q21 = bsk::get<80>(t183_);
            auto alt_q22 = bsk::get<81>(t183_);
            auto alt_d_q22 = bsk::get<82>(t183_);
            auto t184_ = _three_pool_assemble_jvp(alt_free, alt_d_free, alt_pool_b, alt_d_pool_b, alt_pool_c, alt_d_pool_c, alt_a00, alt_d_a00, alt_a01, alt_d_a01, alt_a02, alt_d_a02, alt_a10, alt_d_a10, alt_a11, alt_d_a11, alt_a20, alt_d_a20, alt_a22, alt_d_a22, alt_s00, alt_d_s00, alt_s11, alt_d_s11, alt_s22, alt_d_s22, alt_minors, alt_d_minors, alt_sum_flat, alt_sum_linear, alt_sum_square, alt_d_sum_flat, alt_d_sum_linear, alt_d_sum_square, alt_lift, alt_d_lift, alt_low, alt_middle, alt_d_low, alt_d_middle, alt_leading, alt_d_leading, alt_first, alt_d_first, alt_second, alt_d_second, alt_determinant, alt_d_determinant, alt_high, alt_d_high, alt_radius, alt_d_radius, alt_cube, alt_raw, alt_d_raw, alt_argument, alt_inside_limit, alt_angle, alt_d_angle, alt_centre, alt_d_centre, alt_trailing, alt_d_trailing, alt_guarded, alt_d_guarded, alt_q00, alt_d_q00, alt_q01, alt_d_q01, alt_q02, alt_d_q02, alt_q10, alt_d_q10, alt_q11, alt_d_q11, alt_q12, alt_d_q12, alt_q20, alt_d_q20, alt_q21, alt_d_q21, alt_q22, alt_d_q22, narrow);
            auto alt_def_00 = bsk::get<0>(t184_);
            auto alt_dif_00 = bsk::get<1>(t184_);
            auto alt_def_01 = bsk::get<2>(t184_);
            auto alt_dif_01 = bsk::get<3>(t184_);
            auto alt_def_02 = bsk::get<4>(t184_);
            auto alt_dif_02 = bsk::get<5>(t184_);
            auto alt_def_10 = bsk::get<6>(t184_);
            auto alt_dif_10 = bsk::get<7>(t184_);
            auto alt_def_11 = bsk::get<8>(t184_);
            auto alt_dif_11 = bsk::get<9>(t184_);
            auto alt_def_12 = bsk::get<10>(t184_);
            auto alt_dif_12 = bsk::get<11>(t184_);
            auto alt_def_20 = bsk::get<12>(t184_);
            auto alt_dif_20 = bsk::get<13>(t184_);
            auto alt_def_21 = bsk::get<14>(t184_);
            auto alt_dif_21 = bsk::get<15>(t184_);
            auto alt_def_22 = bsk::get<16>(t184_);
            auto alt_dif_22 = bsk::get<17>(t184_);
            auto t185_ = _three_pool_step_adjoint_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, row_dt, unit, one_att, att_span, bsk::ld((held + 24), active_atom, 0.0f), nil, bsk::ld((held + 25), active_atom, 0.0f), nil, bsk::ld((held + 26), active_atom, 0.0f), nil, bsk::ld((held + 27), active_atom, 0.0f), nil, bsk::ld((held + 28), active_atom, 0.0f), nil, bsk::ld((held + 29), active_atom, 0.0f), nil, bsk::ld((held + 30), active_atom, 0.0f), nil, bsk::ld((held + 31), active_atom, 0.0f), nil, bsk::ld((held + 32), active_atom, 0.0f), nil, bsk::ld((held + 33), active_atom, 0.0f), nil, bsk::ld((held + 34), active_atom, 0.0f), nil, bsk::ld((held + 35), active_atom, 0.0f), nil, alt_free, alt_d_free, alt_pool_b, alt_d_pool_b, alt_pool_c, alt_d_pool_c, alt_a00, alt_d_a00, alt_a01, alt_d_a01, alt_a02, alt_d_a02, alt_a10, alt_d_a10, alt_a11, alt_d_a11, alt_a20, alt_d_a20, alt_a22, alt_d_a22, alt_s00, alt_d_s00, alt_s11, alt_d_s11, alt_s22, alt_d_s22, alt_minors, alt_d_minors, alt_sum_flat, alt_sum_linear, alt_sum_square, alt_d_sum_flat, alt_d_sum_linear, alt_d_sum_square, alt_lift, alt_d_lift, alt_low, alt_middle, alt_d_low, alt_d_middle, alt_leading, alt_d_leading, alt_first, alt_d_first, alt_second, alt_d_second, alt_determinant, alt_d_determinant, alt_high, alt_d_high, alt_radius, alt_d_radius, alt_cube, alt_raw, alt_d_raw, alt_argument, alt_inside_limit, alt_angle, alt_d_angle, alt_centre, alt_d_centre, alt_trailing, alt_d_trailing, alt_guarded, alt_d_guarded, alt_q00, alt_d_q00, alt_q01, alt_d_q01, alt_q02, alt_d_q02, alt_q10, alt_d_q10, alt_q11, alt_d_q11, alt_q12, alt_d_q12, alt_q20, alt_d_q20, alt_q21, alt_d_q21, alt_q22, alt_d_q22, alt_def_00, alt_dif_00, alt_def_01, alt_dif_01, alt_def_02, alt_dif_02, alt_def_10, alt_dif_10, alt_def_11, alt_dif_11, alt_def_12, alt_dif_12, alt_def_20, alt_dif_20, alt_def_21, alt_dif_21, alt_def_22, alt_dif_22, narrow);
            auto _span_r1_v = bsk::get<0>(t185_);
            auto _span_r1b_v = bsk::get<1>(t185_);
            auto _span_r1c_v = bsk::get<2>(t185_);
            auto _span_exch_v = bsk::get<3>(t185_);
            auto _span_sexch_v = bsk::get<4>(t185_);
            auto _span_bound_v = bsk::get<5>(t185_);
            auto _span_semi_v = bsk::get<6>(t185_);
            auto _span_dt_v = bsk::get<7>(t185_);
            auto _span_att_v = bsk::get<8>(t185_);
            auto span_r1_t = bsk::get<9>(t185_);
            auto span_r1b_t = bsk::get<10>(t185_);
            auto span_r1c_t = bsk::get<11>(t185_);
            auto span_exch_t = bsk::get<12>(t185_);
            auto span_sexch_t = bsk::get<13>(t185_);
            auto span_bound_t = bsk::get<14>(t185_);
            auto span_semi_t = bsk::get<15>(t185_);
            auto _span_dt_t = bsk::get<16>(t185_);
            auto _span_att_t = bsk::get<17>(t185_);
            slope1_v = bsk::truediv(-1000.0f, (atom_t1 * atom_t1));
            slope1_t = bsk::truediv((2000.0f * d_t1), ((atom_t1 * atom_t1) * atom_t1));
            slope1b_v = bsk::truediv(-1000.0f, (atom_t1b * atom_t1b));
            slope1b_t = bsk::truediv((2000.0f * d_t1b), ((atom_t1b * atom_t1b) * atom_t1b));
            slope1c_v = bsk::truediv(-1000.0f, (held_semisolid * held_semisolid));
            slope1c_t = bsk::truediv((2000.0f * d_semisolid_t1), ((held_semisolid * held_semisolid) * held_semisolid));
            auto row_r1_t = (back_r1_t + span_r1_t);
            auto row_r1b_t = (back_r1b_t + span_r1b_t);
            auto row_r1c_t = (back_r1c_t + span_r1c_t);
            g_t1v = (g_t1v + (back_r1_v * slope1_v));
            g_t1t = (g_t1t + ((row_r1_t * slope1_v) + (back_r1_v * slope1_t)));
            g_t1bv = (g_t1bv + (back_r1b_v * slope1b_v));
            g_t1bt = (g_t1bt + ((row_r1b_t * slope1b_v) + (back_r1b_v * slope1b_t)));
            g_t1cv = (g_t1cv + (back_r1c_v * slope1c_v));
            g_t1ct = (g_t1ct + ((row_r1c_t * slope1c_v) + (back_r1c_v * slope1c_t)));
            g_exchv = (g_exchv + back_exch_v);
            g_excht = (g_excht + (back_exch_t + span_exch_t));
            g_sexchv = (g_sexchv + back_sexch_v);
            g_sexcht = (g_sexcht + (back_sexch_t + span_sexch_t));
            g_boundv = (g_boundv + back_bound_v);
            g_boundt = (g_boundt + (back_bound_t + span_bound_t));
            g_semiv = (g_semiv + back_semi_v);
            g_semit = (g_semit + (back_semi_t + span_semi_t));
        }
    }
    auto velocity_v = ((g_flowv * flow_scale) + ((g_washv * direction) * washout_scale));
    auto velocity_t = ((g_flowt * flow_scale) + ((g_washt * direction) * washout_scale));
    if (bsk::truth((pools > 0))) {
        // The fraction also sets where each pool starts, which the walk back
        // reaches last.
        g_boundv = (g_boundv + bsk::sum_x(bsk::where((state == 0), (bbvr - zbvr), 0.0f)));
        g_boundt = (g_boundt + bsk::sum_x(bsk::where((state == 0), (bbtr - zbtr), 0.0f)));
    }
    if (bsk::truth((pools == 3))) {
        g_semiv = (g_semiv + bsk::sum_x(bsk::where((state == 0), (cbvr - zbvr), 0.0f)));
        g_semit = (g_semit + bsk::sum_x(bsk::where((state == 0), (cbtr - zbtr), 0.0f)));
    }
    if (bsk::truth((pools == 1))) {
        base_row = (9 + (2 * (shim_rows - 1)));
        bsk::atomic_add(((grad_tissue_value + (base_row * atom_count)) + atom), g_boundv, active_atom);
        bsk::atomic_add(((grad_tissue_tangent + (base_row * atom_count)) + atom), g_boundt, active_atom);
        bsk::atomic_add(((grad_tissue_value + ((base_row + 1) * atom_count)) + atom), g_exchv, active_atom);
        bsk::atomic_add(((grad_tissue_tangent + ((base_row + 1) * atom_count)) + atom), g_excht, active_atom);
        bsk::atomic_add(((grad_tissue_value + ((base_row + 2) * atom_count)) + atom), g_t1bv, active_atom);
        bsk::atomic_add(((grad_tissue_tangent + ((base_row + 2) * atom_count)) + atom), g_t1bt, active_atom);
    }
    if (bsk::truth((pools == 3))) {
        auto semisolid_row = (9 + (2 * (shim_rows - 1)));
        auto stuck = bsk::make_tup(g_semiv, g_sexchv, g_t1cv);
        auto stuck_tangents = bsk::make_tup(g_semit, g_sexcht, g_t1ct);
        bsk::static_for<0, 3, 1>([&](auto offset_c) {
            constexpr std::int64_t offset = decltype(offset_c)::value;
            bsk::atomic_add(((grad_tissue_value + ((semisolid_row + offset) * atom_count)) + atom), bsk::get<offset>(stuck), active_atom);
            bsk::atomic_add(((grad_tissue_tangent + ((semisolid_row + offset) * atom_count)) + atom), bsk::get<offset>(stuck_tangents), active_atom);
        });
    }
    if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
        base_row = (12 + (2 * (shim_rows - 1)));
        auto rows = bsk::make_tup(g_boundv, g_exchv, g_t1bv, g_t2bv, g_shiftv);
        auto tangent_rows = bsk::make_tup(g_boundt, g_excht, g_t1bt, g_t2bt, g_shiftt);
        bsk::static_for<0, 5, 1>([&](auto offset_c) {
            constexpr std::int64_t offset = decltype(offset_c)::value;
            bsk::atomic_add(((grad_tissue_value + ((base_row + offset) * atom_count)) + atom), bsk::get<offset>(rows), active_atom);
            bsk::atomic_add(((grad_tissue_tangent + ((base_row + offset) * atom_count)) + atom), bsk::get<offset>(tangent_rows), active_atom);
        });
    }
    auto values = bsk::make_tup(g_t1v, g_t2v, g_m0v, g_b1v, g_b1pv, g_b0v, g_invv, g_diffv, velocity_v);
    auto tangents = bsk::make_tup(g_t1t, g_t2t, g_m0t, g_b1t, g_b1pt, g_b0t, g_invt, g_difft, velocity_t);
    bsk::static_for<0, 9, 1>([&](auto parameter_c) {
        constexpr std::int64_t parameter = decltype(parameter_c)::value;
        // The transmit pair went to its shim's row above when there is more
        // than one; the rest sit past whatever rows that pair took.
        if (bsk::truth((bsk::truth((!bsk::truth(shimmed))) || bsk::truth((bsk::truth((parameter != 3)) && bsk::truth((parameter != 4))))))) {
            auto plane = bsk::select(bsk::truth((parameter < 3)), parameter, (parameter + (2 * (shim_rows - 1))));
            bsk::atomic_add(((grad_tissue_value + (plane * atom_count)) + atom), bsk::get<parameter>(values), active_atom);
            bsk::atomic_add(((grad_tissue_tangent + (plane * atom_count)) + atom), bsk::get<parameter>(tangents), active_atom);
        }
    });
}

// Longitudinal and transverse diffusion damping for one interval.
//
// ``rate`` already carries the sequence's gradient geometry, so an interval's
// b-factor is that rate times its duration. Order zero has no longitudinal
// weight, which is what keeps the recovery term undamped.
template <class T0, class T1, class T2>
BSK_HD auto _damping(const T0& rate, const T1& dt, const T2& order) {
    auto b_factor = (rate * dt);
    auto squared = (order * order);
    return bsk::make_tup(bsk::exp(((-b_factor) * squared)), bsk::exp(((-b_factor) * ((squared + order) + 0.3333333333333333f))));
}

// How well the bound pool absorbs a pulse this far off its resonance.
//
// Cubic Hermite between the two knots bracketing the offset, taken in
// magnitude because the lineshape is even, and clamped at the far end. Each
// knot is two floats -- the value then its slope -- so the two a read needs
// are four contiguous ones.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _lineshape_at(const T0& lineshape, const T1& offset_hz, const T2& bins, const T3& step) {
    auto last = (bins - 1);
    auto scaled = bsk::minimum(bsk::truediv(bsk::abs(offset_hz), step), (last + 0.0f));
    auto lower = bsk::minimum(bsk::floor(scaled), (last - 1.0f));
    auto u = (scaled - lower);
    auto u2 = (u * u);
    auto u3 = (u2 * u);
    auto base = (bsk::cast<std::int64_t>(lower) * 2);
    auto near = bsk::ld((lineshape + base));
    auto near_slope = bsk::ld(((lineshape + base) + 1));
    auto far = bsk::ld(((lineshape + base) + 2));
    auto far_slope = bsk::ld(((lineshape + base) + 3));
    return (((((((2.0f * u3) - (3.0f * u2)) + 1.0f) * near) + ((((u3 - (2.0f * u2)) + u) * step) * near_slope)) + (((-2.0f * u3) + (3.0f * u2)) * far)) + (((u3 - u2) * step) * far_slope));
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

// One pool through a hard pulse named by its flip angle and phase.
//
// Pulled out of the kernel body so a second pool can take the same rotation:
// a chemical shift moves where a pool precesses, not what a pulse does to it.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
BSK_HD auto _rotate_flip_phase(const T0& cosine, const T1& sine, const T2& cos_phi, const T3& sin_phi, const T4& cos_2phi, const T5& sin_2phi, const T6& fp_r, const T7& fp_i, const T8& fm_r, const T9& fm_i, const T10& z_r, const T11& z_i) {
    auto cosine_half_sq = (0.5f * (1.0f + cosine));
    auto sine_half_sq = (0.5f * (1.0f - cosine));
    auto half_sine = (0.5f * sine);
    // Every sum of products is one fused multiply-add in a fixed order, so a
    // kernel compiled for fewer terms rounds the rotation as the full one does.
    auto minus_2phi_r = bsk::fma(cos_2phi, fm_r, (-(sin_2phi * fm_i)));
    auto minus_2phi_i = bsk::fma(sin_2phi, fm_r, (cos_2phi * fm_i));
    auto plus_2phi_r = bsk::fma(cos_2phi, fp_r, (sin_2phi * fp_i));
    auto plus_2phi_i = bsk::fma(cos_2phi, fp_i, (-(sin_2phi * fp_r)));
    auto z_turn_a = bsk::fma(sin_phi, z_r, (cos_phi * z_i));
    auto z_turn_b = bsk::fma(sin_phi, z_i, (-(cos_phi * z_r)));
    auto z_turn_c = bsk::fma(sin_phi, z_r, (-(cos_phi * z_i)));
    auto z_turn_d = bsk::fma(cos_phi, z_r, (sin_phi * z_i));
    auto rotated_pr = bsk::fma(sine, z_turn_a, bsk::fma(sine_half_sq, minus_2phi_r, (cosine_half_sq * fp_r)));
    auto rotated_pi = bsk::fma(sine, z_turn_b, bsk::fma(sine_half_sq, minus_2phi_i, (cosine_half_sq * fp_i)));
    auto rotated_mr = bsk::fma(sine, z_turn_c, bsk::fma(cosine_half_sq, fm_r, (sine_half_sq * plus_2phi_r)));
    auto rotated_mi = bsk::fma(sine, z_turn_d, bsk::fma(cosine_half_sq, fm_i, (sine_half_sq * plus_2phi_i)));
    auto plus_turn_r = bsk::fma(sin_phi, fp_r, (-(cos_phi * fp_i)));
    auto minus_turn_r = bsk::fma(sin_phi, fm_r, (cos_phi * fm_i));
    auto plus_turn_i = bsk::fma(cos_phi, fp_r, (sin_phi * fp_i));
    auto minus_turn_i = bsk::fma(cos_phi, fm_r, (-(sin_phi * fm_i)));
    auto rotated_zr = bsk::fma(cosine, z_r, bsk::fma((-half_sine), minus_turn_r, ((-half_sine) * plus_turn_r)));
    auto rotated_zi = bsk::fma(cosine, z_i, bsk::fma(half_sine, minus_turn_i, ((-half_sine) * plus_turn_i)));
    return bsk::make_tup(rotated_pr, rotated_pi, rotated_mr, rotated_mi, rotated_zr, rotated_zi);
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

// The sine and cosine of ``x`` from one reduction by a quarter turn.
//
// Cody and Waite's three-part quarter turn and the single-precision Cephes
// polynomials on the eighth turn either side of zero, to about an ulp where
// ``|x|`` is a flip angle; one reduction serves both where two library calls
// would each make their own.
template <class T0>
BSK_HD auto _sincos(const T0& x) {
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0> | 0, 2)> cosine{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0> | 0, 2)> r{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0> | 0, 2)> sine{};
    auto quarter = bsk::rint((x * 0.6366197723675814f));
    r = bsk::fma((-quarter), 1.5703125f, x);
    r = bsk::fma((-quarter), 0.0004837512969970703f, r);
    r = bsk::fma((-quarter), 7.549789954891882e-08f, r);
    auto r2 = (r * r);
    sine = bsk::fma(-0.00019515295891f, r2, 0.0083321608736f);
    sine = bsk::fma(sine, r2, -0.16666654611f);
    sine = bsk::fma((r * r2), sine, r);
    cosine = bsk::fma(2.443315711809948e-05f, r2, -0.001388731625493765f);
    cosine = bsk::fma(cosine, r2, 0.04166664568298827f);
    cosine = bsk::fma((r2 * r2), cosine, bsk::fma(-0.5f, r2, 1.0f));
    auto q = bsk::band(bsk::cast<std::int32_t>(quarter), 3);
    auto s = bsk::where((q == 0), sine, bsk::where((q == 1), cosine, bsk::where((q == 2), (-sine), (-cosine))));
    auto c = bsk::where((q == 0), cosine, bsk::where((q == 1), (-sine), bsk::where((q == 2), (-cosine), sine)));
    return bsk::make_tup(s, c);
}

// What each pool recovers over the interval, beside the operator itself.
//
// Returns the nine entries and the three recoveries, narrowed to float32
// once they are an operator.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11>
BSK_HD auto _three_pool_recovery(const T0& e00, const T1& e01, const T2& e02, const T3& e10, const T4& e11, const T5& e12, const T6& e20, const T7& e21, const T8& e22, const T9& free, const T10& pool_b, const T11& pool_c) {
    auto grow_free = (free - (((e00 * free) + (e01 * pool_b)) + (e02 * pool_c)));
    auto grow_pool_b = (pool_b - (((e10 * free) + (e11 * pool_b)) + (e12 * pool_c)));
    auto grow_bound = (pool_c - (((e20 * free) + (e21 * pool_b)) + (e22 * pool_c)));
    return bsk::make_tup(bsk::cast<float>(e00), bsk::cast<float>(e01), bsk::cast<float>(e02), bsk::cast<float>(e10), bsk::cast<float>(e11), bsk::cast<float>(e12), bsk::cast<float>(e20), bsk::cast<float>(e21), bsk::cast<float>(e22), bsk::cast<float>(grow_free), bsk::cast<float>(grow_pool_b), bsk::cast<float>(grow_bound));
}

// Read one interval's three-pool operator, and what each pool recovers.
//
// The stored row is undamped, so the washout the event carries is applied
// here and the three recoveries follow from the damped entries -- which is
// what makes one row serve every event of the same length whatever its
// washout.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
BSK_HD auto _three_pool_from_table(const T0& table, const T1& row, const T2& atom, const T3& voxel_count, const T4& mask, const T5& attenuation, const T6& free, const T7& pool_b, const T8& pool_c) {
    auto base = ((table + (row * (9 * voxel_count))) + atom);
    return _three_pool_recovery((attenuation * bsk::ld((base + (0 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (1 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (2 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (3 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (4 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (5 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (6 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (7 * voxel_count)), mask, 0.0f)), (attenuation * bsk::ld((base + (8 * voxel_count)), mask, 0.0f)), free, pool_b, pool_c);
}

// ``[a, b] exp``, from exponentials the caller has already taken.
//
// Near the coalescence ``sinh(d)/d`` is even in the gap, so the series is a
// polynomial in its square; the exponential of the midpoint is reached from
// the lower one by a series too, because over a gap this small it is one.
template <class T0, class T1, class T2, class T3>
BSK_HD auto _exp_difference(const T0& lower, const T1& upper, const T2& exp_lower, const T3& exp_upper) {
    auto half = (0.5f * (upper - lower));
    auto near = (bsk::abs(half) < 0.0001f);
    auto square = (half * half);
    // exp(mid) * sinh(half)/half, with both factors expanded about zero.
    auto series = ((exp_lower * ((1.0f + half) + (0.5f * square))) * (1.0f + bsk::truediv(square, 6.0f)));
    auto gap = bsk::where(near, 1.0f, (upper - lower));
    return bsk::where(near, series, bsk::truediv((exp_upper - exp_lower), gap));
}

// ``expm((K - diag(R1)) t)`` for free water beside both second pools.
//
// Free water is pool a, the chemically exchanging pool b and the semisolid
// pool c; each second pool exchanges with the free water and not with the
// other. Returns the nine entries and the three recoveries, narrowed to
// float32 once they are an operator.
//
// Two branches, by how far apart the eigenvalues are. Where they are close
// the exponential's own series is reduced modulo the characteristic
// polynomial, which forms no root at all. Where they are far apart the
// interpolating polynomial is taken in Newton form at the three roots, each
// of which is non-positive, so a long interval cannot overflow.
//
// ``narrow`` says the caller has bounded the spread below
// :data:`blochsim.sequence._parameters.NARROW_SPREAD` for every voxel and
// every interval it will pass, so
// only the series can be reached. The roots then cost nothing, and the series
// holds the answer to float32 without being carried in double --
// :func:`blochsim.sequence._parameters.narrow_three_pool` is what decides it.
template <class Work, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _three_pool_step_in_precision(const T0& r1_free, const T1& r1_pool_b, const T2& r1_bound, const T3& exchange_b, const T4& exchange_c, const T5& fraction_b, const T6& fraction_c, const T7& dt, const T8& attenuation, const T9& narrow) {
    using Ret = bsk::tup<bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>, bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)>>;
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e00{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e01{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e02{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e10{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e11{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e12{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e20{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e21{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> e22{};
    float factorial{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> square{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> sum_flat{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> sum_linear{};
    bsk::tile_t<Work, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> | 0, 3)> sum_square{};
    auto terms = bsk::select(bsk::truth(narrow), 24, 16);
    auto step = bsk::cast<Work>(dt);
    auto free = bsk::cast<Work>(((Work(1.0) - fraction_b) - fraction_c));
    auto pool_b = bsk::cast<Work>(fraction_b);
    auto pool_c = bsk::cast<Work>(fraction_c);
    auto kab = (bsk::cast<Work>(exchange_b) * pool_b);
    auto kba = (bsk::cast<Work>(exchange_b) * free);
    auto kac = (bsk::cast<Work>(exchange_c) * pool_c);
    auto kca = (bsk::cast<Work>(exchange_c) * free);
    auto a00 = ((((-kab) - kac) - bsk::cast<Work>(r1_free)) * step);
    auto a01 = (kba * step);
    auto a02 = (kca * step);
    auto a10 = (kab * step);
    auto a11 = (((-kba) - bsk::cast<Work>(r1_pool_b)) * step);
    auto a20 = (kac * step);
    auto a22 = (((-kca) - bsk::cast<Work>(r1_bound)) * step);
    auto third = bsk::truediv(((a00 + a11) + a22), Work(3.0));
    auto s00 = (a00 - third);
    auto s11 = (a11 - third);
    auto s22 = (a22 - third);
    // The two second pools do not exchange, so the generator keeps a pair of
    // structural zeros the products below are written around.
    auto minors = (((((s00 * s11) - (a01 * a10)) + (s00 * s22)) - (a02 * a20)) + (s11 * s22));
    auto determinant = ((((s00 * s11) * s22) - (a01 * (a10 * s22))) + (a02 * ((-s11) * a20)));
    // --- close together: the series reduced modulo x^3 + minors x - det ---
    flat = (Work(1.0) + (Work(0.0) * third));
    linear = (Work(0.0) * third);
    square = (Work(0.0) * third);
    sum_flat = flat;
    sum_linear = linear;
    sum_square = square;
    factorial = Work(1.0);
    #pragma unroll
    for (bsk::index_t order = 1; order < terms; order += 1) {
        auto next_flat = (square * determinant);
        auto next_linear = (flat - (square * minors));
        auto next_square = linear;
        flat = next_flat;
        linear = next_linear;
        square = next_square;
        factorial = (factorial * order);
        auto weight = bsk::truediv(Work(1.0), factorial);
        sum_flat = (sum_flat + (weight * flat));
        sum_linear = (sum_linear + (weight * linear));
        sum_square = (sum_square + (weight * square));
    }
    auto q00 = (((s00 * s00) + (a01 * a10)) + (a02 * a20));
    auto q01 = ((s00 * a01) + (a01 * s11));
    auto q02 = ((s00 * a02) + (a02 * s22));
    auto q10 = ((a10 * s00) + (s11 * a10));
    auto q11 = ((a10 * a01) + (s11 * s11));
    auto q12 = (a10 * a02);
    auto q20 = ((a20 * s00) + (s22 * a20));
    auto q21 = (a20 * a01);
    auto q22 = ((a20 * a02) + (s22 * s22));
    auto lift = bsk::exp(third);
    auto c00 = (lift * ((sum_flat + (sum_linear * s00)) + (sum_square * q00)));
    auto c01 = (lift * ((sum_linear * a01) + (sum_square * q01)));
    auto c02 = (lift * ((sum_linear * a02) + (sum_square * q02)));
    auto c10 = (lift * ((sum_linear * a10) + (sum_square * q10)));
    auto c11 = (lift * ((sum_flat + (sum_linear * s11)) + (sum_square * q11)));
    auto c12 = (lift * (sum_square * q12));
    auto c20 = (lift * ((sum_linear * a20) + (sum_square * q20)));
    auto c21 = (lift * (sum_square * q21));
    auto c22 = (lift * ((sum_flat + (sum_linear * s22)) + (sum_square * q22)));
    // --- far apart: the Newton form at the three roots ---
    auto damp = bsk::cast<Work>(attenuation);
    if (bsk::truth(narrow)) {
        e00 = (damp * c00);
        e01 = (damp * c01);
        e02 = (damp * c02);
        e10 = (damp * c10);
        e11 = (damp * c11);
        e12 = (damp * c12);
        e20 = (damp * c20);
        e21 = (damp * c21);
        e22 = (damp * c22);
        return bsk::convert<Ret>(_three_pool_recovery(e00, e01, e02, e10, e11, e12, e20, e21, e22, free, pool_b, pool_c));
    }
    auto radius = bsk::sqrt(bsk::maximum(((-minors) * Work(0.3333333333333333)), Work(1e-300)));
    auto argument = bsk::minimum(bsk::maximum(bsk::truediv((Work(0.5) * determinant), ((radius * radius) * radius)), Work(-0.9999999999999999)), Work(0.9999999999999999));
    auto angle = bsk::truediv(bsk::acos(argument), Work(3.0));
    auto root_a = (((Work(2.0) * radius) * bsk::cos(angle)) + third);
    auto root_b = (((Work(2.0) * radius) * bsk::cos((angle - Work(2.0943951023931957)))) + third);
    auto root_c = (((Work(2.0) * radius) * bsk::cos((angle - Work(4.188790204786391)))) + third);
    auto low = bsk::minimum(bsk::minimum(root_a, root_b), root_c);
    auto high = bsk::maximum(bsk::maximum(root_a, root_b), root_c);
    auto middle = bsk::maximum(bsk::minimum(root_a, root_b), bsk::minimum(bsk::maximum(root_a, root_b), root_c));
    // Three exponentials serve every divided difference between them.
    auto leading = bsk::exp(low);
    auto centre = bsk::exp(middle);
    auto trailing = bsk::exp(high);
    auto first = _exp_difference(low, middle, leading, centre);
    auto span = (high - low);
    auto second = bsk::truediv((_exp_difference(middle, high, centre, trailing) - first), bsk::where((span > Work(0.0)), span, Work(1.0)));
    auto m00 = (a00 - low);
    auto m11 = (a11 - low);
    auto m22 = (a22 - low);
    auto n00 = (a00 - middle);
    auto n11 = (a11 - middle);
    auto n22 = (a22 - middle);
    auto p00 = (((m00 * n00) + (a01 * a10)) + (a02 * a20));
    auto p01 = ((m00 * a01) + (a01 * n11));
    auto p02 = ((m00 * a02) + (a02 * n22));
    auto p10 = ((a10 * n00) + (m11 * a10));
    auto p11 = ((a10 * a01) + (m11 * n11));
    auto p12 = (a10 * a02);
    auto p20 = ((a20 * n00) + (m22 * a20));
    auto p21 = (a20 * a01);
    auto p22 = ((a20 * a02) + (m22 * n22));
    auto d00 = ((leading + (first * m00)) + (second * p00));
    auto d01 = ((first * a01) + (second * p01));
    auto d02 = ((first * a02) + (second * p02));
    auto d10 = ((first * a10) + (second * p10));
    auto d11 = ((leading + (first * m11)) + (second * p11));
    auto d12 = (second * p12);
    auto d20 = ((first * a20) + (second * p20));
    auto d21 = (second * p21);
    auto d22 = ((leading + (first * m22)) + (second * p22));
    // The shifted roots sum to zero, so the sum of their squares is -2 * minors
    // and none is larger than the root of that.
    auto close = ((Work(-2.0) * minors) < Work(1.0));
    e00 = (damp * bsk::where(close, c00, d00));
    e01 = (damp * bsk::where(close, c01, d01));
    e02 = (damp * bsk::where(close, c02, d02));
    e10 = (damp * bsk::where(close, c10, d10));
    e11 = (damp * bsk::where(close, c11, d11));
    e12 = (damp * bsk::where(close, c12, d12));
    e20 = (damp * bsk::where(close, c20, d20));
    e21 = (damp * bsk::where(close, c21, d21));
    e22 = (damp * bsk::where(close, c22, d22));
    return bsk::convert<Ret>(_three_pool_recovery(e00, e01, e02, e10, e11, e12, e20, e21, e22, free, pool_b, pool_c));
}

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _three_pool_step(const T0& r1_free, const T1& r1_pool_b, const T2& r1_bound, const T3& exchange_b, const T4& exchange_c, const T5& fraction_b, const T6& fraction_c, const T7& dt, const T8& attenuation, const T9& narrow) {
    using R = decltype(_three_pool_step_in_precision<double>(r1_free, r1_pool_b, r1_bound, exchange_b, exchange_c, fraction_b, fraction_c, dt, attenuation, narrow));
    if (bsk::truth(narrow)) {
        return bsk::convert<R>(_three_pool_step_in_precision<float>(r1_free, r1_pool_b, r1_bound, exchange_b, exchange_c, fraction_b, fraction_c, dt, attenuation, narrow));
    }
    return _three_pool_step_in_precision<double>(r1_free, r1_pool_b, r1_bound, exchange_b, exchange_c, fraction_b, fraction_c, dt, attenuation, narrow);
}

// The two-pool longitudinal operator over one interval, and its recovery.
//
// ``expm((K - diag(R1)) t)`` in the exact 2x2 closed form. Its discriminant
// is a square plus a product of two non-negative rates, so the root is real
// and the branch a general exponential would need does not exist here.
// ``sinh(d)/d`` is taken by series near the origin, where the root has no
// derivative of its own.
//
// The equilibrium each pool relaxes toward is its own fraction, so the
// recovery is ``(I - E1) (1 - f, f)`` and needs no solve. Returned as
// ``(e11, e12, e21, e22, recovery_free, recovery_bound)``.
template <class T0, class T1, class T2, class T3, class T4, class T5>
BSK_HD auto _two_pool_step(const T0& r1_free, const T1& r1_bound, const T2& exchange, const T3& bound, const T4& dt, const T5& attenuation) {
    auto free = (1.0f - bound);
    auto kab = (exchange * bound);
    auto kba = (exchange * free);
    auto l11 = (((-kab) - r1_free) * dt);
    auto l12 = (kba * dt);
    auto l21 = (kab * dt);
    auto l22 = (((-kba) - r1_bound) * dt);
    auto half_trace = (0.5f * (l11 + l22));
    auto half_gap = (0.5f * (l11 - l22));
    auto square = ((half_gap * half_gap) + (l12 * l21));
    // tau +/- d are the eigenvalues, both non-positive for a decaying system,
    // so their exponentials are bounded by one. Formed that way rather than as
    // e^tau cosh(d), which over a long interval is an underflow times an
    // overflow.
    auto root = bsk::sqrt(bsk::maximum(square, 0.0f));
    auto upper = bsk::exp((half_trace + root));
    auto lower = bsk::exp((half_trace - root));
    auto cosine = (0.5f * (upper + lower));
    auto turning = (square > 1e-12f);
    auto guarded = bsk::where(turning, root, 1.0f);
    auto scale = bsk::where(turning, bsk::truediv((0.5f * (upper - lower)), guarded), (bsk::exp(half_trace) * ((1.0f + bsk::truediv(square, 6.0f)) + bsk::truediv((square * square), 120.0f))));
    auto e11 = (attenuation * (cosine + (scale * half_gap)));
    auto e12 = ((attenuation * scale) * l12);
    auto e21 = ((attenuation * scale) * l21);
    auto e22 = (attenuation * (cosine - (scale * half_gap)));
    return bsk::make_tup(e11, e12, e21, e22, (free - ((e11 * free) + (e12 * bound))), (bound - ((e21 * free) + (e22 * bound))));
}

// The transverse operator of two chemically exchanging pools.
//
// ``expm((K - diag(R2) - 2 pi i diag(df)) t)``, in the closed form the
// longitudinal pair uses -- the numbers have become complex, the algebra has
// not. Returned as the four entries, each a pair of floats.
//
// A semisolid pool holds a share of the voxel without carrying any transverse
// magnetization, so it is absent from this 2x2 and present in ``free`` -- how
// much free water the exchange sees.
//
// There is no recovery term: transverse magnetization relaxes toward zero.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
BSK_HD auto _two_pool_transverse_step(const T0& r2_free, const T1& r2_bound, const T2& exchange, const T3& bound, const T4& free, const T5& shift_hz, const T6& dt, const T7& attenuation) {
    auto kab = (exchange * bound);
    auto kba = (exchange * free);
    auto l11 = (((-kab) - r2_free) * dt);
    auto l12 = (kba * dt);
    auto l21 = (kab * dt);
    auto l22 = (((-kba) - r2_bound) * dt);
    // Only pool b's offset appears: pool a sits at whatever off-resonance the
    // free precession already carries the whole voxel through.
    auto l22_imag = ((-6.283185307179586f * shift_hz) * dt);
    auto trace_real = (0.5f * (l11 + l22));
    auto trace_imag = (0.5f * l22_imag);
    auto gap_real = (0.5f * (l11 - l22));
    auto gap_imag = (-0.5f * l22_imag);
    auto square_real = (((gap_real * gap_real) - (gap_imag * gap_imag)) + (l12 * l21));
    auto square_imag = ((2.0f * gap_real) * gap_imag);
    auto t0_ = _complex_sqrt(square_real, square_imag);
    auto root_real = bsk::get<0>(t0_);
    auto root_imag = bsk::get<1>(t0_);
    auto t1_ = _complex_exp((trace_real + root_real), (trace_imag + root_imag));
    auto upper_real = bsk::get<0>(t1_);
    auto upper_imag = bsk::get<1>(t1_);
    auto t2_ = _complex_exp((trace_real - root_real), (trace_imag - root_imag));
    auto lower_real = bsk::get<0>(t2_);
    auto lower_imag = bsk::get<1>(t2_);
    auto cos_real = (0.5f * (upper_real + lower_real));
    auto cos_imag = (0.5f * (upper_imag + lower_imag));
    // ``sinh(d)/d`` by series near the origin, where the root has no
    // derivative of its own.
    auto turning = (((square_real * square_real) + (square_imag * square_imag)) > 1e-24f);
    auto half_real = (0.5f * (upper_real - lower_real));
    auto half_imag = (0.5f * (upper_imag - lower_imag));
    auto guard = bsk::where(turning, ((root_real * root_real) + (root_imag * root_imag)), 1.0f);
    auto divided_real = bsk::truediv(((half_real * root_real) + (half_imag * root_imag)), guard);
    auto divided_imag = bsk::truediv(((half_imag * root_real) - (half_real * root_imag)), guard);
    auto t3_ = _complex_exp(trace_real, trace_imag);
    auto plain_real = bsk::get<0>(t3_);
    auto plain_imag = bsk::get<1>(t3_);
    auto square2_real = ((square_real * square_real) - (square_imag * square_imag));
    auto square2_imag = ((2.0f * square_real) * square_imag);
    auto poly_real = ((1.0f + bsk::truediv(square_real, 6.0f)) + bsk::truediv(square2_real, 120.0f));
    auto poly_imag = (bsk::truediv(square_imag, 6.0f) + bsk::truediv(square2_imag, 120.0f));
    auto series_real = ((plain_real * poly_real) - (plain_imag * poly_imag));
    auto series_imag = ((plain_real * poly_imag) + (plain_imag * poly_real));
    auto scale_real = bsk::where(turning, divided_real, series_real);
    auto scale_imag = bsk::where(turning, divided_imag, series_imag);
    auto off_real = ((scale_real * gap_real) - (scale_imag * gap_imag));
    auto off_imag = ((scale_real * gap_imag) + (scale_imag * gap_real));
    return bsk::make_tup((attenuation * (cos_real + off_real)), (attenuation * (cos_imag + off_imag)), ((attenuation * scale_real) * l12), ((attenuation * scale_imag) * l12), ((attenuation * scale_real) * l21), ((attenuation * scale_imag) * l21), (attenuation * (cos_real - off_real)), (attenuation * (cos_imag - off_imag)));
}

// The fraction of a voxel's spins that stay put over one interval.
//
// Inflowing spins are taken to be fully relaxed and unexcited, which makes
// washout an affine map of the shape longitudinal recovery already has:
//
//     wout * (Z * e1 + (1 - e1)) + win  ==  Z * (e1 * wout) + (1 - e1 * wout)
//
// so scaling both relaxation factors by it carries the whole term. Clamped at
// one, past which the interval has replaced the voxel outright.
template <class T0, class T1>
BSK_HD auto _washout(const T0& rate, const T1& dt) {
    return (1.0f - bsk::minimum((rate * dt), 1.0f));
}

BSK_HD void _epg_kernel(float* t1, float* t2, float* m0, float* b1, float* b1_phase, float* b0, float* inversion_efficiency, float* diffusion, float* velocity, float* bound_fraction, float* bound_exchange, float* t1_bound, float* pool_b_fraction, float* pool_b_exchange, float* t1_pool_b, float* t2_pool_b, float* pool_b_shift, float* duration, std::int32_t* kind, float* flip, float* phase, float* phase_cos, float* phase_sin, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* saturation, float* rf_frequency, float* profile, std::int32_t* profile_index, float* lineshape, float* pairs, std::int32_t* pair_index, std::int32_t* duration_row, float* pool_table, float* output_real, float* output_imag, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, float flow_scale, float washout_scale, float profile_step, float lineshape_step, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shim_rows, bsk::index_t shimmed, bsk::index_t locations, bsk::index_t profiled, bsk::index_t profile_bins, bsk::index_t dynamic, bsk::index_t broadened, bsk::index_t lineshape_bins, bsk::index_t pools, bsk::index_t narrow, bsk::index_t tabulated, bsk::index_t off_axis, bsk::index_t moving, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 2> atom_b0{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_b1_phase{};
    bsk::V<float, 2> atom_bound{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_exchange{};
    bsk::V<float, 2> atom_flow{};
    bsk::V<float, 2> atom_inversion{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 2> atom_r1_bound{};
    bsk::V<float, 2> atom_r1_semisolid{};
    bsk::V<float, 2> atom_r2_bound{};
    bsk::V<float, 2> atom_semisolid{};
    bsk::V<float, 2> atom_semisolid_exchange{};
    bsk::V<float, 2> atom_shift{};
    bsk::V<float, 2> atom_washout{};
    bsk::V<float, 2> b1_cos{};
    bsk::V<float, 2> b1_sin{};
    bsk::V<float, 3> b_rot_mi{};
    bsk::V<float, 3> b_rot_mr{};
    bsk::V<float, 3> b_rot_pi{};
    bsk::V<float, 3> b_rot_pr{};
    bsk::V<float, 3> b_rot_zi{};
    bsk::V<float, 3> b_rot_zr{};
    bsk::V<float, 3> bminus_imag{};
    bsk::V<float, 3> bminus_real{};
    bsk::V<float, 3> bound_imag{};
    bsk::V<float, 3> bound_real{};
    bsk::V<float, 3> bplus_imag{};
    bsk::V<float, 3> bplus_real{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 3> e1{};
    bsk::V<float, 3> e2{};
    bsk::V<float, 3> fminus_imag{};
    bsk::V<float, 3> fminus_real{};
    bsk::V<float, 3> fplus_imag{};
    bsk::V<float, 3> fplus_real{};
    bsk::V<float, 3> free_imag{};
    bsk::V<float, 3> free_real{};
    bsk::V<float, 2> grow_free{};
    bsk::V<float, 2> grow_pool_b{};
    bsk::V<float, 2> grow_semisolid{};
    bsk::V<float, 3> held_imag{};
    bsk::V<float, 3> held_real{};
    bsk::V<float, 3> longitudinal_imag{};
    bsk::V<float, 3> longitudinal_real{};
    bsk::V<float, 3> off_cos{};
    bsk::V<float, 3> off_sin{};
    bsk::V<float, 3> old_real{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> pair{};
    bsk::V<float, 3> read_imag{};
    bsk::V<float, 3> read_real{};
    bsk::V<float, 3> rotated_mi{};
    bsk::V<float, 3> rotated_mr{};
    bsk::V<float, 3> rotated_pi{};
    bsk::V<float, 3> rotated_pr{};
    bsk::V<float, 3> rotated_zi{};
    bsk::V<float, 3> rotated_zr{};
    bsk::V<float, 3> semisolid_imag{};
    bsk::V<float, 3> semisolid_real{};
    bsk::V<float, 3> shaped_mi{};
    bsk::V<float, 3> shaped_mr{};
    bsk::V<float, 3> shaped_pi{};
    bsk::V<float, 3> shaped_pr{};
    bsk::V<float, 3> shaped_zi{};
    bsk::V<float, 3> shaped_zr{};
    bsk::V<float, 2> spun_bi{};
    bsk::V<float, 2> spun_br{};
    bsk::V<float, 2> t11{};
    bsk::V<float, 2> t12{};
    bsk::V<float, 2> t13{};
    bsk::V<float, 2> t21{};
    bsk::V<float, 2> t22{};
    bsk::V<float, 2> t23{};
    bsk::V<float, 2> t31{};
    bsk::V<float, 2> t32{};
    bsk::V<float, 2> t33{};
    bsk::V<float, 3> turn_cos{};
    bsk::V<float, 3> turn_sin{};
    bsk::V<float, 3> turn_t{};
    bsk::V<float, 3> turn_z{};
    bsk::V<float, 2> wout{};
    auto problem = ((bsk::program_id(0) * problems) + bsk::arange_y());
    auto state = bsk::arange_x();
    auto active_atom = (problem < (train_count * atom_count));
    // A partial block carries lanes with no problem behind them, and they must
    // take no part in a reduction or a store.
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    // Voxels are spread over the slice voxel-major, so a voxel's place along
    // the slice is its index modulo the profile's width. One pulse shape holds
    // that many consecutive rows, and the event says which shape it drives.
    auto location = bsk::mod(atom, locations);
    auto empty = bsk::full<float, 3>(0);
    fplus_real = empty;
    fplus_imag = empty;
    fminus_real = empty;
    fminus_imag = empty;
    // A second pool holds its own share of the equilibrium. The semisolid one
    // carries longitudinal states alone -- nothing dephases it, so it reaches
    // the higher orders only through exchange with the free pool's -- while the
    // chemically exchanging one carries a transverse pair of its own.
    //
    // ``bound`` is whichever second pool the longitudinal step pairs the free
    // water with -- the semisolid one when it is the only one, the exchanging
    // one otherwise -- and ``semisolid`` is the third, which only a three-pool
    // run carries.
    atom_bound = 0.0f;
    atom_exchange = 0.0f;
    atom_r1_bound = 0.0f;
    atom_r2_bound = 0.0f;
    atom_shift = 0.0f;
    atom_semisolid = 0.0f;
    atom_semisolid_exchange = 0.0f;
    atom_r1_semisolid = 0.0f;
    if (bsk::truth((pools == 1))) {
        atom_bound = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((bound_exchange + scalar_atom), active_atom, 0.0f);
        atom_r1_bound = bsk::truediv(1000.0f, bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f));
    }
    if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
        atom_bound = bsk::ld((pool_b_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((pool_b_exchange + scalar_atom), active_atom, 0.0f);
        atom_r1_bound = bsk::truediv(1000.0f, bsk::ld((t1_pool_b + scalar_atom), active_atom, 1.0f));
        atom_r2_bound = bsk::truediv(1000.0f, bsk::ld((t2_pool_b + scalar_atom), active_atom, 1.0f));
        atom_shift = bsk::ld((pool_b_shift + scalar_atom), active_atom, 0.0f);
    }
    if (bsk::truth((pools == 3))) {
        atom_semisolid = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_semisolid_exchange = bsk::ld((bound_exchange + scalar_atom), active_atom, 0.0f);
        atom_r1_semisolid = bsk::truediv(1000.0f, bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f));
    }
    // A semisolid pool holds a share of the voxel without carrying any
    // transverse magnetization, so the 2x2 below is blind to it and the
    // exchange inside that 2x2 is not.
    auto atom_free = ((1.0f - atom_bound) - atom_semisolid);
    longitudinal_real = (empty + bsk::where((state == 0), atom_free, 0.0f));
    longitudinal_imag = empty;
    bound_real = (empty + bsk::where((state == 0), (atom_bound + 0.0f), 0.0f));
    bound_imag = empty;
    semisolid_real = (empty + bsk::where((state == 0), (atom_semisolid + 0.0f), 0.0f));
    semisolid_imag = empty;
    bplus_real = empty;
    bplus_imag = empty;
    bminus_real = empty;
    bminus_imag = empty;
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_b1_phase = 0.0f;
    atom_b0 = 0.0f;
    if (bsk::truth(off_axis)) {
        atom_b1_phase = bsk::ld((b1_phase + scalar_atom), active_atom, 0.0f);
        atom_b0 = bsk::ld((b0 + scalar_atom), active_atom, 0.0f);
    }
    b1_cos = bsk::cos(atom_b1_phase);
    b1_sin = bsk::sin(atom_b1_phase);
    atom_inversion = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inversion = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    atom_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
    }
    atom_flow = 0.0f;
    atom_washout = 0.0f;
    if (bsk::truth(moving)) {
        auto atom_velocity = bsk::ld((velocity + scalar_atom), active_atom, 0.0f);
        atom_flow = (atom_velocity * flow_scale);
        atom_washout = (bsk::abs(atom_velocity) * washout_scale);
    }
    auto order = bsk::cast<float>(state);
    auto event_base = (train * event_count);
    for (bsk::index_t event = 0; event < event_count; event += 1) {
        auto dt = _event_value(duration, event_base, event, active_atom, single_train);
        wout = 1.0f;
        if (bsk::truth(moving)) {
            wout = _washout(atom_washout, dt);
        }
        e1 = (bsk::exp(((-bsk::truediv(1000.0f, atom_t1)) * dt)) * wout);
        e2 = (bsk::exp(((-bsk::truediv(1000.0f, atom_t2)) * dt)) * wout);
        damp_z = 1.0f;
        damp_t = 1.0f;
        if (bsk::truth(diffusing)) {
            auto t0_ = _damping(atom_damping, dt, order);
            damp_z = bsk::get<0>(t0_);
            damp_t = bsk::get<1>(t0_);
        }
        turn_z = 0.0f;
        turn_t = 0.0f;
        if (bsk::truth(moving)) {
            auto t1_ = _flow(atom_flow, dt, order);
            turn_z = bsk::get<0>(t1_);
            turn_t = bsk::get<1>(t1_);
        }
        auto recovery = (1.0f - e1);
        e1 = (e1 * damp_z);
        e2 = (e2 * damp_t);
        off_cos = 1.0f;
        off_sin = 0.0f;
        if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
            // Flow winds the transverse states through the same rotation
            // off-resonance does, so the two phases add before either is taken.
            auto off_phase = (((-6.283185307179586f * atom_b0) * dt) + turn_t);
            off_cos = bsk::cos(off_phase);
            off_sin = bsk::sin(off_phase);
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // Both pools take the same off-resonance and the same per-order
            // damping; what separates them is the chemical shift, which the
            // exchange operator already carries.
            auto t2_ = _two_pool_transverse_step(bsk::truediv(1000.0f, atom_t2), atom_r2_bound, atom_exchange, atom_bound, atom_free, atom_shift, dt, wout);
            auto x11r = bsk::get<0>(t2_);
            auto x11i = bsk::get<1>(t2_);
            auto x12r = bsk::get<2>(t2_);
            auto x12i = bsk::get<3>(t2_);
            auto x21r = bsk::get<4>(t2_);
            auto x21i = bsk::get<5>(t2_);
            auto x22r = bsk::get<6>(t2_);
            auto x22i = bsk::get<7>(t2_);
            auto mixed_pr = ((((x11r * fplus_real) - (x11i * fplus_imag)) + (x12r * bplus_real)) - (x12i * bplus_imag));
            auto mixed_pi = ((((x11r * fplus_imag) + (x11i * fplus_real)) + (x12r * bplus_imag)) + (x12i * bplus_real));
            auto mixed_br = ((((x21r * fplus_real) - (x21i * fplus_imag)) + (x22r * bplus_real)) - (x22i * bplus_imag));
            auto mixed_bi = ((((x21r * fplus_imag) + (x21i * fplus_real)) + (x22r * bplus_imag)) + (x22i * bplus_real));
            // ``F-`` follows the conjugate of the operator entry by entry, not
            // its transpose: it is the conjugate state, and the map it takes is
            // the conjugate map.
            auto mixed_mr = ((((x11r * fminus_real) + (x11i * fminus_imag)) + (x12r * bminus_real)) + (x12i * bminus_imag));
            auto mixed_mi = ((((x11r * fminus_imag) - (x11i * fminus_real)) + (x12r * bminus_imag)) - (x12i * bminus_real));
            auto mixed_nr = ((((x21r * fminus_real) + (x21i * fminus_imag)) + (x22r * bminus_real)) + (x22i * bminus_imag));
            auto mixed_ni = ((((x21r * fminus_imag) - (x21i * fminus_real)) + (x22r * bminus_imag)) - (x22i * bminus_real));
            fplus_real = (damp_t * ((mixed_pr * off_cos) - (mixed_pi * off_sin)));
            fplus_imag = (damp_t * ((mixed_pr * off_sin) + (mixed_pi * off_cos)));
            bplus_real = (damp_t * ((mixed_br * off_cos) - (mixed_bi * off_sin)));
            bplus_imag = (damp_t * ((mixed_br * off_sin) + (mixed_bi * off_cos)));
            fminus_real = (damp_t * ((mixed_mr * off_cos) + (mixed_mi * off_sin)));
            fminus_imag = (damp_t * (((-mixed_mr) * off_sin) + (mixed_mi * off_cos)));
            bminus_real = (damp_t * ((mixed_nr * off_cos) + (mixed_ni * off_sin)));
            bminus_imag = (damp_t * (((-mixed_nr) * off_sin) + (mixed_ni * off_cos)));
        } else {
            old_real = fplus_real;
            fplus_real = (e2 * ((old_real * off_cos) - (fplus_imag * off_sin)));
            fplus_imag = (e2 * ((old_real * off_sin) + (fplus_imag * off_cos)));
            old_real = fminus_real;
            fminus_real = (e2 * ((old_real * off_cos) + (fminus_imag * off_sin)));
            fminus_imag = (e2 * (((-old_real) * off_sin) + (fminus_imag * off_cos)));
        }
        // The longitudinal states carry a phase of their own, which nothing
        // else in the state machine gives them.
        turn_cos = 1.0f;
        turn_sin = 0.0f;
        if (bsk::truth(moving)) {
            turn_cos = bsk::cos(turn_z);
            turn_sin = bsk::sin(turn_z);
        }
        if (bsk::truth((pools == 3))) {
            // Three pools mix through a 3x3 formed in double; every pool takes
            // the same per-order damping and flow phase, their order-n states
            // describing one dephasing configuration.
            if (bsk::truth(tabulated)) {
                auto t3_ = _three_pool_from_table(pool_table, bsk::ld(((duration_row + event_base) + event), active_atom, 0), atom, atom_count, active_atom, wout, atom_free, atom_bound, atom_semisolid);
                t11 = bsk::get<0>(t3_);
                t12 = bsk::get<1>(t3_);
                t13 = bsk::get<2>(t3_);
                t21 = bsk::get<3>(t3_);
                t22 = bsk::get<4>(t3_);
                t23 = bsk::get<5>(t3_);
                t31 = bsk::get<6>(t3_);
                t32 = bsk::get<7>(t3_);
                t33 = bsk::get<8>(t3_);
                grow_free = bsk::get<9>(t3_);
                grow_pool_b = bsk::get<10>(t3_);
                grow_semisolid = bsk::get<11>(t3_);
            } else {
                auto t4_ = _three_pool_step(bsk::truediv(1000.0f, atom_t1), atom_r1_bound, atom_r1_semisolid, atom_exchange, atom_semisolid_exchange, atom_bound, atom_semisolid, dt, wout, narrow);
                t11 = bsk::get<0>(t4_);
                t12 = bsk::get<1>(t4_);
                t13 = bsk::get<2>(t4_);
                t21 = bsk::get<3>(t4_);
                t22 = bsk::get<4>(t4_);
                t23 = bsk::get<5>(t4_);
                t31 = bsk::get<6>(t4_);
                t32 = bsk::get<7>(t4_);
                t33 = bsk::get<8>(t4_);
                grow_free = bsk::get<9>(t4_);
                grow_pool_b = bsk::get<10>(t4_);
                grow_semisolid = bsk::get<11>(t4_);
            }
            free_real = (((t11 * longitudinal_real) + (t12 * bound_real)) + (t13 * semisolid_real));
            free_imag = (((t11 * longitudinal_imag) + (t12 * bound_imag)) + (t13 * semisolid_imag));
            held_real = (((t21 * longitudinal_real) + (t22 * bound_real)) + (t23 * semisolid_real));
            held_imag = (((t21 * longitudinal_imag) + (t22 * bound_imag)) + (t23 * semisolid_imag));
            auto stuck_real = (((t31 * longitudinal_real) + (t32 * bound_real)) + (t33 * semisolid_real));
            auto stuck_imag = (((t31 * longitudinal_imag) + (t32 * bound_imag)) + (t33 * semisolid_imag));
            longitudinal_real = (damp_z * ((free_real * turn_cos) - (free_imag * turn_sin)));
            longitudinal_imag = (damp_z * ((free_real * turn_sin) + (free_imag * turn_cos)));
            bound_real = (damp_z * ((held_real * turn_cos) - (held_imag * turn_sin)));
            bound_imag = (damp_z * ((held_real * turn_sin) + (held_imag * turn_cos)));
            semisolid_real = (damp_z * ((stuck_real * turn_cos) - (stuck_imag * turn_sin)));
            semisolid_imag = (damp_z * ((stuck_real * turn_sin) + (stuck_imag * turn_cos)));
            longitudinal_real = (longitudinal_real + bsk::where((state == 0), grow_free, 0.0f));
            bound_real = (bound_real + bsk::where((state == 0), grow_pool_b, 0.0f));
            semisolid_real = (semisolid_real + bsk::where((state == 0), grow_semisolid, 0.0f));
        } else if (bsk::truth((pools > 0))) {
            // The exchange operator is a property of the interval, not of a
            // dephasing order, so it is formed once and the per-order damping
            // multiplies it. Both pools take that damping and the flow phase:
            // their order-n states describe one dephasing configuration, and a
            // second pool has no diffusion coefficient of its own to damp by.
            auto t5_ = _two_pool_step(bsk::truediv(1000.0f, atom_t1), atom_r1_bound, atom_exchange, atom_bound, dt, wout);
            auto e11 = bsk::get<0>(t5_);
            auto e12 = bsk::get<1>(t5_);
            auto e21 = bsk::get<2>(t5_);
            auto e22 = bsk::get<3>(t5_);
            grow_free = bsk::get<4>(t5_);
            auto grow_bound = bsk::get<5>(t5_);
            free_real = ((e11 * longitudinal_real) + (e12 * bound_real));
            free_imag = ((e11 * longitudinal_imag) + (e12 * bound_imag));
            held_real = ((e21 * longitudinal_real) + (e22 * bound_real));
            held_imag = ((e21 * longitudinal_imag) + (e22 * bound_imag));
            longitudinal_real = (damp_z * ((free_real * turn_cos) - (free_imag * turn_sin)));
            longitudinal_imag = (damp_z * ((free_real * turn_sin) + (free_imag * turn_cos)));
            bound_real = (damp_z * ((held_real * turn_cos) - (held_imag * turn_sin)));
            bound_imag = (damp_z * ((held_real * turn_sin) + (held_imag * turn_cos)));
            longitudinal_real = (longitudinal_real + bsk::where((state == 0), grow_free, 0.0f));
            bound_real = (bound_real + bsk::where((state == 0), grow_bound, 0.0f));
        } else {
            old_real = longitudinal_real;
            longitudinal_real = (e1 * ((old_real * turn_cos) - (longitudinal_imag * turn_sin)));
            longitudinal_imag = (e1 * ((old_real * turn_sin) + (longitudinal_imag * turn_cos)));
            longitudinal_real = (longitudinal_real + bsk::where((state == 0), recovery, 0.0f));
        }
        // Every program reads the same event, so a branch on what it does is
        // taken by all of them alike: an event pays only for what it does.
        auto event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        if (bsk::truth((bsk::band(event_action, 1) != 0))) {
            auto t6_ = _shift(fplus_real, fplus_imag, fminus_real, fminus_imag, state, state_mask, state_count);
            fplus_real = bsk::get<0>(t6_);
            fplus_imag = bsk::get<1>(t6_);
            fminus_real = bsk::get<2>(t6_);
            fminus_imag = bsk::get<3>(t6_);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t7_ = _shift(bplus_real, bplus_imag, bminus_real, bminus_imag, state, state_mask, state_count);
                bplus_real = bsk::get<0>(t7_);
                bplus_imag = bsk::get<1>(t7_);
                bminus_real = bsk::get<2>(t7_);
                bminus_imag = bsk::get<3>(t7_);
            }
        }
        auto event_kind = bsk::ld((kind + event));
        auto is_rf = (event_kind == 1);
        auto is_inversion = (bsk::band(event_action, 4) != 0);
        auto invert = bsk::band(is_rf, is_inversion);
        longitudinal_real = bsk::where(invert, ((-atom_inversion) * longitudinal_real), longitudinal_real);
        longitudinal_imag = bsk::where(invert, ((-atom_inversion) * longitudinal_imag), longitudinal_imag);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // A chemically exchanging pool is free water and turns over like
            // any other; a semisolid one is saturated instead, which its own
            // saturation term already carries.
            bound_real = bsk::where(invert, ((-atom_inversion) * bound_real), bound_real);
            bound_imag = bsk::where(invert, ((-atom_inversion) * bound_imag), bound_imag);
        }
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse a row of its own.
        if (bsk::truth(shimmed)) {
            auto row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            atom_b1 = 1.0f;
            if (bsk::truth(transmit)) {
                atom_b1 = bsk::ld(((b1 + row) + atom), active_atom, 1.0f);
            }
            if (bsk::truth(off_axis)) {
                atom_b1_phase = bsk::ld(((b1_phase + row) + atom), active_atom, 0.0f);
                b1_cos = bsk::cos(atom_b1_phase);
                b1_sin = bsk::sin(atom_b1_phase);
            }
        }
        if (bsk::truth(bsk::band((event_kind == 1), (bsk::band(event_action, 4) == 0)))) {
            auto alpha = (_event_value(flip, event_base, event, active_atom, single_train) * atom_b1);
            // The pulse's phase, read off the cosine and sine the launch took
            // of it, turned by the transmit field's own.
            auto cos_event = _event_value(phase_cos, event_base, event, active_atom, single_train);
            auto sin_event = _event_value(phase_sin, event_base, event, active_atom, single_train);
            auto cos_phi = bsk::fma(cos_event, b1_cos, (-(sin_event * b1_sin)));
            auto sin_phi = bsk::fma(sin_event, b1_cos, (cos_event * b1_sin));
            if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                // Either pair is built at zero RF phase, which turns the rotation
                // axis and so reaches ``b`` alone.
                if (bsk::truth(dynamic)) {
                    // Already integrated at this pulse's own flip, so the flip is
                    // inside the pair rather than read against it.
                    pair = _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, active_atom);
                } else {
                    pair = _profile_pair(profile, _table_row(profile_index, event, location, locations), alpha, profile_bins, profile_step);
                }
                auto turn_r = cos_phi;
                auto turn_i = (-sin_phi);
                spun_br = ((bsk::get<2>(pair) * turn_r) - (bsk::get<3>(pair) * turn_i));
                spun_bi = ((bsk::get<2>(pair) * turn_i) + (bsk::get<3>(pair) * turn_r));
                auto t8_ = _rotate_spinor(bsk::get<0>(pair), bsk::get<1>(pair), spun_br, spun_bi, fplus_real, fplus_imag, fminus_real, fminus_imag, longitudinal_real, longitudinal_imag);
                shaped_pr = bsk::get<0>(t8_);
                shaped_pi = bsk::get<1>(t8_);
                shaped_mr = bsk::get<2>(t8_);
                shaped_mi = bsk::get<3>(t8_);
                shaped_zr = bsk::get<4>(t8_);
                shaped_zi = bsk::get<5>(t8_);
            }
            auto t9_ = _sincos(alpha);
            auto sine = bsk::get<0>(t9_);
            auto cosine = bsk::get<1>(t9_);
            auto cos_2phi = bsk::fma(cos_phi, cos_phi, (-(sin_phi * sin_phi)));
            auto sin_2phi = ((2.0f * sin_phi) * cos_phi);
            auto t10_ = _rotate_flip_phase(cosine, sine, cos_phi, sin_phi, cos_2phi, sin_2phi, fplus_real, fplus_imag, fminus_real, fminus_imag, longitudinal_real, longitudinal_imag);
            rotated_pr = bsk::get<0>(t10_);
            rotated_pi = bsk::get<1>(t10_);
            rotated_mr = bsk::get<2>(t10_);
            rotated_mi = bsk::get<3>(t10_);
            rotated_zr = bsk::get<4>(t10_);
            rotated_zi = bsk::get<5>(t10_);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t11_ = _rotate_flip_phase(cosine, sine, cos_phi, sin_phi, cos_2phi, sin_2phi, bplus_real, bplus_imag, bminus_real, bminus_imag, bound_real, bound_imag);
                b_rot_pr = bsk::get<0>(t11_);
                b_rot_pi = bsk::get<1>(t11_);
                b_rot_mr = bsk::get<2>(t11_);
                b_rot_mi = bsk::get<3>(t11_);
                b_rot_zr = bsk::get<4>(t11_);
                b_rot_zi = bsk::get<5>(t11_);
                if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                    // The same pulse, the same rotation: a chemical shift moves
                    // where a pool precesses, not what a pulse does to it.
                    auto t12_ = _rotate_spinor(bsk::get<0>(pair), bsk::get<1>(pair), spun_br, spun_bi, bplus_real, bplus_imag, bminus_real, bminus_imag, bound_real, bound_imag);
                    b_rot_pr = bsk::get<0>(t12_);
                    b_rot_pi = bsk::get<1>(t12_);
                    b_rot_mr = bsk::get<2>(t12_);
                    b_rot_mi = bsk::get<3>(t12_);
                    b_rot_zr = bsk::get<4>(t12_);
                    b_rot_zi = bsk::get<5>(t12_);
                }
            }
            if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                rotated_pr = shaped_pr;
                rotated_pi = shaped_pi;
                rotated_mr = shaped_mr;
                rotated_mi = shaped_mi;
                rotated_zr = shaped_zr;
                rotated_zi = shaped_zi;
            }
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                bplus_real = b_rot_pr;
                bplus_imag = b_rot_pi;
                bminus_real = b_rot_mr;
                bminus_imag = b_rot_mi;
                bound_real = b_rot_zr;
                bound_imag = b_rot_zi;
            }
            if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
                // The semisolid pool absorbs the power the pulse deposits, so it
                // reads the bare flip the transmit field gives the voxel -- not the
                // slice-shaped rotation the free pool takes from the table.
                auto offset = (bsk::ld((rf_frequency + event)) - atom_b0);
                auto absorbed = bsk::exp((((bsk::ld((saturation + event)) * alpha) * alpha) * _lineshape_at(lineshape, offset, lineshape_bins, lineshape_step)));
                if (bsk::truth((pools == 1))) {
                    bound_real = (absorbed * bound_real);
                    bound_imag = (absorbed * bound_imag);
                } else {
                    semisolid_real = (absorbed * semisolid_real);
                    semisolid_imag = (absorbed * semisolid_imag);
                }
            }
            fplus_real = rotated_pr;
            fplus_imag = rotated_pi;
            fminus_real = rotated_mr;
            fminus_imag = rotated_mi;
            longitudinal_real = rotated_zr;
            longitudinal_imag = rotated_zi;
        }
        if (bsk::truth(bsk::band((bsk::band(event_action, 32) != 0), (event_kind == 2)))) {
            auto adc_cos = _event_value(phase_cos, event_base, event, active_atom, single_train);
            auto adc_sin = _event_value(phase_sin, event_base, event, active_atom, single_train);
            // A coil sees the whole voxel, so what it records is the sum over
            // pools; each pool's share is already in its own state.
            read_real = fplus_real;
            read_imag = fplus_imag;
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                read_real = (fplus_real + bplus_real);
                read_imag = (fplus_imag + bplus_imag);
            }
            auto signal_real = (atom_m0 * ((read_real * adc_cos) + (read_imag * adc_sin)));
            auto signal_imag = (atom_m0 * ((read_imag * adc_cos) - (read_real * adc_sin)));
            auto out_ = bsk::ld((output_index + event));
            auto output_offset = ((problem * output_count) + out_);
            auto output_mask = bsk::band(bsk::band(active_atom, (state == 0)), (out_ >= 0));
            bsk::st(((output_real + output_offset) + state), signal_real, output_mask);
            bsk::st(((output_imag + output_offset) + state), signal_imag, output_mask);
        }
        if (bsk::truth((bsk::band(event_action, 18) != 0))) {
            auto t13_ = _shift(fplus_real, fplus_imag, fminus_real, fminus_imag, state, state_mask, state_count);
            fplus_real = bsk::get<0>(t13_);
            fplus_imag = bsk::get<1>(t13_);
            fminus_real = bsk::get<2>(t13_);
            fminus_imag = bsk::get<3>(t13_);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t14_ = _shift(bplus_real, bplus_imag, bminus_real, bminus_imag, state, state_mask, state_count);
                bplus_real = bsk::get<0>(t14_);
                bplus_imag = bsk::get<1>(t14_);
                bminus_real = bsk::get<2>(t14_);
                bminus_imag = bsk::get<3>(t14_);
            }
        }
        if (bsk::truth((bsk::band(event_action, 8) != 0))) {
            fplus_real = empty;
            fplus_imag = empty;
            fminus_real = empty;
            fminus_imag = empty;
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                bplus_real = empty;
                bplus_imag = empty;
                bminus_real = empty;
                bminus_imag = empty;
            }
        }
    }
}

// Fill one row of the three-pool operator table.
//
// The row is ``expm((K - diag(R1)) dt)`` with no washout applied, so the
// event that reads it supplies its own attenuation. Laid out
// ``(rows, 9, voxels)`` -- entry-major over the voxel axis -- so the nine
// loads an event makes are each coalesced.
BSK_HD void _three_pool_table_kernel(float* t1, float* t1_pool_b, float* t1_bound, float* pool_b_exchange, float* bound_exchange, float* pool_b_fraction, float* bound_fraction, float* durations, std::int32_t* rows, float* table, bsk::index_t voxel_count, bsk::index_t BLOCK, bsk::index_t narrow) {
    auto row = bsk::ld((rows + bsk::program_id(0)));
    auto atom = ((bsk::program_id(1) * BLOCK) + bsk::arange_x());
    auto live = (atom < voxel_count);
    auto dt = bsk::ld((durations + row));
    auto fraction_b = bsk::ld((pool_b_fraction + atom), live, 0.0f);
    auto fraction_c = bsk::ld((bound_fraction + atom), live, 0.0f);
    auto t0_ = _three_pool_step(bsk::truediv(1000.0f, bsk::ld((t1 + atom), live, 1.0f)), bsk::truediv(1000.0f, bsk::ld((t1_pool_b + atom), live, 1.0f)), bsk::truediv(1000.0f, bsk::ld((t1_bound + atom), live, 1.0f)), bsk::ld((pool_b_exchange + atom), live, 0.0f), bsk::ld((bound_exchange + atom), live, 0.0f), fraction_b, fraction_c, dt, (1.0f + (0.0f * dt)), narrow);
    auto e00 = bsk::get<0>(t0_);
    auto e01 = bsk::get<1>(t0_);
    auto e02 = bsk::get<2>(t0_);
    auto e10 = bsk::get<3>(t0_);
    auto e11 = bsk::get<4>(t0_);
    auto e12 = bsk::get<5>(t0_);
    auto e20 = bsk::get<6>(t0_);
    auto e21 = bsk::get<7>(t0_);
    auto e22 = bsk::get<8>(t0_);
    auto base = ((table + (row * (9 * voxel_count))) + atom);
    bsk::st((base + (0 * voxel_count)), e00, live);
    bsk::st((base + (1 * voxel_count)), e01, live);
    bsk::st((base + (2 * voxel_count)), e02, live);
    bsk::st((base + (3 * voxel_count)), e10, live);
    bsk::st((base + (4 * voxel_count)), e11, live);
    bsk::st((base + (5 * voxel_count)), e12, live);
    bsk::st((base + (6 * voxel_count)), e20, live);
    bsk::st((base + (7 * voxel_count)), e21, live);
    bsk::st((base + (8 * voxel_count)), e22, live);
}

// Fill one row of the three-pool operator table, value and direction.
//
// The row holds nine undamped entries and the nine a direction through the
// tissue gives them, both at ``d_dt`` of zero -- the interval's own share of
// the direction is ``A1 C d_dt``, which the reading event adds because
// ``d_dt`` is its own and the row's is not. Laid out ``(rows, 18, voxels)``,
// the tangent following the value.
BSK_HD void _three_pool_table_jvp_kernel(float* t1, float* t1_pool_b, float* t1_bound, float* pool_b_exchange, float* bound_exchange, float* pool_b_fraction, float* bound_fraction, float* d_t1, float* d_t1_pool_b, float* d_t1_bound, float* d_pool_b_exchange, float* d_bound_exchange, float* d_pool_b_fraction, float* d_bound_fraction, float* durations, std::int32_t* rows, float* table, bsk::index_t voxel_count, bsk::index_t BLOCK, bsk::index_t narrow) {
    auto row = bsk::ld((rows + bsk::program_id(0)));
    auto atom = ((bsk::program_id(1) * BLOCK) + bsk::arange_x());
    auto live = (atom < voxel_count);
    auto dt = bsk::ld((durations + row));
    auto nil = (0.0f * dt);
    auto value_t1 = bsk::ld((t1 + atom), live, 1.0f);
    auto value_t1b = bsk::ld((t1_pool_b + atom), live, 1.0f);
    auto value_t1c = bsk::ld((t1_bound + atom), live, 1.0f);
    auto t0_ = _three_pool_step_jvp(bsk::truediv(1000.0f, value_t1), bsk::truediv((-1000.0f * bsk::ld((d_t1 + atom), live, 0.0f)), (value_t1 * value_t1)), bsk::truediv(1000.0f, value_t1b), bsk::truediv((-1000.0f * bsk::ld((d_t1_pool_b + atom), live, 0.0f)), (value_t1b * value_t1b)), bsk::truediv(1000.0f, value_t1c), bsk::truediv((-1000.0f * bsk::ld((d_t1_bound + atom), live, 0.0f)), (value_t1c * value_t1c)), bsk::ld((pool_b_exchange + atom), live, 0.0f), bsk::ld((d_pool_b_exchange + atom), live, 0.0f), bsk::ld((bound_exchange + atom), live, 0.0f), bsk::ld((d_bound_exchange + atom), live, 0.0f), bsk::ld((pool_b_fraction + atom), live, 0.0f), bsk::ld((d_pool_b_fraction + atom), live, 0.0f), bsk::ld((bound_fraction + atom), live, 0.0f), bsk::ld((d_bound_fraction + atom), live, 0.0f), dt, nil, (1.0f + nil), nil, narrow);
    auto e00 = bsk::get<0>(t0_);
    auto e01 = bsk::get<1>(t0_);
    auto e02 = bsk::get<2>(t0_);
    auto e10 = bsk::get<3>(t0_);
    auto e11 = bsk::get<4>(t0_);
    auto e12 = bsk::get<5>(t0_);
    auto e20 = bsk::get<6>(t0_);
    auto e21 = bsk::get<7>(t0_);
    auto e22 = bsk::get<8>(t0_);
    auto d00 = bsk::get<12>(t0_);
    auto d01 = bsk::get<13>(t0_);
    auto d02 = bsk::get<14>(t0_);
    auto d10 = bsk::get<15>(t0_);
    auto d11 = bsk::get<16>(t0_);
    auto d12 = bsk::get<17>(t0_);
    auto d20 = bsk::get<18>(t0_);
    auto d21 = bsk::get<19>(t0_);
    auto d22 = bsk::get<20>(t0_);
    auto base = ((table + (row * (18 * voxel_count))) + atom);
    bsk::st((base + (0 * voxel_count)), e00, live);
    bsk::st((base + (1 * voxel_count)), e01, live);
    bsk::st((base + (2 * voxel_count)), e02, live);
    bsk::st((base + (3 * voxel_count)), e10, live);
    bsk::st((base + (4 * voxel_count)), e11, live);
    bsk::st((base + (5 * voxel_count)), e12, live);
    bsk::st((base + (6 * voxel_count)), e20, live);
    bsk::st((base + (7 * voxel_count)), e21, live);
    bsk::st((base + (8 * voxel_count)), e22, live);
    bsk::st((base + (9 * voxel_count)), d00, live);
    bsk::st((base + (10 * voxel_count)), d01, live);
    bsk::st((base + (11 * voxel_count)), d02, live);
    bsk::st((base + (12 * voxel_count)), d10, live);
    bsk::st((base + (13 * voxel_count)), d11, live);
    bsk::st((base + (14 * voxel_count)), d12, live);
    bsk::st((base + (15 * voxel_count)), d20, live);
    bsk::st((base + (16 * voxel_count)), d21, live);
    bsk::st((base + (17 * voxel_count)), d22, live);
}

template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _shift_real(const T0& plus, const T1& minus, const T2& state, const T3& state_mask, const T4& state_count) {
    auto shifted_plus = bsk::where(bsk::band((state > 0), state_mask), _up(plus, state), 0.0f);
    auto shifted_minus = bsk::where(bsk::band(((state + 1) < state_count), state_mask), _down(minus, state), 0.0f);
    return bsk::make_tup(bsk::where((state == 0), (-shifted_minus), shifted_plus), shifted_minus);
}

// Transpose of ``_shift_real``.
//
// The ``a0 = -b0`` coupling sends the incoming plus adjoint back onto minus,
// at the index the minus shift moves it to.
template <class T0, class T1, class T2, class T3, class T4>
BSK_HD auto _shift_real_adjoint(const T0& plus_bar, const T1& minus_bar, const T2& state, const T3& state_mask, const T4& state_count) {
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4> | 0, 3)> shifted_minus{};
    auto carry = (-bsk::where(state_mask, _first(plus_bar, state), 0.0f));
    auto shifted_plus = bsk::where(bsk::band(((state + 1) < state_count), state_mask), _down(plus_bar, state), 0.0f);
    shifted_minus = bsk::where(bsk::band((state > 0), state_mask), _up(minus_bar, state), 0.0f);
    shifted_minus = bsk::where((state == 1), (shifted_minus + carry), shifted_minus);
    return bsk::make_tup(shifted_plus, shifted_minus);
}

BSK_HD void _epg_real_vjp_kernel(float* t1, float* t2, float* m0, float* b1, float* inversion_efficiency, float* diffusion, float* duration, std::int32_t* kind, float* flip, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* grad_output_imag, float* grad_tissue, float* grad_flip, float* grad_duration, float* trajectory_value, bsk::index_t problem_base, bsk::index_t problem_end, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shim_rows, bsk::index_t shimmed, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 3> adjoint_mv{};
    bsk::V<float, 3> adjoint_pv{};
    bsk::V<float, 3> alpha_bar_terms_value{};
    bsk::V<float, 2> alpha_value{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_inversion{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 3> bare1_value{};
    bsk::V<float, 3> bare2_value{};
    bsk::V<float, 2> chs_value{};
    bsk::V<float, 2> cosine_value{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_z{};
    bool do_shift{};
    bsk::V<float, 2> dt_value{};
    bsk::V<float, 3> duration_gain_value{};
    bsk::V<float, 3> e1_value{};
    bsk::V<float, 3> e2_value{};
    std::int64_t event{};
    std::int32_t event_action{};
    bsk::V<float, 2> event_flip{};
    std::int32_t event_kind{};
    bsk::V<float, 2> grad_b1_value{};
    bsk::V<float, 2> grad_damping_value{};
    bsk::V<float, 2> grad_e1_value{};
    bsk::V<float, 2> grad_inversion_value{};
    bsk::V<float, 2> grad_m0_value{};
    bsk::V<float, 3> grad_t1_value{};
    bsk::V<float, 3> grad_t2_value{};
    bsk::V<float, 2> half_sine_value{};
    bool invert{};
    bool is_inversion{};
    bool is_rf{};
    bsk::V<float, 3> long_bar_value{};
    bsk::V<float, 3> long_value{};
    bsk::V<float, 3> minus_bar_value{};
    bsk::V<float, 3> minus_value{};
    bsk::V<float, 3> plus_bar_value{};
    bsk::V<float, 3> plus_value{};
    bool pre_shift{};
    bsk::V<std::int32_t, 2> problem{};
    bsk::V<float, 2> pulse_b1{};
    bsk::V<float, 3> recovery_value{};
    bool rotate{};
    bsk::V<float, 3> rotated_mbv{};
    bsk::V<float, 3> rotated_mv{};
    bsk::V<float, 3> rotated_pbv{};
    bsk::V<float, 3> rotated_pv{};
    bsk::V<float, 3> rotated_zbv{};
    bsk::V<float, 3> rotated_zv{};
    bsk::V<float, 3> row_m_value{};
    bsk::V<float, 3> row_p_value{};
    bsk::V<float, 3> row_z_value{};
    bsk::V<float, 3> shifted_mv{};
    bsk::V<float, 3> shifted_pv{};
    std::int64_t shim_row{};
    bsk::V<float, 2> shs_value{};
    bsk::V<float, 2> sine_value{};
    bsk::V<std::int32_t, 3> slot{};
    bool spoil{};
    bsk::V<float, 2> spread_value{};
    bsk::V<float, 3> stage_mv{};
    bsk::V<float, 3> stage_pv{};
    problem = (problem_base + (bsk::program_id(0) * problems));
    problem = (problem + bsk::arange_y());
    auto state = bsk::arange_x();
    // The grid rounds up to whole tiles, so the last program of a wave reaches
    // past it. Those problems are real, but their trajectory rows belong to a
    // later launch and do not exist yet.
    auto active_atom = (problem < problem_end);
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    // The trajectory holds the state entering every event: three planes of
    // configuration orders.
    auto record_stride = (3 * state_count);
    auto trajectory = ((((problem - problem_base) * event_count) * record_stride) + state);
    auto minus_plane = state_count;
    auto long_plane = (2 * state_count);
    auto empty = bsk::full<float, 3>(0);
    plus_value = empty;
    minus_value = empty;
    long_value = (empty + bsk::where((state == 0), 1.0f, 0.0f));
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_inversion = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inversion = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    atom_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
    }
    auto order = bsk::cast<float>(state);
    auto longitudinal_weight = (order * order);
    auto transverse_weight = ((longitudinal_weight + order) + 0.3333333333333333f);
    auto rate1_value = bsk::truediv(1000.0f, atom_t1);
    auto rate2_value = bsk::truediv(1000.0f, atom_t2);
    auto event_base = (train * event_count);
    for (bsk::index_t event = 0; event < event_count; event += 1) {
        slot = (trajectory + (event * record_stride));
        bsk::st((trajectory_value + slot), plus_value, state_mask);
        bsk::st(((trajectory_value + slot) + minus_plane), minus_value, state_mask);
        bsk::st(((trajectory_value + slot) + long_plane), long_value, state_mask);
        dt_value = _event_value(duration, event_base, event, active_atom, single_train);
        e1_value = bsk::exp(((-rate1_value) * dt_value));
        e2_value = bsk::exp(((-rate2_value) * dt_value));
        damp_z = 1.0f;
        damp_t = 1.0f;
        if (bsk::truth(diffusing)) {
            auto t0_ = _damping(atom_damping, dt_value, order);
            damp_z = bsk::get<0>(t0_);
            damp_t = bsk::get<1>(t0_);
        }
        // Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = (1.0f - e1_value);
        bare1_value = e1_value;
        bare2_value = e2_value;
        e1_value = (bare1_value * damp_z);
        e2_value = (bare2_value * damp_t);
        plus_value = (plus_value * e2_value);
        minus_value = (minus_value * e2_value);
        long_value = (long_value * e1_value);
        long_value = (long_value + bsk::where((state == 0), recovery_value, 0.0f));
        event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        pre_shift = (bsk::band(event_action, 1) != 0);
        auto t1_ = _shift_real(plus_value, minus_value, state, state_mask, state_count);
        shifted_pv = bsk::get<0>(t1_);
        shifted_mv = bsk::get<1>(t1_);
        plus_value = bsk::where(pre_shift, shifted_pv, plus_value);
        minus_value = bsk::where(pre_shift, shifted_mv, minus_value);
        event_kind = bsk::ld((kind + event));
        is_rf = (event_kind == 1);
        is_inversion = (bsk::band(event_action, 4) != 0);
        invert = bsk::band(is_rf, is_inversion);
        long_value = bsk::where(invert, ((-atom_inversion) * long_value), long_value);
        event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        pulse_b1 = atom_b1;
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse the row of the shim it drives.
        if (bsk::truth(shimmed)) {
            shim_row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            if (bsk::truth(transmit)) {
                pulse_b1 = bsk::ld(((b1 + shim_row) + atom), active_atom, 1.0f);
            }
        }
        alpha_value = (event_flip * pulse_b1);
        cosine_value = bsk::cos(alpha_value);
        sine_value = bsk::sin(alpha_value);
        chs_value = (0.5f * (1.0f + cosine_value));
        shs_value = (0.5f * (1.0f - cosine_value));
        half_sine_value = (0.5f * sine_value);
        rotated_pv = ((chs_value * plus_value) + (shs_value * minus_value));
        rotated_pv = (rotated_pv - (sine_value * long_value));
        rotated_mv = ((shs_value * plus_value) + (chs_value * minus_value));
        rotated_mv = (rotated_mv + (sine_value * long_value));
        rotated_zv = ((half_sine_value * plus_value) - (half_sine_value * minus_value));
        rotated_zv = (rotated_zv + (cosine_value * long_value));
        rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        plus_value = bsk::where(rotate, rotated_pv, plus_value);
        minus_value = bsk::where(rotate, rotated_mv, minus_value);
        long_value = bsk::where(rotate, rotated_zv, long_value);
        do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        auto t2_ = _shift_real(plus_value, minus_value, state, state_mask, state_count);
        shifted_pv = bsk::get<0>(t2_);
        shifted_mv = bsk::get<1>(t2_);
        plus_value = bsk::where(do_shift, shifted_pv, plus_value);
        minus_value = bsk::where(do_shift, shifted_mv, minus_value);
        spoil = (bsk::band(event_action, 8) != 0);
        plus_value = bsk::where(spoil, 0.0f, plus_value);
        minus_value = bsk::where(spoil, 0.0f, minus_value);
    }
    plus_bar_value = empty;
    minus_bar_value = empty;
    long_bar_value = empty;
    auto zero = bsk::full<float, 2>(0);
    grad_t1_value = zero;
    grad_t2_value = zero;
    grad_m0_value = zero;
    grad_b1_value = zero;
    grad_inversion_value = zero;
    grad_damping_value = zero;
    for (bsk::index_t reverse = 0; reverse < event_count; reverse += 1) {
        event = ((event_count - 1) - reverse);
        slot = (trajectory + (event * record_stride));
        auto entry_pv = bsk::ld((trajectory_value + slot), state_mask, 0.0f);
        auto entry_mv = bsk::ld(((trajectory_value + slot) + minus_plane), state_mask, 0.0f);
        auto entry_zv = bsk::ld(((trajectory_value + slot) + long_plane), state_mask, 0.0f);
        event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        event_kind = bsk::ld((kind + event));
        dt_value = _event_value(duration, event_base, event, active_atom, single_train);
        e1_value = bsk::exp(((-rate1_value) * dt_value));
        e2_value = bsk::exp(((-rate2_value) * dt_value));
        damp_z = 1.0f;
        damp_t = 1.0f;
        if (bsk::truth(diffusing)) {
            auto t3_ = _damping(atom_damping, dt_value, order);
            damp_z = bsk::get<0>(t3_);
            damp_t = bsk::get<1>(t3_);
        }
        // Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = (1.0f - e1_value);
        bare1_value = e1_value;
        bare2_value = e2_value;
        e1_value = (bare1_value * damp_z);
        e2_value = (bare2_value * damp_t);
        // Replay the intra-event stages from the recorded entry state.
        stage_pv = (entry_pv * e2_value);
        stage_mv = (entry_mv * e2_value);
        auto stage_zv = ((entry_zv * e1_value) + bsk::where((state == 0), recovery_value, 0.0f));
        pre_shift = (bsk::band(event_action, 1) != 0);
        auto t4_ = _shift_real(stage_pv, stage_mv, state, state_mask, state_count);
        shifted_pv = bsk::get<0>(t4_);
        shifted_mv = bsk::get<1>(t4_);
        stage_pv = bsk::where(pre_shift, shifted_pv, stage_pv);
        stage_mv = bsk::where(pre_shift, shifted_mv, stage_mv);
        // Undo the trailing spoil or shift.
        do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        spoil = (bsk::band(event_action, 8) != 0);
        auto t5_ = _shift_real_adjoint(plus_bar_value, minus_bar_value, state, state_mask, state_count);
        adjoint_pv = bsk::get<0>(t5_);
        adjoint_mv = bsk::get<1>(t5_);
        auto trailing = bsk::band(do_shift, bsk::bnot(spoil));
        plus_bar_value = bsk::where(spoil, 0.0f, bsk::where(trailing, adjoint_pv, plus_bar_value));
        minus_bar_value = bsk::where(spoil, 0.0f, bsk::where(trailing, adjoint_mv, minus_bar_value));
        is_rf = (event_kind == 1);
        is_inversion = (bsk::band(event_action, 4) != 0);
        invert = bsk::band(is_rf, is_inversion);
        grad_inversion_value = (grad_inversion_value + (-bsk::sum_x(bsk::where(invert, (long_bar_value * stage_zv), 0.0f))));
        long_bar_value = bsk::where(invert, ((-atom_inversion) * long_bar_value), long_bar_value);
        event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        pulse_b1 = atom_b1;
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse the row of the shim it drives.
        if (bsk::truth(shimmed)) {
            shim_row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            if (bsk::truth(transmit)) {
                pulse_b1 = bsk::ld(((b1 + shim_row) + atom), active_atom, 1.0f);
            }
        }
        alpha_value = (event_flip * pulse_b1);
        cosine_value = bsk::cos(alpha_value);
        sine_value = bsk::sin(alpha_value);
        chs_value = (0.5f * (1.0f + cosine_value));
        shs_value = (0.5f * (1.0f - cosine_value));
        half_sine_value = (0.5f * sine_value);
        // d/dalpha of each output row, contracted with the adjoint.
        row_p_value = ((half_sine_value * stage_mv) - (half_sine_value * stage_pv));
        row_p_value = (row_p_value - (cosine_value * stage_zv));
        row_m_value = ((half_sine_value * stage_pv) - (half_sine_value * stage_mv));
        row_m_value = (row_m_value + (cosine_value * stage_zv));
        row_z_value = (((0.5f * cosine_value) * stage_pv) - ((0.5f * cosine_value) * stage_mv));
        row_z_value = (row_z_value - (sine_value * stage_zv));
        alpha_bar_terms_value = (plus_bar_value * row_p_value);
        alpha_bar_terms_value = (alpha_bar_terms_value + (minus_bar_value * row_m_value));
        alpha_bar_terms_value = (alpha_bar_terms_value + (long_bar_value * row_z_value));
        rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        auto grad_alpha_value = bsk::sum_x(bsk::where(rotate, alpha_bar_terms_value, 0.0f));
        // Transpose of the rotation.
        rotated_pbv = ((chs_value * plus_bar_value) + (shs_value * minus_bar_value));
        rotated_pbv = (rotated_pbv + (half_sine_value * long_bar_value));
        rotated_mbv = ((shs_value * plus_bar_value) + (chs_value * minus_bar_value));
        rotated_mbv = (rotated_mbv - (half_sine_value * long_bar_value));
        rotated_zbv = (((-sine_value) * plus_bar_value) + (sine_value * minus_bar_value));
        rotated_zbv = (rotated_zbv + (cosine_value * long_bar_value));
        plus_bar_value = bsk::where(rotate, rotated_pbv, plus_bar_value);
        minus_bar_value = bsk::where(rotate, rotated_mbv, minus_bar_value);
        long_bar_value = bsk::where(rotate, rotated_zbv, long_bar_value);
        auto writes_flip = bsk::band(active_atom, rotate);
        bsk::atomic_add(((grad_flip + event_base) + event), (grad_alpha_value * pulse_b1), writes_flip);
        if (bsk::truth(shimmed)) {
            // A pulse's transmit gradient belongs to the shim it drives, so
            // with several it lands in that shim's row rather than in a
            // register summed over the whole train.
            bsk::atomic_add((((grad_tissue + (3 * atom_count)) + shim_row) + atom), (grad_alpha_value * event_flip), writes_flip);
        } else {
            grad_b1_value = (grad_b1_value + bsk::where(rotate, (grad_alpha_value * event_flip), 0.0f));
        }
        // The sample is i * m0 * plus[0]; only the imaginary seed acts.
        auto record = bsk::band((bsk::band(event_action, 32) != 0), (event_kind == 2));
        auto out_ = bsk::ld((output_index + event));
        auto seed = bsk::ld(((grad_output_imag + (problem * output_count)) + out_), bsk::band(bsk::band(active_atom, record), (out_ >= 0)), 0.0f);
        grad_m0_value = (grad_m0_value + bsk::sum_x(bsk::where((state == 0), (seed * stage_pv), 0.0f)));
        plus_bar_value = (plus_bar_value + bsk::where((state == 0), (seed * atom_m0), 0.0f));
        auto t6_ = _shift_real_adjoint(plus_bar_value, minus_bar_value, state, state_mask, state_count);
        adjoint_pv = bsk::get<0>(t6_);
        adjoint_mv = bsk::get<1>(t6_);
        plus_bar_value = bsk::where(pre_shift, adjoint_pv, plus_bar_value);
        minus_bar_value = bsk::where(pre_shift, adjoint_mv, minus_bar_value);
        auto cot2_value = ((plus_bar_value * entry_pv) + (minus_bar_value * entry_mv));
        auto cot1_value = (long_bar_value * entry_zv);
        auto grad_e2_value = bsk::sum_x((cot2_value * damp_t));
        grad_e1_value = bsk::sum_x((cot1_value * damp_z));
        grad_e1_value = (grad_e1_value - bsk::sum_x(bsk::where((state == 0), long_bar_value, 0.0f)));
        // The rate and the interval multiply every order's b-weight, so both
        // take a weighted sum. Order zero has no longitudinal weight, which
        // keeps recovery out of this.
        spread_value = zero;
        if (bsk::truth(diffusing)) {
            auto weighted_value = ((((cot1_value * bare1_value) * damp_z) * longitudinal_weight) + (((cot2_value * bare2_value) * damp_t) * transverse_weight));
            spread_value = bsk::sum_x(weighted_value);
            grad_damping_value = (grad_damping_value + ((-spread_value) * dt_value));
        }
        plus_bar_value = (plus_bar_value * e2_value);
        minus_bar_value = (minus_bar_value * e2_value);
        long_bar_value = (long_bar_value * e1_value);
        auto inverse1_value = bsk::truediv(1000.0f, (atom_t1 * atom_t1));
        auto inverse2_value = bsk::truediv(1000.0f, (atom_t2 * atom_t2));
        grad_t1_value = (grad_t1_value + (grad_e1_value * ((bare1_value * dt_value) * inverse1_value)));
        grad_t2_value = (grad_t2_value + (grad_e2_value * ((bare2_value * dt_value) * inverse2_value)));
        duration_gain_value = ((-grad_e1_value) * (rate1_value * bare1_value));
        duration_gain_value = (duration_gain_value - (grad_e2_value * (rate2_value * bare2_value)));
        duration_gain_value = (duration_gain_value + ((-spread_value) * atom_damping));
        bsk::atomic_add(((grad_duration + event_base) + event), duration_gain_value, active_atom);
    }
    bsk::atomic_add((grad_tissue + atom), grad_t1_value, active_atom);
    bsk::atomic_add(((grad_tissue + atom_count) + atom), grad_t2_value, active_atom);
    bsk::atomic_add(((grad_tissue + (2 * atom_count)) + atom), grad_m0_value, active_atom);
    if (bsk::truth((!bsk::truth(shimmed)))) {
        bsk::atomic_add(((grad_tissue + (3 * atom_count)) + atom), grad_b1_value, active_atom);
    }
    // The transmit pair takes a row per shim each in the plane the complex
    // path allocates, so the rows past it move even though this kernel leaves
    // the transmit phase at zero throughout.
    auto past_transmit = (2 * (shim_rows - 1));
    bsk::atomic_add(((grad_tissue + ((6 + past_transmit) * atom_count)) + atom), grad_inversion_value, active_atom);
    bsk::atomic_add(((grad_tissue + ((7 + past_transmit) * atom_count)) + atom), grad_damping_value, active_atom);
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

// One pool through a hard pulse, carried alongside a tangent.
//
// Pulled out of the kernel body so a second pool can take the same
// rotation: a chemical shift moves where a pool precesses, not what a
// pulse does to it.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23>
BSK_HD auto _rotate_flip_phase_jvp(const T0& cosine, const T1& dcosine, const T2& sine, const T3& dsine, const T4& cos_phi, const T5& dcos_phi, const T6& sin_phi, const T7& dsin_phi, const T8& cos_2phi, const T9& dcos_2phi, const T10& sin_2phi, const T11& dsin_2phi, const T12& fpr, const T13& fpi, const T14& fmr, const T15& fmi, const T16& zr, const T17& zi, const T18& dfpr, const T19& dfpi, const T20& dfmr, const T21& dfmi, const T22& dzr, const T23& dzi) {
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> dmi_a{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> dmr_a{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> dpi_a{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> dpr_a{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> rotated_dmi{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> rotated_dmr{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> rotated_dpi{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> rotated_dpr{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> rotated_dzi{};
    bsk::tile_t<float, bsk::fit_axes(bsk::joint_axes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> | 0, 3)> rotated_dzr{};
    auto ch = (0.5f * (1.0f + cosine));
    auto sh = (0.5f * (1.0f - cosine));
    auto dch = (0.5f * dcosine);
    auto dsh = (-0.5f * dcosine);
    auto pr_a = ((cos_2phi * fmr) - (sin_2phi * fmi));
    dpr_a = ((dcos_2phi * fmr) + (cos_2phi * dfmr));
    dpr_a = (dpr_a - ((dsin_2phi * fmi) + (sin_2phi * dfmi)));
    auto pr_b = ((sin_phi * zr) + (cos_phi * zi));
    auto dpr_b = ((((dsin_phi * zr) + (sin_phi * dzr)) + (dcos_phi * zi)) + (cos_phi * dzi));
    auto rotated_pr = (((ch * fpr) + (sh * pr_a)) + (sine * pr_b));
    rotated_dpr = ((((dch * fpr) + (ch * dfpr)) + (dsh * pr_a)) + (sh * dpr_a));
    rotated_dpr = (rotated_dpr + ((dsine * pr_b) + (sine * dpr_b)));
    auto pi_a = ((sin_2phi * fmr) + (cos_2phi * fmi));
    dpi_a = ((dsin_2phi * fmr) + (sin_2phi * dfmr));
    dpi_a = (dpi_a + ((dcos_2phi * fmi) + (cos_2phi * dfmi)));
    auto pi_b = ((sin_phi * zi) - (cos_phi * zr));
    auto dpi_b = ((((dsin_phi * zi) + (sin_phi * dzi)) - (dcos_phi * zr)) - (cos_phi * dzr));
    auto rotated_pi = (((ch * fpi) + (sh * pi_a)) + (sine * pi_b));
    rotated_dpi = ((((dch * fpi) + (ch * dfpi)) + (dsh * pi_a)) + (sh * dpi_a));
    rotated_dpi = (rotated_dpi + ((dsine * pi_b) + (sine * dpi_b)));
    auto mr_a = ((cos_2phi * fpr) + (sin_2phi * fpi));
    dmr_a = ((dcos_2phi * fpr) + (cos_2phi * dfpr));
    dmr_a = (dmr_a + ((dsin_2phi * fpi) + (sin_2phi * dfpi)));
    auto mr_b = ((sin_phi * zr) - (cos_phi * zi));
    auto dmr_b = ((((dsin_phi * zr) + (sin_phi * dzr)) - (dcos_phi * zi)) - (cos_phi * dzi));
    auto rotated_mr = (((sh * mr_a) + (ch * fmr)) + (sine * mr_b));
    rotated_dmr = ((((dsh * mr_a) + (sh * dmr_a)) + (dch * fmr)) + (ch * dfmr));
    rotated_dmr = (rotated_dmr + ((dsine * mr_b) + (sine * dmr_b)));
    auto mi_a = (((-sin_2phi) * fpr) + (cos_2phi * fpi));
    dmi_a = (((-dsin_2phi) * fpr) - (sin_2phi * dfpr));
    dmi_a = (dmi_a + ((dcos_2phi * fpi) + (cos_2phi * dfpi)));
    auto mi_b = ((cos_phi * zr) + (sin_phi * zi));
    auto dmi_b = ((((dcos_phi * zr) + (cos_phi * dzr)) + (dsin_phi * zi)) + (sin_phi * dzi));
    auto rotated_mi = (((sh * mi_a) + (ch * fmi)) + (sine * mi_b));
    rotated_dmi = ((((dsh * mi_a) + (sh * dmi_a)) + (dch * fmi)) + (ch * dfmi));
    rotated_dmi = (rotated_dmi + ((dsine * mi_b) + (sine * dmi_b)));
    auto zr_a = ((sin_phi * fpr) - (cos_phi * fpi));
    auto dzr_a = ((((dsin_phi * fpr) + (sin_phi * dfpr)) - (dcos_phi * fpi)) - (cos_phi * dfpi));
    auto zr_b = ((sin_phi * fmr) + (cos_phi * fmi));
    auto dzr_b = ((((dsin_phi * fmr) + (sin_phi * dfmr)) + (dcos_phi * fmi)) + (cos_phi * dfmi));
    auto rotated_zr = ((((-0.5f * sine) * zr_a) - ((0.5f * sine) * zr_b)) + (cosine * zr));
    rotated_dzr = (-0.5f * ((dsine * zr_a) + (sine * dzr_a)));
    rotated_dzr = (rotated_dzr - (0.5f * ((dsine * zr_b) + (sine * dzr_b))));
    rotated_dzr = (rotated_dzr + ((dcosine * zr) + (cosine * dzr)));
    auto zi_a = ((cos_phi * fpr) + (sin_phi * fpi));
    auto dzi_a = ((((dcos_phi * fpr) + (cos_phi * dfpr)) + (dsin_phi * fpi)) + (sin_phi * dfpi));
    auto zi_b = ((cos_phi * fmr) - (sin_phi * fmi));
    auto dzi_b = ((((dcos_phi * fmr) + (cos_phi * dfmr)) - (dsin_phi * fmi)) - (sin_phi * dfmi));
    auto rotated_zi = ((((-0.5f * sine) * zi_a) + ((0.5f * sine) * zi_b)) + (cosine * zi));
    rotated_dzi = (-0.5f * ((dsine * zi_a) + (sine * dzi_a)));
    rotated_dzi = (rotated_dzi + (0.5f * ((dsine * zi_b) + (sine * dzi_b))));
    rotated_dzi = (rotated_dzi + ((dcosine * zi) + (cosine * dzi)));
    return bsk::make_tup(rotated_pr, rotated_pi, rotated_mr, rotated_mi, rotated_zr, rotated_zi, rotated_dpr, rotated_dpi, rotated_dmr, rotated_dmi, rotated_dzr, rotated_dzi);
}

BSK_HD void _epg_jvp_kernel(float* t1, float* t2, float* m0, float* b1, float* b1_phase, float* b0, float* inversion_efficiency, float* diffusion, float* velocity, float* bound_fraction, float* exchange_rate, float* t1_bound, float* pool_b_fraction, float* pool_b_exchange, float* t1_pool_b, float* t2_pool_b, float* pool_b_shift, float* duration, std::int32_t* kind, float* flip, float* phase, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* tangent_t1, float* tangent_t2, float* tangent_m0, float* tangent_b1, float* tangent_b1_phase, float* tangent_b0, float* tangent_inversion_efficiency, float* tangent_diffusion, float* tangent_velocity, float* tangent_bound_fraction, float* tangent_exchange_rate, float* tangent_t1_bound, float* tangent_pool_b_fraction, float* tangent_pool_b_exchange, float* tangent_t1_pool_b, float* tangent_t2_pool_b, float* tangent_pool_b_shift, float* tangent_duration, float* tangent_flip, float* tangent_phase, float* saturation, float* rf_frequency, float* profile, std::int32_t* profile_index, float* lineshape, float* pairs, std::int32_t* pair_index, float* pair_direction, std::int32_t* duration_row, float* pool_table, float* output_real, float* output_imag, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, float flow_scale, float washout_scale, float profile_step, float lineshape_step, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shim_rows, bsk::index_t shimmed, bsk::index_t locations, bsk::index_t profiled, bsk::index_t profile_bins, bsk::index_t dynamic, bsk::index_t broadened, bsk::index_t lineshape_bins, bsk::index_t pools, bsk::index_t narrow, bsk::index_t tabulated, bsk::index_t off_axis, bsk::index_t moving, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 2> atom_b0{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_b1_phase{};
    bsk::V<float, 2> atom_bound{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_exchange{};
    bsk::V<float, 2> atom_flow{};
    bsk::V<float, 2> atom_inversion{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 2> atom_r1_bound{};
    bsk::V<float, 2> atom_r1_semisolid{};
    bsk::V<float, 2> atom_r2_bound{};
    bsk::V<float, 2> atom_semisolid{};
    bsk::V<float, 2> atom_semisolid_exchange{};
    bsk::V<float, 2> atom_shift{};
    bsk::V<float, 2> atom_washout{};
    bsk::V<float, 3> b_dmi{};
    bsk::V<float, 3> b_dmr{};
    bsk::V<float, 3> b_dpi{};
    bsk::V<float, 3> b_dpr{};
    bsk::V<float, 3> b_dzi{};
    bsk::V<float, 3> b_dzr{};
    bsk::V<float, 3> b_mi{};
    bsk::V<float, 3> b_mr{};
    bsk::V<float, 3> b_pi{};
    bsk::V<float, 3> b_pr{};
    bsk::V<float, 3> b_zi{};
    bsk::V<float, 3> b_zr{};
    bsk::V<float, 3> bi{};
    bsk::V<float, 3> bmi{};
    bsk::V<float, 3> bmr{};
    bsk::V<float, 3> bpi{};
    bsk::V<float, 3> bpr{};
    bsk::V<float, 3> br{};
    bsk::V<float, 3> ci{};
    bsk::V<float, 3> cr{};
    bsk::V<float, 2> d_bound{};
    bsk::V<float, 2> d_damping{};
    bsk::V<float, 2> d_exchange{};
    bsk::V<float, 2> d_flow{};
    bsk::V<float, 3> d_free_i{};
    bsk::V<float, 3> d_free_r{};
    bsk::V<float, 2> d_grow_free{};
    bsk::V<float, 2> d_grow_pool_b{};
    bsk::V<float, 2> d_grow_semisolid{};
    bsk::V<float, 3> d_held_i{};
    bsk::V<float, 3> d_held_r{};
    bsk::V<float, 2> d_r1_bound{};
    bsk::V<float, 2> d_r1_semisolid{};
    bsk::V<float, 2> d_r2_bound{};
    bsk::V<float, 2> d_semisolid{};
    bsk::V<float, 2> d_semisolid_exchange{};
    bsk::V<float, 2> d_shift{};
    bsk::V<float, 2> d_t11{};
    bsk::V<float, 2> d_t12{};
    bsk::V<float, 2> d_t13{};
    bsk::V<float, 2> d_t21{};
    bsk::V<float, 2> d_t22{};
    bsk::V<float, 2> d_t23{};
    bsk::V<float, 2> d_t31{};
    bsk::V<float, 2> d_t32{};
    bsk::V<float, 2> d_t33{};
    bsk::V<float, 2> d_washout{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 2> db0{};
    bsk::V<float, 2> db1{};
    bsk::V<float, 2> db1_phase{};
    bsk::V<float, 3> dbi{};
    bsk::V<float, 3> dbmi{};
    bsk::V<float, 3> dbmr{};
    bsk::V<float, 3> dbpi{};
    bsk::V<float, 3> dbpr{};
    bsk::V<float, 3> dbr{};
    bsk::V<float, 3> dci{};
    bsk::V<float, 3> dcr{};
    bsk::V<float, 3> ddamp_t{};
    bsk::V<float, 3> ddamp_z{};
    bsk::V<float, 3> de1{};
    bsk::V<float, 3> de2{};
    bsk::V<float, 3> dfmi{};
    bsk::V<float, 3> dfmr{};
    bsk::V<float, 3> dfpi{};
    bsk::V<float, 3> dfpr{};
    bsk::V<float, 2> dinversion{};
    bsk::V<float, 2> dm0{};
    bsk::V<float, 3> doff_cos{};
    bsk::V<float, 3> doff_sin{};
    bsk::V<float, 2> dot_ai{};
    bsk::V<float, 2> dot_ar{};
    bsk::V<float, 2> dot_bi{};
    bsk::V<float, 2> dot_br{};
    bsk::V<float, 3> dread_i{};
    bsk::V<float, 3> dread_r{};
    bsk::V<float, 3> dspun_hi{};
    bsk::V<float, 3> dspun_hr{};
    bsk::V<float, 3> dturn_cos{};
    bsk::V<float, 3> dturn_sin{};
    bsk::V<float, 3> dturn_t{};
    bsk::V<float, 3> dturn_z{};
    bsk::V<float, 2> dwout{};
    bsk::V<float, 3> dzi{};
    bsk::V<float, 3> dzr{};
    bsk::V<float, 3> e1{};
    bsk::V<float, 3> e2{};
    bsk::V<float, 3> fmi{};
    bsk::V<float, 3> fmr{};
    bsk::V<float, 3> fpi{};
    bsk::V<float, 3> fpr{};
    bsk::V<float, 3> free_i{};
    bsk::V<float, 3> free_r{};
    bsk::V<float, 2> grow_free{};
    bsk::V<float, 2> grow_pool_b{};
    bsk::V<float, 2> grow_semisolid{};
    bsk::V<float, 3> held_dmi{};
    bsk::V<float, 3> held_dmr{};
    bsk::V<float, 3> held_dpi{};
    bsk::V<float, 3> held_dpr{};
    bsk::V<float, 3> held_dzi{};
    bsk::V<float, 3> held_dzr{};
    bsk::V<float, 3> held_i{};
    bsk::V<float, 3> held_mi{};
    bsk::V<float, 3> held_mr{};
    bsk::V<float, 3> held_pi{};
    bsk::V<float, 3> held_pr{};
    bsk::V<float, 3> held_r{};
    bsk::V<float, 2> held_t1{};
    bsk::V<float, 3> held_zi{};
    bsk::V<float, 3> held_zr{};
    bsk::V<float, 3> off_cos{};
    bsk::V<float, 3> off_sin{};
    bsk::V<float, 2> pair_ai{};
    bsk::V<float, 2> pair_ar{};
    bsk::V<float, 2> pair_bi{};
    bsk::V<float, 2> pair_br{};
    bsk::V<float, 3> read_i{};
    bsk::V<float, 3> read_r{};
    bsk::V<float, 3> rotated_dmi{};
    bsk::V<float, 3> rotated_dmr{};
    bsk::V<float, 3> rotated_dpi{};
    bsk::V<float, 3> rotated_dpr{};
    bsk::V<float, 3> rotated_dzi{};
    bsk::V<float, 3> rotated_dzr{};
    bsk::V<float, 3> rotated_mi{};
    bsk::V<float, 3> rotated_mr{};
    bsk::V<float, 3> rotated_pi{};
    bsk::V<float, 3> rotated_pr{};
    bsk::V<float, 3> rotated_zi{};
    bsk::V<float, 3> rotated_zr{};
    bsk::V<float, 3> s_bmi{};
    bsk::V<float, 3> s_bmr{};
    bsk::V<float, 3> s_bpi{};
    bsk::V<float, 3> s_bpr{};
    bsk::V<float, 3> s_dbmi{};
    bsk::V<float, 3> s_dbmr{};
    bsk::V<float, 3> s_dbpi{};
    bsk::V<float, 3> s_dbpr{};
    bsk::V<float, 3> shaped_dmi{};
    bsk::V<float, 3> shaped_dmr{};
    bsk::V<float, 3> shaped_dpi{};
    bsk::V<float, 3> shaped_dpr{};
    bsk::V<float, 3> shaped_dzi{};
    bsk::V<float, 3> shaped_dzr{};
    bsk::V<float, 3> shaped_mi{};
    bsk::V<float, 3> shaped_mr{};
    bsk::V<float, 3> shaped_pi{};
    bsk::V<float, 3> shaped_pr{};
    bsk::V<float, 3> shaped_zi{};
    bsk::V<float, 3> shaped_zr{};
    bsk::V<float, 3> shifted_dmi{};
    bsk::V<float, 3> shifted_dmr{};
    bsk::V<float, 3> shifted_dpi{};
    bsk::V<float, 3> shifted_dpr{};
    bsk::V<float, 3> shifted_mi{};
    bsk::V<float, 3> shifted_mr{};
    bsk::V<float, 3> shifted_pi{};
    bsk::V<float, 3> shifted_pr{};
    bsk::V<float, 3> signal_imag{};
    bsk::V<float, 3> signal_real{};
    bsk::V<float, 3> spun_hi{};
    bsk::V<float, 3> spun_hr{};
    bsk::V<float, 2> t11{};
    bsk::V<float, 2> t12{};
    bsk::V<float, 2> t13{};
    bsk::V<float, 2> t21{};
    bsk::V<float, 2> t22{};
    bsk::V<float, 2> t23{};
    bsk::V<float, 2> t31{};
    bsk::V<float, 2> t32{};
    bsk::V<float, 2> t33{};
    bsk::V<float, 3> turn_cos{};
    bsk::V<float, 3> turn_sin{};
    bsk::V<float, 3> turn_t{};
    bsk::V<float, 3> turn_z{};
    bsk::V<float, 2> wout{};
    bsk::V<float, 3> zi{};
    bsk::V<float, 3> zr{};
    auto problem = ((bsk::program_id(0) * problems) + bsk::arange_y());
    auto state = bsk::arange_x();
    auto active_atom = (problem < (train_count * atom_count));
    // A partial block carries lanes with no problem behind them, and they must
    // take no part in a reduction or a store.
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    // Voxels are spread over the slice voxel-major, so a voxel's place along
    // the slice is its index modulo the profile's width. One pulse shape holds
    // that many consecutive rows, and the event says which shape it drives.
    auto location = bsk::mod(atom, locations);
    auto empty = bsk::full<float, 3>(0);
    fpr = empty;
    fpi = empty;
    fmr = empty;
    fmi = empty;
    // Equilibrium is split between the pools, so a direction along the bound
    // fraction moves magnetization from one to the other before a single event
    // has run.
    atom_bound = 0.0f;
    d_bound = 0.0f;
    atom_exchange = 0.0f;
    d_exchange = 0.0f;
    atom_r1_bound = 0.0f;
    d_r1_bound = 0.0f;
    atom_r2_bound = 0.0f;
    d_r2_bound = 0.0f;
    atom_shift = 0.0f;
    d_shift = 0.0f;
    atom_semisolid = 0.0f;
    d_semisolid = 0.0f;
    atom_semisolid_exchange = 0.0f;
    d_semisolid_exchange = 0.0f;
    atom_r1_semisolid = 0.0f;
    d_r1_semisolid = 0.0f;
    if (bsk::truth((pools == 1))) {
        atom_bound = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        d_bound = bsk::ld((tangent_bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((exchange_rate + scalar_atom), active_atom, 0.0f);
        d_exchange = bsk::ld((tangent_exchange_rate + scalar_atom), active_atom, 0.0f);
        held_t1 = bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f);
        atom_r1_bound = bsk::truediv(1000.0f, held_t1);
        d_r1_bound = bsk::truediv((-1000.0f * bsk::ld((tangent_t1_bound + scalar_atom), active_atom, 0.0f)), (held_t1 * held_t1));
    }
    if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
        atom_bound = bsk::ld((pool_b_fraction + scalar_atom), active_atom, 0.0f);
        d_bound = bsk::ld((tangent_pool_b_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((pool_b_exchange + scalar_atom), active_atom, 0.0f);
        d_exchange = bsk::ld((tangent_pool_b_exchange + scalar_atom), active_atom, 0.0f);
        held_t1 = bsk::ld((t1_pool_b + scalar_atom), active_atom, 1.0f);
        atom_r1_bound = bsk::truediv(1000.0f, held_t1);
        d_r1_bound = bsk::truediv((-1000.0f * bsk::ld((tangent_t1_pool_b + scalar_atom), active_atom, 0.0f)), (held_t1 * held_t1));
        auto held_t2 = bsk::ld((t2_pool_b + scalar_atom), active_atom, 1.0f);
        atom_r2_bound = bsk::truediv(1000.0f, held_t2);
        d_r2_bound = bsk::truediv((-1000.0f * bsk::ld((tangent_t2_pool_b + scalar_atom), active_atom, 0.0f)), (held_t2 * held_t2));
        atom_shift = bsk::ld((pool_b_shift + scalar_atom), active_atom, 0.0f);
        d_shift = bsk::ld((tangent_pool_b_shift + scalar_atom), active_atom, 0.0f);
    }
    if (bsk::truth((pools == 3))) {
        atom_semisolid = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        d_semisolid = bsk::ld((tangent_bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_semisolid_exchange = bsk::ld((exchange_rate + scalar_atom), active_atom, 0.0f);
        d_semisolid_exchange = bsk::ld((tangent_exchange_rate + scalar_atom), active_atom, 0.0f);
        auto held_semisolid = bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f);
        atom_r1_semisolid = bsk::truediv(1000.0f, held_semisolid);
        d_r1_semisolid = bsk::truediv((-1000.0f * bsk::ld((tangent_t1_bound + scalar_atom), active_atom, 0.0f)), (held_semisolid * held_semisolid));
    }
    auto atom_free = ((1.0f - atom_bound) - atom_semisolid);
    auto d_free = ((-d_bound) - d_semisolid);
    zr = (empty + bsk::where((state == 0), atom_free, 0.0f));
    zi = empty;
    br = (empty + bsk::where((state == 0), (atom_bound + 0.0f), 0.0f));
    bi = empty;
    cr = (empty + bsk::where((state == 0), (atom_semisolid + 0.0f), 0.0f));
    ci = empty;
    dcr = (empty + bsk::where((state == 0), (d_semisolid + 0.0f), 0.0f));
    dci = empty;
    dfpr = empty;
    dfpi = empty;
    dfmr = empty;
    dfmi = empty;
    dzr = (empty + bsk::where((state == 0), ((-d_bound) - d_semisolid), 0.0f));
    dzi = empty;
    dbr = (empty + bsk::where((state == 0), (d_bound + 0.0f), 0.0f));
    dbi = empty;
    bpr = empty;
    bpi = empty;
    bmr = empty;
    bmi = empty;
    dbpr = empty;
    dbpi = empty;
    dbmr = empty;
    dbmi = empty;
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_b1_phase = 0.0f;
    atom_b0 = 0.0f;
    if (bsk::truth(off_axis)) {
        atom_b1_phase = bsk::ld((b1_phase + scalar_atom), active_atom, 0.0f);
        atom_b0 = bsk::ld((b0 + scalar_atom), active_atom, 0.0f);
    }
    atom_inversion = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inversion = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    atom_damping = 0.0f;
    d_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
        d_damping = bsk::ld((tangent_diffusion + scalar_atom), active_atom, 0.0f);
    }
    atom_flow = 0.0f;
    d_flow = 0.0f;
    atom_washout = 0.0f;
    d_washout = 0.0f;
    if (bsk::truth(moving)) {
        auto atom_velocity = bsk::ld((velocity + scalar_atom), active_atom, 0.0f);
        auto d_velocity = bsk::ld((tangent_velocity + scalar_atom), active_atom, 0.0f);
        atom_flow = (atom_velocity * flow_scale);
        d_flow = (d_velocity * flow_scale);
        // |v| has no derivative at the origin, so a still voxel contributes
        // none.
        auto direction = (bsk::cast<float>((atom_velocity > 0.0f)) - bsk::cast<float>((atom_velocity < 0.0f)));
        atom_washout = (bsk::abs(atom_velocity) * washout_scale);
        d_washout = ((direction * d_velocity) * washout_scale);
    }
    auto order = bsk::cast<float>(state);
    auto dt1 = bsk::ld((tangent_t1 + atom), active_atom, 0.0f);
    auto dt2 = bsk::ld((tangent_t2 + atom), active_atom, 0.0f);
    dm0 = 0.0f;
    if (bsk::truth(density)) {
        dm0 = bsk::ld((tangent_m0 + scalar_atom), active_atom, 0.0f);
    }
    db1 = 0.0f;
    if (bsk::truth(transmit)) {
        db1 = bsk::ld((tangent_b1 + scalar_atom), active_atom, 0.0f);
    }
    db1_phase = 0.0f;
    db0 = 0.0f;
    if (bsk::truth(off_axis)) {
        db1_phase = bsk::ld((tangent_b1_phase + scalar_atom), active_atom, 0.0f);
        db0 = bsk::ld((tangent_b0 + scalar_atom), active_atom, 0.0f);
    }
    dinversion = 0.0f;
    if (bsk::truth(inverting)) {
        dinversion = bsk::ld((tangent_inversion_efficiency + scalar_atom), active_atom, 0.0f);
    }
    auto event_base = (train * event_count);
    for (bsk::index_t event = 0; event < event_count; event += 1) {
        auto event_dt = _event_value(duration, event_base, event, active_atom, single_train);
        auto ddt = _event_value(tangent_duration, event_base, event, active_atom, single_train);
        auto r1 = bsk::truediv(1000.0f, atom_t1);
        auto r2 = bsk::truediv(1000.0f, atom_t2);
        wout = 1.0f;
        dwout = 0.0f;
        if (bsk::truth(moving)) {
            auto t0_ = _washout_jvp(atom_washout, d_washout, event_dt, ddt);
            wout = bsk::get<0>(t0_);
            dwout = bsk::get<1>(t0_);
        }
        auto dry1 = bsk::exp(((-r1) * event_dt));
        auto dry2 = bsk::exp(((-r2) * event_dt));
        e1 = (dry1 * wout);
        e2 = (dry2 * wout);
        de1 = ((e1 * (bsk::truediv(((1000.0f * event_dt) * dt1), (atom_t1 * atom_t1)) - (r1 * ddt))) + (dry1 * dwout));
        de2 = ((e2 * (bsk::truediv(((1000.0f * event_dt) * dt2), (atom_t2 * atom_t2)) - (r2 * ddt))) + (dry2 * dwout));
        damp_z = 1.0f;
        ddamp_z = 0.0f;
        damp_t = 1.0f;
        ddamp_t = 0.0f;
        if (bsk::truth(diffusing)) {
            auto t1_ = _damping_jvp(atom_damping, d_damping, event_dt, ddt, order);
            damp_z = bsk::get<0>(t1_);
            ddamp_z = bsk::get<1>(t1_);
            damp_t = bsk::get<2>(t1_);
            ddamp_t = bsk::get<3>(t1_);
        }
        // Order zero is undamped, so the recovery term keeps the bare factor.
        auto t2_ = bsk::make_tup((1.0f - e1), (-de1));
        auto recovery = bsk::get<0>(t2_);
        auto drecovery = bsk::get<1>(t2_);
        de1 = ((de1 * damp_z) + (e1 * ddamp_z));
        e1 = (e1 * damp_z);
        de2 = ((de2 * damp_t) + (e2 * ddamp_t));
        e2 = (e2 * damp_t);
        turn_z = 0.0f;
        turn_t = 0.0f;
        dturn_z = 0.0f;
        dturn_t = 0.0f;
        if (bsk::truth(moving)) {
            auto t3_ = _flow(atom_flow, event_dt, order);
            turn_z = bsk::get<0>(t3_);
            turn_t = bsk::get<1>(t3_);
            auto d_turn = ((d_flow * event_dt) + (atom_flow * ddt));
            dturn_z = ((-order) * d_turn);
            dturn_t = ((-(order + 0.5f)) * d_turn);
        }
        off_cos = 1.0f;
        off_sin = 0.0f;
        doff_cos = 0.0f;
        doff_sin = 0.0f;
        if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
            // Flow winds the transverse states through the same rotation
            // off-resonance does, so the two phases add before either is taken.
            auto off_phase = (((-6.283185307179586f * atom_b0) * event_dt) + turn_t);
            auto doff_phase = ((-6.283185307179586f * ((db0 * event_dt) + (atom_b0 * ddt))) + dturn_t);
            off_cos = bsk::cos(off_phase);
            off_sin = bsk::sin(off_phase);
            doff_cos = ((-off_sin) * doff_phase);
            doff_sin = (off_cos * doff_phase);
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // Both pools take the same off-resonance and per-order damping;
            // what separates them is the chemical shift, which the exchange
            // operator already carries.
            auto t4_ = _two_pool_transverse_step_jvp(r2, bsk::truediv((-1000.0f * dt2), (atom_t2 * atom_t2)), atom_r2_bound, d_r2_bound, atom_exchange, d_exchange, atom_bound, d_bound, atom_free, d_free, atom_shift, d_shift, event_dt, ddt, wout, dwout);
            auto x11r = bsk::get<0>(t4_);
            auto x11i = bsk::get<1>(t4_);
            auto x12r = bsk::get<2>(t4_);
            auto x12i = bsk::get<3>(t4_);
            auto x21r = bsk::get<4>(t4_);
            auto x21i = bsk::get<5>(t4_);
            auto x22r = bsk::get<6>(t4_);
            auto x22i = bsk::get<7>(t4_);
            auto d11r = bsk::get<8>(t4_);
            auto d11i = bsk::get<9>(t4_);
            auto d12r = bsk::get<10>(t4_);
            auto d12i = bsk::get<11>(t4_);
            auto d21r = bsk::get<12>(t4_);
            auto d21i = bsk::get<13>(t4_);
            auto d22r = bsk::get<14>(t4_);
            auto d22i = bsk::get<15>(t4_);
            auto mix_pr = ((((x11r * fpr) - (x11i * fpi)) + (x12r * bpr)) - (x12i * bpi));
            auto mix_pi = ((((x11r * fpi) + (x11i * fpr)) + (x12r * bpi)) + (x12i * bpr));
            auto dmix_pr = ((((((((d11r * fpr) + (x11r * dfpr)) - (d11i * fpi)) - (x11i * dfpi)) + (d12r * bpr)) + (x12r * dbpr)) - (d12i * bpi)) - (x12i * dbpi));
            auto dmix_pi = ((((((((d11r * fpi) + (x11r * dfpi)) + (d11i * fpr)) + (x11i * dfpr)) + (d12r * bpi)) + (x12r * dbpi)) + (d12i * bpr)) + (x12i * dbpr));
            auto mix_br = ((((x21r * fpr) - (x21i * fpi)) + (x22r * bpr)) - (x22i * bpi));
            auto mix_bi = ((((x21r * fpi) + (x21i * fpr)) + (x22r * bpi)) + (x22i * bpr));
            auto dmix_br = ((((((((d21r * fpr) + (x21r * dfpr)) - (d21i * fpi)) - (x21i * dfpi)) + (d22r * bpr)) + (x22r * dbpr)) - (d22i * bpi)) - (x22i * dbpi));
            auto dmix_bi = ((((((((d21r * fpi) + (x21r * dfpi)) + (d21i * fpr)) + (x21i * dfpr)) + (d22r * bpi)) + (x22r * dbpi)) + (d22i * bpr)) + (x22i * dbpr));
            // ``F-`` follows the conjugate of the operator entry by entry.
            auto mix_mr = ((((x11r * fmr) + (x11i * fmi)) + (x12r * bmr)) + (x12i * bmi));
            auto mix_mi = ((((x11r * fmi) - (x11i * fmr)) + (x12r * bmi)) - (x12i * bmr));
            auto dmix_mr = ((((((((d11r * fmr) + (x11r * dfmr)) + (d11i * fmi)) + (x11i * dfmi)) + (d12r * bmr)) + (x12r * dbmr)) + (d12i * bmi)) + (x12i * dbmi));
            auto dmix_mi = ((((((((d11r * fmi) + (x11r * dfmi)) - (d11i * fmr)) - (x11i * dfmr)) + (d12r * bmi)) + (x12r * dbmi)) - (d12i * bmr)) - (x12i * dbmr));
            auto mix_nr = ((((x21r * fmr) + (x21i * fmi)) + (x22r * bmr)) + (x22i * bmi));
            auto mix_ni = ((((x21r * fmi) - (x21i * fmr)) + (x22r * bmi)) - (x22i * bmr));
            auto dmix_nr = ((((((((d21r * fmr) + (x21r * dfmr)) + (d21i * fmi)) + (x21i * dfmi)) + (d22r * bmr)) + (x22r * dbmr)) + (d22i * bmi)) + (x22i * dbmi));
            auto dmix_ni = ((((((((d21r * fmi) + (x21r * dfmi)) - (d21i * fmr)) - (x21i * dfmr)) + (d22r * bmi)) + (x22r * dbmi)) - (d22i * bmr)) - (x22i * dbmr));
            // The damping and off-resonance both pools share, applied after.
            auto carry_r = (damp_t * off_cos);
            auto carry_i = (damp_t * off_sin);
            auto dcarry_r = ((ddamp_t * off_cos) + (damp_t * doff_cos));
            auto dcarry_i = ((ddamp_t * off_sin) + (damp_t * doff_sin));
            fpr = ((mix_pr * carry_r) - (mix_pi * carry_i));
            fpi = ((mix_pr * carry_i) + (mix_pi * carry_r));
            dfpr = ((((dmix_pr * carry_r) + (mix_pr * dcarry_r)) - (dmix_pi * carry_i)) - (mix_pi * dcarry_i));
            dfpi = ((((dmix_pr * carry_i) + (mix_pr * dcarry_i)) + (dmix_pi * carry_r)) + (mix_pi * dcarry_r));
            bpr = ((mix_br * carry_r) - (mix_bi * carry_i));
            bpi = ((mix_br * carry_i) + (mix_bi * carry_r));
            dbpr = ((((dmix_br * carry_r) + (mix_br * dcarry_r)) - (dmix_bi * carry_i)) - (mix_bi * dcarry_i));
            dbpi = ((((dmix_br * carry_i) + (mix_br * dcarry_i)) + (dmix_bi * carry_r)) + (mix_bi * dcarry_r));
            fmr = ((mix_mr * carry_r) + (mix_mi * carry_i));
            fmi = (((-mix_mr) * carry_i) + (mix_mi * carry_r));
            dfmr = ((((dmix_mr * carry_r) + (mix_mr * dcarry_r)) + (dmix_mi * carry_i)) + (mix_mi * dcarry_i));
            dfmi = (((((-dmix_mr) * carry_i) - (mix_mr * dcarry_i)) + (dmix_mi * carry_r)) + (mix_mi * dcarry_r));
            bmr = ((mix_nr * carry_r) + (mix_ni * carry_i));
            bmi = (((-mix_nr) * carry_i) + (mix_ni * carry_r));
            dbmr = ((((dmix_nr * carry_r) + (mix_nr * dcarry_r)) + (dmix_ni * carry_i)) + (mix_ni * dcarry_i));
            dbmi = (((((-dmix_nr) * carry_i) - (mix_nr * dcarry_i)) + (dmix_ni * carry_r)) + (mix_ni * dcarry_r));
        } else {
            auto old_fpr = fpr;
            auto old_fpi = fpi;
            auto old_dfpr = dfpr;
            auto old_dfpi = dfpi;
            fpr = (e2 * ((old_fpr * off_cos) - (old_fpi * off_sin)));
            fpi = (e2 * ((old_fpr * off_sin) + (old_fpi * off_cos)));
            dfpr = (de2 * ((old_fpr * off_cos) - (old_fpi * off_sin)));
            dfpr = (dfpr + (e2 * ((((old_dfpr * off_cos) + (old_fpr * doff_cos)) - (old_dfpi * off_sin)) - (old_fpi * doff_sin))));
            dfpi = (de2 * ((old_fpr * off_sin) + (old_fpi * off_cos)));
            dfpi = (dfpi + (e2 * ((((old_dfpr * off_sin) + (old_fpr * doff_sin)) + (old_dfpi * off_cos)) + (old_fpi * doff_cos))));
            auto old_fmr = fmr;
            auto old_fmi = fmi;
            auto old_dfmr = dfmr;
            auto old_dfmi = dfmi;
            fmr = (e2 * ((old_fmr * off_cos) + (old_fmi * off_sin)));
            fmi = (e2 * (((-old_fmr) * off_sin) + (old_fmi * off_cos)));
            dfmr = (de2 * ((old_fmr * off_cos) + (old_fmi * off_sin)));
            dfmr = (dfmr + (e2 * ((((old_dfmr * off_cos) + (old_fmr * doff_cos)) + (old_dfmi * off_sin)) + (old_fmi * doff_sin))));
            dfmi = (de2 * (((-old_fmr) * off_sin) + (old_fmi * off_cos)));
            dfmi = (dfmi + (e2 * (((((-old_dfmr) * off_sin) - (old_fmr * doff_sin)) + (old_dfmi * off_cos)) + (old_fmi * doff_cos))));
        }
        // The longitudinal states carry a phase of their own, which nothing
        // else in the state machine gives them.
        turn_cos = 1.0f;
        turn_sin = 0.0f;
        dturn_cos = 0.0f;
        dturn_sin = 0.0f;
        if (bsk::truth(moving)) {
            turn_cos = bsk::cos(turn_z);
            turn_sin = bsk::sin(turn_z);
            dturn_cos = ((-turn_sin) * dturn_z);
            dturn_sin = (turn_cos * dturn_z);
        }
        auto old_zr = zr;
        auto old_zi = zi;
        auto old_dzr = dzr;
        auto old_dzi = dzi;
        auto spun_zr = ((old_zr * turn_cos) - (old_zi * turn_sin));
        auto spun_zi = ((old_zr * turn_sin) + (old_zi * turn_cos));
        auto dspun_zr = ((((old_dzr * turn_cos) + (old_zr * dturn_cos)) - (old_dzi * turn_sin)) - (old_zi * dturn_sin));
        auto dspun_zi = ((((old_dzr * turn_sin) + (old_zr * dturn_sin)) + (old_dzi * turn_cos)) + (old_zi * dturn_cos));
        if (bsk::truth((pools == 3))) {
            // Three pools mix through a 3x3 formed in double, tangent and all:
            // a direction through an operator this ill-conditioned needs the
            // width as much as the value does.
            if (bsk::truth(tabulated)) {
                auto t5_ = _three_pool_from_table_jvp(pool_table, bsk::ld(((duration_row + event_base) + event), active_atom, 0), atom, atom_count, active_atom, r1, atom_r1_bound, atom_r1_semisolid, atom_exchange, atom_semisolid_exchange, atom_bound, d_bound, atom_semisolid, d_semisolid, ddt, wout, dwout);
                t11 = bsk::get<0>(t5_);
                t12 = bsk::get<1>(t5_);
                t13 = bsk::get<2>(t5_);
                t21 = bsk::get<3>(t5_);
                t22 = bsk::get<4>(t5_);
                t23 = bsk::get<5>(t5_);
                t31 = bsk::get<6>(t5_);
                t32 = bsk::get<7>(t5_);
                t33 = bsk::get<8>(t5_);
                grow_free = bsk::get<9>(t5_);
                grow_pool_b = bsk::get<10>(t5_);
                grow_semisolid = bsk::get<11>(t5_);
                d_t11 = bsk::get<12>(t5_);
                d_t12 = bsk::get<13>(t5_);
                d_t13 = bsk::get<14>(t5_);
                d_t21 = bsk::get<15>(t5_);
                d_t22 = bsk::get<16>(t5_);
                d_t23 = bsk::get<17>(t5_);
                d_t31 = bsk::get<18>(t5_);
                d_t32 = bsk::get<19>(t5_);
                d_t33 = bsk::get<20>(t5_);
                d_grow_free = bsk::get<21>(t5_);
                d_grow_pool_b = bsk::get<22>(t5_);
                d_grow_semisolid = bsk::get<23>(t5_);
            } else {
                auto t6_ = _three_pool_step_jvp(r1, bsk::truediv((-1000.0f * dt1), (atom_t1 * atom_t1)), atom_r1_bound, d_r1_bound, atom_r1_semisolid, d_r1_semisolid, atom_exchange, d_exchange, atom_semisolid_exchange, d_semisolid_exchange, atom_bound, d_bound, atom_semisolid, d_semisolid, event_dt, ddt, wout, dwout, narrow);
                t11 = bsk::get<0>(t6_);
                t12 = bsk::get<1>(t6_);
                t13 = bsk::get<2>(t6_);
                t21 = bsk::get<3>(t6_);
                t22 = bsk::get<4>(t6_);
                t23 = bsk::get<5>(t6_);
                t31 = bsk::get<6>(t6_);
                t32 = bsk::get<7>(t6_);
                t33 = bsk::get<8>(t6_);
                grow_free = bsk::get<9>(t6_);
                grow_pool_b = bsk::get<10>(t6_);
                grow_semisolid = bsk::get<11>(t6_);
                d_t11 = bsk::get<12>(t6_);
                d_t12 = bsk::get<13>(t6_);
                d_t13 = bsk::get<14>(t6_);
                d_t21 = bsk::get<15>(t6_);
                d_t22 = bsk::get<16>(t6_);
                d_t23 = bsk::get<17>(t6_);
                d_t31 = bsk::get<18>(t6_);
                d_t32 = bsk::get<19>(t6_);
                d_t33 = bsk::get<20>(t6_);
                d_grow_free = bsk::get<21>(t6_);
                d_grow_pool_b = bsk::get<22>(t6_);
                d_grow_semisolid = bsk::get<23>(t6_);
            }
            spun_hr = ((br * turn_cos) - (bi * turn_sin));
            spun_hi = ((br * turn_sin) + (bi * turn_cos));
            dspun_hr = ((((dbr * turn_cos) + (br * dturn_cos)) - (dbi * turn_sin)) - (bi * dturn_sin));
            dspun_hi = ((((dbr * turn_sin) + (br * dturn_sin)) + (dbi * turn_cos)) + (bi * dturn_cos));
            auto spun_cr = ((cr * turn_cos) - (ci * turn_sin));
            auto spun_ci = ((cr * turn_sin) + (ci * turn_cos));
            auto dspun_cr = ((((dcr * turn_cos) + (cr * dturn_cos)) - (dci * turn_sin)) - (ci * dturn_sin));
            auto dspun_ci = ((((dcr * turn_sin) + (cr * dturn_sin)) + (dci * turn_cos)) + (ci * dturn_cos));
            free_r = (((t11 * spun_zr) + (t12 * spun_hr)) + (t13 * spun_cr));
            free_i = (((t11 * spun_zi) + (t12 * spun_hi)) + (t13 * spun_ci));
            held_r = (((t21 * spun_zr) + (t22 * spun_hr)) + (t23 * spun_cr));
            held_i = (((t21 * spun_zi) + (t22 * spun_hi)) + (t23 * spun_ci));
            auto stuck_r = (((t31 * spun_zr) + (t32 * spun_hr)) + (t33 * spun_cr));
            auto stuck_i = (((t31 * spun_zi) + (t32 * spun_hi)) + (t33 * spun_ci));
            d_free_r = ((((((d_t11 * spun_zr) + (t11 * dspun_zr)) + (d_t12 * spun_hr)) + (t12 * dspun_hr)) + (d_t13 * spun_cr)) + (t13 * dspun_cr));
            d_free_i = ((((((d_t11 * spun_zi) + (t11 * dspun_zi)) + (d_t12 * spun_hi)) + (t12 * dspun_hi)) + (d_t13 * spun_ci)) + (t13 * dspun_ci));
            d_held_r = ((((((d_t21 * spun_zr) + (t21 * dspun_zr)) + (d_t22 * spun_hr)) + (t22 * dspun_hr)) + (d_t23 * spun_cr)) + (t23 * dspun_cr));
            d_held_i = ((((((d_t21 * spun_zi) + (t21 * dspun_zi)) + (d_t22 * spun_hi)) + (t22 * dspun_hi)) + (d_t23 * spun_ci)) + (t23 * dspun_ci));
            auto d_stuck_r = ((((((d_t31 * spun_zr) + (t31 * dspun_zr)) + (d_t32 * spun_hr)) + (t32 * dspun_hr)) + (d_t33 * spun_cr)) + (t33 * dspun_cr));
            auto d_stuck_i = ((((((d_t31 * spun_zi) + (t31 * dspun_zi)) + (d_t32 * spun_hi)) + (t32 * dspun_hi)) + (d_t33 * spun_ci)) + (t33 * dspun_ci));
            zr = ((damp_z * free_r) + bsk::where((state == 0), grow_free, 0.0f));
            zi = (damp_z * free_i);
            dzr = (((ddamp_z * free_r) + (damp_z * d_free_r)) + bsk::where((state == 0), d_grow_free, 0.0f));
            dzi = ((ddamp_z * free_i) + (damp_z * d_free_i));
            br = ((damp_z * held_r) + bsk::where((state == 0), grow_pool_b, 0.0f));
            bi = (damp_z * held_i);
            dbr = (((ddamp_z * held_r) + (damp_z * d_held_r)) + bsk::where((state == 0), d_grow_pool_b, 0.0f));
            dbi = ((ddamp_z * held_i) + (damp_z * d_held_i));
            cr = ((damp_z * stuck_r) + bsk::where((state == 0), grow_semisolid, 0.0f));
            ci = (damp_z * stuck_i);
            dcr = (((ddamp_z * stuck_r) + (damp_z * d_stuck_r)) + bsk::where((state == 0), d_grow_semisolid, 0.0f));
            dci = ((ddamp_z * stuck_i) + (damp_z * d_stuck_i));
        } else if (bsk::truth((pools > 0))) {
            // The exchange operator belongs to the interval, not to a dephasing
            // order, so it is formed once and carries its own tangent; the
            // per-order damping multiplies both pools, whose order-n states
            // describe one dephasing configuration.
            auto t7_ = _two_pool_step_jvp(r1, bsk::truediv((-1000.0f * dt1), (atom_t1 * atom_t1)), atom_r1_bound, d_r1_bound, atom_exchange, d_exchange, atom_bound, d_bound, event_dt, ddt, wout, dwout);
            auto e11 = bsk::get<0>(t7_);
            auto e12 = bsk::get<1>(t7_);
            auto e21 = bsk::get<2>(t7_);
            auto e22 = bsk::get<3>(t7_);
            grow_free = bsk::get<4>(t7_);
            auto grow_bound = bsk::get<5>(t7_);
            auto d_e11 = bsk::get<6>(t7_);
            auto d_e12 = bsk::get<7>(t7_);
            auto d_e21 = bsk::get<8>(t7_);
            auto d_e22 = bsk::get<9>(t7_);
            d_grow_free = bsk::get<10>(t7_);
            auto d_grow_bound = bsk::get<11>(t7_);
            auto old_br = br;
            auto old_bi = bi;
            auto old_dbr = dbr;
            auto old_dbi = dbi;
            spun_hr = ((old_br * turn_cos) - (old_bi * turn_sin));
            spun_hi = ((old_br * turn_sin) + (old_bi * turn_cos));
            dspun_hr = ((((old_dbr * turn_cos) + (old_br * dturn_cos)) - (old_dbi * turn_sin)) - (old_bi * dturn_sin));
            dspun_hi = ((((old_dbr * turn_sin) + (old_br * dturn_sin)) + (old_dbi * turn_cos)) + (old_bi * dturn_cos));
            free_r = ((e11 * spun_zr) + (e12 * spun_hr));
            free_i = ((e11 * spun_zi) + (e12 * spun_hi));
            held_r = ((e21 * spun_zr) + (e22 * spun_hr));
            held_i = ((e21 * spun_zi) + (e22 * spun_hi));
            d_free_r = ((((d_e11 * spun_zr) + (e11 * dspun_zr)) + (d_e12 * spun_hr)) + (e12 * dspun_hr));
            d_free_i = ((((d_e11 * spun_zi) + (e11 * dspun_zi)) + (d_e12 * spun_hi)) + (e12 * dspun_hi));
            d_held_r = ((((d_e21 * spun_zr) + (e21 * dspun_zr)) + (d_e22 * spun_hr)) + (e22 * dspun_hr));
            d_held_i = ((((d_e21 * spun_zi) + (e21 * dspun_zi)) + (d_e22 * spun_hi)) + (e22 * dspun_hi));
            zr = ((damp_z * free_r) + bsk::where((state == 0), grow_free, 0.0f));
            zi = (damp_z * free_i);
            dzr = (((ddamp_z * free_r) + (damp_z * d_free_r)) + bsk::where((state == 0), d_grow_free, 0.0f));
            dzi = ((ddamp_z * free_i) + (damp_z * d_free_i));
            br = ((damp_z * held_r) + bsk::where((state == 0), grow_bound, 0.0f));
            bi = (damp_z * held_i);
            dbr = (((ddamp_z * held_r) + (damp_z * d_held_r)) + bsk::where((state == 0), d_grow_bound, 0.0f));
            dbi = ((ddamp_z * held_i) + (damp_z * d_held_i));
        } else {
            dzr = (((dspun_zr * e1) + (spun_zr * de1)) + bsk::where((state == 0), drecovery, 0.0f));
            dzi = ((dspun_zi * e1) + (spun_zi * de1));
            zr = ((spun_zr * e1) + bsk::where((state == 0), recovery, 0.0f));
            zi = (spun_zi * e1);
        }
        auto event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        auto pre_shift = (bsk::band(event_action, 1) != 0);
        auto t8_ = _shift(fpr, fpi, fmr, fmi, state, state_mask, state_count);
        shifted_pr = bsk::get<0>(t8_);
        shifted_pi = bsk::get<1>(t8_);
        shifted_mr = bsk::get<2>(t8_);
        shifted_mi = bsk::get<3>(t8_);
        auto t9_ = _shift(dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count);
        shifted_dpr = bsk::get<0>(t9_);
        shifted_dpi = bsk::get<1>(t9_);
        shifted_dmr = bsk::get<2>(t9_);
        shifted_dmi = bsk::get<3>(t9_);
        fpr = bsk::where(pre_shift, shifted_pr, fpr);
        fpi = bsk::where(pre_shift, shifted_pi, fpi);
        fmr = bsk::where(pre_shift, shifted_mr, fmr);
        fmi = bsk::where(pre_shift, shifted_mi, fmi);
        dfpr = bsk::where(pre_shift, shifted_dpr, dfpr);
        dfpi = bsk::where(pre_shift, shifted_dpi, dfpi);
        dfmr = bsk::where(pre_shift, shifted_dmr, dfmr);
        dfmi = bsk::where(pre_shift, shifted_dmi, dfmi);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t10_ = _shift(bpr, bpi, bmr, bmi, state, state_mask, state_count);
            s_bpr = bsk::get<0>(t10_);
            s_bpi = bsk::get<1>(t10_);
            s_bmr = bsk::get<2>(t10_);
            s_bmi = bsk::get<3>(t10_);
            auto t11_ = _shift(dbpr, dbpi, dbmr, dbmi, state, state_mask, state_count);
            s_dbpr = bsk::get<0>(t11_);
            s_dbpi = bsk::get<1>(t11_);
            s_dbmr = bsk::get<2>(t11_);
            s_dbmi = bsk::get<3>(t11_);
            bpr = bsk::where(pre_shift, s_bpr, bpr);
            bpi = bsk::where(pre_shift, s_bpi, bpi);
            bmr = bsk::where(pre_shift, s_bmr, bmr);
            bmi = bsk::where(pre_shift, s_bmi, bmi);
            dbpr = bsk::where(pre_shift, s_dbpr, dbpr);
            dbpi = bsk::where(pre_shift, s_dbpi, dbpi);
            dbmr = bsk::where(pre_shift, s_dbmr, dbmr);
            dbmi = bsk::where(pre_shift, s_dbmi, dbmi);
        }
        auto event_kind = bsk::ld((kind + event));
        auto is_rf = (event_kind == 1);
        auto is_inversion = (bsk::band(event_action, 4) != 0);
        auto invert = bsk::band(is_rf, is_inversion);
        dzr = bsk::where(invert, (((-dinversion) * zr) - (atom_inversion * dzr)), dzr);
        dzi = bsk::where(invert, (((-dinversion) * zi) - (atom_inversion * dzi)), dzi);
        zr = bsk::where(invert, ((-atom_inversion) * zr), zr);
        zi = bsk::where(invert, ((-atom_inversion) * zi), zi);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // A chemically exchanging pool is free water and turns over like
            // any other; a semisolid one is saturated instead.
            dbr = bsk::where(invert, (((-dinversion) * br) - (atom_inversion * dbr)), dbr);
            dbi = bsk::where(invert, (((-dinversion) * bi) - (atom_inversion * dbi)), dbi);
            br = bsk::where(invert, ((-atom_inversion) * br), br);
            bi = bsk::where(invert, ((-atom_inversion) * bi), bi);
        }
        auto event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        auto event_phase = _event_value(phase, event_base, event, active_atom, single_train);
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse a row of its own.
        if (bsk::truth(shimmed)) {
            auto row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            atom_b1 = 1.0f;
            if (bsk::truth(transmit)) {
                atom_b1 = bsk::ld(((b1 + row) + atom), active_atom, 1.0f);
            }
            db1 = bsk::ld(((tangent_b1 + row) + atom), active_atom, 0.0f);
            if (bsk::truth(off_axis)) {
                atom_b1_phase = bsk::ld(((b1_phase + row) + atom), active_atom, 0.0f);
                db1_phase = bsk::ld(((tangent_b1_phase + row) + atom), active_atom, 0.0f);
            }
        }
        auto alpha = (event_flip * atom_b1);
        auto dalpha = ((_event_value(tangent_flip, event_base, event, active_atom, single_train) * atom_b1) + (event_flip * db1));
        auto phi = (event_phase + atom_b1_phase);
        auto dphi = (_event_value(tangent_phase, event_base, event, active_atom, single_train) + db1_phase);
        if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
            // The semisolid pool absorbs the power the pulse deposits, so it reads
            // the bare flip the transmit field gives the voxel. The offset
            // reaches it through the voxel's own off-resonance, which is where
            // the lineshape's slope enters a forward direction.
            auto offset = (bsk::ld((rf_frequency + event)) - atom_b0);
            auto t12_ = _lineshape_at_slope(lineshape, offset, lineshape_bins, lineshape_step);
            auto shape = bsk::get<0>(t12_);
            auto shape_slope = bsk::get<1>(t12_);
            auto deposited = bsk::ld((saturation + event));
            auto absorbed = bsk::exp((((deposited * alpha) * alpha) * shape));
            auto d_exponent = (deposited * ((((2.0f * alpha) * dalpha) * shape) - (((alpha * alpha) * shape_slope) * db0)));
            auto saturating = bsk::band(is_rf, bsk::bnot(is_inversion));
            if (bsk::truth((pools == 1))) {
                dbr = bsk::where(saturating, (absorbed * (dbr + (br * d_exponent))), dbr);
                dbi = bsk::where(saturating, (absorbed * (dbi + (bi * d_exponent))), dbi);
                br = bsk::where(saturating, (absorbed * br), br);
                bi = bsk::where(saturating, (absorbed * bi), bi);
            } else {
                dcr = bsk::where(saturating, (absorbed * (dcr + (cr * d_exponent))), dcr);
                dci = bsk::where(saturating, (absorbed * (dci + (ci * d_exponent))), dci);
                cr = bsk::where(saturating, (absorbed * cr), cr);
                ci = bsk::where(saturating, (absorbed * ci), ci);
            }
        }
        if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
            if (bsk::truth(dynamic)) {
                // The array was resolved outside the kernel, so a direction
                // along it arrives already carried through the pulse integral.
                auto held = _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, active_atom);
                auto moved = _dynamic_pair_at(pair_direction, pair_index, event_base, event, atom, atom_count, active_atom);
                auto t13_ = held;
                pair_ar = bsk::get<0>(t13_);
                pair_ai = bsk::get<1>(t13_);
                pair_br = bsk::get<2>(t13_);
                pair_bi = bsk::get<3>(t13_);
                auto t14_ = moved;
                dot_ar = bsk::get<0>(t14_);
                dot_ai = bsk::get<1>(t14_);
                dot_br = bsk::get<2>(t14_);
                dot_bi = bsk::get<3>(t14_);
            } else {
                auto read = _profile_pair_slope(profile, _table_row(profile_index, event, location, locations), alpha, profile_bins, profile_step);
                // The flip angle carries the tangent into the table.
                auto t15_ = bsk::make_tup(bsk::get<0>(read), bsk::get<2>(read));
                pair_ar = bsk::get<0>(t15_);
                pair_ai = bsk::get<1>(t15_);
                auto t16_ = bsk::make_tup(bsk::get<4>(read), bsk::get<6>(read));
                pair_br = bsk::get<0>(t16_);
                pair_bi = bsk::get<1>(t16_);
                auto t17_ = bsk::make_tup((bsk::get<1>(read) * dalpha), (bsk::get<3>(read) * dalpha));
                dot_ar = bsk::get<0>(t17_);
                dot_ai = bsk::get<1>(t17_);
                auto t18_ = bsk::make_tup((bsk::get<5>(read) * dalpha), (bsk::get<7>(read) * dalpha));
                dot_br = bsk::get<0>(t18_);
                dot_bi = bsk::get<1>(t18_);
            }
            // The RF phase turns the axis after the pair comes out, and so
            // reaches ``b`` alone.
            auto turn_r = bsk::cos(phi);
            auto turn_i = (-bsk::sin(phi));
            auto spun_br = ((pair_br * turn_r) - (pair_bi * turn_i));
            auto spun_bi = ((pair_br * turn_i) + (pair_bi * turn_r));
            auto slope_br = dot_br;
            auto slope_bi = dot_bi;
            auto t19_ = _rotate_spinor_dual(pair_ar, pair_ai, spun_br, spun_bi, dot_ar, dot_ai, (((slope_br * turn_r) - (slope_bi * turn_i)) + (dphi * spun_bi)), (((slope_br * turn_i) + (slope_bi * turn_r)) - (dphi * spun_br)), fpr, fpi, fmr, fmi, zr, zi, dfpr, dfpi, dfmr, dfmi, dzr, dzi);
            shaped_pr = bsk::get<0>(t19_);
            shaped_pi = bsk::get<1>(t19_);
            shaped_mr = bsk::get<2>(t19_);
            shaped_mi = bsk::get<3>(t19_);
            shaped_zr = bsk::get<4>(t19_);
            shaped_zi = bsk::get<5>(t19_);
            shaped_dpr = bsk::get<6>(t19_);
            shaped_dpi = bsk::get<7>(t19_);
            shaped_dmr = bsk::get<8>(t19_);
            shaped_dmi = bsk::get<9>(t19_);
            shaped_dzr = bsk::get<10>(t19_);
            shaped_dzi = bsk::get<11>(t19_);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                // The same pulse, the same rotation.
                auto t20_ = _rotate_spinor_dual(pair_ar, pair_ai, spun_br, spun_bi, dot_ar, dot_ai, (((slope_br * turn_r) - (slope_bi * turn_i)) + (dphi * spun_bi)), (((slope_br * turn_i) + (slope_bi * turn_r)) - (dphi * spun_br)), bpr, bpi, bmr, bmi, br, bi, dbpr, dbpi, dbmr, dbmi, dbr, dbi);
                held_pr = bsk::get<0>(t20_);
                held_pi = bsk::get<1>(t20_);
                held_mr = bsk::get<2>(t20_);
                held_mi = bsk::get<3>(t20_);
                held_zr = bsk::get<4>(t20_);
                held_zi = bsk::get<5>(t20_);
                held_dpr = bsk::get<6>(t20_);
                held_dpi = bsk::get<7>(t20_);
                held_dmr = bsk::get<8>(t20_);
                held_dmi = bsk::get<9>(t20_);
                held_dzr = bsk::get<10>(t20_);
                held_dzi = bsk::get<11>(t20_);
            }
        }
        auto cosine = bsk::cos(alpha);
        auto sine = bsk::sin(alpha);
        auto dcosine = ((-sine) * dalpha);
        auto dsine = (cosine * dalpha);
        auto cos_phi = bsk::cos(phi);
        auto sin_phi = bsk::sin(phi);
        auto cos_2phi = bsk::cos((2.0f * phi));
        auto sin_2phi = bsk::sin((2.0f * phi));
        auto dcos_phi = ((-sin_phi) * dphi);
        auto dsin_phi = (cos_phi * dphi);
        auto dcos_2phi = ((-2.0f * sin_2phi) * dphi);
        auto dsin_2phi = ((2.0f * cos_2phi) * dphi);
        auto t21_ = _rotate_flip_phase_jvp(cosine, dcosine, sine, dsine, cos_phi, dcos_phi, sin_phi, dsin_phi, cos_2phi, dcos_2phi, sin_2phi, dsin_2phi, fpr, fpi, fmr, fmi, zr, zi, dfpr, dfpi, dfmr, dfmi, dzr, dzi);
        rotated_pr = bsk::get<0>(t21_);
        rotated_pi = bsk::get<1>(t21_);
        rotated_mr = bsk::get<2>(t21_);
        rotated_mi = bsk::get<3>(t21_);
        rotated_zr = bsk::get<4>(t21_);
        rotated_zi = bsk::get<5>(t21_);
        rotated_dpr = bsk::get<6>(t21_);
        rotated_dpi = bsk::get<7>(t21_);
        rotated_dmr = bsk::get<8>(t21_);
        rotated_dmi = bsk::get<9>(t21_);
        rotated_dzr = bsk::get<10>(t21_);
        rotated_dzi = bsk::get<11>(t21_);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t22_ = _rotate_flip_phase_jvp(cosine, dcosine, sine, dsine, cos_phi, dcos_phi, sin_phi, dsin_phi, cos_2phi, dcos_2phi, sin_2phi, dsin_2phi, bpr, bpi, bmr, bmi, br, bi, dbpr, dbpi, dbmr, dbmi, dbr, dbi);
            b_pr = bsk::get<0>(t22_);
            b_pi = bsk::get<1>(t22_);
            b_mr = bsk::get<2>(t22_);
            b_mi = bsk::get<3>(t22_);
            b_zr = bsk::get<4>(t22_);
            b_zi = bsk::get<5>(t22_);
            b_dpr = bsk::get<6>(t22_);
            b_dpi = bsk::get<7>(t22_);
            b_dmr = bsk::get<8>(t22_);
            b_dmi = bsk::get<9>(t22_);
            b_dzr = bsk::get<10>(t22_);
            b_dzi = bsk::get<11>(t22_);
        }
        if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
            rotated_pr = shaped_pr;
            rotated_pi = shaped_pi;
            rotated_mr = shaped_mr;
            rotated_mi = shaped_mi;
            rotated_zr = shaped_zr;
            rotated_zi = shaped_zi;
            rotated_dpr = shaped_dpr;
            rotated_dpi = shaped_dpi;
            rotated_dmr = shaped_dmr;
            rotated_dmi = shaped_dmi;
            rotated_dzr = shaped_dzr;
            rotated_dzi = shaped_dzi;
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                b_pr = held_pr;
                b_pi = held_pi;
                b_mr = held_mr;
                b_mi = held_mi;
                b_zr = held_zr;
                b_zi = held_zi;
                b_dpr = held_dpr;
                b_dpi = held_dpi;
                b_dmr = held_dmr;
                b_dmi = held_dmi;
                b_dzr = held_dzr;
                b_dzi = held_dzi;
            }
        }
        auto rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        fpr = bsk::where(rotate, rotated_pr, fpr);
        fpi = bsk::where(rotate, rotated_pi, fpi);
        fmr = bsk::where(rotate, rotated_mr, fmr);
        fmi = bsk::where(rotate, rotated_mi, fmi);
        zr = bsk::where(rotate, rotated_zr, zr);
        zi = bsk::where(rotate, rotated_zi, zi);
        dfpr = bsk::where(rotate, rotated_dpr, dfpr);
        dfpi = bsk::where(rotate, rotated_dpi, dfpi);
        dfmr = bsk::where(rotate, rotated_dmr, dfmr);
        dfmi = bsk::where(rotate, rotated_dmi, dfmi);
        dzr = bsk::where(rotate, rotated_dzr, dzr);
        dzi = bsk::where(rotate, rotated_dzi, dzi);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            bpr = bsk::where(rotate, b_pr, bpr);
            bpi = bsk::where(rotate, b_pi, bpi);
            bmr = bsk::where(rotate, b_mr, bmr);
            bmi = bsk::where(rotate, b_mi, bmi);
            br = bsk::where(rotate, b_zr, br);
            bi = bsk::where(rotate, b_zi, bi);
            dbpr = bsk::where(rotate, b_dpr, dbpr);
            dbpi = bsk::where(rotate, b_dpi, dbpi);
            dbmr = bsk::where(rotate, b_dmr, dbmr);
            dbmi = bsk::where(rotate, b_dmi, dbmi);
            dbr = bsk::where(rotate, b_dzr, dbr);
            dbi = bsk::where(rotate, b_dzi, dbi);
        }
        auto record = bsk::band((bsk::band(event_action, 32) != 0), (event_kind == 2));
        auto adc_cos = bsk::cos(event_phase);
        auto adc_sin = bsk::sin(event_phase);
        auto dadc_phase = _event_value(tangent_phase, event_base, event, active_atom, single_train);
        auto dadc_cos = ((-adc_sin) * dadc_phase);
        auto dadc_sin = (adc_cos * dadc_phase);
        read_r = fpr;
        read_i = fpi;
        dread_r = dfpr;
        dread_i = dfpi;
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            read_r = (fpr + bpr);
            read_i = (fpi + bpi);
            dread_r = (dfpr + dbpr);
            dread_i = (dfpi + dbpi);
        }
        signal_real = (dm0 * ((read_r * adc_cos) + (read_i * adc_sin)));
        signal_real = (signal_real + (atom_m0 * ((((dread_r * adc_cos) + (read_r * dadc_cos)) + (dread_i * adc_sin)) + (read_i * dadc_sin))));
        signal_imag = (dm0 * ((read_i * adc_cos) - (read_r * adc_sin)));
        signal_imag = (signal_imag + (atom_m0 * ((((dread_i * adc_cos) + (read_i * dadc_cos)) - (dread_r * adc_sin)) - (read_r * dadc_sin))));
        auto out_ = bsk::ld((output_index + event));
        auto output_offset = ((problem * output_count) + out_);
        auto output_mask = bsk::band(bsk::band(bsk::band(active_atom, (state == 0)), record), (out_ >= 0));
        bsk::st(((output_real + output_offset) + state), signal_real, output_mask);
        bsk::st(((output_imag + output_offset) + state), signal_imag, output_mask);
        auto do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        auto t23_ = _shift(fpr, fpi, fmr, fmi, state, state_mask, state_count);
        shifted_pr = bsk::get<0>(t23_);
        shifted_pi = bsk::get<1>(t23_);
        shifted_mr = bsk::get<2>(t23_);
        shifted_mi = bsk::get<3>(t23_);
        auto t24_ = _shift(dfpr, dfpi, dfmr, dfmi, state, state_mask, state_count);
        shifted_dpr = bsk::get<0>(t24_);
        shifted_dpi = bsk::get<1>(t24_);
        shifted_dmr = bsk::get<2>(t24_);
        shifted_dmi = bsk::get<3>(t24_);
        fpr = bsk::where(do_shift, shifted_pr, fpr);
        fpi = bsk::where(do_shift, shifted_pi, fpi);
        fmr = bsk::where(do_shift, shifted_mr, fmr);
        fmi = bsk::where(do_shift, shifted_mi, fmi);
        dfpr = bsk::where(do_shift, shifted_dpr, dfpr);
        dfpi = bsk::where(do_shift, shifted_dpi, dfpi);
        dfmr = bsk::where(do_shift, shifted_dmr, dfmr);
        dfmi = bsk::where(do_shift, shifted_dmi, dfmi);
        auto spoil = (bsk::band(event_action, 8) != 0);
        fpr = bsk::where(spoil, 0.0f, fpr);
        fpi = bsk::where(spoil, 0.0f, fpi);
        fmr = bsk::where(spoil, 0.0f, fmr);
        fmi = bsk::where(spoil, 0.0f, fmi);
        dfpr = bsk::where(spoil, 0.0f, dfpr);
        dfpi = bsk::where(spoil, 0.0f, dfpi);
        dfmr = bsk::where(spoil, 0.0f, dfmr);
        dfmi = bsk::where(spoil, 0.0f, dfmi);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t25_ = _shift(bpr, bpi, bmr, bmi, state, state_mask, state_count);
            s_bpr = bsk::get<0>(t25_);
            s_bpi = bsk::get<1>(t25_);
            s_bmr = bsk::get<2>(t25_);
            s_bmi = bsk::get<3>(t25_);
            auto t26_ = _shift(dbpr, dbpi, dbmr, dbmi, state, state_mask, state_count);
            s_dbpr = bsk::get<0>(t26_);
            s_dbpi = bsk::get<1>(t26_);
            s_dbmr = bsk::get<2>(t26_);
            s_dbmi = bsk::get<3>(t26_);
            bpr = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_bpr, bpr));
            bpi = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_bpi, bpi));
            bmr = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_bmr, bmr));
            bmi = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_bmi, bmi));
            dbpr = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_dbpr, dbpr));
            dbpi = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_dbpi, dbpi));
            dbmr = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_dbmr, dbmr));
            dbmi = bsk::where(spoil, 0.0f, bsk::where(do_shift, s_dbmi, dbmi));
        }
    }
}

BSK_HD void _epg_real_vjp_jvp_kernel(float* t1, float* t2, float* m0, float* b1, float* inversion_efficiency, float* diffusion, float* duration, std::int32_t* kind, float* flip, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* dot_t1, float* dot_t2, float* dot_m0, float* dot_b1, float* dot_inversion_efficiency, float* dot_diffusion, float* dot_duration, float* dot_flip, float* grad_output_imag, float* grad_tissue_value, float* grad_tissue_tangent, float* grad_flip_value, float* grad_flip_tangent, float* grad_duration_value, float* grad_duration_tangent, float* trajectory_value, float* trajectory_tangent, bsk::index_t problem_base, bsk::index_t problem_end, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shim_rows, bsk::index_t shimmed, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 3> adjoint_mt{};
    bsk::V<float, 3> adjoint_mv{};
    bsk::V<float, 3> adjoint_pt{};
    bsk::V<float, 3> adjoint_pv{};
    bsk::V<float, 3> alpha_bar_terms_tangent{};
    bsk::V<float, 3> alpha_bar_terms_value{};
    bsk::V<float, 2> alpha_tangent{};
    bsk::V<float, 2> alpha_value{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_dot_b1{};
    bsk::V<float, 2> atom_dot_damping{};
    bsk::V<float, 2> atom_dot_inversion{};
    bsk::V<float, 2> atom_dot_m0{};
    bsk::V<float, 2> atom_inversion{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 3> bare1_tangent{};
    bsk::V<float, 3> bare1_value{};
    bsk::V<float, 3> bare2_tangent{};
    bsk::V<float, 3> bare2_value{};
    bsk::V<float, 2> chs_tangent{};
    bsk::V<float, 2> chs_value{};
    bsk::V<float, 2> cosine_tangent{};
    bsk::V<float, 2> cosine_value{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_t_tangent{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 3> damp_z_tangent{};
    bool do_shift{};
    bsk::V<float, 2> dt_tangent{};
    bsk::V<float, 2> dt_value{};
    bsk::V<float, 3> duration_gain_tangent{};
    bsk::V<float, 3> duration_gain_value{};
    bsk::V<float, 3> e1_tangent{};
    bsk::V<float, 3> e1_value{};
    bsk::V<float, 3> e2_tangent{};
    bsk::V<float, 3> e2_value{};
    std::int64_t event{};
    std::int32_t event_action{};
    bsk::V<float, 2> event_dot_flip{};
    bsk::V<float, 2> event_flip{};
    std::int32_t event_kind{};
    bsk::V<float, 2> flip_gain_tangent{};
    bsk::V<float, 2> grad_b1_tangent{};
    bsk::V<float, 2> grad_b1_value{};
    bsk::V<float, 2> grad_damping_tangent{};
    bsk::V<float, 2> grad_damping_value{};
    bsk::V<float, 2> grad_e1_tangent{};
    bsk::V<float, 2> grad_e1_value{};
    bsk::V<float, 2> grad_inversion_tangent{};
    bsk::V<float, 2> grad_inversion_value{};
    bsk::V<float, 2> grad_m0_tangent{};
    bsk::V<float, 2> grad_m0_value{};
    bsk::V<float, 3> grad_t1_tangent{};
    bsk::V<float, 3> grad_t1_value{};
    bsk::V<float, 3> grad_t2_tangent{};
    bsk::V<float, 3> grad_t2_value{};
    bsk::V<float, 2> half_sine_tangent{};
    bsk::V<float, 2> half_sine_value{};
    bool invert{};
    bsk::V<float, 3> inverted_tangent{};
    bool is_inversion{};
    bool is_rf{};
    bsk::V<float, 3> long_bar_tangent{};
    bsk::V<float, 3> long_bar_value{};
    bsk::V<float, 3> long_tangent{};
    bsk::V<float, 3> long_value{};
    bsk::V<float, 3> minus_bar_tangent{};
    bsk::V<float, 3> minus_bar_value{};
    bsk::V<float, 3> minus_tangent{};
    bsk::V<float, 3> minus_value{};
    bsk::V<float, 3> plus_bar_tangent{};
    bsk::V<float, 3> plus_bar_value{};
    bsk::V<float, 3> plus_tangent{};
    bsk::V<float, 3> plus_value{};
    bool pre_shift{};
    bsk::V<std::int32_t, 2> problem{};
    bsk::V<float, 2> pulse_b1{};
    bsk::V<float, 2> pulse_dot_b1{};
    bsk::V<float, 3> recovery_tangent{};
    bsk::V<float, 3> recovery_value{};
    bool rotate{};
    bsk::V<float, 3> rotated_mbt{};
    bsk::V<float, 3> rotated_mbv{};
    bsk::V<float, 3> rotated_mt{};
    bsk::V<float, 3> rotated_mv{};
    bsk::V<float, 3> rotated_pbt{};
    bsk::V<float, 3> rotated_pbv{};
    bsk::V<float, 3> rotated_pt{};
    bsk::V<float, 3> rotated_pv{};
    bsk::V<float, 3> rotated_zbt{};
    bsk::V<float, 3> rotated_zbv{};
    bsk::V<float, 3> rotated_zt{};
    bsk::V<float, 3> rotated_zv{};
    bsk::V<float, 3> row_m_tangent{};
    bsk::V<float, 3> row_m_value{};
    bsk::V<float, 3> row_p_tangent{};
    bsk::V<float, 3> row_p_value{};
    bsk::V<float, 3> row_z_tangent{};
    bsk::V<float, 3> row_z_value{};
    bsk::V<float, 3> scale1_tangent{};
    bsk::V<float, 3> scale2_tangent{};
    bsk::V<float, 3> shifted_mt{};
    bsk::V<float, 3> shifted_mv{};
    bsk::V<float, 3> shifted_pt{};
    bsk::V<float, 3> shifted_pv{};
    std::int64_t shim_row{};
    bsk::V<float, 2> shs_tangent{};
    bsk::V<float, 2> shs_value{};
    bsk::V<float, 2> sine_tangent{};
    bsk::V<float, 2> sine_value{};
    bsk::V<std::int32_t, 3> slot{};
    bool spoil{};
    bsk::V<float, 2> spread_tangent{};
    bsk::V<float, 2> spread_value{};
    bsk::V<float, 3> stage_mt{};
    bsk::V<float, 3> stage_mv{};
    bsk::V<float, 3> stage_pt{};
    bsk::V<float, 3> stage_pv{};
    bsk::V<float, 3> stage_zt{};
    problem = (problem_base + (bsk::program_id(0) * problems));
    problem = (problem + bsk::arange_y());
    auto state = bsk::arange_x();
    // The grid rounds up to whole tiles, so the last program of a wave reaches
    // past it. Those problems are real, but their trajectory rows belong to a
    // later launch and do not exist yet.
    auto active_atom = (problem < problem_end);
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    // The trajectory holds the state entering every event: three planes of
    // configuration orders, for the value and the tangent alike.
    auto record_stride = (3 * state_count);
    auto trajectory = ((((problem - problem_base) * event_count) * record_stride) + state);
    auto minus_plane = state_count;
    auto long_plane = (2 * state_count);
    auto empty = bsk::full<float, 3>(0);
    plus_value = empty;
    plus_tangent = empty;
    minus_value = empty;
    minus_tangent = empty;
    long_value = (empty + bsk::where((state == 0), 1.0f, 0.0f));
    long_tangent = empty;
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_inversion = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inversion = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    auto atom_dot_t1 = bsk::ld((dot_t1 + atom), active_atom, 0.0f);
    auto atom_dot_t2 = bsk::ld((dot_t2 + atom), active_atom, 0.0f);
    atom_dot_m0 = 0.0f;
    if (bsk::truth(density)) {
        atom_dot_m0 = bsk::ld((dot_m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_dot_b1 = 0.0f;
    if (bsk::truth(transmit)) {
        atom_dot_b1 = bsk::ld((dot_b1 + scalar_atom), active_atom, 0.0f);
    }
    atom_dot_inversion = 0.0f;
    if (bsk::truth(inverting)) {
        atom_dot_inversion = bsk::ld((dot_inversion_efficiency + scalar_atom), active_atom, 0.0f);
    }
    atom_damping = 0.0f;
    atom_dot_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
        atom_dot_damping = bsk::ld((dot_diffusion + scalar_atom), active_atom, 0.0f);
    }
    auto order = bsk::cast<float>(state);
    auto longitudinal_weight = (order * order);
    auto transverse_weight = ((longitudinal_weight + order) + 0.3333333333333333f);
    auto rate1_value = bsk::truediv(1000.0f, atom_t1);
    auto rate1_tangent = bsk::truediv((-1000.0f * atom_dot_t1), (atom_t1 * atom_t1));
    auto rate2_value = bsk::truediv(1000.0f, atom_t2);
    auto rate2_tangent = bsk::truediv((-1000.0f * atom_dot_t2), (atom_t2 * atom_t2));
    auto event_base = (train * event_count);
    for (bsk::index_t event = 0; event < event_count; event += 1) {
        slot = (trajectory + (event * record_stride));
        bsk::st((trajectory_value + slot), plus_value, state_mask);
        bsk::st(((trajectory_value + slot) + minus_plane), minus_value, state_mask);
        bsk::st(((trajectory_value + slot) + long_plane), long_value, state_mask);
        bsk::st((trajectory_tangent + slot), plus_tangent, state_mask);
        bsk::st(((trajectory_tangent + slot) + minus_plane), minus_tangent, state_mask);
        bsk::st(((trajectory_tangent + slot) + long_plane), long_tangent, state_mask);
        dt_value = _event_value(duration, event_base, event, active_atom, single_train);
        dt_tangent = _event_value(dot_duration, event_base, event, active_atom, single_train);
        e1_value = bsk::exp(((-rate1_value) * dt_value));
        e1_tangent = ((-e1_value) * ((rate1_value * dt_tangent) + (rate1_tangent * dt_value)));
        e2_value = bsk::exp(((-rate2_value) * dt_value));
        e2_tangent = ((-e2_value) * ((rate2_value * dt_tangent) + (rate2_tangent * dt_value)));
        damp_z = 1.0f;
        damp_z_tangent = 0.0f;
        damp_t = 1.0f;
        damp_t_tangent = 0.0f;
        if (bsk::truth(diffusing)) {
            auto t0_ = _damping_jvp(atom_damping, atom_dot_damping, dt_value, dt_tangent, order);
            damp_z = bsk::get<0>(t0_);
            damp_z_tangent = bsk::get<1>(t0_);
            damp_t = bsk::get<2>(t0_);
            damp_t_tangent = bsk::get<3>(t0_);
        }
        // Order zero is undamped, so recovery keeps the bare longitudinal factor.
        auto t1_ = bsk::make_tup((1.0f - e1_value), (-e1_tangent));
        recovery_value = bsk::get<0>(t1_);
        recovery_tangent = bsk::get<1>(t1_);
        auto t2_ = bsk::make_tup(e1_value, e1_tangent);
        bare1_value = bsk::get<0>(t2_);
        bare1_tangent = bsk::get<1>(t2_);
        auto t3_ = bsk::make_tup(e2_value, e2_tangent);
        bare2_value = bsk::get<0>(t3_);
        bare2_tangent = bsk::get<1>(t3_);
        e1_tangent = ((e1_tangent * damp_z) + (bare1_value * damp_z_tangent));
        e1_value = (bare1_value * damp_z);
        e2_tangent = ((e2_tangent * damp_t) + (bare2_value * damp_t_tangent));
        e2_value = (bare2_value * damp_t);
        plus_tangent = ((plus_value * e2_tangent) + (plus_tangent * e2_value));
        plus_value = (plus_value * e2_value);
        minus_tangent = ((minus_value * e2_tangent) + (minus_tangent * e2_value));
        minus_value = (minus_value * e2_value);
        long_tangent = ((long_value * e1_tangent) + (long_tangent * e1_value));
        long_value = (long_value * e1_value);
        long_value = (long_value + bsk::where((state == 0), recovery_value, 0.0f));
        long_tangent = (long_tangent + bsk::where((state == 0), recovery_tangent, 0.0f));
        event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        pre_shift = (bsk::band(event_action, 1) != 0);
        auto t4_ = _shift_real(plus_value, minus_value, state, state_mask, state_count);
        shifted_pv = bsk::get<0>(t4_);
        shifted_mv = bsk::get<1>(t4_);
        auto t5_ = _shift_real(plus_tangent, minus_tangent, state, state_mask, state_count);
        shifted_pt = bsk::get<0>(t5_);
        shifted_mt = bsk::get<1>(t5_);
        plus_value = bsk::where(pre_shift, shifted_pv, plus_value);
        minus_value = bsk::where(pre_shift, shifted_mv, minus_value);
        plus_tangent = bsk::where(pre_shift, shifted_pt, plus_tangent);
        minus_tangent = bsk::where(pre_shift, shifted_mt, minus_tangent);
        event_kind = bsk::ld((kind + event));
        is_rf = (event_kind == 1);
        is_inversion = (bsk::band(event_action, 4) != 0);
        invert = bsk::band(is_rf, is_inversion);
        auto inverted_value = ((-atom_inversion) * long_value);
        inverted_tangent = ((-atom_inversion) * long_tangent);
        inverted_tangent = (inverted_tangent - (atom_dot_inversion * long_value));
        long_value = bsk::where(invert, inverted_value, long_value);
        long_tangent = bsk::where(invert, inverted_tangent, long_tangent);
        event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        event_dot_flip = _event_value(dot_flip, event_base, event, active_atom, single_train);
        pulse_b1 = atom_b1;
        pulse_dot_b1 = atom_dot_b1;
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse the row of the shim it drives.
        if (bsk::truth(shimmed)) {
            shim_row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            if (bsk::truth(transmit)) {
                pulse_b1 = bsk::ld(((b1 + shim_row) + atom), active_atom, 1.0f);
            }
            pulse_dot_b1 = bsk::ld(((dot_b1 + shim_row) + atom), active_atom, 0.0f);
        }
        alpha_value = (event_flip * pulse_b1);
        alpha_tangent = ((event_dot_flip * pulse_b1) + (event_flip * pulse_dot_b1));
        cosine_value = bsk::cos(alpha_value);
        sine_value = bsk::sin(alpha_value);
        cosine_tangent = ((-sine_value) * alpha_tangent);
        sine_tangent = (cosine_value * alpha_tangent);
        chs_value = (0.5f * (1.0f + cosine_value));
        chs_tangent = (0.5f * cosine_tangent);
        shs_value = (0.5f * (1.0f - cosine_value));
        shs_tangent = (-0.5f * cosine_tangent);
        half_sine_value = (0.5f * sine_value);
        half_sine_tangent = (0.5f * sine_tangent);
        rotated_pv = ((chs_value * plus_value) + (shs_value * minus_value));
        rotated_pv = (rotated_pv - (sine_value * long_value));
        rotated_pt = ((chs_value * plus_tangent) + (chs_tangent * plus_value));
        rotated_pt = (rotated_pt + ((shs_value * minus_tangent) + (shs_tangent * minus_value)));
        rotated_pt = (rotated_pt - ((sine_value * long_tangent) + (sine_tangent * long_value)));
        rotated_mv = ((shs_value * plus_value) + (chs_value * minus_value));
        rotated_mv = (rotated_mv + (sine_value * long_value));
        rotated_mt = ((shs_value * plus_tangent) + (shs_tangent * plus_value));
        rotated_mt = (rotated_mt + ((chs_value * minus_tangent) + (chs_tangent * minus_value)));
        rotated_mt = (rotated_mt + ((sine_value * long_tangent) + (sine_tangent * long_value)));
        rotated_zv = ((half_sine_value * plus_value) - (half_sine_value * minus_value));
        rotated_zv = (rotated_zv + (cosine_value * long_value));
        rotated_zt = ((half_sine_value * plus_tangent) + (half_sine_tangent * plus_value));
        rotated_zt = (rotated_zt - ((half_sine_value * minus_tangent) + (half_sine_tangent * minus_value)));
        rotated_zt = (rotated_zt + ((cosine_value * long_tangent) + (cosine_tangent * long_value)));
        rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        plus_value = bsk::where(rotate, rotated_pv, plus_value);
        plus_tangent = bsk::where(rotate, rotated_pt, plus_tangent);
        minus_value = bsk::where(rotate, rotated_mv, minus_value);
        minus_tangent = bsk::where(rotate, rotated_mt, minus_tangent);
        long_value = bsk::where(rotate, rotated_zv, long_value);
        long_tangent = bsk::where(rotate, rotated_zt, long_tangent);
        do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        auto t6_ = _shift_real(plus_value, minus_value, state, state_mask, state_count);
        shifted_pv = bsk::get<0>(t6_);
        shifted_mv = bsk::get<1>(t6_);
        auto t7_ = _shift_real(plus_tangent, minus_tangent, state, state_mask, state_count);
        shifted_pt = bsk::get<0>(t7_);
        shifted_mt = bsk::get<1>(t7_);
        plus_value = bsk::where(do_shift, shifted_pv, plus_value);
        minus_value = bsk::where(do_shift, shifted_mv, minus_value);
        plus_tangent = bsk::where(do_shift, shifted_pt, plus_tangent);
        minus_tangent = bsk::where(do_shift, shifted_mt, minus_tangent);
        spoil = (bsk::band(event_action, 8) != 0);
        plus_value = bsk::where(spoil, 0.0f, plus_value);
        minus_value = bsk::where(spoil, 0.0f, minus_value);
        plus_tangent = bsk::where(spoil, 0.0f, plus_tangent);
        minus_tangent = bsk::where(spoil, 0.0f, minus_tangent);
    }
    plus_bar_value = empty;
    plus_bar_tangent = empty;
    minus_bar_value = empty;
    minus_bar_tangent = empty;
    long_bar_value = empty;
    long_bar_tangent = empty;
    auto zero = bsk::full<float, 2>(0);
    grad_t1_value = zero;
    grad_t1_tangent = zero;
    grad_t2_value = zero;
    grad_t2_tangent = zero;
    grad_m0_value = zero;
    grad_m0_tangent = zero;
    grad_b1_value = zero;
    grad_b1_tangent = zero;
    grad_inversion_value = zero;
    grad_inversion_tangent = zero;
    grad_damping_value = zero;
    grad_damping_tangent = zero;
    for (bsk::index_t reverse = 0; reverse < event_count; reverse += 1) {
        event = ((event_count - 1) - reverse);
        slot = (trajectory + (event * record_stride));
        auto entry_pv = bsk::ld((trajectory_value + slot), state_mask, 0.0f);
        auto entry_mv = bsk::ld(((trajectory_value + slot) + minus_plane), state_mask, 0.0f);
        auto entry_zv = bsk::ld(((trajectory_value + slot) + long_plane), state_mask, 0.0f);
        auto entry_pt = bsk::ld((trajectory_tangent + slot), state_mask, 0.0f);
        auto entry_mt = bsk::ld(((trajectory_tangent + slot) + minus_plane), state_mask, 0.0f);
        auto entry_zt = bsk::ld(((trajectory_tangent + slot) + long_plane), state_mask, 0.0f);
        event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        event_kind = bsk::ld((kind + event));
        dt_value = _event_value(duration, event_base, event, active_atom, single_train);
        dt_tangent = _event_value(dot_duration, event_base, event, active_atom, single_train);
        e1_value = bsk::exp(((-rate1_value) * dt_value));
        e1_tangent = ((-e1_value) * ((rate1_value * dt_tangent) + (rate1_tangent * dt_value)));
        e2_value = bsk::exp(((-rate2_value) * dt_value));
        e2_tangent = ((-e2_value) * ((rate2_value * dt_tangent) + (rate2_tangent * dt_value)));
        damp_z = 1.0f;
        damp_z_tangent = 0.0f;
        damp_t = 1.0f;
        damp_t_tangent = 0.0f;
        if (bsk::truth(diffusing)) {
            auto t8_ = _damping_jvp(atom_damping, atom_dot_damping, dt_value, dt_tangent, order);
            damp_z = bsk::get<0>(t8_);
            damp_z_tangent = bsk::get<1>(t8_);
            damp_t = bsk::get<2>(t8_);
            damp_t_tangent = bsk::get<3>(t8_);
        }
        // Order zero is undamped, so recovery keeps the bare longitudinal factor.
        auto t9_ = bsk::make_tup((1.0f - e1_value), (-e1_tangent));
        recovery_value = bsk::get<0>(t9_);
        recovery_tangent = bsk::get<1>(t9_);
        auto t10_ = bsk::make_tup(e1_value, e1_tangent);
        bare1_value = bsk::get<0>(t10_);
        bare1_tangent = bsk::get<1>(t10_);
        auto t11_ = bsk::make_tup(e2_value, e2_tangent);
        bare2_value = bsk::get<0>(t11_);
        bare2_tangent = bsk::get<1>(t11_);
        e1_tangent = ((e1_tangent * damp_z) + (bare1_value * damp_z_tangent));
        e1_value = (bare1_value * damp_z);
        e2_tangent = ((e2_tangent * damp_t) + (bare2_value * damp_t_tangent));
        e2_value = (bare2_value * damp_t);
        // Replay the intra-event stages from the recorded entry state.
        stage_pv = (entry_pv * e2_value);
        stage_pt = ((entry_pv * e2_tangent) + (entry_pt * e2_value));
        stage_mv = (entry_mv * e2_value);
        stage_mt = ((entry_mv * e2_tangent) + (entry_mt * e2_value));
        auto stage_zv = ((entry_zv * e1_value) + bsk::where((state == 0), recovery_value, 0.0f));
        stage_zt = ((entry_zv * e1_tangent) + (entry_zt * e1_value));
        stage_zt = (stage_zt + bsk::where((state == 0), recovery_tangent, 0.0f));
        pre_shift = (bsk::band(event_action, 1) != 0);
        auto t12_ = _shift_real(stage_pv, stage_mv, state, state_mask, state_count);
        shifted_pv = bsk::get<0>(t12_);
        shifted_mv = bsk::get<1>(t12_);
        auto t13_ = _shift_real(stage_pt, stage_mt, state, state_mask, state_count);
        shifted_pt = bsk::get<0>(t13_);
        shifted_mt = bsk::get<1>(t13_);
        stage_pv = bsk::where(pre_shift, shifted_pv, stage_pv);
        stage_mv = bsk::where(pre_shift, shifted_mv, stage_mv);
        stage_pt = bsk::where(pre_shift, shifted_pt, stage_pt);
        stage_mt = bsk::where(pre_shift, shifted_mt, stage_mt);
        // Undo the trailing spoil or shift.
        do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        spoil = (bsk::band(event_action, 8) != 0);
        auto t14_ = _shift_real_adjoint(plus_bar_value, minus_bar_value, state, state_mask, state_count);
        adjoint_pv = bsk::get<0>(t14_);
        adjoint_mv = bsk::get<1>(t14_);
        auto t15_ = _shift_real_adjoint(plus_bar_tangent, minus_bar_tangent, state, state_mask, state_count);
        adjoint_pt = bsk::get<0>(t15_);
        adjoint_mt = bsk::get<1>(t15_);
        auto trailing = bsk::band(do_shift, bsk::bnot(spoil));
        plus_bar_value = bsk::where(spoil, 0.0f, bsk::where(trailing, adjoint_pv, plus_bar_value));
        minus_bar_value = bsk::where(spoil, 0.0f, bsk::where(trailing, adjoint_mv, minus_bar_value));
        plus_bar_tangent = bsk::where(spoil, 0.0f, bsk::where(trailing, adjoint_pt, plus_bar_tangent));
        minus_bar_tangent = bsk::where(spoil, 0.0f, bsk::where(trailing, adjoint_mt, minus_bar_tangent));
        is_rf = (event_kind == 1);
        is_inversion = (bsk::band(event_action, 4) != 0);
        invert = bsk::band(is_rf, is_inversion);
        auto inversion_gain = (-bsk::sum_x(bsk::where(invert, (long_bar_value * stage_zv), 0.0f)));
        auto inversion_gain_tangent = (-bsk::sum_x(bsk::where(invert, ((long_bar_value * stage_zt) + (long_bar_tangent * stage_zv)), 0.0f)));
        grad_inversion_value = (grad_inversion_value + inversion_gain);
        grad_inversion_tangent = (grad_inversion_tangent + inversion_gain_tangent);
        auto inverted_bar_value = ((-atom_inversion) * long_bar_value);
        auto inverted_bar_tangent = (((-atom_inversion) * long_bar_tangent) - (atom_dot_inversion * long_bar_value));
        long_bar_value = bsk::where(invert, inverted_bar_value, long_bar_value);
        long_bar_tangent = bsk::where(invert, inverted_bar_tangent, long_bar_tangent);
        event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        event_dot_flip = _event_value(dot_flip, event_base, event, active_atom, single_train);
        pulse_b1 = atom_b1;
        pulse_dot_b1 = atom_dot_b1;
        // One shim is the whole sequence's transmit field, loaded once above;
        // several give each pulse the row of the shim it drives.
        if (bsk::truth(shimmed)) {
            shim_row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            if (bsk::truth(transmit)) {
                pulse_b1 = bsk::ld(((b1 + shim_row) + atom), active_atom, 1.0f);
            }
            pulse_dot_b1 = bsk::ld(((dot_b1 + shim_row) + atom), active_atom, 0.0f);
        }
        alpha_value = (event_flip * pulse_b1);
        alpha_tangent = ((event_dot_flip * pulse_b1) + (event_flip * pulse_dot_b1));
        cosine_value = bsk::cos(alpha_value);
        sine_value = bsk::sin(alpha_value);
        cosine_tangent = ((-sine_value) * alpha_tangent);
        sine_tangent = (cosine_value * alpha_tangent);
        chs_value = (0.5f * (1.0f + cosine_value));
        chs_tangent = (0.5f * cosine_tangent);
        shs_value = (0.5f * (1.0f - cosine_value));
        shs_tangent = (-0.5f * cosine_tangent);
        half_sine_value = (0.5f * sine_value);
        half_sine_tangent = (0.5f * sine_tangent);
        // d/dalpha of each output row, contracted with the adjoint.
        row_p_value = ((half_sine_value * stage_mv) - (half_sine_value * stage_pv));
        row_p_value = (row_p_value - (cosine_value * stage_zv));
        row_p_tangent = ((half_sine_value * stage_mt) + (half_sine_tangent * stage_mv));
        row_p_tangent = (row_p_tangent - ((half_sine_value * stage_pt) + (half_sine_tangent * stage_pv)));
        row_p_tangent = (row_p_tangent - ((cosine_value * stage_zt) + (cosine_tangent * stage_zv)));
        row_m_value = ((half_sine_value * stage_pv) - (half_sine_value * stage_mv));
        row_m_value = (row_m_value + (cosine_value * stage_zv));
        row_m_tangent = ((half_sine_value * stage_pt) + (half_sine_tangent * stage_pv));
        row_m_tangent = (row_m_tangent - ((half_sine_value * stage_mt) + (half_sine_tangent * stage_mv)));
        row_m_tangent = (row_m_tangent + ((cosine_value * stage_zt) + (cosine_tangent * stage_zv)));
        row_z_value = (((0.5f * cosine_value) * stage_pv) - ((0.5f * cosine_value) * stage_mv));
        row_z_value = (row_z_value - (sine_value * stage_zv));
        row_z_tangent = (0.5f * ((cosine_value * stage_pt) + (cosine_tangent * stage_pv)));
        row_z_tangent = (row_z_tangent - (0.5f * ((cosine_value * stage_mt) + (cosine_tangent * stage_mv))));
        row_z_tangent = (row_z_tangent - ((sine_value * stage_zt) + (sine_tangent * stage_zv)));
        alpha_bar_terms_value = (plus_bar_value * row_p_value);
        alpha_bar_terms_value = (alpha_bar_terms_value + (minus_bar_value * row_m_value));
        alpha_bar_terms_value = (alpha_bar_terms_value + (long_bar_value * row_z_value));
        alpha_bar_terms_tangent = (plus_bar_value * row_p_tangent);
        alpha_bar_terms_tangent = (alpha_bar_terms_tangent + (plus_bar_tangent * row_p_value));
        alpha_bar_terms_tangent = (alpha_bar_terms_tangent + (minus_bar_value * row_m_tangent));
        alpha_bar_terms_tangent = (alpha_bar_terms_tangent + (minus_bar_tangent * row_m_value));
        alpha_bar_terms_tangent = (alpha_bar_terms_tangent + (long_bar_value * row_z_tangent));
        alpha_bar_terms_tangent = (alpha_bar_terms_tangent + (long_bar_tangent * row_z_value));
        rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        auto grad_alpha_value = bsk::sum_x(bsk::where(rotate, alpha_bar_terms_value, 0.0f));
        auto grad_alpha_tangent = bsk::sum_x(bsk::where(rotate, alpha_bar_terms_tangent, 0.0f));
        // Transpose of the rotation.
        rotated_pbv = ((chs_value * plus_bar_value) + (shs_value * minus_bar_value));
        rotated_pbv = (rotated_pbv + (half_sine_value * long_bar_value));
        rotated_pbt = ((chs_value * plus_bar_tangent) + (chs_tangent * plus_bar_value));
        rotated_pbt = (rotated_pbt + ((shs_value * minus_bar_tangent) + (shs_tangent * minus_bar_value)));
        rotated_pbt = (rotated_pbt + (half_sine_value * long_bar_tangent));
        rotated_pbt = (rotated_pbt + (half_sine_tangent * long_bar_value));
        rotated_mbv = ((shs_value * plus_bar_value) + (chs_value * minus_bar_value));
        rotated_mbv = (rotated_mbv - (half_sine_value * long_bar_value));
        rotated_mbt = ((shs_value * plus_bar_tangent) + (shs_tangent * plus_bar_value));
        rotated_mbt = (rotated_mbt + ((chs_value * minus_bar_tangent) + (chs_tangent * minus_bar_value)));
        rotated_mbt = (rotated_mbt - (half_sine_value * long_bar_tangent));
        rotated_mbt = (rotated_mbt - (half_sine_tangent * long_bar_value));
        rotated_zbv = (((-sine_value) * plus_bar_value) + (sine_value * minus_bar_value));
        rotated_zbv = (rotated_zbv + (cosine_value * long_bar_value));
        rotated_zbt = (((-sine_value) * plus_bar_tangent) - (sine_tangent * plus_bar_value));
        rotated_zbt = (rotated_zbt + ((sine_value * minus_bar_tangent) + (sine_tangent * minus_bar_value)));
        rotated_zbt = (rotated_zbt + ((cosine_value * long_bar_tangent) + (cosine_tangent * long_bar_value)));
        plus_bar_value = bsk::where(rotate, rotated_pbv, plus_bar_value);
        plus_bar_tangent = bsk::where(rotate, rotated_pbt, plus_bar_tangent);
        minus_bar_value = bsk::where(rotate, rotated_mbv, minus_bar_value);
        minus_bar_tangent = bsk::where(rotate, rotated_mbt, minus_bar_tangent);
        long_bar_value = bsk::where(rotate, rotated_zbv, long_bar_value);
        long_bar_tangent = bsk::where(rotate, rotated_zbt, long_bar_tangent);
        auto flip_gain_value = (grad_alpha_value * pulse_b1);
        flip_gain_tangent = (grad_alpha_tangent * pulse_b1);
        flip_gain_tangent = (flip_gain_tangent + (grad_alpha_value * pulse_dot_b1));
        auto writes_flip = bsk::band(active_atom, rotate);
        bsk::atomic_add(((grad_flip_value + event_base) + event), flip_gain_value, writes_flip);
        bsk::atomic_add(((grad_flip_tangent + event_base) + event), flip_gain_tangent, writes_flip);
        if (bsk::truth(shimmed)) {
            // A pulse's transmit gradient belongs to the shim it drives, so
            // with several it lands in that shim's row rather than in a
            // register summed over the whole train.
            bsk::atomic_add((((grad_tissue_value + (3 * atom_count)) + shim_row) + atom), (grad_alpha_value * event_flip), writes_flip);
            bsk::atomic_add((((grad_tissue_tangent + (3 * atom_count)) + shim_row) + atom), ((grad_alpha_tangent * event_flip) + (grad_alpha_value * event_dot_flip)), writes_flip);
        } else {
            grad_b1_value = (grad_b1_value + bsk::where(rotate, (grad_alpha_value * event_flip), 0.0f));
            grad_b1_tangent = (grad_b1_tangent + bsk::where(rotate, ((grad_alpha_tangent * event_flip) + (grad_alpha_value * event_dot_flip)), 0.0f));
        }
        // The sample is i * m0 * plus[0]; only the imaginary seed acts.
        auto record = bsk::band((bsk::band(event_action, 32) != 0), (event_kind == 2));
        auto out_ = bsk::ld((output_index + event));
        auto seed = bsk::ld(((grad_output_imag + (problem * output_count)) + out_), bsk::band(bsk::band(active_atom, record), (out_ >= 0)), 0.0f);
        grad_m0_value = (grad_m0_value + bsk::sum_x(bsk::where((state == 0), (seed * stage_pv), 0.0f)));
        grad_m0_tangent = (grad_m0_tangent + bsk::sum_x(bsk::where((state == 0), (seed * stage_pt), 0.0f)));
        plus_bar_value = (plus_bar_value + bsk::where((state == 0), (seed * atom_m0), 0.0f));
        plus_bar_tangent = (plus_bar_tangent + bsk::where((state == 0), (seed * atom_dot_m0), 0.0f));
        auto t16_ = _shift_real_adjoint(plus_bar_value, minus_bar_value, state, state_mask, state_count);
        adjoint_pv = bsk::get<0>(t16_);
        adjoint_mv = bsk::get<1>(t16_);
        auto t17_ = _shift_real_adjoint(plus_bar_tangent, minus_bar_tangent, state, state_mask, state_count);
        adjoint_pt = bsk::get<0>(t17_);
        adjoint_mt = bsk::get<1>(t17_);
        plus_bar_value = bsk::where(pre_shift, adjoint_pv, plus_bar_value);
        minus_bar_value = bsk::where(pre_shift, adjoint_mv, minus_bar_value);
        plus_bar_tangent = bsk::where(pre_shift, adjoint_pt, plus_bar_tangent);
        minus_bar_tangent = bsk::where(pre_shift, adjoint_mt, minus_bar_tangent);
        auto cot2_value = ((plus_bar_value * entry_pv) + (minus_bar_value * entry_mv));
        auto cot2_tangent = ((((plus_bar_value * entry_pt) + (plus_bar_tangent * entry_pv)) + (minus_bar_value * entry_mt)) + (minus_bar_tangent * entry_mv));
        auto cot1_value = (long_bar_value * entry_zv);
        auto cot1_tangent = ((long_bar_value * entry_zt) + (long_bar_tangent * entry_zv));
        auto grad_e2_value = bsk::sum_x((cot2_value * damp_t));
        auto grad_e2_tangent = bsk::sum_x(((cot2_value * damp_t_tangent) + (cot2_tangent * damp_t)));
        grad_e1_value = bsk::sum_x((cot1_value * damp_z));
        grad_e1_value = (grad_e1_value - bsk::sum_x(bsk::where((state == 0), long_bar_value, 0.0f)));
        grad_e1_tangent = bsk::sum_x(((cot1_value * damp_z_tangent) + (cot1_tangent * damp_z)));
        grad_e1_tangent = (grad_e1_tangent - bsk::sum_x(bsk::where((state == 0), long_bar_tangent, 0.0f)));
        // The rate and the interval multiply every order's b-weight, so both
        // take a weighted sum. Order zero has no longitudinal weight, which
        // keeps recovery out of this.
        spread_value = zero;
        spread_tangent = zero;
        if (bsk::truth(diffusing)) {
            auto weighted_value = ((((cot1_value * bare1_value) * damp_z) * longitudinal_weight) + (((cot2_value * bare2_value) * damp_t) * transverse_weight));
            auto weighted_tangent = ((((((cot1_tangent * bare1_value) * damp_z) + ((cot1_value * bare1_tangent) * damp_z)) + ((cot1_value * bare1_value) * damp_z_tangent)) * longitudinal_weight) + (((((cot2_tangent * bare2_value) * damp_t) + ((cot2_value * bare2_tangent) * damp_t)) + ((cot2_value * bare2_value) * damp_t_tangent)) * transverse_weight));
            spread_value = bsk::sum_x(weighted_value);
            spread_tangent = bsk::sum_x(weighted_tangent);
            grad_damping_value = (grad_damping_value + ((-spread_value) * dt_value));
            grad_damping_tangent = (grad_damping_tangent + (-((spread_value * dt_tangent) + (spread_tangent * dt_value))));
        }
        plus_bar_tangent = ((plus_bar_value * e2_tangent) + (plus_bar_tangent * e2_value));
        plus_bar_value = (plus_bar_value * e2_value);
        minus_bar_tangent = ((minus_bar_value * e2_tangent) + (minus_bar_tangent * e2_value));
        minus_bar_value = (minus_bar_value * e2_value);
        long_bar_tangent = ((long_bar_value * e1_tangent) + (long_bar_tangent * e1_value));
        long_bar_value = (long_bar_value * e1_value);
        auto inverse1_value = bsk::truediv(1000.0f, (atom_t1 * atom_t1));
        auto inverse1_tangent = bsk::truediv((-2000.0f * atom_dot_t1), ((atom_t1 * atom_t1) * atom_t1));
        auto inverse2_value = bsk::truediv(1000.0f, (atom_t2 * atom_t2));
        auto inverse2_tangent = bsk::truediv((-2000.0f * atom_dot_t2), ((atom_t2 * atom_t2) * atom_t2));
        auto scale1_value = ((bare1_value * dt_value) * inverse1_value);
        scale1_tangent = ((bare1_tangent * dt_value) * inverse1_value);
        scale1_tangent = (scale1_tangent + ((bare1_value * dt_tangent) * inverse1_value));
        scale1_tangent = (scale1_tangent + ((bare1_value * dt_value) * inverse1_tangent));
        auto scale2_value = ((bare2_value * dt_value) * inverse2_value);
        scale2_tangent = ((bare2_tangent * dt_value) * inverse2_value);
        scale2_tangent = (scale2_tangent + ((bare2_value * dt_tangent) * inverse2_value));
        scale2_tangent = (scale2_tangent + ((bare2_value * dt_value) * inverse2_tangent));
        grad_t1_value = (grad_t1_value + (grad_e1_value * scale1_value));
        grad_t1_tangent = (grad_t1_tangent + (grad_e1_value * scale1_tangent));
        grad_t1_tangent = (grad_t1_tangent + (grad_e1_tangent * scale1_value));
        grad_t2_value = (grad_t2_value + (grad_e2_value * scale2_value));
        grad_t2_tangent = (grad_t2_tangent + (grad_e2_value * scale2_tangent));
        grad_t2_tangent = (grad_t2_tangent + (grad_e2_tangent * scale2_value));
        auto decay1_value = (rate1_value * bare1_value);
        auto decay1_tangent = ((rate1_value * bare1_tangent) + (rate1_tangent * bare1_value));
        auto decay2_value = (rate2_value * bare2_value);
        auto decay2_tangent = ((rate2_value * bare2_tangent) + (rate2_tangent * bare2_value));
        duration_gain_value = ((-grad_e1_value) * decay1_value);
        duration_gain_value = (duration_gain_value - (grad_e2_value * decay2_value));
        duration_gain_tangent = (-((grad_e1_value * decay1_tangent) + (grad_e1_tangent * decay1_value)));
        duration_gain_tangent = (duration_gain_tangent - ((grad_e2_value * decay2_tangent) + (grad_e2_tangent * decay2_value)));
        duration_gain_value = (duration_gain_value + ((-spread_value) * atom_damping));
        duration_gain_tangent = (duration_gain_tangent + (-((spread_value * atom_dot_damping) + (spread_tangent * atom_damping))));
        bsk::atomic_add(((grad_duration_value + event_base) + event), duration_gain_value, active_atom);
        bsk::atomic_add(((grad_duration_tangent + event_base) + event), duration_gain_tangent, active_atom);
    }
    bsk::atomic_add((grad_tissue_value + atom), grad_t1_value, active_atom);
    bsk::atomic_add((grad_tissue_tangent + atom), grad_t1_tangent, active_atom);
    bsk::atomic_add(((grad_tissue_value + atom_count) + atom), grad_t2_value, active_atom);
    bsk::atomic_add(((grad_tissue_tangent + atom_count) + atom), grad_t2_tangent, active_atom);
    bsk::atomic_add(((grad_tissue_value + (2 * atom_count)) + atom), grad_m0_value, active_atom);
    bsk::atomic_add(((grad_tissue_tangent + (2 * atom_count)) + atom), grad_m0_tangent, active_atom);
    if (bsk::truth((!bsk::truth(shimmed)))) {
        bsk::atomic_add(((grad_tissue_value + (3 * atom_count)) + atom), grad_b1_value, active_atom);
        bsk::atomic_add(((grad_tissue_tangent + (3 * atom_count)) + atom), grad_b1_tangent, active_atom);
    }
    // The transmit pair takes a row per shim each in the plane the complex
    // path allocates, so the rows past it move even though this kernel leaves
    // the transmit phase at zero throughout.
    auto past_transmit = (2 * (shim_rows - 1));
    bsk::atomic_add(((grad_tissue_value + ((6 + past_transmit) * atom_count)) + atom), grad_inversion_value, active_atom);
    bsk::atomic_add(((grad_tissue_tangent + ((6 + past_transmit) * atom_count)) + atom), grad_inversion_tangent, active_atom);
    bsk::atomic_add(((grad_tissue_value + ((7 + past_transmit) * atom_count)) + atom), grad_damping_value, active_atom);
    bsk::atomic_add(((grad_tissue_tangent + ((7 + past_transmit) * atom_count)) + atom), grad_damping_tangent, active_atom);
}

BSK_HD void _epg_real_kernel(float* t1, float* t2, float* m0, float* b1, float* inversion_efficiency, float* diffusion, float* duration, std::int32_t* kind, float* flip, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* output_real, float* output_imag, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shimmed, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 2> alpha{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_inversion{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 2> dt{};
    bsk::V<float, 2> e1{};
    bsk::V<float, 2> e2{};
    bsk::V<float, 2> last_dt{};
    bsk::V<float, 3> longitudinal{};
    bsk::V<float, 3> minus{};
    bsk::V<float, 3> plus{};
    bsk::V<float, 2> pulse_b1{};
    bsk::V<bool, 2> relaxes{};
    auto problem = ((bsk::program_id(0) * problems) + bsk::arange_y());
    auto state = bsk::arange_x();
    auto active_atom = (problem < (train_count * atom_count));
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    auto empty = bsk::full<float, 3>(0);
    plus = empty;
    minus = empty;
    longitudinal = (empty + bsk::where((state == 0), 1.0f, 0.0f));
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_inversion = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inversion = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    auto rate1 = bsk::truediv(1000.0f, atom_t1);
    auto rate2 = bsk::truediv(1000.0f, atom_t2);
    atom_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
    }
    auto order = bsk::cast<float>(state);
    // The relaxation factors depend on the event only through its duration, and
    // a train repeats its intervals: an interval as long as the last one reuses
    // the factors rather than taking the two exponentials again. Where several
    // trains share the program the durations differ across its lanes and there
    // is nothing uniform to compare, so only a single-train launch memoizes.
    last_dt = -1.0f;
    e1 = ((rate1 * 0.0f) + 1.0f);
    e2 = ((rate2 * 0.0f) + 1.0f);
    auto event_base = (train * event_count);
    // Two events to an iteration. A repetition is several events -- a pulse,
    // a sample, an interval -- so the loop runs longer than the sequence is
    // repetitions, and unrolling lets one back-edge and one set of event
    // bookkeeping serve two of them. Two is where it stops paying: four was
    // measured slower, and the body is already large enough that widening it
    // costs registers.
    for (bsk::index_t event = 0; event < event_count; event += 1) {
        // Read here rather than through the helper: one train gives a duration
        // the whole program shares, and the skip and the memo below both want
        // to compare it as the single number it is.
        if (bsk::truth(single_train)) {
            dt = bsk::ld((duration + event));
        } else {
            dt = bsk::ld(((duration + event_base) + event), active_atom, 0.0f);
        }
        // An event of no duration relaxes nothing: both factors are one and the
        // recovery term is zero. Half the events of a spoiled repetition are
        // instantaneous, and reducing over the trains this program carries makes
        // that a branch the whole program agrees on rather than a tile of
        // multiplies by one.
        if (bsk::truth(single_train)) {
            relaxes = (dt != 0.0f);
        } else {
            relaxes = (bsk::max_all(dt) != 0.0f);
        }
        if (bsk::truth(relaxes)) {
            if (bsk::truth(single_train)) {
                if (bsk::truth((dt != last_dt))) {
                    e1 = bsk::exp(((-rate1) * dt));
                    e2 = bsk::exp(((-rate2) * dt));
                    last_dt = dt;
                }
            } else {
                e1 = bsk::exp(((-rate1) * dt));
                e2 = bsk::exp(((-rate2) * dt));
            }
            damp_z = 1.0f;
            damp_t = 1.0f;
            if (bsk::truth(diffusing)) {
                auto t0_ = _damping(atom_damping, dt, order);
                damp_z = bsk::get<0>(t0_);
                damp_t = bsk::get<1>(t0_);
            }
            auto recovery = (1.0f - e1);
            plus = (plus * (e2 * damp_t));
            minus = (minus * (e2 * damp_t));
            longitudinal = ((longitudinal * (e1 * damp_z)) + bsk::where((state == 0), recovery, 0.0f));
        }
        // Every flag below is read from a per-event array with no atom index, so
        // it is uniform across the program and can steer real control flow. A
        // `tl.where` would make every event pay for every operator: a spoiled
        // repetition is four events and needs one rotation and one shift.
        auto event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        if (bsk::truth((bsk::band(event_action, 1) != 0))) {
            auto t1_ = _shift_real(plus, minus, state, state_mask, state_count);
            plus = bsk::get<0>(t1_);
            minus = bsk::get<1>(t1_);
        }
        auto event_kind = bsk::ld((kind + event));
        auto is_rf = (event_kind == 1);
        auto is_inversion = (bsk::band(event_action, 4) != 0);
        if (bsk::truth((bsk::truth(is_rf) && bsk::truth(is_inversion)))) {
            longitudinal = ((-atom_inversion) * longitudinal);
        } else if (bsk::truth(is_rf)) {
            alpha = _event_value(flip, event_base, event, active_atom, single_train);
            pulse_b1 = atom_b1;
            // One shim is the whole sequence's transmit field, loaded once
            // above; several give each pulse the row of the shim it drives.
            if (bsk::truth((bsk::truth(shimmed) && bsk::truth(transmit)))) {
                auto shim_row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
                pulse_b1 = bsk::ld(((b1 + shim_row) + atom), active_atom, 1.0f);
            }
            alpha = (alpha * pulse_b1);
            auto cosine = bsk::cos(alpha);
            auto sine = bsk::sin(alpha);
            auto cosine_half_sq = (0.5f * (1.0f + cosine));
            auto sine_half_sq = (0.5f * (1.0f - cosine));
            auto half_sine = (0.5f * sine);
            auto rotated_p = (((cosine_half_sq * plus) + (sine_half_sq * minus)) - (sine * longitudinal));
            auto rotated_m = (((sine_half_sq * plus) + (cosine_half_sq * minus)) + (sine * longitudinal));
            longitudinal = (((half_sine * plus) - (half_sine * minus)) + (cosine * longitudinal));
            plus = rotated_p;
            minus = rotated_m;
        }
        if (bsk::truth((bsk::truth((bsk::band(event_action, 32) != 0)) && bsk::truth((event_kind == 2))))) {
            auto out_ = bsk::ld((output_index + event));
            auto output_offset = ((problem * output_count) + out_);
            auto output_mask = bsk::band(bsk::band(active_atom, (state == 0)), (out_ >= 0));
            bsk::st(((output_real + output_offset) + state), empty, output_mask);
            bsk::st(((output_imag + output_offset) + state), (atom_m0 * plus), output_mask);
        }
        if (bsk::truth((bsk::truth((bsk::band(event_action, 2) != 0)) || bsk::truth((bsk::band(event_action, 16) != 0))))) {
            auto t2_ = _shift_real(plus, minus, state, state_mask, state_count);
            plus = bsk::get<0>(t2_);
            minus = bsk::get<1>(t2_);
        }
        if (bsk::truth((bsk::band(event_action, 8) != 0))) {
            plus = empty;
            minus = empty;
        }
    }
}

// Seven of the nine rotation coefficients; the rest follow by symmetry.
//
// ``t11`` repeats ``t00`` and ``t10`` is the conjugate of ``t01``, so the
// caller derives those. Feeding ``(cos, sin)`` gives the rotation itself and
// ``(sin, cos)`` rearranged gives its derivative in the flip angle, which is
// why this is one routine rather than two.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
BSK_HD auto _rotation_coefficients(const T0& a, const T1& b, const T2& c, const T3& d, const T4& p1r, const T5& p1i, const T6& p2r, const T7& p2i, const T8& pcr, const T9& pci) {
    auto t00 = bsk::make_tup(a, (0.0f * a));
    auto t01 = bsk::make_tup((b * p2r), (b * p2i));
    auto t02 = _complex_mul((0.0f * c), (-c), p1r, p1i);
    auto t12 = _complex_mul((0.0f * c), c, pcr, pci);
    auto t20 = _complex_mul((0.0f * c), (-0.5f * c), pcr, pci);
    auto t21 = _complex_mul((0.0f * c), (0.5f * c), p1r, p1i);
    auto t22 = bsk::make_tup(d, (0.0f * d));
    return bsk::make_tup(t00, t01, t02, t12, t20, t21, t22);
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

// Send the cotangent on one pulse's rotation to its row.
//
// Summed over the dephasing orders first: the pair multiplies every one of
// them, so what reaches the row is the sum. The block is padded to a power of
// two and the orders past the last carry whatever the sweep left there, so
// the sum is taken over the orders that exist rather than over the block.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12>
BSK_HD auto _store_pair_gradient(const T0& grad_pair, const T1& pair_index, const T2& event_base, const T3& event, const T4& atom, const T5& atom_count, const T6& turning, const T7& mask, const T8& state_mask, const T9& grad_ar, const T10& grad_ai, const T11& grad_br, const T12& grad_bi) {
    auto row = bsk::cast<std::int64_t>(bsk::ld(((pair_index + event_base) + event)));
    auto entry = (((row * atom_count) + atom) * 4);
    auto keep = bsk::band(turning, state_mask);
    bsk::atomic_add(((grad_pair + entry) + 0), bsk::sum_x(bsk::where(keep, grad_ar, 0.0f)), mask);
    bsk::atomic_add(((grad_pair + entry) + 1), bsk::sum_x(bsk::where(keep, grad_ai, 0.0f)), mask);
    bsk::atomic_add(((grad_pair + entry) + 2), bsk::sum_x(bsk::where(keep, grad_br, 0.0f)), mask);
    bsk::atomic_add(((grad_pair + entry) + 3), bsk::sum_x(bsk::where(keep, grad_bi, 0.0f)), mask);
}

// What one interval's cotangents give its length and its attenuation.
//
// The generator is proportional to the interval, so ``dE/d(dt) == A1 E`` and
// the length's gradient needs the operator and 27 multiplies rather than the
// eigenvalues -- which is what lets every other gradient be pooled over the
// events that share a length while this one stays per event.
//
// Returns the two contractions, in the order
// :func:`_three_pool_step_adjoint_jvp` returns them.
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10, class T11, class T12, class T13, class T14, class T15, class T16, class T17, class T18, class T19, class T20, class T21, class T22, class T23, class T24>
BSK_HD auto _three_pool_interval_adjoint(const T0& table, const T1& row, const T2& atom, const T3& voxel_count, const T4& mask, const T5& r1_free, const T6& r1_pool_b, const T7& r1_bound, const T8& exchange_b, const T9& exchange_c, const T10& fraction_b, const T11& fraction_c, const T12& attenuation, const T13& b11, const T14& b12, const T15& b13, const T16& b21, const T17& b22, const T18& b23, const T19& b31, const T20& b32, const T21& b33, const T22& bfree, const T23& bpool_b, const T24& bbound) {
    auto free = ((1.0f - fraction_b) - fraction_c);
    auto a00 = ((((-exchange_b) * fraction_b) - (exchange_c * fraction_c)) - r1_free);
    auto a01 = (exchange_b * free);
    auto a02 = (exchange_c * free);
    auto a10 = (exchange_b * fraction_b);
    auto a11 = (((-exchange_b) * free) - r1_pool_b);
    auto a20 = (exchange_c * fraction_c);
    auto a22 = (((-exchange_c) * free) - r1_bound);
    auto base = ((table + (row * (9 * voxel_count))) + atom);
    auto c00 = bsk::ld((base + (0 * voxel_count)), mask, 0.0f);
    auto c01 = bsk::ld((base + (1 * voxel_count)), mask, 0.0f);
    auto c02 = bsk::ld((base + (2 * voxel_count)), mask, 0.0f);
    auto c10 = bsk::ld((base + (3 * voxel_count)), mask, 0.0f);
    auto c11 = bsk::ld((base + (4 * voxel_count)), mask, 0.0f);
    auto c12 = bsk::ld((base + (5 * voxel_count)), mask, 0.0f);
    auto c20 = bsk::ld((base + (6 * voxel_count)), mask, 0.0f);
    auto c21 = bsk::ld((base + (7 * voxel_count)), mask, 0.0f);
    auto c22 = bsk::ld((base + (8 * voxel_count)), mask, 0.0f);
    // A1 C, the second and third rows of A1 having no entry off their own pool.
    auto p00 = (((a00 * c00) + (a01 * c10)) + (a02 * c20));
    auto p01 = (((a00 * c01) + (a01 * c11)) + (a02 * c21));
    auto p02 = (((a00 * c02) + (a01 * c12)) + (a02 * c22));
    auto p10 = ((a10 * c00) + (a11 * c10));
    auto p11 = ((a10 * c01) + (a11 * c11));
    auto p12 = ((a10 * c02) + (a11 * c12));
    auto p20 = ((a20 * c00) + (a22 * c20));
    auto p21 = ((a20 * c01) + (a22 * c21));
    auto p22 = ((a20 * c02) + (a22 * c22));
    auto grad_dt = (attenuation * _three_pool_contract(p00, p01, p02, p10, p11, p12, p20, p21, p22, b11, b12, b13, b21, b22, b23, b31, b32, b33, bfree, bpool_b, bbound, free, fraction_b, fraction_c));
    auto grad_att = _three_pool_contract(c00, c01, c02, c10, c11, c12, c20, c21, c22, b11, b12, b13, b21, b22, b23, b31, b32, b33, bfree, bpool_b, bbound, free, fraction_b, fraction_c);
    return bsk::make_tup(grad_dt, grad_att);
}

BSK_HD void _epg_vjp_kernel(float* t1, float* t2, float* m0, float* b1, float* b1_phase, float* b0, float* inversion_efficiency, float* diffusion, float* velocity, float* bound_fraction, float* exchange_rate, float* t1_bound, float* pool_b_fraction, float* pool_b_exchange, float* t1_pool_b, float* t2_pool_b, float* pool_b_shift, float* duration, std::int32_t* kind, float* flip, float* phase, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* saturation, float* rf_frequency, float* lineshape, float* profile, std::int32_t* profile_index, float* pairs, std::int32_t* pair_index, std::int32_t* duration_row, float* pool_table, float* pool_bars, float* pool_durations, bsk::index_t row_count, float* grad_pair, float* grad_output_real, float* grad_output_imag, float* grad_tissue, float* grad_flip, float* grad_phase, float* grad_duration, float* trajectory_r, float* trajectory_i, bsk::index_t problem_base, bsk::index_t problem_end, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, float flow_scale, float washout_scale, bsk::index_t shim_rows, float profile_step, float lineshape_step, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shimmed, bsk::index_t locations, bsk::index_t profiled, bsk::index_t profile_bins, bsk::index_t dynamic, bsk::index_t broadened, bsk::index_t lineshape_bins, bsk::index_t pools, bsk::index_t narrow, bsk::index_t tabulated, bsk::index_t off_axis, bsk::index_t moving, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t recording, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 2> _d11{};
    bsk::V<float, 2> _d12{};
    bsk::V<float, 2> _d21{};
    bsk::V<float, 2> _d22{};
    bsk::V<double, 2> _dgb{};
    bsk::V<double, 2> _dgf{};
    bsk::V<double, 2> _dgs{};
    bsk::V<float, 2> _drb{};
    bsk::V<float, 2> _drf{};
    bsk::V<double, 2> _dw11{};
    bsk::V<double, 2> _dw12{};
    bsk::V<double, 2> _dw13{};
    bsk::V<double, 2> _dw21{};
    bsk::V<double, 2> _dw22{};
    bsk::V<double, 2> _dw23{};
    bsk::V<double, 2> _dw31{};
    bsk::V<double, 2> _dw32{};
    bsk::V<double, 2> _dw33{};
    bsk::V<float, 2> _q1{};
    bsk::V<float, 2> _q2{};
    bsk::V<float, 2> _q3{};
    bsk::V<float, 2> _q4{};
    bsk::V<float, 2> _q5{};
    bsk::V<float, 2> _q6{};
    bsk::V<float, 2> _q7{};
    bsk::V<float, 2> _q8{};
    bsk::V<float, 2> _q9{};
    bsk::V<float, 2> a11i{};
    bsk::V<float, 2> a11r{};
    bsk::V<float, 2> a12i{};
    bsk::V<float, 2> a12r{};
    bsk::V<float, 2> a21i{};
    bsk::V<float, 2> a21r{};
    bsk::V<float, 2> a22i{};
    bsk::V<float, 2> a22r{};
    bsk::V<float, 2> absorbed_value{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> across{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> add1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> add2{};
    bsk::V<float, 3> alpha_v{};
    bsk::V<float, 2> alpha_value{};
    bsk::V<float, 3> angle_value{};
    bsk::V<float, 2> atom_b0{};
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_b1_phase{};
    bsk::V<float, 2> atom_bound{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_exchange{};
    bsk::V<float, 2> atom_flow{};
    bsk::V<float, 2> atom_free{};
    bsk::V<float, 2> atom_inv{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 2> atom_semisolid{};
    bsk::V<float, 2> atom_semisolid_exchange{};
    bsk::V<float, 2> atom_shift{};
    bsk::V<float, 2> atom_t1b{};
    bsk::V<float, 2> atom_t1c{};
    bsk::V<float, 2> atom_t2b{};
    bsk::V<float, 2> atom_washout{};
    bsk::V<float, 2> attenuation_v{};
    bsk::V<float, 3> avi{};
    bsk::V<float, 3> avr{};
    bsk::V<float, 2> back_att{};
    bsk::V<float, 2> back_bound{};
    bsk::V<float, 2> back_dt{};
    bsk::V<float, 2> back_exch{};
    bsk::V<float, 3> back_i{};
    bsk::V<float, 3> back_mi{};
    bsk::V<float, 3> back_mr{};
    bsk::V<float, 3> back_pi{};
    bsk::V<float, 3> back_pr{};
    bsk::V<float, 3> back_r{};
    bsk::V<float, 2> back_r1{};
    bsk::V<float, 2> back_r1b{};
    bsk::V<float, 2> back_r1c{};
    bsk::V<float, 2> back_semi{};
    bsk::V<float, 2> back_sexch{};
    bsk::V<float, 3> back_zi{};
    bsk::V<float, 3> back_zr{};
    bsk::V<float, 3> bare1_value{};
    bsk::V<float, 3> bare2_value{};
    bsk::V<float, 3> bare_cot_v{};
    std::int32_t base_row{};
    bsk::V<float, 3> bmvi{};
    bsk::V<float, 3> bmvr{};
    bsk::V<float, 3> bpvi{};
    bsk::V<float, 3> bpvr{};
    bsk::V<float, 3> bvi{};
    bsk::V<float, 3> bvr{};
    bsk::V<float, 3> cari{};
    bsk::V<float, 3> carr{};
    bsk::V<float, 3> col_bi{};
    bsk::V<float, 3> col_br{};
    bsk::V<float, 3> col_ci{};
    bsk::V<float, 3> col_cr{};
    bsk::V<float, 3> col_fi{};
    bsk::V<float, 3> col_fr{};
    bsk::V<float, 2> cos_value{};
    bsk::V<float, 3> cot2_v{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 2> direction{};
    bool do_shift{};
    bsk::V<float, 2> dt_value{};
    bsk::V<float, 3> duration_v{};
    bsk::V<float, 3> e1_v{};
    bsk::V<float, 3> e1_value{};
    bsk::V<float, 3> e2_value{};
    std::int64_t event{};
    std::int32_t event_action{};
    bsk::V<float, 2> event_flip{};
    std::int32_t event_kind{};
    bsk::V<float, 2> event_phase{};
    float event_saturation{};
    bsk::V<float, 3> f11i{};
    bsk::V<float, 3> f11r{};
    bsk::V<float, 3> f12i{};
    bsk::V<float, 3> f12r{};
    bsk::V<float, 3> g21i{};
    bsk::V<float, 3> g21r{};
    bsk::V<float, 3> g22i{};
    bsk::V<float, 3> g22r{};
    bsk::V<float, 2> g_b0v{};
    bsk::V<float, 2> g_b1pv{};
    bsk::V<float, 2> g_b1v{};
    bsk::V<float, 2> g_boundv{};
    bsk::V<float, 2> g_diffv{};
    bsk::V<float, 2> g_exchv{};
    bsk::V<float, 2> g_flowv{};
    bsk::V<float, 2> g_invv{};
    bsk::V<float, 2> g_m0v{};
    bsk::V<float, 2> g_semiv{};
    bsk::V<float, 2> g_sexchv{};
    bsk::V<float, 2> g_shiftv{};
    bsk::V<float, 2> g_t1bv{};
    bsk::V<float, 2> g_t1cv{};
    bsk::V<float, 3> g_t1v{};
    bsk::V<float, 2> g_t2bv{};
    bsk::V<float, 3> g_t2v{};
    bsk::V<float, 2> g_washv{};
    bsk::V<float, 2> grad_alpha_v{};
    bsk::V<float, 2> grad_angle_v{};
    bsk::V<float, 2> grad_e1_v{};
    bsk::V<float, 2> grow_free{};
    bsk::V<float, 2> grow_pool_b{};
    bsk::V<float, 2> grow_semisolid{};
    bsk::V<float, 3> h11i{};
    bsk::V<float, 3> h11r{};
    bsk::V<float, 3> h12i{};
    bsk::V<float, 3> h12r{};
    bsk::V<float*, 2> held{};
    bsk::V<float, 2> hold_value{};
    bool invert{};
    bool is_inversion{};
    bool is_rf{};
    bsk::V<float, 3> k21i{};
    bsk::V<float, 3> k21r{};
    bsk::V<float, 3> k22i{};
    bsk::V<float, 3> k22r{};
    bsk::V<float, 3> long_damp_v{};
    bsk::V<float, 3> lvi{};
    bsk::V<float, 3> lvr{};
    bsk::V<float, 3> mbvi{};
    bsk::V<float, 3> mbvr{};
    bsk::V<float, 3> mix_bi{};
    bsk::V<float, 3> mix_br{};
    bsk::V<float, 3> mix_ci{};
    bsk::V<float, 3> mix_cr{};
    bsk::V<float, 3> mix_fi{};
    bsk::V<float, 3> mix_fr{};
    bsk::V<float, 3> mvi{};
    bsk::V<float, 3> mvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> n0{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> n1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> n2{};
    bsk::V<float, 2> nil{};
    bsk::V<float, 2> offset_value{};
    bsk::V<float, 3> ovi{};
    bsk::V<float, 3> ovr{};
    bsk::V<float, 2> p1i{};
    bsk::V<float, 2> p1r{};
    bsk::V<float, 2> p2i{};
    bsk::V<float, 2> p2r{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>, bsk::V<float, 2>> pair{};
    bsk::V<float, 3> part_i{};
    bsk::V<float, 3> part_r{};
    bsk::V<float, 3> pbvi{};
    bsk::V<float, 3> pbvr{};
    bsk::V<float, 2> pe11{};
    bsk::V<float, 2> pe12{};
    bsk::V<float, 2> pe21{};
    bsk::V<float, 2> pe22{};
    bsk::V<float, 3> per_angle_v{};
    bsk::V<float, 3> per_state{};
    bsk::V<float, 3> phi_v{};
    bsk::V<float, 2> phi_value{};
    bsk::V<float, 3> pool_angle_v{};
    bsk::V<float, 3> pool_back_mi{};
    bsk::V<float, 3> pool_back_mr{};
    bsk::V<float, 3> pool_back_pi{};
    bsk::V<float, 3> pool_back_pr{};
    bsk::V<float, 3> pool_back_zi{};
    bsk::V<float, 3> pool_back_zr{};
    bsk::V<std::int32_t, 2> pool_row{};
    bsk::V<float, 3> pool_shaped_mbi{};
    bsk::V<float, 3> pool_shaped_mbr{};
    bsk::V<float, 3> pool_shaped_pbi{};
    bsk::V<float, 3> pool_shaped_pbr{};
    bsk::V<float, 3> pool_shaped_zbi{};
    bsk::V<float, 3> pool_shaped_zbr{};
    bsk::V<float, 3> poolbi{};
    bsk::V<float, 3> poolbr{};
    bsk::V<float, 3> poolvi{};
    bsk::V<float, 3> poolvr{};
    bsk::V<float, 2> power_value{};
    bool pre_shift{};
    bsk::V<float, 2> prec_b{};
    bsk::V<float, 2> prec_f{};
    bsk::V<std::int32_t, 2> problem{};
    bsk::V<float, 2> pulse_b1{};
    bsk::V<float, 2> pulse_b1_phase{};
    bsk::V<float, 3> pvi{};
    bsk::V<float, 3> pvr{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> q0{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> q1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> q2{};
    bsk::V<float, 3> qi{};
    bsk::V<float, 3> qr{};
    bsk::V<float, 2> r1b_value{};
    bsk::V<float, 2> r1c_value{};
    bsk::V<float, 2> r2b_value{};
    bsk::V<float, 3> rbmvi{};
    bsk::V<float, 3> rbmvr{};
    bsk::V<float, 3> rbpvi{};
    bsk::V<float, 3> rbpvr{};
    bsk::V<float, 3> rbvi{};
    bsk::V<float, 3> rbvr{};
    bsk::V<float, 3> rcvi{};
    bsk::V<float, 3> rcvr{};
    bsk::V<float, 3> reci{};
    bsk::V<float, 3> recovery_value{};
    bsk::V<float, 3> recr{};
    bsk::V<float, 3> rmvi{};
    bsk::V<float, 3> rmvr{};
    bool rotate{};
    std::int64_t row{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> row0{};
    bsk::V<float, 3> rpvi{};
    bsk::V<float, 3> rpvr{};
    bsk::V<float, 3> rzvi{};
    bsk::V<float, 3> rzvr{};
    bsk::V<float, 2> sat_alpha_v{};
    bsk::V<float, 2> sat_b0_v{};
    bool saturating{};
    bsk::V<float, 3> sbi{};
    bsk::V<float, 3> sbmvi{};
    bsk::V<float, 3> sbmvr{};
    bsk::V<float, 3> sbpvi{};
    bsk::V<float, 3> sbpvr{};
    bsk::V<float, 3> sbr{};
    bsk::V<float, 3> semibi{};
    bsk::V<float, 3> semibr{};
    bsk::V<float, 3> semivi{};
    bsk::V<float, 3> semivr{};
    bsk::V<float, 3> sfi{};
    bsk::V<float, 3> sfr{};
    bsk::V<float, 2> shape_value{};
    bsk::V<float, 2> shaped_ai{};
    bsk::V<float, 2> shaped_ar{};
    bsk::V<float, 2> shaped_bi{};
    bsk::V<float, 2> shaped_br{};
    bsk::V<float, 3> shaped_mbi{};
    bsk::V<float, 3> shaped_mbr{};
    bsk::V<float, 3> shaped_pbi{};
    bsk::V<float, 3> shaped_pbr{};
    bsk::V<float, 3> shaped_zbi{};
    bsk::V<float, 3> shaped_zbr{};
    bsk::V<float, 2> sin_value{};
    bsk::V<float, 2> slope_ai{};
    bsk::V<float, 2> slope_ar{};
    bsk::V<float, 2> slope_bi{};
    bsk::V<float, 2> slope_br{};
    bsk::V<std::int32_t, 3> slot{};
    bsk::V<float, 3> spin_i{};
    bsk::V<float, 3> spin_r{};
    bool spoil{};
    bsk::V<float, 2> spread_v{};
    bsk::V<float, 3> spun_bi{};
    bsk::V<float, 3> spun_br{};
    bsk::V<float, 3> spun_fi{};
    bsk::V<float, 3> spun_fr{};
    bsk::V<float, 3> spun_mi{};
    bsk::V<float, 3> spun_mr{};
    bsk::V<float, 3> spun_pi{};
    bsk::V<float, 3> spun_pr{};
    bsk::V<float, 3> spun_zi{};
    bsk::V<float, 3> spun_zr{};
    bsk::V<float, 3> svi{};
    bsk::V<float, 3> svr{};
    bsk::V<float, 3> szi{};
    bsk::V<float, 3> szr{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t00{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t01{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t02{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t12{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t20{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t21{};
    bsk::tup<bsk::V<float, 2>, bsk::V<float, 2>> t22{};
    bsk::V<double, 2> three_a00{};
    bsk::V<double, 2> three_a01{};
    bsk::V<double, 2> three_a02{};
    bsk::V<double, 2> three_a10{};
    bsk::V<double, 2> three_a11{};
    bsk::V<double, 2> three_a20{};
    bsk::V<double, 2> three_a22{};
    bsk::V<double, 2> three_angle{};
    bsk::V<double, 2> three_argument{};
    bsk::V<double, 2> three_centre{};
    bsk::V<double, 2> three_cube{};
    bsk::V<double, 2> three_d_a00{};
    bsk::V<double, 2> three_d_a01{};
    bsk::V<double, 2> three_d_a02{};
    bsk::V<double, 2> three_d_a10{};
    bsk::V<double, 2> three_d_a11{};
    bsk::V<double, 2> three_d_a20{};
    bsk::V<double, 2> three_d_a22{};
    bsk::V<double, 2> three_d_angle{};
    bsk::V<double, 2> three_d_centre{};
    bsk::V<double, 2> three_d_determinant{};
    bsk::V<double, 2> three_d_first{};
    bsk::V<double, 2> three_d_free{};
    bsk::V<double, 2> three_d_guarded{};
    bsk::V<double, 2> three_d_high{};
    bsk::V<double, 2> three_d_leading{};
    bsk::V<double, 2> three_d_lift{};
    bsk::V<double, 2> three_d_low{};
    bsk::V<double, 2> three_d_middle{};
    bsk::V<double, 2> three_d_minors{};
    bsk::V<double, 2> three_d_pool_b{};
    bsk::V<double, 2> three_d_pool_c{};
    bsk::V<double, 2> three_d_q00{};
    bsk::V<double, 2> three_d_q01{};
    bsk::V<double, 2> three_d_q02{};
    bsk::V<double, 2> three_d_q10{};
    bsk::V<double, 2> three_d_q11{};
    bsk::V<double, 2> three_d_q12{};
    bsk::V<double, 2> three_d_q20{};
    bsk::V<double, 2> three_d_q21{};
    bsk::V<double, 2> three_d_q22{};
    bsk::V<double, 2> three_d_radius{};
    bsk::V<double, 2> three_d_raw{};
    bsk::V<double, 2> three_d_s00{};
    bsk::V<double, 2> three_d_s11{};
    bsk::V<double, 2> three_d_s22{};
    bsk::V<double, 2> three_d_second{};
    bsk::V<double, 2> three_d_sum_flat{};
    bsk::V<double, 2> three_d_sum_linear{};
    bsk::V<double, 2> three_d_sum_square{};
    bsk::V<double, 2> three_d_trailing{};
    bsk::V<double, 2> three_def_00{};
    bsk::V<double, 2> three_def_01{};
    bsk::V<double, 2> three_def_02{};
    bsk::V<double, 2> three_def_10{};
    bsk::V<double, 2> three_def_11{};
    bsk::V<double, 2> three_def_12{};
    bsk::V<double, 2> three_def_20{};
    bsk::V<double, 2> three_def_21{};
    bsk::V<double, 2> three_def_22{};
    bsk::V<double, 2> three_determinant{};
    bsk::V<double, 2> three_dif_00{};
    bsk::V<double, 2> three_dif_01{};
    bsk::V<double, 2> three_dif_02{};
    bsk::V<double, 2> three_dif_10{};
    bsk::V<double, 2> three_dif_11{};
    bsk::V<double, 2> three_dif_12{};
    bsk::V<double, 2> three_dif_20{};
    bsk::V<double, 2> three_dif_21{};
    bsk::V<double, 2> three_dif_22{};
    bsk::V<double, 2> three_first{};
    bsk::V<double, 2> three_free{};
    bsk::V<double, 2> three_guarded{};
    bsk::V<double, 2> three_high{};
    bsk::V<bool, 2> three_inside_limit{};
    bsk::V<double, 2> three_leading{};
    bsk::V<double, 2> three_lift{};
    bsk::V<double, 2> three_low{};
    bsk::V<double, 2> three_middle{};
    bsk::V<double, 2> three_minors{};
    bsk::V<double, 2> three_pool_b{};
    bsk::V<double, 2> three_pool_c{};
    bsk::V<double, 2> three_q00{};
    bsk::V<double, 2> three_q01{};
    bsk::V<double, 2> three_q02{};
    bsk::V<double, 2> three_q10{};
    bsk::V<double, 2> three_q11{};
    bsk::V<double, 2> three_q12{};
    bsk::V<double, 2> three_q20{};
    bsk::V<double, 2> three_q21{};
    bsk::V<double, 2> three_q22{};
    bsk::V<double, 2> three_radius{};
    bsk::V<double, 2> three_raw{};
    bsk::V<double, 2> three_s00{};
    bsk::V<double, 2> three_s11{};
    bsk::V<double, 2> three_s22{};
    bsk::V<double, 2> three_second{};
    bsk::V<double, 2> three_sum_flat{};
    bsk::V<double, 2> three_sum_linear{};
    bsk::V<double, 2> three_sum_square{};
    bsk::V<double, 2> three_trailing{};
    bsk::V<float, 3> turn_t{};
    bsk::V<float, 3> turn_z{};
    bsk::V<float, 3> turned_mi{};
    bsk::V<float, 3> turned_mr{};
    bsk::V<float, 3> turned_pi{};
    bsk::V<float, 3> turned_pr{};
    bsk::V<float, 3> turned_zi{};
    bsk::V<float, 3> turned_zr{};
    bsk::V<float, 2> two_pool_dt_v{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> u1{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> u2{};
    bsk::V<float, 3> ubvi{};
    bsk::V<float, 3> ubvr{};
    bsk::V<float, 3> ui{};
    bsk::V<float, 3> ur{};
    bsk::V<float, 3> vi_{};
    bsk::V<float, 3> vr_{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> w0{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> w1{};
    bsk::V<float, 2> w11{};
    bsk::V<float, 2> w12{};
    bsk::V<float, 2> w13{};
    bsk::tup<bsk::V<float, 3>, bsk::V<float, 3>> w2{};
    bsk::V<float, 2> w21{};
    bsk::V<float, 2> w22{};
    bsk::V<float, 2> w23{};
    bsk::V<float, 2> w31{};
    bsk::V<float, 2> w32{};
    bsk::V<float, 2> w33{};
    bsk::V<float, 2> wash_v{};
    bsk::V<float, 3> wbvi{};
    bsk::V<float, 3> wbvr{};
    bsk::V<float, 2> wound_v{};
    bsk::V<float, 2> wout_value{};
    bsk::V<float, 3> wvi{};
    bsk::V<float, 3> wvr{};
    bsk::V<float, 3> xbmvi{};
    bsk::V<float, 3> xbmvr{};
    bsk::V<float, 3> xbpvi{};
    bsk::V<float, 3> xbpvr{};
    bsk::V<float, 3> xbvi{};
    bsk::V<float, 3> xbvr{};
    bsk::V<float, 3> xcvi{};
    bsk::V<float, 3> xcvr{};
    bsk::V<float, 2> xversal_att{};
    bsk::V<float, 2> xversal_dt{};
    bsk::V<float, 3> yi{};
    bsk::V<float, 3> yr{};
    bsk::V<float, 3> zangle_v{};
    bsk::V<float, 3> zbvi{};
    bsk::V<float, 3> zbvr{};
    bsk::V<float, 3> zvi{};
    bsk::V<float, 3> zvr{};
    problem = (problem_base + (bsk::program_id(0) * problems));
    problem = (problem + bsk::arange_y());
    auto state = bsk::arange_x();
    auto active_atom = (problem < problem_end);
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    auto local = (problem - problem_base);
    auto record_stride = (bsk::select(bsk::truth((pools == 3)), 7, bsk::select(bsk::truth((pools == 2)), 6, bsk::select(bsk::truth((pools == 1)), 4, 3))) * state_count);
    auto trajectory = (((local * event_count) * record_stride) + state);
    auto minus_plane = state_count;
    auto long_plane = (2 * state_count);
    auto bound_plane = (3 * state_count);
    auto bplus_plane = (4 * state_count);
    auto bminus_plane = (5 * state_count);
    auto semisolid_plane = (6 * state_count);
    auto empty = bsk::full<float, 3>(0);
    pvr = empty;
    pvi = empty;
    mvr = empty;
    mvi = empty;
    zvr = (empty + bsk::where((state == 0), 1.0f, 0.0f));
    zvi = empty;
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_b1_phase = 0.0f;
    atom_b0 = 0.0f;
    if (bsk::truth(off_axis)) {
        atom_b1_phase = bsk::ld((b1_phase + scalar_atom), active_atom, 0.0f);
        atom_b0 = bsk::ld((b0 + scalar_atom), active_atom, 0.0f);
    }
    atom_inv = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inv = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    atom_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
    }
    atom_flow = 0.0f;
    direction = 0.0f;
    atom_washout = 0.0f;
    if (bsk::truth(moving)) {
        auto atom_velocity = bsk::ld((velocity + scalar_atom), active_atom, 0.0f);
        atom_flow = (atom_velocity * flow_scale);
        // |v| has no derivative at the origin, so a still voxel contributes
        // none.
        direction = (bsk::cast<float>((atom_velocity > 0.0f)) - bsk::cast<float>((atom_velocity < 0.0f)));
        atom_washout = (bsk::abs(atom_velocity) * washout_scale);
    }
    auto order = bsk::cast<float>(state);
    auto longitudinal_weight = (order * order);
    auto transverse_weight = ((longitudinal_weight + order) + 0.3333333333333333f);
    auto r1_value = bsk::truediv(1000.0f, atom_t1);
    auto r2_value = bsk::truediv(1000.0f, atom_t2);
    auto location = bsk::mod(atom, locations);
    // A semisolid pool rides along as a plane of its own: the pulse deposits
    // into it and it exchanges with the free water, so the reverse sweep cannot
    // replay it from the free pool's.
    atom_bound = 0.0f;
    atom_exchange = 0.0f;
    atom_t1b = 1.0f;
    atom_t2b = 1.0f;
    atom_shift = 0.0f;
    r1b_value = 0.0f;
    r2b_value = 0.0f;
    atom_semisolid = 0.0f;
    atom_semisolid_exchange = 0.0f;
    atom_t1c = 1.0f;
    r1c_value = 0.0f;
    atom_free = 1.0f;
    poolvr = empty;
    poolvi = empty;
    bpvr = empty;
    bpvi = empty;
    bmvr = empty;
    bmvi = empty;
    semivr = empty;
    semivi = empty;
    if (bsk::truth((pools == 1))) {
        atom_bound = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((exchange_rate + scalar_atom), active_atom, 0.0f);
        atom_t1b = bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f);
        r1b_value = bsk::truediv(1000.0f, atom_t1b);
    }
    if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
        atom_bound = bsk::ld((pool_b_fraction + scalar_atom), active_atom, 0.0f);
        atom_exchange = bsk::ld((pool_b_exchange + scalar_atom), active_atom, 0.0f);
        atom_t1b = bsk::ld((t1_pool_b + scalar_atom), active_atom, 1.0f);
        r1b_value = bsk::truediv(1000.0f, atom_t1b);
        atom_t2b = bsk::ld((t2_pool_b + scalar_atom), active_atom, 1.0f);
        r2b_value = bsk::truediv(1000.0f, atom_t2b);
        atom_shift = bsk::ld((pool_b_shift + scalar_atom), active_atom, 0.0f);
    }
    if (bsk::truth((pools == 3))) {
        // The semisolid pool takes the rows a run with it alone would take, so
        // the two second pools never contend for one.
        atom_semisolid = bsk::ld((bound_fraction + scalar_atom), active_atom, 0.0f);
        atom_semisolid_exchange = bsk::ld((exchange_rate + scalar_atom), active_atom, 0.0f);
        atom_t1c = bsk::ld((t1_bound + scalar_atom), active_atom, 1.0f);
        r1c_value = bsk::truediv(1000.0f, atom_t1c);
        semivr = (empty + bsk::where((state == 0), (atom_semisolid + 0.0f), 0.0f));
    }
    if (bsk::truth((pools > 0))) {
        // The fractions split the equilibrium at t = 0.
        atom_free = ((1.0f - atom_bound) - atom_semisolid);
        zvr = (empty + bsk::where((state == 0), atom_free, 0.0f));
        poolvr = (empty + bsk::where((state == 0), (atom_bound + 0.0f), 0.0f));
    }
    auto event_base = (train * event_count);
    // The forward half records the trajectory the reverse half walks back,
    // and the two are launched separately: each compiles the sweep it is
    // asked for and no more.
    if (bsk::truth(recording)) {
        for (bsk::index_t event = 0; event < event_count; event += 1) {
            slot = (trajectory + (event * record_stride));
            bsk::st((trajectory_r + slot), pvr, state_mask);
            bsk::st((trajectory_i + slot), pvi, state_mask);
            bsk::st(((trajectory_r + slot) + minus_plane), mvr, state_mask);
            bsk::st(((trajectory_i + slot) + minus_plane), mvi, state_mask);
            bsk::st(((trajectory_r + slot) + long_plane), zvr, state_mask);
            bsk::st(((trajectory_i + slot) + long_plane), zvi, state_mask);
            if (bsk::truth((pools > 0))) {
                bsk::st(((trajectory_r + slot) + bound_plane), poolvr, state_mask);
                bsk::st(((trajectory_i + slot) + bound_plane), poolvi, state_mask);
            }
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                bsk::st(((trajectory_r + slot) + bplus_plane), bpvr, state_mask);
                bsk::st(((trajectory_i + slot) + bplus_plane), bpvi, state_mask);
                bsk::st(((trajectory_r + slot) + bminus_plane), bmvr, state_mask);
                bsk::st(((trajectory_i + slot) + bminus_plane), bmvi, state_mask);
            }
            if (bsk::truth((pools == 3))) {
                bsk::st(((trajectory_r + slot) + semisolid_plane), semivr, state_mask);
                bsk::st(((trajectory_i + slot) + semisolid_plane), semivi, state_mask);
            }
            dt_value = _event_value(duration, event_base, event, active_atom, single_train);
            wout_value = 1.0f;
            if (bsk::truth(moving)) {
                wout_value = _washout(atom_washout, dt_value);
            }
            e1_value = (bsk::exp(((-r1_value) * dt_value)) * wout_value);
            e2_value = (bsk::exp(((-r2_value) * dt_value)) * wout_value);
            damp_z = 1.0f;
            damp_t = 1.0f;
            if (bsk::truth(diffusing)) {
                auto t0_ = _damping(atom_damping, dt_value, order);
                damp_z = bsk::get<0>(t0_);
                damp_t = bsk::get<1>(t0_);
            }
            // Order zero is undamped, so recovery keeps the bare longitudinal factor.
            recovery_value = (1.0f - e1_value);
            bare1_value = e1_value;
            bare2_value = e2_value;
            e1_value = (bare1_value * damp_z);
            e2_value = (bare2_value * damp_t);
            turn_t = 0.0f;
            auto t1_ = bsk::make_tup(1.0f, 0.0f);
            szr = bsk::get<0>(t1_);
            szi = bsk::get<1>(t1_);
            if (bsk::truth(moving)) {
                auto t2_ = _flow(atom_flow, dt_value, order);
                turn_z = bsk::get<0>(t2_);
                turn_t = bsk::get<1>(t2_);
                auto t3_ = bsk::make_tup(bsk::cos(turn_z), bsk::sin(turn_z));
                szr = bsk::get<0>(t3_);
                szi = bsk::get<1>(t3_);
            }
            auto t4_ = bsk::make_tup(1.0f, 0.0f);
            qr = bsk::get<0>(t4_);
            qi = bsk::get<1>(t4_);
            if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
                angle_value = ((-6.283185307179586f * (atom_b0 * dt_value)) + turn_t);
                auto t5_ = bsk::make_tup(bsk::cos(angle_value), bsk::sin(angle_value));
                qr = bsk::get<0>(t5_);
                qi = bsk::get<1>(t5_);
            }
            auto t6_ = bsk::make_tup((e2_value * qr), (e2_value * qi));
            ovr = bsk::get<0>(t6_);
            ovi = bsk::get<1>(t6_);
            auto t7_ = bsk::make_tup((e1_value * szr), (e1_value * szi));
            lvr = bsk::get<0>(t7_);
            lvi = bsk::get<1>(t7_);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                // With an exchanging pool the transverse relaxation sits inside the
                // operator instead of in the scalar the free pool alone multiplies.
                across = _two_pool_transverse_step_jvp(r2_value, 0.0f, r2b_value, 0.0f, atom_exchange, 0.0f, atom_bound, 0.0f, atom_free, 0.0f, atom_shift, 0.0f, dt_value, 0.0f, wout_value, 0.0f);
                auto t8_ = bsk::make_tup(bsk::get<0>(across), bsk::get<1>(across));
                a11r = bsk::get<0>(t8_);
                a11i = bsk::get<1>(t8_);
                auto t9_ = bsk::make_tup(bsk::get<2>(across), bsk::get<3>(across));
                a12r = bsk::get<0>(t9_);
                a12i = bsk::get<1>(t9_);
                auto t10_ = bsk::make_tup(bsk::get<4>(across), bsk::get<5>(across));
                a21r = bsk::get<0>(t10_);
                a21i = bsk::get<1>(t10_);
                auto t11_ = bsk::make_tup(bsk::get<6>(across), bsk::get<7>(across));
                a22r = bsk::get<0>(t11_);
                a22i = bsk::get<1>(t11_);
                auto t12_ = bsk::make_tup((damp_t * qr), (damp_t * qi));
                carr = bsk::get<0>(t12_);
                cari = bsk::get<1>(t12_);
                auto t13_ = _complex_mul(a11r, a11i, pvr, pvi);
                f11r = bsk::get<0>(t13_);
                f11i = bsk::get<1>(t13_);
                auto t14_ = _complex_mul(a12r, a12i, bpvr, bpvi);
                f12r = bsk::get<0>(t14_);
                f12i = bsk::get<1>(t14_);
                auto t15_ = _complex_mul(a21r, a21i, pvr, pvi);
                g21r = bsk::get<0>(t15_);
                g21i = bsk::get<1>(t15_);
                auto t16_ = _complex_mul(a22r, a22i, bpvr, bpvi);
                g22r = bsk::get<0>(t16_);
                g22i = bsk::get<1>(t16_);
                // ``F-`` takes the conjugate of the operator entry by entry, not its
                // transpose: it is the conjugate state following the conjugate map.
                auto t17_ = _complex_mul(a11r, (-a11i), mvr, mvi);
                h11r = bsk::get<0>(t17_);
                h11i = bsk::get<1>(t17_);
                auto t18_ = _complex_mul(a12r, (-a12i), bmvr, bmvi);
                h12r = bsk::get<0>(t18_);
                h12i = bsk::get<1>(t18_);
                auto t19_ = _complex_mul(a21r, (-a21i), mvr, mvi);
                k21r = bsk::get<0>(t19_);
                k21i = bsk::get<1>(t19_);
                auto t20_ = _complex_mul(a22r, (-a22i), bmvr, bmvi);
                k22r = bsk::get<0>(t20_);
                k22i = bsk::get<1>(t20_);
                auto t21_ = _complex_mul((f11r + f12r), (f11i + f12i), carr, cari);
                pvr = bsk::get<0>(t21_);
                pvi = bsk::get<1>(t21_);
                auto t22_ = _complex_mul((g21r + g22r), (g21i + g22i), carr, cari);
                bpvr = bsk::get<0>(t22_);
                bpvi = bsk::get<1>(t22_);
                auto t23_ = _complex_mul((h11r + h12r), (h11i + h12i), carr, (-cari));
                mvr = bsk::get<0>(t23_);
                mvi = bsk::get<1>(t23_);
                auto t24_ = _complex_mul((k21r + k22r), (k21i + k22i), carr, (-cari));
                bmvr = bsk::get<0>(t24_);
                bmvi = bsk::get<1>(t24_);
            } else {
                auto t25_ = _complex_mul(ovr, ovi, pvr, pvi);
                pvr = bsk::get<0>(t25_);
                pvi = bsk::get<1>(t25_);
                auto t26_ = _complex_mul(ovr, (-ovi), mvr, mvi);
                mvr = bsk::get<0>(t26_);
                mvi = bsk::get<1>(t26_);
            }
            if (bsk::truth((pools == 3))) {
                // Three pools mix through a 3x3 formed once for the interval; each
                // second pool exchanges with the free water and not with the other.
                nil = (0.0f * dt_value);
                hold_value = (wout_value + nil);
                if (bsk::truth(tabulated)) {
                    auto t27_ = _three_pool_from_table(pool_table, bsk::ld(((duration_row + event_base) + event), active_atom, 0), atom, atom_count, active_atom, hold_value, atom_free, atom_bound, atom_semisolid);
                    w11 = bsk::get<0>(t27_);
                    w12 = bsk::get<1>(t27_);
                    w13 = bsk::get<2>(t27_);
                    w21 = bsk::get<3>(t27_);
                    w22 = bsk::get<4>(t27_);
                    w23 = bsk::get<5>(t27_);
                    w31 = bsk::get<6>(t27_);
                    w32 = bsk::get<7>(t27_);
                    w33 = bsk::get<8>(t27_);
                    grow_free = bsk::get<9>(t27_);
                    grow_pool_b = bsk::get<10>(t27_);
                    grow_semisolid = bsk::get<11>(t27_);
                } else {
                    auto t28_ = _three_pool_step_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, dt_value, nil, hold_value, nil, narrow);
                    w11 = bsk::get<0>(t28_);
                    w12 = bsk::get<1>(t28_);
                    w13 = bsk::get<2>(t28_);
                    w21 = bsk::get<3>(t28_);
                    w22 = bsk::get<4>(t28_);
                    w23 = bsk::get<5>(t28_);
                    w31 = bsk::get<6>(t28_);
                    w32 = bsk::get<7>(t28_);
                    w33 = bsk::get<8>(t28_);
                    grow_free = bsk::get<9>(t28_);
                    grow_pool_b = bsk::get<10>(t28_);
                    grow_semisolid = bsk::get<11>(t28_);
                    _dw11 = bsk::get<12>(t28_);
                    _dw12 = bsk::get<13>(t28_);
                    _dw13 = bsk::get<14>(t28_);
                    _dw21 = bsk::get<15>(t28_);
                    _dw22 = bsk::get<16>(t28_);
                    _dw23 = bsk::get<17>(t28_);
                    _dw31 = bsk::get<18>(t28_);
                    _dw32 = bsk::get<19>(t28_);
                    _dw33 = bsk::get<20>(t28_);
                    _dgf = bsk::get<21>(t28_);
                    _dgb = bsk::get<22>(t28_);
                    _dgs = bsk::get<23>(t28_);
                }
                auto t29_ = bsk::make_tup((damp_z * szr), (damp_z * szi));
                spin_r = bsk::get<0>(t29_);
                spin_i = bsk::get<1>(t29_);
                mix_fr = (((w11 * zvr) + (w12 * poolvr)) + (w13 * semivr));
                mix_fi = (((w11 * zvi) + (w12 * poolvi)) + (w13 * semivi));
                mix_br = (((w21 * zvr) + (w22 * poolvr)) + (w23 * semivr));
                mix_bi = (((w21 * zvi) + (w22 * poolvi)) + (w23 * semivi));
                mix_cr = (((w31 * zvr) + (w32 * poolvr)) + (w33 * semivr));
                mix_ci = (((w31 * zvi) + (w32 * poolvi)) + (w33 * semivi));
                auto t30_ = _complex_mul(spin_r, spin_i, mix_fr, mix_fi);
                zvr = bsk::get<0>(t30_);
                zvi = bsk::get<1>(t30_);
                auto t31_ = _complex_mul(spin_r, spin_i, mix_br, mix_bi);
                poolvr = bsk::get<0>(t31_);
                poolvi = bsk::get<1>(t31_);
                auto t32_ = _complex_mul(spin_r, spin_i, mix_cr, mix_ci);
                semivr = bsk::get<0>(t32_);
                semivi = bsk::get<1>(t32_);
                zvr = (zvr + bsk::where((state == 0), grow_free, 0.0f));
                poolvr = (poolvr + bsk::where((state == 0), grow_pool_b, 0.0f));
                semivr = (semivr + bsk::where((state == 0), grow_semisolid, 0.0f));
            } else if (bsk::truth((pools > 0))) {
                // The pools exchange while they relax, so the longitudinal step is a
                // 2x2 the interval forms once and the per-order damping and turn
                // multiply. Read from the dual helper with no direction to follow:
                // what only its tangents reach, the compiler drops.
                auto t33_ = _two_pool_step_jvp(r1_value, 0.0f, r1b_value, 0.0f, atom_exchange, 0.0f, atom_bound, 0.0f, dt_value, 0.0f, wout_value, 0.0f);
                pe11 = bsk::get<0>(t33_);
                pe12 = bsk::get<1>(t33_);
                pe21 = bsk::get<2>(t33_);
                pe22 = bsk::get<3>(t33_);
                prec_f = bsk::get<4>(t33_);
                prec_b = bsk::get<5>(t33_);
                _d11 = bsk::get<6>(t33_);
                _d12 = bsk::get<7>(t33_);
                _d21 = bsk::get<8>(t33_);
                _d22 = bsk::get<9>(t33_);
                _drf = bsk::get<10>(t33_);
                _drb = bsk::get<11>(t33_);
                auto t34_ = bsk::make_tup((damp_z * szr), (damp_z * szi));
                spin_r = bsk::get<0>(t34_);
                spin_i = bsk::get<1>(t34_);
                mix_fr = ((pe11 * zvr) + (pe12 * poolvr));
                mix_fi = ((pe11 * zvi) + (pe12 * poolvi));
                mix_br = ((pe21 * zvr) + (pe22 * poolvr));
                mix_bi = ((pe21 * zvi) + (pe22 * poolvi));
                auto t35_ = _complex_mul(spin_r, spin_i, mix_fr, mix_fi);
                zvr = bsk::get<0>(t35_);
                zvi = bsk::get<1>(t35_);
                auto t36_ = _complex_mul(spin_r, spin_i, mix_br, mix_bi);
                poolvr = bsk::get<0>(t36_);
                poolvi = bsk::get<1>(t36_);
                zvr = (zvr + bsk::where((state == 0), prec_f, 0.0f));
                poolvr = (poolvr + bsk::where((state == 0), prec_b, 0.0f));
            } else {
                auto t37_ = _complex_mul(lvr, lvi, zvr, zvi);
                zvr = bsk::get<0>(t37_);
                zvi = bsk::get<1>(t37_);
                zvr = (zvr + bsk::where((state == 0), recovery_value, 0.0f));
            }
            event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
            pre_shift = (bsk::band(event_action, 1) != 0);
            auto t38_ = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count);
            svr = bsk::get<0>(t38_);
            svi = bsk::get<1>(t38_);
            wvr = bsk::get<2>(t38_);
            wvi = bsk::get<3>(t38_);
            pvr = bsk::where(pre_shift, svr, pvr);
            pvi = bsk::where(pre_shift, svi, pvi);
            mvr = bsk::where(pre_shift, wvr, mvr);
            mvi = bsk::where(pre_shift, wvi, mvi);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t39_ = _shift(bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count);
                svr = bsk::get<0>(t39_);
                svi = bsk::get<1>(t39_);
                wvr = bsk::get<2>(t39_);
                wvi = bsk::get<3>(t39_);
                bpvr = bsk::where(pre_shift, svr, bpvr);
                bpvi = bsk::where(pre_shift, svi, bpvi);
                bmvr = bsk::where(pre_shift, wvr, bmvr);
                bmvi = bsk::where(pre_shift, wvi, bmvi);
            }
            event_kind = bsk::ld((kind + event));
            is_rf = (event_kind == 1);
            is_inversion = (bsk::band(event_action, 4) != 0);
            invert = bsk::band(is_rf, is_inversion);
            zvr = bsk::where(invert, ((-atom_inv) * zvr), zvr);
            zvi = bsk::where(invert, ((-atom_inv) * zvi), zvi);
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                // A chemically exchanging pool is free water and inverts like any
                // other; a semisolid one is saturated instead, by the pulse's own
                // saturation term.
                poolvr = bsk::where(invert, ((-atom_inv) * poolvr), poolvr);
                poolvi = bsk::where(invert, ((-atom_inv) * poolvi), poolvi);
            }
            event_flip = _event_value(flip, event_base, event, active_atom, single_train);
            event_phase = _event_value(phase, event_base, event, active_atom, single_train);
            pulse_b1 = atom_b1;
            pulse_b1_phase = atom_b1_phase;
            // One shim is the whole sequence's transmit field, loaded once above;
            // several give each pulse a row of its own.
            if (bsk::truth(shimmed)) {
                row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
                if (bsk::truth(transmit)) {
                    pulse_b1 = bsk::ld(((b1 + row) + atom), active_atom, 1.0f);
                }
                if (bsk::truth(off_axis)) {
                    pulse_b1_phase = bsk::ld(((b1_phase + row) + atom), active_atom, 0.0f);
                }
            }
            alpha_value = (event_flip * pulse_b1);
            phi_value = (event_phase + pulse_b1_phase);
            if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
                // The pool absorbs the power the pulse deposits, read at the offset
                // the pulse is played less the voxel's own.
                offset_value = (bsk::ld((rf_frequency + event)) - atom_b0);
                auto t40_ = _lineshape_at_slope(lineshape, offset_value, lineshape_bins, lineshape_step);
                shape_value = bsk::get<0>(t40_);
                auto _shape_slope = bsk::get<1>(t40_);
                event_saturation = bsk::ld((saturation + event));
                power_value = ((event_saturation * alpha_value) * alpha_value);
                absorbed_value = bsk::exp((power_value * shape_value));
                saturating = bsk::band(is_rf, bsk::bnot(is_inversion));
                if (bsk::truth((pools == 1))) {
                    poolvr = bsk::where(saturating, (absorbed_value * poolvr), poolvr);
                    poolvi = bsk::where(saturating, (absorbed_value * poolvi), poolvi);
                } else {
                    semivr = bsk::where(saturating, (absorbed_value * semivr), semivr);
                    semivi = bsk::where(saturating, (absorbed_value * semivi), semivi);
                }
            }
            cos_value = bsk::cos(alpha_value);
            sin_value = bsk::sin(alpha_value);
            auto t41_ = bsk::make_tup(bsk::cos(phi_value), bsk::sin(phi_value));
            p1r = bsk::get<0>(t41_);
            p1i = bsk::get<1>(t41_);
            auto t42_ = _complex_mul(p1r, p1i, p1r, p1i);
            p2r = bsk::get<0>(t42_);
            p2i = bsk::get<1>(t42_);
            auto t43_ = _rotation_coefficients((0.5f * (1.0f + cos_value)), (0.5f * (1.0f - cos_value)), sin_value, cos_value, p1r, p1i, p2r, p2i, p1r, (-p1i));
            t00 = bsk::get<0>(t43_);
            t01 = bsk::get<1>(t43_);
            t02 = bsk::get<2>(t43_);
            t12 = bsk::get<3>(t43_);
            t20 = bsk::get<4>(t43_);
            t21 = bsk::get<5>(t43_);
            t22 = bsk::get<6>(t43_);
            auto a0 = _complex_mul(bsk::get<0>(t00), bsk::get<1>(t00), pvr, pvi);
            auto a1 = _complex_mul(bsk::get<0>(t01), bsk::get<1>(t01), mvr, mvi);
            auto a2 = _complex_mul(bsk::get<0>(t02), bsk::get<1>(t02), zvr, zvi);
            auto b0_ = _complex_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), pvr, pvi);
            auto b1_ = _complex_mul(bsk::get<0>(t00), bsk::get<1>(t00), mvr, mvi);
            auto b2 = _complex_mul(bsk::get<0>(t12), bsk::get<1>(t12), zvr, zvi);
            auto c0 = _complex_mul(bsk::get<0>(t20), bsk::get<1>(t20), pvr, pvi);
            auto c1 = _complex_mul(bsk::get<0>(t21), bsk::get<1>(t21), mvr, mvi);
            auto c2 = _complex_mul(bsk::get<0>(t22), bsk::get<1>(t22), zvr, zvi);
            turned_pr = ((bsk::get<0>(a0) + bsk::get<0>(a1)) + bsk::get<0>(a2));
            turned_pi = ((bsk::get<1>(a0) + bsk::get<1>(a1)) + bsk::get<1>(a2));
            turned_mr = ((bsk::get<0>(b0_) + bsk::get<0>(b1_)) + bsk::get<0>(b2));
            turned_mi = ((bsk::get<1>(b0_) + bsk::get<1>(b1_)) + bsk::get<1>(b2));
            turned_zr = ((bsk::get<0>(c0) + bsk::get<0>(c1)) + bsk::get<0>(c2));
            turned_zi = ((bsk::get<1>(c0) + bsk::get<1>(c1)) + bsk::get<1>(c2));
            if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                if (bsk::truth(dynamic)) {
                    pair = _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, active_atom);
                    auto t44_ = bsk::make_tup(bsk::get<0>(pair), bsk::get<1>(pair));
                    shaped_ar = bsk::get<0>(t44_);
                    shaped_ai = bsk::get<1>(t44_);
                    // The pair is integrated at zero RF phase, so the event's own
                    // phase turns the axis afterwards.
                    auto t45_ = _complex_mul(bsk::get<2>(pair), bsk::get<3>(pair), p1r, (-p1i));
                    shaped_br = bsk::get<0>(t45_);
                    shaped_bi = bsk::get<1>(t45_);
                } else {
                    auto t46_ = _profile_pair(profile, _table_row(profile_index, event, location, locations), alpha_value, profile_bins, profile_step);
                    shaped_ar = bsk::get<0>(t46_);
                    shaped_ai = bsk::get<1>(t46_);
                    shaped_br = bsk::get<2>(t46_);
                    shaped_bi = bsk::get<3>(t46_);
                    auto t47_ = _complex_mul(shaped_br, shaped_bi, p1r, (-p1i));
                    shaped_br = bsk::get<0>(t47_);
                    shaped_bi = bsk::get<1>(t47_);
                }
                auto t48_ = _rotate_spinor(shaped_ar, shaped_ai, shaped_br, shaped_bi, pvr, pvi, mvr, mvi, zvr, zvi);
                turned_pr = bsk::get<0>(t48_);
                turned_pi = bsk::get<1>(t48_);
                turned_mr = bsk::get<2>(t48_);
                turned_mi = bsk::get<3>(t48_);
                turned_zr = bsk::get<4>(t48_);
                turned_zi = bsk::get<5>(t48_);
            }
            rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                // The same pulse, the same rotation. A chemical shift moves where a
                // pool precesses, not what a pulse does to it.
                auto e0 = _complex_mul(bsk::get<0>(t00), bsk::get<1>(t00), bpvr, bpvi);
                auto e1_ = _complex_mul(bsk::get<0>(t01), bsk::get<1>(t01), bmvr, bmvi);
                auto e2_ = _complex_mul(bsk::get<0>(t02), bsk::get<1>(t02), poolvr, poolvi);
                auto f0 = _complex_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), bpvr, bpvi);
                auto f1 = _complex_mul(bsk::get<0>(t00), bsk::get<1>(t00), bmvr, bmvi);
                auto f2 = _complex_mul(bsk::get<0>(t12), bsk::get<1>(t12), poolvr, poolvi);
                auto h0 = _complex_mul(bsk::get<0>(t20), bsk::get<1>(t20), bpvr, bpvi);
                auto h1 = _complex_mul(bsk::get<0>(t21), bsk::get<1>(t21), bmvr, bmvi);
                auto h2 = _complex_mul(bsk::get<0>(t22), bsk::get<1>(t22), poolvr, poolvi);
                auto t49_ = bsk::make_tup(((bsk::get<0>(e0) + bsk::get<0>(e1_)) + bsk::get<0>(e2_)), ((bsk::get<1>(e0) + bsk::get<1>(e1_)) + bsk::get<1>(e2_)));
                spun_pr = bsk::get<0>(t49_);
                spun_pi = bsk::get<1>(t49_);
                auto t50_ = bsk::make_tup(((bsk::get<0>(f0) + bsk::get<0>(f1)) + bsk::get<0>(f2)), ((bsk::get<1>(f0) + bsk::get<1>(f1)) + bsk::get<1>(f2)));
                spun_mr = bsk::get<0>(t50_);
                spun_mi = bsk::get<1>(t50_);
                auto t51_ = bsk::make_tup(((bsk::get<0>(h0) + bsk::get<0>(h1)) + bsk::get<0>(h2)), ((bsk::get<1>(h0) + bsk::get<1>(h1)) + bsk::get<1>(h2)));
                spun_zr = bsk::get<0>(t51_);
                spun_zi = bsk::get<1>(t51_);
                if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                    auto t52_ = _rotate_spinor(shaped_ar, shaped_ai, shaped_br, shaped_bi, bpvr, bpvi, bmvr, bmvi, poolvr, poolvi);
                    spun_pr = bsk::get<0>(t52_);
                    spun_pi = bsk::get<1>(t52_);
                    spun_mr = bsk::get<2>(t52_);
                    spun_mi = bsk::get<3>(t52_);
                    spun_zr = bsk::get<4>(t52_);
                    spun_zi = bsk::get<5>(t52_);
                }
                bpvr = bsk::where(rotate, spun_pr, bpvr);
                bpvi = bsk::where(rotate, spun_pi, bpvi);
                bmvr = bsk::where(rotate, spun_mr, bmvr);
                bmvi = bsk::where(rotate, spun_mi, bmvi);
                poolvr = bsk::where(rotate, spun_zr, poolvr);
                poolvi = bsk::where(rotate, spun_zi, poolvi);
            }
            pvr = bsk::where(rotate, turned_pr, pvr);
            pvi = bsk::where(rotate, turned_pi, pvi);
            mvr = bsk::where(rotate, turned_mr, mvr);
            mvi = bsk::where(rotate, turned_mi, mvi);
            zvr = bsk::where(rotate, turned_zr, zvr);
            zvi = bsk::where(rotate, turned_zi, zvi);
            do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t53_ = _shift(bpvr, bpvi, bmvr, bmvi, state, state_mask, state_count);
                svr = bsk::get<0>(t53_);
                svi = bsk::get<1>(t53_);
                wvr = bsk::get<2>(t53_);
                wvi = bsk::get<3>(t53_);
                auto spoil_b = (bsk::band(event_action, 8) != 0);
                bpvr = bsk::where(spoil_b, 0.0f, bsk::where(do_shift, svr, bpvr));
                bpvi = bsk::where(spoil_b, 0.0f, bsk::where(do_shift, svi, bpvi));
                bmvr = bsk::where(spoil_b, 0.0f, bsk::where(do_shift, wvr, bmvr));
                bmvi = bsk::where(spoil_b, 0.0f, bsk::where(do_shift, wvi, bmvi));
            }
            auto t54_ = _shift(pvr, pvi, mvr, mvi, state, state_mask, state_count);
            svr = bsk::get<0>(t54_);
            svi = bsk::get<1>(t54_);
            wvr = bsk::get<2>(t54_);
            wvi = bsk::get<3>(t54_);
            pvr = bsk::where(do_shift, svr, pvr);
            pvi = bsk::where(do_shift, svi, pvi);
            mvr = bsk::where(do_shift, wvr, mvr);
            mvi = bsk::where(do_shift, wvi, mvi);
            spoil = (bsk::band(event_action, 8) != 0);
            pvr = bsk::where(spoil, 0.0f, pvr);
            pvi = bsk::where(spoil, 0.0f, pvi);
            mvr = bsk::where(spoil, 0.0f, mvr);
            mvi = bsk::where(spoil, 0.0f, mvi);
        }
        return;
    }
    // ---- reverse ----
    pbvr = empty;
    pbvi = empty;
    mbvr = empty;
    mbvi = empty;
    zbvr = empty;
    zbvi = empty;
    auto zero = bsk::full<float, 2>(0);
    g_diffv = zero;
    g_flowv = zero;
    g_washv = zero;
    g_t1v = zero;
    g_t2v = zero;
    g_m0v = zero;
    g_b1v = zero;
    g_b1pv = zero;
    g_b0v = zero;
    g_invv = zero;
    g_boundv = zero;
    g_exchv = zero;
    g_t1bv = zero;
    g_t2bv = zero;
    g_shiftv = zero;
    g_semiv = zero;
    g_sexchv = zero;
    g_t1cv = zero;
    poolbr = empty;
    poolbi = empty;
    semibr = empty;
    semibi = empty;
    ubvr = empty;
    ubvi = empty;
    wbvr = empty;
    wbvi = empty;
    for (bsk::index_t reverse = 0; reverse < event_count; reverse += 1) {
        event = ((event_count - 1) - reverse);
        slot = (trajectory + (event * record_stride));
        auto xpvr = bsk::ld((trajectory_r + slot), state_mask, 0.0f);
        auto xpvi = bsk::ld((trajectory_i + slot), state_mask, 0.0f);
        auto xmvr = bsk::ld(((trajectory_r + slot) + minus_plane), state_mask, 0.0f);
        auto xmvi = bsk::ld(((trajectory_i + slot) + minus_plane), state_mask, 0.0f);
        auto xzvr = bsk::ld(((trajectory_r + slot) + long_plane), state_mask, 0.0f);
        auto xzvi = bsk::ld(((trajectory_i + slot) + long_plane), state_mask, 0.0f);
        xbvr = empty;
        xbvi = empty;
        xcvr = empty;
        xcvi = empty;
        xbpvr = empty;
        xbpvi = empty;
        xbmvr = empty;
        xbmvi = empty;
        rbpvr = empty;
        rbpvi = empty;
        rbmvr = empty;
        rbmvi = empty;
        if (bsk::truth((pools > 0))) {
            xbvr = bsk::ld(((trajectory_r + slot) + bound_plane), state_mask, 0.0f);
            xbvi = bsk::ld(((trajectory_i + slot) + bound_plane), state_mask, 0.0f);
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            xbpvr = bsk::ld(((trajectory_r + slot) + bplus_plane), state_mask, 0.0f);
            xbpvi = bsk::ld(((trajectory_i + slot) + bplus_plane), state_mask, 0.0f);
            xbmvr = bsk::ld(((trajectory_r + slot) + bminus_plane), state_mask, 0.0f);
            xbmvi = bsk::ld(((trajectory_i + slot) + bminus_plane), state_mask, 0.0f);
        }
        if (bsk::truth((pools == 3))) {
            xcvr = bsk::ld(((trajectory_r + slot) + semisolid_plane), state_mask, 0.0f);
            xcvi = bsk::ld(((trajectory_i + slot) + semisolid_plane), state_mask, 0.0f);
        }
        event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        event_kind = bsk::ld((kind + event));
        dt_value = _event_value(duration, event_base, event, active_atom, single_train);
        wout_value = 1.0f;
        if (bsk::truth(moving)) {
            wout_value = _washout(atom_washout, dt_value);
        }
        auto dry1_value = bsk::exp(((-r1_value) * dt_value));
        auto dry2_value = bsk::exp(((-r2_value) * dt_value));
        e1_value = (dry1_value * wout_value);
        e2_value = (dry2_value * wout_value);
        damp_z = 1.0f;
        damp_t = 1.0f;
        if (bsk::truth(diffusing)) {
            auto t55_ = _damping(atom_damping, dt_value, order);
            damp_z = bsk::get<0>(t55_);
            damp_t = bsk::get<1>(t55_);
        }
        // Order zero is undamped, so recovery keeps the bare longitudinal factor.
        recovery_value = (1.0f - e1_value);
        bare1_value = e1_value;
        bare2_value = e2_value;
        e1_value = (bare1_value * damp_z);
        e2_value = (bare2_value * damp_t);
        turn_t = 0.0f;
        auto t56_ = bsk::make_tup(1.0f, 0.0f);
        szr = bsk::get<0>(t56_);
        szi = bsk::get<1>(t56_);
        if (bsk::truth(moving)) {
            auto t57_ = _flow(atom_flow, dt_value, order);
            turn_z = bsk::get<0>(t57_);
            turn_t = bsk::get<1>(t57_);
            auto t58_ = bsk::make_tup(bsk::cos(turn_z), bsk::sin(turn_z));
            szr = bsk::get<0>(t58_);
            szi = bsk::get<1>(t58_);
        }
        auto t59_ = bsk::make_tup(1.0f, 0.0f);
        qr = bsk::get<0>(t59_);
        qi = bsk::get<1>(t59_);
        if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
            angle_value = ((-6.283185307179586f * (atom_b0 * dt_value)) + turn_t);
            auto t60_ = bsk::make_tup(bsk::cos(angle_value), bsk::sin(angle_value));
            qr = bsk::get<0>(t60_);
            qi = bsk::get<1>(t60_);
        }
        auto t61_ = bsk::make_tup((e2_value * qr), (e2_value * qi));
        ovr = bsk::get<0>(t61_);
        ovi = bsk::get<1>(t61_);
        auto t62_ = bsk::make_tup((e1_value * szr), (e1_value * szi));
        lvr = bsk::get<0>(t62_);
        lvi = bsk::get<1>(t62_);
        // Replay the intra-event stages from the recorded entry state.
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // With an exchanging pool the transverse relaxation sits inside the
            // operator instead of in the scalar the free pool alone multiplies.
            across = _two_pool_transverse_step_jvp(r2_value, 0.0f, r2b_value, 0.0f, atom_exchange, 0.0f, atom_bound, 0.0f, atom_free, 0.0f, atom_shift, 0.0f, dt_value, 0.0f, wout_value, 0.0f);
            auto t63_ = bsk::make_tup(bsk::get<0>(across), bsk::get<1>(across));
            a11r = bsk::get<0>(t63_);
            a11i = bsk::get<1>(t63_);
            auto t64_ = bsk::make_tup(bsk::get<2>(across), bsk::get<3>(across));
            a12r = bsk::get<0>(t64_);
            a12i = bsk::get<1>(t64_);
            auto t65_ = bsk::make_tup(bsk::get<4>(across), bsk::get<5>(across));
            a21r = bsk::get<0>(t65_);
            a21i = bsk::get<1>(t65_);
            auto t66_ = bsk::make_tup(bsk::get<6>(across), bsk::get<7>(across));
            a22r = bsk::get<0>(t66_);
            a22i = bsk::get<1>(t66_);
            auto t67_ = bsk::make_tup((damp_t * qr), (damp_t * qi));
            carr = bsk::get<0>(t67_);
            cari = bsk::get<1>(t67_);
            auto t68_ = _complex_mul(a11r, a11i, xpvr, xpvi);
            f11r = bsk::get<0>(t68_);
            f11i = bsk::get<1>(t68_);
            auto t69_ = _complex_mul(a12r, a12i, xbpvr, xbpvi);
            f12r = bsk::get<0>(t69_);
            f12i = bsk::get<1>(t69_);
            auto t70_ = _complex_mul(a21r, a21i, xpvr, xpvi);
            g21r = bsk::get<0>(t70_);
            g21i = bsk::get<1>(t70_);
            auto t71_ = _complex_mul(a22r, a22i, xbpvr, xbpvi);
            g22r = bsk::get<0>(t71_);
            g22i = bsk::get<1>(t71_);
            // ``F-`` takes the conjugate of the operator entry by entry, not its
            // transpose: it is the conjugate state following the conjugate map.
            auto t72_ = _complex_mul(a11r, (-a11i), xmvr, xmvi);
            h11r = bsk::get<0>(t72_);
            h11i = bsk::get<1>(t72_);
            auto t73_ = _complex_mul(a12r, (-a12i), xbmvr, xbmvi);
            h12r = bsk::get<0>(t73_);
            h12i = bsk::get<1>(t73_);
            auto t74_ = _complex_mul(a21r, (-a21i), xmvr, xmvi);
            k21r = bsk::get<0>(t74_);
            k21i = bsk::get<1>(t74_);
            auto t75_ = _complex_mul(a22r, (-a22i), xbmvr, xbmvi);
            k22r = bsk::get<0>(t75_);
            k22i = bsk::get<1>(t75_);
            auto t76_ = _complex_mul((f11r + f12r), (f11i + f12i), carr, cari);
            rpvr = bsk::get<0>(t76_);
            rpvi = bsk::get<1>(t76_);
            auto t77_ = _complex_mul((g21r + g22r), (g21i + g22i), carr, cari);
            rbpvr = bsk::get<0>(t77_);
            rbpvi = bsk::get<1>(t77_);
            auto t78_ = _complex_mul((h11r + h12r), (h11i + h12i), carr, (-cari));
            rmvr = bsk::get<0>(t78_);
            rmvi = bsk::get<1>(t78_);
            auto t79_ = _complex_mul((k21r + k22r), (k21i + k22i), carr, (-cari));
            rbmvr = bsk::get<0>(t79_);
            rbmvi = bsk::get<1>(t79_);
        } else {
            auto t80_ = _complex_mul(ovr, ovi, xpvr, xpvi);
            rpvr = bsk::get<0>(t80_);
            rpvi = bsk::get<1>(t80_);
            auto t81_ = _complex_mul(ovr, (-ovi), xmvr, xmvi);
            rmvr = bsk::get<0>(t81_);
            rmvi = bsk::get<1>(t81_);
        }
        rbvr = empty;
        rbvi = empty;
        rcvr = empty;
        rcvi = empty;
        if (bsk::truth((pools == 3))) {
            nil = (0.0f * dt_value);
            hold_value = (wout_value + nil);
            if (bsk::truth(tabulated)) {
                // The walk back needs the operator itself, which the row
                // already holds -- and pooling the cotangents took what
                // the eigenvalues were formed for, so nothing here reads
                // them.
                pool_row = bsk::ld(((duration_row + event_base) + event), active_atom, 0);
                auto t82_ = _three_pool_from_table(pool_table, pool_row, atom, atom_count, active_atom, hold_value, atom_free, atom_bound, atom_semisolid);
                w11 = bsk::get<0>(t82_);
                w12 = bsk::get<1>(t82_);
                w13 = bsk::get<2>(t82_);
                w21 = bsk::get<3>(t82_);
                w22 = bsk::get<4>(t82_);
                w23 = bsk::get<5>(t82_);
                w31 = bsk::get<6>(t82_);
                w32 = bsk::get<7>(t82_);
                w33 = bsk::get<8>(t82_);
                grow_free = bsk::get<9>(t82_);
                grow_pool_b = bsk::get<10>(t82_);
                grow_semisolid = bsk::get<11>(t82_);
            } else {
                auto t83_ = _three_pool_pieces_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, dt_value, nil, narrow);
                three_free = bsk::get<0>(t83_);
                three_d_free = bsk::get<1>(t83_);
                three_pool_b = bsk::get<2>(t83_);
                three_d_pool_b = bsk::get<3>(t83_);
                three_pool_c = bsk::get<4>(t83_);
                three_d_pool_c = bsk::get<5>(t83_);
                three_a00 = bsk::get<6>(t83_);
                three_d_a00 = bsk::get<7>(t83_);
                three_a01 = bsk::get<8>(t83_);
                three_d_a01 = bsk::get<9>(t83_);
                three_a02 = bsk::get<10>(t83_);
                three_d_a02 = bsk::get<11>(t83_);
                three_a10 = bsk::get<12>(t83_);
                three_d_a10 = bsk::get<13>(t83_);
                three_a11 = bsk::get<14>(t83_);
                three_d_a11 = bsk::get<15>(t83_);
                three_a20 = bsk::get<16>(t83_);
                three_d_a20 = bsk::get<17>(t83_);
                three_a22 = bsk::get<18>(t83_);
                three_d_a22 = bsk::get<19>(t83_);
                three_s00 = bsk::get<20>(t83_);
                three_d_s00 = bsk::get<21>(t83_);
                three_s11 = bsk::get<22>(t83_);
                three_d_s11 = bsk::get<23>(t83_);
                three_s22 = bsk::get<24>(t83_);
                three_d_s22 = bsk::get<25>(t83_);
                three_minors = bsk::get<26>(t83_);
                three_d_minors = bsk::get<27>(t83_);
                three_sum_flat = bsk::get<28>(t83_);
                three_sum_linear = bsk::get<29>(t83_);
                three_sum_square = bsk::get<30>(t83_);
                three_d_sum_flat = bsk::get<31>(t83_);
                three_d_sum_linear = bsk::get<32>(t83_);
                three_d_sum_square = bsk::get<33>(t83_);
                three_lift = bsk::get<34>(t83_);
                three_d_lift = bsk::get<35>(t83_);
                three_low = bsk::get<36>(t83_);
                three_middle = bsk::get<37>(t83_);
                three_d_low = bsk::get<38>(t83_);
                three_d_middle = bsk::get<39>(t83_);
                three_leading = bsk::get<40>(t83_);
                three_d_leading = bsk::get<41>(t83_);
                three_first = bsk::get<42>(t83_);
                three_d_first = bsk::get<43>(t83_);
                three_second = bsk::get<44>(t83_);
                three_d_second = bsk::get<45>(t83_);
                three_determinant = bsk::get<46>(t83_);
                three_d_determinant = bsk::get<47>(t83_);
                three_high = bsk::get<48>(t83_);
                three_d_high = bsk::get<49>(t83_);
                three_radius = bsk::get<50>(t83_);
                three_d_radius = bsk::get<51>(t83_);
                three_cube = bsk::get<52>(t83_);
                three_raw = bsk::get<53>(t83_);
                three_d_raw = bsk::get<54>(t83_);
                three_argument = bsk::get<55>(t83_);
                three_inside_limit = bsk::get<56>(t83_);
                three_angle = bsk::get<57>(t83_);
                three_d_angle = bsk::get<58>(t83_);
                three_centre = bsk::get<59>(t83_);
                three_d_centre = bsk::get<60>(t83_);
                three_trailing = bsk::get<61>(t83_);
                three_d_trailing = bsk::get<62>(t83_);
                three_guarded = bsk::get<63>(t83_);
                three_d_guarded = bsk::get<64>(t83_);
                three_q00 = bsk::get<65>(t83_);
                three_d_q00 = bsk::get<66>(t83_);
                three_q01 = bsk::get<67>(t83_);
                three_d_q01 = bsk::get<68>(t83_);
                three_q02 = bsk::get<69>(t83_);
                three_d_q02 = bsk::get<70>(t83_);
                three_q10 = bsk::get<71>(t83_);
                three_d_q10 = bsk::get<72>(t83_);
                three_q11 = bsk::get<73>(t83_);
                three_d_q11 = bsk::get<74>(t83_);
                three_q12 = bsk::get<75>(t83_);
                three_d_q12 = bsk::get<76>(t83_);
                three_q20 = bsk::get<77>(t83_);
                three_d_q20 = bsk::get<78>(t83_);
                three_q21 = bsk::get<79>(t83_);
                three_d_q21 = bsk::get<80>(t83_);
                three_q22 = bsk::get<81>(t83_);
                three_d_q22 = bsk::get<82>(t83_);
                auto t84_ = _three_pool_assemble_jvp(three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, narrow);
                three_def_00 = bsk::get<0>(t84_);
                three_dif_00 = bsk::get<1>(t84_);
                three_def_01 = bsk::get<2>(t84_);
                three_dif_01 = bsk::get<3>(t84_);
                three_def_02 = bsk::get<4>(t84_);
                three_dif_02 = bsk::get<5>(t84_);
                three_def_10 = bsk::get<6>(t84_);
                three_dif_10 = bsk::get<7>(t84_);
                three_def_11 = bsk::get<8>(t84_);
                three_dif_11 = bsk::get<9>(t84_);
                three_def_12 = bsk::get<10>(t84_);
                three_dif_12 = bsk::get<11>(t84_);
                three_def_20 = bsk::get<12>(t84_);
                three_dif_20 = bsk::get<13>(t84_);
                three_def_21 = bsk::get<14>(t84_);
                three_dif_21 = bsk::get<15>(t84_);
                three_def_22 = bsk::get<16>(t84_);
                three_dif_22 = bsk::get<17>(t84_);
                auto t85_ = _three_pool_weigh_jvp(three_def_00, three_dif_00, three_def_01, three_dif_01, three_def_02, three_dif_02, three_def_10, three_dif_10, three_def_11, three_dif_11, three_def_12, three_dif_12, three_def_20, three_dif_20, three_def_21, three_dif_21, three_def_22, three_dif_22, three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, hold_value, nil, narrow);
                w11 = bsk::get<0>(t85_);
                w12 = bsk::get<1>(t85_);
                w13 = bsk::get<2>(t85_);
                w21 = bsk::get<3>(t85_);
                w22 = bsk::get<4>(t85_);
                w23 = bsk::get<5>(t85_);
                w31 = bsk::get<6>(t85_);
                w32 = bsk::get<7>(t85_);
                w33 = bsk::get<8>(t85_);
                grow_free = bsk::get<9>(t85_);
                grow_pool_b = bsk::get<10>(t85_);
                grow_semisolid = bsk::get<11>(t85_);
                _dw11 = bsk::get<12>(t85_);
                _dw12 = bsk::get<13>(t85_);
                _dw13 = bsk::get<14>(t85_);
                _dw21 = bsk::get<15>(t85_);
                _dw22 = bsk::get<16>(t85_);
                _dw23 = bsk::get<17>(t85_);
                _dw31 = bsk::get<18>(t85_);
                _dw32 = bsk::get<19>(t85_);
                _dw33 = bsk::get<20>(t85_);
                _dgf = bsk::get<21>(t85_);
                _dgb = bsk::get<22>(t85_);
                _dgs = bsk::get<23>(t85_);
                // The operator is O(1) once formed, so the per-order loop below
                // takes it at the width the states are carried in.
                w11 = bsk::cast<float>(w11);
                w12 = bsk::cast<float>(w12);
                w13 = bsk::cast<float>(w13);
                w21 = bsk::cast<float>(w21);
                w22 = bsk::cast<float>(w22);
                w23 = bsk::cast<float>(w23);
                w31 = bsk::cast<float>(w31);
                w32 = bsk::cast<float>(w32);
                w33 = bsk::cast<float>(w33);
                grow_free = bsk::cast<float>(grow_free);
                grow_pool_b = bsk::cast<float>(grow_pool_b);
                grow_semisolid = bsk::cast<float>(grow_semisolid);
            }
            auto t86_ = bsk::make_tup((damp_z * szr), (damp_z * szi));
            spin_r = bsk::get<0>(t86_);
            spin_i = bsk::get<1>(t86_);
            mix_fr = (((w11 * xzvr) + (w12 * xbvr)) + (w13 * xcvr));
            mix_fi = (((w11 * xzvi) + (w12 * xbvi)) + (w13 * xcvi));
            mix_br = (((w21 * xzvr) + (w22 * xbvr)) + (w23 * xcvr));
            mix_bi = (((w21 * xzvi) + (w22 * xbvi)) + (w23 * xcvi));
            mix_cr = (((w31 * xzvr) + (w32 * xbvr)) + (w33 * xcvr));
            mix_ci = (((w31 * xzvi) + (w32 * xbvi)) + (w33 * xcvi));
            auto t87_ = _complex_mul(spin_r, spin_i, mix_fr, mix_fi);
            rzvr = bsk::get<0>(t87_);
            rzvi = bsk::get<1>(t87_);
            auto t88_ = _complex_mul(spin_r, spin_i, mix_br, mix_bi);
            rbvr = bsk::get<0>(t88_);
            rbvi = bsk::get<1>(t88_);
            auto t89_ = _complex_mul(spin_r, spin_i, mix_cr, mix_ci);
            rcvr = bsk::get<0>(t89_);
            rcvi = bsk::get<1>(t89_);
            rzvr = (rzvr + bsk::where((state == 0), grow_free, 0.0f));
            rbvr = (rbvr + bsk::where((state == 0), grow_pool_b, 0.0f));
            rcvr = (rcvr + bsk::where((state == 0), grow_semisolid, 0.0f));
        } else if (bsk::truth((pools > 0))) {
            auto t90_ = _two_pool_step_jvp(r1_value, 0.0f, r1b_value, 0.0f, atom_exchange, 0.0f, atom_bound, 0.0f, dt_value, 0.0f, wout_value, 0.0f);
            pe11 = bsk::get<0>(t90_);
            pe12 = bsk::get<1>(t90_);
            pe21 = bsk::get<2>(t90_);
            pe22 = bsk::get<3>(t90_);
            prec_f = bsk::get<4>(t90_);
            prec_b = bsk::get<5>(t90_);
            _d11 = bsk::get<6>(t90_);
            _d12 = bsk::get<7>(t90_);
            _d21 = bsk::get<8>(t90_);
            _d22 = bsk::get<9>(t90_);
            _drf = bsk::get<10>(t90_);
            _drb = bsk::get<11>(t90_);
            auto t91_ = bsk::make_tup((damp_z * szr), (damp_z * szi));
            spin_r = bsk::get<0>(t91_);
            spin_i = bsk::get<1>(t91_);
            mix_fr = ((pe11 * xzvr) + (pe12 * xbvr));
            mix_fi = ((pe11 * xzvi) + (pe12 * xbvi));
            mix_br = ((pe21 * xzvr) + (pe22 * xbvr));
            mix_bi = ((pe21 * xzvi) + (pe22 * xbvi));
            auto t92_ = _complex_mul(spin_r, spin_i, mix_fr, mix_fi);
            rzvr = bsk::get<0>(t92_);
            rzvi = bsk::get<1>(t92_);
            auto t93_ = _complex_mul(spin_r, spin_i, mix_br, mix_bi);
            rbvr = bsk::get<0>(t93_);
            rbvi = bsk::get<1>(t93_);
            rzvr = (rzvr + bsk::where((state == 0), prec_f, 0.0f));
            rbvr = (rbvr + bsk::where((state == 0), prec_b, 0.0f));
        } else {
            auto t94_ = _complex_mul(lvr, lvi, xzvr, xzvi);
            rzvr = bsk::get<0>(t94_);
            rzvi = bsk::get<1>(t94_);
            rzvr = (rzvr + bsk::where((state == 0), recovery_value, 0.0f));
        }
        pre_shift = (bsk::band(event_action, 1) != 0);
        auto t95_ = _shift(rpvr, rpvi, rmvr, rmvi, state, state_mask, state_count);
        svr = bsk::get<0>(t95_);
        svi = bsk::get<1>(t95_);
        wvr = bsk::get<2>(t95_);
        wvi = bsk::get<3>(t95_);
        auto spvr = bsk::where(pre_shift, svr, rpvr);
        auto spvi = bsk::where(pre_shift, svi, rpvi);
        auto smvr = bsk::where(pre_shift, wvr, rmvr);
        auto smvi = bsk::where(pre_shift, wvi, rmvi);
        sbpvr = empty;
        sbpvi = empty;
        sbmvr = empty;
        sbmvi = empty;
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t96_ = _shift(rbpvr, rbpvi, rbmvr, rbmvi, state, state_mask, state_count);
            svr = bsk::get<0>(t96_);
            svi = bsk::get<1>(t96_);
            wvr = bsk::get<2>(t96_);
            wvi = bsk::get<3>(t96_);
            sbpvr = bsk::where(pre_shift, svr, rbpvr);
            sbpvi = bsk::where(pre_shift, svi, rbpvi);
            sbmvr = bsk::where(pre_shift, wvr, rbmvr);
            sbmvi = bsk::where(pre_shift, wvi, rbmvi);
        }
        // Undo the trailing spoil or shift.
        do_shift = bsk::bor((bsk::band(event_action, 2) != 0), (bsk::band(event_action, 16) != 0));
        spoil = (bsk::band(event_action, 8) != 0);
        auto t97_ = _shift_adjoint(pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count);
        avr = bsk::get<0>(t97_);
        avi = bsk::get<1>(t97_);
        bvr = bsk::get<2>(t97_);
        bvi = bsk::get<3>(t97_);
        auto trailing = bsk::band(do_shift, bsk::bnot(spoil));
        pbvr = bsk::where(spoil, 0.0f, bsk::where(trailing, avr, pbvr));
        pbvi = bsk::where(spoil, 0.0f, bsk::where(trailing, avi, pbvi));
        mbvr = bsk::where(spoil, 0.0f, bsk::where(trailing, bvr, mbvr));
        mbvi = bsk::where(spoil, 0.0f, bsk::where(trailing, bvi, mbvi));
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t98_ = _shift_adjoint(ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count);
            avr = bsk::get<0>(t98_);
            avi = bsk::get<1>(t98_);
            bvr = bsk::get<2>(t98_);
            bvi = bsk::get<3>(t98_);
            ubvr = bsk::where(spoil, 0.0f, bsk::where(trailing, avr, ubvr));
            ubvi = bsk::where(spoil, 0.0f, bsk::where(trailing, avi, ubvi));
            wbvr = bsk::where(spoil, 0.0f, bsk::where(trailing, bvr, wbvr));
            wbvi = bsk::where(spoil, 0.0f, bsk::where(trailing, bvi, wbvi));
        }
        event_flip = _event_value(flip, event_base, event, active_atom, single_train);
        event_phase = _event_value(phase, event_base, event, active_atom, single_train);
        pulse_b1 = atom_b1;
        pulse_b1_phase = atom_b1_phase;
        if (bsk::truth(shimmed)) {
            row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
            if (bsk::truth(transmit)) {
                pulse_b1 = bsk::ld(((b1 + row) + atom), active_atom, 1.0f);
            }
            if (bsk::truth(off_axis)) {
                pulse_b1_phase = bsk::ld(((b1_phase + row) + atom), active_atom, 0.0f);
            }
        }
        // ---- recorded sample ----
        auto record = bsk::band((bsk::band(event_action, 32) != 0), (event_kind == 2));
        auto out_ = bsk::ld((output_index + event));
        auto seed_mask = bsk::band(bsk::band(active_atom, record), (out_ >= 0));
        auto seed_real = bsk::ld(((grad_output_real + (problem * output_count)) + out_), seed_mask, 0.0f);
        auto seed_imag = bsk::ld(((grad_output_imag + (problem * output_count)) + out_), seed_mask, 0.0f);
        auto t99_ = bsk::make_tup(bsk::cos((-event_phase)), bsk::sin((-event_phase)));
        auto dvr = bsk::get<0>(t99_);
        auto dvi = bsk::get<1>(t99_);
        // grad_m0 = Re(conj(seed) * recorded * demodulation)
        auto t100_ = bsk::make_tup(spvr, spvi);
        recr = bsk::get<0>(t100_);
        reci = bsk::get<1>(t100_);
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            auto t101_ = bsk::make_tup((spvr + sbpvr), (spvi + sbpvi));
            recr = bsk::get<0>(t101_);
            reci = bsk::get<1>(t101_);
        }
        auto t102_ = _complex_mul(recr, reci, dvr, dvi);
        auto wr = bsk::get<0>(t102_);
        auto wi = bsk::get<1>(t102_);
        g_m0v = (g_m0v + bsk::sum_x(bsk::where((state == 0), ((seed_real * wr) + (seed_imag * wi)), 0.0f)));
        // grad_phase = Re(conj(seed) * m0 * recorded * (-i) * demodulation)
        auto t103_ = bsk::make_tup((atom_m0 * recr), (atom_m0 * reci));
        yr = bsk::get<0>(t103_);
        yi = bsk::get<1>(t103_);
        auto t104_ = bsk::make_tup(yi, (-yr));
        yr = bsk::get<0>(t104_);
        yi = bsk::get<1>(t104_);
        auto t105_ = _complex_mul(yr, yi, dvr, dvi);
        yr = bsk::get<0>(t105_);
        yi = bsk::get<1>(t105_);
        bsk::atomic_add(((grad_phase + event_base) + event), bsk::sum_x(bsk::where((state == 0), ((seed_real * yr) + (seed_imag * yi)), 0.0f)), seed_mask);
        // fplus_bar[0] += conj(m0 * demodulation) * seed
        auto t106_ = bsk::make_tup((atom_m0 * dvr), (atom_m0 * dvi));
        auto kr = bsk::get<0>(t106_);
        auto ki = bsk::get<1>(t106_);
        auto t107_ = _complex_mul(kr, (-ki), seed_real, seed_imag);
        auto sr = bsk::get<0>(t107_);
        auto si = bsk::get<1>(t107_);
        pbvr = (pbvr + bsk::where((state == 0), sr, 0.0f));
        pbvi = (pbvi + bsk::where((state == 0), si, 0.0f));
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            ubvr = (ubvr + bsk::where((state == 0), sr, 0.0f));
            ubvi = (ubvi + bsk::where((state == 0), si, 0.0f));
        }
        // ---- RF adjoint ----
        is_rf = (event_kind == 1);
        is_inversion = (bsk::band(event_action, 4) != 0);
        invert = bsk::band(is_rf, is_inversion);
        g_invv = (g_invv + bsk::sum_x(bsk::where(invert, ((zbvr * (-rzvr)) + (zbvi * (-rzvi))), 0.0f)));
        zbvr = bsk::where(invert, ((-atom_inv) * zbvr), zbvr);
        zbvi = bsk::where(invert, ((-atom_inv) * zbvi), zbvi);
        alpha_value = (event_flip * pulse_b1);
        phi_value = (event_phase + pulse_b1_phase);
        cos_value = bsk::cos(alpha_value);
        sin_value = bsk::sin(alpha_value);
        auto t108_ = bsk::make_tup(bsk::cos(phi_value), bsk::sin(phi_value));
        p1r = bsk::get<0>(t108_);
        p1i = bsk::get<1>(t108_);
        auto t109_ = _complex_mul(p1r, p1i, p1r, p1i);
        p2r = bsk::get<0>(t109_);
        p2i = bsk::get<1>(t109_);
        auto t110_ = _rotation_coefficients((0.5f * (1.0f + cos_value)), (0.5f * (1.0f - cos_value)), sin_value, cos_value, p1r, p1i, p2r, p2i, p1r, (-p1i));
        t00 = bsk::get<0>(t110_);
        t01 = bsk::get<1>(t110_);
        t02 = bsk::get<2>(t110_);
        t12 = bsk::get<3>(t110_);
        t20 = bsk::get<4>(t110_);
        t21 = bsk::get<5>(t110_);
        t22 = bsk::get<6>(t110_);
        auto t111_ = _rotation_coefficients((-0.5f * sin_value), (0.5f * sin_value), cos_value, (-sin_value), p1r, p1i, p2r, p2i, p1r, (-p1i));
        auto d00 = bsk::get<0>(t111_);
        auto d01 = bsk::get<1>(t111_);
        auto d02 = bsk::get<2>(t111_);
        auto d12 = bsk::get<3>(t111_);
        auto d20 = bsk::get<4>(t111_);
        auto d21 = bsk::get<5>(t111_);
        auto d22 = bsk::get<6>(t111_);
        sat_alpha_v = zero;
        sat_b0_v = zero;
        if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
            // The pulse scales every order of the pool by one real number, so
            // its cotangent is a single sum over the states it multiplied.
            offset_value = (bsk::ld((rf_frequency + event)) - atom_b0);
            auto t112_ = _lineshape_at_slope(lineshape, offset_value, lineshape_bins, lineshape_step);
            shape_value = bsk::get<0>(t112_);
            auto shape_slope = bsk::get<1>(t112_);
            event_saturation = bsk::ld((saturation + event));
            power_value = ((event_saturation * alpha_value) * alpha_value);
            absorbed_value = bsk::exp((power_value * shape_value));
            if (bsk::truth((pools == 1))) {
                per_state = ((poolbr * rbvr) + (poolbi * rbvi));
            } else {
                per_state = ((semibr * rcvr) + (semibi * rcvi));
            }
            auto grad_absorbed = bsk::sum_x(per_state);
            auto grad_exponent = (grad_absorbed * absorbed_value);
            auto twice = (event_saturation * 2.0f);
            sat_alpha_v = (grad_exponent * ((twice * alpha_value) * shape_value));
            // The lineshape is read at the pulse's offset from the voxel, so a
            // step in the voxel's own off-resonance moves the read the other way.
            sat_b0_v = ((-grad_exponent) * (power_value * shape_slope));
            saturating = bsk::band(is_rf, bsk::bnot(is_inversion));
            if (bsk::truth((pools == 1))) {
                poolbr = bsk::where(saturating, (absorbed_value * poolbr), poolbr);
                poolbi = bsk::where(saturating, (absorbed_value * poolbi), poolbi);
            } else {
                semibr = bsk::where(saturating, (absorbed_value * semibr), semibr);
                semibi = bsk::where(saturating, (absorbed_value * semibi), semibi);
            }
        }
        // d/dalpha, contracted with the adjoint.
        row0 = _complex_mul(bsk::get<0>(d00), bsk::get<1>(d00), spvr, spvi);
        add1 = _complex_mul(bsk::get<0>(d01), bsk::get<1>(d01), smvr, smvi);
        add2 = _complex_mul(bsk::get<0>(d02), bsk::get<1>(d02), rzvr, rzvi);
        alpha_v = (pbvr * ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2)));
        alpha_v = (alpha_v + (pbvi * ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2))));
        row0 = _complex_mul(bsk::get<0>(d01), (-bsk::get<1>(d01)), spvr, spvi);
        add1 = _complex_mul(bsk::get<0>(d00), bsk::get<1>(d00), smvr, smvi);
        add2 = _complex_mul(bsk::get<0>(d12), bsk::get<1>(d12), rzvr, rzvi);
        alpha_v = (alpha_v + (mbvr * ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2))));
        alpha_v = (alpha_v + (mbvi * ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2))));
        row0 = _complex_mul(bsk::get<0>(d20), bsk::get<1>(d20), spvr, spvi);
        add1 = _complex_mul(bsk::get<0>(d21), bsk::get<1>(d21), smvr, smvi);
        add2 = _complex_mul(bsk::get<0>(d22), bsk::get<1>(d22), rzvr, rzvi);
        alpha_v = (alpha_v + (zbvr * ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2))));
        alpha_v = (alpha_v + (zbvi * ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2))));
        // d/dphi, where only the phase factors carry the dependence.
        u1 = _complex_mul(bsk::get<0>(t01), bsk::get<1>(t01), smvr, smvi);
        u2 = _complex_mul(bsk::get<0>(t02), bsk::get<1>(t02), rzvr, rzvi);
        auto t113_ = bsk::make_tup((-((2.0f * bsk::get<1>(u1)) + bsk::get<1>(u2))), ((2.0f * bsk::get<0>(u1)) + bsk::get<0>(u2)));
        ur = bsk::get<0>(t113_);
        ui = bsk::get<1>(t113_);
        phi_v = ((pbvr * ur) + (pbvi * ui));
        u1 = _complex_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), spvr, spvi);
        u2 = _complex_mul(bsk::get<0>(t12), bsk::get<1>(t12), rzvr, rzvi);
        auto t114_ = bsk::make_tup(((2.0f * bsk::get<1>(u1)) + bsk::get<1>(u2)), ((-2.0f * bsk::get<0>(u1)) - bsk::get<0>(u2)));
        ur = bsk::get<0>(t114_);
        ui = bsk::get<1>(t114_);
        phi_v = (phi_v + ((mbvr * ur) + (mbvi * ui)));
        u1 = _complex_mul(bsk::get<0>(t20), bsk::get<1>(t20), spvr, spvi);
        u2 = _complex_mul(bsk::get<0>(t21), bsk::get<1>(t21), smvr, smvi);
        auto t115_ = bsk::make_tup((-(bsk::get<1>(u2) - bsk::get<1>(u1))), (bsk::get<0>(u2) - bsk::get<0>(u1)));
        ur = bsk::get<0>(t115_);
        ui = bsk::get<1>(t115_);
        phi_v = (phi_v + ((zbvr * ur) + (zbvi * ui)));
        if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
            auto t116_ = bsk::make_tup(0.0f, 0.0f, 0.0f, 0.0f);
            slope_ar = bsk::get<0>(t116_);
            slope_ai = bsk::get<1>(t116_);
            slope_br = bsk::get<2>(t116_);
            slope_bi = bsk::get<3>(t116_);
            if (bsk::truth(dynamic)) {
                pair = _dynamic_pair_at(pairs, pair_index, event_base, event, atom, atom_count, active_atom);
                auto t117_ = bsk::make_tup(bsk::get<0>(pair), bsk::get<1>(pair));
                shaped_ar = bsk::get<0>(t117_);
                shaped_ai = bsk::get<1>(t117_);
                auto t118_ = _complex_mul(bsk::get<2>(pair), bsk::get<3>(pair), p1r, (-p1i));
                shaped_br = bsk::get<0>(t118_);
                shaped_bi = bsk::get<1>(t118_);
            } else {
                auto t119_ = _profile_pair_slope(profile, _table_row(profile_index, event, location, locations), alpha_value, profile_bins, profile_step);
                shaped_ar = bsk::get<0>(t119_);
                slope_ar = bsk::get<1>(t119_);
                shaped_ai = bsk::get<2>(t119_);
                slope_ai = bsk::get<3>(t119_);
                shaped_br = bsk::get<4>(t119_);
                slope_br = bsk::get<5>(t119_);
                shaped_bi = bsk::get<6>(t119_);
                slope_bi = bsk::get<7>(t119_);
                auto t120_ = _complex_mul(shaped_br, shaped_bi, p1r, (-p1i));
                shaped_br = bsk::get<0>(t120_);
                shaped_bi = bsk::get<1>(t120_);
                auto t121_ = _complex_mul(slope_br, slope_bi, p1r, (-p1i));
                slope_br = bsk::get<0>(t121_);
                slope_bi = bsk::get<1>(t121_);
            }
            auto t122_ = _spinor_adjoint(shaped_ar, shaped_ai, shaped_br, shaped_bi, spvr, spvi, smvr, smvi, rzvr, rzvi, pbvr, pbvi, mbvr, mbvi, zbvr, zbvi);
            auto grad_ar = bsk::get<0>(t122_);
            auto grad_ai = bsk::get<1>(t122_);
            auto grad_br = bsk::get<2>(t122_);
            auto grad_bi = bsk::get<3>(t122_);
            shaped_pbr = bsk::get<4>(t122_);
            shaped_pbi = bsk::get<5>(t122_);
            shaped_mbr = bsk::get<6>(t122_);
            shaped_mbi = bsk::get<7>(t122_);
            shaped_zbr = bsk::get<8>(t122_);
            shaped_zbi = bsk::get<9>(t122_);
            if (bsk::truth(dynamic)) {
                // The flip is inside the pair rather than read against it, so
                // it has no gradient here: the cotangent goes out on the
                // rotation and whatever integrated it carries the rest. ``b``
                // was turned by the phase after the pair came out, so the
                // cotangent turns back the other way.
                alpha_v = (alpha_v * 0.0f);
                auto t123_ = _complex_mul(grad_br, grad_bi, p1r, p1i);
                back_r = bsk::get<0>(t123_);
                back_i = bsk::get<1>(t123_);
                _store_pair_gradient(grad_pair, pair_index, event_base, event, atom, atom_count, bsk::band(is_rf, bsk::bnot(is_inversion)), active_atom, state_mask, grad_ar, grad_ai, back_r, back_i);
            } else {
                alpha_v = ((grad_ar * slope_ar) + (grad_ai * slope_ai));
                alpha_v = (alpha_v + ((grad_br * slope_br) + (grad_bi * slope_bi)));
            }
            // d(b e^{-i phi})/dphi is -i times it, and nothing else moves.
            phi_v = ((grad_br * shaped_bi) - (grad_bi * shaped_br));
            if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
                auto t124_ = _spinor_adjoint(shaped_ar, shaped_ai, shaped_br, shaped_bi, sbpvr, sbpvi, sbmvr, sbmvi, rbvr, rbvi, ubvr, ubvi, wbvr, wbvi, poolbr, poolbi);
                auto pool_ar = bsk::get<0>(t124_);
                auto pool_ai = bsk::get<1>(t124_);
                auto pool_pair_br = bsk::get<2>(t124_);
                auto pool_pair_bi = bsk::get<3>(t124_);
                pool_shaped_pbr = bsk::get<4>(t124_);
                pool_shaped_pbi = bsk::get<5>(t124_);
                pool_shaped_mbr = bsk::get<6>(t124_);
                pool_shaped_mbi = bsk::get<7>(t124_);
                pool_shaped_zbr = bsk::get<8>(t124_);
                pool_shaped_zbi = bsk::get<9>(t124_);
                if (bsk::truth(dynamic)) {
                    // The same pulse turned this pool, so its cotangent lands
                    // on the same row.
                    auto t125_ = _complex_mul(pool_pair_br, pool_pair_bi, p1r, p1i);
                    back_r = bsk::get<0>(t125_);
                    back_i = bsk::get<1>(t125_);
                    _store_pair_gradient(grad_pair, pair_index, event_base, event, atom, atom_count, bsk::band(is_rf, bsk::bnot(is_inversion)), active_atom, state_mask, pool_ar, pool_ai, back_r, back_i);
                } else {
                    alpha_v = (alpha_v + ((pool_ar * slope_ar) + (pool_ai * slope_ai)));
                    alpha_v = (alpha_v + ((pool_pair_br * slope_br) + (pool_pair_bi * slope_bi)));
                }
                phi_v = (phi_v + ((pool_pair_br * shaped_bi) - (pool_pair_bi * shaped_br)));
            }
        }
        if (bsk::truth((bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3)))) && bsk::truth((!bsk::truth(profiled))) && bsk::truth((!bsk::truth(dynamic)))))) {
            // The same pulse turns the exchanging pool, so its cotangent adds to
            // the flip and phase the free pool already left.
            row0 = _complex_mul(bsk::get<0>(d00), bsk::get<1>(d00), sbpvr, sbpvi);
            add1 = _complex_mul(bsk::get<0>(d01), bsk::get<1>(d01), sbmvr, sbmvi);
            add2 = _complex_mul(bsk::get<0>(d02), bsk::get<1>(d02), rbvr, rbvi);
            alpha_v = (alpha_v + (ubvr * ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2))));
            alpha_v = (alpha_v + (ubvi * ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2))));
            row0 = _complex_mul(bsk::get<0>(d01), (-bsk::get<1>(d01)), sbpvr, sbpvi);
            add1 = _complex_mul(bsk::get<0>(d00), bsk::get<1>(d00), sbmvr, sbmvi);
            add2 = _complex_mul(bsk::get<0>(d12), bsk::get<1>(d12), rbvr, rbvi);
            alpha_v = (alpha_v + (wbvr * ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2))));
            alpha_v = (alpha_v + (wbvi * ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2))));
            row0 = _complex_mul(bsk::get<0>(d20), bsk::get<1>(d20), sbpvr, sbpvi);
            add1 = _complex_mul(bsk::get<0>(d21), bsk::get<1>(d21), sbmvr, sbmvi);
            add2 = _complex_mul(bsk::get<0>(d22), bsk::get<1>(d22), rbvr, rbvi);
            alpha_v = (alpha_v + (poolbr * ((bsk::get<0>(row0) + bsk::get<0>(add1)) + bsk::get<0>(add2))));
            alpha_v = (alpha_v + (poolbi * ((bsk::get<1>(row0) + bsk::get<1>(add1)) + bsk::get<1>(add2))));
            u1 = _complex_mul(bsk::get<0>(t01), bsk::get<1>(t01), sbmvr, sbmvi);
            u2 = _complex_mul(bsk::get<0>(t02), bsk::get<1>(t02), rbvr, rbvi);
            auto t126_ = bsk::make_tup((-((2.0f * bsk::get<1>(u1)) + bsk::get<1>(u2))), ((2.0f * bsk::get<0>(u1)) + bsk::get<0>(u2)));
            ur = bsk::get<0>(t126_);
            ui = bsk::get<1>(t126_);
            phi_v = (phi_v + ((ubvr * ur) + (ubvi * ui)));
            u1 = _complex_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), sbpvr, sbpvi);
            u2 = _complex_mul(bsk::get<0>(t12), bsk::get<1>(t12), rbvr, rbvi);
            auto t127_ = bsk::make_tup(((2.0f * bsk::get<1>(u1)) + bsk::get<1>(u2)), ((-2.0f * bsk::get<0>(u1)) - bsk::get<0>(u2)));
            ur = bsk::get<0>(t127_);
            ui = bsk::get<1>(t127_);
            phi_v = (phi_v + ((wbvr * ur) + (wbvi * ui)));
            u1 = _complex_mul(bsk::get<0>(t20), bsk::get<1>(t20), sbpvr, sbpvi);
            u2 = _complex_mul(bsk::get<0>(t21), bsk::get<1>(t21), sbmvr, sbmvi);
            auto t128_ = bsk::make_tup((-(bsk::get<1>(u2) - bsk::get<1>(u1))), (bsk::get<0>(u2) - bsk::get<0>(u1)));
            ur = bsk::get<0>(t128_);
            ui = bsk::get<1>(t128_);
            phi_v = (phi_v + ((poolbr * ur) + (poolbi * ui)));
        }
        rotate = bsk::band(is_rf, bsk::bnot(is_inversion));
        grad_alpha_v = bsk::sum_x(bsk::where(rotate, alpha_v, 0.0f));
        auto grad_phi_v = bsk::sum_x(bsk::where(rotate, phi_v, 0.0f));
        if (bsk::truth((bsk::truth((pools == 1)) || bsk::truth((pools == 3))))) {
            auto turning = bsk::where(rotate, 1.0f, 0.0f);
            grad_alpha_v = (grad_alpha_v + (sat_alpha_v * turning));
            g_b0v = (g_b0v + (sat_b0_v * turning));
        }
        // Conjugate transpose of the rotation.
        n0 = _complex_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), pbvr, pbvi);
        n1 = _complex_mul(bsk::get<0>(t01), bsk::get<1>(t01), mbvr, mbvi);
        n2 = _complex_mul(bsk::get<0>(t20), (-bsk::get<1>(t20)), zbvr, zbvi);
        q0 = _complex_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), pbvr, pbvi);
        q1 = _complex_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), mbvr, mbvi);
        q2 = _complex_mul(bsk::get<0>(t21), (-bsk::get<1>(t21)), zbvr, zbvi);
        w0 = _complex_mul(bsk::get<0>(t02), (-bsk::get<1>(t02)), pbvr, pbvi);
        w1 = _complex_mul(bsk::get<0>(t12), (-bsk::get<1>(t12)), mbvr, mbvi);
        w2 = _complex_mul(bsk::get<0>(t22), (-bsk::get<1>(t22)), zbvr, zbvi);
        back_pr = ((bsk::get<0>(n0) + bsk::get<0>(n1)) + bsk::get<0>(n2));
        back_pi = ((bsk::get<1>(n0) + bsk::get<1>(n1)) + bsk::get<1>(n2));
        back_mr = ((bsk::get<0>(q0) + bsk::get<0>(q1)) + bsk::get<0>(q2));
        back_mi = ((bsk::get<1>(q0) + bsk::get<1>(q1)) + bsk::get<1>(q2));
        back_zr = ((bsk::get<0>(w0) + bsk::get<0>(w1)) + bsk::get<0>(w2));
        back_zi = ((bsk::get<1>(w0) + bsk::get<1>(w1)) + bsk::get<1>(w2));
        if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
            // A shaped pulse turned the states, so its own adjoint is what
            // goes back rather than the instant rotation's.
            auto t129_ = bsk::make_tup(shaped_pbr, shaped_pbi);
            back_pr = bsk::get<0>(t129_);
            back_pi = bsk::get<1>(t129_);
            auto t130_ = bsk::make_tup(shaped_mbr, shaped_mbi);
            back_mr = bsk::get<0>(t130_);
            back_mi = bsk::get<1>(t130_);
            auto t131_ = bsk::make_tup(shaped_zbr, shaped_zbi);
            back_zr = bsk::get<0>(t131_);
            back_zi = bsk::get<1>(t131_);
        }
        pbvr = bsk::where(rotate, back_pr, pbvr);
        pbvi = bsk::where(rotate, back_pi, pbvi);
        mbvr = bsk::where(rotate, back_mr, mbvr);
        mbvi = bsk::where(rotate, back_mi, mbvi);
        zbvr = bsk::where(rotate, back_zr, zbvr);
        zbvi = bsk::where(rotate, back_zi, zbvi);
        auto writes_flip = bsk::band(active_atom, rotate);
        bsk::atomic_add(((grad_flip + event_base) + event), (grad_alpha_v * pulse_b1), writes_flip);
        bsk::atomic_add(((grad_phase + event_base) + event), grad_phi_v, writes_flip);
        if (bsk::truth(shimmed)) {
            // A pulse's transmit gradient belongs to the shim it drives, so with
            // several it lands in that shim's row rather than in a register
            // summed over the whole train.
            bsk::atomic_add((((grad_tissue + (3 * atom_count)) + row) + atom), (grad_alpha_v * event_flip), writes_flip);
            bsk::atomic_add((((grad_tissue + (((4 + shim_rows) - 1) * atom_count)) + row) + atom), grad_phi_v, writes_flip);
        } else {
            g_b1v = (g_b1v + (grad_alpha_v * event_flip));
            g_b1pv = (g_b1pv + grad_phi_v);
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            n0 = _complex_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), ubvr, ubvi);
            n1 = _complex_mul(bsk::get<0>(t01), bsk::get<1>(t01), wbvr, wbvi);
            n2 = _complex_mul(bsk::get<0>(t20), (-bsk::get<1>(t20)), poolbr, poolbi);
            q0 = _complex_mul(bsk::get<0>(t01), (-bsk::get<1>(t01)), ubvr, ubvi);
            q1 = _complex_mul(bsk::get<0>(t00), (-bsk::get<1>(t00)), wbvr, wbvi);
            q2 = _complex_mul(bsk::get<0>(t21), (-bsk::get<1>(t21)), poolbr, poolbi);
            w0 = _complex_mul(bsk::get<0>(t02), (-bsk::get<1>(t02)), ubvr, ubvi);
            w1 = _complex_mul(bsk::get<0>(t12), (-bsk::get<1>(t12)), wbvr, wbvi);
            w2 = _complex_mul(bsk::get<0>(t22), (-bsk::get<1>(t22)), poolbr, poolbi);
            pool_back_pr = ((bsk::get<0>(n0) + bsk::get<0>(n1)) + bsk::get<0>(n2));
            pool_back_pi = ((bsk::get<1>(n0) + bsk::get<1>(n1)) + bsk::get<1>(n2));
            pool_back_mr = ((bsk::get<0>(q0) + bsk::get<0>(q1)) + bsk::get<0>(q2));
            pool_back_mi = ((bsk::get<1>(q0) + bsk::get<1>(q1)) + bsk::get<1>(q2));
            pool_back_zr = ((bsk::get<0>(w0) + bsk::get<0>(w1)) + bsk::get<0>(w2));
            pool_back_zi = ((bsk::get<1>(w0) + bsk::get<1>(w1)) + bsk::get<1>(w2));
            if (bsk::truth((bsk::truth(profiled) || bsk::truth(dynamic)))) {
                // A shaped pulse turned this pool too, so its own adjoint is
                // what goes back rather than the instant rotation's.
                auto t132_ = bsk::make_tup(pool_shaped_pbr, pool_shaped_pbi);
                pool_back_pr = bsk::get<0>(t132_);
                pool_back_pi = bsk::get<1>(t132_);
                auto t133_ = bsk::make_tup(pool_shaped_mbr, pool_shaped_mbi);
                pool_back_mr = bsk::get<0>(t133_);
                pool_back_mi = bsk::get<1>(t133_);
                auto t134_ = bsk::make_tup(pool_shaped_zbr, pool_shaped_zbi);
                pool_back_zr = bsk::get<0>(t134_);
                pool_back_zi = bsk::get<1>(t134_);
            }
            ubvr = bsk::where(rotate, pool_back_pr, ubvr);
            ubvi = bsk::where(rotate, pool_back_pi, ubvi);
            wbvr = bsk::where(rotate, pool_back_mr, wbvr);
            wbvi = bsk::where(rotate, pool_back_mi, wbvi);
            poolbr = bsk::where(rotate, pool_back_zr, poolbr);
            poolbi = bsk::where(rotate, pool_back_zi, poolbi);
            // An inversion turns the exchanging pool's longitudinal state as
            // well, so the efficiency carries what both left behind.
            g_invv = (g_invv + bsk::sum_x(bsk::where(invert, ((poolbr * (-rbvr)) + (poolbi * (-rbvi))), 0.0f)));
            poolbr = bsk::where(invert, ((-atom_inv) * poolbr), poolbr);
            poolbi = bsk::where(invert, ((-atom_inv) * poolbi), poolbi);
            auto t135_ = _shift_adjoint(ubvr, ubvi, wbvr, wbvi, state, state_mask, state_count);
            avr = bsk::get<0>(t135_);
            avi = bsk::get<1>(t135_);
            bvr = bsk::get<2>(t135_);
            bvi = bsk::get<3>(t135_);
            ubvr = bsk::where(pre_shift, avr, ubvr);
            ubvi = bsk::where(pre_shift, avi, ubvi);
            wbvr = bsk::where(pre_shift, bvr, wbvr);
            wbvi = bsk::where(pre_shift, bvi, wbvi);
        }
        auto t136_ = _shift_adjoint(pbvr, pbvi, mbvr, mbvi, state, state_mask, state_count);
        avr = bsk::get<0>(t136_);
        avi = bsk::get<1>(t136_);
        bvr = bsk::get<2>(t136_);
        bvi = bsk::get<3>(t136_);
        pbvr = bsk::where(pre_shift, avr, pbvr);
        pbvi = bsk::where(pre_shift, avi, pbvi);
        mbvr = bsk::where(pre_shift, bvr, mbvr);
        mbvi = bsk::where(pre_shift, bvi, mbvi);
        // ---- relaxation and off-resonance adjoint ----
        // The damping is homogeneous of degree one in every transverse state it
        // acts on, so its gradient times the damping itself is the cotangent
        // taken against the states the interval leaves.
        auto pq = _complex_mul(qr, qi, xpvr, xpvi);
        auto mq = _complex_mul(qr, (-qi), xmvr, xmvi);
        bare_cot_v = ((pbvr * bsk::get<0>(pq)) + (pbvi * bsk::get<1>(pq)));
        bare_cot_v = (bare_cot_v + ((mbvr * bsk::get<0>(mq)) + (mbvi * bsk::get<1>(mq))));
        auto grad_e2_v = bsk::sum_x((bare_cot_v * damp_t));
        cot2_v = ((bare_cot_v * bare2_value) * damp_t);
        pool_angle_v = empty;
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // With an exchanging pool the damping sits inside the operator, so
            // the cotangent the interval leaves is taken against the states it
            // produced rather than against a scalar the free pool multiplies.
            auto plus_r = ((((pbvr * rpvr) + (pbvi * rpvi)) + (ubvr * rbpvr)) + (ubvi * rbpvi));
            auto plus_i = ((((pbvr * rpvi) - (pbvi * rpvr)) + (ubvr * rbpvi)) - (ubvi * rbpvr));
            auto minus_r = ((((mbvr * rmvr) + (mbvi * rmvi)) + (wbvr * rbmvr)) + (wbvi * rbmvi));
            auto minus_i = ((((mbvr * rmvi) - (mbvi * rmvr)) + (wbvr * rbmvi)) - (wbvi * rbmvr));
            cot2_v = (plus_r + minus_r);
            pool_angle_v = (minus_i - plus_i);
        }
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            // ``F-`` follows the conjugate of the operator, so its cotangent
            // lands on the entry itself rather than on the conjugate of it.
            auto t137_ = bsk::make_tup(carr, cari);
            auto def_r = bsk::get<0>(t137_);
            auto def_i = bsk::get<1>(t137_);
            auto t138_ = _complex_mul(((((pbvr * xpvr) + (pbvi * xpvi)) + (mbvr * xmvr)) + (mbvi * xmvi)), ((((pbvr * xpvi) - (pbvi * xpvr)) - (mbvr * xmvi)) + (mbvi * xmvr)), def_r, def_i);
            auto t11r = bsk::get<0>(t138_);
            auto t11i = bsk::get<1>(t138_);
            auto t139_ = _complex_mul(((((pbvr * xbpvr) + (pbvi * xbpvi)) + (mbvr * xbmvr)) + (mbvi * xbmvi)), ((((pbvr * xbpvi) - (pbvi * xbpvr)) - (mbvr * xbmvi)) + (mbvi * xbmvr)), def_r, def_i);
            auto t12r = bsk::get<0>(t139_);
            auto t12i = bsk::get<1>(t139_);
            auto t140_ = _complex_mul(((((ubvr * xpvr) + (ubvi * xpvi)) + (wbvr * xmvr)) + (wbvi * xmvi)), ((((ubvr * xpvi) - (ubvi * xpvr)) - (wbvr * xmvi)) + (wbvi * xmvr)), def_r, def_i);
            auto t21r = bsk::get<0>(t140_);
            auto t21i = bsk::get<1>(t140_);
            auto t141_ = _complex_mul(((((ubvr * xbpvr) + (ubvi * xbpvi)) + (wbvr * xbmvr)) + (wbvi * xbmvi)), ((((ubvr * xbpvi) - (ubvi * xbpvr)) - (wbvr * xbmvi)) + (wbvi * xbmvr)), def_r, def_i);
            auto t22r = bsk::get<0>(t141_);
            auto t22i = bsk::get<1>(t141_);
            auto qbar11 = bsk::make_tup(bsk::sum_x(t11r), bsk::sum_x(t11i), zero, zero);
            auto qbar12 = bsk::make_tup(bsk::sum_x(t12r), bsk::sum_x(t12i), zero, zero);
            auto qbar21 = bsk::make_tup(bsk::sum_x(t21r), bsk::sum_x(t21i), zero, zero);
            auto qbar22 = bsk::make_tup(bsk::sum_x(t22r), bsk::sum_x(t22i), zero, zero);
            auto t142_ = _two_pool_transverse_adjoint_jvp(r2_value, 0.0f, r2b_value, 0.0f, atom_exchange, 0.0f, atom_bound, 0.0f, atom_free, 0.0f, atom_shift, 0.0f, dt_value, 0.0f, wout_value, 0.0f, qbar11, qbar12, qbar21, qbar22);
            auto back_r2 = bsk::get<0>(t142_);
            _q1 = bsk::get<1>(t142_);
            auto back_r2b = bsk::get<2>(t142_);
            _q2 = bsk::get<3>(t142_);
            auto back_xexch = bsk::get<4>(t142_);
            _q3 = bsk::get<5>(t142_);
            auto back_xbound = bsk::get<6>(t142_);
            _q4 = bsk::get<7>(t142_);
            auto back_xfree = bsk::get<8>(t142_);
            _q5 = bsk::get<9>(t142_);
            auto back_shift = bsk::get<10>(t142_);
            _q6 = bsk::get<11>(t142_);
            auto back_xdt = bsk::get<12>(t142_);
            _q7 = bsk::get<13>(t142_);
            auto back_xatt = bsk::get<14>(t142_);
            _q8 = bsk::get<15>(t142_);
            g_t2v = (g_t2v + (back_r2 * bsk::truediv(-1000.0f, (atom_t2 * atom_t2))));
            g_t2bv = (g_t2bv + (back_r2b * bsk::truediv(-1000.0f, (atom_t2b * atom_t2b))));
            g_exchv = (g_exchv + back_xexch);
            // The free fraction is one less the pool's, so what reaches it
            // arrives at the pool's own with the sign turned.
            g_boundv = (g_boundv + (back_xbound - back_xfree));
            if (bsk::truth((pools == 3))) {
                // The free share is one less both fractions, so what the
                // transverse operator leaves on it reaches the semisolid too.
                g_semiv = (g_semiv - back_xfree);
            }
            g_shiftv = (g_shiftv + back_shift);
            xversal_dt = back_xdt;
            xversal_att = back_xatt;
            // The pool's transverse cotangents go back through the same
            // operator, transposed.
            auto t143_ = _complex_mul(a11r, (-a11i), pbvr, pbvi);
            ur = bsk::get<0>(t143_);
            ui = bsk::get<1>(t143_);
            auto t144_ = _complex_mul(a21r, (-a21i), ubvr, ubvi);
            vr_ = bsk::get<0>(t144_);
            vi_ = bsk::get<1>(t144_);
            auto t145_ = _complex_mul((ur + vr_), (ui + vi_), carr, (-cari));
            auto nub_pr = bsk::get<0>(t145_);
            auto nub_pi = bsk::get<1>(t145_);
            auto t146_ = _complex_mul(a12r, (-a12i), pbvr, pbvi);
            ur = bsk::get<0>(t146_);
            ui = bsk::get<1>(t146_);
            auto t147_ = _complex_mul(a22r, (-a22i), ubvr, ubvi);
            vr_ = bsk::get<0>(t147_);
            vi_ = bsk::get<1>(t147_);
            auto t148_ = _complex_mul((ur + vr_), (ui + vi_), carr, (-cari));
            auto nub_qr = bsk::get<0>(t148_);
            auto nub_qi = bsk::get<1>(t148_);
            auto t149_ = _complex_mul(a11r, a11i, mbvr, mbvi);
            ur = bsk::get<0>(t149_);
            ui = bsk::get<1>(t149_);
            auto t150_ = _complex_mul(a21r, a21i, wbvr, wbvi);
            vr_ = bsk::get<0>(t150_);
            vi_ = bsk::get<1>(t150_);
            auto t151_ = _complex_mul((ur + vr_), (ui + vi_), carr, cari);
            auto nwb_pr = bsk::get<0>(t151_);
            auto nwb_pi = bsk::get<1>(t151_);
            auto t152_ = _complex_mul(a12r, a12i, mbvr, mbvi);
            ur = bsk::get<0>(t152_);
            ui = bsk::get<1>(t152_);
            auto t153_ = _complex_mul(a22r, a22i, wbvr, wbvi);
            vr_ = bsk::get<0>(t153_);
            vi_ = bsk::get<1>(t153_);
            auto t154_ = _complex_mul((ur + vr_), (ui + vi_), carr, cari);
            auto nwb_qr = bsk::get<0>(t154_);
            auto nwb_qi = bsk::get<1>(t154_);
            auto t155_ = bsk::make_tup(nub_pr, nub_pi);
            pbvr = bsk::get<0>(t155_);
            pbvi = bsk::get<1>(t155_);
            auto t156_ = bsk::make_tup(nub_qr, nub_qi);
            ubvr = bsk::get<0>(t156_);
            ubvi = bsk::get<1>(t156_);
            auto t157_ = bsk::make_tup(nwb_pr, nwb_pi);
            mbvr = bsk::get<0>(t157_);
            mbvi = bsk::get<1>(t157_);
            auto t158_ = bsk::make_tup(nwb_qr, nwb_qi);
            wbvr = bsk::get<0>(t158_);
            wbvi = bsk::get<1>(t158_);
        }
        per_angle_v = pool_angle_v;
        if (bsk::truth((bsk::truth((pools != 2)) && bsk::truth((pools != 3)) && bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))))) {
            auto po = _complex_mul(ovr, ovi, xpvr, xpvi);
            auto mo = _complex_mul(ovr, (-ovi), xmvr, xmvi);
            // A turn of the transverse states and the off-resonance angle are
            // the same derivative; only the weight each order carries differs.
            per_angle_v = ((pbvr * (-bsk::get<1>(po))) + (pbvi * bsk::get<0>(po)));
            per_angle_v = (per_angle_v - ((mbvr * (-bsk::get<1>(mo))) + (mbvi * bsk::get<0>(mo))));
        }
        grad_angle_v = zero;
        if (bsk::truth((bsk::truth(off_axis) || bsk::truth(moving)))) {
            grad_angle_v = bsk::sum_x(per_angle_v);
        }
        e1_v = empty;
        grad_e1_v = zero;
        long_damp_v = empty;
        attenuation_v = zero;
        two_pool_dt_v = zero;
        if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
            attenuation_v = (attenuation_v + xversal_att);
            two_pool_dt_v = (two_pool_dt_v + xversal_dt);
        }
        zangle_v = empty;
        if (bsk::truth((pools == 3))) {
            // The nine entries of the mixing operator and the three
            // recoveries, summed over the orders that share them, then pushed
            // back through the closed form once for the whole interval.
            auto t159_ = _complex_mul(spin_r, spin_i, xzvr, xzvi);
            spun_fr = bsk::get<0>(t159_);
            spun_fi = bsk::get<1>(t159_);
            auto t160_ = _complex_mul(spin_r, spin_i, xbvr, xbvi);
            spun_br = bsk::get<0>(t160_);
            spun_bi = bsk::get<1>(t160_);
            auto t161_ = _complex_mul(spin_r, spin_i, xcvr, xcvi);
            auto spun_cr = bsk::get<0>(t161_);
            auto spun_ci = bsk::get<1>(t161_);
            auto e11_v = ((zbvr * spun_fr) + (zbvi * spun_fi));
            auto e12_v = ((zbvr * spun_br) + (zbvi * spun_bi));
            auto e13_v = ((zbvr * spun_cr) + (zbvi * spun_ci));
            auto e21_v = ((poolbr * spun_fr) + (poolbi * spun_fi));
            auto e22_v = ((poolbr * spun_br) + (poolbi * spun_bi));
            auto e23_v = ((poolbr * spun_cr) + (poolbi * spun_ci));
            auto e31_v = ((semibr * spun_fr) + (semibi * spun_fi));
            auto e32_v = ((semibr * spun_br) + (semibi * spun_bi));
            auto e33_v = ((semibr * spun_cr) + (semibi * spun_ci));
            if (bsk::truth(tabulated)) {
                // Every gradient but the interval's own is linear in these
                // twelve, so the events sharing a length pool them here and
                // pay the closed form once each after the walk back.
                auto bar11 = bsk::sum_x(e11_v);
                auto bar12 = bsk::sum_x(e12_v);
                auto bar13 = bsk::sum_x(e13_v);
                auto bar21 = bsk::sum_x(e21_v);
                auto bar22 = bsk::sum_x(e22_v);
                auto bar23 = bsk::sum_x(e23_v);
                auto bar31 = bsk::sum_x(e31_v);
                auto bar32 = bsk::sum_x(e32_v);
                auto bar33 = bsk::sum_x(e33_v);
                auto bar_free = bsk::sum_x(bsk::where((state == 0), zbvr, nil));
                auto bar_pool_b = bsk::sum_x(bsk::where((state == 0), poolbr, nil));
                auto bar_bound = bsk::sum_x(bsk::where((state == 0), semibr, nil));
                held = (pool_bars + (((local * row_count) + pool_row) * 12));
                bsk::st((held + 0), (bsk::ld((held + 0), active_atom, 0.0f) + bar11), active_atom);
                bsk::st((held + 1), (bsk::ld((held + 1), active_atom, 0.0f) + bar12), active_atom);
                bsk::st((held + 2), (bsk::ld((held + 2), active_atom, 0.0f) + bar13), active_atom);
                bsk::st((held + 3), (bsk::ld((held + 3), active_atom, 0.0f) + bar21), active_atom);
                bsk::st((held + 4), (bsk::ld((held + 4), active_atom, 0.0f) + bar22), active_atom);
                bsk::st((held + 5), (bsk::ld((held + 5), active_atom, 0.0f) + bar23), active_atom);
                bsk::st((held + 6), (bsk::ld((held + 6), active_atom, 0.0f) + bar31), active_atom);
                bsk::st((held + 7), (bsk::ld((held + 7), active_atom, 0.0f) + bar32), active_atom);
                bsk::st((held + 8), (bsk::ld((held + 8), active_atom, 0.0f) + bar33), active_atom);
                bsk::st((held + 9), (bsk::ld((held + 9), active_atom, 0.0f) + bar_free), active_atom);
                bsk::st((held + 10), (bsk::ld((held + 10), active_atom, 0.0f) + bar_pool_b), active_atom);
                bsk::st((held + 11), (bsk::ld((held + 11), active_atom, 0.0f) + bar_bound), active_atom);
                auto t162_ = _three_pool_interval_adjoint(pool_table, pool_row, atom, atom_count, active_atom, r1_value, r1b_value, r1c_value, atom_exchange, atom_semisolid_exchange, atom_bound, atom_semisolid, hold_value, bar11, bar12, bar13, bar21, bar22, bar23, bar31, bar32, bar33, bar_free, bar_pool_b, bar_bound);
                back_dt = bsk::get<0>(t162_);
                back_att = bsk::get<1>(t162_);
                attenuation_v = (attenuation_v + back_att);
                two_pool_dt_v = (two_pool_dt_v + back_dt);
            } else {
                auto t163_ = _three_pool_step_adjoint_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, dt_value, nil, hold_value, nil, bsk::sum_x(e11_v), nil, bsk::sum_x(e12_v), nil, bsk::sum_x(e13_v), nil, bsk::sum_x(e21_v), nil, bsk::sum_x(e22_v), nil, bsk::sum_x(e23_v), nil, bsk::sum_x(e31_v), nil, bsk::sum_x(e32_v), nil, bsk::sum_x(e33_v), nil, bsk::sum_x(bsk::where((state == 0), zbvr, nil)), nil, bsk::sum_x(bsk::where((state == 0), poolbr, nil)), nil, bsk::sum_x(bsk::where((state == 0), semibr, nil)), nil, three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, three_def_00, three_dif_00, three_def_01, three_dif_01, three_def_02, three_dif_02, three_def_10, three_dif_10, three_def_11, three_dif_11, three_def_12, three_dif_12, three_def_20, three_dif_20, three_def_21, three_dif_21, three_def_22, three_dif_22, narrow);
                back_r1 = bsk::get<0>(t163_);
                back_r1b = bsk::get<1>(t163_);
                back_r1c = bsk::get<2>(t163_);
                back_exch = bsk::get<3>(t163_);
                back_sexch = bsk::get<4>(t163_);
                back_bound = bsk::get<5>(t163_);
                back_semi = bsk::get<6>(t163_);
                back_dt = bsk::get<7>(t163_);
                back_att = bsk::get<8>(t163_);
                _q1 = bsk::get<9>(t163_);
                _q2 = bsk::get<10>(t163_);
                _q3 = bsk::get<11>(t163_);
                _q4 = bsk::get<12>(t163_);
                _q5 = bsk::get<13>(t163_);
                _q6 = bsk::get<14>(t163_);
                _q7 = bsk::get<15>(t163_);
                _q8 = bsk::get<16>(t163_);
                _q9 = bsk::get<17>(t163_);
                g_t1v = (g_t1v + (back_r1 * bsk::truediv(-1000.0f, (atom_t1 * atom_t1))));
                g_t1bv = (g_t1bv + (back_r1b * bsk::truediv(-1000.0f, (atom_t1b * atom_t1b))));
                g_t1cv = (g_t1cv + (back_r1c * bsk::truediv(-1000.0f, (atom_t1c * atom_t1c))));
                g_exchv = (g_exchv + back_exch);
                g_sexchv = (g_sexchv + back_sexch);
                g_boundv = (g_boundv + back_bound);
                g_semiv = (g_semiv + back_semi);
                // Both halves of the interval reach the same two, so the
                // transverse pass has already put its share here.
                attenuation_v = (attenuation_v + back_att);
                two_pool_dt_v = (two_pool_dt_v + back_dt);
            }
            // All three pools take the same per-order damping and turn, so each
            // collects the cotangent of the mixture that reached it.
            auto t164_ = _complex_mul(spin_r, spin_i, mix_fr, mix_fi);
            sfr = bsk::get<0>(t164_);
            sfi = bsk::get<1>(t164_);
            auto t165_ = _complex_mul(spin_r, spin_i, mix_br, mix_bi);
            sbr = bsk::get<0>(t165_);
            sbi = bsk::get<1>(t165_);
            auto t166_ = _complex_mul(spin_r, spin_i, mix_cr, mix_ci);
            auto scr = bsk::get<0>(t166_);
            auto sci = bsk::get<1>(t166_);
            long_damp_v = ((((zbvr * sfr) + (zbvi * sfi)) + ((poolbr * sbr) + (poolbi * sbi))) + ((semibr * scr) + (semibi * sci)));
            if (bsk::truth(moving)) {
                zangle_v = ((((zbvr * (-sfi)) + (zbvi * sfr)) + ((poolbr * (-sbi)) + (poolbi * sbr))) + ((semibr * (-sci)) + (semibi * scr)));
            }
            auto t167_ = _complex_mul((w11 * spin_r), (-(w11 * spin_i)), zbvr, zbvi);
            col_fr = bsk::get<0>(t167_);
            col_fi = bsk::get<1>(t167_);
            auto t168_ = _complex_mul((w21 * spin_r), (-(w21 * spin_i)), poolbr, poolbi);
            part_r = bsk::get<0>(t168_);
            part_i = bsk::get<1>(t168_);
            auto t169_ = bsk::make_tup((col_fr + part_r), (col_fi + part_i));
            col_fr = bsk::get<0>(t169_);
            col_fi = bsk::get<1>(t169_);
            auto t170_ = _complex_mul((w31 * spin_r), (-(w31 * spin_i)), semibr, semibi);
            part_r = bsk::get<0>(t170_);
            part_i = bsk::get<1>(t170_);
            auto t171_ = bsk::make_tup((col_fr + part_r), (col_fi + part_i));
            col_fr = bsk::get<0>(t171_);
            col_fi = bsk::get<1>(t171_);
            auto t172_ = _complex_mul((w12 * spin_r), (-(w12 * spin_i)), zbvr, zbvi);
            col_br = bsk::get<0>(t172_);
            col_bi = bsk::get<1>(t172_);
            auto t173_ = _complex_mul((w22 * spin_r), (-(w22 * spin_i)), poolbr, poolbi);
            part_r = bsk::get<0>(t173_);
            part_i = bsk::get<1>(t173_);
            auto t174_ = bsk::make_tup((col_br + part_r), (col_bi + part_i));
            col_br = bsk::get<0>(t174_);
            col_bi = bsk::get<1>(t174_);
            auto t175_ = _complex_mul((w32 * spin_r), (-(w32 * spin_i)), semibr, semibi);
            part_r = bsk::get<0>(t175_);
            part_i = bsk::get<1>(t175_);
            auto t176_ = bsk::make_tup((col_br + part_r), (col_bi + part_i));
            col_br = bsk::get<0>(t176_);
            col_bi = bsk::get<1>(t176_);
            auto t177_ = _complex_mul((w13 * spin_r), (-(w13 * spin_i)), zbvr, zbvi);
            col_cr = bsk::get<0>(t177_);
            col_ci = bsk::get<1>(t177_);
            auto t178_ = _complex_mul((w23 * spin_r), (-(w23 * spin_i)), poolbr, poolbi);
            part_r = bsk::get<0>(t178_);
            part_i = bsk::get<1>(t178_);
            auto t179_ = bsk::make_tup((col_cr + part_r), (col_ci + part_i));
            col_cr = bsk::get<0>(t179_);
            col_ci = bsk::get<1>(t179_);
            auto t180_ = _complex_mul((w33 * spin_r), (-(w33 * spin_i)), semibr, semibi);
            part_r = bsk::get<0>(t180_);
            part_i = bsk::get<1>(t180_);
            auto t181_ = bsk::make_tup((col_cr + part_r), (col_ci + part_i));
            col_cr = bsk::get<0>(t181_);
            col_ci = bsk::get<1>(t181_);
            auto t182_ = bsk::make_tup(col_fr, col_fi);
            zbvr = bsk::get<0>(t182_);
            zbvi = bsk::get<1>(t182_);
            auto t183_ = bsk::make_tup(col_br, col_bi);
            poolbr = bsk::get<0>(t183_);
            poolbi = bsk::get<1>(t183_);
            auto t184_ = bsk::make_tup(col_cr, col_ci);
            semibr = bsk::get<0>(t184_);
            semibi = bsk::get<1>(t184_);
        } else if (bsk::truth((pools > 0))) {
            // The four entries of the exchange operator and the two recoveries,
            // summed over the orders that share them, then pushed back through
            // the closed form once for the whole interval.
            auto t185_ = _complex_mul(spin_r, spin_i, xzvr, xzvi);
            spun_fr = bsk::get<0>(t185_);
            spun_fi = bsk::get<1>(t185_);
            auto t186_ = _complex_mul(spin_r, spin_i, xbvr, xbvi);
            spun_br = bsk::get<0>(t186_);
            spun_bi = bsk::get<1>(t186_);
            auto bar_e11 = bsk::sum_x(((zbvr * spun_fr) + (zbvi * spun_fi)));
            auto bar_e12 = bsk::sum_x(((zbvr * spun_br) + (zbvi * spun_bi)));
            auto bar_e21 = bsk::sum_x(((poolbr * spun_fr) + (poolbi * spun_fi)));
            auto bar_e22 = bsk::sum_x(((poolbr * spun_br) + (poolbi * spun_bi)));
            auto rec_f = bsk::sum_x(bsk::where((state == 0), zbvr, 0.0f));
            auto rec_b = bsk::sum_x(bsk::where((state == 0), poolbr, 0.0f));
            auto t187_ = _two_pool_step_adjoint_jvp(r1_value, 0.0f, r1b_value, 0.0f, atom_exchange, 0.0f, atom_bound, 0.0f, dt_value, 0.0f, wout_value, 0.0f, bar_e11, 0.0f, bar_e12, 0.0f, bar_e21, 0.0f, bar_e22, 0.0f, rec_f, 0.0f, rec_b, 0.0f);
            back_r1 = bsk::get<0>(t187_);
            back_r1b = bsk::get<1>(t187_);
            back_exch = bsk::get<2>(t187_);
            back_bound = bsk::get<3>(t187_);
            back_dt = bsk::get<4>(t187_);
            back_att = bsk::get<5>(t187_);
            auto _t1 = bsk::get<6>(t187_);
            auto _t2 = bsk::get<7>(t187_);
            auto _t3 = bsk::get<8>(t187_);
            auto _t4 = bsk::get<9>(t187_);
            auto _t5 = bsk::get<10>(t187_);
            auto _t6 = bsk::get<11>(t187_);
            // r1 = 1000/t1, so a rate gradient reaches the time through the
            // square of it.
            g_t1v = (g_t1v + (back_r1 * bsk::truediv(-1000.0f, (atom_t1 * atom_t1))));
            g_t1bv = (g_t1bv + (back_r1b * bsk::truediv(-1000.0f, (atom_t1b * atom_t1b))));
            g_exchv = (g_exchv + back_exch);
            g_boundv = (g_boundv + back_bound);
            // Both halves of the interval reach the same two, so the
            // transverse pass has already put its share here.
            attenuation_v = (attenuation_v + back_att);
            two_pool_dt_v = (two_pool_dt_v + back_dt);
            // Both pools take the same per-order damping and turn, so each
            // collects the cotangent of the mixture that reached it.
            auto t188_ = _complex_mul(spin_r, spin_i, mix_fr, mix_fi);
            sfr = bsk::get<0>(t188_);
            sfi = bsk::get<1>(t188_);
            auto t189_ = _complex_mul(spin_r, spin_i, mix_br, mix_bi);
            sbr = bsk::get<0>(t189_);
            sbi = bsk::get<1>(t189_);
            long_damp_v = (((zbvr * sfr) + (zbvi * sfi)) + ((poolbr * sbr) + (poolbi * sbi)));
            if (bsk::truth(moving)) {
                zangle_v = (((zbvr * (-sfi)) + (zbvi * sfr)) + ((poolbr * (-sbi)) + (poolbi * sbr)));
            }
            auto t190_ = _complex_mul((pe11 * spin_r), (-(pe11 * spin_i)), zbvr, zbvi);
            back_zr = bsk::get<0>(t190_);
            back_zi = bsk::get<1>(t190_);
            auto t191_ = _complex_mul((pe21 * spin_r), (-(pe21 * spin_i)), poolbr, poolbi);
            auto cross_zr = bsk::get<0>(t191_);
            auto cross_zi = bsk::get<1>(t191_);
            auto t192_ = _complex_mul((pe12 * spin_r), (-(pe12 * spin_i)), zbvr, zbvi);
            auto back_br = bsk::get<0>(t192_);
            auto back_bi = bsk::get<1>(t192_);
            auto t193_ = _complex_mul((pe22 * spin_r), (-(pe22 * spin_i)), poolbr, poolbi);
            auto cross_br = bsk::get<0>(t193_);
            auto cross_bi = bsk::get<1>(t193_);
            poolbr = (back_br + cross_br);
            poolbi = (back_bi + cross_bi);
            zbvr = (back_zr + cross_zr);
            zbvi = (back_zi + cross_zi);
        } else {
            auto spun = _complex_mul(szr, szi, xzvr, xzvi);
            e1_v = ((zbvr * bsk::get<0>(spun)) + (zbvi * bsk::get<1>(spun)));
            grad_e1_v = bsk::sum_x((e1_v * damp_z));
            grad_e1_v = (grad_e1_v - bsk::sum_x(bsk::where((state == 0), zbvr, 0.0f)));
            // The longitudinal states turn too, and by a whole order rather
            // than the transverse half-order more.
            if (bsk::truth(moving)) {
                auto zo = _complex_mul(lvr, lvi, xzvr, xzvi);
                zangle_v = ((zbvr * (-bsk::get<1>(zo))) + (zbvi * bsk::get<0>(zo)));
            }
            long_damp_v = ((e1_v * bare1_value) * damp_z);
            auto t194_ = _complex_mul(lvr, (-lvi), zbvr, zbvi);
            zbvr = bsk::get<0>(t194_);
            zbvi = bsk::get<1>(t194_);
        }
        spread_v = zero;
        if (bsk::truth(diffusing)) {
            // The rate and the interval multiply every order's b-weight, so
            // both take a weighted sum rather than one scalar. Order zero
            // carries no longitudinal weight, which keeps recovery out of this.
            auto weighted_v = ((long_damp_v * longitudinal_weight) + (cot2_v * transverse_weight));
            spread_v = bsk::sum_x(weighted_v);
            g_diffv = (g_diffv + ((-spread_v) * dt_value));
        }
        wound_v = zero;
        wash_v = zero;
        if (bsk::truth(moving)) {
            wound_v = bsk::sum_x(((per_angle_v * (order + 0.5f)) + (zangle_v * order)));
            g_flowv = (g_flowv + ((-wound_v) * dt_value));
            // Washout scales both relaxation factors, so its gradient is the
            // one they already carry, taken against the factors before that
            // scaling. Past the clamp the interval has replaced the voxel
            // outright and nothing further depends on the rate.
            auto live = bsk::cast<float>(((atom_washout * dt_value) < 1.0f));
            auto transverse_dry = bsk::select(bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3)))), zero, (grad_e2_v * dry2_value));
            wash_v = ((-live) * (((grad_e1_v * dry1_value) + transverse_dry) + attenuation_v));
            g_washv = (g_washv + (wash_v * dt_value));
        }
        if (bsk::truth((bsk::truth((pools != 2)) && bsk::truth((pools != 3))))) {
            auto t195_ = _complex_mul(ovr, (-ovi), pbvr, pbvi);
            pbvr = bsk::get<0>(t195_);
            pbvi = bsk::get<1>(t195_);
            auto t196_ = _complex_mul(ovr, ovi, mbvr, mbvi);
            mbvr = bsk::get<0>(t196_);
            mbvi = bsk::get<1>(t196_);
        }
        auto inverse1_value = bsk::truediv(1000.0f, (atom_t1 * atom_t1));
        auto inverse2_value = bsk::truediv(1000.0f, (atom_t2 * atom_t2));
        g_t1v = (g_t1v + (grad_e1_v * ((bare1_value * dt_value) * inverse1_value)));
        if (bsk::truth((bsk::truth((pools != 2)) && bsk::truth((pools != 3))))) {
            g_t2v = (g_t2v + (grad_e2_v * ((bare2_value * dt_value) * inverse2_value)));
        }
        auto turn = -6.283185307179586f;
        g_b0v = (g_b0v + (grad_angle_v * (turn * dt_value)));
        duration_v = ((-grad_e1_v) * (r1_value * bare1_value));
        if (bsk::truth((bsk::truth((pools != 2)) && bsk::truth((pools != 3))))) {
            duration_v = (duration_v - (grad_e2_v * (r2_value * bare2_value)));
        }
        duration_v = (duration_v + ((grad_angle_v * (turn * atom_b0)) + two_pool_dt_v));
        duration_v = (duration_v + (((-spread_v) * atom_damping) - (wound_v * atom_flow)));
        duration_v = (duration_v + (wash_v * atom_washout));
        bsk::atomic_add(((grad_duration + event_base) + event), duration_v, active_atom);
    }
    if (bsk::truth((bsk::truth((pools == 3)) && bsk::truth(tabulated)))) {
        // One closed form per distinct length rather than one per event. The
        // walk back pooled the cotangents the eigenvalues are pushed through,
        // and the closed form is linear in them, so the pieces of the sum are
        // the sum of the pieces.
        for (bsk::index_t row = 0; row < row_count; row += 1) {
            held = (pool_bars + (((local * row_count) + row) * 12));
            auto row_dt = (bsk::ld((pool_durations + row)) + zero);
            auto one_att = bsk::select(bsk::truth(moving), _washout(atom_washout, row_dt), (1.0f + (0.0f * row_dt)));
            nil = (0.0f * row_dt);
            auto t197_ = _three_pool_pieces_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, row_dt, nil, narrow);
            three_free = bsk::get<0>(t197_);
            three_d_free = bsk::get<1>(t197_);
            three_pool_b = bsk::get<2>(t197_);
            three_d_pool_b = bsk::get<3>(t197_);
            three_pool_c = bsk::get<4>(t197_);
            three_d_pool_c = bsk::get<5>(t197_);
            three_a00 = bsk::get<6>(t197_);
            three_d_a00 = bsk::get<7>(t197_);
            three_a01 = bsk::get<8>(t197_);
            three_d_a01 = bsk::get<9>(t197_);
            three_a02 = bsk::get<10>(t197_);
            three_d_a02 = bsk::get<11>(t197_);
            three_a10 = bsk::get<12>(t197_);
            three_d_a10 = bsk::get<13>(t197_);
            three_a11 = bsk::get<14>(t197_);
            three_d_a11 = bsk::get<15>(t197_);
            three_a20 = bsk::get<16>(t197_);
            three_d_a20 = bsk::get<17>(t197_);
            three_a22 = bsk::get<18>(t197_);
            three_d_a22 = bsk::get<19>(t197_);
            three_s00 = bsk::get<20>(t197_);
            three_d_s00 = bsk::get<21>(t197_);
            three_s11 = bsk::get<22>(t197_);
            three_d_s11 = bsk::get<23>(t197_);
            three_s22 = bsk::get<24>(t197_);
            three_d_s22 = bsk::get<25>(t197_);
            three_minors = bsk::get<26>(t197_);
            three_d_minors = bsk::get<27>(t197_);
            three_sum_flat = bsk::get<28>(t197_);
            three_sum_linear = bsk::get<29>(t197_);
            three_sum_square = bsk::get<30>(t197_);
            three_d_sum_flat = bsk::get<31>(t197_);
            three_d_sum_linear = bsk::get<32>(t197_);
            three_d_sum_square = bsk::get<33>(t197_);
            three_lift = bsk::get<34>(t197_);
            three_d_lift = bsk::get<35>(t197_);
            three_low = bsk::get<36>(t197_);
            three_middle = bsk::get<37>(t197_);
            three_d_low = bsk::get<38>(t197_);
            three_d_middle = bsk::get<39>(t197_);
            three_leading = bsk::get<40>(t197_);
            three_d_leading = bsk::get<41>(t197_);
            three_first = bsk::get<42>(t197_);
            three_d_first = bsk::get<43>(t197_);
            three_second = bsk::get<44>(t197_);
            three_d_second = bsk::get<45>(t197_);
            three_determinant = bsk::get<46>(t197_);
            three_d_determinant = bsk::get<47>(t197_);
            three_high = bsk::get<48>(t197_);
            three_d_high = bsk::get<49>(t197_);
            three_radius = bsk::get<50>(t197_);
            three_d_radius = bsk::get<51>(t197_);
            three_cube = bsk::get<52>(t197_);
            three_raw = bsk::get<53>(t197_);
            three_d_raw = bsk::get<54>(t197_);
            three_argument = bsk::get<55>(t197_);
            three_inside_limit = bsk::get<56>(t197_);
            three_angle = bsk::get<57>(t197_);
            three_d_angle = bsk::get<58>(t197_);
            three_centre = bsk::get<59>(t197_);
            three_d_centre = bsk::get<60>(t197_);
            three_trailing = bsk::get<61>(t197_);
            three_d_trailing = bsk::get<62>(t197_);
            three_guarded = bsk::get<63>(t197_);
            three_d_guarded = bsk::get<64>(t197_);
            three_q00 = bsk::get<65>(t197_);
            three_d_q00 = bsk::get<66>(t197_);
            three_q01 = bsk::get<67>(t197_);
            three_d_q01 = bsk::get<68>(t197_);
            three_q02 = bsk::get<69>(t197_);
            three_d_q02 = bsk::get<70>(t197_);
            three_q10 = bsk::get<71>(t197_);
            three_d_q10 = bsk::get<72>(t197_);
            three_q11 = bsk::get<73>(t197_);
            three_d_q11 = bsk::get<74>(t197_);
            three_q12 = bsk::get<75>(t197_);
            three_d_q12 = bsk::get<76>(t197_);
            three_q20 = bsk::get<77>(t197_);
            three_d_q20 = bsk::get<78>(t197_);
            three_q21 = bsk::get<79>(t197_);
            three_d_q21 = bsk::get<80>(t197_);
            three_q22 = bsk::get<81>(t197_);
            three_d_q22 = bsk::get<82>(t197_);
            auto t198_ = _three_pool_assemble_jvp(three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, narrow);
            three_def_00 = bsk::get<0>(t198_);
            three_dif_00 = bsk::get<1>(t198_);
            three_def_01 = bsk::get<2>(t198_);
            three_dif_01 = bsk::get<3>(t198_);
            three_def_02 = bsk::get<4>(t198_);
            three_dif_02 = bsk::get<5>(t198_);
            three_def_10 = bsk::get<6>(t198_);
            three_dif_10 = bsk::get<7>(t198_);
            three_def_11 = bsk::get<8>(t198_);
            three_dif_11 = bsk::get<9>(t198_);
            three_def_12 = bsk::get<10>(t198_);
            three_dif_12 = bsk::get<11>(t198_);
            three_def_20 = bsk::get<12>(t198_);
            three_dif_20 = bsk::get<13>(t198_);
            three_def_21 = bsk::get<14>(t198_);
            three_dif_21 = bsk::get<15>(t198_);
            three_def_22 = bsk::get<16>(t198_);
            three_dif_22 = bsk::get<17>(t198_);
            auto t199_ = _three_pool_step_adjoint_jvp(r1_value, nil, r1b_value, nil, r1c_value, nil, atom_exchange, nil, atom_semisolid_exchange, nil, atom_bound, nil, atom_semisolid, nil, row_dt, nil, one_att, nil, bsk::ld((held + 0), active_atom, 0.0f), nil, bsk::ld((held + 1), active_atom, 0.0f), nil, bsk::ld((held + 2), active_atom, 0.0f), nil, bsk::ld((held + 3), active_atom, 0.0f), nil, bsk::ld((held + 4), active_atom, 0.0f), nil, bsk::ld((held + 5), active_atom, 0.0f), nil, bsk::ld((held + 6), active_atom, 0.0f), nil, bsk::ld((held + 7), active_atom, 0.0f), nil, bsk::ld((held + 8), active_atom, 0.0f), nil, bsk::ld((held + 9), active_atom, 0.0f), nil, bsk::ld((held + 10), active_atom, 0.0f), nil, bsk::ld((held + 11), active_atom, 0.0f), nil, three_free, three_d_free, three_pool_b, three_d_pool_b, three_pool_c, three_d_pool_c, three_a00, three_d_a00, three_a01, three_d_a01, three_a02, three_d_a02, three_a10, three_d_a10, three_a11, three_d_a11, three_a20, three_d_a20, three_a22, three_d_a22, three_s00, three_d_s00, three_s11, three_d_s11, three_s22, three_d_s22, three_minors, three_d_minors, three_sum_flat, three_sum_linear, three_sum_square, three_d_sum_flat, three_d_sum_linear, three_d_sum_square, three_lift, three_d_lift, three_low, three_middle, three_d_low, three_d_middle, three_leading, three_d_leading, three_first, three_d_first, three_second, three_d_second, three_determinant, three_d_determinant, three_high, three_d_high, three_radius, three_d_radius, three_cube, three_raw, three_d_raw, three_argument, three_inside_limit, three_angle, three_d_angle, three_centre, three_d_centre, three_trailing, three_d_trailing, three_guarded, three_d_guarded, three_q00, three_d_q00, three_q01, three_d_q01, three_q02, three_d_q02, three_q10, three_d_q10, three_q11, three_d_q11, three_q12, three_d_q12, three_q20, three_d_q20, three_q21, three_d_q21, three_q22, three_d_q22, three_def_00, three_dif_00, three_def_01, three_dif_01, three_def_02, three_dif_02, three_def_10, three_dif_10, three_def_11, three_dif_11, three_def_12, three_dif_12, three_def_20, three_dif_20, three_def_21, three_dif_21, three_def_22, three_dif_22, narrow);
            back_r1 = bsk::get<0>(t199_);
            back_r1b = bsk::get<1>(t199_);
            back_r1c = bsk::get<2>(t199_);
            back_exch = bsk::get<3>(t199_);
            back_sexch = bsk::get<4>(t199_);
            back_bound = bsk::get<5>(t199_);
            back_semi = bsk::get<6>(t199_);
            back_dt = bsk::get<7>(t199_);
            back_att = bsk::get<8>(t199_);
            _q1 = bsk::get<9>(t199_);
            _q2 = bsk::get<10>(t199_);
            _q3 = bsk::get<11>(t199_);
            _q4 = bsk::get<12>(t199_);
            _q5 = bsk::get<13>(t199_);
            _q6 = bsk::get<14>(t199_);
            _q7 = bsk::get<15>(t199_);
            _q8 = bsk::get<16>(t199_);
            _q9 = bsk::get<17>(t199_);
            g_t1v = (g_t1v + (back_r1 * bsk::truediv(-1000.0f, (atom_t1 * atom_t1))));
            g_t1bv = (g_t1bv + (back_r1b * bsk::truediv(-1000.0f, (atom_t1b * atom_t1b))));
            g_t1cv = (g_t1cv + (back_r1c * bsk::truediv(-1000.0f, (atom_t1c * atom_t1c))));
            g_exchv = (g_exchv + back_exch);
            g_sexchv = (g_sexchv + back_sexch);
            g_boundv = (g_boundv + back_bound);
            g_semiv = (g_semiv + back_semi);
        }
    }
    auto velocity_v = ((g_flowv * flow_scale) + ((g_washv * direction) * washout_scale));
    auto values = bsk::make_tup(g_t1v, g_t2v, g_m0v, g_b1v, g_b1pv, g_b0v, g_invv, g_diffv, velocity_v);
    if (bsk::truth((pools > 0))) {
        // The fraction also sets where each pool starts, which the walk back
        // reaches last.
        g_boundv = (g_boundv + bsk::sum_x(bsk::where((state == 0), (poolbr - zbvr), 0.0f)));
    }
    if (bsk::truth((pools == 3))) {
        g_semiv = (g_semiv + bsk::sum_x(bsk::where((state == 0), (semibr - zbvr), 0.0f)));
        auto semisolid_row = (9 + (2 * (shim_rows - 1)));
        bsk::atomic_add(((grad_tissue + (semisolid_row * atom_count)) + atom), g_semiv, active_atom);
        bsk::atomic_add(((grad_tissue + ((semisolid_row + 1) * atom_count)) + atom), g_sexchv, active_atom);
        bsk::atomic_add(((grad_tissue + ((semisolid_row + 2) * atom_count)) + atom), g_t1cv, active_atom);
    }
    if (bsk::truth((bsk::truth((pools == 2)) || bsk::truth((pools == 3))))) {
        base_row = (12 + (2 * (shim_rows - 1)));
        bsk::atomic_add(((grad_tissue + (base_row * atom_count)) + atom), g_boundv, active_atom);
        bsk::atomic_add(((grad_tissue + ((base_row + 1) * atom_count)) + atom), g_exchv, active_atom);
        bsk::atomic_add(((grad_tissue + ((base_row + 2) * atom_count)) + atom), g_t1bv, active_atom);
        bsk::atomic_add(((grad_tissue + ((base_row + 3) * atom_count)) + atom), g_t2bv, active_atom);
        bsk::atomic_add(((grad_tissue + ((base_row + 4) * atom_count)) + atom), g_shiftv, active_atom);
    }
    if (bsk::truth((pools == 1))) {
        base_row = (9 + (2 * (shim_rows - 1)));
        bsk::atomic_add(((grad_tissue + (base_row * atom_count)) + atom), g_boundv, active_atom);
        bsk::atomic_add(((grad_tissue + ((base_row + 1) * atom_count)) + atom), g_exchv, active_atom);
        bsk::atomic_add(((grad_tissue + ((base_row + 2) * atom_count)) + atom), g_t1bv, active_atom);
    }
    bsk::static_for<0, 9, 1>([&](auto parameter_c) {
        constexpr std::int64_t parameter = decltype(parameter_c)::value;
        // The transmit pair went to its shim's row above when there is more
        // than one; the rest sit past whatever rows that pair took.
        if (bsk::truth((bsk::truth((!bsk::truth(shimmed))) || bsk::truth((bsk::truth((parameter != 3)) && bsk::truth((parameter != 4))))))) {
            auto plane = bsk::select(bsk::truth((parameter < 3)), parameter, (parameter + (2 * (shim_rows - 1))));
            bsk::atomic_add(((grad_tissue + (plane * atom_count)) + atom), bsk::get<parameter>(values), active_atom);
        }
    });
}

BSK_HD void _epg_real_jvp_kernel(float* t1, float* t2, float* m0, float* b1, float* inversion_efficiency, float* diffusion, float* duration, std::int32_t* kind, float* flip, std::uint8_t* action, std::int32_t* output_index, std::int32_t* shim_index, float* tangent_t1, float* tangent_t2, float* tangent_m0, float* tangent_b1, float* tangent_inversion_efficiency, float* tangent_diffusion, float* tangent_duration, float* tangent_flip, float* output_real, float* output_imag, bsk::index_t atom_count, bsk::index_t train_count, bsk::index_t event_count, bsk::index_t output_count, bsk::index_t state_count, bsk::index_t single_train, bsk::index_t atom_stride, bsk::index_t shimmed, bsk::index_t diffusing, bsk::index_t transmit, bsk::index_t density, bsk::index_t inverting, bsk::index_t block_states, bsk::index_t problems) {
    bsk::V<float, 2> atom_b1{};
    bsk::V<float, 2> atom_damping{};
    bsk::V<float, 2> atom_inversion{};
    bsk::V<float, 2> atom_m0{};
    bsk::V<float, 3> damp_t{};
    bsk::V<float, 3> damp_z{};
    bsk::V<float, 3> ddamp_t{};
    bsk::V<float, 3> ddamp_z{};
    bsk::V<float, 2> dot_b1{};
    bsk::V<float, 2> dot_damping{};
    bsk::V<float, 3> dot_e1{};
    bsk::V<float, 3> dot_e2{};
    bsk::V<float, 2> dot_inversion{};
    bsk::V<float, 3> dot_longitudinal{};
    bsk::V<float, 2> dot_m0{};
    bsk::V<float, 3> dot_minus{};
    bsk::V<float, 3> dot_plus{};
    bsk::V<float, 3> e1{};
    bsk::V<float, 3> e2{};
    bsk::V<float, 3> longitudinal{};
    bsk::V<float, 3> minus{};
    bsk::V<float, 3> plus{};
    bsk::V<float, 2> pulse_b1{};
    bsk::V<float, 2> pulse_dot_b1{};
    bsk::V<float, 3> rotated_dm{};
    bsk::V<float, 3> rotated_dp{};
    bsk::V<float, 3> rotated_dz{};
    auto problem = ((bsk::program_id(0) * problems) + bsk::arange_y());
    auto state = bsk::arange_x();
    auto active_atom = (problem < (train_count * atom_count));
    auto state_mask = bsk::band((state < state_count), active_atom);
    auto atom = bsk::mod(problem, atom_count);
    // A property given as one value for the whole tissue is read at one
    // address by every voxel, which is a stride of zero through it.
    auto scalar_atom = (atom * atom_stride);
    auto train = bsk::floordiv(problem, atom_count);
    auto empty = bsk::full<float, 3>(0);
    plus = empty;
    minus = empty;
    longitudinal = (empty + bsk::where((state == 0), 1.0f, 0.0f));
    dot_plus = empty;
    dot_minus = empty;
    dot_longitudinal = empty;
    auto atom_t1 = bsk::ld((t1 + atom), active_atom, 1.0f);
    auto atom_t2 = bsk::ld((t2 + atom), active_atom, 1.0f);
    atom_m0 = 1.0f;
    if (bsk::truth(density)) {
        atom_m0 = bsk::ld((m0 + scalar_atom), active_atom, 0.0f);
    }
    atom_b1 = 1.0f;
    if (bsk::truth(transmit)) {
        atom_b1 = bsk::ld((b1 + scalar_atom), active_atom, 1.0f);
    }
    atom_inversion = 1.0f;
    if (bsk::truth(inverting)) {
        atom_inversion = bsk::ld((inversion_efficiency + scalar_atom), active_atom, 1.0f);
    }
    auto dot_t1 = bsk::ld((tangent_t1 + atom), active_atom, 0.0f);
    auto dot_t2 = bsk::ld((tangent_t2 + atom), active_atom, 0.0f);
    dot_m0 = 0.0f;
    if (bsk::truth(density)) {
        dot_m0 = bsk::ld((tangent_m0 + scalar_atom), active_atom, 0.0f);
    }
    dot_b1 = 0.0f;
    if (bsk::truth(transmit)) {
        dot_b1 = bsk::ld((tangent_b1 + scalar_atom), active_atom, 0.0f);
    }
    dot_inversion = 0.0f;
    if (bsk::truth(inverting)) {
        dot_inversion = bsk::ld((tangent_inversion_efficiency + scalar_atom), active_atom, 0.0f);
    }
    auto rate1 = bsk::truediv(1000.0f, atom_t1);
    auto rate2 = bsk::truediv(1000.0f, atom_t2);
    atom_damping = 0.0f;
    dot_damping = 0.0f;
    if (bsk::truth(diffusing)) {
        atom_damping = bsk::ld((diffusion + scalar_atom), active_atom, 0.0f);
        dot_damping = bsk::ld((tangent_diffusion + scalar_atom), active_atom, 0.0f);
    }
    auto order = bsk::cast<float>(state);
    auto event_base = (train * event_count);
    // Two events to an iteration. A repetition is several events -- a pulse,
    // a sample, an interval -- so the loop runs longer than the sequence is
    // repetitions, and unrolling lets one back-edge and one set of event
    // bookkeeping serve two of them. Two is where it stops paying: four was
    // measured slower, and the body is already large enough that widening it
    // costs registers.
    for (bsk::index_t event = 0; event < event_count; event += 1) {
        auto dt = _event_value(duration, event_base, event, active_atom, single_train);
        auto dot_dt = _event_value(tangent_duration, event_base, event, active_atom, single_train);
        // An event of no duration relaxes nothing, and carries no tangent along
        // the relaxation either: both factors are one and both their derivatives
        // are zero.
        if (bsk::truth((bsk::truth((bsk::max_all(dt) != 0.0f)) || bsk::truth((bsk::max_all(dot_dt) != 0.0f))))) {
            e1 = bsk::exp(((-rate1) * dt));
            e2 = bsk::exp(((-rate2) * dt));
            dot_e1 = (e1 * (bsk::truediv(((1000.0f * dt) * dot_t1), (atom_t1 * atom_t1)) - (rate1 * dot_dt)));
            dot_e2 = (e2 * (bsk::truediv(((1000.0f * dt) * dot_t2), (atom_t2 * atom_t2)) - (rate2 * dot_dt)));
            damp_z = 1.0f;
            ddamp_z = 0.0f;
            damp_t = 1.0f;
            ddamp_t = 0.0f;
            if (bsk::truth(diffusing)) {
                auto t0_ = _damping_jvp(atom_damping, dot_damping, dt, dot_dt, order);
                damp_z = bsk::get<0>(t0_);
                ddamp_z = bsk::get<1>(t0_);
                damp_t = bsk::get<2>(t0_);
                ddamp_t = bsk::get<3>(t0_);
            }
            // Order zero is undamped, so the recovery keeps the bare factor.
            auto t1_ = bsk::make_tup((1.0f - e1), (-dot_e1));
            auto recovery = bsk::get<0>(t1_);
            auto dot_recovery = bsk::get<1>(t1_);
            dot_e1 = ((dot_e1 * damp_z) + (e1 * ddamp_z));
            e1 = (e1 * damp_z);
            dot_e2 = ((dot_e2 * damp_t) + (e2 * ddamp_t));
            e2 = (e2 * damp_t);
            dot_plus = ((dot_plus * e2) + (plus * dot_e2));
            dot_minus = ((dot_minus * e2) + (minus * dot_e2));
            dot_longitudinal = ((dot_longitudinal * e1) + (longitudinal * dot_e1));
            dot_longitudinal = (dot_longitudinal + bsk::where((state == 0), dot_recovery, 0.0f));
            plus = (plus * e2);
            minus = (minus * e2);
            longitudinal = ((longitudinal * e1) + bsk::where((state == 0), recovery, 0.0f));
        }
        // Every flag below is read from a per-event array with no atom index, so
        // it is uniform across the program and can steer real control flow.
        auto event_action = bsk::cast<std::int32_t>(bsk::ld((action + event)));
        if (bsk::truth((bsk::band(event_action, 1) != 0))) {
            auto t2_ = _shift_real(plus, minus, state, state_mask, state_count);
            plus = bsk::get<0>(t2_);
            minus = bsk::get<1>(t2_);
            auto t3_ = _shift_real(dot_plus, dot_minus, state, state_mask, state_count);
            dot_plus = bsk::get<0>(t3_);
            dot_minus = bsk::get<1>(t3_);
        }
        auto event_kind = bsk::ld((kind + event));
        auto is_rf = (event_kind == 1);
        auto is_inversion = (bsk::band(event_action, 4) != 0);
        if (bsk::truth((bsk::truth(is_rf) && bsk::truth(is_inversion)))) {
            dot_longitudinal = (((-atom_inversion) * dot_longitudinal) - (dot_inversion * longitudinal));
            longitudinal = ((-atom_inversion) * longitudinal);
        } else if (bsk::truth(is_rf)) {
            auto event_flip = _event_value(flip, event_base, event, active_atom, single_train);
            auto dot_flip = _event_value(tangent_flip, event_base, event, active_atom, single_train);
            pulse_b1 = atom_b1;
            pulse_dot_b1 = dot_b1;
            // One shim is the whole sequence's transmit field, loaded once above;
            // several give each pulse the row of the shim it drives.
            if (bsk::truth(shimmed)) {
                auto shim_row = (bsk::cast<std::int64_t>(bsk::ld((shim_index + event))) * atom_count);
                if (bsk::truth(transmit)) {
                    pulse_b1 = bsk::ld(((b1 + shim_row) + atom), active_atom, 1.0f);
                }
                pulse_dot_b1 = bsk::ld(((tangent_b1 + shim_row) + atom), active_atom, 0.0f);
            }
            auto alpha = (event_flip * pulse_b1);
            auto dot_alpha = ((dot_flip * pulse_b1) + (event_flip * pulse_dot_b1));
            auto cosine = bsk::cos(alpha);
            auto sine = bsk::sin(alpha);
            auto cosine_half_sq = (0.5f * (1.0f + cosine));
            auto sine_half_sq = (0.5f * (1.0f - cosine));
            auto half_sine = (0.5f * sine);
            auto dot_cosine = ((-sine) * dot_alpha);
            auto dot_sine = (cosine * dot_alpha);
            auto dot_cosine_half_sq = ((-0.5f * sine) * dot_alpha);
            auto dot_sine_half_sq = ((0.5f * sine) * dot_alpha);
            auto dot_half_sine = ((0.5f * cosine) * dot_alpha);
            rotated_dp = ((cosine_half_sq * dot_plus) + (dot_cosine_half_sq * plus));
            rotated_dp = (rotated_dp + ((sine_half_sq * dot_minus) + (dot_sine_half_sq * minus)));
            rotated_dp = (rotated_dp - ((sine * dot_longitudinal) + (dot_sine * longitudinal)));
            rotated_dm = ((sine_half_sq * dot_plus) + (dot_sine_half_sq * plus));
            rotated_dm = (rotated_dm + ((cosine_half_sq * dot_minus) + (dot_cosine_half_sq * minus)));
            rotated_dm = (rotated_dm + ((sine * dot_longitudinal) + (dot_sine * longitudinal)));
            rotated_dz = ((half_sine * dot_plus) + (dot_half_sine * plus));
            rotated_dz = (rotated_dz - ((half_sine * dot_minus) + (dot_half_sine * minus)));
            rotated_dz = (rotated_dz + ((cosine * dot_longitudinal) + (dot_cosine * longitudinal)));
            auto rotated_p = (((cosine_half_sq * plus) + (sine_half_sq * minus)) - (sine * longitudinal));
            auto rotated_m = (((sine_half_sq * plus) + (cosine_half_sq * minus)) + (sine * longitudinal));
            auto rotated_z = (((half_sine * plus) - (half_sine * minus)) + (cosine * longitudinal));
            plus = rotated_p;
            minus = rotated_m;
            longitudinal = rotated_z;
            dot_plus = rotated_dp;
            dot_minus = rotated_dm;
            dot_longitudinal = rotated_dz;
        }
        if (bsk::truth((bsk::truth((bsk::band(event_action, 32) != 0)) && bsk::truth((event_kind == 2))))) {
            auto out_ = bsk::ld((output_index + event));
            auto output_offset = ((problem * output_count) + out_);
            auto output_mask = bsk::band(bsk::band(active_atom, (state == 0)), (out_ >= 0));
            auto signal_imag = ((dot_m0 * plus) + (atom_m0 * dot_plus));
            bsk::st(((output_real + output_offset) + state), empty, output_mask);
            bsk::st(((output_imag + output_offset) + state), signal_imag, output_mask);
        }
        if (bsk::truth((bsk::truth((bsk::band(event_action, 2) != 0)) || bsk::truth((bsk::band(event_action, 16) != 0))))) {
            auto t4_ = _shift_real(plus, minus, state, state_mask, state_count);
            plus = bsk::get<0>(t4_);
            minus = bsk::get<1>(t4_);
            auto t5_ = _shift_real(dot_plus, dot_minus, state, state_mask, state_count);
            dot_plus = bsk::get<0>(t5_);
            dot_minus = bsk::get<1>(t5_);
        }
        if (bsk::truth((bsk::band(event_action, 8) != 0))) {
            plus = empty;
            minus = empty;
            dot_plus = empty;
            dot_minus = empty;
        }
    }
}
