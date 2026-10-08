// Which kernels have a layout, and their arguments read by name.
#define BLOCHSIM_TABLE_ONLY 1

#include "_layout.hpp"
#include "_special.hpp"

namespace blochsim_layout {
namespace {

// An argument by name, or nothing where this kernel has no such parameter.
struct Reader {
    int kernel;
    const bsk::Arg* a;

    int at(const char* name) const { return bsk::param_index(kernel, name); }
    const float* floats(const char* name) const {
        const int index = at(name);
        return index < 0 ? nullptr : static_cast<const float*>(a[index].p);
    }
    const int* ints(const char* name) const {
        const int index = at(name);
        return index < 0 ? nullptr : static_cast<const int*>(a[index].p);
    }
    float* outputs(const char* name) const { return static_cast<float*>(a[at(name)].p); }
    long long integer(const char* name) const { return a[at(name)].i; }
    float real(const char* name) const { return static_cast<float>(a[at(name)].f); }
    bool flag(const char* name) const { return a[at(name)].i != 0; }
};

// What the forward kernel and the adjoint read alike.
epg::Params complex_params_common(const Reader& r) {
    epg::Params p{};
    p.t1 = r.floats("t1");
    p.t2 = r.floats("t2");
    p.m0 = r.floats("m0");
    p.b1 = r.floats("b1");
    p.b1_phase = r.floats("b1_phase");
    p.b0 = r.floats("b0");
    p.inversion_efficiency = r.floats("inversion_efficiency");
    p.diffusion = r.floats("diffusion");
    p.velocity = r.floats("velocity");
    p.bound_fraction = r.floats("bound_fraction");
    p.bound_exchange = r.at("bound_exchange") >= 0 ? r.floats("bound_exchange") : r.floats("exchange_rate");
    p.t1_bound = r.floats("t1_bound");
    p.pool_b_fraction = r.floats("pool_b_fraction");
    p.pool_b_exchange = r.floats("pool_b_exchange");
    p.t1_pool_b = r.floats("t1_pool_b");
    p.t2_pool_b = r.floats("t2_pool_b");
    p.pool_b_shift = r.floats("pool_b_shift");
    p.duration = r.floats("duration");
    p.flip = r.floats("flip");
    p.phase = r.floats("phase");
    p.phase_cos = r.floats("phase_cos");
    p.phase_sin = r.floats("phase_sin");
    p.profile = r.floats("profile");
    p.saturation = r.floats("saturation");
    p.rf_frequency = r.floats("rf_frequency");
    p.lineshape = r.floats("lineshape");
    p.pairs = r.floats("pairs");
    p.pool_table = r.floats("pool_table");
    p.kind = r.ints("kind");
    p.output_index = r.ints("output_index");
    p.shim_index = r.ints("shim_index");
    p.profile_index = r.ints("profile_index");
    p.pair_index = r.ints("pair_index");
    p.duration_row = r.ints("duration_row");
    p.action = static_cast<const unsigned char*>(r.a[r.at("action")].p);
    p.atom_count = static_cast<int>(r.integer("atom_count"));
    p.train_count = static_cast<int>(r.integer("train_count"));
    p.event_count = static_cast<int>(r.integer("event_count"));
    p.output_count = static_cast<int>(r.integer("output_count"));
    p.state_count = static_cast<int>(r.integer("state_count"));
    p.width = static_cast<int>(r.integer("block_states"));
    p.locations = static_cast<int>(r.integer("locations"));
    p.profile_bins = static_cast<int>(r.integer("profile_bins"));
    p.lineshape_bins = static_cast<int>(r.integer("lineshape_bins"));
    p.flow_scale = r.real("flow_scale");
    p.washout_scale = r.real("washout_scale");
    p.profile_step = r.real("profile_step");
    p.lineshape_step = r.real("lineshape_step");
    p.single_train = r.flag("single_train");
    p.atom_stride = r.flag("atom_stride");
    p.shimmed = r.flag("shimmed");
    p.off_axis = r.flag("off_axis");
    p.moving = r.flag("moving");
    p.diffusing = r.flag("diffusing");
    p.transmit = r.flag("transmit");
    p.density = r.flag("density");
    p.inverting = r.flag("inverting");
    return p;
}

epg::Params complex_params(const Reader& r) {
    epg::Params p = complex_params_common(r);
    p.d_t1 = r.floats("tangent_t1");
    p.d_t2 = r.floats("tangent_t2");
    p.d_m0 = r.floats("tangent_m0");
    p.d_b1 = r.floats("tangent_b1");
    p.d_b1_phase = r.floats("tangent_b1_phase");
    p.d_b0 = r.floats("tangent_b0");
    p.d_inversion_efficiency = r.floats("tangent_inversion_efficiency");
    p.d_diffusion = r.floats("tangent_diffusion");
    p.d_velocity = r.floats("tangent_velocity");
    p.d_bound_fraction = r.floats("tangent_bound_fraction");
    p.d_bound_exchange = r.floats("tangent_exchange_rate");
    p.d_t1_bound = r.floats("tangent_t1_bound");
    p.d_pool_b_fraction = r.floats("tangent_pool_b_fraction");
    p.d_pool_b_exchange = r.floats("tangent_pool_b_exchange");
    p.d_t1_pool_b = r.floats("tangent_t1_pool_b");
    p.d_t2_pool_b = r.floats("tangent_t2_pool_b");
    p.d_pool_b_shift = r.floats("tangent_pool_b_shift");
    p.d_duration = r.floats("tangent_duration");
    p.d_flip = r.floats("tangent_flip");
    p.d_phase = r.floats("tangent_phase");
    p.pair_direction = r.floats("pair_direction");
    p.output_real = r.outputs("output_real");
    p.output_imag = r.outputs("output_imag");
    return p;
}

layout_real::Params real_params(const Reader& r) {
    layout_real::Params p{};
    p.t1 = r.floats("t1");
    p.t2 = r.floats("t2");
    p.m0 = r.floats("m0");
    p.b1 = r.floats("b1");
    p.inversion_efficiency = r.floats("inversion_efficiency");
    p.diffusion = r.floats("diffusion");
    p.duration = r.floats("duration");
    p.flip = r.floats("flip");
    p.d_t1 = r.floats("tangent_t1");
    p.d_t2 = r.floats("tangent_t2");
    p.d_m0 = r.floats("tangent_m0");
    p.d_b1 = r.floats("tangent_b1");
    p.d_inversion_efficiency = r.floats("tangent_inversion_efficiency");
    p.d_diffusion = r.floats("tangent_diffusion");
    p.d_duration = r.floats("tangent_duration");
    p.d_flip = r.floats("tangent_flip");
    p.kind = r.ints("kind");
    p.output_index = r.ints("output_index");
    p.shim_index = r.ints("shim_index");
    p.action = static_cast<const unsigned char*>(r.a[r.at("action")].p);
    p.output_real = r.outputs("output_real");
    p.output_imag = r.outputs("output_imag");
    p.atom_count = static_cast<int>(r.integer("atom_count"));
    p.train_count = static_cast<int>(r.integer("train_count"));
    p.event_count = static_cast<int>(r.integer("event_count"));
    p.output_count = static_cast<int>(r.integer("output_count"));
    p.state_count = static_cast<int>(r.integer("state_count"));
    p.width = static_cast<int>(r.integer("block_states"));
    p.single_train = r.flag("single_train");
    p.atom_stride = r.flag("atom_stride");
    p.shimmed = r.flag("shimmed");
    p.diffusing = r.flag("diffusing");
    p.transmit = r.flag("transmit");
    p.density = r.flag("density");
    p.inverting = r.flag("inverting");
    return p;
}

int complex_launch(bool jvp, const Reader& r, cudaStream_t stream) {
    const epg::Params p = complex_params(r);
    const int rf = r.flag("dynamic") ? epg::DYNAMIC : (r.flag("profiled") ? epg::PROFILE : epg::HARD);
    const int mode = r.flag("tabulated") ? epg::TABLE : (r.flag("narrow") ? epg::NARROW : epg::ROOTS);
    switch (static_cast<int>(r.integer("pools")) + (jvp ? 4 : 0)) {
        case 0: return complex_forward_0(p, rf, mode, stream);
        case 1: return complex_forward_1(p, rf, mode, stream);
        case 2: return complex_forward_2(p, rf, mode, stream);
        case 3: return complex_forward_3(p, rf, mode, stream);
        case 4: return complex_jvp_0(p, rf, mode, stream);
        case 5: return complex_jvp_1(p, rf, mode, stream);
        case 6: return complex_jvp_2(p, rf, mode, stream);
        case 7: return complex_jvp_3(p, rf, mode, stream);
        default: return -1;
    }
}

// The complex adjoint's arguments, for one sweep or its derivative.
epg_vjp::Params adjoint_params(bool dual, const Reader& r) {
    epg_vjp::Params v{};
    epg::Params& p = v.f;
    p = complex_params_common(r);
    p.d_t1 = r.floats("dot_t1");
    p.d_t2 = r.floats("dot_t2");
    p.d_m0 = r.floats("dot_m0");
    p.d_b1 = r.floats("dot_b1");
    p.d_b1_phase = r.floats("dot_b1_phase");
    p.d_b0 = r.floats("dot_b0");
    p.d_inversion_efficiency = r.floats("dot_inversion_efficiency");
    p.d_diffusion = r.floats("dot_diffusion");
    p.d_velocity = r.floats("dot_velocity");
    p.d_bound_fraction = r.floats("dot_bound_fraction");
    p.d_bound_exchange = r.floats("dot_exchange_rate");
    p.d_t1_bound = r.floats("dot_t1_bound");
    p.d_pool_b_fraction = r.floats("dot_pool_b_fraction");
    p.d_pool_b_exchange = r.floats("dot_pool_b_exchange");
    p.d_t1_pool_b = r.floats("dot_t1_pool_b");
    p.d_t2_pool_b = r.floats("dot_t2_pool_b");
    p.d_pool_b_shift = r.floats("dot_pool_b_shift");
    p.d_duration = r.floats("dot_duration");
    p.d_flip = r.floats("dot_flip");
    p.d_phase = r.floats("dot_phase");
    // A pair with no direction of its own along this sweep.
    p.pair_direction = r.at("directed") >= 0 && r.flag("directed") ? r.floats("pair_direction") : nullptr;
    v.grad_output_real = r.floats("grad_output_real");
    v.grad_output_imag = r.floats("grad_output_imag");
    auto out = [&](const char* name) { return r.at(name) < 0 ? nullptr : r.outputs(name); };
    if (dual) {
        v.grad_tissue = out("grad_tissue_value");
        v.grad_tissue_t = out("grad_tissue_tangent");
        v.grad_flip = out("grad_flip_value");
        v.grad_flip_t = out("grad_flip_tangent");
        v.grad_phase = out("grad_phase_value");
        v.grad_phase_t = out("grad_phase_tangent");
        v.grad_duration = out("grad_duration_value");
        v.grad_duration_t = out("grad_duration_tangent");
        v.grad_pair = out("grad_pair_value");
        v.grad_pair_t = out("grad_pair_tangent");
        v.trajectory_r = out("trajectory_vr");
        v.trajectory_i = out("trajectory_vi");
        v.trajectory_tr = out("trajectory_tr");
        v.trajectory_ti = out("trajectory_ti");
    } else {
        v.grad_tissue = out("grad_tissue");
        v.grad_flip = out("grad_flip");
        v.grad_phase = out("grad_phase");
        v.grad_duration = out("grad_duration");
        v.grad_pair = out("grad_pair");
        v.trajectory_r = out("trajectory_r");
        v.trajectory_i = out("trajectory_i");
    }
    v.problem_base = static_cast<int>(r.integer("problem_base"));
    v.problem_end = static_cast<int>(r.integer("problem_end"));
    v.shim_rows = static_cast<int>(r.integer("shim_rows"));
    v.mode = r.flag("tabulated") ? epg::TABLE : (r.flag("narrow") ? epg::NARROW : epg::ROOTS);
    return v;
}

int adjoint_launch(bool dual, const Reader& r, cudaStream_t stream) {
    // One launch walks both ways; the recording launch has nothing to do.
    if (r.flag("recording")) return cudaSuccess;
    const epg_vjp::Params v = adjoint_params(dual, r);
    const int rf = r.flag("dynamic") ? epg::DYNAMIC : (r.flag("profiled") ? epg::PROFILE : epg::HARD);
    switch (static_cast<int>(r.integer("pools")) + (dual ? 4 : 0)) {
        case 0: return complex_vjp_0(v, rf, stream);
        case 1: return complex_vjp_1(v, rf, stream);
        case 2: return complex_vjp_2(v, rf, stream);
        case 3: return complex_vjp_3(v, rf, stream);
        case 4: return complex_vjp_jvp_0(v, rf, stream);
        case 5: return complex_vjp_jvp_1(v, rf, stream);
        case 6: return complex_vjp_jvp_2(v, rf, stream);
        case 7: return complex_vjp_jvp_3(v, rf, stream);
        default: return -1;
    }
}

layout_real_vjp::Params real_adjoint_params(bool dual, const Reader& r) {
    layout_real_vjp::Params p{};
    p.t1 = r.floats("t1");
    p.t2 = r.floats("t2");
    p.m0 = r.floats("m0");
    p.b1 = r.floats("b1");
    p.inversion_efficiency = r.floats("inversion_efficiency");
    p.diffusion = r.floats("diffusion");
    p.duration = r.floats("duration");
    p.flip = r.floats("flip");
    p.d_t1 = r.floats("dot_t1");
    p.d_t2 = r.floats("dot_t2");
    p.d_m0 = r.floats("dot_m0");
    p.d_b1 = r.floats("dot_b1");
    p.d_inversion_efficiency = r.floats("dot_inversion_efficiency");
    p.d_diffusion = r.floats("dot_diffusion");
    p.d_duration = r.floats("dot_duration");
    p.d_flip = r.floats("dot_flip");
    p.kind = r.ints("kind");
    p.output_index = r.ints("output_index");
    p.shim_index = r.ints("shim_index");
    p.action = static_cast<const unsigned char*>(r.a[r.at("action")].p);
    p.grad_output_imag = r.floats("grad_output_imag");
    if (dual) {
        p.grad_tissue = r.outputs("grad_tissue_value");
        p.grad_tissue_t = r.outputs("grad_tissue_tangent");
        p.grad_flip = r.outputs("grad_flip_value");
        p.grad_flip_t = r.outputs("grad_flip_tangent");
        p.grad_duration = r.outputs("grad_duration_value");
        p.grad_duration_t = r.outputs("grad_duration_tangent");
        p.trajectory = r.outputs("trajectory_value");
        p.trajectory_t = r.outputs("trajectory_tangent");
    } else {
        p.grad_tissue = r.outputs("grad_tissue");
        p.grad_flip = r.outputs("grad_flip");
        p.grad_duration = r.outputs("grad_duration");
        p.trajectory = r.outputs("trajectory_value");
    }
    p.problem_base = static_cast<int>(r.integer("problem_base"));
    p.problem_end = static_cast<int>(r.integer("problem_end"));
    p.atom_count = static_cast<int>(r.integer("atom_count"));
    p.train_count = static_cast<int>(r.integer("train_count"));
    p.event_count = static_cast<int>(r.integer("event_count"));
    p.output_count = static_cast<int>(r.integer("output_count"));
    p.state_count = static_cast<int>(r.integer("state_count"));
    p.width = static_cast<int>(r.integer("block_states"));
    p.shim_rows = static_cast<int>(r.integer("shim_rows"));
    p.single_train = r.flag("single_train");
    p.atom_stride = r.flag("atom_stride");
    p.shimmed = r.flag("shimmed");
    p.diffusing = r.flag("diffusing");
    p.transmit = r.flag("transmit");
    p.density = r.flag("density");
    p.inverting = r.flag("inverting");
    return p;
}

// What the many-pool forward kernel and its adjoint read alike.
epg_pooled::Params pooled_params(const Reader& r, long long programs) {
    epg_pooled::Params p{};
    p.m0 = r.floats("m0");
    p.b1 = r.floats("b1");
    p.b1_phase = r.floats("b1_phase");
    p.b0 = r.floats("b0");
    p.efficiency = r.floats("efficiency");
    p.diffusion = r.floats("diffusion");
    p.velocity = r.floats("velocity");
    p.dm0 = r.floats("dm0");
    p.db1 = r.floats("db1");
    p.db1_phase = r.floats("db1_phase");
    p.db0 = r.floats("db0");
    p.defficiency = r.floats("defficiency");
    p.ddiffusion = r.floats("ddiffusion");
    p.dvelocity = r.floats("dvelocity");
    p.duration = r.floats("duration");
    p.flip = r.floats("flip");
    p.phase = r.floats("phase");
    p.saturation = r.floats("saturation");
    p.rf_frequency = r.floats("rf_frequency");
    p.dduration = r.floats("dduration");
    p.dflip = r.floats("dflip");
    p.dphase = r.floats("dphase");
    p.table = r.floats("table");
    p.dtable = r.floats("dtable");
    p.profile = r.floats("profile");
    p.lineshape = r.floats("lineshape");
    p.pairs = r.floats("pairs");
    p.dpairs = r.floats("dpairs");
    p.kind = r.ints("kind");
    p.output_index = r.ints("output_index");
    p.shim_index = r.ints("shim_index");
    p.pool_index = r.ints("pool_index");
    p.profile_index = r.ints("profile_index");
    p.pair_index = r.ints("pair_index");
    p.action = static_cast<const unsigned char*>(r.a[r.at("action")].p);
    p.base = r.integer("base");
    p.problems = static_cast<int>(programs);
    p.atom_count = static_cast<int>(r.integer("atom_count"));
    p.event_count = static_cast<int>(r.integer("event_count"));
    p.output_count = static_cast<int>(r.integer("output_count"));
    p.state_count = static_cast<int>(r.integer("state_count"));
    p.rows = static_cast<int>(r.integer("rows"));
    p.width = static_cast<int>(r.integer("S"));
    p.m = static_cast<int>(r.integer("m"));
    p.blocks = static_cast<int>(r.integer("blocks"));
    p.locations = static_cast<int>(r.integer("locations"));
    p.profile_bins = static_cast<int>(r.integer("profile_bins"));
    p.lineshape_bins = static_cast<int>(r.integer("lineshape_bins"));
    p.flow_scale = r.real("flow_scale");
    p.washout_scale = r.real("washout_scale");
    p.profile_step = r.real("profile_step");
    p.lineshape_step = r.real("lineshape_step");
    p.atom_stride = r.flag("atom_stride");
    p.shimmed = r.flag("shimmed");
    p.directed_pairs = r.flag("directed_pairs");
    p.directed_table = r.flag("directed_table");
    p.off_axis = r.flag("off_axis");
    p.moving = r.flag("moving");
    p.diffusing = r.flag("diffusing");
    p.transmit = r.flag("transmit");
    p.density = r.flag("density");
    p.inverting = r.flag("inverting");
    return p;
}

epg_pooled::Adjoint pooled_adjoint(const Reader& r) {
    epg_pooled::Adjoint g{};
    g.grad_real = r.floats("grad_real");
    g.grad_imag = r.floats("grad_imag");
    g.grad_tissue = r.outputs("grad_tissue");
    g.dgrad_tissue = r.outputs("dgrad_tissue");
    g.grad_duration = r.outputs("grad_duration");
    g.dgrad_duration = r.outputs("dgrad_duration");
    g.grad_flip = r.outputs("grad_flip");
    g.dgrad_flip = r.outputs("dgrad_flip");
    g.grad_phase = r.outputs("grad_phase");
    g.dgrad_phase = r.outputs("dgrad_phase");
    g.grad_table = r.outputs("grad_table");
    g.dgrad_table = r.outputs("dgrad_table");
    g.grad_pairs = r.outputs("grad_pairs");
    g.dgrad_pairs = r.outputs("dgrad_pairs");
    g.trajectory = r.outputs("trajectory");
    g.m0_row = static_cast<int>(r.integer("m0_row"));
    g.b1_row = static_cast<int>(r.integer("b1_row"));
    g.b1_phase_row = static_cast<int>(r.integer("b1_phase_row"));
    g.b0_row = static_cast<int>(r.integer("b0_row"));
    g.efficiency_row = static_cast<int>(r.integer("efficiency_row"));
    g.diffusion_row = static_cast<int>(r.integer("diffusion_row"));
    g.velocity_row = static_cast<int>(r.integer("velocity_row"));
    return g;
}

// Whether the many-pool layouts carry this many pools and orders.
bool pooled_fits(long long n, long long width) { return n >= 2 && n <= 8 && width <= 32; }

int pooled_launch(bool adjoint, const Reader& r, long long programs, cudaStream_t stream) {
    const long long n = r.integer("n");
    if (!pooled_fits(n, r.integer("S"))) {
        return -1;
    }
    // A recording for the tile adjoint is the tile kernel's to make.
    if (!adjoint && r.flag("keep")) {
        return -1;
    }
    epg_pooled::Params p = pooled_params(r, programs);
    if (!adjoint) {
        p.output_real = r.outputs("output_real");
        p.output_imag = r.outputs("output_imag");
    }
    const int rf = r.flag("dynamic") ? epg::DYNAMIC : (r.flag("profiled") ? epg::PROFILE : epg::HARD);
    const bool dual = r.flag("following");
    if (adjoint) {
        const epg_pooled::Adjoint g = pooled_adjoint(r);
        switch (n) {
#define BLOCHSIM_LAYOUT_POOLED_CASE(pools) \
    case pools: return pooled_adjoint_##pools(p, g, rf, dual, stream);
            BLOCHSIM_LAYOUT_POOLED(BLOCHSIM_LAYOUT_POOLED_CASE)
#undef BLOCHSIM_LAYOUT_POOLED_CASE
        }
    } else {
        switch (n) {
#define BLOCHSIM_LAYOUT_POOLED_CASE(pools) \
    case pools: return pooled_forward_##pools(p, rf, dual, stream);
            BLOCHSIM_LAYOUT_POOLED(BLOCHSIM_LAYOUT_POOLED_CASE)
#undef BLOCHSIM_LAYOUT_POOLED_CASE
        }
    }
    return -1;
}

const int POOLED = bsk::kernel_index("_pooled_kernel");
const int POOLED_ADJOINT = bsk::kernel_index("_pooled_adjoint_kernel");
const int COMPLEX = bsk::kernel_index("_epg_kernel");
const int COMPLEX_VJP = bsk::kernel_index("_epg_vjp_kernel");
const int COMPLEX_VJP_JVP = bsk::kernel_index("_epg_vjp_jvp_kernel");
const int REAL_VJP = bsk::kernel_index("_epg_real_vjp_kernel");
const int REAL_VJP_JVP = bsk::kernel_index("_epg_real_vjp_jvp_kernel");
const int COMPLEX_JVP = bsk::kernel_index("_epg_jvp_kernel");
const int REAL = bsk::kernel_index("_epg_real_kernel");
const int REAL_JVP = bsk::kernel_index("_epg_real_jvp_kernel");

}  // namespace

long long pooled_adjoint_floats(long long problems, int event_count, int n, int m, int width, bool dual) {
    if (!pooled_fits(n, width)) {
        return -1;
    }
    const long long programs = (problems + 32 / width - 1) / (32 / width);
    return problems * epg_pooled::adjoint_kept_floats(n, width, event_count, POOLED_SEGMENT, dual) +
           programs * epg_pooled::adjoint_scratch_floats(n, m, 32, POOLED_SEGMENT, dual);
}

int launch(int kernel, const bsk::Arguments& arguments, long long programs, cudaStream_t stream) {
    const Reader r{kernel, arguments.a};
    if (kernel == POOLED || kernel == POOLED_ADJOINT) {
        return pooled_launch(kernel == POOLED_ADJOINT, r, programs, stream);
    }
    const bool adjoint = kernel == COMPLEX_VJP || kernel == COMPLEX_VJP_JVP || kernel == REAL_VJP || kernel == REAL_VJP_JVP;
    if (kernel != COMPLEX && kernel != COMPLEX_JVP && kernel != REAL && kernel != REAL_JVP && !adjoint) {
        return -1;
    }
    // A layout's rows are at most a warp wide.
    if (r.integer("block_states") > 32) {
        return -1;
    }
    if (kernel == COMPLEX || kernel == COMPLEX_JVP) {
        return complex_launch(kernel == COMPLEX_JVP, r, stream);
    }
    if (kernel == COMPLEX_VJP || kernel == COMPLEX_VJP_JVP) {
        return adjoint_launch(kernel == COMPLEX_VJP_JVP, r, stream);
    }
    if (kernel == REAL_VJP || kernel == REAL_VJP_JVP) {
        const layout_real_vjp::Params p = real_adjoint_params(kernel == REAL_VJP_JVP, r);
        return kernel == REAL_VJP ? real_vjp(p, stream) : real_vjp_jvp(p, stream);
    }
    const layout_real::Params p = real_params(r);
    return kernel == REAL ? real_forward(p, stream) : real_jvp(p, stream);
}

}  // namespace blochsim_layout
