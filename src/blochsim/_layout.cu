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

epg::Params complex_params(const Reader& r) {
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

const int COMPLEX = bsk::kernel_index("_epg_kernel");
const int COMPLEX_JVP = bsk::kernel_index("_epg_jvp_kernel");
const int REAL = bsk::kernel_index("_epg_real_kernel");
const int REAL_JVP = bsk::kernel_index("_epg_real_jvp_kernel");

}  // namespace

int launch(int kernel, const bsk::Arguments& arguments, cudaStream_t stream) {
    const Reader r{kernel, arguments.a};
    if (kernel != COMPLEX && kernel != COMPLEX_JVP && kernel != REAL && kernel != REAL_JVP) {
        return -1;
    }
    // A layout's rows are at most a warp wide.
    if (r.integer("block_states") > 32) {
        return -1;
    }
    if (kernel == COMPLEX || kernel == COMPLEX_JVP) {
        return complex_launch(kernel == COMPLEX_JVP, r, stream);
    }
    const layout_real::Params p = real_params(r);
    return kernel == REAL ? real_forward(p, stream) : real_jvp(p, stream);
}

}  // namespace blochsim_layout
