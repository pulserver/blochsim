#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "_threads.hpp"

// The Cayley-Klein pair of a shaped pulse for a batch of spins, composed over
// its samples in play order: the forward pass of compose_spinor. A spin's
// samples depend on each other and its neighbours' do not, so the spins are
// split into blocks, one block to a work item, and a block walks the samples
// together, reading each sample's row of the fields once.

namespace {

using blochsim::WorkerPool;
using blochsim::worker_count;
using Complex = std::complex<double>;

constexpr std::int64_t SPINS_PER_BLOCK = 64;

struct Pulse {
    const Complex* drive;  // (samples,) shared, or (samples, spins)
    const double* turn;    // (spins,) held, or (samples, spins)
    Complex* a;            // (spins,)
    Complex* b;            // (spins,)
    std::int64_t samples;
    std::int64_t spins;
    bool drive_per_spin;
    bool turn_varies;
};

// One sample's SU(2) rotation about (turn_x, turn_y, turn), right-handed by
// the vector's length. Near zero both coefficients are taken as series in the
// squared length, as the differentiable composition takes them.
inline void step(
    const Complex drive, const double turn, Complex& step_a, Complex& step_b
) {
    const double turn_x = drive.real();
    const double turn_y = drive.imag();
    const double square = turn_x * turn_x + turn_y * turn_y + turn * turn;
    double cosine = 0.0;
    double scale = 0.0;
    if (square > 1e-18) {
        const double angle = std::sqrt(square);
        cosine = std::cos(0.5 * angle);
        scale = std::sin(0.5 * angle) / angle;
    } else {
        cosine = 1.0 - square / 8.0 + square * square / 384.0;
        scale = 0.5 - square / 48.0 + square * square / 3840.0;
    }
    step_a = Complex(cosine, -turn * scale);
    step_b = Complex(-turn_y * scale, -turn_x * scale);
}

void compose_block(const Pulse& pulse, const std::int64_t block) {
    const std::int64_t first = block * SPINS_PER_BLOCK;
    const std::int64_t count = std::min(SPINS_PER_BLOCK, pulse.spins - first);
    std::array<Complex, SPINS_PER_BLOCK> a;
    std::array<Complex, SPINS_PER_BLOCK> b;
    a.fill(Complex(1.0, 0.0));
    b.fill(Complex(0.0, 0.0));
    for (std::int64_t sample = 0; sample < pulse.samples; ++sample) {
        const Complex* drive = pulse.drive_per_spin
            ? pulse.drive + sample * pulse.spins + first
            : pulse.drive + sample;
        const double* turn = pulse.turn_varies
            ? pulse.turn + sample * pulse.spins + first
            : pulse.turn + first;
        for (std::int64_t spin = 0; spin < count; ++spin) {
            Complex step_a;
            Complex step_b;
            step(pulse.drive_per_spin ? drive[spin] : drive[0], turn[spin], step_a, step_b);
            const Complex next_a = step_a * a[spin] - step_b * std::conj(b[spin]);
            b[spin] = step_b * std::conj(a[spin]) + step_a * b[spin];
            a[spin] = next_a;
        }
    }
    std::copy(a.begin(), a.begin() + count, pulse.a + first);
    std::copy(b.begin(), b.begin() + count, pulse.b + first);
}

bool parse_pointer(PyObject* sequence, const Py_ssize_t index, void** pointer) {
    PyObject* value = PySequence_GetItem(sequence, index);
    if (value == nullptr) {
        return false;
    }
    *pointer = PyLong_AsVoidPtr(value);
    Py_DECREF(value);
    return *pointer != nullptr || !PyErr_Occurred();
}

// The buffers a call hands over, as data pointers in a fixed order.
bool unpack(
    PyObject* pointers, const Py_ssize_t expected, std::vector<void*>& raw
) {
    if (PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffers");
        return false;
    }
    raw.resize(static_cast<std::size_t>(expected));
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(
                pointers, index, &raw[static_cast<std::size_t>(index)]
            )) {
            return false;
        }
    }
    return true;
}

PyObject* compose(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long samples = 0;
    long long spins = 0;
    int drive_per_spin = 0;
    int turn_varies = 0;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLppi", &pointers, &samples, &spins, &drive_per_spin,
            &turn_varies, &threads
        )) {
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 4, raw)) {
        return nullptr;
    }
    const Pulse pulse{
        static_cast<const Complex*>(raw[0]),
        static_cast<const double*>(raw[1]),
        static_cast<Complex*>(raw[2]),
        static_cast<Complex*>(raw[3]),
        samples,
        spins,
        drive_per_spin != 0,
        turn_varies != 0,
    };
    Py_BEGIN_ALLOW_THREADS
    const std::int64_t blocks = (pulse.spins + SPINS_PER_BLOCK - 1) / SPINS_PER_BLOCK;
    const unsigned int slots = worker_count(threads, blocks);
    WorkerPool::instance().run(slots, [&pulse, blocks, slots](const unsigned int slot) {
        for (std::int64_t block = slot; block < blocks; block += slots) {
            compose_block(pulse, block);
        }
    });
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"compose", compose, METH_VARARGS,
     "Cayley-Klein pair of a shaped pulse for a batch of spins."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "blochsim._spinor_cpu", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__spinor_cpu() { return PyModule_Create(&module); }
