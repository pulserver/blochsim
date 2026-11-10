#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "_ifunc.hpp"
#include "_real_subspace.hpp"
#include "_threads.hpp"

// Refocusing trains whose echoes follow a prescription, and the echoes a
// train gives, for a batch of trains at once.
//
// A train is an ideal excitation a quarter turn from its refocusing pulses,
// then one pulse per echo, each midway between two echoes: the CPMG
// arrangement. The excited magnetization lies along the pulses' axis, so its
// states are the real subspace along it (_real_subspace.hpp). What recovers
// during the train is excited by the pulses across the axis, into the other
// subspace. It is not carried: a pathway born at a pulse is back at the
// zeroth order an even number of half intervals later, which is at a pulse
// again and never at an echo.
//
// These are design kernels: a schedule is solved once, on the host, and the
// solve is a shooting problem whose residual has to resolve a change of a
// billionth of a degree. They are in double precision for that reason, and
// there is no counterpart of them on a card.

#if BLOCHSIM_MULTIVERSIONED
#define BLOCHSIM_TRAIN_CLONES \
    __attribute__((target_clones("default", "avx2", "avx512f")))
#else
#define BLOCHSIM_TRAIN_CLONES
#endif

namespace {

using blochsim::WorkerPool;
using blochsim::worker_count;
namespace subspace = blochsim::real_subspace;

constexpr double PI = 3.14159265358979323846;

// A train is tens of microseconds of work, so a worker is worth waking only
// for a few of them.
constexpr std::int64_t TRAINS_PER_WORKER = 8;

// What every train of a batch shares: the batch's shape, and the relaxation
// over half an echo interval -- the first interval's, and every later one's.
struct Batch {
    std::int64_t count;
    std::int64_t echoes;
    const std::int64_t* lengths;
    double e1_first;
    double e2_first;
    double e1;
    double e2;

    bool relaxing() const {
        return e1_first != 1.0 || e2_first != 1.0 || e1 != 1.0 || e2 != 1.0;
    }
};

// The states of one real subspace, and how many of its orders a train has
// reached: an order beyond that is still empty, so nothing is spent on it.
struct States {
    std::vector<double> plus;
    std::vector<double> minus;
    std::vector<double> longitudinal;
    std::size_t occupied = 0;

    explicit States(const std::size_t capacity)
        : plus(capacity), minus(capacity), longitudinal(capacity) {}

    void clear(const std::size_t states) {
        std::fill(plus.begin(), plus.begin() + states, 0.0);
        std::fill(minus.begin(), minus.begin() + states, 0.0);
        std::fill(longitudinal.begin(), longitudinal.begin() + states, 0.0);
        occupied = 1;
    }

    BLOCHSIM_ALWAYS_INLINE void relax(const double e1, const double e2) {
        for (std::size_t state = 0; state < occupied; ++state) {
            plus[state] *= e2;
            minus[state] *= e2;
            longitudinal[state] *= e1;
        }
    }

    BLOCHSIM_ALWAYS_INLINE void shift(const std::size_t states, const double coupling) {
        occupied = std::min(occupied + 1, states);
        subspace::shift(plus.data(), minus.data(), occupied, coupling);
    }

    BLOCHSIM_ALWAYS_INLINE void rotate(const double angle) {
        subspace::rotate(
            plus.data(), minus.data(), longitudinal.data(), occupied,
            std::cos(angle), std::sin(angle)
        );
    }
};

// The flip angle in [0, pi] whose echo is ``target``, from the first-order
// states the pulse is about to turn.
//
// The echo is the zeroth F- the pulse leaves, a + b cos(alpha) + c sin(alpha)
// = a + r cos(alpha - delta), scaled by the transverse relaxation ``e2`` of
// the half interval after the pulse. A target within reach has two solutions,
// delta -/+ turn, and the smaller rotation is taken. One out of reach has
// none, and the angle whose echo comes nearest is taken instead: an end of
// the interval, or the extremum at delta. The candidates are scored by their
// distance from the target plus a term that orders equal scores by angle.
inline double refocusing_angle(
    const double plus, const double minus, const double longitudinal,
    const double target, const double e2
) {
    if (std::isnan(target)) {
        return target;
    }
    const double a = 0.5 * (plus + minus);
    const double b = 0.5 * (minus - plus);
    const double c = longitudinal;
    const double delta = std::atan2(c, b);
    const double radius = std::hypot(b, c);
    const double reach = std::min(
        1.0, std::max(-1.0, (target / e2 - a) / std::max(radius, 1e-16))
    );
    const double turn = std::acos(reach);
    const double met = a + radius * reach;
    const double angles[8] = {
        0.0, PI, delta, delta + PI, delta - turn, delta + turn,
        delta - turn + 2.0 * PI, delta + turn + 2.0 * PI,
    };
    const double echoes[8] = {
        a + b, a - b, a + radius, a - radius, met, met, met, met,
    };
    double best = 0.0;
    double best_score = INFINITY;
    for (int candidate = 0; candidate < 8; ++candidate) {
        const double angle = angles[candidate];
        if (!(angle >= 0.0 && angle <= PI)) {
            continue;
        }
        const double score =
            std::fabs(echoes[candidate] * e2 - target) + 1e-12 * angle;
        if (score < best_score) {
            best_score = score;
            best = angle;
        }
    }
    return best;
}

struct Inverse {
    Batch batch;
    const double* target;
    // The first pulse's flip angle per train, or null for it to be solved
    // like the others.
    const double* first_flip;
    double* flip;
    double* achieved;
};

// Trains ``begin`` to ``end``: the flip angles that give the prescribed
// echoes, and the echoes they do give. Echoes past a train's length are left
// at zero.
BLOCHSIM_TRAIN_CLONES
void inverse_range(
    const Inverse& problem, const std::int64_t begin, const std::int64_t end
) {
    const Batch& batch = problem.batch;
    const std::size_t echoes = static_cast<std::size_t>(batch.echoes);
    const bool relaxing = batch.relaxing();
    States along(echoes + 1);
    for (std::int64_t train = begin; train < end; ++train) {
        const std::size_t row = static_cast<std::size_t>(train) * echoes;
        const std::size_t length = static_cast<std::size_t>(batch.lengths[train]);
        // An order is reached one shift at a time and has as many shifts to
        // come back by, so none past the train's length reaches an echo.
        const std::size_t states = length + 1;
        along.clear(states);
        along.plus[0] = 1.0;
        along.minus[0] = 1.0;
        for (std::size_t echo = 0; echo < length; ++echo) {
            const double e1 = echo == 0 ? batch.e1_first : batch.e1;
            const double e2 = echo == 0 ? batch.e2_first : batch.e2;
            if (relaxing) {
                along.relax(e1, e2);
            }
            along.shift(states, subspace::ALONG_AXIS<double>);
            const double angle = (echo == 0 && problem.first_flip != nullptr)
                ? problem.first_flip[train]
                : refocusing_angle(
                    along.plus[1], along.minus[1], along.longitudinal[1],
                    problem.target[row + echo], e2
                );
            along.rotate(angle);
            if (relaxing) {
                along.relax(e1, e2);
            }
            along.shift(states, subspace::ALONG_AXIS<double>);
            problem.flip[row + echo] = angle;
            problem.achieved[row + echo] = along.plus[0];
        }
        for (std::size_t echo = length; echo < echoes; ++echo) {
            problem.flip[row + echo] = 0.0;
            problem.achieved[row + echo] = 0.0;
        }
    }
}

struct Forward {
    Batch batch;
    const double* flip;
    double* echo;
};

// Trains ``begin`` to ``end``: the echoes of given flip angles.
BLOCHSIM_TRAIN_CLONES
void forward_range(
    const Forward& problem, const std::int64_t begin, const std::int64_t end
) {
    const Batch& batch = problem.batch;
    const std::size_t echoes = static_cast<std::size_t>(batch.echoes);
    const bool relaxing = batch.relaxing();
    States along(echoes + 1);
    for (std::int64_t train = begin; train < end; ++train) {
        const std::size_t row = static_cast<std::size_t>(train) * echoes;
        const std::size_t length = static_cast<std::size_t>(batch.lengths[train]);
        const std::size_t states = length + 1;
        along.clear(states);
        along.plus[0] = 1.0;
        along.minus[0] = 1.0;
        for (std::size_t echo = 0; echo < length; ++echo) {
            const double e1 = echo == 0 ? batch.e1_first : batch.e1;
            const double e2 = echo == 0 ? batch.e2_first : batch.e2;
            if (relaxing) {
                along.relax(e1, e2);
            }
            along.shift(states, subspace::ALONG_AXIS<double>);
            along.rotate(problem.flip[row + echo]);
            if (relaxing) {
                along.relax(e1, e2);
            }
            along.shift(states, subspace::ALONG_AXIS<double>);
            problem.echo[row + echo] = along.plus[0];
        }
        for (std::size_t echo = length; echo < echoes; ++echo) {
            problem.echo[row + echo] = 0.0;
        }
    }
}

// Deal the trains to the pool in contiguous shares.
template <typename Problem, typename Kernel>
void run(const Problem& problem, const int requested_threads, const Kernel kernel) {
    const std::int64_t count = problem.batch.count;
    // ``worker_count`` gives a slot to every MIN_WORK_PER_THREAD items of an
    // unrequested count, so the trains are counted in the units that makes
    // TRAINS_PER_WORKER of them.
    const unsigned int slots = worker_count(
        requested_threads,
        requested_threads > 0
            ? count
            : count * blochsim::MIN_WORK_PER_THREAD / TRAINS_PER_WORKER
    );
    const std::int64_t share = (count + slots - 1) / slots;
    WorkerPool::instance().run(slots, [&](const unsigned int slot) {
        const std::int64_t begin = static_cast<std::int64_t>(slot) * share;
        const std::int64_t end = std::min(count, begin + share);
        if (begin < end) {
            kernel(problem, begin, end);
        }
    });
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

bool unpack(PyObject* pointers, const Py_ssize_t expected, std::vector<void*>& raw) {
    if (PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffers");
        return false;
    }
    raw.resize(static_cast<std::size_t>(expected));
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[static_cast<std::size_t>(index)])) {
            return false;
        }
    }
    return true;
}

// A batch with nothing in it has nothing to compute; one whose shape does not
// describe its buffers must not be walked.
bool describe(
    Batch& batch, const long long count, const long long echoes,
    const void* lengths
) {
    if (count < 0 || echoes < 1 || lengths == nullptr) {
        PyErr_SetString(PyExc_ValueError, "a batch needs its lengths and an echo");
        return false;
    }
    batch.count = count;
    batch.echoes = echoes;
    batch.lengths = static_cast<const std::int64_t*>(lengths);
    for (long long train = 0; train < count; ++train) {
        if (batch.lengths[train] < 1 || batch.lengths[train] > echoes) {
            PyErr_SetString(
                PyExc_ValueError, "a train's length is outside its batch's echoes"
            );
            return false;
        }
    }
    return true;
}

PyObject* inverse(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long count = 0;
    long long echoes = 0;
    Inverse problem{};
    Batch& batch = problem.batch;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLddddi", &pointers, &count, &echoes, &batch.e1_first,
            &batch.e2_first, &batch.e1, &batch.e2, &threads
        )) {
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 5, raw) || !describe(batch, count, echoes, raw[0])) {
        return nullptr;
    }
    problem.target = static_cast<const double*>(raw[1]);
    problem.first_flip = static_cast<const double*>(raw[2]);
    problem.flip = static_cast<double*>(raw[3]);
    problem.achieved = static_cast<double*>(raw[4]);
    Py_BEGIN_ALLOW_THREADS
    run(problem, threads, inverse_range);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyObject* forward(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long count = 0;
    long long echoes = 0;
    Forward problem{};
    Batch& batch = problem.batch;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLddddi", &pointers, &count, &echoes, &batch.e1_first,
            &batch.e2_first, &batch.e1, &batch.e2, &threads
        )) {
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 3, raw) || !describe(batch, count, echoes, raw[0])) {
        return nullptr;
    }
    problem.flip = static_cast<const double*>(raw[1]);
    problem.echo = static_cast<double*>(raw[2]);
    Py_BEGIN_ALLOW_THREADS
    run(problem, threads, forward_range);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"inverse", inverse, METH_VARARGS,
     "Flip angles whose CPMG echoes follow a prescription."},
    {"forward", forward, METH_VARARGS, "CPMG echoes of given flip angles."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "blochsim._trains_cpu",
    "Torch-free refocusing-train design kernels.", -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__trains_cpu() { return PyModule_Create(&module); }
