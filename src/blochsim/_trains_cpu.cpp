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
// for the best part of a millisecond of them.
constexpr std::int64_t TRAINS_PER_WORKER = 64;

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

    // The derivative of a pulse along one direction: this tangent turned by
    // the pulse, plus the derivative of the turn itself applied to ``primal``
    // before it, times the angle's own derivative. Call it before the primal
    // is turned.
    BLOCHSIM_ALWAYS_INLINE void rotate_tangent(
        const States& primal, const double angle, const double slope
    ) {
        subspace::rotate(
            plus.data(), minus.data(), longitudinal.data(), occupied,
            std::cos(angle), std::sin(angle)
        );
        if (slope == 0.0) {
            return;
        }
        const double half_sine = 0.5 * std::sin(angle) * slope;
        const double cosine = std::cos(angle) * slope;
        const double sine = std::sin(angle) * slope;
        for (std::size_t state = 0; state < occupied; ++state) {
            const double p = primal.plus[state];
            const double m = primal.minus[state];
            const double z = primal.longitudinal[state];
            plus[state] += -half_sine * p + half_sine * m - cosine * z;
            minus[state] += half_sine * p - half_sine * m + cosine * z;
            longitudinal[state] += 0.5 * cosine * (p - m) - sine * z;
        }
    }

    BLOCHSIM_ALWAYS_INLINE void rotate(const double angle) {
        subspace::rotate(
            plus.data(), minus.data(), longitudinal.data(), occupied,
            std::cos(angle), std::sin(angle)
        );
    }
};

// How a solved angle moves with what it was solved from.
enum class Reached {
    // An end of [0, pi]: it stays there under a small change.
    END,
    // The angle whose echo comes nearest an unreachable target: it follows
    // the extremum of the echo, delta or delta + pi.
    EXTREMUM,
    // The echo meets the target, and the angle follows the target.
    MET,
};

struct Solved {
    double angle;
    Reached reached;
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
inline Solved refocusing_angle(
    const double plus, const double minus, const double longitudinal,
    const double target, const double e2
) {
    // A prescription that is not a number has no angle, and neither has
    // any echo after it: the states it leaves are not numbers either.
    if (!std::isfinite(target)) {
        return {NAN, Reached::END};
    }
    const double a = 0.5 * (plus + minus);
    const double b = 0.5 * (minus - plus);
    const double c = longitudinal;
    const double delta = std::atan2(c, b);
    const double radius = std::hypot(b, c);
    const double wanted = (target / e2 - a) / std::max(radius, 1e-16);
    const double reach = std::min(1.0, std::max(-1.0, wanted));
    const double turn = std::acos(reach);
    const double met = a + radius * reach;
    const bool within = wanted > -1.0 && wanted < 1.0;
    const double angles[8] = {
        0.0, PI, delta, delta + PI, delta - turn, delta + turn,
        delta - turn + 2.0 * PI, delta + turn + 2.0 * PI,
    };
    const double echoes[8] = {
        a + b, a - b, a + radius, a - radius, met, met, met, met,
    };
    Solved best{NAN, Reached::END};
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
            best.angle = angle;
            best.reached = candidate < 2   ? Reached::END
                : candidate < 4 || !within ? Reached::EXTREMUM
                                           : Reached::MET;
        }
    }
    return best;
}

// The derivative of a solved angle along one direction, from the derivatives
// of the states it was solved from and of its target.
//
// A met target holds a + b cos(alpha) + c sin(alpha) = target / e2, so the
// angle moves by the implicit function theorem; an extremum moves with
// delta = atan2(c, b); an end of the interval does not move.
inline double refocusing_slope(
    const Solved& solved, const double plus, const double minus,
    const double longitudinal, const double d_plus, const double d_minus,
    const double d_longitudinal, const double d_target, const double e2
) {
    if (solved.reached == Reached::END || !std::isfinite(solved.angle)) {
        return 0.0;
    }
    const double b = 0.5 * (minus - plus);
    const double c = longitudinal;
    const double d_a = 0.5 * (d_plus + d_minus);
    const double d_b = 0.5 * (d_minus - d_plus);
    const double d_c = d_longitudinal;
    if (solved.reached == Reached::MET) {
        const double cosine = std::cos(solved.angle);
        const double sine = std::sin(solved.angle);
        const double steepness = c * cosine - b * sine;
        if (steepness != 0.0) {
            return (d_target / e2 - d_a - d_b * cosine - d_c * sine) / steepness;
        }
    }
    const double radius_sq = b * b + c * c;
    return radius_sq > 0.0 ? (b * d_c - c * d_b) / radius_sq : 0.0;
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
                ).angle;
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

// Derivatives ride along a train as one copy of its states per direction,
// laid out ``(trains, directions, echoes)`` like the echoes they belong to.
struct Tangents {
    std::int64_t directions;

    std::size_t at(
        const std::int64_t train, const std::int64_t direction,
        const std::size_t echoes
    ) const {
        return (static_cast<std::size_t>(train * directions + direction)) * echoes;
    }
};

// The derivative states of one train, one copy per direction. What is linear
// in the states -- relaxation and the shift -- moves them as it moves the
// states themselves.
struct Slopes {
    std::vector<States> along;

    Slopes(const std::int64_t directions, const std::size_t capacity)
        : along(static_cast<std::size_t>(directions), States(capacity)) {}

    void clear(const std::size_t states) {
        for (States& slope : along) {
            slope.clear(states);
        }
    }

    void relax(const double e1, const double e2) {
        for (States& slope : along) {
            slope.relax(e1, e2);
        }
    }

    void shift(const std::size_t states) {
        for (States& slope : along) {
            slope.shift(states, subspace::ALONG_AXIS<double>);
        }
    }

    States& operator[](const std::int64_t direction) {
        return along[static_cast<std::size_t>(direction)];
    }
};

// A train's states and their derivatives, back at the excitation.
inline void start(States& states, Slopes& slopes, const std::size_t count) {
    states.clear(count);
    states.plus[0] = 1.0;
    states.minus[0] = 1.0;
    slopes.clear(count);
}

// Relaxation over half an interval, then the shift, for states and slopes.
inline void advance(
    States& states, Slopes& slopes, const std::size_t count, const bool relaxing,
    const double e1, const double e2
) {
    if (relaxing) {
        states.relax(e1, e2);
        slopes.relax(e1, e2);
    }
    states.shift(count, subspace::ALONG_AXIS<double>);
    slopes.shift(count);
}

struct InverseTangents {
    Batch batch;
    Tangents along;
    const double* target;
    const double* first_flip;
    const double* d_target;
    // ``(trains, directions)``, or null when the first angle is solved too.
    const double* d_first_flip;
    double* flip;
    double* achieved;
    double* d_flip;
    double* d_achieved;
};

// What ``inverse_range`` computes, and its derivative along each direction of
// the targets and the first angles.
BLOCHSIM_TRAIN_CLONES
void inverse_tangents_range(
    const InverseTangents& problem, const std::int64_t begin, const std::int64_t end
) {
    const Batch& batch = problem.batch;
    const std::size_t echoes = static_cast<std::size_t>(batch.echoes);
    const std::int64_t directions = problem.along.directions;
    const bool relaxing = batch.relaxing();
    States along(echoes + 1);
    Slopes slopes(directions, echoes + 1);
    for (std::int64_t train = begin; train < end; ++train) {
        const std::size_t row = static_cast<std::size_t>(train) * echoes;
        const std::size_t length = static_cast<std::size_t>(batch.lengths[train]);
        const std::size_t states = length + 1;
        start(along, slopes, states);
        for (std::size_t echo = 0; echo < length; ++echo) {
            const double e1 = echo == 0 ? batch.e1_first : batch.e1;
            const double e2 = echo == 0 ? batch.e2_first : batch.e2;
            advance(along, slopes, states, relaxing, e1, e2);
            const bool given = echo == 0 && problem.first_flip != nullptr;
            const Solved solved = given
                ? Solved{problem.first_flip[train], Reached::END}
                : refocusing_angle(
                    along.plus[1], along.minus[1], along.longitudinal[1],
                    problem.target[row + echo], e2
                );
            for (std::int64_t direction = 0; direction < directions; ++direction) {
                States& slope = slopes[direction];
                const std::size_t at = problem.along.at(train, direction, echoes);
                const double d_angle = given
                    ? (problem.d_first_flip == nullptr
                           ? 0.0
                           : problem.d_first_flip[train * directions + direction])
                    : refocusing_slope(
                        solved, along.plus[1], along.minus[1],
                        along.longitudinal[1], slope.plus[1], slope.minus[1],
                        slope.longitudinal[1], problem.d_target[at + echo], e2
                    );
                slope.rotate_tangent(along, solved.angle, d_angle);
                problem.d_flip[at + echo] = d_angle;
            }
            along.rotate(solved.angle);
            advance(along, slopes, states, relaxing, e1, e2);
            problem.flip[row + echo] = solved.angle;
            problem.achieved[row + echo] = along.plus[0];
            for (std::int64_t direction = 0; direction < directions; ++direction) {
                problem.d_achieved[problem.along.at(train, direction, echoes) + echo] =
                    slopes[direction].plus[0];
            }
        }
        for (std::size_t echo = length; echo < echoes; ++echo) {
            problem.flip[row + echo] = 0.0;
            problem.achieved[row + echo] = 0.0;
            for (std::int64_t direction = 0; direction < directions; ++direction) {
                const std::size_t at = problem.along.at(train, direction, echoes);
                problem.d_flip[at + echo] = 0.0;
                problem.d_achieved[at + echo] = 0.0;
            }
        }
    }
}

struct ForwardTangents {
    Batch batch;
    Tangents along;
    const double* flip;
    const double* d_flip;
    double* echo;
    double* d_echo;
};

// What ``forward_range`` computes, and its derivative along each direction of
// the flip angles.
BLOCHSIM_TRAIN_CLONES
void forward_tangents_range(
    const ForwardTangents& problem, const std::int64_t begin, const std::int64_t end
) {
    const Batch& batch = problem.batch;
    const std::size_t echoes = static_cast<std::size_t>(batch.echoes);
    const std::int64_t directions = problem.along.directions;
    const bool relaxing = batch.relaxing();
    States along(echoes + 1);
    Slopes slopes(directions, echoes + 1);
    for (std::int64_t train = begin; train < end; ++train) {
        const std::size_t row = static_cast<std::size_t>(train) * echoes;
        const std::size_t length = static_cast<std::size_t>(batch.lengths[train]);
        const std::size_t states = length + 1;
        start(along, slopes, states);
        for (std::size_t echo = 0; echo < length; ++echo) {
            const double e1 = echo == 0 ? batch.e1_first : batch.e1;
            const double e2 = echo == 0 ? batch.e2_first : batch.e2;
            advance(along, slopes, states, relaxing, e1, e2);
            const double angle = problem.flip[row + echo];
            for (std::int64_t direction = 0; direction < directions; ++direction) {
                slopes[direction].rotate_tangent(
                    along, angle,
                    problem.d_flip[problem.along.at(train, direction, echoes) + echo]
                );
            }
            along.rotate(angle);
            advance(along, slopes, states, relaxing, e1, e2);
            problem.echo[row + echo] = along.plus[0];
            for (std::int64_t direction = 0; direction < directions; ++direction) {
                problem.d_echo[problem.along.at(train, direction, echoes) + echo] =
                    slopes[direction].plus[0];
            }
        }
        for (std::size_t echo = length; echo < echoes; ++echo) {
            problem.echo[row + echo] = 0.0;
            for (std::int64_t direction = 0; direction < directions; ++direction) {
                problem.d_echo[problem.along.at(train, direction, echoes) + echo] = 0.0;
            }
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

PyObject* inverse_tangents(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long count = 0;
    long long echoes = 0;
    long long directions = 0;
    InverseTangents problem{};
    Batch& batch = problem.batch;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLLddddi", &pointers, &count, &echoes, &directions,
            &batch.e1_first, &batch.e2_first, &batch.e1, &batch.e2, &threads
        )) {
        return nullptr;
    }
    if (directions < 1) {
        PyErr_SetString(PyExc_ValueError, "tangents need a direction");
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 9, raw) || !describe(batch, count, echoes, raw[0])) {
        return nullptr;
    }
    problem.along.directions = directions;
    problem.target = static_cast<const double*>(raw[1]);
    problem.first_flip = static_cast<const double*>(raw[2]);
    problem.d_target = static_cast<const double*>(raw[3]);
    problem.d_first_flip = static_cast<const double*>(raw[4]);
    problem.flip = static_cast<double*>(raw[5]);
    problem.achieved = static_cast<double*>(raw[6]);
    problem.d_flip = static_cast<double*>(raw[7]);
    problem.d_achieved = static_cast<double*>(raw[8]);
    Py_BEGIN_ALLOW_THREADS
    run(problem, threads, inverse_tangents_range);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyObject* forward_tangents(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long count = 0;
    long long echoes = 0;
    long long directions = 0;
    ForwardTangents problem{};
    Batch& batch = problem.batch;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLLddddi", &pointers, &count, &echoes, &directions,
            &batch.e1_first, &batch.e2_first, &batch.e1, &batch.e2, &threads
        )) {
        return nullptr;
    }
    if (directions < 1) {
        PyErr_SetString(PyExc_ValueError, "tangents need a direction");
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 5, raw) || !describe(batch, count, echoes, raw[0])) {
        return nullptr;
    }
    problem.along.directions = directions;
    problem.flip = static_cast<const double*>(raw[1]);
    problem.d_flip = static_cast<const double*>(raw[2]);
    problem.echo = static_cast<double*>(raw[3]);
    problem.d_echo = static_cast<double*>(raw[4]);
    Py_BEGIN_ALLOW_THREADS
    run(problem, threads, forward_tangents_range);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"inverse", inverse, METH_VARARGS,
     "Flip angles whose CPMG echoes follow a prescription."},
    {"forward", forward, METH_VARARGS, "CPMG echoes of given flip angles."},
    {"inverse_tangents", inverse_tangents, METH_VARARGS,
     "Prescribed flip angles and their derivatives along given directions."},
    {"forward_tangents", forward_tangents, METH_VARARGS,
     "CPMG echoes and their derivatives along given directions."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "blochsim._trains_cpu",
    "Torch-free refocusing-train design kernels.", -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__trains_cpu() { return PyModule_Create(&module); }
