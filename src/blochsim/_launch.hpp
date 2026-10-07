// The Python side of a kernel launch, shared by the CUDA build and the host
// one: the table of kernels, and an argument tuple read into the array a
// kernel takes. Host code only; the kernels themselves are in _kernels.hpp.
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstring>

#include "_kernels.hpp"

namespace blochsim_launch {

inline int find_kernel(const char* name) {
    int index = 0;
    for (const auto& info : bsk::KERNELS) {
        if (std::strcmp(info.name, name) == 0) {
            return index;
        }
        ++index;
    }
    return -1;
}

inline PyObject* kernel_table(PyObject*, PyObject*) {
    PyObject* table = PyDict_New();
    if (table == nullptr) {
        return nullptr;
    }
    for (const auto& info : bsk::KERNELS) {
        PyObject* entry = Py_BuildValue("(ssi)", info.params, info.kinds, info.lanes);
        if (entry == nullptr || PyDict_SetItemString(table, info.name, entry) < 0) {
            Py_XDECREF(entry);
            Py_DECREF(table);
            return nullptr;
        }
        Py_DECREF(entry);
    }
    return table;
}

struct Launch {
    int kernel = -1;
    std::int64_t grid[2] = {1, 1};
    int block[2] = {1, 1};
    // The length of z, which a program holds in each thread.
    int z = 1;
    // The rows each thread holds on a card.
    int lanes = 1;
    bsk::Arguments arguments{};
};

// ``name``, ``grid`` and ``args`` of a launch: the grid a tuple of one or two
// program counts, the arguments one Python number per parameter, pointers as
// the integer address of their first element.
inline bool read_launch(PyObject* name, PyObject* grid, PyObject* args, Launch& launch) {
    const char* text = PyUnicode_AsUTF8AndSize(name, nullptr);
    if (text == nullptr) {
        return false;
    }
    launch.kernel = find_kernel(text);
    if (launch.kernel < 0) {
        PyErr_Format(PyExc_KeyError, "no kernel named %s", text);
        return false;
    }
    const bsk::KernelInfo& info = bsk::KERNELS[launch.kernel];
    if (!PyTuple_Check(grid) || PyTuple_Size(grid) < 1 || PyTuple_Size(grid) > 2) {
        PyErr_SetString(PyExc_ValueError, "the grid is a tuple of one or two counts");
        return false;
    }
    for (Py_ssize_t axis = 0; axis < PyTuple_Size(grid); ++axis) {
        launch.grid[axis] = PyLong_AsLongLong(PyTuple_GetItem(grid, axis));
        if (PyErr_Occurred()) {
            return false;
        }
    }
    const Py_ssize_t count = static_cast<Py_ssize_t>(std::strlen(info.kinds));
    if (!PyTuple_Check(args) || PyTuple_Size(args) != count) {
        PyErr_Format(PyExc_TypeError, "%s takes %zd arguments", info.name, count);
        return false;
    }
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* item = PyTuple_GetItem(args, i);
        bsk::Arg& arg = launch.arguments.a[i];
        switch (info.kinds[i]) {
            case 'p':
                arg.p = PyLong_AsVoidPtr(item);
                break;
            case 'f':
                arg.f = PyFloat_AsDouble(item);
                break;
            default:
                arg.i = PyLong_AsLongLong(item);
                // The EPG kernels index in 32 bits, as Triton did for an
                // argument that fit; one that does not is refused, not cut.
                if (!PyErr_Occurred() && (arg.i > 2147483647LL || arg.i < -2147483648LL)) {
                    PyErr_Format(PyExc_ValueError,
                                 "%s: argument %zd is %lld, past what a kernel indexes with",
                                 info.name, i, static_cast<long long>(arg.i));
                    return false;
                }
                break;
        }
        if (PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError, "argument %zd of %s is not a number", i, info.name);
            return false;
        }
    }
    launch.block[0] = info.x < 0 ? 1 : static_cast<int>(launch.arguments.a[info.x].i);
    launch.block[1] = info.y < 0 ? 1 : static_cast<int>(launch.arguments.a[info.y].i);
    launch.z = info.z < 0 ? 1 : static_cast<int>(launch.arguments.a[info.z].i);
    launch.lanes = info.lanes;
    if (launch.block[0] < 1 || launch.block[1] < 1 || launch.block[0] * launch.block[1] > 1024 * launch.lanes) {
        PyErr_Format(PyExc_ValueError, "%s: a block of %d by %d threads is more than a card runs",
                     info.name, launch.block[0], launch.block[1] / launch.lanes);
        return false;
    }
    if (launch.z < 1 || launch.z > bsk::MAX_Z) {
        PyErr_Format(PyExc_ValueError, "%s: %d pools is more than a program holds", info.name,
                     launch.z);
        return false;
    }
    return true;
}

}  // namespace blochsim_launch
