// The GPU kernels run on the host, one program at a time, over host buffers.
//
// The same source the CUDA build compiles, with a tile held as a whole array
// rather than an element per thread. It is how the kernels are checked on a
// machine with no card; nothing in the package dispatches to it otherwise.
#include "_launch.hpp"

#include <new>

namespace {

using host_call = void (*)(const bsk::Arg*);

#define BLOCHSIM_HOST_ENTRY(name) &bsk::call##name,
constexpr host_call CALLS[] = {BLOCHSIM_FOR_EACH_KERNEL(BLOCHSIM_HOST_ENTRY)};
#undef BLOCHSIM_HOST_ENTRY

PyObject* launch(PyObject*, PyObject* args) {
    PyObject* name = nullptr;
    PyObject* grid = nullptr;
    PyObject* values = nullptr;
    if (!PyArg_ParseTuple(args, "UOO", &name, &grid, &values)) {
        return nullptr;
    }
    blochsim_launch::Launch request;
    if (!blochsim_launch::read_launch(name, grid, values, request)) {
        return nullptr;
    }
    bool failed = false;
    Py_BEGIN_ALLOW_THREADS
    try {
        bsk::program.nx = request.block[0];
        bsk::program.ny = request.block[1];
        bsk::program.nz = request.z;
        for (std::int64_t y = 0; y < request.grid[1]; ++y) {
            for (std::int64_t x = 0; x < request.grid[0]; ++x) {
                bsk::program.pid[0] = x;
                bsk::program.pid[1] = y;
                CALLS[request.kernel](request.arguments.a);
            }
        }
    } catch (const std::bad_alloc&) {
        failed = true;
    }
    Py_END_ALLOW_THREADS
    if (failed) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}

PyMethodDef METHODS[] = {
    {"kernels", blochsim_launch::kernel_table, METH_NOARGS,
     "Each kernel's parameter names and kinds."},
    {"launch", launch, METH_VARARGS, "Run a kernel over a grid of programs on the host."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT, "_gpu_host", "The GPU kernels, run on the host.", -1, METHODS,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__gpu_host(void) { return PyModule_Create(&MODULE); }
