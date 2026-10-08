// The GPU kernels, compiled ahead of time for the card.
//
// One CUDA block runs one program: its threads hold a tile an element each,
// x along a row and y across the rows. The module links the CUDA runtime
// statically, so a machine needs the driver and nothing else, and it calls no
// PyTorch API: a launch takes the addresses of the tensors' data and the
// stream PyTorch is queueing on.
//
// The kernels themselves are compiled one to a file, from _gpu_kernel.cu.in.
#define BLOCHSIM_TABLE_ONLY 1

#include "_launch.hpp"
#include "_layout.hpp"

#include <cuda_runtime.h>

#include <string>
#include <utility>
#include <vector>

#define BLOCHSIM_DEVICE_ENTRY(name)                                \
    __global__ void kernel##name##_256(bsk::Arguments arguments, int z); \
    __global__ void kernel##name##_1024(bsk::Arguments arguments, int z);
BLOCHSIM_FOR_EACH_KERNEL(BLOCHSIM_DEVICE_ENTRY)
#undef BLOCHSIM_DEVICE_ENTRY

namespace {

// Each kernel bounded to 256 threads, then to 1024.
#define BLOCHSIM_DEVICE_POINTER(name)                \
    {reinterpret_cast<const void*>(&kernel##name##_256), \
     reinterpret_cast<const void*>(&kernel##name##_1024)},
const void* const KERNEL_FUNCTIONS[][2] = {BLOCHSIM_FOR_EACH_KERNEL(BLOCHSIM_DEVICE_POINTER)};
#undef BLOCHSIM_DEVICE_POINTER

// Whether a launch may run its kernel's layout (_layout.hpp), and how many have.
bool laying_out = true;
unsigned long long layout_launches = 0;

PyObject* cuda_error(cudaError_t status, const char* what) {
    PyErr_Format(PyExc_RuntimeError, "%s: %s", what, cudaGetErrorString(status));
    return nullptr;
}

PyObject* launch(PyObject*, PyObject* args) {
    PyObject* name = nullptr;
    PyObject* grid = nullptr;
    PyObject* values = nullptr;
    int device = 0;
    unsigned long long stream = 0;
    if (!PyArg_ParseTuple(args, "UOOiK", &name, &grid, &values, &device, &stream)) {
        return nullptr;
    }
    blochsim_launch::Launch request;
    if (!blochsim_launch::read_launch(name, grid, values, request)) {
        return nullptr;
    }
    // A thread holds ``lanes`` of a program's rows, so the rows fill whole threads.
    if (request.block[1] % request.lanes != 0) {
        PyErr_Format(PyExc_ValueError, "%s: %d rows do not fill threads of %d",
                     bsk::KERNELS[request.kernel].name, request.block[1], request.lanes);
        return nullptr;
    }
    if (request.grid[0] > 2147483647LL || request.grid[1] > 65535) {
        PyErr_SetString(PyExc_ValueError, "the grid is larger than a launch can hold");
        return nullptr;
    }
    if (request.grid[0] == 0 || request.grid[1] == 0) {
        Py_RETURN_NONE;
    }
    // The caller's current device is put back, whatever this one was.
    int previous = 0;
    cudaError_t status = cudaGetDevice(&previous);
    if (status == cudaSuccess && previous != device) {
        status = cudaSetDevice(device);
    }
    if (status != cudaSuccess) {
        return cuda_error(status, "selecting the device");
    }
    if (laying_out) {
        const int laid = blochsim_layout::launch(request.kernel, request.arguments, request.grid[0],
                                                 reinterpret_cast<cudaStream_t>(stream));
        if (laid >= 0) {
            ++layout_launches;
            if (previous != device) {
                cudaSetDevice(previous);
            }
            if (laid != cudaSuccess) {
                return cuda_error(static_cast<cudaError_t>(laid), bsk::KERNELS[request.kernel].name);
            }
            Py_RETURN_NONE;
        }
    }
    const dim3 blocks(static_cast<unsigned>(request.grid[0]), static_cast<unsigned>(request.grid[1]));
    const dim3 threads(static_cast<unsigned>(request.block[0]), static_cast<unsigned>(request.block[1] / request.lanes));
    // A row's reduction and gather go through one word per thread, and a
    // product with an operator over pools through as many again.
    const std::size_t shared =
        sizeof(unsigned long long) * (threads.x * threads.y + bsk::MAX_Z * bsk::MAX_Z);
    void* parameters[] = {&request.arguments, &request.z};
    const int bounded = threads.x * threads.y > 256 ? 1 : 0;
    const void* function = KERNEL_FUNCTIONS[request.kernel][bounded];
    status = cudaLaunchKernel(function, blocks, threads, parameters, shared,
                              reinterpret_cast<cudaStream_t>(stream));
    if (previous != device) {
        cudaSetDevice(previous);
    }
    if (status != cudaSuccess) {
        return cuda_error(status, bsk::KERNELS[request.kernel].name);
    }
    Py_RETURN_NONE;
}

PyObject* use_layouts(PyObject*, PyObject* args) {
    int on = 1;
    if (!PyArg_ParseTuple(args, "p", &on)) {
        return nullptr;
    }
    const bool previous = laying_out;
    laying_out = on != 0;
    return PyBool_FromLong(previous);
}

PyObject* layout_launch_count(PyObject*, PyObject*) {
    return PyLong_FromUnsignedLongLong(layout_launches);
}

PyObject* pooled_layout_floats(PyObject*, PyObject* args) {
    long long problems = 0;
    int event_count = 0, n = 0, m = 0, width = 0, dual = 0;
    if (!PyArg_ParseTuple(args, "Liiiip", &problems, &event_count, &n, &m, &width, &dual)) {
        return nullptr;
    }
    if (!laying_out) {
        return PyLong_FromLong(-1);
    }
    return PyLong_FromLongLong(blochsim_layout::pooled_adjoint_floats(problems, event_count, n, m, width, dual != 0));
}

PyMethodDef METHODS[] = {
    {"kernels", blochsim_launch::kernel_table, METH_NOARGS,
     "Each kernel's parameter names and kinds."},
    {"launch", launch, METH_VARARGS,
     "Queue a kernel over a grid of programs on a device's stream."},
    {"use_layouts", use_layouts, METH_VARARGS,
     "Whether launches may run their kernel's layout; returns the previous setting."},
    {"layout_launches", layout_launch_count, METH_NOARGS,
     "How many launches have run a kernel's layout."},
    {"pooled_layout_floats", pooled_layout_floats, METH_VARARGS,
     "The floats the many-pool adjoint's layout takes, or -1 where the tile kernels run it."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT, "_gpu", "The GPU kernels, compiled for the card.", -1, METHODS,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__gpu(void) { return PyModule_Create(&MODULE); }
