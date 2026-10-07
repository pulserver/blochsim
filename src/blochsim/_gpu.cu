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
#include "_special.hpp"
#include "_special_table.hpp"

#include <cuda_runtime.h>

#include <string>
#include <utility>
#include <vector>

#define BLOCHSIM_DEVICE_ENTRY(name)                                \
    __global__ void kernel##name##_256(bsk::Arguments arguments, int z); \
    __global__ void kernel##name##_1024(bsk::Arguments arguments, int z);
BLOCHSIM_FOR_EACH_KERNEL(BLOCHSIM_DEVICE_ENTRY)
#undef BLOCHSIM_DEVICE_ENTRY

#define BLOCHSIM_SPECIAL_DECLARATION(index, kernel, fixed) \
    __global__ void special##index(bsk::Arguments arguments, int z);
BLOCHSIM_FOR_EACH_SPECIAL(BLOCHSIM_SPECIAL_DECLARATION)
#undef BLOCHSIM_SPECIAL_DECLARATION

namespace {

// Each kernel bounded to 256 threads, then to 1024.
#define BLOCHSIM_DEVICE_POINTER(name)                \
    {reinterpret_cast<const void*>(&kernel##name##_256), \
     reinterpret_cast<const void*>(&kernel##name##_1024)},
const void* const KERNEL_FUNCTIONS[][2] = {BLOCHSIM_FOR_EACH_KERNEL(BLOCHSIM_DEVICE_POINTER)};
#undef BLOCHSIM_DEVICE_POINTER

// The kernels compiled for one combination of their switches, as CMake listed
// them, and parsed once into the arguments each fixes.
struct SpecialSource {
    const char* kernel;
    const char* fixed;
    const void* function;
};

#define BLOCHSIM_SPECIAL_SOURCE(index, kernel, fixed) \
    {#kernel, fixed, reinterpret_cast<const void*>(&special##index)},
const SpecialSource SPECIAL_SOURCES[] = {
    BLOCHSIM_FOR_EACH_SPECIAL(BLOCHSIM_SPECIAL_SOURCE){nullptr, nullptr, nullptr}};
#undef BLOCHSIM_SPECIAL_SOURCE

struct Special {
    std::vector<std::pair<int, long long>> fixed;
    const void* function;
};

const std::vector<std::vector<Special>>& specials() {
    static const std::vector<std::vector<Special>> table = [] {
        std::vector<std::vector<Special>> out(sizeof(bsk::KERNELS) / sizeof(bsk::KERNELS[0]));
        for (const SpecialSource& source : SPECIAL_SOURCES) {
            if (source.kernel == nullptr) {
                break;
            }
            const int kernel = blochsim_launch::find_kernel(source.kernel);
            Special special{{}, source.function};
            const std::string text = source.fixed;
            std::size_t start = 0;
            while (start < text.size()) {
                const std::size_t end = text.find(',', start);
                const std::string pair = text.substr(start, end - start);
                const std::size_t equals = pair.find('=');
                special.fixed.emplace_back(bsk::param_index(kernel, pair.substr(0, equals).c_str()),
                                           std::stoll(pair.substr(equals + 1)));
                start = end + 1;
            }
            out[kernel].push_back(std::move(special));
        }
        return out;
    }();
    return table;
}

// Whether a launch may run a specialized kernel, and how many have.
bool specializing = true;
unsigned long long specialized_launches = 0;

// The specialized kernel whose fixed switches this launch matches, if any.
const void* matching(const blochsim_launch::Launch& request) {
    for (const Special& special : specials()[request.kernel]) {
        bool match = true;
        for (const auto& [index, value] : special.fixed) {
            if (request.arguments.a[index].i != value) {
                match = false;
                break;
            }
        }
        if (match) {
            return special.function;
        }
    }
    return nullptr;
}

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
    const dim3 blocks(static_cast<unsigned>(request.grid[0]), static_cast<unsigned>(request.grid[1]));
    const dim3 threads(static_cast<unsigned>(request.block[0]), static_cast<unsigned>(request.block[1] / request.lanes));
    // A row's reduction and gather go through one word per thread, and a
    // product with an operator over pools through as many again.
    const std::size_t shared =
        sizeof(unsigned long long) * (threads.x * threads.y + bsk::MAX_Z * bsk::MAX_Z);
    void* parameters[] = {&request.arguments, &request.z};
    const int bounded = threads.x * threads.y > 256 ? 1 : 0;
    // A specialized kernel is compiled for 256 threads; a wider block runs the
    // kernel compiled for every combination.
    const void* function = KERNEL_FUNCTIONS[request.kernel][bounded];
    if (specializing && !bounded) {
        if (const void* special = matching(request)) {
            function = special;
            ++specialized_launches;
        }
    }
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

PyObject* specializations(PyObject*, PyObject*) {
    PyObject* out = PyList_New(0);
    if (out == nullptr) {
        return nullptr;
    }
    for (const SpecialSource& source : SPECIAL_SOURCES) {
        if (source.kernel == nullptr) {
            break;
        }
        PyObject* entry = Py_BuildValue("(ss)", source.kernel, source.fixed);
        if (entry == nullptr || PyList_Append(out, entry) < 0) {
            Py_XDECREF(entry);
            Py_DECREF(out);
            return nullptr;
        }
        Py_DECREF(entry);
    }
    return out;
}

PyObject* use_specializations(PyObject*, PyObject* args) {
    int on = 1;
    if (!PyArg_ParseTuple(args, "p", &on)) {
        return nullptr;
    }
    const bool previous = specializing;
    specializing = on != 0;
    return PyBool_FromLong(previous);
}

PyObject* specialized_launch_count(PyObject*, PyObject*) {
    return PyLong_FromUnsignedLongLong(specialized_launches);
}

PyMethodDef METHODS[] = {
    {"kernels", blochsim_launch::kernel_table, METH_NOARGS,
     "Each kernel's parameter names and kinds."},
    {"launch", launch, METH_VARARGS,
     "Queue a kernel over a grid of programs on a device's stream."},
    {"specializations", specializations, METH_NOARGS,
     "Each specialized kernel, and the switches it was compiled for."},
    {"use_specializations", use_specializations, METH_VARARGS,
     "Whether launches may run specialized kernels; returns the previous setting."},
    {"specialized_launches", specialized_launch_count, METH_NOARGS,
     "How many launches have run a specialized kernel."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef MODULE = {
    PyModuleDef_HEAD_INIT, "_gpu", "The GPU kernels, compiled for the card.", -1, METHODS,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__gpu(void) { return PyModule_Create(&MODULE); }
