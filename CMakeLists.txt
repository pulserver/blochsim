cmake_minimum_required(VERSION 3.26)

project(blochsim LANGUAGES CXX)

# The kernels are plain CPython extensions against the stable ABI: they call
# no PyTorch API and link no PyTorch library, which is why one wheel per
# platform serves every supported interpreter and why that wheel is a couple of
# megabytes rather than the size of libtorch.
set(BLOCHSIM_ABI3_VERSION 3.10)

find_package(Python 3.10 REQUIRED
    COMPONENTS Interpreter Development.Module Development.SABIModule
)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(blochsim_add_kernel name source)
    python_add_library(${name} MODULE
        USE_SABI ${BLOCHSIM_ABI3_VERSION}
        WITH_SOABI
        "${source}"
    )
    # The shared headers sit beside the sources they are shared by.
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/blochsim")
    target_link_libraries(${name} PRIVATE Threads::Threads)

    # Nothing the C++ runtime owns crosses the module boundary: the kernel
    # takes raw buffers from the interpreter and hands back objects the
    # interpreter allocated. So the runtime goes inside the extension, and a
    # wheel carries no library beside it and needs no redistributable
    # installed to import.
    if(MSVC)
        set_property(TARGET ${name} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        )
    endif()

    # The mask a host kernel reads decides which terms it evaluates, and the
    # tests hold a gated launch to a whole one bit for bit. Contraction is free
    # to fuse a multiply and an add on one side of that and not the other, which
    # moves the last place of a result that has to be identical. MSVC does not
    # contract across statements under its default /fp:precise.
    if(MSVC)
        target_compile_options(${name} PRIVATE /O2)
    else()
        target_compile_options(${name} PRIVATE -O3 -ffp-contract=off)
    endif()

    install(TARGETS ${name} DESTINATION blochsim)
endfunction()

blochsim_add_kernel(_epg_cpu  src/blochsim/_epg_cpu.cpp)
blochsim_add_kernel(_perk_cpu src/blochsim/_perk_cpu.cpp)

# The GPU kernels compiled for the host, one program at a time. Nothing in the
# package dispatches to them; they are how the suite checks the GPU kernels on
# a machine with no card. Linux only, like the card's build.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_blochsim_host_default ON)
else()
    set(_blochsim_host_default OFF)
endif()
option(BLOCHSIM_HOST_KERNELS "Compile the GPU kernels for the host, for the tests"
    ${_blochsim_host_default})
if(BLOCHSIM_HOST_KERNELS)
    blochsim_add_kernel(_gpu_host src/blochsim/_gpu_host.cpp)
endif()

# The GPU kernels compiled ahead of time for the card, wherever a CUDA
# compiler is found unless BLOCHSIM_CUDA says otherwise.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_blochsim_cuda_default ON)
else()
    set(_blochsim_cuda_default OFF)
endif()
option(BLOCHSIM_CUDA "Compile the GPU kernels for the card" ${_blochsim_cuda_default})

if(BLOCHSIM_CUDA)
    # Real code for each generation and PTX for the newest, which the driver
    # compiles for a card newer than any listed.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # One file per kernel, so they compile in parallel.
    set(_blochsim_kernels
        _three_pool_table_jvp_kernel _three_pool_table_kernel
        _epg_vjp_kernel _epg_vjp_jvp_kernel _epg_real_vjp_jvp_kernel
        _epg_real_vjp_kernel _epg_real_kernel _epg_real_jvp_kernel
        _epg_kernel _epg_jvp_kernel
        _pooled_kernel _pooled_adjoint_kernel
        _regress_kernel _regress_vjp_kernel
    )
    set(_blochsim_gpu_sources src/blochsim/_gpu.cu)
    foreach(NAME IN LISTS _blochsim_kernels)
        set(_source "${CMAKE_CURRENT_BINARY_DIR}/kernels/${NAME}.cu")
        configure_file(src/blochsim/_gpu_kernel.cu.in "${_source}" @ONLY)
        list(APPEND _blochsim_gpu_sources "${_source}")
    endforeach()

    # The EPG kernels written for their layouts (_layout.hpp): a unit per
    # kernel and pool layout, so they compile side by side.
    list(APPEND _blochsim_gpu_sources src/blochsim/_layout.cu src/blochsim/_layout_real.cu
         src/blochsim/_layout_real_vjp.cu)
    foreach(KIND IN ITEMS forward jvp)
        if(KIND STREQUAL "forward")
            set(TYPE "float")
        else()
            set(TYPE "num::Dual")
        endif()
        foreach(POOLS RANGE 3)
            set(_source "${CMAKE_CURRENT_BINARY_DIR}/layout/complex_${KIND}_${POOLS}.cu")
            configure_file(src/blochsim/_layout_complex.cu.in "${_source}" @ONLY)
            list(APPEND _blochsim_gpu_sources "${_source}")
        endforeach()
    endforeach()
    foreach(KIND IN ITEMS vjp vjp_jvp)
        if(KIND STREQUAL "vjp")
            set(TYPE "float")
        else()
            set(TYPE "num::Dual")
        endif()
        foreach(POOLS RANGE 3)
            set(_source "${CMAKE_CURRENT_BINARY_DIR}/layout/complex_${KIND}_${POOLS}.cu")
            configure_file(src/blochsim/_layout_complex_vjp.cu.in "${_source}" @ONLY)
            list(APPEND _blochsim_gpu_sources "${_source}")
        endforeach()
    endforeach()

    python_add_library(_gpu MODULE
        USE_SABI ${BLOCHSIM_ABI3_VERSION}
        WITH_SOABI
        ${_blochsim_gpu_sources}
    )
    target_include_directories(_gpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/blochsim")
    # The runtime is linked in, so a machine needs the driver and nothing else.
    target_link_libraries(_gpu PRIVATE CUDA::cudart_static)
    # Warning 221 is a double constant that rounds to zero in float, which the
    # kernels write deliberately as the floor of a clamp.
    target_compile_options(_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=221>)
    install(TARGETS _gpu DESTINATION blochsim)
endif()
