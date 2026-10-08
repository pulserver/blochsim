"""Load the compiled kernels out of an installed wheel, and report their size.

Run by cibuildwheel against every wheel it builds, on the interpreter that
wheel claims to support, and against each blochsim-cudaNN wheel installed
beside a CUDA build of torch. It loads each extension by file path rather than
by importing :mod:`blochsim`, so the check needs no PyTorch and says something
about the binary alone: that the stable-ABI module initialises on this
interpreter, that it carries no vendored library, and -- for the card's module
-- that the CUDA runtime it links is the one in torch's nvidia wheels, found
from the module's own directory.

    python scripts/check_wheel.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

#: A kernel calls no PyTorch API and links no PyTorch library, so anything
#: approaching this size means something was bundled that should not have been.
LARGEST_REASONABLE_BYTES = 16 * 1024 * 1024
#: The card's module carries compressed machine code for each architecture it
#: was compiled for, and links the CUDA runtime rather than carrying it.
LARGEST_REASONABLE_GPU_BYTES = 96 * 1024 * 1024
#: The CUDA major versions a blochsim-cudaNN package is built for.
CUDA_MAJORS = (12, 13)


def package_directory(name: str) -> pathlib.Path | None:
    """Where an installed package lives, without executing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def extensions(root: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    """The compiled extensions in ``root`` whose names match ``pattern``."""
    return [
        path
        for path in sorted(root.glob(pattern))
        if path.suffix in {".so", ".pyd", ".dylib"}
    ]


def load(path: pathlib.Path, package: str, largest: int) -> None:
    """Initialise the extension at ``path`` and hold it to a size."""
    name = path.name.split(".")[0]
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"no loader for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    size = path.stat().st_size
    print(f"{package}/{path.name}: {size} bytes, {len(dir(module))} attributes")
    if size > largest:
        raise SystemExit(f"{path.name} is {size} bytes; something was bundled")


def runtime_loaded(major: int) -> pathlib.Path:
    """The CUDA runtime this process mapped, which must be an nvidia wheel's."""
    soname = f"libcudart.so.{major}"
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
        if line.endswith(soname) or f"/{soname}" in line:
            path = pathlib.Path(line.split()[-1])
            if "nvidia" not in path.parts:
                raise SystemExit(f"{soname} came from {path}, not torch's nvidia wheel")
            return path
    raise SystemExit(f"{soname} is not mapped")


def main() -> int:
    """Load every compiled kernel installed."""
    root = package_directory("blochsim")
    if root is None:
        raise SystemExit("blochsim is not installed")
    kernels = extensions(root, "_*_cpu.*")
    if len(kernels) != 2:
        raise SystemExit(f"expected two kernels in {root}, found {kernels}")
    for path in kernels:
        load(path, "blochsim", LARGEST_REASONABLE_BYTES)
    # A build from source keeps the card's module beside the package.
    for path in extensions(root, "_gpu.*"):
        load(path, "blochsim", LARGEST_REASONABLE_GPU_BYTES)

    # Loading the card's module needs no card: nothing reaches the driver
    # until a launch. It does need the runtime, which its rpath finds.
    for major in CUDA_MAJORS:
        build = package_directory(f"blochsim_cuda{major}")
        if build is None:
            continue
        modules = extensions(build, "_gpu.*")
        if len(modules) != 1:
            raise SystemExit(f"expected the card's module in {build}, found {modules}")
        load(modules[0], f"blochsim_cuda{major}", LARGEST_REASONABLE_GPU_BYTES)
        print(f"blochsim_cuda{major}: the runtime is {runtime_loaded(major)}")

    print(f"ok on {sys.implementation.name} {'.'.join(map(str, sys.version_info[:3]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
