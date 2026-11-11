"""Designing a train does not import PyTorch, and the package is whole on first use.

``import blochsim`` brings :mod:`blochsim.trains` alone; everything built on
PyTorch arrives the first time it is asked for, by any of the routes an eager
import served.
"""

import os
import subprocess
import sys

import pytest

import blochsim


def run(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()


def test_a_train_is_designed_without_pytorch():
    out = run(
        "import sys, blochsim\n"
        "from blochsim.trains import busse_centre_echo, busse_flips\n"
        "busse_flips(66); busse_centre_echo(60.0, 66, 3.6)\n"
        "print('torch' in sys.modules)"
    )
    assert out == ["False"]


@pytest.mark.parametrize(
    "use",
    [
        "blochsim.fse_sim",
        "blochsim.Simulator",
        "blochsim.simulators.FSESimulator",
        "blochsim.optim.Bounded",
        "blochsim._execution",
        "getattr(__import__('blochsim.simulators').simulators, 'FSESimulator')",
        "dir(blochsim)",
        "blochsim.__all__",
    ],
)
def test_the_package_loads_on_first_use(use):
    assert run(f"import sys, blochsim\n{use}\nprint('torch' in sys.modules)") == [
        "True"
    ]


def test_a_star_import_names_what_the_package_exports():
    names = run(
        "from blochsim import *\nprint(*sorted(k for k in dir() if k[0] != '_'))"
    )
    exported = {
        *blochsim.sequence.__all__,
        *blochsim._functional.__all__,
        "ModelOperator",
        "Simulator",
        "Subspace",
    }
    assert set(names) == exported
    for name in exported:
        assert getattr(blochsim, name) is not None


@pytest.mark.parametrize(
    ("first_use", "meanwhile"),
    [
        ("blochsim.Simulator", "blochsim.estimators"),
        ("blochsim.fse_sim", "blochsim.simulators"),
        ("blochsim.__all__", "blochsim.optim"),
    ],
)
def test_two_threads_load_the_package_without_waiting_on_each_other(
    first_use, meanwhile
):
    # One thread asks the package for a name while another imports one of
    # its subpackages, which asks it for an optional module in turn.
    code = f"""
import threading
import blochsim
threads = [
    threading.Thread(target=lambda: {first_use}),
    threading.Thread(target=lambda: __import__({meanwhile!r})),
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(blochsim.Simulator.__name__, blochsim.estimators.__name__)
"""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["Simulator", "blochsim.estimators"]


def test_an_optional_module_that_was_not_built_does_not_load_the_package():
    out = run(
        "import sys, blochsim\n"
        "try:\n    from blochsim import _no_such_module\n"
        "except ImportError:\n    pass\n"
        "print('torch' in sys.modules)"
    )
    assert out == ["False"]


def test_what_the_package_lacks_is_an_attribute_error():
    with pytest.raises(AttributeError):
        _ = blochsim.no_such_thing
    with pytest.raises(ImportError):
        from blochsim import no_such_thing  # noqa: F401


@pytest.mark.skipif(not hasattr(os, "fork"), reason="the platform does not fork")
def test_a_forked_child_runs_its_own_workers():
    code = """
import os, signal
import numpy as np
from blochsim.trains import prescribed_flips
target = np.full((256, 32), 0.5)
before, _ = prescribed_flips(target, threads=4)
pid = os.fork()
if pid == 0:
    signal.alarm(20)
    after, _ = prescribed_flips(target, threads=4)
    os._exit(0 if np.array_equal(before, after) else 3)
_, status = os.waitpid(pid, 0)
print(os.WIFEXITED(status) and os.WEXITSTATUS(status))
"""
    assert run(code) == ["0"]
