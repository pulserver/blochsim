"""The GPU kernels for tabulated pools, run on the host and held to the C++
ones.

Each case runs one pass of both backends on the same tables. The C++ kernels
are held to the state machine written out in torch by ``test_many_pools.py``,
so agreeing with them is agreeing with that.
"""

from __future__ import annotations

import pytest

from utils.host_pools import check

PASSES = ("forward", "jvp", "vjp", "vjp_jvp")


@pytest.mark.parametrize("pass_name", PASSES)
@pytest.mark.parametrize(
    ("pools", "rotation"),
    [("2f", "instant"), ("3s", "profile"), ("4s", "dynamic"), ("4s", "shimmed")],
)
def test_the_gpu_kernels_agree_with_the_cpp_ones(pass_name, pools, rotation):
    """Two to four exchanging pools, with and without a semisolid one, turned
    by a hard pulse, a tabulated one, a rotation per voxel and a transmit row
    per shim."""
    check(pass_name, pools, rotation)


@pytest.mark.parametrize("pass_name", ("vjp", "vjp_jvp"))
def test_an_adjoint_in_waves_over_two_trains_agrees_with_the_cpp_one(pass_name):
    """Every problem recorded in a wave of its own, over two trains of
    different lengths: the trajectory and the per-problem table cotangents are
    indexed from the wave's base, and summed over the trains."""
    check(pass_name, "4s", "dynamic", "trains", "chunked")


@pytest.mark.parametrize("pass_name", PASSES)
def test_the_kernels_agree_with_every_optional_term_off(pass_name):
    """A tissue declaring nothing and a sequence moving nothing turns every
    optional term off in both kernels."""
    check(pass_name, "3s", "instant", "undeclared")
