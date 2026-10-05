"""MRF tests."""

import numpy as np
from pytest import fixture

from blochsim import (
    mrf_sim,
)


@fixture
def flip():
    return np.linspace(5.0, 60.0, 100, dtype=np.float32)


def test_scalar_forward(flip):
    sig = mrf_sim(flip, TR=10.0, T1=1000.0, T2=100.0)
    assert sig.shape == (100,)


def test_multiple_forward(flip):
    sig = mrf_sim(flip, TR=10.0, T1=(200, 500, 1000.0), T2=100.0)
    assert sig.shape == (3, 100)


def test_scalar_derivative(flip):
    _, dsig = mrf_sim(flip, TR=10.0, T1=1000.0, T2=100.0, diff="T1")
    assert dsig.shape == (100,)


def test_multiple_derivative(flip):
    _, dsig = mrf_sim(flip, TR=10.0, T1=(200, 500, 1000.0), T2=100.0, diff="T1")
    assert dsig.shape == (3, 100)


def test_scalar_gradient(flip):
    _, grad = mrf_sim(flip, TR=10.0, T1=1000.0, T2=100.0, diff=("T1", "T2"))
    assert grad.shape == (2, 100)


def test_multiple_gradient(flip):
    _, grad = mrf_sim(flip, TR=10.0, T1=(200, 500, 1000.0), T2=100.0, diff=("T1", "T2"))
    assert grad.shape == (3, 2, 100)


def test_current_simulator_protocol_arguments_pass_through(flip):
    """The convenience call accepts protocol fields added to MRFSimulator."""
    from blochsim.simulators import MRFSimulator

    phases = np.linspace(0.0, 117.0, flip.size, dtype=np.float32)
    got = mrf_sim(
        flip,
        TR=10.0,
        T1=1000.0,
        T2=100.0,
        phases=phases,
        B0=23.0,
    )
    expected = MRFSimulator(flip=flip, TR=10.0, phases=phases).simulate(
        T1=1000.0, T2=100.0, B0=23.0
    )
    np.testing.assert_allclose(got, expected)
