"""The flips an inversion returns give the echoes it was asked for.

The echoes are checked by isochromat summation (``utils.isochromats``), which
shares nothing with the phase graph the inversion walks, and against the two
closed forms a CPMG train has: a 180 degree pulse returns the whole echo, and
the first echo of any train is the squared sine of half its first angle.
"""

import numpy as np
import pytest

from blochsim.trains import prescribed_flips
from utils.isochromats import cpmg_echoes


def test_unit_echoes_are_a_train_of_180_degree_pulses():
    flip, achieved = prescribed_flips(np.ones(12))
    np.testing.assert_allclose(flip, 180.0, atol=1e-12)
    np.testing.assert_allclose(achieved, 1.0, atol=1e-14)


@pytest.mark.parametrize("amplitude", [0.05, 0.25, 0.5, 0.9])
def test_first_echo_is_the_squared_sine_of_half_the_first_angle(amplitude):
    flip, achieved = prescribed_flips([amplitude])
    assert flip[0] == pytest.approx(2 * np.rad2deg(np.arcsin(np.sqrt(amplitude))))
    assert achieved[0] == pytest.approx(amplitude, abs=1e-15)


@pytest.mark.parametrize("amplitude", [0.3, 0.5, 0.8])
def test_isochromats_give_a_constant_prescription(amplitude):
    target = np.full(40, amplitude)
    flip, achieved = prescribed_flips(target)
    np.testing.assert_allclose(achieved, target, atol=1e-13)
    echoes = cpmg_echoes(flip)
    np.testing.assert_allclose(echoes.real, target, atol=1e-12)
    np.testing.assert_allclose(echoes.imag, 0.0, atol=1e-12)


def test_isochromats_give_a_prescription_under_relaxation():
    timing = {"ESP": 4.0, "first_ESP": 5.0, "T1": 900.0, "T2": 70.0}
    echo_time = 5.0 + 4.0 * np.arange(48)
    target = 0.45 * np.exp(-echo_time / 250.0)
    flip, achieved = prescribed_flips(target, **timing)
    np.testing.assert_allclose(achieved, target, atol=1e-13)
    # What recovers during the train is excited across the axis and
    # refocuses at the pulses: no echo holds any of it.
    echoes = cpmg_echoes(flip, **timing)
    np.testing.assert_allclose(echoes.real, target, atol=1e-12)
    np.testing.assert_allclose(echoes.imag, 0.0, atol=1e-12)


def test_the_first_angle_can_be_given_and_its_echo_follows():
    flip, achieved = prescribed_flips([0.0, 0.5, 0.5, 0.5], first_flip=120.0)
    assert flip[0] == pytest.approx(120.0)
    assert achieved[0] == pytest.approx(np.sin(np.deg2rad(60.0)) ** 2)
    np.testing.assert_allclose(achieved[1:], 0.5, atol=1e-14)
    np.testing.assert_allclose(cpmg_echoes(flip).real, achieved, atol=1e-12)


def test_of_two_angles_that_meet_an_echo_the_smaller_is_taken():
    target = [0.0, 0.7]
    flip, _ = prescribed_flips(target, first_flip=100.0)
    scan = np.linspace(0.0, 180.0, 3601)
    second = np.array([cpmg_echoes([100.0, angle]).real[1] for angle in scan])
    crossings = np.flatnonzero(np.diff(np.sign(second - target[1])))
    assert len(crossings) == 2
    assert scan[crossings[0]] <= flip[1] <= scan[crossings[0] + 1]


@pytest.mark.parametrize(
    ("amplitude", "angle", "echo"), [(1.5, 180.0, 1.0), (-0.5, 0.0, 0.0)]
)
def test_an_echo_out_of_reach_gets_the_nearest_angle(amplitude, angle, echo):
    flip, achieved = prescribed_flips([amplitude])
    assert flip[0] == pytest.approx(angle, abs=1e-12)
    assert achieved[0] == pytest.approx(echo, abs=1e-15)


def test_trains_of_a_batch_are_solved_as_they_are_alone():
    rng = np.random.default_rng(0)
    target = rng.uniform(0.2, 0.7, (3, 5, 24))
    lengths = rng.integers(4, 25, (3, 5))
    flip, achieved = prescribed_flips(target, lengths=lengths, threads=4)
    assert flip.shape == achieved.shape == target.shape
    for where in np.ndindex(3, 5):
        length = lengths[where]
        alone, echoes = prescribed_flips(target[where][:length], threads=1)
        np.testing.assert_array_equal(flip[where][:length], alone)
        np.testing.assert_array_equal(achieved[where][:length], echoes)
        assert not flip[where][length:].any()
        assert not achieved[where][length:].any()


@pytest.mark.parametrize(
    "arguments",
    [
        {"signal": []},
        {"signal": 0.5},
        {"T2": 80.0},
        {"first_ESP": 5.0},
        {"ESP": 0.0},
        {"ESP": 4.0, "T1": -1.0},
        {"ESP": 4.0, "T2": np.nan},
        {"ESP": 4.0, "first_ESP": np.inf},
        {"lengths": 0},
        {"lengths": 9},
    ],
)
def test_an_invalid_prescription_is_refused(arguments):
    with pytest.raises(ValueError):
        prescribed_flips(**({"signal": np.full(8, 0.5)} | arguments))
