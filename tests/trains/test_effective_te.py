"""The contrast-equivalent echo time is the T2 decay an echo actually took.

Two things pin it from outside the phase graph: a train of 180 degree pulses
decays with the echo time itself, and for any other train the echo and its
relaxation-free twin can both be summed over isochromats
(``utils.isochromats``).
"""

import numpy as np
import pytest

from blochsim.trains import effective_te
from utils.isochromats import cpmg_echoes


def test_180_degree_pulses_give_the_echo_time():
    equivalent = effective_te(np.full(6, 180.0), 4.0, first_ESP=6.0)
    np.testing.assert_allclose(equivalent, 6.0 + 4.0 * np.arange(6), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_isochromats_give_the_same_decay(seed):
    flip = np.random.default_rng(seed).uniform(30.0, 170.0, 40)
    timing = {"ESP": 3.6, "first_ESP": 4.2, "T1": 1200.0, "T2": 85.0}
    relaxed = np.abs(cpmg_echoes(flip, **timing))
    free = np.abs(cpmg_echoes(flip))
    np.testing.assert_allclose(
        effective_te(flip, **timing), -85.0 * np.log(relaxed / free), atol=1e-9
    )


def test_reduced_angles_shorten_the_equivalent_echo_time():
    equivalent = effective_te(np.full(30, 60.0), 4.0)
    echo_time = 4.0 * np.arange(1, 31)
    assert equivalent[0] == pytest.approx(4.0)
    assert np.all(equivalent[1:] < echo_time[1:])
    assert np.all(np.diff(equivalent) > 0)


def test_a_batch_keeps_its_shape_and_its_trains():
    flip = np.random.default_rng(2).uniform(20.0, 180.0, (2, 3, 16))
    together = effective_te(flip, 5.0, threads=3)
    assert together.shape == flip.shape
    for where in np.ndindex(2, 3):
        np.testing.assert_array_equal(together[where], effective_te(flip[where], 5.0))


def test_an_echo_without_signal_has_no_echo_time():
    assert np.isnan(effective_te([0.0, 90.0], 4.0)[0])


@pytest.mark.parametrize(
    "arguments",
    [
        {"flip": []},
        {"flip": [181.0]},
        {"flip": [-1.0]},
        {"flip": [np.nan]},
        {"ESP": 0.0},
        {"ESP": np.inf},
        {"first_ESP": -1.0},
        {"T1": np.inf},
        {"T2": 0.0},
    ],
)
def test_an_invalid_train_is_refused(arguments):
    with pytest.raises(ValueError):
        effective_te(**({"flip": [120.0, 90.0], "ESP": 4.0} | arguments))
