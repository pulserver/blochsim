"""A Busse train plays the angles it is asked for and gives the signal it prescribes.

Pinned from outside the phase graph: the pseudo-steady-state amplitude
against SciPy's Legendre function, the 106-pulse train of Busse et al. 2008,
Figure 1, against a frozen reconstruction of it, and the echoes of any train
by isochromat summation (``utils.isochromats``), which must rise as the
prescription says they do.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_legendre

from blochsim.trains import busse_flips
from blochsim.trains._busse import _pss
from utils.isochromats import cpmg_echoes


@pytest.fixture(scope="module")
def figure1():
    return json.loads((Path(__file__).parent / "busse_figure1.json").read_text())


def test_pseudo_steady_state_is_the_legendre_form():
    angle = np.linspace(1.0, 180.0, 359)
    sine = np.sin(np.deg2rad(angle) / 2)
    argument = sine * (1 + np.sin(np.deg2rad(angle)) ** 2 / (8 * sine**4))
    np.testing.assert_allclose(
        _pss(angle), np.sqrt(sine) * eval_legendre(-0.5, argument), rtol=1e-13
    )
    assert _pss(np.array(180.0)) == pytest.approx(1.0)


def test_the_default_train_is_figure_1(figure1):
    flip = busse_flips(figure1["length"])
    np.testing.assert_allclose(flip, figure1["angles_deg"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        cpmg_echoes(flip, count=2048), figure1["static_signal"], atol=1e-11, rtol=0
    )


@pytest.mark.parametrize("acquired", [40, 80, 100, 120, 160])
def test_a_train_plays_the_angles_asked_for(acquired):
    flip = busse_flips(acquired + 6)
    centre = 6 + (acquired - 1) // 2
    assert flip.shape == (acquired + 6,)
    np.testing.assert_allclose(
        [flip[1:centre].min(), flip[centre], flip[-1]], [25, 70, 120], atol=1e-8
    )
    assert flip[0] == pytest.approx(120.0, abs=1e-12)
    assert flip.max() <= 120.0 + 1e-8
    assert flip.min() >= 25.0 - 1e-8


def test_the_echoes_rise_linearly_past_the_centre():
    flip = busse_flips(126, centre_echo=50)
    echoes = cpmg_echoes(flip).real
    past = echoes[50:]
    np.testing.assert_allclose(np.diff(past, 2), 0.0, atol=1e-11)
    assert np.all(np.diff(past) > 0)
    # Before it they rise faster than linearly: a power above one.
    assert np.all(np.diff(echoes[20:51], 2) > 0)


def test_trains_of_a_batch_are_designed_as_they_are_alone():
    lengths = np.array([46, 51, 66, 86, 106])
    controls = np.array(
        [[25, 70, 120], [27, 68, 118], [30, 65, 115], [22, 74, 125], [20, 75, 130]],
        dtype=float,
    )
    together = busse_flips(
        lengths,
        min_flip=controls[:, 0],
        centre_flip=controls[:, 1],
        max_flip=controls[:, 2],
        centre_echo=25,
        threads=3,
    )
    assert together.shape == (5, 106)
    for row, (length, wanted) in enumerate(zip(lengths, controls, strict=True)):
        alone = busse_flips(
            int(length),
            min_flip=wanted[0],
            centre_flip=wanted[1],
            max_flip=wanted[2],
            centre_echo=25,
            threads=1,
        )
        np.testing.assert_array_equal(together[row, :length], alone)
        assert not together[row, length:].any()


def test_arguments_broadcast_to_the_batch_shape():
    flip = busse_flips([[60], [66]], centre_flip=[60.0, 70.0, 80.0], centre_echo=30)
    assert flip.shape == (2, 3, 66)
    np.testing.assert_allclose(flip[..., 30], [[60, 70, 80]] * 2, atol=1e-8)


def test_a_train_centred_on_its_first_acquired_echo_is_found():
    # Newton steps leave the reachable echoes here; the hybrid solver does not.
    flip = busse_flips(66, centre_echo=6)
    np.testing.assert_allclose(
        [flip[1:6].min(), flip[6], flip[-1]], [25, 70, 120], atol=1e-6
    )
    echoes = cpmg_echoes(flip).real
    # The opening transient is below the tolerance from the fortieth echo.
    np.testing.assert_allclose(np.diff(echoes[40:], 2), 0.0, atol=1e-11)


def test_a_prescription_no_train_meets_is_refused():
    with pytest.raises(ValueError, match="no Busse train"):
        busse_flips(66, max_flip=100.0, centre_echo=6)
    with pytest.raises(ValueError, match="trains 1"):
        busse_flips(66, max_flip=[120.0, 100.0], centre_echo=6)


@pytest.mark.parametrize(
    "arguments",
    [
        {"length": 0},
        {"length": 106.5},
        {"length": True},
        {"startup_echoes": -1},
        {"startup_echoes": 106},
        {"startup_echoes": 6.5},
        {"centre_echo": 5},
        {"centre_echo": 105},
        {"centre_echo": 55.5},
        {"min_flip": 0.0},
        {"min_flip": np.nan},
        {"centre_flip": 20.0},
        {"max_flip": 180.0},
        {"initial_decay": 0.0},
        {"initial_decay": np.inf},
    ],
)
def test_an_invalid_prescription_is_refused(arguments):
    with pytest.raises(ValueError):
        busse_flips(**({"length": 106} | arguments))
