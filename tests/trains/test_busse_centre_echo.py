"""The centre echo chosen is the nearest one in contrast-equivalent echo time.

Held to the definition: design the train for every centre, one at a time,
take :func:`effective_te` of each centre echo, and pick the nearest by hand.
"""

import numpy as np
import pytest

from blochsim.trains import busse_centre_echo, busse_flips, effective_te

TIMING = {"ESP": 3.416, "first_ESP": 4.162}


def one_at_a_time(length, startup=6, **prescription):
    """Equivalent echo time of every centre a train can be designed for."""
    equivalent = {}
    for centre in range(max(startup, 6), length - 1):
        try:
            flip = busse_flips(length, centre_echo=centre, **prescription)
        except ValueError:
            continue
        equivalent[centre] = effective_te(flip[: centre + 1], **TIMING)[-1]
    return equivalent


@pytest.mark.parametrize("TE", [20.0, 54.0, 80.0, 400.0])
def test_the_centre_is_the_nearest_of_every_feasible_one(TE):
    equivalent = one_at_a_time(46)
    expected = min(
        equivalent, key=lambda centre: (abs(equivalent[centre] - TE), centre)
    )
    centre, achieved = busse_centre_echo(TE, 46, **TIMING)
    assert centre == expected
    assert achieved == pytest.approx(equivalent[expected], abs=1e-12)


def test_the_equivalent_echo_time_grows_with_the_centre():
    equivalent = one_at_a_time(66, max_flip=100.0)
    assert np.all(np.diff([equivalent[centre] for centre in sorted(equivalent)]) > 0)


def test_a_centre_is_kept_only_when_every_train_has_a_design():
    # Centred on its first acquired echo, the 120 degree train has a design
    # and the 100 degree one has none.
    alone, _ = busse_centre_echo(1.0, 66, **TIMING)
    assert alone == 6
    shared, _ = busse_centre_echo(1.0, 66, max_flip=[120.0, 100.0], **TIMING)
    assert shared == 7
    with pytest.raises(ValueError):
        busse_flips(66, max_flip=100.0, centre_echo=6)


def test_the_echo_time_is_the_first_train_s():
    centre, achieved = busse_centre_echo(
        60.0, [66, 46], centre_flip=[70.0, 60.0], **TIMING
    )
    flip = busse_flips(66, centre_echo=centre)
    assert achieved == pytest.approx(effective_te(flip[: centre + 1], **TIMING)[-1])
    assert centre < 45


def test_startup_echoes_are_not_centres():
    centre, _ = busse_centre_echo(1.0, 66, startup_echoes=12, **TIMING)
    assert centre == 12


@pytest.mark.parametrize(
    "arguments",
    [
        {"TE": 0.0},
        {"TE": np.nan},
        {"ESP": 0.0},
        {"T2": np.inf},
        {"length": 7},
        {"startup_echoes": -1},
        {"max_flip": 180.0},
        {"length": 46, "startup_echoes": 45},
    ],
)
def test_an_invalid_request_is_refused(arguments):
    with pytest.raises(ValueError):
        busse_centre_echo(**({"TE": 60.0, "length": 46, "ESP": 3.4} | arguments))
