"""Busse refocusing trains: a prescribed pseudo-steady-state signal, inverted."""

from __future__ import annotations

__all__ = ["busse_centre_echo", "busse_flips"]

from operator import index

import numpy as np
import numpy.typing as npt

from ._prescribed import AUTO_THREADS, equivalent_te, inverse_rows, relaxation

#: No relaxation: the prescription is of coherence, not of a tissue's signal.
_FREE = (1.0, 1.0, 1.0, 1.0)
#: Echoes the opening transient is given before the ramp to the centre starts.
_RAMP_START = 5
#: The step the shooting Jacobian is differenced over, in degrees.
_STEP_DEG = 1e-3
#: Newton steps a train is given, the halvings a step is, the steps a
#: bracket is closed in, and the residual a train stops at, in degrees.
_NEWTON_STEPS = 40
_HALVINGS = 8
_BRACKET_STEPS = 80
_CONVERGED_DEG = 1e-10
#: What a solved train is held to: its three angles, in degrees, and the
#: echoes it gives against the ones prescribed.
_ANGLE_TOLERANCE_DEG = 1e-6
_SIGNAL_TOLERANCE = 1e-10
#: Degrees of residual a unit of echo out of reach counts for.
_SHORTFALL_DEG = 360.0


def _pss(flip_deg: np.ndarray) -> np.ndarray:
    """Pseudo-steady-state echo amplitude of a constant train of one flip angle.

    ``sqrt(s) * P(s * (1 + sin(alpha)**2 / (8 * s**4)))`` with ``s`` the sine of
    half the angle and ``P`` the Legendre function of degree -1/2, which is
    the reciprocal of an arithmetic-geometric mean.
    """
    angle = np.deg2rad(flip_deg)
    sine = np.sin(angle / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        argument = sine * (1 + np.sin(angle) ** 2 / (8 * sine**4))
        arithmetic = np.ones_like(argument)
        geometric = np.sqrt((1 + argument) / 2)
        for _ in range(10):
            arithmetic, geometric = (
                (arithmetic + geometric) / 2,
                np.sqrt(arithmetic * geometric),
            )
        return np.sqrt(sine) / arithmetic


def _target(
    controls: np.ndarray,
    lengths: np.ndarray,
    centres: np.ndarray,
    max_flip: np.ndarray,
    decay: np.ndarray,
    echoes: int,
) -> np.ndarray:
    """The prescribed echoes of ``(trains, 3)`` pseudo-steady-state controls.

    The controls are flip angles whose pseudo-steady-state amplitudes the
    signal passes through: the first at the end of the opening transient, the
    second at the centre echo, the third at the last echo.
    """
    low, middle, final = _pss(controls.T)[:, :, None]
    echo = np.arange(echoes)[None, :]
    centre = centres[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ramp = np.clip((echo - _RAMP_START) / (centre - _RAMP_START), 0, 1) ** 1.5
        signal = low + (middle - low) * ramp
        signal = np.where(
            echo > centre,
            middle
            + (final - middle) * (echo - centre) / (lengths[:, None] - 1 - centre),
            signal,
        )
        first = np.sin(np.deg2rad(max_flip[:, None] / 2)) ** 2
        return signal + (first - low) * np.exp(-echo / decay[:, None])


class _Trains:
    """A batch of trains to solve: what each is asked for, and how to try it.

    The inversion is causal: the angles up to an echo answer to the echoes
    prescribed up to it and to nothing after. So the first two controls, which
    shape the prescription up to the centre echo, are solved against the
    lowest angle before the centre and the angle at it, on the trains cut at
    their centres; the third, which only sets the rise past the centre, is
    then solved against the last angle alone.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        centres: np.ndarray,
        wanted: np.ndarray,
        decay: np.ndarray,
        threads: int,
    ) -> None:
        self.lengths = lengths
        self.centres = centres
        self.wanted = wanted
        self.decay = decay
        self.threads = threads

    def evaluate(
        self, controls: np.ndarray, rows: np.ndarray, opening: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Try ``controls`` on the trains ``rows``, which may repeat.

        Returns the three angles each trial plays less the ones it is asked
        for, with what its echoes fall short by added in; its flip angles in
        degrees; its echoes; and the echoes prescribed. With ``opening`` the
        trains are cut after their centre echoes, which is all the first two
        controls reach, and the third residual is not formed.

        An echo out of reach gets the nearest angle whatever the controls
        are, so the angles alone stop answering to them. What the echoes
        fall short by does not, and it is added where the level it answers
        to is: before the centre, at it, and past it. A train that meets its
        prescription falls short of nothing, so the roots are the same.
        """
        centres = self.centres[rows]
        lengths = centres + 1 if opening else self.lengths[rows]
        wanted = self.wanted[rows]
        echoes = int(lengths.max())
        target = _target(
            controls,
            self.lengths[rows],
            centres,
            wanted[:, 2],
            self.decay[rows],
            echoes,
        )
        flip, achieved = inverse_rows(
            target, lengths, np.deg2rad(wanted[:, 2]), _FREE, self.threads
        )
        flip = np.rad2deg(flip)
        echo = np.arange(echoes)[None, :]
        centre = centres[:, None]
        each = np.arange(len(rows))
        short = _SHORTFALL_DEG * np.where(
            echo < lengths[:, None], target - achieved, 0.0
        )
        residual = np.column_stack(
            (
                np.where((echo >= 1) & (echo < centre), flip, np.inf).min(axis=1)
                + np.where(echo < centre, short, 0.0).sum(axis=1),
                flip[each, centres] + short[each, centres],
                flip[each, lengths - 1]
                + np.where(echo > centre, short, 0.0).sum(axis=1),
            )
        )
        return residual - wanted, flip, achieved, target

    def accepted(
        self, controls: np.ndarray, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Whether each trial is a train that meets its prescription, and its flips."""
        residual, flip, achieved, target = self.evaluate(controls, rows)
        wanted = self.wanted[rows]
        played = np.arange(flip.shape[1])[None, :] < self.lengths[rows, None]
        with np.errstate(invalid="ignore"):
            valid = (
                np.all(np.isfinite(np.where(played, flip, 0.0)), axis=1)
                & (np.abs(residual).max(axis=1) < _ANGLE_TOLERANCE_DEG)
                & np.all(
                    np.where(played, np.abs(achieved - target), 0.0)
                    < _SIGNAL_TOLERANCE,
                    axis=1,
                )
                & np.all(
                    np.where(
                        played, flip >= wanted[:, :1] - _ANGLE_TOLERANCE_DEG, True
                    ),
                    axis=1,
                )
                & np.all(
                    np.where(
                        played, flip <= wanted[:, 2:] + _ANGLE_TOLERANCE_DEG, True
                    ),
                    axis=1,
                )
                & (controls[:, 0] > 0)
                & np.all(np.diff(controls, axis=1) > 0, axis=1)
                & (controls[:, 2] < 180)
            )
        return valid, np.where(played, flip, 0.0)

    def _opening(self, controls: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """The two residuals the first two controls answer to."""
        return self.evaluate(controls, rows, opening=True)[0][:, :2]

    def newton(self, controls: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """The first two controls of ``rows`` by damped Newton steps, in place.

        The Jacobian is differenced, and a step is halved until it brings
        the residual down: the residual has kinks, where the lowest angle
        moves to another echo or an echo comes within reach. A train stops
        at its own residual and takes no further part, so what it converges
        to does not depend on the batch it was solved in. Returns which
        trains converged.
        """
        active = np.arange(len(rows))
        base = self._opening(controls, rows)
        converged = np.zeros(len(rows), dtype=bool)
        offsets = _STEP_DEG * np.eye(3)[:2]
        for _ in range(_NEWTON_STEPS):
            with np.errstate(invalid="ignore"):
                done = np.abs(base).max(axis=1) < _CONVERGED_DEG
            converged[active[done]] = True
            active, base = active[~done], base[~done]
            if not active.size:
                break
            count = active.size
            probes = np.concatenate([controls[active] + offset for offset in offsets])
            probed = self._opening(probes, np.tile(rows[active], 2))
            jacobian = np.stack(
                [
                    (probed[:count] - base) / _STEP_DEG,
                    (probed[count:] - base) / _STEP_DEG,
                ],
                axis=-1,
            )
            with np.errstate(invalid="ignore"):
                determinant = np.linalg.det(jacobian)
                solvable = (
                    np.isfinite(determinant)
                    & (np.abs(determinant) > 1e-300)
                    & np.all(np.isfinite(base), axis=1)
                )
            active, base = active[solvable], base[solvable]
            correction = np.clip(
                np.linalg.solve(jacobian[solvable], base[..., None])[..., 0], -10, 10
            )
            size = np.linalg.norm(base, axis=1)
            pending = np.arange(active.size)
            taken = np.zeros(active.size, dtype=bool)
            for halving in range(_HALVINGS + 1):
                if not pending.size:
                    break
                trial = controls[active[pending]].copy()
                trial[:, :2] -= correction[pending] / 2**halving
                tried = self._opening(trial, rows[active[pending]])
                with np.errstate(invalid="ignore"):
                    better = np.linalg.norm(tried, axis=1) < size[pending]
                controls[active[pending[better]]] = trial[better]
                base[pending[better]] = tried[better]
                taken[pending[better]] = True
                pending = pending[~better]
            # A train no step improves has stalled, and is left where it is.
            active, base = active[taken], base[taken]
        return converged

    def hybrid(self, row: int) -> np.ndarray:
        """The first two controls of one train from MINPACK's hybrid method.

        It follows a root across kinks a damped Newton step can stall at.
        """
        from scipy.optimize import root

        rows = np.array([row])
        trial = self.wanted[row : row + 1].copy()

        def residual(controls: np.ndarray) -> np.ndarray:
            trial[0, :2] = controls
            return self._opening(trial, rows)[0]

        trial[0, :2] = root(residual, self.wanted[row, :2], tol=1e-9).x
        return trial

    def closing(self, controls: np.ndarray, rows: np.ndarray) -> None:
        """The third control of ``rows``, in place, bracketed and closed in on.

        Between the second control, where the prescription is flat past the
        centre, and 180 degrees, where it rises to the whole echo, the last
        angle crosses the one asked for. The bracket is closed by false
        position with the Illinois correction, which cannot leave it.
        """

        def last(third: np.ndarray, which: np.ndarray) -> np.ndarray:
            trial = controls[which].copy()
            trial[:, 2] = third
            return self.evaluate(trial, rows[which])[0][:, 2]

        every = np.arange(len(rows))
        low = controls[:, 1].copy()
        high = np.full(len(rows), 180.0 - 1e-9)
        at_low, at_high = last(low, every), last(high, every)
        # Which end the last step left in place: -1 the low one, 1 the high.
        kept = np.zeros(len(rows), dtype=np.int8)
        with np.errstate(invalid="ignore"):
            active = np.flatnonzero((at_low < 0) & (at_high > 0))
        controls[:, 2] = np.nan
        for _ in range(_BRACKET_STEPS):
            if not active.size:
                break
            a, b = low[active], high[active]
            fa, fb = at_low[active], at_high[active]
            third = b - fb * (b - a) / (fb - fa)
            value = last(third, active)
            controls[active, 2] = third
            below = value < 0
            # An end left in place twice running has its value halved, or
            # it would hold the bracket open while the other end crept up.
            again = kept[active] == np.where(below, 1, -1)
            at_high[active] = np.where(below, np.where(again, fb / 2, fb), value)
            at_low[active] = np.where(below, value, np.where(again, fa / 2, fa))
            high[active] = np.where(below, b, third)
            low[active] = np.where(below, third, a)
            kept[active] = np.where(below, 1, -1)
            with np.errstate(invalid="ignore"):
                active = active[
                    np.isfinite(value)
                    & (np.abs(value) >= _CONVERGED_DEG)
                    & (high[active] - low[active] > 1e-13)
                ]

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Flip angles in degrees, zero past each train, and which trains exist."""
        rows = np.arange(len(self.lengths))
        controls = self.wanted.copy()
        converged = self.newton(controls, rows)
        for row in np.flatnonzero(~converged):
            one = rows[row : row + 1]
            trial = self.hybrid(int(row))
            if self.newton(trial, one)[0]:
                controls[one] = trial
        self.closing(controls, rows)
        return self.accepted(controls, rows)[::-1]


def _integers(values: npt.ArrayLike, name: str) -> np.ndarray:
    """``values`` as integers, refusing what only rounds to one."""
    array = np.asarray(values)
    if array.dtype == bool or not (
        np.issubdtype(array.dtype, np.integer)
        or (array.dtype == object and all(_is_index(v) for v in array.ravel()))
    ):
        raise ValueError(f"{name} must be an integer")
    return array.astype(np.int64)


def _is_index(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        index(value)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def _prescription(
    length: npt.ArrayLike,
    min_flip: npt.ArrayLike,
    centre_flip: npt.ArrayLike,
    max_flip: npt.ArrayLike,
    initial_decay: npt.ArrayLike,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    """The batch shape, and each train's length, three angles and decay."""
    lengths = _integers(length, "length")
    arrays = np.broadcast_arrays(
        lengths,
        np.asarray(min_flip, dtype=float),
        np.asarray(centre_flip, dtype=float),
        np.asarray(max_flip, dtype=float),
        np.asarray(initial_decay, dtype=float),
    )
    shape = arrays[0].shape
    lengths = arrays[0].reshape(-1)
    wanted = np.stack([array.reshape(-1) for array in arrays[1:4]], axis=1)
    decay = arrays[4].reshape(-1)
    if np.any(lengths < 1):
        raise ValueError("length must be positive")
    with np.errstate(invalid="ignore"):
        ordered = (
            np.all(np.isfinite(wanted), axis=1)
            & (wanted[:, 0] > 0)
            & (wanted[:, 0] < wanted[:, 1])
            & (wanted[:, 1] < wanted[:, 2])
            & (wanted[:, 2] < 180)
        )
    if not np.all(ordered):
        raise ValueError(
            "flip angles must satisfy 0 < min_flip < centre_flip < max_flip < 180"
        )
    if not np.all(np.isfinite(decay)) or np.any(decay <= 0):
        raise ValueError("initial_decay must be positive and finite")
    return shape, lengths, wanted, decay


def busse_flips(
    length: npt.ArrayLike,
    *,
    min_flip: npt.ArrayLike = 25.0,
    centre_flip: npt.ArrayLike = 70.0,
    max_flip: npt.ArrayLike = 120.0,
    centre_echo: npt.ArrayLike | None = None,
    startup_echoes: npt.ArrayLike = 6,
    initial_decay: npt.ArrayLike = 1.3,
    threads: int = AUTO_THREADS,
) -> np.ndarray:
    """Variable refocusing flip angles of a long echo train, after Busse.

    The echo signal is prescribed without relaxation: an opening transient
    from the first echo, a rise through the centre echo, and a linear rise to
    the last one, each level the pseudo-steady-state amplitude of a flip
    angle [1]_ [2]_. :func:`prescribed_flips` inverts it, and the three levels
    are found by shooting so that the angles played are the ones asked for:
    the lowest before the centre is ``min_flip``, the centre echo's is
    ``centre_flip``, and the first and the last are ``max_flip``.

    Every argument may be an array; they broadcast, and each element is a
    train of its own. The trains are solved together.

    Parameters
    ----------
    length : int or array-like
        Refocusing pulses of the train, startup pulses included.
    min_flip : float or array-like, default=25.0
        Lowest flip angle before the centre echo, in degrees.
    centre_flip : float or array-like, default=70.0
        Flip angle at the centre echo, in degrees.
    max_flip : float or array-like, default=120.0
        Flip angle of the first and the last pulse, in degrees.
    centre_echo : int or array-like, default=None
        Zero-based index of the pulse whose echo samples the centre of
        k-space, startup pulses included. By default the middle of what
        follows the startup pulses,
        ``startup_echoes + (length - startup_echoes - 1) // 2``.
    startup_echoes : int or array-like, default=6
        Leading pulses whose echoes are not acquired; they only place the
        default centre echo.
    initial_decay : float or array-like, default=1.3
        Time constant of the opening transient, in echoes.
    threads : int, default=0
        Threads to spread the trains over; 0 sizes the pool to the batch.

    Returns
    -------
    numpy.ndarray
        Flip angles in degrees: shape ``(length,)`` for one train, and
        ``(..., max(length))`` for a batch, zero past each train's last pulse.

    Raises
    ------
    ValueError
        If a count or an angle is not valid, a centre echo is not after the
        sixth pulse and before the last, or no train meets a prescription.

    Notes
    -----
    The rise to the centre follows a power of 1.5 and the one after it is
    linear [2]_. The flip angles do not depend on the echo spacing; what
    relaxation makes of them is :func:`effective_te`'s and the simulators'.

    References
    ----------
    .. [1] Busse RF, Hariharan H, Vu A, Brittain JH. Fast spin echo sequences
       with very long echo trains: design of variable refocusing flip angle
       schedules and generation of clinical T2 contrast. Magn Reson Med.
       2006;55:1030-1037.
    .. [2] Busse RF, Brau ACS, Vu A, et al. Effects of refocusing flip angle
       modulation and view ordering in 3D fast spin echo. Magn Reson Med.
       2008;60:640-649.

    Examples
    --------
    >>> import numpy as np
    >>> from blochsim.trains import busse_flips
    >>> flip = busse_flips(106)
    >>> bool(np.allclose([flip[1:55].min(), flip[55], flip[-1]], [25, 70, 120]))
    True

    """
    # The centre and the startup echoes shape the batch like the rest.
    every = [length, min_flip, centre_flip, max_flip, initial_decay, startup_echoes]
    if centre_echo is not None:
        every.append(centre_echo)
    shape = np.broadcast_shapes(*(np.shape(value) for value in every))
    shape, lengths, wanted, decay = _prescription(
        np.broadcast_to(_integers(length, "length"), shape),
        min_flip,
        centre_flip,
        max_flip,
        initial_decay,
    )
    startup = np.broadcast_to(_integers(startup_echoes, "startup_echoes"), shape)
    startup = startup.reshape(-1)
    if np.any(startup < 0) or np.any(startup >= lengths):
        raise ValueError("startup_echoes must lie in [0, length)")
    if centre_echo is None:
        centres = startup + (lengths - startup - 1) // 2
    else:
        centres = np.broadcast_to(_integers(centre_echo, "centre_echo"), shape)
        centres = centres.reshape(-1)
    if np.any(centres <= _RAMP_START) or np.any(centres >= lengths - 1):
        raise ValueError(
            "centre_echo must be after the sixth pulse and before the last"
        )
    flip, feasible = _Trains(lengths, centres, wanted, decay, threads).solve()
    if not np.all(feasible):
        missing = np.flatnonzero(~feasible)
        raise ValueError(
            "no Busse train meets the prescription of "
            + (
                "the train"
                if len(lengths) == 1
                else f"trains {', '.join(map(str, missing[:8]))}"
            )
        )
    return flip.reshape(*shape, flip.shape[-1])


def busse_centre_echo(
    TE: float,
    length: npt.ArrayLike,
    ESP: float,
    *,
    first_ESP: float | None = None,
    min_flip: npt.ArrayLike = 25.0,
    centre_flip: npt.ArrayLike = 70.0,
    max_flip: npt.ArrayLike = 120.0,
    startup_echoes: int = 6,
    initial_decay: float = 1.3,
    T1: float = 1000.0,
    T2: float = 100.0,
    threads: int = AUTO_THREADS,
) -> tuple[int, float]:
    """The centre echo whose contrast-equivalent echo time is nearest ``TE``.

    A Busse train is designed around its centre echo, so moving the centre
    changes the train and with it the contrast the centre is acquired at.
    Every centre the trains allow is tried at once: the train is designed for
    it, and :func:`effective_te` of its centre echo is compared with ``TE``.

    ``length`` and the three angles broadcast to several trains that share
    the centre echo -- the trains of a shot-dependent schedule. A centre is
    kept only when every one of them can be designed for it, and the echo
    time is that of the first.

    Parameters
    ----------
    TE : float
        Contrast-equivalent echo time asked for, in ms.
    length : int or array-like
        Refocusing pulses of each train, startup pulses included.
    ESP : float
        Echo spacing, in ms.
    first_ESP : float, default=None
        Time from the excitation to the first echo, in ms; ``ESP`` by default.
    min_flip : float or array-like, default=25.0
        Lowest flip angle before the centre echo, in degrees.
    centre_flip : float or array-like, default=70.0
        Flip angle at the centre echo, in degrees.
    max_flip : float or array-like, default=120.0
        Flip angle of the first and the last pulse, in degrees.
    startup_echoes : int, default=6
        Leading pulses whose echoes are not acquired. The centre is an
        acquired echo.
    initial_decay : float, default=1.3
        Time constant of the opening transient, in echoes.
    T1 : float, default=1000.0
        Longitudinal relaxation time of the reference tissue, in ms.
    T2 : float, default=100.0
        Transverse relaxation time of the reference tissue, in ms.
    threads : int, default=0
        Threads to spread the trains over; 0 sizes the pool to the batch.

    Returns
    -------
    centre_echo : int
        Zero-based index of the centre pulse, startup pulses included. Of two
        centres equally near, the earlier.
    TE : float
        Its contrast-equivalent echo time, in ms.

    Raises
    ------
    ValueError
        If ``TE``, the timing or a prescription is not valid, or no centre
        gives every train a design.

    Examples
    --------
    >>> from blochsim.trains import busse_centre_echo
    >>> centre, te = busse_centre_echo(60.0, 66, 3.672, max_flip=100.0)
    >>> centre
    38

    """
    if not np.isfinite(TE) or TE <= 0:
        raise ValueError("TE must be positive and finite")
    if not np.all(np.isfinite([T1, T2])):
        raise ValueError("the reference T1 and T2 must be finite")
    relax = relaxation(ESP, first_ESP, T1, T2)
    _, lengths, wanted, _ = _prescription(
        length, min_flip, centre_flip, max_flip, initial_decay
    )
    startup = int(_integers(startup_echoes, "startup_echoes"))
    if startup < 0:
        raise ValueError("startup_echoes must not be negative")
    # Trains asked for twice are designed once; the first stays first.
    trains = np.column_stack((lengths, wanted))
    _, first = np.unique(trains, axis=0, return_index=True)
    keep = np.sort(first)
    lengths, wanted = lengths[keep], wanted[keep]
    candidates = np.arange(
        max(startup, _RAMP_START + 1), int(lengths.min()) - 1, dtype=np.int64
    )
    if not candidates.size:
        raise ValueError("the trains are too short to have a centre echo")
    trials, shared = len(candidates), len(lengths)
    flip, feasible = _Trains(
        np.tile(lengths, trials),
        np.repeat(candidates, shared),
        np.tile(wanted, (trials, 1)),
        np.full(trials * shared, float(initial_decay)),
        threads,
    ).solve()
    feasible = feasible.reshape(trials, shared).all(axis=1)
    equivalent = equivalent_te(
        np.deg2rad(flip[::shared]), candidates + 1, relax, T2, threads
    )[np.arange(trials), candidates]
    usable = feasible & np.isfinite(equivalent)
    if not np.any(usable):
        raise ValueError(
            "no centre echo gives every train a Busse design at these angles"
        )
    miss = np.where(usable, np.abs(equivalent - TE), np.inf)
    best = int(np.argmin(miss))
    return int(candidates[best]), float(equivalent[best])
