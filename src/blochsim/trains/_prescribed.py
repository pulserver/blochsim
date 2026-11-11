"""The train kernels of the C++ extension, over NumPy arrays.

The extension takes data pointers, so what it is given has to be contiguous,
of the type it reads, and alive for the call. That is what the private
functions here are for; the public ones are those two with units.
"""

from __future__ import annotations

__all__ = ["effective_te", "prescribed_flips"]

import numpy as np
import numpy.typing as npt

from .._trains_cpu import forward, inverse

#: Let the extension's pool size itself against the work.
AUTO_THREADS = 0


def relaxation(
    ESP: float | None, first_ESP: float | None, T1: float, T2: float
) -> tuple[float, float, float, float]:
    """Relaxation over half an echo interval: ``(e1_first, e2_first, e1, e2)``.

    The first pair is the interval from the excitation to the first echo, the
    second every later one. Times are in ms; an infinite relaxation time is no
    relaxation, and without any the spacing is not needed.
    """
    times = np.asarray([T1, T2], dtype=float)
    if np.any(np.isnan(times)) or np.any(times <= 0):
        raise ValueError("T1 and T2 must be positive")
    if ESP is None:
        if np.all(np.isinf(times)) and first_ESP is None:
            return 1.0, 1.0, 1.0, 1.0
        raise ValueError("relaxation and first_ESP need the echo spacing ESP")
    first = ESP if first_ESP is None else first_ESP
    spacing = np.asarray([first, ESP], dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("ESP and first_ESP must be positive and finite")
    e1_first, e1 = np.exp(-spacing / 2 / T1)
    e2_first, e2 = np.exp(-spacing / 2 / T2)
    return float(e1_first), float(e2_first), float(e1), float(e2)


def _rows(values: npt.ArrayLike, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """``values`` as contiguous float64 ``(trains, echoes)``, and its batch shape."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError(f"{name} needs its echoes on the last axis")
    return np.ascontiguousarray(array.reshape(-1, array.shape[-1])), array.shape[:-1]


def _lengths(
    lengths: npt.ArrayLike | None, shape: tuple[int, ...], echoes: int
) -> np.ndarray:
    """The echoes each train of a batch of ``shape`` plays, one per train."""
    if lengths is None:
        return np.full(int(np.prod(shape)), echoes, dtype=np.int64)
    played = np.ascontiguousarray(np.broadcast_to(lengths, shape), dtype=np.int64)
    played = played.reshape(-1)
    if np.any(played < 1) or np.any(played > echoes):
        raise ValueError("lengths must lie in [1, echoes]")
    return played


def inverse_rows(
    target: np.ndarray,
    lengths: np.ndarray,
    first_rad: np.ndarray | None,
    relax: tuple[float, float, float, float],
    threads: int = AUTO_THREADS,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip angles (rad) and achieved echoes of ``(trains, echoes)`` targets."""
    target = np.ascontiguousarray(target, dtype=np.float64)
    lengths = np.ascontiguousarray(lengths, dtype=np.int64)
    count, echoes = target.shape
    flip = np.empty_like(target)
    achieved = np.empty_like(target)
    if first_rad is not None:
        first_rad = np.ascontiguousarray(first_rad, dtype=np.float64)
    inverse(
        (
            lengths.ctypes.data,
            target.ctypes.data,
            0 if first_rad is None else first_rad.ctypes.data,
            flip.ctypes.data,
            achieved.ctypes.data,
        ),
        count,
        echoes,
        *relax,
        threads,
    )
    return flip, achieved


def forward_rows(
    flip_rad: np.ndarray,
    lengths: np.ndarray,
    relax: tuple[float, float, float, float],
    threads: int = AUTO_THREADS,
) -> np.ndarray:
    """Echoes of ``(trains, echoes)`` flip angles, in radians."""
    flip_rad = np.ascontiguousarray(flip_rad, dtype=np.float64)
    lengths = np.ascontiguousarray(lengths, dtype=np.int64)
    count, echoes = flip_rad.shape
    echo = np.empty_like(flip_rad)
    forward(
        (lengths.ctypes.data, flip_rad.ctypes.data, echo.ctypes.data),
        count,
        echoes,
        *relax,
        threads,
    )
    return echo


def equivalent_te(
    flip_rad: np.ndarray,
    lengths: np.ndarray,
    relax: tuple[float, float, float, float],
    T2: float,
    threads: int = AUTO_THREADS,
) -> np.ndarray:
    """Contrast-equivalent echo times (ms) of ``(trains, echoes)`` flip angles."""
    relaxed = np.abs(forward_rows(flip_rad, lengths, relax, threads))
    free = np.abs(forward_rows(flip_rad, lengths, (1.0, 1.0, 1.0, 1.0), threads))
    with np.errstate(invalid="ignore", divide="ignore"):
        equivalent = -T2 * np.log(relaxed / free)
    return np.where(free > 1e-12, equivalent, np.nan)


def prescribed_flips(
    signal: npt.ArrayLike,
    *,
    first_flip: npt.ArrayLike | None = None,
    ESP: float | None = None,
    first_ESP: float | None = None,
    T1: float = np.inf,
    T2: float = np.inf,
    lengths: npt.ArrayLike | None = None,
    threads: int = AUTO_THREADS,
) -> tuple[np.ndarray, np.ndarray]:
    """Refocusing flip angles whose echoes take prescribed amplitudes.

    The train is the CPMG arrangement: an ideal 90 degree excitation a quarter
    turn from the refocusing pulses, each pulse midway between two echoes. The
    extended phase graph is inverted one echo at a time: the states before a
    pulse fix the echo after it as a function of its flip angle alone, and the
    smallest angle in [0, 180] degrees giving the prescribed amplitude is
    taken. An amplitude no angle reaches gets the angle that comes nearest,
    so the returned echoes are the test of a prescription: they equal it
    exactly where it can be met.

    Parameters
    ----------
    signal : array-like
        Prescribed echo amplitudes, as fractions of the excited magnetization,
        shape ``(..., echoes)``.
    first_flip : float or array-like, default=None
        Flip angle of the first pulse, in degrees, one per train. The first
        echo is then what that pulse gives and its prescription is not read.
    ESP : float, default=None
        Echo spacing, in ms. Needed only with relaxation.
    first_ESP : float, default=None
        Time from the excitation to the first echo, in ms; ``ESP`` by default.
    T1 : float, default=np.inf
        Longitudinal relaxation time the amplitudes are prescribed under, in
        ms. The default is no relaxation.
    T2 : float, default=np.inf
        Transverse relaxation time the amplitudes are prescribed under, in ms.
        The default is no relaxation.
    lengths : int or array-like, default=None
        Echoes each train plays, when the trains of a batch differ in length;
        what follows a train's last echo is returned as zero.
    threads : int, default=0
        Threads to spread the trains over; 0 sizes the pool to the batch.

    Returns
    -------
    flip : numpy.ndarray
        Refocusing flip angles, in degrees, shaped like ``signal``.
    achieved : numpy.ndarray
        The echo amplitudes those angles give, shaped like ``signal``.

    Raises
    ------
    ValueError
        If an amplitude or a first angle is not finite, or the timing, the
        relaxation times or the lengths are not valid.

    Notes
    -----
    The magnetization that recovers during the train is excited across the
    refocusing axis and refocuses at the pulses, not at the echoes, so an
    echo holds none of it: T1 enters only through the decay of what the
    train stores along z.

    Examples
    --------
    A constant train of 180 degree pulses is what unit echoes ask for:

    >>> import numpy as np
    >>> from blochsim.trains import prescribed_flips
    >>> flip, achieved = prescribed_flips(np.ones(8))
    >>> bool(np.allclose(flip, 180.0))
    True

    """
    target, shape = _rows(signal, "signal")
    count, echoes = target.shape
    if not np.all(np.isfinite(target)):
        raise ValueError("signal must be finite")
    relax = relaxation(ESP, first_ESP, T1, T2)
    first = None
    if first_flip is not None:
        first = np.deg2rad(np.broadcast_to(first_flip, shape).reshape(count))
        if not np.all(np.isfinite(first)):
            raise ValueError("first_flip must be finite")
    flip, achieved = inverse_rows(
        target, _lengths(lengths, shape, echoes), first, relax, threads
    )
    return (
        np.rad2deg(flip).reshape(*shape, echoes),
        achieved.reshape(*shape, echoes),
    )


def effective_te(
    flip: npt.ArrayLike,
    ESP: float,
    *,
    first_ESP: float | None = None,
    T1: float = 1000.0,
    T2: float = 100.0,
    threads: int = AUTO_THREADS,
) -> np.ndarray:
    """Contrast-equivalent echo time of every echo of a refocusing train.

    The echo of a train of reduced flip angles has decayed less than its echo
    time says, because part of its history was spent stored along z. Its
    contrast-equivalent echo time is the time a spin echo would need for the
    same T2 decay: ``-T2 * log(S / S0)``, with ``S`` the echo of a reference
    tissue and ``S0`` the same echo without relaxation.

    Parameters
    ----------
    flip : array-like
        Refocusing flip angles, in degrees, shape ``(..., echoes)``, each in
        [0, 180].
    ESP : float
        Echo spacing, in ms.
    first_ESP : float, default=None
        Time from the excitation to the first echo, in ms; ``ESP`` by default.
    T1 : float, default=1000.0
        Longitudinal relaxation time of the reference tissue, in ms.
    T2 : float, default=100.0
        Transverse relaxation time of the reference tissue, in ms.
    threads : int, default=0
        Threads to spread the trains over; 0 sizes the pool to the batch.

    Returns
    -------
    numpy.ndarray
        Contrast-equivalent echo times, in ms, shaped like ``flip``. An echo
        with no signal has none, and is ``nan``.

    Raises
    ------
    ValueError
        If an angle is outside [0, 180] degrees, or the timing or the
        relaxation times are not positive and finite.

    Notes
    -----
    The train is the one :func:`prescribed_flips` designs: an ideal 90 degree
    excitation of fully relaxed magnetization and instantaneous CPMG
    refocusing, with no slice profile, off-resonance or transmit variation.
    The value depends on the reference tissue; at 180 degrees it is the echo
    time itself.

    Examples
    --------
    >>> import numpy as np
    >>> from blochsim.trains import effective_te
    >>> bool(np.allclose(effective_te([180.0] * 3, 4.0), [4.0, 8.0, 12.0]))
    True

    """
    angles, shape = _rows(flip, "flip")
    if not np.all(np.isfinite(angles)) or np.any((angles < 0) | (angles > 180 + 1e-6)):
        raise ValueError("flip must be finite degrees in [0, 180]")
    if not np.all(np.isfinite([T1, T2])):
        raise ValueError("the reference T1 and T2 must be finite")
    relax = relaxation(ESP, first_ESP, T1, T2)
    echoes = angles.shape[-1]
    equivalent = equivalent_te(
        np.deg2rad(angles), _lengths(None, shape, echoes), relax, T2, threads
    )
    return equivalent.reshape(*shape, echoes)
