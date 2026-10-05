"""Balanced Steady State Free Precession simulator."""

__all__ = ["bssfp_sim"]

from typing import Any

import numpy.typing as npt
import torch

from ..simulators.bssfp import bSSFPSimulator
from ._run import evaluated


def bssfp_sim(
    flip: float | npt.ArrayLike,
    TE: float | npt.ArrayLike,
    TR: float,
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    diff: str | tuple[str, ...] | None = None,
    phase_inc: float = 180.0,
    B0: float | npt.ArrayLike = 0.0,
    chemshift: float | npt.ArrayLike = 0.0,
    M0: float | npt.ArrayLike = 1.0,
    device: str | torch.device | None = None,
    **values: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced steady-state free-precession simulator wrapper.

    Parameters
    ----------
    flip : float | npt.ArrayLike
        Flip angle train in degrees.
    TE : float | npt.ArrayLike
        Echo time in milliseconds.
    TR : float | npt.ArrayLike
        Repetition time in milliseconds.
    T1 : float | npt.ArrayLike
        Longitudinal relaxation time in milliseconds.
    T2 : float | npt.ArrayLike
        Transverse relaxation time in milliseconds.
    diff : str | tuple[str], optional
        Arguments to get the signal derivative with respect to.
        The default is ``None`` (no differentation).
    phase_inc : float, optional
        Linear phase-cycle increment in degrees.
        The default is ``180.0``
    B0 : float | npt.ArrayLike, optional
        Frequency offset map in Hz, default is ``0.0.``
    chemshift : float | npt.ArrayLik, optional
        Chemical shift in Hz, default is ``0.0``.
    M0 : float or array-like, optional
        Proton density scaling factor, default is ``1.0``.
    **values : optional
        Additional tissue, protocol or run settings accepted by the corresponding
        ``Simulator`` class. Explicit wrapper arguments remain available for
        backwards compatibility.
    device : str | torch.device, optional
        Computational device for simulation.
        The default is ``None`` (infer from input).

    Returns
    -------
    sig : npt.ArrayLike
        Signal evolution of shape ``(..., len(flip))``.
    jac : npt.ArrayLike
        Derivatives of signal wrt ``diff`` parameters,
        of shape ``(..., len(diff), len(flip))``.
        Not returned if ``diff`` is ``None``.

    """
    return evaluated(
        bSSFPSimulator(),
        diff,
        device,
        T1=T1,
        T2=T2,
        M0=M0,
        B0=B0,
        chemshift=chemshift,
        flip=flip,
        TR=TR,
        TE=TE,
        phase_inc=phase_inc,
        **values,
    )
