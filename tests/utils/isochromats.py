"""A CPMG train by isochromat summation, which shares nothing with a phase graph.

Every isochromat precesses through its own fixed angle in each half echo
interval -- a whole number of turns across the ensemble, as a crusher pair
leaves it -- and is turned by every refocusing pulse about the axis the
excitation left it along. The echo is the mean transverse magnetization: its
real part lies along the refocusing axis and its imaginary part across it.
"""

import numpy as np


def cpmg_echoes(flip_deg, ESP=None, first_ESP=None, T1=np.inf, T2=np.inf, count=4096):
    """Complex echoes of one train of refocusing flip angles, in degrees."""
    precession = np.exp(2j * np.pi * np.arange(count) / count)
    transverse = np.ones(count, dtype=complex)
    longitudinal = np.zeros(count)
    echoes = []
    for echo, angle in enumerate(np.deg2rad(flip_deg)):
        half = (
            0.0 if ESP is None else (first_ESP if echo == 0 and first_ESP else ESP) / 2
        )
        e1, e2 = np.exp(-half / T1), np.exp(-half / T2)
        for after_pulse in (False, True):
            if after_pulse:
                across = transverse.imag.copy()
                transverse = transverse.real + 1j * (
                    np.cos(angle) * across - np.sin(angle) * longitudinal
                )
                longitudinal = np.sin(angle) * across + np.cos(angle) * longitudinal
            transverse = transverse * e2
            longitudinal = longitudinal * e1 + (1 - e1)
            transverse = transverse * precession
        echoes.append(transverse.mean())
    return np.array(echoes)
