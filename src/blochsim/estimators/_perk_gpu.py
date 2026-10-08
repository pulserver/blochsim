"""The fused PERK kernels on a card.

Estimating a parameter from a signal is, once PERK is fitted, one line of
arithmetic per voxel::

    y = parameter_mean + (scale * cos(W @ x + b) - feature_mean) @ weight.T

Written as Torch operations that line builds the whole ``(voxels, features)``
matrix, writes it to memory and reads it back to contract it away again. On a
million voxels at a thousand features that is nearly four gigabytes of traffic
carrying nothing the answer needs. The kernels form a block of features and
consume it into the output while it is still in registers, so the matrix never
exists; the adjoint forms the angles again rather than keeping them.
"""

from __future__ import annotations

__all__ = ["regress", "regress_vjp"]

import math
from functools import cache

import torch

from .._gpu_launch import Kernel, cdiv

#: Threads per program, the voxels a program holds, and the features the
#: forward pass and the adjoint form at once: ``THREADS``, ``THREADS * VOXELS``,
#: ``FEATURES`` and ``ADJOINT_FEATURES`` in ``_perk_kernels.hpp``.
_THREADS = 64
_BLOCK_VOXELS = 128
_FEATURES = 32
_ADJOINT_FEATURES = 16
#: Programs per multiprocessor a launch is split to reach.
_PROGRAMS_PER_SM = 4

_regress_kernel = Kernel("_regress_kernel")
_regress_vjp_kernel = Kernel("_regress_vjp_kernel")


def _ready(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous float32 tensor the kernels can read row by row."""
    return tensor.detach().to(torch.float32).contiguous()


@cache
def _multiprocessors(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    return torch.cuda.get_device_properties(device).multi_processor_count


def _splits(device: torch.device, voxels: int, features: int, block: int) -> int:
    """How many programs share a block of voxels' features.

    One, unless the voxels' programs alone leave the card's multiprocessors
    short of work; then as many as fill them, a block of features at least
    each.
    """
    programs = cdiv(voxels, _BLOCK_VOXELS)
    wanted = _PROGRAMS_PER_SM * _multiprocessors(device)
    return max(1, min(cdiv(features, block), wanted // programs))


def regress(
    signals: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
    feature_mean: torch.Tensor,
    weight: torch.Tensor,
    parameter_mean: torch.Tensor,
) -> torch.Tensor:
    """Estimate parameters from ``(voxels, contrasts)`` signals.

    Returns
    -------
    torch.Tensor
        ``(voxels, parameters)``.
    """
    signals = _ready(signals)
    voxels, contrasts = signals.shape
    features = frequency.shape[0]
    parameters = weight.shape[0]
    splits = _splits(signals.device, voxels, features, _FEATURES)
    output = (torch.zeros if splits > 1 else torch.empty)(
        (voxels, parameters), dtype=torch.float32, device=signals.device
    )
    if voxels:
        _regress_kernel[(cdiv(voxels, _BLOCK_VOXELS), splits)](
            signals,
            _ready(frequency),
            _ready(phase),
            _ready(feature_mean),
            _ready(weight),
            _ready(parameter_mean),
            output,
            voxels,
            contrasts,
            features,
            parameters,
            math.sqrt(2.0 / features),
            splits,
            _THREADS,
        )
    return output


def regress_vjp(
    cotangent: torch.Tensor,
    signals: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The derivative of :func:`regress` with respect to ``signals``.

    Returns
    -------
    torch.Tensor
        ``(voxels, contrasts)``.
    """
    signals = _ready(signals)
    voxels, contrasts = signals.shape
    features = frequency.shape[0]
    parameters = weight.shape[0]
    splits = _splits(signals.device, voxels, features, _ADJOINT_FEATURES)
    output = (torch.zeros_like if splits > 1 else torch.empty_like)(signals)
    if voxels:
        _regress_vjp_kernel[(cdiv(voxels, _BLOCK_VOXELS), splits)](
            signals,
            _ready(frequency),
            _ready(phase),
            _ready(weight),
            _ready(cotangent),
            output,
            voxels,
            contrasts,
            features,
            parameters,
            math.sqrt(2.0 / features),
            splits,
            _THREADS,
        )
    return output
