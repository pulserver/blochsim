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

import torch

from .._gpu_launch import Kernel, cdiv

#: Voxels per program, one per thread.
_BLOCK_VOXELS = 128

_regress_kernel = Kernel("_regress_kernel")
_regress_vjp_kernel = Kernel("_regress_vjp_kernel")


def _ready(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous float32 tensor the kernels can read row by row."""
    return tensor.detach().to(torch.float32).contiguous()


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
    output = torch.empty(
        (voxels, parameters), dtype=torch.float32, device=signals.device
    )
    if voxels:
        _regress_kernel[(cdiv(voxels, _BLOCK_VOXELS),)](
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
            _BLOCK_VOXELS,
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
    output = torch.empty_like(signals)
    if voxels:
        _regress_vjp_kernel[(cdiv(voxels, _BLOCK_VOXELS),)](
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
            _BLOCK_VOXELS,
        )
    return output
