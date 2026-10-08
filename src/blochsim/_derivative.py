"""Directional derivatives taken in reverse mode.

PyTorch's forward mode loads, the first time it makes a dual tensor in a
process, decompositions it registers through TorchScript. What the package
works out about its own structure -- how a packing moves with its arguments,
how a transition table moves with its flip -- is therefore taken in reverse
mode, so that a simulation and its gradients compile nothing at run time.
"""

from __future__ import annotations

__all__ = ["directional_derivatives"]

from collections.abc import Callable, Sequence
from typing import Any

import torch


def directional_derivatives(
    function: Callable[..., Any],
    primals: Sequence[torch.Tensor],
    directions: Sequence[Sequence[torch.Tensor]],
) -> tuple[Any, tuple[Any, ...]]:
    """``function`` at ``primals``, and its derivative along each direction.

    What :func:`torch.func.jvp` returns for each direction, by reverse passes:
    the vector-Jacobian product is linear in its cotangent, so its own
    vector-Jacobian product, taken with a direction as the cotangent, is the
    Jacobian times that direction. One forward pass serves every direction, and
    the passes compose with an enclosing :mod:`torch.func` transform.

    Parameters
    ----------
    function:
        Takes the primals positionally and returns a tensor or a tuple of
        tensors, real or complex.
    primals:
        Floating-point tensors.
    directions:
        Each a tangent per primal, of the primal's shape.

    Returns
    -------
    tuple
        The outputs, as ``function`` returns them, and per direction their
        derivatives in the same form.
    """
    shape: dict[str, Any] = {}

    def real(*given: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = function(*given)
        shape["single"] = isinstance(outputs, torch.Tensor)
        held = (outputs,) if shape["single"] else tuple(outputs)
        shape["complex"] = tuple(out.is_complex() for out in held)
        return tuple(
            torch.view_as_real(out) if out.is_complex() else out for out in held
        )

    values, pull = torch.func.vjp(real, *primals)
    _, push = torch.func.vjp(
        lambda *cotangents: pull(cotangents), *(torch.zeros_like(v) for v in values)
    )

    def shaped(parts: Sequence[torch.Tensor]) -> Any:
        out = tuple(
            torch.view_as_complex(part.contiguous()) if is_complex else part
            for part, is_complex in zip(parts, shape["complex"], strict=True)
        )
        return out[0] if shape["single"] else out

    slopes = tuple(
        shaped(
            push(
                tuple(
                    tangent.to(primal.dtype)
                    for tangent, primal in zip(direction, primals, strict=True)
                )
            )
        )
        for direction in directions
    )
    return shaped(values), slopes
