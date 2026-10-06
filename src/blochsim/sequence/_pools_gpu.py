"""The GPU kernels for a tissue whose pools are tabulated per interval.

The device side of ``simulate_pooled`` and ``simulate_pooled_adjoint`` in
``_epg_cpu.cpp``, reading the tables ``blochsim.sequence._pools`` builds. One
program carries one (train, voxel) problem, and its states are tiles of pools
by dephasing orders, so an interval's relaxation and exchange is a product of
the tabulated operator with the tile and a pulse turns every exchanging pool at
once.

Each body is written once over dual numbers. A complex dual is the tuple
``(real, imag, tangent real, tangent imag)`` and a real one ``(value,
tangent)``; with ``following`` off the helpers leave the tangents at zero and
compute none of them, so the pass that follows no direction pays for none.
"""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

import torch

from .._gpu_launch import Kernel, next_power_of_2
from ._accelerators import _shim_count, _train_count
from ._epg_gpu import _TRAJECTORY_BUDGET_BYTES, _atom_stride, _output_shape
from ._parameters import (
    FLOAT_NAMES,
    NO_GEOMETRY,
    TISSUE_NAMES,
    Geometry,
    tissue_gradient_bases,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from ._parameters import feature_flags as _feature_flags

# The seven per-voxel properties a tabulated tissue still reads, in packing
# order. Relaxation, exchange and the pools' shares are inside the tables.
_VOXEL = (
    "m0",
    "b1",
    "b1_phase_rad",
    "b0_hz",
    "inversion_efficiency",
    "diffusion_um2_per_ms",
    "velocity_m_per_s",
)
_VOXEL_INDEX = tuple(TISSUE_NAMES.index(name) for name in _VOXEL)

_pooled_kernel = Kernel("_pooled_kernel")
_pooled_adjoint_kernel = Kernel("_pooled_adjoint_kernel")


# ---------------------------------------------------------------------------
# Launchers.
# ---------------------------------------------------------------------------


def _tiles(layout: Any, state_count: int) -> dict[str, int]:
    """The tile shape a launch runs: pools along ``P`` and orders along ``S``."""
    pools = next_power_of_2(layout.longitudinal)
    states = next_power_of_2(state_count)
    return {
        "n": layout.longitudinal,
        "m": layout.transverse,
        "blocks": layout.blocks,
        "P": pools,
        "S": states,
    }


def _inputs(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    pools: Any,
    profile: Any,
    lineshape: Any,
    dynamic: Any,
    tissue_tangents: tuple[torch.Tensor, ...] | None,
    event_tangents: tuple[torch.Tensor, ...] | None,
    dynamic_direction: Any,
) -> tuple[torch.Tensor, ...]:
    """The buffers both kernels open with, in their order.

    A buffer whose branch is compiled out is still an argument, so one the
    launch already holds stands in for it.
    """
    device = tissue[0].device
    voxel = tuple(tissue[index] for index in _VOXEL_INDEX)
    moving = (
        voxel
        if tissue_tangents is None
        else tuple(tissue_tangents[index] for index in _VOXEL_INDEX)
    )
    duration, kind, flip, phase = events[:4]
    stepping = (duration, flip, phase) if event_tangents is None else event_tangents
    pairs = duration if dynamic is None else dynamic.packed(device)
    return (
        *voxel,
        *moving,
        *events[:9],
        *stepping,
        pools.values,
        pools.values if pools.direction is None else pools.direction,
        pools.index,
        duration if profile is None else profile.packed(device),
        kind if profile is None else profile.rows(device),
        duration if lineshape is None else lineshape.packed(device),
        pairs,
        kind
        if dynamic is None
        else dynamic.rows_per_event(_train_count(events), kind.numel()).to(device),
        pairs if dynamic_direction is None else dynamic_direction.to(device),
    )


def _scalars(
    base: int,
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_count: int,
    state_count: int,
    pools: Any,
    geometry: Geometry,
    profile: Any,
    lineshape: Any,
) -> tuple[Any, ...]:
    """The scalar arguments both kernels take after their buffers."""
    return (
        base,
        tissue[0].numel(),
        events[1].numel(),
        output_count,
        state_count,
        pools.layout.rows,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        1 if profile is None else profile.points,
        0 if profile is None else profile.bins,
        0 if lineshape is None else lineshape.bins,
    )


def _switches(
    tissue: tuple[torch.Tensor, ...],
    moving: tuple[torch.Tensor, ...] | None,
    pools: Any,
    profile: Any,
    dynamic: Any,
    dynamic_direction: Any,
    features: frozenset[str] | None,
    geometry: Geometry,
    following: bool,
) -> dict[str, Any]:
    """The constexpr switches both kernels take."""
    return {
        "atom_stride": _atom_stride(tissue)
        if moving is None
        else _atom_stride(tissue, moving),
        "shimmed": _shim_count(tissue) > 1,
        "profiled": profile is not None and profile.bins > 0,
        "dynamic": dynamic is not None,
        "directed_pairs": following and dynamic_direction is not None,
        "directed_table": following and pools.direction is not None,
        "following": following,
        **_feature_flags(features, geometry),
    }


def _forward(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...] | None,
    event_tangents: tuple[torch.Tensor, ...] | None,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry,
    profile: Any,
    lineshape: Any,
    dynamic: Any,
    dynamic_direction: Any,
    features: frozenset[str] | None,
    pools: Any,
) -> torch.Tensor:
    following = tissue_tangents is not None
    atoms = tissue[0].numel()
    total = _train_count(events) * atoms
    output_real = torch.zeros(
        _output_shape(_train_count(events), atoms, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.zeros_like(output_real)
    if total:
        _pooled_kernel[(total,)](
            *_inputs(
                tissue,
                events,
                pools,
                profile,
                lineshape,
                dynamic,
                tissue_tangents,
                event_tangents,
                dynamic_direction,
            ),
            output_real,
            output_imag,
            output_real,
            *_scalars(
                0,
                tissue,
                events,
                output_count,
                state_count,
                pools,
                geometry,
                profile,
                lineshape,
            ),  # fmt: skip
            planes=12 if following else 6,
            keep=False,
            **_switches(
                tissue,
                tissue_tangents,
                pools,
                profile,
                dynamic,
                dynamic_direction,
                features,
                geometry,
                following,
            ),
            **_tiles(pools.layout, state_count),
        )
    return torch.complex(output_real, output_imag)


def simulate(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> torch.Tensor:
    """The signals of a tissue whose pools are tabulated, on CUDA."""
    return _forward(
        tissue,
        events,
        None,
        None,
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=None,
        features=features,
        pools=pools,
    )


def simulate_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> torch.Tensor:
    """The signals' derivative along a direction, for a tabulated tissue.

    ``pools.direction`` is the direction along the tables, if they move.
    """
    return _forward(
        tissue,
        events,
        tissue_tangents,
        tuple(event_tangents),
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        features=features,
        pools=pools,
    )


def _adjoint(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...] | None,
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry,
    profile: Any,
    lineshape: Any,
    dynamic: Any,
    dynamic_direction: Any,
    features: frozenset[str] | None,
    pools: Any,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Both planes of the adjoint: the cotangents, then their derivative.

    Each plane holds the tissue's gradient rows, then duration, flip and
    phase, then the pair's and the tables' cotangents where there are any.
    The derivative plane is zero unless ``tangents`` gives a direction.
    """
    following = tangents is not None
    device = tissue[0].device
    atoms = tissue[0].numel()
    trains = _train_count(events)
    event_count = events[1].numel()
    total = trains * atoms
    shims = _shim_count(tissue)
    tissue_tangents = None if tangents is None else tangents[: len(TISSUE_NAMES)]
    event_tangents = (
        None
        if tangents is None
        else tuple(tangents[FLOAT_NAMES.index(name)] for name in _EVENT)
    )
    inputs = _inputs(
        tissue,
        events,
        pools,
        profile,
        lineshape,
        dynamic,
        tissue_tangents,
        event_tangents,
        dynamic_direction,
    )
    switches = _switches(
        tissue,
        tissue_tangents,
        pools,
        profile,
        dynamic,
        dynamic_direction,
        features,
        geometry,
        following,
    )
    tiles = _tiles(pools.layout, state_count)
    planes = 12 if following else 6
    held = planes * tiles["P"] * tiles["S"] * max(1, event_count)
    wave = max(1, min(total, _TRAJECTORY_BUDGET_BYTES // (4 * held)))
    trajectory = torch.empty(wave * held, dtype=torch.float32, device=device)

    grad_output = grad_output.resolve_conj()
    grad_real = grad_output.real.contiguous()
    grad_imag = grad_output.imag.contiguous()
    duration, _kind, flip, phase = events[:4]
    pairs = inputs[_PAIRS]

    def plane() -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(
                tissue_gradient_height(shims) * atoms,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros_like(duration),
            torch.zeros_like(flip),
            torch.zeros_like(phase),
            torch.zeros(
                (trains, *pools.values.shape), dtype=torch.float32, device=device
            ),
            torch.zeros_like(pairs),
        )

    value, tangent = plane(), plane()
    rows = tissue_gradient_bases(shims)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        scalars = _scalars(
            base, tissue, events, output_count, state_count, pools, geometry,
            profile, lineshape,
        )  # fmt: skip
        _pooled_kernel[(span,)](
            *inputs,
            grad_real,
            grad_imag,
            trajectory,
            *scalars,
            planes=planes,
            keep=True,
            **switches,
            **tiles,
        )
        _pooled_adjoint_kernel[(span,)](
            *inputs,
            grad_real,
            grad_imag,
            *(entry for pair in zip(value, tangent, strict=True) for entry in pair),
            trajectory,
            *scalars,
            *(rows[TISSUE_NAMES.index(name)] for name in _VOXEL),
            planes=planes,
            **switches,
            **tiles,
        )

    def gathered(side: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        voxel = tuple(
            side[0][start * atoms : (start + count) * atoms]
            for start, count in zip(rows, tissue_gradient_rows(shims), strict=True)
        )
        return (
            *voxel,
            side[1],
            side[2],
            side[3],
            *(() if dynamic is None else (side[5],)),
            side[4].sum(0),
        )

    return gathered(value), gathered(tangent)


# Where the pair sits among the buffers ``_inputs`` returns.
_PAIRS = 2 * len(_VOXEL) + 9 + 3 + 3 + 3

_EVENT = ("duration", "flip", "phase")


def simulate_vjp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    dynamic: Any = None,
    lineshape: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> tuple[torch.Tensor, ...]:
    """The first-order adjoint of a tabulated tissue, on CUDA.

    Returns the tissue's gradient rows, then duration, flip and phase, then the
    pair's cotangent where there is a pair and the tables' last.
    """
    value, _ = _adjoint(
        tissue,
        events,
        None,
        grad_output,
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=None,
        features=features,
        pools=pools,
    )
    return value


def simulate_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: Any,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse through a tabulated tissue, on CUDA.

    Returns the gradients with respect to the primal inputs -- the derivative
    of the adjoint along ``tangents`` -- and then with respect to the tangent
    inputs, which is the adjoint itself.
    """
    value, tangent = _adjoint(
        tissue,
        events,
        tangents,
        grad_output,
        state_count=state_count,
        output_count=output_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        features=features,
        pools=pools,
    )
    return tangent, value
