"""The EPG state machines on a card, through the compiled kernels."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

import torch

from .._gpu_launch import Kernel, cdiv, next_power_of_2
from ._accelerators import _shim_count, _train_count
from ._parameters import FLOAT_NAMES as _FLOAT_NAMES
from ._parameters import (
    NARROW_SPREAD,
    NO_GEOMETRY,
    Geometry,
    narrow_three_pool,
    three_pool_spread_rate,
    tissue_gradient_bases,
    tissue_gradient_height,
    tissue_gradient_rows,
)
from ._parameters import (
    feature_flags as _feature_flags,
)

# Where the two event directions the real-subspace adjoint follows sit among
# the differentiable inputs. Named rather than counted, so a tissue parameter
# added ahead of them moves them instead of silently renaming a neighbour.
_DURATION_SEED = _FLOAT_NAMES.index("duration")
_FLIP_SEED = _FLOAT_NAMES.index("flip")

_epg_kernel = Kernel("_epg_kernel")
_epg_jvp_kernel = Kernel("_epg_jvp_kernel")
_epg_vjp_kernel = Kernel("_epg_vjp_kernel")
_epg_vjp_jvp_kernel = Kernel("_epg_vjp_jvp_kernel")
_epg_real_kernel = Kernel("_epg_real_kernel")
_epg_real_jvp_kernel = Kernel("_epg_real_jvp_kernel")
_epg_real_vjp_kernel = Kernel("_epg_real_vjp_kernel")
_epg_real_vjp_jvp_kernel = Kernel("_epg_real_vjp_jvp_kernel")
_three_pool_table_kernel = Kernel("_three_pool_table_kernel")
_three_pool_table_jvp_kernel = Kernel("_three_pool_table_jvp_kernel")


def _pool_flag(lineshape: Any, exchanging: bool) -> int:
    """Which pools a launch is to carry, as the kernels' own constexpr reads it.

    Kept in one place so a launcher cannot describe the tissue one way and the
    kernel read it another.
    """
    if lineshape is not None and exchanging:
        return 3
    if exchanging:
        return 2
    return 1 if lineshape is not None else 0


# The operator table holds nine entries per voxel per distinct interval and
# the adjoint's cotangent table twelve, both float32.
_TABLE_FLOATS_PER_ROW = 9
_BAR_FLOATS_PER_ROW = 12

# What the two tables may take of what the card can spare. The trajectory is
# the larger claim on the same memory and is allocated after them.
_TABLE_SHARE = 0.25
_TABLE_FLOOR_BYTES = 64 << 20


def _three_pool_table_bytes(
    tissue: tuple[torch.Tensor, ...],
    rows: int,
    *,
    problems: int | None,
    dual: bool,
) -> int:
    """What the tables would take -- the operator's, and the adjoint's bars.

    The operator table holds a row of voxels; the cotangent table holds a row
    of problems, which is voxels times trains cut to what one chunk carries.
    ``problems`` of ``None`` is a caller that builds no cotangent table.

    A ``dual`` launch stores the operator twice over, value and direction, and
    pools its cotangents three times over: the value bars, the tangent bars,
    and the value bars weighted by each event's own interval direction.
    """
    entries = _TABLE_FLOATS_PER_ROW * (2 if dual else 1)
    total = int(tissue[0].numel()) * int(rows) * entries
    if problems is not None:
        pooled = _BAR_FLOATS_PER_ROW * (3 if dual else 1)
        total += int(problems) * int(rows) * pooled
    return total * 4


def _table_budget(device: torch.device) -> int:
    """How many bytes the three-pool tables may claim on this device."""
    if device.type != "cuda":
        return _TABLE_FLOOR_BYTES
    free, _total = torch.cuda.mem_get_info(device)
    return max(_TABLE_FLOOR_BYTES, int(free * _TABLE_SHARE))


def _three_pool_table_jvp(
    tissue: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    durations: torch.Tensor,
) -> torch.Tensor:
    """The three-pool operator and a direction through it, per distinct length.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    tangents:
        The directions along them, in the same order.
    durations:
        The distinct interval lengths, in seconds.

    Returns
    -------
    torch.Tensor
        ``(rows, 18, voxels)`` float32, undamped and at ``d_dt`` of zero.
    """
    voxels = int(tissue[0].numel())
    rows = int(durations.numel())
    table = torch.empty(
        (rows, 18, voxels), dtype=torch.float32, device=tissue[0].device
    )
    order = (0, 14, 11, 13, 10, 12, 9)
    block = min(1024, next_power_of_2(max(voxels, 1)))
    spread = durations.abs().to(torch.float64) * three_pool_spread_rate(tissue)
    for picked, narrow in (
        (torch.nonzero(spread <= NARROW_SPREAD).flatten(), True),
        (torch.nonzero(spread > NARROW_SPREAD).flatten(), False),
    ):
        if picked.numel() == 0:
            continue
        _three_pool_table_jvp_kernel[(picked.numel(), cdiv(voxels, block))](
            *(tissue[index] for index in order),
            *(tangents[index] for index in order),
            durations.to(torch.float32),
            picked.to(torch.int32),
            table,
            voxels,
            BLOCK=block,
            narrow=narrow,
        )
    return table


def _tabulate_three_pool(
    tissue: tuple[torch.Tensor, ...],
    duration: torch.Tensor,
    *,
    pools: int,
    narrow: bool,
    problems: int | None = None,
    tangents: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """The operator table an event loop should read, or ``None`` to form it.

    Only a wide launch has anything to gain: under ``narrow`` the operator is
    already 504 float32 instructions and forming it per event costs less than
    a round trip through memory.

    A wide launch is wide because of its longest interval, and one preparation
    delay is enough -- so the events that pay the roots in double are mostly
    events whose own length would have taken the series. Splitting the table by
    row is what lets each interval take the branch its own spread asks for,
    which is worth more than sharing a row between events and does not need a
    row to be shared at all.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    duration:
        The packed event durations, in seconds.
    pools:
        Which pool model the launch carries.
    narrow:
        Whether every interval keeps the eigenvalues close together.
    problems:
        How many problems a chunk of the adjoint carries, which is the height
        of the cotangent table it allocates. ``None`` for a caller that builds
        no such table.
    tangents:
        The directions along the tissue, for a caller that follows one. The
        table then carries the direction through the operator beside its
        value, at twice the width.

    Returns
    -------
    tuple
        The per-event row index, the table and the distinct lengths, or
        ``(None, None, None)``.
    """
    if pools != 3 or narrow:
        return None, None, None
    distinct, inverse = torch.unique(duration.detach(), return_inverse=True)
    # A row costs a formation, an event costs one too, so a train whose
    # lengths are all different has nothing to gain and a table to write.
    if distinct.numel() >= duration.numel():
        return None, None, None
    if _three_pool_table_bytes(
        tissue, distinct.numel(), problems=problems, dual=tangents is not None
    ) > _table_budget(tissue[0].device):
        # A pathological train has as many lengths as events, and the tables
        # grow with their product. Forming the operator per event is slower
        # and always fits, so that is what an unbounded one falls back to.
        return None, None, None
    lengths = distinct.to(torch.float32).contiguous()
    if tangents is not None:
        built = _three_pool_table_jvp(tissue, tangents, lengths)
    else:
        built = _three_pool_table(tissue, lengths)
    return inverse.reshape(duration.shape).to(torch.int32), built, lengths


def _three_pool_table(
    tissue: tuple[torch.Tensor, ...], durations: torch.Tensor
) -> torch.Tensor:
    """The three-pool operator for each distinct interval, over every voxel.

    Parameters
    ----------
    tissue:
        The prepared per-voxel buffers, in ``TISSUE_NAMES`` order.
    durations:
        The distinct interval lengths, in seconds.

    Returns
    -------
    torch.Tensor
        ``(rows, 9, voxels)`` float32, undamped -- the reading event applies
        its own washout.
    """
    (
        t1,
        _t2,
        _m0,
        _b1,
        _b1_phase,
        _b0,
        _inversion,
        _diffusion,
        _velocity,
        bound_fraction,
        bound_exchange,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        _t2_pool_b,
        _pool_b_shift,
    ) = tissue
    voxels = t1.numel()
    rows = durations.numel()
    table = torch.empty((rows, 9, voxels), dtype=torch.float32, device=t1.device)
    # The spread a row reaches is its own length times the rate, so the split
    # is exact per row rather than one verdict for the whole table.
    spread = durations.abs().to(torch.float64) * three_pool_spread_rate(tissue)
    block = min(1024, next_power_of_2(max(voxels, 1)))
    narrow_rows = torch.nonzero(spread <= NARROW_SPREAD, as_tuple=False).flatten()
    wide_rows = torch.nonzero(spread > NARROW_SPREAD, as_tuple=False).flatten()
    for picked, narrow in ((narrow_rows, True), (wide_rows, False)):
        if picked.numel() == 0:
            continue
        _three_pool_table_kernel[(picked.numel(), cdiv(voxels, block))](
            t1,
            t1_pool_b,
            t1_bound,
            pool_b_exchange,
            bound_exchange,
            pool_b_fraction,
            bound_fraction,
            durations.to(torch.float32),
            picked.to(torch.int32),
            table,
            voxels,
            BLOCK=block,
            narrow=narrow,
        )
    return table


# Threads of one program: two warps, their lanes along the states and, where
# the states are fewer, across rows of problems. A program of one warp caps a
# card at as many warps as it runs blocks, short of what the registers allow.
_PROGRAM_THREADS = 64


def _atom_stride(*tuples: tuple[torch.Tensor, ...]) -> int:
    """How far to step through a property to reach one voxel's value.

    Zero where every optional property was given as one value for the whole
    tissue: each is then read at one address by every voxel and needs no room
    per voxel. The relaxation times lead each tuple and are stepped by one
    whatever this says, since a tissue is its two relaxation times before it is
    anything else.

    One stride serves the values and the directions followed beside them, so a
    pass carrying tangents is asked about both: a direction laid out per voxel
    has to be stepped through even where the value it follows is one number.
    """
    return (
        0 if all(value.numel() <= 1 for values in tuples for value in values[2:]) else 1
    )


def _problems_per_program(block_states: int, kernel: Kernel) -> int:
    """How many independent problems to carry on one program's lane axis.

    A warp's lanes cost about the same whether they are used or not, so packing
    several problems into one program is close to free. Each thread then holds
    ``kernel.lanes`` problems of its row in registers, and the work it does once
    an event -- reading it, branching on it -- serves all of them.

    It depends on the state count alone, and deliberately not on how many
    problems the launch has. A run cut into chunks would otherwise compile a
    different tile from the same run whole, and the two tiles reassociate their
    arithmetic differently -- so a streamed volume would answer a little
    differently from an unstreamed one, which is a difference a caller has no
    way to account for. ``tests/sequence/test_both_pools.py`` pins that.

    The result sizes a block of threads, so it must be a power of two.
    """
    rows = max(1, _PROGRAM_THREADS // block_states)
    return (1 << (rows.bit_length() - 1)) * kernel.lanes


def _output_shape(
    train_count: int, atom_count: int, output_count: int
) -> tuple[int, ...]:
    """Signal shape, matching what the CPU kernels return."""
    if train_count == 1:
        return (atom_count, output_count)
    return (train_count, atom_count, output_count)


def _only_scalars(flags: dict) -> dict:
    """The switches a real-subspace kernel takes.

    Off-resonance and flow are not in its representation to begin with -- it
    carries three real planes where the complex kernels carry four -- so it is
    given the terms that survive that reduction and no others.
    """
    return {
        name: flags[name] for name in ("diffusing", "transmit", "density", "inverting")
    }


def simulate(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
    pools: Any = None,
) -> torch.Tensor:
    """Run a packed state machine on CUDA and return complex signals.

    ``real_axis`` of 1 selects the real-subspace kernel; see
    ``real_subspace_axis`` for when that is legitimate.
    """
    if pools is not None:
        from . import _pools_gpu

        return _pools_gpu.simulate(
            tissue,
            events,
            state_count=state_count,
            output_count=output_count,
            geometry=geometry,
            profile=profile,
            lineshape=lineshape,
            dynamic=dynamic,
            features=features,
            pools=pools,
        )
    train_count = _train_count(events)
    atom_count = tissue[0].numel()
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.empty_like(output_real)
    simulate_into(
        tissue,
        events,
        output_real,
        output_imag,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        exchanging=exchanging,
        dynamic=dynamic,
        features=features,
    )
    return torch.complex(output_real, output_imag)


def simulate_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    features: frozenset[str] | None = None,
) -> None:
    """Run the forward machine into buffers the caller owns.

    Streaming reuses one set of buffers per chunk, so allocating here would put
    an allocation in the loop -- and an allocation that reaches ``cudaMalloc``
    synchronizes the device, which is exactly what the streams exist to avoid.

    ``atom_count`` is given rather than taken from ``tissue`` because a chunk's
    buffers are sized for the largest chunk and the last one is shorter.
    """
    (
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        bound_fraction,
        bound_exchange,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
    ) = tissue
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
    ) = events
    train_count = _train_count(events)
    shims = _shim_count(tissue)
    pools = _pool_flag(lineshape, exchanging)
    block_states = next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(block_states, _epg_real_kernel if real_axis == 1 else _epg_kernel)
    grid = (cdiv(total, problems),)
    # A kernel argument has to be a tensor even where the branch reading it is
    # compiled out, so an unprofiled launch passes one it already has.
    table = None if profile is None else profile.packed(t1.device)
    pairs = None if dynamic is None else dynamic.packed(t1.device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, kind.numel()).to(t1.device)
    )
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    duration_row, pool_table, _lengths = _tabulate_three_pool(
        tissue, duration, pools=pools, narrow=narrow
    )

    # Phases grow without bound under RF spoiling, so their cosines and sines
    # are taken once here, in double precision, rather than in every program.
    phase_cos = torch.cos(phase.double()).to(torch.float32)
    phase_sin = torch.sin(phase.double()).to(torch.float32)

    if real_axis == 1:
        _epg_real_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            output_real,
            output_imag,
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            shimmed=_shim_count(tissue) > 1,
            **_only_scalars(_feature_flags(features, geometry)),
            block_states=block_states,
            problems=problems,
        )
        return

    _epg_kernel[grid](
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        bound_fraction,
        bound_exchange,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
        duration,
        kind,
        flip,
        phase,
        phase_cos,
        phase_sin,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
        t1 if table is None else table,
        kind if table_rows is None else table_rows,
        t1 if absorption is None else absorption,
        t1 if pairs is None else pairs,
        kind if pair_rows is None else pair_rows,
        kind if duration_row is None else duration_row,
        t1 if pool_table is None else pool_table,
        output_real,
        output_imag,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        state_count=state_count,
        single_train=train_count == 1,
        atom_stride=_atom_stride(tissue),
        shim_rows=shims,
        shimmed=shims > 1,
        locations=1 if profile is None else profile.points,
        profiled=profile is not None and profile.bins > 0,
        profile_bins=0 if profile is None else profile.bins,
        dynamic=dynamic is not None,
        broadened=lineshape is not None and lineshape.bins > 0,
        lineshape_bins=0 if lineshape is None else lineshape.bins,
        pools=pools,
        narrow=narrow,
        tabulated=pool_table is not None,
        **_feature_flags(features, geometry),
        block_states=block_states,
        problems=problems,
    )


def simulate_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: Any = None,
) -> torch.Tensor:
    """Run one fused state-machine Jacobian-vector product on CUDA.

    ``real_axis`` of 1 selects the real-subspace kernel, which produces no
    derivative along ``b1_phase``, ``b0`` or the RF phase -- seeds along those
    directions leave the subspace, so the caller must rule them out.
    """
    if pools is not None:
        from . import _pools_gpu

        return _pools_gpu.simulate_jvp(
            tissue,
            events,
            tissue_tangents,
            event_tangents,
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
    train_count = _train_count(events)
    atom_count = tissue[0].numel()
    output_real = torch.empty(
        _output_shape(train_count, atom_count, output_count),
        dtype=torch.float32,
        device=tissue[0].device,
    )
    output_imag = torch.empty_like(output_real)
    simulate_jvp_into(
        tissue,
        events,
        tissue_tangents,
        event_tangents,
        output_real,
        output_imag,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        exchanging=exchanging,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        features=features,
    )
    return torch.complex(output_real, output_imag)


def simulate_jvp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tissue_tangents: tuple[torch.Tensor, ...],
    event_tangents: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
) -> None:
    """Run one Jacobian-vector product into buffers the caller owns.

    See ``simulate_into`` for why the streaming path needs this.
    """
    (
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        bound_fraction,
        bound_exchange,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
    ) = tissue
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
    ) = events
    tangent_duration, tangent_flip, tangent_phase = event_tangents
    train_count = _train_count(events)
    pools = _pool_flag(lineshape, exchanging)
    shims = _shim_count(tissue)
    block_states = next_power_of_2(state_count)
    total = train_count * atom_count
    problems = _problems_per_program(block_states, _epg_real_jvp_kernel if real_axis == 1 else _epg_jvp_kernel)
    grid = (cdiv(total, problems),)

    if real_axis == 1:
        _epg_real_jvp_kernel[grid](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            tissue_tangents[0],
            tissue_tangents[1],
            tissue_tangents[2],
            tissue_tangents[3],
            tissue_tangents[6],
            tissue_tangents[7],
            tangent_duration,
            tangent_flip,
            output_real,
            output_imag,
            atom_count,
            train_count,
            kind.numel(),
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue, tissue_tangents),
            shimmed=shims > 1,
            **_only_scalars(_feature_flags(features, geometry)),
            block_states=block_states,
            problems=problems,
        )
        return

    table = None if profile is None else profile.packed(t1.device)
    pairs = None if dynamic is None else dynamic.packed(t1.device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, kind.numel()).to(t1.device)
    )
    pair_direction = (
        None if dynamic_direction is None else dynamic_direction.to(t1.device)
    )
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    duration_row, pool_table, _lengths = _tabulate_three_pool(
        tissue, duration, pools=pools, narrow=narrow, tangents=tissue_tangents
    )
    _epg_jvp_kernel[grid](
        *tissue,
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        *tissue_tangents,
        tangent_duration,
        tangent_flip,
        tangent_phase,
        saturation,
        rf_frequency,
        t1 if table is None else table,
        kind if table_rows is None else table_rows,
        t1 if absorption is None else absorption,
        t1 if pairs is None else pairs,
        kind if pair_rows is None else pair_rows,
        t1 if pair_direction is None else pair_direction,
        kind if duration_row is None else duration_row,
        t1 if pool_table is None else pool_table,
        output_real,
        output_imag,
        atom_count,
        train_count,
        kind.numel(),
        output_count,
        geometry.flow_scale,
        geometry.washout_scale,
        1.0 if profile is None else profile.step,
        1.0 if lineshape is None else lineshape.step,
        state_count=state_count,
        single_train=train_count == 1,
        atom_stride=_atom_stride(tissue, tissue_tangents),
        shim_rows=shims,
        shimmed=shims > 1,
        locations=1 if profile is None else profile.points,
        profiled=profile is not None and profile.bins > 0,
        profile_bins=0 if profile is None else profile.bins,
        dynamic=dynamic is not None,
        broadened=lineshape is not None and lineshape.bins > 0,
        lineshape_bins=0 if lineshape is None else lineshape.bins,
        pools=pools,
        narrow=narrow,
        tabulated=pool_table is not None,
        **_feature_flags(features, geometry),
        block_states=block_states,
        problems=problems,
    )


# How much device memory the recorded trajectory may hold at once. Beyond this
# the problems are run in waves, which the gradient buffers absorb because they
# accumulate rather than being written.
_TRAJECTORY_BUDGET_BYTES = 256 << 20


def _trajectory_wave(
    event_count: int, state_count: int, total: int, planes: int, blocks: int = 3
) -> int:
    """How many problems can record their trajectory in one launch."""
    per_problem = event_count * blocks * state_count * planes * 4
    return max(1, min(total, _TRAJECTORY_BUDGET_BYTES // max(1, per_problem)))


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
    exchanging: bool = False,
    features: frozenset[str] | None = None,
    pools: Any = None,
) -> tuple[torch.Tensor, ...]:
    """The first-order adjoint on CUDA, for a whole volume on one device.

    Returns the gradients in the differentiable-input order -- every tissue
    property, then event duration, flip and phase, and the pair's cotangent
    where one is given. A shard takes this same kernel a level down and a
    streamed volume has chunked launchers of its own;
    :func:`blochsim.sequence._accelerators` decides which route a run takes.

    Carrying no forward direction, this records two trajectory planes per
    recorded state where that pass records four, and holds one state where it
    holds a dual.
    """
    if pools is not None:
        from . import _pools_gpu

        return _pools_gpu.simulate_vjp(
            tissue,
            events,
            grad_output,
            state_count=state_count,
            output_count=output_count,
            geometry=geometry,
            profile=profile,
            dynamic=dynamic,
            lineshape=lineshape,
            features=features,
            pools=pools,
        )
    (
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        bound_fraction,
        exchange_rate,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
    ) = tissue
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
    ) = events[:9]
    atom_count = t1.numel()
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = next_power_of_2(state_count)
    device = t1.device
    shims = max(1, b1.numel() // atom_count) if atom_count else 1
    table = None if profile is None else profile.packed(device)
    table_rows = None if profile is None else profile.rows().to(device)
    pairs = None if dynamic is None else dynamic.packed(device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, event_count).to(device)
    )
    grad_pair = None if dynamic is None else torch.zeros_like(pairs)
    locations = 1 if profile is None else profile.points
    absorption = None if lineshape is None else lineshape.packed(device)

    grad_tissue = torch.zeros(
        tissue_gradient_height(shims) * atom_count,
        dtype=torch.float32,
        device=device,
    )
    grad_flip = torch.zeros_like(flip)
    grad_phase = torch.zeros_like(phase)
    grad_duration = torch.zeros_like(duration)
    grad_output = grad_output.resolve_conj()
    grad_real = grad_output.real.contiguous()
    grad_imag = grad_output.imag.contiguous()

    # A semisolid pool records a plane of its own beside the three the free
    # water keeps; a chemically exchanging one three, and the two together
    # four.
    pools = _pool_flag(lineshape, exchanging)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    blocks = 7 if pools == 3 else (6 if pools == 2 else (4 if pools == 1 else 3))
    wave = _trajectory_wave(event_count, state_count, total, 2, blocks)
    duration_row, pool_table, pool_durations = _tabulate_three_pool(
        tissue, duration, pools=pools, narrow=narrow, problems=wave
    )
    row_count = 0 if pool_durations is None else pool_durations.numel()
    pool_bars = None
    if pool_table is not None:
        # A slot per problem the chunk carries, so the walk back accumulates
        # into memory it owns and no two programs contend for a row. The
        # chunks run one after another, so one chunk's worth is enough.
        pool_bars = torch.zeros(
            wave * row_count * 12, dtype=torch.float32, device=device
        )
    trajectory = [
        torch.empty(
            (wave, event_count * blocks * state_count),
            dtype=torch.float32,
            device=device,
        )
        for _ in range(2)
    ]

    problems = _problems_per_program(block_states, _epg_vjp_kernel)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        if pool_bars is not None:
            # The slots are per chunk, so each chunk starts from nothing.
            pool_bars.zero_()
        # The trajectory is written by one launch and walked back by the
        # next, so each compiles one sweep instead of both.
        for recording in (True, False):
            _epg_vjp_kernel[(cdiv(span, problems),)](
                t1,
                t2,
                m0,
                b1,
                b1_phase,
                b0,
                inversion_efficiency,
                diffusion,
                velocity,
                bound_fraction,
                exchange_rate,
                t1_bound,
                pool_b_fraction,
                pool_b_exchange,
                t1_pool_b,
                t2_pool_b,
                pool_b_shift,
                duration,
                kind,
                flip,
                phase,
                action,
                output_index,
                shim_index,
                saturation,
                rf_frequency,
                absorption,
                table,
                table_rows,
                pairs,
                pair_rows,
                duration_row,
                pool_table,
                pool_bars,
                pool_durations,
                row_count,
                grad_pair,
                grad_real,
                grad_imag,
                grad_tissue,
                grad_flip,
                grad_phase,
                grad_duration,
                *trajectory,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                geometry.flow_scale,
                geometry.washout_scale,
                shims,
                1.0 if profile is None else profile.step,
                1.0 if lineshape is None else lineshape.step,
                state_count=state_count,
                single_train=train_count == 1,
                atom_stride=_atom_stride(tissue),
                shimmed=shims > 1,
                locations=locations,
                profiled=profile is not None and profile.bins > 0,
                profile_bins=0 if profile is None else profile.bins,
                dynamic=dynamic is not None,
                broadened=lineshape is not None and lineshape.bins > 0,
                lineshape_bins=0 if lineshape is None else lineshape.bins,
                pools=pools,
                narrow=narrow,
                tabulated=pool_table is not None,
                recording=recording,
                block_states=block_states,
                problems=problems,
                **_feature_flags(features, geometry),
            )
    voxel = tuple(
        grad_tissue[base * atom_count : (base + rows) * atom_count]
        for base, rows in zip(
            tissue_gradient_bases(shims), tissue_gradient_rows(shims), strict=True
        )
    )
    if dynamic is not None:
        return (*voxel, grad_duration, grad_flip, grad_phase, grad_pair)
    return (*voxel, grad_duration, grad_flip, grad_phase)


def simulate_real_vjp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """The first-order adjoint through the real subspace, on CUDA.

    Returns the gradients in the differentiable-input order -- every tissue
    property, then event duration, flip and phase. The representation divides
    the RF phase out, so transmit phase, off-resonance, velocity and RF phase
    come back at zero and callers must not ask for those.

    Carrying no forward direction, this records one trajectory plane where the
    forward-over-reverse pass records two, and holds one state where it holds a
    dual.
    """
    (
        t1,
        t2,
        m0,
        b1,
        _b1_phase,
        _b0,
        inversion_efficiency,
        diffusion,
        *_rest,
    ) = tissue
    duration, kind, flip, phase, action, output_index, shim_index = events[:7]
    atom_count = t1.numel()
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = next_power_of_2(state_count)
    device = t1.device
    shims = _shim_count(tissue)

    grad_tissue = torch.zeros(
        tissue_gradient_height(shims) * atom_count,
        dtype=torch.float32,
        device=device,
    )
    grad_flip = torch.zeros_like(flip)
    grad_duration = torch.zeros_like(duration)
    grad_phase = torch.zeros_like(phase)
    grad_imag = grad_output.resolve_conj().imag.contiguous()

    wave = _trajectory_wave(event_count, state_count, total, 1)
    trajectory = torch.empty(
        (wave, event_count * 3 * state_count), dtype=torch.float32, device=device
    )

    problems = _problems_per_program(block_states, _epg_real_vjp_kernel)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        _epg_real_vjp_kernel[(cdiv(span, problems),)](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            grad_imag,
            grad_tissue,
            grad_flip,
            grad_duration,
            trajectory,
            base,
            base + span,
            atom_count,
            train_count,
            event_count,
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            shim_rows=shims,
            shimmed=shims > 1,
            **_only_scalars(_feature_flags(features, NO_GEOMETRY)),
            block_states=block_states,
            problems=problems,
        )
    voxel = tuple(
        grad_tissue[base * atom_count : (base + rows) * atom_count]
        for base, rows in zip(
            tissue_gradient_bases(shims), tissue_gradient_rows(shims), strict=True
        )
    )
    return (*voxel, grad_duration, grad_flip, grad_phase)


class AdjointBuffers:
    """Device memory a forward-over-reverse pass writes into.

    Sized for ``chunk`` voxels and reusable for any narrower one. Per-voxel
    gradients are cleared before each pass; per-event gradients accumulate over
    every pass the buffers serve and are read out with ``event_gradients``.

    ``real_axis`` of 1 halves the state planes, so buffers built for one
    representation cannot be handed to the other.
    """

    def __init__(
        self,
        events: tuple[torch.Tensor, ...],
        chunk: int,
        *,
        state_count: int,
        output_count: int,
        real_axis: int | None = None,
        shims: int = 1,
        pools: int = 0,
    ) -> None:
        (
            duration,
            kind,
            flip,
            phase,
            _action,
            _output_index,
            _shim,
            _saturation,
            _rf_frequency,
        ) = events
        device = kind.device
        train_count = _train_count(events)
        event_count = kind.numel()
        self.planes = 2 if real_axis == 1 else 4
        # A bound pool records a fourth block of states per event: the RF
        # operator scales it, so the reverse sweep cannot replay it from the
        # free pool's.
        self.blocks = 3 + (4 if pools == 3 else (3 if pools == 2 else pools))
        self.chunk = chunk
        self.shims = shims
        self.rows = tissue_gradient_height(shims)
        self.state_count = state_count
        self.output_count = output_count
        self.train_count = train_count
        # One dual accumulator per plane: value is the gradient w.r.t. the
        # tangent inputs, tangent the gradient w.r.t. the primal ones.
        self.tissue = [
            torch.zeros(self.rows * chunk, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        self.flip = [torch.zeros_like(flip) for _ in range(2)]
        self.duration = [torch.zeros_like(duration) for _ in range(2)]
        self.phase = [torch.zeros_like(phase) for _ in range(2)]
        self.cotangent = [
            torch.empty(
                train_count * chunk * output_count,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self.wave = _trajectory_wave(
            event_count,
            state_count,
            train_count * chunk,
            self.planes,
            self.blocks,
        )
        self.trajectory = [
            torch.empty(
                (self.wave, event_count * self.blocks * state_count),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.planes)
        ]

    def tissue_gradients(self, atom_count: int) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The per-voxel gradients of the last pass, one entry per parameter.

        Each is flat and as wide as the buffer it belongs to, so the transmit
        pair spans every shim. Ordered to match ``event_gradients``: tangent
        plane first.
        """
        return tuple(
            tuple(
                self.tissue[plane][base * atom_count : (base + rows) * atom_count]
                for base, rows in zip(
                    tissue_gradient_bases(self.shims),
                    tissue_gradient_rows(self.shims),
                    strict=True,
                )
            )
            for plane in (1, 0)
        )

    def event_gradients(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """The per-event gradients summed over every pass so far.

        Ordered ``(duration, flip, phase)`` to match the tail of the
        differentiable-input order, tangent plane first.
        """
        return tuple(
            (self.duration[plane], self.flip[plane], self.phase[plane])
            for plane in (1, 0)
        )


class GradientBuffers:
    """Device memory a first-order adjoint writes into.

    Half of what the forward-over-reverse pass needs: one accumulator per
    gradient rather than a dual, and one trajectory plane per real state rather
    than a plane per component of one. Sized for ``chunk`` voxels and reusable
    for any narrower one; per-event gradients accumulate over every pass the
    buffers serve.

    ``real_axis`` of 1 halves the planes again, so buffers built for one
    representation cannot be handed to the other.
    """

    def __init__(
        self,
        events: tuple[torch.Tensor, ...],
        chunk: int,
        *,
        state_count: int,
        output_count: int,
        real_axis: int | None = None,
    ) -> None:
        duration, kind, flip, phase = events[:4]
        device = kind.device
        train_count = _train_count(events)
        event_count = kind.numel()
        self.real_axis = real_axis
        self.planes = 1 if real_axis == 1 else 2
        self.chunk = chunk
        self.rows = tissue_gradient_height(1)
        self.state_count = state_count
        self.output_count = output_count
        self.train_count = train_count
        self.tissue = torch.zeros(self.rows * chunk, dtype=torch.float32, device=device)
        self.flip = torch.zeros_like(flip)
        self.duration = torch.zeros_like(duration)
        self.phase = torch.zeros_like(phase)
        self.cotangent = [
            torch.empty(
                train_count * chunk * output_count,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self.wave = _trajectory_wave(
            event_count, state_count, train_count * chunk, self.planes
        )
        self.trajectory = [
            torch.empty(
                (self.wave, event_count * 3 * state_count),
                dtype=torch.float32,
                device=device,
            )
            for _ in range(self.planes)
        ]

    def tissue_gradients(self, atom_count: int) -> tuple[torch.Tensor, ...]:
        """The per-voxel gradients of the last pass, one entry per parameter."""
        return tuple(
            self.tissue[base * atom_count : (base + rows) * atom_count]
            for base, rows in zip(
                tissue_gradient_bases(1), tissue_gradient_rows(1), strict=True
            )
        )

    def event_gradients(self) -> tuple[torch.Tensor, ...]:
        """``(duration, flip, phase)``, summed over every pass so far."""
        return (self.duration, self.flip, self.phase)


def simulate_vjp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: GradientBuffers,
    *,
    state_count: int,
    output_count: int,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """One chunk of a first-order adjoint, into buffers the caller owns.

    ``grad_output`` is already on the device. Returns the per-voxel gradients
    of this chunk; the per-event ones accumulate in ``buffers``.
    """
    (
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        bound_fraction,
        exchange_rate,
        t1_bound,
        pool_b_fraction,
        pool_b_exchange,
        t1_pool_b,
        t2_pool_b,
        pool_b_shift,
    ) = tissue
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        saturation,
        rf_frequency,
    ) = events[:9]
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = next_power_of_2(state_count)

    buffers.tissue.zero_()
    grad_output = grad_output.resolve_conj()
    size = total * output_count
    grad_real = buffers.cotangent[0][:size]
    grad_imag = buffers.cotangent[1][:size]
    grad_real.copy_(grad_output.real.reshape(-1))
    grad_imag.copy_(grad_output.imag.reshape(-1))

    problems = _problems_per_program(block_states, _epg_vjp_kernel)
    for base in range(0, total, buffers.wave):
        span = min(buffers.wave, total - base)
        # The trajectory is written by one launch and walked back by the
        # next, so each compiles one sweep instead of both.
        for recording in (True, False):
            _epg_vjp_kernel[(cdiv(span, problems),)](
                t1,
                t2,
                m0,
                b1,
                b1_phase,
                b0,
                inversion_efficiency,
                diffusion,
                velocity,
                bound_fraction,
                exchange_rate,
                t1_bound,
                pool_b_fraction,
                pool_b_exchange,
                t1_pool_b,
                t2_pool_b,
                pool_b_shift,
                duration,
                kind,
                flip,
                phase,
                action,
                output_index,
                shim_index,
                saturation,
                rf_frequency,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                grad_real,
                grad_imag,
                buffers.tissue,
                buffers.flip,
                buffers.phase,
                buffers.duration,
                *buffers.trajectory,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                geometry.flow_scale,
                geometry.washout_scale,
                1,
                1.0,
                1.0,
                state_count=state_count,
                single_train=train_count == 1,
                atom_stride=_atom_stride(tissue),
                shimmed=False,
                locations=1,
                profiled=False,
                profile_bins=0,
                dynamic=False,
                broadened=False,
                lineshape_bins=0,
                pools=0,
                narrow=False,
                tabulated=False,
                recording=recording,
                block_states=block_states,
                problems=problems,
                **_feature_flags(features, geometry),
            )
    return buffers.tissue_gradients(atom_count)


def simulate_real_vjp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: GradientBuffers,
    *,
    state_count: int,
    output_count: int,
    atom_count: int,
    features: frozenset[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """The same, for a train the real subspace covers."""
    (
        t1,
        t2,
        m0,
        b1,
        _b1_phase,
        _b0,
        inversion_efficiency,
        diffusion,
        *_rest,
    ) = tissue
    duration, kind, flip, _phase, action, output_index, shim_index = events[:7]
    train_count = _train_count(events)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = next_power_of_2(state_count)

    buffers.tissue.zero_()
    size = total * output_count
    grad_imag = buffers.cotangent[1][:size]
    grad_imag.copy_(grad_output.resolve_conj().imag.reshape(-1))

    problems = _problems_per_program(block_states, _epg_real_vjp_kernel)
    for base in range(0, total, buffers.wave):
        span = min(buffers.wave, total - base)
        _epg_real_vjp_kernel[(cdiv(span, problems),)](
            t1,
            t2,
            m0,
            b1,
            inversion_efficiency,
            diffusion,
            duration,
            kind,
            flip,
            action,
            output_index,
            shim_index,
            grad_imag,
            buffers.tissue,
            buffers.flip,
            buffers.duration,
            buffers.trajectory[0],
            base,
            base + span,
            atom_count,
            train_count,
            event_count,
            output_count,
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue),
            # The streamed route carries one shim, as the complex one does:
            # ``GradientBuffers`` sizes its gradient plane for a single row.
            shim_rows=1,
            shimmed=False,
            block_states=block_states,
            problems=problems,
            **_only_scalars(_feature_flags(features, NO_GEOMETRY)),
        )
    return buffers.tissue_gradients(atom_count)


def simulate_vjp_jvp_into(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    buffers: AdjointBuffers,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    atom_count: int,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    dynamic_gradients: tuple[torch.Tensor, torch.Tensor] | None = None,
    features: frozenset[str] | None = None,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Forward-over-reverse for one chunk of voxels, into caller-owned buffers.

    Returns the per-voxel gradients of this chunk -- tangent plane first, one
    entry per tissue parameter, views into ``buffers`` that the next call
    overwrites. The per-event gradients accumulate inside ``buffers`` instead,
    because every chunk contributes to all of them.

    ``atom_count`` is this chunk's width, which may be narrower than the one
    the buffers were built for.
    """
    (
        t1,
        t2,
        m0,
        b1,
        b1_phase,
        b0,
        inversion_efficiency,
        diffusion,
        velocity,
        _bound_fraction,
        _bound_exchange,
        _t1_bound,
        _pool_b_fraction,
        _pool_b_exchange,
        _t1_pool_b,
        _t2_pool_b,
        _pool_b_shift,
    ) = tissue
    (
        duration,
        kind,
        flip,
        phase,
        action,
        output_index,
        shim_index,
        _saturation,
        _rf_frequency,
    ) = events
    train_count = _train_count(events)
    pools = _pool_flag(lineshape, exchanging)
    event_count = kind.numel()
    total = train_count * atom_count
    block_states = next_power_of_2(state_count)
    real = real_axis == 1

    grad_output = grad_output.resolve_conj()
    size = total * output_count
    grad_real, grad_imag = (
        plane[:size].view(grad_output.shape) for plane in buffers.cotangent
    )
    grad_real.copy_(grad_output.real)
    grad_imag.copy_(grad_output.imag)
    grad_tissue = [plane[: buffers.rows * atom_count] for plane in buffers.tissue]
    for plane in grad_tissue:
        plane.zero_()
    grad_flip, grad_duration, grad_phase = buffers.flip, buffers.duration, buffers.phase
    trajectory = buffers.trajectory
    table = None if profile is None else profile.packed(t1.device)
    pairs = None if dynamic is None else dynamic.packed(t1.device)
    pair_rows = (
        None
        if dynamic is None
        else dynamic.rows_per_event(train_count, kind.numel()).to(t1.device)
    )
    pair_direction = (
        None if dynamic_direction is None else dynamic_direction.to(t1.device)
    )
    grad_pair_value = None if dynamic_gradients is None else dynamic_gradients[0]
    grad_pair_tangent = None if dynamic_gradients is None else dynamic_gradients[1]
    table_rows = None if profile is None else profile.rows(kind.device)
    absorption = None if lineshape is None else lineshape.packed(t1.device)
    narrow = narrow_three_pool(tissue, duration, pools=pools)
    wave = buffers.wave
    duration_row, pool_table, pool_durations = _tabulate_three_pool(
        tissue,
        duration,
        pools=pools,
        narrow=narrow,
        tangents=tangents,
        problems=wave,
    )
    row_count = 0 if pool_durations is None else pool_durations.numel()
    pool_bars = None
    if pool_table is not None:
        # A slot per problem the chunk carries, so the walk back accumulates
        # into memory it owns and no two programs contend for a row. Three
        # sets of twelve: the value cotangents, their directions, and the
        # value cotangents weighted by each event's own interval direction.
        pool_bars = torch.zeros(
            wave * row_count * 36, dtype=torch.float32, device=t1.device
        )

    problems = _problems_per_program(block_states, _epg_real_vjp_jvp_kernel if real else _epg_vjp_jvp_kernel)
    for base in range(0, total, wave):
        span = min(wave, total - base)
        if pool_bars is not None:
            # The slots are per chunk, so each chunk starts from nothing.
            pool_bars.zero_()
        grid = (cdiv(span, problems),)
        shape = dict(
            state_count=state_count,
            single_train=train_count == 1,
            atom_stride=_atom_stride(tissue, tangents),
            block_states=block_states,
            problems=problems,
        )
        if real:
            _epg_real_vjp_jvp_kernel[grid](
                t1,
                t2,
                m0,
                b1,
                inversion_efficiency,
                diffusion,
                duration,
                kind,
                flip,
                action,
                output_index,
                shim_index,
                tangents[0],
                tangents[1],
                tangents[2],
                tangents[3],
                tangents[6],
                tangents[7],
                tangents[_DURATION_SEED],
                tangents[_FLIP_SEED],
                grad_imag,
                *grad_tissue,
                *grad_flip,
                *grad_duration,
                *trajectory,
                base,
                base + span,
                atom_count,
                train_count,
                event_count,
                output_count,
                shim_rows=_shim_count(tissue),
                shimmed=_shim_count(tissue) > 1,
                **_only_scalars(_feature_flags(features, geometry)),
                **shape,
            )
        else:
            # The trajectory is written by one launch and walked back by the
            # next, so each compiles one sweep instead of both.
            for recording in (True, False):
                _epg_vjp_jvp_kernel[grid](
                    *tissue,
                    *events,
                    t1 if table is None else table,
                    kind if table_rows is None else table_rows,
                    t1 if absorption is None else absorption,
                    t1 if pairs is None else pairs,
                    kind if pair_rows is None else pair_rows,
                    t1 if pair_direction is None else pair_direction,
                    t1 if grad_pair_value is None else grad_pair_value,
                    t1 if grad_pair_tangent is None else grad_pair_tangent,
                    *tangents,
                    kind if duration_row is None else duration_row,
                    t1 if pool_table is None else pool_table,
                    t1 if pool_bars is None else pool_bars,
                    t1 if pool_durations is None else pool_durations,
                    row_count,
                    grad_real,
                    grad_imag,
                    *grad_tissue,
                    *grad_flip,
                    *grad_phase,
                    *grad_duration,
                    *trajectory,
                    base,
                    base + span,
                    atom_count,
                    train_count,
                    event_count,
                    output_count,
                    geometry.flow_scale,
                    geometry.washout_scale,
                    1.0 if profile is None else profile.step,
                    1.0 if lineshape is None else lineshape.step,
                    shim_rows=_shim_count(tissue),
                    shimmed=_shim_count(tissue) > 1,
                    locations=1 if profile is None else profile.points,
                    profiled=profile is not None and profile.bins > 0,
                    profile_bins=0 if profile is None else profile.bins,
                    dynamic=dynamic is not None,
                    directed=dynamic_direction is not None,
                    broadened=lineshape is not None and lineshape.bins > 0,
                    lineshape_bins=0 if lineshape is None else lineshape.bins,
                    pools=pools,
                    narrow=narrow,
                    tabulated=pool_table is not None,
                    recording=recording,
                    **_feature_flags(features, geometry),
                    **shape,
                )

    # Plane 1 is the tangent part -> d/d(primal inputs); plane 0 the value part.
    return buffers.tissue_gradients(atom_count)


def simulate_vjp_jvp(
    tissue: tuple[torch.Tensor, ...],
    events: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
    *,
    state_count: int,
    output_count: int,
    real_axis: int | None = None,
    geometry: Geometry = NO_GEOMETRY,
    profile: Any = None,
    lineshape: Any = None,
    exchanging: bool = False,
    dynamic: Any = None,
    dynamic_direction: Any = None,
    features: frozenset[str] | None = None,
    pools: Any = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Forward-over-reverse through the state machine on CUDA.

    ``tangents`` follows the differentiable-input order -- every tissue
    property, then event duration, flip and phase -- and the two returned
    tuples, gradients with respect to the primal inputs then to the tangent
    inputs, follow it too.

    ``real_axis`` of 1 selects the real-subspace adjoint. That representation
    divides the RF phase out, so it leaves ``b1_phase``, ``b0`` and ``phase`` at
    zero and callers must not ask for those; the complex adjoint produces every
    one of them.

    Gradients land through atomic accumulation, so repeated runs agree to
    floating-point tolerance rather than bit for bit.
    """
    if pools is not None:
        from . import _pools_gpu

        return _pools_gpu.simulate_vjp_jvp(
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
    atom_count = tissue[0].numel()
    gradients = None
    if dynamic is not None:
        held = dynamic.packed(tissue[0].device)
        gradients = (torch.zeros_like(held), torch.zeros_like(held))
    buffers = AdjointBuffers(
        events,
        atom_count,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        shims=_shim_count(tissue),
        pools=_pool_flag(lineshape, exchanging),
    )
    voxel_grads = simulate_vjp_jvp_into(
        tissue,
        events,
        tangents,
        grad_output,
        buffers,
        state_count=state_count,
        output_count=output_count,
        real_axis=real_axis,
        atom_count=atom_count,
        geometry=geometry,
        profile=profile,
        lineshape=lineshape,
        exchanging=exchanging,
        dynamic=dynamic,
        dynamic_direction=dynamic_direction,
        dynamic_gradients=gradients,
        features=features,
    )
    sides = tuple(
        (*voxels, *per_event)
        for voxels, per_event in zip(
            voxel_grads, buffers.event_gradients(), strict=True
        )
    )
    if gradients is None:
        return sides
    # The value plane is the adjoint and the tangent plane its own derivative,
    # which is the split the tissue gradients take; the sides come back in the
    # order the caller reads them, curvature first.
    return (*sides[0], gradients[1]), (*sides[1], gradients[0])
