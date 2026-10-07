"""Whether a kernel compiled for its switches or its layout computes what the general one does.

A kernel that quietly did not run agrees perfectly, so every case also asserts
that one did.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from blochsim import _gpu_launch, rf_definition
from blochsim.sequence import EpgEngine, exact_slice_profile, fse_description
from blochsim.sequence._simulation import TissueProperties

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _gpu_launch.specializations(),
    reason="needs a card and the specialized kernels",
)


def _tissue(atoms: int = 300) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "t1_ms": (200 + 2800 * torch.rand(atoms, generator=generator)).cuda(),
        "t2_ms": (10 + 290 * torch.rand(atoms, generator=generator)).cuda(),
    }


def _sinc_pulse(description):
    grid = np.linspace(-2.0, 2.0, 128)
    envelope = np.sinc(grid) * (0.54 + 0.46 * np.cos(np.pi * grid / 2.0))
    definition = rf_definition(
        envelope.astype(np.complex128),
        dwell_s=1e-5,
        bandwidth_hz=2000.0,
        definition_id=0,
    )
    return replace(description, rf_definitions={definition.id: definition})


def _forward(phase: float, profiled: bool = False) -> torch.Tensor:
    description = fse_description(
        torch.deg2rad(torch.full((32,), 150.0)),
        echo_spacing_s=5e-3,
        phases_rad=phase,
        excitation_phase_rad=torch.pi / 2,
    )
    across = None
    if profiled:
        description = _sinc_pulse(description)
        across = exact_slice_profile(9)
    return (
        EpgEngine()
        .simulate(
            description, TissueProperties(**_tissue()), across_slice=across, nstates=32
        )
        .signal
    )


def _gradient(phase: float) -> torch.Tensor:
    description = fse_description(
        torch.deg2rad(torch.full((32,), 150.0)), echo_spacing_s=5e-3, phases_rad=phase
    )
    tissue = _tissue()
    tissue["t2_ms"] = tissue["t2_ms"].clone().requires_grad_()
    signal = (
        EpgEngine().simulate(description, TissueProperties(**tissue), nstates=32).signal
    )
    signal.abs().sum().backward()
    return tissue["t2_ms"].grad


def _fast_launches() -> int:
    return _gpu_launch.specialized_launches() + _gpu_launch.layout_launches()


CASES = {
    "real forward": lambda: _forward(torch.pi / 2),
    "complex forward": lambda: _forward(0.0),
    "slice profile": lambda: _forward(torch.pi / 2, profiled=True),
    "real gradient": lambda: _gradient(torch.pi / 2),
    "complex gradient": lambda: _gradient(0.0),
}


@pytest.mark.parametrize("case", CASES)
def test_a_specialized_kernel_computes_what_the_general_one_does(case) -> None:
    with _gpu_launch.generic_kernels():
        general = CASES[case]()
    before = _fast_launches()
    special = CASES[case]()

    assert _fast_launches() > before
    error = (special - general).abs().max()
    scale = general.abs().max()
    assert float(error / scale) < 1e-5, f"{float(error):.3e} against {float(scale):.3e}"


def test_the_general_kernels_run_where_asked() -> None:
    before = _fast_launches()
    with _gpu_launch.generic_kernels():
        _forward(torch.pi / 2)

    assert _fast_launches() == before
