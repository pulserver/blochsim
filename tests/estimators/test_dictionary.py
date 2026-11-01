"""Tests for memory-bounded dictionary matching."""

from __future__ import annotations

import pytest
import torch

from blochsim.estimators import DictionaryMatcher
from blochsim.simulators import MultiEchoSimulator


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_dictionary_matcher_recovers_complex_scaled_atoms(device: str) -> None:
    parameter = torch.linspace(0.1, 2.0, 67, device=device)[:, None]
    dictionary = torch.cat(
        (torch.exp(1j * parameter), torch.exp(2j * parameter)), dim=-1
    )
    scale = torch.tensor([2.0 - 0.5j, -0.2 + 1.1j], device=device)
    selected = torch.tensor([3, 52], device=device)
    signals = scale[:, None] * dictionary[selected]
    matcher = DictionaryMatcher(
        dictionary=dictionary,
        parameters=parameter,
        query_chunk_size=1,
        dictionary_chunk_size=11,
        top_k=2,
    )

    result = matcher.match(signals)

    torch.testing.assert_close(result.indices[:, 0], selected)
    torch.testing.assert_close(result.parameters[:, 0, 0], parameter[selected, 0])
    torch.testing.assert_close(result.scales[:, 0], scale)
    torch.testing.assert_close(result.scores[:, 0], torch.ones(2, device=device))


def test_dictionary_matcher_forward_returns_indices_without_parameters() -> None:
    dictionary = torch.eye(4)
    matcher = DictionaryMatcher(dictionary=dictionary, dictionary_chunk_size=2)

    actual = matcher(dictionary[[2, 0]])

    torch.testing.assert_close(actual, torch.tensor([2, 0]))


def test_a_complex_measurement_of_a_real_model_keeps_both_its_parts() -> None:
    """The phase a voxel carries must not decide whether it matches.

    A model that is real is stored real, and half the arithmetic of a complex
    one. The measurement of it is not: proton density and receive phase put a
    complex scale on every voxel. Reading only the real part leaves
    ``Re(rho)`` times the signal, and normalizing puts the size back -- so on
    noiseless data it matches anyway, and the mistake does not show. What it
    also multiplies by ``Re(rho)`` is the signal-to-noise ratio, and a voxel
    whose phase approaches a quarter turn is then matching noise.
    """
    grid = torch.linspace(20.0, 300.0, 128)
    acquisition = MultiEchoSimulator(TE=torch.linspace(10.0, 200.0, 16))
    atoms = torch.as_tensor(acquisition.simulate(T2=grid))
    assert not torch.is_complex(atoms)
    matcher = DictionaryMatcher(dictionary=atoms, parameters=grid[:, None])
    truth = grid[::7]
    generator = torch.Generator().manual_seed(0)
    clean = torch.as_tensor(acquisition.simulate(T2=truth)).to(torch.complex64)
    noise = 0.01 * torch.complex(
        torch.randn(clean.shape, generator=generator),
        torch.randn(clean.shape, generator=generator),
    )
    step = float(torch.diff(grid).max())

    in_phase = None
    for turn in (0.0, 0.15, 0.25, 0.5):
        phase = torch.exp(2j * torch.pi * torch.tensor(turn))

        found = matcher(clean * phase + noise)[:, 0]

        error = float((found - truth).abs().mean())
        if turn == 0.0:
            in_phase = error
            assert error < step, f"the match misses in phase, by {error:.1f} ms"
        # Reading only the real part scales the signal-to-noise ratio by
        # ``Re(rho)``, which sends the quarter turn to matching noise across
        # the whole grid -- tens of milliseconds, not a fraction more than the
        # same data in phase costs.
        assert error < 2.0 * in_phase, (
            f"at {turn} turns the match drifted {error:.1f} ms, "
            f"against {in_phase:.1f} ms in phase"
        )


def _lattice_dictionary() -> tuple[torch.Tensor, torch.Tensor]:
    """Atoms over a triangular (T1, T2) lattice, T2 below T1, whose likeness
    falls with their distance in the logarithm: a bump centred on each atom's
    parameters, sampled at fixed points of the log plane."""
    t1 = torch.logspace(2, 3.5, 40)
    t2 = torch.logspace(1, 3, 35)
    g1, g2 = torch.meshgrid(t1, t2, indexing="ij")
    keep = g2 < g1
    parameters = torch.stack((g1[keep], g2[keep]), -1)
    u, v = torch.meshgrid(
        torch.linspace(2, 3.5, 30), torch.linspace(1, 3, 30), indexing="ij"
    )
    log = parameters.log10()
    distance = (log[:, :1] - u.reshape(1, -1)).square() + (
        log[:, 1:] - v.reshape(1, -1)
    ).square()
    return torch.exp(-distance / 0.02).to(torch.complex64) * (0.3 + 0.8j), parameters


@pytest.mark.parametrize("top_k", [1, 3])
def test_coarse_to_fine_search_finds_the_exhaustive_match(top_k: int) -> None:
    dictionary, parameters = _lattice_dictionary()
    torch.manual_seed(0)
    chosen = torch.randint(0, dictionary.shape[0], (200,))
    scale = torch.randn(200, dtype=torch.complex64)
    signals = scale[:, None] * dictionary[chosen]
    signals = signals + 0.002 * torch.randn_like(signals)
    exhaustive = DictionaryMatcher(
        dictionary=dictionary, parameters=parameters, top_k=top_k
    )
    lattice = DictionaryMatcher(
        dictionary=dictionary,
        parameters=parameters,
        top_k=top_k,
        coarse=3,
        query_chunk_size=64,
    )

    want, got = exhaustive.match(signals), lattice.match(signals)

    torch.testing.assert_close(got.indices[:, 0], want.indices[:, 0])
    torch.testing.assert_close(got.scales[:, 0], want.scales[:, 0])
    torch.testing.assert_close(got.scores, want.scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got.parameters, want.parameters)


def test_without_refinement_the_coarse_lattice_alone_is_searched() -> None:
    dictionary, parameters = _lattice_dictionary()
    matcher = DictionaryMatcher(
        dictionary=dictionary, parameters=parameters, coarse=4, refine=False
    )
    found = matcher.match(dictionary)

    on_coarse = set(matcher.lattice.coarse.tolist())
    assert set(found.indices[:, 0].tolist()) <= on_coarse
    assert len(on_coarse) < dictionary.shape[0] / 10
    matcher.refine = True
    torch.testing.assert_close(
        matcher.match(dictionary).indices[:, 0], torch.arange(dictionary.shape[0])
    )


def test_a_coarse_search_needs_parameters_and_no_groups() -> None:
    dictionary, parameters = _lattice_dictionary()
    with pytest.raises(ValueError, match="parameters"):
        DictionaryMatcher(dictionary=dictionary, coarse=2)
    with pytest.raises(ValueError, match="not both"):
        DictionaryMatcher(
            dictionary=dictionary, parameters=parameters, coarse=2, groups=8
        )
    with pytest.raises(ValueError, match="positive"):
        DictionaryMatcher(coarse=0)
    with pytest.raises(ValueError, match="same parameter values"):
        DictionaryMatcher(
            dictionary=torch.eye(3), parameters=torch.tensor([1.0, 1.0, 2.0]), coarse=1
        )


def test_a_coarse_search_follows_the_matcher_to_another_device() -> None:
    dictionary, parameters = _lattice_dictionary()
    matcher = DictionaryMatcher(dictionary=dictionary, parameters=parameters, coarse=3)
    moved = (
        matcher.to(torch.float64) if not torch.cuda.is_available() else matcher.cuda()
    )
    assert moved.lattice.table.device == moved.dictionary.device
    found = moved.match(dictionary[:5].to(moved.dictionary.device))
    torch.testing.assert_close(found.indices[:, 0].cpu(), torch.arange(5))
