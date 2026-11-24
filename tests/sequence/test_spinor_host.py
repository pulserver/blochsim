"""The host kernel behind ``compose_spinor``, against matrix exponentials.

A call that carries no derivatives and holds everything on the host is
composed by ``_spinor_cpu``. Each test pins that path against the pulse
exponentiated sample by sample as 2x2 matrices, which shares none of the
kernel's algebra, or against the differentiable composition it stands in for.
"""

import pytest
import torch

from blochsim import compose_spinor


def _exponentiated(drive: torch.Tensor, turns: torch.Tensor) -> torch.Tensor:
    """``(spins, 2, 2)`` products of ``expm(-i/2 (x sx + y sy + z sz))`` in play order.

    ``drive`` is ``(samples, spins)`` complex and ``turns`` ``(samples,
    spins)`` real, in radians per sample.
    """
    pauli = torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex128,
    )
    spins = drive.shape[1]
    net = torch.eye(2, dtype=torch.complex128).repeat(spins, 1, 1)
    for sample in range(drive.shape[0]):
        field = (
            drive[sample].real[:, None, None] * pauli[0]
            + drive[sample].imag[:, None, None] * pauli[1]
            + turns[sample][:, None, None] * pauli[2]
        )
        net = torch.matrix_exp(-0.5j * field) @ net
    return net


def _pulse(samples: int, spins: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    drive = 0.05 * torch.complex(
        torch.randn(samples, generator=generator, dtype=torch.float64),
        torch.randn(samples, generator=generator, dtype=torch.float64),
    )
    turns = 0.02 * torch.randn(samples, spins, generator=generator, dtype=torch.float64)
    return drive, turns


def test_a_pulse_under_a_held_detuning_is_the_product_of_its_exponentials():
    drive, turns = _pulse(200, 37, seed=1)
    held = turns[0]

    a, b = compose_spinor(drive, held)

    expected = _exponentiated(drive[:, None].expand(-1, 37), held.expand(200, -1))
    assert torch.allclose(a, expected[:, 0, 0], atol=1e-12)
    assert torch.allclose(b, expected[:, 0, 1], atol=1e-12)


def test_a_pulse_under_a_moving_gradient_is_the_product_of_its_exponentials():
    drive, turns = _pulse(200, 130, seed=2)

    a, b = compose_spinor(drive, list(turns))

    expected = _exponentiated(drive[:, None].expand(-1, 130), turns)
    assert torch.allclose(a, expected[:, 0, 0], atol=1e-12)
    assert torch.allclose(b, expected[:, 0, 1], atol=1e-12)


def test_a_drive_of_its_own_per_spin_is_the_product_of_its_exponentials():
    generator = torch.Generator().manual_seed(3)
    drive = 0.05 * torch.randn(90, 2, 3, generator=generator, dtype=torch.complex128)
    held = 0.01 * torch.randn(2, 3, generator=generator, dtype=torch.float64)

    a, b = compose_spinor(drive, held)

    expected = _exponentiated(drive.reshape(90, 6), held.reshape(1, 6).expand(90, -1))
    assert a.shape == held.shape
    assert torch.allclose(a.reshape(-1), expected[:, 0, 0], atol=1e-12)
    assert torch.allclose(b.reshape(-1), expected[:, 0, 1], atol=1e-12)


def test_the_host_kernel_gives_the_pair_the_differentiable_composition_does():
    drive, turns = _pulse(300, 70, seed=4)
    tracked = [turn.clone().requires_grad_() for turn in turns]

    fast = compose_spinor(drive, list(turns))
    slow = compose_spinor(drive, tracked)

    assert slow[0].requires_grad
    for native, composed in zip(fast, slow, strict=True):
        assert torch.allclose(native, composed.detach(), atol=1e-13)


def test_a_moving_gradient_shorter_than_the_drive_is_refused():
    drive, turns = _pulse(20, 4, seed=5)

    with pytest.raises(ValueError, match="samples"):
        compose_spinor(drive, list(turns[:-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_a_pulse_on_the_card_gives_the_pair_the_host_kernel_does():
    drive, turns = _pulse(200, 50, seed=6)

    host = compose_spinor(drive, list(turns))
    card = compose_spinor(drive.cuda(), [turn.cuda() for turn in turns])

    for on_host, on_card in zip(host, card, strict=True):
        assert torch.allclose(on_host, on_card.cpu(), atol=1e-12)
