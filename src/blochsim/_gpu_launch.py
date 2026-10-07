"""Launching the compiled GPU kernels from the tensors a simulation holds.

A kernel is named and indexed with its grid, then called with its arguments in
the order its signature lists them, positionally or by name. Tensors on a card
are queued on PyTorch's current stream for that card; tensors on the host run
the same kernel source compiled for the host, one program at a time, which is
how the kernels are checked without a card.
"""

from __future__ import annotations

__all__: list[str] = []

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

import torch


def next_power_of_2(value: int) -> int:
    """The smallest power of two no less than ``value``."""
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def cdiv(numerator: int, denominator: int) -> int:
    """``numerator / denominator`` rounded up."""
    return -(-int(numerator) // int(denominator))


@cache
def _module(device_type: str) -> Any:
    if device_type == "cuda":
        from blochsim import _gpu

        return _gpu
    from blochsim import _gpu_host

    return _gpu_host


@cache
def _entry(name: str) -> tuple[tuple[str, ...], str, int]:
    params, kinds, lanes = _module("cuda" if available() else "cpu").kernels()[name]
    return tuple(params.split(",")), kinds, lanes


def _signature(name: str) -> tuple[tuple[str, ...], str]:
    names, kinds, _ = _entry(name)
    return names, kinds


@cache
def available() -> bool:
    """Whether this installation carries the kernels compiled for a card."""
    try:
        _module("cuda")
    except ImportError:
        return False
    return True


def specializations() -> list[tuple[str, dict[str, int]]]:
    """Each kernel compiled for one combination of its switches, and the switches.

    Empty where this installation carries no kernels for a card.
    """
    if not available():
        return []
    return [
        (
            kernel,
            {
                name: int(value)
                for name, value in (
                    pair.split("=") for pair in fixed.split(",") if pair
                )
            },
        )
        for kernel, fixed in _module("cuda").specializations()
    ]


def specialized_launches() -> int:
    """How many launches on a card have run a specialized kernel."""
    return _module("cuda").specialized_launches() if available() else 0


@contextmanager
def generic_kernels() -> Iterator[None]:
    """Run every launch inside on the kernels compiled for all combinations."""
    if not available():
        yield
        return
    previous = _module("cuda").use_specializations(False)
    try:
        yield
    finally:
        _module("cuda").use_specializations(previous)


class Kernel:
    """A compiled kernel, launched as ``kernel[grid](*arguments)``."""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def lanes(self) -> int:
        """Rows of a program's y axis each thread holds on a card."""
        return _entry(self.name)[2]

    def __getitem__(self, grid: tuple[int, ...]) -> Any:
        def run(*args: Any, **kwargs: Any) -> None:
            self.launch(tuple(int(count) for count in grid), args, kwargs)

        return run

    def launch(
        self, grid: tuple[int, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        names, kinds = _signature(self.name)
        values = list(args)
        for name in names[len(args) :]:
            if name not in kwargs:
                raise TypeError(f"{self.name} is missing argument {name!r}")
            values.append(kwargs[name])
        device = None
        packed: list[int | float] = []
        for name, kind, value in zip(names, kinds, values, strict=True):
            if value is None or (kind == "p" and isinstance(value, int)):
                # An argument the launch leaves out, which the kernel never reads.
                packed.append(0)
                continue
            if kind == "p":
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{self.name}: {name} must be a tensor")
                if device is None:
                    device = value.device
                elif value.device != device:
                    raise ValueError(
                        f"{self.name}: {name} is on {value.device}, not {device}"
                    )
                packed.append(value.data_ptr())
            elif kind == "f":
                packed.append(float(value))
            else:
                packed.append(int(value))
        if device is None or device.type == "cpu":
            _module("cpu").launch(self.name, grid, tuple(packed))
            return
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        stream = torch.cuda.current_stream(device).cuda_stream
        _module("cuda").launch(self.name, grid, tuple(packed), index, stream)
