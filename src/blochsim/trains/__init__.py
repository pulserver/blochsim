"""Refocusing trains whose echoes follow a prescription.

Everything here runs on the host in double precision and takes and returns
NumPy arrays; importing it does not import PyTorch.
"""

from __future__ import annotations

__all__ = [
    "busse_centre_echo",
    "busse_flips",
    "effective_te",
    "prescribed_flips",
]

from ._busse import busse_centre_echo, busse_flips
from ._prescribed import effective_te, prescribed_flips
