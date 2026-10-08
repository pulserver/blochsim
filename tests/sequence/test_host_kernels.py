"""The GPU kernels, run on the host and held to what they specialize or to an
oracle.

The host build of the kernels runs the source the CUDA build compiles one
program at a time over host tensors, so these reach the plumbing without a
card: the argument alignment of the launchers, the trajectory planes a pool
model claims, the operator table's row indexing, and the washout and
recoveries the reading event applies.
"""

from __future__ import annotations

import pytest

pytest.importorskip("blochsim._gpu_host", reason="the host build is Linux only")

from utils import host_kernels


@pytest.mark.parametrize(
    "case",
    [
        "narrow",
        "wide",
        "chunked",
        "streamed",
        "washed",
        "shimmed",
        "profiled",
        "one_pool",
        "two_pools",
        "real",
        "real_shimmed",
        "spoiled",
        "narrowed",
    ],
)
def test_the_kernels_agree_with_what_they_specialize(case: str) -> None:
    """Each case runs one launch two ways and holds the two to each other.

    The table cases differ only in where the operator comes from; the pool
    cases hold a kernel to an oracle sharing no code with it, and to the pass
    it specializes.

    ``narrow`` forces a table onto a train the launch-wide gate calls narrow,
    so every row takes the series and the two arms agree to the bit. ``wide``
    is the case that ships -- an inversion makes the launch decline the gate,
    and the table carries series rows beside a roots row. ``chunked`` is the
    same launch cut into chunks, which is what tells a chunk-local index for
    the cotangent table from a global one. ``streamed`` runs the chunked
    launcher, whose fixed positional list is what a grown kernel signature
    misaligns first. ``washed`` gives the interval a washout, so a pooled row
    has to carry its own attenuation rather than one. ``shimmed`` drives a
    transmit row per shim, which is what tells the three-pool row index from
    the shim row it sits beside. ``profiled`` turns a shaped pulse through its
    own table, which is the one launch that answers whether a table is read
    separately from how many knots it holds.
    ``one_pool`` and ``two_pools`` reach the pool models the table cases
    never do, against the packed reference and against the
    forward-over-reverse pass. ``real`` reaches the real-subspace kernels,
    which carry three real planes where every other case here carries four
    components -- against the reference, against the complex adjoint, and
    with the gradients the representation cannot hold held to exactly zero.
    ``real_shimmed`` gives those kernels a transmit row per shim and leaves
    the last row undriven, so a layout that is merely wide enough cannot pass
    for one that reads the index.
    """
    host_kernels.run(case)
