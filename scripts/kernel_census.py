"""Which feature switches the GPU kernels are launched with, and the list to compile.

As a pytest plugin it records, for every launch on a card, the kernel and the
values of its feature switches, and writes them out when the session ends::

    PYTHONPATH=scripts pytest -p kernel_census tests/ --census census.json

Run as a script it turns one or more such records into the entries of
``src/blochsim/_specializations.json``: one per distinct combination of a
kernel's switches, whatever else differed between the launches::

    python scripts/kernel_census.py census.json > src/blochsim/_specializations.json
"""

from __future__ import annotations

import collections
import json
import sys

#: The switches a kernel is compiled for when it is specialized: every argument
#: that only turns terms on and off. Tile sizes are left out; they shape the
#: launch, not the code.
SWITCHES = (
    "single_train",
    "atom_stride",
    "shimmed",
    "profiled",
    "dynamic",
    "broadened",
    "pools",
    "narrow",
    "tabulated",
    "off_axis",
    "moving",
    "diffusing",
    "transmit",
    "density",
    "inverting",
    "recording",
    "directed",
)

#: The kernels specialized. The pooled kernels' tiles are their pools, and the
#: three-pool tables and PERK have no switches worth a kernel of their own.
SPECIALIZED = (
    "_epg_kernel",
    "_epg_jvp_kernel",
    "_epg_vjp_kernel",
    "_epg_vjp_jvp_kernel",
    "_epg_real_kernel",
    "_epg_real_jvp_kernel",
    "_epg_real_vjp_kernel",
    "_epg_real_vjp_jvp_kernel",
)

#: Switches with which a kernel is left to its general build. Fixing them
#: does not shrink the second-order kernel's compile but multiplies it: one such
#: entry took as long to compile as a dozen without, and they are rare.
COSTLY = {
    "_epg_vjp_jvp_kernel": (
        "profiled",
        "moving",
        "pools",
        "dynamic",
        "broadened",
        "narrow",
        "tabulated",
    ),
}

_seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--census", default="census.json", help="where to write the census"
    )


def pytest_configure(config) -> None:
    from blochsim import _gpu_launch

    original = _gpu_launch.Kernel.launch

    def recorded(self, grid, args, kwargs):
        names, _ = _gpu_launch._signature(self.name)
        values = dict(
            zip(
                names,
                list(args) + [kwargs.get(n) for n in names[len(args) :]],
                strict=True,
            )
        )
        fixed = {n: int(values[n]) for n in SWITCHES if n in values}
        _seen[self.name][json.dumps(fixed, sort_keys=True)] += 1
        return original(self, grid, args, kwargs)

    _gpu_launch.Kernel.launch = recorded


def pytest_unconfigure(config) -> None:
    census = {name: dict(counts) for name, counts in sorted(_seen.items())}
    with open(config.getoption("--census"), "w") as stream:
        json.dump(census, stream, indent=1, sort_keys=True)


def entries(*censuses: dict) -> list[dict]:
    """One entry per distinct combination of a specialized kernel's switches.

    A combination turning on a switch :data:`COSTLY` names for its kernel is
    left out.
    """
    combinations = {
        (name, key)
        for census in censuses
        for name, counts in census.items()
        if name in SPECIALIZED
        for key in counts
        if not any(json.loads(key).get(switch) for switch in COSTLY.get(name, ()))
    }
    return [
        {"kernel": name, "fixed": json.loads(key)} for name, key in sorted(combinations)
    ]


if __name__ == "__main__":
    loaded = [json.load(open(path)) for path in sys.argv[1:]]
    json.dump({"specializations": entries(*loaded)}, sys.stdout, indent=1)
    sys.stdout.write("\n")
