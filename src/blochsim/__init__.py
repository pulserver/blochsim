"""Main BlochSim API.

The simulators, the estimators and the design tools are built on PyTorch,
and importing it takes far longer than anything :mod:`blochsim.trains` does.
So ``import blochsim`` imports :mod:`blochsim.trains` alone, and the rest of
the package the first time anything else is asked of it.
"""

from __future__ import annotations

import threading
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from . import trains  # noqa: F401

if TYPE_CHECKING:
    from . import (  # noqa: F401
        _functional,
        estimators,
        optim,
        sequence,
        simulators,
    )
    from ._functional import *  # noqa: F403
    from ._model import Simulator  # noqa: F401
    from ._operator import ModelOperator  # noqa: F401
    from ._subspace import Subspace  # noqa: F401
    from .sequence import *  # noqa: F403

#: Set on the thread that is importing the rest of the package, while it does.
_loading = threading.local()
_loaded = False


def _load() -> None:
    """Import everything built on PyTorch, and name it here.

    No lock is held: the import system serializes each module's import and
    detects a cycle between threads, which a lock of this module's own would
    hide from it. Two threads arriving together both run this, to the same
    end.
    """
    global _loaded
    _loading.here = True
    try:
        from . import _functional, estimators, optim, sequence, simulators
        from ._model import Simulator
        from ._operator import ModelOperator
        from ._subspace import Subspace

        namespace = globals()
        namespace.update(
            estimators=estimators,
            optim=optim,
            sequence=sequence,
            simulators=simulators,
            _functional=_functional,
            Simulator=Simulator,
            ModelOperator=ModelOperator,
            Subspace=Subspace,
        )
        names = ["ModelOperator", "Simulator", "Subspace"]
        for module in (sequence, _functional):
            for name in module.__all__:
                namespace[name] = getattr(module, name)
            names.extend(module.__all__)
        namespace["__all__"] = names
        _loaded = True
    finally:
        _loading.here = False


def __getattr__(name: str) -> Any:
    missing = AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name.startswith("__") and name != "__all__":
        raise missing
    # A submodule that imports a sibling while the package loads is asking
    # the import system, and an AttributeError is what sends it there.
    if _loaded or getattr(_loading, "here", False):
        raise missing
    if name != "__all__":
        # A submodule is the import system's to import.
        if find_spec(f"{__name__}.{name}") is not None:
            return import_module(f"{__name__}.{name}")
        # Everything the package exports is public. A private name that is
        # no submodule is an optional module that was not built, asked for
        # by ``from blochsim import``: it is missing, and loading the
        # package from inside another module's import would not find it.
        if name.startswith("_"):
            raise missing
    _load()
    try:
        return globals()[name]
    except KeyError:
        raise missing from None


def __dir__() -> list[str]:
    _load()
    return sorted(globals())
