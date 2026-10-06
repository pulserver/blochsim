# Simulators

The sequences that ship with BlochSim. A constructor takes the keywords
{meth}`~blochsim.model.Simulator.simulate` takes and fixes them, so a
sequence and the tissue it is being asked about are written down together and
what is left to give at the call is whatever is actually varying.

## Closed form

Signals that have an analytic expression, evaluated in one pass.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.simulators.bSSFPSimulator
   blochsim.simulators.SPGRSimulator
   blochsim.simulators.MP2RAGESimulator
   blochsim.simulators.InversionRecoverySimulator
   blochsim.simulators.MultiEchoSimulator
   blochsim.simulators.DoubleAngleSimulator
   blochsim.simulators.SpinEchoSimulator
   blochsim.simulators.LookLockerSimulator
   blochsim.simulators.MOLLISimulator
   blochsim.simulators.IRbSSFPSimulator
   blochsim.simulators.HSFPSimulator
   blochsim.simulators.MultiGradientEchoSimulator
   blochsim.simulators.IRMultiGradientEchoSimulator
   blochsim.simulators.DiffusionSimulator
   blochsim.simulators.LorentzianSimulator
   blochsim.simulators.ASLSimulator
```

## State machine

Trains that have to be played out, run on the extended phase graph engine.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.simulators.FSESimulator
   blochsim.simulators.HyperechoSimulator
   blochsim.simulators.StimulatedEchoSimulator
   blochsim.simulators.MPRAGESimulator
   blochsim.simulators.MPnRAGESimulator
   blochsim.simulators.MRFSimulator
   blochsim.simulators.FLASHSimulator
   blochsim.simulators.TrueFISPSimulator
   blochsim.simulators.fmSSFPSimulator
   blochsim.simulators.CESTSimulator
```

## Functional wrappers

The `*_sim` functions are convenience calls for a subset of the shipped
simulators: protocol and tissue go into one function call, with an optional
Jacobian. They are not a second extension API. New sequence families are
implemented as {class}`~blochsim.model.Simulator` subclasses, so they can use
both an offline {meth}`~blochsim.model.Simulator.layout` and the same handlers
when a Pulseq/MRD description arrives from a scanner.

Prefer the simulator classes whenever the same sequence is reused, bound to a
protocol, passed to an estimator/reconstruction/design object, or constructed
from a description. The wrappers remain useful for compact one-off calls.

### Analytical

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.bssfp_sim
   blochsim.spgr_sim
```

### Iterative

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   blochsim.fse_sim
   blochsim.mprage_sim
   blochsim.mp2rage_sim
   blochsim.mpnrage_sim
   blochsim.mrf_sim
```
