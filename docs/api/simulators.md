# Simulators

The sequences that ship with TorchSim. A constructor takes the keywords
{meth}`~torchsim.model.Simulator.simulate` takes and fixes them, so a
sequence and the tissue it is being asked about are written down together and
what is left to give at the call is whatever is actually varying.

## Closed form

Signals that have an analytic expression, evaluated in one pass.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.simulators.bSSFPSimulator
   torchsim.simulators.SPGRSimulator
   torchsim.simulators.MP2RAGESimulator
   torchsim.simulators.InversionRecoverySimulator
   torchsim.simulators.MultiEchoSimulator
   torchsim.simulators.DoubleAngleSimulator
   torchsim.simulators.SpinEchoSimulator
   torchsim.simulators.LookLockerSimulator
   torchsim.simulators.MOLLISimulator
   torchsim.simulators.IRbSSFPSimulator
   torchsim.simulators.HSFPSimulator
   torchsim.simulators.MultiGradientEchoSimulator
   torchsim.simulators.IRMultiGradientEchoSimulator
   torchsim.simulators.DiffusionSimulator
   torchsim.simulators.LorentzianSimulator
   torchsim.simulators.ASLSimulator
```

## State machine

Trains that have to be played out, run on the extended phase graph engine.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.simulators.FSESimulator
   torchsim.simulators.HyperechoSimulator
   torchsim.simulators.StimulatedEchoSimulator
   torchsim.simulators.MPRAGESimulator
   torchsim.simulators.MPnRAGESimulator
   torchsim.simulators.MRFSimulator
   torchsim.simulators.FLASHSimulator
   torchsim.simulators.TrueFISPSimulator
   torchsim.simulators.fmSSFPSimulator
   torchsim.simulators.CESTSimulator
```

## Functional wrappers

The `*_sim` functions are convenience calls for a subset of the shipped
simulators: protocol and tissue go into one function call, with an optional
Jacobian. They are not a second extension API. New sequence families are
implemented as {class}`~torchsim.model.Simulator` subclasses, so they can use
both an offline {meth}`~torchsim.model.Simulator.layout` and the same handlers
when a Pulseq/MRD description arrives from a scanner.

Prefer the simulator classes whenever the same sequence is reused, bound to a
protocol, passed to an estimator/reconstruction/design object, or constructed
from a description. The wrappers remain useful for compact one-off calls.

### Analytical

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.bssfp_sim
   torchsim.spgr_sim
```

### Iterative

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   torchsim.fse_sim
   torchsim.mprage_sim
   torchsim.mp2rage_sim
   torchsim.mpnrage_sim
   torchsim.mrf_sim
```
