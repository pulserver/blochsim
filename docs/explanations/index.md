# Explanations

The concepts behind BlochSim, separate from the step-by-step examples and the
API reference.

{doc}`signal-model`
: What a signal model is: handlers, layout, and the descriptions that replace
  the layout.

{doc}`description`
: The sequence representation BlochSim consumes: events, RF definitions,
  readout roles, Pulseq input, and the MRD description a running scanner can
  send.

{doc}`epg`
: Extended phase graphs: configuration states, RF transitions, gradient
  shifts, relaxation, diffusion, flow, exchange, and the assumptions behind
  the model.

{doc}`implementation`
: How a {class}`~blochsim.model.Simulator` turns either an offline layout or
  an incoming sequence description into the fused CPU/GPU state machine, and
  how differentiation and execution are arranged.

```{toctree}
:hidden:
:maxdepth: 1

signal-model
description
epg
implementation
```
