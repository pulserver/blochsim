(general_examples)=

# Examples

The examples have two jobs: a short **Course** teaches the framework itself,
then **Tours** show what can be built on top of it. Every page is executable
and uses the same public interfaces documented in the API reference.

## Course

Read the four **Framework** lessons in order. They are the shortest path from a
shipped sequence to one of your own:

1. **Getting started** — run a shipped {class}`torchsim.model.Simulator`,
   inspect its sequence description, differentiate the signal, and see where
   execution happens.
2. **Expanded physics** — turn on off-resonance, transmit variation,
   diffusion, flow, exchange and magnetization transfer without changing the
   sequence abstraction.
3. **Writing a simulator** — the extension point of the framework:
   subclass {class}`torchsim.model.Simulator`, choose the command handlers
   that interpret an incoming Pulseq/MRD event stream, and implement
   {meth}`torchsim.model.Simulator.layout` for offline construction.
4. **Custom operator** — package a preparation or readout as an operator while
   leaving the state-machine kernels untouched.

A simulator is deliberately the common object in both directions. Offline,
`layout()` builds the sequence description. During acquisition,
`Simulator.from_description()` takes the description decoded from the
scanner's MRD stream and replays the RF/ADC commands through the simulator's
handlers. The physics, differentiation, execution policy, estimators and
reconstruction code therefore see the same object in either case.

## Tours

The remaining sections are standalone applications. They assume the Course,
but not one another.

**Parameter inference** compares dictionary matching, lookup tables, nonlinear
least squares and PERK on the same mapping problem.

**Sequence optimization** differentiates through the simulator to design echo
trains, quantitative schedules and RF pulses.

**Model-based imaging** places the simulator inside the forward model, through
a linear subspace or nonlinear inversion.

**Miscellaneous** contains complete pipelines that do not belong to the linear
course.
