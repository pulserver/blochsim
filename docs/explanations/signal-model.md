# Signal model

```{admonition} TL;DR
:class: tldr

- A signal model is a {class}`~blochsim.model.Simulator` subclass. It names
  a *handler* for each kind of event and a *layout*, the events of one
  repetition in playing order.
- The layout is the default sequence description, built by the constructor
  from the sequence parameters. `from_pulseq` and `from_description` replace
  it with a description read from a file or a scanner stream.
- The handlers play every event, whatever the description came from. They
  also define the dephasing between events, which a description does not
  carry.
- Everything else, including batching, derivatives, devices, parameter
  estimation and sequence design, is written against this interface.
```

You give blochsim a sequence and a tissue, and it returns the signal and its
derivatives. The sequence is written once, as a signal model: which operator
plays each kind of event, and in what order the events of one repetition are
played. The 26 simulators that ship with blochsim are written this way, and so
is any sequence you add.

A signal model separates the sequence from the physics. The sequence is a
list of events with timestamps; the physics is the set of tissue properties a
voxel carries, from T1 and T2 to B1, off resonance, a second pool, diffusion
and flow. Either can change without the other, and a model pays only for the
physics a call asks for.

## What blochsim does

:::{container} capabilities

- **Builds a sequence from a subclass that names a handler per kind of event and implements `layout`.** The handlers are `excitation`, `refocusing`, `inversion`, `saturation`, `readout` and `delay`; one left unnamed plays the plain operator.

  Code: {class}`~blochsim.model.Simulator`, {class}`~blochsim.model.EventOperators`. Tests: *a protocol with no layout says so* (`test_state_machine.py`); *a trigger table defaults to the bare operators* (`test_state_machine.py`).
- **Composes operators with `@`, each starting where the previous one ends.** A handler can be a function that composes shipped operators, as a saturation pulse followed by a spoiler.

  Code: `Operator.__matmul__`. Test: *a bare operator follows the one before it* (`test_operators.py`).
- **Takes the sequence parameters at construction, and lets a call override them.**

  Code: {meth}`~blochsim.model.Simulator.bind`. Test: *the protocol may be overridden per call* (`test_state_machine.py`).
- **Returns the sequence description the layout produced, with the timestamp of every event.**

  Code: {meth}`~blochsim.model.Simulator.describe`. Test: *the repetition lasts as long as the operators do* (`test_description_helper.py`).
- **Simulates a description read from a Pulseq file or an MRD stream, skipping the layout.**

  Code: {meth}`~blochsim.model.Simulator.from_pulseq`, {meth}`~blochsim.model.Simulator.from_description`. Tests: *a description handed over whole skips the layout* (`test_state_machine.py`); *the steady state is the spoiled gradient echo closed form* (`test_pulseq_reader.py`).
- **Plays a description through the handlers of the class it is given to.** The same events give a different signal as a refocused train and as an unbalanced one.

  Code: {meth}`~blochsim.model.Simulator.from_description`. Test: *which simulator reads a stream is what decides the dephasing* (`test_state_machine.py`).
- **Carries a tissue property into the kernels only when a call gives it as a map.** A property left out, or given as its default scalar, costs nothing.

  Code: {class}`~blochsim.model.SpinPhysics`. Tests: *a property the model does not declare stays out of the kernel* (`test_signal_model.py`); *a declared property given a map reaches the kernel* (`test_signal_model.py`).
- **Accepts a closed form in place of a layout.** A simulator that implements `evaluate` never builds events, and gets the same batching and derivatives.

  Code: {meth}`~blochsim.model.Simulator.evaluate`. Test: *a closed form travels the same way* (`test_array_backends.py`).
- **Returns the signal in the array library it was given: NumPy, CuPy or PyTorch.**

  Code: {meth}`~blochsim.model.Simulator.simulate`. Test: *the signal comes back in the library it was asked in* (`test_array_backends.py`).

:::

## What blochsim leaves out

- **Gradient waveforms and spatial encoding.** An event says that a gradient
  dephases by a whole configuration order, not by how much. Waveforms,
  k-space trajectories and their timing belong to the sequence design tools,
  such as pypulseqpp.
- **The reconstruction encoding.** A model-based reconstruction composes a
  blochsim signal model with an encoding operator from a reconstruction
  library, such as mri-nufft or bartorch.
- **Safety.** A signal model does not check gradient, PNS or SAR limits.

## How it works

```{figure} /generated/figures/signal_model.png
:width: 100%
:alt: Three sources of a description, the description, the handlers, and the signal.

A description comes from the layout, from a Pulseq file or from an MRD stream.
The handlers of the class play each of its events, and the kernels return the
signal and its derivatives.
```

### Handlers

A sequence is a series of RF pulses, ADC windows and waits. A handler is the
operator that plays one kind of event: whether an excitation is an ideal
rotation or a shaped pulse, and whether a readout is followed by an unbalanced
gradient, by ideal spoiling, or by a rewinder. The handlers are class
attributes:

```python
class SaturationRecovery(Simulator):
    saturation = Saturate
    excitation = Excitation
    delay = Delay
    readout = SPGRReadout
```

The shipped tables `SPOILED`, `UNBALANCED`, `BALANCED` and `REFOCUSED` in
{mod}`blochsim.model` assign the readout and refocusing handlers
of the four sequence families.

### Layout

The layout returns the operators of one repetition in playing order, built
from the handlers through `self.operators`. Its keyword arguments are the
sequence parameters, in the public units: milliseconds and degrees. The
operators take seconds and radians. The first call builds the description
from the layout; later calls rebind the numbers onto the same structure.

### Descriptions from elsewhere

`from_pulseq` reads the RF pulses, ADC windows and timing of a `.seq` file,
and `from_description` takes a {class}`~blochsim.SequenceDescription` from
any source, such as the MRD stream pulserver sends during an acquisition.
Neither carries gradients. The handlers of the class supply the dephasing, so
the class you read a description with is part of the model.
{doc}`description` describes what is read and how.

### Tissue

A voxel carries the properties the model declares, such as T1 and T2. B1,
off resonance, a second pool, diffusion and flow enter the kernels
only when a call passes them as maps. {doc}`epg` describes the physics of each.

## See it run

- {doc}`../generated/autoexamples/01-framework/01-first-simulation`: a shipped
  simulator, batched over three tissues.
- {doc}`../generated/autoexamples/01-framework/03-custom-signal-model`: a new
  signal model, written and checked against its closed form.
- {doc}`../generated/autoexamples/01-framework/04-description-based-simulation`:
  the same events read with two sets of handlers, and a Pulseq file.
- {doc}`../generated/autoexamples/01-framework/05-custom-operator`: new
  operators composed with `@`.
- {doc}`../api/model`: {class}`~blochsim.model.Simulator`,
  {class}`~blochsim.model.EventOperators` and
  {class}`~blochsim.model.SpinPhysics`.
