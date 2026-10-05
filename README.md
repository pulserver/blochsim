# TorchSim

TorchSim is a differentiable MR signal simulator built on PyTorch, with
closed-form signal models and a fused extended-phase-graph (EPG) state machine
for pulse trains.

[![codecov](https://codecov.io/gh/pulserver/torchsim/graph/badge.svg?token=l8xhIVORYm)](https://codecov.io/gh/pulserver/torchsim)
[![Tests](https://github.com/pulserver/torchsim/actions/workflows/test.yml/badge.svg)](https://github.com/pulserver/torchsim/actions/workflows/test.yml)
[![Lint](https://github.com/pulserver/torchsim/actions/workflows/lint.yml/badge.svg)](https://github.com/pulserver/torchsim/actions/workflows/lint.yml)
[![License](https://img.shields.io/github/license/pulserver/torchsim)](https://github.com/pulserver/torchsim/blob/main/LICENSE.txt)
[![Documentation](https://github.com/pulserver/torchsim/actions/workflows/docs.yml/badge.svg)](https://pulserver.github.io/torchsim/)
[![PyPi](https://img.shields.io/pypi/v/torchsim)](https://pypi.org/project/torchsim)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PythonVersion](https://img.shields.io/badge/Python-%3E=3.10-blue?logo=python&logoColor=white)](https://python.org)

## What it provides

- Vectorized signal simulation over voxels/atoms on CPU and NVIDIA GPU.
- Forward-mode Jacobians with respect to tissue properties and reverse-mode
  differentiation with respect to sequence parameters.
- Closed-form models and an EPG state machine with relaxation,
  off-resonance, diffusion, flow, transmit variation, magnetization transfer
  and chemical exchange.
- Parameter inference, model-based reconstruction and sequence-design tools
  written against the same simulator interface.
- Pulseq and MRD sequence-description input, so the same sequence model can be
  used offline or driven from a scanner stream.

## Installation

Install the PyTorch build appropriate for your machine first, then TorchSim:

```bash
pip install torchsim
```

See the [User Guide](https://pulserver.github.io/torchsim/latest/user_guide.html)
for CPU, CUDA, macOS and source-build details.

## Basic usage

The central public object is a `Simulator`. A shipped simulator fixes the
sequence; `simulate` and `jacobian` evaluate it over the tissue you pass:

```python
import numpy as np
from torchsim.simulators import MRFSimulator

flip = np.concatenate(
    (np.linspace(5.0, 60.0, 300), np.linspace(60.0, 2.0, 300), np.full(280, 2.0))
)
sequence = MRFSimulator(flip=flip, TR=10.0)

signal, jacobian = sequence.jacobian(
    ("T1", "T2"),
    T1=1000.0,
    T2=100.0,
)
```

Functional helpers such as `torchsim.mrf_sim(...)` remain convenient for
one-off calls. The class interface is the canonical one for reusable models,
parameter estimation, reconstruction, optimization, Pulseq input and scanner
descriptions.

## Implementing a sequence

Subclass `torchsim.model.Simulator`. For a state-machine sequence you define:

1. the event handlers that say how excitation, refocusing, inversion,
   saturation, readout and delay commands are interpreted; and
2. `layout()`, which returns those operators in order for offline use.

An incoming Pulseq/MRD description already supplies the layout, so
`Simulator.from_description()` replays its commands through the same handlers.
That gives offline design and scanner-driven simulation one public sequence
abstraction.

The executable
[Framework course](https://pulserver.github.io/torchsim/latest/generated/autoexamples/01-framework/index.html)
walks through the complete pattern.

## Development

```bash
git clone git@github.com:pulserver/torchsim
cd torchsim
pip install -e ".[dev]"
pre-commit install
```

The install compiles the two C++ kernels, so it needs a C++17 compiler. CMake
and Ninja arrive as build-time dependencies. `pre-commit` runs the same Ruff
format/lint checks as CI.

## Related projects

The documentation's
[Related projects](https://pulserver.github.io/torchsim/latest/misc/related.html)
page places TorchSim among other MR simulators and links to the relevant
packages and literature.
