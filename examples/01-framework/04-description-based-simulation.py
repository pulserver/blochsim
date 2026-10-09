"""
================================
4. Description-based simulation
================================

In lesson 3 your layout produced the events of the sequence. Often the events
already exist: in a Pulseq file you wrote with another tool, or in the
sequence a scanner is running. In this lesson you simulate those, without
retyping a single sequence parameter.

**Learning objectives**

- Read a sequence description: the events, their timestamps and the RF
  pulses they play.
- Simulate a description with ``from_description``, and see why the
  simulator you call it on matters.
- Simulate a Pulseq file with ``from_pulseq``.

Previous: :doc:`03-custom-signal-model`. Next: :doc:`05-custom-operator`, where
you add a preparation or a readout of your own.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install "blochsim[pulseq]" matplotlib
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/examples/01-framework/fse.seq

# sphinx_gallery_start_ignore
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")
# The gallery runs a script from its own directory; a shell may not.
SEQ_FILE = next(
    candidate
    for candidate in (Path("fse.seq"), Path("examples/01-framework/fse.seq"))
    if candidate.exists()
)
# sphinx_gallery_end_ignore

# %%
# A sequence description
# ----------------------
#
# A *sequence description* is the list of events of one repetition: RF
# pulses, ADC windows and waits, each with its timestamp and its numbers (flip
# angle, phase, which pulse shape). It is a
# :class:`~blochsim.SequenceDescription`, and it is what blochsim actually
# simulates.
#
# Every simulator has a default description: the one its layout builds from
# the parameters you pass. ``describe`` returns it. Here is the fast spin echo
# of lesson 1, with 60-degree refocusing:
import torch

from blochsim.simulators import FSESimulator

fse = FSESimulator(ESP=5.0, TR=3000.0)
flip = torch.full((48,), 60.0)

description = fse.describe(flip=flip, ESP=5.0, TR=3000.0)
print(f"{len(description.events)} events, {len(description.adc_events)} readouts")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 0.35 * PAGE_WIDTH))
description.plot(ax, upto_s=6 * 5e-3)
ax.set(ylim=(-8, 105))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The first 30 ms: the excitation, then each refocusing pulse halfway between
# two readouts. This is all a description holds. In particular it holds no
# gradients: a Pulseq file has them, but the stream a scanner sends does not,
# and blochsim does not need them.
#
# Simulating a description
# ------------------------
#
# ``from_description`` builds a simulator from a description instead of from
# a layout. You give it the tissue and nothing about the sequence:
from_events = FSESimulator.from_description(description, states=48, T1=1000.0, T2=80.0)
signal = from_events.simulate()

# %%
# The layout is skipped, but the *handlers* of the class you call it on are
# not: they play the events, and they decide what happens between them, which
# is where the missing gradients come back. ``FSESimulator`` crushes around
# each refocusing pulse, as an FSE does. Read the same events with
# :class:`~blochsim.simulators.MRFSimulator`, whose readout winds the
# magnetization on by one configuration state after every sample, and you get
# a different sequence:
from blochsim.simulators import MRFSimulator

as_mrf = MRFSimulator.from_description(description, states=48, T1=1000.0, T2=80.0)
wrong = as_mrf.simulate()

# sphinx_gallery_start_ignore
reference = fse.simulate(flip=flip, T1=1000.0, T2=80.0, states=48)
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
echoes = range(1, 49)
ax.plot(
    echoes, reference.abs().reshape(-1), color=MUTED, lw=3, label="FSESimulator, layout"
)
ax.plot(
    echoes, signal.abs().reshape(-1), color=SERIES[0], label="FSE handlers, description"
)
ax.plot(
    echoes, wrong.abs().reshape(-1), color=SERIES[1], label="MRF handlers, description"
)
ax.set(xlabel="echo", ylabel="|signal| (a.u.)", xlim=(1, 48), ylim=(0, None))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# With the FSE handlers the description reproduces the shipped simulator, but
# for a small offset: ``FSESimulator`` also adds the recovery between one
# train and the next, which is not an event and so is not in the description.
# With the MRF handlers the echoes are never refocused and the signal is gone
# within a few echoes. Choosing the class is how you say what kind of sequence
# the events belong to.
#
# A description from a scanner
# ----------------------------
#
# During a scan, pulserver streams the description of the running sequence
# with the raw data. :func:`~blochsim.read_mrd_description` reads it from an
# MRD file, and ``from_description`` simulates it exactly as above. That is
# how a reconstruction can simulate the sequence the scanner actually played.
#
# A Pulseq file
# -------------
#
# ``from_pulseq`` reads the description from a ``.seq`` file. It needs the
# ``pulseq`` extra (``pip install "blochsim[pulseq]"``), which parses the file.
# This file is a short FSE written with another tool:
from_file = FSESimulator.from_pulseq(SEQ_FILE, states=20)
file_signal = from_file.simulate(T1=1000.0, T2=torch.tensor([80.0, 110.0]))

# sphinx_gallery_start_ignore
read = from_file.describe()
echo_ms = torch.tensor([event.timestamp_us for event in read.adc_events]) * 1e-3
echo_ms = echo_ms - echo_ms[0] + (echo_ms[1] - echo_ms[0])
print(f"{len(read.adc_events)} echoes, {echo_ms[1] - echo_ms[0]:.1f} ms apart")
fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 0.65 * PAGE_WIDTH))
read.plot(axes[0])
axes[0].set(ylim=(-8, 200))
legend_outside(axes[0])
for row, name in enumerate(("white matter", "grey matter")):
    axes[1].plot(echo_ms, file_signal[row].abs(), "o-", color=SERIES[row], label=name)
axes[1].set(xlabel="time after excitation (ms)", ylabel="|signal| (a.u.)")
legend_outside(axes[1])
plt.show()
# sphinx_gallery_end_ignore

# %%
# You never gave the echo spacing, the number of echoes, the refocusing angle
# or the pulse shapes: they came from the file. What you chose is the tissue,
# and the class whose handlers play the events.
#
# The file states how many blocks make one repetition (its ``TRSize``
# definition). Each readout's timestamp is placed on the sample where the
# trajectory crosses the centre of k-space, which is when the echo forms.
#
# As a spec
# ---------
#
# What this lesson did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, read the sequence in fse.seq with FSESimulator.from_pulseq
#    and simulate it in white matter and grey matter (T1 1 s; T2 80 and
#    110 ms). Plot the events of one repetition and the signal at each echo
#    time. Separately, take the description of a 48-echo, 60-degree FSE and
#    simulate it with the FSE handlers and with the MRF handlers, to show that
#    the handlers decide what happens between the events.
