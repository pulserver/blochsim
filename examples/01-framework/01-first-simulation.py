"""
========================
1. Your first simulation
========================

blochsim computes the signal an MR sequence produces in a tissue, and how
that signal changes when the tissue or the sequence changes. In this lesson
you simulate a fast spin echo (FSE) that ships with blochsim: first in three
tissues at once, then with a different refocusing angle, then its derivative
with respect to T2.

**Learning objectives**

- Build a shipped simulator from the sequence parameters and run it over
  several tissues at once.
- Take the derivative of the signal with respect to a tissue property.
- Choose how many configuration states to keep, and where the simulation runs.

Next: :doc:`02-advanced-physics`, where the same sequence meets B1, off
resonance, magnetization transfer, diffusion and flow.
"""

# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install blochsim matplotlib
#    !wget --quiet --no-clobber https://raw.githubusercontent.com/pulserver/blochsim/main/docs/figure_style.py

# sphinx_gallery_start_ignore
import warnings

import matplotlib.pyplot as plt

from figure_style import MUTED, PAGE_WIDTH, SERIES, legend_outside

warnings.filterwarnings("ignore")
TISSUES = ("white matter", "grey matter", "CSF")
# sphinx_gallery_end_ignore

# %%
# Your sequence
# -------------
#
# A fast spin echo excites the magnetization once, then plays a train of
# refocusing pulses, and records one echo between each pair of them. Here the
# train has 48 echoes, 5 ms apart, and the whole train is repeated every 3 s.
#
# A shipped sequence is a *simulator*: a class you construct with the
# sequence parameters. Times are in milliseconds and angles in degrees, as on a
# scanner console. The refocusing angles are a list with one entry per echo,
# so a train whose angles change along the echoes is written the same way.
import torch

from blochsim.simulators import FSESimulator

fse = FSESimulator(ESP=5.0, TR=3000.0)
flip = torch.full((48,), 180.0)

# %%
# Your tissue
# -----------
#
# The tissue is what you pass when you call the simulator: here T1 and T2, in
# milliseconds. A number is one voxel. A tensor is many voxels, and blochsim
# simulates all of them at once, so a whole map costs one call. Here are white
# matter, grey matter and cerebrospinal fluid, which share a T1 of 1 s and
# differ in T2:
T2 = torch.tensor([80.0, 110.0, 2000.0])

signal = fse.simulate(flip=flip, T1=1000.0, T2=T2)
print(signal.shape)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
for row, name in enumerate(TISSUES):
    ax.plot(range(1, 49), signal[row].abs(), color=SERIES[row], label=name)
ax.set(xlabel="echo", ylabel="|signal| (a.u.)", xlim=(1, 48), ylim=(0, None))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The signal has one row per voxel and one column per echo. Each echo train
# decays with its own T2, as it would on a scanner: CSF keeps most of its
# signal, white matter loses it within the train.
#
# Changing the sequence
# ---------------------
#
# A clinical FSE rarely refocuses at 180 degrees, because the power the pulses
# deposit grows with the square of the angle. Lower the refocusing angle to 60
# degrees and simulate again. A low angle opens more echo pathways, so ask
# for more of them to be kept (``states``, explained at the end of the lesson):
signal60 = fse.simulate(flip=torch.full((48,), 60.0), T1=1000.0, T2=T2, states=48)

# sphinx_gallery_start_ignore
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 0.38 * PAGE_WIDTH), sharey=True)
for ax, values, title in (
    (axes[0], signal, "180° refocusing"),
    (axes[1], signal60, "60° refocusing"),
):
    for row, name in enumerate(TISSUES):
        ax.plot(range(1, 49), values[row].abs(), color=SERIES[row], label=name)
    ax.set(title=title, xlabel="echo", xlim=(1, 48), ylim=(0, None))
axes[0].set_ylabel("|signal| (a.u.)")
legend_outside(fig)
plt.show()
# sphinx_gallery_end_ignore

# %%
# A 60-degree pulse refocuses only part of the magnetization. The rest is
# stored along z and comes back as a stimulated echo a few pulses later, so
# the first echoes oscillate and then settle to a slow decay, well above the
# 180-degree train at the end. blochsim follows every one of these echo
# pathways, with the extended phase graph (EPG) formalism: the
# :doc:`EPG explanation </explanations/epg>` says how.
#
# Derivative with respect to the tissue
# -------------------------------------
#
# A T2 map is found by adjusting T2 until the simulated train matches the
# measured one, and that search needs to know how the signal changes with T2.
# ``jacobian`` returns the signal and its derivative with respect to the
# properties you name:
signal, dT2 = fse.jacobian("T2", flip=flip, T1=1000.0, T2=T2)

# %%
# The derivative is exact, not a finite difference: blochsim carries it
# through the simulation alongside the signal. Compare it with a finite
# difference over a 1 ms step:
step = 1.0
finite = (fse.simulate(flip=flip, T1=1000.0, T2=T2 + step) - signal) / step

error = (dT2 - finite).abs().max() / dT2.abs().max()
print(f"largest difference: {error:.1e}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
for row, name in enumerate(TISSUES[:2]):
    ax.plot(range(1, 49), dT2[row].real, color=SERIES[row], label=name)
    ax.plot(range(1, 49), finite[row].real, "o", ms=3, mfc="none", color=SERIES[row])
ax.plot([], [], "o", ms=3, mfc="none", color=MUTED, label="finite difference")
ax.set(xlabel="echo", ylabel="∂signal / ∂T2 (1/ms)", xlim=(1, 48))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The two agree to the size of the step. The derivative is largest in the
# middle of the train: those are the echoes that tell one T2 from another.
# Name several properties, ``jacobian(("T1", "T2"), ...)``, and you get one
# derivative per property.
#
# How exact, and where it runs
# ----------------------------
#
# Two settings decide what a simulation costs. You can give either to the
# constructor or to the call.
#
# ``states`` is how many configuration states (echo pathways) the simulation
# keeps. Each refocusing pulse can open one more pathway, so a train of 48
# echoes needs at most 48. The FSE keeps 10 by default, enough for 180-degree
# pulses, where few pathways carry signal. Too few gives a wrong answer, not a
# less precise one, so check against the full number when you lower the angle:
flip60 = torch.full((48,), 60.0)
reference = fse.simulate(flip=flip60, T1=1000.0, T2=T2, states=48)
for states in (10, 24, 32):
    approximate = fse.simulate(flip=flip60, T1=1000.0, T2=T2, states=states)
    error = (approximate - reference).abs().max() / reference.abs().max()
    print(f"{states:2d} states: {error:.0%} off")

# %%
# The 60-degree train needs all 48. That is why the figure above asked for
# them.
#
# ``blochsim.execution`` says where the simulation runs. The default,
# ``"auto"``, keeps small problems on the CPU and sends large ones to a GPU if
# there is one, in pieces if they do not fit. Naming a device forces it:
import blochsim

with blochsim.execution("cpu"):
    on_cpu = fse.simulate(flip=flip, T1=1000.0, T2=T2)

# %%
# A simulator is worth keeping. The first call works out the structure of the
# sequence, and later calls only change the numbers, so build it once and call
# it in your loop.
#
# For a one-off simulation, every shipped sequence also has a function form,
# which builds the simulator and calls it in one line:
signal, dT2 = blochsim.fse_sim(
    flip=flip, ESP=5.0, TR=3000.0, T1=1000.0, T2=T2, diff="T2"
)

# %%
# As a spec
# ---------
#
# What this lesson did, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    With blochsim, simulate a fast spin echo of 48 echoes, 5 ms apart, TR 3 s,
#    in white matter, grey matter and CSF (T1 1 s; T2 80, 110 and 2000 ms).
#    Compare 180 and 60 degree refocusing trains. Return the derivative of the
#    signal with respect to T2, checked against a finite difference, and check
#    that the number of configuration states kept is enough for the 60 degree
#    train.
