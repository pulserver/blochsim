"""
=======================
3. Custom signal model
=======================

In lessons 1 and 2 you simulated sequences that ship with blochsim. In this
lesson you write one yourself, saturation recovery, one piece at a time, and
check after each piece what blochsim does with it. By the end your sequence
does everything a shipped one does: it runs over many voxels at once, has
derivatives with respect to tissue and to the sequence, and takes any of the
physics of lesson 2.

**Learning objectives**

- Choose the handlers: which operator plays each kind of event.
- Write the layout: the events of one repetition, in order.
- Check your sequence against its closed form.
- Look at the events your layout produces.
- Differentiate it, with respect to the tissue and to the sequence.

Previous: :doc:`02-advanced-physics`. Next:
:doc:`04-description-based-simulation`, where the events come from a Pulseq
file or a scanner instead of from your layout.
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
# Saturation recovery measures T1 in three steps. A 90-degree pulse followed by
# a spoiler gradient destroys all the magnetization. During a wait, the
# saturation time TS, the longitudinal magnetization recovers towards its
# equilibrium. A small flip angle then reads how much has come back. Repeat
# for several saturation times and the signal traces out
#
# .. math::
#
#    S(T_S) = M_0 \sin\alpha \, \left(1 - e^{-T_S/T_1}\right).
#
# The closed form is what you check your simulation against at the end.
#
# The handlers
# ------------
#
# A sequence is made of events: pulses, readouts and waits. The *handlers* say
# what each kind of event does to the magnetization. They are class attributes
# of a :class:`~blochsim.model.Simulator`, one per kind of event:
# ``excitation``, ``refocusing``, ``inversion``, ``saturation``, ``readout``
# and ``delay`` (a wait: free precession, no RF and no ADC). One you do not set
# keeps its default.
#
# Saturation recovery needs four. Its saturation is a 90-degree pulse followed
# by a spoiler, and ``@`` composes the two into one operator. Its readout is
# :func:`~blochsim.SPGRReadout`, which records a sample and then spoils, so
# nothing is left in the transverse plane for the next block:
import torch

from blochsim import Delay, Excitation, SPGRReadout, Spoil
from blochsim.model import Simulator


def Saturate(flip_rad=torch.pi / 2, phase_rad=0.0):
    """A 90-degree pulse, then a spoiler."""
    return Excitation(flip_rad, phase_rad) @ Spoil()


class SaturationRecovery(Simulator):
    saturation = Saturate  # plays a saturation pulse
    excitation = Excitation  # plays an excitation pulse
    delay = Delay  # plays a wait
    readout = SPGRReadout  # plays a readout, then spoils
    states = 1


# %%
# ``states = 1`` keeps a single configuration state. Every block starts by
# spoiling, so no echo pathway survives from one block to the next and one
# state is the whole answer.
#
# The handlers matter beyond your own sequence. A Pulseq file or a scanner
# stream carries pulses, ADC windows and their timing, but no gradients. When
# blochsim plays one (lesson 4), your handlers decide what happens between the
# events: whether a readout spoils, winds the states on, or rewinds them.
#
# The layout
# ----------
#
# The *layout* is the list of events of one repetition, in the order they are
# played, written with the handlers through ``self.operators``. It is your
# sequence's default description: the constructor calls it with the
# parameters you pass. You never write timestamps; each operator holds the
# timeline for as long as it lasts.
#
# The parameters are the keyword arguments of ``layout``. Here they are the
# saturation times in milliseconds and the readout flip angle in degrees;
# operators take seconds and radians.


class SaturationRecovery(SaturationRecovery):
    def layout(self, *, TS, flip):
        angle = torch.deg2rad(torch.as_tensor(flip))
        parts = []
        for wait in torch.as_tensor(TS) * 1e-3:  # ms to s
            parts += [self.operators.saturation(), self.operators.delay(wait)]
            parts += [self.operators.excitation(angle), self.operators.readout()]
        return parts


# %%
# Run it
# ------
#
# Your simulator now works like a shipped one. Construct it with the sequence
# parameters, call it with the tissue:
TS = torch.logspace(1.3, 3.7, 40)  # 20 ms to 5 s
T1 = torch.tensor([830.0, 1330.0, 4000.0])

recovery = SaturationRecovery(TS=TS, flip=10.0)
signal = recovery.simulate(T1=T1, T2=torch.tensor([80.0, 110.0, 2000.0]))

closed_form = torch.sin(torch.deg2rad(torch.tensor(10.0))) * (
    1 - torch.exp(-TS / T1[:, None])
)
print(f"largest difference: {(signal.abs() - closed_form).abs().max():.1e}")

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.42 * PAGE_WIDTH))
for row, name in enumerate(TISSUES):
    ax.plot(TS, signal[row].abs(), "o", ms=4, color=SERIES[row], label=name)
    ax.plot(TS, closed_form[row], color=SERIES[row], lw=1)
ax.plot([], [], color=MUTED, lw=1, label="closed form")
ax.set(xscale="log", xlabel="TS (ms)", ylabel="|signal| (a.u.)", ylim=(0, None))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# The simulation lands on the closed form. blochsim was never told what the
# events add up to; it played them one by one, so the agreement checks your
# layout.
#
# Look at the events
# ------------------
#
# ``describe`` returns the events your layout produced, with their
# timestamps, as a :class:`~blochsim.SequenceDescription`. Its ``plot`` draws
# them. When a simulation looks wrong, this is the first place to look:
description = recovery.describe(TS=[50.0, 100.0, 200.0], flip=10.0)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 0.35 * PAGE_WIDTH))
description.plot(ax)
ax.set(ylim=(-8, 100))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# Three blocks, each a saturation pulse, a wait that grows from block to
# block, a 10-degree excitation and a readout. This description is the same
# kind of object lesson 4 reads from a Pulseq file or a scanner.
#
# Derivatives
# -----------
#
# Everything lesson 1 did with a shipped simulator works on yours without
# another line. ``jacobian`` gives the derivative with respect to tissue, which
# is what a T1 fit needs:
signal, dT1 = recovery.jacobian("T1", T1=T1, T2=80.0)

# %%
# The sequence parameters are PyTorch tensors, so the derivative with respect
# to the sequence is PyTorch's own ``backward``. Here is how the total signal
# changes with each saturation time:
TS_designed = TS.clone().requires_grad_(True)
total = recovery.simulate(TS=TS_designed, T1=830.0, T2=80.0).abs().sum()
total.backward()

# sphinx_gallery_start_ignore
analytic = torch.sin(torch.deg2rad(torch.tensor(10.0))) * torch.exp(-TS / 830.0) / 830.0
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
ax.plot(TS, TS_designed.grad, "o", ms=4, color=SERIES[0], label="backward()")
ax.plot(TS, analytic, color=MUTED, lw=1, label="closed form")
ax.set(xscale="log", xlabel="TS (ms)", ylabel="∂signal / ∂TS (1/ms)")
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# This is what sequence design descends: choose the saturation times, the flip
# angles or the timings that make a cost smallest. The sequence design
# Applications do it for real protocols.
#
# More physics
# ------------
#
# The properties of lesson 2 are tissue properties, so your sequence takes them
# as they are. A transmit field 20% low turns the saturation pulse into a
# 72-degree pulse, which leaves part of the magnetization behind:
low_b1 = recovery.simulate(T1=830.0, T2=80.0, B1=0.8)
ideal = recovery.simulate(T1=830.0, T2=80.0)

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH))
ax.plot(TS, ideal.abs(), color=SERIES[0], label="B1 = 1")
ax.plot(TS, low_b1.abs(), color=SERIES[1], label="B1 = 0.8")
ax.set(xscale="log", xlabel="TS (ms)", ylabel="|signal| (a.u.)", ylim=(0, None))
legend_outside(ax)
plt.show()
# sphinx_gallery_end_ignore

# %%
# At short saturation times the low-B1 signal is higher, because the
# magnetization the saturation missed adds to what recovered. A T1 fit that
# ignores B1 reads that as a shorter T1.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a blochsim Simulator for saturation recovery. Handlers: a saturation
#    that is a 90-degree pulse followed by a spoiler, an ideal excitation, a
#    delay, and a spoiled readout; keep one configuration state. The layout
#    takes the saturation times TS (ms) and the readout flip angle (deg), and
#    plays saturate, wait TS, excite, read for each TS. Check it against
#    M0 sin(flip) (1 - exp(-TS/T1)) in white matter, grey matter and CSF.
