# Refocusing trains

```{eval-rst}
.. currentmodule:: blochsim.trains
```

Refocusing flip angles designed from the echo signal they are to give. The
functions take and return NumPy arrays in degrees and ms, run on the host in
double precision, and do not import PyTorch: `import blochsim.trains` costs
what NumPy costs.

{func}`prescribed_flips` inverts the extended phase graph of a CPMG train one
echo at a time: the flip angle of each pulse is the smallest whose echo has the
prescribed amplitude, with or without relaxation. {func}`busse_flips` builds
on it the variable-flip-angle trains of long 3D fast spin echo readouts, and
{func}`busse_centre_echo` places their centre echo for a contrast-equivalent
echo time, which {func}`effective_te` computes for any train. A batch of
trains is solved in one call, across threads.

| Object | Description |
| --- | --- |
| {obj}`~blochsim.trains.prescribed_flips` | Flip angles whose echoes take prescribed amplitudes, and the echoes they give. |
| {obj}`~blochsim.trains.effective_te` | Contrast-equivalent echo time of every echo of a train. |
| {obj}`~blochsim.trains.busse_flips` | Variable flip angles with a prescribed lowest, centre and highest angle. |
| {obj}`~blochsim.trains.busse_centre_echo` | Centre echo whose contrast-equivalent echo time is nearest a target. |
