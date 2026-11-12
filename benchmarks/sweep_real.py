"""Time MRF-FISP (500 TRs, 32 states) forward and T1/T2 Jacobian per kernel variant.

Run from the blochsim-bench env with a scratch build of the sweep branch:
    python sweep.py --atoms 100000 [--profile]
"""

import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--atoms", type=int, default=100_000)
    cli.add_argument("--length", type=int, default=500)
    cli.add_argument("--states", type=int, default=32)
    cli.add_argument("--repeats", type=int, default=30)
    cli.add_argument("--profile", action="store_true")
    cli.add_argument("--no-sweep", action="store_true")
    args = cli.parse_args()

    from blochsim.simulators import MRFSimulator

    device = torch.device("cuda")
    from _common import flip_train, tissue_grid

    flip = torch.tensor(flip_train(args.length), dtype=torch.float32, device=device)
    t1, t2 = tissue_grid(args.atoms)
    T1 = torch.tensor(t1, dtype=torch.float32, device=device)
    T2 = torch.tensor(t2, dtype=torch.float32, device=device)
    seq = MRFSimulator(flip=flip, TR=10.0, states=args.states).to(device)
    seq.simulate(T1=T1[:1], T2=T2[:1])
    seq.jacobian(("T1", "T2"), T1=T1[:1], T2=T2[:1])

    def timed(fn, repeats=args.repeats):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
        times.sort()
        return times[len(times) // 2]

    fwd = lambda: seq.simulate(T1=T1, T2=T2)  # noqa: E731
    jac = lambda: seq.jacobian(("T1", "T2"), T1=T1, T2=T2)  # noqa: E731
    reference = fwd()
    print(f"forward default: {timed(fwd):.2f} ms", flush=True)
    print(f"jacobian default: {timed(jac, 10):.2f} ms", flush=True)
    print(f"checksum {complex(reference.sum().cpu())}", flush=True)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        for name, fn in (("forward", fwd), ("jacobian", jac)):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
            print(f"== {name} (5 calls)")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    if args.no_sweep:
        return
    for y, w, occ in itertools.product((2, 4, 8), (2, 4, 8), (0, 16, 24, 32)):
        os.environ["BLOCHSIM_REAL_SWEEP"] = f"{y},{w},{occ}"
        try:
            out = fwd()
            err = (out - reference).abs().max().item()
            print(f"fwd Y={y} W={w} occ={occ}: {timed(fwd):.2f} ms  maxerr {err:.1e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"fwd Y={y} W={w} occ={occ}: failed {e}", flush=True)
    os.environ.pop("BLOCHSIM_REAL_SWEEP")
    for y, w, occ in itertools.product((1, 2, 4), (2, 4, 8), (0, 16, 24, 32)):
        os.environ["BLOCHSIM_REAL_JVP_SWEEP"] = f"{y},{w},{occ}"
        try:
            print(f"jvp Y={y} W={w} occ={occ}: jacobian {timed(jac, 10):.2f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"jvp Y={y} W={w} occ={occ}: failed {e}", flush=True)


if __name__ == "__main__":
    main()
