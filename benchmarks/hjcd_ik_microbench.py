"""
hjcd_ik_microbench — HJCD-IK in-process latency stability benchmark.

Measures whether per-call HJCD-IK latency is stable across repeated invocations
in a single process.  Healthy behaviour: call 1 may be warm (after --warmup-calls)
but calls 2..N should remain within 2x of call 1.

Usage::

    python -m benchmarks.hjcd_ik_microbench
    python -m benchmarks.hjcd_ik_microbench --trials 20 --batch-size 128 --num-solutions 8
    python -m benchmarks.hjcd_ik_microbench --trials 20 --batch-size 128 --vary-seed
    python -m benchmarks.hjcd_ik_microbench --simulate-trial-gap 30  # sleep between calls

The --simulate-trial-gap flag sleeps N seconds between calls to mimic the
GPU-idle time that occurs during MuJoCo simulation in eval_baselines.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.solver.ds.factory import warmup_hjcdik
from src.solver.ik.hjcd_wrapper import solve_batch

_DUMMY_TARGET = {"position": [0.5, 0.0, 0.5], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}


def _run(args) -> None:
    print(f"HJCD-IK microbench  batch={args.batch_size}  solutions={args.num_solutions}  "
          f"trials={args.trials}  warmup_calls={args.warmup_calls}")
    if args.simulate_trial_gap > 0:
        print(f"  simulate_trial_gap={args.simulate_trial_gap}s (sleep between calls)")
    print()

    # One-time cold warmup to absorb CUDA context init
    print(f"[cold warmup] running ...", end=" ", flush=True)
    t0 = time.perf_counter()
    warmup_hjcdik(batch_size=args.batch_size, num_solutions=args.num_solutions)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"{cold_ms:.0f}ms  ← expected large (CUDA context init)")

    # Additional warmup calls if requested
    for i in range(args.warmup_calls - 1):
        print(f"[warmup {i+2}/{args.warmup_calls}] ...", end=" ", flush=True)
        t0 = time.perf_counter()
        warmup_hjcdik(batch_size=args.batch_size, num_solutions=args.num_solutions)
        print(f"{(time.perf_counter()-t0)*1000:.0f}ms")

    print()
    print(f"{'Call':>6}  {'Seed':>6}  {'total_ms':>9}  {'n_sol':>5}  {'status'}")
    print("-" * 55)

    latencies = []
    for i in range(args.trials):
        seed = args.seed + i if args.vary_seed else args.seed
        target = _DUMMY_TARGET

        if args.simulate_trial_gap > 0 and i > 0:
            print(f"         sleeping {args.simulate_trial_gap}s to simulate GPU idle ...", flush=True)
            time.sleep(args.simulate_trial_gap)
            if args.rewarm_after_gap:
                if args.deep_rewarm:
                    # Tight loop until a single call settles < 15ms (GPU at P0)
                    t_rw = time.perf_counter()
                    n_rw = 0
                    while True:
                        t_c = time.perf_counter()
                        warmup_hjcdik(batch_size=args.batch_size, num_solutions=1)
                        c_ms = (time.perf_counter() - t_c) * 1000.0
                        elapsed = (time.perf_counter() - t_rw) * 1000.0
                        n_rw += 1
                        if c_ms < 8.0 or elapsed >= 1500.0:
                            break
                    rw_ms = (time.perf_counter() - t_rw) * 1000.0
                    print(f"         deep_rewarm: {rw_ms:.0f}ms ({n_rw} calls, last={c_ms:.1f}ms)")
                else:
                    t_rw = time.perf_counter()
                    warmup_hjcdik(batch_size=args.batch_size, num_solutions=1)
                    rw_ms = (time.perf_counter() - t_rw) * 1000.0
                    print(f"         rewarm: {rw_ms:.0f}ms")

        t0 = time.perf_counter()
        try:
            result = solve_batch(
                target_pose=target,
                batch_size=args.batch_size,
                num_solutions=args.num_solutions,
            )
            n_sol = len(result.solutions)
            status = "ok"
        except Exception as exc:
            n_sol = 0
            status = f"FAIL:{exc}"
        total_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(total_ms)

        flag = ""
        if i > 0 and latencies[0] > 0 and total_ms > latencies[0] * 3.0:
            flag = "  ← SLOW (>3x call-1)"
        print(f"{i+1:>6}  {seed:>6}  {total_ms:>9.1f}  {n_sol:>5}  {status}{flag}")

    print()
    if latencies:
        print(f"Summary:")
        print(f"  call 1 (post-warmup) : {latencies[0]:.1f} ms")
        if len(latencies) > 1:
            tail = latencies[1:]
            print(f"  calls 2..{args.trials}           : min={min(tail):.1f}  max={max(tail):.1f}  "
                  f"mean={sum(tail)/len(tail):.1f} ms")
            ratio = max(tail) / latencies[0] if latencies[0] > 0 else float("inf")
            verdict = "PASS" if ratio < 3.0 else "FAIL — latency degrades across trials"
            print(f"  max/call-1 ratio     : {ratio:.2f}x  →  {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials",        type=int,   default=20,
                        help="Number of timed IK calls (default: 20)")
    parser.add_argument("--batch-size",    type=int,   default=128,
                        help="HJCD-IK batch_size per call (default: 128)")
    parser.add_argument("--num-solutions", type=int,   default=8,
                        help="Number of solutions to request (default: 8)")
    parser.add_argument("--warmup-calls",  type=int,   default=1,
                        help="Warmup calls before timed section (default: 1)")
    parser.add_argument("--seed",          type=int,   default=0,
                        help="Base RNG seed (default: 0)")
    parser.add_argument("--vary-seed",     action="store_true",
                        help="Increment seed each call (default: fixed seed)")
    parser.add_argument("--simulate-trial-gap", type=float, default=0.0, metavar="SECS",
                        help="Sleep N seconds between calls to simulate GPU idle (default: 0)")
    parser.add_argument("--rewarm-after-gap", action="store_true",
                        help="Run a rewarm call after each simulated gap (tests rewarm effectiveness)")
    parser.add_argument("--deep-rewarm", action="store_true",
                        help="Use tight-loop rewarm (runs until GPU settles to <15ms) instead of single call")
    args = parser.parse_args()

    _run(args)


if __name__ == "__main__":
    main()
