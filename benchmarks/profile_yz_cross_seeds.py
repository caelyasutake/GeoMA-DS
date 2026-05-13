"""Multi-seed reliability profiler for frontal_yz_cross.

Usage:
    python -m benchmarks.profile_yz_cross_seeds [--num-seeds N] [--method METHOD]

Runs frontal_yz_cross N times (seeds 0..N-1) with the full boundary-escape +
backtrack-staging pipeline and reports:

    success_rate
    median_grasp_err_m
    median_min_clearance_m
    median_backtrack_dist_at_escape_rad  (dist_rad from [backtrack_to_escape] log)
    median_n_esc_sw                      (escape switches per trial)
    median_n_back_sw                     (backtrack activations per trial)
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import contextlib
from typing import List, Optional

import numpy as np

from benchmarks.eval_baselines import (
    _PLANNER_PROFILES,
    _build_frontal_yz_cross,
    build_solver,
    run_trial,
    warmup_hjcdik,
)

_N_STEPS = 3000   # same as eval_baselines default

_RE_BT_ESCAPE = re.compile(
    r"\[backtrack_to_escape .*?dist_rad=([0-9.]+)"
)


class _Tee:
    """Write to both a real stream and a StringIO buffer."""
    def __init__(self, real, buf: io.StringIO):
        self._real = real
        self._buf  = buf

    def write(self, s: str):
        self._real.write(s)
        self._buf.write(s)

    def flush(self):
        self._real.flush()
        self._buf.flush()


@contextlib.contextmanager
def _capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = _Tee(old, buf)
    try:
        yield buf
    finally:
        sys.stdout = old


def _parse_bt_dist(captured: str) -> Optional[float]:
    """Return the last backtrack_to_escape dist_rad seen in captured output."""
    matches = _RE_BT_ESCAPE.findall(captured)
    return float(matches[-1]) if matches else None


def _run_seed(
    seed: int,
    method: str,
    attractor_generation_mode: str = "online",
    planner_profile: str = "robust",
) -> dict:
    spec = _build_frontal_yz_cross()
    _profile = _PLANNER_PROFILES[planner_profile]

    with _capture_stdout() as buf:
        solver, clearance_fn = build_solver(
            method=method,
            spec=spec,
            ik_source=_profile["ik_source"],
            ik_batch_size=_profile["ik_batch_size"],
            ik_num_solutions=_profile["ik_num_solutions"],
            ik_filter_mode=_profile["ik_filter_mode"],
            enable_clearance_recovery=True,
            enable_boundary_escape_waypoints=True,
            enable_backtrack_staging=True,
            enable_pre_goal_waypoint=True,
            attractor_generation_mode=attractor_generation_mode,
            enable_yz_expansion=_profile.get("enable_yz_expansion", True),
            online_escape_ik_batch_size=_profile.get("online_escape_batch", 64),
            online_pregoal_ik_batch_size=_profile.get("online_pregoal_batch", 64),
            online_escape_candidate_policy=_profile.get("online_escape_candidate_policy", "score_first"),
            online_ik_max_ms=_profile.get("online_ik_max_ms", 0.0),
        )
        result, _ = run_trial(
            solver, spec, clearance_fn,
            n_steps=_N_STEPS,
        )

    bt_dist = _parse_bt_dist(buf.getvalue())

    return {
        "seed":            seed,
        "success":         result.success,
        "grasp_err_m":     result.final_grasp_err_m,
        "min_clearance_m": result.min_clearance_m,
        "n_esc_sw":        result.n_esc_sw,
        "n_back_sw":       result.n_back_sw,
        "n_reentry_fail":  result.n_reentry_fail,
        "n_pregoal_sw":    result.n_pregoal_sw,
        "bt_dist_at_esc":  bt_dist,
        "dnf_reason":      result.dnf_reason,
    }


def _median_opt(vals: list) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return float(np.median(clean)) if clean else None


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed reliability profiler for frontal_yz_cross"
    )
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--method", default="geo_ma_ds",
        choices=["geo_ma_ds"],
    )
    parser.add_argument(
        "--attractor-generation", dest="attractor_generation",
        choices=["online", "prebuild_debug"], default="online",
        help="Attractor generation mode: 'online' (default, fair timing) or "
             "'prebuild_debug' (precompute at setup, not valid for benchmarks)",
    )
    parser.add_argument(
        "--planner-profile", dest="planner_profile",
        choices=list(_PLANNER_PROFILES.keys()), default="robust",
        help="IK planner preset (default: robust)",
    )
    args = parser.parse_args()

    if args.attractor_generation == "prebuild_debug":
        print("[warn] prebuild_debug excludes online attractor generation cost from "
              "runtime behavior; do not use for fair timing")

    print(f"[profile_yz_cross_seeds] method={args.method} num_seeds={args.num_seeds} "
          f"attractor_generation={args.attractor_generation} "
          f"planner_profile={args.planner_profile}")
    print("[warmup] HJCD-IK CUDA warmup ...", end=" ", flush=True)
    print(f"{warmup_hjcdik():.0f}ms")

    rows: List[dict] = []
    for seed in range(args.num_seeds):
        print(f"\n--- seed {seed}/{args.num_seeds - 1} ---", flush=True)
        row = _run_seed(
            seed, args.method,
            attractor_generation_mode=args.attractor_generation,
            planner_profile=args.planner_profile,
        )
        rows.append(row)
        status = "SUCCESS" if row["success"] else f"FAIL({row['dnf_reason']})"
        bt_str = f"{row['bt_dist_at_esc']:.3f}" if row["bt_dist_at_esc"] is not None else "n/a"
        print(
            f"  seed={seed} {status} "
            f"grasp={row['grasp_err_m']:.4f}m "
            f"cl={row['min_clearance_m']:.4f}m "
            f"esc_sw={row['n_esc_sw']} "
            f"back_sw={row['n_back_sw']} "
            f"reentry_fail={row['n_reentry_fail']} "
            f"pregoal_sw={row['n_pregoal_sw']} "
            f"bt_dist_at_esc={bt_str}",
            flush=True,
        )

    successes = [r for r in rows if r["success"]]
    success_rate = len(successes) / len(rows)

    print("\n" + "=" * 60)
    print(f"  seeds evaluated            : {len(rows)}")
    print(f"  success_rate               : {success_rate:.2%}  ({len(successes)}/{len(rows)})")

    def _fmt(label: str, val: Optional[float], unit: str = "") -> str:
        return f"  {label:<30}: {val:.4f}{unit}" if val is not None else f"  {label:<30}: n/a"

    print(_fmt("median_grasp_err_m",         _median_opt([r["grasp_err_m"]     for r in rows]), "m"))
    print(_fmt("median_min_clearance_m",      _median_opt([r["min_clearance_m"] for r in rows]), "m"))
    print(_fmt("median_n_esc_sw",             _median_opt([r["n_esc_sw"]          for r in rows])))
    print(_fmt("median_n_back_sw",            _median_opt([r["n_back_sw"]         for r in rows])))
    print(_fmt("median_n_reentry_fail",       _median_opt([r["n_reentry_fail"]    for r in rows])))
    print(_fmt("median_n_pregoal_sw",         _median_opt([r["n_pregoal_sw"]      for r in rows])))
    bt_med = _median_opt([r["bt_dist_at_esc"] for r in rows])
    if bt_med is not None:
        print(f"  {'median_bt_dist_at_esc_rad':<30}: {bt_med:.3f} rad")
    else:
        print(f"  {'median_bt_dist_at_esc_rad':<30}: n/a (no backtrack→escape observed)")
    print("=" * 60)


if __name__ == "__main__":
    main()
