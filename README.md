# GeoMultiAttractorDS

This repository contains the code for **GeoMA-DS: Geometrically-Guided Multi-Attractor Dynamical Systems for Reactive Control**.~~~~

---

## Problem

Standard task-space DS controllers (DiffIK + CBF) get permanently trapped at homotopy barriers: the safety filter repels the arm from the only collision-free path. Planning-based methods (BiRRT) find paths but add latency and can't react online.

**GeoMA-DS replaces pre-planned paths with a reactive, geometry-guided multi-attractor velocity field** that resolves homotopy barriers at runtime with ~8 ms per replan event.

---

## Method

```
HJCD-IK (GPU, batch=256–1000, ~4ms)
    │  diverse joint-space goal configs
    ▼
Build: IK attractors + geometric scoring + escape candidates
    │  attractor set, boundary waypoints
    ▼
GeoMultiAttractorDS (500 Hz)
    ├─ f_att  = -K_c (q - q_active*)   conservative pull to active IK goal
    ├─ f_tan  = α·w·(P_tan f_att - f_att)  obstacle surface tangent shaping
    └─ f_null = K_n (q_ref - q)         nullspace regularization
    │
    ├─ Attractor scoring + switching (clearance · horizon · hysteresis)
    └─ Escape state machine → pregoal cascade (online homotopy jump)
    │
    ▼
ImpedanceController → MuJoCo (500 Hz)
```

### Online homotopy jump (cross-barrier)

When the arm must cross a narrow slot (homotopy gap too large for any build-time IK), the escape state machine:
1. Builds a backtrack waypoint to maximize maneuverability
2. Stages two escape attractors (below → lateral) that thread the arm through the slot
3. Generates an intermediate pregoal IK at 20% of EE distance to goal (q_dist ≈ 1.2 rad from post-escape arm)
4. Upgrades to full-goal IK from the closer position: picks min-q_dist candidate among clearance-valid IK solutions (q_dist ≈ 4.5 rad, clearance ≈ 0.06 m)

All IK calls use HJCD-IK with `num_solutions=8` across 3 EE orientations, seeded from current arm configuration.

---

## Installation

### Prerequisites

- Python 3.11 (Anaconda)
- NVIDIA GPU + CUDA 12.x (required for HJCD-IK)
- Windows 11 or Linux

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/caelyasutake/GeoMA-DS
cd GeoMA-DS
```

### 2. Create conda environment

```bash
conda create -n geoma-ds python=3.11
conda activate geoma-ds
pip install numpy scipy matplotlib pytest
```

### 3. Build HJCD-IK

```bash
# Windows
.\external\HJCD-IK\scripts\bootstrap_windows.bat
pip install -e external/HJCD-IK

# Linux
./external/HJCD-IK/scripts/bootstrap.sh
pip install -e external/HJCD-IK
```

### 4. Install MuJoCo

```bash
pip install mujoco
```

### 5. Verify

```bash
conda activate geoma-ds
python -c "import hjcdik; print('HJCD-IK ok')"
python -m pytest tests/unit/ -q
```

---

## Running Benchmarks

```bash
conda activate geoma-ds

# All three scenarios, 10 trials each
python -m benchmarks.eval_baselines --scenarios open_reach i_barrier cross_barrier --trials 10

# Single scenario, 50 trials
python -m benchmarks.eval_baselines --scenarios cross_barrier --trials 50

# GeoMA-DS only (skip CBF baseline)
python -m benchmarks.eval_baselines --scenarios cross_barrier --methods geo_ma_ds --trials 10

# Render one trial (saves video to outputs/)
python -m benchmarks.eval_baselines --scenarios cross_barrier --methods geo_ma_ds --trials 1 --render
```

Results are printed to stdout and saved to `outputs/summary.csv` and `outputs/summary.json`.

---

## Running Tests

```bash
conda activate geoma-ds

# All unit tests
python -m pytest tests/unit/ -v

# Specific test
python -m pytest tests/unit/test_geo_multi_attractor_ds.py -v
```

---

## Project Structure

```
GeoMA-DS/
├── benchmarks/
│   └── eval_baselines.py        # Main benchmark: geo_ma_ds vs diffik_ds_cbf
├── external/
│   └── HJCD-IK/                 # Git submodule: GPU IK solver
└── src/
    ├── scenarios/
    │   ├── scenario_builders.py # open_reach, i_barrier, cross_barrier
    │   └── scenario_schema.py   # ScenarioSpec, Obstacle
    ├── simulation/              # MuJoCo physics wrapper
    └── solver/
        ├── ds/
        │   ├── geo_multi_attractor_ds.py  # GeoMA-DS core
        │   └── factory.py                 # Build: IK→attractors, escape candidates
        ├── ik/
        │   └── hjcd_wrapper.py  # HJCD-IK Python interface
        └── planner/
            └── collision.py     # Clearance function (SDF queries)
```

---

## Key Configuration

`GeoMultiAttractorDSConfig` in `src/solver/ds/geo_multi_attractor_ds.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_c` | 2.0 | Attractor gain (rad/s per rad) |
| `alpha_max` | 1.0 | Max tangent shaping weight |
| `h_alpha_on` | 0.12 | Clearance threshold to activate shaping |
| `enable_pre_goal_waypoint` | False | Enable homotopy jump via pregoal cascade |
| `pregoal_beta_sweep` | [0.20, 0.40, 0.60, 0.80] | Intermediate EE fractions toward goal |
| `pregoal_upgrade_on_near_goal` | True | Upgrade to full-goal IK when near intermediate |
| `pregoal_max_qdist` | 5.0 | Max joint-space distance for intermediate IK (rad) |
| `online_pregoal_ik_batch_size` | 256 | IK batch size for pregoal calls |

The `cross_barrier` scenario automatically enables `enable_pre_goal_waypoint=True` via the factory.

---

## Scenarios

| Name | Obstacle | Goal side | Notes |
|------|----------|-----------|-------|
| `open_reach` | None | Same as start | Free-space convergence baseline |
| `i_barrier` | I-shaped vertical plate | Same Y side | Arm must route around the plate |
| `cross_barrier` | YZ-cross (4-quadrant wall with slot) | Opposite Y side | Requires online homotopy jump |

Build a scenario programmatically:

```python
from src.scenarios.scenario_builders import build_frontal_yz_cross
from src.solver.ds.factory import build_geo_multi_attractor_ds

spec = build_frontal_yz_cross()
ds, diag = build_geo_multi_attractor_ds(spec)

# Run one step
import numpy as np
q = np.array(spec.q_start)
qdot, result = ds.compute(q)
print(f"clearance: {result.clearance:.3f}  active: {result.active_attractor_kind}")
```
