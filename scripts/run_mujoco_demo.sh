#!/usr/bin/env bash
# run_mujoco_demo.sh — MuJoCo Franka Panda visualization demo
#
# Usage:
#   bash scripts/run_mujoco_demo.sh
#   bash scripts/run_mujoco_demo.sh --viewer
#   bash scripts/run_mujoco_demo.sh --viewer --viewer-speed 0.5
#   bash scripts/run_mujoco_demo.sh --scenario free_space --seed 1
#   bash scripts/run_mujoco_demo.sh --headless --no-animation

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "ds-iks" ]; then
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ds-iks 2>/dev/null || true
    fi
fi

echo "=== Multi-IK DS MuJoCo Demo ==="
echo "Repo: $REPO_ROOT"
echo "Python: $(python --version 2>&1)"
echo ""

python -m src.visualization.mujoco_demo "$@"
