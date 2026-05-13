#!/usr/bin/env bash
# run_demo.sh — convenience wrapper for the Multi-IK DS visualization demo
#
# Usage:
#   bash scripts/run_demo.sh
#   bash scripts/run_demo.sh --scenario free_space --seed 1
#   bash scripts/run_demo.sh --headless --no-animation

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda environment if available and not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "ds-iks" ]; then
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ds-iks 2>/dev/null || true
    fi
fi

echo "=== Multi-IK DS Demo ==="
echo "Repo: $REPO_ROOT"
echo "Python: $(python --version 2>&1)"
echo ""

python -m src.visualization.demo_example "$@"
