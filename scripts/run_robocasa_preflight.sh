#!/usr/bin/env bash
set -euo pipefail
: "${ROBOCASA_PYTHON:?Set ROBOCASA_PYTHON to the simulator Python executable}"
: "${ROBOCASA_SOURCE:?Set ROBOCASA_SOURCE to the directory containing the robocasa package}"
: "${ROBOSUITE_SOURCE:?Set ROBOSUITE_SOURCE to the directory containing the robosuite package}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${ROBOCASA_SOURCE}:${ROBOSUITE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYNPUT_BACKEND=dummy MUJOCO_GL=disable
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$ROBOCASA_PYTHON" "$PROJECT_DIR/scripts/robocasa_preflight.py" "$@"
