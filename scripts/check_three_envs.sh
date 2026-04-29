#!/usr/bin/env bash
set -e

PROJECT_ROOT="/home/pm/Desktop/Project/robot_stack"
GRASPNET_ROOT="/home/pm/Desktop/Project/graspnet-baseline"
LOG_DIR="/tmp/robot_pipeline/env_check_logs"

ISAAC_PY="/home/pm/miniconda3/envs/isaaclab/bin/python"
CUROBO_PY="/home/pm/miniconda3/envs/curobo_env/bin/python"
GRASPNET_PY="/home/pm/miniconda3/envs/graspnet_env/bin/python"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

echo "============================================================"
echo "[1/3] Checking Isaac Lab environment"
echo "============================================================"
"${ISAAC_PY}" scripts/check_env.py \
  --mode isaaclab \
  --project-root "${PROJECT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/isaaclab_check.log"

echo ""
echo "============================================================"
echo "[2/3] Checking cuRobo environment"
echo "============================================================"
"${CUROBO_PY}" scripts/check_env.py \
  --mode curobo \
  --project-root "${PROJECT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/curobo_check.log"

echo ""
echo "============================================================"
echo "[3/3] Checking GraspNet environment"
echo "============================================================"
"${GRASPNET_PY}" scripts/check_env.py \
  --mode graspnet \
  --project-root "${PROJECT_ROOT}" \
  --graspnet-root "${GRASPNET_ROOT}" \
  2>&1 | tee "${LOG_DIR}/graspnet_check.log"

echo ""
echo "============================================================"
echo "All checks finished."
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"