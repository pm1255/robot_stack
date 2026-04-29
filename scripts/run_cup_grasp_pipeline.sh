#!/usr/bin/env bash
set -e

PROJECT_ROOT="/home/pm/Desktop/Project/robot_stack"

ISAAC_PY="/home/pm/miniconda3/envs/isaaclab/bin/python"
GRASPNET_PY="/home/pm/miniconda3/envs/graspnet_env/bin/python"
CUROBO_PY="/home/pm/miniconda3/envs/curobo_env/bin/python"

SCENE_USD="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd"
CUP_PATH="/World/office1/Room_seed123_idx000/furniture/tea_table/cup"

export ROBOT_STACK_GRASPNET_PY="${GRASPNET_PY}"
export ROBOT_STACK_CUROBO_PY="${CUROBO_PY}"

cd "${PROJECT_ROOT}"

"${ISAAC_PY}" -u isaac_collector/run_cup_grasp_pipeline.py \
  --project-root "${PROJECT_ROOT}" \
  --scene-usd "${SCENE_USD}" \
  --cup-path "${CUP_PATH}" \
  --grasp-mode mock \
  --curobo-mode mock \
  --execution-mode direct_cup_debug \
  --output-dir /tmp/robot_pipeline/cup_grasp_run \
  --keep-open \
  --/persistent/isaac/asset_root/default="/opt/Assets/isaacsim_assets/Assets/Isaac/5.1"