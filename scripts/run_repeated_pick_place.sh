#!/usr/bin/env bash
set -e

PROJECT_ROOT="/home/pm/Desktop/Project/robot_stack"

ISAAC_PY="/home/pm/miniconda3/envs/isaaclab/bin/python"
GRASPNET_PY="/home/pm/miniconda3/envs/graspnet_env/bin/python"
CUROBO_PY="/home/pm/miniconda3/envs/curobo_env_cu13/bin/python"

SCENE_USD="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd"
CUP_PATH="/World/office1/Room_seed123_idx000/furniture/tea_table/cup"

GRASPNET_CKPT="/home/pm/Desktop/Project/robot_stack/third_party/graspnet-baseline/logs/log_rs/checkpoint.tar"

# 这个文件现在还没有，后面生成
A2D_CUROBO_YML="/home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody_ignore_adjacent.yml"

cd "${PROJECT_ROOT}"


"${ISAAC_PY}" -u isaac_collector/run_repeated_pick_place.py \
  --project-root "${PROJECT_ROOT}" \
  --scene-usd "${SCENE_USD}" \
  --cup-path "${CUP_PATH}" \
  --graspnet-python "${GRASPNET_PY}" \
  --curobo-python "${CUROBO_PY}" \
  --graspnet-checkpoint "${GRASPNET_CKPT}" \
  --curobo-robot-config "${A2D_CUROBO_YML}" \
  --grasp-mode real \
  --curobo-mode real \
  --num-pick-place 1 \
  --right-distance 0.25 \
  --right-axis x \
  --headless \
  --/persistent/isaac/asset_root/default="/opt/Assets/isaacsim_assets/Assets/Isaac/5.1"