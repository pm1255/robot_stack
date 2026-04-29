#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/pm/Desktop/Project/robot_stack}"
PATCH_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[PATCH] project root: ${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/isaac_collector/runtime"
mkdir -p "${PROJECT_ROOT}/configs/tasks"
mkdir -p "${PROJECT_ROOT}/configs/scenes"
mkdir -p "${PROJECT_ROOT}/scripts"

cp "${PATCH_ROOT}/isaac_collector/runtime/target_registry.py" \
   "${PROJECT_ROOT}/isaac_collector/runtime/target_registry.py"

cp "${PATCH_ROOT}/isaac_collector/runtime/sim_target_pointcloud.py" \
   "${PROJECT_ROOT}/isaac_collector/runtime/sim_target_pointcloud.py"

cp "${PATCH_ROOT}/isaac_collector/runtime/action_specs.py" \
   "${PROJECT_ROOT}/isaac_collector/runtime/action_specs.py"

cp "${PATCH_ROOT}/isaac_collector/runtime/episode_logging.py" \
   "${PROJECT_ROOT}/isaac_collector/runtime/episode_logging.py"

cp "${PATCH_ROOT}/isaac_collector/run_atomic_manipulation.py" \
   "${PROJECT_ROOT}/isaac_collector/run_atomic_manipulation.py"

cp "${PATCH_ROOT}/configs/tasks/pick_place_cup_right10cm.json" \
   "${PROJECT_ROOT}/configs/tasks/pick_place_cup_right10cm.json"

cp "${PATCH_ROOT}/configs/scenes/mutil_room001.example.json" \
   "${PROJECT_ROOT}/configs/scenes/mutil_room001.example.json"

cp "${PATCH_ROOT}/scripts/run_atomic_sim_pointcloud_pick_place.sh" \
   "${PROJECT_ROOT}/scripts/run_atomic_sim_pointcloud_pick_place.sh"
chmod +x "${PROJECT_ROOT}/scripts/run_atomic_sim_pointcloud_pick_place.sh"

echo "[PATCH] installed scalable atomic manipulation runner."
echo "[PATCH] If your registry is currently at isaac_collector/configs/scenes/mutil_room001.json, keep using it."
echo "[PATCH] Example registry template copied to configs/scenes/mutil_room001.example.json"
echo "[PATCH] Run:"
echo "  cd ${PROJECT_ROOT}"
echo "  ./scripts/run_atomic_sim_pointcloud_pick_place.sh"
