#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/pm/Desktop/Project/robot_stack}"
PATCH_ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_ROOT"

mkdir -p isaac_collector/runtime
mkdir -p configs/scenes

cp "$PATCH_ROOT/isaac_collector/runtime/operable_scene_registry.py" \
   isaac_collector/runtime/operable_scene_registry.py

cp "$PATCH_ROOT/configs/scenes/mutil_room001.operable.example.json" \
   configs/scenes/mutil_room001.operable.example.json

python "$PATCH_ROOT/patch_run_atomic_to_operable_registry.py"

echo "[DONE] OperableSceneRegistry patch applied."
echo "[INFO] Your current registry can keep using legacy objects/cup paths."
echo "[INFO] Preferred new format example: configs/scenes/mutil_room001.operable.example.json"
echo "[INFO] Run:"
echo "  ./scripts/run_atomic_sim_pointcloud_pick_place.sh"
