#!/usr/bin/env bash
set -euo pipefail

cd /home/pm/Desktop/Project/robot_stack

rm -rf /tmp/robot_pipeline/repeated_pick_place

PYTHONFAULTHANDLER=1 python -X faulthandler -u isaac_collector/run_repeated_pick_place.py \
  --scene-usd /home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd \
  --cup-path /World/office1/Room_seed123_idx000/furniture/tea_table/cup \
  --observation-source head_camera \
  --grasp-mode mock \
  --curobo-mode real \
  --curobo-robot-config /home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody_ignore_adjacent.yml \
  --execution-mode a2d_replay \
  --attach-cup-during-place \
  --ee-path /World/A2D/Link7_r \
  --target-source manual \
  --pickup-target-robot 0.45 0.00 0.80 \
  --place-target-robot 0.45 0.10 0.85 \
  --save-executed-trajectory \
  --trajectory-save-stride 100 \
  --enable-rgbd-recording \
  --use-head-camera \
  --use-wrist-cameras \
  --rgbd-width 320 \
  --rgbd-height 240 \
  --head-camera-parent /World/A2D/link_pitch_head \
  --head-camera-path /World/A2D/link_pitch_head/RGBD_Head_Camera \
  --head-camera-eye-world 2.83 -2.20 1.45 \
  --head-camera-target-world 2.83 -1.87 0.78 \
  --wrist-camera-target-world 2.83 -1.87 0.78 \
  --rgbd-record-stride 100 \
  --num-pick-place 1 \
  --headless \
  2>&1 | tee /tmp/test_three_robot_bound_cameras_head_graspnet_v2.log

echo "EXIT_CODE=${PIPESTATUS[0]}"
