#!/usr/bin/env bash
# This measured schedule did NOT yield qualified errors; retained for diagnosis.
set -euo pipefail
python -m robot_stack.calibrate --backend maniskill --tasks PokeCube-v1 \
  --episodes 2 --seed-start 4100 --validation-episodes 5 --validation-seed-start 4105 \
  --scales .125 .25 .5 --max-steps 1000 --case-timeout 600 --workers 3 \
  --adapter-options '{"native_horizon":1200,"max_replans":3}' \
  --schedule examples/perturbations/maniskill-late.json \
  --output "${1:?Provide a new output directory}"
