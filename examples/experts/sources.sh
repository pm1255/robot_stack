#!/usr/bin/env bash
# Run from the repository root, inside the selected simulator's environment.
# Each output directory must be new. Failures are retained by the collector.
set -euo pipefail
TASK=${1:?Usage: bash examples/experts/sources.sh TASK OUTPUT [SEED] [EPISODES]}
OUTPUT=${2:?Provide a new output directory}
SEED=${3:-4105}
EPISODES=${4:-5}
case "$TASK" in
  PickSingleYCB-v1|PokeCube-v1)
    BACKEND=maniskill; BUDGET=1000
    OPTIONS='{"native_horizon":1200,"max_replans":3}' ;;
  PickPlaceSinkToCounter|CheesyBread|PackDessert)
    BACKEND=robocasa; BUDGET=1200
    OPTIONS='{"layout":1,"style":1}' ;;
  *) echo "Unsupported example task: $TASK" >&2; exit 2 ;;
esac
python -m robot_stack.collect --backend "$BACKEND" --task "$TASK" \
  --source-only --seed-start "$SEED" --episodes "$EPISODES" \
  --max-steps "$BUDGET" --adapter-options "$OPTIONS" --output "$OUTPUT"
python -m robot_stack.audit "$OUTPUT" --output "$OUTPUT/audit.json"
