#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Output root must not already exist.
output=${1:?Usage: bash scripts/run_lift_comparison.sh NEW_OUTPUT_DIR [SEED_START] [EPISODES]}
seed_start=${2:-100}
episodes=${3:-20}
python_bin=${SIM_PYTHON:-python}
export MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir "$output"
"$python_bin" -m robosuite_collector.collect --output "$output/clean" --seed-start "$seed_start" --episodes "$episodes" --max-attempts 1 --label clean | tee "$output/clean.log"
"$python_bin" -m robosuite_collector.collect --output "$output/perturbed" --seed-start "$seed_start" --episodes "$episodes" --max-attempts 1 --first-grasp-offset .06 0 0 --label perturbed | tee "$output/perturbed.log"
"$python_bin" -m robosuite_collector.collect --output "$output/recovery" --seed-start "$seed_start" --episodes "$episodes" --max-attempts 3 --first-grasp-offset .06 0 0 --label recovery | tee "$output/recovery.log"
"$python_bin" scripts/audit_lift_comparison.py --input "$output" --output "$output/comparison.json"
"$python_bin" scripts/build_collection_report.py --input "$output" --output "$output/report.json"
