#!/usr/bin/env bash
set -euo pipefail
# Paths are caller-supplied; run from repository root in the configured GPU container.
: "${MANISKILL_PYTHON:?}"
: "${ROBOTWIN_PYTHON:?}"
: "${MANISKILL_DATA:?}"
: "${ROBOTWIN_ROOT:?}"
: "${ROBOTWIN_RESULT:?}"
: "${VERIFY_OUTPUT:?}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir "$VERIFY_OUTPUT"
ms_status=0
"$MANISKILL_PYTHON" scripts/verify_maniskill.py --input "$MANISKILL_DATA" --output "$VERIFY_OUTPUT/maniskill" --video || ms_status=$?
rt_status=0
"$ROBOTWIN_PYTHON" -m benchmark_collector.replay_robotwin "$ROBOTWIN_RESULT" --root "$ROBOTWIN_ROOT" --output "$VERIFY_OUTPUT/robotwin-cup.json" || rt_status=$?
if (( ms_status != 0 || rt_status != 0 )); then exit 1; fi
