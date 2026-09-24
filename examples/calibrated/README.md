# Empirically validated perturbation schedules

These schedules passed the recorded small-sample held-out gate on 2026-09-24. They are conditional on the task, expert, controller options, action budget and simulator version in the experiment report; they do not guarantee success on new scenes.

`manifest.json` records every included schedule, clean baseline counts, held-out counts, calibration/validation seeds, adapter options and the evidence path. Only tasks whose status is `passed_configured_validation` are included. Failed candidate profiles remain in the full reports under `docs/evidence/calibration/`.

```bash
pip install -e .
robot-stack-collect --backend maniskill --task PickCube-v1 \
  --episodes 10 --seed-start 4000 --max-steps 1000 \
  --adapter-options '{"native_horizon":1200,"max_replans":4}' \
  --schedule examples/calibrated/maniskill/PickCube-v1.json \
  --output outputs/pickcube-new-scenes
python -m robot_stack.audit outputs/pickcube-new-scenes \
  --output outputs/pickcube-new-scenes/audit.json
```

The matching native simulator and assets must already be installed. For MetaWorld, a changed random seed alone does not ensure a different MT1 task instance: use unused `task_index` values and inspect `split_group`. The calibration CLI handles disjoint calibration/validation indices automatically.

A successful source, a failed replay from the perturbed state, and a successful recovery from that same state are required for a qualified correction. An intervention that leaves the original policy successful is not a qualified error. All outcomes, including failures and errors, must remain in the denominator.

See [calibration instructions](../../docs/task-calibration.md) and the [task readiness dashboard](https://pm1255.github.io/robot_stack/readiness.html).
