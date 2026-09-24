# Scheduled perturbation experiments — 2026-09-24

All task classes in the installed MetaWorld 3.1.1 registry now have **at least one qualified correction sample** across the first sweep and a separately reported tool-task follow-up. This is existential task-class coverage, not a claim that every native instance, perturbation, or rollout recovers. The tool-task follow-up was chosen after inspecting first-sweep failures and must not be treated as a held-out evaluation.

| Experiment | Tasks / instances | Attempts | Successful sources | Qualified corrections | Audited HDF5 |
| --- | --- | ---: | ---: | ---: | ---: |
| MetaWorld first sweep | 50 classes, index 0, seed 600; simple + random schedules | 100 | 98 | 69 | 296 |
| MetaWorld diversity | push, pick-place, door-open, hammer; indices 0–2, seeds 610–612; multi-error + reverse + gripper | 36 | 36 | 24 | 108 |
| MetaWorld tool follow-up | stick-pull + stick-push; indices 0–9, seeds 620–629; tool-use + reverse | 40 | 34 | 18 | 108 |
| AI2-THOR six-goal route | FloorPlan1, seeds 700–702; three source-milestone heading errors | 3 | 3 | 3 | 9 |

All experiments use a 500-action cap; failed sources produce one file, successful sources produce three. There were no execution/configuration exceptions in these four batches. All **521 saved trajectories** passed structural, checksum, budget and correction-evidence audit. There are **114 qualified pairs in 179 attempts** in this set, with different configurations and potentially shared initial-state ancestry; this aggregate is not a benchmark score or a count of independent scenes.

MetaWorld first-sweep wall time was 127.63 s on the existing eval-server CPU environment, including native task generation. No controlled throughput comparison is claimed. MetaWorld uses upstream experts and privileged observations; AI2-THOR uses the native reachable grid.

The first sweep covered 48 classes with qualified samples. `stick-pull-v3` failed to produce a successful source at index 0. `stick-push-v3` either failed recovery or allowed the original continuation to succeed. The follow-up preserved all failures and obtained 14/20 qualified pull samples and 4/20 push samples. No success predicate or physics was relaxed.

## Videos and independent replay

Every MetaWorld task class has one independently replayed triplet: **50 classes × 3 branches = 150/150 passes**, maximum state error **0**, with matching native success traces. This verifies the selected coverage examples; the remaining trajectories were audited, not all independently re-executed. [Per-trajectory replay evidence](evidence/metaworld50-independent-replay.json).

- Multi-error pick-place, seed 610: source 50 actions / control 74 / recovery 103, all three independently replayed with maximum state error 0 and matching native success traces.
- Six-goal AI2-THOR, seed 700: source 105 / control 112 / recovery 111, all three independently replayed with pose/progress error 0 and matching native success traces.
- Tool-use stick-pull, seed 621: recovery 186 actions; a more complex manipulation example, not a mobile household task.

Video speed follows the viewing frame rate, not physical time. The six-goal example is sequential PointNav, not navigation-plus-manipulation or official ObjectNav. Composite household tasks such as clearing a table or retrieving an object from a cupboard remain future adapter/policy work.

## Reproduce

Use [perturbations.md](perturbations.md) for installation and initial full sweep. Additional batches:

```bash
python -m robot_stack.suite --tasks push-v3 pick-place-v3 door-open-v3 hammer-v3 \
  --episodes 3 --seed-start 610 \
  --adapter-options '{"error_features":"manipulation"}' \
  --schedules examples/perturbations/multi-error.json \
  examples/perturbations/reverse.json examples/perturbations/gripper.json \
  --output outputs/diversity

python -m robot_stack.suite --tasks stick-pull-v3 stick-push-v3 \
  --episodes 10 --seed-start 620 \
  --schedules examples/perturbations/tool-use.json examples/perturbations/reverse.json \
  --output outputs/tools

# Independently replay one qualifying source/control/recovery triplet per task.
python scripts/replay_task_coverage.py outputs/metaworld-all outputs/tools \
  --output outputs/coverage-replay
```

Evidence: [first sweep](evidence/metaworld50-scheduled-summary.json), [first audit](evidence/metaworld50-scheduled-audit.json), [diversity](evidence/metaworld-diversity-summary.json), [tools](evidence/metaworld-tools-summary.json), [six-goal navigation](evidence/ai2thor-long-summary.json). Interactive examples: [playground](playground.html).
