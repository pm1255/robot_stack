# Configurable perturbations and recovery

One collection engine, reusable native action interventions, and task-specific experts. A new task does **not** need a new dataset driver. It does need a policy that can act from off-demonstration states, a native success predicate, and any semantic milestones used by its schedule. An adapter or task name alone cannot guarantee recovery.

## Choose where, what and how many

```json
{
  "version": 1,
  "events": [
    {
      "at": {"fraction": 0.2},
      "type": "cartesian_offset",
      "steps": 8,
      "strength": 0.8,
      "repeat": 3,
      "gap": 8,
      "parameters": {"direction": [1, -1, 1], "gripper": -1}
    }
  ]
}
```

This injects three 8-action bursts. The first occurs after 20% of the **successful source's actions**; subsequent bursts are separated by eight normal actions. `gap: 0` gives adjacent bursts. All 24 injected actions count toward `max_steps`. `strength` is in `(0,1]`, within the native action bounds. Add entries with different `type` values to mix interventions.

| Parameter | Meaning |
| --- | --- |
| `at.fraction` | Fraction of the successful source's action count, in `(0,1)` |
| `at.step` | Insert after this many source actions, positive integer |
| `at.event` | First state carrying an adapter milestone in the successful source |
| `type` | Native intervention from the table below, or adapter extension |
| `steps` | Actions in each burst; positive integer |
| `repeat` / `gap` | Number of bursts / normal actions between repeated bursts |
| `strength` | Bounded action magnitude; discrete AI2-THOR requires 1 |
| `parameters` | Type-specific direction or gripper command |
| `--max-steps` | Total action budget per branch, including every injected action |
| `--error-threshold` | Minimum norm of recorded feature displacement (default 0.03) |
| `--episodes` / `--seed-start` | Attempt count / reproducible initial seed |

Use exactly one `at` field. Unknown fields, unsupported interventions, absent milestones, out-of-source anchors and impossible budgets are errors, not silently changed settings. More perturbations do not imply more qualified data.

**Event semantics:** milestone anchors are resolved on the successful source, then translated into the same nominal action ticks in both branches. They are not live semantic triggers on the already perturbed branch. Later branches may be in different task stages. A future live-trigger protocol must report that different causal comparison explicitly.

## Native intervention catalog

There are **19 configurable action interventions across five adapters**, including hold controls. These are not 19 demonstrated failure mechanisms; whether an intervention creates an error is measured per episode. No synthetic success flags, teleportation, forced attachment, or physics weakening are used.

| Adapter | Types | Parameters |
| --- | --- | --- |
| MetaWorld | `cartesian_offset`, `random_cartesian`, `gripper_open`, `gripper_close`, `action_reverse`, `action_hold` | Cartesian direction `[x,y,z]` and gripper `[-1,1]` for `cartesian_offset`; seeded per-action Cartesian noise for `random_cartesian` |
| RoboCasa NavigateKitchen | `base_yaw`, `base_translation`, `base_reverse`, `base_hold` | Translation direction `[x,y]`; bounded normalized base velocity |
| AI2-THOR PointNav | `wrong_heading`, `backtrack`, `lateral_drift`, `navigation_hold` | Native RotateRight / MoveBack / MoveRight / Pass; discrete grid spacing |
| Custom adapter | `make_perturbation(event, rng)` | Caller-defined physical action generator with strict action count |
| Legacy adapter | `native_default` | Existing `perturbation_actions(steps)`, no strength scaling |

Random seeds determine commands, not whether the physics makes them effective. Collision-blocked moves and holds often induce no measurable displacement and are retained as unqualified controls.

MetaWorld defaults to EEF xyz features for compatibility. For gripper/object experiments use `--adapter-options '{"error_features":"manipulation"}'`: EEF xyz + normalized aperture × 0.05 + primary-object xyz. This mixed feature norm is a configured error proxy, not a task failure predicate. Optional milestones are `gripper_closed` (opening < 0.25, **not** a grasp assertion) and `object_raised_3cm` (relative to reset). They may be absent or already true at reset; unusable anchors fail explicitly.

## Launch examples

Install each simulator in a separate environment; core install alone does not install its simulator or assets. Each output directory must be new.

```bash
# 1. Simple manipulation: one perturbation
pip install -e '.[metaworld]'
python -m robot_stack.collect --backend metaworld --task push-v3 \
  --episodes 3 --seed-start 600 --schedule examples/perturbations/simple.json \
  --output outputs/push-single

# 2. Three separated perturbations in one recovery trajectory
python -m robot_stack.collect --backend metaworld --task pick-place-v3 \
  --episodes 3 --seed-start 600 --schedule examples/perturbations/multi-error.json \
  --output outputs/pick-place-multi

# 3. Enumerate all installed native MetaWorld experts
python -m robot_stack.suite --backend metaworld --tasks all --list

# 4. All discovered tasks × two schedules; preserve every unsuccessful attempt
python -m robot_stack.suite --backend metaworld --tasks all --episodes 1 \
  --seed-start 600 --schedules examples/perturbations/simple.json \
  examples/perturbations/random.json --output outputs/metaworld-all

# 5. Gripper intervention with gripper/object error features
python -m robot_stack.collect --backend metaworld --task pick-place-v3 \
  --adapter-options '{"error_features":"manipulation"}' \
  --schedule examples/perturbations/gripper.json --output outputs/gripper

# 6. RoboCasa kitchen navigation (matching RoboCasa / robosuite assets required)
python -m robot_stack.collect --backend robocasa --task NavigateKitchen \
  --adapter-options '{"layout":1,"style":1}' --episodes 3 \
  --schedule examples/perturbations/robocasa.json --output outputs/kitchen

# 7. AI2-THOR single-goal navigation (Unity CloudRendering + GPU/Vulkan required)
python -m robot_stack.collect --backend ai2thor --task FloorPlan1 \
  --schedule examples/perturbations/ai2thor.json --output outputs/navigation

# 8. Six sequential navigation goals, three source-milestone perturbations
python -m robot_stack.collect --backend ai2thor --task FloorPlan1 \
  --episodes 3 --seed-start 700 --max-steps 500 --adapter-options '{"waypoints":6}' \
  --schedule examples/perturbations/ai2thor-long.json --output outputs/long-navigation

# 9. Python interface
python examples/perturbations/python_api.py

# 10. Audit saved evidence, then independently replay native actions
python -m robot_stack.audit outputs/pick-place-multi --output outputs/multi-audit.json
python -m robot_stack.replay outputs/pick-place-multi/ep_0600/recovery.hdf5 \
  --output outputs/multi-replay.json --video outputs/multi-recovery.mp4
```

Use `--tasks push-v3 pick-place-v3` for a subset. Automatic `all` expert discovery currently applies to MetaWorld only. The original eight-task expert-only CLI remains compatible.

## Multi-error evidence and causality

The source uses the expert with no perturbations. Both branches physically replay the same prefix. The control then reuses the source's remaining actions; recovery uses state feedback. Both receive identical intervention commands at identical nominal ticks, excluding inserted actions.

The **first error state must match**. After feedback starts, later error states can differ and are reported individually in `branch_state_max_abs_errors`; we do not claim identical states for all interventions. This measures a scheduled intervention experiment, not isolated causal attribution for every later error. To isolate one error, collect independent single-event triplets at each source anchor.

A qualified record requires: successful source; every requested event fully executed and above the configured feature threshold in both branches; unsuccessful open-loop control; successful native final recovery; identical first error; matching physical prefix. Success before a later scheduled event means that event was skipped and the sample is not qualified. Failed sources, failed recoveries, ineffective interventions and configuration errors remain in the results.

The HDF5 alignment stays T actions / T+1 states. `events_json` stores T+1 milestone snapshots. `perturbation_events` records each half-open action interval, type, displacement, threshold and completion. `split_group` keeps descendants of the same initial task state together. The audit recomputes event timing, command equality, displacement, budget and the verdict from saved evidence.

## Extending other benchmarks

For a working native adapter, pass `--adapter-options '{"adapter_factory":"my_package.adapters:make_adapter"}'`. The factory receives `(task, **options)` and must return the core Adapter protocol. `reset(seed)` must reproduce physics **and policy state**. `expert_action()` must be side-effect free with respect to physics. Expose `events()` for milestone labels and optionally `make_perturbation(event, rng)` for native interventions. Replay stores and reuses the factory/options.

RoboDojo's existing bridge accepts a caller-provided native policy and state readers. ManiSkill and RoboTwin now connect native experts to this scheduler through action streaming and current-state replanning; verified task coverage and remaining limitations are listed in [task coverage](task-coverage.md). A method that only replays a successful action tape is not a general recovery expert. RoboCasa composite household manipulation and all-task RoboDojo recovery remain integration work. The six-goal AI2-THOR example is **longer sequential navigation, not a difficult navigation-plus-manipulation household benchmark**.

MimicGen's verified Lift example remains available in [mimicgen.md](mimicgen.md). New tasks require their own object-relative subtask definitions and environment interface. Generated successes are not automatically correction pairs; run perturbation/recovery validation separately and preserve source ancestry when splitting data.


## Native ManiSkill and RoboTwin planners

The same JSON schedule now supports ManiSkill (`joint_offset`, `gripper_open`, `gripper_close`, `joint_hold`) and experimental RoboTwin (`joint_offset`). Native experts are streamed into the collector one action at a time; after an intervention, replanning starts from current state. RoboTwin durations count individual physics steps; ManiSkill durations count control steps. The web configurator exposes both backends, JSON parameters, adapter options, budgets, episode counts and seeds. [Installation, all-expert commands and measured coverage](task-coverage.md).

Use `examples/perturbations/maniskill-late.json`, `robotwin-carry.json` or `robotwin-first-carry.json` as task-specific starting points. A different task may require a different insertion time or recovery policy. Original experts are collection policies, not proof of arbitrary-error recovery.
