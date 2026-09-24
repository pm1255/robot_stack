# Coverage, collection yield and planner improvements

The coverage table counts task classes with at least one observed successful case. For example, RoboCasa 2/317 means two task classes have a verified controller and successful examples in the installed inventory. It is not a 0.63% execution success rate. A whole-benchmark success rate needs a fixed task distribution, scenes, seeds and trial budget, including failures.

Our collection experts may use privileged simulator state, native success predicates, inverse kinematics, motion planning and scripted skills. They are demonstration generators, not learned policies evaluated under restricted benchmark observations. Saved metadata marks `oracle_policy=true` and `standard_benchmark_score=false`.

## What the current code actually uses

- **AI2-THOR:** native `GetReachablePositions`, shortest-path BFS on a uniform grid, physical move/rotate actions, and replanning around failed edges. A* is a possible efficiency upgrade on larger maps, but replacing BFS with A* on the same connected uniform grid does not by itself add navigation-and-manipulation task semantics.
- **ManiSkill:** native synchronous task experts and MPlib, streamed into the collector so every action is recorded and budgeted. Most upstream experts request screw-interpolated motions. An optional planner portfolio now tries RRTConnect after screw planning fails and stops the current script if both fail, instead of blindly continuing to its next grasp/place stage.
- **RoboTwin:** upstream task scripts and the native robot planning interface. The tested `aloha-agilex` embodiment already sets `planner: curobo`; adding cuRobo by name is therefore not the missing integration. Restarting `play_once()` after an arbitrary disturbance can violate its assumptions about object poses, completed stages and grasp state.
- **RoboCasa:** privileged base feedback and an experimental Cartesian manipulation state machine. PickPlaceCounterToSink has fixed-scene evidence, not a controller covering the remaining kitchen task classes.
- **RoboDojo:** native initialization/physics smoke and an adapter bridge; built-in successful task experts and recovery policies remain missing.

Official references: [ManiSkill motion-planned demonstration generation](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html), [cuRobo MotionGen](https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html). Motion planning reaches a supplied goal pose; choosing that goal, verifying a grasp, preserving completed subgoals and recovering a dropped object still need task logic and feedback.

## Optional ManiSkill planner portfolio

After installing the native ManiSkill environment and assets:

```bash
robot-stack-suite --backend maniskill \
  --tasks PickCube-v1 PlaceSphere-v1 LiftPegUpright-v1 PullCubeTool-v1 PlugCharger-v1 \
  --episodes 2 --seed-start 810 --max-steps 1000 --case-timeout 180 \
  --adapter-options '{"native_horizon":1200,"max_replans":4,"planner_fallback":true}' \
  --schedules examples/perturbations/maniskill-late.json \
  --output outputs/planner-portfolio
```

`planner_fallback` is opt-in and defaults to false. It clones the upstream solver's module globals; installed native files are not modified. The portfolio performs dry-run planning before executing one selected path. RRTConnect uses the installed MPlib planner configuration; this does not add missing obstacle geometry or guarantee contact-rich task success. Source and recovery branches record planning request, fallback success and failure counters. Sampling-based planning is not promised bitwise reproducible; saved action replay is the independent validation mechanism.

A separate fix preserves the last physically commanded gripper value during policy hold actions. Explicit `joint_offset` and `joint_hold` interventions retain their documented opening behavior, so this fix does not weaken the requested disturbance.

## Matched native experiment: 2026-09-24

Five task classes, seeds 810/811, identical 15-step joint disturbance at 60% of the source, offset 0.8 radians, total budget 1,000 actions and four replanning attempts. Both runs completed without setup or worker errors on the same single-4090 job.

| Metric | Upstream baseline | Optional portfolio + hold fix |
|---|---:|---:|
| Successful source / attempts | 9/10 | 9/10 |
| Recovery / induced-error cases with failed control | 2/9 | 2/9 |
| Qualified correction / attempts | 2/10 | 2/10 |
| Saved trajectories passing audit | 28/28 | 28/28 |

**No success-rate improvement was demonstrated in this experiment.** All nine successful source tapes are action-identical between versions; source-state and first-error-state differences are zero. The two successful recoveries are PickCube seed 810 and LiftPegUpright seed 810. Recovery diagnostics recorded five RRT fallback requests and no successful RRT fallback; many unsuccessful cases had nominally successful planning requests but did not achieve the physical task.

The option stays experimental and disabled by default. These ten attempts target previously difficult cases and are not an unbiased whole-benchmark estimate. The collector holds the last gripper command when a policy has no next action; this fixes a control inconsistency but did not improve aggregate success in this sample.

[Full comparison](evidence/planner-comparison/comparison.json) · [Baseline report](evidence/planner-comparison/baseline-summary.json) · [Candidate report](evidence/planner-comparison/portfolio-summary.json) · [Identical source/error-state checks](evidence/planner-comparison/matched-source-states.json) · [Planning diagnostics](evidence/planner-comparison/portfolio-planning-diagnostics.json) · [Baseline audit](evidence/planner-comparison/baseline-audit.json) · [Candidate audit](evidence/planner-comparison/portfolio-audit.json)

## Why more planning tools may not help a failed case

In the matched seed-811 PickCube diagnostic, a 15-step, 0.8-radian joint intervention at 60% of the successful source throws the cube off the tabletop. Its saved position reaches approximately `[-0.837, 2.758, -0.900]` metres ([state evidence and trajectory checksum](evidence/planner-comparison/pickcube-unreachable-diagnostic.json)). A fixed-base arm cannot recover that displaced object with the tested embodiment. Replanning cannot repair physical unreachability.

Keep these samples, but label their failure cause separately. To build a recoverable-error dataset, evaluate a predeclared severity grid and report each tier's attempts and outcomes. Do not silently discard difficult seeds, weaken the success predicate, teleport objects back, or claim success-rate improvement by changing the denominator. An object falling within reach and an object leaving the workspace are different recovery problems.

## Measure the right quantities

Report these separately for every task and severity tier:

1. Successful source trajectories / all attempted initializations, with setup errors explicitly counted.
2. Completed correction verdicts / successful source attempts.
3. Recovery success / completed cases with induced error and a failed open-loop control.
4. Qualified correction groups / all attempts.
5. Wall time and saved physical actions, so yield improvements can be distinguished from additional retries.

```bash
python scripts/compare_collection_runs.py \
  outputs/planner-baseline/summary.json \
  outputs/planner-portfolio/summary.json \
  --output outputs/planner-comparison.json
```

The comparison rejects mismatched or duplicate task/seed/schedule keys. Also compare full run configurations, action budgets, simulator versions, and source/error states. Identical schedule names alone do not guarantee identical experiments. Old trajectory counts remain a dated snapshot until a new server-wide inventory is published.

## Next controller work

Prioritize state-aware recovery skills: test native grasp/contact predicates, preserve completed subgoals, lift out of contact before replanning, generate multiple feasible grasp candidates, and try alternate approach poses. Add task families behind the shared collector instead of treating every registered environment ID as solved. Use A* for navigation where appropriate, MPlib/cuRobo for arm trajectories, and a task-level state machine for the skill sequence. Validate on held-out seeds and scenes before promoting a controller to broad coverage.
