# Paired correction protocol

The framework is task-independent; the policy and success oracle are not. A task name alone cannot supply a reliable expert. Current adapters use privileged simulator state, so results are collection checks rather than learned-policy evaluations.

## Purpose and counterfactual interpretation

Robot Stack aims to provide experience with unexpected execution states, alongside successful demonstrations. Its intended research uses include error recognition, corrective action selection and recovery during navigation, manipulation and long tasks. These are data and training objectives, not claims of measured learning gains.

“Counterfactual” refers here to controlled alternative executions: a successful unperturbed source, an original-action continuation after intervention, and a feedback recovery from the same verified first post-intervention state. The source and control compare execution with and without an injected disturbance; control and recovery compare two continuations after that disturbance. The qualification rules below determine which comparisons actually demonstrate correction.

For schedules with multiple interventions, the first post-intervention state is matched. Later interventions share commands and nominal ticks, but branch states can diverge after feedback begins; later states must not be described as identical. Numeric state checks are limited to what each adapter records. This construction does not establish a general causal effect, equalize controller computation, or demonstrate transfer to unseen failures or real robots. See the [project motivation](../README.md#why-we-build-robot-stack).

## What qualifies as a correction?

1. Collect a successful source demonstration with real environment actions.
2. Select a prefix strictly before source completion. Recreate the same seeded environment and physically re-execute that prefix. Reject mismatching numeric states.
3. Inject a bounded action sequence. Verify a physical deviation in documented error features and confirm the task is not already successful.
4. In the **perturbed control**, execute the original remaining source actions without feedback. Save its final success/failure.
5. In the **recovery branch**, reconstruct the same prefix and perturbation, verify the same resulting state, then run a feedback policy from that state. No reset or state restoration is allowed during recovery.
6. Mark a qualified correction only when the source succeeded, the deviation was verified in both branches, the perturbed control failed, and recovery succeeded. Retain every non-qualifying attempt separately.

The open-loop control and closed-loop recovery have different lengths. Both share the same maximum action budget, including prefix and perturbation. This comparison demonstrates recoverability under this injected error; it is not a compute-matched policy benchmark or proof of improvement over a different closed-loop policy.

## HDF5 schema: `robot_stack.correction.v1`

| Field | Shape / meaning |
| --- | --- |
| `states` | T+1 numeric states: initial, then one per executed action |
| `error_features` | T+1 task-relevant physical features; units documented by adapter |
| `actions_json` | T executed actions; supports continuous vectors and native discrete action dictionaries |
| `phases` | T labels: expert, source_prefix, perturbation, open_loop_continuation, recovery |
| `success` | T post-action task-oracle flags |
| HDF5 `metadata` | backend, task, seed, versions, budget, source ID and branching provenance |

Each HDF5 has an `*_episode_result.json` with SHA256. `correction.json` links the source/control/recovery verdicts. All action ranges are half-open `[start, end)`; the state immediately after action `i` is `states[i+1]`. `branch_step` is the number of original actions executed before perturbation. `recovery_action_range` includes all feedback recovery actions up to termination or budget.

Use `source_demo_id` / `split_group` as the split unit. Never split a source and its perturbation descendants between training and validation. The current state data is suitable for native replay; image observations must be rendered from replay if a vision policy needs them. AI2-THOR records agent pose in a static scene, not a complete Unity snapshot.

## Extending the interface

Implement `reset`, `state`, `error_features`, `step`, `expert_action`, `perturbation_actions`, `success`, `terminal`, and `close`. Call `collect_triplet(adapter, seed, new_output_dir, budget=...)`. Optional `render` enables replay video. Keep all simulator-specific imports outside the core.

For RoboDojo, use `RoboDojoAdapter` with a caller-created native `EvalEnv`, a real policy callback, a numeric state reader and a physical perturbation generator. Its native `success` field starts as True, so the bridge requires `end_flag`, `success` and at least one actually executed action. The bridge is not a substitute for installing Isaac Sim, loading assets, launching XPolicyLab or validating a capable policy.

## MimicGen boundary

These files are not yet MimicGen demonstrations. Environment-specific object/EEF frames, `datagen_info`, subtask termination annotations and a successful generation/replay test are still required. Do not rename the files and claim MimicGen compatibility.
