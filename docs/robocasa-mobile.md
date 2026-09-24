# RoboCasa: mobile manipulation and recovery

The experimental `PickPlaceCounterToSink` adapter runs the native PandaOmron robot: navigate to the object, approach, grasp, lift clear of the counter, carry to the sink, place, release and retreat. It currently targets **layout 1 / style 1 / apple objects**. It is not a controller for every RoboCasa task, scene or object category. Scene distractors, collisions and the native task success check remain active.

The collection policy reads privileged base, end-effector and object poses, and the native grasp predicate. It sends bounded base velocity, Cartesian arm and gripper commands; it does not teleport the object or restore an error state. After an intervention, the controller chooses a recovery stage from the current grasp and task state. A lost object triggers another physical approach and grasp attempt.

## Launch three different interventions

Install RoboCasa assets and a compatible native environment first. Validated runtime: RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1, Python 3.10; Objaverse and Lightwheel object registries. Simulator dependencies should be isolated from RoboTwin / ManiSkill environments.

```bash
export MUJOCO_GL=egl PYNPUT_BACKEND=dummy
python -m robot_stack.suite \
  --backend robocasa --tasks PickPlaceCounterToSink \
  --episodes 1 --seed-start 1100 --max-steps 700 \
  --schedules examples/perturbations/robocasa-mobile-navigation.json \
              examples/perturbations/robocasa-mobile-arm.json \
              examples/perturbations/robocasa-mobile-gripper.json \
  --output outputs/kitchen-mobile
python -m robot_stack.audit outputs/kitchen-mobile \
  --output outputs/kitchen-mobile/audit.json
```

Each attempt saves the successful source, perturbed open-loop control and state-feedback recovery (when the source succeeds). Keep failures and inspect `qualified_correction`; running a schedule does not guarantee an induced failure or recovery.

| Example | Insertion point on the source demonstration | Physical intervention |
|---|---|---|
| navigation | First `navigate_sink` milestone | 20 native base yaw commands; preserve the grasp command |
| arm | 75% of source actions | 20 normalized Cartesian translation commands, direction `[1,0,0]`, strength 0.6 |
| gripper | First `object_lifted` milestone | Open the gripper for 25 control actions |

Milestones anchor to the source tape. Later events use nominal scheduled ticks; they are not live feedback predicates. `object_lifted` means object center more than 10 cm above its initial height. All intervention and recovery actions count toward the 700-action budget. At control frequency 20 Hz, one action is nominally 0.05 simulated seconds. Video frames may be subsampled.

The mobile adapter also accepts `base_translation`, `base_reverse`, `base_hold`, `gripper_close` and `arm_hold`. Arm translation directions are in the configured robot base controller reference frame. These are action perturbations, not object displacement or sensor corruption. Holds and small commands may not create an error.

## Reproducible initial scenes

This upstream version's `Counter.get_reset_regions` deduplicates XML elements with `set`, producing identity-dependent left/right ordering. The adapter sorts returned counter regions by geometry during initialization. It also deterministically seeds otherwise unseeded fixture RNGs within that scope. Both wrappers and external Python/NumPy RNG states are restored, including on errors; installed simulator files are not edited. Run environments serially or in isolated processes, not concurrent threads.

This defines a documented deterministic initialization variant; its seed-to-scene mapping can differ from unpatched upstream runs. It does not change region dimensions, placement constraints, physics, task success or recovery state. Each branch still resets by seed and physically re-executes the source prefix with the strict state comparison. Replay uses the same compatibility layer.

Independent replay example (use a saved branch path from your run):

```bash
python -m robot_stack.replay \
  outputs/kitchen-mobile/PickPlaceCounterToSink/robocasa-mobile-navigation/ep_1100/recovery.hdf5 \
  --output outputs/mobile-replay.json \
  --video outputs/mobile-recovery.mp4 --video-stride 3
```

The final fixed-scene experiment produced 3/3 qualified pairs; an earlier placement-policy version produced 2/3 and is retained. All three schedules share one initial scene and must stay in the same dataset split.

Rendering uses `mujoco.Renderer` with visual geometry only, with a fixed overview and the native robot camera side by side. Unlike upstream offscreen initialization, it does not introduce an extra dynamics forward call before reset. Independent replay still checks every integration state and success flag.

See the [live mobile experiment](https://pm1255.github.io/robot_stack/robocasa-mobile.html) for measured outcomes and retained failures, and the [per-benchmark perturbation reference](benchmark-perturbations.md) for units and recovery mechanisms.
