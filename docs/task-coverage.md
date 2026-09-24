# Task coverage, native planners and examples

Robot Stack aims to collect three related kinds of data for every supported task: successful source demonstrations, deliberate execution errors, and successful recovery from those errors. An adapter alone does not solve a task. The task needs a source policy, a native success oracle and a policy that can act from a disturbed state. The same collector, storage format, schedule and audit run across adapters; a separate collection pipeline is not required for every task.

[Search the live coverage table and 50 examples](https://pm1255.github.io/robot_stack/gallery.html) · [Machine-readable coverage](evidence/task-coverage.json) · [Schedule API](perturbations.md)

## What “all tasks” means

`robot_stack.inventory` reads the installed upstream task registry. `robot_stack.suite --tasks all` enumerates **installed supported experts** for MetaWorld or ManiSkill. These are intentionally separate sets: ManiSkill 3.0.1 registers 74 task IDs, while its Panda motion-planning examples provide 12 experts. RoboCasa 1.0.1 in this checkout lists 317 unique tasks in its atomic/composite dataset registry; that is a versioned inventory, not a claim about every task in every RoboCasa release. RoboTwin's source contains 50 matching task classes and experts. RoboDojo's checked-out public task definitions contain 54 classes; its eval-only interface does not supply a general expert policy.

The coverage report marks four observed properties separately: source success, effective perturbation, recovery success and qualified correction. A qualified pair additionally requires the perturbed open-loop control to fail. A green cell means at least one positive example in the linked reports; it is not a success rate, complete seed coverage or a standard benchmark score. Unsupported or untested tasks remain visible.

The current inventory is not a census of all robotics benchmarks. LIBERO, RLBench, CALVIN, Habitat and BEHAVIOR are examples of additional integrations that have not been validated here. Registering their names would not establish data-generation support.

## All MetaWorld task classes

```bash
pip install -e '.[metaworld]'
python -m robot_stack.inventory --backend metaworld --output outputs/metaworld-inventory.json
MUJOCO_GL=egl python -m robot_stack.suite \
  --backend metaworld --tasks all --episodes 1 --seed-start 600 \
  --schedules examples/perturbations/simple.json examples/perturbations/multi-error.json \
  --max-steps 500 --output outputs/metaworld-all
python -m robot_stack.audit outputs/metaworld-all --output outputs/metaworld-all-audit.json
```

The first sweep and separate tool-task follow-up jointly produced qualified samples for all 50 task classes. The README embeds one recovery animation per class. All 150 selected source/control/recovery trajectories passed independent native action replay, and the 50 gallery recoveries were replayed again for rendering with zero recorded state error. Failed attempts remain in the reports. [Experimental conditions](scheduled-results.md).

## ManiSkill: stream native planners into the same protocol

Use an independent Python 3.10 environment with the upstream assets, compatible Torch/CUDA/SAPIEN packages and a GPU-capable Vulkan runtime. A CPU physics backend still requires the native rendering libraries to load assets. The optional extra does not replace upstream platform setup.

```bash
pip install -e '.[maniskill]'
python -m robot_stack.inventory --backend maniskill --output outputs/maniskill-inventory.json
python -m robot_stack.suite --backend maniskill --tasks all --list

# A single task, source + deliberately disturbed control + replanned recovery
python -m robot_stack.collect --backend maniskill --task PlaceSphere-v1 \
  --episodes 1 --seed-start 800 --max-steps 700 \
  --adapter-options '{"native_horizon":1000,"max_replans":3}' \
  --schedule examples/perturbations/maniskill-joint.json --output outputs/sphere

# All 12 installed experts; this does not silently promise all 74 registered IDs
python -m robot_stack.suite --backend maniskill --tasks all \
  --episodes 2 --seed-start 810 --max-steps 1000 \
  --adapter-options '{"native_horizon":1200,"max_replans":4}' \
  --schedules examples/perturbations/maniskill-late.json --output outputs/maniskill-all
python -m robot_stack.audit outputs/maniskill-all --output outputs/maniskill-all-audit.json
```

`ActionStream` yields every action from the synchronous native planner before it can execute physics through the proxy. The planner's initial `reset()` receives the current observation. A second or late reset is rejected. After an intervention, the old planning stack is unwound and planning starts from the current physical state. The collection engine checks that this hook did not change the recorded state. Replanning is bounded and does not guarantee recovery.

The ManiSkill 3.0.1 PandaStick expert passes an obsolete visualization argument to its base constructor. A local function clone substitutes a compatible constructor only for that version; no installed upstream files or global planner classes are modified. `svgpathtools` is required by DrawSVG and is pinned in the extra. Initial errors remain in the pre-fix experiment report.

Perturbations support `joint_offset`, `gripper_open`, `gripper_close` and `joint_hold`. `joint_offset` adds a bounded absolute joint-target offset while opening the gripper; `joint_hold` holds arm targets with the gripper open. Gripper-only perturbations hold the arm. A stick embodiment rejects gripper perturbations. Features combine TCP position and robot joint positions, so the default error threshold is a numerical state-deviation criterion, not a pure metric distance.

The first four-task trial produced 4/4 successful sources and one qualified pair (PlaceSphere). The subsequent 24-attempt, 12-expert sweep produced 19 successful sources and eight qualified pairs, with four setup errors and one unsuccessful source. Ten distinct task classes generated successful sources; combined with the first trial, seven classes have qualified examples. The original sweep predates the drawing compatibility fixes. A later drawing follow-up reached a native mplib/pybind11 GIL error and a process crash; drawing-task recovery remains unverified. ManiSkill sweeps now isolate each attempt in a subprocess, retain crash/timeout records and continue with the remaining tasks. `--case-timeout` bounds each worker (default 900 seconds); other backends can opt in with `--isolate`. Isolated attempts save trajectories under `case_SEED/ep_SEED/`. [First trial](evidence/maniskill-smoke-summary.json), [full sweep](evidence/maniskill-all-summary.json), [58-file sweep audit](evidence/maniskill-all-audit.json). One selected triplet from each of the seven covered classes also passed **21/21 independent native replays with state error 0**. [Replay evidence](evidence/maniskill-seven-task-replay.json).

## RoboTwin: all native source experts and experimental recovery

Keep the upstream checkout/assets and its compatible SAPIEN 3.0.0b1 / cuRobo environment. Install this project and `greenlet==3.2.4` into an owned environment or a project-owned dependency directory. Do not mix ManiSkill's SAPIEN 3.0.3 packages into that environment.

```bash
# ROBOTWIN_ROOT points to an installed upstream checkout with assets.
python -m robot_stack.inventory --backend robotwin --root "$ROBOTWIN_ROOT" \
  --output outputs/robotwin-inventory.json
python -m benchmark_collector.robotwin --root "$ROBOTWIN_ROOT" \
  --tasks all --episodes 1 --seed-start 900 --output outputs/robotwin-all-source
```

For three-branch collection, supply the checkout path as JSON (replace the example absolute path):

```bash
python -m robot_stack.suite --backend robotwin --tasks stack_blocks_two \
  --episodes 2 --seed-start 100 --max-steps 15000 \
  --adapter-options '{"root":"/absolute/path/to/RoboTwin"}' \
  --schedules examples/perturbations/robotwin-carry.json \
              examples/perturbations/robotwin-first-carry.json \
  --output outputs/robotwin-carry
python -m robot_stack.audit outputs/robotwin-carry --output outputs/robotwin-carry-audit.json
```

The adapter yields at each native `scene.step()`, records motor targets, velocities, forces and gripper commands, and applies exactly one physical step per recorded action. Budgets are **physics steps**, not the downsampled frames in the original RoboTwin HDF5 files; 240 injected steps must not be compared directly with 15 ManiSkill control steps. Restarting `play_once()` from current poses is experimental: scripts can assume untouched objects or redo completed subgoals. Actor/articulation poses, velocities and joint positions are recorded, not a complete solver-internal SAPIEN snapshot. Independent replay is therefore required to test the practical determinism of a case.

The first full RoboTwin source sweep attempted all 50 tasks at seed 900: **37 successful sources, nine task failures and four errors**. This is an unfiltered fixed-seed result, including setup instability and planning errors. [All source records](evidence/robotwin-all-summary.json).

The initial stack_blocks_two trial collected a successful source, effective disturbance and successful replanning, with identical prefix/error states. Its control also succeeded, so it yielded **zero qualified correction pairs**. This result is retained. Investigation found that both arms share one articulation in ALOHA; the earlier motor layout overwrote the left intervention with the second full motor vector. The adapter now records one vector per unique articulation and preserves both arms' edits. Old duplicated-vector files can still replay the command actually applied by that version. The fixed four-attempt stack_blocks_two experiment (seeds 100–101, two schedules) produced **4/4 successful sources and 3/4 qualified correction pairs**; all 12 trajectory files passed audit. [Results](evidence/robotwin-carry-summary.json), [audit](evidence/robotwin-carry-audit.json). The seed 101 second-carry triplet independently replayed with state error 0 on all three branches; recovery took 7,748 native physics steps. [Replay evidence](evidence/robotwin-carry-replay.json). Python task-source discovery also uses Python's declared encoding, independent of native libraries changing the process locale. Fixed-version experiments are reported separately.

## Navigation, RoboDojo and MimicGen

RoboCasa currently has evidence for native NavigateKitchen; AI2-THOR has evidence for custom PointNav in FloorPlan1, including six sequential destinations and three interventions. Neither establishes general mobile manipulation, ObjectNav or complete scene/task coverage. [Navigation interfaces](navigation.md).

RoboDojo has an `EvalEnv` bridge and a real stack_bowls reset/physics smoke test. It still needs a capable policy and task-specific state/perturbation readers. Its initial `success=True` flag is not task-completion evidence. A source policy checkpoint or planner is a substantive dependency, not something the collector can invent.

MimicGen's native Lift pipeline generated ten trajectories from two demonstrations; seven succeeded and all ten independently replayed. These are augmentation results, not automatically correction pairs. New tasks require appropriate object frames, action conversion and subtask annotations. Recovery trajectories can have new retry branches and must be annotated accordingly before augmentation. [MimicGen commands](mimicgen.md).

The correction HDF5 files currently store state/action trajectories, per-step success, intervention phases and provenance. The gallery videos are separate visualizations; they do not imply that synchronized multi-camera images, depth and language labels are already present in every HDF5 file. Those modalities need an explicit native observation export for a particular training recipe.

## Rebuild coverage for your installation

```bash
python -m robot_stack.coverage \
  --inventories outputs/metaworld-inventory.json outputs/maniskill-inventory.json \
  --reports outputs/metaworld-all/summary.json outputs/maniskill-all/summary.json \
  --output outputs/task-coverage.json
```

Pass final summaries once, not both progress files and summaries from the same run. Registry-only entries never become verified just because they have an expert. Keep all attempts when calculating rates. The dashboard's green cells count task classes with evidence and are deliberately distinct from episode counts.
