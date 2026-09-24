# Navigation and simulator integration

## RoboCasa 1.0.1

`RoboCasaNavigationAdapter` uses the native `NavigateKitchen` task, PandaOmron's base action vector, and RoboCasa's own position/orientation success condition. It does not substitute a toy kitchen. The controller uses the native base Jacobian and bounded friction compensation. It does not change model friction, teleport the robot, or relax native success thresholds. It remains a target-pose controller, not a general collision-aware route planner.

Install the matching [official RoboCasa source and assets](https://robocasa.ai/docs/build/html/introduction/installation.html) in a dedicated environment. Do not mix the earlier MetaWorld MuJoCo pin into this environment.

```bash
MUJOCO_GL=egl PYNPUT_BACKEND=dummy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m robot_stack.collect --backend robocasa --task NavigateKitchen \
  --output outputs/kitchen-navigation --episodes 3 --seed-start 220 \
  --perturb-steps 20 --max-steps 500 \
  --adapter-options '{"layout":1,"style":1}'
```

The current server originally lacked Lightwheel fixtures; reset failed before any robot action. The official asset URL returned HTTP 404 on 2026-09-24. Both missing Lightwheel packs were installed into an isolated source directory after full SHA256 verification; their pinned provenance is in [asset-manifest.json](asset-manifest.json). Import success is not a passed reset, and a passed reset is not a successful navigation episode.

The original uncompensated controller failed seed 220 after 500 real actions. With the same layout/style, seed and budget, the corrected controller succeeded. A fixed three-seed check (220–222, layout/style 1, 20 perturbation actions, 500 total actions) then produced **3/3 qualified correction pairs**: all sources/recoveries succeeded and all open-loop perturbed controls failed. Prefix and paired error states matched exactly. Independent action replay of seed 220's source, perturbed and recovery trajectories had zero integration-state error and identical success labels (136/156/183 actions). Video rendering samples every five actions; replay still verifies every state. This is a three-seed smoke check, not an all-layout navigation benchmark.

## AI2-THOR 5.0.0

`AI2ThorNavigationAdapter` uses a native FloorPlan scene and CloudRendering. It obtains a privileged reachable grid, chooses a seeded distant goal, plans a grid path, and executes only movement/rotation actions. The error injection rotates the heading; recovery replans from the current agent pose. No teleportation occurs in the recorded trajectory.

```bash
pip install -e '.[ai2thor]'
AI2THOR_CACHE_DIR=/path/to/writable/thor-cache \
  python -m robot_stack.collect --backend ai2thor --task FloorPlan1 \
  --output outputs/thor-navigation --episodes 3 --seed-start 220 \
  --perturb-steps 3 --max-steps 160
```

Use a Linux container with the required Vulkan/GPU driver. Python installation alone does not install or validate the Unity executable. The pinned PyPI package uses official build `f0825767cd50d69f666c7f282e54abfe58f1e917`. The adapter allows a caller-selected cache directory or `executable_path` for a verified preinstalled build, and does not require runtime files in the repository. Official public downloads use HTTPS. On 2026-09-24, the evaluation server successfully resumed the official 835,983,275-byte archive and verified its full SHA256 from [asset-manifest.json](asset-manifest.json). A bounded single-4090 cctl job then produced **3/3 qualified correction pairs** in FloorPlan1, seeds 220–222, three perturbation rotations and a 160-action cap. All nine trajectories passed data audit; independent action replay of seed 220 recovery had zero agent-pose error and matching success labels. This demonstrates that the evaluation server can download and run the native build; the training server had a different TLS failure path.

The success rule is distance <= 0.10 m from the chosen reachable point. This is an explicitly defined oracle PointNav task, not AI2-THOR's official ObjectNav evaluation. Agent pose is sufficient for this static navigation replay check, but it is not a complete save/restore state for arbitrary object manipulation.

## RoboDojo bridge

Checked against official [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) commit `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`. The current upstream release separates simulator evaluation from XPolicyLab policies. We do not ship a universal RoboDojo expert or claim a successful native rollout.

A subsequent read-only inventory found a complete shared official RoboDojo checkout at the same upstream commit, including Robots, Object/RoboDojo, Eval_Layout/RoboDojo and Material assets, plus Python 3.11 / Isaac Sim 5.1 / IsaacLab 0.54.3. The assets are linked into a separate project-owned checkout; no third-party policy checkpoint is used. Initial GPU startup exposed a C++ ABI mismatch and duplicate Vulkan ICDs. Native task success remains unverified. `scripts/robodojo_native_probe.py` is a bounded reset/physics probe, not a task policy or a benchmark score.

From a running native Isaac application, create a real `EvalEnv` and supply it through `env_factory` to `RoboDojoAdapter`. Supply a real policy callable, numeric `state_reader`, relevant `feature_reader`, a physical `perturbation` callback, and the upstream commit. Then call `collect_triplet(...)`. Each factory instance must reproduce the same seeded scene. GPU/PhysX nondeterminism can invalidate strict prefix equivalence; do not relax tolerances without measured justification.

The bridge requires one environment and uses native `take_action`. It rejects silent no-op actions and refuses to continue after native termination. Success requires a completed native episode plus the native success flag and actual action execution; the initial True flag alone is never accepted.

## Honest support levels

- **Adapter implemented:** code matches an upstream interface, but may not have run native dynamics.
- **Reset/step verified:** assets load and actions execute; no task-success claim.
- **Collection verified:** fixed attempts, actual actions and native task outcomes are saved.
- **Correction verified:** source success, induced error, failing control, matching branch state and successful physical recovery are all demonstrated.

Keep these levels separate in issues, reports and PRs. Unit-test fixtures verify contracts, not simulator performance.
