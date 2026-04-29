"""
Inspect loaded scene, registered objects, A2D joint names and body names.

Run from your project root:
    cd /home/pm/Desktop/Project
    python ./isaac_collector/scripts/inspect_loaded_scene.py \
      --scene-name mutilrooms_fixed \
      --robot-model a2d \
      --ee-keyword gripper

If --ee-keyword fails, try:
    --ee-keyword hand
    --ee-keyword wrist
    --ee-keyword tcp
"""

import argparse
import sys
from pathlib import Path


"""
Inspect loaded scene, registered objects, A2D joint names and body names.
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before importing IsaacLab / torch-heavy modules.
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")

try:
    import torch

    for name, value in [
        ("_jit_set_nvfuser_enabled", False),
        ("_jit_override_can_fuse_on_gpu", False),
        ("_jit_set_texpr_fuser_enabled", False),
        ("_jit_set_profiling_executor", False),
        ("_jit_set_profiling_mode", False),
    ]:
        fn = getattr(torch._C, name, None)
        if fn is not None:
            try:
                fn(value)
            except Exception:
                pass

    try:
        torch.jit._state.disable()
    except Exception:
        pass

    print("[GUARD] Torch JIT CUDA fusers disabled", flush=True)
except Exception as e:
    print("[WARN] Torch JIT guard failed:", repr(e), flush=True)

from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser()
parser.add_argument("--scene-name", type=str, default="mutilrooms_fixed")
parser.add_argument("--robot-model", type=str, default="a2d")
parser.add_argument("--ee-keyword", type=str, default="gripper")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Make imports work from robot_stack/scripts/debug.
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaac_collector.runtime.scene_loader import load_scene_with_robot
from isaac_collector.controllers import (
    A2DRobotAdapter,
    ManipulationController,
    RuleBasedGraspGenerator,
    ManualJointPlanner,
)



def main():
    loaded = load_scene_with_robot(
        scene_name=args.scene_name,
        robot_model=args.robot_model,
        num_envs=1,
        device="cuda:0",
        register_task_objects=True,
        verbose=True,
    )

    adapter = A2DRobotAdapter(
        loaded.robot,
        ee_keyword=args.ee_keyword,
    )
    planner = ManualJointPlanner(robot_adapter=adapter)
    grasp_generator = RuleBasedGraspGenerator()

    controller = ManipulationController(
        loaded=loaded,
        robot_adapter=adapter,
        grasp_generator=grasp_generator,
        motion_planner=planner,
        grasp_mode="attach",
    )

    adapter.print_robot_info()
    controller.print_objects()

    print("\n[INFO] Inspection complete. Close the window or wait a few seconds.")
    controller.step(steps=120)


if __name__ == "__main__":
    main()
    simulation_app.close()
