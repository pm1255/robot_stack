from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Guard for GB10 / Torch JIT NVRTC issue.
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene-name", default="mutilrooms")
    parser.add_argument("--robot-model", default="a2d")
    parser.add_argument("--ee-keyword", default="Link7_l")

    parser.add_argument(
        "--plan-json",
        default="/tmp/robot_pipeline/repeated_pick_place/ep_0000_pickup_plan.json",
    )

    parser.add_argument("--steps-per-point", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--open-gripper", action="store_true")
    parser.add_argument("--close-gripper", action="store_true")

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def load_plan(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Plan JSON not found: {p}")
    data = json.loads(p.read_text())
    if not data.get("success"):
        raise RuntimeError(f"Plan is not successful: status={data.get('status')}")
    if not data.get("positions"):
        raise RuntimeError("Plan has no positions.")
    if not data.get("joint_names"):
        raise RuntimeError("Plan has no joint_names.")
    return data


def build_full_trajectory_from_curobo_plan(adapter, plan: dict):
    """
    cuRobo plan contains only the joints in cuRobo config.
    IsaacLab A2D has more joints, including head and gripper joints.
    This function maps cuRobo joint trajectory into full IsaacLab joint vector.
    """
    device = adapter.robot.data.joint_pos.device
    dtype = adapter.robot.data.joint_pos.dtype

    plan_joint_names = list(plan["joint_names"])
    plan_positions = plan["positions"]

    q0 = adapter.get_joint_pos()  # [num_envs, num_joints]
    if q0.ndim != 2:
        raise RuntimeError(f"Unexpected q0 shape: {tuple(q0.shape)}")

    full_traj = []

    missing = []
    for name in plan_joint_names:
        if name not in adapter.joint_names:
            missing.append(name)

    if missing:
        print("[WARN] cuRobo plan joints missing in IsaacLab robot:", missing, flush=True)

    for step_idx, q_plan in enumerate(plan_positions):
        if len(q_plan) != len(plan_joint_names):
            raise RuntimeError(
                f"Step {step_idx}: len(position)={len(q_plan)} "
                f"!= len(joint_names)={len(plan_joint_names)}"
            )

        q_full = q0.clone()

        for name, value in zip(plan_joint_names, q_plan):
            if name not in adapter.joint_names:
                continue
            jid = adapter.joint_names.index(name)
            q_full[:, jid] = float(value)

        full_traj.append(q_full)

    traj = torch.stack(full_traj, dim=0).to(device=device, dtype=dtype)
    return traj


def main():
    args = parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        project_root = Path("/home/pm/Desktop/Project/robot_stack")
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from isaac_collector.runtime.scene_loader import (
            load_scene_with_robot,
            step_loaded_scene,
        )
        from isaac_collector.controllers import A2DRobotAdapter

        print("[1] Loading scene with A2D Articulation", flush=True)
        loaded = load_scene_with_robot(
            scene_name=args.scene_name,
            robot_model=args.robot_model,
            num_envs=1,
            device="cuda:0",
            register_task_objects=True,
            verbose=True,
        )

        print("[2] Creating A2DRobotAdapter", flush=True)
        adapter = A2DRobotAdapter(
            loaded.robot,
            ee_keyword=args.ee_keyword,
        )

        adapter.print_robot_info()

        print("[3] Loading cuRobo plan:", args.plan_json, flush=True)
        plan = load_plan(args.plan_json)

        print("[PLAN] source:", plan.get("source"), flush=True)
        print("[PLAN] num points:", len(plan.get("positions", [])), flush=True)
        print("[PLAN] plan joint_names:", plan.get("joint_names"), flush=True)

        print("[4] Building full IsaacLab joint trajectory", flush=True)
        traj = build_full_trajectory_from_curobo_plan(adapter, plan)
        print("[TRAJ] shape:", tuple(traj.shape), flush=True)

        print("[5] Settling scene", flush=True)
        step_loaded_scene(loaded, steps=args.settle_steps)

        if args.open_gripper:
            print("[6] Opening gripper", flush=True)
            adapter.open_gripper()
            step_loaded_scene(loaded, steps=60)

        print("[7] Executing trajectory on A2D", flush=True)
        adapter.follow_joint_trajectory(
            traj,
            step_fn=lambda steps: step_loaded_scene(loaded, steps=steps),
            steps_per_point=args.steps_per_point,
        )

        if args.close_gripper:
            print("[8] Closing gripper", flush=True)
            adapter.close_gripper()
            step_loaded_scene(loaded, steps=60)

        print("[DONE] cuRobo plan replay finished.", flush=True)
        step_loaded_scene(loaded, steps=args.settle_steps)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
