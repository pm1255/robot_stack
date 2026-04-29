from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    p.add_argument("--scene-usd", required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--robot-path", default="/World/A2D")
    p.add_argument(
        "--table-path",
        default="/World/office1/Room_seed123_idx000/furniture/tea_table",
    )
    p.add_argument(
        "--cup-path",
        default="/World/office1/Room_seed123_idx000/furniture/tea_table/cup",
    )
    return p.parse_args()


def wait_frames(simulation_app, n: int):
    for _ in range(n):
        simulation_app.update()


def print_pose(stage, label, path):
    from isaac_collector.runtime.load_scene import get_prim_world_matrix

    m = np.asarray(get_prim_world_matrix(stage, path), dtype=float)
    xyz = m[3, 0:3].copy()

    print("=" * 80, flush=True)
    print(label, flush=True)
    print(path, flush=True)
    print(m, flush=True)
    print("translation:", xyz.tolist(), flush=True)

    return m


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    try:
        import omni.timeline
        from isaac_collector.runtime.load_scene import (
            open_usd_stage,
            get_prim_world_matrix,
            set_prim_world_matrix,
        )

        print("[1] open scene", flush=True)
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)

        print_pose(stage, "[POSE] initial robot", args.robot_path)
        print_pose(stage, "[POSE] initial table", args.table_path)
        print_pose(stage, "[POSE] initial cup", args.cup_path)

        print("\n[2] apply bringup robot pose only", flush=True)
        robot_world = np.asarray(get_prim_world_matrix(stage, args.robot_path), dtype=float)
        robot_world[3, 0] = 2.827275532134659
        robot_world[3, 1] = -2.5853563806447653
        robot_world[3, 2] = -0.039999272674322135
        set_prim_world_matrix(stage, args.robot_path, robot_world)
        simulation_app.update()

        print_pose(stage, "[POSE] after bringup robot", args.robot_path)
        print_pose(stage, "[POSE] after bringup table", args.table_path)
        print_pose(stage, "[POSE] after bringup cup", args.cup_path)

        print("\n[3] play timeline and wait", flush=True)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        wait_frames(simulation_app, 120)

        print_pose(stage, "[POSE] after timeline robot", args.robot_path)
        print_pose(stage, "[POSE] after timeline table", args.table_path)
        print_pose(stage, "[POSE] after timeline cup", args.cup_path)

        print("[DONE] pose check finished", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
