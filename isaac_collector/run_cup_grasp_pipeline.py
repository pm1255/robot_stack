from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-usd",
        required=True,
        help="Path to the USD scene.",
    )
    parser.add_argument(
        "--cup-path",
        required=True,
        help="Prim path of the target cup.",
    )
    parser.add_argument(
        "--project-root",
        default="/home/pm/Desktop/Project/robot_stack",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/robot_pipeline/cup_grasp_run",
    )

    parser.add_argument(
        "--grasp-mode",
        choices=["mock", "real"],
        default="mock",
    )
    parser.add_argument(
        "--curobo-mode",
        choices=["mock", "real"],
        default="mock",
    )

    parser.add_argument(
        "--graspnet-checkpoint",
        default=None,
    )

    parser.add_argument(
        "--execution-mode",
        choices=["direct_cup_debug", "robot_trajectory"],
        default="direct_cup_debug",
        help=(
            "direct_cup_debug: move cup directly for pipeline visualization; "
            "robot_trajectory: execute real joint trajectory."
        ),
    )

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--frames-per-waypoint", type=int, default=80)

    args, _ = parser.parse_known_args()
    return args


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def execute_direct_cup_debug(
    *,
    simulation_app,
    stage,
    cup_path: str,
    trajectory_result: dict,
    frames_per_waypoint: int,
):
    from isaac_collector.runtime.load_scene import (
        get_prim_world_matrix,
        set_prim_world_matrix,
        interpolate_pose_translation,
        wait_frames,
    )

    waypoints = trajectory_result.get("cartesian_waypoints_world", [])
    if not waypoints:
        raise RuntimeError("No cartesian_waypoints_world in trajectory result.")

    current = get_prim_world_matrix(stage, cup_path)

    print("[EXEC] direct_cup_debug mode")
    print("[EXEC] This only visualizes the grasp pipeline by moving the cup.")
    print("[EXEC] It is not real robot joint control.")

    for idx, waypoint in enumerate(waypoints):
        target = np.asarray(waypoint, dtype=np.float64)
        poses = interpolate_pose_translation(
            current,
            target,
            steps=frames_per_waypoint,
        )

        print(f"[EXEC] Move cup to waypoint {idx + 1}/{len(waypoints)}")

        for pose in poses:
            set_prim_world_matrix(stage, cup_path, pose)
            simulation_app.update()

        current = target
        wait_frames(simulation_app, 10)


def execute_robot_trajectory(
    *,
    simulation_app,
    stage,
    trajectory_result: dict,
):
    """
    这里是后面接你真实 robot_adapter / manipulation_controller 的地方。

    真实 cuRobo worker 应该输出：
    - joint_names
    - positions
    - dt

    然后这里调用你的 ManipulationController 执行。
    """

    joint_names = trajectory_result.get("joint_names", [])
    positions = trajectory_result.get("positions", [])
    dt = trajectory_result.get("dt", 0.02)

    if not joint_names or not positions:
        raise RuntimeError(
            "trajectory_result does not contain joint trajectory. "
            "Use --execution-mode direct_cup_debug first, or implement real cuRobo worker."
        )

    print("[EXEC] robot_trajectory mode")
    print(f"[EXEC] joint_names: {joint_names}")
    print(f"[EXEC] num trajectory points: {len(positions)}")
    print(f"[EXEC] dt: {dt}")

    # 这里不能凭空猜你的 ManipulationController API。
    # 后面你真实接入时，应该类似这样：
    #
    # from isaac_collector.controllers.manipulation_controller import ManipulationController
    # controller = ManipulationController(stage=stage, ...)
    # controller.execute_joint_trajectory(
    #     joint_names=joint_names,
    #     positions=positions,
    #     dt=dt,
    # )
    #
    # 现在先明确报错，避免伪装成真实机器人抓取。
    raise NotImplementedError(
        "Please connect this function to your existing manipulation_controller.py API."
    )


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # IsaacLab / Omniverse 必须在 AppLauncher 后再 import omni / pxr。
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    try:
        from isaac_collector.runtime.load_scene import (
            open_usd_stage,
            get_prim_world_matrix,
            wait_frames,
        )
        from isaac_collector.ipc.graspnet_client import GraspNetClient
        from isaac_collector.ipc.curobo_client import CuRoboClient

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[1/5] Opening USD scene")
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)
        wait_frames(simulation_app, 30)

        print("[2/5] Reading cup pose")
        cup_pose_world = get_prim_world_matrix(stage, args.cup_path)

        print("[INFO] cup path:", args.cup_path)
        print("[INFO] cup pose world:")
        print(cup_pose_world)

        scene_state = {
            "scene_usd": str(Path(args.scene_usd).expanduser().resolve()),
            "cup_path": args.cup_path,
            "object_pose_world": cup_pose_world.tolist(),
        }
        save_json(output_dir / "scene_state.json", scene_state)

        print("[3/5] Calling GraspNet worker")
        grasp_client = GraspNetClient(
            project_root=str(project_root),
            mode=args.grasp_mode,
            checkpoint=args.graspnet_checkpoint,
        )

        grasp_request = {
            "scene_usd": scene_state["scene_usd"],
            "object_path": args.cup_path,
            "object_pose_world": cup_pose_world.tolist(),
        }

        grasp_result = grasp_client.predict(grasp_request)
        save_json(output_dir / "grasp_result.json", grasp_result)

        print("[INFO] grasp_result:")
        print(json.dumps(grasp_result, indent=2))

        print("[4/5] Calling cuRobo worker")
        curobo_client = CuRoboClient(
            project_root=str(project_root),
            mode=args.curobo_mode,
        )

        curobo_request = {
            "scene_state": scene_state,
            "grasp_result": grasp_result,
            "robot_state": {},
            "world_collision": {},
        }

        trajectory_result = curobo_client.plan(curobo_request)
        save_json(output_dir / "trajectory_result.json", trajectory_result)

        print("[INFO] trajectory_result:")
        print(json.dumps(trajectory_result, indent=2))

        print("[5/5] Executing result in Isaac")
        if args.execution_mode == "direct_cup_debug":
            execute_direct_cup_debug(
                simulation_app=simulation_app,
                stage=stage,
                cup_path=args.cup_path,
                trajectory_result=trajectory_result,
                frames_per_waypoint=args.frames_per_waypoint,
            )
        else:
            execute_robot_trajectory(
                simulation_app=simulation_app,
                stage=stage,
                trajectory_result=trajectory_result,
            )

        print("[DONE] Pipeline finished.")

        if args.keep_open and not args.headless:
            print("[INFO] keep-open enabled. Close the Isaac window to exit.")
            while simulation_app.is_running():
                simulation_app.update()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()