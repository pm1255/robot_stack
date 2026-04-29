from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    p.add_argument("--scene-usd", required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--robot-path", default="/World/A2D")

    # Start conservative. We only test right_hand_joint1 first.
    p.add_argument("--dof-name", default="right_hand_joint1")
    p.add_argument("--open-position", type=float, default=0.0)
    p.add_argument("--close-position", type=float, default=0.7)
    p.add_argument("--hold-frames", type=int, default=120)
    p.add_argument("--cycles", type=int, default=2)
    return p.parse_args()


def wait_frames(simulation_app, n: int):
    for _ in range(n):
        simulation_app.update()


def patch_a2d_articulation_root(stage, robot_path: str):
    from pxr import UsdPhysics

    try:
        from pxr import PhysxSchema
    except Exception:
        PhysxSchema = None

    old_root_path = f"{robot_path}/root_joint"
    new_root_path = f"{robot_path}/base_link"

    old_root = stage.GetPrimAtPath(old_root_path)
    new_root = stage.GetPrimAtPath(new_root_path)

    print("[PATCH] articulation root check", flush=True)
    print(f"[PATCH] old root: {old_root_path}, valid={old_root.IsValid() if old_root else False}", flush=True)
    print(f"[PATCH] new root: {new_root_path}, valid={new_root.IsValid() if new_root else False}", flush=True)

    if not new_root or not new_root.IsValid():
        raise RuntimeError(f"Cannot find A2D base_link: {new_root_path}")

    if old_root and old_root.IsValid():
        try:
            if old_root.HasAPI(UsdPhysics.ArticulationRootAPI):
                old_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[PATCH] removed ArticulationRootAPI from {old_root_path}", flush=True)
        except Exception as e:
            print(f"[PATCH][WARN] remove old ArticulationRootAPI failed: {e!r}", flush=True)

        if PhysxSchema is not None:
            try:
                if old_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    old_root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                    print(f"[PATCH] removed PhysxArticulationAPI from {old_root_path}", flush=True)
            except Exception as e:
                print(f"[PATCH][WARN] remove old PhysxArticulationAPI failed: {e!r}", flush=True)

    if not new_root.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(new_root)
        print(f"[PATCH] applied ArticulationRootAPI to {new_root_path}", flush=True)

    if PhysxSchema is not None:
        if not new_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(new_root)
            print(f"[PATCH] applied PhysxArticulationAPI to {new_root_path}", flush=True)


def acquire_articulation(dc, robot_path: str):
    candidates = [
        f"{robot_path}/base_link",
        robot_path,
        f"{robot_path}/root_joint",
    ]

    for p in candidates:
        art = dc.get_articulation(p)
        ok = art is not None and not (isinstance(art, int) and art == 0)
        print(f"[ART] try articulation {p} -> ok={ok}, handle={art}", flush=True)

        if ok:
            dc.wake_up_articulation(art)
            print(f"[ART] using articulation path: {p}", flush=True)
            return art

    raise RuntimeError(f"Cannot acquire articulation under {robot_path}")


def find_dof_by_name(dc, art, target_name: str):
    n = dc.get_articulation_dof_count(art)

    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        if name == target_name:
            print(f"[DOF] found {target_name} at index {i}, handle={dof}", flush=True)
            return dof

    print("[DOF] available names:", flush=True)
    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        print(f"{i:02d}  {dc.get_dof_name(dof)}", flush=True)

    raise RuntimeError(f"Cannot find DOF: {target_name}")


def command_dof(simulation_app, dc, dof, name: str, target: float, hold_frames: int):
    print(f"[GRIPPER] command {name} -> target={target}", flush=True)

    try:
        before = float(dc.get_dof_position(dof))
    except Exception:
        before = None

    dc.set_dof_position_target(dof, float(target))

    for _ in range(hold_frames):
        simulation_app.update()

    try:
        after = float(dc.get_dof_position(dof))
    except Exception:
        after = None

    print(f"[GRIPPER] {name} before={before}, after={after}", flush=True)


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
        from omni.isaac.dynamic_control import _dynamic_control
        from isaac_collector.runtime.load_scene import open_usd_stage

        print("[1] Open Isaac scene", flush=True)
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)
        from isaac_collector.runtime.load_scene import get_prim_world_matrix, set_prim_world_matrix

        robot_world = get_prim_world_matrix(stage, args.robot_path)
        robot_world[3, 0] = 2.827275532134659
        robot_world[3, 1] = -2.5853563806447653
        robot_world[3, 2] = -0.039999272674322135

        set_prim_world_matrix(stage, args.robot_path, robot_world)
        simulation_app.update()

        print("[BRINGUP] moved A2D to known pick-place bringup pose", flush=True)
        print(robot_world, flush=True)

        patch_a2d_articulation_root(stage, args.robot_path)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        wait_frames(simulation_app, 60)

        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = acquire_articulation(dc, args.robot_path)

        dof = find_dof_by_name(dc, art, args.dof_name)

        print("[TEST] start gripper open/close cycles", flush=True)

        for i in range(args.cycles):
            print("=" * 80, flush=True)
            print(f"[CYCLE {i + 1}/{args.cycles}] open", flush=True)
            command_dof(
                simulation_app,
                dc,
                dof,
                args.dof_name,
                args.open_position,
                args.hold_frames,
            )

            print("=" * 80, flush=True)
            print(f"[CYCLE {i + 1}/{args.cycles}] close", flush=True)
            command_dof(
                simulation_app,
                dc,
                dof,
                args.dof_name,
                args.close_position,
                args.hold_frames,
            )

        print("[DONE] gripper test finished", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
