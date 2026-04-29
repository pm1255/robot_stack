from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def wait_frames(simulation_app, n: int):
    for _ in range(n):
        simulation_app.update()


def patch_a2d_articulation_root(stage, robot_path: str):
    """
    Same in-memory patch as run_repeated_pick_place.py.

    Some A2D USDs put ArticulationRootAPI on:
        /World/A2D/root_joint

    dynamic_control expects a root rigid body, so for replay/listing we move the
    articulation root API to:
        /World/A2D/base_link
    """
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
    print(
        f"[PATCH] old root: {old_root_path}, "
        f"valid={old_root.IsValid() if old_root else False}",
        flush=True,
    )
    print(
        f"[PATCH] new root: {new_root_path}, "
        f"valid={new_root.IsValid() if new_root else False}",
        flush=True,
    )

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


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # IMPORTANT:
    # Isaac / Omniverse modules must be imported only after AppLauncher creates SimulationApp.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    try:
        # Import Isaac/Omniverse-dependent modules only after SimulationApp is created.
        import omni.timeline
        from omni.isaac.dynamic_control import _dynamic_control

        from isaac_collector.runtime.load_scene import open_usd_stage

        print("[1] Open Isaac scene", flush=True)
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)

        robot_path = "/World/A2D"

        patch_a2d_articulation_root(stage, robot_path)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        wait_frames(simulation_app, 60)

        dc = _dynamic_control.acquire_dynamic_control_interface()

        candidates = [
            f"{robot_path}/base_link",
            robot_path,
            f"{robot_path}/root_joint",
        ]

        art = None
        for p in candidates:
            h = dc.get_articulation(p)
            ok = h is not None and not (isinstance(h, int) and h == 0)
            print(f"[ART] try articulation {p} -> ok={ok}, handle={h}", flush=True)

            if ok:
                art = h
                dc.wake_up_articulation(art)
                print(f"[ART] using articulation path: {p}", flush=True)
                break

        if art is None:
            raise RuntimeError("Cannot acquire A2D articulation")

        n = dc.get_articulation_dof_count(art)
        print("=" * 80, flush=True)
        print("[DOF] count:", n, flush=True)
        print("=" * 80, flush=True)

        for i in range(n):
            dof = dc.get_articulation_dof(art, i)
            name = dc.get_dof_name(dof)

            try:
                pos = float(dc.get_dof_position(dof))
            except Exception:
                pos = None

            print(f"{i:02d}  {name}  pos={pos}", flush=True)

        print("=" * 80, flush=True)
        print("[FILTER] possible gripper/finger/hand DOFs", flush=True)
        print("=" * 80, flush=True)

        keywords = [
            "gripper",
            "finger",
            "hand",
            "claw",
            "left_finger",
            "right_finger",
            "l_finger",
            "r_finger",
            "left",
            "right",
        ]

        for i in range(n):
            dof = dc.get_articulation_dof(art, i)
            name = dc.get_dof_name(dof)
            lname = name.lower()

            if any(k in lname for k in keywords):
                try:
                    pos = float(dc.get_dof_position(dof))
                except Exception:
                    pos = None

                print(f"{i:02d}  {name}  pos={pos}", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
