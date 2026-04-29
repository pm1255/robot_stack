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

    p.add_argument(
        "--dof-names",
        nargs="+",
        default=["right_Left_0_Joint", "right_Right_0_Joint"],
    )

    p.add_argument(
        "--open-positions",
        nargs="+",
        type=float,
        default=[0.0, -6.55],
        help="Open target for each DOF. Must match --dof-names length.",
    )

    p.add_argument(
        "--close-positions",
        nargs="+",
        type=float,
        default=[0.25, -0.28],
        help="Close target for each DOF. Must match --dof-names length.",
    )

    p.add_argument("--hold-frames", type=int, default=240)
    p.add_argument("--cycles", type=int, default=3)
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


def get_dof_map(dc, art):
    n = dc.get_articulation_dof_count(art)
    out = {}

    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        out[name] = dof

    return out


def read_positions(dc, dof_map, names):
    out = {}
    for name in names:
        dof = dof_map[name]
        try:
            out[name] = float(dc.get_dof_position(dof))
        except Exception:
            out[name] = None
    return out


def command_multi(simulation_app, dc, dof_map, names, targets, label, hold_frames):
    print("=" * 80, flush=True)
    print(f"[GRIPPER_MULTI] {label}", flush=True)

    before = read_positions(dc, dof_map, names)
    print("[GRIPPER_MULTI] before:", before, flush=True)

    for name, target in zip(names, targets):
        dof = dof_map[name]
        print(f"[GRIPPER_MULTI] set {name} -> {float(target)}", flush=True)
        dc.set_dof_position_target(dof, float(target))

    for _ in range(hold_frames):
        simulation_app.update()

    after = read_positions(dc, dof_map, names)
    print("[GRIPPER_MULTI] after:", after, flush=True)

    return {
        "label": label,
        "before": before,
        "after": after,
        "targets": {name: float(t) for name, t in zip(names, targets)},
    }


def main():
    args = parse_args()

    if len(args.dof_names) != len(args.open_positions):
        raise ValueError("--dof-names and --open-positions length mismatch")
    if len(args.dof_names) != len(args.close_positions):
        raise ValueError("--dof-names and --close-positions length mismatch")

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

        patch_a2d_articulation_root(stage, args.robot_path)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        wait_frames(simulation_app, 60)

        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = acquire_articulation(dc, args.robot_path)
        dof_map = get_dof_map(dc, art)

        print("[GRIPPER_MULTI] using DOFs:", args.dof_names, flush=True)
        for name in args.dof_names:
            if name not in dof_map:
                raise RuntimeError(f"Cannot find DOF: {name}")
            print(f"  {name}", flush=True)

        for i in range(args.cycles):
            print("\n" + "#" * 80, flush=True)
            print(f"[CYCLE {i + 1}/{args.cycles}]", flush=True)

            command_multi(
                simulation_app,
                dc,
                dof_map,
                args.dof_names,
                args.open_positions,
                label="open",
                hold_frames=args.hold_frames,
            )

            command_multi(
                simulation_app,
                dc,
                dof_map,
                args.dof_names,
                args.close_positions,
                label="close",
                hold_frames=args.hold_frames,
            )

        print("[DONE] multi-DOF gripper test finished", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
