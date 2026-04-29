from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    p.add_argument("--scene-usd", required=True)
    p.add_argument(
        "--robot-config",
        default="/home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody_ignore_adjacent.yml",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--robot-path", default="/World/A2D")

    p.add_argument("--dof-name", default="right_Left_0_Joint")
    p.add_argument("--open-position", type=float, default=0.0)
    p.add_argument("--close-position", type=float, default=0.25)
    p.add_argument("--hold-frames", type=int, default=300)
    p.add_argument("--cycles", type=int, default=5)
    return p.parse_args()


def wait_frames(simulation_app, n: int):
    for _ in range(n):
        simulation_app.update()


def load_retract_robot_state(robot_config_path: str):
    cfg = yaml.safe_load(Path(robot_config_path).read_text(encoding="utf-8"))
    cspace = cfg["robot_cfg"]["kinematics"]["cspace"]
    return {
        "joint_names": list(cspace["joint_names"]),
        "positions": [float(x) for x in cspace["retract_config"]],
    }


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


def read_dof(dc, dof):
    try:
        return float(dc.get_dof_position(dof))
    except Exception:
        return None


def reset_retract_joints(simulation_app, dc, dof_map, robot_state, hold_frames=120):
    print("[RESET] reset cuRobo-controlled joints to retract_config", flush=True)

    for name, q in zip(robot_state["joint_names"], robot_state["positions"]):
        if name not in dof_map:
            print(f"[RESET][WARN] joint not found in USD DOF map: {name}", flush=True)
            continue

        dof = dof_map[name]
        q = float(q)

        try:
            dc.set_dof_position(dof, q)
        except Exception as e:
            print(f"[RESET][WARN] set_dof_position failed for {name}: {e!r}", flush=True)

        dc.set_dof_position_target(dof, q)

    for _ in range(hold_frames):
        simulation_app.update()

    print("[RESET] retract reset done", flush=True)


def build_freeze_targets(dc, dof_map, exclude_names):
    freeze_targets = {}

    for name, dof in dof_map.items():
        if name in exclude_names:
            continue

        pos = read_dof(dc, dof)
        if pos is not None:
            freeze_targets[name] = float(pos)

    print(f"[FREEZE] freezing {len(freeze_targets)} non-command DOFs", flush=True)
    return freeze_targets


def apply_freeze_targets(dc, dof_map, freeze_targets):
    for name, q in freeze_targets.items():
        dc.set_dof_position_target(dof_map[name], float(q))


def command_gripper_isolated(
    *,
    simulation_app,
    dc,
    dof_map,
    command_name: str,
    dof_name: str,
    target: float,
    freeze_targets,
    hold_frames: int,
):
    dof = dof_map[dof_name]

    before = read_dof(dc, dof)
    print("=" * 80, flush=True)
    print(f"[GRIPPER_ISOLATED] {command_name}: {dof_name} -> {target}", flush=True)
    print(f"[GRIPPER_ISOLATED] before={before}", flush=True)

    for _ in range(hold_frames):
        apply_freeze_targets(dc, dof_map, freeze_targets)
        dc.set_dof_position_target(dof, float(target))
        simulation_app.update()

    after = read_dof(dc, dof)
    print(f"[GRIPPER_ISOLATED] after={after}", flush=True)


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
        from isaac_collector.runtime.load_scene import (
            open_usd_stage,
            get_prim_world_matrix,
            set_prim_world_matrix,
        )

        print("[1] Open Isaac scene", flush=True)
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)

        # Match the main pick-place script bringup robot root pose.
        robot_world = get_prim_world_matrix(stage, args.robot_path)
        robot_world[3, 0] = 2.827275532134659
        robot_world[3, 1] = -2.5853563806447653
        robot_world[3, 2] = -0.039999272674322135
        set_prim_world_matrix(stage, args.robot_path, robot_world)
        simulation_app.update()
        print("[BRINGUP] moved A2D to known pick-place bringup pose", flush=True)

        patch_a2d_articulation_root(stage, args.robot_path)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        wait_frames(simulation_app, 60)

        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = acquire_articulation(dc, args.robot_path)
        dof_map = get_dof_map(dc, art)

        if args.dof_name not in dof_map:
            raise RuntimeError(f"Cannot find DOF: {args.dof_name}")

        robot_state = load_retract_robot_state(args.robot_config)
        reset_retract_joints(
            simulation_app,
            dc,
            dof_map,
            robot_state,
            hold_frames=120,
        )

        freeze_targets = build_freeze_targets(
            dc,
            dof_map,
            exclude_names={args.dof_name},
        )

        print("[TEST] start isolated gripper cycles", flush=True)

        for i in range(args.cycles):
            print("\n" + "#" * 80, flush=True)
            print(f"[CYCLE {i + 1}/{args.cycles}]", flush=True)

            command_gripper_isolated(
                simulation_app=simulation_app,
                dc=dc,
                dof_map=dof_map,
                command_name="open",
                dof_name=args.dof_name,
                target=args.open_position,
                freeze_targets=freeze_targets,
                hold_frames=args.hold_frames,
            )

            command_gripper_isolated(
                simulation_app=simulation_app,
                dc=dc,
                dof_map=dof_map,
                command_name="close",
                dof_name=args.dof_name,
                target=args.close_position,
                freeze_targets=freeze_targets,
                hold_frames=args.hold_frames,
            )

        print("[DONE] isolated gripper test finished", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
