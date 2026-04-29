#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    parser.add_argument(
        "--scene-usd",
        default="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd",
    )
    parser.add_argument("--robot-path", default="/World/A2D")

    parser.add_argument(
        "--curobo-robot-config",
        default="/home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody_ignore_adjacent.yml",
    )

    parser.add_argument(
        "--pickup-plan",
        default="/tmp/robot_pipeline/repeated_pick_place/ep_0000_pickup_plan.json",
    )
    parser.add_argument(
        "--place-plan",
        default="/tmp/robot_pipeline/repeated_pick_place/ep_0000_place_plan.json",
    )

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-open", action="store_true")

    parser.add_argument("--load-wait", type=int, default=120)
    parser.add_argument("--pre-wait", type=int, default=30)
    parser.add_argument("--post-wait", type=int, default=60)

    parser.add_argument(
        "--steps-per-position",
        type=int,
        default=1,
        help="How many Isaac updates to hold each cuRobo waypoint.",
    )
    parser.add_argument(
        "--speed-stride",
        type=int,
        default=1,
        help="Use every N-th cuRobo waypoint. 1 means full trajectory.",
    )

    parser.add_argument(
        "--move-robot-to-bringup-pose",
        action="store_true",
        help="Move /World/A2D to the same known bring-up pose used in run_repeated_pick_place.py.",
    )

    args, _ = parser.parse_known_args()
    return args


def load_json(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_curobo_joint_names(robot_config_path: str | Path) -> List[str]:
    path = Path(robot_config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cuRobo robot config not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cspace = cfg["robot_cfg"]["kinematics"]["cspace"]
    return list(cspace["joint_names"])


def load_plan_positions(plan_path: str | Path) -> np.ndarray:
    plan = load_json(plan_path)

    if not plan.get("success", False):
        raise RuntimeError(f"Plan is not successful: {plan_path}\nPlan: {plan}")

    positions = plan.get("positions", None)
    if positions is None:
        raise KeyError(f"Plan has no 'positions': {plan_path}")

    arr = np.asarray(positions, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected positions shape [T, J], got {arr.shape}")

    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"Empty positions in plan: {arr.shape}")

    return arr


def get_prim_world_matrix(stage, prim_path: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    return np.array(mat, dtype=np.float64)


def set_prim_world_matrix(stage, prim_path: str, mat: np.ndarray):
    """
    Simple bring-up setter for Xform prims.
    Uses the project's existing runtime helper if available.
    """
    try:
        from isaac_collector.runtime.load_scene import set_prim_world_matrix as project_set

        project_set(stage, prim_path, mat)
        return
    except Exception:
        pass

    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op = xform.AddTransformOp()
    op.Set(Gf.Matrix4d(mat.tolist()))


def move_a2d_close_to_cup_for_bringup(stage, simulation_app, robot_path: str):
    """
    Same fixed robot placement used in run_repeated_pick_place.py.
    This makes the replay scene consistent with the planning scene.
    """
    robot_world = get_prim_world_matrix(stage, robot_path)
    robot_world = np.asarray(robot_world, dtype=np.float64)

    robot_world[3, 0] = 2.827275532134659
    robot_world[3, 1] = -2.5853563806447653
    robot_world[3, 2] = -0.039999272674322135

    set_prim_world_matrix(stage, robot_path, robot_world)

    for _ in range(10):
        simulation_app.update()

    print("[SETUP] moved A2D to bring-up pose", flush=True)
    print("[SETUP] robot translation row:", robot_world[3, 0:3], flush=True)



def patch_a2d_articulation_root(stage, robot_path: str):
    """
    Some exported A2D USDs put PhysicsArticulationRootAPI on:
        /World/A2D/root_joint

    In Isaac Sim / dynamic_control, this can fail because root_joint is a
    PhysicsFixedJoint, not the root rigid body. For replay bring-up, move the
    articulation root API to:
        /World/A2D/base_link

    This is an in-memory patch only. It does not modify the source USD file.
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
    print(f"[PATCH] old root: {old_root_path}, valid={old_root.IsValid() if old_root else False}", flush=True)
    print(f"[PATCH] new root: {new_root_path}, valid={new_root.IsValid() if new_root else False}", flush=True)

    if not new_root or not new_root.IsValid():
        print("[PATCH] base_link not found; skip articulation root patch", flush=True)
        return

    # Remove APIs from the old fixed joint root if possible.
    if old_root and old_root.IsValid():
        try:
            if old_root.HasAPI(UsdPhysics.ArticulationRootAPI):
                old_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[PATCH] removed UsdPhysics.ArticulationRootAPI from {old_root_path}", flush=True)
        except Exception as e:
            print(f"[PATCH][WARN] failed to remove ArticulationRootAPI from old root: {e!r}", flush=True)

        if PhysxSchema is not None:
            try:
                if old_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    old_root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                    print(f"[PATCH] removed PhysxArticulationAPI from {old_root_path}", flush=True)
            except Exception as e:
                print(f"[PATCH][WARN] failed to remove PhysxArticulationAPI from old root: {e!r}", flush=True)

    # Apply APIs to the root rigid body.
    try:
        if not new_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(new_root)
            print(f"[PATCH] applied UsdPhysics.ArticulationRootAPI to {new_root_path}", flush=True)
        else:
            print(f"[PATCH] {new_root_path} already has UsdPhysics.ArticulationRootAPI", flush=True)
    except Exception as e:
        print(f"[PATCH][WARN] failed to apply ArticulationRootAPI to base_link: {e!r}", flush=True)

    if PhysxSchema is not None:
        try:
            if not new_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(new_root)
                print(f"[PATCH] applied PhysxArticulationAPI to {new_root_path}", flush=True)
            else:
                print(f"[PATCH] {new_root_path} already has PhysxArticulationAPI", flush=True)
        except Exception as e:
            print(f"[PATCH][WARN] failed to apply PhysxArticulationAPI to base_link: {e!r}", flush=True)

    print("[PATCH] articulation root patch finished", flush=True)



def wait_frames(simulation_app, n: int):
    for _ in range(n):
        simulation_app.update()


def acquire_dynamic_control():
    try:
        from omni.isaac.dynamic_control import _dynamic_control
    except Exception:
        from isaacsim.core.api import SimulationContext  # noqa: F401
        from omni.isaac.dynamic_control import _dynamic_control

    return _dynamic_control.acquire_dynamic_control_interface()


def get_articulation_handle(dc, robot_path: str):
    """
    Try several likely articulation handles.

    For A2D, after patching, the expected working path is:
        /World/A2D/base_link
    """
    candidates = [
        f"{robot_path}/base_link",
        robot_path,
        f"{robot_path}/root_joint",
    ]

    last_art = None

    for p in candidates:
        try:
            art = dc.get_articulation(p)
            last_art = art
            ok = art is not None and not (isinstance(art, int) and art == 0)

            print(f"[DYNAMIC_CONTROL] try articulation {p} -> ok={ok}, handle={art}", flush=True)

            if ok:
                dc.wake_up_articulation(art)
                print(f"[DYNAMIC_CONTROL] using articulation path: {p}", flush=True)
                return art

        except Exception as e:
            print(f"[DYNAMIC_CONTROL] try articulation {p} -> ERROR: {e!r}", flush=True)

    raise RuntimeError(
        f"Failed to get articulation for candidates under {robot_path}. "
        f"Last handle={last_art}. "
        f"This means the USD articulation is still not recognized by dynamic_control."
    )


def get_articulation_dof_names_and_handles(dc, art) -> Tuple[List[str], Dict[str, object]]:
    n = dc.get_articulation_dof_count(art)

    names: List[str] = []
    handles: Dict[str, object] = {}

    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        names.append(name)
        handles[name] = dof

    return names, handles


def normalize_name(s: str) -> str:
    return s.lower().replace("/", "_").replace(":", "_")


def build_joint_mapping(
    *,
    curobo_joint_names: List[str],
    usd_dof_names: List[str],
    usd_dof_handles: Dict[str, object],
) -> Dict[str, object]:
    """
    Map cuRobo cspace joint names to USD articulation DOF handles.

    Priority:
      1. exact name match
      2. normalized exact match
      3. suffix match
      4. contains match

    If this fails, print all USD DOFs so we can patch the aliases.
    """
    mapping: Dict[str, object] = {}

    norm_to_usd = {normalize_name(n): n for n in usd_dof_names}

    for cj in curobo_joint_names:
        if cj in usd_dof_handles:
            mapping[cj] = usd_dof_handles[cj]
            continue

        norm_cj = normalize_name(cj)
        if norm_cj in norm_to_usd:
            usd_name = norm_to_usd[norm_cj]
            mapping[cj] = usd_dof_handles[usd_name]
            continue

        suffix_candidates = [
            n for n in usd_dof_names
            if normalize_name(n).endswith(norm_cj) or norm_cj.endswith(normalize_name(n))
        ]
        if len(suffix_candidates) == 1:
            usd_name = suffix_candidates[0]
            mapping[cj] = usd_dof_handles[usd_name]
            continue

        contains_candidates = [
            n for n in usd_dof_names
            if norm_cj in normalize_name(n) or normalize_name(n) in norm_cj
        ]
        if len(contains_candidates) == 1:
            usd_name = contains_candidates[0]
            mapping[cj] = usd_dof_handles[usd_name]
            continue

        print("\n[ERROR] Cannot map cuRobo joint:", cj, flush=True)
        print("[ERROR] USD articulation DOF names:", flush=True)
        for i, n in enumerate(usd_dof_names):
            print(f"  [{i:02d}] {n}", flush=True)
        raise RuntimeError(f"Cannot map cuRobo joint to USD DOF: {cj}")

    return mapping


def set_joint_targets(
    *,
    dc,
    joint_mapping: Dict[str, object],
    curobo_joint_names: List[str],
    q: np.ndarray,
):
    q = np.asarray(q, dtype=np.float32)

    if q.shape[0] != len(curobo_joint_names):
        raise ValueError(
            f"q length mismatch: q={q.shape[0]}, joint_names={len(curobo_joint_names)}"
        )

    for j, name in enumerate(curobo_joint_names):
        dof = joint_mapping[name]
        value = float(q[j])
        dc.set_dof_position_target(dof, value)


def replay_plan(
    *,
    name: str,
    simulation_app,
    dc,
    joint_mapping: Dict[str, object],
    curobo_joint_names: List[str],
    positions: np.ndarray,
    steps_per_position: int,
    speed_stride: int,
):
    if speed_stride < 1:
        speed_stride = 1

    positions = positions[::speed_stride]

    print("\n" + "=" * 80, flush=True)
    print(f"[REPLAY] {name}", flush=True)
    print("=" * 80, flush=True)
    print(f"[REPLAY] positions shape: {positions.shape}", flush=True)
    print(f"[REPLAY] first q: {positions[0].tolist()}", flush=True)
    print(f"[REPLAY] last  q: {positions[-1].tolist()}", flush=True)

    for t, q in enumerate(positions):
        set_joint_targets(
            dc=dc,
            joint_mapping=joint_mapping,
            curobo_joint_names=curobo_joint_names,
            q=q,
        )

        for _ in range(steps_per_position):
            simulation_app.update()

        if t % 25 == 0 or t == positions.shape[0] - 1:
            print(f"[REPLAY] {name}: {t + 1}/{positions.shape[0]}", flush=True)

    print(f"[DONE] replay finished: {name}", flush=True)


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    try:
        try:
            from isaac_collector.runtime.load_scene import open_usd_stage
        except Exception:
            open_usd_stage = None

        if open_usd_stage is not None:
            print("[1] Open USD stage with project helper", flush=True)
            stage = open_usd_stage(args.scene_usd, simulation_app, wait=args.load_wait)
        else:
            print("[1] Open USD stage with omni.usd", flush=True)
            import omni.usd

            omni.usd.get_context().open_stage(args.scene_usd)
            wait_frames(simulation_app, args.load_wait)
            stage = omni.usd.get_context().get_stage()

        if args.move_robot_to_bringup_pose:
            move_a2d_close_to_cup_for_bringup(
                stage=stage,
                simulation_app=simulation_app,
                robot_path=args.robot_path,
            )

        print("[2] Load cuRobo joint names", flush=True)
        curobo_joint_names = load_curobo_joint_names(args.curobo_robot_config)
        print("[INFO] cuRobo joint names:", curobo_joint_names, flush=True)

        print("[3] Load saved cuRobo plans", flush=True)
        pickup_positions = load_plan_positions(args.pickup_plan)
        place_positions = load_plan_positions(args.place_plan)

        print("[PLAN] pickup positions:", pickup_positions.shape, flush=True)
        print("[PLAN] place  positions:", place_positions.shape, flush=True)

        if pickup_positions.shape[1] != len(curobo_joint_names):
            raise RuntimeError(
                f"pickup plan width {pickup_positions.shape[1]} != "
                f"num cuRobo joints {len(curobo_joint_names)}"
            )

        if place_positions.shape[1] != len(curobo_joint_names):
            raise RuntimeError(
                f"place plan width {place_positions.shape[1]} != "
                f"num cuRobo joints {len(curobo_joint_names)}"
            )

        print("[4] Patch articulation root and acquire A2D articulation", flush=True)
        patch_a2d_articulation_root(stage, args.robot_path)

        print("[5] Start timeline and acquire A2D articulation", flush=True)
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        wait_frames(simulation_app, args.pre_wait)

        dc = acquire_dynamic_control()
        art = get_articulation_handle(dc, args.robot_path)
        usd_dof_names, usd_dof_handles = get_articulation_dof_names_and_handles(dc, art)

        print("[INFO] USD articulation DOF count:", len(usd_dof_names), flush=True)
        print("[INFO] USD articulation DOF names:", flush=True)
        for i, n in enumerate(usd_dof_names):
            print(f"  [{i:02d}] {n}", flush=True)

        print("[6] Build cuRobo joint → USD DOF mapping", flush=True)
        joint_mapping = build_joint_mapping(
            curobo_joint_names=curobo_joint_names,
            usd_dof_names=usd_dof_names,
            usd_dof_handles=usd_dof_handles,
        )

        print("[MAPPING] cuRobo → USD DOF", flush=True)
        for name in curobo_joint_names:
            dof = joint_mapping[name]
            usd_name = dc.get_dof_name(dof)
            print(f"  {name}  ->  {usd_name}", flush=True)

        if args.dry_run:
            print("[DRY_RUN] Mapping succeeded. No replay executed.", flush=True)
            return

        print("[7] Replay pickup plan on A2D", flush=True)
        replay_plan(
            name="pickup",
            simulation_app=simulation_app,
            dc=dc,
            joint_mapping=joint_mapping,
            curobo_joint_names=curobo_joint_names,
            positions=pickup_positions,
            steps_per_position=args.steps_per_position,
            speed_stride=args.speed_stride,
        )

        wait_frames(simulation_app, 30)

        print("[8] Replay place plan on A2D", flush=True)
        replay_plan(
            name="place",
            simulation_app=simulation_app,
            dc=dc,
            joint_mapping=joint_mapping,
            curobo_joint_names=curobo_joint_names,
            positions=place_positions,
            steps_per_position=args.steps_per_position,
            speed_stride=args.speed_stride,
        )

        wait_frames(simulation_app, args.post_wait)

        print("[DONE] saved cuRobo pickup/place plans replayed on A2D.", flush=True)

        if args.keep_open and not args.headless:
            print("[INFO] keep-open enabled. Close Isaac window to exit.", flush=True)
            while simulation_app.is_running():
                simulation_app.update()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()