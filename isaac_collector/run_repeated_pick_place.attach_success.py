from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--cup-path", required=True)

    parser.add_argument(
        "--observation-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "rgbd_file"],
        help="Where the GraspNet observation comes from.",
    )

    parser.add_argument(
        "--rgbd-observation-npz",
        type=str,
        default="/tmp/robot_pipeline/repeated_pick_place/ep_0000_observation_rgbd.npz",
        help="Existing RGB-D observation npz exported from IsaacLab camera.",
    )

    parser.add_argument(
        "--graspnet-python",
        default="/home/pm/miniconda3/envs/graspnet_env/bin/python",
    )
    parser.add_argument(
        "--curobo-python",
        default="/home/pm/miniconda3/envs/curobo_env_cu13/bin/python",
    )

    parser.add_argument("--grasp-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--curobo-mode", choices=["mock", "real"], default="real")

    parser.add_argument("--graspnet-checkpoint", default=None)
    parser.add_argument("--curobo-robot-config", default=None)

    parser.add_argument("--num-pick-place", type=int, default=1)
    parser.add_argument("--right-distance", type=float, default=0.25)
    parser.add_argument("--right-axis", choices=["x", "y"], default="x")

    parser.add_argument("--frames-per-segment", type=int, default=80)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")

    parser.add_argument(
        "--execution-mode",
        choices=["debug_object", "a2d_replay"],
        default="debug_object",
        help="debug_object directly moves cup; a2d_replay executes cuRobo joint plans on A2D.",
    )

    parser.add_argument(
        "--steps-per-position",
        type=int,
        default=1,
        help="How many Isaac updates to hold each cuRobo waypoint during A2D replay.",
    )

    parser.add_argument(
        "--speed-stride",
        type=int,
        default=1,
        help="Use every N-th cuRobo waypoint during A2D replay.",
    )

    parser.add_argument(
        "--attach-cup-during-place",
        action="store_true",
        help="After pickup replay, attach cup to EE by kinematic following during place replay.",
    )

    parser.add_argument(
        "--ee-path",
        type=str,
        default="/World/A2D/Link7_r",
        help="End-effector prim used for kinematic cup attachment.",
    )

    args, _ = parser.parse_known_args()
    return args


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_retract_robot_state(robot_config_path: str | None) -> dict:
    """
    Temporary bring-up robot_state.

    This reads cuRobo cspace.joint_names and retract_config from the robot yaml.
    Later this should be replaced by the real IsaacLab robot joint state.
    """
    if robot_config_path is None:
        return {}

    cfg_path = Path(robot_config_path).expanduser().resolve()
    if not cfg_path.exists():
        return {}

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cspace = cfg["robot_cfg"]["kinematics"]["cspace"]

    return {
        "joint_names": list(cspace["joint_names"]),
        "positions": [float(x) for x in cspace["retract_config"]],
    }


def pose_matrix_to_world_position(pose_world: np.ndarray) -> np.ndarray:
    """
    Extract world position from the current USD/Gf-style 4x4 matrix.

    In this project:
    - USD world translation is in row 3: m[3, 0:3].
    - mock GraspNet adds approach/grasp/lift offsets in column 3.
    """
    m = np.asarray(pose_world, dtype=float)

    row_t = m[3, 0:3].copy()
    col_t = m[0:3, 3].copy()

    if np.linalg.norm(row_t) > 1e-8:
        return row_t + col_t

    return col_t


def world_position_to_robot_position(
    position_world: np.ndarray,
    robot_world: np.ndarray,
) -> np.ndarray:
    """
    USD/Gf row-vector convention:
        p_world_h = p_robot_h @ T_world_robot
        p_robot_h = p_world_h @ inv(T_world_robot)
    """
    p = np.asarray(position_world, dtype=float)
    t_robot = np.asarray(robot_world, dtype=float)

    p_h = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    p_robot_h = p_h @ np.linalg.inv(t_robot)

    return p_robot_h[0:3]


def make_robot_frame_target_matrix(
    pose_world: np.ndarray,
    robot_world: np.ndarray,
    *,
    fixed_orientation: bool = True,
    clamp_to_known_reachable: bool = True,
) -> np.ndarray:
    """
    Convert a world-frame target matrix into a robot-frame target matrix.

    For bring-up, we clamp to a known reachable target:
        [0.45, 0.0, 0.8]
    because cuRobo reachability scan showed this target succeeds.
    Later remove this clamp and use the real transformed target.
    """
    position_world = pose_matrix_to_world_position(pose_world)
    position_robot = world_position_to_robot_position(position_world, robot_world)

    if clamp_to_known_reachable:
        position_robot[0] = 0.45
        position_robot[1] = 0.0
        position_robot[2] = 0.80

    target_robot = np.eye(4, dtype=float)

    if not fixed_orientation:
        # Disabled for now: current USD matrices contain scale.
        target_robot[0:3, 0:3] = np.asarray(pose_world, dtype=float)[0:3, 0:3]

    # Row-vector convention: translation is in row 3.
    target_robot[3, 0:3] = position_robot

    return target_robot


def make_place_pose(current_pose: np.ndarray, *, axis: str, distance: float) -> np.ndarray:
    """
    Move the object in USD row-vector convention.
    Translation is pose[3, 0:3], not pose[0:3, 3].
    """
    pose = np.asarray(current_pose, dtype=float).copy()

    if axis == "x":
        pose[3, 0] += distance
    else:
        pose[3, 1] += distance

    return pose


def save_synthetic_observation_npz(
    path: Path,
    *,
    cup_pose_world: np.ndarray,
    num_points: int = 2048,
):
    """
    Temporary observation file for testing GraspNet real-mode plumbing.

    This is not real RGB-D. It only creates a point cloud-like npz so that
    graspnet_service.py --mode real skeleton can verify file input.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    cup_center = np.asarray(cup_pose_world, dtype=float)[3, 0:3]
    points = cup_center[None, :] + 0.02 * np.random.randn(num_points, 3)
    colors = np.ones((num_points, 3), dtype=np.float32)

    np.savez(
        path,
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
        object_pose_world=np.asarray(cup_pose_world, dtype=np.float32),
    )


def save_observation_from_existing_rgbd_npz(src_npz_path, dst_npz_path):
    """
    Copy an existing RGB-D observation npz into the episode observation path.

    This is an adapter layer:
      src: RGB-D exporter output
      dst: current GraspNet skeleton expected observation path

    Required fields:
      points: [N, 3], float32
      colors: [N, 3], float32, range 0~1
    """
    src_npz_path = Path(src_npz_path)
    dst_npz_path = Path(dst_npz_path)

    if not src_npz_path.exists():
        raise FileNotFoundError(f"RGB-D observation npz not found: {src_npz_path}")

    obs = np.load(src_npz_path, allow_pickle=True)

    if "points" not in obs.files or "colors" not in obs.files:
        raise KeyError(
            f"RGB-D observation must contain 'points' and 'colors'. "
            f"Available keys: {obs.files}"
        )

    points = obs["points"].astype(np.float32)
    colors = obs["colors"].astype(np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid points shape: {points.shape}")

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Invalid colors shape: {colors.shape}")

    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points/colors count mismatch: {points.shape[0]} vs {colors.shape[0]}"
        )

    if not np.isfinite(points).all():
        raise ValueError("points contain NaN/Inf")

    if not np.isfinite(colors).all():
        raise ValueError("colors contain NaN/Inf")

    dst_npz_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "points": points,
        "colors": colors,
    }

    for k in [
        "rgb",
        "depth",
        "intrinsics",
        "points_camera",
        "valid_depth_mask",
        "target_xyz",
        "camera_position",
        "camera_path",
        "target_path",
        "scene_usd",
        "depth_source",
    ]:
        if k in obs.files:
            save_dict[k] = obs[k]

    np.savez_compressed(dst_npz_path, **save_dict)

    print(f"[OBS] copied RGB-D observation", flush=True)
    print(f"[OBS] src: {src_npz_path}", flush=True)
    print(f"[OBS] dst: {dst_npz_path}", flush=True)
    print(f"[OBS] points: {points.shape} {points.dtype}", flush=True)
    print(f"[OBS] colors: {colors.shape} {colors.dtype}", flush=True)
    print(f"[OBS] points min: {points.min(axis=0)}", flush=True)
    print(f"[OBS] points max: {points.max(axis=0)}", flush=True)
    print(f"[OBS] colors min/max: {colors.min()} / {colors.max()}", flush=True)


def execute_cartesian_debug_on_object(
    *,
    simulation_app,
    stage,
    cup_path: str,
    target_pose_world: np.ndarray,
    frames: int,
):
    """
    Debug execution: directly move cup.

    This is not final robot execution.

    Final execution should be:
    - cuRobo returns joint trajectory
    - A2D Articulation executes joint trajectory
    - close gripper
    - attach / physics grasp
    """
    from isaac_collector.runtime.load_scene import (
        get_prim_world_matrix,
        set_prim_world_matrix,
        interpolate_pose_translation,
    )

    start = get_prim_world_matrix(stage, cup_path)
    poses = interpolate_pose_translation(start, target_pose_world, frames)

    for pose in poses:
        set_prim_world_matrix(stage, cup_path, pose)
        simulation_app.update()


def wait_frames(simulation_app, n: int):
    for _ in range(n):
        simulation_app.update()


def patch_a2d_articulation_root(stage, robot_path: str):
    """
    In-memory patch.

    Some A2D USDs put ArticulationRootAPI on:
        /World/A2D/root_joint

    dynamic_control expects a root rigid body, so for replay we move the
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


def acquire_dynamic_control():
    from omni.isaac.dynamic_control import _dynamic_control
    return _dynamic_control.acquire_dynamic_control_interface()


def get_articulation_handle(dc, robot_path: str):
    candidates = [
        f"{robot_path}/base_link",
        robot_path,
        f"{robot_path}/root_joint",
    ]

    for p in candidates:
        art = dc.get_articulation(p)
        ok = art is not None and not (isinstance(art, int) and art == 0)
        print(f"[DYNAMIC_CONTROL] try articulation {p} -> ok={ok}, handle={art}", flush=True)

        if ok:
            dc.wake_up_articulation(art)
            print(f"[DYNAMIC_CONTROL] using articulation path: {p}", flush=True)
            return art

    raise RuntimeError(f"Failed to acquire A2D articulation under {robot_path}")


def get_articulation_dof_names_and_handles(dc, art):
    n = dc.get_articulation_dof_count(art)

    names = []
    handles = {}

    for i in range(n):
        dof = dc.get_articulation_dof(art, i)
        name = dc.get_dof_name(dof)
        names.append(name)
        handles[name] = dof

    return names, handles


def normalize_joint_name(s: str) -> str:
    return s.lower().replace("/", "_").replace(":", "_")


def build_joint_mapping(curobo_joint_names, usd_dof_names, usd_dof_handles):
    mapping = {}
    norm_to_usd = {normalize_joint_name(n): n for n in usd_dof_names}

    for cj in curobo_joint_names:
        if cj in usd_dof_handles:
            mapping[cj] = usd_dof_handles[cj]
            continue

        norm_cj = normalize_joint_name(cj)
        if norm_cj in norm_to_usd:
            usd_name = norm_to_usd[norm_cj]
            mapping[cj] = usd_dof_handles[usd_name]
            continue

        candidates = [
            n
            for n in usd_dof_names
            if normalize_joint_name(n).endswith(norm_cj)
            or norm_cj.endswith(normalize_joint_name(n))
        ]

        if len(candidates) == 1:
            mapping[cj] = usd_dof_handles[candidates[0]]
            continue

        print("[ERROR] Cannot map cuRobo joint:", cj, flush=True)
        print("[ERROR] USD DOF names:", usd_dof_names, flush=True)
        raise RuntimeError(f"Cannot map cuRobo joint to USD DOF: {cj}")

    return mapping


def init_a2d_replay_controller(stage, simulation_app, robot_path: str, curobo_joint_names):
    import omni.timeline

    patch_a2d_articulation_root(stage, robot_path)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    wait_frames(simulation_app, 60)

    dc = acquire_dynamic_control()
    art = get_articulation_handle(dc, robot_path)
    usd_dof_names, usd_dof_handles = get_articulation_dof_names_and_handles(dc, art)

    print("[REPLAY_INIT] USD articulation DOF count:", len(usd_dof_names), flush=True)

    mapping = build_joint_mapping(
        curobo_joint_names=curobo_joint_names,
        usd_dof_names=usd_dof_names,
        usd_dof_handles=usd_dof_handles,
    )

    print("[REPLAY_INIT] cuRobo → USD DOF mapping ready", flush=True)
    for name in curobo_joint_names:
        print(f"  {name} -> {dc.get_dof_name(mapping[name])}", flush=True)

    return {
        "dc": dc,
        "art": art,
        "joint_mapping": mapping,
        "curobo_joint_names": list(curobo_joint_names),
    }


def invert_row_pose(T: np.ndarray) -> np.ndarray:
    """
    Invert a row-vector transform:
        p_world = p_local @ T
    """
    return np.linalg.inv(np.asarray(T, dtype=float))


def compute_cup_to_ee_row_offset(stage, *, cup_path: str, ee_path: str) -> np.ndarray:
    """
    Compute fixed cup offset relative to EE.

    Row-vector convention:
        cup_world = cup_in_ee @ ee_world
        cup_in_ee = cup_world @ inv(ee_world)
    """
    from isaac_collector.runtime.load_scene import get_prim_world_matrix

    cup_world = np.asarray(get_prim_world_matrix(stage, cup_path), dtype=float)
    ee_world = np.asarray(get_prim_world_matrix(stage, ee_path), dtype=float)

    cup_in_ee = cup_world @ invert_row_pose(ee_world)

    print("[ATTACH] computed cup_in_ee offset", flush=True)
    print("[ATTACH] cup path:", cup_path, flush=True)
    print("[ATTACH] ee path:", ee_path, flush=True)
    print("[ATTACH] cup_world:", flush=True)
    print(cup_world, flush=True)
    print("[ATTACH] ee_world:", flush=True)
    print(ee_world, flush=True)
    print("[ATTACH] cup_in_ee:", flush=True)
    print(cup_in_ee, flush=True)

    return cup_in_ee


def dc_pose_to_position(pose) -> np.ndarray:
    """
    dynamic_control pose.p -> numpy xyz.
    """
    return np.array([float(pose.p.x), float(pose.p.y), float(pose.p.z)], dtype=float)


def get_ee_position_from_dynamic_control(replay_controller, ee_path: str) -> np.ndarray:
    """
    Read runtime EE rigid body position from dynamic_control.

    This is necessary because USD stage transforms may not reflect PhysX-updated
    articulation link poses during replay.
    """
    dc = replay_controller["dc"]

    ee_body = replay_controller.get("ee_body_handle", None)
    cached_ee_path = replay_controller.get("ee_body_path", None)

    if ee_body is None or cached_ee_path != ee_path:
        ee_body = dc.get_rigid_body(ee_path)
        ok = ee_body is not None and not (isinstance(ee_body, int) and ee_body == 0)

        print(f"[ATTACH] get_rigid_body {ee_path} -> ok={ok}, handle={ee_body}", flush=True)

        if not ok:
            raise RuntimeError(f"Failed to get EE rigid body from dynamic_control: {ee_path}")

        replay_controller["ee_body_handle"] = ee_body
        replay_controller["ee_body_path"] = ee_path

    pose = dc.get_rigid_body_pose(ee_body)
    return dc_pose_to_position(pose)


def compute_cup_to_ee_translation_offset(stage, replay_controller, *, cup_path: str, ee_path: str) -> np.ndarray:
    """
    Compute cup translation offset relative to runtime EE position.

    First version only attaches translation:
        cup_pos = ee_pos + offset
    """
    from isaac_collector.runtime.load_scene import get_prim_world_matrix

    cup_world = np.asarray(get_prim_world_matrix(stage, cup_path), dtype=float)
    cup_pos = cup_world[3, 0:3].copy()

    ee_pos = get_ee_position_from_dynamic_control(replay_controller, ee_path)

    offset = cup_pos - ee_pos

    print("[ATTACH] computed cup translation offset from dynamic_control EE", flush=True)
    print("[ATTACH] cup_pos:", cup_pos, flush=True)
    print("[ATTACH] ee_pos:", ee_pos, flush=True)
    print("[ATTACH] offset:", offset, flush=True)

    return offset


def set_cup_follow_ee_translation(stage, replay_controller, *, cup_path: str, ee_path: str, cup_offset: np.ndarray):
    """
    Kinematic attach by translation only:
        cup_pos = ee_pos + cup_offset

    Keep current cup orientation/scale unchanged.
    """
    from isaac_collector.runtime.load_scene import (
        get_prim_world_matrix,
        set_prim_world_matrix,
    )

    ee_pos = get_ee_position_from_dynamic_control(replay_controller, ee_path)

    cup_world = np.asarray(get_prim_world_matrix(stage, cup_path), dtype=float)
    cup_world[3, 0:3] = ee_pos + np.asarray(cup_offset, dtype=float)

    set_prim_world_matrix(stage, cup_path, cup_world)

def set_cup_follow_ee(stage, *, cup_path: str, ee_path: str, cup_in_ee: np.ndarray):
    """
    Kinematic attach: force cup pose to follow EE pose every frame.
    """
    from isaac_collector.runtime.load_scene import (
        get_prim_world_matrix,
        set_prim_world_matrix,
    )

    ee_world = np.asarray(get_prim_world_matrix(stage, ee_path), dtype=float)
    cup_world = np.asarray(cup_in_ee, dtype=float) @ ee_world

    set_prim_world_matrix(stage, cup_path, cup_world)


def execute_curobo_plan_on_a2d(
    *,
    simulation_app,
    replay_controller,
    plan: dict,
    name: str,
    steps_per_position: int = 1,
    speed_stride: int = 1,
    stage=None,
    cup_path: str | None = None,
    ee_path: str | None = None,
    cup_offset: np.ndarray | None = None,
):
    if not plan.get("success", False):
        raise RuntimeError(f"Cannot replay failed plan: {name}, plan={plan}")

    positions = np.asarray(plan.get("positions", []), dtype=np.float32)

    if positions.ndim != 2:
        raise ValueError(f"Invalid plan positions shape for {name}: {positions.shape}")

    if speed_stride < 1:
        speed_stride = 1

    positions = positions[::speed_stride]

    dc = replay_controller["dc"]
    mapping = replay_controller["joint_mapping"]
    joint_names = replay_controller["curobo_joint_names"]

    attach_enabled = (
        stage is not None
        and cup_path is not None
        and ee_path is not None
        and cup_offset is not None
    )

    print("\n" + "=" * 80, flush=True)
    print(f"[A2D_REPLAY] {name}", flush=True)
    print("=" * 80, flush=True)
    print(f"[A2D_REPLAY] positions shape: {positions.shape}", flush=True)
    print(f"[A2D_REPLAY] attach_enabled: {attach_enabled}", flush=True)

    for t, q in enumerate(positions):
        if q.shape[0] != len(joint_names):
            raise ValueError(
                f"{name} q width mismatch: {q.shape[0]} vs {len(joint_names)}"
            )

        for j, joint_name in enumerate(joint_names):
            dc.set_dof_position_target(mapping[joint_name], float(q[j]))

        for _ in range(steps_per_position):
            simulation_app.update()

            if attach_enabled:
                set_cup_follow_ee_translation(
                    stage,
                    replay_controller,
                    cup_path=cup_path,
                    ee_path=ee_path,
                    cup_offset=cup_offset,
                )
                simulation_app.update()

        if t % 25 == 0 or t == positions.shape[0] - 1:
            print(f"[A2D_REPLAY] {name}: {t + 1}/{positions.shape[0]}", flush=True)

    print(f"[DONE] A2D replay finished: {name}", flush=True)


def print_plan_summary(name: str, plan: dict):
    print(f"[PLAN] {name} success:", plan.get("success"), flush=True)
    print(f"[PLAN] {name} source:", plan.get("source"), flush=True)
    print(f"[PLAN] {name} status:", plan.get("status"), flush=True)
    print(f"[PLAN] {name} num_positions:", len(plan.get("positions", [])), flush=True)
    print(f"[PLAN] {name} target_pose_world:", plan.get("target_pose_world"), flush=True)


def move_a2d_close_to_cup_for_bringup(
    *,
    stage,
    simulation_app,
    robot_path: str,
):
    """
    Temporary bring-up placement.

    This moves A2D near the current cup so that cuRobo can plan to a known
    reachable robot-frame target. Later replace this with navigation/base control.
    """
    from isaac_collector.runtime.load_scene import get_prim_world_matrix, set_prim_world_matrix

    robot_world = get_prim_world_matrix(stage, robot_path)
    robot_world = np.asarray(robot_world, dtype=float)

    # Known good placement from previous debugging.
    robot_world[3, 0] = 2.827275532134659
    robot_world[3, 1] = -2.5853563806447653
    robot_world[3, 2] = -0.039999272674322135

    set_prim_world_matrix(stage, robot_path, robot_world)
    simulation_app.update()


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    grasp_service = None
    curobo_service = None

    try:
        from isaac_collector.runtime.load_scene import (
            open_usd_stage,
            get_prim_world_matrix,
        )
        from isaac_collector.ipc.jsonl_service import PersistentJsonService

        print("[1] Open Isaac scene", flush=True)
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)

        robot_path = "/World/A2D"

        move_a2d_close_to_cup_for_bringup(
            stage=stage,
            simulation_app=simulation_app,
            robot_path=robot_path,
        )

        print("[2] Start persistent GraspNet service", flush=True)
        grasp_service = PersistentJsonService(
            name="graspnet",
            python_exe=args.graspnet_python,
            worker_file=str(project_root / "isaac_collector/services/graspnet_service.py"),
            project_root=str(project_root),
            args=[
                "--mode",
                args.grasp_mode,
                *([] if args.graspnet_checkpoint is None else ["--checkpoint", args.graspnet_checkpoint]),
            ],
        )

        print("[3] Start persistent cuRobo service", flush=True)
        curobo_service = PersistentJsonService(
            name="curobo",
            python_exe=args.curobo_python,
            worker_file=str(project_root / "isaac_collector/services/curobo_service.py"),
            project_root=str(project_root),
            args=[
                "--mode",
                args.curobo_mode,
                *([] if args.curobo_robot_config is None else ["--robot-config", args.curobo_robot_config]),
            ],
        )

        output_dir = Path("/tmp/robot_pipeline/repeated_pick_place")
        output_dir.mkdir(parents=True, exist_ok=True)

        robot_state = load_retract_robot_state(args.curobo_robot_config)
        if args.curobo_mode == "real":
            print("[STATE] using cuRobo retract_config robot_state", flush=True)
            print("[STATE] robot_state joint_names:", robot_state.get("joint_names", []), flush=True)
            print("[STATE] robot_state positions:", robot_state.get("positions", []), flush=True)

        replay_controller = None
        if args.execution_mode == "a2d_replay":
            replay_controller = init_a2d_replay_controller(
                stage=stage,
                simulation_app=simulation_app,
                robot_path=robot_path,
                curobo_joint_names=robot_state.get("joint_names", []),
            )

        robot_world = get_prim_world_matrix(stage, robot_path)

        print("[DEBUG] world pose check", flush=True)
        for debug_path in [robot_path, args.cup_path, args.ee_path]:
            try:
                m = get_prim_world_matrix(stage, debug_path)
                print(f"[DEBUG] {debug_path}", flush=True)
                print(m, flush=True)
                print(
                    "[DEBUG] translation row:",
                    [float(m[3][0]), float(m[3][1]), float(m[3][2])],
                    flush=True,
                )
            except Exception as e:
                print(f"[DEBUG] failed to get pose for {debug_path}: {e!r}", flush=True)

        print("[DEBUG] robot_world matrix:", flush=True)
        print(robot_world, flush=True)

        print("[4] Start repeated pickup / move-right / putdown loop", flush=True)

        for ep in range(args.num_pick_place):
            print("\n" + "=" * 80, flush=True)
            print(f"[EP {ep + 1}/{args.num_pick_place}]", flush=True)
            print("=" * 80, flush=True)

            cup_offset = None

            cup_pose = get_prim_world_matrix(stage, args.cup_path)
            print("[STATE] cup pose:", flush=True)
            print(cup_pose, flush=True)

            obs_path = output_dir / f"ep_{ep:04d}_observation.npz"

            if args.observation_source == "synthetic":
                save_synthetic_observation_npz(
                    obs_path,
                    cup_pose_world=cup_pose,
                )
                print("[OBS] saved synthetic observation_npz:", obs_path, flush=True)

            elif args.observation_source == "rgbd_file":
                save_observation_from_existing_rgbd_npz(
                    src_npz_path=args.rgbd_observation_npz,
                    dst_npz_path=obs_path,
                )
                print("[OBS] saved RGB-D observation_npz:", obs_path, flush=True)

            else:
                raise ValueError(f"Unknown observation_source: {args.observation_source}")

            grasp_result = grasp_service.call(
                "predict",
                {
                    "object_path": args.cup_path,
                    "object_pose_world": cup_pose.tolist(),
                    "observation_npz": str(obs_path),
                },
            )

            save_json(output_dir / f"ep_{ep:04d}_grasp.json", grasp_result)
            print("[DEBUG] saved grasp json", flush=True)
            print("[DEBUG] grasp source:", grasp_result.get("source"), flush=True)

            lift_world = np.asarray(grasp_result["lift_pose_world"], dtype=float)

            pickup_target_robot = make_robot_frame_target_matrix(
                lift_world,
                robot_world,
                fixed_orientation=True,
                clamp_to_known_reachable=True,
            )

            print("[DEBUG] pickup target robot-frame matrix:", flush=True)
            print(pickup_target_robot, flush=True)

            pickup_plan = curobo_service.call(
                "plan",
                {
                    "task": "pickup",
                    "robot_state": robot_state,
                    "waypoints_world": [pickup_target_robot.tolist()],
                },
            )

            print_plan_summary("pickup", pickup_plan)
            save_json(output_dir / f"ep_{ep:04d}_pickup_plan.json", pickup_plan)

            print("[EXEC] pickup: approach -> close gripper -> lift", flush=True)

            if args.execution_mode == "debug_object":
                execute_cartesian_debug_on_object(
                    simulation_app=simulation_app,
                    stage=stage,
                    cup_path=args.cup_path,
                    target_pose_world=lift_world,
                    frames=args.frames_per_segment,
                )

            elif args.execution_mode == "a2d_replay":
                execute_curobo_plan_on_a2d(
                    simulation_app=simulation_app,
                    replay_controller=replay_controller,
                    plan=pickup_plan,
                    name="pickup",
                    steps_per_position=args.steps_per_position,
                    speed_stride=args.speed_stride,
                )

            else:
                raise ValueError(args.execution_mode)

            wait_frames(simulation_app, 20)

            if args.execution_mode == "a2d_replay" and args.attach_cup_during_place:
                cup_offset = compute_cup_to_ee_translation_offset(
                    stage,
                    replay_controller,
                    cup_path=args.cup_path,
                    ee_path=args.ee_path,
                )
                print("[ATTACH] cup attached to EE for place replay", flush=True)

            current_after_lift = get_prim_world_matrix(stage, args.cup_path)
            place_world = make_place_pose(
                current_after_lift,
                axis=args.right_axis,
                distance=args.right_distance,
            )

            place_target_robot = make_robot_frame_target_matrix(
                place_world,
                robot_world,
                fixed_orientation=True,
                clamp_to_known_reachable=True,
            )

            print("[DEBUG] place target robot-frame matrix:", flush=True)
            print(place_target_robot, flush=True)

            place_plan = curobo_service.call(
                "plan",
                {
                    "task": "putdown",
                    "robot_state": robot_state,
                    "world_collision": {},
                    "attached_object": args.cup_path,
                    "waypoints_world": [place_target_robot.tolist()],
                },
            )

            print_plan_summary("place", place_plan)
            save_json(output_dir / f"ep_{ep:04d}_place_plan.json", place_plan)

            print("[EXEC] move-right -> putdown -> open gripper", flush=True)

            if args.execution_mode == "debug_object":
                execute_cartesian_debug_on_object(
                    simulation_app=simulation_app,
                    stage=stage,
                    cup_path=args.cup_path,
                    target_pose_world=place_world,
                    frames=args.frames_per_segment,
                )

            elif args.execution_mode == "a2d_replay":
                execute_curobo_plan_on_a2d(
                    simulation_app=simulation_app,
                    replay_controller=replay_controller,
                    plan=place_plan,
                    name="place",
                    steps_per_position=args.steps_per_position,
                    speed_stride=args.speed_stride,
                    stage=stage if args.attach_cup_during_place else None,
                    cup_path=args.cup_path if args.attach_cup_during_place else None,
                    ee_path=args.ee_path if args.attach_cup_during_place else None,
                    cup_offset=cup_offset if args.attach_cup_during_place else None,
                )

            else:
                raise ValueError(args.execution_mode)

            wait_frames(simulation_app, 30)

            if args.attach_cup_during_place:
                final_cup_pose = get_prim_world_matrix(stage, args.cup_path)
                print("[ATTACH] final cup pose after place replay:", flush=True)
                print(final_cup_pose, flush=True)

            print(f"[EP {ep + 1}] done", flush=True)

        print("[DONE] repeated pick-place collection loop finished.", flush=True)

        if args.keep_open and not args.headless:
            print("[INFO] keep-open enabled.", flush=True)
            while simulation_app.is_running():
                simulation_app.update()

    finally:
        if grasp_service is not None:
            grasp_service.close()
        if curobo_service is not None:
            curobo_service.close()
        simulation_app.close()


if __name__ == "__main__":
    main()