#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def quat_wxyz_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to scalar-first quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotmat_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotmat_from_quat_wxyz(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def look_at_quat_world(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Build a quaternion for Isaac camera_axes='world':
      camera local +X points forward,
      local +Z is approximately world up.
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    forward = target - camera_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("camera_pos and target are too close.")
    x_axis = forward / norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(x_axis, up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    y_axis = np.cross(up, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return quat_wxyz_from_rotmat(R)


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def get_prim_bbox_center(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedRange()

    mn = np.array(aligned.GetMin(), dtype=np.float64)
    mx = np.array(aligned.GetMax(), dtype=np.float64)

    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
        return None
    if np.linalg.norm(mx - mn) < 1e-8:
        return None

    return 0.5 * (mn + mx)


def ensure_dome_light(stage):
    from pxr import Sdf, UsdLux

    light_path = Sdf.Path("/World/RGBD_DomeLight")
    if stage.GetPrimAtPath(str(light_path)).IsValid():
        return

    light = UsdLux.DomeLight.Define(stage, light_path)
    light.CreateIntensityAttr(500.0)


def rgb_to_uint8(rgb_like: np.ndarray, h: int, w: int) -> np.ndarray:
    arr = np.asarray(rgb_like)

    if arr.ndim == 1:
        if arr.size == h * w * 4:
            arr = arr.reshape(h, w, 4)
        elif arr.size == h * w * 3:
            arr = arr.reshape(h, w, 3)
        else:
            raise RuntimeError(f"Cannot reshape RGB/RGBA buffer with shape {arr.shape}")

    if arr.ndim == 2 and arr.shape[0] == h * w and arr.shape[1] in (3, 4):
        arr = arr.reshape(h, w, arr.shape[1])

    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB/RGBA shape: {arr.shape}")

    if arr.shape[:2] != (h, w):
        if arr.shape[:2] == (w, h):
            arr = np.transpose(arr, (1, 0, 2))
        else:
            raise RuntimeError(f"RGB shape {arr.shape} does not match depth shape {(h, w)}")

    rgb = arr[..., :3]

    if rgb.dtype == np.uint8:
        return rgb

    rgb = rgb.astype(np.float32)
    if np.nanmax(rgb) <= 1.5:
        rgb = rgb * 255.0

    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_preview_images(out_path: Path, rgb_u8: np.ndarray, depth: np.ndarray):
    try:
        from PIL import Image
    except Exception as e:
        print(f"[WARN] PIL not available, skip preview images: {e}")
        return

    rgb_png = out_path.with_suffix(".rgb.png")
    depth_png = out_path.with_suffix(".depth.png")

    Image.fromarray(rgb_u8).save(rgb_png)

    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        d = depth.copy()
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        depth_u8 = (d * 255).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)

    Image.fromarray(depth_u8).save(depth_png)
    print(f"[SAVE] preview rgb:   {rgb_png}")
    print(f"[SAVE] preview depth: {depth_png}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-usd",
        type=str,
        default="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default="/World/office1/Room_seed123_idx000/furniture/tea_table/cup",
        help="Use this prim bbox center as camera target if valid.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.8],
        help="Fallback target xyz if target-path is invalid.",
    )
    parser.add_argument(
        "--camera-offset",
        nargs=3,
        type=float,
        default=[0.75, -0.75, 0.65],
        help="Camera position = target + camera_offset.",
    )
    parser.add_argument("--camera-path", type=str, default="/World/RGBD_Camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=5.0)
    parser.add_argument(
        "--crop-radius",
        type=float,
        default=1.0,
        help="Crop point cloud around target in world frame. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Randomly downsample saved point cloud. Set <=0 to keep all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument(
        "--out",
        type=str,
        default="/tmp/robot_pipeline/repeated_pick_place/ep_0000_observation_rgbd.npz",
    )
    parser.add_argument("--no-preview", action="store_true")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, unknown = parser.parse_known_args()

    if unknown:
        print(f"[INFO] Keep unknown Kit args: {unknown}")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import Isaac/Omniverse modules only after SimulationApp/AppLauncher is created.
    import omni.usd

    try:
        from isaacsim.core.utils.stage import open_stage
    except Exception:
        from omni.isaac.core.utils.stage import open_stage

    try:
        from isaacsim.sensors.camera import Camera
    except Exception as e:
        raise RuntimeError(
            "Cannot import isaacsim.sensors.camera.Camera. "
            "Try running with Isaac Sim / IsaacLab python and ensure extension "
            "'isaacsim.sensors.camera' is enabled."
        ) from e

    scene_usd = Path(args_cli.scene_usd)
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene usd not found: {scene_usd}")

    out_path = Path(args_cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] scene: {scene_usd}")
    open_stage(str(scene_usd))

    for _ in range(args_cli.load_steps):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    ensure_dome_light(stage)

    target = None
    if args_cli.target_path:
        target = get_prim_bbox_center(stage, args_cli.target_path)

    if target is None:
        target = np.array(args_cli.target, dtype=np.float64)
        print(f"[WARN] target-path invalid or empty: {args_cli.target_path}")
        print(f"[INFO] use fallback target: {target}")
    else:
        print(f"[INFO] target from prim bbox: {args_cli.target_path}")
        print(f"[INFO] target xyz: {target}")

    camera_offset = np.array(args_cli.camera_offset, dtype=np.float64)
    camera_pos = target + camera_offset
    camera_quat = look_at_quat_world(camera_pos, target)

    print(f"[INFO] camera path: {args_cli.camera_path}")
    print(f"[INFO] camera pos:  {camera_pos}")
    print(f"[INFO] camera quat: {camera_quat}")

    camera = Camera(
        prim_path=args_cli.camera_path,
        name="rgbd_camera",
        resolution=(args_cli.width, args_cli.height),
    )
    camera.initialize()
    camera.set_world_pose(
        position=camera_pos,
        orientation=camera_quat,
        camera_axes="world",
    )
    camera.set_clipping_range(near_distance=0.01, far_distance=10.0)
    camera.add_distance_to_image_plane_to_frame()

    for _ in range(args_cli.warmup_steps):
        simulation_app.update()

    depth = to_numpy(camera.get_depth()).astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth shape: {depth.shape}")

    h, w = depth.shape

    rgba = to_numpy(camera.get_rgba())
    rgb_u8 = rgb_to_uint8(rgba, h=h, w=w)
    rgb_f32 = rgb_u8.astype(np.float32) / 255.0

    K = to_numpy(camera.get_intrinsics_matrix()).astype(np.float32)

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    uv_all = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float32)

    depth_flat = depth.reshape(-1)
    valid = (
        np.isfinite(depth_flat)
        & (depth_flat > args_cli.depth_min)
        & (depth_flat < args_cli.depth_max)
    )

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        raise RuntimeError("No valid depth points. Camera may not see the scene/cup.")

    uv_valid = uv_all[valid_idx]
    depth_valid = depth_flat[valid_idx]
    colors_valid = rgb_f32.reshape(-1, 3)[valid_idx]
    colors_u8_valid = rgb_u8.reshape(-1, 3)[valid_idx]

    points_camera = to_numpy(
        camera.get_camera_points_from_image_coords(uv_valid, depth_valid)
    ).astype(np.float32)

    points_world = to_numpy(
        camera.get_world_points_from_image_coords(uv_valid, depth_valid)
    ).astype(np.float32)

    # Optional crop around cup/table area in world frame.
    if args_cli.crop_radius > 0:
        dist = np.linalg.norm(points_world - target.reshape(1, 3), axis=1)
        keep = dist < args_cli.crop_radius

        points_camera = points_camera[keep]
        points_world = points_world[keep]
        colors_valid = colors_valid[keep]
        colors_u8_valid = colors_u8_valid[keep]
        uv_valid = uv_valid[keep]
        depth_valid = depth_valid[keep]

    # Optional downsample for GraspNet-friendly observation size.
    if args_cli.max_points > 0 and points_camera.shape[0] > args_cli.max_points:
        rng = np.random.default_rng(args_cli.seed)
        ids = rng.choice(points_camera.shape[0], size=args_cli.max_points, replace=False)

        points_camera = points_camera[ids]
        points_world = points_world[ids]
        colors_valid = colors_valid[ids]
        colors_u8_valid = colors_u8_valid[ids]
        uv_valid = uv_valid[ids]
        depth_valid = depth_valid[ids]

    pos_world, quat_world = camera.get_world_pose(camera_axes="world")
    pos_ros, quat_ros = camera.get_world_pose(camera_axes="ros")
    pos_usd, quat_usd = camera.get_world_pose(camera_axes="usd")

    pos_world = to_numpy(pos_world).astype(np.float32)
    quat_world = to_numpy(quat_world).astype(np.float32)
    pos_ros = to_numpy(pos_ros).astype(np.float32)
    quat_ros = to_numpy(quat_ros).astype(np.float32)
    pos_usd = to_numpy(pos_usd).astype(np.float32)
    quat_usd = to_numpy(quat_usd).astype(np.float32)

    T_world_cam_world_axes = pose_matrix_from_pos_quat(pos_world, quat_world).astype(np.float32)
    T_world_cam_ros_axes = pose_matrix_from_pos_quat(pos_ros, quat_ros).astype(np.float32)
    T_world_cam_usd_axes = pose_matrix_from_pos_quat(pos_usd, quat_usd).astype(np.float32)

    np.savez_compressed(
        out_path,
        # Full rendered observation.
        rgb=rgb_u8,
        depth=depth.astype(np.float32),
        intrinsics=K.astype(np.float32),
        valid_depth_mask=valid.reshape(h, w),

        # GraspNet-friendly aliases.
        # points/colors are camera-frame by default.
        points=points_camera.astype(np.float32),
        colors=colors_valid.astype(np.float32),

        # Explicit coordinate versions.
        points_camera=points_camera.astype(np.float32),
        points_world=points_world.astype(np.float32),
        colors_uint8=colors_u8_valid.astype(np.uint8),
        uv=uv_valid.astype(np.float32),
        depth_samples=depth_valid.astype(np.float32),

        # Camera poses in several conventions for later debugging.
        camera_position_world_axes=pos_world,
        camera_quat_world_axes=quat_world,
        camera_pose_world_axes=T_world_cam_world_axes,

        camera_position_ros_axes=pos_ros,
        camera_quat_ros_axes=quat_ros,
        camera_pose_ros_axes=T_world_cam_ros_axes,

        camera_position_usd_axes=pos_usd,
        camera_quat_usd_axes=quat_usd,
        camera_pose_usd_axes=T_world_cam_usd_axes,

        # Metadata.
        camera_path=np.array(args_cli.camera_path),
        target_path=np.array(args_cli.target_path),
        target_xyz=target.astype(np.float32),
        scene_usd=np.array(str(scene_usd)),
    )

    print(f"[SAVE] observation npz: {out_path}")
    print(f"[SHAPE] rgb:           {rgb_u8.shape} {rgb_u8.dtype}")
    print(f"[SHAPE] depth:         {depth.shape} {depth.dtype}")
    print(f"[SHAPE] intrinsics:    {K.shape} {K.dtype}")
    print(f"[SHAPE] points_camera: {points_camera.shape} {points_camera.dtype}")
    print(f"[SHAPE] points_world:  {points_world.shape} {points_world.dtype}")
    print(f"[RANGE] depth valid:   min={depth_valid.min():.4f}, max={depth_valid.max():.4f}")
    print(f"[RANGE] camera xyz:    min={points_camera.min(axis=0)}, max={points_camera.max(axis=0)}")
    print(f"[RANGE] world xyz:     min={points_world.min(axis=0)}, max={points_world.max(axis=0)}")
    print(f"[INFO] K:\n{K}")

    if not args_cli.no_preview:
        save_preview_images(out_path, rgb_u8, depth)

    simulation_app.close()


if __name__ == "__main__":
    main()
