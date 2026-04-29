#!/usr/bin/env python3
import argparse
import os
import traceback
from pathlib import Path

import numpy as np


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def rgb_to_uint8(rgb):
    rgb = np.asarray(rgb)

    if rgb.ndim == 4:
        rgb = rgb[0]

    if rgb.ndim != 3:
        raise RuntimeError(f"Unexpected rgb shape: {rgb.shape}")

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if rgb.dtype == np.uint8:
        return rgb

    rgb_f = rgb.astype(np.float32)

    # Some Isaac outputs are 0~1 float, some are already 0~255 float.
    if np.nanmax(rgb_f) <= 1.5:
        rgb_f = rgb_f * 255.0

    return np.clip(rgb_f, 0, 255).astype(np.uint8)


def save_preview_images(out_path, rgb_u8, depth):
    try:
        from PIL import Image
    except Exception as e:
        print(f"[WARN] PIL unavailable, skip preview images: {e}", flush=True)
        return

    out_path = Path(out_path)
    rgb_png = out_path.with_suffix(".rgb.png")
    depth_png = out_path.with_suffix(".depth.png")

    Image.fromarray(rgb_u8).save(rgb_png)

    depth = np.squeeze(np.asarray(depth)).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [2, 98])
        depth_vis = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
        depth_u8 = (depth_vis * 255).astype(np.uint8)
    else:
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)

    Image.fromarray(depth_u8).save(depth_png)

    print(f"[SAVE] rgb preview:   {rgb_png}", flush=True)
    print(f"[SAVE] depth preview: {depth_png}", flush=True)


def get_prim_bbox_center(stage, prim_path):
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
    aligned_range = bbox.ComputeAlignedRange()

    mn = np.array(aligned_range.GetMin(), dtype=np.float64)
    mx = np.array(aligned_range.GetMax(), dtype=np.float64)

    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
        return None

    if np.linalg.norm(mx - mn) < 1e-8:
        return None

    return 0.5 * (mn + mx)


def ensure_dome_light(stage):
    from pxr import Sdf, UsdLux

    path = Sdf.Path("/World/RGBD_Debug_DomeLight")
    if stage.GetPrimAtPath(str(path)).IsValid():
        return

    light = UsdLux.DomeLight.Define(stage, path)
    light.CreateIntensityAttr(1000.0)
    print(f"[INFO] added dome light: {path}", flush=True)


def unproject_depth_to_camera_points(depth, K):
    """
    Convert image-plane depth to camera-frame point cloud.

    Camera-frame convention used here:
      x: right
      y: down
      z: forward

    This is suitable as a first-stage GraspNet input format.
    """
    depth = np.squeeze(depth).astype(np.float32)

    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth shape for unprojection: {depth.shape}")

    h, w = depth.shape

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        raise RuntimeError(f"Invalid intrinsics fx/fy: fx={fx}, fy={fy}")

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u.astype(np.float32) - cx) / fx * z
    y = (v.astype(np.float32) - cy) / fy * z

    points = np.stack([x, y, z], axis=-1)
    return points.reshape(-1, 3)


def print_depth_stats(name, depth):
    depth = np.squeeze(np.asarray(depth)).astype(np.float32)
    finite = np.isfinite(depth)
    positive = finite & (depth > 0)

    print(f"[DEBUG] {name} shape: {depth.shape} {depth.dtype}", flush=True)
    print(f"[DEBUG] {name} finite count:   {int(finite.sum())}", flush=True)
    print(f"[DEBUG] {name} positive count: {int(positive.sum())}", flush=True)

    if np.any(finite):
        print(
            f"[DEBUG] {name} finite min/max: "
            f"{float(np.nanmin(depth[finite])):.6f} / {float(np.nanmax(depth[finite])):.6f}",
            flush=True,
        )

    if np.any(positive):
        print(
            f"[DEBUG] {name} positive min/max: "
            f"{float(np.nanmin(depth[positive])):.6f} / {float(np.nanmax(depth[positive])):.6f}",
            flush=True,
        )


def simulation_step_with_render(sim, simulation_app):
    """
    Different IsaacLab versions expose slightly different step/render APIs.
    This helper tries the common paths.
    """
    try:
        sim.step(render=True)
        return
    except TypeError:
        pass

    sim.step()

    # Force an app update as an extra nudge for camera/render products.
    try:
        simulation_app.update()
    except Exception:
        pass


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-usd",
        type=str,
        required=True,
        help="Path to the USD scene.",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default="",
        help="Cup/table/object prim path. Used only to place the debug camera.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output observation npz path.",
    )

    parser.add_argument("--camera-path", type=str, default="/World/RGBD_Debug_Camera")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)

    parser.add_argument(
        "--camera-offset",
        nargs=3,
        type=float,
        default=[1.5, -1.5, 1.5],
        help="camera_pos = target + camera_offset. Default is a high oblique view.",
    )

    parser.add_argument("--depth-min", type=float, default=0.001)
    parser.add_argument("--depth-max", type=float, default=100000.0)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--load-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=80)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()

    # Very important: force camera/rendering mode on.
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    if unknown:
        print(f"[INFO] unknown args kept by parser: {unknown}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import omni.usd
    import isaaclab.sim as sim_utils

    try:
        from isaaclab.sensors import Camera, CameraCfg
    except Exception:
        from isaaclab.sensors.camera import Camera, CameraCfg

    try:
        from isaacsim.core.utils.stage import open_stage
    except Exception:
        from omni.isaac.core.utils.stage import open_stage

    scene_usd = Path(args.scene_usd)
    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD not found: {scene_usd}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] scene: {scene_usd}", flush=True)
    open_stage(str(scene_usd))

    for _ in range(args.load_steps):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    ensure_dome_light(stage)

    target = None
    if args.target_path:
        target = get_prim_bbox_center(stage, args.target_path)

    if target is None:
        target = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        print(f"[WARN] target-path invalid or bbox empty: {args.target_path}", flush=True)
        print(f"[INFO] fallback target xyz: {target}", flush=True)
    else:
        print(f"[INFO] target from bbox: {args.target_path}", flush=True)
        print(f"[INFO] target xyz: {target}", flush=True)

    camera_offset = np.array(args.camera_offset, dtype=np.float64)
    camera_pos = target + camera_offset

    print(f"[INFO] camera path:   {args.camera_path}", flush=True)
    print(f"[INFO] camera pos:    {camera_pos}", flush=True)
    print(f"[INFO] camera target: {target}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    camera_cfg = CameraCfg(
        prim_path=args.camera_path,
        update_period=0.0,
        height=args.cam_height,
        width=args.cam_width,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "distance_to_camera",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.001, 100000.0),
        ),
    )

    camera = Camera(cfg=camera_cfg)

    sim.reset()

    cam_pos_t = torch.tensor(
        np.asarray([camera_pos], dtype=np.float32),
        dtype=torch.float32,
        device=sim.device,
    )
    target_t = torch.tensor(
        np.asarray([target], dtype=np.float32),
        dtype=torch.float32,
        device=sim.device,
    )

    camera.set_world_poses_from_view(cam_pos_t, target_t)

    rgb = None
    depth_plane = None
    depth_camera = None

    for i in range(args.warmup_steps):
        simulation_step_with_render(sim, simulation_app)

        try:
            camera.update(dt=sim.get_physics_dt(), force_compute=True)
        except TypeError:
            camera.update(dt=sim.get_physics_dt())

        outputs = camera.data.output

        if "rgb" in outputs and outputs["rgb"] is not None:
            rgb = outputs["rgb"]

        if "distance_to_image_plane" in outputs and outputs["distance_to_image_plane"] is not None:
            depth_plane = outputs["distance_to_image_plane"]

        if "distance_to_camera" in outputs and outputs["distance_to_camera"] is not None:
            depth_camera = outputs["distance_to_camera"]

    if rgb is None:
        print("[ERROR] camera output missing rgb.", flush=True)
        np.savez_compressed(
            out_path.with_name(out_path.stem + "_missing_rgb_debug.npz"),
            target_xyz=target.astype(np.float32),
            camera_position=camera_pos.astype(np.float32),
        )
        return

    rgb_np = to_numpy(rgb)
    rgb_u8 = rgb_to_uint8(rgb_np)

    print(f"[SHAPE] rgb: {rgb_u8.shape} {rgb_u8.dtype}", flush=True)

    depth_plane_np = None
    depth_camera_np = None

    if depth_plane is not None:
        depth_plane_np = np.squeeze(to_numpy(depth_plane)).astype(np.float32)
        print_depth_stats("distance_to_image_plane", depth_plane_np)

    if depth_camera is not None:
        depth_camera_np = np.squeeze(to_numpy(depth_camera)).astype(np.float32)
        print_depth_stats("distance_to_camera", depth_camera_np)

    # Prefer image-plane depth. If it is unusable, fall back to distance_to_camera for raw debugging.
    depth_for_points = None
    depth_name = None

    if depth_plane_np is not None:
        valid_plane = np.isfinite(depth_plane_np) & (depth_plane_np > args.depth_min) & (depth_plane_np < args.depth_max)
        if int(valid_plane.sum()) > 0:
            depth_for_points = depth_plane_np
            depth_name = "distance_to_image_plane"

    if depth_for_points is None and depth_camera_np is not None:
        valid_camera = np.isfinite(depth_camera_np) & (depth_camera_np > args.depth_min) & (depth_camera_np < args.depth_max)
        if int(valid_camera.sum()) > 0:
            depth_for_points = depth_camera_np
            depth_name = "distance_to_camera_fallback"

    if depth_for_points is None:
        print("[ERROR] no valid depth from either distance_to_image_plane or distance_to_camera.", flush=True)

        # Save whatever we have for offline inspection.
        debug_npz = out_path.with_name(out_path.stem + "_no_valid_depth_debug.npz")
        np.savez_compressed(
            debug_npz,
            rgb=rgb_u8.astype(np.uint8),
            distance_to_image_plane=np.array([]) if depth_plane_np is None else depth_plane_np.astype(np.float32),
            distance_to_camera=np.array([]) if depth_camera_np is None else depth_camera_np.astype(np.float32),
            target_xyz=target.astype(np.float32),
            camera_position=camera_pos.astype(np.float32),
        )
        print(f"[SAVE] no-valid-depth debug npz: {debug_npz}", flush=True)

        if depth_plane_np is not None:
            save_preview_images(out_path, rgb_u8, depth_plane_np)
        elif depth_camera_np is not None:
            save_preview_images(out_path, rgb_u8, depth_camera_np)

        return

    print(f"[INFO] use depth source for points: {depth_name}", flush=True)

    if depth_for_points.shape[:2] != rgb_u8.shape[:2]:
        raise RuntimeError(
            f"RGB/depth shape mismatch: rgb={rgb_u8.shape}, depth={depth_for_points.shape}"
        )

    K = to_numpy(camera.data.intrinsic_matrices[0]).astype(np.float32)
    print(f"[INFO] intrinsics K:\n{K}", flush=True)

    points_all = unproject_depth_to_camera_points(depth_for_points, K)
    colors_all = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0

    depth_flat = depth_for_points.reshape(-1)
    valid_flat = (
        np.isfinite(depth_flat)
        & (depth_flat > args.depth_min)
        & (depth_flat < args.depth_max)
        & np.all(np.isfinite(points_all), axis=1)
    )

    print(f"[DEBUG] total pixels: {depth_flat.size}", flush=True)
    print(f"[DEBUG] valid depth pixels for point cloud: {int(valid_flat.sum())}", flush=True)

    points = points_all[valid_flat]
    colors = colors_all[valid_flat]

    if points.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(points.shape[0], size=args.max_points, replace=False)
        points = points[ids]
        colors = colors[ids]

    print(f"[SHAPE] points: {points.shape} {points.dtype}", flush=True)

    if points.shape[0] > 0:
        print(f"[RANGE] points camera min: {points.min(axis=0)}", flush=True)
        print(f"[RANGE] points camera max: {points.max(axis=0)}", flush=True)
    else:
        print("[ERROR] points is empty after filtering.", flush=True)

    save_preview_images(out_path, rgb_u8, depth_for_points)

    np.savez_compressed(
        out_path,
        rgb=rgb_u8.astype(np.uint8),
        depth=depth_for_points.astype(np.float32),
        depth_source=np.array(depth_name),
        intrinsics=K.astype(np.float32),

        # GraspNet-friendly fields.
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),

        # Debug fields.
        points_camera=points.astype(np.float32),
        valid_depth_mask=valid_flat.reshape(depth_for_points.shape).astype(bool),
        target_xyz=target.astype(np.float32),
        camera_position=camera_pos.astype(np.float32),
        camera_path=np.array(args.camera_path),
        target_path=np.array(args.target_path),
        scene_usd=np.array(str(scene_usd)),
    )

    print(f"[SAVE] observation npz: {out_path}", flush=True)
    print("[DONE] RGB-D observation export finished.", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        run()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        print("[FORCE_EXIT] os._exit now, avoid Isaac/Replicator cleanup hang.", flush=True)
        os._exit(exit_code)