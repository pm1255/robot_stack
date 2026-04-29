from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def add_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-usd",
        type=str,
        default="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd",
    )
    parser.add_argument(
        "--cup-path",
        type=str,
        default="/World/office1/Room_seed123_idx000/furniture/tea_table/cup",
    )
    parser.add_argument(
        "--camera-path",
        type=str,
        default="",
        help="如果为空，会自动找第一个 Camera prim。建议后续显式传机器人头部/腕部相机路径。",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--crop-margin", type=float, default=0.06)

    parser.add_argument(
        "--robot-path",
        type=str,
        default="",
        help="机器人 prim path。为空时会按 agibot/a2d/robot 关键词自动找。",
    )
    parser.add_argument(
        "--debug-camera-path",
        type=str,
        default="/World/RobotDebugCamera",
        help="当场景里没有 Camera prim 时，自动创建的调试相机路径。",
    )
    parser.add_argument(
        "--create-debug-camera",
        action="store_true",
        help="即使场景里已有相机，也强制创建/使用调试相机。",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/pm/Desktop/Project/robot_stack/results/graspnet_cup_test",
    )

    parser.add_argument(
        "--graspnet-env",
        type=str,
        default="graspnet_env",
    )
    parser.add_argument(
        "--graspnet-root",
        type=str,
        default="/home/pm/Desktop/Project/graspnet-baseline",
    )
    parser.add_argument(
        "--graspnet-checkpoint",
        type=str,
        required=True,
        help="例如 /home/pm/Desktop/Project/graspnet-baseline/logs/log_kn/checkpoint-rs.tar",
    )
    parser.add_argument(
        "--graspnet-worker",
        type=str,
        default="/home/pm/Desktop/Project/robot_stack/isaac_collector/services/graspnet_worker.py",
    )
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--skip-graspnet", action="store_true")

    return parser


def gf_matrix_to_np(m):
    """
    USD/Gf Matrix4d internally follows row-vector convention, where translation
    is stored in the last row. We convert it to the standard column-vector
    convention used by NumPy here, where translation is in T[:3, 3].
    """
    raw = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    return raw.T


def transform_points(T, points):
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1
    )
    out = points_h @ T.T
    return out[:, :3]


def _prim_is_valid(stage, path: str) -> bool:
    if not path:
        return False
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def find_robot_prim_path(stage, preferred_robot_path: str = ""):
    if preferred_robot_path and _prim_is_valid(stage, preferred_robot_path):
        return preferred_robot_path

    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        low = path.lower()
        if any(k in low for k in ["agibot", "a2d", "robot"]):
            candidates.append(path)

    if not candidates:
        return ""

    # 越短通常越接近机器人根节点
    candidates = sorted(candidates, key=lambda x: (len(x), x))
    return candidates[0]


def create_camera_looking_at(stage, camera_path: str, eye, target):
    from pxr import UsdGeom, Sdf, Gf

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-8)

    # USD camera looks along local -Z, with local +Y as up.
    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # 如果视线几乎和 z-up 平行，换一个 up
    if abs(float(np.dot(up, z_axis))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-8)

    cam = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    prim = cam.GetPrim()

    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(15.2908)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    # Gf.Matrix4d uses row-vector convention:
    # rows are local axes in world coordinates, last row is translation.
    mat = Gf.Matrix4d(
        float(x_axis[0]), float(x_axis[1]), float(x_axis[2]), 0.0,
        float(y_axis[0]), float(y_axis[1]), float(y_axis[2]), 0.0,
        float(z_axis[0]), float(z_axis[1]), float(z_axis[2]), 0.0,
        float(eye[0]),    float(eye[1]),    float(eye[2]),    1.0,
    )

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(mat)

    return camera_path


def find_or_create_camera(
    stage,
    preferred_camera_path: str,
    debug_camera_path: str,
    robot_path: str,
    cup_center,
    force_create: bool = False,
):
    from pxr import UsdGeom

    if preferred_camera_path:
        prim = stage.GetPrimAtPath(preferred_camera_path)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
            return preferred_camera_path

    if not force_create:
        cameras = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cameras.append(str(prim.GetPath()))

        if cameras:
            def score(path: str):
                low = path.lower()
                s = 0
                for kw in ["robot", "a2d", "agibot", "head", "wrist", "camera", "rgb", "realsense"]:
                    if kw in low:
                        s += 1
                return s

            cameras = sorted(cameras, key=score, reverse=True)
            return cameras[0]

    robot_path = find_robot_prim_path(stage, robot_path)

    target = np.asarray(cup_center, dtype=np.float64) + np.array([0.0, 0.0, 0.05])

    if robot_path:
        T_world_robot = get_world_matrix(stage, robot_path)
        robot_pos = T_world_robot[:3, 3]

        direction = target - robot_pos
        direction[2] = 0.0
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        direction = direction / np.linalg.norm(direction)

        # 近似机器人头部/胸前视角：机器人根节点上方 1.35m，向 cup 方向略微前移
        eye = robot_pos + np.array([0.0, 0.0, 1.35]) + 0.25 * direction
        print(f"[INFO] Created debug robot-view camera from robot prim: {robot_path}")
    else:
        # 找不到机器人时退化为 cup 前上方视角
        eye = target + np.array([-1.2, 0.0, 0.75])
        print("[WARN] Robot prim not found; created fallback camera looking at cup.")

    return create_camera_looking_at(stage, debug_camera_path, eye, target)

def get_camera_intrinsics(stage, camera_path: str, width: int, height: int):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    cam = UsdGeom.Camera(prim)

    focal = float(cam.GetFocalLengthAttr().Get())
    h_ap = float(cam.GetHorizontalApertureAttr().Get())

    v_ap_attr = cam.GetVerticalApertureAttr().Get()
    if v_ap_attr is None or float(v_ap_attr) <= 0:
        v_ap = h_ap * height / width
    else:
        v_ap = float(v_ap_attr)

    fx = width * focal / h_ap
    fy = height * focal / v_ap
    cx = width / 2.0
    cy = height / 2.0

    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal_length": focal,
        "horizontal_aperture": h_ap,
        "vertical_aperture": v_ap,
    }


def get_world_matrix(stage, prim_path: str):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim 不存在: {prim_path}")

    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    return gf_matrix_to_np(mat)


def get_world_bbox(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"cup prim 不存在: {prim_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bmin = np.array(box.GetMin(), dtype=np.float64)
    bmax = np.array(box.GetMax(), dtype=np.float64)
    return bmin, bmax


def depth_to_pointcloud_usd_and_cv(depth, intrinsics):
    """
    输出两套点云：

    1. points_cam_usd:
       USD Camera local frame:
       +X right, +Y up, -Z forward

    2. points_cam_cv:
       常见 RGB-D / GraspNet camera frame:
       +X right, +Y down, +Z forward
    """
    h, w = depth.shape
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)

    valid = np.isfinite(z) & (z > 1e-4) & (z < 10.0)

    x_cv = (us - cx) * z / fx
    y_cv = (vs - cy) * z / fy
    z_cv = z

    points_cam_cv = np.stack([x_cv, y_cv, z_cv], axis=-1)

    x_usd = x_cv
    y_usd = -y_cv
    z_usd = -z_cv

    points_cam_usd = np.stack([x_usd, y_usd, z_usd], axis=-1)

    return points_cam_usd, points_cam_cv, valid


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def capture_rgbd(camera_path: str, width: int, height: int, simulation_app):
    """
    Capture RGB-D using Replicator annotators.

    In this pipeline, Replicator is more stable than isaacsim.sensors.camera.Camera
    under IsaacLab headless rendering. The Camera wrapper may block on
    get_rgba()/get_depth(), while Replicator annotators let us step frames
    explicitly and fail with a timeout.
    """
    import time
    import numpy as np

    print("[INFO] Importing omni.replicator.core ...", flush=True)
    import omni.replicator.core as rep

    print(f"[INFO] Creating render product for camera: {camera_path}", flush=True)
    render_product = rep.create.render_product(camera_path, (width, height))

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")

    rgb_annot.attach([render_product])
    depth_annot.attach([render_product])

    rgb = None
    depth = None

    max_steps = 120
    t0 = time.time()

    print("[INFO] Warming up renderer and waiting for RGB-D frame ...", flush=True)

    for i in range(max_steps):
        if i % 10 == 0:
            print(f"[INFO] RGB-D warmup step {i}/{max_steps}", flush=True)

        # rep.orchestrator.step() is the important part here.
        # simulation_app.update() alone may not flush annotator data.
        try:
            rep.orchestrator.step()
        except TypeError:
            rep.orchestrator.step(delta_time=0.0)

        simulation_app.update()

        try:
            rgb_data = rgb_annot.get_data()
            depth_data = depth_annot.get_data()
        except Exception as e:
            if i % 10 == 0:
                print(f"[WARN] annotator get_data failed at step {i}: {e}", flush=True)
            continue

        if rgb_data is None or depth_data is None:
            continue

        rgb_arr = np.asarray(rgb_data)
        depth_arr = np.asarray(depth_data)

        if rgb_arr.size == 0 or depth_arr.size == 0:
            continue

        # Sometimes the first few frames are all zero / invalid.
        if depth_arr.ndim >= 2:
            finite = np.isfinite(depth_arr)
            valid_depth = finite & (depth_arr > 1e-4) & (depth_arr < 20.0)
            valid_count = int(valid_depth.sum())
        else:
            valid_count = 0

        print(
            f"[INFO] step={i}, rgb_shape={rgb_arr.shape}, "
            f"depth_shape={depth_arr.shape}, valid_depth={valid_count}",
            flush=True,
        )

        if valid_count > 50:
            rgb = rgb_arr
            depth = depth_arr
            break

        if time.time() - t0 > 60:
            raise RuntimeError(
                "Timed out waiting for valid RGB-D frame. "
                "The camera exists, but renderer did not produce valid depth within 60s."
            )

    if rgb is None or depth is None:
        raise RuntimeError(
            "Failed to capture RGB-D after warmup. "
            "Try running non-headless first, or lower resolution with --width 320 --height 240."
        )

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.5:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return rgb, depth

def save_rgb_depth(results_dir: Path, rgb, depth):
    from PIL import Image

    Image.fromarray(rgb).save(results_dir / "rgb.png")
    np.save(results_dir / "depth.npy", depth)

    d = depth.copy()
    d[~np.isfinite(d)] = 0
    positive = d[d > 0]
    if positive.size > 0:
        lo, hi = np.percentile(positive, [1, 99])
        d_vis = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        d_vis = np.zeros_like(d)

    Image.fromarray((d_vis * 255).astype(np.uint8)).save(results_dir / "depth_vis.png")


def call_graspnet_worker(args, input_npz: Path, output_npz: Path):
    cmd = [
        "conda",
        "run",
        "-n",
        args.graspnet_env,
        "python",
        args.graspnet_worker,
        "--input",
        str(input_npz),
        "--output",
        str(output_npz),
        "--graspnet-root",
        args.graspnet_root,
        "--checkpoint",
        args.graspnet_checkpoint,
        "--topk",
        str(args.topk),
        "--num-point",
        str(args.num_point),
    ]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    print("[INFO] Running GraspNet worker:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, env=env)


def convert_grasps_camera_cv_to_world(grasps_npz: Path, T_world_cam_usd: np.ndarray):
    data = np.load(grasps_npz)

    scores = data["scores"]
    translations = data["translations"]
    rotations = data["rotations"]

    T_usd_from_cv = np.eye(4, dtype=np.float64)
    T_usd_from_cv[:3, :3] = np.diag([1.0, -1.0, -1.0])

    candidates = []

    for i in range(len(scores)):
        T_cam_cv_grasp = np.eye(4, dtype=np.float64)
        T_cam_cv_grasp[:3, :3] = rotations[i]
        T_cam_cv_grasp[:3, 3] = translations[i]

        T_world_grasp = T_world_cam_usd @ T_usd_from_cv @ T_cam_cv_grasp

        candidates.append(
            {
                "rank": int(i),
                "score": float(scores[i]),
                "translation_camera_cv": translations[i].tolist(),
                "rotation_camera_cv": rotations[i].tolist(),
                "T_world_grasp": T_world_grasp.tolist(),
            }
        )

    return candidates


def build_curobo_request_stub(candidates, results_dir: Path):
    """
    这里只生成后续 cuRobo 要吃的候选目标位姿。
    真正规划时，curobo_worker 需要读取 robot config、world collision、当前关节状态等。
    """
    requests = []

    for c in candidates:
        T = np.array(c["T_world_grasp"], dtype=np.float64)

        approach_axis = T[:3, 0]
        approach_axis = approach_axis / max(np.linalg.norm(approach_axis), 1e-8)

        T_pre = T.copy()
        T_pre[:3, 3] = T[:3, 3] - 0.08 * approach_axis

        T_lift = T.copy()
        T_lift[:3, 3] = T[:3, 3] + np.array([0.0, 0.0, 0.12])

        requests.append(
            {
                "rank": c["rank"],
                "score": c["score"],
                "pregrasp_T_world": T_pre.tolist(),
                "grasp_T_world": T.tolist(),
                "lift_T_world": T_lift.tolist(),
                "note": "后续顺序：plan_to_pregrasp -> approach -> close_gripper -> lift",
            }
        )

    save_json(results_dir / "curobo_request_candidates.json", requests)


def main():
    parser = add_args()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import omni.usd
    from PIL import Image  # noqa: F401

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Opening scene: {args.scene_usd}")

    ctx = omni.usd.get_context()
    ctx.open_stage(args.scene_usd)

    for _ in range(120):
        simulation_app.update()

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Stage 加载失败。")

    cup_T_world = get_world_matrix(stage, args.cup_path)
    cup_bmin, cup_bmax = get_world_bbox(stage, args.cup_path)
    cup_center = 0.5 * (cup_bmin + cup_bmax)

    camera_path = find_or_create_camera(
        stage=stage,
        preferred_camera_path=args.camera_path,
        debug_camera_path=args.debug_camera_path,
        robot_path=args.robot_path,
        cup_center=cup_center,
        force_create=args.create_debug_camera,
    )
    print(f"[INFO] Using camera: {camera_path}")

    print(f"[INFO] Cup path: {args.cup_path}")
    print(f"[INFO] Cup world position: {cup_T_world[:3, 3].tolist()}")
    print(f"[INFO] Cup bbox min: {cup_bmin.tolist()}")
    print(f"[INFO] Cup bbox max: {cup_bmax.tolist()}")

    intr = get_camera_intrinsics(stage, camera_path, args.width, args.height)
    T_world_cam_usd = get_world_matrix(stage, camera_path)

    save_json(results_dir / "camera_intrinsics.json", intr)
    save_json(
        results_dir / "camera_pose_world.json",
        {
            "camera_path": camera_path,
            "T_world_cam_usd": T_world_cam_usd.tolist(),
        },
    )
    save_json(
        results_dir / "cup_pose_world.json",
        {
            "cup_path": args.cup_path,
            "T_world_cup": cup_T_world.tolist(),
            "bbox_min": cup_bmin.tolist(),
            "bbox_max": cup_bmax.tolist(),
        },
    )

    rgb, depth = capture_rgbd(camera_path, args.width, args.height, simulation_app)
    save_rgb_depth(results_dir, rgb, depth)

    points_cam_usd, points_cam_cv, valid = depth_to_pointcloud_usd_and_cv(depth, intr)

    flat_usd = points_cam_usd.reshape(-1, 3)
    flat_cv = points_cam_cv.reshape(-1, 3)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    flat_valid = valid.reshape(-1)

    world_points = transform_points(T_world_cam_usd, flat_usd)

    margin = args.crop_margin
    crop_min = cup_bmin - margin
    crop_max = cup_bmax + margin

    crop_mask = (
        flat_valid
        & np.all(world_points >= crop_min[None, :], axis=1)
        & np.all(world_points <= crop_max[None, :], axis=1)
    )

    cup_points = flat_cv[crop_mask]
    cup_colors = flat_rgb[crop_mask]

    print(f"[INFO] Full valid points: {int(flat_valid.sum())}")
    print(f"[INFO] Cup crop points: {len(cup_points)}")

    if len(cup_points) < 200:
        raise RuntimeError(
            f"裁剪到的 cup 点云太少: {len(cup_points)}。"
            f"请检查 cup-path、camera-path，或者增大 --crop-margin。"
        )

    input_npz = results_dir / "graspnet_input_cup_crop.npz"
    np.savez_compressed(
        input_npz,
        points=cup_points.astype(np.float32),
        colors=cup_colors.astype(np.float32),
        cup_T_world=cup_T_world.astype(np.float64),
        T_world_cam_usd=T_world_cam_usd.astype(np.float64),
    )

    print(f"[INFO] Saved GraspNet input: {input_npz}")

    output_npz = results_dir / "graspnet_topk_camera_cv.npz"

    if not args.skip_graspnet:
        call_graspnet_worker(args, input_npz, output_npz)

        candidates = convert_grasps_camera_cv_to_world(output_npz, T_world_cam_usd)

        save_json(results_dir / "graspnet_topk_world.json", candidates)
        build_curobo_request_stub(candidates, results_dir)

        print(f"[OK] Saved top-k grasps: {results_dir / 'graspnet_topk_world.json'}")
        print(f"[OK] Saved cuRobo request stub: {results_dir / 'curobo_request_candidates.json'}")
    else:
        print("[INFO] skip-graspnet enabled; only saved RGB-D and cropped point cloud.")

    simulation_app.close()


if __name__ == "__main__":
    main()