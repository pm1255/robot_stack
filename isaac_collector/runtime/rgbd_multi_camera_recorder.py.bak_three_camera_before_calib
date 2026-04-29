from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _as_np3(x, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}: {x}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf: {x}")
    return arr


def _safe_camera_tag(camera_path: str) -> str:
    s = camera_path.strip("/")
    s = s.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s or "camera"


def _to_gf_matrix4d(m: np.ndarray):
    from pxr import Gf

    m = np.asarray(m, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {m.shape}")

    return Gf.Matrix4d(
        float(m[0, 0]), float(m[0, 1]), float(m[0, 2]), float(m[0, 3]),
        float(m[1, 0]), float(m[1, 1]), float(m[1, 2]), float(m[1, 3]),
        float(m[2, 0]), float(m[2, 1]), float(m[2, 2]), float(m[2, 3]),
        float(m[3, 0]), float(m[3, 1]), float(m[3, 2]), float(m[3, 3]),
    )


def _normalize(v: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def _make_usd_camera_world_matrix_row(
    eye: Sequence[float],
    target: Sequence[float],
    up_hint: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Build a USD/Gf-style row-vector camera world transform.

    Project convention used by your code:
        p_world_h = p_local_h @ T_world

    USD camera convention:
        local -Z = forward/view direction
        local +Y = up
        local +X = right

    Therefore matrix rows are:
        row 0 = local +X axis in world
        row 1 = local +Y axis in world
        row 2 = local +Z axis in world = backward
        row 3 = translation
    """
    eye = _as_np3(eye, name="eye")
    target = _as_np3(target, name="target")
    up_hint = _as_np3(up_hint, name="up_hint")

    forward = _normalize(target - eye)

    # Avoid degenerate up direction.
    if abs(float(np.dot(forward, _normalize(up_hint)))) > 0.98:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=float)

    backward = -forward
    right = _normalize(np.cross(forward, up_hint))
    up = _normalize(np.cross(backward, right))

    m = np.eye(4, dtype=float)
    m[0, 0:3] = right
    m[1, 0:3] = up
    m[2, 0:3] = backward
    m[3, 0:3] = eye
    return m


def _ensure_xform_path(stage, path: str):
    """
    Define missing Xform parents for paths like /World/Foo/Bar.
    Does not redefine valid existing prims.
    """
    from pxr import Sdf, UsdGeom

    sdf_path = Sdf.Path(path)
    prefixes = sdf_path.GetPrefixes()

    # Skip the last prefix if the target itself is not intended to be a parent.
    for p in prefixes:
        if str(p) == "/":
            continue
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            UsdGeom.Xform.Define(stage, p)


def _define_camera(stage, camera_path: str):
    from pxr import Sdf, UsdGeom

    sdf_path = Sdf.Path(camera_path)
    parent_path = str(sdf_path.GetParentPath())

    if parent_path not in ("", "/"):
        _ensure_xform_path(stage, parent_path)

    cam = UsdGeom.Camera.Define(stage, sdf_path)
    return cam


def _set_xform_matrix(stage, prim_path: str, matrix: np.ndarray):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot set transform; prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(_to_gf_matrix4d(matrix))


def create_table_camera(
    stage,
    *,
    camera_path: str = "/World/RGBD_Table_Camera",
    eye: Sequence[float] = (3.4, -3.0, 2.2),
    target: Sequence[float] = (2.8273, -1.8654, 0.72),
    focal_length: float = 24.0,
    horizontal_aperture: float = 20.955,
    clipping_range: Tuple[float, float] = (0.01, 20.0),
):
    """
    Create a fixed RGB-D table camera without using viewport utilities.

    This is the key replacement for headless mode:
    - no omni.kit.viewport
    - no isaacsim.core.utils.viewports
    - only USD camera + Replicator render product later
    """
    from pxr import Gf

    cam = _define_camera(stage, camera_path)

    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(horizontal_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))

    camera_world = _make_usd_camera_world_matrix_row(
        eye=eye,
        target=target,
        up_hint=(0.0, 0.0, 1.0),
    )
    _set_xform_matrix(stage, camera_path, camera_world)

    print("[RGBD_CAMERA] created table camera", flush=True)
    print("[RGBD_CAMERA] path:", camera_path, flush=True)
    print("[RGBD_CAMERA] eye:", list(map(float, eye)), flush=True)
    print("[RGBD_CAMERA] target:", list(map(float, target)), flush=True)
    print("[RGBD_CAMERA] world matrix:", flush=True)
    print(camera_world, flush=True)

    return camera_path


def set_camera_local_transform(
    stage,
    camera_path: str,
    *,
    translate: Sequence[float] = (0.06, 0.0, 0.02),
    rotate_xyz_deg: Sequence[float] = (0.0, 90.0, 0.0),
    focal_length: float = 18.0,
    horizontal_aperture: float = 20.955,
    clipping_range: Tuple[float, float] = (0.01, 10.0),
):
    """
    Create a camera under an existing parent link and set its local transform.

    Example:
        /World/A2D/Link7_r/RGBD_Right_Wrist_Camera

    The parent link must already exist. This function intentionally does not
    use viewport utilities.
    """
    from pxr import Gf, Sdf, UsdGeom

    sdf_path = Sdf.Path(camera_path)
    parent_path = str(sdf_path.GetParentPath())
    parent = stage.GetPrimAtPath(parent_path)

    if not parent or not parent.IsValid():
        raise RuntimeError(
            f"Wrist camera parent does not exist: {parent_path}. "
            f"Cannot create camera: {camera_path}"
        )

    cam = UsdGeom.Camera.Define(stage, sdf_path)
    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(horizontal_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))

    prim = stage.GetPrimAtPath(camera_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    t = _as_np3(translate, name="translate")
    r = _as_np3(rotate_xyz_deg, name="rotate_xyz_deg")

    xform.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(r[0]), float(r[1]), float(r[2])))

    print("[RGBD_CAMERA] created local camera", flush=True)
    print("[RGBD_CAMERA] path:", camera_path, flush=True)
    print("[RGBD_CAMERA] parent:", parent_path, flush=True)
    print("[RGBD_CAMERA] translate:", t.tolist(), flush=True)
    print("[RGBD_CAMERA] rotate_xyz_deg:", r.tolist(), flush=True)

    return camera_path


def _extract_data(x):
    """
    Replicator annotators may return either an ndarray or a dict with 'data'.
    """
    if isinstance(x, dict):
        if "data" in x:
            x = x["data"]
        elif "value" in x:
            x = x["value"]
        else:
            # Last-resort: use the first array-like value.
            for v in x.values():
                try:
                    return np.asarray(v)
                except Exception:
                    pass
            raise ValueError(f"Cannot extract array from annotator dict keys={list(x.keys())}")
    return np.asarray(x)


def _rgb_to_uint8(rgb) -> np.ndarray:
    rgb = _extract_data(rgb)

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Invalid RGB shape: {rgb.shape}")

    if np.issubdtype(rgb.dtype, np.floating):
        if float(np.nanmax(rgb)) <= 1.5:
            rgb = np.clip(rgb * 255.0, 0, 255)
        else:
            rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


def _depth_to_float32(depth) -> np.ndarray:
    depth = _extract_data(depth)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 2:
        raise ValueError(f"Invalid depth shape: {depth.shape}")

    depth = depth.astype(np.float32)
    return depth


def _get_camera_intrinsics(stage, camera_path: str, *, width: int, height: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert USD camera focal length / aperture to pinhole intrinsics.

    USD focal_length and apertures are both in tenths of a scene unit-like camera
    unit; their ratio is what matters here:
        fx = width  * focal_length / horizontal_aperture
        fy = height * focal_length / vertical_aperture
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_path}")

    cam = UsdGeom.Camera(prim)

    focal_length = cam.GetFocalLengthAttr().Get()
    horizontal_aperture = cam.GetHorizontalApertureAttr().Get()
    vertical_aperture = cam.GetVerticalApertureAttr().Get()

    if focal_length is None:
        focal_length = 24.0
    if horizontal_aperture is None or float(horizontal_aperture) <= 0:
        horizontal_aperture = 20.955
    if vertical_aperture is None or float(vertical_aperture) <= 0:
        vertical_aperture = float(horizontal_aperture) * float(height) / float(width)

    fx = float(width) * float(focal_length) / float(horizontal_aperture)
    fy = float(height) * float(focal_length) / float(vertical_aperture)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    meta = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
        "vertical_aperture": float(vertical_aperture),
    }

    return K, meta


def _backproject_depth_to_usd_camera_points(
    depth: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backproject depth image to camera-local points.

    Replicator distance_to_image_plane depth is treated as positive distance
    along the view direction.

    USD camera local frame:
        +X right
        +Y up
        -Z forward

    Image frame:
        u right
        v down

    Therefore:
        x =  (u - cx) / fx * d
        y = -(v - cy) / fy * d
        z = -d
    """
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    valid = np.isfinite(depth) & (depth > 0.0)

    if not bool(valid.any()):
        return np.zeros((0, 3), dtype=np.float32), valid

    vv, uu = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    d = depth[valid]
    u = uu[valid]
    v = vv[valid]

    x = (u - cx) / fx * d
    y = -(v - cy) / fy * d
    z = -d

    points_camera = np.stack([x, y, z], axis=1).astype(np.float32)
    return points_camera, valid


def _transform_points_row(points_local: np.ndarray, T_world: np.ndarray) -> np.ndarray:
    """
    Transform points using row-vector convention:
        p_world_h = p_local_h @ T_world
    """
    points_local = np.asarray(points_local, dtype=np.float32)

    if points_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    T_world = np.asarray(T_world, dtype=float)
    ones = np.ones((points_local.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_local, ones], axis=1).astype(np.float64)
    world_h = points_h @ T_world
    return world_h[:, 0:3].astype(np.float32)


class MultiCameraRGBDRecorder:
    """
    Multi-camera RGB-D recorder for Isaac Sim / IsaacLab headless execution.

    It uses:
    - USD Camera prims
    - Replicator render products
    - rgb annotator
    - distance_to_image_plane annotator
    - manual depth backprojection to points / points_camera

    Output layout:
        output_dir/
          ep_0000/
            rgbd_frames/
              pickup/
                step_000000/
                  World__RGBD_Table_Camera.npz
              place/
                step_000000/
                  World__RGBD_Table_Camera.npz
    """

    def __init__(
        self,
        *,
        stage,
        camera_paths: Sequence[str],
        output_dir: Path | str,
        width: int = 640,
        height: int = 480,
        rt_subframes: int = 1,
        warmup_steps: int = 5,
    ):
        if not camera_paths:
            raise ValueError("camera_paths cannot be empty")

        self.stage = stage
        self.camera_paths = [str(p) for p in camera_paths]
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.width = int(width)
        self.height = int(height)
        self.rt_subframes = int(rt_subframes)
        self.warmup_steps = int(warmup_steps)

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid resolution: {self.width}x{self.height}")

        # Import Replicator only after SimulationApp/AppLauncher has started.
        import omni.kit.app

        self._kit_app = omni.kit.app.get_app()

        try:
            import omni.replicator.core as rep
        except ModuleNotFoundError:
            print("[RGBD] omni.replicator.core not loaded; trying to enable extension", flush=True)

            try:
                from isaacsim.core.utils.extensions import enable_extension
            except Exception:
                try:
                    from omni.isaac.core.utils.extensions import enable_extension
                except Exception:
                    enable_extension = None

            if enable_extension is not None:
                enable_extension("omni.replicator.core")
            else:
                ext_mgr = self._kit_app.get_extension_manager()
                ext_mgr.set_extension_enabled_immediate("omni.replicator.core", True)

            for _ in range(20):
                self._kit_app.update()

            import omni.replicator.core as rep

        self._rep = rep

        self._items = []

        for camera_path in self.camera_paths:
            prim = self.stage.GetPrimAtPath(camera_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Camera prim does not exist: {camera_path}")

            render_product = rep.create.render_product(camera_path, (self.width, self.height))

            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")

            self._attach_annotator(rgb_annotator, render_product)
            self._attach_annotator(depth_annotator, render_product)

            K, intr_meta = _get_camera_intrinsics(
                self.stage,
                camera_path,
                width=self.width,
                height=self.height,
            )

            self._items.append(
                {
                    "camera_path": camera_path,
                    "render_product": render_product,
                    "rgb_annotator": rgb_annotator,
                    "depth_annotator": depth_annotator,
                    "intrinsics": K,
                    "intrinsics_meta": intr_meta,
                }
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(max(0, self.warmup_steps)):
            self._kit_app.update()

        print("[RGBD] MultiCameraRGBDRecorder initialized", flush=True)
        print("[RGBD] output_dir:", str(self.output_dir), flush=True)
        print("[RGBD] resolution:", f"{self.width}x{self.height}", flush=True)
        print("[RGBD] cameras:", self.camera_paths, flush=True)

    @staticmethod
    def _attach_annotator(annotator, render_product):
        try:
            annotator.attach([render_product])
        except TypeError:
            annotator.attach(render_product)

    def _step_render(self):
        """
        Conservative render stepping.

        In this data-collection loop, physics / joint replay is already advanced
        by simulation_app.update(). Here we only need to let Kit refresh render
        products and annotator buffers. This avoids first-call blocking observed
        with rep.orchestrator.step() in some headless rendering setups.
        """
        for _ in range(max(1, int(self.rt_subframes))):
            self._kit_app.update()

    def capture(
        self,
        *,
        ep: int,
        phase: str,
        step_index: int,
        extra: Optional[dict] = None,
    ) -> Dict[str, str]:
        """
        Capture all configured cameras and return:
            {camera_path: saved_npz_path}
        """
        ep = int(ep)
        step_index = int(step_index)
        phase = str(phase)
        extra = extra or {}

        self._step_render()

        saved: Dict[str, str] = {}

        for item in self._items:
            camera_path = item["camera_path"]
            camera_tag = _safe_camera_tag(camera_path)

            out_dir = (
                self.output_dir
                / f"ep_{ep:04d}"
                / "rgbd_frames"
                / phase
                / f"step_{step_index:06d}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{camera_tag}.npz"

            rgb_raw = item["rgb_annotator"].get_data()
            depth_raw = item["depth_annotator"].get_data()

            rgb = _rgb_to_uint8(rgb_raw)
            depth = _depth_to_float32(depth_raw)

            if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
                raise ValueError(
                    f"RGB shape mismatch for {camera_path}: "
                    f"got {rgb.shape}, expected ({self.height}, {self.width}, 3)"
                )

            if depth.shape[0] != self.height or depth.shape[1] != self.width:
                raise ValueError(
                    f"Depth shape mismatch for {camera_path}: "
                    f"got {depth.shape}, expected ({self.height}, {self.width})"
                )

            from isaac_collector.runtime.load_scene import get_prim_world_matrix

            camera_world = np.asarray(
                get_prim_world_matrix(self.stage, camera_path),
                dtype=float,
            )

            K = np.asarray(item["intrinsics"], dtype=np.float32)
            points_camera, valid_mask = _backproject_depth_to_usd_camera_points(depth, K)
            points_world = _transform_points_row(points_camera, camera_world)

            rgb_float = rgb.astype(np.float32) / 255.0
            colors = rgb_float[valid_mask].reshape(-1, 3).astype(np.float32)

            if points_world.shape[0] != colors.shape[0]:
                raise ValueError(
                    f"points/colors mismatch for {camera_path}: "
                    f"{points_world.shape[0]} vs {colors.shape[0]}"
                )

            meta = {
                "camera_path": camera_path,
                "phase": phase,
                "step_index": step_index,
                "ep": ep,
                "width": self.width,
                "height": self.height,
                "extra": extra,
                "intrinsics_meta": item["intrinsics_meta"],
            }

            np.savez_compressed(
                out_path,
                rgb=rgb,
                depth=depth.astype(np.float32),
                valid_depth_mask=valid_mask.astype(bool),
                points=points_world.astype(np.float32),
                points_camera=points_camera.astype(np.float32),
                colors=colors.astype(np.float32),
                intrinsics=K.astype(np.float32),
                camera_world=camera_world.astype(np.float32),
                camera_path=np.array(camera_path),
                phase=np.array(phase),
                step_index=np.array(step_index, dtype=np.int64),
                ep=np.array(ep, dtype=np.int64),
                metadata_json=np.array(json.dumps(meta, ensure_ascii=False)),
            )

            print(
                f"[RGBD] saved {phase} step={step_index} camera={camera_path} "
                f"points={points_world.shape[0]} -> {out_path}",
                flush=True,
            )

            saved[camera_path] = str(out_path)

        return saved

def _path_join(parent: str, child: str) -> str:
    parent = parent.rstrip("/")
    child = child.strip("/")
    return parent + "/" + child


def find_first_valid_prim_path(stage, candidates):
    """
    Return the first valid prim path from a candidate list.
    """
    for p in candidates:
        if not p:
            continue
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            return str(p)
    raise RuntimeError(f"Cannot find any valid prim from candidates: {candidates}")


def create_bound_camera_look_at(
    stage,
    *,
    camera_path: str,
    parent_path: str,
    eye_world,
    target_world,
    focal_length: float = 18.0,
    horizontal_aperture: float = 20.955,
    clipping_range=(0.01, 20.0),
):
    """
    Create a USD Camera under parent_path, but initialize it by a world-space
    eye/target look-at pose.

    After creation, the camera is a child of parent_path, so it will move with
    the robot when the robot navigates.

    Row-vector convention used by this project:
        child_world = child_local @ parent_world
        child_local = child_world @ inv(parent_world)
    """
    from pxr import Gf, Sdf, UsdGeom
    from isaac_collector.runtime.load_scene import get_prim_world_matrix

    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        raise RuntimeError(f"Camera parent does not exist: {parent_path}")

    # If the user passed empty camera_path, create under parent.
    if not camera_path:
        camera_path = _path_join(parent_path, "RGBD_Camera")

    sdf_path = Sdf.Path(camera_path)
    expected_parent = str(sdf_path.GetParentPath())

    if expected_parent != parent_path:
        raise ValueError(
            f"camera_path must be directly under parent_path.\n"
            f"  camera_path: {camera_path}\n"
            f"  parent_path: {parent_path}\n"
            f"  actual parent of camera_path: {expected_parent}"
        )

    cam = UsdGeom.Camera.Define(stage, sdf_path)
    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(horizontal_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))

    camera_world = _make_usd_camera_world_matrix_row(
        eye=eye_world,
        target=target_world,
        up_hint=(0.0, 0.0, 1.0),
    )

    parent_world = np.asarray(get_prim_world_matrix(stage, parent_path), dtype=float)
    camera_local = camera_world @ np.linalg.inv(parent_world)

    _set_xform_matrix(stage, camera_path, camera_local)

    print("[RGBD_CAMERA_BOUND] created robot-bound camera", flush=True)
    print("[RGBD_CAMERA_BOUND] camera_path:", camera_path, flush=True)
    print("[RGBD_CAMERA_BOUND] parent_path:", parent_path, flush=True)
    print("[RGBD_CAMERA_BOUND] eye_world:", np.asarray(eye_world, dtype=float).tolist(), flush=True)
    print("[RGBD_CAMERA_BOUND] target_world:", np.asarray(target_world, dtype=float).tolist(), flush=True)
    print("[RGBD_CAMERA_BOUND] camera_world:", flush=True)
    print(camera_world, flush=True)
    print("[RGBD_CAMERA_BOUND] camera_local:", flush=True)
    print(camera_local, flush=True)

    return camera_path


def create_bound_camera_from_parent_position_look_at(
    stage,
    *,
    camera_path: str,
    parent_path: str,
    target_world,
    world_offset=(0.0, 0.0, 0.03),
    focal_length: float = 18.0,
    horizontal_aperture: float = 20.955,
    clipping_range=(0.01, 10.0),
):
    """
    Create a robot-bound camera whose initial eye position is:
        parent_world_translation + world_offset

    This is convenient for wrist cameras:
        parent_path = /World/A2D/Link7_r or /World/A2D/Link7_l
        target_world = cup/table region
    """
    from isaac_collector.runtime.load_scene import get_prim_world_matrix

    parent_world = np.asarray(get_prim_world_matrix(stage, parent_path), dtype=float)
    parent_xyz = parent_world[3, 0:3].copy()

    eye_world = parent_xyz + np.asarray(world_offset, dtype=float)

    return create_bound_camera_look_at(
        stage,
        camera_path=camera_path,
        parent_path=parent_path,
        eye_world=eye_world,
        target_world=target_world,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        clipping_range=clipping_range,
    )


def create_camera_under_parent_look_at_world(
    stage,
    *,
    camera_path: str,
    parent_path: str,
    eye_world,
    target_world,
    focal_length: float = 18.0,
    horizontal_aperture: float = 20.955,
    clipping_range=(0.01, 10.0),
):
    """
    Create a camera under parent_path, initialized by a world-space look-at pose.

    The camera is a child of parent_path, so when the robot/base/link moves,
    the camera moves with it.

    Project row-vector convention:
        child_world = child_local @ parent_world
        child_local = child_world @ inv(parent_world)
    """
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom
    from isaac_collector.runtime.load_scene import get_prim_world_matrix

    parent = stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        raise RuntimeError(f"Camera parent does not exist: {parent_path}")

    sdf_path = Sdf.Path(camera_path)
    actual_parent = str(sdf_path.GetParentPath())
    if actual_parent != parent_path:
        raise ValueError(
            f"camera_path must be directly under parent_path\n"
            f"camera_path={camera_path}\n"
            f"parent_path={parent_path}\n"
            f"actual_parent={actual_parent}"
        )

    cam = UsdGeom.Camera.Define(stage, sdf_path)
    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(horizontal_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))

    camera_world = _make_usd_camera_world_matrix_row(
        eye=eye_world,
        target=target_world,
        up_hint=(0.0, 0.0, 1.0),
    )

    parent_world = np.asarray(get_prim_world_matrix(stage, parent_path), dtype=float)
    camera_local = camera_world @ np.linalg.inv(parent_world)

    _set_xform_matrix(stage, camera_path, camera_local)

    print("[RGBD_CAMERA_BOUND] created camera", flush=True)
    print("[RGBD_CAMERA_BOUND] camera_path:", camera_path, flush=True)
    print("[RGBD_CAMERA_BOUND] parent_path:", parent_path, flush=True)
    print("[RGBD_CAMERA_BOUND] eye_world:", np.asarray(eye_world, dtype=float).tolist(), flush=True)
    print("[RGBD_CAMERA_BOUND] target_world:", np.asarray(target_world, dtype=float).tolist(), flush=True)
    print("[RGBD_CAMERA_BOUND] camera_local:", flush=True)
    print(camera_local, flush=True)

    return camera_path


def find_first_existing_prim(stage, candidates):
    for p in candidates:
        if not p:
            continue
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            return p
    raise RuntimeError(f"Cannot find valid prim from candidates: {candidates}")
