from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def _lazy_import_usd():
    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    return omni.usd, Usd, UsdGeom, Gf


def wait_frames(simulation_app, num_frames: int = 30):
    for _ in range(num_frames):
        simulation_app.update()


def open_usd_stage(scene_usd: str, simulation_app, wait: int = 60):
    omni_usd, _, _, _ = _lazy_import_usd()

    scene_path = Path(scene_usd).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene USD not found: {scene_path}")

    ctx = omni_usd.get_context()
    ctx.open_stage(str(scene_path))

    wait_frames(simulation_app, wait)

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {scene_path}")

    return stage


def gf_matrix_to_np(mat) -> np.ndarray:
    return np.array([[float(mat[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def np_to_gf_matrix(arr: np.ndarray):
    _, _, _, Gf = _lazy_import_usd()

    arr = np.asarray(arr, dtype=np.float64)
    assert arr.shape == (4, 4)

    return Gf.Matrix4d(
        float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2]), float(arr[0, 3]),
        float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2]), float(arr[1, 3]),
        float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2]), float(arr[2, 3]),
        float(arr[3, 0]), float(arr[3, 1]), float(arr[3, 2]), float(arr[3, 3]),
    )


def get_prim_world_matrix(stage, prim_path: str) -> np.ndarray:
    _, Usd, UsdGeom, _ = _lazy_import_usd()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    world_mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return gf_matrix_to_np(world_mat)


def set_prim_world_matrix(stage, prim_path: str, world_matrix: np.ndarray):
    _, Usd, UsdGeom, _ = _lazy_import_usd()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    world_gf = np_to_gf_matrix(world_matrix)

    parent = prim.GetParent()
    if parent and parent.IsValid() and str(parent.GetPath()) != "/":
        parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local_gf = world_gf * parent_world.GetInverse()
    else:
        local_gf = world_gf

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(local_gf)


def interpolate_pose_translation(
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
) -> List[np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    poses = []
    for i in range(steps):
        alpha = (i + 1) / steps
        pose = start.copy()
        pose[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * target[:3, 3]
        pose[:3, :3] = target[:3, :3]
        poses.append(pose)

    return poses