from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np


def _collect_mesh_prims(root_prim) -> List:
    from pxr import UsdGeom

    meshes: List = []
    if root_prim.IsA(UsdGeom.Mesh):
        meshes.append(root_prim)
    for child in root_prim.GetChildren():
        meshes.extend(_collect_mesh_prims(child))
    return meshes


def _collect_boundable_prims(root_prim) -> List:
    """Collect non-Xform geometric primitives as fallback geometry.

    This catches USD analytic geometry / Gprim / Boundable assets that are
    renderable but not stored as UsdGeom.Mesh.
    """
    from pxr import UsdGeom

    out: List = []
    try:
        if (
            root_prim.IsA(UsdGeom.Boundable)
            or root_prim.IsA(UsdGeom.Gprim)
            or root_prim.IsA(UsdGeom.Mesh)
        ):
            out.append(root_prim)
    except Exception:
        pass

    for child in root_prim.GetChildren():
        out.extend(_collect_boundable_prims(child))
    return out


def _points_local_to_world(mesh_prim, points_local: np.ndarray) -> np.ndarray:
    from pxr import Gf, Usd, UsdGeom

    xform = UsdGeom.Xformable(mesh_prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    out = np.empty_like(points_local, dtype=np.float32)
    for i, p in enumerate(points_local):
        q = mat.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        out[i] = [q[0], q[1], q[2]]
    return out


def _triangulate_mesh_world(mesh_prim) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)

    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    if points is None or counts is None or indices is None:
        return np.zeros((0, 3, 3), dtype=np.float32)

    points_local = np.asarray(points, dtype=np.float32)
    points_world = _points_local_to_world(mesh_prim, points_local)

    triangles = []
    cursor = 0

    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count

        if count < 3:
            continue

        v0 = face[0]
        for j in range(1, count - 1):
            triangles.append(points_world[[v0, face[j], face[j + 1]]])

    if not triangles:
        return np.zeros((0, 3, 3), dtype=np.float32)

    return np.asarray(triangles, dtype=np.float32)


def _sample_triangles(triangles: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise RuntimeError(f"Invalid triangles shape: {triangles.shape}")

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    valid = np.isfinite(areas) & (areas > 1e-12)

    triangles = triangles[valid]
    areas = areas[valid]

    if len(triangles) == 0:
        raise RuntimeError("All mesh triangles are degenerate.")

    probs = areas / areas.sum()
    rng = np.random.default_rng(seed)

    ids = rng.choice(len(triangles), size=int(n_points), replace=True, p=probs)
    tri = triangles[ids]

    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]

    r1 = rng.random(int(n_points), dtype=np.float32)
    r2 = rng.random(int(n_points), dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)

    pts = (
        (1.0 - sqrt_r1)[:, None] * a
        + (sqrt_r1 * (1.0 - r2))[:, None] * b
        + (sqrt_r1 * r2)[:, None] * c
    )

    return pts.astype(np.float32)


def _world_aligned_bbox_for_prim(prim):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=True,
    )
    bbox = cache.ComputeWorldBound(prim)
    box = bbox.ComputeAlignedBox()

    mn = box.GetMin()
    mx = box.GetMax()

    mn = np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=np.float32)
    mx = np.array([float(mx[0]), float(mx[1]), float(mx[2])], dtype=np.float32)

    if not np.isfinite(mn).all() or not np.isfinite(mx).all():
        return None

    if np.any(mx < mn):
        return None

    extent = mx - mn
    if float(np.linalg.norm(extent)) < 1e-8:
        return None

    return mn, mx


def _box_surface_area(mn: np.ndarray, mx: np.ndarray) -> float:
    d = np.maximum(mx - mn, 0.0)
    dx, dy, dz = [float(x) for x in d]
    return 2.0 * (dx * dy + dx * dz + dy * dz)


def _sample_box_surface(mn: np.ndarray, mx: np.ndarray, n: int, rng) -> np.ndarray:
    d = np.maximum(mx - mn, 0.0)
    dx, dy, dz = [float(x) for x in d]

    face_areas = np.array(
        [
            dx * dy,  # z min
            dx * dy,  # z max
            dx * dz,  # y min
            dx * dz,  # y max
            dy * dz,  # x min
            dy * dz,  # x max
        ],
        dtype=np.float64,
    )

    if face_areas.sum() <= 1e-12:
        center = 0.5 * (mn + mx)
        return np.repeat(center[None, :], int(n), axis=0).astype(np.float32)

    probs = face_areas / face_areas.sum()
    face_ids = rng.choice(6, size=int(n), replace=True, p=probs)

    u = rng.random(int(n), dtype=np.float32)
    v = rng.random(int(n), dtype=np.float32)

    pts = np.zeros((int(n), 3), dtype=np.float32)

    for i, f in enumerate(face_ids):
        if f == 0:
            pts[i] = [mn[0] + u[i] * dx, mn[1] + v[i] * dy, mn[2]]
        elif f == 1:
            pts[i] = [mn[0] + u[i] * dx, mn[1] + v[i] * dy, mx[2]]
        elif f == 2:
            pts[i] = [mn[0] + u[i] * dx, mn[1], mn[2] + v[i] * dz]
        elif f == 3:
            pts[i] = [mn[0] + u[i] * dx, mx[1], mn[2] + v[i] * dz]
        elif f == 4:
            pts[i] = [mn[0], mn[1] + u[i] * dy, mn[2] + v[i] * dz]
        else:
            pts[i] = [mx[0], mn[1] + u[i] * dy, mn[2] + v[i] * dz]

    return pts


def _sample_boundable_boxes_world(boundable_prims: List, n_points: int, seed: int) -> Tuple[np.ndarray, Dict]:
    boxes = []
    for prim in boundable_prims:
        box = _world_aligned_bbox_for_prim(prim)
        if box is None:
            continue
        mn, mx = box
        area = _box_surface_area(mn, mx)
        if area <= 1e-12:
            continue
        boxes.append((str(prim.GetPath()), mn, mx, area))

    if not boxes:
        raise RuntimeError("No valid Boundable/Gprim world bounding boxes found for target object.")

    areas = np.array([b[3] for b in boxes], dtype=np.float64)
    probs = areas / areas.sum()

    rng = np.random.default_rng(seed)
    box_ids = rng.choice(len(boxes), size=int(n_points), replace=True, p=probs)

    counts = np.bincount(box_ids, minlength=len(boxes))
    chunks = []
    used_paths = []

    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        path, mn, mx, _area = boxes[idx]
        chunks.append(_sample_box_surface(mn, mx, int(count), rng))
        used_paths.append(path)

    points = np.concatenate(chunks, axis=0).astype(np.float32)

    # Shuffle so box order does not create blocks.
    perm = rng.permutation(points.shape[0])
    points = points[perm]

    meta = {
        "geometry_source": "boundable_bbox_fallback",
        "boundable_paths": used_paths,
        "num_boundable_boxes": len(used_paths),
    }
    return points, meta


def sample_prim_surface_points_world(
    stage,
    prim_path: str,
    *,
    n_points: int = 20000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid target prim path: {prim_path}")

    mesh_prims = _collect_mesh_prims(prim)

    all_triangles = []
    mesh_paths = []

    for mesh_prim in mesh_prims:
        tris = _triangulate_mesh_world(mesh_prim)
        if len(tris) > 0:
            all_triangles.append(tris)
            mesh_paths.append(str(mesh_prim.GetPath()))

    if all_triangles:
        triangles = np.concatenate(all_triangles, axis=0)
        points = _sample_triangles(triangles, n_points=n_points, seed=seed)
        geom_meta = {
            "geometry_source": "mesh_triangles",
            "mesh_paths": mesh_paths,
            "num_meshes": len(mesh_paths),
            "num_triangles": int(len(triangles)),
        }
    else:
        boundable_prims = _collect_boundable_prims(prim)
        if not boundable_prims:
            raise RuntimeError(
                f"No Mesh or Boundable/Gprim descendants under target prim: {prim_path}"
            )
        points, geom_meta = _sample_boundable_boxes_world(
            boundable_prims,
            n_points=n_points,
            seed=seed,
        )

    colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_extent = bbox_max - bbox_min

    meta = {
        "target_prim_path": prim_path,
        "num_points": int(points.shape[0]),
        "frame": "world",
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "bbox_center": bbox_center.astype(float).tolist(),
        "bbox_extent": bbox_extent.astype(float).tolist(),
    }
    meta.update(geom_meta)

    return points, colors, meta


def save_sim_target_cloud_npz(
    stage,
    prim_path: str,
    out_path: str | Path,
    *,
    target_class: str,
    n_points: int = 20000,
    seed: int = 0,
    extra_meta: Dict | None = None,
) -> Dict:
    points, colors, meta = sample_prim_surface_points_world(
        stage,
        prim_path,
        n_points=n_points,
        seed=seed,
    )

    meta = dict(meta)
    meta["target_class"] = target_class

    if extra_meta:
        meta.update(extra_meta)

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
        frame=np.array("world"),
        target_class=np.array(target_class),
        target_path=np.array(prim_path),
        target_prim_path=np.array(prim_path),
        bbox_min=np.asarray(meta["bbox_min"], dtype=np.float32),
        bbox_max=np.asarray(meta["bbox_max"], dtype=np.float32),
        bbox_center=np.asarray(meta["bbox_center"], dtype=np.float32),
        bbox_extent=np.asarray(meta["bbox_extent"], dtype=np.float32),
        metadata_json=np.array(json.dumps(meta)),
    )

    meta["observation_npz"] = str(out_path)
    return meta
