from __future__ import annotations

from typing import Optional

from pxr import Gf, Usd, UsdGeom


HEAD_CAMERA_PATH = "/World/A2D/link_pitch_head/RGBD_Head_Camera"
RIGHT_WRIST_CAMERA_PATH = "/World/A2D/Link7_r/RGBD_Right_Wrist_Camera"
LEFT_WRIST_CAMERA_PATH = "/World/A2D/Link7_l/RGBD_Left_Wrist_Camera"

HEAD_LINK_PATH = "/World/A2D/link_pitch_head"
RIGHT_WRIST_LINK_PATH = "/World/A2D/Link7_r"
LEFT_WRIST_LINK_PATH = "/World/A2D/Link7_l"


def _get_world_pos(stage: Usd.Stage, prim_path: str) -> Optional[Gf.Vec3d]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[CAM_CALIB][WARN] invalid prim: {prim_path}", flush=True)
        return None

    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    p = mat.ExtractTranslation()
    return Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))


def _set_camera_world_lookat(
    stage: Usd.Stage,
    camera_path: str,
    eye_world: Gf.Vec3d,
    target_world: Gf.Vec3d,
    focal_length: float = 16.0,
    horizontal_aperture: float = 28.0,
    clipping_near: float = 0.01,
    clipping_far: float = 20.0,
) -> bool:
    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim or not cam_prim.IsValid():
        print(f"[CAM_CALIB][WARN] invalid camera: {camera_path}", flush=True)
        return False

    parent_prim = cam_prim.GetParent()
    if not parent_prim or not parent_prim.IsValid():
        print(f"[CAM_CALIB][WARN] invalid camera parent: {camera_path}", flush=True)
        return False

    if (target_world - eye_world).GetLength() < 1e-6:
        print(f"[CAM_CALIB][WARN] eye and target too close: {camera_path}", flush=True)
        return False

    up = Gf.Vec3d(0.0, 0.0, 1.0)

    world_mat = Gf.Matrix4d(1.0)
    world_mat.SetLookAt(eye_world, target_world, up)

    cache = UsdGeom.XformCache()
    parent_world = cache.GetLocalToWorldTransform(parent_prim)

    # USD xform composition here follows local * parent = world.
    local_mat = world_mat * parent_world.GetInverse()

    xformable = UsdGeom.Xformable(cam_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(local_mat)

    cam = UsdGeom.Camera(cam_prim)
    cam.GetFocalLengthAttr().Set(float(focal_length))
    cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(clipping_near), float(clipping_far)))

    print(
        f"[CAM_CALIB] set camera={camera_path}\n"
        f"            eye=({eye_world[0]:.4f}, {eye_world[1]:.4f}, {eye_world[2]:.4f})\n"
        f"         target=({target_world[0]:.4f}, {target_world[1]:.4f}, {target_world[2]:.4f})",
        flush=True,
    )
    return True


def calibrate_robot_rgbd_cameras(stage: Usd.Stage, cup_path: str) -> None:
    print("[CAM_CALIB] start robot RGB-D camera calibration", flush=True)

    cup_pos = _get_world_pos(stage, cup_path)
    head_pos = _get_world_pos(stage, HEAD_LINK_PATH)
    right_wrist_pos = _get_world_pos(stage, RIGHT_WRIST_LINK_PATH)
    left_wrist_pos = _get_world_pos(stage, LEFT_WRIST_LINK_PATH)

    if cup_pos is None:
        print(f"[CAM_CALIB][WARN] cannot find cup: {cup_path}", flush=True)
        return

    print(f"[CAM_CALIB] cup_pos=({cup_pos[0]:.4f}, {cup_pos[1]:.4f}, {cup_pos[2]:.4f})", flush=True)
    if head_pos is not None:
        print(f"[CAM_CALIB] head_pos=({head_pos[0]:.4f}, {head_pos[1]:.4f}, {head_pos[2]:.4f})", flush=True)
    if right_wrist_pos is not None:
        print(f"[CAM_CALIB] right_wrist_pos=({right_wrist_pos[0]:.4f}, {right_wrist_pos[1]:.4f}, {right_wrist_pos[2]:.4f})", flush=True)
    if left_wrist_pos is not None:
        print(f"[CAM_CALIB] left_wrist_pos=({left_wrist_pos[0]:.4f}, {left_wrist_pos[1]:.4f}, {left_wrist_pos[2]:.4f})", flush=True)

    cup_target = cup_pos + Gf.Vec3d(0.0, 0.0, 0.05)

    # 头部主相机：不要再默认看沙发，直接看杯子/桌面区域。
    # 第一版优先使用“杯子斜上方”稳定主视角，而不是机器人头部自身朝向。
    head_eye = cup_pos + Gf.Vec3d(0.70, -0.70, 0.60)
    _set_camera_world_lookat(
        stage,
        HEAD_CAMERA_PATH,
        head_eye,
        cup_target,
        focal_length=18.0,
        horizontal_aperture=24.0,
        clipping_near=0.01,
        clipping_far=20.0,
    )

    # 右腕相机：目前 depth 约 0.036m，说明太贴近 Link7。
    # 先把视点拉到右腕外侧 30cm、高 18cm，再看向杯子。
    if right_wrist_pos is not None:
        right_eye = right_wrist_pos + Gf.Vec3d(0.00, -0.30, 0.18)
        _set_camera_world_lookat(
            stage,
            RIGHT_WRIST_CAMERA_PATH,
            right_eye,
            cup_target,
            focal_length=14.0,
            horizontal_aperture=32.0,
            clipping_near=0.01,
            clipping_far=10.0,
        )

    # 左腕相机：同理，放到左腕外侧。
    if left_wrist_pos is not None:
        left_eye = left_wrist_pos + Gf.Vec3d(0.00, 0.30, 0.18)
        _set_camera_world_lookat(
            stage,
            LEFT_WRIST_CAMERA_PATH,
            left_eye,
            cup_target,
            focal_length=14.0,
            horizontal_aperture=32.0,
            clipping_near=0.01,
            clipping_far=10.0,
        )

    print("[CAM_CALIB] finished robot RGB-D camera calibration", flush=True)
