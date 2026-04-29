
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene-name", type=str, default="mutilrooms")
parser.add_argument("--robot-model", type=str, default="a2d")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

THIS = Path(__file__).resolve()
ISAAC_COLLECTOR = THIS.parents[1]
if str(ISAAC_COLLECTOR) not in sys.path:
    sys.path.insert(0, str(ISAAC_COLLECTOR))

from scene_loader import load_scene_with_robot, get_stage
from pxr import UsdGeom, UsdPhysics

def main():
    loaded = load_scene_with_robot(
        scene_name=args.scene_name,
        robot_model=args.robot_model,
        num_envs=1,
        device="cuda:0",
        register_task_objects=True,
        verbose=False,
    )

    stage = get_stage()

    keywords = [
        "tea_table",
        "office_desk",
        "cup",
    ]

    print("\n[CHECK] relevant prims:")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        path_lc = path.lower()

        if not any(k in path_lc for k in keywords):
            continue

        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        visible = "unknown"

        try:
            imageable = UsdGeom.Imageable(prim)
            visible = imageable.ComputeVisibility()
        except Exception:
            pass

        try:
            xform = UsdGeom.Xformable(prim)
            mat = xform.ComputeLocalToWorldTransform(0)
            pos = mat.ExtractTranslation()
            pos = [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)]
        except Exception:
            pos = None

        print(f"path={path}")
        print(f"  type={prim.GetTypeName()}, rigid={has_rb}, visible={visible}, world_pos={pos}")

    loaded.sim.set_camera_view([4.0, -4.2, 2.2], [2.8, -1.9, 0.8])
    for _ in range(120):
        simulation_app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()
