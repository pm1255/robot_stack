import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene-name", type=str, default="mutilrooms_fixed")
parser.add_argument("--robot-model", type=str, default="a2d")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

THIS = Path(__file__).resolve()
ISAAC_COLLECTOR = THIS.parents[1]
if str(ISAAC_COLLECTOR) not in sys.path:
    sys.path.insert(0, str(ISAAC_COLLECTOR))

from scene_loader import load_scene_with_robot
import omni.usd

loaded = load_scene_with_robot(
    scene_name=args.scene_name,
    robot_model=args.robot_model,
    num_envs=1,
    device="cuda:0",
    register_task_objects=False,
    verbose=False,
)

stage = omni.usd.get_context().get_stage()

print("\n========== A2D PRIMS AND REFERENCES ==========")

for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "A2D" not in path and "a2d" not in path:
        continue

    print("\nPRIM:", path)
    print("TYPE:", prim.GetTypeName())

    refs = prim.GetMetadata("references")
    payload = prim.GetMetadata("payload")
    asset_info = prim.GetAssetInfo()

    if refs:
        print("REFERENCES:", refs)
    if payload:
        print("PAYLOAD:", payload)
    if asset_info:
        print("ASSET_INFO:", asset_info)

    stack = prim.GetPrimStack()
    for spec in stack:
        layer = spec.layer
        print("  layer:", layer.identifier)

simulation_app.close()
