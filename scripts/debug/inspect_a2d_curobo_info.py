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

loaded = load_scene_with_robot(
    scene_name=args.scene_name,
    robot_model=args.robot_model,
    num_envs=1,
    device="cuda:0",
    register_task_objects=True,
    verbose=True,
)

robot = loaded.robot

print("\n========== ROBOT BASIC INFO ==========")
print("robot:", robot)
print("robot class:", type(robot))
print("robot cfg:", getattr(robot, "cfg", None))

print("\n========== ROBOT PRIM PATH ==========")
try:
    print("prim_path:", robot.cfg.prim_path)
except Exception as e:
    print("cannot read robot.cfg.prim_path:", repr(e))

print("\n========== TRY READ USD PATH ==========")
try:
    spawn = robot.cfg.spawn
    print("spawn:", spawn)
    print("spawn class:", type(spawn))
    for key in ["usd_path", "asset_path", "path"]:
        if hasattr(spawn, key):
            print(f"spawn.{key} =", getattr(spawn, key))
except Exception as e:
    print("cannot read spawn usd path:", repr(e))

print("\n========== JOINT NAMES ==========")
for i, name in enumerate(robot.joint_names):
    print(f"{i:03d}: {name}")

print("\n========== BODY NAMES ==========")
for i, name in enumerate(robot.body_names):
    print(f"{i:03d}: {name}")

print("\n========== RIGHT/LEFT CANDIDATES ==========")
keywords = ["right", "left", "r_", "l_", "arm", "shoulder", "elbow", "wrist", "gripper", "hand"]
for i, name in enumerate(robot.joint_names):
    low = name.lower()
    if any(k in low for k in keywords):
        print(f"joint {i:03d}: {name}")

print("\n--- body candidates ---")
for i, name in enumerate(robot.body_names):
    low = name.lower()
    if any(k in low for k in keywords):
        print(f"body  {i:03d}: {name}")

simulation_app.close()
