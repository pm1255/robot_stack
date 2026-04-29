import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene-name", type=str, default="mutilrooms_fixed")
parser.add_argument("--robot-model", type=str, default="a2d")
parser.add_argument("--out", type=str, default="/tmp/a2d_robot_info.json")
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
    register_task_objects=False,  # only export robot info; avoid RigidObject NVRTC issue
    verbose=True,
)

robot = loaded.robot

info = {}

info["robot_class"] = str(type(robot))
info["robot_cfg_type"] = str(type(getattr(robot, "cfg", None)))

try:
    info["prim_path"] = robot.cfg.prim_path
except Exception as e:
    info["prim_path_error"] = repr(e)

try:
    spawn = robot.cfg.spawn
    info["spawn_type"] = str(type(spawn))
    for key in ["usd_path", "asset_path", "path"]:
        if hasattr(spawn, key):
            info[f"spawn.{key}"] = str(getattr(spawn, key))
except Exception as e:
    info["spawn_error"] = repr(e)

info["joint_names"] = list(robot.joint_names)
info["body_names"] = list(robot.body_names)

keywords = ["right", "left", "arm", "shoulder", "elbow", "wrist", "gripper", "hand", "finger"]
info["joint_candidates"] = [
    [i, name] for i, name in enumerate(robot.joint_names)
    if any(k in name.lower() for k in keywords)
]
info["body_candidates"] = [
    [i, name] for i, name in enumerate(robot.body_names)
    if any(k in name.lower() for k in keywords)
]

Path(args.out).write_text(json.dumps(info, indent=2), encoding="utf-8")
print(f"[SAVED] {args.out}")

simulation_app.close()
