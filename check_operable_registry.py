from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    p.add_argument("--scene-usd", required=True)
    p.add_argument("--scene-registry-json", required=True)
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from isaaclab.app import AppLauncher
    from isaac_collector.runtime.load_scene import open_usd_stage
    from isaac_collector.runtime.operable_scene_registry import load_operable_objects_from_registry

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    try:
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)
        objs = load_operable_objects_from_registry(
            stage=stage,
            registry_json=args.scene_registry_json,
            validate_in_stage=True,
            require_mesh=True,
        )
        print("[OPERABLE] count:", len(objs), flush=True)
        for obj in objs:
            print(obj.to_dict(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
