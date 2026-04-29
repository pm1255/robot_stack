from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def run_mock_curobo(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug version.

    真实 cuRobo 应该输出 joint_names + positions。
    这里先输出 cartesian waypoints，方便 IsaacLab 侧验证整个调用链路。
    """

    grasp_result = request["grasp_result"]

    return {
        "success": True,
        "source": "mock_curobo_cartesian_waypoints",
        "trajectory_type": "cartesian_waypoints_debug",
        "cartesian_waypoints_world": [
            grasp_result["pregrasp_pose_world"],
            grasp_result["grasp_pose_world"],
            grasp_result["lift_pose_world"],
        ],
        "joint_names": [],
        "positions": [],
        "dt": 0.02,
    }


def run_real_curobo(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    后面真正接 cuRobo 的地方。

    真实版本应该读取：
    - robot_state
    - world_collision
    - grasp_result

    然后输出：
    {
      "success": true,
      "trajectory_type": "joint_trajectory",
      "joint_names": [...],
      "positions": [[...], [...], ...],
      "dt": 0.02
    }
    """

    raise NotImplementedError(
        "Real cuRobo planning is not connected yet. "
        "Run with --mode mock first to verify the pipeline."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    try:
        if args.mode == "mock":
            response = run_mock_curobo(request)
        else:
            response = run_real_curobo(request)

    except Exception as e:
        response = {
            "success": False,
            "error": repr(e),
        }

    Path(args.response).write_text(json.dumps(response, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()