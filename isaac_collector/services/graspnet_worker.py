from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def make_mock_top_down_grasp(object_pose_world: np.ndarray) -> Dict[str, Any]:
    """
    Debug version.

    用 cup 的世界位姿构造一个从上往下抓取的 grasp pose。
    这不是最终 GraspNet 输出，只是为了先跑通完整三环境链路。
    """

    object_pose_world = np.asarray(object_pose_world, dtype=np.float64)
    assert object_pose_world.shape == (4, 4)

    grasp = object_pose_world.copy()
    grasp[:3, :3] = np.eye(3)

    # 抓取点略高于杯子中心，避免直接穿模。
    grasp[:3, 3] = object_pose_world[:3, 3] + np.array([0.0, 0.0, 0.08])

    pregrasp = grasp.copy()
    pregrasp[:3, 3] += np.array([0.0, 0.0, 0.18])

    lift = grasp.copy()
    lift[:3, 3] += np.array([0.0, 0.0, 0.35])

    return {
        "success": True,
        "source": "mock_top_down_grasp",
        "score": 1.0,
        "width": 0.06,
        "pregrasp_pose_world": pregrasp.tolist(),
        "grasp_pose_world": grasp.tolist(),
        "lift_pose_world": lift.tolist(),
    }


def run_real_graspnet(request: Dict[str, Any], checkpoint: str | None) -> Dict[str, Any]:
    """
    后面真正接 GraspNet 的地方。

    真实版本应该读取：
    - observation npz
    - RGB-D
    - camera intrinsics
    - point cloud

    然后输出：
    - pregrasp_pose_world
    - grasp_pose_world
    - lift_pose_world
    """

    raise NotImplementedError(
        "Real GraspNet is not connected yet. "
        "Run with --mode mock first to verify the pipeline."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    try:
        if args.mode == "mock":
            object_pose = np.asarray(request["object_pose_world"], dtype=np.float64)
            response = make_mock_top_down_grasp(object_pose)
        else:
            response = run_real_graspnet(request, args.checkpoint)

    except Exception as e:
        response = {
            "success": False,
            "error": repr(e),
        }

    Path(args.response).write_text(json.dumps(response, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()