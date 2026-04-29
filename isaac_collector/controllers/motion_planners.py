"""
motion_planners.py

Unified motion planning interface.

Current implementation:
  - ManualJointPlanner: reads a user-provided joint trajectory / waypoint list.

Future implementation:
  - CuRoboPlanner: plan current joint state -> target EE pose with collision checking.
  - DifferentialIKPlanner: Isaac Lab differential IK for local movements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch


class BaseMotionPlanner:
    def plan_to_pose(self, *, robot_state: Dict[str, torch.Tensor], target_pose_w: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ManualJointPlanner(BaseMotionPlanner):
    """
    A planner for debugging before CuRobo is installed.

    It does NOT solve IK. It simply returns a stored joint trajectory.
    You can provide waypoints from a JSON file:
        [
          {"joint_a": 0.1, "joint_b": -0.2},
          {"joint_a": 0.2, "joint_b": -0.4}
        ]

    This is enough to test:
      - scene_loader
      - robot joint command path
      - gripper open/close
      - attach-mode object following
    """

    def __init__(self, *, robot_adapter, waypoint_file: Optional[str] = None):
        self.robot_adapter = robot_adapter
        self.waypoints: List[Dict[str, float]] = []
        if waypoint_file:
            self.load_waypoints(waypoint_file)

    def load_waypoints(self, waypoint_file: str) -> None:
        path = Path(waypoint_file)
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Waypoint JSON must be a list of dicts.")
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each waypoint must be a dict: {joint_name: value}.")
        self.waypoints = data

    def plan_to_pose(self, *, robot_state: Dict[str, torch.Tensor], target_pose_w: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.waypoints:
            raise RuntimeError(
                "ManualJointPlanner has no waypoints. It cannot solve IK. "
                "Provide --waypoint-file or use a real planner such as CuRobo."
            )

        q0 = robot_state["joint_pos"]
        num_envs, num_joints = q0.shape
        traj = []

        for wp in self.waypoints:
            q = q0.clone()
            for name, value in wp.items():
                if name not in self.robot_adapter.joint_names:
                    raise KeyError(f"Unknown joint in waypoint file: {name}")
                j = self.robot_adapter.joint_names.index(name)
                q[:, j] = float(value)
            traj.append(q)

        return torch.stack(traj, dim=0)


class CuRoboPlanner(BaseMotionPlanner):
    """
    Placeholder for future CuRobo integration.

    Once cuRobo is installed, this class should:
      1. Build a cuRobo robot config for A2D.
      2. Build a world collision config from Isaac Sim / USD.
      3. Convert target_pose_w to cuRobo goal pose.
      4. Return a joint trajectory tensor compatible with A2DRobotAdapter.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CuRoboPlanner is a placeholder. Install cuRobo later and implement "
            "this wrapper without changing ManipulationController."
        )
