"""
grasp_generators.py

Unified grasp generation interface.

Current implementation:
  - RuleBasedGraspGenerator: does not require GraspNet; creates a simple top-down
    grasp pose from object pose.

Future implementation:
  - GraspNetGenerator: render RGB-D / point cloud from Isaac Sim, call GraspNet,
    transform grasp candidates into world frame.
  - AnyGraspGenerator: same interface, different model backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class GraspCandidate:
    pose_w: torch.Tensor  # shape [num_envs, 7], [x,y,z,qw,qx,qy,qz]
    score: float = 1.0
    width: Optional[float] = None
    source: str = "rule_based"


class BaseGraspGenerator:
    def generate(self, *, object_name: str, object_pose_w: torch.Tensor, **kwargs) -> List[GraspCandidate]:
        raise NotImplementedError


class RuleBasedGraspGenerator(BaseGraspGenerator):
    """
    A minimal grasp generator for debugging.

    It creates a grasp pose above the object by lifting z by approach_height.
    The orientation is kept as identity quaternion by default.

    This is NOT a real grasp predictor. It only allows the manipulation pipeline
    to be developed before GraspNet / AnyGrasp is installed.
    """

    def __init__(
        self,
        *,
        approach_height: float = 0.12,
        identity_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ):
        self.approach_height = approach_height
        self.identity_quat_wxyz = identity_quat_wxyz

    def generate(self, *, object_name: str, object_pose_w: torch.Tensor, **kwargs) -> List[GraspCandidate]:
        pose = object_pose_w.clone()
        pose[:, 2] += self.approach_height

        quat = torch.tensor(
            self.identity_quat_wxyz,
            dtype=pose.dtype,
            device=pose.device,
        ).view(1, 4).repeat(pose.shape[0], 1)

        pose[:, 3:7] = quat
        return [GraspCandidate(pose_w=pose, score=1.0, source="rule_based")]


class GraspNetGenerator(BaseGraspGenerator):
    """
    Placeholder for future GraspNet integration.

    Expected future input:
      - RGB-D image or point cloud
      - camera intrinsics
      - camera pose
      - optional object mask / crop

    Expected output:
      - List[GraspCandidate], each in world frame.

    Do not instantiate this class until GraspNet is installed and the wrapper
    is implemented.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GraspNetGenerator is a placeholder. Install GraspNet/AnyGrasp later "
            "and implement this wrapper without changing ManipulationController."
        )
