"""
robot_adapter.py

Robot-specific adapter layer for Isaac Lab Articulation.

This adapter is intentionally independent of GraspNet and CuRobo.
It only provides:
- joint/body inspection
- end-effector body selection
- arm/gripper joint selection
- joint target execution
- gripper open/close
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _get_names(asset, attr_name: str) -> List[str]:
    if hasattr(asset, attr_name):
        names = _as_list(getattr(asset, attr_name))
        if names:
            return names

    if hasattr(asset, "data") and hasattr(asset.data, attr_name):
        names = _as_list(getattr(asset.data, attr_name))
        if names:
            return names

    return []


class A2DRobotAdapter:
    """
    Adapter for A2D-like dual-arm robot.

    Recommended EE keywords:
    - Link7_l : left arm wrist / end link
    - Link7_r : right arm wrist / end link

    This adapter selects:
    - Link7_l -> left_arm_joint1-7 + left gripper joints
    - Link7_r -> right_arm_joint1-7 + right gripper joints
    """

    def __init__(
        self,
        robot,
        *,
        ee_keyword: str = "Link7_l",
        explicit_arm_joints: Optional[Sequence[str]] = None,
        explicit_gripper_joints: Optional[Sequence[str]] = None,
        gripper_open_value: float = 0.035,
        gripper_close_value: float = 0.0,
    ):
        self.robot = robot
        self.device = getattr(robot, "device", "cuda:0")

        self.joint_names: List[str] = _get_names(robot, "joint_names")
        self.body_names: List[str] = _get_names(robot, "body_names")

        if not self.joint_names:
            raise RuntimeError(
                "Cannot read robot joint_names. Make sure robot is an Isaac Lab Articulation "
                "and sim.reset() has been called."
            )

        if not self.body_names:
            raise RuntimeError(
                "Cannot read robot body_names. Make sure robot is an Isaac Lab Articulation "
                "and sim.reset() has been called."
            )

        self.ee_keyword = ee_keyword
        self.ee_body_id = self.find_body_id(ee_keyword)

        ee_lc = ee_keyword.lower()

        # ------------------------------------------------------------
        # Select arm joints
        # ------------------------------------------------------------
        if explicit_arm_joints is not None:
            self.arm_joint_names = list(explicit_arm_joints)
        elif self._is_left_ee(ee_lc):
            self.arm_joint_names = [
                f"left_arm_joint{i}"
                for i in range(1, 8)
                if f"left_arm_joint{i}" in self.joint_names
            ]
        elif self._is_right_ee(ee_lc):
            self.arm_joint_names = [
                f"right_arm_joint{i}"
                for i in range(1, 8)
                if f"right_arm_joint{i}" in self.joint_names
            ]
        else:
            self.arm_joint_names = [
                name for name in self.joint_names
                if "arm_joint" in name.lower()
            ]

        # ------------------------------------------------------------
        # Select gripper joints
        # ------------------------------------------------------------
        if explicit_gripper_joints is not None:
            self.gripper_joint_names = list(explicit_gripper_joints)
        elif self._is_left_ee(ee_lc):
            self.gripper_joint_names = [
                name for name in ("left_Left_2_Joint", "left_Right_2_Joint")
                if name in self.joint_names
            ]
        elif self._is_right_ee(ee_lc):
            self.gripper_joint_names = [
                name for name in ("right_Left_2_Joint", "right_Right_2_Joint")
                if name in self.joint_names
            ]
        else:
            self.gripper_joint_names = [
                name for name in self.joint_names
                if "2_joint" in name.lower()
            ]

        self._validate_selected_joints()

        self.arm_joint_ids = self.joint_names_to_ids(self.arm_joint_names)
        self.gripper_joint_ids = self.joint_names_to_ids(self.gripper_joint_names)

        self.gripper_open_value = float(gripper_open_value)
        self.gripper_close_value = float(gripper_close_value)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _is_left_ee(self, ee_lc: str) -> bool:
        return (
            ee_lc.endswith("_l")
            or "link7_l" in ee_lc
            or "left" in ee_lc
            or ee_lc.startswith("left_")
        )

    def _is_right_ee(self, ee_lc: str) -> bool:
        return (
            ee_lc.endswith("_r")
            or "link7_r" in ee_lc
            or "right" in ee_lc
            or ee_lc.startswith("right_")
        )

    def _validate_selected_joints(self) -> None:
        missing_arm = [name for name in self.arm_joint_names if name not in self.joint_names]
        missing_gripper = [name for name in self.gripper_joint_names if name not in self.joint_names]

        if missing_arm:
            raise KeyError(f"Selected arm joints not found: {missing_arm}")

        if missing_gripper:
            raise KeyError(f"Selected gripper joints not found: {missing_gripper}")

        if not self.arm_joint_names:
            print("[WARN] No arm joints selected. Check ee_keyword or pass explicit_arm_joints.")

        if not self.gripper_joint_names:
            print("[WARN] No gripper joints selected. Check ee_keyword or pass explicit_gripper_joints.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def print_robot_info(self) -> None:
        print("\n[ROBOT] joint names:")
        for i, name in enumerate(self.joint_names):
            tags = []
            if name in self.arm_joint_names:
                tags.append("ARM")
            if name in self.gripper_joint_names:
                tags.append("GRIPPER")

            suffix = f"  <-- {','.join(tags)}" if tags else ""
            print(f"  [{i:03d}] {name}{suffix}")

        print("\n[ROBOT] body names:")
        for i, name in enumerate(self.body_names):
            suffix = "  <-- EE" if i == self.ee_body_id else ""
            print(f"  [{i:03d}] {name}{suffix}")

        print("\n[ROBOT] selected EE body:")
        print(f"  id={self.ee_body_id}, name={self.body_names[self.ee_body_id]}")

        print("\n[ROBOT] selected arm joints:")
        if self.arm_joint_names:
            for name in self.arm_joint_names:
                print(f"  - {name}")
        else:
            print("  <none>")

        print("\n[ROBOT] selected gripper joints:")
        if self.gripper_joint_names:
            for name in self.gripper_joint_names:
                print(f"  - {name}")
        else:
            print("  <none>")

    def find_body_id(self, keyword: str) -> int:
        keyword_lc = keyword.lower()

        # First exact match.
        for i, name in enumerate(self.body_names):
            if name.lower() == keyword_lc:
                return i

        # Then substring match.
        for i, name in enumerate(self.body_names):
            if keyword_lc in name.lower():
                return i

        msg = [f"Cannot find EE body with keyword='{keyword}'. Available bodies:"]
        msg += [f"  [{i}] {name}" for i, name in enumerate(self.body_names)]
        raise RuntimeError("\n".join(msg))

    def joint_names_to_ids(self, names: Sequence[str]) -> List[int]:
        ids: List[int] = []

        for name in names:
            if name not in self.joint_names:
                raise KeyError(f"Joint '{name}' not found in robot.joint_names.")
            ids.append(self.joint_names.index(name))

        return ids

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.clone()

    def get_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.clone()

    def get_robot_state(self) -> Dict[str, torch.Tensor]:
        return {
            "joint_pos": self.get_joint_pos(),
            "joint_vel": self.get_joint_vel(),
        }

    def get_ee_pose_w(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self.robot.data.body_pos_w[:, self.ee_body_id].clone()
        quat = self.robot.data.body_quat_w[:, self.ee_body_id].clone()
        return pos, quat

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_joint_position_target_full(self, q_target: torch.Tensor) -> None:
        self.robot.set_joint_position_target(q_target)

    def set_named_joint_targets(self, joint_targets: Dict[str, float]) -> None:
        q = self.robot.data.joint_pos.clone()

        for joint_name, value in joint_targets.items():
            if joint_name not in self.joint_names:
                raise KeyError(
                    f"Unknown joint name '{joint_name}'. "
                    f"Available joint names: {self.joint_names}"
                )

            joint_id = self.joint_names.index(joint_name)
            q[:, joint_id] = float(value)

        self.robot.set_joint_position_target(q)

    def follow_joint_trajectory(
        self,
        trajectory: torch.Tensor,
        *,
        step_fn,
        steps_per_point: int = 2,
    ) -> None:
        """
        trajectory shape:
            [T, num_joints]
            or [T, num_envs, num_joints]
        """
        if trajectory.ndim == 2:
            trajectory = trajectory[:, None, :]

        for t in range(trajectory.shape[0]):
            q = trajectory[t].to(self.robot.data.joint_pos.device)
            self.robot.set_joint_position_target(q)
            step_fn(steps=steps_per_point)

    def open_gripper(self) -> None:
        if not self.gripper_joint_names:
            print("[WARN] No gripper joints selected; open_gripper() skipped.")
            return

        targets = {
            name: self.gripper_open_value
            for name in self.gripper_joint_names
        }
        self.set_named_joint_targets(targets)

    def close_gripper(self) -> None:
        if not self.gripper_joint_names:
            print("[WARN] No gripper joints selected; close_gripper() skipped.")
            return

        targets = {
            name: self.gripper_close_value
            for name in self.gripper_joint_names
        }
        self.set_named_joint_targets(targets)
