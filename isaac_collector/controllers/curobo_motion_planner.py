from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class CuroboPlanResult:
    success: bool
    joint_names: List[str]
    position: Optional[torch.Tensor] = None  # [T, dof]
    message: str = ""


class CuroboMotionPlanner:
    """
    Minimal cuRobo planner wrapper.

    Input:
        - current joint state from IsaacLab robot
        - target EE pose in world/base frame: [x, y, z, qw, qx, qy, qz]

    Output:
        - joint trajectory: [T, dof]

    Important:
        cuRobo needs a robot config yaml / urdf.
        The joint_names in cuRobo config must match IsaacLab robot joint names.
    """

    def __init__(
        self,
        robot_adapter,
        robot_cfg_path: str,
        device: str = "cuda:0",
        interpolation_dt: float = 1.0 / 60.0,
        collision_free: bool = False,
    ):
        self.robot_adapter = robot_adapter
        self.robot_cfg_path = str(Path(robot_cfg_path).expanduser().resolve())
        self.device = device
        self.interpolation_dt = interpolation_dt
        self.collision_free = collision_free

        self._init_curobo()

    def _init_curobo(self):
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.types.file_path import ContentPath
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
            from curobo.geom.types import WorldConfig
        except Exception as e:
            raise ImportError(
                "Failed to import cuRobo. Please install cuRobo inside the isaaclab environment first."
            ) from e

        self.TensorDeviceType = TensorDeviceType
        self.MotionGen = MotionGen
        self.MotionGenConfig = MotionGenConfig
        self.MotionGenPlanConfig = MotionGenPlanConfig
        self.WorldConfig = WorldConfig

        self.tensor_args = TensorDeviceType(device=torch.device(self.device))

        # 第一版先用空世界，不做场景碰撞。
        # 后面可以替换为从 Isaac stage 解析出来的 WorldConfig。
        world_cfg = WorldConfig()

        # 注意：
        # 不同 cuRobo 版本 API 可能略有变化。
        # 如果这里报 ContentPath 或 load_from_robot_config 相关错误，
        # 说明你装的是 cuRobo v0.8 新 API，需要按新版 API 改。
        robot_cfg = {
            "robot_cfg": {
                "kinematics": {
                    "use_usd_kinematics": False,
                }
            }
        }

        # 更推荐直接传 yaml 文件路径给 load_from_robot_config。
        # 但不同版本对参数支持不完全一致，所以这里保留最直接写法。
        try:
            config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg_path,
                world_cfg,
                tensor_args=self.tensor_args,
                interpolation_dt=self.interpolation_dt,
            )
        except TypeError:
            config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg_path,
                world_cfg,
                self.tensor_args,
                interpolation_dt=self.interpolation_dt,
            )

        self.motion_gen = MotionGen(config)

        print("[cuRobo] Warming up MotionGen...")
        self.motion_gen.warmup()
        print("[cuRobo] MotionGen ready.")

        self.plan_config = MotionGenPlanConfig(
            max_attempts=4,
            enable_graph=True,
            enable_graph_attempt=2,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5,
        )

        # 从 cuRobo robot config 里读出来的 joint names。
        # 不同版本字段可能不同，所以做兼容处理。
        self.curobo_joint_names = None
        try:
            self.curobo_joint_names = list(self.motion_gen.kinematics.joint_names)
        except Exception:
            pass

        if self.curobo_joint_names is None:
            print("[WARN] Cannot read cuRobo joint names automatically.")
            print("[WARN] You may need to set self.curobo_joint_names manually.")

    def _get_current_joint_state(self):
        """
        Return cuRobo JointState from IsaacLab robot.

        这里要求：
        cuRobo yaml 的 joint_names 必须能在 IsaacLab robot.joint_names 里找到。
        """
        from curobo.types.robot import JointState

        if self.curobo_joint_names is None:
            raise RuntimeError(
                "curobo_joint_names is None. Please set it according to your A2D cuRobo yaml."
            )

        robot = self.robot_adapter.robot
        isaac_joint_names = list(robot.joint_names)

        joint_ids = []
        missing = []
        for name in self.curobo_joint_names:
            if name not in isaac_joint_names:
                missing.append(name)
            else:
                joint_ids.append(isaac_joint_names.index(name))

        if missing:
            raise RuntimeError(
                "Some cuRobo joints are not found in IsaacLab robot.joint_names:\n"
                f"{missing}\n\n"
                "This means your cuRobo yaml joint names do not match IsaacLab A2D joint names."
            )

        q = robot.data.joint_pos[0, joint_ids].detach().to(self.device).float().unsqueeze(0)

        return JointState.from_position(
            q,
            joint_names=self.curobo_joint_names,
        )

    def plan_to_pose(self, target_pose_w: torch.Tensor) -> CuroboPlanResult:
        """
        target_pose_w:
            shape [7] or [1, 7]
            format [x, y, z, qw, qx, qy, qz]

        Returns:
            CuroboPlanResult with position [T, dof]
        """
        from curobo.types.math import Pose

        if target_pose_w.ndim == 2:
            target_pose_w = target_pose_w[0]

        target_pose_w = target_pose_w.detach().to(self.device).float()

        position = target_pose_w[0:3].view(1, 3)
        quaternion = target_pose_w[3:7].view(1, 4)

        goal_pose = Pose(
            position=position,
            quaternion=quaternion,
        )

        start_state = self._get_current_joint_state()

        result = self.motion_gen.plan_single(
            start_state,
            goal_pose,
            self.plan_config,
        )

        success = bool(result.success.item())

        if not success:
            return CuroboPlanResult(
                success=False,
                joint_names=self.curobo_joint_names,
                position=None,
                message=f"cuRobo planning failed. status={getattr(result, 'status', None)}",
            )

        traj = result.get_interpolated_plan()

        return CuroboPlanResult(
            success=True,
            joint_names=self.curobo_joint_names,
            position=traj.position.detach(),
            message="ok",
        )