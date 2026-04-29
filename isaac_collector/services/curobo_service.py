from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict


def send(obj: Dict[str, Any]):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class CuRoboService:
    def __init__(self, *, mode: str, robot_config: str | None):
        self.mode = mode
        self.robot_config = robot_config

        print(f"[CuRoboService] python: {sys.executable}", file=sys.stderr)
        print(f"[CuRoboService] mode: {mode}", file=sys.stderr)
        print(f"[CuRoboService] robot_config: {robot_config}", file=sys.stderr)

        if self.mode == "real":
            if robot_config is None or not Path(robot_config).exists():
                raise FileNotFoundError(
                    f"Real cuRobo needs robot config, but got: {robot_config}"
                )
            self._load_real_curobo()
        else:
            self.motion_gen = None

    def _load_real_curobo(self):
        """
        Initialize real cuRobo MotionGen.

        Current version:
        - CUDA 13 / cu130 environment.
        - Dummy far obstacle to avoid empty-world collision error.
        - Only initializes MotionGen; real planning is implemented separately.
        """
        import os

        os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")

        import torch

        for name, value in [
            ("_jit_set_nvfuser_enabled", False),
            ("_jit_override_can_fuse_on_gpu", False),
            ("_jit_set_texpr_fuser_enabled", False),
            ("_jit_set_profiling_executor", False),
            ("_jit_set_profiling_mode", False),
        ]:
            fn = getattr(torch._C, name, None)
            if fn is not None:
                try:
                    fn(value)
                except Exception:
                    pass

        try:
            torch.jit._state.disable()
        except Exception:
            pass

        from curobo.types.base import TensorDeviceType
        from curobo.geom.types import WorldConfig, Cuboid
        from curobo.wrap.reacher.motion_gen import (
            MotionGen,
            MotionGenConfig,
            MotionGenPlanConfig,
        )

        self.torch = torch
        self.device = "cuda:0"
        self.tensor_args = TensorDeviceType(device=torch.device(self.device))

        self.world_cfg = WorldConfig(
            cuboid=[
                Cuboid(
                    name="dummy_far_obstacle",
                    pose=[100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.01, 0.01, 0.01],
                )
            ]
        )

        print("[CuRoboService] loading MotionGenConfig...", file=sys.stderr)
        try:
            self.motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_config,
                self.world_cfg,
                tensor_args=self.tensor_args,
                interpolation_dt=1.0 / 60.0,
            )
        except TypeError:
            self.motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_config,
                self.world_cfg,
                self.tensor_args,
                interpolation_dt=1.0 / 60.0,
            )

        print("[CuRoboService] constructing MotionGen...", file=sys.stderr)
        self.motion_gen = MotionGen(self.motion_gen_config)

        print("[CuRoboService] warming up MotionGen...", file=sys.stderr)
        self.motion_gen.warmup()
        print("[CuRoboService] MotionGen ready.", file=sys.stderr)

        self.MotionGenPlanConfig = MotionGenPlanConfig
        self.plan_config = MotionGenPlanConfig(
            max_attempts=4,
            enable_graph=True,
            enable_graph_attempt=2,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.5,
        )

        self.curobo_joint_names = list(self.motion_gen.kinematics.joint_names)
        print(f"[CuRoboService] joint_names: {self.curobo_joint_names}", file=sys.stderr)

    def plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "mock":
            return self._mock_plan(params)
        return self._real_plan(params)

    def _mock_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        调试用。

        真实 cuRobo 应该输出 joint trajectory。
        mock 先把目标 waypoint 原样返回，验证 IsaacLab 主循环。
        """

        waypoints = params["waypoints_world"]

        return {
            "source": "mock_curobo",
            "trajectory_type": "cartesian_waypoints_debug",
            "cartesian_waypoints_world": waypoints,
            "joint_names": [],
            "positions": [],
            "dt": 0.02,
        }

    def _real_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real cuRobo planning.

        Required:
        - robot_state.joint_names
        - robot_state.positions

        Target:
        - target_pose_world: [x, y, z, qw, qx, qy, qz]
          or
        - waypoints_world: list of 4x4 matrices; current version plans to the last waypoint.
        """
        import math

        torch = self.torch

        def _norm3(v):
            return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)

        def _rot_to_quat_wxyz(r):
            # r is 3x3 row-major rotation matrix.
            r00, r01, r02 = r[0]
            r10, r11, r12 = r[1]
            r20, r21, r22 = r[2]
            tr = r00 + r11 + r22

            if tr > 0.0:
                s = math.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * s
                qx = (r21 - r12) / s
                qy = (r02 - r20) / s
                qz = (r10 - r01) / s
            elif r00 > r11 and r00 > r22:
                s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
                qw = (r21 - r12) / s
                qx = 0.25 * s
                qy = (r01 + r10) / s
                qz = (r02 + r20) / s
            elif r11 > r22:
                s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
                qw = (r02 - r20) / s
                qx = (r01 + r10) / s
                qy = 0.25 * s
                qz = (r12 + r21) / s
            else:
                s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
                qw = (r10 - r01) / s
                qx = (r02 + r20) / s
                qy = (r12 + r21) / s
                qz = 0.25 * s

            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            return [qw / n, qx / n, qy / n, qz / n]

        def _matrix4_to_pose7(m):
            # Supports both normal column-translation matrices and USD/Gf-style row translation.
            m = [[float(x) for x in row] for row in m]
            r = [
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ]

            row_t = [m[3][0], m[3][1], m[3][2]]
            col_t = [m[0][3], m[1][3], m[2][3]]

            if _norm3(row_t) >= _norm3(col_t):
                pos = row_t
                # Compatibility for the current mock GraspNet matrices:
                # vertical offsets were stored in m[2][3].
                if abs(m[0][3]) < 1e-8 and abs(m[1][3]) < 1e-8 and abs(m[2][3]) > 1e-8:
                    pos = [pos[0], pos[1], pos[2] + m[2][3]]
            else:
                pos = col_t

            quat = _rot_to_quat_wxyz(r)
            return [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]

        def _extract_target_pose7(params):
            if "target_pose_world" in params:
                pose = params["target_pose_world"]
                if len(pose) != 7:
                    raise ValueError(f"target_pose_world must have length 7, got: {pose}")
                return [float(x) for x in pose]

            waypoints = params.get("waypoints_world", None)
            if not waypoints:
                raise ValueError("Missing target_pose_world or waypoints_world")

            target = waypoints[-1]

            # 4x4 matrix
            if isinstance(target, list) and len(target) == 4 and all(isinstance(row, list) and len(row) == 4 for row in target):
                return _matrix4_to_pose7(target)

            # Already pose7
            if isinstance(target, list) and len(target) == 7:
                return [float(x) for x in target]

            raise ValueError(f"Unsupported target format: {target}")

        robot_state = params.get("robot_state", {})
        joint_names = robot_state.get("joint_names", None)
        positions = robot_state.get("positions", None)

        if not joint_names or positions is None:
            return {
                "success": False,
                "source": "real_curobo",
                "error": "Missing robot_state.joint_names or robot_state.positions",
                "expected_joint_names": self.curobo_joint_names,
            }

        name_to_pos = {name: float(pos) for name, pos in zip(joint_names, positions)}

        missing = [name for name in self.curobo_joint_names if name not in name_to_pos]
        if missing:
            return {
                "success": False,
                "source": "real_curobo",
                "error": "robot_state is missing cuRobo joints",
                "missing": missing,
                "expected_joint_names": self.curobo_joint_names,
                "given_joint_names": list(joint_names),
            }

        q = [name_to_pos[name] for name in self.curobo_joint_names]

        from curobo.types.robot import JointState
        from curobo.types.math import Pose

        start_state = JointState.from_position(
            torch.tensor(q, device=self.device, dtype=torch.float32).view(1, -1),
            joint_names=self.curobo_joint_names,
        )

        pose7 = _extract_target_pose7(params)

        goal_pose = Pose(
            position=torch.tensor(pose7[0:3], device=self.device, dtype=torch.float32).view(1, 3),
            quaternion=torch.tensor(pose7[3:7], device=self.device, dtype=torch.float32).view(1, 4),
        )

        result = self.motion_gen.plan_single(
            start_state,
            goal_pose,
            self.plan_config,
        )

        try:
            success = bool(result.success.item())
        except Exception:
            success = bool(result.success)

        if not success:
            return {
                "success": False,
                "source": "real_curobo",
                "trajectory_type": "joint_trajectory",
                "joint_names": self.curobo_joint_names,
                "positions": [],
                "target_pose_world": pose7,
                "status": str(getattr(result, "status", "")),
            }

        traj = result.get_interpolated_plan()
        pos = traj.position.detach().cpu()

        # Some cuRobo versions return [1, T, dof]; normalize to [T, dof].
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos[0]

        return {
            "success": True,
            "source": "real_curobo",
            "trajectory_type": "joint_trajectory",
            "joint_names": self.curobo_joint_names,
            "positions": pos.tolist(),
            "dt": 1.0 / 60.0,
            "target_pose_world": pose7,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--robot-config", default=None)
    args = parser.parse_args()

    try:
        with redirect_stdout(sys.stderr):
            service = CuRoboService(mode=args.mode, robot_config=args.robot_config)
        send({
            "type": "ready",
            "service": "curobo",
            "python": sys.executable,
            "mode": args.mode,
        })
    except Exception as e:
        send({
            "type": "ready",
            "service": "curobo",
            "error": repr(e),
        })
        raise

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req = json.loads(line)
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "shutdown":
            send({"id": req_id, "success": True, "result": {"shutdown": True}})
            break

        try:
            if method == "plan":
                with redirect_stdout(sys.stderr):
                    result = service.plan(params)
            else:
                raise ValueError(f"Unknown method: {method}")

            send({
                "id": req_id,
                "success": True,
                "result": result,
            })

        except Exception as e:
            send({
                "id": req_id,
                "success": False,
                "error": repr(e),
            })


if __name__ == "__main__":
    main()