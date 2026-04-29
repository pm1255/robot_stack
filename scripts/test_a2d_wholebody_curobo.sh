#!/usr/bin/env bash
set -e

CUROBO_PY="/home/pm/miniconda3/envs/curobo_env/bin/python"
A2D_CUROBO_YML="/home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody.yml"

"${CUROBO_PY}" - <<PY
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import load_yaml

path = "${A2D_CUROBO_YML}"
cfg = load_yaml(path)["robot_cfg"]

tensor_args = TensorDeviceType()
robot_cfg = RobotConfig.from_dict(cfg, tensor_args)
model = CudaRobotModel(robot_cfg.kinematics)

joint_names = cfg["kinematics"]["cspace"]["joint_names"]
q0 = cfg["kinematics"]["cspace"]["retract_config"]

print("[OK] loaded config:", path)
print("[OK] dof:", model.get_dof())
print("[OK] joint_names:", joint_names)
print("[OK] retract_config length:", len(q0))

q = torch.tensor([q0], device=tensor_args.device, dtype=torch.float32)
state = model.get_state(q)

print("[OK] ee_position:", state.ee_position)
print("[OK] ee_quaternion:", state.ee_quaternion)
PY