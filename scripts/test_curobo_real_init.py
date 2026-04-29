from pathlib import Path
import sys
import traceback
import torch

robot_cfg_path = Path("/home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody.yml")

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

print("robot cfg exists:", robot_cfg_path.exists())
print("robot cfg:", robot_cfg_path)

try:
    from curobo.types.base import TensorDeviceType
    from curobo.geom.types import WorldConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    world_cfg = WorldConfig()

    print("[1] loading MotionGenConfig...")
    try:
        config = MotionGenConfig.load_from_robot_config(
            str(robot_cfg_path),
            world_cfg,
            tensor_args=tensor_args,
            interpolation_dt=1.0 / 60.0,
        )
    except TypeError:
        config = MotionGenConfig.load_from_robot_config(
            str(robot_cfg_path),
            world_cfg,
            tensor_args,
            interpolation_dt=1.0 / 60.0,
        )

    print("[2] constructing MotionGen...")
    motion_gen = MotionGen(config)

    print("[3] warming up...")
    motion_gen.warmup()

    print("[OK] cuRobo MotionGen init + warmup success")

    try:
        print("joint names:", motion_gen.kinematics.joint_names)
    except Exception as e:
        print("[WARN] cannot read joint names:", repr(e))

except Exception:
    print("[FAIL] cuRobo init failed")
    traceback.print_exc()
    raise
