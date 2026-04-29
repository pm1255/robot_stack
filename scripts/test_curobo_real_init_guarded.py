from pathlib import Path
import os
import sys
import traceback

# 必须在 import curobo 之前设置
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")

import torch

def disable_torch_jit_fusers():
    print("[GUARD] disabling TorchScript CUDA fusers")

    funcs = [
        ("_jit_set_nvfuser_enabled", False),
        ("_jit_override_can_fuse_on_gpu", False),
        ("_jit_set_texpr_fuser_enabled", False),
        ("_jit_set_profiling_executor", False),
        ("_jit_set_profiling_mode", False),
    ]

    for name, value in funcs:
        fn = getattr(torch._C, name, None)
        if fn is None:
            print(f"[GUARD] torch._C.{name} not found")
            continue
        try:
            fn(value)
            print(f"[GUARD] {name}({value}) ok")
        except Exception as e:
            print(f"[GUARD] {name} failed: {repr(e)}")

    try:
        torch.jit._state.disable()
        print("[GUARD] torch.jit._state.disable() ok")
    except Exception as e:
        print("[GUARD] torch.jit._state.disable() failed:", repr(e))

disable_torch_jit_fusers()

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
    print("[1] importing cuRobo MotionGen...")
    from curobo.types.base import TensorDeviceType
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    print("[2] cuRobo import success")

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="dummy_far_obstacle",
                pose=[100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
                dims=[0.01, 0.01, 0.01],
            )
        ]
    )

    print("[3] loading MotionGenConfig...")
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

    print("[4] constructing MotionGen...")
    motion_gen = MotionGen(config)

    print("[5] warming up MotionGen...")
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
