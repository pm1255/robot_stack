from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse
import re
import omni.usd

parser = argparse.ArgumentParser()
parser.add_argument("--input-usd", required=True)
parser.add_argument("--output-usd", required=True)
args = parser.parse_args()

ctx = omni.usd.get_context()
ctx.open_stage(args.input_usd)
stage = ctx.get_stage()

# 只删夹爪联动/闭环相关 joint，不删 body/head/left_arm/right_arm 主链。
REMOVE_NAME_PATTERNS = [
    r"^left_.*_(0|1|2)_Joint$",
    r"^right_.*_(0|1|2)_Joint$",
    r"^left_.*_Support_Joint$",
    r"^right_.*_Support_Joint$",
    r"^left_.*_RevoluteJoint$",
    r"^right_.*_RevoluteJoint$",
    r"^left_hand_joint1$",
    r"^right_hand_joint1$",
]

def should_remove(name: str, type_name: str) -> bool:
    # 只处理 joint prim，避免误删 mesh/xform。
    if "Joint" not in type_name and "joint" not in type_name.lower() and "Joint" not in name:
        return False
    return any(re.match(p, name) for p in REMOVE_NAME_PATTERNS)

to_remove = []
for prim in stage.Traverse():
    name = prim.GetName()
    type_name = prim.GetTypeName()
    if should_remove(name, type_name):
        to_remove.append(str(prim.GetPath()))

print("===== Removing gripper mechanism joints =====")
for p in to_remove:
    print("[REMOVE]", p)

# 从深到浅删除，避免父子路径顺序影响
for p in sorted(to_remove, key=lambda x: len(x), reverse=True):
    stage.RemovePrim(p)

stage.GetRootLayer().Export(args.output_usd)
print(f"[OK] saved simplified planning USD: {args.output_usd}")

simulation_app.close()