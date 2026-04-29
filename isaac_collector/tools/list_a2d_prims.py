from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse
import omni.usd

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
args = parser.parse_args()

ctx = omni.usd.get_context()
ctx.open_stage(args.usd)
stage = ctx.get_stage()

print("===== Root prims =====")
for prim in stage.GetPseudoRoot().GetChildren():
    print(prim.GetPath(), prim.GetTypeName())

print("\n===== Candidate joint/gripper prims =====")
for prim in stage.Traverse():
    name = prim.GetName()
    type_name = prim.GetTypeName()
    path = str(prim.GetPath())

    if (
        "Joint" in name
        or "joint" in name
        or "gripper" in name.lower()
        or "hand" in name.lower()
        or "left_" in name
        or "right_" in name
    ):
        print(f"{path:100s}  type={type_name}")

simulation_app.close()