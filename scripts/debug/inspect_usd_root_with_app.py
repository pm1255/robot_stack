import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd-path",
    type=str,
    required=True,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd

stage = Usd.Stage.Open(args.usd_path)

print("\n========== USD ROOT INFO ==========")
print("Opened:", args.usd_path)

default_prim = stage.GetDefaultPrim()
print("DefaultPrim:", default_prim.GetPath() if default_prim else None)

print("\nRoot children:")
for prim in stage.GetPseudoRoot().GetChildren():
    print(" ", prim.GetPath(), prim.GetTypeName())

print("\nTop-level traverse:")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.count("/") <= 2:
        print(" ", path, prim.GetTypeName())

simulation_app.close()
