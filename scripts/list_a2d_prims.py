from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    from isaac_collector.runtime.load_scene import open_usd_stage

    scene_usd = "/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd"
    stage = open_usd_stage(scene_usd, simulation_app, wait=120)

    print("[A2D-related prims]")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        if "a2d" in path.lower() or "agibot" in path.lower() or "robot" in name:
            print(path, prim.GetTypeName())

finally:
    simulation_app.close()
