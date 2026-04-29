from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

try:
    from isaac_collector.runtime.load_scene import open_usd_stage, get_prim_world_matrix

    scene_usd = "/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd"

    print("[DEBUG] before open_usd_stage", flush=True)
    stage = open_usd_stage(scene_usd, simulation_app, wait=120)
    print("[DEBUG] after open_usd_stage", flush=True)

    paths = [
        "/World/A2D",
        "/World/office1/Room_seed123_idx000/furniture/tea_table/cup",
    ]

    for path in paths:
        print("\n===", path, "===", flush=True)
        prim = stage.GetPrimAtPath(path)
        print("valid:", prim.IsValid(), flush=True)

        if not prim.IsValid():
            continue

        m = get_prim_world_matrix(stage, path)
        print(m, flush=True)

        print(
            "translation row:",
            [float(m[3][0]), float(m[3][1]), float(m[3][2])],
            flush=True,
        )
        print(
            "translation col:",
            [float(m[0][3]), float(m[1][3]), float(m[2][3])],
            flush=True,
        )

finally:
    simulation_app.close()