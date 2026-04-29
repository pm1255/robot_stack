#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-usd",
        default="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd",
    )
    parser.add_argument("--root-path", default="/World/A2D")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import omni.usd
        import omni.timeline
        from pxr import Usd, UsdPhysics

        try:
            from isaacsim.core.utils.stage import open_stage
        except Exception:
            from omni.isaac.core.utils.stage import open_stage

        scene_usd = Path(args.scene_usd)
        if not scene_usd.exists():
            raise FileNotFoundError(scene_usd)

        print(f"[LOAD] {scene_usd}", flush=True)
        open_stage(str(scene_usd))

        for _ in range(120):
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()

        root = stage.GetPrimAtPath(args.root_path)
        print(f"[CHECK] root path: {args.root_path}", flush=True)
        print(f"[CHECK] root valid: {root.IsValid() if root else False}", flush=True)

        if root and root.IsValid():
            print(f"[CHECK] root type: {root.GetTypeName()}", flush=True)
            print(f"[CHECK] root applied schemas: {root.GetAppliedSchemas()}", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("[1] Prims under root that look physics/articulation/joint-related", flush=True)
        print("=" * 80, flush=True)

        interesting = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(args.root_path):
                continue

            type_name = str(prim.GetTypeName())
            schemas = list(prim.GetAppliedSchemas())

            text = " ".join([path, type_name, " ".join(schemas)]).lower()
            if (
                "articulation" in text
                or "joint" in text
                or "physics" in text
                or "revolute" in text
                or "prismatic" in text
            ):
                interesting.append((path, type_name, schemas))

        for path, type_name, schemas in interesting[:300]:
            print(f"[PRIM] {path}", flush=True)
            print(f"       type={type_name}", flush=True)
            print(f"       schemas={schemas}", flush=True)

        print(f"[INFO] num interesting prims: {len(interesting)}", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("[2] Prims with UsdPhysics.ArticulationRootAPI", flush=True)
        print("=" * 80, flush=True)

        articulation_roots = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(str(prim.GetPath()))

        if articulation_roots:
            for p in articulation_roots:
                print(f"[ARTICULATION_ROOT_API] {p}", flush=True)
        else:
            print("[ARTICULATION_ROOT_API] none found", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("[3] Try dynamic_control get_articulation on candidates", flush=True)
        print("=" * 80, flush=True)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for _ in range(60):
            simulation_app.update()

        from omni.isaac.dynamic_control import _dynamic_control

        dc = _dynamic_control.acquire_dynamic_control_interface()

        candidates = []
        candidates.append(args.root_path)
        candidates.extend(articulation_roots)

        # Also try some common child prims under A2D.
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(args.root_path):
                schemas = " ".join(prim.GetAppliedSchemas()).lower()
                type_name = str(prim.GetTypeName()).lower()
                if "articulation" in schemas or "articulation" in type_name:
                    candidates.append(path)

        # Deduplicate.
        seen = set()
        candidates = [x for x in candidates if not (x in seen or seen.add(x))]

        for p in candidates:
            try:
                art = dc.get_articulation(p)
                ok = art is not None and not (isinstance(art, int) and art == 0)
                print(f"[DYNAMIC_CONTROL] {p} -> ok={ok}, handle={art}", flush=True)

                if ok:
                    n = dc.get_articulation_dof_count(art)
                    print(f"  dof_count={n}", flush=True)
                    for i in range(n):
                        dof = dc.get_articulation_dof(art, i)
                        print(f"  [{i:02d}] {dc.get_dof_name(dof)}", flush=True)
            except Exception as e:
                print(f"[DYNAMIC_CONTROL] {p} -> ERROR: {e!r}", flush=True)

        print("\n[DONE] articulation inspection finished.", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
