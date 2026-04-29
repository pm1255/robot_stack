from pathlib import Path
import re

p = Path("isaac_collector/run_atomic_manipulation.py")
if not p.exists():
    raise FileNotFoundError(p)

s = p.read_text(encoding="utf-8")
backup = p.with_suffix(".py.bak_operable_registry")
backup.write_text(s, encoding="utf-8")
print(f"[BACKUP] {backup}")

s = s.replace(
    "    from isaac_collector.runtime.target_registry import resolve_target\n",
    "    from isaac_collector.runtime.operable_scene_registry import query_operable_object, load_operable_objects_from_registry\n",
)

if "operable_scene_registry import query_operable_object" not in s:
    marker = "    from isaac_collector.runtime.sim_target_pointcloud import save_sim_target_cloud_npz\n"
    if marker not in s:
        raise RuntimeError("Cannot find import insertion marker for sim_target_pointcloud.")
    s = s.replace(
        marker,
        "    from isaac_collector.runtime.operable_scene_registry import query_operable_object, load_operable_objects_from_registry\n" + marker,
    )

if "--target-not-found-exit-code" not in s:
    marker = '    parser.add_argument("--keep-open", action="store_true")\n'
    if marker not in s:
        raise RuntimeError("Cannot find parser keep-open marker.")
    added = '''
    parser.add_argument(
        "--target-not-found-exit-code",
        type=int,
        default=20,
        help="Process exit code used when the requested target is not registered as operable.",
    )

'''
    s = s.replace(marker, marker + added)

pattern = re.compile(
    r'''        print\("\[3\] resolve target object", flush=True\)\n.*?        print\("\[4\] start persistent GraspNet service", flush=True\)\n''',
    re.DOTALL,
)

replacement = '''        print("[3] load registered operable objects from Isaac scene registry", flush=True)
        operable_objects = load_operable_objects_from_registry(
            stage=stage,
            registry_json=args.scene_registry_json,
            validate_in_stage=True,
            require_mesh=True,
        )
        print("[SCENE_REGISTRY] num valid operable objects:", len(operable_objects), flush=True)
        for obj in operable_objects:
            print(
                f"[SCENE_REGISTRY] id={obj.object_id} class={obj.class_name} "
                f"path={obj.prim_path} affordances={obj.affordances}",
                flush=True,
            )

        requested_target_class = pick_action.target_class or args.target_class
        requested_target_path = pick_action.target_path or args.target_path

        target_query = query_operable_object(
            stage=stage,
            registry_json=args.scene_registry_json,
            target_class=requested_target_class,
            explicit_path=requested_target_path,
            required_affordance="pick",
        )
        save_json(output_dir / "operable_scene_objects.json", target_query.to_dict())

        if not target_query.success or target_query.selected is None:
            failure = {
                "success": False,
                "status": "target_not_found",
                "stage": "query_operable_scene_registry",
                "missing_target_class": requested_target_class,
                "requested_target_path": requested_target_path,
                "scene_usd": str(args.scene_usd),
                "scene_registry_json": str(args.scene_registry_json),
                "available_operable_objects": [
                    x.to_dict() for x in target_query.all_operable_objects
                ],
                "message": target_query.message,
            }
            print("[TARGET_NOT_FOUND]", failure, flush=True)
            save_json(output_dir / "planner_result.json", failure)
            save_json(output_dir / "resolved_target_failed.json", failure)

            # Critical: do not start GraspNet or cuRobo when target is not registered as operable.
            return int(getattr(args, "target_not_found_exit_code", 20))

        selected_obj = target_query.selected
        target_path = selected_obj.prim_path

        # Keep the downstream variable name 'target' compatible with the previous runner.
        target = selected_obj
        print("[TARGET] resolved from operable scene registry:", selected_obj.to_dict(), flush=True)
        save_json(output_dir / "resolved_target.json", selected_obj.to_dict())

        print("[4] start persistent GraspNet service", flush=True)
'''

s2, n = pattern.subn(replacement, s)
if n != 1:
    raise RuntimeError(
        f"Expected to replace exactly one target resolution block, replaced {n}. "
        "Please inspect run_atomic_manipulation.py around '[3] resolve target object'."
    )
s = s2

if 'ret = main()' not in s:
    s = s.replace(
        'if __name__ == "__main__":\n    main()\n',
        'if __name__ == "__main__":\n    ret = main()\n    if isinstance(ret, int):\n        raise SystemExit(ret)\n',
    )

p.write_text(s, encoding="utf-8")
print(f"[PATCHED] {p}")
