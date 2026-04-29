from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scalable atomic manipulation runner. "
            "This is the simulator-target-pointcloud baseline: USD target mesh -> point cloud -> GraspNet -> cuRobo -> A2D replay."
        )
    )

    parser.add_argument("--project-root", default="/home/pm/Desktop/Project/robot_stack")
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument(
        "--scene-registry-json",
        default="/home/pm/Desktop/Project/robot_stack/isaac_collector/configs/scenes/mutil_room001.json",
        help="Flexible mapping file for object class -> USD prim path.",
    )

    parser.add_argument("--robot-path", default="/World/A2D")
    parser.add_argument("--ee-path", default="/World/A2D/Link7_r")

    parser.add_argument("--target-class", default="cup")
    parser.add_argument("--target-path", default=None)

    parser.add_argument(
        "--task-json",
        default=None,
        help="Optional JSON action list. If omitted, creates pick(target_class) + place(offset_robot).",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the cup and place down the right 10cm",
        help="Stored as metadata. LLM parsing should produce --task-json in the next version.",
    )
    parser.add_argument(
        "--place-offset-robot",
        nargs=3,
        type=float,
        default=[0.0, 0.10, 0.0],
        help="Default place offset in robot frame. For your old convention, +Y means right if robot right is +Y.",
    )

    parser.add_argument(
        "--observation-mode",
        choices=["sim_object_pointcloud"],
        default="sim_object_pointcloud",
    )
    parser.add_argument("--target-cloud-npoints", type=int, default=20000)
    parser.add_argument("--target-cloud-seed", type=int, default=0)

    parser.add_argument("--graspnet-python", default="/home/pm/miniconda3/envs/graspnet_env/bin/python")
    parser.add_argument("--curobo-python", default="/home/pm/miniconda3/envs/curobo_env_cu13/bin/python")
    parser.add_argument("--grasp-mode", choices=["mock", "real"], default="real")
    parser.add_argument("--curobo-mode", choices=["mock", "real"], default="real")
    parser.add_argument("--graspnet-checkpoint", default=None)
    parser.add_argument("--curobo-robot-config", default=None)
    parser.add_argument("--graspnet-target-key", default="lift_pose_world")

    parser.add_argument("--execution-mode", choices=["a2d_replay", "debug_object"], default="a2d_replay")
    parser.add_argument("--attach-target-during-place", action="store_true")
    parser.add_argument("--steps-per-position", type=int, default=1)
    parser.add_argument("--speed-stride", type=int, default=1)
    parser.add_argument("--trajectory-save-stride", type=int, default=5)
    parser.add_argument("--save-executed-trajectory", action="store_true")

    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/robot_pipeline/atomic_manipulation")

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument(
        "--no-reset-joints",
        action="store_true",
        help="Do not reset A2D joints to retract_config before planning/replay.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Run scene/query/GraspNet/cuRobo planning but do not execute A2D replay.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Open the USD scene in GUI and do not start services, reset joints, plan, or replay.",
    )

    parser.add_argument(
        "--target-not-found-exit-code",
        type=int,
        default=20,
        help="Process exit code used when the requested target is not registered as operable.",
    )


    # Important: default is NO root teleport. This fixes your current issue.
    parser.add_argument(
        "--teleport-robot-for-debug",
        action="store_true",
        help="Opt-in debug only. Default keeps the robot root pose stored in the USD.",
    )
    parser.add_argument(
        "--reset-robot-root-each-episode",
        action="store_true",
        help="Opt-in. Reset robot root to the initial USD root pose at episode start. Default false.",
    )
    parser.add_argument(
        "--reset-target-each-episode",
        action="store_true",
        help="Reset target object pose to initial pose before each episode. Useful for repeated data collection.",
    )

    parser.add_argument("--pickup-target-z", type=float, default=None)
    parser.add_argument("--place-target-z", type=float, default=None)

    return parser.parse_args()


def _ensure_project_on_path(project_root: Path):
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _service_args(mode: str, extra: list[str]) -> list[str]:
    return ["--mode", mode] + extra


def _call_graspnet(grasp_service, *, target_path: str, target_pose_world, observation_npz: Path) -> dict:
    return grasp_service.call(
        "predict",
        {
            "object_path": target_path,
            "target_path": target_path,
            "object_pose_world": np.asarray(target_pose_world, dtype=float).tolist(),
            "observation_npz": str(observation_npz),
            "observation_source": "sim_object_pointcloud",
        },
    )


def _plan_curobo(curobo_service, *, task: str, robot_state: dict, target_robot: np.ndarray, attached_object: str | None = None) -> dict:
    request = {
        "task": task,
        "robot_state": robot_state,
        "waypoints_world": [np.asarray(target_robot, dtype=float).tolist()],
    }
    if attached_object:
        request["attached_object"] = attached_object
        request["world_collision"] = {}
    return curobo_service.call("plan", request)



def _candidate_to_grasp_result(base_grasp_result: dict, candidate: dict) -> dict:
    """
    Convert one GraspNet candidate into the old top-level grasp_result format
    expected by make_targets_from_graspnet_result(...).

    This keeps backward compatibility with the old runner code.
    """
    out = dict(base_grasp_result)
    for k in [
        "score",
        "width",
        "depth",
        "translation_world",
        "rotation_world",
        "pregrasp_pose_world",
        "grasp_pose_world",
        "lift_pose_world",
    ]:
        if k in candidate:
            out[k] = candidate[k]
    out["selected_candidate"] = candidate
    return out


def _select_feasible_grasp_with_curobo(
    *,
    grasp_result: dict,
    curobo_service,
    robot_state: dict,
    robot_world: np.ndarray,
    place_offset_robot,
    args,
    make_targets_from_graspnet_result,
    summarize_plan,
    place_action,
):
    """
    Try GraspNet candidates one by one and use cuRobo as feasibility filter.

    Returns:
      selected_index, selected_grasp_result, pickup_target_robot,
      place_target_robot, pickup_plan, candidate_logs
    """
    raw_candidates = grasp_result.get("candidates", None)

    if raw_candidates:
        candidates = list(raw_candidates)
        print(f"[FEASIBILITY] GraspNet candidates: {len(candidates)}", flush=True)
    else:
        candidates = [grasp_result]
        print("[FEASIBILITY] no candidates field; fallback to top-1 grasp_result", flush=True)

    max_try = int(__import__("os").environ.get("MAX_GRASP_CANDIDATES", "20"))
    candidates = candidates[:max_try]

    candidate_logs = []

    for i, cand in enumerate(candidates):
        cand_result = _candidate_to_grasp_result(grasp_result, cand)

        try:
            pickup_target_robot, place_target_robot = make_targets_from_graspnet_result(
                grasp_result=cand_result,
                robot_world=robot_world,
                preferred_key=args.graspnet_target_key,
                place_offset_robot=place_offset_robot,
                pickup_target_z=args.pickup_target_z,
                place_target_z=args.place_target_z,
            )

            # Explicit place target can override relative offset.
            if place_action.target_robot is not None:
                place_target_robot = np.eye(4, dtype=float)
                place_target_robot[3, 0:3] = np.asarray(place_action.target_robot, dtype=float)

            # DEBUG mode remains available, but it should be off for real GraspNet testing.
            if bool(int(__import__("os").environ.get("DEBUG_FORCE_REACHABLE_PICK", "0"))):
                print("[DEBUG_FORCE] overriding pickup/place targets before cuRobo plan", flush=True)
                pickup_target_robot = np.eye(4, dtype=float)
                pickup_target_robot[3, 0:3] = [0.45, 0.00, 0.80]
                place_target_robot = np.eye(4, dtype=float)
                place_target_robot[3, 0:3] = [0.45, 0.10, 0.85]

            print(
                f"[FEASIBILITY] try candidate {i}: "
                f"score={cand.get('score')} width={cand.get('width')} "
                f"pickup_xyz={pickup_target_robot[3, 0:3].tolist()}",
                flush=True,
            )

            pickup_plan = _plan_curobo(
                curobo_service,
                task="pickup",
                robot_state=robot_state,
                target_robot=pickup_target_robot,
            )

            plan_summary = summarize_plan(pickup_plan)
            print(f"[FEASIBILITY] candidate {i} pickup plan: {plan_summary}", flush=True)

            candidate_logs.append(
                {
                    "candidate_index": i,
                    "candidate": cand,
                    "pickup_target_robot": pickup_target_robot.tolist(),
                    "place_target_robot": place_target_robot.tolist(),
                    "pickup_plan_summary": plan_summary,
                    "success": bool(pickup_plan.get("success", False)),
                }
            )

            if pickup_plan.get("success", False):
                print(f"[FEASIBILITY] selected candidate {i}", flush=True)
                return (
                    i,
                    cand_result,
                    pickup_target_robot,
                    place_target_robot,
                    pickup_plan,
                    candidate_logs,
                )

        except Exception as e:
            print(f"[FEASIBILITY][WARN] candidate {i} failed before/inside cuRobo: {repr(e)}", flush=True)
            candidate_logs.append(
                {
                    "candidate_index": i,
                    "candidate": cand,
                    "success": False,
                    "error": repr(e),
                }
            )

    return None, None, None, None, None, candidate_logs


def _select_task_plan(args):
    from isaac_collector.runtime.action_specs import (
        load_task_plan,
        default_pick_place_plan,
        validate_pick_place_plan,
    )

    if args.task_json:
        plan = load_task_plan(args.task_json)
    else:
        plan = default_pick_place_plan(
            target_class=args.target_class,
            target_path=args.target_path,
            place_offset_robot=args.place_offset_robot,
            instruction=args.instruction,
        )
    validate_pick_place_plan(plan)
    return plan


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    _ensure_project_on_path(project_root)

    # Existing stable helpers from your current monolithic script.
    # We reuse them first, then gradually move them into runtime/ modules later.
    from isaac_collector.run_repeated_pick_place import (
        wait_frames,
        load_retract_robot_state,
        init_a2d_replay_controller,
        reset_a2d_joints_to_robot_state,
        make_targets_from_graspnet_result,
        execute_curobo_plan_on_a2d,
        compute_cup_to_ee_translation_offset,
        execute_cartesian_debug_on_object,
        move_a2d_close_to_cup_for_bringup,
        make_place_pose,
    )

    from isaac_collector.runtime.operable_scene_registry import query_operable_object, load_operable_objects_from_registry
    from isaac_collector.runtime.sim_target_pointcloud import save_sim_target_cloud_npz
    from isaac_collector.runtime.episode_logging import (
        save_json,
        matrix_to_list,
        translation_row,
        summarize_plan,
    )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    grasp_service = None
    curobo_service = None

    try:
        from isaac_collector.runtime.load_scene import (
            open_usd_stage,
            get_prim_world_matrix,
            set_prim_world_matrix,
        )
        from isaac_collector.ipc.jsonl_service import PersistentJsonService

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[1] open USD scene", flush=True)
        stage = open_usd_stage(args.scene_usd, simulation_app, wait=120)

        if args.inspect_only:
            print(
                "[INSPECT_ONLY] scene opened. No robot teleport, no joint reset, "
                "no GraspNet, no cuRobo, no replay.",
                flush=True,
            )
            if args.keep_open and not args.headless:
                print("[INSPECT_ONLY] keep-open enabled. Close Isaac window to exit.", flush=True)
                while simulation_app.is_running():
                    simulation_app.update()
            return 0

        if args.teleport_robot_for_debug:
            print("[ROBOT][DEBUG] teleport enabled by --teleport-robot-for-debug", flush=True)
            move_a2d_close_to_cup_for_bringup(
                stage=stage,
                simulation_app=simulation_app,
                robot_path=args.robot_path,
            )
        else:
            print("[ROBOT] root teleport disabled; using robot pose stored in USD", flush=True)

        print("[2] resolve task plan", flush=True)
        task_plan = _select_task_plan(args)
        save_json(output_dir / "task_plan.json", task_plan.to_dict())
        print("[TASK]", task_plan.to_dict(), flush=True)

        pick_action = task_plan.actions[0]
        place_action = task_plan.actions[1]

        print("[3] load registered operable objects from Isaac scene registry", flush=True)
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

            # Do not start GraspNet or cuRobo when target is absent/unregistered.
            return int(getattr(args, "target_not_found_exit_code", 20))

        selected_obj = target_query.selected
        target_path = selected_obj.prim_path

        # Keep downstream variable name compatible.
        target = selected_obj
        print("[TARGET] resolved from operable scene registry:", selected_obj.to_dict(), flush=True)
        save_json(output_dir / "resolved_target.json", selected_obj.to_dict())

        print("[4] start persistent GraspNet service", flush=True)
        grasp_service = PersistentJsonService(
            name="graspnet",
            python_exe=args.graspnet_python,
            worker_file=str(project_root / "isaac_collector/services/graspnet_service.py"),
            project_root=str(project_root),
            args=_service_args(
                args.grasp_mode,
                [] if args.graspnet_checkpoint is None else ["--checkpoint", args.graspnet_checkpoint],
            ),
        )

        print("[5] start persistent cuRobo service", flush=True)
        curobo_service = PersistentJsonService(
            name="curobo",
            python_exe=args.curobo_python,
            worker_file=str(project_root / "isaac_collector/services/curobo_service.py"),
            project_root=str(project_root),
            args=_service_args(
                args.curobo_mode,
                [] if args.curobo_robot_config is None else ["--robot-config", args.curobo_robot_config],
            ),
        )

        robot_state = load_retract_robot_state(args.curobo_robot_config)
        print("[STATE] robot_state joint_names:", robot_state.get("joint_names", []), flush=True)

        replay_controller = None
        if args.execution_mode == "a2d_replay":
            print("[6] init A2D replay controller", flush=True)
            replay_controller = init_a2d_replay_controller(
                stage=stage,
                simulation_app=simulation_app,
                robot_path=args.robot_path,
                curobo_joint_names=robot_state.get("joint_names", []),
            )
            print("[6] A2D replay controller ready", flush=True)

        initial_robot_world = np.asarray(get_prim_world_matrix(stage, args.robot_path), dtype=float).copy()
        initial_target_world = np.asarray(get_prim_world_matrix(stage, target_path), dtype=float).copy()
        robot_world = initial_robot_world.copy()

        save_json(
            output_dir / "initial_state.json",
            {
                "robot_path": args.robot_path,
                "robot_world": matrix_to_list(initial_robot_world),
                "target_path": target_path,
                "target_world": matrix_to_list(initial_target_world),
                "target_xyz": translation_row(initial_target_world),
            },
        )

        print("[7] execute episodes", flush=True)

        for ep in range(args.num_episodes):
            print("\n" + "=" * 80, flush=True)
            print(f"[EP {ep + 1}/{args.num_episodes}]", flush=True)
            print("=" * 80, flush=True)

            ep_dir = output_dir
            if args.reset_robot_root_each_episode:
                print("[RESET] reset robot root to initial USD pose", flush=True)
                set_prim_world_matrix(stage, args.robot_path, initial_robot_world)

            if args.reset_target_each_episode:
                print("[RESET] reset target object to initial USD pose", flush=True)
                set_prim_world_matrix(stage, target_path, initial_target_world)

            wait_frames(simulation_app, 10)

            if args.execution_mode == "a2d_replay" and not args.no_reset_joints:
                reset_a2d_joints_to_robot_state(
                    simulation_app=simulation_app,
                    replay_controller=replay_controller,
                    robot_state=robot_state,
                    hold_frames=60,
                )
            elif args.no_reset_joints:
                print("[NO_RESET_JOINTS] skip reset_a2d_joints_to_robot_state", flush=True)

            wait_frames(simulation_app, 20)

            target_pose_world = np.asarray(get_prim_world_matrix(stage, target_path), dtype=float).copy()
            robot_world = np.asarray(get_prim_world_matrix(stage, args.robot_path), dtype=float).copy()

            obs_path = ep_dir / f"ep_{ep:04d}_target_cloud_world.npz"
            obs_meta = save_sim_target_cloud_npz(
                stage,
                target_path,
                obs_path,
                target_class=target.class_name,
                n_points=args.target_cloud_npoints,
                seed=args.target_cloud_seed + ep,
                extra_meta={
                    "episode": ep,
                    "target_resolution": target.to_dict(),
                    "scene_usd": str(args.scene_usd),
                    "observation_mode": args.observation_mode,
                },
            )
            save_json(ep_dir / f"ep_{ep:04d}_target_cloud_meta.json", obs_meta)
            print("[OBS] saved simulator target point cloud:", obs_path, flush=True)
            print("[OBS] bbox_center:", obs_meta.get("bbox_center"), flush=True)
            print("[OBS] bbox_extent:", obs_meta.get("bbox_extent"), flush=True)

            print("[8] GraspNet predict", flush=True)
            grasp_result = _call_graspnet(
                grasp_service,
                target_path=target_path,
                target_pose_world=target_pose_world,
                observation_npz=obs_path,
            )
            save_json(ep_dir / f"ep_{ep:04d}_grasp.json", grasp_result)
            print("[GRASP] source:", grasp_result.get("source"), flush=True)
            print("[GRASP] keys:", list(grasp_result.keys()), flush=True)

            place_offset_robot = (
                place_action.offset_robot
                if place_action.offset_robot is not None
                else args.place_offset_robot
            )

            print("[9] cuRobo feasibility filter over GraspNet candidates", flush=True)
            (
                selected_candidate_index,
                selected_grasp_result,
                pickup_target_robot,
                place_target_robot,
                pickup_plan,
                candidate_logs,
            ) = _select_feasible_grasp_with_curobo(
                grasp_result=grasp_result,
                curobo_service=curobo_service,
                robot_state=robot_state,
                robot_world=robot_world,
                place_offset_robot=place_offset_robot,
                args=args,
                make_targets_from_graspnet_result=make_targets_from_graspnet_result,
                summarize_plan=summarize_plan,
                place_action=place_action,
            )

            save_json(ep_dir / f"ep_{ep:04d}_grasp_candidate_plans.json", candidate_logs)

            if pickup_plan is None or not pickup_plan.get("success", False):
                save_json(
                    ep_dir / f"ep_{ep:04d}_planner_result.json",
                    {
                        "success": False,
                        "status": "no_feasible_grasp",
                        "num_candidates_tried": len(candidate_logs),
                        "candidate_logs": candidate_logs,
                    },
                )
                raise RuntimeError("Pickup plan failed: no feasible GraspNet candidate found by cuRobo")

            # From here on, use the selected candidate as the episode grasp.
            grasp_result = selected_grasp_result

            save_json(
                ep_dir / f"ep_{ep:04d}_selected_grasp.json",
                {
                    "selected_candidate_index": selected_candidate_index,
                    "selected_grasp_result": selected_grasp_result,
                    "pickup_target_robot": matrix_to_list(pickup_target_robot),
                    "place_target_robot": matrix_to_list(place_target_robot),
                    "place_offset_robot": place_offset_robot,
                },
            )

            save_json(
                ep_dir / f"ep_{ep:04d}_targets.json",
                {
                    "selected_candidate_index": selected_candidate_index,
                    "pickup_target_robot": matrix_to_list(pickup_target_robot),
                    "place_target_robot": matrix_to_list(place_target_robot),
                    "place_offset_robot": place_offset_robot,
                },
            )

            save_json(ep_dir / f"ep_{ep:04d}_pickup_plan.json", pickup_plan)
            print("[PLAN] pickup:", summarize_plan(pickup_plan), flush=True)

            pickup_executed = {}
            if args.plan_only:
                print("[PLAN_ONLY] skip A2D pickup replay", flush=True)
            elif args.execution_mode == "a2d_replay":
                pickup_executed = execute_curobo_plan_on_a2d(
                    simulation_app=simulation_app,
                    replay_controller=replay_controller,
                    plan=pickup_plan,
                    name="pickup",
                    steps_per_position=args.steps_per_position,
                    speed_stride=args.speed_stride,
                    stage=stage if args.save_executed_trajectory else None,
                    cup_path=target_path if args.save_executed_trajectory else None,
                    ee_path=args.ee_path if args.save_executed_trajectory else None,
                    save_stride=args.trajectory_save_stride,
                )
            elif args.execution_mode == "debug_object":
                lift_world = np.asarray(grasp_result.get("lift_pose_world", target_pose_world), dtype=float)
                execute_cartesian_debug_on_object(
                    simulation_app=simulation_app,
                    stage=stage,
                    cup_path=target_path,
                    target_pose_world=lift_world,
                    frames=80,
                )

            wait_frames(simulation_app, 20)

            target_offset = None
            if args.execution_mode == "a2d_replay" and args.attach_target_during_place:
                target_offset = compute_cup_to_ee_translation_offset(
                    stage,
                    replay_controller,
                    cup_path=target_path,
                    ee_path=args.ee_path,
                )
                print("[ATTACH] target attached to EE by translation following", flush=True)

            print("[10] cuRobo plan place", flush=True)
            place_plan = _plan_curobo(
                curobo_service,
                task="putdown",
                robot_state=robot_state,
                target_robot=place_target_robot,
                attached_object=target_path,
            )
            save_json(ep_dir / f"ep_{ep:04d}_place_plan.json", place_plan)
            print("[PLAN] place:", summarize_plan(place_plan), flush=True)

            if not place_plan.get("success", False):
                raise RuntimeError(f"Place plan failed: {place_plan.get('status')}")

            place_executed = {}
            if args.plan_only:
                print("[PLAN_ONLY] skip A2D place replay", flush=True)
            elif args.execution_mode == "a2d_replay":
                place_executed = execute_curobo_plan_on_a2d(
                    simulation_app=simulation_app,
                    replay_controller=replay_controller,
                    plan=place_plan,
                    name="place",
                    steps_per_position=args.steps_per_position,
                    speed_stride=args.speed_stride,
                    stage=stage if (args.attach_target_during_place or args.save_executed_trajectory) else None,
                    cup_path=target_path if (args.attach_target_during_place or args.save_executed_trajectory) else None,
                    ee_path=args.ee_path if (args.attach_target_during_place or args.save_executed_trajectory) else None,
                    cup_offset=target_offset if args.attach_target_during_place else None,
                    save_stride=args.trajectory_save_stride,
                )
            elif args.execution_mode == "debug_object":
                current = np.asarray(get_prim_world_matrix(stage, target_path), dtype=float)
                place_world = make_place_pose(current, axis="y", distance=float(place_offset_robot[1]))
                execute_cartesian_debug_on_object(
                    simulation_app=simulation_app,
                    stage=stage,
                    cup_path=target_path,
                    target_pose_world=place_world,
                    frames=80,
                )

            wait_frames(simulation_app, 30)

            final_target_world = np.asarray(get_prim_world_matrix(stage, target_path), dtype=float).copy()
            episode_result = {
                "episode": ep,
                "success": bool(pickup_plan.get("success", False) and place_plan.get("success", False)),
                "target": target.to_dict(),
                "observation_npz": str(obs_path),
                "initial_target_pose": matrix_to_list(target_pose_world),
                "initial_target_xyz": translation_row(target_pose_world),
                "final_target_pose": matrix_to_list(final_target_world),
                "final_target_xyz": translation_row(final_target_world),
                "pickup_plan": summarize_plan(pickup_plan),
                "place_plan": summarize_plan(place_plan),
                "pickup_target_robot": matrix_to_list(pickup_target_robot),
                "place_target_robot": matrix_to_list(place_target_robot),
                "trajectory_files": {},
            }

            if args.save_executed_trajectory and args.execution_mode == "a2d_replay":
                pickup_traj_path = ep_dir / f"ep_{ep:04d}_pickup_executed_trajectory.json"
                place_traj_path = ep_dir / f"ep_{ep:04d}_place_executed_trajectory.json"
                save_json(pickup_traj_path, pickup_executed)
                save_json(place_traj_path, place_executed)
                episode_result["trajectory_files"] = {
                    "pickup_executed": str(pickup_traj_path),
                    "place_executed": str(place_traj_path),
                }

            save_json(ep_dir / f"ep_{ep:04d}_episode_result.json", episode_result)
            print("[EP DONE]", episode_result, flush=True)

        print("[DONE] atomic manipulation runner finished", flush=True)

        if args.keep_open and not args.headless:
            print("[INFO] keep-open enabled", flush=True)
            while simulation_app.is_running():
                simulation_app.update()

    finally:
        if grasp_service is not None:
            grasp_service.close()
        if curobo_service is not None:
            curobo_service.close()
        simulation_app.close()


if __name__ == "__main__":
    ret = main()
    if isinstance(ret, int):
        raise SystemExit(ret)
