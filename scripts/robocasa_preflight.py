"""Check a real RoboCasa reset and physics steps; this is not task evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="PickPlaceCounterToSink")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    os.environ.setdefault("MUJOCO_GL", "disable")
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    report = {"purpose": "environment_preflight_only", "task": args.task,
              "seed": args.seed, "layout": args.layout, "style": args.style,
              "python": platform.python_version(), "hostname": platform.node(),
              "status": "started", "steps_completed": 0, "rendering": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Use a new output file; refusing to overwrite an earlier check")

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    started = time.monotonic()
    env = None
    save()
    try:
        import numpy as np
        import robocasa
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        report["versions"] = {name: importlib.metadata.version(name)
                              for name in ("mujoco", "numpy", "h5py")}
        report["versions"].update(robocasa=robocasa.__version__, robosuite=robosuite.__version__)
        report["sources"] = {"robocasa": robocasa.__file__, "robosuite": robosuite.__file__}
        report["status"] = "creating_environment"
        save()
        env = robosuite.make(
            env_name=args.task, robots="PandaOmron",
            controller_configs=load_composite_controller_config(robot="PandaOmron"),
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            use_object_obs=True, seed=args.seed, layout_ids=args.layout, style_ids=args.style,
            generative_textures=None, randomize_cameras=False, control_freq=20,
        )
        report["status"] = "resetting"
        save()
        t = time.monotonic()
        obs = env.reset()
        report["reset_seconds"] = time.monotonic() - t
        report["action_dim"] = int(env.action_dim)
        report["observation_shapes"] = {k: list(np.asarray(v).shape) for k, v in obs.items()}
        report["initial_task_success"] = bool(env._check_success())
        report["status"] = "stepping"
        save()
        t = time.monotonic()
        for _ in range(args.steps):
            action = np.zeros(env.action_dim)
            obs, _, _, _ = env.step(action)
            if not np.isfinite(env.sim.data.qpos).all() or not np.isfinite(env.sim.data.qvel).all():
                raise RuntimeError("Non-finite simulator state")
            report["steps_completed"] += 1
        report["step_seconds"] = time.monotonic() - t
        report["physics_steps_per_second"] = args.steps / report["step_seconds"]
        report["final_task_success"] = bool(env._check_success())
        report["status"] = "passed"
        return_code = 0
    except Exception as exc:
        report["failed_stage"] = report["status"]
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        return_code = 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["close_error"] = repr(exc)
                report["status"] = "failed"
                return_code = 1
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        print(json.dumps(report, indent=2), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
