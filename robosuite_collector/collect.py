"""Run seeded Lift episodes and save state/action HDF5 plus auditable results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def make_env(seed, max_steps):
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    cfg = load_composite_controller_config(robot='Panda')
    cfg['body_parts']['right']['input_ref_frame'] = 'world'
    config = dict(robots='Panda', controller_configs=cfg, has_renderer=False,
                  has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
                  initialization_noise=None, seed=seed, horizon=max_steps + 1,
                  ignore_done=False, control_freq=20)
    env = robosuite.make('Lift', **config)
    return env, config


def save_trajectory(path, controller, xml, config, result):
    import numpy as np
    import h5py
    if len(controller.states) != len(controller.actions) + 1:
        raise ValueError('Each action requires a pre-state and a post-state')
    temporary = path.with_suffix('.hdf5.tmp')
    with h5py.File(temporary, 'x') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({'env_name': 'Lift', 'type': 1, 'env_kwargs': config})
        data.attrs['total'] = len(controller.actions)
        data.attrs['source_type'] = 'scripted_state_feedback'
        demo = data.create_group('demo_0')
        demo.attrs['model_file'] = xml
        demo.attrs['num_samples'] = len(controller.actions)
        demo.attrs['ep_meta'] = json.dumps({'seed': result['seed'], 'task': 'Lift'})
        demo.attrs['success'] = result['success']
        demo.attrs['collector_result'] = json.dumps(result)
        demo.create_dataset('states', data=np.asarray(controller.states[:-1]), compression='gzip')
        demo.create_dataset('actions', data=np.asarray(controller.actions).reshape(-1, 7), compression='gzip')
        demo.create_dataset('final_state', data=controller.states[-1])
        if controller.samples:
            for name in ('eef_xyz', 'cube_xyz', 'grasped', 'environment_success', 'verified_success', 'attempt'):
                demo.create_dataset('trace/' + name, data=np.asarray([r[name] for r in controller.samples]))
            demo.create_dataset('trace/phase', data=[r['phase'] for r in controller.samples],
                                dtype=h5py.string_dtype())
        demo.attrs['function_events'] = json.dumps(controller.events)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-attempts', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--first-grasp-offset', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--label', default='lift_collection')
    args = parser.parse_args()
    if min(args.episodes, args.max_attempts, args.max_steps) < 1:
        parser.error('Episodes, attempts and step budget must be positive')
    os.environ.setdefault('MUJOCO_GL', 'disable')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    import numpy as np
    import mujoco
    import robosuite
    from .lift import LiftSkills
    if not np.isfinite(args.first_grasp_offset).all():
        parser.error('Grasp offsets must be finite')
    args.output.mkdir(parents=True, exist_ok=False)
    run_config = {**vars(args), 'output': str(args.output), 'task': 'Lift',
                  'backend': 'robosuite', 'versions': {'robosuite': robosuite.__version__,
                  'mujoco': mujoco.__version__, 'numpy': np.__version__},
                  'created_at': datetime.now(timezone.utc).isoformat(),
                  'observation_source': 'simulator_ground_truth',
                  'success_criterion': 'Lift oracle + two-sided grasp + >=10cm height gain, all held for 10 control steps',
                  'perturbation_type': 'first_attempt_cartesian_grasp_target_offset',
                  'state_restore_during_recovery': False, 'rendering': False}
    write_json(args.output / 'run_config.json', run_config)
    run_started = time.monotonic()
    results = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        started = time.monotonic()
        env, controller = None, None
        path = args.output / f'ep_{seed:04d}_episode_result.json'
        write_json(path, {'seed': seed, 'status': 'incomplete', 'success': False, 'validation_version': 1})
        result = {'seed': seed, 'episode': seed, 'backend': 'robosuite', 'task': 'Lift',
                  'validation_version': 1, 'debug': False, 'label': args.label,
                  'perturbation': args.first_grasp_offset, 'max_steps': args.max_steps,
                  'max_attempts': args.max_attempts, 'planning_success': None,
                  'control_success': False}
        try:
            env, config = make_env(seed, args.max_steps)
            obs = env.reset()
            if env._check_success():
                raise RuntimeError('Episode is already successful after reset')
            controller = LiftSkills(env, obs, max_steps=args.max_steps)
            initial_xml = env.sim.model.get_xml()
            outcome = controller.run(max_attempts=args.max_attempts, first_grasp_offset=args.first_grasp_offset)
            result.update(outcome, task_success=outcome['success'], execution_success=True,
                          control_success=outcome['success'],
                          status='success' if outcome['success'] else 'task_failed',
                          steps=len(controller.actions),
                          initial_cube_z=controller.initial_cube_z,
                          final_cube_xyz=controller.cube.tolist(),
                          recovery_success=outcome['success'] and outcome['attempts'] > 1,
                          initial_state_sha256=hashlib.sha256(controller.states[0].tobytes()).hexdigest())
            result['duration_seconds'] = time.monotonic() - started
            data_path = args.output / f'ep_{seed:04d}.hdf5'
            result['trajectory_sha256'] = save_trajectory(data_path, controller, initial_xml, config, result)
            result['trajectory_file'] = data_path.name
        except Exception as exc:
            result.update(success=False, task_success=False, execution_success=False, control_success=False,
                          status='error', error=repr(exc), traceback=traceback.format_exc())
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    result['close_error'] = repr(exc)
            result['duration_seconds'] = time.monotonic() - started
            write_json(path, result)
        results.append(result)
        print(json.dumps({k: result.get(k) for k in ['seed', 'status', 'steps', 'attempts', 'failures', 'error', 'duration_seconds']}), flush=True)
    elapsed = time.monotonic() - run_started
    successful = sum(r['success'] for r in results)
    summary = {'episodes': len(results), 'successful': successful,
               'first_attempt_successes': sum(r['success'] and r.get('first_attempt_success', False) for r in results),
               'recovery_successes': sum(r['success'] and r.get('recovery_success', False) for r in results),
               'errors': sum(r['status'] == 'error' for r in results),
               'success_rate': successful / len(results), 'wall_seconds': elapsed,
               'successful_trajectories_per_hour': successful * 3600 / elapsed,
               'includes_reset_recording_cleanup': True, 'excludes_import_and_queue_time': True,
               'total_actions': sum(r.get('steps', 0) for r in results)}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
