"""Collect native MetaWorld v3 scripted experts, with seeded tasks and replayable states.

This is a selected-task collection check, not the complete MT10/MT50 evaluation.
"""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np

from robosuite_collector.collect import write_json


TASKS = {
    'reach-v3': 'SawyerReachV3Policy',
    'push-v3': 'SawyerPushV3Policy',
    'pick-place-v3': 'SawyerPickPlaceV3Policy',
    'door-open-v3': 'SawyerDoorOpenV3Policy',
    'drawer-open-v3': 'SawyerDrawerOpenV3Policy',
    'button-press-v3': 'SawyerButtonPressV3Policy',
    'window-open-v3': 'SawyerWindowOpenV3Policy',
    'faucet-open-v3': 'SawyerFaucetOpenV3Policy',
}


def state(env):
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, result, spec)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite MuJoCo integration state')
    return result


def create(task, benchmark_seed, task_index, reset_seed, render=False):
    import metaworld
    benchmark = metaworld.MT1(task, seed=benchmark_seed)
    env = benchmark.train_classes[task](render_mode='rgb_array' if render else None)
    native_task = benchmark.train_tasks[task_index]
    env.set_task(native_task)
    obs, _ = env.reset(seed=reset_seed)
    return env, obs, native_task


def save(path, states, actions, raw_actions, observations, rewards, flags, metadata, task_data):
    temporary = path.with_suffix('.hdf5.tmp')
    with h5py.File(temporary, 'x') as f:
        f.attrs['schema'] = 'robot_stack.metaworld.v1'
        f.attrs['metadata'] = json.dumps(metadata)
        f.attrs['state_spec'] = 'mujoco.mjtState.mjSTATE_INTEGRATION'
        f.attrs['alignment'] = 'states/observations have T+1 rows; actions/rewards/success have T rows'
        for name, data in [('states', states), ('actions', actions), ('raw_actions', raw_actions),
                           ('observations', observations), ('rewards', rewards), ('success', flags)]:
            f.create_dataset(name, data=np.asarray(data), compression='gzip')
        # Kept for provenance; replay regenerates tasks instead of unpickling this field.
        f.create_dataset('native_task_bytes', data=np.frombuffer(task_data, dtype=np.uint8))
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import metaworld
    from metaworld import policies
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {name: version(name) for name in ('metaworld', 'mujoco', 'numpy', 'gymnasium', 'h5py')}
    write_json(args.output / 'run_config.json', {
        'backend': 'metaworld', 'tasks': args.tasks, 'episodes_per_task': args.episodes,
        'benchmark_seed': args.benchmark_seed, 'seed_start': args.seed_start,
        'task_indices': list(range(args.episodes)), 'max_steps': args.max_steps,
        'policy_source': 'upstream MetaWorld scripted experts, simulator observations',
        'success_criterion': 'native info[success], stop on first success', 'versions': versions,
        'scope': 'selected-task expert collection, not MT10/MT50 benchmark score'})
    results, start = [], time.monotonic()
    for task in args.tasks:
        directory = args.output / task
        directory.mkdir()
        setup_start = time.monotonic()
        benchmark = metaworld.MT1(task, seed=args.benchmark_seed)
        setup_seconds = time.monotonic() - setup_start
        for index in range(args.episodes):
            seed = args.seed_start + index
            path = directory / f'ep_{seed:04d}_episode_result.json'
            result = dict(backend='metaworld', task=task, seed=seed, task_index=index,
                          benchmark_seed=args.benchmark_seed, versions=versions,
                          validation_version=1, label=f'MetaWorld / {task}', debug=False,
                          status='incomplete', success=False, planning_success=None,
                          execution_success=False, control_success=False, task_success=False)
            write_json(path, result)
            started = time.monotonic()
            env = None
            try:
                env = benchmark.train_classes[task]()
                native_task = benchmark.train_tasks[index]
                env.set_task(native_task)
                obs, _ = env.reset(seed=seed)
                policy = getattr(policies, TASKS[task])()
                states, observations = [state(env)], [np.array(obs).copy()]
                actions, raw_actions, rewards, flags = [], [], [], []
                for _ in range(min(args.max_steps, env.max_path_length)):
                    raw = np.asarray(policy.get_action(obs), dtype=np.float64)
                    if raw.shape != env.action_space.shape or not np.isfinite(raw).all():
                        raise ValueError('Invalid native expert action')
                    action = np.clip(raw, env.action_space.low, env.action_space.high)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(reward):
                        raise ValueError('Non-finite observation or reward')
                    flag = bool(info['success'])
                    states.append(state(env)); observations.append(np.array(obs).copy())
                    actions.append(action); raw_actions.append(raw); rewards.append(float(reward)); flags.append(flag)
                    if flag or terminated or truncated:
                        break
                success = bool(flags and flags[-1])
                result.update(status='success' if success else 'task_failed', success=success,
                              task_success=success, control_success=success, execution_success=True,
                              steps=len(actions), native_task_sha256=hashlib.sha256(native_task.data).hexdigest(),
                              initial_state_sha256=hashlib.sha256(states[0].tobytes()).hexdigest(),
                              native_final_success=success, max_steps=args.max_steps,
                              task_generation_seconds=setup_seconds if index == 0 else 0)
                trajectory = directory / f'ep_{seed:04d}.hdf5'
                result['trajectory_sha256'] = save(trajectory, states, actions, raw_actions,
                    observations, rewards, flags, result, native_task.data)
                result['trajectory_file'] = trajectory.name
            except Exception as exc:
                result.update(status='error', success=False, task_success=False,
                              control_success=False, execution_success=False,
                              error=repr(exc), traceback=traceback.format_exc())
            finally:
                if env is not None:
                    env.close()
                result['duration_seconds'] = time.monotonic() - started
                write_json(path, result)
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ('task', 'seed', 'status', 'steps', 'error')}), flush=True)
    by_task = {}
    for task in args.tasks:
        rows = [r for r in results if r['task'] == task]
        by_task[task] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                         'errors': sum(r['status'] == 'error' for r in rows)}
    summary = {'backend': 'metaworld', 'by_task': by_task, 'episodes': len(results),
               'successes': sum(r['success'] for r in results),
               'errors': sum(r['status'] == 'error' for r in results),
               'wall_seconds': time.monotonic() - start, 'includes_task_generation_and_saving': True,
               'excludes_imports_queue_and_rendering': True}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary['errors'] > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=100)
    p.add_argument('--benchmark-seed', type=int, default=20260924)
    p.add_argument('--max-steps', type=int, default=500)
    args = p.parse_args()
    if not 1 <= args.episodes <= 50 or args.max_steps < 1:
        p.error('Use 1..50 distinct goal indices per task and a positive action budget')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
