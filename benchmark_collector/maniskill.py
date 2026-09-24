"""Fixed-seed ManiSkill native motion-planning collection using CPU PhysX."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
import traceback

import gymnasium as gym
import h5py
import numpy as np

from robosuite_collector.collect import write_json


class ActionBudget(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps, self.steps = max_steps, 0

    def reset(self, *args, **kwargs):
        self.steps = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        if self.steps >= self.max_steps:
            raise RuntimeError('Episode action budget exhausted')
        array = action.detach().cpu().numpy() if hasattr(action, 'detach') else np.asarray(action)
        if not np.isfinite(array).all():
            raise ValueError('Non-finite planner action')
        self.steps += 1
        return self.env.step(action)


def inspect_trajectory(path):
    with h5py.File(path, 'r') as f:
        groups = list(f.keys())
        if len(groups) != 1:
            raise ValueError(f'Expected one recorded episode, found {groups}')
        group = f[groups[0]]
        steps = len(group['actions'])
        if steps < 1:
            raise ValueError('Empty trajectory')
        sizes = {}
        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                sizes[name] = list(value.shape)
                if value.dtype.kind in 'fc' and not np.isfinite(value[:]).all():
                    raise ValueError(f'Non-finite recorded data: {name}')
                if name.startswith('env_states/') and len(value) != steps + 1:
                    raise ValueError(f'Environment state/action alignment mismatch: {name}')
        group.visititems(visit)
        if not any(k.startswith('env_states/') for k in sizes):
            raise ValueError('Native recorder did not save environment states')
        return steps, sizes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['PickCube-v1', 'PushCube-v1', 'StackCube-v1', 'PlaceSphere-v1'])
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=500)
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        p.error('Episode and action budgets must be positive')
    import mani_skill.envs
    import torch
    from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
    from mani_skill.utils.wrappers.record import RecordEpisode
    unknown = set(args.tasks) - set(MP_SOLUTIONS)
    if unknown:
        p.error(f'Unsupported native planner tasks: {sorted(unknown)}')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {n: version(n) for n in ('mani_skill', 'sapien', 'mplib', 'torch', 'numpy', 'gymnasium')}
    write_json(args.output / 'run_config.json', {
        'backend': 'maniskill', 'tasks': args.tasks, 'episodes_per_task': args.episodes,
        'seed_start': args.seed_start, 'max_steps': args.max_steps, 'versions': versions,
        'sim_backend': 'physx_cpu', 'render_backend': 'none', 'control_mode': 'pd_joint_pos',
        'policy_source': 'upstream ManiSkill Panda motion-planning solutions',
        'success_criterion': 'native evaluate()[success] at final physical state',
        'scope': 'bounded expert collection, custom 500-step budget; not a learned-policy benchmark score'})
    records, all_started = [], time.monotonic()
    for task in args.tasks:
        folder = args.output / task
        folder.mkdir()
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            result_path = folder / f'ep_{seed:04d}_episode_result.json'
            result = dict(backend='maniskill', task=task, seed=seed, label=f'ManiSkill / {task}',
                          validation_version=1, debug=False, status='incomplete', success=False,
                          planning_success=False, execution_success=False,
                          control_success=False, task_success=False)
            write_json(result_path, result)
            env = None
            started = time.monotonic()
            try:
                base = gym.make(task, obs_mode='state', control_mode='pd_joint_pos',
                                sim_backend='physx_cpu', render_backend='none',
                                max_episode_steps=args.max_steps)
                limited = ActionBudget(base, args.max_steps)
                env = RecordEpisode(limited, output_dir=str(folder), trajectory_name=f'ep_{seed:04d}',
                                    save_video=False, save_on_reset=False, record_env_state=True,
                                    source_type='motionplanning', source_desc='Upstream ManiSkill expert')
                solver_result = MP_SOLUTIONS[task](env, seed=seed, debug=False, vis=False)
                planned = not (isinstance(solver_result, (int, np.integer)) and solver_result == -1)
                native = bool(env.unwrapped.evaluate()['success'].item())
                env.flush_trajectory()
                env.close()
                env = None
                trajectory = folder / f'ep_{seed:04d}.h5'
                steps, sizes = inspect_trajectory(trajectory)
                success = planned and native
                result.update(status='success' if success else 'task_failed', success=success,
                              planning_success=planned, execution_success=True,
                              task_success=native, control_success=success,
                              steps=steps, max_steps=args.max_steps, dataset_shapes=sizes,
                              native_final_success=native, trajectory_file=trajectory.name,
                              trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest())
            except Exception as exc:
                result.update(status='error', success=False, task_success=False,
                              execution_success=False, control_success=False,
                              error=repr(exc), traceback=traceback.format_exc())
                if env is not None and getattr(env, '_trajectory_buffer', None) is not None:
                    try:
                        env.flush_trajectory()
                    except Exception as save_exc:
                        result['save_error'] = repr(save_exc)
            finally:
                if env is not None:
                    env.close()
                result['duration_seconds'] = time.monotonic() - started
                write_json(result_path, result)
            records.append(result)
            print(json.dumps({k: result.get(k) for k in ('task', 'seed', 'status', 'steps', 'error')}), flush=True)
    by_task = {}
    for task in args.tasks:
        rows = [r for r in records if r['task'] == task]
        by_task[task] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                         'errors': sum(r['status'] == 'error' for r in rows)}
    summary = {'backend': 'maniskill', 'by_task': by_task, 'episodes': len(records),
               'successes': sum(r['success'] for r in records),
               'errors': sum(r['status'] == 'error' for r in records),
               'wall_seconds': time.monotonic() - all_started,
               'includes_environment_planning_execution_saving': True,
               'excludes_imports_queue_and_rendering': True}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
