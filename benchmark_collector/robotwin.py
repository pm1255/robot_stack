"""Bounded RoboTwin native expert collection on fixed seeds (no success-seed search)."""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import h5py
import numpy as np
import yaml

from robosuite_collector.collect import write_json


def config(root, output, task):
    cfg = yaml.safe_load((root / 'task_config/demo_clean.yml').read_text())
    embodiment = yaml.safe_load((root / 'task_config/_embodiment_config.yml').read_text())['aloha-agilex']['file_path']
    robot = yaml.safe_load((root / embodiment / 'config.yml').read_text())
    cfg.update(task_name=task, task_config='demo_clean',
               left_robot_file=embodiment, right_robot_file=embodiment,
               dual_arm_embodied=True, left_embodiment_config=copy.deepcopy(robot),
               right_embodiment_config=copy.deepcopy(robot), head_camera_h=240, head_camera_w=320,
               eval_mode=False, render_freq=0, save_data=True, collect_data=True,
               eval_video_log=False, need_plan=True, save_freq=30, save_path=str(output))
    cfg['camera']['collect_wrist_camera'] = False
    return cfg


def inspect_hdf5(path):
    sizes = {}
    with h5py.File(path, 'r') as f:
        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                sizes[name] = list(value.shape)
                if value.dtype.kind in 'fc' and not np.isfinite(value[:]).all():
                    raise ValueError(f'Non-finite native trajectory dataset: {name}')
        f.visititems(visit)
    if not sizes or not any('joint' in k for k in sizes) or not any('endpose' in k for k in sizes):
        raise ValueError('Native trajectory missing robot joint or end-effector data')
    return sizes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['stack_blocks_two', 'turn_switch', 'place_empty_cup'])
    p.add_argument('--seed-start', type=int, default=100)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--replay-first', action='store_true')
    args = p.parse_args()
    if args.episodes < 1:
        p.error('episodes must be positive')
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'description/utils'))
    import torch
    from importlib.metadata import version
    torch.set_num_threads(4)
    versions = {n: version(n) for n in ('sapien', 'torch', 'numpy', 'mplib', 'h5py')}
    sources = ['envs/_base_task.py', 'task_config/demo_clean.yml'] + [f'envs/{t}.py' for t in args.tasks]
    write_json(output / 'run_config.json', {
        'backend': 'robotwin', 'tasks': args.tasks, 'seed_start': args.seed_start,
        'episodes_per_task': args.episodes, 'versions': versions,
        'success_criterion': 'native check_success() after physical play_once(), and plan_success',
        'policy_source': 'upstream native scripted experts + cuRobo',
        'configuration': 'demo_clean, aloha-agilex, 320x240 head RGB, save_freq=30',
        'seed_search': False, 'source_root': str(root),
        'source_hashes': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in sources}})
    records, started_all = [], time.monotonic()
    for task in args.tasks:
        cls = getattr(importlib.import_module('envs.' + task), task)
        directory = output / task
        directory.mkdir()
        for index in range(args.episodes):
            seed = args.seed_start + index
            episode_dir = directory / f'ep_{seed:04d}'
            episode_dir.mkdir()
            result_path = directory / f'ep_{seed:04d}_episode_result.json'
            result = dict(backend='robotwin', task=task, seed=seed, label=f'RoboTwin / {task}',
                          validation_version=1, debug=False, success=False, status='incomplete',
                          planning_success=False, control_success=False,
                          execution_success=False, task_success=False)
            write_json(result_path, result)
            env, replay_env = None, None
            started = time.monotonic()
            try:
                random.seed(seed)
                env = cls()
                cfg = config(root, episode_dir, task)
                env.setup_demo(now_ep_num=0, seed=seed, **cfg)
                native_info = env.play_once()
                native_success = bool(env.check_success())
                planned = bool(env.plan_success)
                paths = {'left_joint_path': copy.deepcopy(env.left_joint_path),
                         'right_joint_path': copy.deepcopy(env.right_joint_path)}
                env.save_traj_data(0)
                env.merge_pkl_to_hdf5_video()
                trajectory = episode_dir / 'data/episode0.hdf5'
                sizes = inspect_hdf5(trajectory)
                success = native_success and planned
                result.update(success=success, status='success' if success else 'task_failed',
                              planning_success=planned, execution_success=True,
                              task_success=native_success, control_success=success,
                              native_frames=int(env.FRAME_IDX), dataset_shapes=sizes,
                              trajectory_file=str(trajectory.relative_to(directory)),
                              trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                              native_info=native_info)
                env.close_env(clear_cache=True)
                env = None
                if args.replay_first and index == 0 and success:
                    random.seed(seed)
                    replay_env = cls()
                    replay_cfg = dict(cfg, need_plan=False, save_data=False, collect_data=False,
                                      **paths)
                    replay_env.setup_demo(now_ep_num=0, seed=seed, **replay_cfg)
                    replay_env.set_path_lst(replay_cfg)
                    replay_env.play_once()
                    result['native_plan_replay_success'] = bool(replay_env.check_success())
                    result['replay_mode'] = 'same-seed native stored joint paths, no success-state restore'
            except Exception as exc:
                result.update(status='error', success=False, execution_success=False,
                              control_success=False, task_success=False,
                              error=repr(exc), traceback=traceback.format_exc())
            finally:
                for instance in (env, replay_env):
                    if instance is not None:
                        try:
                            instance.close_env(clear_cache=True)
                        except Exception as exc:
                            result['close_error'] = repr(exc)
                result['duration_seconds'] = time.monotonic() - started
                # Only upstream instruction metadata is converted for serialization.
                if 'native_info' in result:
                    result['native_info'] = json.loads(json.dumps(result['native_info'], default=str))
                write_json(result_path, result)
            records.append(result)
            print('ROBOT_STACK_RESULT', json.dumps({k: result.get(k) for k in
                ('task', 'seed', 'status', 'native_frames', 'native_plan_replay_success', 'error')}), flush=True)
    by_task = {}
    for task in args.tasks:
        rows = [r for r in records if r['task'] == task]
        by_task[task] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                         'errors': sum(r['status'] == 'error' for r in rows)}
    summary = {'backend': 'robotwin', 'by_task': by_task, 'episodes': len(records),
               'successes': sum(r['success'] for r in records),
               'errors': sum(r['status'] == 'error' for r in records),
               'wall_seconds': time.monotonic() - started_all,
               'includes_rendering_recording_and_requested_replays': True}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
