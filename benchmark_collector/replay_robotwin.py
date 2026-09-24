"""Replay this collector's trusted native RoboTwin joint-path files on the same seed."""
import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys

from .robotwin import config
from robosuite_collector.collect import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', type=Path)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.result, args.output = args.result.resolve(), args.output.resolve()
    record = json.loads(args.result.read_text())
    if not record['success'] or not record['planning_success']:
        raise ValueError('Select a successfully collected native demonstration')
    episode_dir = (args.result.parent / record['trajectory_file']).resolve().parent.parent
    with open(episode_dir / '_traj_data/episode0.pkl', 'rb') as f:
        paths = pickle.load(f)
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'description/utils'))
    cls = getattr(importlib.import_module('envs.' + record['task']), record['task'])
    cfg = config(root, episode_dir, record['task'])
    cfg.update(need_plan=False, save_data=False, collect_data=False, **paths)
    random.seed(record['seed'])
    env = cls()
    try:
        env.setup_demo(now_ep_num=0, seed=record['seed'], **cfg)
        env.set_path_lst(cfg)
        env.play_once()
        success = bool(env.check_success())
        result = {'backend': 'robotwin', 'task': record['task'], 'seed': record['seed'],
                  'native_final_success': success, 'passed': success,
                  'replay_mode': 'native stored joint paths on same-seed reset',
                  'numeric_state_comparison': False, 'restored_success_state': False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)
        return int(not success)
    finally:
        env.close_env(clear_cache=True)


if __name__ == '__main__':
    raise SystemExit(main())
