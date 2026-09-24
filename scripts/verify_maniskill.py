"""Audit all recorded files and replay the first successful demo per selected task."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_collector.maniskill import inspect_trajectory
from benchmark_collector.replay_maniskill import replay
from robosuite_collector.collect import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checked, chosen = 0, {}
    for path in sorted(args.input.rglob('*_episode_result.json')):
        record = json.loads(path.read_text())
        trajectory = path.parent / record['trajectory_file']
        if hashlib.sha256(trajectory.read_bytes()).hexdigest() != record['trajectory_sha256']:
            raise ValueError(f'Checksum mismatch: {trajectory}')
        steps, _ = inspect_trajectory(trajectory)
        if steps != record['steps']:
            raise ValueError(f'Action count mismatch: {trajectory}')
        checked += 1
        if record['success']:
            chosen.setdefault(record['task'], trajectory)
    results = []
    for task, trajectory in chosen.items():
        video = args.output / f'{task}.mp4' if args.video else None
        result = replay(trajectory, video)
        write_json(args.output / f'{task}.json', result)
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {'hdf5_verified': checked, 'action_replays': results,
               'all_replays_passed': bool(results) and all(r['passed'] for r in results),
               'sample_selection': 'first successfully collected demonstration per task'}
    write_json(args.output / 'audit.json', summary)
    raise SystemExit(0 if summary['all_replays_passed'] else 1)
