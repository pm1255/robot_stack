"""Audit matched clean / perturbed / recovery runs and build correction links."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def audit(root):
    names = ('clean', 'perturbed', 'recovery')
    runs, summaries = {}, {}
    checked = 0
    for name in names:
        folder = root / name
        summaries[name] = json.loads((folder / 'summary.json').read_text())
        records = {}
        for p in sorted(folder.glob('*_episode_result.json')):
            record = json.loads(p.read_text())
            if record['status'] == 'error' or 'trajectory_file' not in record:
                raise ValueError(f'Cannot pair an incomplete/error trajectory: {p}')
            trajectory = folder / record['trajectory_file']
            if hashlib.sha256(trajectory.read_bytes()).hexdigest() != record['trajectory_sha256']:
                raise ValueError(f'Trajectory checksum mismatch: {trajectory}')
            with h5py.File(trajectory, 'r') as f:
                demo = f['data/demo_0']
                actions, states = demo['actions'][:], demo['states'][:]
                if actions.shape != (record['steps'], 7) or len(states) != len(actions):
                    raise ValueError(f'State/action alignment error: {trajectory}')
                if not all(np.isfinite(a).all() for a in (actions, states, demo['final_state'][:])):
                    raise ValueError(f'Non-finite trajectory: {trajectory}')
                if hashlib.sha256(states[0].tobytes()).hexdigest() != record['initial_state_sha256']:
                    raise ValueError(f'Initial state hash mismatch: {trajectory}')
                flags = demo['trace/verified_success'][:]
                if record['success'] and (len(flags) < 10 or not flags[-10:].all()):
                    raise ValueError(f'Success missing stable physical evidence: {trajectory}')
                for dataset in demo['trace'].values():
                    if len(dataset) != len(actions):
                        raise ValueError(f'Post-action trace alignment error: {trajectory}')
                records[record['seed']] = dict(record=record, actions=actions, states=states,
                                              final_state=demo['final_state'][:],
                                              env_args=f['data'].attrs['env_args'])
            checked += 1
        if len(records) != summaries[name]['episodes']:
            raise ValueError(f'Missing or duplicate episodes: {name}')
        runs[name] = records
    seeds = set(runs['clean'])
    if not seeds or any(set(runs[n]) != seeds for n in names):
        raise ValueError('Seed sets differ or are empty')
    pairs = []
    for seed in sorted(seeds):
        clean, bad, recovery = [runs[n][seed] for n in names]
        if len({r['record']['initial_state_sha256'] for r in (clean, bad, recovery)}) != 1:
            raise ValueError(f'Initial states differ at seed {seed}')
        if len({r['env_args'] for r in (clean, bad, recovery)}) != 1:
            raise ValueError(f'Environment configurations differ at seed {seed}')
        n = len(bad['actions'])
        if not (np.array_equal(bad['actions'], recovery['actions'][:n])
                and np.array_equal(bad['states'], recovery['states'][:n])):
            raise ValueError(f'Perturbed action/state prefix differs at seed {seed}')
        continued_state = recovery['states'][n] if n < len(recovery['states']) else recovery['final_state']
        if not np.array_equal(bad['final_state'], continued_state):
            raise ValueError(f'Recovery does not continue the failure state at seed {seed}')
        rec = recovery['record']
        is_correction = clean['record']['success'] and not bad['record']['success'] and rec['success']
        pairs.append({'source_demo_id': f'clean/ep_{seed:04d}.hdf5',
                      'failed_demo_id': f'perturbed/ep_{seed:04d}.hdf5',
                      'recovery_demo_id': f'recovery/ep_{seed:04d}.hdf5',
                      'seed': seed, 'source_split_group': f'Lift_seed_{seed}',
                      'construction': 'matched_seed_rerun_with_first_grasp_target_offset',
                      'restored_mid_trajectory': False,
                      'perturbation_xyz_m': rec['perturbation'],
                      'failure_detected_after_action_count': n if not bad['record']['success'] else None,
                      'recovery_suffix_start_action_index': n if is_correction else None,
                      'recovery_suffix_end_exclusive': rec['steps'] if is_correction else None,
                      'verified_correction': bool(is_correction)})
    return {'schema_version': 1, 'task': 'robosuite Lift / Panda', 'seeds': sorted(seeds),
            'summaries': summaries, 'hdf5_files_verified': checked,
            'matched_initial_states': len(seeds), 'identical_perturbed_prefixes': len(seeds),
            'verified_correction_pairs': sum(p['verified_correction'] for p in pairs),
            'scope': 'Controlled 6cm first-grasp perturbation; simulator truth feedback; not a kitchen or vision benchmark',
            'pairs': pairs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('pairs', 'seeds')}, indent=2))
