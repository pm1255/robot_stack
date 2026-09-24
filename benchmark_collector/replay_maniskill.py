"""Independent native ManiSkill action replay, without simulator-state substitution."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

from robosuite_collector.collect import write_json
from robosuite_collector.replay import VideoWriter


def leaves(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for name, child in value.items():
            result.update(leaves(child, prefix + '/' + name if prefix else name))
        return result
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return {prefix: np.asarray(value).reshape(-1)}


def replay(path, video=None, tolerance=1e-5):
    import mani_skill.envs
    import torch
    torch.set_num_threads(1)
    meta = json.loads(path.with_suffix('.json').read_text())
    if len(meta['episodes']) != 1:
        raise ValueError('Expected one native episode')
    episode, spec = meta['episodes'][0], meta['env_info']
    states = {}
    with h5py.File(path, 'r') as f:
        group = f[f'traj_{episode["episode_id"]}']
        actions, expected_flags = group['actions'][:], group['success'][:]
        def visit(name, value):
            if isinstance(value, h5py.Dataset):
                states[name] = value[:]
        group['env_states'].visititems(visit)
    if any(len(values) != len(actions) + 1 for values in states.values()):
        raise ValueError('Native state/action alignment mismatch')
    kwargs = dict(spec['env_kwargs'], max_episode_steps=spec['max_episode_steps'])
    if video:
        kwargs.update(render_backend='gpu', render_mode='rgb_array')
    env = gym.make(spec['env_id'], **kwargs)
    writer = None
    try:
        env.reset(**episode['reset_kwargs'])
        def error(index):
            actual = leaves(env.unwrapped.get_state_dict())
            if set(actual) != set(states):
                raise ValueError('Native state entities differ from saved trajectory')
            errors = [float(np.max(np.abs(actual[k] - states[k][index].reshape(-1)))) for k in states]
            if not np.isfinite(errors).all():
                raise ValueError('Non-finite replay states')
            return max(errors)
        initial_error = error(0)
        max_error, flags = initial_error, []
        for i, action in enumerate(actions):
            _, _, _, _, info = env.step(action)
            flags.append(bool(info['success'].item()))
            max_error = max(max_error, error(i + 1))
            if video:
                frame = env.render()
                if hasattr(frame, 'detach'):
                    frame = frame.detach().cpu().numpy()
                if frame.ndim == 4:
                    frame = frame[0]
                if writer is None:
                    video.parent.mkdir(parents=True, exist_ok=True)
                    writer = VideoWriter(video, width=frame.shape[1], height=frame.shape[0],
                                         fps=env.unwrapped.control_freq)
                writer.append_data(frame)
        if writer is not None:
            writer.close()
            writer = None
        matches = bool(np.array_equal(flags, expected_flags))
        native = bool(env.unwrapped.evaluate()['success'].item())
        passed = max_error <= tolerance and matches and native == bool(episode['success'])
        return {'task': spec['env_id'], 'seed': episode['episode_seed'], 'actions': len(actions),
                'initial_max_abs_error': initial_error, 'max_abs_state_error': max_error,
                'state_tolerance': tolerance, 'success_trace_matches': matches,
                'native_final_success': native, 'expected_success': bool(episode['success']),
                'passed': bool(passed), 'video': video.name if video else None,
                'replay_mode': 'actions_from_same_seed_reset_no_state_restore'}
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video', type=Path)
    args = p.parse_args()
    result = replay(args.trajectory, args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
