"""Re-execute MetaWorld actions from the original seeded task and reset."""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .metaworld import create, state
from robosuite_collector.collect import write_json
from robosuite_collector.replay import VideoWriter


def replay(path, video=None):
    with h5py.File(path, 'r') as f:
        meta = json.loads(f.attrs['metadata'])
        states, actions = f['states'][:], f['actions'][:]
        observations, expected_flags = f['observations'][:], f['success'][:]
    if len(states) != len(actions) + 1 or len(observations) != len(states):
        raise ValueError('State/action alignment mismatch')
    env, obs, _ = create(meta['task'], meta['benchmark_seed'], meta['task_index'], meta['seed'], bool(video))
    writer = None
    try:
        initial_error = float(np.max(np.abs(state(env) - states[0])))
        max_error = initial_error
        obs_error = float(np.max(np.abs(obs - observations[0])))
        flags = []
        for index, action in enumerate(actions):
            obs, _, _, _, info = env.step(action)
            max_error = max(max_error, float(np.max(np.abs(state(env) - states[index + 1]))))
            obs_error = max(obs_error, float(np.max(np.abs(obs - observations[index + 1]))))
            flags.append(bool(info['success']))
            if video:
                frame = env.render()
                if writer is None:
                    video.parent.mkdir(parents=True, exist_ok=True)
                    writer = VideoWriter(video, width=frame.shape[1], height=frame.shape[0], fps=20)
                writer.append_data(frame)
        if writer is not None:
            writer.close()
            writer = None
        matches = np.array_equal(flags, expected_flags)
        passed = max_error <= 1e-8 and obs_error <= 1e-8 and matches
        return {'task': meta['task'], 'seed': meta['seed'], 'actions': len(actions),
                'initial_max_abs_error': initial_error, 'max_abs_state_error': max_error,
                'max_abs_observation_error': obs_error, 'success_trace_matches': bool(matches),
                'native_final_success': bool(flags[-1]), 'passed': bool(passed),
                'replay_mode': 'actions_from_regenerated_seeded_native_task',
                'video': video.name if video else None,
                'video_playback_fps': 20 if video else None,
                'video_time_is_physics_time': False if video else None}
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    result = replay(args.trajectory, args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
