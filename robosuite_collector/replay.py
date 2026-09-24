"""Independently replay saved actions from the seeded reset; optionally render video.

This validates dynamics, not just state playback. Exact version/config/seed match
is required; no saved-state substitution occurs during replay.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


class VideoWriter:
    """Stream RGB frames to an installed FFmpeg; no frame cache is required."""
    def __init__(self, path, *, width=640, height=480, fps=5):
        executable = shutil.which('ffmpeg')
        if executable is None:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        self.process = subprocess.Popen([
            executable, '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
            '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def append_data(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin.closed:
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        error = self.process.stderr.read().decode(errors='replace')
        if self.process.wait() != 0:
            raise RuntimeError('Video encoding failed: ' + error)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video', type=Path)
    args = p.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl' if args.video else 'disable')
    if args.video:
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import h5py
    import numpy as np
    import robosuite
    from .collect import write_json
    from .lift import LiftSkills
    with h5py.File(args.trajectory, 'r') as f:
        data, demo = f['data'], f['data/demo_0']
        env_args = json.loads(data.attrs['env_args'])
        states, actions, final = demo['states'][:], demo['actions'][:], demo['final_state'][:]
        expected = bool(demo.attrs['success'])
        expected_flags = demo['trace/verified_success'][:]
    if len(actions) != len(states) or len(actions) == 0:
        raise ValueError('Trajectory has inconsistent or empty state/action arrays')
    cfg = env_args['env_kwargs']
    cfg['has_offscreen_renderer'] = bool(args.video)
    env = robosuite.make(env_args['env_name'], **cfg)
    writer = None
    try:
        obs = env.reset()
        controller = LiftSkills(env, obs)
        initial_error = float(np.max(np.abs(controller.state() - states[0])))
        if initial_error > 1e-9:
            raise ValueError(f'Seeded reset differs from saved initial state: {initial_error}')
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = VideoWriter(args.video)
        max_error, flags = initial_error, []
        for i, action in enumerate(actions):
            obs, _, _, _ = env.step(action)
            controller.obs = obs
            target = states[i + 1] if i + 1 < len(states) else final
            error = float(np.max(np.abs(controller.state() - target)))
            if not np.isfinite(error):
                raise ValueError('Non-finite state replay error')
            max_error = max(max_error, error)
            flags.append(controller.success())
            if writer is not None and (i % 4 == 0 or i == len(actions) - 1):
                frame = env.sim.render(width=640, height=480, camera_name='agentview')[::-1]
                writer.append_data(frame)
        stable_success = len(flags) >= 10 and all(flags[-10:])
        passed = (max_error <= 1e-8 and stable_success == expected
                  and np.array_equal(flags, expected_flags))
        if writer is not None:
            writer.close()
            writer = None
        result = {'trajectory': args.trajectory.name, 'actions': len(actions),
                  'initial_max_abs_error': initial_error, 'max_abs_state_error': max_error,
                  'expected_success': expected, 'replayed_stable_success': stable_success,
                  'success_trace_matches': bool(np.array_equal(flags, expected_flags)),
                  'passed': bool(passed), 'replay_mode': 'actions_from_seeded_reset',
                  'video': args.video.name if args.video else None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)
        return 0 if passed else 1
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
