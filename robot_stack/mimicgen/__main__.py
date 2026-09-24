"""Prepare, generate, and independently replay native MimicGen Lift data."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_state(actual, expected):
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-7):
        raise ValueError('Native action replay disagrees with recorded state')


def write_info(group, infos):
    if not infos:
        raise ValueError('Empty datagen information')
    for key in ('eef_pose', 'target_pose', 'gripper_action'):
        group.create_dataset(key, data=np.asarray([getattr(x, key) for x in infos]))
    for key in ('object_poses', 'subtask_term_signals'):
        for name in getattr(infos[0], key):
            values = np.asarray([getattr(x, key)[name] for x in infos])
            if key == 'subtask_term_signals':
                values = np.maximum.accumulate(values.astype(np.int32))
            group.create_dataset(key + '/' + name, data=values)
    group.attrs['env_interface_name'] = 'LiftInterface'
    group.attrs['env_interface_type'] = 'robot_stack_robosuite15'


def prepare(args):
    from .lift import LiftEnvironment, LiftInterface
    temporary = args.output.with_suffix('.tmp.hdf5')
    if args.output.exists() or temporary.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(temporary, 'x') as output:
            data = output.create_group('data')
            total = 0
            for i, path in enumerate(args.sources):
                with h5py.File(path, 'r') as source:
                    original = source['data/demo_0']
                    result = json.loads(original.attrs['collector_result'])
                    if not result['success']:
                        raise ValueError(f'Unsuccessful source: {path}')
                    env = LiftEnvironment(result['seed'], result['max_steps'])
                    try:
                        env.reset()
                        interface = LiftInterface(env)
                        infos = []
                        for state, action in zip(original['states'], original['actions'], strict=True):
                            check_state(env.state(), state)
                            infos.append(interface.get_datagen_info(action))
                            check_state(interface.target_pose_to_action(interface.action_to_target_pose(action)), action[:6])
                            env.step(action)
                        check_state(env.state(), original['final_state'][:])
                        if not env.is_success()['task']:
                            raise ValueError('Source failed final stable native success check')
                        grasp = [x.subtask_term_signals['grasp'] for x in infos]
                        if not grasp or grasp[0] or not any(grasp):
                            raise ValueError('Source has no observable 0→1 grasp boundary')
                        demo = data.create_group(f'demo_{i}')
                        for key in ('states', 'actions', 'final_state'):
                            demo.create_dataset(key, data=original[key][:], compression='gzip')
                        for key in original.attrs:
                            demo.attrs[key] = original.attrs[key]
                        demo.attrs['source_sha256'] = digest(path)
                        demo.attrs['source_name'] = path.name
                        write_info(demo.create_group('datagen_info'), infos)
                        data.attrs['env_args'] = json.dumps(env.serialize())
                        total += len(infos)
                    finally:
                        env.close()
            data.attrs['total'] = total
            data.attrs['preparation'] = 'native action replay; no state restoration; verified final stable success'
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def generate(args):
    from mimicgen.configs.task_spec import MG_TaskSpec
    from mimicgen.datagen.data_generator import DataGenerator
    from .lift import LiftEnvironment, LiftInterface
    args.output.mkdir(parents=True, exist_ok=False)
    spec = MG_TaskSpec()
    for signal in ('grasp', None):
        spec.add_subtask(object_ref='cube', subtask_term_signal=signal,
                         selection_strategy='random', action_noise=0.,
                         num_interpolation_steps=5, num_fixed_steps=0)
    with h5py.File(args.source, 'r') as source:
        keys = sorted(source['data'].keys())
    generator = DataGenerator(spec, str(args.source), demo_keys=keys)
    versions = {}
    for package in ('numpy', 'mujoco', 'robosuite', 'robomimic', 'torch'):
        versions[package] = importlib.metadata.version(package)
    config = {'source_sha256': digest(args.source), 'versions': versions,
              'seed_start': args.seed_start, 'episodes': args.episodes, 'max_steps': args.max_steps,
              'task': 'Lift', 'object_reference': 'cube', 'subtasks': ['grasp', None],
              'selection_strategy': 'random', 'action_noise': 0., 'interpolation_steps': 5,
              'transform_first_robot_pose': True, 'select_src_per_subtask': False,
              'upstream_commit': '72bd767c255545f462e7ccfb2731f2e5d4c1d9bb'}
    (args.output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    results = []
    started = time.monotonic()
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        np.random.seed(seed)
        env = LiftEnvironment(seed, args.max_steps)
        result = {'seed': seed, 'success': False, 'source_sha256': digest(args.source),
                  'source_demo_keys': keys, 'max_steps': args.max_steps,
                  'split_group': 'mimicgen-source:' + config['source_sha256'],
                  'upstream_commit': '72bd767c255545f462e7ccfb2731f2e5d4c1d9bb'}
        generated = None
        try:
            generated = generator.generate(env, LiftInterface(env), select_src_per_subtask=False,
                                           transform_first_robot_pose=True)
            result.update(success=env.is_success()['task'], upstream_ever_success=bool(generated['success']),
                          source_demo_indices=np.asarray(generated['src_demo_inds']).tolist())
        except Exception as exc:
            result['error'] = repr(exc)
        finally:
            if env.native is not None:
                result['steps'] = len(env.actions)
                with h5py.File(args.output / f'ep_{seed:04d}.hdf5', 'x') as f:
                    data = f.create_group('data')
                    data.attrs['env_args'] = json.dumps(env.serialize())
                    data.attrs['total'] = len(env.actions)
                    demo = data.create_group('demo_0')
                    demo.attrs['model_file'] = env.initial_xml
                    demo.attrs['result'] = json.dumps(result)
                    demo.create_dataset('states', data=np.asarray(env.states[:-1]))
                    demo.create_dataset('actions', data=np.asarray(env.actions).reshape(-1, 7))
                    demo.create_dataset('final_state', data=env.states[-1])
                    demo.create_dataset('stable_success', data=env.stable_flags)
                    demo.create_dataset('native_success_with_grasp_height', data=env.flags)
                    if generated is not None:
                        write_info(demo.create_group('datagen_info'), generated['datagen_infos'])
                        demo.create_dataset('src_demo_labels', data=generated['src_demo_labels'])
                env.close()
        results.append(result)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    summary = {'attempts': len(results), 'successful': sum(x['success'] for x in results),
               'errors': sum('error' in x for x in results), 'wall_seconds': time.monotonic() - started,
               'definition': 'native Lift + two-sided grasp + 10cm gain for final 10 steps',
               'correction_pairs': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return int(summary['errors'] > 0)


def replay(args):
    from .environment import LiftEnvironment
    from robosuite_collector.replay import VideoWriter
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in args.trajectories:
        with h5py.File(path, 'r') as f:
            demo = f['data/demo_0']
            result = json.loads(demo.attrs['result'])
            env = LiftEnvironment(result['seed'], result['max_steps'], render=bool(args.video_dir))
            writer = None
            try:
                env.reset()
                if args.video_dir:
                    writer = VideoWriter(args.video_dir / (path.stem + '.mp4'))
                for index, (state, action) in enumerate(zip(demo['states'], demo['actions'], strict=True)):
                    check_state(env.state(), state)
                    env.step(action)
                    if writer and index % 4 == 0:
                        writer.append_data(env.native.sim.render(width=640, height=480, camera_name='agentview')[::-1])
                check_state(env.state(), demo['final_state'][:])
                if env.is_success()['task'] != result['success']:
                    raise ValueError('Replay success differs from recorded outcome')
                if not np.array_equal(env.stable_flags, demo['stable_success'][:]):
                    raise ValueError('Replay success history differs')
                reports.append({'file': path.name, 'verified': True, 'success': result['success']})
            finally:
                if writer:
                    writer.close()
                env.close()
    args.output.write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports), flush=True)


def main():
    os.environ.setdefault('MUJOCO_GL', 'disable')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--sources', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('generate')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=400)
    p.add_argument('--max-steps', type=int, default=600)
    p = sub.add_parser('replay')
    p.add_argument('--trajectories', type=Path, nargs='+', required=True)
    p.add_argument('--video-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'generate' and min(args.episodes, args.max_steps) < 1:
        parser.error('Episodes and max steps must be positive')
    if args.command == 'replay' and args.video_dir:
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    return globals()[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
