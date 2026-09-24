"""python -m robot_stack.collect --backend metaworld --task reach-v3 --output RUN"""
import argparse
import importlib
from pathlib import Path
import time
import traceback
from .core import collect_source, collect_triplet, write_json


def load_adapter(backend, task, options):
    options = dict(options)
    factory = options.pop('adapter_factory', None)
    if factory:
        module, name = factory.split(':', 1)
        adapter = getattr(importlib.import_module(module), name)(task, **options)
        if adapter.backend != backend or adapter.task != task:
            raise ValueError('Custom adapter identity does not match requested backend/task')
        adapter.metadata['adapter_options'] = dict(options, adapter_factory=factory)
        return adapter
    names = {'metaworld': ('metaworld', 'MetaWorldAdapter'),
             'maniskill': ('maniskill', 'ManiSkillAdapter'),
             'robotwin': ('robotwin', 'RoboTwinAdapter'),
             'robocasa': ('robocasa', 'RoboCasaNavigationAdapter'),
             'ai2thor': ('ai2thor', 'AI2ThorNavigationAdapter')}
    if backend not in names:
        raise ValueError(f'{backend} requires an adapter_factory and a verified native recovery policy')
    module, cls = names[backend]
    if backend == 'robocasa' and task != 'NavigateKitchen':
        module, cls = 'robocasa_mobile', 'RoboCasaMobileManipulationAdapter'
    adapter = getattr(importlib.import_module('robot_stack.adapters.' + module), cls)(task, **options)
    adapter.metadata['adapter_options'] = options
    return adapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=200)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--perturb-steps', type=int, default=15)
    p.add_argument('--branch-fraction', type=float, default=.35)
    p.add_argument('--adapter-options', default='{}', help='JSON object of adapter-specific options')
    p.add_argument('--schedule', type=Path, help='JSON perturbation schedule (overrides branch-fraction/perturb-steps)')
    p.add_argument('--error-threshold', type=float, default=.03)
    p.add_argument('--source-only', action='store_true', help='Clean baseline: save expert actions without any perturbation')
    args = p.parse_args()
    if args.source_only and args.schedule:
        p.error('source-only cannot be combined with schedule')
    if args.output is not None:
        args.output = args.output.resolve()
    import json
    options = json.loads(args.adapter_options)
    schedule = json.loads(args.schedule.read_text()) if args.schedule else None
    if schedule is not None:
        from .schedule import validate_schedule
        validate_schedule(schedule)
    if args.episodes < 1 or not isinstance(options, dict):
        p.error('Positive episodes and JSON object options required')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'run_config.json', dict(vars(args), output=str(args.output),
                                                     adapter_options=options, schedule=schedule))
    started, rows = time.monotonic(), []
    for index in range(args.episodes):
        seed = args.seed_start + index
        adapter = None
        try:
            episode_options = dict(options)
            if args.backend == 'metaworld' and 'task_index' not in options:
                if index >= 50:
                    raise ValueError('A MetaWorld MT1 split supplies 50 distinct task indices')
                episode_options['task_index'] = index
            adapter = load_adapter(args.backend, args.task, episode_options)
            if args.source_only:
                result = collect_source(adapter, seed, args.output / f'ep_{seed:04d}', budget=args.max_steps)
            else:
                result = collect_triplet(adapter, seed, args.output / f'ep_{seed:04d}',
                                     budget=args.max_steps, branch_fraction=args.branch_fraction,
                                     perturb_steps=args.perturb_steps, schedule=schedule,
                                     error_threshold=args.error_threshold)
        except Exception as exc:
            result = {'seed': seed, 'status': 'error', 'qualified_correction': False,
                      'error': repr(exc), 'traceback': traceback.format_exc()}
            write_json(args.output / f'ep_{seed:04d}_error.json', result)
        finally:
            if adapter is not None:
                adapter.close()
        rows.append(result)
        print(json.dumps(result), flush=True)
    summary = {'backend': args.backend, 'task': args.task, 'source_attempts': len(rows),
               'successful_sources': sum(r.get('source_success', False) for r in rows),
               'qualified_corrections': sum(r['qualified_correction'] for r in rows),
               'errors': sum(r.get('status') == 'error' for r in rows),
               'wall_seconds': time.monotonic() - started, 'results': rows}
    write_json(args.output / 'summary.json', summary)
    return int(summary['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
