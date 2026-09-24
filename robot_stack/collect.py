"""python -m robot_stack.collect --backend metaworld --task reach-v3 --output RUN"""
import argparse
import importlib
from pathlib import Path
import time
import traceback
from .core import collect_triplet, write_json


def load_adapter(backend, task, options):
    names = {'metaworld': ('metaworld', 'MetaWorldAdapter'),
             'robocasa': ('robocasa', 'RoboCasaNavigationAdapter'),
             'ai2thor': ('ai2thor', 'AI2ThorNavigationAdapter')}
    module, cls = names[backend]
    return getattr(importlib.import_module('robot_stack.adapters.' + module), cls)(task, **options)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['metaworld', 'robocasa', 'ai2thor'], required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=200)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--perturb-steps', type=int, default=15)
    p.add_argument('--branch-fraction', type=float, default=.35)
    p.add_argument('--adapter-options', default='{}', help='JSON object of adapter-specific options')
    args = p.parse_args()
    import json
    options = json.loads(args.adapter_options)
    if args.episodes < 1 or not isinstance(options, dict):
        p.error('Positive episodes and JSON object options required')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'run_config.json', dict(vars(args), output=str(args.output), adapter_options=options))
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
            result = collect_triplet(adapter, seed, args.output / f'ep_{seed:04d}',
                                     budget=args.max_steps, branch_fraction=args.branch_fraction,
                                     perturb_steps=args.perturb_steps)
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
