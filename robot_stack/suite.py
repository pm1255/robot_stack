"""Enumerate native MetaWorld experts and run a bounded task × schedule sweep."""
import argparse
import json
from pathlib import Path
import time
import traceback
from .collect import load_adapter
from .core import collect_triplet, write_json
from .schedule import validate_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='metaworld')
    parser.add_argument('--tasks', nargs='+', default=['all'])
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--schedules', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed-start', type=int, default=600)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--adapter-options', default='{}')
    args = parser.parse_args()
    tasks = args.tasks
    if tasks == ['all']:
        if args.backend != 'metaworld':
            parser.error('Automatic all-task expert discovery currently supports MetaWorld only')
        from benchmark_collector.metaworld import discover_tasks
        tasks = list(discover_tasks())
    if args.list:
        print(json.dumps({'backend':args.backend,'tasks':tasks,'count':len(tasks),
                          'status':'expert_registered_not_recovery_verified'}, indent=2))
        return 0
    if not args.output or not args.schedules or not 1 <= args.episodes <= 50:
        parser.error('output, schedules and episodes in [1,50] required')
    options = json.loads(args.adapter_options)
    configs = [(p.stem, validate_schedule(json.loads(p.read_text()))) for p in args.schedules]
    if len({name for name,_ in configs}) != len(configs):
        parser.error('Schedule basenames must be distinct')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'run_config.json', dict(backend=args.backend, tasks=tasks,
        schedules=dict(configs), episodes=args.episodes, seed_start=args.seed_start,
        max_steps=args.max_steps, adapter_options=options))
    started, rows = time.monotonic(), []
    for task in tasks:
        for name, schedule in configs:
            for index in range(args.episodes):
                adapter, seed = None, args.seed_start+index
                try:
                    kwargs = dict(options)
                    if args.backend == 'metaworld':
                        kwargs.setdefault('task_index', index)
                    adapter = load_adapter(args.backend, task, kwargs)
                    row = collect_triplet(adapter, seed, args.output/task/name/f'ep_{seed:04d}',
                                          budget=args.max_steps, schedule=schedule)
                except Exception as exc:
                    row = dict(task=task,seed=seed,status='error',qualified_correction=False,
                               error=repr(exc),traceback=traceback.format_exc())
                finally:
                    if adapter is not None:
                        adapter.close()
                row['schedule_name'] = name
                rows.append(row)
                write_json(args.output/'progress.json', {'results':rows})
                print(json.dumps(row), flush=True)
    summary = dict(backend=args.backend, task_count=len(tasks), attempts=len(rows),
                   successful_sources=sum(r.get('source_success',False) for r in rows),
                   qualified_corrections=sum(r['qualified_correction'] for r in rows),
                   errors=sum(r.get('status')=='error' for r in rows),
                   wall_seconds=time.monotonic()-started, results=rows)
    write_json(args.output/'summary.json', summary)
    return int(summary['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
