"""Enumerate native MetaWorld experts and run a bounded task × schedule sweep."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import time
import traceback
from .collect import load_adapter
from .core import collect_triplet, write_json
from .schedule import validate_schedule


def run_isolated(backend, task, seed, folder, *, budget, schedule, options, timeout, source_only=False):
    """Contain native crashes/timeouts so the remaining task sweep can finish."""
    folder = Path(folder).resolve()
    folder.parent.mkdir(parents=True, exist_ok=True)
    spec = folder.with_name(folder.name + '_schedule.json')
    log = folder.with_name(folder.name + '_worker.log')
    if not source_only:
        write_json(spec, schedule)
    command = [sys.executable, '-m', 'robot_stack.collect', '--backend', backend,
        '--task', task, '--episodes', '1', '--seed-start', str(seed),
        '--max-steps', str(budget),
        '--adapter-options', json.dumps(options), '--output', str(folder)]
    command += ['--source-only'] if source_only else ['--schedule', str(spec)]
    with log.open('wb') as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=timeout)
            error = f'Native worker exited with code {process.returncode}'
        except subprocess.TimeoutExpired:
            error = f'Native worker exceeded {timeout} seconds'
    summary = folder/'summary.json'
    if summary.exists():
        row = json.loads(summary.read_text(encoding='utf-8'))['results'][0]
    else:
        row = dict(backend=backend, task=task, seed=seed, status='error',
                   qualified_correction=False, error=error)
    source = folder/f'ep_{seed:04d}'/'source_episode_result.json'
    if 'source_success' not in row and source.exists():
        row['source_success'] = json.loads(source.read_text(encoding='utf-8')).get('success') is True
    row.setdefault('backend', backend)
    row.setdefault('task', task)
    row.setdefault('seed', seed)
    row['worker_log'] = log.name
    row['isolated_worker'] = True
    return row


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
    parser.add_argument('--isolate', action='store_true', help='Run each attempt in a separate process (automatic for ManiSkill)')
    parser.add_argument('--case-timeout', type=int, default=900, help='Seconds per isolated worker')
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.case_timeout < 1:
        parser.error('case-timeout must be positive')
    isolated = args.isolate or args.backend == 'maniskill'
    tasks = args.tasks
    if tasks == ['all']:
        if args.backend == 'metaworld':
            from benchmark_collector.metaworld import discover_tasks
        elif args.backend == 'maniskill':
            from .adapters.maniskill import discover_tasks
        else:
            parser.error('Automatic expert discovery supports MetaWorld and ManiSkill; other backends need explicit tasks')
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
        max_steps=args.max_steps, adapter_options=options,
        isolated_workers=isolated, case_timeout=args.case_timeout if isolated else None))
    started, rows = time.monotonic(), []
    for task in tasks:
        for name, schedule in configs:
            for index in range(args.episodes):
                adapter, seed = None, args.seed_start+index
                try:
                    kwargs = dict(options)
                    if args.backend == 'metaworld':
                        kwargs.setdefault('task_index', index)
                    if isolated:
                        row = run_isolated(args.backend, task, seed, args.output/task/name/f'case_{seed:04d}',
                            budget=args.max_steps, schedule=schedule, options=kwargs, timeout=args.case_timeout)
                    else:
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
