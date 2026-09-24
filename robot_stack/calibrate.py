"""Per-task clean baselines, bounded severity search and held-out validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from .core import write_json
from .schedule import validate_schedule
from .suite import run_isolated


def scale_schedule(schedule, scale, axis='strength'):
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Scales must be finite and in (0, 1]')
    result = deepcopy(validate_schedule(schedule))
    if axis not in ('strength', 'steps'):
        raise ValueError('Axis must be strength or steps')
    for event in result['events']:
        if axis == 'strength':
            event['strength'] = event.get('strength', 1.) * scale
        else:
            event['steps'] = max(1, math.ceil(event['steps'] * scale))
    return validate_schedule(result)


def metrics(rows):
    n = len(rows)
    source = sum(r.get('source_success') is True for r in rows)
    completed = [r for r in rows if 'recovery_success' in r]
    failed_controls = [r for r in completed if r.get('induced_error') is True
                       and r.get('perturbed_success') is False]
    qualified = sum(r.get('qualified_correction') is True for r in rows)
    return dict(attempts=n, successful_sources=source,
                source_success_rate=source/n if n else None,
                completed_verdicts=len(completed), failed_controls=len(failed_controls),
                recovery_successes=sum(r.get('recovery_success') is True for r in failed_controls),
                recovery_success_rate=(sum(r.get('recovery_success') is True for r in failed_controls)
                                       / len(failed_controls) if failed_controls else None),
                qualified_corrections=qualified, qualified_yield=qualified/n if n else None,
                errors=sum(r.get('status') == 'error' for r in rows))


def select_profile(levels, min_source_rate, min_recovery_rate):
    # Lower amplitude is preferred only if it still produces a qualified error.
    eligible = [level for level in levels
                if (level['metrics']['source_success_rate'] or 0) >= min_source_rate
                and level['metrics']['qualified_corrections'] > 0
                and (level['metrics']['recovery_success_rate'] or 0) >= min_recovery_rate]
    return min(eligible, key=lambda level: level['scale']) if eligible else None


def validation_status(rows, training_groups, *, min_source_rate, min_recovery_rate,
                      min_episodes):
    groups = [r.get('split_group') for r in rows]
    if any(group in training_groups for group in groups if group):
        return 'scene_overlap_with_calibration'
    if len(rows) < min_episodes or len(set(g for g in groups if g)) < min_episodes:
        return 'insufficient_validation_scenes'
    m = metrics(rows)
    if (m['source_success_rate'] or 0) < min_source_rate:
        return 'baseline_below_target_on_validation'
    if not m['qualified_corrections']:
        return 'no_qualified_error_on_validation'
    if (m['recovery_success_rate'] or 0) < min_recovery_rate:
        return 'recovery_below_target_on_validation'
    return 'passed_configured_validation'


def run(args):
    options = json.loads(args.adapter_options)
    if not isinstance(options, dict):
        raise ValueError('adapter-options must be an object')
    if args.inventory:
        inventory = json.loads(args.inventory.read_text())
        if inventory['backend'] != args.backend:
            raise ValueError('Inventory backend mismatch')
    elif args.tasks == ['all']:
        from .inventory import discover
        inventory = discover(args.backend, root=options.get('root'))
    else:
        inventory = dict(backend=args.backend, tasks=[dict(task=t, expert_available=True)
                                                       for t in args.tasks])
    registered = {row['task']: row for row in inventory['tasks']}
    tasks = sorted(registered) if args.tasks == ['all'] else args.tasks
    if len(set(tasks)) != len(tasks) or any(t not in registered for t in tasks):
        raise ValueError('Tasks must be unique and present in the supplied inventory')
    if any(not isinstance(t, str) or t in ('.', '..') or Path(t).name != t for t in tasks):
        raise ValueError('Task identifiers must be single path components')
    train_seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    validation_seeds = list(range(args.validation_seed_start,
                                  args.validation_seed_start + args.validation_episodes))
    if set(train_seeds) & set(validation_seeds):
        raise ValueError('Calibration and validation seeds must be disjoint')
    if args.backend == 'metaworld' and args.episodes + args.validation_episodes > 50:
        raise ValueError('MetaWorld requires disjoint task indices within its 50-instance split')
    base = validate_schedule(json.loads(args.schedule.read_text()))
    if args.backend == 'ai2thor' and args.axis == 'strength' and any(s != 1 for s in args.scales):
        raise ValueError('AI2-THOR discrete actions require --axis steps')
    levels = [dict(name=f'level_{i:02d}', scale=s, schedule=scale_schedule(base, s, args.axis))
              for i, s in enumerate(sorted(set(args.scales)))]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(backend=args.backend, tasks=tasks, train_seeds=train_seeds,
                  validation_seeds=validation_seeds, levels=levels, axis=args.axis,
                  max_steps=args.max_steps, adapter_options=options, workers=args.workers,
                  case_timeout=args.case_timeout, min_source_rate=args.min_source_rate,
                  min_recovery_rate=args.min_recovery_rate,
                  min_validation_episodes=args.min_validation_episodes)
    write_json(output/'run_config.json', config)
    report = dict(schema='robot_stack.task_calibration.v1', backend=args.backend,
                  semantics='Empirical task/severity profiles, not universal benchmark guarantees. '
                            'Selection uses calibration only; all attempts including failures are retained.',
                  config=config, tasks={}, results=[])
    for task in tasks:
        report['tasks'][task] = dict(task=task, status='pending_baseline', levels=[])
        if not registered[task].get('expert_available') and not options.get('adapter_factory'):
            report['tasks'][task]['status'] = 'missing_expert'
    started = time.monotonic()

    def case(spec):
        task, stage, seed, index, level = spec
        name = level['name'] if level else 'clean'
        folder = output/task/stage/name/f'case_{seed:04d}'
        kwargs = dict(options)
        if args.backend == 'metaworld':
            kwargs['task_index'] = index
        try:
            row = run_isolated(args.backend, task, seed, folder, budget=args.max_steps,
                schedule=level['schedule'] if level else None, options=kwargs,
                timeout=args.case_timeout, source_only=level is None)
            source = folder/f'ep_{seed:04d}'/'source_episode_result.json'
            if source.exists():
                row['split_group'] = json.loads(source.read_text()).get('split_group')
        except Exception as exc:
            row = dict(backend=args.backend, task=task, seed=seed, status='error',
                       error=repr(exc), qualified_correction=False)
        row.update(stage=stage, schedule_name=name, scale=level['scale'] if level else None,
                   evidence=str(folder.relative_to(output)))
        return row

    def batch(specs):
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(case, spec) for spec in specs]):
                row = future.result(); rows.append(row); report['results'].append(row)
                write_json(output/'progress.json', report)
                print(json.dumps(row), flush=True)
        return sorted(rows, key=lambda r: (r['task'], r['schedule_name'], r['seed']))

    active = [t for t in tasks if report['tasks'][t]['status'] == 'pending_baseline']
    baseline = batch([(t, 'baseline', seed, i, None)
                      for t in active for i, seed in enumerate(train_seeds)])
    eligible = []
    for task in active:
        profile = report['tasks'][task]
        profile['baseline'] = metrics([r for r in baseline if r['task'] == task])
        if (profile['baseline']['source_success_rate'] or 0) >= args.min_source_rate:
            eligible.append(task); profile['status'] = 'calibrating'
        else:
            profile['status'] = 'baseline_below_target'
    calibrated = batch([(t, 'calibration', seed, i, level) for t in eligible
                        for level in levels for i, seed in enumerate(train_seeds)])
    selected = {}
    for task in eligible:
        profile = report['tasks'][task]
        profile['levels'] = [dict(level, metrics=metrics([r for r in calibrated
            if r['task'] == task and r['schedule_name'] == level['name']])) for level in levels]
        choice = select_profile(profile['levels'], args.min_source_rate, args.min_recovery_rate)
        if choice:
            selected[task] = choice; profile['selected'] = choice; profile['status'] = 'pending_validation'
        else:
            profile['status'] = 'no_qualified_profile'
    # Freeze chosen schedules before looking at any held-out outcomes.
    write_json(output/'selected_profiles.json', selected)
    (output/'selected').mkdir()
    for task, level in selected.items():
        write_json(output/'selected'/(task+'.json'), level['schedule'])
    validated = batch([(t, 'validation', seed, args.episodes+i, level)
                       for t, level in selected.items() for i, seed in enumerate(validation_seeds)])
    for task in selected:
        rows = [r for r in validated if r['task'] == task]
        training_groups = {r['split_group'] for r in baseline+calibrated
                           if r['task'] == task and r.get('split_group')}
        profile = report['tasks'][task]
        profile['validation'] = metrics(rows)
        profile['status'] = validation_status(rows, training_groups,
            min_source_rate=args.min_source_rate, min_recovery_rate=args.min_recovery_rate,
            min_episodes=args.min_validation_episodes)
    report['wall_seconds'] = time.monotonic()-started
    report['results'].sort(key=lambda r: (r['task'], r['stage'], r['schedule_name'], r['seed']))
    report['task_count'] = len(tasks)
    report['passed_tasks'] = sum(p['status'] == 'passed_configured_validation'
                                 for p in report['tasks'].values())
    write_json(output/'summary.json', report)
    print(json.dumps(dict(task_count=len(tasks), passed_tasks=report['passed_tasks'],
                         statuses={t:p['status'] for t,p in report['tasks'].items()})), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', required=True)
    p.add_argument('--tasks', nargs='+', default=['all'])
    p.add_argument('--inventory', type=Path)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--scales', type=float, nargs='+', default=[.125, .25, .5, 1.])
    p.add_argument('--axis', choices=['strength', 'steps'], default='strength')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=800)
    p.add_argument('--validation-episodes', type=int, default=5)
    p.add_argument('--validation-seed-start', type=int, default=1800)
    p.add_argument('--min-validation-episodes', type=int, default=5)
    p.add_argument('--min-source-rate', type=float, default=.8)
    p.add_argument('--min-recovery-rate', type=float, default=.5)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--case-timeout', type=int, default=180)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--adapter-options', default='{}')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for name in ('episodes','validation_episodes','min_validation_episodes','max_steps','case_timeout','workers'):
        if getattr(args, name) < 1:
            p.error(name+' must be positive')
    if min(args.seed_start, args.validation_seed_start) < 0:
        p.error('Seeds must be nonnegative')
    if not 0 < args.min_source_rate <= 1 or not 0 < args.min_recovery_rate <= 1:
        p.error('Rate thresholds must be in (0, 1]')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
