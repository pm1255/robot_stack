"""Paired multi-intervention collection, with explicit causal limits.

The first intervention has identical starting states. Later interventions use
identical commands at identical nominal source ticks, but branch states can
differ because feedback has already acted. Every failed/skipped event is kept.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import Episode, jsonable, rollout, save_episode, write_json, intervention_ended
from .schedule import validate_schedule, resolve_schedule
from .perturbations import make_actions


def collect_scheduled(adapter, seed, folder, *, schedule, budget=500,
                      error_threshold=.03, replay_tolerance=1e-7):
    validate_schedule(schedule)
    if budget < 3 or not np.isfinite(error_threshold) or error_threshold <= 0:
        raise ValueError('Positive error threshold and action budget >= 3 required')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    adapter.reset(seed)
    source = Episode.start(adapter)
    split = hashlib.sha256((adapter.backend+'/'+adapter.task).encode()+source.states[0].tobytes()).hexdigest()[:20]
    metadata = dict(adapter.metadata, backend=adapter.backend, task=adapter.task, seed=seed,
                    max_steps=budget, protocol='scheduled_interventions.v1', schedule=schedule,
                    oracle_policy=True, standard_benchmark_score=False, split_group=split,
                    replay_tolerance=replay_tolerance,
                    state_reconstruction='same_seed_real_action_prefix')
    group = hashlib.sha256(json.dumps(jsonable(metadata), sort_keys=True).encode()).hexdigest()[:20]
    metadata['source_demo_id'] = group
    rollout(adapter, source, budget, 'expert')
    source_record = save_episode(folder, 'source', source, dict(metadata, label='source'))
    summary = dict(backend=adapter.backend, task=adapter.task, seed=seed, source_demo_id=group,
                   split_group=split, source_success=source.success, qualified_correction=False,
                   protocol='scheduled_interventions.v1')
    if not source.success or len(source.actions) < 2:
        summary['reason'] = 'No successful source with a nonempty prefix'
        write_json(folder/'correction.json', summary)
        return summary
    events = resolve_schedule(schedule, source)
    if events[-1]['source_step'] + sum(e['steps'] for e in events) >= budget:
        raise ValueError('Scheduled injections leave no recovery action budget')
    first = events[0]['source_step']
    rows, trajectories, frozen = {}, {}, {}
    for label in ('perturbed', 'recovery'):
        adapter.reset(seed)
        episode = Episode.start(adapter)
        prefix_error = float(np.max(np.abs(episode.states[0]-source.states[0])))
        for i, action in enumerate(source.actions[:first]):
            episode.step(adapter, action, 'source_prefix', budget)
            prefix_error = max(prefix_error, float(np.max(np.abs(episode.states[-1]-source.states[i+1]))))
        if prefix_error > replay_tolerance:
            raise ValueError(f'Prefix replay differs from source: {prefix_error}')
        tick, cursor, logs = first, 0, []
        while len(episode.actions) < budget and not adapter.terminal():
            if label == 'recovery' and episode.success:
                break
            while cursor < len(events) and events[cursor]['source_step'] == tick:
                event = events[cursor]
                if cursor not in frozen:
                    # Independent streams keep each event reproducible under adapter use.
                    rng = np.random.default_rng(np.random.SeedSequence([seed, cursor]))
                    frozen[cursor] = jsonable(make_actions(adapter, event, rng))
                    if len(frozen[cursor]) != event['steps']:
                        raise ValueError('Adapter changed requested perturbation action count')
                start = len(episode.actions)
                before = episode.features[-1]
                for action in frozen[cursor]:
                    if adapter.terminal() or len(episode.actions) >= budget:
                        break
                    episode.step(adapter, action, 'perturbation', budget)
                end = len(episode.actions)
                displacement = float(np.linalg.norm(episode.features[-1]-before))
                logs.append(dict(id=event['id'], type=event['type'], source_step=tick,
                                 action_range=[start,end], requested_steps=event['steps'],
                                 complete=end-start == event['steps'],
                                 feature_displacement=displacement, threshold=error_threshold,
                                 induced_error=displacement >= error_threshold and not adapter.success()))
                intervention_ended(adapter)
                cursor += 1
                if adapter.terminal() or len(episode.actions) >= budget:
                    break
            if adapter.terminal() or len(episode.actions) >= budget:
                break
            if label == 'perturbed':
                if tick >= len(source.actions):
                    break
                action, phase = source.actions[tick], 'open_loop_continuation'
            else:
                if episode.success:
                    break
                action, phase = adapter.expert_action(), 'recovery'
            episode.step(adapter, action, phase, budget)
            tick += 1
        all_applied = len(logs) == len(events) and all(e['complete'] for e in logs)
        all_induced = all_applied and all(e['induced_error'] for e in logs)
        meta = dict(metadata, label=label, source_trajectory='source.hdf5',
                    source_sha256=source_record['trajectory_sha256'], branch_step=first,
                    resolved_schedule=events, perturbation_events=logs,
                    requested_event_count=len(events), applied_event_count=sum(e['complete'] for e in logs),
                    all_events_applied=all_applied, induced_error=all_induced,
                    prefix_max_abs_error=prefix_error,
                    pairing='identical_first_error; later_commands_and_nominal_ticks_matched')
        rows[label] = save_episode(folder, label, episode, meta)
        trajectories[label] = episode
    control, recovery = rows['perturbed'], rows['recovery']
    deltas = []
    for a, b in zip(control['perturbation_events'], recovery['perturbation_events']):
        deltas.append(float(np.max(np.abs(trajectories['perturbed'].states[a['action_range'][1]]
                                         - trajectories['recovery'].states[b['action_range'][1]]))))
    qualified = (control['induced_error'] and recovery['induced_error'] and not control['success']
                 and recovery['success'] and bool(deltas) and deltas[0] <= replay_tolerance)
    summary.update(qualified_correction=bool(qualified), perturbed_success=control['success'],
                   recovery_success=recovery['success'], induced_error=recovery['induced_error'],
                   requested_event_count=len(events), applied_event_count=recovery['applied_event_count'],
                   branch_state_max_abs_errors=deltas, branch_step=first,
                   recovery_steps=recovery['steps'])
    write_json(folder/'correction.json', summary)
    return summary
