"""Simulator-independent paired correction collection.

Adapters own dynamics and task oracles. This module never sets simulator state:
branches reconstruct their prefix with real actions, then continue from the
perturbed state. All injected and recovery actions count toward the same budget.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Adapter(Protocol):
    """One initialized task. reset(seed) must reproduce scene and policy state."""
    backend: str
    task: str
    metadata: dict
    def reset(self, seed: int) -> None: ...
    def state(self) -> np.ndarray: ...
    def error_features(self) -> np.ndarray: ...
    def step(self, action: Any) -> None: ...
    def expert_action(self) -> Any: ...
    def perturbation_actions(self, steps: int) -> list: ...
    def success(self) -> bool: ...
    def terminal(self) -> bool: ...
    def close(self) -> None: ...


@dataclass
class Episode:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    success_flags: list = field(default_factory=list)
    features: list = field(default_factory=list)

    @classmethod
    def start(cls, adapter):
        return cls(states=[checked(adapter.state())], features=[checked(adapter.error_features())])

    @property
    def success(self):
        return bool(self.success_flags and self.success_flags[-1])

    def step(self, adapter, action, phase, budget):
        if len(self.actions) >= budget:
            raise ValueError('Action budget exhausted')
        saved_action = jsonable(action)  # some native controllers mutate arrays in place
        adapter.step(action)
        self.actions.append(saved_action)
        self.phases.append(phase)
        self.states.append(checked(adapter.state()))
        self.features.append(checked(adapter.error_features()))
        self.success_flags.append(bool(adapter.success()))


def checked(value):
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError('Empty or non-finite simulator state/features')
    return result


def rollout(adapter, episode, budget, phase):
    while len(episode.actions) < budget and not episode.success and not adapter.terminal():
        episode.step(adapter, adapter.expert_action(), phase, budget)


def save_episode(folder, name, episode, metadata):
    import h5py
    if len(episode.states) != len(episode.actions) + 1:
        raise ValueError('State/action alignment violation')
    path = folder / (name + '.hdf5')
    strings = h5py.string_dtype('utf-8')
    with h5py.File(path, 'x') as f:
        f.attrs['schema'] = 'robot_stack.correction.v1'
        f.attrs['metadata'] = json.dumps(jsonable(metadata), allow_nan=False)
        f.attrs['alignment'] = 'states/features: T+1; actions/phases/success: T'
        f.create_dataset('states', data=np.asarray(episode.states), compression='gzip')
        f.create_dataset('error_features', data=np.asarray(episode.features), compression='gzip')
        f.create_dataset('actions_json', data=[json.dumps(a, allow_nan=False) for a in episode.actions], dtype=strings)
        f.create_dataset('phases', data=episode.phases, dtype=strings)
        f.create_dataset('success', data=episode.success_flags, dtype=np.bool_)
    record = dict(metadata, success=episode.success, task_success=episode.success,
                  control_success=episode.success, execution_success=True,
                  validation_version=1, debug=False,
                  status='success' if episode.success else 'task_failed',
                  steps=len(episode.actions), trajectory_file=path.name,
                  trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(folder / (name + '_episode_result.json'), record)
    return record


def collect_triplet(adapter, seed, folder, *, budget=500, branch_fraction=.35,
                    perturb_steps=15, error_threshold=.03, replay_tolerance=1e-7):
    if not 0 < branch_fraction < 1 or perturb_steps < 1 or budget < 3:
        raise ValueError('Invalid branch or action budget')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    adapter.reset(seed)
    base_meta = dict(adapter.metadata, backend=adapter.backend, task=adapter.task, seed=seed,
                     max_steps=budget, state_reconstruction='same_seed_real_action_prefix',
                     replay_tolerance=replay_tolerance,
                     oracle_policy=True, standard_benchmark_score=False)
    group = hashlib.sha256(json.dumps(jsonable(base_meta), sort_keys=True).encode()).hexdigest()[:20]
    source = Episode.start(adapter)
    # Different reset seeds can map to the same native scene/task instance.
    # Conservatively keep identical initial states for a task in one split.
    split_group = hashlib.sha256((adapter.backend + '/' + adapter.task).encode()
                                 + source.states[0].tobytes()).hexdigest()[:20]
    base_meta.update(source_demo_id=group, split_group=split_group)
    rollout(adapter, source, budget, 'expert')
    source_record = save_episode(folder, 'source', source, dict(base_meta, label='source'))
    summary = dict(backend=adapter.backend, task=adapter.task, seed=seed, source_demo_id=group,
                   source_success=source.success, qualified_correction=False)
    if not source.success or len(source.actions) < 2:
        summary['reason'] = 'No successful source with a nonempty prefix'
        write_json(folder / 'correction.json', summary)
        return summary
    branch = max(1, min(len(source.actions) - 1, int(len(source.actions) * branch_fraction)))
    if branch + perturb_steps >= budget:
        raise ValueError('Perturbation leaves no recovery budget')
    branch_records, injected_states = {}, []
    injection = None
    max_prefix_error = 0.0
    for label in ['perturbed', 'recovery']:
        adapter.reset(seed)
        episode = Episode.start(adapter)
        errors = [float(np.max(np.abs(episode.states[0] - source.states[0])))]
        for index, action in enumerate(source.actions[:branch]):
            episode.step(adapter, action, 'source_prefix', budget)
            errors.append(float(np.max(np.abs(episode.states[-1] - source.states[index + 1]))))
        prefix_error = max(errors)
        max_prefix_error = max(max_prefix_error, prefix_error)
        if prefix_error > replay_tolerance:
            raise ValueError(f'Prefix replay differs from source: {prefix_error}')
        if injection is None:
            injection = [jsonable(a) for a in adapter.perturbation_actions(perturb_steps)]
            if len(injection) != perturb_steps:
                raise ValueError('Adapter changed perturbation action count')
        for action in injection:
            if adapter.terminal():
                break
            episode.step(adapter, action, 'perturbation', budget)
        error_end = len(episode.actions)
        displacement = float(np.linalg.norm(episode.features[-1] - source.features[branch]))
        induced_error = displacement >= error_threshold and not adapter.success()
        injected_states.append(episode.states[-1])
        if label == 'perturbed':
            for action in source.actions[branch:]:
                if len(episode.actions) >= budget or adapter.terminal():
                    break
                episode.step(adapter, action, 'open_loop_continuation', budget)
        else:
            rollout(adapter, episode, budget, 'recovery')
        meta = dict(base_meta, label=label, source_trajectory='source.hdf5',
                    source_sha256=source_record['trajectory_sha256'], branch_step=branch,
                    perturbation_action_range=[branch, error_end],
                    recovery_action_range=[error_end, len(episode.actions)] if label == 'recovery' else None,
                    perturbation={'type': adapter.metadata['perturbation_type'],
                                  'requested_steps': perturb_steps, 'seed': seed,
                                  'feature_displacement': displacement, 'threshold': error_threshold},
                    induced_error=induced_error, prefix_max_abs_error=prefix_error)
        branch_records[label] = save_episode(folder, label, episode, meta)
    same_error = float(np.max(np.abs(injected_states[0] - injected_states[1])))
    control, recovery = branch_records['perturbed'], branch_records['recovery']
    qualified = (control['induced_error'] and recovery['induced_error'] and not control['success']
                 and recovery['success'] and same_error <= replay_tolerance)
    summary.update(branch_step=branch, perturbed_success=control['success'],
                   recovery_success=recovery['success'], induced_error=recovery['induced_error'],
                   prefix_max_abs_error=max_prefix_error, branch_state_max_abs_error=same_error,
                   qualified_correction=bool(qualified), split_group=split_group,
                   recovery_action_range=recovery['recovery_action_range'])
    write_json(folder / 'correction.json', summary)
    return summary
