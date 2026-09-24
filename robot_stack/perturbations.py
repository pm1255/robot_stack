"""Native action interventions; no state teleportation or fake grasp attachment."""
import numpy as np


CATALOG = {
    'metaworld': ['cartesian_offset','random_cartesian','gripper_open','gripper_close','action_reverse','action_hold'],
    'robocasa': ['base_yaw','base_translation','base_reverse','base_hold'],
    'ai2thor': ['wrong_heading','backtrack','lateral_drift','navigation_hold'],
}


def make_actions(adapter, event, rng):
    kind, steps = event['type'], event['steps']
    strength, params = event.get('strength', 1.), event.get('parameters', {})
    if kind == 'native_default':
        if params or strength != 1:
            raise ValueError('native_default accepts no parameters or strength scaling')
        return adapter.perturbation_actions(steps)
    custom = getattr(adapter, 'make_perturbation', None)
    if custom:
        return custom(event, rng)
    if kind not in CATALOG.get(adapter.backend, []):
        raise ValueError(f'Unsupported perturbation {kind} for {adapter.backend}')
    allowed = {'direction','gripper'} if kind in {'cartesian_offset','base_translation'} else set()
    if set(params) - allowed:
        raise ValueError(f'Unsupported parameters for {kind}: {sorted(set(params)-allowed)}')
    if adapter.backend == 'metaworld':
        expert = np.asarray(adapter.expert_action(), dtype=float).copy()
        if kind == 'random_cartesian':
            actions = np.tile(expert, (steps,1))
            actions[:,:3] = rng.uniform(-strength, strength, (steps,3))
            return actions.tolist()
        action = expert.copy()
        if kind == 'cartesian_offset':
            direction = np.asarray(params.get('direction', [1,-1,1]), dtype=float)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.max(np.abs(direction)) > 1:
                raise ValueError('direction requires three finite normalized values in [-1,1]')
            action[:3] = strength * direction
            action[3] = params.get('gripper', -1.)
            if not np.isfinite(action[3]) or not -1 <= action[3] <= 1:
                raise ValueError('gripper must be in [-1,1]')
        elif kind in {'gripper_open','gripper_close'}:
            action[:3] = 0
            action[3] = (-1 if kind == 'gripper_open' else 1) * strength
        elif kind == 'action_reverse':
            action[:3] *= -strength
        else:
            action[:3] = 0
        return [action.tolist() for _ in range(steps)]
    if adapter.backend == 'robocasa':
        velocity = np.zeros(3)
        if kind == 'base_yaw':
            velocity[2] = strength
        elif kind in {'base_translation','base_reverse'}:
            direction = np.asarray(params.get('direction', [1,0]), dtype=float)
            if direction.shape != (2,) or not np.isfinite(direction).all() or np.max(np.abs(direction)) > 1:
                raise ValueError('direction requires two finite normalized values')
            velocity[:2] = direction * strength * (-1 if kind == 'base_reverse' else 1)
        return [adapter.action(velocity).tolist() for _ in range(steps)]
    if strength != 1:
        raise ValueError('Discrete grid navigation requires strength=1; vary steps instead')
    action = {'wrong_heading': {'action':'RotateRight','degrees':90},
              'backtrack': {'action':'MoveBack','moveMagnitude':adapter.grid},
              'lateral_drift': {'action':'MoveRight','moveMagnitude':adapter.grid},
              'navigation_hold': {'action':'Pass'}}[kind]
    return [dict(action) for _ in range(steps)]
