"""Strict, portable perturbation schedules. Anchors refer to the clean source.

Repeat gap counts normal actions, excluding injected actions. Event anchors are
resolved on the successful source, not guessed from a perturbed trajectory.
"""
import math


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def validate_schedule(schedule):
    if (not isinstance(schedule, dict) or set(schedule) != {'version', 'events'}
            or type(schedule['version']) is not int or schedule['version'] != 1):
        raise ValueError('Schedule requires version=1 and events')
    if not isinstance(schedule['events'], list) or not schedule['events']:
        raise ValueError('At least one perturbation event is required')
    for event in schedule['events']:
        if not isinstance(event, dict) or set(event) - {'at','type','steps','repeat','gap','strength','parameters'}:
            raise ValueError('Unknown perturbation fields')
        if not {'at','type','steps'} <= set(event) or not isinstance(event['type'], str):
            raise ValueError('Each event requires at, type and steps')
        at = event['at']
        if not isinstance(at, dict) or len(at) != 1 or not set(at) <= {'step','fraction','event'}:
            raise ValueError('at must contain exactly one of step, fraction, event')
        if 'step' in at:
            positive_int(at['step'], 'step')
        if 'fraction' in at:
            f = at['fraction']
            if isinstance(f, bool) or not isinstance(f, (int,float)) or not math.isfinite(f) or not 0 < f < 1:
                raise ValueError('fraction must be finite and in (0, 1)')
        if 'event' in at and (not isinstance(at['event'], str) or not at['event']):
            raise ValueError('event must be a nonempty source milestone name')
        positive_int(event['steps'], 'steps')
        positive_int(event.get('repeat', 1), 'repeat')
        positive_int(event.get('gap', 0), 'gap', 0)
        strength = event.get('strength', 1.)
        if isinstance(strength, bool) or not isinstance(strength, (int,float)) or not math.isfinite(strength) or not 0 < strength <= 1:
            raise ValueError('strength must be finite and in (0, 1]')
        if not isinstance(event.get('parameters', {}), dict):
            raise ValueError('parameters must be an object')
    return schedule


def resolve_schedule(schedule, source):
    validate_schedule(schedule)
    resolved = []
    for index, event in enumerate(schedule['events']):
        at = event['at']
        if 'step' in at:
            start = at['step']
        elif 'fraction' in at:
            start = max(1, int(len(source.actions) * at['fraction']))
        else:
            candidates = [i for i, labels in enumerate(source.events) if at['event'] in labels]
            if not candidates:
                raise ValueError(f"Source milestone absent: {at['event']}")
            start = candidates[0]
        for repetition in range(event.get('repeat', 1)):
            step = start + repetition * event.get('gap', 0)
            if not 1 <= step < len(source.actions):
                raise ValueError(f'Insertion step {step} must be inside successful source trajectory')
            resolved.append(dict(event, source_step=step, id=f'{index}:{repetition}'))
    return sorted(resolved, key=lambda item: item['source_step'])
