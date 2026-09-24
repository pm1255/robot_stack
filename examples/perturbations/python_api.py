"""Run after installing robot-stack[metaworld]; output must not already exist."""
from pathlib import Path
from robot_stack.collect import load_adapter
from robot_stack.core import collect_triplet

adapter = load_adapter('metaworld', 'pick-place-v3', {'task_index': 0})
try:
    result = collect_triplet(adapter, 600, Path('outputs/python-example'), budget=500,
        schedule={'version': 1, 'events': [
            {'at': {'fraction': .2}, 'type': 'cartesian_offset', 'steps': 8,
             'strength': .8, 'repeat': 3, 'gap': 8}
        ]})
    print(result)  # qualification comes from evidence, never presumed from config
finally:
    adapter.close()
