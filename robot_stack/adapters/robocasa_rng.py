"""Scoped initialization compatibility for native RoboCasa scene sampling.

Run native environments in separate processes. NumPy's factory is temporarily
wrapped while constructing a scene and always restored, including on errors.
Explicitly seeded generator calls retain their original behavior. Counter reset
regions are sorted by geometry because upstream deduplicates XML elements with
a set (identity-dependent order). No installed simulator files are modified.
"""
from contextlib import contextmanager
import random
from threading import RLock
import numpy as np

_lock = RLock()


def ordered_counter_regions(regions):
    """Counter.get_reset_regions uses set(XML elements), ordered by identity.

    Canonicalize by geometry, preserving every region and its dimensions. Native
    counter keys are generated geom_N labels, so regenerate them in this order.
    """
    def key(region):
        return tuple(float(x) for x in region['offset']) + tuple(float(x) for x in region['size'])
    return {f'geom_{i}': region for i, region in enumerate(sorted(regions.values(), key=key))}


@contextmanager
def seeded_scene_initialization(seed, counter_class=None):
    with _lock:
        original_regions = counter_class.get_reset_regions if counter_class is not None else None
        factory = np.random.default_rng
        numpy_state, python_state = np.random.get_state(), random.getstate()
        sequence = np.random.SeedSequence([int(seed), 0x524353])

        def deterministic_factory(seed=None):
            return factory(sequence.spawn(1)[0] if seed is None else seed)

        np.random.seed(seed)
        random.seed(seed)
        np.random.default_rng = deterministic_factory
        if counter_class is not None:
            def stable_regions(self, *args, **kwargs):
                return ordered_counter_regions(original_regions(self, *args, **kwargs))
            counter_class.get_reset_regions = stable_regions
        try:
            yield
        finally:
            if counter_class is not None:
                counter_class.get_reset_regions = original_regions
            np.random.default_rng = factory
            np.random.set_state(numpy_state)
            random.setstate(python_state)
