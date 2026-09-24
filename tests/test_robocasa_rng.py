import random
import unittest
import numpy as np
from robot_stack.adapters.robocasa_rng import seeded_scene_initialization


class SceneRandomnessContracts(unittest.TestCase):
    def test_unseeded_children_repeat_and_explicit_seeds_keep_semantics(self):
        original=np.random.default_rng
        expected=original(8).random(4)
        samples=[]
        for _ in range(2):
            with seeded_scene_initialization(1100):
                samples.append([np.random.default_rng().random(4),np.random.default_rng().random(4)])
                np.testing.assert_array_equal(np.random.default_rng(8).random(4),expected)
            self.assertIs(np.random.default_rng,original)
        np.testing.assert_array_equal(samples[0],samples[1])
        self.assertFalse(np.array_equal(samples[0][0],samples[0][1]))

    def test_failed_initialization_restores_global_random_state(self):
        original=np.random.default_rng
        np_state=np.random.get_state();py_state=random.getstate()
        with self.assertRaisesRegex(RuntimeError,'fixture failed'):
            with seeded_scene_initialization(1100):
                np.random.random();random.random()
                raise RuntimeError('fixture failed')
        self.assertIs(np.random.default_rng,original)
        after=np.random.get_state()
        self.assertEqual(np_state[0],after[0])
        np.testing.assert_array_equal(np_state[1],after[1])
        self.assertEqual(np_state[2:],after[2:])
        self.assertEqual(random.getstate(),py_state)

    def test_counter_geometry_order_is_stable_and_wrapper_is_scoped(self):
        left={'offset':[-1,0,1], 'size':[.5,.6]}
        right={'offset':[1,0,1], 'size':[.5,.6]}
        class Counter:
            flipped=False
            def get_reset_regions(self):
                self.flipped=not self.flipped
                values=[left,right] if self.flipped else [right,left]
                return dict(enumerate(values))
        original=Counter.get_reset_regions
        counter=Counter()
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with seeded_scene_initialization(7,Counter):
                self.assertEqual(counter.get_reset_regions(),counter.get_reset_regions())
                self.assertEqual(list(counter.get_reset_regions().values()),[left,right])
                raise RuntimeError('stop')
        self.assertIs(Counter.get_reset_regions,original)
