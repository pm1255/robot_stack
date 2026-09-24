import importlib.util
import unittest
import numpy as np
from robot_stack.action_stream import ActionStream
from robot_stack.core import intervention_ended


@unittest.skipUnless(importlib.util.find_spec('greenlet'), 'optional native planner dependency')
class ActionStreamContracts(unittest.TestCase):
    def test_planner_cannot_execute_physics_or_reset_scene(self):
        class Env:
            steps=0
            resets=0
            unwrapped=None
            def reset(self,*args,**kwargs):self.resets+=1
            def step(self,action):self.steps+=1;return ('observation',self.steps)
        env=Env();received=[]
        def solve(proxy,**kwargs):
            received.append(proxy.reset(seed=kwargs['seed']))
            received.append(proxy.step([.1,.2]))
            return proxy.step([.3,.4])
        stream=ActionStream(env,solve,5,lambda:('current',{}))
        np.testing.assert_array_equal(stream.next_action(),[.1,.2])
        self.assertEqual((env.resets,env.steps),(0,0))
        result=env.step([.1,.2])
        np.testing.assert_array_equal(stream.next_action(result),[.3,.4])
        self.assertEqual(received,[('current',{}),('observation',1)])
        self.assertIsNone(stream.next_action(env.step([.3,.4])))
        self.assertEqual((env.resets,env.steps),(0,2))
        stream.close()

    def test_late_reset_is_rejected_and_cancel_unwinds(self):
        cleaned=[]
        def bad(proxy,**kwargs):
            proxy.step([1])
            proxy.reset()
        stream=ActionStream(object(),bad,1,lambda:(None,{}))
        stream.next_action()
        with self.assertRaisesRegex(RuntimeError,'non-setup reset'):stream.next_action(None)
        def infinite(proxy,**kwargs):
            try:
                while True:proxy.step([1])
            finally:cleaned.append(True)
        stream=ActionStream(object(),infinite,1,lambda:(None,{}))
        stream.next_action();stream.close()
        self.assertEqual(cleaned,[True])


class ReplanningContracts(unittest.TestCase):
    def test_replanning_cannot_modify_physics(self):
        class Bad:
            x=0
            def state(self):return [self.x]
            def after_intervention(self):self.x=1
        with self.assertRaisesRegex(ValueError,'changed simulator state'):
            intervention_ended(Bad())
