import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_corrections import LineWorld
from robot_stack.core import collect_triplet
from robot_stack.audit import audit
from robot_stack.schedule import validate_schedule, resolve_schedule
from robot_stack.perturbations import make_actions


def schedule(**kwargs):
    return {'version':1,'events':[dict(at={'step':1},type='native_default',steps=1,**kwargs)]}


class ScheduledContracts(unittest.TestCase):
    def test_multiple_errors_recover_and_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            result=collect_triplet(LineWorld(),4,root,budget=30,schedule=schedule(repeat=3,gap=1))
            self.assertTrue(result['qualified_correction'])
            self.assertEqual(result['applied_event_count'],3)
            self.assertEqual(result['branch_state_max_abs_errors'][0],0)
            self.assertEqual(audit(root)['qualified_corrections'],1)
            row=json.loads((root/'recovery_episode_result.json').read_text())
            self.assertEqual([e['action_range'] for e in row['perturbation_events']],[[1,2],[3,4],[5,6]])
            row['perturbation_events'][1]['action_range']=[2,3]
            (root/'recovery_episode_result.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):audit(root)

    def test_skipped_later_events_cannot_qualify(self):
        class Jump(LineWorld):
            def make_perturbation(self,event,rng):return [4.]*event['steps']
        config={'version':1,'events':[{'at':{'step':1},'type':'jump','steps':1},
                                     {'at':{'step':3},'type':'jump','steps':1}]}
        with tempfile.TemporaryDirectory() as d:
            row=collect_triplet(Jump(),1,Path(d)/'run',budget=20,schedule=config)
            self.assertFalse(row['qualified_correction'])
            self.assertEqual(row['applied_event_count'],1)
            audit(Path(d)/'run')

    def test_event_anchor_and_missing_event(self):
        class EventWorld(LineWorld):
            def events(self):return ['halfway'] if self.x >= 2 else []
        config=schedule();config['events'][0]['at']={'event':'halfway'}
        with tempfile.TemporaryDirectory() as d:
            row=collect_triplet(EventWorld(),0,Path(d)/'run',budget=20,schedule=config)
            self.assertEqual(row['branch_step'],2)
            audit(Path(d)/'run')
            with self.assertRaisesRegex(ValueError,'Source milestone absent'):
                collect_triplet(LineWorld(),0,Path(d)/'missing',budget=20,schedule=config)

    def test_bad_schedule_is_rejected(self):
        for field,value in [('steps',0),('repeat',-1),('gap',1.2),('strength',float('nan')),('typo',3)]:
            config=schedule();config['events'][0][field]=value
            with self.assertRaises(ValueError):validate_schedule(config)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'no recovery action budget'):
                collect_triplet(LineWorld(),1,Path(d)/'run',budget=6,schedule=schedule(repeat=8))

    def test_random_commands_reproducible_and_unknown_type_rejected(self):
        class Fake:
            backend='metaworld'
            def expert_action(self):return [0,0,0,-1]
        event={'type':'random_cartesian','steps':20,'strength':.2}
        a=make_actions(Fake(),event,np.random.default_rng(123))
        b=make_actions(Fake(),event,np.random.default_rng(123))
        self.assertEqual(a,b)
        self.assertLessEqual(np.abs(np.array(a)[:,:3]).max(),.2)
        with self.assertRaises(ValueError):make_actions(Fake(),dict(event,type='typo'),np.random.default_rng())
