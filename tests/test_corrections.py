"""Contracts that prevent a successful endpoint being mislabelled as correction."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from robot_stack.core import Episode, collect_triplet
from robot_stack.adapters.robodojo import RoboDojoAdapter


class LineWorld:
    backend='test'; task='line'
    metadata={'perturbation_type':'backwards_action'}
    def reset(self,seed): self.x=0.; self.count=0
    def state(self): return np.array([self.x,self.count])
    def error_features(self): return np.array([self.x])
    def step(self,a): self.x+=float(a);self.count+=1
    def expert_action(self): return min(1.,5-self.x)
    def perturbation_actions(self,steps): return [-1.]*steps
    def success(self): return abs(self.x-5)<1e-9
    def terminal(self): return False
    def close(self): pass


class CorrectionContracts(unittest.TestCase):
    def test_real_error_and_recovery_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run'
            result=collect_triplet(LineWorld(),4,root,budget=20,perturb_steps=2)
            self.assertTrue(result['qualified_correction'])
            self.assertFalse(result['perturbed_success'])
            self.assertEqual(result['branch_state_max_abs_error'],0)
            import h5py
            with h5py.File(root/'recovery.hdf5') as f:
                self.assertEqual(len(f['states']),len(f['actions_json'])+1)
                self.assertIn('perturbation',f['phases'].asstr()[:])
                self.assertIn('recovery',f['phases'].asstr()[:])
            record=json.loads((root/'recovery_episode_result.json').read_text())
            self.assertEqual(record['source_demo_id'],result['source_demo_id'])
            self.assertLessEqual(record['steps'],20)
            from robot_stack.audit import audit
            self.assertEqual(audit(root)['qualified_corrections'],1)
            record['induced_error']=False
            (root/'recovery_episode_result.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Error predicate'):
                audit(root)

    def test_no_error_is_not_correction(self):
        class NoError(LineWorld):
            def perturbation_actions(self,steps): return [0.]*steps
        with tempfile.TemporaryDirectory() as d:
            result=collect_triplet(NoError(),0,Path(d)/'run',budget=20,perturb_steps=2)
            self.assertFalse(result['qualified_correction'])
            self.assertFalse(result['induced_error'])

    def test_identical_initial_states_share_split_despite_different_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            a=collect_triplet(LineWorld(),10,Path(d)/'a',budget=20,perturb_steps=2)
            b=collect_triplet(LineWorld(),11,Path(d)/'b',budget=20,perturb_steps=2)
            self.assertNotEqual(a['source_demo_id'],b['source_demo_id'])
            self.assertEqual(a['split_group'],b['split_group'])

    def test_replay_mismatch_rejected(self):
        class Drift(LineWorld):
            calls=0
            def reset(self,seed):
                super().reset(seed); self.x+=self.calls*.1; self.calls+=1
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Prefix replay differs'):
                collect_triplet(Drift(),0,Path(d)/'run',budget=20,perturb_steps=2)

    def test_injected_steps_count_in_budget(self):
        with tempfile.TemporaryDirectory() as d:
            result=collect_triplet(LineWorld(),0,Path(d)/'run',budget=6,perturb_steps=2)
            self.assertFalse(result['qualified_correction'])
            self.assertFalse(result['recovery_success'])

    def test_episode_never_steps_over_budget(self):
        env=LineWorld();env.reset(0);e=Episode.start(env)
        e.step(env,1,'expert',1)
        with self.assertRaises(ValueError): e.step(env,1,'recovery',1)
        self.assertEqual(env.count,1)

    def test_robodojo_initial_true_is_not_completion(self):
        class Native:
            num_envs=1
            def reset(self,seed): self.success=[True];self.end_flag=[False];self.take_action_cnt=[0]
            def take_action(self,action): self.take_action_cnt[0]+=1;self.end_flag=[True]
            def close(self): pass
        adapter=RoboDojoAdapter('test',env_factory=Native,policy=lambda obs:{},
            state_reader=lambda env:[0],feature_reader=lambda env:[0],
            perturbation=lambda env,n:[{}]*n,upstream_commit='contract-test')
        adapter.reset(0)
        self.assertFalse(adapter.success())
        adapter.step({});self.assertTrue(adapter.success())
        with self.assertRaises(RuntimeError):adapter.step({})
