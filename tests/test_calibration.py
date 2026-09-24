import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from robot_stack.calibrate import scale_schedule, metrics, select_profile, validation_status, run
from robot_stack.core import collect_source
from test_corrections import LineWorld


class CalibrationContracts(unittest.TestCase):
    def test_source_only_never_injects_and_preserves_failed_attempt(self):
        class CleanOnly(LineWorld):
            def perturbation_actions(self, steps):
                raise AssertionError('No perturbation is permitted in baseline')
        with tempfile.TemporaryDirectory() as d:
            result = collect_source(CleanOnly(), 1, Path(d)/'clean', budget=10)
            self.assertTrue(result['source_success'])
            self.assertEqual(len(list((Path(d)/'clean').glob('*.hdf5'))), 1)
            failed = collect_source(CleanOnly(), 2, Path(d)/'failed', budget=2)
            self.assertFalse(failed['source_success'])
            self.assertTrue((Path(d)/'failed/source.hdf5').exists())
            from robot_stack.audit import audit
            checked = audit(Path(d))
            self.assertEqual(checked['clean_baseline_attempts'],2)
            self.assertEqual(checked['incomplete_source_attempts'],0)

    def test_scaling_preserves_other_parameters_and_discrete_steps(self):
        spec = {'version':1,'events':[{'at':{'fraction':.6},'type':'joint_offset',
                'steps':15,'strength':.8,'parameters':{'offset':.8}}]}
        before = deepcopy(spec)
        scaled = scale_schedule(spec,.25)
        self.assertAlmostEqual(scaled['events'][0]['strength'], .2)
        self.assertEqual(scaled['events'][0]['steps'],15)
        self.assertEqual(scale_schedule(spec,.125,'steps')['events'][0]['steps'],2)
        self.assertEqual(spec,before)
        with self.assertRaises(ValueError):scale_schedule(spec,0)

    def test_easy_no_error_does_not_win_and_validation_leakage_is_blocked(self):
        good = dict(source_success=True, induced_error=True, perturbed_success=False,
                    recovery_success=True, qualified_correction=True)
        easy = dict(source_success=True, induced_error=False, perturbed_success=True,
                    recovery_success=True, qualified_correction=False)
        levels = [dict(scale=.1,metrics=metrics([easy])),dict(scale=.25,metrics=metrics([good]))]
        self.assertEqual(select_profile(levels,.8,.5)['scale'],.25)
        self.assertIsNone(select_profile(levels[:1],.8,.5))
        rows = [dict(good,split_group='same')]
        kw=dict(min_source_rate=.8,min_recovery_rate=.5,min_episodes=1)
        self.assertEqual(validation_status(rows,{'same'},**kw),'scene_overlap_with_calibration')
        self.assertEqual(validation_status(rows,set(),**kw),'passed_configured_validation')
        self.assertEqual(validation_status(rows*5,set(),**dict(kw,min_episodes=5)),
                         'insufficient_validation_scenes')

    def test_pipeline_keeps_missing_experts_and_freezes_before_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            inventory=root/'inventory.json';schedule=root/'schedule.json'
            inventory.write_text(json.dumps(dict(backend='test',tasks=[
                dict(task='supported',expert_available=True),dict(task='missing',expert_available=False)])))
            schedule.write_text(json.dumps(dict(version=1,events=[dict(at={'step':1},type='offset',steps=2)])))
            args=SimpleNamespace(adapter_options='{}',inventory=inventory,backend='test',tasks=['all'],
                seed_start=10,episodes=1,validation_seed_start=20,validation_episodes=1,
                schedule=schedule,axis='strength',scales=[.25,1.],output=root/'run',workers=1,
                max_steps=20,case_timeout=10,min_source_rate=.8,min_recovery_rate=.5,
                min_validation_episodes=1)
            def worker(backend,task,seed,folder,**kwargs):
                if seed==20:
                    self.assertTrue((args.output/'selected_profiles.json').exists())
                source=folder/f'ep_{seed:04d}';source.mkdir(parents=True)
                (source/'source_episode_result.json').write_text(json.dumps(dict(split_group=str(seed))))
                row=dict(task=task,seed=seed,source_success=True,qualified_correction=False)
                if not kwargs['source_only']:
                    row.update(induced_error=True,perturbed_success=False,recovery_success=True,
                               qualified_correction=True)
                return row
            with patch('robot_stack.calibrate.run_isolated',side_effect=worker):run(args)
            report=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(report['tasks']['missing']['status'],'missing_expert')
            self.assertEqual(report['tasks']['supported']['selected']['scale'],.25)
            self.assertEqual(report['passed_tasks'],1)
            self.assertEqual(report['tasks']['supported']['baseline']['attempts'],1)


class ReadinessContracts(unittest.TestCase):
    def test_historical_success_is_not_baseline_rate_or_calibrated_pass(self):
        from robot_stack.readiness import build_readiness
        coverage={'tasks':[dict(backend='native',task='one',expert_available=True,
                              source_success=True,qualified_correction=True),
                           dict(backend='native',task='two',expert_available=False,
                              source_success=False,qualified_correction=False)]}
        report=build_readiness(coverage,[])
        self.assertEqual(report['task_count'],2)
        self.assertEqual(report['tasks'][0]['status'],'awaiting_calibration')
        self.assertIsNone(report['tasks'][0]['clean_baseline'])
        self.assertEqual(report['tasks'][1]['status'],'missing_expert')
