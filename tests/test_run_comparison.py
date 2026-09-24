import unittest
from scripts.compare_collection_runs import compare, metrics


class RunComparisonTests(unittest.TestCase):
    def test_failure_denominators_are_explicit(self):
        result = metrics([
            {'status': 'error'},
            {'source_success': False},
            {'source_success': True, 'perturbed_success': True, 'recovery_success': True},
            {'source_success': True, 'induced_error': True, 'perturbed_success': False,
             'recovery_success': True, 'qualified_correction': True},
            {'source_success': True, 'induced_error': True, 'perturbed_success': False,
             'recovery_success': False},
        ])
        self.assertEqual(result['source_success_per_attempt']['rate'], .6)
        self.assertEqual(result['recovery_success_after_failed_control']['rate'], .5)
        self.assertEqual(result['qualified_correction_per_attempt']['rate'], .2)
        self.assertIsNone(metrics([])['source_success_per_attempt']['rate'])

    def test_mismatched_or_duplicate_attempts_are_rejected(self):
        row = dict(task='PickCube-v1', seed=1, schedule_name='late')
        with self.assertRaises(ValueError):
            compare({'results': [row, row]}, {'results': [row]})
        with self.assertRaises(ValueError):
            compare({'results': [row]}, {'results': []})
