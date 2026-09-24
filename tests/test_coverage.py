import unittest
from robot_stack.coverage import build_coverage


class CoverageContracts(unittest.TestCase):
    def test_inventory_and_success_flag_do_not_claim_recovery(self):
        inventory = dict(backend='native', tasks=[dict(task='one',expert_available=True)])
        report = dict(backend='native',results=[dict(task='one',success=True)])
        row = build_coverage([inventory],[('smoke',report)])['tasks'][0]
        self.assertTrue(row['expert_available'])
        self.assertFalse(row['source_success'])
        self.assertFalse(row['qualified_correction'])

    def test_qualified_requires_failure_of_control_and_actual_error(self):
        row = dict(task='one',source_success=True,induced_error=True,
                   perturbed_success=False,recovery_success=True,qualified_correction=True)
        def coverage(data):
            return build_coverage([], [('experiment',dict(backend='native',results=[data]))])['tasks'][0]
        self.assertTrue(coverage(row)['qualified_correction'])
        self.assertFalse(coverage(dict(row,perturbed_success=True))['qualified_correction'])
        self.assertFalse(coverage(dict(row,induced_error=False))['qualified_correction'])
        self.assertFalse(coverage(dict(row,source_success=False))['qualified_correction'])

    def test_native_source_success_and_task_coverage_are_distinct(self):
        row=dict(task='one',success=True,execution_success=True,task_success=True)
        result=build_coverage([], [('experiment',dict(backend='native',records=[row,row]))])
        self.assertEqual(result['totals']['native']['source_success'],1)
        self.assertEqual(result['totals']['native']['qualified_correction'],0)
        self.assertEqual(result['tasks'][0]['evidence'],['experiment'])
