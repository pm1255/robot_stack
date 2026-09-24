import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from robot_stack.suite import run_isolated


class IsolatedSuiteContracts(unittest.TestCase):
    def test_native_crash_retains_saved_source_and_marks_incomplete_correction(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/'case'
            def crash(command, **kwargs):
                source=folder/'ep_0003';source.mkdir(parents=True)
                (source/'source_episode_result.json').write_text(json.dumps(dict(success=True)))
                return subprocess.CompletedProcess(command,-11)
            with patch('robot_stack.suite.subprocess.run',side_effect=crash):
                row=run_isolated('native','task',3,folder,budget=30,
                    schedule={},options={},timeout=20)
            self.assertEqual(row['status'],'error')
            self.assertTrue(row['source_success'])
            self.assertFalse(row['qualified_correction'])
            self.assertIn('-11',row['error'])

    def test_timeout_becomes_an_attempt_record(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('robot_stack.suite.subprocess.run',side_effect=subprocess.TimeoutExpired('worker',20)):
                row=run_isolated('native','task',3,Path(root)/'case',budget=30,
                    schedule={},options={},timeout=20)
            self.assertEqual(row['status'],'error')
            self.assertFalse(row['qualified_correction'])
            self.assertIn('20 seconds',row['error'])
