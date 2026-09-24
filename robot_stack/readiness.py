"""Keep every inventoried task visible while overlaying measured calibration gates."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .core import write_json


def build_readiness(coverage, calibrations):
    rows = {}
    for task in coverage['tasks']:
        key = task['backend'], task['task']
        rows[key] = dict(backend=key[0], task=key[1],
            historical_source_evidence=bool(task['source_success']),
            historical_correction_evidence=bool(task['qualified_correction']),
            status=('awaiting_calibration' if task['expert_available'] or task['source_success']
                    else 'missing_expert'),
            clean_baseline=None, validation=None, selected=None,
            evidence=list(task.get('evidence', [])))
    seen = set()
    for reference, report in calibrations:
        if report.get('schema') != 'robot_stack.task_calibration.v1':
            raise ValueError('Expected a completed task calibration summary')
        if 'passed_tasks' not in report:
            raise ValueError('In-progress calibration cannot certify task readiness')
        for name, profile in report['tasks'].items():
            key = report['backend'], name
            if key in seen:
                raise ValueError('Choose one explicit current calibration per task')
            seen.add(key)
            row = rows.setdefault(key, dict(backend=key[0],task=key[1],
                historical_source_evidence=False,historical_correction_evidence=False,evidence=[]))
            row.update(status=profile['status'], clean_baseline=profile.get('baseline'),
                       validation=profile.get('validation'), selected=profile.get('selected'),
                       levels=profile.get('levels',[]), criteria={k:report['config'][k] for k in
                       ('min_source_rate','min_recovery_rate','min_validation_episodes')},
                       calibration_report=reference)
    return dict(schema='robot_stack.task_readiness.v1',
                semantics='Historical examples do not satisfy measured baseline and held-out perturbation gates. '
                          'Null rate means not measured, never zero. Each pass is conditional on the recorded '
                          'task, controller, schedule, seeds and empirical thresholds, not a general guarantee.',
                tasks=[rows[key] for key in sorted(rows)], task_count=len(rows),
                status_counts=dict(Counter(row['status'] for row in rows.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--calibrations',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    reports=[(str(path),json.loads(path.read_text())) for path in args.calibrations]
    result=build_readiness(json.loads(args.coverage.read_text()),reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps(result['status_counts']))


if __name__=='__main__':main()
