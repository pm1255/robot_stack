"""Merge installed task inventories with explicit experiment summaries.

Coverage means at least one observed case, never a task success rate. Registry
entries and untested adapters cannot turn a verification column green.
"""
import argparse
import json
from pathlib import Path


def build_coverage(inventories, reports):
    tasks = {}
    for inventory in inventories:
        backend = inventory['backend']
        for item in inventory['tasks']:
            task = item['task']
            tasks[(backend, task)] = dict(backend=backend, task=task,
                registered=True, expert_available=bool(item.get('expert_available')),
                source_success=False, induced_error=False, recovery_success=False,
                qualified_correction=False, evidence=[])
    for name, report in reports:
        # Only explicit summaries are accepted; do not recursively double count
        # progress.json, individual episodes and the enclosing summary.
        rows = report.get('results', report.get('records', []))
        for row in rows:
            backend = row.get('backend', report.get('backend'))
            task = row.get('task', report.get('task'))
            if not backend or not task:
                continue
            key = (backend, task)
            target = tasks.setdefault(key, dict(backend=backend, task=task,
                registered=False, expert_available=False, source_success=False,
                induced_error=False, recovery_success=False,
                qualified_correction=False, evidence=[]))
            # Correction summaries use source_success; native collectors use
            # success + execution_success + task_success. A generic success
            # field alone (e.g. a simulator smoke check) is insufficient.
            source = row.get('source_success') is True or (
                row.get('success') is True and row.get('execution_success') is True
                and row.get('task_success') is True)
            eligible = (source and row.get('induced_error') is True
                        and row.get('perturbed_success') is False
                        and row.get('recovery_success') is True
                        and row.get('qualified_correction') is True)
            values = dict(source_success=source, induced_error=row.get('induced_error') is True,
                          recovery_success=row.get('recovery_success') is True,
                          qualified_correction=eligible)
            for field, value in values.items():
                target[field] |= value
            if name not in target['evidence']:
                target['evidence'].append(name)
    ordered = [tasks[key] for key in sorted(tasks)]
    totals = {}
    for row in ordered:
        summary = totals.setdefault(row['backend'], dict(tasks=0, registered=0,
            expert_available=0, source_success=0, induced_error=0,
            recovery_success=0, qualified_correction=0))
        for field in summary:
            summary[field] += 1 if field == 'tasks' else int(row[field])
    return dict(schema='robot_stack.task_coverage.v1',
        semantics='At least one observed case per task; false means no positive evidence in these reports. Not success rates or all benchmark variants.',
        inventories=[{k:v for k,v in i.items() if k!='tasks'} for i in inventories],
        totals=totals, tasks=ordered)


def main():
    from .core import write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventories', nargs='+', type=Path, required=True)
    p.add_argument('--reports', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    inventories = [json.loads(path.read_text()) for path in args.inventories]
    reports = [(str(path), json.loads(path.read_text())) for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, build_coverage(inventories, reports))


if __name__ == '__main__':
    main()
