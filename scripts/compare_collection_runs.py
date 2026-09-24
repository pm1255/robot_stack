"""Compare matched collection attempts without confusing coverage with rates."""
import argparse
import json
from pathlib import Path


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def metrics(rows):
    sources = [row for row in rows if row.get('source_success') is True]
    verdicts = [row for row in sources if 'recovery_success' in row]
    errors = [row for row in verdicts if row.get('induced_error') is True
              and row.get('perturbed_success') is False]
    return dict(
        attempts=len(rows),
        infrastructure_or_policy_errors=sum(row.get('status') == 'error' for row in rows),
        source_success_per_attempt=fraction(len(sources), len(rows)),
        completed_verdicts_per_successful_source=fraction(len(verdicts), len(sources)),
        recovery_success_after_failed_control=fraction(
            sum(row.get('recovery_success') is True for row in errors), len(errors)),
        qualified_correction_per_attempt=fraction(
            sum(row.get('qualified_correction') is True for row in rows), len(rows)),
    )


def compare(baseline, candidate):
    def keyed(report):
        result = {}
        for row in report['results']:
            key = (row['task'], row['seed'], row['schedule_name'])
            if key in result:
                raise ValueError('Duplicate task/seed/schedule; compare unambiguous attempts')
            result[key] = row
        return result
    left, right = keyed(baseline), keyed(candidate)
    if left.keys() != right.keys():
        raise ValueError('Runs must contain identical task/seed/schedule attempts')
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        fields = ('source_success', 'perturbed_success', 'recovery_success',
                  'qualified_correction', 'status', 'error')
        rows.append(dict(task=key[0], seed=key[1], schedule=key[2],
                         baseline={k:a[k] for k in fields if k in a},
                         candidate={k:b[k] for k in fields if k in b}))
    return dict(
        baseline=metrics(list(left.values())), candidate=metrics(list(right.values())),
        cases=rows,
        limitations='Matched task/seed/schedule names only. Verify run_config, budgets, '
                    'native versions and saved branch states before causal interpretation. '
                    'Aggregate conditional rates can have different denominators. '
                    'These results are not whole-benchmark success rates.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('baseline', 'candidate')}, indent=2))


if __name__ == '__main__':
    main()
