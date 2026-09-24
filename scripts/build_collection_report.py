"""Export whitelisted episode metadata for the static collection dashboard."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def build_report(root):
    rows = []
    for path in sorted(Path(root).rglob('*_episode_result.json')):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            status = data.get('status', 'legacy_unverified')
            verified = (data.get('validation_version') == 1 and data.get('success') is True
                        and data.get('task_success') is True and not data.get('debug', False)
                        and status == 'success' and data.get('execution_success') is True
                        and (data.get('planning_success') is True or
                             (data.get('backend') in {'robosuite', 'metaworld', 'maniskill', 'robotwin'}
                              and data.get('control_success') is True)))
            if data.get('validation_version') != 1 or (status == 'success' and not verified):
                status = 'legacy_unverified'
            seconds = data.get('duration_seconds')
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds <= 0):
                seconds = None
            # Do not publish absolute paths, scene assets, stack traces or raw records.
            rows.append({'id': str(path.relative_to(root)), 'status': str(status),
                         'verified_success': verified, 'duration_seconds': seconds,
                         'planning_success': data.get('planning_success') is True,
                         'control_success': data.get('control_success') is True,
                         'backend': str(data.get('backend', 'isaac')),
                         'task': str(data.get('task', 'unspecified')),
                         'label': str(data.get('label', 'unspecified')),
                         'execution_success': data.get('execution_success') is True})
        except (ValueError, OSError):
            rows.append({'id': str(path.relative_to(root)), 'status': 'invalid_record',
                         'verified_success': False, 'duration_seconds': None,
                         'planning_success': False, 'execution_success': False})
    successes = sum(r['verified_success'] for r in rows)
    return {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {'attempts': len(rows), 'verified_successes': successes,
                        'confirmed_fraction': successes / len(rows) if rows else None,
                        'status_counts': dict(Counter(r['status'] for r in rows))},
            'episodes': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error('--input must be an existing directory')
    result = build_report(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Exported {len(result['episodes'])} attempts to {args.output}")
