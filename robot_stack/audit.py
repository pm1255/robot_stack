"""Validate trajectory files and independently recompute correction eligibility."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .core import write_json
from .schedule import resolve_schedule


def audit_scheduled(folder, source, control, recovery):
    """Recompute each event from arrays, including commands and full source prefix."""
    from types import SimpleNamespace
    with h5py.File(folder/source['trajectory_file'], 'r') as src, \
         h5py.File(folder/control['trajectory_file'], 'r') as a, \
         h5py.File(folder/recovery['trajectory_file'], 'r') as b:
        resolved = resolve_schedule(control['schedule'], SimpleNamespace(
            actions=src['actions_json'][:], events=[json.loads(x) for x in src['events_json'].asstr()[:]]))
        first, tolerance = resolved[0]['source_step'], recovery['replay_tolerance']
        first_ends = []
        for f, row in ((a,control),(b,recovery)):
            # Sidecars must agree with metadata covered by the HDF5 checksum.
            meta = json.loads(f.attrs['metadata'])
            for key in ('schedule','resolved_schedule','perturbation_events','induced_error',
                        'all_events_applied','requested_event_count','applied_event_count'):
                if row[key] != meta[key]:
                    raise ValueError('Scheduled sidecar disagrees with checksummed metadata')
            if row['resolved_schedule'] != resolved or row['requested_event_count'] != len(resolved):
                raise ValueError('Incorrect resolved schedule')
            prefix = float(np.max(np.abs(f['states'][:first+1]-src['states'][:first+1])))
            if prefix > tolerance or not np.array_equal(f['actions_json'][:first], src['actions_json'][:first]):
                raise ValueError('Scheduled prefix does not match source')
            logs = row['perturbation_events']
            offset, induced_flags = 0, []
            for i, event in enumerate(logs):
                spec = resolved[i]
                start, end = event['action_range']
                if (event['id'] != spec['id'] or event['source_step'] != spec['source_step']
                        or start != spec['source_step'] + offset or not start < end <= len(f['actions_json'])
                        or end-start > spec['steps']):
                    raise ValueError('Invalid scheduled event interval')
                offset += end-start
                if not all(x == 'perturbation' for x in f['phases'].asstr()[start:end]):
                    raise ValueError('Missing perturbation phase labels')
                displacement = float(np.linalg.norm(f['error_features'][end]-f['error_features'][start]))
                induced = displacement >= event['threshold'] and not bool(f['success'][end-1])
                if induced != event['induced_error'] or event['complete'] != (end-start == spec['steps']):
                    raise ValueError('Scheduled error predicate disagrees with physical features')
                induced_flags.append(induced)
            applied = len(logs) == len(resolved) and all(e['complete'] for e in logs)
            if applied != row['all_events_applied'] or (applied and all(induced_flags)) != row['induced_error']:
                raise ValueError('Scheduled aggregate error predicate incorrect')
            first_ends.append(logs[0]['action_range'][1] if logs else None)
        for x,y in zip(control['perturbation_events'], recovery['perturbation_events']):
            s,e = x['action_range']; t,u = y['action_range']
            common = min(e-s, u-t)
            if s != t or not np.array_equal(a['actions_json'][s:s+common], b['actions_json'][t:t+common]):
                raise ValueError('Branch intervention commands differ')
        same = (first_ends[0] is not None and first_ends[1] is not None and
                np.max(np.abs(a['states'][first_ends[0]]-b['states'][first_ends[1]])) <= tolerance)
        return bool(source['success'] and not control['success'] and recovery['success']
                    and control['induced_error'] and recovery['induced_error'] and same)


def audit(root):
    records={}
    for path in sorted(root.rglob('*_episode_result.json')):
        data=json.loads(path.read_text())
        trajectory=path.parent/data['trajectory_file']
        if hashlib.sha256(trajectory.read_bytes()).hexdigest()!=data['trajectory_sha256']:
            raise ValueError(f'Checksum mismatch: {trajectory}')
        with h5py.File(trajectory,'r') as f:
            if f.attrs.get('schema')!='robot_stack.correction.v1':continue
            n=len(f['actions_json'])
            for key in ['states','error_features']:
                if len(f[key])!=n+1 or not np.isfinite(f[key][:]).all():
                    raise ValueError(f'Invalid aligned state dataset: {trajectory}/{key}')
            if len(f['phases'])!=n or len(f['success'])!=n or n!=data['steps']:
                raise ValueError('Action or label alignment mismatch')
            flags=f['success'][:]
            if bool(n and flags[-1])!=data['success']:
                raise ValueError('Result disagrees with native success trace')
            if n>data['max_steps']:raise ValueError('Action budget exceeded')
        records[str(path.relative_to(root))]=data
    qualified,total=0,0
    for path in sorted(root.rglob('correction.json')):
        verdict=json.loads(path.read_text());total+=1
        if not verdict.get('source_success'):continue
        folder=path.parent
        rows=[json.loads((folder/(name+'_episode_result.json')).read_text()) for name in ['source','perturbed','recovery']]
        source,control,recovery=rows
        if len({r['source_demo_id'] for r in rows})!=1 or len({r['split_group'] for r in rows})!=1:
            raise ValueError('Broken correction provenance')
        for r in rows[1:]:
            if r['source_sha256']!=source['trajectory_sha256']:raise ValueError('Wrong source hash')
        if verdict.get('protocol') == 'scheduled_interventions.v1':
            eligible = audit_scheduled(folder, source, control, recovery)
            if eligible != verdict['qualified_correction']:
                raise ValueError('Correction verdict disagrees with saved evidence')
            qualified += int(eligible)
            continue
        # Recompute paired branch-state equality from the actual saved arrays.
        with h5py.File(folder/control['trajectory_file'],'r') as a, h5py.File(folder/recovery['trajectory_file'],'r') as b:
            end=control['perturbation_action_range'][1]
            if end!=recovery['perturbation_action_range'][1]:raise ValueError('Different perturbation ranges')
            same=float(np.max(np.abs(a['states'][end]-b['states'][end])))
            for f,r in [(a,control),(b,recovery)]:
                start=r['branch_step']
                displacement=float(np.linalg.norm(f['error_features'][end]-f['error_features'][start]))
                induced=displacement>=r['perturbation']['threshold'] and not bool(f['success'][end-1])
                if bool(induced)!=r['induced_error']:raise ValueError('Error predicate disagrees with physical features')
        tolerance=recovery.get('replay_tolerance',1e-7)
        eligible=(source['success'] and not control['success'] and recovery['success']
                  and control['induced_error'] and recovery['induced_error'] and same<=tolerance
                  and max(control['prefix_max_abs_error'],recovery['prefix_max_abs_error'])<=tolerance)
        if bool(eligible)!=verdict['qualified_correction']:
            raise ValueError('Correction verdict disagrees with saved evidence')
        qualified+=int(eligible)
    sources = [Path(name).parent for name, row in records.items() if row.get('label') == 'source']
    incomplete = sum(not (root/folder/'correction.json').exists() for folder in sources)
    return {'hdf5_verified':len(records),'source_attempts':len(sources),
            'completed_correction_verdicts':total, 'incomplete_source_attempts':incomplete,
            'qualified_corrections':qualified,
            'checks':['SHA256','finite_states','T+1_alignment','native_success_trace','budget','paired_error_state','provenance'],
            'passed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
