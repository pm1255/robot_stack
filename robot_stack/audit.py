"""Validate trajectory files and independently recompute correction eligibility."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .core import write_json


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
    return {'hdf5_verified':len(records),'source_attempts':total,'qualified_corrections':qualified,
            'checks':['SHA256','finite_states','T+1_alignment','native_success_trace','budget','paired_error_state','provenance'],
            'passed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
