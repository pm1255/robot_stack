"""Inventory saved files; shared sources and failed attempts remain explicit."""
from pathlib import Path
import hashlib,json,collections,h5py
import argparse
p=argparse.ArgumentParser(description='Count HDF5 trajectories and exact-file duplicates; does not certify training quality.')
p.add_argument('root',type=Path)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
root=args.root.resolve()
if not root.is_dir():p.error('root must be a directory')
rows=[];seen={};groups=collections.defaultdict(lambda:collections.Counter())
for p in sorted(root.rglob('*')):
 if p.suffix not in ('.hdf5','.h5') or not p.is_file():continue
 rel=str(p.relative_to(root));run=rel.split('/')[0]
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 digest=h.hexdigest();record={'path':rel,'run':run,'bytes':p.stat().st_size,'sha256':digest}
 if digest in seen:record['exact_duplicate_of']=seen[digest]
 else:seen[digest]=rel
 try:
  with h5py.File(p) as f:
   record['schema']=str(f.attrs.get('schema','native'))
   raw=f.attrs.get('metadata','{}');meta=json.loads(raw) if isinstance(raw,(str,bytes)) else {}
   record['metadata']=meta
   if 'actions_json' in f and 'states' in f:
    record.update(kind='correction',trajectories=1,branch=p.stem,actions=len(f['actions_json']),alignment_valid=len(f['states'])==len(f['actions_json'])+1,success=bool(f['success'][-1]) if len(f['success']) else False)
   elif 'data' in f and isinstance(f['data'],h5py.Group):
    demos=[n for n in f['data'] if isinstance(f['data'][n],h5py.Group) and 'actions' in f['data'][n]]
    record.update(kind='demo_container',trajectories=len(demos),demos=demos)
   elif any(k.startswith('traj_') and isinstance(f[k],h5py.Group) and 'actions' in f[k] for k in f):
    count=sum(k.startswith('traj_') and isinstance(f[k],h5py.Group) and 'actions' in f[k] for k in f)
    record.update(kind='native_container',trajectories=count,root_keys=list(f))
   elif 'actions' in f or 'joint_action' in f:
    record.update(kind='native',trajectories=1,root_keys=list(f))
   else:
    raise ValueError('Unrecognized trajectory layout; not counted')
  groups[run]['files']+=1;groups[run]['trajectories']+=record['trajectories']
  if 'exact_duplicate_of' not in record:groups[run]['unique_files']+=1;groups[run]['byte_unique_trajectories']+=record['trajectories']
  if record.get('kind')=='correction':
   groups[run][record['branch']]+=1
   if record['success']:groups[run][record['branch']+'_successful']+=1
 except Exception as exc:record['error']=repr(exc);groups[run]['unreadable']+=1
 rows.append(record)
result={'scope':str(root),'definition':'HDF5 files and contained action trajectories; exact-byte file duplicates excluded separately. Replays/videos are not new trajectories. Distinct files may share source demonstrations; counts are not independent initial states. Native success is not inferred without a result record.','files':len(rows),'unique_files':len(seen),'contained_trajectories':sum(r.get('trajectories',0) for r in rows),'byte_unique_trajectories':sum(r.get('trajectories',0) for r in rows if 'exact_duplicate_of' not in r),'by_run':dict(groups),'records':rows}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
