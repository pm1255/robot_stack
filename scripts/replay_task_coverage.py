"""Independently replay one qualified triplet per task from supplied suite reports."""
import argparse
import json
from pathlib import Path
import sys
from robot_stack.replay import main as replay
from robot_stack.core import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    selected={}
    for root in a.runs:
        for row in json.loads((root/'summary.json').read_text())['results']:
            if row['qualified_correction']:
                selected.setdefault(row['task'],root/row['task']/row['schedule_name']/f"ep_{row['seed']:04d}")
    records=[]
    for task,folder in sorted(selected.items()):
        for branch in ('source','perturbed','recovery'):
            target=a.output/(task+'-'+branch+'.json')
            old=sys.argv
            try:
                sys.argv=['replay',str(folder/(branch+'.hdf5')),'--output',str(target)]
                code=replay()
            finally:
                sys.argv=old
            record=json.loads(target.read_text())
            records.append(dict(record,branch=branch,trajectory=str(folder/(branch+'.hdf5'))))
            if code:
                raise RuntimeError(f'Native replay failed: {task}/{branch}')
    write_json(a.output/'summary.json',dict(task_count=len(selected),trajectories=len(records),
        passed=all(r['passed'] for r in records),results=records))


if __name__=='__main__':main()
