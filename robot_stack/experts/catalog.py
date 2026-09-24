"""List project-owned experts without treating registration as validation."""
import argparse
import json
from pathlib import Path
from .maniskill import EXPERT_SPECS
from ..adapters.robocasa_mobile import MOBILE_TASKS


def project_specs(backend):
    if backend=='maniskill':
        return {name:dict(spec) for name,spec in EXPERT_SPECS.items()}
    if backend=='robocasa':
        specs={name:dict(provider='robot_stack',skill_family='mobile_grasp_transport_place',
            status='implemented_requires_native_validation',robot='PandaOmron',
            native_success_oracle=True,privileged_state=True,
            manipulated_object=source,destination_object=destination,
            scope='Experimental task expert; see measured layout, style and object conditions')
            for name,(source,destination) in MOBILE_TASKS.items()}
        specs['NavigateKitchen']=dict(provider='robot_stack',skill_family='base_navigation',
            status='implemented_requires_native_validation',robot='PandaOmron',
            native_success_oracle=True,privileged_state=True)
        return specs
    return {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['maniskill','robocasa'],required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    data=dict(backend=args.backend,scope='Project-owned implementations only; upstream experts remain available through the adapter inventory. This catalog is not a success report.',experts=project_specs(args.backend))
    payload=json.dumps(data,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(payload)
    print(payload,end='')


if __name__=='__main__':main()
