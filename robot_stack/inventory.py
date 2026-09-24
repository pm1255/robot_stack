"""Discover installed task registries without claiming execution or recovery."""
import argparse
import ast
import hashlib
import tokenize
from importlib.metadata import version
import json
from pathlib import Path


def file_tasks(folder):
    result = []
    for path in sorted(Path(folder).glob('*.py')):
        if path.stem.startswith('_'):
            continue
        with tokenize.open(path) as source:
            tree = ast.parse(source.read())
        if any(isinstance(node, ast.ClassDef) and node.name == path.stem for node in tree.body):
            result.append(path.stem)
    return result


def discover(backend, root=None):
    experts = []
    if backend == 'metaworld':
        import metaworld
        from benchmark_collector.metaworld import discover_tasks
        tasks, experts = sorted(metaworld.ALL_V3_ENVIRONMENTS), sorted(discover_tasks())
        revision = version('metaworld')
    elif backend == 'maniskill':
        import mani_skill.envs
        from mani_skill.utils.registration import REGISTERED_ENVS
        from .adapters.maniskill import discover_tasks
        tasks, experts = sorted(REGISTERED_ENVS), sorted(discover_tasks())
        revision = version('mani_skill')
    elif backend == 'robocasa':
        import robocasa
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
        # Official dataset tasks, excluding helper/base environment classes.
        tasks = sorted(set(ATOMIC_TASK_DATASETS) | set(COMPOSITE_TASK_DATASETS))
        experts, revision = ['NavigateKitchen'], robocasa.__version__
    elif backend in {'robotwin','robodojo'}:
        if root is None:
            raise ValueError('A local official source root is required')
        folder = Path(root)/('envs' if backend=='robotwin' else 'task/RoboDojo/tasks')
        tasks = file_tasks(folder)
        experts = tasks if backend=='robotwin' else []
        revision = 'source_files_sha256:' + hashlib.sha256(''.join(
            t + hashlib.sha256((folder/(t+'.py')).read_bytes()).hexdigest()
            for t in tasks).encode()).hexdigest()
    else:
        raise ValueError('Supported inventories: metaworld, maniskill, robocasa, robotwin, robodojo')
    return dict(backend=backend, version=revision, task_count=len(tasks),
                registered_expert_count=sum(t in experts for t in tasks),
                additional_adapter_tasks=sorted(set(experts)-set(tasks)), scope='installed native registry, not complete literature coverage',
                tasks=[dict(task=t,registered=True,expert_available=t in experts,
                            source_success_verified=False,perturbation_verified=False,
                            recovery_success_verified=False,status='not_evaluated_by_inventory') for t in tasks])


def main():
    from .core import write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',required=True)
    p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=discover(args.backend,args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,data)
    print(json.dumps({k:v for k,v in data.items() if k!='tasks'}))


if __name__=='__main__':main()
