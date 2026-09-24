"""Replay correction HDF5 actions in native dynamics, optionally record RGB video."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from .collect import load_adapter
from .core import write_json
from robosuite_collector.replay import VideoWriter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video',type=Path)
    p.add_argument('--video-stride',type=int,default=1,help='Render every N actions; verify every state regardless')
    p.add_argument('--adapter-options',default='{}')
    p.add_argument('--tolerance',type=float,default=1e-7)
    args=p.parse_args()
    if args.video_stride < 1: p.error("video-stride must be positive")
    with h5py.File(args.trajectory,'r') as f:
        if f.attrs['schema'] != 'robot_stack.correction.v1':
            raise ValueError('Unsupported trajectory schema')
        metadata=json.loads(f.attrs['metadata'])
        states=f['states'][:]; flags=f['success'][:]
        actions=[json.loads(a) for a in f['actions_json'].asstr()[:]]
    if len(states)!=len(actions)+1 or len(flags)!=len(actions):
        raise ValueError('Trajectory alignment mismatch')
    options=json.loads(args.adapter_options)
    if metadata['backend'] == 'metaworld':
        for key in ('task_index', 'benchmark_seed'):
            options.setdefault(key, metadata[key])
    elif metadata['backend'] == 'robocasa':
        for key in ('layout', 'style'):
            options.setdefault(key, metadata[key])
    options['render']=bool(args.video)
    adapter=load_adapter(metadata['backend'],metadata['task'],options)
    writer=None
    try:
        adapter.reset(metadata['seed'])
        error=float(np.max(np.abs(adapter.state()-states[0])))
        observed=[]
        for index,action in enumerate(actions):
            adapter.step(action)
            error=max(error,float(np.max(np.abs(adapter.state()-states[index+1]))))
            observed.append(adapter.success())
            if args.video and ((index + 1) % args.video_stride == 0 or index == len(actions)-1):
                frame=adapter.render()
                if writer is None:
                    args.video.parent.mkdir(parents=True,exist_ok=True)
                    writer=VideoWriter(args.video,width=frame.shape[1],height=frame.shape[0],fps=20)
                writer.append_data(frame)
        if writer:
            writer.close(); writer=None
        matched=bool(np.array_equal(flags,observed))
        result={'backend':metadata['backend'],'task':metadata['task'],'seed':metadata['seed'],
                'steps':len(actions),'max_abs_state_error':error,'success_trace_matches':matched,
                'final_success':bool(adapter.success()),'tolerance':args.tolerance,
                'passed':bool(np.isfinite(error) and error<=args.tolerance and matched),
                'replay_mode':'same_seed_reset_then_real_actions_no_state_restore',
                'video_time_is_physics_time':False,'video_stride':args.video_stride}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.output,result)
        print(json.dumps(result),flush=True)
        return int(not result['passed'])
    finally:
        if writer: writer.close()
        adapter.close()

if __name__=='__main__':
    raise SystemExit(main())
