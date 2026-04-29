#!/usr/bin/env python3
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs",
        type=str,
        default="/tmp/robot_pipeline/repeated_pick_place/ep_0000_observation_rgbd.npz",
    )
    args = parser.parse_args()

    obs = np.load(args.obs, allow_pickle=True)

    points = obs["points"].astype(np.float32)
    colors = obs["colors"].astype(np.float32)

    print("[LOAD]", args.obs)
    print("[SHAPE] points:", points.shape, points.dtype)
    print("[SHAPE] colors:", colors.shape, colors.dtype)

    print("[RANGE] points min:", points.min(axis=0))
    print("[RANGE] points max:", points.max(axis=0))
    print("[RANGE] colors min/max:", colors.min(), colors.max())

    assert points.ndim == 2 and points.shape[1] == 3
    assert colors.ndim == 2 and colors.shape[1] == 3
    assert points.shape[0] == colors.shape[0]
    assert np.isfinite(points).all()
    assert np.isfinite(colors).all()

    print("[OK] This observation can be passed to GraspNet skeleton.")


if __name__ == "__main__":
    main()
