#!/usr/bin/env python3
from pathlib import Path
import argparse


def find_files(root: Path, patterns):
    hits = []
    for pat in patterns:
        hits.extend(root.rglob(pat))
    return sorted(set(hits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default="/home/pm/Desktop/Project",
    )
    parser.add_argument(
        "--graspnet-root",
        default="/home/pm/Desktop/Project/graspnet-baseline",
    )
    parser.add_argument(
        "--curobo-root",
        default="/home/pm/Desktop/Project/third_party/curobo",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    graspnet_root = Path(args.graspnet_root).expanduser().resolve()
    curobo_root = Path(args.curobo_root).expanduser().resolve()

    print("=" * 80)
    print("GraspNet checkpoint check")
    print("=" * 80)

    checkpoint_patterns = [
        "checkpoint-rs.tar",
        "checkpoint-kn.tar",
        "*.pth",
        "*.pt",
        "*.ckpt",
    ]

    search_roots = [
        project_root,
        graspnet_root,
    ]

    all_hits = []
    for root in search_roots:
        if root.exists():
            all_hits.extend(find_files(root, checkpoint_patterns))

    all_hits = sorted(set(all_hits))

    if not all_hits:
        print("[FAIL] No GraspNet-like checkpoint found.")
        print("       You probably have not downloaded checkpoint-rs.tar / checkpoint-kn.tar.")
    else:
        for p in all_hits:
            size_gb = p.stat().st_size / 1024**3
            print(f"[FOUND] {p}  ({size_gb:.2f} GB)")

    print()
    print("=" * 80)
    print("GraspNet code check")
    print("=" * 80)

    expected = [
        graspnet_root / "demo.py",
        graspnet_root / "models",
        graspnet_root / "pointnet2",
        graspnet_root / "knn",
    ]

    for p in expected:
        if p.exists():
            print(f"[ OK ] {p}")
        else:
            print(f"[FAIL] missing: {p}")

    print()
    print("=" * 80)
    print("cuRobo code/config check")
    print("=" * 80)

    expected_curobo = [
        curobo_root,
        curobo_root / "src",
        curobo_root / "examples",
        curobo_root / "src" / "curobo",
    ]

    for p in expected_curobo:
        if p.exists():
            print(f"[ OK ] {p}")
        else:
            print(f"[WARN] missing: {p}")

    print()
    print("[INFO] cuRobo usually does not need a neural checkpoint.")
    print("[INFO] What you need for cuRobo is robot config + collision world + working CUDA/NVRTC.")


if __name__ == "__main__":
    main()