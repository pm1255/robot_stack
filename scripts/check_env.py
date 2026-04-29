#!/usr/bin/env python3
import argparse
import ctypes
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_cmd(cmd):
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        return True, out.strip()
    except Exception as e:
        return False, str(e)


def check_import(module_name: str):
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"[FAIL] import {module_name}: module not found")
            return False

        importlib.import_module(module_name)
        print(f"[ OK ] import {module_name}")
        return True

    except Exception as e:
        print(f"[FAIL] import {module_name}")
        print(f"       {type(e).__name__}: {e}")
        return False


def check_basic():
    print_section("Basic Environment")

    print(f"python executable : {sys.executable}")
    print(f"python version    : {sys.version.replace(chr(10), ' ')}")
    print(f"platform          : {platform.platform()}")
    print(f"cwd               : {os.getcwd()}")
    print(f"CONDA_PREFIX      : {os.environ.get('CONDA_PREFIX')}")
    print(f"CONDA_DEFAULT_ENV : {os.environ.get('CONDA_DEFAULT_ENV')}")
    print(f"LD_LIBRARY_PATH   : {os.environ.get('LD_LIBRARY_PATH', '')}")

    print("\nPATH python:")
    ok, out = run_cmd(["which", "python"])
    print(out if ok else f"[WARN] {out}")

    print("\nCompiler:")
    for tool in ["gcc", "g++", "cmake", "make", "ninja", "nvcc"]:
        path = shutil.which(tool)
        if path is None:
            print(f"[WARN] {tool}: not found")
        else:
            ok, out = run_cmd([tool, "--version"])
            first_line = out.splitlines()[0] if ok and out else out
            print(f"[ OK ] {tool}: {path} | {first_line}")


def check_torch():
    print_section("Torch / CUDA / NVRTC")

    try:
        import torch

        print(f"torch.__version__       : {torch.__version__}")
        print(f"torch.version.cuda      : {torch.version.cuda}")
        print(f"torch.cuda.is_available : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"gpu name                : {torch.cuda.get_device_name(0)}")
            print(f"capability              : {torch.cuda.get_device_capability(0)}")
            print(f"arch list               : {torch.cuda.get_arch_list()}")

            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print(f"[ OK ] simple CUDA matmul: {float(y[0, 0]):.6f}")
        else:
            print("[FAIL] CUDA is not available in torch")

    except Exception as e:
        print("[FAIL] torch check failed")
        print(type(e).__name__, e)
        traceback.print_exc()

    # NVRTC
    try:
        lib = ctypes.CDLL("libnvrtc.so.12")
        major = ctypes.c_int()
        minor = ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        print(f"NVRTC version           : {major.value}.{minor.value}")

        # Try to locate actual shared library path.
        try:
            class DlInfo(ctypes.Structure):
                _fields_ = [
                    ("dli_fname", ctypes.c_char_p),
                    ("dli_fbase", ctypes.c_void_p),
                    ("dli_sname", ctypes.c_char_p),
                    ("dli_saddr", ctypes.c_void_p),
                ]

            libdl = ctypes.CDLL("libdl.so.2")
            info = DlInfo()
            fn_ptr = ctypes.cast(lib.nvrtcVersion, ctypes.c_void_p)
            ret = libdl.dladdr(fn_ptr, ctypes.byref(info))
            if ret and info.dli_fname:
                print(f"NVRTC path              : {info.dli_fname.decode()}")
        except Exception:
            pass

        print("[ OK ] libnvrtc.so.12 load success")

    except Exception as e:
        print("[FAIL] cannot load libnvrtc.so.12")
        print(f"       {type(e).__name__}: {e}")


def check_isaaclab(smoke: bool):
    print_section("Isaac Lab Check")

    modules = [
        "isaaclab",
        "isaaclab.app",
        "isaaclab.sim",
        "isaaclab.assets",
        "isaaclab.scene",
    ]

    for m in modules:
        check_import(m)

    if not smoke:
        print("\n[INFO] Skip Isaac SimulationApp smoke test.")
        print("       To test real Isaac startup, run with: --smoke-isaac")
        return

    print("\n[INFO] Starting Isaac Lab SimulationApp smoke test...")

    try:
        from isaaclab.app import AppLauncher

        # Different Isaac Lab versions support slightly different constructors.
        try:
            app_launcher = AppLauncher(headless=True)
        except TypeError:
            app_launcher = AppLauncher({"headless": True})

        simulation_app = app_launcher.app
        print("[ OK ] Isaac SimulationApp started")

        simulation_app.close()
        print("[ OK ] Isaac SimulationApp closed")

    except Exception as e:
        print("[FAIL] Isaac SimulationApp smoke test failed")
        print(f"       {type(e).__name__}: {e}")
        traceback.print_exc()


def check_curobo():
    print_section("cuRobo Check")

    modules = [
        "curobo",
        "curobo.cuda_robot_model",
        "curobo.geom",
        "curobo.wrap",
        "curobo.types",
    ]

    for m in modules:
        check_import(m)

    # cuRobo often depends on torch + CUDA JIT/NVRTC.
    print("\n[INFO] cuRobo environment should have working torch CUDA and NVRTC above.")
    print("[INFO] If cuRobo later fails during planning, the next check should be a small cuRobo planning script.")


def check_graspnet(graspnet_root: str):
    print_section("GraspNet Check")

    root = Path(graspnet_root).expanduser().resolve()
    print(f"graspnet root: {root}")

    if not root.exists():
        print(f"[FAIL] graspnet root does not exist: {root}")
        return

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print(f"[ OK ] added to sys.path: {root}")

    expected_paths = [
        root / "models",
        root / "pointnet2",
        root / "knn",
    ]

    for p in expected_paths:
        if p.exists():
            print(f"[ OK ] exists: {p}")
        else:
            print(f"[WARN] missing: {p}")

    # Basic packages
    for m in [
        "numpy",
        "scipy",
        "torch",
        "open3d",
    ]:
        check_import(m)

    # GraspNet-baseline specific modules.
    # Some may fail if CUDA extensions were not compiled.
    candidate_modules = [
        "models.graspnet",
        "pointnet2",
        "pointnet2._ext",
        "knn",
    ]

    for m in candidate_modules:
        check_import(m)

    print("\n[INFO] If pointnet2._ext or knn fails, GraspNet CUDA extensions are not built correctly.")


def check_project_files(project_root: str):
    print_section("Project File Check")

    root = Path(project_root).expanduser().resolve()
    print(f"project root: {root}")

    paths = [
        root / "isaac_collector",
        root / "graspnet_runner",
        root / "curobo_runner",
        root / "common",
    ]

    for p in paths:
        if p.exists():
            print(f"[ OK ] exists: {p}")
        else:
            print(f"[WARN] missing: {p}")

    pipeline_dir = Path("/tmp/robot_pipeline")
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    test_file = pipeline_dir / "write_test.json"
    try:
        test_file.write_text(json.dumps({"ok": True}, indent=2))
        print(f"[ OK ] can write pipeline dir: {pipeline_dir}")
    except Exception as e:
        print(f"[FAIL] cannot write pipeline dir: {pipeline_dir}")
        print(f"       {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["isaaclab", "curobo", "graspnet", "basic"],
    )
    parser.add_argument(
        "--project-root",
        default="/home/pm/Desktop/Project/robot_stack",
    )
    parser.add_argument(
        "--graspnet-root",
        default="/home/pm/Desktop/Project/graspnet-baseline",
    )
    parser.add_argument(
        "--smoke-isaac",
        action="store_true",
        help="Actually start Isaac SimulationApp in headless mode.",
    )

    args = parser.parse_args()

    check_basic()
    check_torch()
    check_project_files(args.project_root)

    if args.mode == "isaaclab":
        check_isaaclab(smoke=args.smoke_isaac)
    elif args.mode == "curobo":
        check_curobo()
    elif args.mode == "graspnet":
        check_graspnet(args.graspnet_root)
    elif args.mode == "basic":
        pass

    print_section("Done")
    print("[INFO] Environment check finished.")


if __name__ == "__main__":
    main()