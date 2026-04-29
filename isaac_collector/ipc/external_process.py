from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_json_worker(
    *,
    python_exe: str,
    worker_file: str,
    request: Dict[str, Any],
    project_root: str,
    extra_args: Optional[List[str]] = None,
    timeout_sec: int = 600,
) -> Dict[str, Any]:
    project_root = str(Path(project_root).expanduser().resolve())
    worker_file = str(Path(worker_file).expanduser().resolve())

    ipc_dir = Path("/tmp/robot_pipeline/ipc")
    ipc_dir.mkdir(parents=True, exist_ok=True)

    stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
    request_path = ipc_dir / f"request_{stamp}.json"
    response_path = ipc_dir / f"response_{stamp}.json"

    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

    cmd = [
        python_exe,
        "-u",
        worker_file,
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print("[IPC] Running worker:")
    print(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
    )

    if proc.stdout.strip():
        print("[IPC][stdout]")
        print(proc.stdout)

    if proc.stderr.strip():
        print("[IPC][stderr]")
        print(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed with return code {proc.returncode}: {worker_file}"
        )

    if not response_path.exists():
        raise RuntimeError(f"Worker did not create response file: {response_path}")

    response = json.loads(response_path.read_text(encoding="utf-8"))

    if not response.get("success", False):
        raise RuntimeError(f"Worker returned failure: {json.dumps(response, indent=2)}")

    return response