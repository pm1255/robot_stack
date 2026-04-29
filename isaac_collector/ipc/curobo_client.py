from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from isaac_collector.ipc.external_process import run_json_worker


class CuRoboClient:
    def __init__(
        self,
        *,
        project_root: str,
        python_exe: str | None = None,
        mode: str = "mock",
    ):
        self.project_root = str(Path(project_root).expanduser().resolve())
        self.python_exe = python_exe or os.environ.get(
            "ROBOT_STACK_CUROBO_PY",
            "/home/pm/miniconda3/envs/curobo_env/bin/python",
        )
        self.mode = mode

        self.worker_file = str(
            Path(self.project_root)
            / "isaac_collector"
            / "services"
            / "curobo_worker.py"
        )

    def plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return run_json_worker(
            python_exe=self.python_exe,
            worker_file=self.worker_file,
            request=request,
            project_root=self.project_root,
            extra_args=["--mode", self.mode],
        )