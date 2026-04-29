from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from isaac_collector.ipc.external_process import run_json_worker


class GraspNetClient:
    def __init__(
        self,
        *,
        project_root: str,
        python_exe: str | None = None,
        mode: str = "mock",
        checkpoint: str | None = None,
    ):
        self.project_root = str(Path(project_root).expanduser().resolve())
        self.python_exe = python_exe or os.environ.get(
            "ROBOT_STACK_GRASPNET_PY",
            "/home/pm/miniconda3/envs/graspnet_env/bin/python",
        )
        self.mode = mode
        self.checkpoint = checkpoint

        self.worker_file = str(
            Path(self.project_root)
            / "isaac_collector"
            / "services"
            / "graspnet_worker.py"
        )

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        extra_args = ["--mode", self.mode]
        if self.checkpoint:
            extra_args += ["--checkpoint", self.checkpoint]

        return run_json_worker(
            python_exe=self.python_exe,
            worker_file=self.worker_file,
            request=request,
            project_root=self.project_root,
            extra_args=extra_args,
        )