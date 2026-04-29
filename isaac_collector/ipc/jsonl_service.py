from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class PersistentJsonService:
    """
    IsaacLab 主进程用它启动并持续调用外部 worker。

    worker 通过 stdin/stdout 使用 JSON Lines 协议：
    - Isaac 写入一行 JSON request
    - worker 返回一行 JSON response
    """

    def __init__(
        self,
        *,
        name: str,
        python_exe: str,
        worker_file: str,
        project_root: str,
        args: Optional[List[str]] = None,
        log_path: Optional[str] = None,
        startup_timeout: float = 120.0,
    ):
        self.name = name
        self.python_exe = str(Path(python_exe).expanduser().resolve())
        self.worker_file = str(Path(worker_file).expanduser().resolve())
        self.project_root = str(Path(project_root).expanduser().resolve())
        self.args = args or []

        log_dir = Path("/tmp/robot_pipeline/service_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else log_dir / f"{name}.log"
        self.log_fp = self.log_path.open("w", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = self.project_root + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            self.python_exe,
            "-u",
            self.worker_file,
            *self.args,
        ]

        print(f"[SERVICE] starting {self.name}")
        print("[SERVICE] cmd:", " ".join(cmd))
        print("[SERVICE] log:", self.log_path)

        self.proc = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_fp,
            text=True,
            bufsize=1,
        )

        self._queue: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        ready = self._read_json(timeout=startup_timeout)
        if ready.get("type") != "ready":
            raise RuntimeError(f"{self.name} did not send ready message: {ready}")

        print(f"[SERVICE] {self.name} ready:")
        print(json.dumps(ready, indent=2))

    def _reader_loop(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self._queue.put(line)

    def _read_json(self, timeout: float) -> Dict[str, Any]:
        start = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited with code {self.proc.returncode}. "
                    f"See log: {self.log_path}"
                )

            remain = timeout - (time.time() - start)
            if remain <= 0:
                raise TimeoutError(f"Timeout waiting response from {self.name}")

            try:
                line = self._queue.get(timeout=min(0.5, remain))
            except queue.Empty:
                continue

            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON from {self.name}: {line}") from e

    def call(self, method: str, params: Dict[str, Any], timeout: float = 600.0) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"{self.name} already exited with code {self.proc.returncode}. "
                f"See log: {self.log_path}"
            )

        request_id = str(uuid.uuid4())
        req = {
            "id": request_id,
            "method": method,
            "params": params,
        }

        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

        while True:
            resp = self._read_json(timeout=timeout)

            if resp.get("type") == "ready":
                continue

            if resp.get("id") != request_id:
                print(f"[WARN] ignore unmatched response from {self.name}: {resp}")
                continue

            if not resp.get("success", False):
                raise RuntimeError(
                    f"{self.name}.{method} failed:\n"
                    f"{json.dumps(resp, indent=2)}\n"
                    f"See log: {self.log_path}"
                )

            return resp["result"]

    def close(self):
        try:
            if self.proc.poll() is None:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"id": "shutdown", "method": "shutdown", "params": {}}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            self.log_fp.close()