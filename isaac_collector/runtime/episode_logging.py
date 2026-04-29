from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import numpy as np


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    p.write_text(json.dumps(data, indent=2, default=convert), encoding="utf-8")


def matrix_to_list(x) -> list:
    return np.asarray(x, dtype=float).tolist()


def translation_row(x) -> list:
    m = np.asarray(x, dtype=float)
    return [float(m[3, 0]), float(m[3, 1]), float(m[3, 2])]


def summarize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": bool(plan.get("success", False)),
        "source": plan.get("source"),
        "status": str(plan.get("status")),
        "num_positions": len(plan.get("positions", [])),
        "target_pose_world": plan.get("target_pose_world"),
    }
