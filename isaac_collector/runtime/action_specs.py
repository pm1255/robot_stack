from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class AtomicAction:
    """One executable atomic action.

    Current supported actions:
      - pick: target_class required
      - place: offset_robot or target_robot required
    """
    name: str
    target_class: Optional[str] = None
    target_path: Optional[str] = None
    offset_robot: Optional[List[float]] = None
    target_robot: Optional[List[float]] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    instruction: str
    actions: List[AtomicAction]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "actions": [asdict(a) for a in self.actions],
            "metadata": self.metadata,
        }


def load_task_plan(path: str | Path) -> TaskPlan:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    actions = [AtomicAction(**x) for x in data["actions"]]
    return TaskPlan(
        instruction=data.get("instruction", ""),
        actions=actions,
        metadata=data.get("metadata", {}),
    )


def save_task_plan(path: str | Path, plan: TaskPlan) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")


def default_pick_place_plan(
    *,
    target_class: str,
    place_offset_robot: List[float],
    instruction: str = "",
    target_path: str | None = None,
) -> TaskPlan:
    return TaskPlan(
        instruction=instruction or f"pick up the {target_class} and place by offset {place_offset_robot}",
        actions=[
            AtomicAction(name="pick", target_class=target_class, target_path=target_path),
            AtomicAction(name="place", target_class=target_class, offset_robot=[float(x) for x in place_offset_robot]),
        ],
        metadata={"source": "default_pick_place_plan"},
    )


def validate_pick_place_plan(plan: TaskPlan) -> None:
    names = [a.name for a in plan.actions]
    if names != ["pick", "place"]:
        raise ValueError(
            "This first scalable entrypoint currently executes exactly [pick, place]. "
            f"Got actions={names}. Add new action handlers in ManipulationExecutor for more actions."
        )
    if not plan.actions[0].target_class and not plan.actions[0].target_path:
        raise ValueError("pick action requires target_class or target_path.")
    place = plan.actions[1]
    if place.offset_robot is None and place.target_robot is None:
        raise ValueError("place action requires offset_robot or target_robot.")
