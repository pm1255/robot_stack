"""Simulator-independent checks shared by collection and regression tests."""
from __future__ import annotations

import numpy as np


def sample_trajectory(positions, stride: int):
    """Subsample without dropping the goal waypoint."""
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or not all(q.shape) or not np.isfinite(q).all():
        raise ValueError("Trajectory must be a nonempty finite [steps, joints] array")
    if stride < 1:
        raise ValueError("Trajectory stride must be positive")
    indices = list(range(0, len(q), stride))
    if indices[-1] != len(q) - 1:
        indices.append(len(q) - 1)
    return q[indices]


def read_robot_state(controller):
    """Read measured DOF positions, in the planner's joint order."""
    names = list(controller["curobo_joint_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("Planner joint names must be nonempty and unique")
    dc, mapping = controller["dc"], controller["joint_mapping"]
    positions = [float(dc.get_dof_position(mapping[name])) for name in names]
    if not np.isfinite(positions).all():
        raise ValueError("Simulator returned non-finite joint positions")
    return {"joint_names": names, "positions": positions}


def terminal_robot_state(plan, expected_names):
    """Predicted state for planning-only chaining; never a measured state."""
    if not plan.get("success"):
        raise ValueError("Cannot chain a failed plan")
    names = list(plan.get("joint_names", []))
    q = sample_trajectory(plan.get("positions", []), 1)
    if len(names) != q.shape[1] or len(set(names)) != len(names):
        raise ValueError("Plan joint names do not match its trajectory")
    by_name = dict(zip(names, q[-1].tolist()))
    return {"joint_names": list(expected_names),
            "positions": [by_name[name] for name in expected_names]}


def classify_outcome(*, planning_success, execution_success, plan_only=False,
                     debug=False, task_success=None):
    """Keep planning, execution and a task oracle's verdict separate.

    None means no task oracle was evaluated. A planner or a debug attachment
    can never certify a successful physical demonstration.
    """
    verified = (not plan_only and not debug and bool(planning_success)
                and bool(execution_success) and task_success is True)
    if plan_only:
        status = "planning_only"
    elif debug:
        status = "debug_only"
    elif not planning_success:
        status = "planning_failed"
    elif not execution_success:
        status = "execution_failed"
    elif task_success is None:
        status = "task_unverified"
    else:
        status = "success" if verified else "task_failed"
    return {"success": bool(verified), "status": status,
            "planning_success": bool(planning_success),
            "execution_success": bool(execution_success) and not plan_only,
            "task_success": task_success, "debug": bool(debug),
            "validation_version": 1}
