"""Opt-in native MPlib planner portfolio; never changes simulator state."""
from numbers import Integral
from types import FunctionType


class PlanningFailure(RuntimeError):
    """No candidate plan; abandon this script instead of executing later stages."""


def failed(result):
    return isinstance(result, Integral) and result == -1


def with_planner_fallback(solve, diagnostics):
    """Clone expert globals so the installed upstream planner is not patched.

    Screw interpolation is preferred. RRTConnect is attempted only when it
    fails. Dry runs stay dry, and a failed execution request stops the script.
    The adapter owns bounded replanning and every physical control step.
    """
    namespace = dict(solve.__globals__)
    for name in ('PandaArmMotionPlanningSolver', 'PandaStickMotionPlanningSolver'):
        base = namespace.get(name)
        if base is None:
            continue

        def make_solver(parent):
            class PortfolioSolver(parent):
                def move_to_pose_with_screw(self, pose, dry_run=False, refine_steps=0):
                    diagnostics['screw_requests'] += 1
                    result = super().move_to_pose_with_screw(pose, dry_run=True)
                    if failed(result):
                        diagnostics['rrt_requests'] += 1
                        result = super().move_to_pose_with_RRTConnect(pose, dry_run=True)
                        if not failed(result):
                            diagnostics['rrt_successes'] += 1
                    if failed(result):
                        diagnostics['failed_requests'] += 1
                        if dry_run:
                            return -1
                        raise PlanningFailure('Screw and RRTConnect both failed')
                    if dry_run:
                        return result
                    return self.follow_path(result, refine_steps=refine_steps)
            return PortfolioSolver

        namespace[name] = make_solver(base)
    clone = FunctionType(solve.__code__, namespace, solve.__name__,
                         solve.__defaults__, solve.__closure__)
    clone.__kwdefaults__ = solve.__kwdefaults__
    return clone


def new_diagnostics():
    return dict(screw_requests=0, rrt_requests=0, rrt_successes=0,
                failed_requests=0, aborted_scripts=0)
