"""Geometry-based Panda skills executed through native MPlib and physics.

Only observation reads, planner calls and env.step are used after reset. Native
success criteria remain authoritative. No object/robot poses are assigned.
"""
import numpy as np


def _planner(env, debug, vis, speed=.7):
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    return PandaArmMotionPlanningSolver(env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis,
        print_env_info=False, joint_vel_limits=speed, joint_acc_limits=.7)


def _move(planner, pose, refine=0):
    # Plan before execution so a failed segment cannot silently advance stages.
    from robot_stack.adapters.maniskill_planning import PlanningFailure, failed
    result=planner.move_to_pose_with_screw(pose,dry_run=True)
    if failed(result):
        result=planner.move_to_pose_with_RRTConnect(pose,dry_run=True)
    if failed(result):
        raise PlanningFailure('No physical motion plan for expert waypoint')
    return planner.follow_path(result,refine_steps=refine)


def _grasp(planner, env, actor, *, center=None, closing=None):
    import sapien
    from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb, compute_grasp_info_by_obb
    approaching=np.array([0.,0.,-1.])
    if closing is None:
        target=env.agent.tcp.pose.to_transformation_matrix()[0,:3,1].cpu().numpy()
        info=compute_grasp_info_by_obb(get_actor_obb(actor),approaching=approaching,
                                     target_closing=target,depth=.025)
        closing=info['closing']
        if center is None:center=info['center']
    if center is None:center=actor.pose.sp.p
    pose=env.agent.build_grasp_pose(approaching,closing,center)
    planner.open_gripper()
    _move(planner,pose*sapien.Pose([0,0,-.08]))
    _move(planner,pose,refine=8)
    planner.close_gripper()
    _move(planner,sapien.Pose([0,0,.10])*pose,refine=8)
    return pose


def solve_pick_ycb(env,seed=None,debug=False,vis=False):
    """Grasp an observed YCB object and preserve its TCP offset at the goal."""
    import sapien
    env.reset(seed=seed)
    planner=_planner(env,debug,vis)
    native=env.unwrapped
    try:
        if not bool(native.agent.is_grasping(native.obj).item()):
            _grasp(planner,native,native.obj)
        # Use measured TCP-to-object offset after lifting; irregular objects do
        # not generally have their center at the gripper's grasp point.
        tcp=native.agent.tcp.pose.sp
        offset=native.goal_site.pose.sp.p-native.obj.pose.sp.p
        return _move(planner,sapien.Pose(tcp.p+offset,tcp.q),refine=25)
    finally:
        planner.close()


def solve_poke_cube(env,seed=None,debug=False,vis=False):
    """Pick the peg, align its tip behind the cube, and physically push."""
    import sapien
    env.reset(seed=seed)
    planner=_planner(env,debug,vis,speed=.5)
    native=env.unwrapped
    try:
        if not bool(native.agent.is_grasping(native.peg).item()):
            # A top-down grasp closes across the thin peg, not along its length.
            qmat=native.peg.pose.sp.to_transformation_matrix()[:3,:3]
            _grasp(planner,native,native.peg,center=native.peg.pose.sp.p,closing=qmat[:,1])
        tcp=native.agent.tcp.pose.sp
        tip=native.peg_head_pose.sp.p
        cube=native.cube.pose.sp.p
        goal=native.goal_region.pose.sp.p
        direction=goal[:2]-cube[:2];direction/=max(np.linalg.norm(direction),1e-8)
        # The native peg begins along +x. Rotate the held peg toward the target.
        from transforms3d.euler import euler2quat
        angle=np.arctan2(direction[1],direction[0])
        peg_yaw=np.arctan2(native.peg.pose.sp.to_transformation_matrix()[1,0],native.peg.pose.sp.to_transformation_matrix()[0,0])
        rotation=sapien.Pose(q=euler2quat(0,0,angle-peg_yaw))
        oriented=rotation*sapien.Pose(q=tcp.q)
        _move(planner,sapien.Pose(tcp.p,oriented.q),refine=5)
        tcp=native.agent.tcp.pose.sp;tip=native.peg_head_pose.sp.p
        behind=np.r_[cube[:2]-direction*(native.cube_half_size+.012),native.peg_half_width]
        target_tcp=tcp.p+(behind-tip)
        _move(planner,sapien.Pose(target_tcp+[0,0,.08],tcp.q))
        _move(planner,sapien.Pose(target_tcp,tcp.q),refine=8)
        travel=max(0.,np.linalg.norm(goal[:2]-native.cube.pose.sp.p[:2])-.012)
        result=_move(planner,sapien.Pose(target_tcp+np.r_[direction*travel,0],tcp.q),refine=30)
        return result
    finally:
        planner.close()


EXPERTS={'PickSingleYCB-v1':solve_pick_ycb,'PokeCube-v1':solve_poke_cube}
EXPERT_SPECS={name:dict(provider='robot_stack',skill_family=family,
    status='implemented_requires_native_validation',robot='Panda family',
    native_success_oracle=True,privileged_state=True)
    for name,family in [('PickSingleYCB-v1','grasp_lift_transport'),('PokeCube-v1','grasp_tool_align_push')]}
