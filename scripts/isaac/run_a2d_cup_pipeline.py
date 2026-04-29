from isaac_collector.ipc.graspnet_client import GraspNetClient
from isaac_collector.ipc.curobo_client import CuroboClient


def main():
    # 1. Isaac Lab 加载场景和机器人
    # scene = load_scene(...)
    # robot = RobotAdapter(...)

    # 2. Isaac 渲染 RGB-D，保存到 runtime
    rgb_path = "/home/pm/Desktop/Project/robot_stack/runtime/results/rgb.png"
    depth_path = "/home/pm/Desktop/Project/robot_stack/runtime/results/depth.npy"

    camera_intrinsics = {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
    }

    camera_to_world = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # 3. 调用 GraspNet 环境
    grasp_client = GraspNetClient(conda_env="graspnet_env")
    grasp_result = grasp_client.predict(
        rgb_path=rgb_path,
        depth_path=depth_path,
        camera_intrinsics=camera_intrinsics,
        camera_to_world=camera_to_world,
    )

    best_grasp = grasp_result["grasps"][0]
    goal_ee_pose = best_grasp["pose_world"]

    # 4. 从 Isaac 读取机器人当前关节
    joint_names = [
        # TODO: robot.get_arm_joint_names()
    ]

    start_joint_positions = [
        # TODO: robot.get_joint_positions(joint_names)
    ]

    # 5. 调用 cuRobo 环境
    curobo_client = CuroboClient(conda_env="curobo_env")
    motion_result = curobo_client.plan(
        robot_config_path="/home/pm/Desktop/Project/robot_stack/isaac_collector/configs/curobo/a2d_right_arm.yml",
        joint_names=joint_names,
        start_joint_positions=start_joint_positions,
        goal_ee_pose=goal_ee_pose,
        obstacles=[],
    )

    trajectory = motion_result["trajectory"]

    # 6. Isaac 执行轨迹
    # robot.execute_joint_trajectory(joint_names, trajectory)


if __name__ == "__main__":
    main()