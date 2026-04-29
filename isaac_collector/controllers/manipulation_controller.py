"""
manipulation_controller.py

Generic manipulation controller.

It is object-agnostic:
    controller.pick("cup")
    controller.pick("bottle")
    controller.move_object("apple", target_pose_w)

It is backend-agnostic:
    grasp_generator can be rule-based, GraspNet, AnyGrasp, etc.
    motion_planner can be manual waypoints, CuRobo, Differential IK, etc.

The first runnable mode is grasp_mode="attach":
    - Move robot according to provided motion planner.
    - Close gripper.
    - Attach object to EE in simulation.
    - Carry object by updating object root pose to EE pose + offset.
    - Detach on place.

This mode is for pipeline debugging, not for physically realistic grasping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class ManipulationController:
    def __init__(
        self,
        *,
        loaded,
        robot_adapter,
        grasp_generator,
        motion_planner,
        grasp_mode: str = "attach",
        carry_offset=(0.0, 0.0, -0.08),
    ):
        if grasp_mode not in ("attach", "physics"):
            raise ValueError("grasp_mode must be 'attach' or 'physics'.")

        self.loaded = loaded
        self.sim = loaded.sim
        self.scene = loaded.scene
        self.robot = loaded.robot
        self.objects = loaded.rigid_objects

        self.robot_adapter = robot_adapter
        self.grasp_generator = grasp_generator
        self.motion_planner = motion_planner
        self.grasp_mode = grasp_mode

        self.attached_object_name: Optional[str] = None
        self.attached_object = None
        self.carry_offset = torch.tensor(
            [carry_offset],
            dtype=torch.float32,
            device=self.robot.data.joint_pos.device,
        )

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self, *, steps: int = 1) -> None:
        sim_dt = self.sim.get_physics_dt()

        for _ in range(steps):
            if self.attached_object is not None:
                self._update_attached_object_pose()

            self.scene.write_data_to_sim()
            self.robot.write_data_to_sim()
            for obj in self.objects.values():
                obj.write_data_to_sim()

            self.sim.step()

            self.scene.update(sim_dt)
            self.robot.update(sim_dt)
            for obj in self.objects.values():
                obj.update(sim_dt)

    # ------------------------------------------------------------------
    # Object interface
    # ------------------------------------------------------------------

    def list_objects(self) -> List[str]:
        return list(self.objects.keys())

    def print_objects(self) -> None:
        print("\n[OBJECTS] registered rigid objects:")
        if not self.objects:
            print("  <none>")
            return
        for name in self.list_objects():
            path = self.loaded.object_prim_paths_env0.get(name, "<unknown path>")
            pos, quat = self.get_object_pose(name)
            p = pos[0].detach().cpu().tolist()
            print(f"  - {name}: {path}, pos={p}")

    def get_object(self, object_name: str):
        if object_name not in self.objects:
            raise KeyError(
                f"Unknown object_name='{object_name}'. "
                f"Available objects: {self.list_objects()}"
            )
        return self.objects[object_name]

    def get_object_pose(self, object_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = self.get_object(object_name)
        pos = obj.data.root_pos_w.clone()
        quat = obj.data.root_quat_w.clone()
        return pos, quat

    def get_object_pose7(self, object_name: str) -> torch.Tensor:
        pos, quat = self.get_object_pose(object_name)
        return torch.cat([pos, quat], dim=-1)

    # ------------------------------------------------------------------
    # Robot / motion interface
    # ------------------------------------------------------------------

    def move_ee_to_pose(self, target_pose_w: torch.Tensor, *, steps_per_point: int = 8) -> None:
        robot_state = self.robot_adapter.get_robot_state()
        trajectory = self.motion_planner.plan_to_pose(
            robot_state=robot_state,
            target_pose_w=target_pose_w,
        )
        self.robot_adapter.follow_joint_trajectory(
            trajectory,
            step_fn=self.step,
            steps_per_point=steps_per_point,
        )

    def open_gripper(self, *, settle_steps: int = 60) -> None:
        self.robot_adapter.open_gripper()
        self.step(steps=settle_steps)

    def close_gripper(self, *, settle_steps: int = 60) -> None:
        self.robot_adapter.close_gripper()
        self.step(steps=settle_steps)

    # ------------------------------------------------------------------
    # Grasp / attach interface
    # ------------------------------------------------------------------

    def attach_object(self, object_name: str) -> None:
        self.attached_object_name = object_name
        self.attached_object = self.get_object(object_name)
        print(f"[MANIP] attached object: {object_name}")

    def detach_object(self) -> None:
        print(f"[MANIP] detached object: {self.attached_object_name}")
        self.attached_object_name = None
        self.attached_object = None

    def _update_attached_object_pose(self) -> None:
        ee_pos, ee_quat = self.robot_adapter.get_ee_pose_w()

        obj_pos = ee_pos + self.carry_offset
        obj_quat = ee_quat
        root_pose = torch.cat([obj_pos, obj_quat], dim=-1)

        self.attached_object.write_root_pose_to_sim(root_pose)

        # Keep carried object stable in debug attach mode.
        if hasattr(self.attached_object.data, "root_vel_w"):
            zero_vel = torch.zeros_like(self.attached_object.data.root_vel_w)
            self.attached_object.write_root_velocity_to_sim(zero_vel)

    def select_grasp(self, candidates):
        if not candidates:
            raise RuntimeError("No grasp candidates generated.")
        return sorted(candidates, key=lambda x: x.score, reverse=True)[0]

    def compute_pre_grasp_pose(self, grasp_pose_w: torch.Tensor, *, retreat_distance: float = 0.10) -> torch.Tensor:
        # Minimal placeholder: approach from above by increasing z.
        pose = grasp_pose_w.clone()
        pose[:, 2] += retreat_distance
        return pose

    def compute_lift_pose(self, grasp_pose_w: torch.Tensor, *, lift_height: float = 0.20) -> torch.Tensor:
        pose = grasp_pose_w.clone()
        pose[:, 2] += lift_height
        return pose

    # ------------------------------------------------------------------
    # High-level primitives
    # ------------------------------------------------------------------

    def pick(self, object_name: str) -> None:
        object_pose_w = self.get_object_pose7(object_name)

        candidates = self.grasp_generator.generate(
            object_name=object_name,
            object_pose_w=object_pose_w,
            loaded=self.loaded,
        )
        grasp = self.select_grasp(candidates)
        grasp_pose_w = grasp.pose_w
        pre_grasp_pose_w = self.compute_pre_grasp_pose(grasp_pose_w)

        print(f"[MANIP] pick('{object_name}') using grasp source={grasp.source}, score={grasp.score}")

        # Move to pre-grasp and grasp. With ManualJointPlanner these calls just replay
        # user-provided waypoints; with CuRobo they should become real pose plans.
        self.move_ee_to_pose(pre_grasp_pose_w)
        self.move_ee_to_pose(grasp_pose_w)

        self.close_gripper()

        if self.grasp_mode == "attach":
            self.attach_object(object_name)
        else:
            print("[MANIP] physics grasp mode: relying on real contact/friction.")

        lift_pose_w = self.compute_lift_pose(grasp_pose_w)
        self.move_ee_to_pose(lift_pose_w)

    def place(self, object_name: str, target_pose_w: torch.Tensor) -> None:
        if target_pose_w.ndim == 1:
            target_pose_w = target_pose_w.view(1, 7)

        pre_place_pose_w = self.compute_pre_grasp_pose(target_pose_w)
        print(f"[MANIP] place('{object_name}')")

        self.move_ee_to_pose(pre_place_pose_w)
        self.move_ee_to_pose(target_pose_w)

        if self.grasp_mode == "attach":
            self.detach_object()

        self.open_gripper()

    def move_object(self, object_name: str, target_pose_w: torch.Tensor) -> None:
        self.pick(object_name)
        self.place(object_name, target_pose_w)
