"""
scene_loader.py

A reusable scene-loading utility for Isaac Lab.

Main idea:
1. Map a logical scene name, e.g. "mutilrooms", to a USD asset path.
2. Import that USD scene into /World/envs/env_i/Scene.
3. Automatically scan and register task-level interactive objects as RigidObject.
4. Register or spawn a specified robot model, e.g. "a2d".
5. Return a LoadedScene object that exposes sim, scene, robot, and rigid_objects.

Important:
- This file intentionally avoids importing Isaac Lab / omni modules at top level.
- Import and call this module after AppLauncher has created simulation_app.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCENE_LOADER_VERSION = "fixed_logdir_joints_v5_leaf_filter"


# --------------------------------------------------------------------------------------
# Registry specs
# --------------------------------------------------------------------------------------


@dataclass
class SceneSpec:
    """Description of a reusable scene asset."""

    name: str
    usd_path: str

    # Relative folders under /World/envs/env_0/Scene that should contain task objects.
    # If these folders do not exist, the loader falls back to scanning the whole scene
    # with keyword filters.
    task_object_roots: Tuple[str, ...] = (
        "TaskObjects",
        "Objects",
        "Interactables",
        "Props",
        "SmallObjects",
    )

    # Used when falling back to scanning broad scene roots.
    include_keywords: Tuple[str, ...] = (
        "apple",
        "banana",
        "orange",
        "fruit",
        "cup",
        "mug",
        "bottle",
        "plate",
        "bowl",
        "box",
        "cube",
        "can",
        "tool",
        "screw",
        "block",
    )

    exclude_keywords: Tuple[str, ...] = (
        # room / structure
        "wall",
        "floor",
        "ceiling",
        "roof",
        "window",
        "light",
        "lamp",

        # large furniture
        "table",
        "desk",
        "chair",
        "sofa",
        "bed",
        "cabinet",
        "shelf",
        "wardrobe",
        "counter",

        # articulated or joint-based objects
        "door",
        "drawer",
        "hinge",
        "handle",

        # robot-related
        "robot",
        "agibot",
        "a2d",
        "link",
        "joint",
    )


@dataclass
class RobotSpec:
    """Description of a supported robot model."""

    name: str

    # Python module and attribute name for the Isaac Lab robot cfg.
    cfg_module: str
    cfg_name: str

    # Keywords used to detect an already-existing articulation inside the scene USD.
    match_keywords: Tuple[str, ...] = ("robot",)

    # Where to spawn the robot if it is not already inside the imported USD.
    # This is converted to /World/envs/env_0/Robot or /World/envs/env_.*/Robot.
    default_spawn_path: str = "{ENV_REGEX_NS}/Robot"

    # Optional joint-name aliases for assets whose USD joint names differ slightly
    # from the joint names expected by the robot config.
    joint_name_aliases: Tuple[Tuple[str, str], ...] = tuple()

    # Optional regex-expression aliases for actuator configs. These are needed
    # when Isaac Lab's actuator cfg uses regexes such as left_.*_RevoluteJoint,
    # while the USD export uses left_.*_2_Joint.
    joint_regex_aliases: Tuple[Tuple[str, str], ...] = tuple()


@dataclass
class LoadedScene:
    """Returned by load_scene_with_robot."""

    sim: Any
    scene: Any
    robot: Any
    rigid_objects: Dict[str, Any]

    scene_name: str
    scene_usd_path: str
    robot_model: str
    robot_prim_path: str
    robot_was_spawned: bool

    object_prim_paths_env0: Dict[str, str] = field(default_factory=dict)
    articulation_roots_env0: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Default registries
# --------------------------------------------------------------------------------------


DEFAULT_SCENE_REGISTRY: Dict[str, SceneSpec] = {
    # Keep the user's historical typo/name for compatibility.
    # You can also add "multirooms" as an alias below.
    "mutilrooms": SceneSpec(
        name="mutilrooms",
        usd_path="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001.usd",
    ),
    "mutilrooms_fixed": SceneSpec(
        name="mutilrooms_fixed",
        usd_path="/home/pm/Desktop/Project/house_type_usd/mutil_room/mutil_room001_fixed.usd",
    ),
}


DEFAULT_ROBOT_REGISTRY: Dict[str, RobotSpec] = {
    "a2d": RobotSpec(
        name="a2d",
        cfg_module="isaaclab_assets.robots.agibot",
        cfg_name="AGIBOT_A2D_CFG",
        match_keywords=("agibot", "a2d", "robot"),
        default_spawn_path="{ENV_REGEX_NS}/Robot",
        # Some A2D USD exports use *_2_Joint for the gripper revolute joints,
        # while AGIBOT_A2D_CFG expects *_RevoluteJoint. Remap the config to
        # match the USD asset if those target names exist in the stage.
        joint_name_aliases=(
            ("left_Right_RevoluteJoint", "left_Right_2_Joint"),
            ("left_Left_RevoluteJoint", "left_Left_2_Joint"),
            ("right_Right_RevoluteJoint", "right_Right_2_Joint"),
            ("right_Left_RevoluteJoint", "right_Left_2_Joint"),
        ),
        joint_regex_aliases=(
            ("left_.*_RevoluteJoint", "left_.*_2_Joint"),
            ("right_.*_RevoluteJoint", "right_.*_2_Joint"),
        ),
    ),
    "franka": RobotSpec(
        name="franka",
        cfg_module="isaaclab_assets.robots.franka",
        cfg_name="FRANKA_PANDA_CFG",
        match_keywords=("franka", "panda", "robot"),
        default_spawn_path="{ENV_REGEX_NS}/Robot",
    ),
}


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def register_scene_spec(scene_spec: SceneSpec, registry: Optional[Dict[str, SceneSpec]] = None) -> Dict[str, SceneSpec]:
    """Register a new scene spec and return the registry."""
    target = registry if registry is not None else DEFAULT_SCENE_REGISTRY
    target[scene_spec.name] = scene_spec
    return target


def register_robot_spec(robot_spec: RobotSpec, registry: Optional[Dict[str, RobotSpec]] = None) -> Dict[str, RobotSpec]:
    """Register a new robot spec and return the registry."""
    target = registry if registry is not None else DEFAULT_ROBOT_REGISTRY
    target[robot_spec.name] = robot_spec
    return target


def resolve_scene_spec(scene_name_or_usd: str, registry: Optional[Dict[str, SceneSpec]] = None) -> SceneSpec:
    """Resolve a logical scene name or direct USD path into a SceneSpec."""
    scene_registry = registry or DEFAULT_SCENE_REGISTRY

    if scene_name_or_usd in scene_registry:
        return scene_registry[scene_name_or_usd]

    # Allow passing a direct USD path for debugging or ad-hoc use.
    if scene_name_or_usd.endswith(".usd") or scene_name_or_usd.endswith(".usda") or scene_name_or_usd.endswith(".usdc"):
        name = os.path.splitext(os.path.basename(scene_name_or_usd))[0]
        return SceneSpec(name=name, usd_path=scene_name_or_usd)

    available = ", ".join(sorted(scene_registry.keys()))
    raise KeyError(
        f"Unknown scene '{scene_name_or_usd}'. Available scene names: {available}. "
        f"You may also pass a direct .usd/.usda/.usdc path."
    )


def resolve_robot_spec(robot_model: str, registry: Optional[Dict[str, RobotSpec]] = None) -> RobotSpec:
    """Resolve a robot model name into a RobotSpec."""
    robot_registry = registry or DEFAULT_ROBOT_REGISTRY

    if robot_model in robot_registry:
        return robot_registry[robot_model]

    available = ", ".join(sorted(robot_registry.keys()))
    raise KeyError(f"Unknown robot model '{robot_model}'. Available robot models: {available}.")


def import_robot_cfg(robot_spec: RobotSpec) -> Any:
    """Lazily import a robot cfg object."""
    module = importlib.import_module(robot_spec.cfg_module)
    return getattr(module, robot_spec.cfg_name)


def is_under_path(path: str, root: str) -> bool:
    """Return True if path is root itself or under root."""
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


def env_regex_ns(num_envs: int) -> str:
    """Return the namespace expression used to address one or many cloned envs."""
    if num_envs <= 1:
        return "/World/envs/env_0"
    return "/World/envs/env_.*"


def to_env_regex_path(path: str, num_envs: int) -> str:
    """Convert an env_0 path into an env_.* path if num_envs > 1."""
    if num_envs <= 1:
        return path
    return path.replace("/env_0/", "/env_.*/")


def scene_base_path_env0() -> str:
    """The default env_0 scene path used by this loader."""
    return "/World/envs/env_0/Scene"


def safe_object_name_from_path(path: str, existing_names: set[str]) -> str:
    """Convert a USD prim path into a safe dictionary key."""
    base = path.rstrip("/").split("/")[-1]
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", base)

    if not base:
        base = "object"

    if base[0].isdigit():
        base = f"obj_{base}"

    name = base
    idx = 1
    while name in existing_names:
        name = f"{base}_{idx}"
        idx += 1

    existing_names.add(name)
    return name


def _as_tuple_or_none(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if values is None:
        return None
    return tuple(v for v in values if v)


# --------------------------------------------------------------------------------------
# USD scanning utilities
# --------------------------------------------------------------------------------------


def get_stage() -> Any:
    """Get current USD stage. Must be called after Isaac Sim app launch."""
    import omni.usd

    return omni.usd.get_context().get_stage()


def prim_exists(path: str) -> bool:
    """Check whether a prim exists in the current USD stage."""
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def prim_has_collision_under(prim: Any) -> bool:
    """Check whether this prim or any descendant has CollisionAPI."""
    from pxr import Usd, UsdPhysics

    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


def rigid_body_enabled(prim: Any) -> bool:
    """Return whether a prim's RigidBodyAPI is enabled.

    If the attribute is not authored, treat it as enabled. This is the useful
    default for most USD assets.
    """
    from pxr import UsdPhysics

    rb_api = UsdPhysics.RigidBodyAPI(prim)
    attr = rb_api.GetRigidBodyEnabledAttr()
    value = attr.Get() if attr else None
    return value is not False


def rigid_body_is_kinematic(prim: Any) -> bool:
    """Return whether a rigid body is kinematic.

    Kinematic bodies are usually not suitable as task objects for physical grasping,
    so the scanner can exclude them.
    """
    from pxr import UsdPhysics

    rb_api = UsdPhysics.RigidBodyAPI(prim)
    attr = rb_api.GetKinematicEnabledAttr()
    value = attr.Get() if attr else None
    return value is True


def find_articulation_roots_under(prefix: str) -> List[str]:
    """Find all articulation-root prims under a prefix."""
    from pxr import UsdPhysics

    stage = get_stage()
    candidates: List[str] = []

    for prim in stage.Traverse():
        path = prim.GetPath().pathString

        if not is_under_path(path, prefix):
            continue

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(path)

    return candidates


def find_usd_joint_names_under(root_path: str) -> List[str]:
    """Find USD physics joint names around an articulation root.

    Some exported robots put ArticulationRootAPI on a prim named root_joint,
    while the real robot joints are not descendants of that prim. In that case,
    scanning only root_path returns just ["root_joint"]. We therefore also scan
    the parent prim, which is usually the robot root, e.g. /.../A2D.
    """
    from pxr import Usd, UsdPhysics

    stage = get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return []

    scan_roots = [root_prim]

    parent = root_prim.GetParent()
    if parent and parent.IsValid():
        scan_roots.append(parent)

    joint_names: List[str] = []
    seen = set()

    for scan_root in scan_roots:
        for prim in Usd.PrimRange(scan_root):
            name = prim.GetName()
            try:
                # Most physics joint prims should satisfy this.
                is_joint = prim.IsA(UsdPhysics.Joint)
            except Exception:
                is_joint = False

            # Fallback: exported robot joints often contain "joint" in the prim name.
            # This is only for diagnostics and alias validation, not for final physics.
            looks_like_joint = "joint" in name.lower()

            if is_joint or looks_like_joint:
                if name not in seen:
                    seen.add(name)
                    joint_names.append(name)

    return joint_names


def _replace_cfg_field(cfg_obj: Any, **kwargs) -> Any:
    """Replace fields on an Isaac Lab configclass object.

    Most Isaac Lab config classes support .replace(...). If not, fall back to
    in-place assignment.
    """
    if hasattr(cfg_obj, "replace"):
        return cfg_obj.replace(**kwargs)

    for key, value in kwargs.items():
        setattr(cfg_obj, key, value)
    return cfg_obj


def _remap_joint_expr_list(
    joint_exprs: Sequence[str],
    *,
    alias_map: Dict[str, str],
    regex_alias_map: Dict[str, str],
    available_joint_names: Sequence[str],
) -> List[str]:
    """Remap actuator joint expressions using exact-name and regex aliases.

    This handles both:
    - exact names, e.g. left_Right_RevoluteJoint -> left_Right_2_Joint
    - regexes, e.g. left_.*_RevoluteJoint -> left_.*_2_Joint
    """
    remapped: List[str] = []

    for expr in joint_exprs:
        if expr in alias_map:
            remapped.append(alias_map[expr])
            continue

        if expr in regex_alias_map:
            remapped.append(regex_alias_map[expr])
            continue

        # Generic fallback for A2D-like exported gripper naming.
        # This is deliberately conservative: only rewrite expressions that
        # explicitly mention RevoluteJoint.
        if "RevoluteJoint" in expr:
            expr = expr.replace("RevoluteJoint", "2_Joint")

        remapped.append(expr)

    return remapped



def force_patch_a2d_actuator_joint_exprs(robot_cfg: Any, *, verbose: bool = True) -> Any:
    """Force-patch A2D actuator joint expressions in-place.

    This is intentionally more direct than config.replace(...), because some
    Isaac Lab cfg objects may keep nested actuator cfg objects by reference.
    """
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        if verbose:
            print(f"[WARN] robot_cfg.actuators is not a dict: {type(actuators)}")
        return robot_cfg

    for actuator_name, actuator_cfg in actuators.items():
        joint_exprs = getattr(actuator_cfg, "joint_names_expr", None)
        if joint_exprs is None:
            continue

        old_exprs = list(joint_exprs)
        new_exprs = []

        for expr in old_exprs:
            # Exact exported A2D gripper names.
            expr = expr.replace("left_Right_RevoluteJoint", "left_Right_2_Joint")
            expr = expr.replace("left_Left_RevoluteJoint", "left_Left_2_Joint")
            expr = expr.replace("right_Right_RevoluteJoint", "right_Right_2_Joint")
            expr = expr.replace("right_Left_RevoluteJoint", "right_Left_2_Joint")

            # Regex patterns in AGIBOT_A2D_CFG.
            expr = expr.replace("left_.*_RevoluteJoint", "left_.*_2_Joint")
            expr = expr.replace("right_.*_RevoluteJoint", "right_.*_2_Joint")

            # Conservative generic fallback.
            if "RevoluteJoint" in expr:
                expr = expr.replace("RevoluteJoint", "2_Joint")

            new_exprs.append(expr)

        if old_exprs != new_exprs:
            if verbose:
                print(f"[INFO] FORCE remap actuator '{actuator_name}' joint_names_expr:")
                print(f"       old: {old_exprs}")
                print(f"       new: {new_exprs}")

            # Direct in-place assignment. This is the important part.
            try:
                actuator_cfg.joint_names_expr = new_exprs
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Direct assignment failed for actuator '{actuator_name}': {exc}")

            # Also update dict entry in case actuator_cfg.replace(...) is required.
            try:
                actuators[actuator_name] = _replace_cfg_field(
                    actuator_cfg,
                    joint_names_expr=new_exprs,
                )
            except Exception as exc:
                if verbose:
                    print(f"[WARN] replace() failed for actuator '{actuator_name}': {exc}")

    try:
        robot_cfg.actuators = actuators
    except Exception:
        try:
            robot_cfg = _replace_cfg_field(robot_cfg, actuators=actuators)
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to assign patched actuators back to robot_cfg: {exc}")

    return robot_cfg


def adapt_robot_cfg_to_existing_usd_joints(
    robot_cfg: Any,
    *,
    robot_spec: RobotSpec,
    robot_prim_env0: Optional[str],
    verbose: bool = True,
) -> Any:
    """Patch robot cfg joint names to match an already-existing USD robot.

    This is needed when the robot asset inside the scene was exported with
    slightly different joint names from the Isaac Lab asset config.
    """
    if not robot_prim_env0:
        return robot_cfg

    available_joint_names = find_usd_joint_names_under(robot_prim_env0)
    if not available_joint_names:
        if verbose:
            print("[WARN] No USD physics joints found under robot prim; robot cfg was not adapted.")
        return robot_cfg

    alias_map = dict(robot_spec.joint_name_aliases)
    regex_alias_map = dict(robot_spec.joint_regex_aliases)

    if verbose:
        print("[INFO] USD robot joints detected:")
        for name in available_joint_names:
            print(f"  - {name}")

    # 1. Patch initial joint positions if present.
    try:
        init_state = getattr(robot_cfg, "init_state", None)
        joint_pos = getattr(init_state, "joint_pos", None) if init_state is not None else None

        if isinstance(joint_pos, dict):
            new_joint_pos = {}
            for key, value in joint_pos.items():
                new_key = alias_map.get(key, key)
                if new_key != key and verbose:
                    print(f"[INFO] remap init_state joint_pos: {key} -> {new_key}")
                new_joint_pos[new_key] = value

            new_init_state = _replace_cfg_field(init_state, joint_pos=new_joint_pos)
            robot_cfg = _replace_cfg_field(robot_cfg, init_state=new_init_state)
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to adapt init_state.joint_pos: {exc}")

    # 2. Patch actuator joint name expressions if present.
    try:
        actuators = getattr(robot_cfg, "actuators", None)

        if isinstance(actuators, dict):
            new_actuators = dict(actuators)

            for actuator_name, actuator_cfg in actuators.items():
                joint_exprs = getattr(actuator_cfg, "joint_names_expr", None)

                if joint_exprs is None:
                    continue

                remapped_exprs = _remap_joint_expr_list(
                    list(joint_exprs),
                    alias_map=alias_map,
                    regex_alias_map=regex_alias_map,
                    available_joint_names=available_joint_names,
                )

                if list(joint_exprs) != remapped_exprs:
                    if verbose:
                        print(f"[INFO] remap actuator '{actuator_name}' joint_names_expr:")
                        print(f"       old: {list(joint_exprs)}")
                        print(f"       new: {remapped_exprs}")

                    new_actuators[actuator_name] = _replace_cfg_field(
                        actuator_cfg,
                        joint_names_expr=remapped_exprs,
                    )

            robot_cfg = _replace_cfg_field(robot_cfg, actuators=new_actuators)

    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to adapt actuator joint_names_expr: {exc}")

    return robot_cfg




def choose_existing_robot_prim(
    articulation_roots: Sequence[str],
    robot_spec: RobotSpec,
    *,
    allow_fallback_to_first: bool = False,
) -> Optional[str]:
    """Choose an existing robot articulation from articulation roots.

    By default, this only returns a path if it matches the robot's keywords.
    This avoids accidentally treating a door/drawer articulation as the robot.
    """
    lower_map = {path: path.lower() for path in articulation_roots}

    for key in robot_spec.match_keywords:
        key_lc = key.lower()
        for path, path_lc in lower_map.items():
            if key_lc in path_lc:
                return path

    if allow_fallback_to_first and articulation_roots:
        return articulation_roots[0]

    return None


def resolve_task_scan_roots(scene_spec: SceneSpec, scene_prefix_env0: str) -> Tuple[List[str], bool]:
    """Resolve candidate task-object roots.

    Returns:
        roots: existing roots to scan.
        roots_are_explicit_task_roots: True if at least one configured task root exists.
    """
    roots: List[str] = []

    for root in scene_spec.task_object_roots:
        if not root:
            continue

        if root.startswith("/"):
            abs_root = root
        else:
            abs_root = f"{scene_prefix_env0}/{root.strip('/')}"

        if prim_exists(abs_root):
            roots.append(abs_root)

    if roots:
        return roots, True

    # Fallback: scan the whole scene with keyword filters.
    return [scene_prefix_env0], False


def find_registerable_rigid_objects(
    scan_roots: Sequence[str],
    *,
    exclude_articulation_roots: Sequence[str],
    include_keywords: Optional[Sequence[str]],
    exclude_keywords: Optional[Sequence[str]],
    require_collision: bool = True,
    exclude_kinematic: bool = True,
    max_objects: Optional[int] = None,
) -> Dict[str, str]:
    """Find rigid bodies that should be registered as task-level RigidObject.

    The function is intentionally conservative:
    - it only accepts prims with RigidBodyAPI;
    - it excludes anything under an articulation root;
    - it can require CollisionAPI;
    - it can use include/exclude keyword filters;
    - it skips kinematic rigid bodies by default.
    """
    from pxr import UsdPhysics

    stage = get_stage()

    include_keywords_lc = tuple(k.lower() for k in include_keywords or ())
    exclude_keywords_lc = tuple(k.lower() for k in exclude_keywords or ())

    result: Dict[str, str] = {}
    used_names: set[str] = set()

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        path_lc = path.lower()
        leaf_name_lc = prim.GetName().lower()

        if not any(is_under_path(path, root) for root in scan_roots):
            continue

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        if not rigid_body_enabled(prim):
            continue

        if exclude_kinematic and rigid_body_is_kinematic(prim):
            continue

        # Exclude robots, doors, drawers, and any other articulation-based object.
        if any(is_under_path(path, art_root) for art_root in exclude_articulation_roots):
            continue

        if require_collision and not prim_has_collision_under(prim):
            continue

        # Important: apply include/exclude filters to the leaf prim name, not the full path.
        # Otherwise a cup under /tea_table/cup is wrongly excluded because its parent path
        # contains "table"; similarly pen/laptop under /office_desk would be excluded by "desk".
        if exclude_keywords_lc and any(k in leaf_name_lc for k in exclude_keywords_lc):
            continue

        # If include_keywords is empty/None, accept everything under explicit task roots.
        # If include_keywords is provided, require at least one keyword match on the object name.
        if include_keywords_lc and not any(k in leaf_name_lc for k in include_keywords_lc):
            continue

        object_name = safe_object_name_from_path(path, used_names)
        result[object_name] = path

        if max_objects is not None and len(result) >= max_objects:
            break

    return result


def freeze_non_task_rigid_bodies(
    *,
    scene_prefix_env0: str,
    task_object_paths_env0: Sequence[str],
    articulation_roots_env0: Sequence[str],
    verbose: bool = True,
) -> None:
    """Freeze selected non-task rigid bodies under the scene.

    Conservative version:
    - Do NOT freeze robot links.
    - Do NOT freeze doors or articulated/joint-based objects.
    - Do NOT freeze task objects such as cup.
    - Only freeze furniture/prop bodies that are likely to be static support objects.
    """
    from pxr import UsdPhysics

    stage = get_stage()

    task_paths = list(task_object_paths_env0)

    # Important:
    # A2D's ArticulationRootAPI is on /A2D/root_joint, while actual links are siblings
    # under /A2D. Therefore excluding only articulation_roots_env0 is not enough.
    robot_parent_paths = set()
    for art_root in articulation_roots_env0:
        if "/A2D/" in art_root or art_root.endswith("/A2D/root_joint"):
            robot_parent_paths.add(art_root.rsplit("/", 1)[0])

    freeze_keywords = (
        "table",
        "desk",
        "counter",
        "shelf",
        "cabinet",
        "sofa",
        "bed",
        "chair",
        "wardrobe",
        "plant",
        "laptop",
        "pen",
    )

    skip_keywords = (
        "/A2D/",
        "/door/",
        "/drawer/",
        "/joints/",
        "/joint",
    )

    frozen_count = 0

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        path_lc = path.lower()

        if not is_under_path(path, scene_prefix_env0):
            continue

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        # Do not freeze registered task objects, e.g. cup.
        if any(is_under_path(path, task_path) or is_under_path(task_path, path) for task_path in task_paths):
            continue

        # Do not freeze robot subtree.
        if any(is_under_path(path, robot_parent) for robot_parent in robot_parent_paths):
            continue

        # Extra protection against robot / joint / articulated objects.
        if any(k.lower() in path_lc for k in skip_keywords):
            continue

        # Only freeze known static-ish furniture/support props.
        if not any(k in path_lc for k in freeze_keywords):
            continue

        rb_api = UsdPhysics.RigidBodyAPI(prim)

        # Kinematic keeps it fixed in simulation while still allowing collision.
        rb_api.CreateKinematicEnabledAttr(True)

        # If CCD exists on this body, disable it because kinematic bodies with CCD
        # produce PhysX warnings/errors.
        try:
            physx_rb_api = UsdPhysics.PhysxRigidBodyAPI(prim)
            if physx_rb_api:
                physx_rb_api.CreateEnableCCDAttr(False)
        except Exception:
            pass

        frozen_count += 1

        if verbose:
            print(f"[INFO] freeze static support rigid body: {path}")

    if verbose:
        print(f"[INFO] frozen static support rigid bodies count: {frozen_count}")


# --------------------------------------------------------------------------------------
# Main loader
# --------------------------------------------------------------------------------------


def make_prebuilt_scene_cfg(
    usd_path: str,
    *,
    add_dome_light: bool = True,
    dome_light_intensity: float = 2000.0,
) -> Any:
    """Create an InteractiveSceneCfg subclass for this USD file."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    @configclass
    class PrebuiltSceneCfg(InteractiveSceneCfg):
        scene_usd = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Scene",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        )

        if add_dome_light:
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=dome_light_intensity,
                    color=(0.75, 0.75, 0.75),
                ),
            )

    return PrebuiltSceneCfg


def load_scene_with_robot(
    *,
    scene_name: str,
    robot_model: str = "a2d",
    num_envs: int = 1,
    env_spacing: float = 6.0,
    device: str = "cuda:0",
    robot_prim: Optional[str] = None,
    spawn_robot_if_missing: bool = True,
    register_task_objects: bool = True,
    task_object_roots: Optional[Sequence[str]] = None,
    include_keywords: Optional[Sequence[str]] = None,
    exclude_keywords: Optional[Sequence[str]] = None,
    require_collision: bool = True,
    exclude_kinematic: bool = True,
    max_objects: Optional[int] = None,
    add_dome_light: bool = True,
    dome_light_intensity: float = 2000.0,
    scene_registry: Optional[Dict[str, SceneSpec]] = None,
    robot_registry: Optional[Dict[str, RobotSpec]] = None,
    log_dir: Optional[str] = None,
    save_logs_to_file: bool = True,
    verbose: bool = True,
) -> LoadedScene:
    """Load a scene USD, register/spawn a robot, and register interactive objects.

    Args:
        scene_name: Logical scene name, e.g. "mutilrooms", or a direct USD path.
        robot_model: Robot model key, e.g. "a2d".
        num_envs: Number of cloned environments.
        env_spacing: Spacing between cloned environments.
        device: Simulation device, e.g. "cuda:0" or "cpu".
        robot_prim: Optional explicit existing robot articulation root in env_0.
        spawn_robot_if_missing: If True, spawn the robot if it is not found in the scene USD.
        register_task_objects: Whether to auto-register task rigid objects.
        task_object_roots: Optional override for task-object folders under the scene.
        include_keywords: Optional override for include keywords.
        exclude_keywords: Optional override for exclude keywords.
        require_collision: Only register rigid bodies with collision.
        exclude_kinematic: Exclude kinematic rigid bodies.
        max_objects: Optional cap on registered object count.
        add_dome_light: Add a dome light if needed.
        dome_light_intensity: Dome light intensity.
        scene_registry: Optional custom scene registry.
        robot_registry: Optional custom robot registry.
        log_dir: Optional user-writable Isaac Lab log directory.
        save_logs_to_file: Whether Isaac Lab should save its own logs to file.
        verbose: Print diagnostic information.

    Returns:
        LoadedScene.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    # Resolve logical names.
    scene_spec = resolve_scene_spec(scene_name, scene_registry)
    robot_spec = resolve_robot_spec(robot_model, robot_registry)
    robot_base_cfg = import_robot_cfg(robot_spec)

    if not os.path.exists(scene_spec.usd_path):
        raise FileNotFoundError(
            f"Scene USD does not exist: {scene_spec.usd_path}\n"
            f"Please update DEFAULT_SCENE_REGISTRY or pass a direct USD path."
        )

    # 1. Create simulator and import prebuilt scene USD.
    # Avoid Isaac Lab's default /tmp/isaaclab/logs path because it may be owned by another user.
    if log_dir is None:
        log_dir = os.path.join(os.path.expanduser("~"), ".cache", "isaaclab", "logs")
    os.makedirs(log_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(
        device=device,
        log_dir=log_dir,
        save_logs_to_file=save_logs_to_file,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        [4.0, -4.2, 2.2],
        [2.8, -1.9, 0.8],
    )

    SceneCfgClass = make_prebuilt_scene_cfg(
        scene_spec.usd_path,
        add_dome_light=add_dome_light,
        dome_light_intensity=dome_light_intensity,
    )
    scene_cfg = SceneCfgClass(num_envs=num_envs, env_spacing=env_spacing)
    scene = InteractiveScene(scene_cfg)

    # Let USD references compose. Useful for large nested/referenced USDs.
    import omni.kit.app

    omni.kit.app.get_app().update()

    scene_prefix_env0 = scene_base_path_env0()

    # 2. Find articulation roots already inside the imported scene.
    articulation_roots_env0 = find_articulation_roots_under(scene_prefix_env0)

    if verbose:
        print("[INFO] articulation roots under imported scene:")
        if articulation_roots_env0:
            for path in articulation_roots_env0:
                print(f"  - {path}")
        else:
            print("  <none>")

    # 3. Register existing robot or spawn a new one.
    robot_was_spawned = False

    if robot_prim:
        robot_prim_env0 = robot_prim
        if not prim_exists(robot_prim_env0):
            raise ValueError(f"Explicit --robot-prim does not exist: {robot_prim_env0}")
    else:
        robot_prim_env0 = choose_existing_robot_prim(
            articulation_roots_env0,
            robot_spec,
            allow_fallback_to_first=False,
        )

    if robot_prim_env0:
        robot_prim_expr = to_env_regex_path(robot_prim_env0, num_envs)
        robot_cfg = robot_base_cfg.replace(
            prim_path=robot_prim_expr,
            spawn=None,  # Existing robot inside the imported USD.
        )
        robot_cfg = adapt_robot_cfg_to_existing_usd_joints(
            robot_cfg,
            robot_spec=robot_spec,
            robot_prim_env0=robot_prim_env0,
            verbose=verbose,
        )
        robot_cfg = force_patch_a2d_actuator_joint_exprs(robot_cfg, verbose=verbose)
        if verbose:
            print(f"[INFO] scene_loader version: {SCENE_LOADER_VERSION}")
            print(f"[INFO] registering existing robot: {robot_prim_expr}")
    else:
        if not spawn_robot_if_missing:
            raise RuntimeError(
                f"No existing robot articulation matching model '{robot_model}' was found "
                f"under {scene_prefix_env0}, and spawn_robot_if_missing=False."
            )

        robot_prim_expr = robot_spec.default_spawn_path.replace("{ENV_REGEX_NS}", env_regex_ns(num_envs))
        robot_cfg = robot_base_cfg.replace(
            prim_path=robot_prim_expr,
            # Keep spawn from the original robot cfg so Isaac Lab spawns the robot asset.
        )
        robot_was_spawned = True
        if verbose:
            print(f"[INFO] no existing robot found; spawning '{robot_model}' at: {robot_prim_expr}")

    robot = Articulation(robot_cfg)

    # 4. Auto-register task rigid objects.
    rigid_objects: Dict[str, Any] = {}
    object_prim_paths_env0: Dict[str, str] = {}

    if register_task_objects:
        # Determine scan roots.
        scan_scene_spec = scene_spec
        if task_object_roots is not None:
            scan_scene_spec = SceneSpec(
                name=scene_spec.name,
                usd_path=scene_spec.usd_path,
                task_object_roots=tuple(task_object_roots),
                include_keywords=scene_spec.include_keywords,
                exclude_keywords=scene_spec.exclude_keywords,
            )

        scan_roots, roots_are_explicit_task_roots = resolve_task_scan_roots(
            scan_scene_spec,
            scene_prefix_env0,
        )

        # If explicit task roots exist, accept all rigid bodies under them unless the caller
        # explicitly supplied include_keywords. If falling back to whole-scene scanning,
        # use conservative include keywords.
        user_include = _as_tuple_or_none(include_keywords)
        user_exclude = _as_tuple_or_none(exclude_keywords)

        if user_include is not None:
            effective_include = user_include
        elif roots_are_explicit_task_roots:
            effective_include = tuple()
        else:
            effective_include = scene_spec.include_keywords

        effective_exclude = user_exclude if user_exclude is not None else scene_spec.exclude_keywords

        # Exclude all articulation roots inside the scene. This avoids registering robot links,
        # door links, drawer bodies, etc. If the user supplied robot_prim, include it too.
        exclude_articulation_roots = list(dict.fromkeys(list(articulation_roots_env0) + ([robot_prim_env0] if robot_prim_env0 else [])))

        object_prim_paths_env0 = find_registerable_rigid_objects(
            scan_roots,
            exclude_articulation_roots=exclude_articulation_roots,
            include_keywords=effective_include,
            exclude_keywords=effective_exclude,
            require_collision=require_collision,
            exclude_kinematic=exclude_kinematic,
            max_objects=max_objects,
        )

        if verbose:
            print("[INFO] task-object scan roots:")
            for root in scan_roots:
                print(f"  - {root}")
            print("[INFO] auto-registerable rigid objects:")
            if object_prim_paths_env0:
                for name, path in object_prim_paths_env0.items():
                    print(f"  - {name}: {path}")
            else:
                print("  <none>")

        for object_name, object_prim_env0 in object_prim_paths_env0.items():
            object_prim_expr = to_env_regex_path(object_prim_env0, num_envs)
            object_cfg = RigidObjectCfg(
                prim_path=object_prim_expr,
                spawn=None,  # Existing rigid body inside the imported USD.
            )
            rigid_objects[object_name] = RigidObject(object_cfg)
        
    # Freeze all non-task rigid bodies such as furniture.
    # freeze_non_task_rigid_bodies(
    #     scene_prefix_env0=scene_prefix_env0,
    #     task_object_paths_env0=list(object_prim_paths_env0.values()),
    #     articulation_roots_env0=articulation_roots_env0,
    #     verbose=verbose,
    # )    

    # 5. Build physics handles.
    sim.reset()

    robot.reset()
    for obj in rigid_objects.values():
        obj.reset()

    if verbose:
        print("[INFO] scene import complete.")
        print(f"[INFO] scene_name      : {scene_spec.name}")
        print(f"[INFO] scene_usd_path  : {scene_spec.usd_path}")
        print(f"[INFO] robot_model     : {robot_model}")
        print(f"[INFO] robot_prim_path : {robot_prim_expr}")
        print(f"[INFO] robot_spawned   : {robot_was_spawned}")
        print(f"[INFO] objects_count   : {len(rigid_objects)}")

    return LoadedScene(
        sim=sim,
        scene=scene,
        robot=robot,
        rigid_objects=rigid_objects,
        scene_name=scene_spec.name,
        scene_usd_path=scene_spec.usd_path,
        robot_model=robot_model,
        robot_prim_path=robot_prim_expr,
        robot_was_spawned=robot_was_spawned,
        object_prim_paths_env0=object_prim_paths_env0,
        articulation_roots_env0=articulation_roots_env0,
    )


def step_loaded_scene(loaded: LoadedScene, *, steps: int = 1) -> None:
    """Step a LoadedScene for a given number of simulation steps."""
    sim_dt = loaded.sim.get_physics_dt()

    for _ in range(steps):
        loaded.scene.write_data_to_sim()
        loaded.robot.write_data_to_sim()

        for obj in loaded.rigid_objects.values():
            obj.write_data_to_sim()

        loaded.sim.step()

        loaded.scene.update(sim_dt)
        loaded.robot.update(sim_dt)

        for obj in loaded.rigid_objects.values():
            obj.update(sim_dt)
