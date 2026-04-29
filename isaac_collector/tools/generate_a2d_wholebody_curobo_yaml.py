from pathlib import Path
import xml.etree.ElementTree as ET

URDF = Path("/home/pm/Desktop/Project/robot_stack/assets/a2d/a2d_wholebody_simplified.urdf")
OUT = Path("/home/pm/Desktop/Project/robot_stack/configs/curobo/a2d_wholebody.yml")

root = ET.parse(URDF).getroot()

joint_child = {}
all_joints = set()
all_links = set()

for link in root.findall("link"):
    all_links.add(link.attrib["name"])

for joint in root.findall("joint"):
    name = joint.attrib["name"]
    all_joints.add(name)
    child = joint.find("child").attrib.get("link")
    joint_child[name] = child

cspace_joints = [
    "joint_lift_body",
    "joint_body_pitch",

    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",

    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
]

missing = [j for j in cspace_joints if j not in all_joints]
if missing:
    print("[WARN] Missing expected joints:")
    for j in missing:
        print("  ", j)

collision_links = []
for j in cspace_joints:
    if j in joint_child:
        collision_links.append(joint_child[j])

# 去重
collision_links = list(dict.fromkeys(collision_links))

# 选择 base link
if "base_link" in all_links:
    base_link = "base_link"
elif "root" in all_links:
    base_link = "root"
else:
    # 粗略选择第一个 link；后面如果 cuRobo 报错，再手动改
    base_link = sorted(all_links)[0]

# 选择 ee link：优先 gripper_center，否则用 left_arm_joint7 的 child link
if "gripper_center" in all_links:
    ee_link = "gripper_center"
else:
    ee_link = joint_child.get("left_arm_joint7", collision_links[-1] if collision_links else base_link)

retract = [
    0.1995,
    0.6025,

    -1.0817,
    0.5907,
    0.3442,
    -1.2819,
    0.6928,
    1.4725,
    -0.1599,

    1.0817,
    -0.5907,
    -0.3442,
    1.2819,
    -0.6928,
    -0.7,
    0.0,
]

def sphere_radius(link):
    name = link.lower()
    if "body" in name or "base" in name or "torso" in name:
        return 0.18
    if "head" in name:
        return 0.12
    if "arm" in name or "left" in name or "right" in name:
        return 0.07
    return 0.06

lines = []
lines.append("robot_cfg:")
lines.append("  kinematics:")
lines.append("    use_usd_kinematics: False")
lines.append(f'    urdf_path: "{URDF}"')
lines.append('    asset_root_path: "/home/pm/Desktop/Project/robot_stack/assets/a2d"')
lines.append(f'    base_link: "{base_link}"')
lines.append(f'    ee_link: "{ee_link}"')
lines.append("    link_names: null")
lines.append("    extra_links: null")
lines.append("    lock_joints: null")
lines.append("")
lines.append("    collision_link_names:")
for link in collision_links:
    lines.append(f'      - "{link}"')
lines.append("")
lines.append("    collision_spheres:")
for link in collision_links:
    r = sphere_radius(link)
    lines.append(f"      {link}:")
    lines.append(f"        - center: [0.0, 0.0, 0.0]")
    lines.append(f"          radius: {r}")
lines.append("")
lines.append("    collision_sphere_buffer: 0.005")
lines.append("    extra_collision_spheres: {}")
lines.append("    self_collision_ignore: {}")
lines.append("    self_collision_buffer: {}")
lines.append("    use_global_cumul: True")
lines.append("    mesh_link_names: null")
lines.append("")
lines.append("    cspace:")
lines.append("      joint_names:")
for j in cspace_joints:
    lines.append(f'        - "{j}"')
lines.append(f"      retract_config: {retract}")
lines.append(f"      null_space_weight: {[1.0] * len(cspace_joints)}")
lines.append(f"      cspace_distance_weight: {[1.0] * len(cspace_joints)}")
lines.append("      max_jerk: 500.0")
lines.append("      max_acceleration: 15.0")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print("[OK] wrote:", OUT)
print("[INFO] base_link:", base_link)
print("[INFO] ee_link:", ee_link)
print("[INFO] collision_links:")
for link in collision_links:
    print("  ", link)