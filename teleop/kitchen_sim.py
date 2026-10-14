"""
The MuJoCo simulation: loads the kitchen + robot, sends joint commands, steps physics,
and renders the two camera views.
"""
from pathlib import Path

import cv2
import mujoco
import numpy as np

import config

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCENE_XML = PROJECT_DIR / "assets" / "kitchen_scene.xml"

SIDES = ("left", "right")


class KitchenSim:
    def __init__(self, xml_path=SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # In this model every actuator has the same name as the joint it drives.
        self.actuator_ids = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.torso_id = self.model.body("torso_link").id

        # ---- renderers (they draw the scene into numpy images) ----
        kw, kh = config.KITCHEN_VIEW_SIZE
        ew, eh = config.ROBOT_EYES_SIZE
        self.kitchen_renderer = mujoco.Renderer(self.model, height=kh, width=kw)
        self.eyes_renderer = mujoco.Renderer(self.model, height=eh, width=ew)

        # A camera you can orbit with the mouse. Starts across the island, looking at the robot.
        self.kitchen_camera = mujoco.MjvCamera()
        self.kitchen_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.use_fixed_camera = False
        self.reset_camera()

        # press T in the app: draw the IK targets (your pose, resized to the robot) as a skeleton
        self.show_targets = False
        self.targets_to_draw = None

        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        """Put the robot and all the objects back where they started."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.model.joint("neck_pitch_joint").qposadr[0]] = config.HEAD_PITCH_OFFSET
        # every position actuator starts by holding its joint where it is
        for name, actuator_id in self.actuator_ids.items():
            self.data.ctrl[actuator_id] = self.joint_angle(name)
        mujoco.mj_forward(self.model, self.data)

    def reset_camera(self):
        cam = self.kitchen_camera
        cam.lookat[:] = [0.25, 0.0, 0.9]
        cam.distance = 2.2
        cam.azimuth = 180.0     # looking from the far side of the island back at the robot
        cam.elevation = -18.0

    def joint_angle(self, joint_name):
        return float(self.data.qpos[self.model.joint(joint_name).qposadr[0]])

    def robot_body_info(self):
        """
        Measure the robot in its "home" pose (every joint at 0: torso upright, upper arms down,
        forearms pointing forward, hands like a handshake). Everything is in the WORLD frame.
        """
        scratch = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, scratch)

        info = {"torso_pivot": scratch.xpos[self.torso_id].copy()}   # where the waist bends
        for side in SIDES:
            shoulder = scratch.site(f"{side}_shoulder_site").xpos.copy()
            elbow = scratch.site(f"{side}_elbow_site").xpos.copy()
            wrist = scratch.site(f"{side}_hand_ik_site")
            info[side] = {
                "shoulder": shoulder,
                "home_elbow": elbow,
                "home_wrist": wrist.xpos.copy(),
                "home_wrist_rot": wrist.xmat.reshape(3, 3).copy(),
                "upper_arm_length": np.linalg.norm(elbow - shoulder),
                "forearm_length": np.linalg.norm(wrist.xpos - elbow),
            }
        return info

    # ------------------------------------------------------------------ control
    def set_joint_targets(self, targets, max_change=None):
        """
        targets: dict joint name -> desired angle (radians)
        max_change: if given, each target moves at most this much from the previous command
                    (a speed limit, so the robot never makes violent jumps)
        """
        for name, angle in targets.items():
            actuator_id = self.actuator_ids[name]
            low, high = self.model.actuator_ctrlrange[actuator_id]
            angle = np.clip(angle, low, high)
            if max_change is not None:
                previous = self.data.ctrl[actuator_id]
                angle = previous + np.clip(angle - previous, -max_change, max_change)
            self.data.ctrl[actuator_id] = angle

    def step(self, seconds):
        """Advance the physics by about this many seconds."""
        n_steps = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ rendering
    def render_kitchen(self):
        """The 'normal' view of the kitchen and robot, as a BGR image for OpenCV."""
        camera = "kitchen_cam" if self.use_fixed_camera else self.kitchen_camera
        self.kitchen_renderer.update_scene(self.data, camera=camera)
        if self.show_targets and self.targets_to_draw is not None:
            draw_target_skeleton(self.kitchen_renderer.scene, self.targets_to_draw)
        return cv2.cvtColor(self.kitchen_renderer.render(), cv2.COLOR_RGB2BGR)

    def render_robot_eyes(self):
        """What the robot's head camera sees, as a BGR image for OpenCV."""
        self.eyes_renderer.update_scene(self.data, camera="head_cam")
        return cv2.cvtColor(self.eyes_renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self):
        self.kitchen_renderer.close()
        self.eyes_renderer.close()


# ---------------------------------------------------------------------- drawing helpers
def _add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _add_line(scene, start, end, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    scene.ngeom += 1


def draw_target_skeleton(scene, targets):
    """Draw where the IK is trying to put the shoulders, elbows and wrists (+ wrist axes)."""
    yellow = (1.0, 0.85, 0.1, 0.9)
    _add_line(scene, targets.shoulder_pos["left"], targets.shoulder_pos["right"], 0.006, yellow)
    for side in SIDES:
        shoulder, elbow, wrist = targets.shoulder_pos[side], targets.elbow_pos[side], targets.wrist_pos[side]
        _add_line(scene, shoulder, elbow, 0.006, yellow)
        _add_line(scene, elbow, wrist, 0.006, yellow)
        for point in (shoulder, elbow, wrist):
            _add_sphere(scene, point, 0.018, yellow)
        # wrist rotation axes: red = where the fingers point, green = across the palm, blue = index-finger side
        for axis, color in zip(targets.wrist_rot[side].T, [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]):
            _add_line(scene, wrist, wrist + 0.07 * axis, 0.003, color)
