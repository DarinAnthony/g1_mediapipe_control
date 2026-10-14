"""
STEP 2: drive the robot with the KEYBOARD (no webcam needed).

You move the hand targets around and the IK solver figures out the joint angles.
Here the IK only cares about the torso and the hands; the elbows can go wherever they like
(compare with main.py, where your real elbows are tracked too).

Keys (click on the kitchen window first):
    1 / 2       pick the LEFT / RIGHT hand
    W / S       move that hand forward / back
    A / D       move it left / right
    R / F       move it up / down
    SPACE       close / open that hand
    P           palm down  <->  handshake
    I / K       lean the torso forward / back
    J / L       twist the torso left / right
    U / O       turn the head left / right
    Y / H       look up / down
    T           show the IK targets          0  reset everything
    [ ]         zoom                         Q  quit
"""
import cv2
import numpy as np

import config
from teleop.fake_human import rot_x, rot_y, rot_z
from teleop.ik import UpperBodyIK
from teleop.kitchen_sim import KitchenSim
from teleop.retarget import RobotTargets, dex3_grip_angles
from teleop.viewer import OrbitControls, draw_text_lines

config.IK_ELBOW_WEIGHT = 0.0   # no elbow targets in this playground

STEP = 0.01          # meters per key press
ANGLE_STEP = 0.05    # radians per key press


def main():
    sim = KitchenSim()
    robot = sim.robot_body_info()
    ik = UpperBodyIK(sim.model)
    orbit = OrbitControls(sim.kitchen_camera)
    cv2.namedWindow(config.KITCHEN_WINDOW)
    cv2.setMouseCallback(config.KITCHEN_WINDOW, orbit.on_mouse)

    def home():
        return {
            "hand": "right",
            "wrist_pos": {s: robot[s]["home_wrist"].copy() for s in ("left", "right")},
            "palm_down": {"left": False, "right": False},
            "closed": {"left": False, "right": False},
            "lean": 0.0, "twist": 0.0,
            "neck_yaw": 0.0, "neck_pitch": config.HEAD_PITCH_OFFSET,
        }

    state = home()
    print(__doc__)

    while True:
        # ---- build the targets from the keyboard state ----
        torso_rot = rot_z(state["twist"]) @ rot_y(state["lean"])
        pivot = robot["torso_pivot"]
        shoulders = {s: pivot + torso_rot @ (robot[s]["shoulder"] - pivot) for s in ("left", "right")}
        wrist_rot = {}
        hand_joints = {}
        for side in ("left", "right"):
            # handshake = the home rotation; palm down = roll it 90 degrees about the forearm
            roll = 1.0 if side == "left" else -1.0
            wrist_rot[side] = rot_x(roll * np.pi / 2) if state["palm_down"][side] else np.eye(3)
            hand_joints.update(dex3_grip_angles(side, 1.0 if state["closed"][side] else 0.0))

        targets = RobotTargets(
            torso_rot=torso_rot,
            shoulder_pos=shoulders,
            elbow_pos=dict(state["wrist_pos"]),     # ignored (elbow weight is 0)
            wrist_pos=state["wrist_pos"],
            wrist_rot=wrist_rot,
            hand_joints=hand_joints,
            neck_yaw=state["neck_yaw"],
            neck_pitch=state["neck_pitch"],
        )

        # ---- IK + physics ----
        commands = ik.solve(sim.data.qpos, targets)
        commands.update(targets.hand_joints)
        commands["neck_yaw_joint"] = targets.neck_yaw
        commands["neck_pitch_joint"] = targets.neck_pitch
        sim.set_joint_targets(commands, max_change=config.MAX_JOINT_SPEED / 30)
        sim.step(1 / 30)
        sim.targets_to_draw = targets

        # ---- draw ----
        kitchen = sim.render_kitchen()
        side = state["hand"]
        draw_text_lines(kitchen, [
            f"moving the {side.upper()} hand   wrist error {100 * ik.errors.get(side + '_wrist', 0):.1f} cm",
            "1/2 hand  WASD RF move  SPACE grip  P palm",
            "IK lean  JL twist  UO YH head  T targets  0 reset  Q quit",
        ])
        cv2.imshow(config.KITCHEN_WINDOW, kitchen)
        cv2.imshow(config.ROBOT_EYES_WINDOW, sim.render_robot_eyes())

        # ---- keyboard ----
        key = cv2.waitKey(1) & 0xFF
        pos = state["wrist_pos"][side]
        moves = {"w": (0, STEP), "s": (0, -STEP), "a": (1, STEP), "d": (1, -STEP), "r": (2, STEP), "f": (2, -STEP)}
        if key in (ord("q"), 27):
            break
        elif chr(key) in moves:
            axis, amount = moves[chr(key)]
            pos[axis] += amount
        elif key == ord("1"):
            state["hand"] = "left"
        elif key == ord("2"):
            state["hand"] = "right"
        elif key == ord(" "):
            state["closed"][side] = not state["closed"][side]
        elif key == ord("p"):
            state["palm_down"][side] = not state["palm_down"][side]
        elif key == ord("i"):
            state["lean"] += ANGLE_STEP
        elif key == ord("k"):
            state["lean"] -= ANGLE_STEP
        elif key == ord("j"):
            state["twist"] += ANGLE_STEP
        elif key == ord("l"):
            state["twist"] -= ANGLE_STEP
        elif key == ord("u"):
            state["neck_yaw"] += ANGLE_STEP
        elif key == ord("o"):
            state["neck_yaw"] -= ANGLE_STEP
        elif key == ord("y"):
            state["neck_pitch"] -= ANGLE_STEP
        elif key == ord("h"):
            state["neck_pitch"] += ANGLE_STEP
        elif key == ord("t"):
            sim.show_targets = not sim.show_targets
        elif key == ord("0"):
            state = home()
            sim.reset()
            ik.reset()
        elif key == ord("["):
            orbit.zoom(0.9)
        elif key == ord("]"):
            orbit.zoom(1.1)

    sim.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
