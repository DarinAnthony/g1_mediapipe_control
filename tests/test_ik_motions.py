"""
Check that retargeting + IK follow a bunch of upper-body motions (no webcam needed).

For each pretend pose it:
    1. runs the Retargeter and the UpperBodyIK, just like main.py does
    2. measures how close the robot gets to every target (torso, elbows, wrists, hand rotation)
    3. runs the physics for a moment to check the motors can actually hold the pose
    4. (with --images) renders the robot with the yellow target skeleton on top

Run it from PyCharm, or:   python tests/test_ik_motions.py --images
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import cv2
import mujoco
import numpy as np

import config
from teleop.geometry import normalize, rotation_angle
from teleop.ik import UpperBodyIK
from teleop.kitchen_sim import KitchenSim, draw_target_skeleton
from teleop.retarget import Retargeter
from teleop.fake_human import (BACK, DOWN, FORWARD, LEFT, RIGHT, UP, hand_frame, make_human, torso_rot)

d2 = lambda *v: normalize(np.array(v, dtype=float))   # direction helper

PALM_DOWN = {"left": hand_frame(FORWARD, DOWN, "left"), "right": hand_frame(FORWARD, DOWN, "right")}

# name: keyword arguments for make_human()
MOTIONS = {
    "ready (home)": dict(),
    "arms straight forward": dict(left_arm=(FORWARD, FORWARD), right_arm=(FORWARD, FORWARD),
                                  left_hand=PALM_DOWN["left"], right_hand=PALM_DOWN["right"]),
    "T-pose": dict(left_arm=(LEFT, LEFT), right_arm=(RIGHT, RIGHT),
                   left_hand=hand_frame(LEFT, DOWN, "left"), right_hand=hand_frame(RIGHT, DOWN, "right")),
    "arms up": dict(left_arm=(UP, UP), right_arm=(UP, UP),
                    left_hand=hand_frame(UP, FORWARD, "left"), right_hand=hand_frame(UP, FORWARD, "right")),
    "reach down to counter": dict(left_arm=(d2(0.5, 0, -1), d2(1, 0, -0.7)), right_arm=(d2(0.5, 0, -1), d2(1, 0, -0.7)),
                                  left_hand=hand_frame(d2(1, 0, -0.7), DOWN, "left"),
                                  right_hand=hand_frame(d2(1, 0, -0.7), DOWN, "right"), closure=(0.5, 0.5)),
    "lean forward 20 + reach": dict(torso=(0, 20, 0),
                                    left_arm=(d2(0.6, 0, -1), d2(1, 0, -0.5)), right_arm=(d2(0.6, 0, -1), d2(1, 0, -0.5)),
                                    left_hand=PALM_DOWN["left"], right_hand=PALM_DOWN["right"]),
    "lean back 10": dict(torso=(0, -10, 0)),
    "twist left 30": dict(torso=(30, 0, 0),
                          left_arm=(DOWN, torso_rot(30) @ FORWARD), right_arm=(DOWN, torso_rot(30) @ FORWARD),
                          left_hand=torso_rot(30) @ hand_frame(FORWARD, RIGHT, "left"),
                          right_hand=torso_rot(30) @ hand_frame(FORWARD, LEFT, "right")),
    "side lean right 15": dict(torso=(0, 0, 15)),
    "chicken wings (elbows out)": dict(left_arm=(d2(0, 1, -1), FORWARD), right_arm=(d2(0, -1, -1), FORWARD)),
    "hands together at chest": dict(left_arm=(d2(0.3, 0.3, -1), d2(0.6, -1, 0.3)),
                                    right_arm=(d2(0.3, -0.3, -1), d2(0.6, 1, 0.3)),
                                    left_hand=hand_frame(d2(0.6, -1, 0.3), RIGHT, "left"),
                                    right_hand=hand_frame(d2(0.6, 1, 0.3), LEFT, "right")),
    "right hand wave": dict(right_arm=(d2(0, -1, 0.2), UP), right_hand=hand_frame(UP, FORWARD, "right"),
                            closure=(0.0, 0.0)),
    "cross-body reach (right hand left)": dict(right_arm=(d2(0.7, 0.2, -0.7), d2(0.4, 1, -0.3)),
                                               right_hand=hand_frame(d2(0.4, 1, -0.3), DOWN, "right")),
    "fists + head turn/nod": dict(closure=(1.0, 1.0), head=(40, 20)),
}


def robot_contacts(sim):
    """Short description of what the robot is touching ("counter", "itself", objects...)."""
    m, d = sim.model, sim.data
    pelvis = m.body("pelvis").id
    touching = set()
    for contact in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
        robot1, robot2 = m.body_rootid[b1] == pelvis, m.body_rootid[b2] == pelvis
        if robot1 and robot2:
            touching.add("itself")
        elif robot1 or robot2:
            other = b2 if robot1 else b1
            touching.add("counter" if m.body(other).name == "island" else m.body(other).name)
    return ", ".join(sorted(touching))


def evaluate(sim, ik, targets, solution):
    """Forward kinematics of the IK answer -> errors for every task."""
    d = mujoco.MjData(sim.model)
    d.qpos[:] = sim.data.qpos
    for name, angle in solution.items():
        d.qpos[sim.model.joint(name).qposadr[0]] = angle
    mujoco.mj_kinematics(sim.model, d)
    torso = d.body("torso_link").xmat.reshape(3, 3)
    result = {"torso_deg": np.degrees(rotation_angle(targets.torso_rot.T @ torso))}
    for side in ("left", "right"):
        elbow = d.site(f"{side}_elbow_site").xpos
        wrist = d.site(f"{side}_hand_ik_site")
        result[f"{side}_elbow_cm"] = 100 * np.linalg.norm(elbow - targets.elbow_pos[side])
        result[f"{side}_wrist_cm"] = 100 * np.linalg.norm(wrist.xpos - targets.wrist_pos[side])
        result[f"{side}_rot_deg"] = np.degrees(rotation_angle(targets.wrist_rot[side].T @ wrist.xmat.reshape(3, 3)))
    at_limit = [n for n, i in zip(ik.joint_names, range(len(ik.q)))
                if min(ik.q[i] - ik.lower[i], ik.upper[i] - ik.q[i]) < 1e-3]
    return result, at_limit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", action="store_true", help="save a picture of every pose")
    args = parser.parse_args()

    config.SMOOTHING = 0.0       # no smoothing: we want the exact answer for each pose
    sim = KitchenSim()
    robot = sim.robot_body_info()
    out_dir = PROJECT_DIR / "tests" / "output"
    pictures = []

    header = f"{'motion':36s} {'torso':>6s} {'L elb':>6s} {'L wri':>6s} {'L rot':>6s} {'R elb':>6s} {'R wri':>6s} {'R rot':>6s} {'hold':>6s}  {'robot touches':22s} joints at a limit"
    print(header)
    print(f"{'':36s} {'deg':>6s} {'cm':>6s} {'cm':>6s} {'deg':>6s} {'cm':>6s} {'cm':>6s} {'deg':>6s} {'deg':>6s}")
    print("-" * len(header))

    failures = []
    for name, kwargs in MOTIONS.items():
        sim.reset()
        retargeter = Retargeter(robot)
        ik = UpperBodyIK(sim.model)
        human = make_human(**kwargs)

        targets = retargeter.update(human)
        for _ in range(5):                       # 5 frames of IK, like 5 webcam frames
            solution = ik.solve(sim.data.qpos, targets)
        errors, at_limit = evaluate(sim, ik, targets, solution)

        # physics: can the motors actually hold this pose?
        commands = dict(solution)
        commands.update(targets.hand_joints)
        commands["neck_yaw_joint"], commands["neck_pitch_joint"] = targets.neck_yaw, targets.neck_pitch
        for _ in range(100):   # 2 seconds, moving no faster than the app's speed limit
            sim.set_joint_targets(commands, max_change=config.MAX_JOINT_SPEED * 0.02)
            sim.step(0.02)
        hold_error = max(abs(sim.joint_angle(j) - a) for j, a in solution.items())
        touching = robot_contacts(sim)

        print(f"{name:36s} {errors['torso_deg']:6.1f} "
              f"{errors['left_elbow_cm']:6.1f} {errors['left_wrist_cm']:6.1f} {errors['left_rot_deg']:6.1f} "
              f"{errors['right_elbow_cm']:6.1f} {errors['right_wrist_cm']:6.1f} {errors['right_rot_deg']:6.1f} "
              f"{np.degrees(hold_error):6.1f}  {touching:22s} {', '.join(j.replace('_joint', '') for j in at_limit)}")

        if name == "fists + head turn/nod":
            fingers = {k: round(v, 2) for k, v in targets.hand_joints.items() if "index" in k or "thumb" in k}
            print(f"{'':36s} neck yaw {np.degrees(targets.neck_yaw):.0f} deg, "
                  f"neck pitch {np.degrees(targets.neck_pitch - config.HEAD_PITCH_OFFSET):.0f} deg (+offset); "
                  f"fingers {fingers}")

        worst_wrist = max(errors["left_wrist_cm"], errors["right_wrist_cm"])
        if worst_wrist > 3.0 or errors["torso_deg"] > 5.0:
            failures.append(name)

        if args.images:
            renderer = sim.kitchen_renderer
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.05, 0.0, 1.0]
            cam.distance, cam.azimuth, cam.elevation = 2.0, 145.0, -12.0
            renderer.update_scene(sim.data, camera=cam)
            draw_target_skeleton(renderer.scene, targets)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            image = image[60:-60, 180:-180]
            cv2.putText(image, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            pictures.append(image)

    print("-" * len(header))
    print("wrist/elbow = distance from target, rot = hand rotation error, "
          "hold = biggest joint error after 2 s of physics\n"
          "(a big 'hold' error together with 'robot touches' means something is physically in the way)")
    if failures:
        print("Poses the robot could not fully reach (wrist > 3 cm or torso > 5 deg off):", ", ".join(failures))
    else:
        print("All poses reached!")

    if pictures:
        out_dir.mkdir(exist_ok=True)
        columns = 4
        while len(pictures) % columns:
            pictures.append(np.full_like(pictures[0], 255))
        rows = [np.hstack(pictures[i:i + columns]) for i in range(0, len(pictures), columns)]
        montage = np.vstack(rows)
        path = out_dir / "ik_motions.png"
        cv2.imwrite(str(path), montage)
        print("saved", path)


if __name__ == "__main__":
    main()
