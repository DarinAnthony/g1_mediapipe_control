"""
G1 kitchen teleoperation with MediaPipe.

    webcam -> MediaPipe (body, hands, head) -> retargeting -> IK (waist + arms) -> MuJoCo G1 robot

Two windows open:
    "Kitchen view"               the kitchen island and the robot (drag the mouse to orbit)
    "Robot eyes (first person)"  what the robot's head camera sees (it follows YOUR head)

Keys (click on one of the windows first):
    SPACE  pause / resume copying your motion
    C      calibrate: sit/stand up straight, face the screen and look at it, then press C
    X      clear the calibration
    R      reset the robot and the objects
    T      show / hide the IK targets (your pose, resized to the robot, drawn in yellow)
    V      show / hide the webcam picture
    K      switch between the mouse camera and the fixed kitchen camera
    H      hand mode: open_close -> grip -> fingers (see config.py)
    F      fine mode on/off: the robot's hands move only 40% as far as yours (precise grasping)
    [ ]    zoom in / out
    Q/ESC  quit

Optional:
    python main.py --video some_video.mp4    use a video file instead of the webcam
    python main.py --record my_session.mp4   also save the webcam video (to replay with --video)
"""
import os

# Our matrices are tiny (17x17). Letting numpy's math library use many threads for them just
# makes it fight with MediaPipe's threads and slows everything down ~10x. (Must be set before
# numpy is imported.)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time

import cv2

import config
from teleop.human_tracker import HumanTracker, TrackerThread
from teleop.ik import UpperBodyIK
from teleop.kitchen_sim import KitchenSim
from teleop.retarget import Retargeter
from teleop.viewer import OrbitControls, draw_text_lines, paste_inset
from teleop.webcam import Webcam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="video file to use instead of the webcam")
    parser.add_argument("--record", default=None, help="save the webcam video to this .mp4 file")
    args = parser.parse_args()

    # ---------------------------------------------------------------- set everything up
    webcam = Webcam(args.video or config.WEBCAM_INDEX, config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT,
                    record_path=args.record)
    tracking = TrackerThread(webcam, HumanTracker())   # MediaPipe runs in the background
    sim = KitchenSim()
    retargeter = Retargeter(sim.robot_body_info())
    ik = UpperBodyIK(sim.model)

    cv2.namedWindow(config.KITCHEN_WINDOW)
    cv2.namedWindow(config.ROBOT_EYES_WINDOW)
    cv2.moveWindow(config.KITCHEN_WINDOW, 0, 0)
    cv2.moveWindow(config.ROBOT_EYES_WINDOW, config.KITCHEN_VIEW_SIZE[0] + 10, 0)
    orbit = OrbitControls(sim.kitchen_camera)
    cv2.setMouseCallback(config.KITCHEN_WINDOW, orbit.on_mouse)

    teleop_on = True
    show_webcam = config.SHOW_WEBCAM_INSET
    last_human = None
    last_time = time.time()
    fps = 0.0

    print(__doc__)

    while True:
        # ---------------------------------------------------------------- 1. see the human
        human, webcam_picture = tracking.latest()

        # ---------------------------------------------------------------- 2. human -> robot goals
        if human is not None and human is not last_human and teleop_on:
            targets = retargeter.update(human)   # a new measurement arrived
            last_human = human
        else:
            targets = retargeter.targets()       # nothing new (or paused): keep the last goals

        # ---------------------------------------------------------------- 3. IK: waist + both arms
        joint_targets = ik.solve(sim.data.qpos, targets)
        joint_targets.update(targets.hand_joints)
        joint_targets["neck_yaw_joint"] = targets.neck_yaw
        joint_targets["neck_pitch_joint"] = targets.neck_pitch

        # ---------------------------------------------------------------- 4. move the robot
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)

        sim.set_joint_targets(joint_targets, max_change=config.MAX_JOINT_SPEED * dt)
        sim.step(dt)
        sim.targets_to_draw = targets

        # ---------------------------------------------------------------- 5. draw the windows
        kitchen = sim.render_kitchen()
        eyes = sim.render_robot_eyes()

        if show_webcam and webcam_picture is not None:
            paste_inset(kitchen, cv2.flip(webcam_picture, 1))   # mirrored, like a selfie

        seen = []
        if human is not None and human.body is not None:
            seen.append("body")
        if human is not None and human.left_hand is not None:
            seen.append("left hand")
        if human is not None and human.right_hand is not None:
            seen.append("right hand")
        hands = []
        for side in ("left", "right"):
            state = ""
            if config.HAND_MODE == "open_close":
                state = " CLOSED" if retargeter.hand_closed[side] else " open"
            hands.append(f"{side[0].upper()} {retargeter.grip[side]:.2f}{state}")
        draw_text_lines(kitchen, [
            f"{'TELEOP ON' if teleop_on else 'PAUSED (space)'}"
            f"{'   FINE MODE' if retargeter.fine_anchor is not None else ''}   {fps:4.1f} fps",
            "tracking: " + (", ".join(seen) if seen else "nobody found"),
            f"hands ({config.HAND_MODE}): " + "   ".join(hands),
        ])
        draw_text_lines(kitchen, [
            "SPACE pause  F fine  C calibrate  X clear  R reset  H hand mode",
            "T targets  V webcam  K camera  [ ] zoom  Q quit",
        ], corner="bottom-left", scale=0.5)

        cv2.imshow(config.KITCHEN_WINDOW, kitchen)
        cv2.imshow(config.ROBOT_EYES_WINDOW, eyes)

        # ---------------------------------------------------------------- 6. keyboard
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            teleop_on = not teleop_on
        elif key == ord("c") and human is not None:
            retargeter.calibrate(human)
            print("Calibrated!")
        elif key == ord("x"):
            retargeter.clear_calibration()
            print("Calibration cleared.")
        elif key == ord("r"):
            sim.reset()
            ik.reset()
        elif key == ord("t"):
            sim.show_targets = not sim.show_targets
        elif key == ord("v"):
            show_webcam = not show_webcam
        elif key == ord("f"):
            print("Fine mode", "ON" if retargeter.toggle_fine_mode() else "off")
        elif key == ord("h"):
            modes = ["open_close", "grip", "fingers"]
            config.HAND_MODE = modes[(modes.index(config.HAND_MODE) + 1) % len(modes)]
            print("Hand mode:", config.HAND_MODE)
        elif key == ord("k"):
            sim.use_fixed_camera = not sim.use_fixed_camera
        elif key == ord("["):
            orbit.zoom(0.9)
        elif key == ord("]"):
            orbit.zoom(1.1)

    tracking.stop()
    webcam.release()
    sim.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
