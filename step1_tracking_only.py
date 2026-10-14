"""
STEP 1: just the webcam + MediaPipe, no robot.

Shows your skeleton and hands, and prints the numbers the robot will use:
    - how much your torso is leaning / twisting
    - where your head is pointing
    - your elbow angles (same calc_angle as the bicep curl app)
    - how bent your index, middle and thumb are, and your overall "grip" (0 open .. 1 fist)

Press Q to quit.
"""
import cv2
import numpy as np

import config
from teleop import human_tracker as ht
from teleop.geometry import bend, calc_angle, head_yaw_pitch, torso_rotation
from teleop.human_tracker import HumanTracker
from teleop.retarget import grip_amount
from teleop.viewer import draw_text_lines
from teleop.webcam import Webcam


def describe(human):
    """Turn a HumanPose into a few lines of text."""
    lines = []
    if human.body is None:
        return ["No person found - step back so your upper body is visible"]

    b = human.body
    torso = torso_rotation(b[ht.LEFT_SHOULDER], b[ht.RIGHT_SHOULDER], b[ht.LEFT_HIP], b[ht.RIGHT_HIP])
    forward, up = torso[:, 0], torso[:, 2]
    lines.append(f"torso: lean forward {np.degrees(np.arcsin(up[0])):+4.0f} deg   "
                 f"lean right {np.degrees(np.arcsin(-up[1])):+4.0f} deg   "
                 f"twist left {np.degrees(np.arctan2(forward[1], forward[0])):+4.0f} deg")

    if human.can_see(ht.NOSE, ht.LEFT_EAR, ht.RIGHT_EAR):
        yaw, pitch = head_yaw_pitch(b[ht.NOSE], b[ht.LEFT_EAR], b[ht.RIGHT_EAR])
        lines.append(f"head: turned left {np.degrees(yaw):+4.0f} deg   looking down {np.degrees(pitch):+4.0f} deg")

    for side, (s, e, w) in (("left", (ht.LEFT_SHOULDER, ht.LEFT_ELBOW, ht.LEFT_WRIST)),
                            ("right", (ht.RIGHT_SHOULDER, ht.RIGHT_ELBOW, ht.RIGHT_WRIST))):
        if human.can_see(s, e, w):
            lines.append(f"{side} elbow angle: {calc_angle(b[s], b[e], b[w]):4.0f} deg")

    for side, hand in (("left", human.left_hand), ("right", human.right_hand)):
        if hand is None:
            continue
        index = np.degrees(bend(hand[ht.WRIST], hand[ht.INDEX_MCP], hand[ht.INDEX_PIP]))
        middle = np.degrees(bend(hand[ht.WRIST], hand[ht.MIDDLE_MCP], hand[ht.MIDDLE_PIP]))
        thumb = np.degrees(bend(hand[ht.THUMB_MCP], hand[ht.THUMB_IP], hand[ht.THUMB_TIP]))
        lines.append(f"{side} hand bends: index {index:3.0f}  middle {middle:3.0f}  thumb {thumb:3.0f} deg")
        grip = grip_amount(hand)
        verdict = "closed" if grip > config.GRIP_CLOSE_ABOVE else "open" if grip < config.GRIP_OPEN_BELOW else "in between"
        lines.append(f"{side} hand grip: {grip:.2f} ({verdict})")
    return lines


def main():
    webcam = Webcam(config.WEBCAM_INDEX, config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT)
    tracker = HumanTracker()
    last_frame_id = -1

    while True:
        frame_id, frame = webcam.read()
        if frame is None or frame_id == last_frame_id:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        last_frame_id = frame_id

        human = tracker.process(frame)
        picture = cv2.flip(tracker.draw(frame), 1)   # mirror it so it feels like a mirror
        picture = cv2.resize(picture, (960, 720))
        draw_text_lines(picture, describe(human), scale=0.6)

        cv2.imshow("Step 1: MediaPipe tracking", picture)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
