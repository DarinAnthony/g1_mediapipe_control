"""
Check your TODOs (no webcam needed). Run this after each TODO you finish.

    TODO 1  teleop/human_tracker.py   body landmark numbers
    TODO 2  teleop/human_tracker.py   create the MediaPipe models
    TODO 3  teleop/human_tracker.py   run the models on a picture
    TODO 4  teleop/human_tracker.py   landmarks -> numpy arrays
    TODO 5  teleop/retarget.py        grip_amount (how closed is your hand?)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PHOTO = Path(__file__).resolve().parent / "assets" / "check" / "person.jpg"
BODY_NAMES = ["NOSE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
              "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP"]


def mediapipe_chart():
    """MediaPipe's own numbering of the 33 body landmarks."""
    if hasattr(mp, "solutions"):
        return {name: mp.solutions.pose.PoseLandmark[name].value for name in BODY_NAMES}
    from mediapipe.tasks.python.vision import PoseLandmark
    return {name: PoseLandmark[name].value for name in BODY_NAMES}


# ------------------------------------------------------------------------------ checks
# Each check returns (passed, message).

def check_todo1():
    from teleop import human_tracker as ht
    chart = mediapipe_chart()
    wrong = [name for name in BODY_NAMES if getattr(ht, name) != chart[name]]
    if not wrong:
        return True, "all 11 landmark numbers are right"
    return False, f"these numbers aren't right yet: {', '.join(wrong)}"


class NotReady(Exception):
    pass


def run_tracker_on_photo():
    from teleop.human_tracker import HumanTracker
    tracker = HumanTracker()
    if tracker.pose is None or tracker.hands is None:
        raise NotReady("finish TODO 2 first")
    image = cv2.imread(str(PHOTO))
    human = None
    for _ in range(3):   # a few frames, like a short video
        human = tracker.process(image)
    return tracker, human


def check_todo2():
    from teleop.human_tracker import HumanTracker
    tracker = HumanTracker()
    if tracker.pose is None or tracker.hands is None:
        missing = [name for name, model in (("self.pose", tracker.pose), ("self.hands", tracker.hands)) if model is None]
        return False, f"{' and '.join(missing)} still None"
    if not hasattr(tracker.pose, "process") or not hasattr(tracker.hands, "process"):
        return False, "self.pose / self.hands should be MediaPipe models (mp_pose.Pose(...), mp_hands.Hands(...))"
    return True, "both models were created"


def check_todo3():
    tracker, _ = run_tracker_on_photo()
    if tracker.pose_results is None or tracker.hand_results is None:
        return False, "self.pose_results / self.hand_results are still None"
    if not tracker.pose_results.pose_landmarks:
        return False, "MediaPipe ran, but found nobody in the test photo (did you convert BGR to RGB?)"
    if not tracker.hand_results.multi_hand_landmarks:
        return False, "found the body but no hands in the test photo (did you give Hands the RGB image?)"
    return True, "MediaPipe found the person and a hand in the test photo"


def check_todo4():
    from teleop.geometry import to_robot_frame
    tracker, human = run_tracker_on_photo()
    if human.body is None:
        return False, "no body yet (finish TODO 3 first)"
    if human.body_visibility is None or np.shape(human.body_visibility) != (33,):
        return False, f"visibility should have shape (33,), yours is {np.shape(human.body_visibility)}"
    if np.shape(human.body) != (33, 3):
        return False, f"points should have shape (33, 3), yours is {np.shape(human.body)}"
    if np.all(human.body == 0) or np.all(human.body_visibility == 0):
        return False, "points / visibility are still all zeros"
    landmarks = tracker.pose_results.pose_world_landmarks.landmark
    expected_points = to_robot_frame([[p.x, p.y, p.z] for p in landmarks])
    if not np.allclose(human.body, expected_points, atol=1e-6):
        return False, "the numbers in points don't match the landmarks: use p.x, p.y, p.z in that order"
    if not np.allclose(human.body_visibility, [p.visibility for p in landmarks], atol=1e-6):
        return False, "visibility doesn't match: use p.visibility for each landmark"
    return True, "33 points with x, y, z and visibility, and they make sense"


def check_todo5():
    from teleop.fake_human import FORWARD, RIGHT, hand_frame, hand_landmarks
    from teleop.retarget import grip_amount
    rotation = hand_frame(FORWARD, RIGHT, "left")
    open_hand = grip_amount(hand_landmarks(rotation, "left", 0.0))
    half = grip_amount(hand_landmarks(rotation, "left", 0.5))
    fist = grip_amount(hand_landmarks(rotation, "left", 1.0))
    big_fist = grip_amount(1.5 * hand_landmarks(rotation, "left", 1.0))   # same fist, bigger hand
    if open_hand == half == fist == 0.0:
        return False, "grip_amount still returns 0 for every hand"
    numbers = f"(open {open_hand:.2f}, half {half:.2f}, fist {fist:.2f})"
    if open_hand > 0.1 or fist < 0.9 or not 0.1 < half < 0.9:
        return False, f"open hand should be ~0, fist ~1, half closed in between {numbers}: check your distances"
    if abs(big_fist - fist) > 0.05:
        return False, "a bigger hand gives a different grip: ratio should be tip_distance / palm_length"
    return True, f"open hand {open_hand:.2f}, half closed {half:.2f}, fist {fist:.2f}"


CHECKS = [
    ("TODO 1", "landmark numbers", check_todo1),
    ("TODO 2", "create the MediaPipe models", check_todo2),
    ("TODO 3", "run the models", check_todo3),
    ("TODO 4", "landmarks -> numpy arrays", check_todo4),
    ("TODO 5", "grip_amount", check_todo5),
]


def main():
    print("Checking your code...\n")
    done = 0
    for todo, title, check in CHECKS:
        try:
            passed, message = check()
        except NotReady as reason:
            passed, message = False, str(reason)
        except Exception as error:
            passed = False
            last_line = traceback.format_exc().strip().splitlines()[-1]
            message = f"crashed with {last_line}"
            if not isinstance(error, (AttributeError, NameError, TypeError, ValueError, IndexError)):
                traceback.print_exc()
        done += passed
        print(f"{todo}  {title:30s} {'PASS' if passed else '----'}   {message}")
    print(f"\n{done} of {len(CHECKS)} TODOs done." + ("  All done, run main.py!" if done == len(CHECKS) else ""))


if __name__ == "__main__":
    main()
