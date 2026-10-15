"""
Runs MediaPipe Pose (your body) and MediaPipe Hands (your fingers) on a webcam frame
and packs the results into a HumanPose.
"""
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

import config
from teleop.geometry import to_robot_frame

# MediaPipe 0.10.x (Python 3.10 - 3.12) has mp.solutions, the same API as the class projects.
# MediaPipe 1.x (Python 3.13+) removed it, so we fall back to a look-alike built on the new API.
HAS_SOLUTIONS = hasattr(mp, "solutions")
if HAS_SOLUTIONS:
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
else:
    from teleop import mediapipe_tasks
    mp_pose = mp_hands = mediapipe_tasks

# ============================================================================ TODO 1
# Body landmark numbers. MediaPipe Pose gives us 33 points, numbered 0 to 32.
# Replace each 0 with the right number.
# Use the list in your bicep curl app, or the picture at
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
NOSE = 0
LEFT_EAR, RIGHT_EAR = 0, 0
LEFT_SHOULDER, RIGHT_SHOULDER = 0, 0
LEFT_ELBOW, RIGHT_ELBOW = 0, 0
LEFT_WRIST, RIGHT_WRIST = 0, 0
LEFT_HIP, RIGHT_HIP = 0, 0
# ==================================================================================

# ----- hand landmark numbers (MediaPipe Hands gives us 21 points per hand) -----
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
PINKY_MCP = 17


@dataclass
class HumanPose:
    """
    Everything we measured about the person in one webcam frame.
    All 3D points are in ROBOT axes (x forward, y left, z up) and in meters.
    Anything MediaPipe could not see is None.
    """
    body: np.ndarray = None             # (33, 3) body landmarks, origin between your hips
    body_visibility: np.ndarray = None  # (33,)   0..1, how sure MediaPipe is about each landmark
    left_hand: np.ndarray = None        # (21, 3) hand landmarks, origin near the middle of the hand
    right_hand: np.ndarray = None       # (21, 3)
    timestamp: float = 0.0              # when the webcam frame was processed (seconds)

    def can_see(self, *landmark_ids):
        """True if every listed body landmark is visible enough to trust."""
        if self.body is None:
            return False
        return all(self.body_visibility[i] >= config.MIN_LANDMARK_VISIBILITY for i in landmark_ids)


class HumanTracker:
    def __init__(self):
        # ======================================================================== TODO 2
        # Make the two MediaPipe models, like in the bicep curl app:
        #     mp_pose.Pose()     finds your body (33 points)
        #     mp_hands.Hands()   finds your hands (21 points each, up to 2 hands)
        # (Optional: give them settings like in class, e.g. min_detection_confidence=0.6)
        self.pose = None
        self.hands = None
        # ==============================================================================
        self.pose_results = None
        self.hand_results = None

    def process(self, frame_bgr):
        """Run MediaPipe on one BGR webcam frame and return a HumanPose."""
        human = HumanPose(timestamp=time.monotonic())

        # ======================================================================== TODO 3
        # 1. OpenCV gives us BGR colors, but MediaPipe wants RGB: convert frame_bgr with cv2.cvtColor
        # 2. Run BOTH models on the RGB image and keep what they find:
        #        self.pose_results = ...
        #        self.hand_results = ...
        self.pose_results = None
        self.hand_results = None
        # ==============================================================================
        if self.pose_results is None or self.hand_results is None:
            return human   # (TODO 3 isn't done yet, so we can't see anybody)

        # ---- body ----
        # pose_world_landmarks are real 3D positions in meters (pose_landmarks are 2D, in the image).
        if self.pose_results.pose_world_landmarks:
            landmarks = self.pose_results.pose_world_landmarks.landmark
            # ==================================================================== TODO 4
            # `landmarks` is a list of 33 points. Each point p has p.x, p.y, p.z (meters)
            # and p.visibility (0..1, how sure MediaPipe is that it can see that point).
            # Make two numpy arrays:
            #     points      shape (33, 3)   [[x, y, z], [x, y, z], ...]
            #     visibility  shape (33,)     [v, v, v, ...]
            # Hint: put a list comprehension inside np.array( ... ), e.g. [... for p in landmarks]
            points = np.zeros((33, 3))
            visibility = np.zeros(33)
            # ==========================================================================
            human.body = to_robot_frame(points)
            human.body_visibility = visibility

        # ---- hands ----
        if self.hand_results.multi_hand_world_landmarks:
            sides = self._left_or_right(self.hand_results)
            for side, world_landmarks in zip(sides, self.hand_results.multi_hand_world_landmarks):
                points = np.array([[p.x, p.y, p.z] for p in world_landmarks.landmark])
                if side is None:
                    continue   # not near either of your wrists: probably a mistake
                if side == "left":
                    human.left_hand = to_robot_frame(points)
                else:
                    human.right_hand = to_robot_frame(points)

        return human

    def _left_or_right(self, hand_results):
        """
        Decide which detected hand is your left hand and which is your right.

        MediaPipe Hands guesses "Left"/"Right" by assuming a MIRRORED (selfie) image, and our
        image is not mirrored, so its labels are backwards. A more reliable trick: a hand
        belongs to whichever wrist from MediaPipe Pose is closest to it in the image.
        """
        hand_wrists = [np.array([h.landmark[WRIST].x, h.landmark[WRIST].y])
                       for h in hand_results.multi_hand_landmarks]

        if self.pose_results.pose_landmarks is None:
            # no body found: fall back to MediaPipe's label, flipped because the image isn't mirrored
            labels = [h.classification[0].label for h in hand_results.multi_handedness]
            return ["right" if label == "Left" else "left" for label in labels]

        body = self.pose_results.pose_landmarks.landmark
        left_wrist = np.array([body[LEFT_WRIST].x, body[LEFT_WRIST].y])
        right_wrist = np.array([body[RIGHT_WRIST].x, body[RIGHT_WRIST].y])

        # how much closer each hand is to the left wrist than to the right wrist
        leftness = [np.linalg.norm(w - right_wrist) - np.linalg.norm(w - left_wrist) for w in hand_wrists]

        if len(hand_wrists) == 2:
            # two hands: the "more left" one is the left hand, the other one is the right hand
            sides = ["left", "right"] if leftness[0] > leftness[1] else ["right", "left"]
        else:
            sides = ["left" if leftness[0] > 0 else "right"]

        # a hand that is far from the wrist it was matched to is probably a false detection
        body_wrist = {"left": left_wrist, "right": right_wrist}
        return [side if np.linalg.norm(wrist - body_wrist[side]) < config.HAND_MATCH_MAX_DISTANCE else None
                for side, wrist in zip(sides, hand_wrists)]

    def draw(self, frame_bgr):
        """Draw the skeleton and hands from the last process() call onto a copy of the frame."""
        image = frame_bgr.copy()
        if not HAS_SOLUTIONS:
            return self._draw_without_solutions(image)
        if self.pose_results is not None and self.pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, self.pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
        if self.hand_results is not None and self.hand_results.multi_hand_landmarks:
            for hand_landmarks in self.hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())
        return image

    def _draw_without_solutions(self, image):
        if self.pose_results is not None and self.pose_results.pose_landmarks:
            mediapipe_tasks.draw_landmarks(image, self.pose_results.pose_landmarks,
                                           mediapipe_tasks.POSE_CONNECTIONS, (0, 200, 255))
        if self.hand_results is not None and self.hand_results.multi_hand_landmarks:
            for hand_landmarks in self.hand_results.multi_hand_landmarks:
                mediapipe_tasks.draw_landmarks(image, hand_landmarks,
                                               mediapipe_tasks.HAND_CONNECTIONS, (0, 0, 255))
        return image

    def close(self):
        if self.pose is not None:
            self.pose.close()
        if self.hands is not None:
            self.hands.close()


class TrackerThread:
    """
    Runs MediaPipe on the newest webcam frame in the BACKGROUND, so drawing the robot never has
    to wait for MediaPipe (each one takes ~30 ms; one after the other the app would be slow).

        thread = TrackerThread(webcam, HumanTracker())
        human, picture = thread.latest()     # whenever you want the newest result
    """

    def __init__(self, webcam, tracker):
        self.webcam = webcam
        self.tracker = tracker
        self.human = None        # newest HumanPose
        self.picture = None      # newest webcam frame with the skeleton drawn on it
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        last_frame_id = -1
        while self.running:
            frame_id, frame = self.webcam.read()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)    # no new frame yet: wait a moment
                continue
            last_frame_id = frame_id
            human = self.tracker.process(frame)
            picture = self.tracker.draw(frame)
            with self.lock:
                self.human, self.picture = human, picture

    def latest(self):
        with self.lock:
            return self.human, self.picture

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.tracker.close()
