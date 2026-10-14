"""
Backup plan for NEW MediaPipe versions (1.x, used on Python 3.13+), which removed mp.solutions.

It runs the newer "Tasks" API (PoseLandmarker + HandLandmarker) and repackages the results to
look exactly like the old mp.solutions results, so human_tracker.py works the same either way.
You don't need to read this file for class; human_tracker.py picks it automatically.
"""
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models"


def _with_landmark_list(landmarks):
    """mp.solutions results keep their points in a .landmark list."""
    return SimpleNamespace(landmark=landmarks)


class _VideoClock:
    """The Tasks API wants a timestamp (ms) that always goes up."""

    def __init__(self):
        self.last = -1

    def now(self):
        self.last = max(self.last + 1, int(time.monotonic() * 1000))
        return self.last


def _to_mp_image(rgb_image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))


class Pose:
    """Stand-in for mp.solutions.pose.Pose."""

    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "pose_landmarker_full.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.clock = _VideoClock()

    def process(self, rgb_image):
        result = self.landmarker.detect_for_video(_to_mp_image(rgb_image), self.clock.now())
        if not result.pose_landmarks:
            return SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None)
        world = result.pose_world_landmarks[0]
        for point in world:   # some versions leave visibility empty on world points
            if point.visibility is None:
                point.visibility = 1.0
        return SimpleNamespace(pose_landmarks=_with_landmark_list(result.pose_landmarks[0]),
                               pose_world_landmarks=_with_landmark_list(world))

    def close(self):
        self.landmarker.close()


class Hands:
    """Stand-in for mp.solutions.hands.Hands."""

    def __init__(self, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.clock = _VideoClock()

    def process(self, rgb_image):
        result = self.landmarker.detect_for_video(_to_mp_image(rgb_image), self.clock.now())
        if not result.hand_landmarks:
            return SimpleNamespace(multi_hand_landmarks=None, multi_hand_world_landmarks=None,
                                   multi_handedness=None)
        return SimpleNamespace(
            multi_hand_landmarks=[_with_landmark_list(h) for h in result.hand_landmarks],
            multi_hand_world_landmarks=[_with_landmark_list(h) for h in result.hand_world_landmarks],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label=c[0].category_name)])
                              for c in result.handedness],
        )

    def close(self):
        self.landmarker.close()


POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def draw_landmarks(image, landmarks, connections, color):
    """Simple replacement for mp.solutions.drawing_utils.draw_landmarks."""
    h, w = image.shape[:2]
    points = [(int(p.x * w), int(p.y * h)) for p in landmarks.landmark]
    for start, end in connections:
        cv2.line(image, points[start], points[end], (255, 255, 255), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
