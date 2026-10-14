"""
Small math helpers: changing coordinate frames, angles between points, and
building rotation matrices out of hand landmarks.
"""
import numpy as np

# MediaPipe "world" coordinates (meters):
#     x -> toward the RIGHT side of the webcam image
#     y -> DOWN
#     z -> AWAY from the webcam
#
# Robot coordinates (meters):
#     x -> FORWARD (the way the robot faces)
#     y -> the robot's LEFT
#     z -> UP
#
# You face the webcam and the robot faces you, so "forward" for the robot is
# "toward the camera" (-z in MediaPipe). Your LEFT arm drives the robot's LEFT arm.
# The webcam image is not mirrored, so your left side shows up on the RIGHT of the image (+x).
#
#     robot x = -mediapipe z
#     robot y = +mediapipe x
#     robot z = -mediapipe y
MEDIAPIPE_TO_ROBOT = np.array([
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def to_robot_frame(points):
    """Convert one point (3,) or many points (N, 3) from MediaPipe axes to robot axes."""
    return np.asarray(points, dtype=float) @ MEDIAPIPE_TO_ROBOT.T


def normalize(v):
    """Return v scaled to length 1."""
    return v / (np.linalg.norm(v) + 1e-9)


def calc_angle(a, b, c):
    """
    Angle ABC in DEGREES (the angle at point b), same idea as the bicep curl app.
    A straight line a-b-c gives 180, a sharp fold gives close to 0.
    """
    ba = np.asarray(a) - np.asarray(b)
    bc = np.asarray(c) - np.asarray(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # tiny rounding errors can break arccos
    return np.degrees(np.arccos(cosine_angle))


def bend(a, b, c):
    """How much the joint at b is bent, in RADIANS. Straight = 0."""
    return np.radians(180.0 - calc_angle(a, b, c))


def palm_rotation(hand):
    """
    Build a 3x3 rotation matrix for a hand from its 21 landmarks (already in robot axes).

    The columns are the hand's own x, y, z axes, chosen to match the G1's Dex3 hand:
        x = from the wrist toward the middle finger   (where the fingers point)
        y = straight through the palm                  (out of the palm for a right hand,
                                                        out of the back for a left hand)
        z = x cross y                                   (toward the index finger side)

    Hand landmark numbers: 0 = wrist, 5 = index knuckle, 9 = middle knuckle, 17 = pinky knuckle.
    """
    wrist, index_knuckle, middle_knuckle, pinky_knuckle = hand[0], hand[5], hand[9], hand[17]

    x_axis = normalize(middle_knuckle - wrist)

    # The cross product of two vectors lying in the palm sticks straight out of the palm.
    # Because left and right hands are mirror images (and so are the robot's), this same
    # formula works for BOTH hands.
    y_axis = np.cross(index_knuckle - wrist, pinky_knuckle - wrist)
    y_axis = normalize(y_axis - np.dot(y_axis, x_axis) * x_axis)  # make it exactly 90 deg to x

    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def torso_rotation(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    A 3x3 rotation matrix for your upper body. Its columns are:
        forward (out of your chest), left (toward your left shoulder), up (hips -> shoulders)
    Sitting up straight and facing the camera gives the identity matrix.
    """
    up = normalize((left_shoulder + right_shoulder) / 2.0 - (left_hip + right_hip) / 2.0)
    left = left_shoulder - right_shoulder
    left = normalize(left - np.dot(left, up) * up)   # make it exactly 90 deg to "up"
    forward = np.cross(left, up)
    return np.column_stack([forward, left, up])


def rotation_angle(rotation):
    """How far (radians) a rotation matrix turns things. Identity -> 0."""
    return np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))


def head_yaw_pitch(nose, left_ear, right_ear):
    """
    Estimate where your head is pointing from 3 body landmarks (robot axes).
    Returns (yaw, pitch) in radians:
        yaw   > 0  means you turned your head to YOUR left
        pitch > 0  means you are looking DOWN
    """
    ear_axis = left_ear - right_ear               # points out of your left ear
    yaw = np.arctan2(-ear_axis[0], ear_axis[1])

    ear_middle = (left_ear + right_ear) / 2.0
    nose_direction = nose - ear_middle            # points out of your face
    pitch = np.arctan2(-nose_direction[2], np.hypot(nose_direction[0], nose_direction[1]))
    return yaw, pitch


def closest_rotation(matrix):
    """Snap a 'almost rotation' 3x3 matrix (e.g. after averaging) back to a real rotation."""
    u, _, vt = np.linalg.svd(matrix)
    r = u @ vt
    if np.linalg.det(r) < 0:   # reflection, not a rotation: flip one axis
        u[:, -1] *= -1
        r = u @ vt
    return r
