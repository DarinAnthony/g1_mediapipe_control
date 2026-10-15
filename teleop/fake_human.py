"""
Build a pretend HumanPose (what MediaPipe would give us) from a few simple numbers,
so we can test retargeting + IK without a webcam.

Everything is in ROBOT axes (x forward, y left, z up), origin between the hips,
like HumanPose after the tracker converted it.
"""
import numpy as np

from teleop.human_tracker import HumanPose
from teleop.geometry import normalize

# rough adult proportions (meters)
HIP_HALF_WIDTH = 0.10
TORSO_LENGTH = 0.50
SHOULDER_HALF_WIDTH = 0.19
UPPER_ARM = 0.30
FOREARM = 0.26

FORWARD, BACK = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
LEFT, RIGHT = np.array([0, 1.0, 0]), np.array([0, -1.0, 0])
UP, DOWN = np.array([0, 0, 1.0]), np.array([0, 0, -1.0])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def torso_rot(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """pitch > 0 = lean forward, roll > 0 = lean to your right, yaw > 0 = twist to your left."""
    return rot_z(np.radians(yaw_deg)) @ rot_y(np.radians(pitch_deg)) @ rot_x(np.radians(roll_deg))


def hand_frame(fingers_dir, palm_dir, side):
    """
    Rotation of a hand whose fingers point along fingers_dir and whose palm faces palm_dir,
    using the same axis convention as geometry.palm_rotation.
    """
    x = normalize(fingers_dir)
    n = normalize(palm_dir - np.dot(palm_dir, x) * x)
    y = -n if side == "left" else n
    return np.column_stack([x, y, np.cross(x, y)])


def hand_landmarks(rotation, side, closure=0.0):
    """21 hand landmarks around the origin. closure 0 = open flat hand, 1 = fist-ish."""
    x = np.array([1.0, 0, 0])
    z = np.array([0, 0, 1.0])
    palm_normal = np.array([0, -1.0, 0]) if side == "left" else np.array([0, 1.0, 0])

    def finger(knuckle, lengths, bends):
        points, direction_angle, p = [], 0.0, np.array(knuckle, dtype=float)
        for length, b in zip(lengths, bends):
            direction_angle += b
            p = p + length * (np.cos(direction_angle) * x + np.sin(direction_angle) * palm_normal)
            points.append(p)
        return points

    c = closure
    pts = [np.zeros(3)]                                              # 0 wrist
    # thumb: starts on the index side, swings toward the palm when closing
    cmc = np.array([0.025, 0.0, 0.025]) + 0.01 * palm_normal
    thumb_dir = normalize(0.6 * x + 0.8 * z)
    p = cmc.copy()
    pts.append(p.copy())                                             # 1 thumb CMC
    angle = 0.0
    for length, b in zip((0.04, 0.03, 0.025), (0.0, 0.6 * c, 0.8 * c)):
        angle += b
        p = p + length * (np.cos(angle) * thumb_dir + np.sin(angle) * palm_normal)
        pts.append(p.copy())                                         # 2, 3, 4
    knuckles = {"index": (0.085, 0, 0.022), "middle": (0.09, 0, 0.0),
                "ring": (0.085, 0, -0.018), "pinky": (0.075, 0, -0.035)}
    for name, knuckle in knuckles.items():
        pts.append(np.array(knuckle))                                # MCP
        pts.extend(finger(knuckle, (0.04, 0.025, 0.02), (1.3 * c, 1.5 * c, 1.0 * c)))
    hand = np.array(pts)
    hand -= hand.mean(axis=0)            # MediaPipe hand world landmarks are centered on the hand
    return hand @ rotation.T


def make_human(torso=(0, 0, 0), left_arm=(DOWN, FORWARD), right_arm=(DOWN, FORWARD),
               left_hand=None, right_hand=None, closure=(0.0, 0.0), head=(0.0, 0.0), timestamp=0.0):
    """
    torso:     (yaw, pitch, roll) in degrees
    left_arm:  (upper arm direction, forearm direction), world axes
    left_hand: hand rotation (3x3, from hand_frame), or None for "handshake" (fingers forward)
    closure:   (left, right) how closed each hand is, 0..1
    head:      (yaw, pitch) of your head relative to your torso, degrees (pitch > 0 = look down)
    timestamp: time of this pretend webcam frame, in seconds
    """
    body = np.zeros((33, 3))
    R = torso_rot(*torso)
    body[23] = R @ (HIP_HALF_WIDTH * LEFT)
    body[24] = R @ (HIP_HALF_WIDTH * RIGHT)
    shoulder_mid = R @ (TORSO_LENGTH * UP)
    body[11] = shoulder_mid + R @ (SHOULDER_HALF_WIDTH * LEFT)
    body[12] = shoulder_mid + R @ (SHOULDER_HALF_WIDTH * RIGHT)

    neck = shoulder_mid + R @ (0.08 * UP)
    R_head = R @ rot_z(np.radians(head[0])) @ rot_y(np.radians(head[1]))
    body[0] = neck + R_head @ np.array([0.10, 0, 0.08])                    # nose (a bit below the ears)
    body[7] = neck + R_head @ np.array([0.0, 0.075, 0.10])                 # left ear
    body[8] = neck + R_head @ np.array([0.0, -0.075, 0.10])                # right ear

    hands = {}
    for side, (upper, fore), shoulder_id, elbow_id, wrist_id, rot, c in (
            ("left", left_arm, 11, 13, 15, left_hand, closure[0]),
            ("right", right_arm, 12, 14, 16, right_hand, closure[1])):
        body[elbow_id] = body[shoulder_id] + UPPER_ARM * normalize(upper)
        body[wrist_id] = body[elbow_id] + FOREARM * normalize(fore)
        if rot is None:
            rot = hand_frame(FORWARD, RIGHT if side == "left" else LEFT, side)
        hands[side] = hand_landmarks(rot, side, c)

    visibility = np.zeros(33)
    visibility[[0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]] = 1.0
    return HumanPose(body=body, body_visibility=visibility,
                     left_hand=hands["left"], right_hand=hands["right"], timestamp=timestamp)
