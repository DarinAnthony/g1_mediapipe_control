"""
Retargeting = turning what the HUMAN is doing into goals for the ROBOT.

People and the G1 have different sizes, so we don't copy positions directly. Instead:

    your torso lean / twist         ->  the robot torso should be rotated the same way
    your upper arm + forearm        ->  copy the DIRECTION of each bone, but use the robot's
                                        bone LENGTHS, starting from the robot's shoulder.
                                        That gives target points for the robot's elbow and wrist.
    your palm rotation              ->  the robot's wrist rotation
    your finger bends               ->  robot finger joint angles (direct copy)
    your head turn / nod            ->  robot neck joint angles (direct copy)

The torso, elbow and wrist targets are then handed to the IK solver (ik.py), which finds the
waist + arm joint angles that hit all of them at once.

All robot positions and rotations here are in the WORLD frame (the robot's pelvis is bolted
to the world, facing +x, so world axes = robot axes: x forward, y left, z up).
"""
from dataclasses import dataclass, field

import numpy as np

import config
from teleop import human_tracker as ht
from teleop.geometry import bend, head_yaw_pitch, normalize, palm_rotation, torso_rotation
from teleop.smoothing import RotationSmoother, Smoother

SIDES = ("left", "right")


@dataclass
class RobotTargets:
    """What we want the robot to do right now (world frame)."""
    torso_rot: np.ndarray        # (3, 3) how the robot torso should be rotated
    shoulder_pos: dict           # side -> (3,) where the shoulder ends up (follows from torso_rot)
    elbow_pos: dict              # side -> (3,) where the elbow should be
    wrist_pos: dict              # side -> (3,) where the wrist should be
    wrist_rot: dict              # side -> (3, 3) how the wrist/hand should be rotated
    hand_joints: dict = field(default_factory=dict)   # finger joint name -> angle
    neck_yaw: float = 0.0
    neck_pitch: float = config.HEAD_PITCH_OFFSET


def finger_joint_angles(hand, side):
    """
    Turn 21 hand landmarks into the 7 joint angles of one Dex3 hand.

    The Dex3 hand has a thumb (3 joints), an index finger (2 joints) and a middle finger (2 joints).
    The left and right robot hands are mirror images, so "closing" is a NEGATIVE angle for the
    left fingers and a POSITIVE angle for the right fingers (and the other way around for thumbs).
    """
    finger_sign = -1.0 if side == "left" else 1.0
    thumb_sign = -finger_sign

    # how bent each of your finger joints is (0 = straight)
    index_knuckle = bend(hand[ht.WRIST], hand[ht.INDEX_MCP], hand[ht.INDEX_PIP])
    index_middle_joint = bend(hand[ht.INDEX_MCP], hand[ht.INDEX_PIP], hand[ht.INDEX_DIP])
    middle_knuckle = bend(hand[ht.WRIST], hand[ht.MIDDLE_MCP], hand[ht.MIDDLE_PIP])
    middle_middle_joint = bend(hand[ht.MIDDLE_MCP], hand[ht.MIDDLE_PIP], hand[ht.MIDDLE_DIP])
    thumb_base = bend(hand[ht.THUMB_CMC], hand[ht.THUMB_MCP], hand[ht.THUMB_IP])
    thumb_tip = bend(hand[ht.THUMB_MCP], hand[ht.THUMB_IP], hand[ht.THUMB_TIP])

    g, tg = config.FINGER_GAIN, config.THUMB_GAIN
    return {
        f"{side}_hand_index_0_joint": finger_sign * g * index_knuckle,
        f"{side}_hand_index_1_joint": finger_sign * g * index_middle_joint,
        f"{side}_hand_middle_0_joint": finger_sign * g * middle_knuckle,
        f"{side}_hand_middle_1_joint": finger_sign * g * middle_middle_joint,
        f"{side}_hand_thumb_0_joint": 0.0,   # twists the thumb toward the index or middle finger
        f"{side}_hand_thumb_1_joint": thumb_sign * tg * thumb_base,
        f"{side}_hand_thumb_2_joint": thumb_sign * tg * thumb_tip,
    }


def grip_amount(hand):
    """
    How closed your hand is: 0 = open flat hand, 1 = fist.

    We look at how far your index and middle fingertips are from your wrist, divided by your
    palm length (wrist -> middle knuckle). Dividing makes it work for big and small hands and
    near or far from the camera. Open hand: about 1.8-1.9. Fist: about 0.8-1.0.
    """
    palm_length = np.linalg.norm(hand[ht.MIDDLE_MCP] - hand[ht.WRIST])
    tip_distances = [np.linalg.norm(hand[tip] - hand[ht.WRIST]) for tip in (ht.INDEX_TIP, ht.MIDDLE_TIP)]
    ratio = np.mean(tip_distances) / palm_length
    grip = (config.GRIP_OPEN_RATIO - ratio) / (config.GRIP_OPEN_RATIO - config.GRIP_CLOSED_RATIO)
    return float(np.clip(grip, 0.0, 1.0))


def dex3_grip_angles(side, closure):
    """All 7 Dex3 joints for a hand that is 'closure' closed (0 = open, 1 = grasping)."""
    finger_sign = -1.0 if side == "left" else 1.0
    thumb_sign = -finger_sign
    return {
        f"{side}_hand_index_0_joint": finger_sign * 1.2 * closure,
        f"{side}_hand_index_1_joint": finger_sign * 1.2 * closure,
        f"{side}_hand_middle_0_joint": finger_sign * 1.2 * closure,
        f"{side}_hand_middle_1_joint": finger_sign * 1.2 * closure,
        f"{side}_hand_thumb_0_joint": 0.0,
        f"{side}_hand_thumb_1_joint": thumb_sign * 1.0 * closure,
        f"{side}_hand_thumb_2_joint": thumb_sign * 1.2 * closure,
    }


class Retargeter:
    def __init__(self, robot):
        """
        robot: from KitchenSim.robot_body_info() -- the robot's torso pivot, shoulder positions,
               bone lengths and home pose, measured with every joint at 0.
        """
        self.robot = robot

        # ---- the current goals, starting at the robot's home pose ----
        self.torso_rot = np.eye(3)
        self.upper_arm_dir = {s: normalize(robot[s]["home_elbow"] - robot[s]["shoulder"]) for s in SIDES}
        self.forearm_dir = {s: normalize(robot[s]["home_wrist"] - robot[s]["home_elbow"]) for s in SIDES}
        self.wrist_rot = {s: robot[s]["home_wrist_rot"].copy() for s in SIDES}
        self.hand_joints = {}
        self.grip = {s: 0.0 for s in SIDES}            # 0 = open .. 1 = fist (after smoothing)
        self.hand_closed = {s: False for s in SIDES}   # open/closed state for "open_close" mode
        self.neck_yaw = 0.0
        self.neck_pitch = config.HEAD_PITCH_OFFSET

        # ---- calibration (press C) ----
        # camera_fix undoes a tilted webcam: it rotates your "sitting straight" pose to upright.
        self.camera_fix = np.eye(3)
        self.head_zero = np.array([0.0, config.HEAD_NEUTRAL_PITCH])   # (yaw, pitch) of "looking straight"

        # ---- smoothing ----
        self.torso_smoother = RotationSmoother(config.SMOOTHING)
        self.upper_arm_smoother = {s: Smoother(config.SMOOTHING) for s in SIDES}
        self.forearm_smoother = {s: Smoother(config.SMOOTHING) for s in SIDES}
        self.wrist_rot_smoother = {s: RotationSmoother(config.SMOOTHING) for s in SIDES}
        self.finger_smoother = {s: Smoother(config.FINGER_SMOOTHING) for s in SIDES}
        self.grip_smoother = {s: Smoother(config.GRIP_SMOOTHING) for s in SIDES}
        self.head_smoother = Smoother(config.SMOOTHING)

    # ------------------------------------------------------------------ torso
    def _measure_torso(self, body):
        """Rotation of your torso (already camera-fixed), or None if we can't see your shoulders."""
        if body is None:
            return None
        return torso_rotation(body[ht.LEFT_SHOULDER], body[ht.RIGHT_SHOULDER],
                              body[ht.LEFT_HIP], body[ht.RIGHT_HIP])

    def _update_torso(self, human, body):
        if not human.can_see(ht.LEFT_SHOULDER, ht.RIGHT_SHOULDER):
            return
        if config.TRACK_TORSO:
            self.torso_rot = self.torso_smoother(self._measure_torso(body))

    # ------------------------------------------------------------------ arms
    def _arm_indices(self, side):
        if side == "left":
            return ht.LEFT_SHOULDER, ht.LEFT_ELBOW, ht.LEFT_WRIST
        return ht.RIGHT_SHOULDER, ht.RIGHT_ELBOW, ht.RIGHT_WRIST

    def _update_arm(self, human, body, side):
        shoulder_id, elbow_id, wrist_id = self._arm_indices(side)
        if not human.can_see(shoulder_id, elbow_id, wrist_id):
            return   # can't see this arm: keep the old goal
        shoulder, elbow, wrist = body[shoulder_id], body[elbow_id], body[wrist_id]
        self.upper_arm_dir[side] = normalize(self.upper_arm_smoother[side](normalize(elbow - shoulder)))
        self.forearm_dir[side] = normalize(self.forearm_smoother[side](normalize(wrist - elbow)))

    def _update_hand(self, human, side):
        hand = human.left_hand if side == "left" else human.right_hand
        if hand is None:
            return
        if config.USE_HAND_ORIENTATION:
            palm = self.camera_fix @ palm_rotation(hand)
            self.wrist_rot[side] = self.wrist_rot_smoother[side](palm)

        self.grip[side] = float(self.grip_smoother[side](grip_amount(hand)))

        if config.HAND_MODE == "fingers":
            # copy every finger joint (detailed, but MediaPipe's noise shows up in 7 joints)
            angles = finger_joint_angles(hand, side)
            smoothed = self.finger_smoother[side](list(angles.values()))
            self.hand_joints.update(zip(angles.keys(), smoothed))

        elif config.HAND_MODE == "grip":
            # one number per hand: the robot hand closes exactly as much as yours
            self.hand_joints.update(dex3_grip_angles(side, self.grip[side]))

        else:  # "open_close"
            # Two different thresholds (hysteresis): once closed, the hand stays closed until you
            # open it a lot. With one threshold, a hand right at the edge would flicker open/shut.
            if self.hand_closed[side] and self.grip[side] < config.GRIP_OPEN_BELOW:
                self.hand_closed[side] = False
            elif not self.hand_closed[side] and self.grip[side] > config.GRIP_CLOSE_ABOVE:
                self.hand_closed[side] = True
            self.hand_joints.update(dex3_grip_angles(side, 1.0 if self.hand_closed[side] else 0.0))

    # ------------------------------------------------------------------ head
    def _head_angles(self, human, body):
        """Head yaw/pitch measured RELATIVE TO YOUR TORSO (the robot's neck sits on its torso)."""
        if not human.can_see(ht.NOSE, ht.LEFT_EAR, ht.RIGHT_EAR):
            return None
        torso = self._measure_torso(body) if config.TRACK_TORSO else np.eye(3)
        nose, left_ear, right_ear = (torso.T @ body[i] for i in (ht.NOSE, ht.LEFT_EAR, ht.RIGHT_EAR))
        return np.array(head_yaw_pitch(nose, left_ear, right_ear))

    def _update_head(self, human, body):
        angles = self._head_angles(human, body)
        if angles is None:
            return
        yaw, pitch = self.head_smoother(angles - self.head_zero)
        self.neck_yaw = config.HEAD_YAW_GAIN * yaw
        self.neck_pitch = config.HEAD_PITCH_OFFSET + config.HEAD_PITCH_GAIN * pitch

    # ------------------------------------------------------------------ public
    def update(self, human):
        """Read a HumanPose, update the goals and return them as RobotTargets."""
        body = None if human.body is None else human.body @ self.camera_fix.T
        self._update_torso(human, body)
        for side in SIDES:
            self._update_arm(human, body, side)
            self._update_hand(human, side)
        self._update_head(human, body)
        return self.targets()

    def targets(self):
        """Build robot target points from the current goals, using the ROBOT's bone lengths."""
        pivot = self.robot["torso_pivot"]
        shoulder_pos, elbow_pos, wrist_pos = {}, {}, {}
        for side in SIDES:
            arm = self.robot[side]
            # leaning the torso swings the shoulder around the waist
            shoulder_pos[side] = pivot + self.torso_rot @ (arm["shoulder"] - pivot)
            elbow_pos[side] = shoulder_pos[side] + arm["upper_arm_length"] * self.upper_arm_dir[side]
            wrist_pos[side] = elbow_pos[side] + arm["forearm_length"] * self.forearm_dir[side]
        return RobotTargets(
            torso_rot=self.torso_rot.copy(),
            shoulder_pos=shoulder_pos,
            elbow_pos=elbow_pos,
            wrist_pos=wrist_pos,
            wrist_rot={s: self.wrist_rot[s].copy() for s in SIDES},
            hand_joints=dict(self.hand_joints),
            neck_yaw=self.neck_yaw,
            neck_pitch=self.neck_pitch,
        )

    def calibrate(self, human):
        """
        Call this while you sit/stand up straight, face the screen and look at it.
        Whatever your torso looks like right now becomes "upright", which also cancels out
        a webcam that is tilted up or down. Your head angle right now becomes "straight ahead".
        """
        if human.can_see(ht.LEFT_SHOULDER, ht.RIGHT_SHOULDER):
            self.camera_fix = self._measure_torso(human.body).T
        body = human.body @ self.camera_fix.T if human.body is not None else None
        angles = self._head_angles(human, body)
        if angles is not None:
            self.head_zero = angles
        for smoother in [self.torso_smoother, self.head_smoother,
                         *self.upper_arm_smoother.values(), *self.forearm_smoother.values(),
                         *self.wrist_rot_smoother.values()]:
            smoother.reset()

    def clear_calibration(self):
        self.camera_fix = np.eye(3)
        self.head_zero = np.array([0.0, config.HEAD_NEUTRAL_PITCH])
