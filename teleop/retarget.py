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
from teleop.geometry import bend, head_yaw_pitch, normalize, palm_rotation, rotation_angle, torso_rotation
from teleop.smoothing import (JumpFilter, OneEuroFilter, OneEuroRotationFilter, RotationSmoother,
                              Smoother)

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
    # ============================================================================ TODO 5
    # Like the mouth aspect ratio from smile detection, but for your hand.
    # hand[ht.WRIST], hand[ht.INDEX_TIP], ... are points [x, y, z] in meters, and
    # the distance between two points a and b is np.linalg.norm(a - b)
    palm_length = 0.0     # distance from the wrist to the middle knuckle (ht.MIDDLE_MCP)
    tip_distance = 0.0    # AVERAGE distance from the wrist to ht.INDEX_TIP and ht.MIDDLE_TIP
    ratio = config.GRIP_OPEN_RATIO   # tip_distance divided by palm_length (for now: "open hand")
    # ==================================================================================

    # turn the ratio into a grip number: an open hand (GRIP_OPEN_RATIO) -> 0, a fist (GRIP_CLOSED_RATIO) -> 1
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

        # ---- fine mode (press F): robot hands move only FINE_MODE_SCALE as far as yours ----
        self.fine_anchor = None

        # ---- filters ----
        self.last_time = None
        self.torso_filter = self._make_filter(rotation=True)
        self.upper_arm_filter = {s: self._make_filter() for s in SIDES}
        self.forearm_filter = {s: self._make_filter() for s in SIDES}
        self.wrist_rot_filter = {s: self._make_filter(rotation=True, hand=True) for s in SIDES}
        self.head_filter = self._make_filter()
        self.finger_smoother = {s: Smoother(config.FINGER_SMOOTHING) for s in SIDES}
        self.grip_smoother = {s: Smoother(config.GRIP_SMOOTHING) for s in SIDES}

        # glitch catchers: the last measurement we believed, and how big a jump we allow per frame
        on = config.REJECT_GLITCHES
        self.arm_jump = {s: JumpFilter(config.MAX_ARM_SPEED if on else None, config.GLITCH_PATIENCE)
                         for s in SIDES}
        self.hand_jump = {s: JumpFilter(config.MAX_HAND_TURN_SPEED if on else None, config.GLITCH_PATIENCE)
                          for s in SIDES}
        self.last_arm = {s: None for s in SIDES}
        self.last_palm = {s: None for s in SIDES}

    @staticmethod
    def _make_filter(rotation=False, hand=False):
        """Adaptive 1-Euro filter, or the simple average if ADAPTIVE_SMOOTHING is off."""
        if config.ADAPTIVE_SMOOTHING:
            min_cutoff = config.HAND_ROT_MIN_CUTOFF if hand else config.ONE_EURO_MIN_CUTOFF
            beta = config.HAND_ROT_BETA if hand else config.ONE_EURO_BETA
            return (OneEuroRotationFilter if rotation else OneEuroFilter)(min_cutoff, beta)
        return (RotationSmoother if rotation else Smoother)(config.SMOOTHING)

    @staticmethod
    def _trust(human, landmark_ids):
        """
        0..1: how much to believe these landmarks. MediaPipe gives each body landmark a
        'visibility'. Partly hidden landmarks move the robot only a little, hidden ones not at all.
        """
        visibility = min(human.body_visibility[i] for i in landmark_ids)
        trust = (visibility - config.MIN_LANDMARK_VISIBILITY) / (config.FULL_TRUST_VISIBILITY - config.MIN_LANDMARK_VISIBILITY)
        return float(np.clip(trust, 0.0, 1.0))

    # ------------------------------------------------------------------ torso
    def _measure_torso(self, body):
        """Rotation of your torso (already camera-fixed), or None if we can't see your shoulders."""
        if body is None:
            return None
        return torso_rotation(body[ht.LEFT_SHOULDER], body[ht.RIGHT_SHOULDER],
                              body[ht.LEFT_HIP], body[ht.RIGHT_HIP])

    def _update_torso(self, human, body, dt):
        if not config.TRACK_TORSO or human.body is None:
            return
        trust = self._trust(human, (ht.LEFT_SHOULDER, ht.RIGHT_SHOULDER))
        if trust > 0:
            self.torso_rot = self.torso_filter(self._measure_torso(body), dt, trust)

    # ------------------------------------------------------------------ arms
    def _arm_indices(self, side):
        if side == "left":
            return ht.LEFT_SHOULDER, ht.LEFT_ELBOW, ht.LEFT_WRIST
        return ht.RIGHT_SHOULDER, ht.RIGHT_ELBOW, ht.RIGHT_WRIST

    def _update_arm(self, human, body, side, dt):
        if human.body is None:
            return
        shoulder_id, elbow_id, wrist_id = self._arm_indices(side)
        trust = self._trust(human, (shoulder_id, elbow_id, wrist_id))
        if trust == 0:
            return   # can't see this arm: keep the old goal
        upper = normalize(body[elbow_id] - body[shoulder_id])
        fore = normalize(body[wrist_id] - body[elbow_id])

        # did the arm "teleport" since the last measurement we believed? Then it's probably a glitch.
        if self.last_arm[side] is not None:
            last_upper, last_fore = self.last_arm[side]
            jump = max(np.arccos(np.clip(np.dot(upper, last_upper), -1, 1)),
                       np.arccos(np.clip(np.dot(fore, last_fore), -1, 1)))
            if not self.arm_jump[side].accept(np.degrees(jump), dt):
                return
        self.last_arm[side] = (upper, fore)

        self.upper_arm_dir[side] = normalize(self.upper_arm_filter[side](upper, dt, trust))
        self.forearm_dir[side] = normalize(self.forearm_filter[side](fore, dt, trust))

    # ------------------------------------------------------------------ hands
    def _update_hand(self, human, side, dt):
        hand = human.left_hand if side == "left" else human.right_hand
        if hand is None:
            return
        # a squashed or blown-up hand means MediaPipe is guessing (usually a hidden hand)
        palm_length = np.linalg.norm(hand[ht.MIDDLE_MCP] - hand[ht.WRIST])
        if not 0.05 < palm_length < 0.15:
            return

        palm = self.camera_fix @ palm_rotation(hand)
        if self.last_palm[side] is not None:
            turn = rotation_angle(self.last_palm[side].T @ palm)
            if not self.hand_jump[side].accept(np.degrees(turn), dt):
                return   # the whole hand estimate flipped around: skip this frame (rotation AND grip)
        self.last_palm[side] = palm

        if config.USE_HAND_ORIENTATION:
            self.wrist_rot[side] = self.wrist_rot_filter[side](palm, dt)

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

    def _update_head(self, human, body, dt):
        angles = self._head_angles(human, body)
        if angles is None:
            return
        yaw, pitch = self.head_filter(angles - self.head_zero, dt)
        self.neck_yaw = config.HEAD_YAW_GAIN * yaw
        self.neck_pitch = config.HEAD_PITCH_OFFSET + config.HEAD_PITCH_GAIN * pitch

    # ------------------------------------------------------------------ public
    def update(self, human):
        """Read a HumanPose, update the goals and return them as RobotTargets."""
        # time since the last measurement (the adaptive filters need it)
        dt = 1 / 30 if self.last_time is None else float(np.clip(human.timestamp - self.last_time, 0.005, 0.2))
        self.last_time = human.timestamp

        body = None if human.body is None else human.body @ self.camera_fix.T
        self._update_torso(human, body, dt)
        for side in SIDES:
            self._update_arm(human, body, side, dt)
            self._update_hand(human, side, dt)
        self._update_head(human, body, dt)
        return self.targets()

    def _arm_targets(self):
        """Shoulder, elbow and wrist targets from the current goals, using the ROBOT's bone lengths."""
        pivot = self.robot["torso_pivot"]
        shoulder_pos, elbow_pos, wrist_pos = {}, {}, {}
        for side in SIDES:
            arm = self.robot[side]
            # leaning the torso swings the shoulder around the waist
            shoulder_pos[side] = pivot + self.torso_rot @ (arm["shoulder"] - pivot)
            elbow_pos[side] = shoulder_pos[side] + arm["upper_arm_length"] * self.upper_arm_dir[side]
            wrist_pos[side] = elbow_pos[side] + arm["forearm_length"] * self.forearm_dir[side]
        return shoulder_pos, elbow_pos, wrist_pos

    def targets(self):
        shoulder_pos, elbow_pos, wrist_pos = self._arm_targets()
        if self.fine_anchor is not None:
            # fine mode: only move FINE_MODE_SCALE as far as you do, starting where you pressed F
            for side in SIDES:
                anchor_elbow, anchor_wrist = self.fine_anchor[side]
                elbow_pos[side] = anchor_elbow + config.FINE_MODE_SCALE * (elbow_pos[side] - anchor_elbow)
                wrist_pos[side] = anchor_wrist + config.FINE_MODE_SCALE * (wrist_pos[side] - anchor_wrist)
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

    def toggle_fine_mode(self):
        """F key: precise, slow hands for lining up a grasp (also shrinks any leftover jitter)."""
        if self.fine_anchor is None:
            _, elbow_pos, wrist_pos = self._arm_targets()
            self.fine_anchor = {s: (elbow_pos[s].copy(), wrist_pos[s].copy()) for s in SIDES}
        else:
            self.fine_anchor = None
        return self.fine_anchor is not None

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
        for f in [self.torso_filter, self.head_filter, *self.upper_arm_filter.values(),
                  *self.forearm_filter.values(), *self.wrist_rot_filter.values()]:
            f.reset()
        self.last_arm = {s: None for s in SIDES}
        self.last_palm = {s: None for s in SIDES}
        for f in [*self.arm_jump.values(), *self.hand_jump.values()]:
            f.reset()

    def clear_calibration(self):
        self.camera_fix = np.eye(3)
        self.head_zero = np.array([0.0, config.HEAD_NEUTRAL_PITCH])
