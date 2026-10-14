"""
All the numbers you might want to tweak live in this one file.

Angles are in RADIANS unless the name says otherwise (1 radian is about 57 degrees).
Distances are in METERS.
"""

# ---------------------------------------------------------------- webcam + MediaPipe
WEBCAM_INDEX = 0                 # try 1 or 2 if you have more than one camera
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

POSE_MODEL_COMPLEXITY = 1        # 0 = fastest, 1 = balanced, 2 = most accurate (slow)
HAND_MODEL_COMPLEXITY = 1        # 0 = fastest, 1 = more accurate
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
MIN_LANDMARK_VISIBILITY = 0.5    # ignore a body landmark if MediaPipe is less sure than this

# ---------------------------------------------------------------- windows
KITCHEN_WINDOW = "Kitchen view"
ROBOT_EYES_WINDOW = "Robot eyes (first person)"
KITCHEN_VIEW_SIZE = (960, 720)   # (width, height) in pixels
ROBOT_EYES_SIZE = (800, 600)
SHOW_WEBCAM_INSET = True         # small picture of your webcam in the corner of the kitchen view

# ---------------------------------------------------------------- human -> robot retargeting
# How much to smooth the motion. 0 = no smoothing (jittery), 0.9 = very smooth (laggy).
SMOOTHING = 0.6

# Copy your torso lean/twist onto the robot's waist? (If you sit at a desk and MediaPipe
# guesses your hips badly, try turning this off.)
TRACK_TORSO = True

# Copy the rotation of your palm onto the robot's hand?
USE_HAND_ORIENTATION = True

# ---------------------------------------------------------------- robot hands
# How your hand controls the robot hand (press H in main.py to switch while it runs):
#   "open_close"  make a fist -> robot hand closes, open your hand -> it opens. Most stable,
#                 best for picking things up.
#   "grip"        the robot hand closes exactly as much as yours (one smooth number per hand)
#   "fingers"     copy every finger joint separately. Most detailed, but jittery.
HAND_MODE = "open_close"

# "Grip" = how closed your hand is, 0 (open) .. 1 (fist). It comes from
# (fingertip-to-wrist distance) / (palm length): about 1.8-1.9 for an open hand, 0.8-1.0 for a fist.
GRIP_OPEN_RATIO = 1.8
GRIP_CLOSED_RATIO = 1.0
GRIP_SMOOTHING = 0.6

# open_close mode: close when grip goes ABOVE this, open again only when it drops BELOW this.
# The gap between the two numbers is what stops the hand from flickering.
GRIP_CLOSE_ABOVE = 0.65
GRIP_OPEN_BELOW = 0.4

# fingers mode: robot joint angle = gain * how much your finger joint is bent
FINGER_GAIN = 1.3
THUMB_GAIN = 1.6
FINGER_SMOOTHING = 0.8

# Head -> robot neck
HEAD_YAW_GAIN = 1.0
HEAD_PITCH_GAIN = 1.0
# When you look straight at your screen the robot looks DOWN at the counter by this much.
HEAD_PITCH_OFFSET = 0.6
# Your nose sits a bit lower than your ears, so looking straight ahead measures as ~0.2 rad "down".
# Calibrating (press C) replaces this with your own number.
HEAD_NEUTRAL_PITCH = 0.2

# ---------------------------------------------------------------- inverse kinematics (IK)
IK_ITERATIONS = 10               # more = more accurate but slower
IK_DAMPING = 0.05                # keeps the arms from going crazy when the goal is out of reach

# How much each IK task matters. Every error is multiplied by its weight before they are
# compared. Positions are in meters and rotations in radians, so with a rotation weight of 0.3,
# being 0.1 rad (6 degrees) off counts the same as being 3 cm off (0.3 * 0.1 = 0.03).
IK_TORSO_WEIGHT = 0.5
IK_ELBOW_WEIGHT = 0.5
IK_WRIST_WEIGHT = 1.0
IK_WRIST_ROTATION_WEIGHT = 0.3

IK_MAX_STEP = 0.2                # largest joint change (radians) in a single IK iteration

# Safety: a joint target may not change faster than this (radians per second)
MAX_JOINT_SPEED = 4.0
