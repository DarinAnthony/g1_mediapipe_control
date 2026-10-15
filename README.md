# G1 Kitchen Teleop

Control a simulated **Unitree G1 humanoid with Dex3 hands** using only your webcam.
MediaPipe tracks your upper body, head, and fingers (thumb, index, middle). That motion is
retargeted onto the robot, which stands behind a kitchen island with objects you can pick up.

```
webcam ──► MediaPipe Pose + Hands ──► retargeting ──► IK (waist + both arms) ──► MuJoCo G1
           (body, head, fingers)      (torso, bones,   (torso, elbows, wrists     (position motors,
                                       palm, fingers,   at the same time)          physics, cameras)
                                       head)
```

Two windows open:

| Window | What you see |
|---|---|
| **Kitchen view** | The island and the robot, seen from across the counter. Drag the mouse to orbit. Your webcam is in the corner. |
| **Robot eyes (first person)** | The robot's head camera. It turns with **your** head, so you can look at what you're grabbing. |

## Class TODOs (about 10 minutes)

Five short pieces of MediaPipe code are left for you to write. Search for `TODO` in these files:

| TODO | File | What you write |
|---|---|---|
| 1 | `teleop/human_tracker.py` | The body landmark numbers (nose, ears, shoulders, elbows, wrists, hips) |
| 2 | `teleop/human_tracker.py` | Create the two MediaPipe models: `mp_pose.Pose()` and `mp_hands.Hands()` |
| 3 | `teleop/human_tracker.py` | Convert the webcam picture to RGB and run both models on it |
| 4 | `teleop/human_tracker.py` | Turn the 33 body landmarks into numpy arrays (x, y, z and visibility) |
| 5 | `teleop/retarget.py` | `grip_amount`: how closed your hand is (like the smile mouth ratio!) |

After each TODO, run **Check my code** (`check_my_code.py`). It tests your code on a photo
and on pretend hands (no webcam needed) and tells you what to fix. The app runs the whole
time; the robot just stands still until your TODOs are done.

## Setup

**Works on:** Windows, macOS, and Linux with **Python 3.10 - 3.14** (3.12 recommended, same
as the other class projects). Intel Macs need Python 3.10 - 3.12, because newer MediaPipe
versions don't support Intel Macs.

`requirements.txt` only lists the four packages we use directly (mediapipe, mujoco, numpy,
opencv). pip picks the right helper packages for each computer. On Python 3.13+ it installs
MediaPipe 1.x, which no longer has `mp.solutions`; the code notices and switches to MediaPipe's
newer API by itself (`teleop/mediapipe_tasks.py`, models in `assets/models/`).

### In PyCharm (easiest)

1. **File > Open** and pick this folder (or **File > New Project from Version Control** and
   paste `https://github.com/DarinAnthony/g1_mediapipe_control.git`).
2. If PyCharm asks about the interpreter: **Add New Interpreter > Add Local Interpreter >
   Virtualenv Environment > New**, base interpreter **Python 3.12**, location `.venv` inside
   this folder.
3. Open `requirements.txt` and click **Install requirements** in the yellow bar (or run the
   pip command below in PyCharm's Terminal tab).

### In a terminal

macOS / Linux:

```bash
git clone https://github.com/DarinAnthony/g1_mediapipe_control.git
cd g1_mediapipe_control
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell or Command Prompt):

```bat
git clone https://github.com/DarinAnthony/g1_mediapipe_control.git
cd g1_mediapipe_control
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

In PyCharm, pick a run configuration from the dropdown at the top right and press Run:

| Run configuration | File | Webcam? | What it does |
|---|---|---|---|
| **Check my code** | `check_my_code.py` | no | Checks your class TODOs |
| **Step 1 - tracking only** | `step1_tracking_only.py` | yes | MediaPipe skeleton + the angles the robot will use |
| **Step 2 - IK playground** | `step2_ik_playground.py` | no | Move the robot's hands with the keyboard and watch the IK |
| **Teleop (main)** | `main.py` | yes | The full thing |
| **Test - IK motions** | `tests/test_ik_motions.py` | no | Checks 14 upper-body motions and saves `tests/output/ik_motions.png` |
| **Test - hand modes** | `tests/test_hand_modes.py` | no | Shakes pretend hands like MediaPipe does and measures how jittery each hand mode is |
| **Test - occlusion** | `tests/test_occlusion.py` | no | Fake hidden-hand glitches; compares the old and new filtering |

Or from a terminal with the venv activated: `python step1_tracking_only.py`, `python main.py`, etc.
`python main.py --video clip.mp4` uses a video file instead of the webcam, and
`python main.py --record clip.mp4` saves your webcam session so you can replay it later.

### Teleop controls (click on a window first)

| Key | |
|---|---|
| `SPACE` | pause / resume copying you |
| `F` | **fine mode**: the robot's hands move only 40% as far as yours (precise grasping, less jitter) |
| `C` | **calibrate**: sit/stand up straight, face the screen, look at it, press C |
| `X` | clear calibration |
| `R` | reset robot + objects |
| `T` | show the IK targets (your pose resized to the robot, in yellow) |
| `V` | webcam picture on/off |
| `K` | mouse camera ↔ fixed kitchen camera |
| `H` | hand mode: open_close → grip → fingers |
| `[` `]` | zoom |
| `Q` / `ESC` | quit |

**Tips:** your shoulders, elbows, and hands need to be in the picture. Good lighting helps
MediaPipe a lot. Press `C` once at the start. That also cancels out a webcam that's tilted.

### When your hands get close to the table

A single webcam can't see a hand that's hidden behind the desk, turned edge-on, or out of
frame, so MediaPipe starts **guessing**, and the guesses jump around. Code can't bring back
what the camera can't see, but it can stop believing bad guesses:

* **Visibility trust:** MediaPipe rates how visible each body point is. Half-hidden points
  move the robot only partway, and hidden ones are ignored (the robot holds still).
* **Glitch rejection:** an arm turning faster than 600°/s or a hand faster than 450°/s is
  physically impossible, so that frame is ignored. If it's still there 4 frames later, it was real.
* **Adaptive "1 Euro" smoothing:** heavy smoothing while you move slowly (lining up a grasp),
  light smoothing when you move fast (no lag).
* **Fine mode (`F`):** the robot hand moves 40% as far as you do, which also shrinks any
  leftover jitter to 40%.

In a simulated test with occlusion glitches (`tests/test_occlusion.py`), hand-rotation error
dropped from 15° to 3° (worst case 46° → 6°), jitter while holding still from 1.6 cm to
0.7 cm (0.3 cm in fine mode), and lag during fast moves from 4.0 cm to 1.8 cm.

**What helps most is keeping your hands in view:** sit 1-1.5 m from the camera, tilt the
screen down a bit, and do the task in the air in front of your chest. You don't need to touch
your own desk; the robot copies the *direction* of your arms.

### Hand modes (`HAND_MODE` in `config.py`, or press `H`)

| Mode | What your hand does | Good for |
|---|---|---|
| `open_close` (default) | Make a fist → robot hand closes. Open your hand → it opens. | Picking things up. Never flickers. |
| `grip` | The robot hand closes exactly as much as yours | Gentle, partial grasps |
| `fingers` | Every finger joint is copied separately | Seeing finger tracking, but jittery |

"Grip" is one number per hand: the fingertip-to-wrist distance divided by your palm length
(open hand ≈ 1.8-1.9, fist ≈ 0.8-1.0), rescaled to 0 (open) .. 1 (fist). `open_close` uses two
thresholds (close above 0.65, open below 0.4) so a hand right at the edge can't flicker. The
kitchen window shows your live grip values, and Step 1 prints them, so you can tune the
thresholds for your own hand.

## How it works

1. **Tracking** (`teleop/human_tracker.py`): MediaPipe Pose gives 33 body landmarks in meters,
   and MediaPipe Hands gives 21 landmarks per hand. Everything is converted to robot axes
   (x forward, y left, z up). MediaPipe's left/right hand labels assume a mirrored selfie
   image, so each hand is matched to the closest *body* wrist instead. This runs in a
   background thread so the robot never waits for the camera.
2. **Retargeting** (`teleop/retarget.py`): people and the G1 are different sizes, so we copy
   *directions*, not positions:
   * **torso**: your shoulders vs. hips → how the robot torso should lean/twist (the G1's waist joints)
   * **arms**: the *direction* of your upper arm and forearm × the *robot's* bone lengths,
     starting at the robot's shoulder → elbow and wrist targets
   * **palm**: wrist, index, middle, and pinky knuckles → a rotation matrix for the robot hand
   * **fingers**: by default one "grip" number per hand with open/close thresholds; or how bent
     each finger joint is (`calc_angle`, same as the bicep curl app) → Dex3 joint angles
   * **head**: nose vs. ears, measured relative to your torso → robot neck yaw/pitch
3. **IK** (`teleop/ik.py`): one damped-least-squares solver over 17 joints (3 waist + 2×7 arm)
   tracks the torso orientation, both elbows, both wrists, and both hand rotations at once.
   The weights in `config.py` decide what matters most.
4. **Simulation** (`teleop/kitchen_sim.py`): MuJoCo position motors follow the joint targets,
   with a speed limit so the robot never jerks. Two cameras are rendered off-screen and shown
   with OpenCV (so no `mjpython` is needed on macOS).

All the tunable numbers (smoothing, gains, IK weights, window sizes) are in **`config.py`**.

## The robot and the scene

* **Robot:** `assets/g1_dex3_fixed.xml`, derived from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)
  G1 with Dex3-1 hands (BSD-3). Changes:
  * the pelvis is bolted to the world and the legs are rigid
  * a **virtual 2-DOF neck** (yaw + pitch) carries the head and the `head_cam` camera. The
    real G1 has no neck; this works like the active neck gimbal in
    [Open-TeleVision](https://robot-tv.github.io/).
  * IK sites on the shoulders, elbows, and wrists
  * thumb torque doubled, so one thumb can push back against two fingers when grasping
* **Leaning:** with the pelvis fixed, the G1's **waist** joints (yaw ±150°, roll/pitch ±30°)
  do the forward/back/side leaning and twisting.
* **MediaPipe models:** `assets/models/*.task` are Google's
  [MediaPipe](https://ai.google.dev/edge/mediapipe) pose and hand landmarker models
  (Apache-2.0). They're only used on Python 3.13+.
* **Kitchen:** `assets/kitchen_scene.xml`. The countertop is 0.70 m high (the G1 is 1.32 m
  tall, so that feels like a normal counter to it). The objects are within the G1's short
  reach. There's a tray to drop things into.

## Files

```
main.py                    full teleop
check_my_code.py           checks the class TODOs
step1_tracking_only.py     lesson 1: MediaPipe only
step2_ik_playground.py     lesson 2: keyboard-controlled IK, no webcam
config.py                  every number you might want to change
teleop/
  webcam.py                always-newest-frame webcam reader
  human_tracker.py         MediaPipe Pose + Hands → HumanPose (+ background thread)
  mediapipe_tasks.py       automatic fallback for MediaPipe 1.x (Python 3.13+), no mp.solutions there
  geometry.py              frames, calc_angle, palm rotation, torso rotation, head angles
  smoothing.py             smoothing filters (moving average, 1 Euro) and the glitch filter
  retarget.py              HumanPose → RobotTargets
  ik.py                    upper-body inverse kinematics
  kitchen_sim.py           MuJoCo model, motors, cameras, target drawing
  viewer.py                mouse orbit, text boxes, picture-in-picture
  fake_human.py            builds pretend HumanPoses (for tests / no-webcam demos)
tests/test_ik_motions.py   motion test suite
tests/test_hand_modes.py   finger-noise test for the three hand modes
tests/test_occlusion.py    hidden-hand glitch test for the filtering
assets/                    robot + kitchen MJCF, meshes, MediaPipe model files (for the fallback)
```

## Troubleshooting

* **"Could not open camera"**
  * Mac: System Settings > Privacy & Security > Camera > allow PyCharm (or Terminal), then
    restart PyCharm.
  * Windows: Settings > Privacy & security > Camera > turn on "Let desktop apps access your
    camera". Close Zoom/Teams.
  * Try `WEBCAM_INDEX = 1` in `config.py` if you have more than one camera.
* **`pip install` says "No matching distribution found for mediapipe"**: your Python is too
  new or too old for your computer. Use Python 3.12. On a Windows laptop with a Snapdragon
  (ARM) chip, install the regular **64-bit (x64)** Python from python.org, not the ARM64 one.
* **`cv2.imshow` errors**: only `opencv-contrib-python` should be installed. If
  `opencv-python` or `opencv-python-headless` sneaks in, `pip uninstall` it and reinstall
  `opencv-contrib-python`.
* **Slow**: lower `POSE_MODEL_COMPLEXITY` to 0 or shrink the window sizes in `config.py`.
  `main.py` limits numpy to one thread on purpose (it's ~10x faster here).
