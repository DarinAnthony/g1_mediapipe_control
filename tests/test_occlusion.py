"""
What happens when your hand is half hidden (e.g. reaching down toward the desk)?  (no webcam)

We fake an 8 second session at 20 fps:
    0 - 3 s    slowly reach down toward the counter
    3 - 5 s    hold still (lining up a grasp)
    5 - 5.5 s  swing the arm up FAST
    5.5 - 8 s  hold still
and add the kind of trouble MediaPipe has with hidden hands:
    - normal jitter on every landmark (worst in depth)
    - 15% of frames while the hand is low: the wrist "teleports" 10-20 cm with medium visibility,
      and the hand rotation flips by 50-150 degrees
    - 5% of frames: the hand disappears
    - a 0.4 s stretch where the arm is half hidden (visibility 0.6, 3x the jitter)

Then we compare the robot's wrist targets with where they SHOULD be, for:
    before      the old simple smoothing, no glitch rejection
    after       1-Euro smoothing + visibility trust + glitch rejection (the defaults)
    after+fine  the same, with fine mode switched on at 3 s
Run it from PyCharm, or:   python tests/test_occlusion.py
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

import config
from teleop.fake_human import DOWN, FORWARD, UP, hand_frame, make_human
from teleop.geometry import normalize, rotation_angle
from teleop.kitchen_sim import KitchenSim
from teleop.retarget import Retargeter

FPS = 20
SECONDS = 8.0
JITTER = np.array([0.015, 0.005, 0.005])   # meters (depth, sideways, up)


def slerp_dir(a, b, t):
    return normalize((1 - t) * a + t * b)


def arm_at(time):
    """The true right-arm pose at this time: (upper arm direction, forearm direction)."""
    ready = (DOWN, FORWARD)
    reach = (normalize(np.array([0.4, 0, -1.0])), normalize(np.array([1.0, 0, -0.7])))
    up = (normalize(np.array([0.3, 0, -1.0])), normalize(np.array([0.5, 0, 1.0])))
    if time < 3.0:
        t = time / 3.0
        return slerp_dir(ready[0], reach[0], t), slerp_dir(ready[1], reach[1], t)
    if time < 5.0:
        return reach
    if time < 5.5:
        t = (time - 5.0) / 0.5
        return slerp_dir(reach[0], up[0], t), slerp_dir(reach[1], up[1], t)
    return up


def session(seed=0):
    """Yields (time, noisy HumanPose, true upper arm dir, true forearm dir, true palm rotation)."""
    rng = np.random.default_rng(seed)
    for frame in range(int(SECONDS * FPS)):
        time = frame / FPS
        upper, fore = arm_at(time)
        palm = hand_frame(fore, DOWN, "right")
        noisy = make_human(right_arm=(upper, fore), right_hand=palm, timestamp=time)

        jitter = JITTER.copy()
        hand_low = 1.5 <= time < 5.0
        if 2.2 <= time < 2.6:                       # half-hidden stretch
            jitter *= 3
            noisy.body_visibility[[14, 16]] = 0.6
        noisy.body = noisy.body + rng.normal(0, jitter, noisy.body.shape)

        if hand_low and rng.random() < 0.15:        # MediaPipe guessing a hidden hand
            noisy.body[16] += normalize(rng.normal(size=3)) * rng.uniform(0.10, 0.20)
            noisy.body_visibility[16] = rng.uniform(0.5, 0.8)
            axis = normalize(rng.normal(size=3))
            angle = np.radians(rng.uniform(50, 150))
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            flip = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            noisy.right_hand = noisy.right_hand @ flip.T
        if hand_low and rng.random() < 0.05:        # hand not found at all
            noisy.right_hand = None
        if noisy.right_hand is not None:
            noisy.right_hand = noisy.right_hand + rng.normal(0, [0.006, 0.002, 0.002], noisy.right_hand.shape)
        yield time, noisy, upper, fore, palm


def run(name, robot, fine=False, seeds=range(5)):
    pos_err, rot_err, hold_jitter, fast_err, fast_rot = [], [], [], [], []
    for seed in seeds:
        tracker, truth = Retargeter(robot), Retargeter(robot)
        hold = []
        for time, noisy, upper, fore, palm in session(seed):
            if fine and abs(time - 3.0) < 1e-9:
                tracker.toggle_fine_mode()
            target = tracker.update(noisy)
            # where the wrist SHOULD go: the true arm directions with the robot's bone lengths
            truth.upper_arm_dir["right"], truth.forearm_dir["right"] = upper, fore
            true_wrist = truth.targets().wrist_pos["right"]
            error = 100 * np.linalg.norm(target.wrist_pos["right"] - true_wrist)
            turn = np.degrees(rotation_angle(target.wrist_rot["right"].T @ palm))
            if time < 5.0:
                pos_err.append(error)
                rot_err.append(turn)
            if 3.5 <= time < 5.0:
                hold.append(target.wrist_pos["right"].copy())
            if 5.0 <= time < 5.6:
                fast_err.append(error)
                fast_rot.append(turn)
        hold_jitter.append(100 * np.linalg.norm(np.std(hold, axis=0)))
    return {
        "name": name,
        "pos_rms": np.sqrt(np.mean(np.square(pos_err))),
        "pos_worst": np.percentile(pos_err, 99),
        "rot_rms": np.sqrt(np.mean(np.square(rot_err))),
        "rot_worst": np.percentile(rot_err, 99),
        "hold_jitter": np.mean(hold_jitter),
        "fast_err": np.sqrt(np.mean(np.square(fast_err))),
        "fast_rot": np.sqrt(np.mean(np.square(fast_rot))),
    }


def main():
    robot = KitchenSim().robot_body_info()
    defaults = {k: getattr(config, k) for k in ("ADAPTIVE_SMOOTHING", "REJECT_GLITCHES", "FULL_TRUST_VISIBILITY")}

    # "before": simple smoothing, every visible landmark fully trusted, no glitch rejection
    config.ADAPTIVE_SMOOTHING, config.REJECT_GLITCHES = False, False
    config.FULL_TRUST_VISIBILITY = config.MIN_LANDMARK_VISIBILITY + 1e-6
    results = [run("before", robot)]
    for key, value in defaults.items():
        setattr(config, key, value)
    results.append(run("after", robot))
    results.append(run("after+fine", robot, fine=True))

    print("Right wrist target vs. where it should be (5 runs). Robot arm is ~37 cm long.\n")
    print(f"{'':12s}{'reach + hold (0-5 s)':>40s}{'still (3.5-5 s)':>18s}{'fast swing':>12s}")
    print(f"{'':12s}{'pos err':>10s}{'worst 1%':>10s}{'rot err':>10s}{'worst 1%':>10s}{'jitter':>18s}{'lag err':>12s}")
    for r in results:
        print(f"{r['name']:12s}{r['pos_rms']:8.1f}cm{r['pos_worst']:8.1f}cm{r['rot_rms']:7.0f}deg{r['rot_worst']:7.0f}deg"
              f"{r['hold_jitter']:16.2f}cm{r['fast_err']:10.1f}cm")
    print("\n(pos/rot err for after+fine are bigger on purpose: fine mode moves the hand less than you do)")

    before, after = results[0], results[1]
    ok = (after["pos_worst"] < before["pos_worst"] and after["rot_worst"] < before["rot_worst"]
          and after["hold_jitter"] < before["hold_jitter"] and results[2]["hold_jitter"] < after["hold_jitter"])
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
