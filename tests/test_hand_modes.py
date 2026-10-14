"""
How jittery is each hand mode? (no webcam needed)

We build pretend hands, shake every landmark a little every frame (like MediaPipe does), run
them through the Retargeter, and measure how much the robot's finger commands wobble.

Run it from PyCharm, or:   python tests/test_hand_modes.py
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

import config
from teleop.fake_human import make_human
from teleop.kitchen_sim import KitchenSim
from teleop.retarget import Retargeter

# MediaPipe-like noise on every hand landmark, in meters: depth (x) is the worst
NOISE = np.array([0.008, 0.003, 0.003])
FRAMES = 150


def run(mode, closures, robot, seed=0):
    """Feed a sequence of hand closures (one per frame); return the commands and open/close switches."""
    config.HAND_MODE = mode
    rng = np.random.default_rng(seed)
    retargeter = Retargeter(robot)
    commands, switches, was_closed = [], 0, False
    for closure in closures:
        human = make_human(closure=(closure, closure))
        human.left_hand = human.left_hand + rng.normal(0, NOISE, human.left_hand.shape)
        human.right_hand = human.right_hand + rng.normal(0, NOISE, human.right_hand.shape)
        targets = retargeter.update(human)
        joints = sorted(k for k in targets.hand_joints if k.startswith("left_"))
        commands.append([targets.hand_joints[k] for k in joints])
        if retargeter.hand_closed["left"] != was_closed:
            switches += 1
            was_closed = retargeter.hand_closed["left"]
    return np.array(commands), switches


def main():
    robot = KitchenSim().robot_body_info()
    modes = ["fingers", "grip", "open_close"]
    holds = {"open hand": 0.0, "half closed": 0.5, "fist": 1.0}

    print(f"Finger command wobble while holding still (average std of the 7 joints, degrees)")
    print(f"landmark noise: {NOISE * 1000} mm (depth, sideways, up)\n")
    print(f"{'hand pose':14s}" + "".join(f"{m:>14s}" for m in modes))
    worst = {}
    for name, closure in holds.items():
        row = f"{name:14s}"
        for mode in modes:
            commands, _ = run(mode, [closure] * FRAMES, robot)
            wobble = np.degrees(commands[20:].std(axis=0).mean())   # skip the first frames (filter warm-up)
            worst[mode] = max(worst.get(mode, 0.0), wobble)
            row += f"{wobble:14.1f}"
        print(row)

    # slowly close and re-open the hand: open_close should switch exactly twice
    sweep = list(np.linspace(0, 1, FRAMES)) + list(np.linspace(1, 0, FRAMES))
    _, switches = run("open_close", sweep, robot)
    print(f"\nopen_close: slow close -> open sweep switched {switches} times (want 2)")
    flicker = 0
    for name, closure in holds.items():
        counts = [run("open_close", [closure] * FRAMES, robot, seed=s)[1] for s in range(5)]
        expected = 1 if closure == 1.0 else 0     # holding a fist closes the hand once, then nothing
        flicker += sum(c - expected for c in counts)
        print(f"open_close: holding '{name}' for {FRAMES} frames, 5 tries -> switches {counts} (want {expected} each)")

    ok = switches == 2 and flicker == 0 and worst["open_close"] < 0.5 and worst["grip"] < worst["fingers"]
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
