"""
Smoothing filters. MediaPipe's landmarks wiggle a little every frame; if we sent those
straight to the robot it would shake. Averaging new measurements with old ones fixes that.
"""
import numpy as np

from teleop.geometry import closest_rotation


class Smoother:
    """
    Exponential moving average:
        smoothed = amount * previous_smoothed + (1 - amount) * new_measurement

    amount = 0   -> no smoothing at all
    amount = 0.9 -> very smooth, but the robot reacts slowly
    Works for single numbers and numpy arrays.
    """

    def __init__(self, amount):
        self.amount = amount
        self.value = None

    def __call__(self, measurement, dt=None, trust=1.0):
        measurement = np.asarray(measurement, dtype=float)
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.value + trust * (1.0 - self.amount) * (measurement - self.value)
        return self.value

    def reset(self):
        self.value = None


class RotationSmoother(Smoother):
    """Same as Smoother, but for 3x3 rotation matrices (averaging bends them, so we fix them up)."""

    def __call__(self, rotation, dt=None, trust=1.0):
        self.value = closest_rotation(super().__call__(rotation, dt, trust))
        return self.value


class OneEuroFilter:
    """
    Smoothing that adapts to how fast you move (the "1 Euro filter", Casiez et al. 2012,
    used a lot for VR hand tracking):
        moving SLOWLY (lining up a grasp) -> smooth a lot, so the jitter disappears
        moving FAST (swinging your arm)   -> smooth a little, so the robot doesn't lag behind

    min_cutoff: smoothing when you hold still (lower = smoother)
    beta:       how quickly smoothing switches off as you speed up (higher = less lag)
    trust:      0..1 per call, how much we believe this measurement (0 = ignore it)
    """

    def __init__(self, min_cutoff, beta, speed_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.speed_cutoff = speed_cutoff
        self.value = None
        self.speed = None

    @staticmethod
    def _blend(cutoff, dt):
        """How far (0..1) to move toward the new measurement, for a filter with this cutoff (Hz)."""
        time_constant = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + time_constant / dt)

    def __call__(self, measurement, dt, trust=1.0):
        measurement = np.asarray(measurement, dtype=float)
        if self.value is None:
            self.value = measurement
            self.speed = np.zeros_like(measurement)
            return self.value
        # how fast is it changing? (smoothed, so jitter doesn't look like speed)
        raw_speed = (measurement - self.value) / dt
        self.speed = self.speed + self._blend(self.speed_cutoff, dt) * (raw_speed - self.speed)
        # faster movement -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.speed)
        self.value = self.value + trust * self._blend(cutoff, dt) * (measurement - self.value)
        return self.value

    def reset(self):
        self.value = None
        self.speed = None


class OneEuroRotationFilter(OneEuroFilter):
    """OneEuroFilter for 3x3 rotation matrices."""

    def __call__(self, rotation, dt, trust=1.0):
        smoothed = super().__call__(np.asarray(rotation).flatten(), dt, trust)
        self.value = closest_rotation(smoothed.reshape(3, 3)).flatten()
        return self.value.reshape(3, 3)


class JumpFilter:
    """
    Ignores sudden jumps that a real arm can't make. Those are usually MediaPipe guessing where a
    hidden hand is. The limit is a SPEED (degrees per second), so it works the same on fast and
    slow computers. If the "jump" is still there a few frames later, it was real, so we accept it.
    """

    def __init__(self, max_speed, patience):
        self.max_speed = max_speed      # None = accept everything
        self.patience = patience
        self.rejected_in_a_row = 0

    def accept(self, jump, dt):
        """jump: degrees moved since the last accepted measurement; dt: seconds since the last frame."""
        if self.max_speed is None or jump <= self.max_speed * dt or self.rejected_in_a_row >= self.patience:
            self.rejected_in_a_row = 0
            return True
        self.rejected_in_a_row += 1
        return False

    def reset(self):
        self.rejected_in_a_row = 0
