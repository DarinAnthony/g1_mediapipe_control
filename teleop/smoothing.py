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

    def __call__(self, measurement):
        measurement = np.asarray(measurement, dtype=float)
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.amount * self.value + (1.0 - self.amount) * measurement
        return self.value

    def reset(self):
        self.value = None


class RotationSmoother(Smoother):
    """Same as Smoother, but for 3x3 rotation matrices (averaging bends them, so we fix them up)."""

    def __call__(self, rotation):
        self.value = closest_rotation(super().__call__(rotation))
        return self.value
