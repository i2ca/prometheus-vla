"""Controller finger targets for local simulation while teleoperation is enabled."""
import numpy as np


class SimulationHandControl:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.filtered = np.zeros(4)  # left/right trigger, left/right squeeze

    def update(self, data):
        values = np.array([
            (10.0 - getattr(data, "left_ctrl_triggerValue", 10.0)) / 10.0,
            (10.0 - getattr(data, "right_ctrl_triggerValue", 10.0)) / 10.0,
            getattr(data, "left_ctrl_squeezeValue", 0.0),
            getattr(data, "right_ctrl_squeezeValue", 0.0),
        ], dtype=float)
        values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
        self.filtered = self.alpha * values + (1 - self.alpha) * self.filtered
        trigger_l, trigger_r, squeeze_l, squeeze_r = np.where(self.filtered < 0.05, 0, self.filtered)
        left = squeeze_l * np.array([0., .920, 1.74, -1.57, -1.74, -1.57, -1.74])
        right = squeeze_r * np.array([0., -.920, -1.74, 1.57, 1.74, 1.57, 1.74])
        left += trigger_l * np.array([-.5, .8, .8, 0., 0., -2.2, -2.1])
        right += trigger_r * np.array([-.5, -.8, -.8, 0., 0., 2.2, 2.1])
        signals = {
            "left_grasp_trigger.q": float(trigger_l), "right_grasp_trigger.q": float(trigger_r),
            "left_grasp_squeeze.q": float(squeeze_l), "right_grasp_squeeze.q": float(squeeze_r),
        }
        return left, right, signals
