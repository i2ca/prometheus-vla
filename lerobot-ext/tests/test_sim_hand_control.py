import ast
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_hand_control import SimulationHandControl


def controller(squeeze=0.0, trigger=10.0):
    return SimpleNamespace(left_ctrl_squeezeValue=squeeze, right_ctrl_squeezeValue=squeeze,
                           left_ctrl_triggerValue=trigger, right_ctrl_triggerValue=trigger)


class SimHandTests(unittest.TestCase):
    def test_releasing_grip_and_trigger_returns_both_targets_to_open(self):
        control = SimulationHandControl()
        for _ in range(10):
            left, right, _ = control.update(controller(1.0, 0.0))
        self.assertGreater(np.linalg.norm(right), 1)
        for _ in range(7):
            left, right, signals = control.update(controller())
        np.testing.assert_array_equal(left, np.zeros(7))
        np.testing.assert_array_equal(right, np.zeros(7))
        self.assertTrue(all(value == 0 for value in signals.values()))

    def test_invalid_input_cannot_latch_a_closed_command(self):
        control = SimulationHandControl()
        for _ in range(10):
            control.update(controller(1.0, 0.0))
        for _ in range(7):
            left, right, _ = control.update(controller(float('nan'), float('nan')))
        self.assertTrue(np.isfinite(right).all())
        np.testing.assert_array_equal(right, np.zeros(7))

    def test_actual_entrypoint_pause_freezes_whole_action_in_both_modes(self):
        # Importing the entrypoint would start voice/CLI threads. Execute its
        # actual pause function alone, with a fake downstream robot sender.
        tree = ast.parse((ROOT / 'init_lerobot_teleoparate.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'patched_send_action')
        for simulation in (True, False):
            scope = {'robot_paused': True, 'frozen_action': {'arm.q': .4, 'finger.q': 1.4},
                     'original_send_action': lambda robot, action: action}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<pause-function>', 'exec'), scope)
            robot = SimpleNamespace(config=SimpleNamespace(is_simulation=simulation),
                                    left_hand_joint_names=[], right_hand_joint_names=['finger'])
            for finger in (0.0, 1.7):
                action = scope['patched_send_action'](robot, {'arm.q': .9, 'finger.q': finger})
                self.assertEqual(action, {'arm.q': .4, 'finger.q': 1.4})
            scope['robot_paused'] = False
            resumed = {'arm.q': .9, 'finger.q': 0.0}
            self.assertEqual(scope['patched_send_action'](robot, resumed), resumed)
            self.assertEqual(scope['frozen_action'], resumed)

    def test_actual_xr_pause_freezes_fingers_and_filter_without_calling_ik(self):
        from teleop.xr_g1_arm import XRG1Arm
        teleop = XRG1Arm.__new__(XRG1Arm)
        teleop.config = SimpleNamespace(is_simulation=True, input_mode='controller', ee_type='dex3')
        teleop._is_connected = True
        teleop._has_arm_feedback = True
        teleop.controller_enabled = False
        teleop.last_x_state = teleop.last_y_state = False
        teleop._last_diag_log = time.time()
        teleop._EMA_ALPHA = .5
        teleop._left_hand_names = [f'left_{i}' for i in range(7)]
        teleop._right_hand_names = [f'right_{i}' for i in range(7)]
        teleop.body_joints = {'arm.q': .4}
        teleop.hand_joints = {f'{name}.q': 0.0 for name in teleop._left_hand_names + teleop._right_hand_names}
        data = controller(1.0, 0.0)
        teleop.tv_wrapper = SimpleNamespace(get_tele_data=lambda: data)
        with patch.dict(os.environ, {'G1_LEFT_ARM_LIMP': '0'}):
            for _ in range(10):
                frozen_open = teleop.get_action()
            self.assertEqual(frozen_open['right_6.q'], 0.0)
            self.assertFalse(hasattr(teleop, '_sim_hand_control'))
            # Model a hand already closed before pausing.
            for _ in range(10):
                teleop._update_sim_controller_hands(data)
            frozen = {**teleop.body_joints, **teleop.hand_joints}
            filtered = teleop._sim_hand_control.filtered.copy()
            self.assertGreater(abs(frozen['right_6.q']), 1)
            data = controller()
            for _ in range(7):
                self.assertEqual(teleop.get_action(), frozen)
            np.testing.assert_array_equal(teleop._sim_hand_control.filtered, filtered)
            # Once enabled again, the normal hand update can reopen it.
            teleop.controller_enabled = True
            for _ in range(7):
                teleop._update_sim_controller_hands(data)
        self.assertEqual(teleop.body_joints['arm.q'], .4)
        self.assertEqual(teleop.hand_joints['right_6.q'], 0.0)
        self.assertFalse(hasattr(teleop, 'kp_hacked'))


if __name__ == '__main__':
    unittest.main()
