import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot.unitree_g1.g1_utils import G1_29_JointArmIndex
from teleop.xr_g1_arm import XRG1Arm


def make_teleop():
    teleop = XRG1Arm.__new__(XRG1Arm)
    teleop._has_arm_feedback = False
    teleop.controller_enabled = False
    teleop.clutch_anchored = False
    teleop.current_arm_q = np.zeros(14)
    teleop.current_arm_dq = np.zeros(14)
    teleop.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointArmIndex}
    teleop.hand_joints = {"finger.q": 0.7}
    return teleop


def observation(q=0.4):
    return {f"{motor.name}.{field}": value
            for motor in G1_29_JointArmIndex
            for field, value in (("q", q), ("dq", 0.1))}


def record_wrapper(original):
    # Importar o entrypoint iniciaria threads e patches; testa a função real isolada.
    tree = ast.parse((ROOT / "init_lerobot_record.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_patched_record_loop")
    scope = {"_orig_record_loop": original}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<record-wrapper>", "exec"), scope)
    return scope["_patched_record_loop"]


def run_record(original, processor, **kwargs):
    events = kwargs.pop("events", {"exit_early": False})
    return record_wrapper(original)(object(), events, 30, lambda pair: pair[0],
                                    lambda pair: pair[0], processor, **kwargs)


class RecordFeedbackTests(unittest.TestCase):
    def test_feedback_precedes_action_and_allows_x_with_positional_arguments(self):
        teleop = make_teleop()
        raw = observation()
        processor = Mock(side_effect=lambda obs: {"processed": obs})
        events = {"exit_early": True}
        main = SimpleNamespace(robot_paused=False, global_events=events)

        def loop(robot, events, fps, teleop_action_processor, robot_action_processor,
                 robot_observation_processor, dataset=None, teleop=None):
            processed = robot_observation_processor(raw)
            self.assertTrue(teleop._has_arm_feedback)
            teleop._trigger_record_event("toggle_pause")
            self.assertTrue(teleop.controller_enabled)
            self.assertFalse(events["exit_early"])
            return processed

        with patch.dict(sys.modules, {"__main__": main}):
            result = record_wrapper(loop)(object(), events, 30, None, None,
                                          processor, object(), teleop)
        self.assertEqual(result, {"processed": raw})
        processor.assert_called_once_with(raw)

    def test_missing_or_nonfinite_state_keeps_x_locked(self):
        for raw in ({}, observation(float("nan")), observation(float("inf"))):
            with self.subTest(raw=raw):
                teleop = make_teleop()

                def loop(robot, events, fps, action_processor, robot_action_processor,
                         robot_observation_processor, teleop=None):
                    robot_observation_processor(raw)
                    teleop._trigger_record_event("toggle_pause")
                    self.assertFalse(teleop.controller_enabled)
                    self.assertFalse(teleop._has_arm_feedback)

                with patch.dict(sys.modules, {"__main__": SimpleNamespace(robot_paused=True)}):
                    run_record(loop, lambda obs: obs, teleop=teleop)

    def test_nonfinite_velocity_is_not_feedback(self):
        raw = observation()
        raw[f"{next(iter(G1_29_JointArmIndex)).name}.dq"] = float("nan")
        self.assertIsNone(make_teleop().build_feedback(raw))

    def test_pause_preserves_arm_and_finger_targets_despite_changing_feedback(self):
        teleop = make_teleop()
        teleop.send_feedback(teleop.build_feedback(observation()))
        main = SimpleNamespace(robot_paused=False)
        with patch.dict(sys.modules, {"__main__": main}):
            teleop._trigger_record_event("toggle_pause")
            self.assertTrue(teleop.controller_enabled)
            teleop.body_joints = dict.fromkeys(teleop.body_joints, 0.8)
            teleop._trigger_record_event("toggle_pause")
            self.assertTrue(main.robot_paused)
            frozen = {**teleop.body_joints, **teleop.hand_joints}
            for q in (0.7, 0.5, 0.3):
                teleop.send_feedback(teleop.build_feedback(observation(q)))
                self.assertEqual({**teleop.body_joints, **teleop.hand_joints}, frozen)
                np.testing.assert_allclose(teleop.current_arm_q, q)
            teleop._trigger_record_event("toggle_pause")
            self.assertTrue(teleop.controller_enabled)
            self.assertFalse(main.robot_paused)

    def test_reset_without_dataset_still_delivers_feedback(self):
        teleop = make_teleop()
        events = {"exit_early": True}

        def loop(robot, events, fps, action_processor, robot_action_processor,
                 robot_observation_processor, dataset=None, teleop=None):
            robot_observation_processor(observation())
            self.assertTrue(teleop._has_arm_feedback)
            self.assertTrue(events["exit_early"])

        run_record(loop, lambda obs: obs, events=events, teleop=teleop)

    def test_other_teleoperators_and_policy_only_keep_original_processor(self):
        processor = lambda obs: obs
        for teleop in (None, object(), [object()]):
            def loop(robot, events, fps, action_processor, robot_action_processor,
                     robot_observation_processor, teleop=None):
                self.assertIs(robot_observation_processor, processor)
                return "unchanged"

            self.assertEqual(run_record(loop, processor, teleop=teleop), "unchanged")

    def test_processor_error_propagates_and_processor_is_not_modified(self):
        processor = Mock(side_effect=RuntimeError("processor failed"))
        teleop = make_teleop()

        def loop(robot, events, fps, action_processor, robot_action_processor,
                 robot_observation_processor, teleop=None):
            return robot_observation_processor(observation())

        with self.assertRaisesRegex(RuntimeError, "processor failed"):
            run_record(loop, processor, teleop=teleop)
        processor.assert_called_once()

    def test_installed_lerobot_loop_receives_feedback_before_get_action(self):
        from lerobot.scripts.lerobot_record import record_loop

        teleop = make_teleop()
        events = {"exit_early": False}
        robot = SimpleNamespace(get_observation=lambda: observation(), send_action=Mock())

        def get_action():
            self.assertTrue(teleop._has_arm_feedback)
            teleop._trigger_record_event("toggle_pause")
            self.assertTrue(teleop.controller_enabled)
            events["exit_early"] = True
            return {**teleop.body_joints, **teleop.hand_joints}

        teleop.get_action = get_action
        with patch.dict(sys.modules, {"__main__": SimpleNamespace(robot_paused=False)}):
            record_wrapper(record_loop)(
                robot, events, 30, lambda pair: pair[0], lambda pair: pair[0],
                lambda obs: obs, teleop=teleop, control_time_s=1,
            )
        robot.send_action.assert_called_once()

    def test_record_loop_writes_actual_capture_times_without_changing_observations(self):
        teleop = make_teleop()
        raw = observation()

        def loop(robot, events, fps, action_processor, robot_action_processor,
                 robot_observation_processor, dataset=None, teleop=None):
            for _ in range(3):
                self.assertIs(robot_observation_processor(raw), raw)

        with tempfile.TemporaryDirectory() as root:
            dataset = SimpleNamespace(root=root, episode_buffer={"episode_index": 2})
            with patch("recording_timing.time.perf_counter", side_effect=[100, 100.1, 100.2]):
                record_wrapper(loop)(object(), {}, 10, None, None, lambda obs: obs,
                                     dataset=dataset, teleop=teleop)
            report = json.loads((Path(root) / "timing/episode_000002.json").read_text())
            self.assertEqual(report["frames"], 3)
            self.assertTrue(report["timing_ok"])
            self.assertTrue(teleop._has_arm_feedback)

    def test_interrupted_record_loop_is_not_reported_as_valid_timing(self):
        def loop(robot, events, fps, action_processor, robot_action_processor,
                 robot_observation_processor, dataset=None):
            robot_observation_processor(observation())
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as root:
            dataset = SimpleNamespace(root=root, episode_buffer={"episode_index": 0})
            with self.assertRaises(KeyboardInterrupt):
                run_record(loop, lambda obs: obs, dataset=dataset)
            report = json.loads((Path(root) / "timing/episode_000000.json").read_text())
            self.assertFalse(report["record_loop_completed"])
            self.assertFalse(report["timing_ok"])


if __name__ == "__main__":
    unittest.main()
