import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MujocoRecordProfileTests(unittest.TestCase):
    def test_profile_parses_with_installed_recorder_and_three_local_cameras(self):
        import draccus
        import robot.unitree_g1
        import teleop.unitree_g1
        from lerobot.scripts.lerobot_record import RecordConfig

        config = draccus.parse(RecordConfig, args=[
            "--config_path", str(ROOT / "config/record/record_mujoco_wla.yaml")])
        self.assertTrue(config.robot.is_simulation)
        self.assertTrue(config.teleop.is_simulation)
        self.assertFalse(config.dataset.push_to_hub)
        self.assertEqual(config.dataset.fps, 10)
        self.assertEqual(config.dataset.num_episodes, 1)
        self.assertEqual(config.dataset.video_encoding_batch_size, 1)
        self.assertEqual(set(config.robot.cameras), {
            "head_camera", "left_wrist_camera", "right_wrist_camera"})
        for name, camera in config.robot.cameras.items():
            self.assertEqual(camera.camera_name, name)
            self.assertEqual(camera.server_address, "127.0.0.1")
            self.assertEqual((camera.width, camera.height), (640, 480))

    def test_sim_does_not_reinsert_disabled_depth_and_keeps_default_behavior(self):
        source = ROOT / "robot/unitree_g1/unitree_g1.py"
        tree = ast.parse(source.read_text())
        condition = next(node.test for node in ast.walk(tree)
                         if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
                         and "head_camera_depth" in ast.unparse(node.test))
        code = compile(ast.Expression(condition), str(source), "eval")
        for setting, enabled in (("0", False), ("false", False), ("False", False),
                                 ("", False), ("1", True), (None, True)):
            environment = {} if setting is None else {"CAMERA_EMIT_DEPTH": setting}
            scope = {"os": SimpleNamespace(environ=environment), "lista_de_cameras": ["head_camera"]}
            self.assertEqual(eval(code, scope), enabled)
            scope["lista_de_cameras"].append("head_camera_depth")
            self.assertFalse(eval(code, scope))

    def test_launcher_shell_syntax(self):
        subprocess.run(["bash", "-n", str(ROOT / "start_record_mujoco_wla.sh")], check=True)


if __name__ == "__main__":
    unittest.main()
