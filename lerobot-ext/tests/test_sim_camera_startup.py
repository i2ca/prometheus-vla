"""Exercise the camera startup helper without importing robot/DDS hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[1] / "robot/unitree_g1/unitree_g1.py"
tree = ast.parse(SOURCE.read_text())
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "connect_camera")
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), namespace)
connect_camera = namespace["connect_camera"]


class CameraStartupTests(unittest.TestCase):
    def camera(self, kind="zmq"):
        return Mock(config=SimpleNamespace(type=kind, warmup_s=10))

    def test_simulation_waits_for_one_frame_with_full_timeout(self):
        cam = self.camera()
        connect_camera(cam, simulation=True)
        cam.connect.assert_called_once_with(warmup=False)
        cam.async_read.assert_called_once_with(timeout_ms=10000)
        cam.disconnect.assert_not_called()

    def test_timeout_is_propagated_and_camera_is_closed(self):
        cam = self.camera()
        cam.async_read.side_effect = TimeoutError("no frame")
        with self.assertRaises(TimeoutError):
            connect_camera(cam, simulation=True)
        cam.disconnect.assert_called_once_with()

    def test_hardware_and_other_camera_types_keep_original_warmup(self):
        for simulation, kind in ((False, "zmq"), (True, "opencv")):
            cam = self.camera(kind)
            connect_camera(cam, simulation=simulation)
            cam.connect.assert_called_once_with()
            cam.async_read.assert_not_called()
