"""Regressions for stale VR sessions and camera shutdown (no robot/Quest needed)."""
import ast
import io
import logging
import socket
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from session_ports import require_free_ports
from teleop.xr_g1_arm import XRG1Arm

tree = ast.parse((ROOT / "robot/unitree_g1/unitree_g1.py").read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "disconnect_camera")
scope = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), "<disconnect-camera>", "exec"), scope)
disconnect_camera = scope["disconnect_camera"]


def teleop():
    obj = XRG1Arm.__new__(XRG1Arm)
    obj._is_connected = False
    obj._video_stop = threading.Event()
    obj.tv_wrapper = None
    obj.config = SimpleNamespace(is_simulation=True, input_mode="controller",
                                 display_mode="pass-through", zmq=False, webrtc=False,
                                 img_server_ip="127.0.0.1", ee_type="none")
    return obj


def wrapper():
    obj = Mock()
    obj.tvuer.process.is_alive.return_value = False
    return obj


class SessionPortsTests(unittest.TestCase):
    def test_occupied_port_stops_startup_without_touching_owner(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, f"Porta {port} indisponível"):
                require_free_ports((port,))
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass

    def test_checks_release_their_own_sockets_even_on_failure(self):
        with socket.socket() as busy, socket.socket() as available:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            available.bind(("127.0.0.1", 0))
            free_port = available.getsockname()[1]
            available.close()
            require_free_ports((free_port,))
            with self.assertRaises(RuntimeError):
                require_free_ports((free_port, busy.getsockname()[1]))
            with socket.socket() as reused:
                reused.bind(("127.0.0.1", free_port))


class VRLifecycleTests(unittest.TestCase):
    def test_dead_server_cannot_be_reported_ready(self):
        obj = teleop()
        obj.tv_wrapper = wrapper()
        with self.assertRaisesRegex(RuntimeError, "Vuer encerrou"):
            obj._wait_for_vuer()

    def test_live_process_without_listening_socket_times_out(self):
        obj = teleop()
        obj.tv_wrapper = wrapper()
        obj.tv_wrapper.tvuer.process.is_alive.return_value = True
        with patch("teleop.xr_g1_arm.socket.create_connection", side_effect=ConnectionRefusedError):
            with self.assertRaisesRegex(TimeoutError, "não abriu"):
                obj._wait_for_vuer(timeout=0.01)

    def test_ready_server_requires_a_live_process_and_listening_socket(self):
        obj = teleop()
        obj.tv_wrapper = wrapper()
        obj.tv_wrapper.tvuer.process.is_alive.return_value = True
        with patch("teleop.xr_g1_arm.socket.create_connection") as connect:
            obj._wait_for_vuer()
        connect.assert_called_once_with(("127.0.0.1", 8012), timeout=0.2)

    def test_startup_failure_closes_partial_wrapper(self):
        for failure in (RuntimeError("Vuer failed"), KeyboardInterrupt()):
            obj = teleop()
            vr = wrapper()
            with patch("teleop.xr_g1_arm.require_free_ports"), \
                 patch("teleop.xr_g1_arm.TeleVuerWrapper", return_value=vr), \
                 patch.object(obj, "_wait_for_vuer", side_effect=failure):
                with self.assertRaises(type(failure)):
                    obj.connect()
            vr.close.assert_called_once_with()
            self.assertFalse(obj.is_connected)
            self.assertIsNone(obj.tv_wrapper)

    def test_disconnect_is_idempotent_and_reaps_stubborn_owned_child(self):
        obj = teleop()
        vr = obj.tv_wrapper = wrapper()
        vr.tvuer.process.is_alive.return_value = True
        obj.disconnect()
        obj.disconnect()
        vr.close.assert_called_once_with()
        vr.tvuer.process.kill.assert_called_once_with()
        vr.tvuer.process.join.assert_called_once_with(timeout=2)
        self.assertTrue(obj._video_stop.is_set())

    def test_close_failure_still_reaps_owned_child(self):
        obj = teleop()
        vr = obj.tv_wrapper = wrapper()
        vr.tvuer.process.is_alive.return_value = True
        vr.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            obj.disconnect()
        vr.tvuer.process.kill.assert_called_once_with()
        self.assertIsNone(obj.tv_wrapper)

    def test_terminal_shows_connection_instructions(self):
        tree = ast.parse((ROOT / "init_lerobot_record.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_TermRouter")
        scope = {"_ACTION_REWRITE": [], "_UI": "\x00UI\x00"}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "<router>", "exec"), scope)
        terminal, log = io.StringIO(), io.StringIO()
        scope["_TermRouter"](log, terminal).write("[CONTROLE VR] Servidor VR pronto. Atualize o Quest.\n")
        self.assertIn("Atualize o Quest", terminal.getvalue())


class CameraShutdownTests(unittest.TestCase):
    def test_waits_full_receive_timeout_before_driver_teardown(self):
        camera = Mock(config=SimpleNamespace(type="zmq"), timeout_ms=5000)
        camera.thread.is_alive.side_effect = [True, False]
        disconnect_camera(camera)
        camera.stop_event.set.assert_called_once()
        camera.thread.join.assert_called_once_with(timeout=6)
        self.assertEqual(camera.socket.linger, 0)
        camera.disconnect.assert_called_once()

    def test_does_not_close_socket_used_by_a_stuck_reader(self):
        camera = Mock(config=SimpleNamespace(type="zmq"), timeout_ms=5000)
        camera.thread.is_alive.return_value = True
        with self.assertRaisesRegex(RuntimeError, "não encerrou"):
            disconnect_camera(camera)
        camera.disconnect.assert_not_called()

    def test_other_cameras_keep_original_disconnect(self):
        camera = Mock(config=SimpleNamespace(type="opencv"))
        disconnect_camera(camera)
        camera.disconnect.assert_called_once()
        camera.thread.join.assert_not_called()

    def test_installed_zmq_driver_stops_without_publisher_or_thread_race(self):
        from lerobot.cameras.zmq.camera_zmq import ZMQCamera
        from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig

        with socket.socket() as unused_port:
            unused_port.bind(("127.0.0.1", 0))
            camera = ZMQCamera(ZMQCameraConfig(
                server_address="127.0.0.1", port=unused_port.getsockname()[1],
                width=32, height=24, fps=10, timeout_ms=2200,
            ))
            with patch("threading.excepthook") as thread_error:
                camera.connect(warmup=False)
                reader = camera.thread
                try:
                    disconnect_camera(camera)
                finally:
                    if camera.is_connected:
                        disconnect_camera(camera)
                self.assertFalse(reader.is_alive())
                self.assertIsNone(camera.stop_event)
                self.assertFalse(camera.is_connected)
                thread_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
