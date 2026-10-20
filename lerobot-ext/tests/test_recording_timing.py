import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recording_timing import CaptureTiming


class RecordingTimingTests(unittest.TestCase):
    def capture(self, fps, timestamps):
        timing = CaptureTiming(fps)
        with patch("recording_timing.time.perf_counter", side_effect=timestamps):
            for _ in timestamps:
                timing.observe()
        return timing

    def test_real_ten_fps_passes(self):
        report = self.capture(10, [100 + i / 10 for i in range(101)]).report()
        self.assertTrue(report["timing_ok"])
        self.assertAlmostEqual(report["observed_fps"], 10)
        self.assertEqual(report["capture_timestamps_seconds"][0], 0)

    def test_old_nineteen_hz_declared_as_thirty_is_rejected(self):
        report = self.capture(30, [100 + i / 19 for i in range(101)]).report()
        self.assertFalse(report["timing_ok"])
        self.assertAlmostEqual(report["observed_fps"], 19)

    def test_too_fast_too_slow_and_interrupted_do_not_pass(self):
        for actual in (5, 20):
            report = self.capture(10, [100 + i / actual for i in range(101)]).report()
            self.assertFalse(report["timing_ok"])
        self.assertFalse(self.capture(10, [100, 100.1]).report(completed=False)["timing_ok"])

    def test_empty_single_frame_and_nonmonotonic_do_not_pass(self):
        for timestamps in ([], [100], [100, 100], [100, 99]):
            self.assertFalse(self.capture(10, timestamps).report()["timing_ok"])

    def test_report_is_saved_separately_without_changing_existing_data(self):
        with tempfile.TemporaryDirectory() as root:
            existing = Path(root) / "info.json"
            existing.write_text("unchanged")
            report = self.capture(10, [100, 100.1, 100.2]).save(root, 3)
            path = Path(root) / "timing/episode_000003.json"
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(existing.read_text(), "unchanged")

    def test_invalid_fps_is_rejected(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CaptureTiming(fps)


if __name__ == "__main__":
    unittest.main()
