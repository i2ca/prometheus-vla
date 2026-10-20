"""Compare the LeRobot fixed-rate timeline against actual observation times."""
import json
import math
import time
from pathlib import Path


class CaptureTiming:
    def __init__(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.fps = fps
        self.timestamps = []

    def observe(self):
        self.timestamps.append(time.perf_counter())

    def report(self, completed=True):
        elapsed = [t - self.timestamps[0] for t in self.timestamps]
        wall_duration = elapsed[-1] if elapsed else 0.0
        expected = max(0, len(elapsed) - 1) / self.fps
        error = abs(wall_duration - expected) / expected if expected else None
        intervals = [b - a for a, b in zip(elapsed, elapsed[1:])]
        monotonic = bool(intervals) and all(dt > 0 for dt in intervals)
        return {
            "frames": len(elapsed),
            "configured_fps": self.fps,
            "observed_fps": (len(elapsed) - 1) / wall_duration if monotonic else None,
            "wall_duration_seconds": wall_duration,
            "dataset_duration_seconds": expected,
            "relative_duration_error": error,
            "max_interval_seconds": max(intervals) if intervals else None,
            "record_loop_completed": completed,
            "timing_ok": bool(completed and monotonic and error <= 0.05),
            "capture_timestamps_seconds": elapsed,
        }

    def save(self, root, episode_index, completed=True):
        report = self.report(completed)
        path = Path(root) / "timing" / f"episode_{episode_index:06d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        status = "OK" if report["timing_ok"] else "REVISAR ANTES DO TREINO"
        actual = report["observed_fps"]
        rate = f"{actual:.2f}" if actual is not None else "indisponível"
        print(f"[TIMING] {status}: alvo={self.fps} FPS, real={rate} FPS; {path}", flush=True)
        return report
