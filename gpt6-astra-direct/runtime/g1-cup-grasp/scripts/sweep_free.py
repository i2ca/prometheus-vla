"""Bounded geometry diagnostic; retains every free-cup trial, including failures."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path
from free_cup_test import trial, MODEL, ROOT


def main():
    directory = Path(ROOT) / "results" / ("free-sweep-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir()
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    (directory / "provenance.json").write_text(json.dumps({"model": MODEL, "source_sha256": sources}, indent=2))
    summary = []
    for i, (pitch, dz, gravity) in enumerate(itertools.product([0.0, 0.35, 0.65], [-0.02, -0.04, -0.06, -0.08], [False, True])):
        p = {"reason": "Bounded diagnostic: horizontal historical grasp asks wrist to reach below feasible pose and contacts table. Test wrist inclination and higher grip, with and without gravity compensation.",
             "offset": [-0.02, -0.02, dz], "palm_pitch_rad": pitch,
             "gravity_compensation": gravity, "home_lift": 0.10}
        run = directory / f"trial-{i+1:02d}"
        run.mkdir()
        (run / "parameters.json").write_text(json.dumps(p, indent=2))
        result = trial(p, run)
        summary.append({"directory": str(run), **result})
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        print(i+1, pitch, dz, gravity, "held", result["retained_for_2s"], "table", result["hand_table_frames"], "ik", round(result["max_ik_position_error_m"], 3), "lift", round(result["lift_m"], 3), flush=True)
    print(directory, flush=True)


if __name__ == "__main__":
    main()
