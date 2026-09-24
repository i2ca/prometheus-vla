"""Varredura de posicoes do copo na mesa: uma tentativa completa por posicao (com video e auditoria), resumo em JSON.
Uso: .venv/bin/python scripts/sweep_positions.py experiments/attempt-22.parameters.json results/sweep-positions-01 [--grid 0.30,0.40,0.48:-0.35,-0.20,0.0,0.20]
"""
import json, subprocess, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run_attempt as ra

params = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
grid = "0.30,0.40,0.48:-0.35,-0.20,0.0,0.20"
if "--grid" in sys.argv: grid = sys.argv[sys.argv.index("--grid") + 1]
xs, ys = ([float(v) for v in part.split(",")] for part in grid.split(":"))
rows = []
for x in xs:
    for y in ys:
        run = out / f"pos_x{x:.2f}_y{y:+.2f}"; run.mkdir()
        p = dict(params); p["cup_xy"] = [x, y]
        (run / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
        row = {"cup_xy": [x, y], "dir": str(run)}
        try:
            r = ra.run(p, run)
            row.update({k: r.get(k) for k in ("status", "error", "ik_mode", "reach_probe_arm_only_error_m", "waist_yaw_deg_at_hold", "accepted", "retained_for_2s", "upright_grasp",
                                                "cup_tilt_deg_before_close", "cup_tilt_deg_at_hold", "lift_m", "hand_table_frames", "self_collision_frames", "max_ik_position_error_m")})
            det = json.loads((run / "cup-detection.json").read_text()) if (run / "cup-detection.json").exists() else {}
            row["estimate_error_xy_m"] = det.get("estimate_error_xy_m"); row["yaw_error_deg"] = det.get("yaw_error_deg")
        except Exception as e:
            row.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        subprocess.run([sys.executable, str(Path(__file__).parent / "audit_attempt.py"), str(run)], capture_output=True)
        rows.append(row); (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"x={x:.2f} y={y:+.2f} -> {row.get('status')} modo={row.get('ik_mode')} cintura={row.get('waist_yaw_deg_at_hold')} aceito={row.get('accepted')} tilt={row.get('cup_tilt_deg_at_hold')} erro={row.get('error')}", flush=True)
