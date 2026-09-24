"""Fecha um episodio Direct: monta o video, aplica os MESMOS criterios do controlador
scriptado (retido 2 s e copo em pe) e escreve report.json.

Criterios copiados de run_attempt.py para a comparacao ser legitima:
  retido  = em todos os quadros da janela final de 2 s, copo acima de 0,6*8 cm da
            altura inicial, com pelo menos 2 dedos distintos em contato e sem tocar a mesa
  em pe   = inclinacao do copo abaixo de 15 graus antes do fecho e na janela final
  aceito  = retido e em pe e sem autocolisao e sem mao na mesa

Uso: .venv/bin/python scripts/direct_finish.py results/direct-01/x0.30_y-0.20
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from direct_env import FPS, TILT_LIMIT_DEG

ap = argparse.ArgumentParser()
ap.add_argument("episode_dir")
ap.add_argument("--hold-seconds", type=float, default=2.0)
ap.add_argument("--min-lift-m", type=float, default=0.048)   # 0,6 x 8 cm, como o scriptado
a = ap.parse_args()
ep = Path(a.episode_dir)
meta = json.loads((ep / "episode.json").read_text())

samples = []
for path in sorted((ep / "actions").glob("samples-*.json")):
    samples.extend(json.loads(path.read_text()))
calls = [json.loads(p.read_text()) for p in sorted((ep / "actions").glob("call-*.json"))]
truth0 = json.loads(sorted((ep / "truth").glob("step-*.json"))[0].read_text())
z0 = truth0["cup"][2]

hold_n = int(a.hold_seconds * FPS)
hold = samples[-hold_n:] if len(samples) >= hold_n else []
tilt_before = max([s["cup_tilt_deg"] for s in samples if not s["fingers"]] or [0.0])
tilt_hold = float(np.mean([s["cup_tilt_deg"] for s in hold])) if hold else None
retained = bool(hold) and all(s["cup"][2] > z0 + a.min_lift_m and len(s["fingers"]) >= 2 and not s["cup_table"] for s in hold)
upright = tilt_before < TILT_LIMIT_DEG and tilt_hold is not None and tilt_hold < TILT_LIMIT_DEG
clean = meta.get("aborted") is None
report = {
    "episode": ep.name,
    "policy": "direct",
    "model": meta.get("policy_model", "NAO DECLARADO: preencha policy_model em episode.json"),
    "cup_xy_privileged": meta["case"]["cup_xy"],
    "calls_used": meta["calls"],
    "calls_budget": meta["max_calls"],
    "rejected_calls": sum(1 for c in calls if not c["accepted"]),
    "sim_seconds": round(meta["frames"] / FPS, 2),
    "lift_m": round(float(max([s["cup"][2] for s in samples] or [z0]) - z0), 4),
    "retained_for_2s": bool(retained),
    "cup_tilt_deg_before_close": round(float(tilt_before), 2),
    "cup_tilt_deg_at_hold": None if tilt_hold is None else round(tilt_hold, 2),
    "upright_grasp": bool(upright),
    "aborted": meta.get("aborted"),
    "accepted": bool(retained and upright and clean),
    "criteria": {"tilt_limit_deg": TILT_LIMIT_DEG, "min_lift_m": a.min_lift_m, "hold_seconds": a.hold_seconds,
                 "source": "run_attempt.py (mesmos criterios do controlador scriptado)"},
}
(ep / "report.json").write_text(json.dumps(report, indent=2) + "\n")

frames = sorted((ep / "frames").glob("*.jpg"))
if frames and not (ep / "direct-run.mp4").exists():
    listing = ep / "frames.txt"
    listing.write_text("".join(f"file 'frames/{p.name}'\nduration {1/FPS:.5f}\n" for p in frames))
    subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", "frames.txt", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", f"fps={FPS}", "-preset", "veryfast", "-crf", "20",
                    "direct-run.mp4"], check=False, cwd=ep)
print(json.dumps(report, indent=2, ensure_ascii=False))
