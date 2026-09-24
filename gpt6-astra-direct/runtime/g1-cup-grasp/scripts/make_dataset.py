"""Gera o mini dataset: sorteia posicao e yaw do copo numa regiao, roda run_attempt com record_dataset em paralelo,
separa episodios aceitos pela fisica (retido 2 s, em pe, sem autocolisao) dos rejeitados, e monta o LeRobot v3.0 com
os aceitos (build_lerobot_dataset.py no .venv-lerobot). Cada episodio fica inteiro em <out>/episodes/ep-NNN (video 3x2,
trajetoria, auditoria, parameters.json com o sorteio), os rejeitados em <out>/rejected/ com o motivo em summary.json.

Uso: .venv/bin/python scripts/make_dataset.py experiments/attempt-27.parameters.json results/dataset-01 --n 60 --target 30 --jobs 3 --seed 1
"""
import argparse, json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
ap = argparse.ArgumentParser(); ap.add_argument("parameters"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--target", type=int, default=30)
ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--x", default="0.36,0.48"); ap.add_argument("--y", default="-0.20,0.20"); ap.add_argument("--yaw", default="60,120")
ap.add_argument("--repo-id", default="lewislf/g1_cup_sim_v1")
a = ap.parse_args()
base = json.loads(Path(a.parameters).read_text()); base["record_dataset"] = True
out = Path(a.out); (out / "episodes").mkdir(parents=True, exist_ok=False); (out / "rejected").mkdir(); (out / "tmp").mkdir()
rng = np.random.default_rng(a.seed); rx, ry, ryaw = ([float(v) for v in s.split(",")] for s in (a.x, a.y, a.yaw))
samples = [(float(rng.uniform(*rx)), float(rng.uniform(*ry)), float(rng.uniform(*ryaw))) for _ in range(a.n)]


def run(i):
    x, y, yaw = samples[i]; p = dict(base); p["cup_xy"] = [round(x, 4), round(y, 4)]; p["cup_yaw_deg"] = round(yaw, 2)
    p["reason"] = f"Mini dataset {out.name}, sorteio {i} (seed {a.seed}): copo em x={x:.3f} y={y:.3f} yaw={yaw:.1f}. Controlador da {Path(a.parameters).stem}."
    pf = out / "tmp" / f"sample-{i:03d}.json"; pf.write_text(json.dumps(p, indent=2)); d = out / "tmp" / f"sample-{i:03d}"
    r = subprocess.run([sys.executable, str(HERE / "run_attempt.py"), "--parameters", str(pf), "--run-dir", str(d)], capture_output=True, text=True)
    row = {"sample": i, "cup_xy": p["cup_xy"], "cup_yaw_deg": p["cup_yaw_deg"], "dir": str(d)}
    try:
        rep = json.loads((d / "run-report.json").read_text()); phys = json.loads((d / "physics-report.json").read_text()) if (d / "physics-report.json").exists() else {}
        row.update({k: rep.get(k) for k in ("status", "error", "ik_mode", "waist_yaw_deg_at_hold")}); row.update({k: phys.get(k) for k in ("accepted", "retained_for_2s", "upright_grasp", "cup_tilt_deg_before_close", "cup_tilt_deg_at_hold", "lift_m", "hand_table_frames", "self_collision_frames")})
        det = json.loads((d / "cup-detection.json").read_text()) if (d / "cup-detection.json").exists() else {}
        row["estimate_error_xy_m"] = det.get("estimate_error_xy_m"); row["yaw_error_deg"] = det.get("yaw_error_deg")
    except Exception as e:
        row.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "stderr_tail": r.stderr[-400:]})
    if row.get("status") == "completed":
        subprocess.run([sys.executable, str(HERE / "audit_attempt.py"), str(d)], capture_output=True)
    print(f"[{i:03d}] x={x:.3f} y={y:+.3f} yaw={yaw:5.1f} -> {row.get('status')} aceito={row.get('accepted')} tilt={row.get('cup_tilt_deg_at_hold')} lift={row.get('lift_m')} erro={(row.get('error') or '')[:60]}", flush=True)
    return row


with ThreadPoolExecutor(a.jobs) as ex:
    rows = list(ex.map(run, range(a.n)))
accepted = [r for r in rows if r.get("accepted")]
kept = accepted[:a.target]
for k, r in enumerate(kept):
    dst = out / "episodes" / f"ep-{k:03d}"; shutil.move(r["dir"], dst); r["episode"] = k; r["dir"] = str(dst)
for r in rows:
    if "episode" not in r and Path(r["dir"]).exists():
        dst = out / "rejected" / Path(r["dir"]).name; shutil.move(r["dir"], dst); r["dir"] = str(dst)
        r["reject_reason"] = r.get("error") or ("nao aceito pela fisica: " + ", ".join(k for k in ("retained_for_2s", "upright_grasp") if not r.get(k)) + (f", autocolisao {r['self_collision_frames']} quadros" if r.get("self_collision_frames") else "")) if not r.get("accepted") else "excedente ao alvo"
shutil.rmtree(out / "tmp")
summary = {"parameters_base": a.parameters, "seed": a.seed, "region": {"x": rx, "y": ry, "yaw_deg": ryaw}, "sampled": a.n, "accepted": len(accepted), "kept": len(kept),
           "acceptance_rate": len(accepted) / a.n, "rows": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"aceitos {len(accepted)}/{a.n}, mantidos {len(kept)}")
if kept:
    lp = ROOT / ".venv-lerobot" / "bin" / "python"
    subprocess.run([str(lp), str(HERE / "build_lerobot_dataset.py"), "--out", str(out / "lerobot"), "--repo-id", a.repo_id] + [str(Path(r["dir"]) / "dataset") for r in kept], check=True)
