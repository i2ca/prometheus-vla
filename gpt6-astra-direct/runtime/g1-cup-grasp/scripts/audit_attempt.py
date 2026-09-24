"""Auditoria adversarial de uma tentativa: folhas de contato densas das cameras e bateria de medicoes.
Roda depois de toda tentativa, antes de dizer que esta boa. Uso: .venv/bin/python scripts/audit_attempt.py results/attempt-NN
"""
import collections, json, subprocess, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); out = run / "audit"; out.mkdir(exist_ok=True)
s = json.load(open(run / "trajectory.json")); ik = json.load(open(run / "ik.json")); rep = json.load(open(run / "run-report.json"))
tl = json.load(open(run / "timeline.json")); cams = tl["cameras"]; W, Hh, COLS = 640, 360, 3
lines = []
P = lambda *a: lines.append(" ".join(str(x) for x in a))
P("FASES", [(e["t"], e.get("phase")) for e in tl["events"]])
t = np.array([x["t"] for x in s]); palm = np.array([x["palm"] for x in s]); cup = np.array([x["cup"] for x in s]); ph = [x["phase"] for x in s]
dt = np.diff(t); v = np.linalg.norm(np.diff(palm, axis=0), axis=1) / dt; a = np.diff(v) / dt[1:]; j = np.diff(a) / dt[2:]
P(f"1 palma: v max {v.max():.3f} m/s, media {v.mean():.3f}; |a| max {np.abs(a).max():.2f} m/s2; |jerk| max {np.abs(j).max():.0f} m/s3")
for k in collections.OrderedDict.fromkeys(ph):
    idx = [i for i, p in enumerate(ph) if p == k]
    if len(idx) > 2:
        vv = v[idx[0]:idx[-1]]; P(f"   {k:16s} dur {t[idx[-1]]-t[idx[0]]:.2f}s v max {vv.max():.3f} media {vv.mean():.3f} palma z {palm[idx[0],2]:.3f}->{palm[idx[-1],2]:.3f}")
top = np.argsort(np.abs(a))[-5:]; P("2 picos de |a| (t, fase, a):", [(round(float(t[i+1]), 2), ph[i+1], round(float(a[i]), 2)) for i in sorted(top)])
d = np.linalg.norm(cup - palm, axis=1)
for k in ("close", "lift", "retreat", "hold"):
    idx = [i for i, p in enumerate(ph) if p == k]
    if idx: P(f"3 dist copo-palma em {k}: {d[idx[0]]*100:.1f} -> {d[idx[-1]]*100:.1f} cm (var {(d[idx].max()-d[idx].min())*100:.2f} cm)")
idx = [i for i, p in enumerate(ph) if p in ("approach_behind", "approach_cup", "approach_above", "preshape")]
if idx: P(f"4 copo antes do fecho: deslocou {np.linalg.norm(cup[idx[-1]]-cup[idx[0]])*1000:.1f} mm, inclinou max {max(s[i]['cup_tilt_deg'] for i in idx):.1f} graus")
P("5 mao-mesa por fase:", dict(collections.Counter((x["phase"], tuple(x["hand_table"])) for x in s if x["hand_table"])), f"penetracao max {rep.get('max_substep_table_penetration_m', 0)*1000:.2f} mm")
for k in collections.OrderedDict.fromkeys(ph):
    c = collections.Counter(l for x in s if x["phase"] == k for l in x["finger_links"])
    if c: P(f"6 elos no copo em {k}:", dict(c))
byph = collections.defaultdict(list)
for e in ik: byph[e["phase"]].append(e["position_error_m"])
P("7 erro IK por fase (max mm):", {k: round(max(vv)*1000, 1) for k, vv in byph.items()}, f"| orientacao max {max(e['orientation_error_rad'] for e in ik):.3f} rad")
idx = [i for i, p in enumerate(ph) if p == "hold"]
if idx: P(f"8 espera: amplitude palma z {(palm[idx,2].max()-palm[idx,2].min())*1000:.2f} mm, copo z {(cup[idx,2].max()-cup[idx,2].min())*1000:.2f} mm, inclinacao {s[idx[0]]['cup_tilt_deg']:.1f}->{s[idx[-1]]['cup_tilt_deg']:.1f}")
if rep.get("hold_tips_cup_frame"): P("9 pontas na espera (r, z):", {k: (round(vv["r"], 3), round(vv["z"], 3)) for k, vv in rep["hold_tips_cup_frame"].items()}, "| dentro", rep.get("frames_finger_inside_cup"), "| alca", rep.get("frames_finger_in_handle_gap"))
sc = [(round(float(x["t"]), 2), x["phase"], x["self_collision"]) for x in s if x.get("self_collision")]
P(f"10 autocolisoes (braco direito x tronco/esquerdo): {len(sc)} quadros", sc[:3])
P("11 fisica:", {k: rep.get(k) for k in ("retained_for_2s", "upright_grasp", "accepted", "cup_tilt_deg_before_close", "cup_tilt_deg_at_hold", "lift_m", "warnings", "hand_table_frames", "self_collision_frames")})
P("12 percepcao:", {k: json.load(open(run / "cup-detection.json")).get(k) for k in ("estimate_error_xy_m", "yaw_deg_estimate", "yaw_error_deg", "silhouette_fit_rms_px")} if (run / "cup-detection.json").exists() else "sem deteccao")
# folhas de contato: cada camera, 4 quadros/s
for ci, cam in enumerate(cams):
    x0, y0 = (ci % COLS) * W, (ci // COLS) * Hh
    # 48 quadros cobrindo o video INTEIRO (antes: 4 fps fixos = so os primeiros 12 s; as fases de colocacao ficavam fora das folhas)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(run / "grasp-run.mp4")], capture_output=True, text=True).stdout.strip() or 12)
    fps_sheet = min(4.0, 48.0 / max(dur, 1e-3))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(run / "grasp-run.mp4"), "-vf", f"fps={fps_sheet:.4f},crop={W}:{Hh}:{x0}:{y0},scale=320:180,tile=8x6", "-frames:v", "1", "-update", "1", str(out / f"sheet_{cam}.png")], check=False)
(out / "audit.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print("folhas:", [f.name for f in out.glob("sheet_*.png")])
