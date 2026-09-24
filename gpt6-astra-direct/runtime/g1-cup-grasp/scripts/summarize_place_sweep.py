"""Resumo de uma varredura de colocacao (make_dataset com place_xy): por amostra, posicao inicial, aceite, erro de colocacao,
inclinacao final, contatos mao-coador, aceleracao maxima. Uso: summarize_place_sweep.py results/sweep-place-01"""
import json, sys, numpy as np
from pathlib import Path
from collections import Counter
root = Path(sys.argv[1]); rows = []
for d in sorted(list((root / "episodes").glob("ep-*")) + list((root / "rejected").glob("sample-*")) + list((root / "tmp").glob("sample-*/"))):
    if not (d / "run-report.json").exists(): continue
    r = json.loads((d / "run-report.json").read_text()); p = json.loads((d / "parameters.json").read_text()); pl = r.get("place") or {}
    t = json.loads((d / "trajectory.json").read_text()) if (d / "trajectory.json").exists() else []
    s = t if isinstance(t, list) else t.get("samples", [])
    cont = Counter(p_[1] if p_[0].startswith("right") else p_[0] for x in s for p_ in x.get("env_contacts", []) if len(p_) >= 2 and not ({"copo", "coador"} == {p_[0], p_[1]}))
    palm = np.array([x["palm"] for x in s]) if s else np.zeros((2, 3)); v = np.linalg.norm(np.diff(palm, axis=0), axis=1) * 30; a = np.abs(np.diff(v) * 30).max() if len(v) > 1 else 0
    rows.append({"dir": d.name, "cup_xy": p["cup_xy"], "yaw": p["cup_yaw_deg"], "status": r.get("status"), "accepted": r.get("accepted"), "placed": pl.get("placed"),
                 "err_mm": round((pl.get("xy_error_m") or 9) * 1000, 1), "tilt": round(pl.get("tilt_deg", -1), 1), "self_col": r.get("self_collision_frames"), "table": r.get("hand_table_frames"),
                 "contacts": dict(cont), "a_max": round(float(a), 1), "error": (r.get("error") or "")[:60]})
for x in rows: print(f"{x['dir']:12s} copo {x['cup_xy']} yaw {x['yaw']:5.1f} | {x['status']:9s} aceito {str(x['accepted']):5s} colocado {str(x['placed']):5s} erro {x['err_mm']:6.1f} mm tilt {x['tilt']:5.1f} | autocol {x['self_col']} mesa {x['table']} a_max {x['a_max']} | contatos {x['contacts']} {x['error']}")
ok = [x for x in rows if x["accepted"]]; print(f"\n{len(ok)}/{len(rows)} aceitas; erro mediano {np.median([x['err_mm'] for x in ok]) if ok else None} mm")
json.dump(rows, open(root / "sweep-summary.json", "w"), indent=1)
