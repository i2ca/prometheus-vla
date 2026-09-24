"""Teste do arnes SEM modelo nenhum: uma politica geometrica boba que usa a posicao
VERDADEIRA do copo. Serve so para verificar que start/act/finish, o portao e a
execucao funcionam antes de gastar cota. O resultado NAO e resultado de experimento
e nao deve ser comparado com nada: ele enxerga o que a politica de verdade nao ve.

Uso: .venv/bin/python scripts/direct_selftest.py results/direct-selftest/x0.40_y-0.20 --cup 0.40,-0.20
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from direct_env import DirectEpisode, MAX_TARGET_DISTANCE_M, MAX_TARGET_ROTATION_RAD, quat_of
import direct_cases

ap = argparse.ArgumentParser()
ap.add_argument("episode_dir"); ap.add_argument("--cup", required=True); ap.add_argument("--max-calls", type=int, default=40)
a = ap.parse_args()
x, y = (float(v) for v in a.cup.split(","))
case = direct_cases.case((x, y), direct_cases.home_arm_pose())
ep = DirectEpisode(a.episode_dir)
obs = ep.start(case, max_calls=a.max_calls)
import json as _json
_m = _json.loads((Path(a.episode_dir) / "episode.json").read_text())
_m["policy_model"] = "NENHUM: politica geometrica de teste com a verdade do copo"
(Path(a.episode_dir) / "episode.json").write_text(_json.dumps(_m, indent=2) + "\n")

# alvo grosseiro: 13 cm atras do copo na direcao do robo, altura da pega, orientacao inicial mantida
cup = np.array([x, y, 0.752])
quat = obs["current_palm"]["quaternion_wxyz"]
goal = cup + np.array([-0.10, 0.0, 0.055])
for phase, goal_pos, grip in (("aproxima", goal, "open"), ("fecha", goal, "closed"),
                              ("levanta", goal + np.array([0.0, 0.0, 0.10]), "closed"),
                              ("segura", goal + np.array([0.0, 0.0, 0.10]), "keep")):
    for _ in range(a.max_calls):
        cur = np.array(obs["current_palm"]["position"], float)
        delta = goal_pos - cur
        dist = float(np.linalg.norm(delta))
        if dist < 0.004 and phase not in ("fecha", "segura"):
            break
        scale = 0.9
        want = list(quat)
        for _try in range(4):   # rejeicao do portao: encurta o passo e tenta de novo, como uma politica faria
            step = delta if dist <= MAX_TARGET_DISTANCE_M else delta / dist * MAX_TARGET_DISTANCE_M
            step = step * scale
            obs = ep.load().act({"reason": f"selftest {phase}", "steps": 3,
                                 "target": {"position": (cur + step).tolist(), "quaternion_wxyz": want, "gripper": grip}})
            if not obs.get("rejected"):
                break
            if any("rotacao" in e for e in (obs["previous_execution"] or {}).get("errors", [])):
                # o punho ficou para tras: pede so uma fracao do giro, medida a partir da pose atual
                cq = np.array(obs["current_palm"]["quaternion_wxyz"], float)
                tq = np.array(want, float) * (1 if np.dot(cq, want) >= 0 else -1)
                want = list(cq + 0.5 * (tq - cq) / max(np.linalg.norm(tq - cq), 1e-9) * min(0.9 * MAX_TARGET_ROTATION_RAD, 0.3))
                want = list(np.array(want) / np.linalg.norm(want))
            else:
                scale *= 0.5
        if obs.get("finished"):
            break
        if phase in ("fecha", "segura"):
            break
    if obs.get("finished"):
        break
print(json.dumps({"calls": obs["step_id"], "finished": obs["finished"],
                  "palm": obs["current_palm"]["position"]}, indent=2))
