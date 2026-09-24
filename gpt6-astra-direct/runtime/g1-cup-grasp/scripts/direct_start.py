"""Abre um episodio Direct. Uso:
  .venv/bin/python scripts/direct_start.py results/direct-01/x0.30_y-0.20 --cup 0.30,-0.20 [--max-calls 60] [--no-video]
Imprime a observacao inicial (caminhos das 3 imagens, pose medida da palma, limites e next_call).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from direct_env import DirectEpisode
import direct_cases

ap = argparse.ArgumentParser()
ap.add_argument("episode_dir")
ap.add_argument("--cup", required=True, help="x,y do copo na mesa")
ap.add_argument("--max-calls", type=int, default=60)
ap.add_argument("--max-sim-seconds", type=float, default=40.0)
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--model", required=True, help="modelo que vai atuar como politica (ex.: gpt-6-astra xhigh)")
a = ap.parse_args()

x, y = (float(v) for v in a.cup.split(","))
case = direct_cases.case((x, y), direct_cases.home_arm_pose())
ep = DirectEpisode(a.episode_dir)
obs = ep.start(case, max_calls=a.max_calls, max_sim_seconds=a.max_sim_seconds, video=not a.no_video)
import json as _json
meta = _json.loads((Path(a.episode_dir) / "episode.json").read_text())
meta["policy_model"] = a.model
(Path(a.episode_dir) / "episode.json").write_text(_json.dumps(meta, indent=2) + "\n")
print(json.dumps(obs, indent=2, ensure_ascii=False))
