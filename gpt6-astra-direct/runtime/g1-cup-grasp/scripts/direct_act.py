"""Executa uma acao no episodio. Uso:
  .venv/bin/python scripts/direct_act.py results/direct-01/x0.30_y-0.20 --action acao.json
  .venv/bin/python scripts/direct_act.py <ep> --json '{"reason":"...","steps":3,"target":{...}}'

Formato da acao:
{
  "reason": "evidencia visual e proposito, em uma frase",
  "steps": 1..5,                      # 1 step = 0,2 s de movimento
  "target": {
    "position": [x, y, z],            # metros, origem do ambiente
    "quaternion_wxyz": [w, x, y, z],  # unitario
    "gripper": "keep" | "open" | "closed" | 0.0..1.0
  }
}
Rejeicao nao executa nada e pode ser corrigida; ela consome uma chamada do orcamento.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from direct_env import DirectEpisode

ap = argparse.ArgumentParser()
ap.add_argument("episode_dir")
ap.add_argument("--action", help="arquivo JSON com a acao")
ap.add_argument("--json", help="acao inline")
a = ap.parse_args()
action = json.loads(Path(a.action).read_text() if a.action else a.json)
obs = DirectEpisode(a.episode_dir).load().act(action)
print(json.dumps(obs, indent=2, ensure_ascii=False))
