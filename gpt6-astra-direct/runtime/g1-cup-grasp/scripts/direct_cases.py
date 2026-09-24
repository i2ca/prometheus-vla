"""Casos do experimento Direct, alinhados com results/sweep-positions-03 (mesma cena, mesmo copo, mesmo yaw).

A linha de base e o controlador scriptado da varredura 03. O Direct roda nos mesmos
(x, y), para comparacao caso a caso, como o evaluation_cases.json do GPT-as-Policy.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pose de prontidao usada na varredura 03: semente do braco com o roll do ombro em -0.35
BASE = {
    "home_shoulder_roll_rad": -0.35,
    "left_arm_pose": [0.3, 0.15, 0.0, 0.7, 0.0, 0.0, 0.0],
    "hand_kp": 8,
    "offset": [-0.02, -0.02, -0.02],
    "gain": 1.5,
    "cup_yaw_deg": 90.0,
    "instruction": "Pegue a caneca branca da mesa com a mao direita, levante-a pelo menos 8 cm "
                   "mantendo-a em pe, e segure por 2 segundos.",
}

# resultado do controlador scriptado em results/sweep-positions-03/summary.json
BASELINE = {
    (0.30, -0.35): {"status": "failed", "accepted": False, "why": "percepcao: nenhum candidato branco"},
    (0.30, -0.20): {"status": "failed", "accepted": False, "why": "percepcao: silhueta nao bate (41% da altura)"},
    (0.30, 0.00): {"status": "completed", "accepted": False,
                   "why": "copo derrubado NA APROXIMACAO: inclinacao 114,8 graus antes do fecho e lift -0,75 m (caiu da mesa)"},
    (0.30, 0.20): {"status": "completed", "accepted": False,
                   "why": "pega boa (retido, em pe, 9,7 cm) reprovada por 75 quadros de AUTOCOLISAO"},
    (0.40, -0.35): {"status": "failed", "accepted": False, "why": "percepcao: silhueta nao bate (17%)"},
    (0.40, -0.20): {"status": "completed", "accepted": True, "why": "-"},
    (0.40, 0.00): {"status": "failed", "accepted": None, "why": "INFRAESTRUTURA: ffmpeg SIGSEGV, nao e falha de controle"},
    (0.40, 0.20): {"status": "completed", "accepted": True, "why": "-"},
    (0.48, -0.35): {"status": "failed", "accepted": False, "why": "percepcao: calibracao ruim (marcador coberto, 24,6 px)"},
    (0.48, -0.20): {"status": "completed", "accepted": True, "why": "-"},
    (0.48, 0.00): {"status": "completed", "accepted": True, "why": "-"},
    (0.48, 0.20): {"status": "completed", "accepted": True, "why": "-"},
}

# alvo do experimento: onde o scriptado NAO foi aceito, tirando a falha de infraestrutura
TARGET_CASES = [xy for xy, r in BASELINE.items() if r["accepted"] is False]
CONTROL_CASES = [(0.40, -0.20), (0.48, 0.00)]   # dois casos ja aceitos, para medir regressao


def case(cup_xy, home_arm_pose):
    return {**BASE, "cup_xy": [round(cup_xy[0], 3), round(cup_xy[1], 3)], "home_arm_pose": list(home_arm_pose)}


def home_arm_pose():
    """Mesma pose inicial da varredura 03: primeiro quadro da semente com o roll do ombro alterado."""
    import numpy as np
    from g1_sim import load_seed
    arm, _, _ = load_seed(0)
    home = arm[0].copy()
    home[1] = BASE["home_shoulder_roll_rad"]
    return home.tolist()


def main():
    rows = []
    for xy in TARGET_CASES + CONTROL_CASES:
        rows.append({"cup_xy": list(xy), "group": "target" if xy in TARGET_CASES else "control",
                     "baseline": BASELINE[xy]})
    print(json.dumps({"cases": rows, "n_target": len(TARGET_CASES), "n_control": len(CONTROL_CASES)}, indent=2))


if __name__ == "__main__":
    main()
