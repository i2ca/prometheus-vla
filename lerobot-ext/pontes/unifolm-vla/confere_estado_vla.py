#!/usr/bin/env python
"""A nossa pose de mão cai dentro do que o UnifoLM-VLA-0 viu no treino?

POR QUE ISTO EXISTE, ANTES DE QUALQUER INFERÊNCIA
--------------------------------------------------
O UnifoLM-VLA-0 não fala em ângulo de junta. O `ActionEncoding.EE_R6_G1` do
código deles é, literalmente:

    2 x [EEF XYZ (3) + R6 (6) + Gripper Open/Close (1)] + Waist roll-pitch-yaw (3)

ou seja, 23 números que descrevem ONDE a mão está e COMO está orientada. Isso é
ótimo — independe de quantas juntas o nosso robô tem, e a conversão de volta é a
nossa IK. Mas cria uma armadilha nova: a pose tem que sair na MESMA convenção em
que eles gravaram. Frame de referência, sinal dos eixos, e qual par de colunas da
matriz de rotação vira o R6. Errar qualquer uma dessas não dá erro nenhum — só
faz o modelo achar que a mão aponta para outro lado.

Em 18/09 esse mesmo tipo de erro custou o dia: o robô nascia a 0,906 rad da pose
de treino e o punho entrava normalizado em -5,02 num espaço de [-1, +1]. Só
descobrimos depois de ver o robô fazer besteira. Aqui a ordem se inverte.

O QUE ELE MEDE
--------------
Monta as 23 dimensões a partir de uma pose nossa (cinemática direta com o mesmo
Pinocchio e o mesmo URDF da nossa IK) e compara, dimensão por dimensão, com o
`dataset_statistics.json` das 12 tarefas de G1 do checkpoint deles.

Dentro da faixa  -> a convenção bate, dá para ligar.
Fora da faixa    -> está trocada, e o número diz qual dimensão.

    python confere_estado_vla.py
    python confere_estado_vla.py --pose 0.958 0.683 ... (14 juntas + cintura)
"""
import argparse
import json
import sys

import numpy as np
import pinocchio as pin
from huggingface_hub import hf_hub_download

sys.path.insert(0, "/home/mrwlker/DEV/prometheus-vla/lerobot-ext")

# O `ASSETS_DIR` do `g1_arm_ik.py` aponta para `robot/unitree_g1/assets`, que nesta cópia do
# repositório não existe — o URDF está em `lerobot-ext/assets/g1`. Importar o módulo da IK só
# para pegar a constante também arrastaria casadi e o solver inteiro, que aqui não usamos.
from pathlib import Path
# Caminho ABSOLUTO era uma bomba: so funcionava nesta maquina e neste login. Procurar
# `assets/g1` subindo a partir deste arquivo funciona aqui e na athena.
def _acha_assets(inicio):
    for d in [inicio, *inicio.parents]:
        if (d / "assets" / "g1").is_dir():
            return d / "assets"
    raise SystemExit("nao achei `assets/g1` subindo a partir de " + str(inicio))


ASSETS_DIR = _acha_assets(Path(__file__).resolve().parent)

# Pose em que os 302 episódios do copo começam (média do primeiro quadro), medida
# em 18/09 no `cotreino_completo_2026-09-11`: 14 juntas de braço + yaw do tronco.
POSE_COPO = [0.958, 0.683, 1.476, -0.554, -1.501, 0.505, -0.225,
             -0.941, -0.628, -0.313, 0.388, 0.232, 0.655, 0.337, -0.036]

# ATENÇÃO — a ORDEM aqui NÃO é a que o comentário deles diz. O
# `rlds_dataloader/.../configs.py` escreve
#
#     EE_R6_G1 = 6   # 2 x [EEF XYZ (3) + R6 (6) + Gripper Open/Close (1)] + Waist roll-pitch-yaw (3)
#
# o que faria as garras caírem nas dimensões 9 e 19. Os DADOS dizem outra coisa: no
# `dataset_statistics.json` as dimensões 18 e 19 é que vão de 0,02 a 4,50 (unidade de garra), e a
# 9 vai de 0,175 a 0,411, que é posição em metros. Ou seja, as duas garras estão JUNTAS no fim:
#
#     [ESQ xyz + ESQ r6] + [DIR xyz + DIR r6] + [garra ESQ, garra DIR] + [cintura r, p, y]
#        0..2     3..8       9..11    12..17        18        19          20   21   22
#
# Medido em 18/09 por este próprio script, que acusou "ESQ garra = 1,000 contra faixa de treino
# 0,063..0,432" — garra não tem faixa de 0,06 a 0,43, posição tem.
NOMES_23 = ([f"ESQ pos {e}" for e in "xyz"] + [f"ESQ r6[{i}]" for i in range(6)] +
            [f"DIR pos {e}" for e in "xyz"] + [f"DIR r6[{i}]" for i in range(6)] +
            ["garra ESQ", "garra DIR"] +
            ["cintura roll", "cintura pitch", "cintura yaw"])


def rot_para_r6(R):
    """R6 = as duas PRIMEIRAS COLUNAS da matriz de rotação, empilhadas.

    É a convenção do `rotmat_to_rot6d` deles
    (`rlds_dataloader/datasets/rlds/oxe/utils/droid_utils.py`). Se um dia o
    resultado sair fora da faixa em TODAS as seis dimensões de rotação e dentro
    nas de posição, o suspeito número um é aqui: linhas em vez de colunas.
    """
    return np.concatenate([R[:, 0], R[:, 1]])


def monta_23(juntas_bracos, cintura_yaw, garras=(4.5, 4.5)):
    """14 juntas de braço + yaw do tronco -> as 23 dimensões do EE_R6_G1."""
    robo = pin.RobotWrapper.BuildFromURDF(str(ASSETS_DIR / "g1" / "g1_body29_hand14.urdf"),
                                          str(ASSETS_DIR / "g1"))
    q = pin.neutral(robo.model)
    nomes_juntas = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    for nome, valor in zip(nomes_juntas, juntas_bracos):
        if robo.model.existJointName(nome):
            q[robo.model.idx_qs[robo.model.getJointId(nome)]] = valor
    if robo.model.existJointName("waist_yaw_joint"):
        q[robo.model.idx_qs[robo.model.getJointId("waist_yaw_joint")]] = cintura_yaw

    pin.forwardKinematics(robo.model, robo.data, q)
    pin.updateFramePlacements(robo.model, robo.data)

    saida = []
    for lado in ("left", "right"):
        quadro = None
        for cand in (f"{lado}_wrist_yaw_link", f"{lado}_rubber_hand", f"{lado}_hand_palm_link"):
            if robo.model.existFrame(cand):
                quadro = cand
                break
        if quadro is None:
            raise SystemExit(f"não achei o frame da mão {lado} no URDF")
        T = robo.data.oMf[robo.model.getFrameId(quadro)]
        saida.append(np.concatenate([T.translation, rot_para_r6(T.rotation)]))
        print(f"   frame usado para a mão {lado}: {quadro}")
    # cintura: só temos o yaw; roll e pitch entram em zero, que é o que o nosso robô faz
    return np.concatenate([saida[0], saida[1], list(garras), [0.0, 0.0, cintura_yaw]])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pose", nargs=15, type=float, default=POSE_COPO,
                   help="14 juntas de braço + yaw do tronco")
    args = p.parse_args()

    print("Montando as 23 dimensões a partir da nossa pose (FK com o URDF da nossa IK):")
    nosso = monta_23(args.pose[:14], args.pose[14])

    est = json.load(open(hf_hub_download("unitreerobotics/UnifoLM-VLA-Base",
                                         "dataset_statistics.json")))
    tarefas = list(est)
    # faixa que a UNIÃO das 12 tarefas cobre: é o que o modelo viu ao todo
    lo = np.min([est[t]["proprio"]["min"] for t in tarefas], axis=0)
    hi = np.max([est[t]["proprio"]["max"] for t in tarefas], axis=0)

    print(f"\ncomparando com a união das {len(tarefas)} tarefas de G1 do checkpoint\n")
    print(f"{'dim':<16}{'nosso':>10}{'treino min':>12}{'treino max':>12}   situação")
    fora = 0
    for i, nome in enumerate(NOMES_23):
        v, a, b = nosso[i], lo[i], hi[i]
        folga = (b - a) * 0.10 or 0.01
        if a - folga <= v <= b + folga:
            s = "dentro"
        else:
            s = "FORA <<<"
            fora += 1
        print(f"{nome:<16}{v:>10.3f}{a:>12.3f}{b:>12.3f}   {s}")
    print(f"\n{fora} de 23 fora da faixa.")
    if fora == 0:
        print("A convenção bate. Dá para ligar a inferência.")
    else:
        print("Convenção trocada — as dimensões marcadas dizem onde. NÃO ligar ainda:\n"
              "  posição fora / rotação dentro -> frame de referência errado (ou origem);\n"
              "  rotação fora / posição dentro -> R6 por linhas em vez de colunas;\n"
              "  tudo fora                     -> frame da mão errado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
