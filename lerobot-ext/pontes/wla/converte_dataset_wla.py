#!/usr/bin/env python
"""Converte um dataset NOSSO (29 juntas) para o formato de colunas do UnifoLM-WLA.

POR QUE ISTO EXISTE
-------------------
O carregador deles (`policies/unifolm_wla/dataloader/multi_source_dataset/`) lê LeRobot
nativamente e mistura fontes com `ConcatDataset` — os datasets da Unitree e os nossos na mesma
corrida. O que ele NÃO aceita é o nosso formato de ação.

O espaço unificado deles tem 54 dimensões e é de POSE DE MÃO: `left_ee_pose`, `right_ee_pose`,
as duas garras, cintura, tronco, pernas e base. Não há compartimento nenhum para ângulo de junta
de braço. E os nossos datasets têm uma coluna `action` única com 29 ângulos.

Então a conversão é: juntas -> cinemática direta -> xyz + rotvec, em colunas nomeadas.

    python pontes/wla/converte_dataset_wla.py \
        --origem meu_dataset/pega_copo_sem_prof_2026-09-18 \
        --destino meu_dataset/wla_pega_copo

A FK É A MESMA QUE JÁ FOI VALIDADA
----------------------------------
Sai da classe `Cinematica` de `pontes/unifolm-vla/roda_unifolm_mujoco.py`, escrita em 18/09 para
a ponte do UnifoLM-VLA-0 e conferida contra o `dataset_statistics.json` deles pelo
`confere_estado_vla.py`. Ela usa o modelo REDUZIDO do nosso `G1_29_ArmIK` e os frames `L_ee`/`R_ee`,
que ficam 5 cm à frente do `wrist_yaw`. Duas consequências que precisam estar escritas:

  1. O modelo reduzido TRAVA a cintura em zero. Então a pose sai no referencial do pelvis com o
     tronco reto, e o yaw real vai separado, na coluna `waist_action_joint`. É o mesmo que eles
     fazem: `waist_joint` é campo próprio, fora da pose da mão.

  2. O deslocamento de 5 cm do frame da mão é CONSTANTE. Como ele entra igual no estado e na
     ação, o modelo aprende a relação entre os dois e o viés se cancela. O que não pode é misturar
     dois frames diferentes — por isso estado e ação saem da MESMA função.

`ee_format` de saída é `xyz_rvec` (eixo-ângulo), e não `xyz_rpy`: a FK devolve matriz de rotação,
e rotvec sai dela sem passar por ângulos de Euler, que têm gimbal lock e ambiguidade de sinal.
O `se3_utils.py` deles aceita os dois — declarar `ee_format: "xyz_rvec"` no YAML.

O QUE NÃO TEMOS, E COMO ENTRA
-----------------------------
  cintura roll/pitch  zeros. O nosso schema de 29 só tem yaw.
  pernas, base, altura  ausentes. Usar `arm_type: "dual"` (e não `dual_with_legs`) no YAML.
  fig6d               ausente por ora. A Dex3 tem 3 dedos e 7 juntas; eles falam em garra de 2
                      dedos ou mão de 5. Enquanto não se souber como o `fig6d` codifica isso,
                      a mão entra só pelo escalar de garra, que é o caminho conservador.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "unifolm-vla"))
sys.path.insert(0, str(AQUI.parent.parent))

# Os nossos datasets existem em DOIS schemas, e a diferença é a cintura:
#   v2 (29 dims): 0..13 braços | 14 kWaistYaw | 15..21 mão esq | 22..28 mão dir
#   v1 (28 dims): 0..13 braços |      —       | 14..20 mão esq | 21..27 mão dir
# O de simulação é v2; o do robô real (`dataset_g1_cup_convertido`) é v1. Deduzir pelo
# tamanho evita ter que passar uma bandeira e errá-la em silêncio — um deslocamento de
# uma posição aqui mandaria a mão esquerda para o lugar do yaw sem erro nenhum.
BRACOS = slice(0, 14)


def fatias(n_dims):
    if n_dims == 29:
        return 14, slice(15, 22), slice(22, 29)
    if n_dims == 28:
        return None, slice(14, 21), slice(21, 28)
    raise SystemExit(f"schema de {n_dims} dims desconhecido (espero 28 ou 29)")


def rot_para_rotvec(R):
    """Matriz de rotação -> eixo-ângulo (3), pela fórmula de Rodrigues inversa."""
    t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(t))
    if ang < 1e-8:
        return np.zeros(3)
    if abs(ang - np.pi) < 1e-6:
        # Perto de 180° o seno some e a fórmula geral perde precisão: tira o eixo da
        # diagonal de (R + I), que continua bem condicionada.
        d = np.sqrt(np.clip(np.diag(R) + 1.0, 0.0, None) / 2.0)
        i = int(np.argmax(d))
        eixo = np.zeros(3)
        eixo[i] = d[i]
        for j in range(3):
            if j != i:
                eixo[j] = (R[i, j] + R[j, i]) / (4.0 * d[i]) if d[i] > 1e-8 else 0.0
        eixo /= (np.linalg.norm(eixo) or 1.0)
        return eixo * ang
    s = 2.0 * np.sin(ang)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s * ang


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origem", required=True)
    p.add_argument("--destino", required=True)
    p.add_argument("--tarefa", default=None,
                   help="sobrescreve a frase da tarefa (por padrão mantém a do dataset)")
    args = p.parse_args()

    origem, destino = Path(args.origem), Path(args.destino)
    if destino.exists():
        shutil.rmtree(destino)

    from roda_unifolm_mujoco import Cinematica, dedos_para_garra
    print("⏳ montando a cinemática (Pinocchio)…", flush=True)
    cin = Cinematica()

    # Vídeos e metadados vão inteiros; só as colunas de ação e estado são reescritas.
    print("⏳ copiando vídeos e metadados…", flush=True)
    shutil.copytree(origem, destino)

    arquivos = sorted((destino / "data").rglob("*.parquet"))
    print(f"⏳ convertendo {len(arquivos)} arquivo(s) de dados…", flush=True)
    total = 0
    for arq in arquivos:
        df = pd.read_parquet(arq)
        n = len(df)
        saida = {k: np.zeros((n, d), dtype=np.float32) for k, d in (
            ("left_ee", 6), ("right_ee", 6), ("left_gripper", 1),
            ("right_gripper", 1), ("waist", 3))}
        saida_st = {k: v.copy() for k, v in saida.items()}

        for origem_col, alvo in (("action", saida), ("observation.state", saida_st)):
            vet = np.stack(df[origem_col].values).astype(float)
            CINTURA, MAO_ESQ, MAO_DIR = fatias(vet.shape[1])
            for i in range(n):
                q = vet[i]
                (pe, Re), (pd_, Rd) = cin.fk(q[BRACOS])
                alvo["left_ee"][i] = np.concatenate([pe, rot_para_rotvec(Re)])
                alvo["right_ee"][i] = np.concatenate([pd_, rot_para_rotvec(Rd)])
                alvo["left_gripper"][i] = dedos_para_garra(q[MAO_ESQ], "esq")
                alvo["right_gripper"][i] = dedos_para_garra(q[MAO_DIR], "dir")
                # Só temos yaw; roll e pitch em zero. No schema v1 nem yaw existe.
                alvo["waist"][i] = [0.0, 0.0, q[CINTURA] if CINTURA is not None else 0.0]

        # Os nomes são os que o `configs/unitree.yaml` deles procura.
        for pref, d in (("action", saida), ("observation.state", saida_st)):
            df[f"{pref}.left_ee_pose_gripper_base"] = list(d["left_ee"])
            df[f"{pref}.right_ee_pose_gripper_base"] = list(d["right_ee"])
            # ESCALAR, e não vetor de um elemento. MEDIDO em 21/09 contra o parquet deles:
            # `action.left_gripper` é `float32` 4.5, não `[4.5]`. Com o vetor, o carregador
            # morre em `TypeError: Couldn't cast array of type list<element: float> to float`
            # lá dentro do pyarrow, longe da causa. O `info.json` declara shape [1] nos dois.
            df[f"{pref}.left_gripper"] = d["left_gripper"][:, 0].astype("float32")
            df[f"{pref}.right_gripper"] = d["right_gripper"][:, 0].astype("float32")
        df["action.waist_action_joint"] = list(saida["waist"])
        df["observation.state.waist_state_joint"] = list(saida_st["waist"])
        df.to_parquet(arq, index=False)
        total += n
        print(f"   {arq.relative_to(destino)}: {n} quadros", flush=True)

    # O info.json precisa declarar as colunas novas, senão o LeRobot não as enxerga.
    info = json.load(open(destino / "meta" / "info.json"))
    f = info["features"]
    for pref in ("action", "observation.state"):
        for nome, dim in ((f"{pref}.left_ee_pose_gripper_base", 6),
                          (f"{pref}.right_ee_pose_gripper_base", 6),
                          (f"{pref}.left_gripper", 1),
                          (f"{pref}.right_gripper", 1)):
            f[nome] = {"dtype": "float32", "shape": [dim], "names": None}
    f["action.waist_action_joint"] = {"dtype": "float32", "shape": [3], "names": None}
    f["observation.state.waist_state_joint"] = {"dtype": "float32", "shape": [3], "names": None}
    json.dump(info, open(destino / "meta" / "info.json", "w"), indent=4)

    print(f"\n✅ {total} quadros convertidos em {destino}")
    print("   ee_format: xyz_rvec | arm_type: dual | sem pernas, base nem fig6d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
