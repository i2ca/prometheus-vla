#!/usr/bin/env python
"""Põe o UnifoLM-VLA-0 a dirigir o nosso G1 no MuJoCo.

O QUE ESTA PONTE RESOLVE
------------------------
O `init_lerobot_inference_v3.py` fala com política LeRobot: carrega pesos locais e recebe
ÂNGULO DE JUNTA. O UnifoLM-VLA-0 não faz nada disso — mora num servidor HTTP (a 8777), recebe
imagem + texto + 23 números de POSE DE MÃO, e devolve 25 passos de POSE DE MÃO. Entre os dois
falta tradução nos dois sentidos:

    juntas do MuJoCo --FK--> 23 dimensões --HTTP--> 25 x 23 --IK--> juntas do MuJoCo

Isto é essa tradução. Ver `confere_estado_vla.py` ao lado, que é quem PROVOU a convenção das 23
dimensões antes de qualquer inferência (a ordem real é ESQ xyz+r6, DIR xyz+r6, as DUAS garras
juntas, e só então a cintura — o comentário do código deles está errado).

AS TRÊS DECISÕES QUE IMPORTAM AQUI
----------------------------------
1. FK E IK NO MESMO MODELO. A FK usa o `reduced_robot` do nosso próprio `G1_29_ArmIK` e os
   frames `L_ee`/`R_ee` dele, não o URDF completo. Não é detalhe: o `L_ee` fica 5 cm à frente
   do `left_wrist_yaw_link`, e a cintura entra travada em zero no modelo reduzido. Usando o
   MESMO frame nos dois sentidos, qualquer diferença entre a nossa origem de mão e a deles vira
   um deslocamento CONSTANTE — o modelo lê "minha mão está em X", manda "vá para X+d", e nós
   reproduzimos X+d. O viés se cancela. Misturar os dois frames é que faria a mão errar 5 cm
   sem erro nenhum aparecer no log.

2. A CINTURA NÃO OBEDECE, POR PADRÃO. O modelo devolve roll, pitch e yaw de cintura; o nosso
   robô só tem yaw. Na primeira resposta real ele pediu roll 0,279 e pitch 0,151 — que nós
   simplesmente não conseguimos fazer. Pior: o alvo da mão vem num frame em que a cintura está
   fixa, então girar o tronco move a mão por fora da conta. Com `--cintura` o yaw passa; sem
   ela, o tronco fica parado e o braço faz o trabalho todo.

3. O PUNHO ESQUERDO É UMA CÓPIA DO DIREITO. O nosso MJCF só tem `right_wrist_camera`
   (`g1_29dof_with_hand.xml:495`). O modelo espera três imagens. Mandar duas mudaria a conta de
   tokens de imagem que ele viu no treino, então a de punho direito vai nos dois lugares — e é
   por isso que `--so-direita` é o PADRÃO: com a imagem do punho esquerdo mentindo, o que ele
   mandar para o braço esquerdo não merece confiança. O braço esquerdo fica onde começou.

A CONTA DO TEMPO
----------------
A inferência mede 1,8 s e devolve 25 passos. A 30 Hz esses 25 passos duram 0,83 s — menos do
que a resposta demora, e o robô passaria metade do tempo parado esperando. A 15 Hz duram 1,67 s,
que cobre quase toda a espera. Por isso o padrão é `--fps 15`, e o pedido seguinte sai quando
ainda restam `--gatilho` passos na fila, para a resposta chegar antes de a fila esvaziar.

Como as ações são pose ABSOLUTA (e não incremento), ficar sem fila é seguro: o robô repete o
último alvo e para de pé, em vez de derivar.

    python roda_unifolm_mujoco.py --seco          # sobe tudo e NÃO move o robô
    python roda_unifolm_mujoco.py                 # move
"""
import argparse
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import json_numpy
import numpy as np
import requests

# A raiz do `lerobot-ext` e PROCURADA, e nao contada em `parent.parent`: em 21/09 este
# arquivo saiu de `lerobot-ext/unifolm/` para `lerobot-ext/pontes/unifolm-vla/`, e a
# contagem fixa passou a apontar para `pontes/`, quebrando o caminho do URDF sem que
# nenhuma busca por texto pudesse ver — o caminho era CALCULADO. Subir ate achar
# `assets/g1` sobrevive a proxima mudanca de pasta.
def _acha_raiz(inicio):
    for d in [inicio, *inicio.parents]:
        if (d / "assets" / "g1").is_dir():
            return d
    return inicio.parent.parent


RAIZ = _acha_raiz(Path(__file__).resolve().parent)
sys.path.insert(0, str(RAIZ))

# Pose em que os 302 episódios do copo começam (14 juntas de braço + yaw do tronco), a mesma
# constante do `init_lerobot_inference_v3.py`. Repetida aqui de propósito: importar aquele
# módulo arrastaria torch e o LeRobot inteiro só para pegar uma lista de 15 números.
POSE_PARTIDA_COPO = [0.958, 0.683, 1.476, -0.554, -1.501, 0.505, -0.225,
                     -0.941, -0.628, -0.313, 0.388, 0.232, 0.655, 0.337, -0.036]

NOMES_BRACOS = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q", "kLeftElbow.q",
    "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q", "kRightElbow.q",
    "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
]
NOME_CINTURA = "kWaistYaw.q"
NOMES_MAO_ESQ = ["left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q",
                 "left_hand_thumb_2_joint.q", "left_hand_middle_0_joint.q",
                 "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
                 "left_hand_index_1_joint.q"]
NOMES_MAO_DIR = ["right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q",
                 "right_hand_thumb_2_joint.q", "right_hand_index_0_joint.q",
                 "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
                 "right_hand_middle_1_joint.q"]
NOMES_29 = NOMES_BRACOS + [NOME_CINTURA] + NOMES_MAO_ESQ + NOMES_MAO_DIR

# ── Garra: 1 número deles, 7 juntas nossas ──────────────────────────────────────────────────
# O `dataset_statistics.json` põe as dimensões 18 e 19 entre 0,02 e 4,50, e na primeira resposta
# real o modelo manteve 4,49 enquanto descia a mão — logo 4,5 é ABERTA e 0 é FECHADA.
# A Dex3 tem 7 juntas por mão e limites ASSIMÉTRICOS entre esquerda e direita (`g1_utils.py`),
# por isso as duas posturas de fechar têm sinais trocados. A ordem das juntas também difere:
# esquerda é polegar,polegar,polegar,médio,médio,indicador,indicador; direita troca médio e
# indicador de lugar — mas como o gesto é o mesmo nos quatro dedos, o vetor não muda por isso.
GARRA_ABERTA = 4.5
DEDOS_FECHADOS = {
    "esq": np.array([0.0,  0.5,  1.0, -1.1, -1.2, -1.1, -1.2]),
    "dir": np.array([0.0, -0.5, -1.0,  1.1,  1.2,  1.1,  1.2]),
}


def r6_para_rot(r6):
    """As duas primeiras COLUNAS de volta a uma matriz de rotação (Gram-Schmidt).

    O par que chega do modelo quase nunca é perfeitamente ortonormal; sem reortogonalizar, a
    matriz não é rotação e a IK recebe um alvo que não existe.
    """
    a1, a2 = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    c1 = a1 / (np.linalg.norm(a1) or 1.0)
    a2 = a2 - np.dot(c1, a2) * c1
    c2 = a2 / (np.linalg.norm(a2) or 1.0)
    return np.column_stack([c1, c2, np.cross(c1, c2)])


def rot_para_r6(R):
    """R6 = as duas primeiras COLUNAS empilhadas (convenção `rotmat_to_rot6d` deles)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def dedos_para_garra(q7, lado):
    """7 ângulos de dedo -> o 1 número de garra que o modelo entende."""
    ref = DEDOS_FECHADOS[lado]
    uteis = np.abs(ref) > 1e-6
    fechamento = float(np.clip(np.mean(np.asarray(q7)[uteis] / ref[uteis]), 0.0, 1.0))
    return GARRA_ABERTA * (1.0 - fechamento)


def garra_para_dedos(valor, lado):
    """O 1 número de garra -> os 7 ângulos de dedo."""
    fechamento = float(np.clip(1.0 - valor / GARRA_ABERTA, 0.0, 1.0))
    return DEDOS_FECHADOS[lado] * fechamento


class Cinematica:
    """FK e IK no MESMO modelo reduzido — ver a decisão 1 no cabeçalho."""

    def __init__(self):
        import pinocchio as pin
        from robot.unitree_g1.robot_control import g1_arm_ik as _ik
        # O `ASSETS_DIR` dele aponta para `robot/unitree_g1/assets`, que NESTA copia do
        # repositorio nao existe - o URDF mora em `lerobot-ext/assets/g1`. O mesmo desvio ja
        # estava no `confere_estado_vla.py`, la so para pegar a constante; aqui o solver inteiro
        # precisa dele, entao o caminho e corrigido no modulo antes de construir.
        if not (_ik.ASSETS_DIR / "g1" / "g1_body29_hand14.urdf").exists():
            _ik.ASSETS_DIR = RAIZ / "assets"
        self.pin = pin
        self.ik = _ik.G1_29_ArmIK()
        self.modelo = self.ik.reduced_robot.model
        # `data` NOVO: o `reduced_robot.data` nasceu ANTES de o `G1_29_ArmIK` acrescentar os
        # frames `L_ee`/`R_ee`, e o `oMf` dele não tem espaço para eles.
        self.dados = self.modelo.createData()

    def fk(self, q14):
        q = np.asarray(q14, dtype=float)
        self.pin.forwardKinematics(self.modelo, self.dados, q)
        self.pin.updateFramePlacements(self.modelo, self.dados)
        saida = []
        for ident in (self.ik.L_hand_id, self.ik.R_hand_id):
            T = self.dados.oMf[ident]
            saida.append((np.array(T.translation, float), np.array(T.rotation, float)))
        return saida

    def resolve(self, alvo_esq, alvo_dir, q14_atual):
        """Duas poses (pos, R) -> 14 juntas de braço."""
        def homog(par):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = par[1], par[0]
            return T
        q, _ = self.ik.solve_ik(homog(alvo_esq), homog(alvo_dir), np.asarray(q14_atual, float))
        return np.asarray(q, float)


def monta_estado(cin, q14, cintura_yaw, mao_esq, mao_dir):
    """As 23 dimensões do `EE_R6_G1` a partir do que o robô está fazendo agora."""
    (p_e, R_e), (p_d, R_d) = cin.fk(q14)
    return np.concatenate([
        p_e, rot_para_r6(R_e),
        p_d, rot_para_r6(R_d),
        [dedos_para_garra(mao_esq, "esq"), dedos_para_garra(mao_dir, "dir")],
        [0.0, 0.0, float(cintura_yaw)],          # roll e pitch de cintura: não temos
    ]).astype(np.float32)


def consulta(sessao, url, cabeca, punho_esq, punho_dir, estado, instrucao, tarefa, espera):
    """Um POST no /act. Devolve (25, 23).

    Caminho `encoded` de propósito: o `json_numpy.patch()` mexe no `json` do processo inteiro e
    já quebrou o import do TensorFlow uma vez (18/09). Aqui a codificação é explícita e local.
    """
    obs = {
        "full_image": cabeca,
        # A ordem importa: o servidor junta todas as `full_image` e DEPOIS as chaves que contêm
        # "wrist", na ordem do dicionário. Esquerda antes de direita.
        "left_wrist_image": punho_esq,
        "right_wrist_image": punho_dir,
        "instruction": instrucao,
        "state": np.asarray(estado, dtype=np.float32),
        "task_name": tarefa,
    }
    corpo = {"encoded": json_numpy.dumps({"observations": [obs]})}
    r = sessao.post(url, json=corpo, timeout=espera)
    r.raise_for_status()
    return np.asarray(json_numpy.loads(r.json()), dtype=float)


def prepara_imagem(img):
    """uint8 HxWx3 contíguo — é o que o `check_image_format` do servidor exige."""
    if img is None:
        return None
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def leva_a_pose_de_partida(robot, segundos=3.0, fps=30):
    obs = robot.get_observation() or {}
    inicio = [float(obs.get(n, 0.0)) for n in NOMES_29]
    fim = list(inicio)
    fim[:15] = POSE_PARTIDA_COPO
    passos = max(1, int(segundos * fps))
    for k in range(1, passos + 1):
        a = k / passos
        robot.send_action({n: float((1 - a) * i + a * f)
                           for n, i, f in zip(NOMES_29, inicio, fim)})
        time.sleep(1.0 / fps)
    depois = robot.get_observation() or {}
    erro = max(abs(float(depois.get(n, 0.0)) - f) for n, f in zip(NOMES_29, fim))
    print(f"🎯 pose de partida aplicada | maior erro de junta: {erro:.3f} rad", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--servidor", default="http://127.0.0.1:8777/act")
    p.add_argument("--tarefa", default="pick up the red cup on the table",
                   help="o texto que condiciona o modelo")
    p.add_argument("--chave-norm", default="g1_stack_block",
                   help="qual das 12 tarefas de G1 fornece a normalização (`task_name`)")
    p.add_argument("--fps", type=float, default=15.0, help="ver A CONTA DO TEMPO no cabeçalho")
    p.add_argument("--gatilho", type=int, default=10,
                   help="pede o próximo plano quando restam N passos na fila")
    p.add_argument("--salto-max", type=float, default=0.03,
                   help="metros que a mão pode andar por passo; corta alvo absurdo antes da IK")
    p.add_argument("--cintura", action="store_true", help="deixa o yaw do tronco obedecer")
    p.add_argument("--dois-bracos", action="store_true",
                   help="solta também o braço esquerdo (ver decisão 3: a imagem dele é falsa)")
    p.add_argument("--sem-pose-inicial", action="store_true")
    p.add_argument("--seco", action="store_true",
                   help="roda o laço todo e NÃO manda ação nenhuma ao robô")
    p.add_argument("--passos", type=int, default=0, help="para depois de N passos (0 = sem fim)")
    args = p.parse_args()

    print("⏳ montando FK/IK (Pinocchio + CasADi)…", flush=True)
    cin = Cinematica()

    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print("⏳ subindo o MuJoCo e conectando…", flush=True)
    robot = UnitreeG1Dex3(UnitreeG1Dex3Config(
        robot_ip="10.9.8.73", control_mode="upper_body", use_waist_yaw=True,
        use_depth_camera=False, use_wrist_camera=True, is_simulation=True,
    ))
    robot.connect()
    for cam in robot.cameras.values():
        if hasattr(cam, "timeout_ms"):
            cam.timeout_ms = 800
    print("✅ robô conectado", flush=True)

    if not args.sem_pose_inicial and not args.seco:
        leva_a_pose_de_partida(robot)

    sessao = requests.Session()
    piscina = ThreadPoolExecutor(max_workers=1)
    pedido, fila, ultimo_alvo = None, deque(), None
    congelado_esq = None                 # pose da mão esquerda no arranque, quando ela fica parada
    n, t_infer, chunks = 0, 0.0, 0

    print(f"\n🚀 {'ENSAIO SECO' if args.seco else 'INFERÊNCIA ATIVA'} — UnifoLM-VLA-0")
    print(f'   tarefa: "{args.tarefa}"  |  normalização: {args.chave_norm}')
    print(f"   {args.fps:g} fps  |  braço {'esquerdo e direito' if args.dois_bracos else 'DIREITO só'}"
          f"  |  cintura {'solta' if args.cintura else 'travada'}")
    print("   Ctrl+C para parar.\n", flush=True)

    try:
        while args.passos == 0 or n < args.passos:
            t0 = time.perf_counter()
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"⚠️  timeout de câmera: {e}", flush=True)
                continue
            if not obs:
                continue

            q14 = np.array([float(obs.get(k, 0.0)) for k in NOMES_BRACOS])
            cintura = float(obs.get(NOME_CINTURA, 0.0))
            mao_e = [float(obs.get(k, 0.0)) for k in NOMES_MAO_ESQ]
            mao_d = [float(obs.get(k, 0.0)) for k in NOMES_MAO_DIR]
            (pe, Re), (pd, Rd) = cin.fk(q14)
            if congelado_esq is None:
                congelado_esq = (pe.copy(), Re.copy())
            if ultimo_alvo is None:
                ultimo_alvo = {"esq": (pe.copy(), Re.copy()), "dir": (pd.copy(), Rd.copy()),
                               "garra": (dedos_para_garra(mao_e, "esq"),
                                         dedos_para_garra(mao_d, "dir")),
                               "cintura": cintura}

            # ── pede o próximo plano, sem parar o laço ────────────────────────────────────
            if pedido is None and len(fila) <= args.gatilho:
                cabeca = prepara_imagem(obs.get("head_camera"))
                punho = prepara_imagem(obs.get("right_wrist_camera"))
                if cabeca is not None and punho is not None:
                    estado = monta_estado(cin, q14, cintura, mao_e, mao_d)
                    pedido = piscina.submit(consulta, sessao, args.servidor, cabeca,
                                            punho, punho, estado, args.tarefa,
                                            args.chave_norm, 60.0)
                    pedido._t = time.perf_counter()
            if pedido is not None and pedido.done():
                try:
                    plano = pedido.result()
                    t_infer = time.perf_counter() - pedido._t
                    fila.clear()
                    fila.extend(plano)
                    chunks += 1
                    print(f"📦 plano {chunks}: {plano.shape} em {t_infer:.2f} s", flush=True)
                except Exception as e:                                       # noqa: BLE001
                    print(f"⚠️  o servidor falhou: {type(e).__name__}: {e}", flush=True)
                pedido = None

            # ── executa um passo ──────────────────────────────────────────────────────────
            if fila:
                a = np.asarray(fila.popleft(), float)
                for lado, base in (("esq", 0), ("dir", 9)):
                    pos = a[base:base + 3]
                    ant = ultimo_alvo[lado][0]
                    pos = ant + np.clip(pos - ant, -args.salto_max, args.salto_max)
                    ultimo_alvo[lado] = (pos, r6_para_rot(a[base + 3:base + 9]))
                ultimo_alvo["garra"] = (float(a[18]), float(a[19]))
                ultimo_alvo["cintura"] = float(a[22])

            alvo_esq = ultimo_alvo["esq"] if args.dois_bracos else congelado_esq
            q_novo = cin.resolve(alvo_esq, ultimo_alvo["dir"], q14)
            if not args.dois_bracos:
                q_novo[:7] = POSE_PARTIDA_COPO[:7]      # o braço esquerdo não se mexe

            acao = {k: float(v) for k, v in zip(NOMES_BRACOS, q_novo)}
            acao[NOME_CINTURA] = float(ultimo_alvo["cintura"]) if args.cintura else cintura
            for nome, v in zip(NOMES_MAO_ESQ, garra_para_dedos(ultimo_alvo["garra"][0], "esq")):
                acao[nome] = float(v)
            for nome, v in zip(NOMES_MAO_DIR, garra_para_dedos(ultimo_alvo["garra"][1], "dir")):
                acao[nome] = float(v)
            if not args.seco:
                robot.send_action(acao)

            if n % 15 == 0:
                pp = ultimo_alvo["dir"][0]
                print(f"[{n:5d}] fila {len(fila):2d} | mão dir alvo "
                      f"({pp[0]:+.3f} {pp[1]:+.3f} {pp[2]:+.3f}) | real "
                      f"({pd[0]:+.3f} {pd[1]:+.3f} {pd[2]:+.3f}) | garra "
                      f"{ultimo_alvo['garra'][1]:.2f} | infer {t_infer:.2f}s", flush=True)
            n += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\n🛑 parando…", flush=True)
    finally:
        piscina.shutdown(wait=False)
        robot.disconnect()
        print("✅ encerrado.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
