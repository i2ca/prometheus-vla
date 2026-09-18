#!/usr/bin/env python
"""Política treinada NO simulador da Unitree (IsaacLab) fechando o laço com ele, na PGX.

Diferença para o `pgx/roda_pi05_g1_sim.py`: lá a política vinha de dado REAL e via render do
MuJoCo — medido em 16/09, ela emitia saída constante, presa num ponto fixo, porque a imagem não
se parecia com nada do treino. Aqui a política foi treinada com imagens DESTE simulador
(`carroll511/g1_pick_redblock_dex1_sim`, gravado no `Isaac-PickPlace-RedBlock-G129-Dex1-Joint`),
então a pergunta "o modelo faz a tarefa?" enfim é a pergunta certa.

Como falar com o simulador deles — três coisas que não são óbvias:
  * o DDS do sim fala no **domínio 1**, não no 0 (está escrito no `sim_main.py` deles). Um
    cliente no domínio 0 não recebe nada e não dá erro nenhum;
  * as câmeras NÃO vêm por DDS: são três publicadores ZMQ (55555 cabeça, 55556 punho esquerdo,
    55557 punho direito) mandando JPEG cru. O `teleimager` deles tem um cliente, mas ele puxa
    `opencv-python` não-headless, que briga com o `opencv-python-headless` do nosso env — por
    isso aqui o SUB é escrito à mão, em ~20 linhas;
  * a garra Dex1 tem tópico próprio (`rt/dex1/{left,right}/{state,cmd}`, IDL `unitree_go`) e
    trabalha em unidade de GARRA: 0 fechada, 5.4 aberta. O `rt/lowcmd` só leva os braços.

O simulador ignora kp/kd do `lowcmd` — o `DDSActionProvider` deles lê só as posições das 29
juntas e recorta os braços. Então basta preencher `motor_cmd[j].q`.

    python pgx/roda_politica_g1_isaaclab.py                      # π0.5 de 30k passos
    python pgx/roda_politica_g1_isaaclab.py --politica binabik-ai/act_PickPlaceRedBlock
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

POLITICA = "RooibosT/Sim_pi05_expert_only_absolute_s30k"
# MEDIDO 17/09: o `meta/tasks.parquet` do dataset desta tarefa tem DUAS frases, e esta não é
# nenhuma das duas. As 108 demonstrações aparecem rotuladas duas vezes (216 episódios), com
# "Pick up the red cup on the table." e "Place the red wooden block into the yellow box.".
# O π0.5 é condicionado a linguagem: mandar uma frase que ele nunca viu é entrada fora da
# distribuição. Toda a varredura de π0.5 de 16/09 rodou com a frase errada.
TAREFA = "Place the red wooden block into the yellow box."
# Nome da câmera na política -> porta ZMQ do servidor de imagem do simulador.
CAMERAS = {"cam_left_high": 55555, "cam_left_wrist": 55556, "cam_right_wrist": 55557}
DOMINIO = 1
JUNTAS_BRACOS = list(range(15, 29))  # as 14 juntas dos braços, na ordem do dataset
GARRA_ABERTA, GARRA_FECHADA = 5.4, 0.0
# Pose em que os episódios do dataset COMEÇAM (média dos 20 primeiros de
# `unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim`, desvio ≤ 0,16 rad em cada junta).
# Medido em 17/09: o `rt/reset_pose/cmd` recoloca o CUBO, mas deixa o robô onde estava — 10 das
# 16 dims partiam de outro lugar, até 1,16 rad no punho direito. Clonagem de comportamento
# extrapola desde o primeiro quadro quando isso acontece.
POSE_PARTIDA = [0.106, 0.023, 0.081, -0.002, -0.062, -0.424, -0.141,
                0.280, -0.006, 0.017, -0.288, -0.146, -0.179, -0.012]
GARRAS_PARTIDA = (0.616, 0.616)


class CameraZMQ(threading.Thread):
    """SUB de um publicador de imagem do simulador. Guarda só o último quadro."""

    def __init__(self, nome: str, porta: int, host: str):
        super().__init__(daemon=True)
        self.nome, self.porta, self.host = nome, porta, host
        self._quadro = None
        self._trava = threading.Lock()
        self._rodando = True
        self.recebidos = 0

    def run(self) -> None:
        import cv2
        import zmq

        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 1)  # só o mais recente interessa: o laço é de tempo real
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{self.host}:{self.porta}")
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        poller = zmq.Poller()
        poller.register(s, zmq.POLLIN)
        while self._rodando:
            if s in dict(poller.poll(timeout=100)):
                bgr = cv2.imdecode(np.frombuffer(s.recv(), np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                with self._trava:
                    self._quadro = bgr[:, :, ::-1].copy()  # BGR do JPEG -> RGB da política
                    self.recebidos += 1
        s.close()

    def le(self):
        with self._trava:
            return None if self._quadro is None else self._quadro.copy()

    def para(self) -> None:
        self._rodando = False


def abre_dds():
    """Assinaturas de estado e publicadores de comando, no domínio do simulador."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(DOMINIO)
    estado = {"corpo": None, "garra_e": None, "garra_d": None}

    sub_corpo = ChannelSubscriber("rt/lowstate", hg_LowState)
    sub_corpo.Init(lambda m: estado.__setitem__("corpo", m), 10)
    sub_ge = ChannelSubscriber("rt/dex1/left/state", MotorStates_)
    sub_ge.Init(lambda m: estado.__setitem__("garra_e", m), 10)
    sub_gd = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
    sub_gd.Init(lambda m: estado.__setitem__("garra_d", m), 10)

    pub_corpo = ChannelPublisher("rt/lowcmd", hg_LowCmd)
    pub_corpo.Init()
    pub_ge = ChannelPublisher("rt/dex1/left/cmd", MotorCmds_)
    pub_ge.Init()
    pub_gd = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
    pub_gd.Init()
    # `rt/reset_pose/cmd` recoloca a cena (categoria 1), que é como mandar "de novo" sem
    # descarregar a política — os 7 GB do checkpoint continuam na GPU.
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    pub_reset = ChannelPublisher("rt/reset_pose/cmd", String_)
    pub_reset.Init()

    def manda_garra(pub, valor: float) -> None:
        c = unitree_go_msg_dds__MotorCmd_()
        c.q = float(np.clip(valor, GARRA_FECHADA, GARRA_ABERTA))
        pub.Write(MotorCmds_(cmds=[c]))

    return (estado, pub_corpo, unitree_hg_msg_dds__LowCmd_(), CRC(),
            lambda v: manda_garra(pub_ge, v), lambda v: manda_garra(pub_gd, v),
            lambda: pub_reset.Write(String_(data="1")),
            [sub_corpo, sub_ge, sub_gd])


def redimensiona_com_borda(rgb: np.ndarray, lado: int) -> np.ndarray:
    """Escala preservando a proporção e centraliza numa tela `lado`x`lado` preta.

    É o `resize_with_pad` do openpi: 640x480 vira 224x168 no meio de um quadrado de 224, com
    28 linhas pretas em cima e embaixo. Esticar para 224x224 sem preservar a proporção mudaria
    a geometria que a política aprendeu.
    """
    import cv2

    alt, larg = rgb.shape[:2]
    escala = min(lado / alt, lado / larg)
    nova = (max(1, int(round(larg * escala))), max(1, int(round(alt * escala))))
    peq = cv2.resize(rgb, nova, interpolation=cv2.INTER_LINEAR)
    tela = np.zeros((lado, lado, 3), dtype=rgb.dtype)
    y, x = (lado - nova[1]) // 2, (lado - nova[0]) // 2
    tela[y:y + nova[1], x:x + nova[0]] = peq
    return tela


def leva_a_partida(msg, crc, pub, garra_e, garra_d, estado, segundos: float = 3.0) -> float:
    """Comanda a pose de partida do dataset e espera assentar. Devolve o maior erro medido."""
    for k, j in enumerate(JUNTAS_BRACOS):
        msg.motor_cmd[j].q = float(POSE_PARTIDA[k])
    fim = time.time() + segundos
    while time.time() < fim:
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        garra_e(GARRAS_PARTIDA[0])
        garra_d(GARRAS_PARTIDA[1])
        time.sleep(0.02)
    ls = estado["corpo"]
    q = np.array([ls.motor_state[j].q for j in JUNTAS_BRACOS], dtype=np.float32)
    return float(np.abs(q - np.array(POSE_PARTIDA, dtype=np.float32)).max())


def carrega_politica(repo: str, device: str, passos_acao=None, ensemble=None):
    """Carrega pi05 ou act pelo que o próprio checkpoint declara."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.device = device
    if passos_acao is not None:
        cfg.n_action_steps = int(passos_acao)
    if ensemble is not None:
        # No ACT do LeRobot, `temporal_ensemble_coeff` só vale com n_action_steps=1 — ele guarda
        # as previsões sobrepostas e faz a média exponencial. É a receita do artigo do ACT.
        cfg.temporal_ensemble_coeff = float(ensemble)
        cfg.n_action_steps = 1
    if cfg.type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Classe
    elif cfg.type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as Classe
    elif cfg.type == "smolvla":
        # SmolVLA tem ~450 M de parâmetros contra os 4,14 B da π0.5 — 10x menor. Medido em
        # 17/09: com a π0.5 ligada o simulador cai de 13,8 Hz para 4,8 Hz; o tamanho do modelo
        # é o que decide a fluidez, não a cena.
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Classe
    else:
        raise SystemExit(f"❌ tipo de política '{cfg.type}' não suportado aqui "
                         f"(pi05, act ou smolvla)")
    politica = Classe.from_pretrained(repo, config=cfg)
    politica.eval().to(device)
    pre, pos = make_pre_post_processors(policy_cfg=cfg, pretrained_path=repo)
    torch.backends.cuda.matmul.allow_tf32 = True
    entradas = {k: tuple(v.shape) for k, v in (cfg.input_features or {}).items()}
    print(f"✅ política carregada: {cfg.type} | entradas {entradas} | chunk {cfg.chunk_size}"
          f" | passos de ação {cfg.n_action_steps}"
          f"{' | ensemble ' + str(getattr(cfg, 'temporal_ensemble_coeff', None)) if getattr(cfg, 'temporal_ensemble_coeff', None) else ''}")
    return politica, pre, pos, cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--politica", default=POLITICA)
    p.add_argument("--task", default=TAREFA, help="instrução em texto (só a π0.5 usa)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=30.0, help="o dataset foi gravado a 30 Hz")
    p.add_argument("--passos", type=int, default=0, help="0 = laço infinito, pare com Ctrl-C")
    p.add_argument("--saida", default=str(Path.home() / "g1_isaaclab"))
    p.add_argument("--espera", type=float, default=20.0, help="s para o simulador aparecer")
    p.add_argument("--pose-inicial", action="store_true",
                   help="antes de soltar a política (e depois de cada `reset`), leva os braços e as "
                        "garras à pose em que os episódios do dataset começam. Sem isto o robô "
                        "parte de onde o último comando o deixou, que a política nunca viu")
    p.add_argument("--redimensiona", type=int, default=None, metavar="LADO",
                   help="redimensiona a imagem para LADOxLADO preservando a proporção e "
                        "preenchendo o resto com preto. O `binabik-ai/act_PickPlaceRedBlock` faz "
                        "isso DENTRO do checkpoint (224); ao carregá-lo como ACT puro, o passo "
                        "sai do pré-processador e tem de ser feito aqui")
    p.add_argument("--passos-acao", type=int, default=None, metavar="N",
                   help="quantas ações do chunk executar antes de replanejar. O checkpoint vem com "
                        "50 (1,7 s cego a 30 Hz); com 1 ele replaneja a cada quadro")
    p.add_argument("--ensemble", type=float, default=None, metavar="COEF",
                   help="ensemble temporal do ACT (ex.: 0.01): mistura as previsões sobrepostas "
                        "de vários chunks em vez de trocar de plano em degrau. Exige --passos-acao=1")
    p.add_argument("--comando", default=str(Path.home() / "comando_g1.txt"),
                   help="arquivo lido a cada segundo: escreva uma frase para trocar a instrução "
                        "em tempo de execução, ou a palavra 'reset' para recolocar o cubo. "
                        "Nenhum dos dois recarrega o modelo")
    args = p.parse_args()

    import torch

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    print(f">>> conectando nas câmeras ZMQ de {args.host}: {sorted(CAMERAS.values())}")
    cams = {}
    for nome, porta in CAMERAS.items():
        c = CameraZMQ(nome, porta, args.host)
        c.start()
        cams[nome] = c

    print(f">>> assinando o DDS no domínio {DOMINIO}")
    estado, pub, msg, crc, garra_e, garra_d, reset_cena, _subs = abre_dds()

    limite = time.time() + args.espera
    while time.time() < limite:
        if estado["corpo"] is not None and all(c.le() is not None for c in cams.values()):
            break
        time.sleep(0.2)
    if estado["corpo"] is None:
        raise SystemExit("❌ nenhum `rt/lowstate` — o simulador está rodando? (domínio 1)")
    faltando = [n for n, c in cams.items() if c.le() is None]
    if faltando:
        raise SystemExit(f"❌ sem imagem de {faltando} — confira o `--enable_cameras` do sim")
    print(f"    estado e {len(cams)} câmeras chegando")

    politica, pre, pos, cfg = carrega_politica(args.politica, args.device,
                                              args.passos_acao, args.ensemble)
    usa_texto = cfg.type == "pi05"

    # `mode_machine` vem do próprio simulador; o resto do lowcmd começa na pose atual, para o
    # primeiro passo não dar um tranco.
    ls = estado["corpo"]
    msg.mode_pr = 0
    msg.mode_machine = ls.mode_machine
    for j in range(29):
        msg.motor_cmd[j].mode = 1
        msg.motor_cmd[j].q = ls.motor_state[j].q
        msg.motor_cmd[j].dq = 0.0
        msg.motor_cmd[j].tau = 0.0

    dt = 1.0 / args.fps
    infinito = args.passos <= 0
    linhas = deque(maxlen=200_000)
    ms_politica = 0.0
    print(f">>> {'laço infinito' if infinito else str(args.passos) + ' passos'} a {args.fps:g} Hz"
          f"{' | instrução: ' + repr(args.task) if usa_texto else ' (ACT: sem texto)'}")

    import signal

    parar = {"agora": False}
    for sinal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sinal, lambda *_: parar.__setitem__("agora", True))

    if args.pose_inicial:
        erro = leva_a_partida(msg, crc, pub, garra_e, garra_d, estado)
        print(f">>> pose de partida do dataset aplicada | maior erro de junta: {erro:.3f} rad")

    arq_comando = Path(args.comando).expanduser()
    arq_comando.write_text(args.task + "\n")
    print(f">>> comando em tempo real: escreva no arquivo {arq_comando}")
    print(f"    echo 'pick up the red cube' > {arq_comando}")
    print(f"    echo reset > {arq_comando}      # recoloca a cena, sem recarregar o modelo")
    ultimo_comando = args.task

    t_ini = time.perf_counter()
    feitos = 0
    passo = 0
    while not parar["agora"] and (infinito or passo < args.passos):
        alvo = t_ini + passo * dt
        # Uma leitura por segundo: é um arquivo de poucos bytes, não pesa no laço de 30 Hz.
        if passo % int(args.fps) == 0:
            try:
                texto = arq_comando.read_text().strip()
            except OSError:
                texto = ultimo_comando
            if texto and texto != ultimo_comando:
                if texto.lower() == "reset":
                    reset_cena()
                    time.sleep(1.0)
                    if args.pose_inicial:
                        erro = leva_a_partida(msg, crc, pub, garra_e, garra_d, estado)
                        print(f"    [{passo}] pose de partida reaplicada (erro {erro:.3f} rad)", flush=True)
                    print(f"    [{passo}] cena recolocada; instrução segue {args.task!r}", flush=True)
                    arq_comando.write_text(args.task + "\n")
                    texto = args.task
                else:
                    args.task = texto
                    print(f"    [{passo}] nova instrução: {texto!r}", flush=True)
                ultimo_comando = texto
        ls = estado["corpo"]
        ge, gd = estado["garra_e"], estado["garra_d"]
        # As 16 dims do dataset: 14 juntas dos braços + garra esquerda + garra direita, esta
        # última já em unidade de garra (o simulador converte ao publicar o estado).
        q = np.array([ls.motor_state[j].q for j in JUNTAS_BRACOS], dtype=np.float32)
        qe = float(ge.states[0].q) if ge is not None and len(ge.states) else GARRA_ABERTA
        qd = float(gd.states[0].q) if gd is not None and len(gd.states) else GARRA_ABERTA
        estado_16 = np.concatenate([q, [qe, qd]]).astype(np.float32)

        bruto = {"observation.state": torch.from_numpy(estado_16)}
        for nome, cam in cams.items():
            quadro = cam.le()
            if quadro is None:
                raise SystemExit(f"❌ a câmera {nome} parou de publicar")
            if args.redimensiona:
                quadro = redimensiona_com_borda(quadro, args.redimensiona)
            # float32 em [0,1], que é como o LeRobot entrega imagem de dataset. Mandar uint8
            # 0-255 estoura o normalizador do ACT (MEAN_STD sobre tensor uint8) e, pior, passa
            # CALADO na π0.5, cuja normalização visual é IDENTITY: ela receberia a imagem 255×
            # mais clara do que viu no treino. Medido em 16/09.
            bruto[f"observation.images.{nome}"] = (
                torch.from_numpy(quadro).permute(2, 0, 1).to(torch.float32) / 255.0)
        if usa_texto:
            bruto["task"] = args.task

        t0 = time.perf_counter()
        with torch.inference_mode():
            acao = pos(politica.select_action(pre(bruto)))
        ms_politica += (time.perf_counter() - t0) * 1000
        acao = np.asarray(acao, dtype=np.float32).reshape(-1)
        if acao.shape[0] != 16:
            raise SystemExit(f"❌ ação com {acao.shape[0]} dims, esperava 16 (14 braços + 2 garras)")

        for i, j in enumerate(JUNTAS_BRACOS):
            msg.motor_cmd[j].q = float(np.clip(acao[i], -np.pi, np.pi))
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        garra_e(acao[14])
        garra_d(acao[15])
        linhas.append([passo, *[f"{v:.4f}" for v in acao]])

        if passo % 60 == 0:
            print(f"    {passo}/{'∞' if infinito else args.passos} | "
                  f"{ms_politica / max(passo + 1, 1):.0f} ms por chamada | "
                  f"garras cmd {acao[14]:.2f}/{acao[15]:.2f} (medido {qe:.2f}/{qd:.2f})", flush=True)

        atraso = alvo + dt - time.perf_counter()
        if atraso > 0:
            time.sleep(atraso)
        passo += 1
        feitos = passo

    dur = time.perf_counter() - t_ini
    print(f">>> {feitos} passos em {dur / 60:.1f} min ({ms_politica / max(feitos, 1):.0f} ms por chamada)")
    arq = saida / "acoes.csv"
    with open(arq, "w", newline="") as f:
        w = csv.writer(f)
        nomes = ["LomboPitch", "LomboRoll", "LomboYaw", "Lcotovelo", "LpunhoRoll", "LpunhoPitch", "LpunhoYaw",
                 "RomboPitch", "RomboRoll", "RomboYaw", "Rcotovelo", "RpunhoRoll", "RpunhoPitch", "RpunhoYaw",
                 "garra_esq", "garra_dir"]
        w.writerow(["passo", *nomes])
        w.writerows(linhas)
    print(f">>> ações em {arq}")
    for c in cams.values():
        c.para()
    return 0


if __name__ == "__main__":
    sys.exit(main())
