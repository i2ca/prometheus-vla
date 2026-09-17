#!/usr/bin/env python
"""π0.5 de G1 (`xiaopeng-wu/pi05_unitree_g1`) no simulador MuJoCo do LeRobot, na PGX.

Teste de ponta a ponta com uma política PRONTA, sem robô e sem o nosso dataset: responde se a
montagem simulador → câmera → política → ação fecha o laço na PGX (aarch64 + CUDA 13).

O que este teste é, e o que ele NÃO é:
  * o checkpoint foi treinado em 550 episódios de `nepyope/unitree_box_move_blue_full`, com a
    instrução "move blue box back and forth between tables". É a única tarefa que ele conhece;
  * a ação dele tem 18 dims: as 14 juntas dos braços (índices 15..28 do G1) e 4 eixos do
    controle remoto (`remote.lx/ly/rx/ry`), que comandam o ANDAR. Sem controlador de locomoção
    no laço, os 4 últimos são só registrados, não viram movimento;
  * a entrada visual dele é UMA câmera, `global_view`, em terceira pessoa. Ela é panorâmica: o
    robô ocupa ~30x60 pixels em 640x480, e por isso o movimento "some" na tela. Para PODER VER,
    a `head_camera` é gravada em paralelo — ela não entra na política, só no vídeo e no painel;
  * a cena do simulador está VAZIA: não há as duas mesas nem a caixa azul do treino. O modelo
    roda fora da distribuição dele, então o movimento não tem como cumprir a tarefa.

Por que o comando vai por DDS, e não pela classe `UnitreeG1`:
  o `UnitreeG1.connect()` em modo simulação cria o env ELE MESMO, com as câmeras padrão, e
  usaria o nosso esquema de 29 dims. Aqui o env é criado direto do `env.py` do Hub (única forma
  de escolher a câmera) e o `rt/lowcmd` é publicado à mão, exatamente como a classe faz. O
  `init_channel` do próprio env já inicializa o canal DDS no domínio 0 em `lo`. Que esse
  caminho move o robô foi medido com o `pgx/testa_dds_sim.py`: senoide de ±0,8 rad comandada,
  1,588 rad de amplitude medida.

    python pgx/roda_pi05_g1_sim.py --passos 3600 --tempo-real --v-web=8088
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")  # antes de qualquer import pesado (BLAS/ipopt)
# O backend do GL é escolhido no import do mujoco, muito antes do argparse: o EGL renderiza
# offscreen mas NÃO abre janela, e o GLFW abre janela e também renderiza offscreen (medido).
# Por isso `--na-tela` é lido aqui, na unha, direto do argv.
_NA_TELA = "--na-tela" in sys.argv
os.environ.setdefault("MUJOCO_GL", "glfw" if _NA_TELA else "egl")

import argparse
import csv
import importlib.util
import itertools
import signal
import socket
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_POLITICA = "xiaopeng-wu/pi05_unitree_g1"
REPO_ENV = "lerobot/unitree-g1-mujoco"
TAREFA_TREINO = "move blue box back and forth between tables"
CAMERA_POLITICA = "global_view"
CAMERA_VIDEO = "head_camera"
# As 14 juntas dos braços, na MESMA ordem em que o dataset gravou a ação:
# left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, depois o lado direito.
JUNTAS_BRACOS = list(range(15, 29))
# Postura de pé do dataset de treino (q50 por junta das 15 primeiras: pernas + tronco). O robô
# do dataset é REAL e fica agachado sob um controlador de locomoção; o do simulador nasce de
# perna reta, pendurado na faixa elástica. Medido em 16/09: com perna reta, 15 das 29 juntas
# caem fora da faixa q01–q99 do treino.
POSTURA_TREINO = [-0.333, -0.023, 0.004, 0.656, -0.339, -0.002,
                  -0.390, 0.011, -0.015, 0.651, -0.277, -0.028,
                  -0.004, 0.000, 0.000]
# Pose INICIAL dos braços, também o q50 do treino. Medido em 16/09: a π0.5 emite ação ≈ estado
# atual mais um passo pequeno, então o braço fica onde nasce. No simulador ele nasce com o
# cotovelo em 1,37 rad, acima do q99 do treino (1,26), e as duas políticas testadas ficaram
# travadas justamente ali — não por decisão delas, mas porque copiaram a nossa pose.
BRACOS_TREINO = [-0.029, 0.228, -0.256, -0.034, 0.107, 0.000, 0.000,
                 -0.203, -0.073, 0.009, 0.227, -0.012, 0.000, 0.000]
# Dims em que o estado do treino é ZERO CONSTANTE (std exatamente 0): tronco roll/pitch e
# punho pitch/yaw dos dois lados. O robô do dataset comanda esses eixos mas não os observa —
# a mesma doença das 66 dims mortas do nosso dataset real. Mandar valor vivo neles é ruído
# puro para a política.
DIMS_MORTAS = [13, 14, 20, 21, 27, 28]


def porta_ocupada(porta: int) -> bool:
    """O publicador de imagem do simulador é um subprocesso `spawn`: se o pai morre, ele SOBREVIVE
    segurando a porta, e a rodada seguinte morre com `TimeoutError` na câmera (aconteceu em
    16/09). Melhor falhar aqui, com o motivo escrito."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", porta)) == 0


def carrega_env(cameras: list[str], porta: int, env_dir: str | None = None, na_tela: bool = False):
    """Importa o `env.py` (do Hub ou de uma cópia) e cria o simulador com as câmeras pedidas.

    `env_dir` é a saída do `pgx/prepara_cena_caixa.sh`: a mesma árvore, com a cena das mesas e
    da caixa azul no lugar da vazia. O `base_sim.py` resolve a cena relativa à pasta do
    ambiente, então basta apontar para a cópia.
    """
    from huggingface_hub import snapshot_download

    raiz = Path(env_dir).expanduser() if env_dir else Path(snapshot_download(REPO_ENV))
    if not (raiz / "env.py").is_file():
        raise SystemExit(f"❌ {raiz} não tem env.py — rode antes: bash pgx/prepara_cena_caixa.sh")
    spec = importlib.util.spec_from_file_location("g1_mujoco_env", raiz / "env.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["g1_mujoco_env"] = modulo
    spec.loader.exec_module(modulo)
    return modulo.make_env(
        n_envs=1,
        use_async_envs=False,
        cameras=cameras,
        publish_images=True,
        camera_port=porta,
        # `onscreen=True` abre o viewer passivo do MuJoCo; o `env.step()` já chama o
        # `update_viewer()` na taxa do `VIEWER_DT` (50 Hz), então não há nada a sincronizar aqui.
        onscreen=na_tela,
    )


def carrega_politica(repo: str, device: str):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    cfg = PreTrainedConfig.from_pretrained(repo)
    if cfg.type != "pi05":
        raise SystemExit(f"❌ o checkpoint é '{cfg.type}', este script é para 'pi05'")
    cfg.device = device
    politica = PI05Policy.from_pretrained(repo, config=cfg)
    politica.eval().to(device)
    pre, pos = make_pre_post_processors(policy_cfg=cfg, pretrained_path=repo)
    torch.backends.cuda.matmul.allow_tf32 = True
    entradas = {k: tuple(v.shape) for k, v in (cfg.input_features or {}).items()}
    # A chave da imagem é a que o checkpoint declara, NÃO o nome da câmera do simulador. As duas
    # eram a mesma coisa por acaso (`global_view`), e isso escondia a pergunta: a `global_view`
    # do dataset é a câmera da CABEÇA de um robô real, em close na caixa — não uma vista em
    # terceira pessoa. Com a chave desacoplada dá para alimentar a `head_camera` do simulador,
    # que é a análoga de verdade.
    chave_img = next(k for k in (cfg.input_features or {}) if k.startswith("observation.images."))
    print(f"✅ política carregada: {cfg.type} | entradas {entradas} | chunk {cfg.chunk_size}")
    return politica, pre, pos, chave_img


def abre_camera(camera: str, porta: int):
    from lerobot.cameras.zmq.camera_zmq import ZMQCamera
    from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig

    cam = ZMQCamera(
        ZMQCameraConfig(
            server_address="127.0.0.1", port=porta, camera_name=camera,
            width=640, height=480, fps=30, warmup_s=5, timeout_ms=10000,
        )
    )
    cam.connect()
    return cam


def abre_dds():
    """Publisher de `rt/lowcmd` e subscriber de `rt/lowstate`, como a classe UnitreeG1 faz."""
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
    from unitree_sdk2py.utils.crc import CRC

    pub = ChannelPublisher("rt/lowcmd", hg_LowCmd)
    pub.Init()
    estado = {"msg": None}
    sub = ChannelSubscriber("rt/lowstate", hg_LowState)
    sub.Init(lambda m: estado.__setitem__("msg", m), 1)
    return pub, sub, estado, unitree_hg_msg_dds__LowCmd_(), CRC()


def salva_mp4(quadros: list, caminho: Path, fps: float) -> None:
    """PyAV, e não imageio: o env da PGX não tem `imageio` nem `imageio-ffmpeg`, e a 1ª rodada
    de 900 passos morreu AQUI, depois de gravar o CSV. O wheel aarch64 do `av` traz o libx264."""
    import av

    altura, largura = quadros[0].shape[:2]
    altura, largura = altura - altura % 2, largura - largura % 2  # yuv420p exige lado par
    with av.open(str(caminho), mode="w") as saida:
        fluxo = saida.add_stream("libx264", rate=int(round(fps)))
        fluxo.width, fluxo.height, fluxo.pix_fmt = largura, altura, "yuv420p"
        for q in quadros:
            vq = av.VideoFrame.from_ndarray(np.ascontiguousarray(q[:altura, :largura]), format="rgb24")
            for pacote in fluxo.encode(vq):
                saida.mux(pacote)
        for pacote in fluxo.encode():
            saida.mux(pacote)
    print(f">>> vídeo em {caminho} ({len(quadros)} quadros)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--politica", default=REPO_POLITICA)
    p.add_argument("--task", default=TAREFA_TREINO, help="instrução em texto dada à política")
    p.add_argument("--camera", default=CAMERA_POLITICA, help="a câmera que a POLÍTICA enxerga")
    p.add_argument("--camera-video", default=CAMERA_VIDEO,
                   help="câmera extra, só para ver (vazio = só a da política)")
    p.add_argument("--porta", type=int, default=5555)
    p.add_argument("--env-dir", default=None,
                   help="pasta do ambiente (padrão: cache do HF). Use a cópia com as mesas e a "
                        "caixa azul feita por pgx/prepara_cena_caixa.sh")
    p.add_argument("--passos", type=int, default=3600,
                   help="passos de controle; 0 = laço infinito, até Ctrl-C ou kill")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--tempo-real", action="store_true",
                   help="segura o laço em 1/fps; sem isto a simulação corre mais rápido que o relógio")
    p.add_argument("--device", default="cuda")
    p.add_argument("--saida", default=str(Path.home() / "pi05_g1_sim"))
    p.add_argument("--sem-video", action="store_true")
    p.add_argument("--postura-treino", action="store_true",
                   help="pernas e tronco na postura de pé do dataset (agachado) e braços na pose "
                        "mediana do treino. Sem isto o robô nasce de perna reta e cotovelo dobrado, "
                        "poses que a política nunca viu — e ela copia a pose em que encontra o robô")
    p.add_argument("--zera-mortos", action="store_true",
                   help="zera no estado as 6 dims que no treino são zero constante (tronco "
                        "roll/pitch, punho pitch/yaw)")
    p.add_argument("--na-tela", action="store_true",
                   help="abre a janela do MuJoCo no monitor da máquina (backend GLFW, exige DISPLAY). "
                        "Sem isto, roda headless em EGL e você só vê pelo --v-web")
    p.add_argument("--v-web", nargs="?", const=8088, type=int, default=None, metavar="PORTA",
                   help="serve as câmeras ao vivo por HTTP (padrão: 8088). A PGX não tem tela: "
                        "abra http://192.168.123.165:<porta>/ do notebook")
    args = p.parse_args()

    import torch

    if args.na_tela and not os.environ.get("DISPLAY"):
        raise SystemExit("❌ --na-tela sem DISPLAY: rode de um terminal da sessão gráfica, ou "
                         "exporte DISPLAY=:1 antes de chamar.")

    if porta_ocupada(args.porta):
        raise SystemExit(f"❌ a porta {args.porta} já está ocupada — provavelmente sobrou o publicador "
                         f"de imagem de uma rodada anterior. Encerre aquele processo ou use --porta.")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    cam_video = args.camera_video.strip()
    cameras = [args.camera] + ([cam_video] if cam_video and cam_video != args.camera else [])

    print(f">>> criando o simulador | política vê {args.camera!r} | vídeo de {cameras}"
          f"{' | cena: ' + args.env_dir if args.env_dir else ''}")
    env = carrega_env(cameras, args.porta, args.env_dir, args.na_tela)
    obs, _ = env.reset()
    print(f"    observação do env: {obs.shape} (as 29 primeiras são body_q)")

    cam = abre_camera(args.camera, args.porta)
    cam2 = abre_camera(cam_video, args.porta) if len(cameras) > 1 else None
    pub, _sub, estado, msg, crc = abre_dds()

    # Espera o primeiro lowstate para copiar `mode_machine` e a pose atual, senão o robô
    # recebe um alvo de junta vindo do nada e dá um tranco no primeiro passo.
    limite = time.time() + 10.0
    while estado["msg"] is None:
        env.step()
        if time.time() > limite:
            raise SystemExit("❌ nenhum `rt/lowstate` em 10 s — o simulador não subiu o DDS")
    lowstate = estado["msg"]

    from lerobot.robots.unitree_g1.config_unitree_g1 import _DEFAULT_KD, _DEFAULT_KP

    msg.mode_pr = 0
    msg.mode_machine = lowstate.mode_machine
    for j in range(29):
        msg.motor_cmd[j].mode = 1
        msg.motor_cmd[j].kp = _DEFAULT_KP[j]
        msg.motor_cmd[j].kd = _DEFAULT_KD[j]
        msg.motor_cmd[j].q = lowstate.motor_state[j].q
        msg.motor_cmd[j].qd = 0.0
        msg.motor_cmd[j].tau = 0.0

    if args.postura_treino:
        # As 15 primeiras juntas (pernas + tronco) vão para a postura do dataset e FICAM lá: a
        # política só comanda os braços, então quem segura o corpo somos nós. Assenta antes do
        # laço começar, senão o primeiro quadro pega o robô no meio do agachamento.
        for j, q in enumerate(POSTURA_TREINO + BRACOS_TREINO):
            msg.motor_cmd[j].q = float(q)
        msg.crc = crc.Crc(msg)
        for _ in range(int(2.0 / 0.004)):  # 2 s a 250 Hz
            pub.Write(msg)
            env.step()
        q_medido = np.array([estado["msg"].motor_state[j].q for j in range(29)], dtype=np.float32)
        erro = float(np.abs(q_medido - np.array(POSTURA_TREINO + BRACOS_TREINO, dtype=np.float32)).max())
        print(f"    postura de treino assentada | maior erro de junta: {erro:.3f} rad")

    politica, pre, pos, chave_img = carrega_politica(args.politica, args.device)
    if chave_img != f"observation.images.{args.camera}":
        print(f"    a câmera {args.camera!r} entra na política como {chave_img!r}")

    dt = 1.0 / args.fps
    passos_fisica = max(1, int(round(dt / 0.004)))  # SIMULATE_DT do config = 250 Hz
    # Em laço infinito os quadros NÃO podem se acumular: 3.600 quadros de 640x480 já são
    # ~3,3 GB. `deque` com teto guarda só o último minuto de cada câmera, que é o que
    # interessa no vídeo do fim.
    infinito = args.passos <= 0
    teto = int(60 * args.fps) if infinito else None
    quadros_pol = deque(maxlen=teto)
    quadros_vid = deque(maxlen=teto)
    linhas = deque(maxlen=200_000 if infinito else None)
    ms_politica = 0.0
    print(f">>> {'laço infinito' if infinito else str(args.passos) + ' passos'} a {args.fps:g} Hz"
          f"{' (tempo real)' if args.tempo_real else ' (o mais rápido que der)'}"
          f" | instrução: {args.task!r}")

    # Painel HTTP reaproveitado do FastWAM-D: MJPEG, sem tela, `cv2.imencode` do build headless.
    # Ele recebe a imagem em RGB e converte para BGR por dentro. Sem autenticação, como lá.
    painel = None
    if args.v_web:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from viz_debug_fastwamd import PainelWeb

        painel = PainelWeb(porta=int(args.v_web), host="0.0.0.0", fps=min(args.fps, 15.0))
        painel.create()
        for url in painel.urls():
            print(f"    painel ao vivo: {url}")

    # Parada limpa: sem isto, um Ctrl-C ou `kill` no laço infinito mata o processo antes de
    # gravar CSV e vídeo, e a rodada inteira se perde.
    parar = {"agora": False}
    for sinal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sinal, lambda *_: parar.__setitem__("agora", True))

    t_ini = time.perf_counter()
    feitos = 0
    for passo in itertools.count():
        if parar["agora"] or (not infinito and passo >= args.passos):
            break
        feitos = passo + 1
        alvo_relogio = t_ini + passo * dt
        estado_q = np.asarray(obs[:29], dtype=np.float32)
        if args.zera_mortos:
            estado_q[DIMS_MORTAS] = 0.0
        quadro = cam.read()  # RGB HWC uint8 — o que a política vê
        quadro_vid = cam2.read() if cam2 is not None else None
        if not args.sem_video:
            quadros_pol.append(quadro.copy())
            if quadro_vid is not None:
                quadros_vid.append(quadro_vid.copy())

        bruto = {
            "observation.state": torch.from_numpy(estado_q),
            chave_img: torch.from_numpy(quadro).permute(2, 0, 1),
            "task": args.task,
        }
        t0 = time.perf_counter()
        with torch.inference_mode():
            lote = pre(bruto)
            acao = politica.select_action(lote)
            acao = pos(acao)
        ms_politica += (time.perf_counter() - t0) * 1000
        acao = np.asarray(acao, dtype=np.float32).reshape(-1)
        if acao.shape[0] != 18:
            raise SystemExit(f"❌ ação com {acao.shape[0]} dims, esperava 18 (14 braços + 4 do controle)")

        for i, j in enumerate(JUNTAS_BRACOS):
            msg.motor_cmd[j].q = float(np.clip(acao[i], -np.pi, np.pi))
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        linhas.append([passo, *[f"{v:.4f}" for v in acao]])

        if painel is not None:
            if quadro_vid is not None:
                import cv2

                h = min(quadro.shape[0], quadro_vid.shape[0])
                mosaico = np.hstack([
                    cv2.resize(quadro, (int(quadro.shape[1] * h / quadro.shape[0]), h)),
                    cv2.resize(quadro_vid, (int(quadro_vid.shape[1] * h / quadro_vid.shape[0]), h)),
                ])
            else:
                mosaico = quadro
            painel.define_imagens(rgb_mosaico=mosaico)
            painel.define_cabecalho(
                f"pi05_unitree_g1 | passo {passo + 1}/{'∞' if infinito else args.passos} | "
                f"{ms_politica / (passo + 1):.0f} ms/chamada | {args.camera} + {cam_video} | {args.task}")
            painel.show()

        for _ in range(passos_fisica):
            obs, _r, _t, _tr, _i = env.step()

        if args.tempo_real:
            atraso = alvo_relogio + dt - time.perf_counter()
            if atraso > 0:
                time.sleep(atraso)
        if passo % 60 == 0:
            total = "∞" if infinito else str(args.passos)
            print(f"    {passo}/{total} | {ms_politica / max(passo + 1, 1):.0f} ms por chamada"
                  f" | controle remoto: {np.round(acao[14:], 3).tolist()}", flush=True)

    dur = time.perf_counter() - t_ini
    if parar["agora"]:
        print(">>> parada pedida (sinal) — gravando o que já foi feito")
    print(f">>> {feitos} passos em {dur / 60:.1f} min ({ms_politica / max(feitos, 1):.0f} ms por chamada)")

    csv_saida = saida / "acoes.csv"
    with open(csv_saida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["passo", *[f"braco_{i}" for i in range(14)], "remote_lx", "remote_ly", "remote_rx", "remote_ry"])
        w.writerows(linhas)
    print(f">>> ações em {csv_saida}")

    if quadros_pol:
        salva_mp4(quadros_pol, saida / f"{args.camera}.mp4", args.fps)
    if quadros_vid:
        salva_mp4(quadros_vid, saida / f"{cam_video}.mp4", args.fps)

    if painel is not None:
        painel.destroy()
    cam.disconnect()
    if cam2 is not None:
        cam2.disconnect()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
