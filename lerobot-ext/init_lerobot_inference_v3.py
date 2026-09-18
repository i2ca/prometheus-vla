#!/usr/bin/env python
"""
Inference Entry Point V3 — Universal
Suporta: actdepth, pi05depth, openvladepth

Políticas condicionadas por linguagem (`pi05depth`, `openvladepth`) EXIGEM
`--task`: é o texto desse argumento que vira o prompt e decide qual tarefa o
robô executa. `actdepth` não usa linguagem e ignora o argumento.

Uso:
  python init_lerobot_inference_v3.py --checkpoint=<CAMINHO> [OPÇÕES]

Opções:
  --checkpoint=<PATH>    (obrigatório) Caminho para o pretrained_model
  --task="<TEXTO>"       Comando em linguagem natural (obrigatório para VLAs)
  --pose-inicial         Leva os braços à pose em que os episódios do dataset
                         começam, ANTES de soltar a política. Sem isto o robô
                         parte de uma pose que o modelo nunca viu.
  --interactive          Permite trocar o comando digitando durante a execução
  --sim                  Modo simulação (sem robô real)
  --cam-robot=<IP>       Stream ZMQ de câmera externa
  --port-cam=<PORTA>     Porta do stream (padrão: 5555)
  --fake-video=<PATH>    Injeta imagem ou vídeo na câmera
  --uncertainty=<FLOAT>  Ativa o uncertainty gate (ex: 0.1)
  --v                    Abre janela de visualização da câmera
  --debug                Loga ações no terminal em tempo real
  --v-web[=PORTA]        Painel de debug no navegador (só pi05depth; padrão 8088):
                         atenção das ações sobre as câmeras, depth, nuvem e o
                         chunk em execução. Abra http://<ip-desta-máquina>:PORTA/
  -h, --help             Mostra esta mensagem

Exemplos:
  # ACT-D (sem linguagem):
  python init_lerobot_inference_v3.py \
      --checkpoint=train_output/pick_up_the_cup_nodepth/best_val_checkpoint/pretrained_model

  # OpenVLA-Depth por comando de texto:
  python init_lerobot_inference_v3.py \
      --checkpoint=train/output/openvla_depth_cup_2026-06-09/best_val_checkpoint/pretrained_model \
      --task="pick up the white mug and place it to the right" \
      --cam-robot=192.168.123.164 --v

  # Trocando o comando durante a execução:
  python init_lerobot_inference_v3.py --checkpoint=<PATH> \
      --task="pick up the white mug and place it to the right" --interactive

O texto precisa casar com o que foi gravado no campo `task` do dataset. Ver
docs/INFERENCIA_COMANDO_TEXTO.md.
"""

import os
import sys
import time
import torch
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# 1. REGISTRO DAS POLÍTICAS CUSTOMIZADAS
# ─────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import policies  # registra actdepth, pi05depth, etc.
except ImportError as e:
    print(f"[ERRO]: Falha ao carregar o registry 'policies': {e}")
    sys.exit(1)

from lerobot.configs.policies import PreTrainedConfig


# ─────────────────────────────────────────────────────────────────────
# 2. LOADER UNIVERSAL
# ─────────────────────────────────────────────────────────────────────
def load_policy(checkpoint_dir: str, device: torch.device):
    print(f"⏳ Carregando política de: {checkpoint_dir}")

    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_type = getattr(config, "type", "desconhecido")
    print(f"   Tipo detectado: {policy_type}")

    from safetensors.torch import load_file
    import importlib

    _POLICY_CLASS_MAP = {
        "actdepth":     ("policies.act_depth.modeling_act", "ACTPolicy"),
        "pi05depth":    ("policies.pi0_depth.modeling_pi05", "PI05DEPTHPolicy"),
        "openvladepth": ("policies.openvla_depth.modeling_openvla", "OPENVLADEPTHPolicy"),
    }

    if policy_type in _POLICY_CLASS_MAP:
        module_path, class_name = _POLICY_CLASS_MAP[policy_type]
        module = importlib.import_module(module_path)
        PolicyClass = getattr(module, class_name)
        policy = PolicyClass(config)
        print(f"   Instanciado: {module_path}.{class_name}")
    else:
        raise ValueError(f"Tipo '{policy_type}' não mapeado. Adicione em _POLICY_CLASS_MAP.")

    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"model.safetensors não encontrado em {checkpoint_dir}")

    state_dict = load_file(model_file)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)

    if missing:
        # Pesos do VAE encoder são esperados ausentes em inferência (não são usados)
        vae_prefixes = (
            "model.vae_encoder", "model.vae_encoder_cls_embed",
            "model.vae_encoder_robot_state_input_proj",
            "model.vae_encoder_action_input_proj",
            "model.vae_encoder_latent_output_proj",
        )
        real_missing = [k for k in missing if not any(k.startswith(p) for p in vae_prefixes)]
        if real_missing:
            print(f"   ⚠️  {len(real_missing)} pesos ausentes inesperados:")
            for k in real_missing[:10]:
                print(f"      - {k}")
        else:
            print(f"   ✅ {len(missing)} ausentes = VAE encoder (normal em inferência)")
    if unexpected:
        print(f"   ⚠️  {len(unexpected)} pesos inesperados")

    policy.eval()
    policy.to(device)
    print(f"✅ Política '{policy_type}' carregada!")
    return policy, policy_type


# ─────────────────────────────────────────────────────────────────────
# 3. CARREGA PREPROCESSOR E POSTPROCESSOR DO CHECKPOINT
# ─────────────────────────────────────────────────────────────────────
def load_pre_post_processors(checkpoint_dir: str, policy):
    """
    Carrega preprocessor e postprocessor salvos junto com o checkpoint.

    Ambas as políticas usam arquivos externos de normalização:

    ACT  (MEAN_STD):
      preprocessor → rename, to_batch, device, normalize(images+state com mean/std)
      postprocessor → unnormalize(action com mean/std inverso), cpu

    PI05 (QUANTILES):
      preprocessor → rename, to_batch, normalize(state com q01/q99), discretize, tokenize, device
      postprocessor → unnormalize(action com q01/q99 inverso), cpu

    Os pesos vêm dos safetensors no checkpoint:
      policy_preprocessor_step_N_normalizer_processor.safetensors
      policy_postprocessor_step_0_unnormalizer_processor.safetensors
    """
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
    )
    print("✅ Preprocessor e postprocessor carregados do checkpoint.")
    return preprocessor, postprocessor


# ─────────────────────────────────────────────────────────────────────
# 4. MONTA OBSERVAÇÃO BRUTA (ACT e PI05 usam a mesma função)
# ─────────────────────────────────────────────────────────────────────
def _depth_to_tensor(depth: "np.ndarray", device=None) -> "torch.Tensor":
    """Mapa de profundidade → `[1, H, W]` float32 em MILÍMETROS.

    Formato nativo da 0.6.1: 1 canal, valor métrico. O caminho antigo replicava
    em 3 canais e dividia por 255, porque a profundidade era gravada como
    imagem de 8 bits (0–2000 mm espremidos em 0–255). Fazer isso hoje não
    quebra nada visivelmente — só entrega milímetros divididos por 255 à
    política, que espera milímetros. Erro silencioso, o pior tipo.

    A política converte mm → metros na projeção 3D
    (`policies/act_depth/depth_encoder.py::depth_to_pointcloud`, `depth_unit`).
    """
    import numpy as _np

    depth = _np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(
            f"Profundidade deveria ser um mapa de 1 canal [H, W], veio {depth.shape}. "
            "Se o servidor ainda publica cinza de 3 canais, atualize-o "
            "(Scripts_Prometheus_int/full_realsenser_server.py)."
        )
    tensor = torch.from_numpy(_np.ascontiguousarray(depth)).float().unsqueeze(0)
    return tensor if device is None else tensor.to(device)


# MEDIDO 18/09/2026 — pose em que os 302 episódios de "place the white cup on the dripper"
# COMEÇAM, no `cotreino_completo_2026-09-11` (média do primeiro quadro de cada episódio).
#
# Por que isto existe: o MuJoCo nasce com os braços caídos, e essa pose fica a **0,906 rad em
# média** da pose de partida do dataset — os dois punhos dobrados 2 rad (115°) para o lado
# errado. Depois da normalização por quantis do π0.5, o punho esquerdo entra em **-5,02** num
# espaço onde o modelo só viu [-1, +1]. A primeira observação já está fora da distribuição e a
# política extrapola a partir do primeiro quadro — é o mesmo bug que o `--pose-inicial` do
# `pgx/roda_politica_g1_isaaclab.py` corrige no IsaacLab.
#
# São 15 valores: as 14 juntas dos braços e o yaw do tronco, na ordem do `info.json`. As 14
# dimensões das mãos ficam em zero porque no dado do copo elas SÃO zero (desvio 0,000).
POSE_PARTIDA_COPO = [0.958, 0.683, 1.476, -0.554, -1.501, 0.505, -0.225,
                     -0.941, -0.628, -0.313, 0.388, 0.232, 0.655, 0.337,
                     -0.036]


def leva_a_pose_de_partida(robot, joint_names, alvo=None, segundos=3.0, fps=30) -> None:
    """Interpola da pose atual até a pose de partida do dataset, antes de soltar a política."""
    import numpy as _np

    alvo = list(alvo or POSE_PARTIDA_COPO)
    obs = robot.get_observation() or {}
    inicio = [float(obs.get(n, 0.0)) for n in joint_names]
    fim = list(inicio)
    fim[:len(alvo)] = alvo                       # mãos ficam onde estão (zero no dado do copo)
    passos = max(1, int(segundos * fps))
    for k in range(1, passos + 1):
        a = k / passos
        q = [(1 - a) * i + a * f for i, f in zip(inicio, fim)]
        robot.send_action({n: float(v) for n, v in zip(joint_names, q)})
        time.sleep(1.0 / fps)
    depois = robot.get_observation() or {}
    erro = max(abs(float(depois.get(n, 0.0)) - f) for n, f in zip(joint_names, fim))
    print(f"🎯 pose de partida do dataset aplicada | maior erro de junta: {erro:.3f} rad")


def make_raw_obs(
    obs: dict,
    joint_names: list[str],
    has_depth: bool = False,
    has_pressure: bool = False,
    task: str | None = None,
    image_keys: list[str] | None = None,
) -> dict:
    """
    Monta o dict de observação SEM batch dim e SEM normalização.

    O preprocessor cuida de tudo:
      - Batch dim (to_batch_processor)
      - ACT:  normaliza imagens com ImageNet mean/std e estado com MEAN_STD do dataset
      - PI05: normaliza estado com QUANTILES, discretiza e tokeniza

    Parâmetros:
      task: None para ACT (não usa linguagem), "pick up the cup" para PI05
    """
    raw = {}

    # Estado das juntas [28] — radianos brutos, sem normalização
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    raw["observation.state"] = torch.tensor(state_vector, dtype=torch.float32)

    # RGB [C, H, W] em [0, 1] — preprocessor aplica ImageNet mean/std (ACT) ou nada (PI05 IDENTITY)
    #
    # TODAS as câmeras RGB que o checkpoint declara (`image_keys`), e não só a
    # cabeça. Até 15/09/2026 só a cabeça entrava: um modelo treinado com cabeça +
    # pulso rodava SEM o pulso e sem erro nenhum — o π0.5 preenche câmera
    # ausente com imagem vazia mascarada e segue.
    for chave in image_keys or ["observation.images.head_camera"]:
        nome = chave.rsplit(".", 1)[-1]
        if nome.endswith("depth"):
            continue
        rgb = obs.get(nome)
        if rgb is not None:
            raw[chave] = (
                torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
            )

    # Depth [C, H, W] em [0, 1]
    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            raw["observation.images.head_camera_depth"] = _depth_to_tensor(depth)

    # Pressão [33] — sem batch dim
    if has_pressure:
        for side in ["left", "right"]:
            val = obs.get(f"{side}_hand_pressure")
            if val is not None:
                raw[f"observation.{side}_hand_pressure"] = torch.from_numpy(
                    np.array(val, dtype=np.float32)
                )

    # Task string — só PI05 usa; o preprocessor tokeniza internamente
    if task is not None:
        raw["task"] = task

    return raw


# ─────────────────────────────────────────────────────────────────────
# 5. SETUP DE CÂMERAS
# ─────────────────────────────────────────────────────────────────────
def setup_cameras(cam_robot_ip, cam_port, fake_video_path):
    from Scripts_Prometheus_int.sim.sensor_utils import SensorClient, ImageUtils

    stream_client = None
    fake_cap = None
    fake_img_rgb = None

    if cam_robot_ip:
        stream_client = SensorClient()
        stream_client.start_client(server_ip=cam_robot_ip, port=int(cam_port))
        print(f"📡 Conectando ao ZMQ SensorServer em tcp://{cam_robot_ip}:{cam_port}...")
    elif fake_video_path:
        if not os.path.exists(fake_video_path):
            print(f"❌ ERRO: Arquivo fake não encontrado: {fake_video_path}")
            sys.exit(1)
        if fake_video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            fake_cap = cv2.VideoCapture(fake_video_path)
            print("✅ Vídeo fake carregado! (Modo Loop)")
        else:
            img_bgr = cv2.imread(fake_video_path)
            fake_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            print("✅ Imagem fake carregada!")

    return stream_client, fake_cap, fake_img_rgb


# ─────────────────────────────────────────────────────────────────────
# 6. LEITURA DE FRAME
# ─────────────────────────────────────────────────────────────────────
def get_camera_frames(obs, stream_client, fake_cap, fake_img_rgb):
    if stream_client is not None:
        from Scripts_Prometheus_int.sim.sensor_utils import ImageUtils
        msg = stream_client.receive_message()
        if msg and "images" in msg:
            # TODAS as câmeras do stream, não só a cabeça. O MuJoCo publica
            # cabeça, depth e pulso no MESMO stream (5555); copiando só as duas
            # primeiras, a `right_wrist_camera` chegava e era descartada aqui.
            for nome, dado in msg["images"].items():
                obs[nome] = ImageUtils.decode_image(dado)
    elif fake_cap is not None:
        ret, frame = fake_cap.read()
        if not ret:
            fake_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = fake_cap.read()
        if ret:
            fake_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obs["head_camera"] = fake_img_rgb
    elif fake_img_rgb is not None:
        obs["head_camera"] = fake_img_rgb

    return obs, fake_img_rgb


# ─────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    if any(f in sys.argv for f in ["-h", "--help"]):
        print(__doc__)
        sys.exit(0)

    checkpoint_dir = None
    is_sim = False
    fake_video_path = None
    cam_robot_ip = None
    cam_port = "5555"
    debug_mode = False
    show_video = False
    uncertainty_threshold = 0.0
    remote_sim_ip = None
    fps = 30
    task_cli = None
    interactive = False
    pose_inicial = False
    web_porta = None

    for arg in sys.argv[1:]:
        if arg.startswith("--checkpoint="):
            checkpoint_dir = arg.split("=", 1)[1]
        elif arg in ["--sim", "--simulation=true"]:
            is_sim = True
        elif arg.startswith("--fake-video="):
            fake_video_path = arg.split("=", 1)[1]
        elif arg.startswith("--cam-robot="):
            cam_robot_ip = arg.split("=", 1)[1]
        elif arg.startswith("--port-cam="):
            cam_port = arg.split("=", 1)[1]
        elif arg.startswith("--uncertainty="):
            uncertainty_threshold = float(arg.split("=", 1)[1])
        elif arg == "--debug":
            debug_mode = True
        elif arg == "--v":
            show_video = True
            print("[INFO]: Visualização de câmera ativada (--v)")
        elif arg.startswith("--remote-sim="):
            remote_sim_ip = arg.split("=", 1)[1]
        elif arg.startswith("--fps="):
            fps = int(arg.split("=", 1)[1])
        elif arg.startswith("--task="):
            task_cli = arg.split("=", 1)[1]
        elif arg == "--interactive":
            interactive = True
        elif arg == "--pose-inicial":
            pose_inicial = True
        elif arg == "--v-web":
            web_porta = 8088
        elif arg.startswith("--v-web="):
            web_porta = int(arg.split("=", 1)[1])

    if checkpoint_dir is None:
        print("❌ ERRO: --checkpoint obrigatório.")
        print("   Uso: python init_lerobot_inference_v3.py --checkpoint=<CAMINHO>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando device: {device}")

    # ── Carrega política ──────────────────────────────────────────────
    policy, policy_type = load_policy(checkpoint_dir, device)

    has_depth    = getattr(policy.config, "use_depth_3d", False)
    # MEDIDO 18/09/2026: `use_wrist_camera` nasce False no `UnitreeG1Dex3Config`, e este script
    # nunca passava o parâmetro. Resultado: o MuJoCo subia com DUAS câmeras (head + depth), o
    # checkpoint pedia `right_wrist_camera`, e ela entrava faltando — o aviso "câmeras que o
    # checkpoint espera e NÃO chegaram" era impresso e a inferência seguia com essa entrada
    # ausente. Agora quem decide é o próprio checkpoint.
    has_wrist    = "observation.images.right_wrist_camera" in (policy.config.image_features or {})
    has_pressure = getattr(policy.config, "use_pressure", False)
    print(f"   Depth 3D: {has_depth} | Pressão: {has_pressure}")

    if uncertainty_threshold > 0:
        policy.config.scene_uncertainty_threshold = uncertainty_threshold
        print(f"✅ Uncertainty Gate ativado: threshold={uncertainty_threshold}")

    # ── Preprocessor e Postprocessor ─────────────────────────────────
    # Ambas as políticas (ACT e PI05) têm preprocessor/postprocessor
    # salvos no checkpoint com seus pesos de normalização.
    # O preprocessor normaliza entradas; o postprocessor desnormaliza a saída.
    preprocessor, postprocessor = load_pre_post_processors(checkpoint_dir, policy)

    # ── Comando em texto ──────────────────────────────────────────────
    # `pi05depth` e `openvladepth` são condicionados por linguagem: o texto aqui
    # vira o prompt do modelo e é o que decide qual tarefa ele executa. `actdepth`
    # não usa linguagem nenhuma e ignora isto.
    #
    # O comando fica numa lista de um elemento para poder ser trocado em tempo de
    # execução pela thread de --interactive (ver command_listener).
    LANGUAGE_POLICIES = {"pi05depth", "openvladepth"}
    task_box = [None]

    if policy_type in LANGUAGE_POLICIES:
        if task_cli is None:
            print(
                f"\n❌ ERRO: a política '{policy_type}' é condicionada por linguagem e "
                f"precisa de um comando.\n"
                f'   Use: --task="pick up the white mug and place it to the right"\n'
                f"   O texto deve casar com o que foi gravado no campo `task` do dataset.\n"
            )
            sys.exit(1)
        task_box[0] = task_cli
        print(f'🗣️  Comando: "{task_cli}"')
    elif task_cli is not None:
        print(f"⚠️  --task ignorado: '{policy_type}' não usa linguagem.")

    # Troca de comando em tempo de execução: cada linha digitada vira o novo
    # prompt. `policy.reset()` é essencial — sem ele o robô termina o chunk
    # antigo (até 50 passos, ~1,7 s a 30 fps) antes de obedecer ao comando novo.
    if interactive and policy_type in LANGUAGE_POLICIES:
        import threading

        def command_listener():
            print("\n💬 Modo interativo: digite um comando e Enter para trocar a tarefa.")
            print("   (Ctrl+C encerra a inferência)\n")
            for line in sys.stdin:
                novo = line.strip()
                if not novo:
                    continue
                task_box[0] = novo
                policy.reset()          # descarta o chunk em execução
                print(f'🗣️  Comando trocado para: "{novo}"')

        threading.Thread(target=command_listener, daemon=True, name="CommandListener").start()

    # ── Câmeras ───────────────────────────────────────────────────────
    stream_client, fake_cap, fake_img_rgb = setup_cameras(
        cam_robot_ip, cam_port, fake_video_path
    )

    # ── Robô ─────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        robot_ip="10.9.8.73",
        #robot_ip="192.168.123.164",
        control_mode="upper_body",
        use_waist_yaw=True,     # sem isto o robô não expõe nem comanda kWaistYaw.q
        # Só abre o stream de profundidade se o checkpoint usa: um π0.5 sem
        # `use_depth_3d` roda contra `run_sim.py --so-rgb` sem esperar por ela.
        use_depth_camera=has_depth,
        use_wrist_camera=has_wrist,
        is_simulation=is_sim,
        remote_sim_ip=remote_sim_ip,
    )
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô conectado!")

    for cam in robot.cameras.values():
        if hasattr(cam, 'timeout_ms'):
            cam.timeout_ms = 800

    # Nomes das juntas na mesma ordem do dataset (info.json)
    joint_names = [
        "kLeftShoulderPitch.q",  "kLeftShoulderRoll.q",  "kLeftShoulderYaw.q",
        "kLeftElbow.q",          "kLeftWristRoll.q",      "kLeftWristPitch.q",
        "kLeftWristyaw.q",
        "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
        "kRightElbow.q",         "kRightWristRoll.q",     "kRightWristPitch.q",
        "kRightWristYaw.q",
        # Schema v2 (SCHEMA_G1_V2.md): 29 juntas, o yaw do tronco na dim 14.
        # Sem ele o estado entra com as mãos deslocadas uma posição e a ação
        # volta para as juntas erradas — sem erro nenhum.
        "kWaistYaw.q",
        "left_hand_thumb_0_joint.q",  "left_hand_thumb_1_joint.q",
        "left_hand_thumb_2_joint.q",  "left_hand_middle_0_joint.q",
        "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
        "left_hand_index_1_joint.q",
        "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q",
        "right_hand_thumb_2_joint.q", "right_hand_index_0_joint.q",
        "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
        "right_hand_middle_1_joint.q",
    ]

    dim_acao = policy.config.output_features["action"].shape[0]
    if len(joint_names) != dim_acao:
        raise SystemExit(
            f"❌ o checkpoint prevê {dim_acao} juntas e o script conhece {len(joint_names)}. "
            "Casar a lista `joint_names` com o `info.json` do dataset de treino."
        )

    # ── Painel de debug no navegador (--v-web) ───────────────────────
    painel = None
    linha_base = None
    chunks_painel = None
    infer_ms = 0.0
    massa = None
    passo_global = 0
    if web_porta is not None:
        if policy_type != "pi05depth":
            print(f"⚠️  --v-web só existe para pi05depth; '{policy_type}' segue sem painel.")
        else:
            from viz_debug_fastwamd import PainelWeb, INTRINSECOS_PADRAO
            from policies.pi0_depth.debug_atencao_pi05 import (
                CapturaAtencaoPI05, LinhaDeBaseDaAtencao, mosaico_como_o_modelo,
                profundidade_mm_para_painel,
            )
            chaves_imagem = list(policy.config.image_features)
            # A nuvem de pontos precisa dos intrínsecos da câmera que está
            # mandando a profundidade. No MuJoCo é a `head_camera_depth` do MJCF:
            # fovy 58° a 848×480 → f = 240 / tan(29°) ≈ 433 px, pixel quadrado.
            intrinsecos = ({"fx": 433.0, "fy": 433.0, "cx": 424.0, "cy": 240.0} if is_sim
                           else dict(INTRINSECOS_PADRAO))
            painel = PainelWeb(porta=web_porta, fps=10, intrinsecos=intrinsecos)
            painel.nome_atencao = "pi0.5 (acoes -> cameras)"
            painel.create()
            linha_base = LinhaDeBaseDaAtencao()
            print(f"📊 Painel de debug: {', '.join(painel.urls())}")

    # Uma thread para o laço: a IK do braço roda a cada quadro e, com o ipopt multithread do
    # conda-forge, ela sai de 0,8 ms para 83 ms. Aqui e não no ambiente porque o MESMO processo
    # acabou de carregar 9,3 GB de pesos, e essa parte quer TODAS as threads.
    torch.set_num_threads(1)

    if pose_inicial:
        leva_a_pose_de_partida(robot, joint_names, fps=fps)

    print(f"\n🚀 INFERÊNCIA ATIVA [{policy_type.upper()}] — O robô vai se mover!")
    if show_video:
        print("   📺 Janela de câmera ativa.")
    print(f"   ⏱️  FPS: {fps} ({1000/fps:.1f}ms por ciclo)")
    print("   Ctrl+C para parar.\n")

    # ─────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────────
    try:
        while True:
            start_t = time.perf_counter()

            # 1. Observação do robô
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"⚠️  Timeout de câmera: {e}. Pulando frame...")
                continue
            if not obs:
                continue

            # 2. Câmeras externas
            obs, fake_img_rgb = get_camera_frames(
                obs, stream_client, fake_cap, fake_img_rgb
            )

            # 3. Visualização
            rgb = obs.get("head_camera")
            if show_video and rgb is not None:
                cv2.imshow("Visao da IA", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # 4. Monta observação bruta (sem batch dim, sem normalização)
            raw_obs = make_raw_obs(
                obs=obs,
                joint_names=joint_names,
                has_depth=has_depth,
                has_pressure=has_pressure,
                task=task_box[0],   # lido a cada passo: --interactive pode ter trocado
                image_keys=list(policy.config.image_features),
            )
            if passo_global == 0:
                faltando = [k for k in policy.config.image_features
                            if k not in raw_obs and not k.endswith("depth")]
                if faltando:
                    print(f"\n⚠️  câmeras que o checkpoint espera e NÃO chegaram: {faltando}")
                else:
                    print(f"📷 câmeras entregues ao modelo: {list(policy.config.image_features)}")

            # 5. Preprocessor: normaliza + batch dim + device
            #    ACT:  mean/std nas imagens (ImageNet) e estado (dataset stats)
            #    PI05: quantiles no estado, discretiza, tokeniza
            batch = preprocessor(raw_obs)

            # Remove "action" do batch — o preprocessor define essa chave nos features
            # para normalizar targets durante treino, mas em inferência ela não existe
            # e fica None. Com ela no batch o ACT entra no VAE encoder e crasha.
            batch.pop("action", None)

            # Filtra o batch: mantém só tensors.
            # O preprocessor pode deixar valores escalares (floats, ints, strings)
            # de metadata interna no dict. O modelo faz next(iter(batch.values())).shape[0]
            # para inferir B, então qualquer não-tensor causa AttributeError.
            batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Workaround: garante batch dim na pressão se o preprocessor não adicionar
            if has_pressure:
                for side in ["left", "right"]:
                    k = f"observation.{side}_hand_pressure"
                    if k in batch and batch[k].dim() == 1:
                        batch[k] = batch[k].unsqueeze(0)

            # 6. Inferência
            #    Com o painel, só o passo que calcula um chunk NOVO (fila vazia)
            #    roda dentro da captura de atenção; os outros só tiram da fila.
            chunk_novo = painel is not None and len(policy._action_queue) == 0
            t_inferencia = time.perf_counter()
            with torch.inference_mode():
                if chunk_novo:
                    with CapturaAtencaoPI05(policy) as captura:
                        action = policy.select_action(batch)
                else:
                    action = policy.select_action(batch)

            if chunk_novo:
                infer_ms = (time.perf_counter() - t_inferencia) * 1000
                resumo = captura.resumo()
                if resumo is not None:
                    painel.define_debug_servidor(linha_base.payload(resumo))
                    massa = resumo["massa"]
                # O chunk inteiro, em radianos, para o quadrante 4: a ação deste
                # passo mais as que ficaram na fila.
                try:
                    with torch.inference_mode():
                        chunk_norm = torch.stack([action] + list(policy._action_queue), dim=1)
                        chunk_rad = postprocessor(chunk_norm)
                    if isinstance(chunk_rad, dict):
                        chunk_rad = chunk_rad["action"]
                    chunks_painel = [{"inicio": passo_global,
                                      "chunk": chunk_rad[0].float().cpu().numpy()}]
                except Exception as erro:
                    print(f"\n⚠️  painel: chunk não desnormalizado ({erro})")
                    chunks_painel = None

            # 7. Postprocessor: desnormaliza ação → radianos reais → CPU
            #    ACT:  action * std + mean   (MEAN_STD inverso)
            #    PI05: (action + 1) / 2 * (q99 - q01) + q01   (QUANTILES inverso)
            action = postprocessor(action)

            # 8. Converte para numpy
            if isinstance(action, dict):
                action_numpy = action["action"].squeeze(0).cpu().numpy()
            else:
                action_numpy = action.squeeze(0).cpu().numpy()

            # 9. Debug — valores devem estar em radianos reais, não em [-1, 1]
            if debug_mode:
                arm = " | ".join([f"{v:.3f}" for v in action_numpy[:7]])
                print(f"\r🤖 [{policy_type}] braço E: [{arm}]", end="", flush=True)

            # 10. Envia ao robô
            action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}
            robot.send_action(action_dict)

            # 10b. Painel: só deposita os dados; o desenho roda na thread dele.
            if painel is not None:
                painel.define_imagens(rgb_mosaico=mosaico_como_o_modelo(obs, chaves_imagem),
                                      depth_mm=profundidade_mm_para_painel(obs.get("head_camera_depth")))
                painel.define_chunks(chunks_painel, passo_global, 0)
                no_chunk = policy.config.n_action_steps - len(policy._action_queue)
                atencao = ""
                if massa:
                    atencao = "  |  atencao: " + "  ".join(
                        f"{k.replace('_camera', '')} {v:.0%}" for k, v in massa.items())
                painel.define_cabecalho(
                    f"pi0.5 '{task_box[0]}'  |  passo {passo_global}  |  chunk {no_chunk}/"
                    f"{policy.config.n_action_steps}  |  infer {infer_ms:.0f} ms{atencao}")
            passo_global += 1

            # 11. Limita ao fps configurado (padrão: 30Hz)
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.0, (1.0 / fps) - elapsed)
            if debug_mode and elapsed > (1.0 / fps):
                print(f"\n⚠️  Loop lento: {elapsed*1000:.1f}ms (limite: {1000/fps:.1f}ms)")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Parando inferência...")

    finally:
        if fake_cap is not None:
            fake_cap.release()
        if stream_client is not None:
            stream_client.stop_client()
        robot.disconnect()
        if show_video:
            cv2.destroyAllWindows()
        if painel is not None:
            painel.destroy()
        print("✅ Encerrado com segurança.")


if __name__ == "__main__":
    main()