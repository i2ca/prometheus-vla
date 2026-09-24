#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# GPU: NVIDIA por PRIME offload — LIGADO POR PADRÃO.
#
# Ganho medido no render offscreen 640×480: Intel 57 fps vs RTX 5070 1141 fps.
# Na Intel a renderização das câmeras vira o gargalo do loop.
#
# Para desligar:  PROMETHEUS_FORCE_NVIDIA=0 python run_sim.py
#
# Precisa vir antes de qualquer import que crie contexto OpenGL — o libGL lê na
# criação do contexto, depois não adianta.
# ─────────────────────────────────────────────────────────────────────────────
import os

if os.environ.get("PROMETHEUS_FORCE_NVIDIA") != "0":
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import sys
from pathlib import Path
import time
# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from sim.simulator_factory import SimulatorFactory, init_channel

def main(n_envs=1, use_async_envs: bool = False, 
             publish_images=True, camera_port=5555, cameras=None, **kwargs):
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with default values
    enable_offscreen = publish_images or config.get("ENABLE_OFFSCREEN", False)

    # Configure cameras if requested
    camera_configs = {}
    if enable_offscreen:
        # REMOVIDO: "d435i_rgb"
        camera_list = cameras or ["head_camera", "head_camera_depth", "right_wrist_camera"]

        # Resolução por câmera.
        #
        # A câmera de pulso vai a 224×224 de propósito: é exatamente o que a torre
        # visual do OpenVLA consome. Gravar maior só gastaria disco, porque o
        # `_preprocess_images` redimensiona tudo para 224×224 antes do modelo — e
        # gravar em 4:3 ainda introduziria uma deformação no reescalonamento para
        # quadrado. 224×224 é o menor tamanho que não perde nada.
        #
        # A cabeça fica maior porque o depth entra na nuvem de pontos em resolução
        # NATIVA (ali resolução importa de verdade) e o RGB da cabeça também
        # alimenta o VR.
        #
        # 848×480, igual à D435i do robô real, ao `UnitreeG1Dex3Config` (que
        # espera 848×480 também em simulação) e aos datasets de simulação do
        # `gerar_dataset_mujoco.py`. Com 640×480 a política recebia outra
        # proporção de imagem do que viu no treino — e o `ZMQCamera` não confere
        # tamanho, então nada acusava.
        CAMERA_RESOLUTIONS = {
            "head_camera":        {"height": 480, "width": 848},
            "head_camera_depth":  {"height": 480, "width": 848},
            "right_wrist_camera": {"height": 224, "width": 224},
        }
        for cam_name in camera_list:
            camera_configs[cam_name] = CAMERA_RESOLUTIONS.get(
                cam_name, {"height": 480, "width": 640}
            )
        # SIM_CAMERAS_ANTIGAS=1: cabeça e pulso saem das poses que valiam até 21/09
        # (`*_antiga` no MJCF), com os MESMOS nomes publicados. É com elas que os
        # datasets de simulação até 18/09 foram gravados (o `pega_copo_sem_prof`,
        # que treinou o π0.5); com as poses corrigidas a política vê outra cena.
        if os.environ.get("SIM_CAMERAS_ANTIGAS") == "1":
            for nome in ("head_camera", "head_camera_depth", "right_wrist_camera"):
                if nome in camera_configs:
                    base = nome.replace("_depth", "")
                    camera_configs[nome]["mjcf"] = f"{base}_antiga"
            print("📷 SIM_CAMERAS_ANTIGAS=1: cabeça e pulso nas poses de antes de 21/09")
        resumo = ", ".join(
            f"{n} {camera_configs[n]['width']}x{camera_configs[n]['height']}" for n in camera_list
        )
        print(f"📷 Cameras: {resumo} → ZMQ port {camera_port}")
    
    print("="*60)
    
    # Initialize DDS channel
    init_channel(config=config)
    
    # Create simulator
    sim = SimulatorFactory.create_simulator(
        config=config,
        env_name="default",
        onscreen=config.get("ENABLE_ONSCREEN", True),
        offscreen=enable_offscreen,
        camera_configs=camera_configs,
    )
    
    # Start simulator
    print("\nSimulator running. Press Ctrl+C to exit.")
    if enable_offscreen and publish_images:
        print(f"Camera images publishing on tcp://localhost:{camera_port}")
    try:
        if publish_images:
            sim.start_image_publish_subprocess(
                start_method="spawn",
                camera_port=camera_port,
            )
            time.sleep(1)
        sim.start()
    except KeyboardInterrupt:
        print("+++++Simulator interrupted by user.")
    except Exception as e:
        print(f"++++error in simulator: {e} ++++")
    finally:
        print("++++closing simulator ++++")
        sim.close()

if __name__ == "__main__":
    # --so-rgb: publica só cabeça e pulso, sem `head_camera_depth`. Para testar
    # políticas que não usam profundidade (o π0.5 sem `use_depth_3d`) sem gastar
    # render e banda com ela. O padrão continua com as três câmeras, porque a
    # teleoperação no simulador grava profundidade.
    #
    # Do lado da política, a câmera de profundidade tem que sair também do robô
    # (`UnitreeG1Dex3Config.use_depth_camera=False`, que o v3 já faz quando o
    # checkpoint não usa profundidade) — senão o `connect()` espera por ela.
    if "--so-rgb" in sys.argv:
        main(cameras=["head_camera", "right_wrist_camera"])
    else:
        main()