import pyrealsense2 as rs
import numpy as np
import cv2
import zmq
import time
import sys
from pathlib import Path

# Garante que o Python ache o utils do seu projeto
sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import SensorServer, ImageUtils

def start_real_robot_cameras():
    # ==========================================================
    # CONFIGURAÇÕES DE RESOLUÇÃO
    # ==========================================================
    HEAD_WIDTH, HEAD_HEIGHT = 640, 480
    DEPTH_WIDTH, DEPTH_HEIGHT = 640, 480 
    FPS = 30

    # ==========================================================
    # 1. INICIALIZA A INTEL REALSENSE D435i
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, HEAD_WIDTH, HEAD_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print(f"[RealSense D435i] Iniciada com sucesso!")
        print(f" -> RGB (head_camera): {HEAD_WIDTH}x{HEAD_HEIGHT} (Visão do VR)")
        print(f" -> Depth (head_camera_depth): {DEPTH_WIDTH}x{DEPTH_HEIGHT} (Visão em Cores da IA)")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    # ==========================================================
    # 2. INICIALIZA O SERVIDOR ZMQ
    # ==========================================================
    server = SensorServer()
    # Enviamos TUDO pela mesma porta agora
    server.start_server(port=5555)

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("[ZMQ] Servidor de Visão ativo na porta 5555. Aguardando LeRobot...")

    try:
        time.sleep(2.0)
        
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print("[Aviso] Câmera engasgou (Timeout). Tentando novamente...")
                continue

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # --- TRATAMENTO RGB ---
            img_bgr = np.asanyarray(color_frame.get_data())
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # --- TRATAMENTO DEPTH (A MÁGICA DA COR) ---
            depth_raw = np.asanyarray(depth_frame.get_data())
            # Convertemos a profundidade crua (16-bit) para uma escala de 8-bit (0-255).
            # O alpha=255.0/2000.0 significa que o alcance ideal é até 2 metros (2000mm).
            # Tudo além de 2m ficará com a mesma cor "de fundo".
            depth_8bit = cv2.convertScaleAbs(depth_raw, alpha=255.0 / 2000.0)
            
            # Aplica o mapa de calor (Vermelho = perto, Azul = longe)
            depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            # Converte BGR para RGB para o LeRobot receber as cores corretas
            depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

            # ==========================================================
            # EMPACOTAMENTO E ENVIO
            # ==========================================================
            current_time = time.time()
            
            message = {
                "images": {
                    "head_camera": ImageUtils.encode_image(img_rgb),
                    "head_camera_depth": ImageUtils.encode_image(depth_rgb), # O LeRobot entende isso nativamente!
                },
                "timestamps": {
                    "head_camera": current_time,
                    "head_camera_depth": current_time,
                }
            }

            server.send_message(message)

    except KeyboardInterrupt:
        print("\nEncerrando transmissão...")
    finally:
        pipeline.stop()
        server.stop_server()

if __name__ == "__main__":
    start_real_robot_cameras()