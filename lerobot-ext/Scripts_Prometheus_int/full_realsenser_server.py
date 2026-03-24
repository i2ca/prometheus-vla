import pyrealsense2 as rs
import numpy as np
import cv2
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
    # Câmera RGB da RealSense (Agora é a nossa head_camera oficial) -> HD
    HEAD_WIDTH, HEAD_HEIGHT = 640, 480
    
    # Câmera de Profundidade (Depth)
    DEPTH_WIDTH, DEPTH_HEIGHT = 640, 480 
    
    FPS = 30

    # ==========================================================
    # 1. INICIALIZA A INTEL REALSENSE D435i
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    # Habilita apenas RGB e Depth (Sem as lentes IR para economizar USB/CPU)
    config.enable_stream(rs.stream.color, HEAD_WIDTH, HEAD_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
        
        # Pega a escala de profundidade real da câmera
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print(f"[RealSense D435i] Iniciada com sucesso!")
        print(f" -> RGB (head_camera): {HEAD_WIDTH}x{HEAD_HEIGHT} (Visão do VR)")
        print(f" -> Depth (d435i_depth): {DEPTH_WIDTH}x{DEPTH_HEIGHT} (Visão da IA)")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    # ==========================================================
    # 2. INICIALIZA O SERVIDOR ZMQ
    # ==========================================================
    server = SensorServer()
    server.start_server(port=5555)
    print("[ZMQ] Servidor de Visão ativo na porta 5555. Aguardando LeRobot...")

    try:
        while True:
            # Puxa os frames sincronizados da RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # ==========================================================
            # TRATAMENTO DAS IMAGENS
            # ==========================================================
            
            # RGB assume a identidade da head_camera
            img_rgb = np.asanyarray(color_frame.get_data())

            # Profundidade (Convertendo para o formato visual de 3 canais)
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_raw * depth_scale
            depth_clipped = np.clip(depth_meters, 0.0, 3.0) 
            depth_8u = (depth_clipped * (255.0 / 3.0)).astype(np.uint8)
            img_depth = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR) 

            # ==========================================================
            # EMPACOTAMENTO E ENVIO
            # ==========================================================
            current_time = time.time()
            
            message = {
                "images": {
                    "head_camera": ImageUtils.encode_image(img_rgb),
                    "head_camera_depth": ImageUtils.encode_image(img_depth)
                },
                "timestamps": {
                    "head_camera": current_time,
                    "head_camera_depth": current_time
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