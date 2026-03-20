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
    # 1. INICIALIZA A INTEL REALSENSE D435i
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    # Define a resolução máxima para o mundo real
    WIDTH, HEIGHT, FPS = 640, 480, 30

    # Habilita as 4 lentes da RealSense
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS) # IR Esquerdo
    config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, FPS) # IR Direito

    try:
        profile = pipeline.start(config)
        
        # Pega a escala de profundidade real da câmera (geralmente 0.001 metros por unidade)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print("[RealSense D435i] 4 Lentes iniciadas com sucesso.")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    # ==========================================================
    # 2. INICIALIZA A CÂMERA DE FÁBRICA DO UNITREE G1
    # ==========================================================
    # No Linux do robô, geralmente é a porta 0. Se não abrir, tente 1, 2, ou 4.
    head_cam_index = 0 
    head_cap = cv2.VideoCapture(head_cam_index)
    
    if not head_cap.isOpened():
        print(f"[Aviso] Câmera de fábrica (Index {head_cam_index}) não encontrada. Tente outro index.")
    else:
        head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        head_cap.set(cv2.CAP_PROP_FPS, FPS)
        print("[Unitree G1 Head] Câmera de fábrica iniciada.")

    # ==========================================================
    # 3. INICIALIZA O SERVIDOR ZMQ
    # ==========================================================
    server = SensorServer()
    server.start_server(port=5555)
    print("[ZMQ] Servidor de Visão ativo na porta 5555. Aguardando LeRobot...")

    try:
        while True:
            # Puxa o frame da cabeça do G1
            ret, head_frame = head_cap.read()
            if not ret:
                # Se falhar, manda uma tela preta para não travar a IA
                head_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # Puxa os frames da RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)

            if not color_frame or not depth_frame or not ir_left_frame or not ir_right_frame:
                continue

            # ==========================================================
            # TRATAMENTO DAS IMAGENS (Padrão LeRobot)
            # ==========================================================
            
            # 1. RGB (Já vem pronto em BGR 3 canais)
            img_rgb = np.asanyarray(color_frame.get_data())

            # 2. Depth (Converte Milímetros -> Metros -> Escala de Cinza 8-bits -> 3 Canais)
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_raw * depth_scale
            depth_clipped = np.clip(depth_meters, 0.0, 3.0) # Limita a visão até 3 metros
            depth_8u = (depth_clipped * (255.0 / 3.0)).astype(np.uint8)
            img_depth = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR) # Triplica os canais

            # 3. IRs (Converte 8-bits 1 canal -> 3 Canais)
            ir_left_raw = np.asanyarray(ir_left_frame.get_data())
            img_ir_left = cv2.cvtColor(ir_left_raw, cv2.COLOR_GRAY2BGR)

            ir_right_raw = np.asanyarray(ir_right_frame.get_data())
            img_ir_right = cv2.cvtColor(ir_right_raw, cv2.COLOR_GRAY2BGR)

            # ==========================================================
            # EMPACOTAMENTO E ENVIO
            # ==========================================================
            current_time = time.time()
            
            message = {
                "images": {
                    "head_camera": ImageUtils.encode_image(head_frame),
                    "d435i_rgb": ImageUtils.encode_image(img_rgb),
                    "d435i_depth": ImageUtils.encode_image(img_depth),
                    "d435i_ir_left": ImageUtils.encode_image(img_ir_left),
                    "d435i_ir_right": ImageUtils.encode_image(img_ir_right)
                },
                "timestamps": {
                    "head_camera": current_time,
                    "d435i_rgb": current_time,
                    "d435i_depth": current_time,
                    "d435i_ir_left": current_time,
                    "d435i_ir_right": current_time
                }
            }

            server.send_message(message)

    except KeyboardInterrupt:
        print("\nEncerrando transmissão...")

    finally:
        pipeline.stop()
        if head_cap.isOpened():
            head_cap.release()
        server.stop_server()

if __name__ == "__main__":
    start_real_robot_cameras()