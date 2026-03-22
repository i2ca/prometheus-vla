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
    # CONFIGURAÇÕES DE RESOLUÇÃO HÍBRIDA
    # ==========================================================
    # Câmera da cabeça (Visão do VR) -> HD para nitidez
    HEAD_WIDTH, HEAD_HEIGHT = 1280, 720
    
    # Câmeras da RealSense (Inteligência Artificial) -> Padrão
    RS_WIDTH, RS_HEIGHT = 640, 480 
    
    FPS = 30

    # ==========================================================
    # 1. INICIALIZA A INTEL REALSENSE D435i
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    # Habilita as 4 lentes da RealSense com a resolução dela
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, 1, RS_WIDTH, RS_HEIGHT, rs.format.y8, FPS) # IR Esquerdo
    config.enable_stream(rs.stream.infrared, 2, RS_WIDTH, RS_HEIGHT, rs.format.y8, FPS) # IR Direito

    try:
        profile = pipeline.start(config)
        
        # Pega a escala de profundidade real da câmera
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        print(f"[RealSense D435i] 4 Lentes iniciadas em {RS_WIDTH}x{RS_HEIGHT}.")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    # ==========================================================
    # 2. INICIALIZA A CÂMERA DE FÁBRICA DO UNITREE G1 (VR)
    # ==========================================================
    head_cam_index = 0 
    head_cap = cv2.VideoCapture(head_cam_index)
    
    if not head_cap.isOpened():
        print(f"[Aviso] Câmera de fábrica (Index {head_cam_index}) não encontrada. Tente outro index.")
    else:
        # Força a câmera da cabeça a rodar em HD (720p)
        head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, HEAD_WIDTH)
        head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEAD_HEIGHT)
        head_cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Lê de volta para garantir que a câmera aceitou o comando HD
        actual_w = head_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = head_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Unitree G1 Head] Câmera iniciada em {int(actual_w)}x{int(actual_h)} para o VR.")

    # ==========================================================
    # 3. INICIALIZA O SERVIDOR ZMQ
    # ==========================================================
    server = SensorServer()
    server.start_server(port=5555)
    print("[ZMQ] Servidor de Visão ativo na porta 5555. Aguardando LeRobot...")

    try:
        while True:
            # Puxa o frame da cabeça do G1 (HD)
            ret, head_frame = head_cap.read()
            if not ret:
                # Manda tela preta do tamanho certo (HD) se falhar
                head_frame = np.zeros((HEAD_HEIGHT, HEAD_WIDTH, 3), dtype=np.uint8)

            # Puxa os frames da RealSense (Padrão 640x480)
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
            
            img_rgb = np.asanyarray(color_frame.get_data())

            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_raw * depth_scale
            depth_clipped = np.clip(depth_meters, 0.0, 3.0) 
            depth_8u = (depth_clipped * (255.0 / 3.0)).astype(np.uint8)
            img_depth = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR) 

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