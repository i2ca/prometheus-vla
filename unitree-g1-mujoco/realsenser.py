#!/usr/bin/env python3
"""
Live camera viewer for MuJoCo simulator using Pure OpenCV
Ultra-Fast - 30+ FPS with multiple windows
"""
import argparse
import sys
import time
from pathlib import Path

# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from sim.sensor_utils import SensorClient, ImageUtils

class CameraViewer:
    def __init__(self, host, port):
        self.client = SensorClient()
        self.client.start_client(server_ip=host, port=port)
        
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        
    def _process_image(self, cam_name, img_data):
        """Decode and process image based on camera type"""
        if isinstance(img_data, str):
            try:
                # O simulador e a câmera real agora mandam TUDO em JPG 8-bits 3-canais (RGB)
                img = ImageUtils.decode_image(img_data)
            except Exception:
                return None
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            return None

        if img is None or not isinstance(img, np.ndarray):
            return None

        # ==========================================================
        # RENDERIZAÇÃO ULTRA-RÁPIDA (OpenCV)
        # ==========================================================
        # Como a imagem (tanto RGB quanto Depth) já vem colorida e pronta da fonte, 
        # nós só precisamos inverter de RGB (Padrão IA) para BGR (Padrão Monitor/OpenCV).
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_out = img
            
        return img_out

    def start(self):
        """Start the live viewer with OpenCV"""
        print("⏳ Aguardando conexão com as lentes do simulador...")
        
        # Espera o primeiro frame para ter certeza que conectou
        data = self.client.receive_message()
        if not data or "images" not in data:
            print("❌ Nenhuma câmera encontrada no stream!")
            return

        print(f"\n{'='*60}")
        print("📹 Live Multi-Camera Viewer (OPENCV TURBO) started!")
        print("Pressione a tecla 'Q' com qualquer janela focada para sair.")
        print(f"{'='*60}\n")
        
        try:
            while True:
                data = self.client.receive_message()
                
                # Cálculo de FPS contínuo
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time

                images_dict = data.get("images", {})
                
                for cam_name, img_data in images_dict.items():
                    img_bgr = self._process_image(cam_name, img_data)
                    if img_bgr is None:
                        continue

                    # Se for a câmera HD, encolhe pela metade só para caber melhor na tela do PC
                    if cam_name == "head_camera":
                        img_bgr = cv2.resize(img_bgr, (640, 360))
                    
                    # Desenha a tarja preta transparente e o texto verde do FPS
                    cv2.rectangle(img_bgr, (5, 5), (150, 45), (0, 0, 0), -1)
                    cv2.putText(img_bgr, f'FPS: {self.fps:.1f}', (15, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Renderiza a janela na hora!
                    cv2.imshow(f"Sensor: {cam_name}", img_bgr)
                
                # Permite framerate ilimitado aguardando tecla de escape (Q)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Tecla 'Q' pressionada. Encerrando...")
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping viewer...")
        finally:
            self.client.stop_client()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Live camera viewer for MuJoCo simulator")
    parser.add_argument("--host", type=str, default="localhost", help="Simulator host address")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    args = parser.parse_args()
    
    viewer = CameraViewer(host=args.host, port=args.port)
    viewer.start()

if __name__ == "__main__":
    main()