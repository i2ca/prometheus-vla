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
                # O sensor_utils sabe qual método de descompressão usar baseado no nome
                if 'depth' in cam_name.lower():
                    img = ImageUtils.decode_depth_image(img_data)
                else:
                    img = ImageUtils.decode_image(img_data)
            except Exception:
                return None
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            return None

        if img is None or not isinstance(img, np.ndarray):
            return None

        # RENDERIZAÇÃO ULTRA-RÁPIDA (OpenCV)
        if 'depth' in cam_name.lower():
            # A imagem chegou em milímetros reais (uint16). 
            # Recorta visão para o limite de 3 metros (3000mm) para gerar um bom Heatmap.
            depth_clipped = np.clip(img, 0, 3000)
            depth_8u = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_out = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        else:
            # As câmeras RGB e IR já foram corrigidas na memória pelo image_publish_utils.py.
            # O OpenCV carrega JPGs do IR como BGR-Grayscale automaticamente, então é só passar reto.
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