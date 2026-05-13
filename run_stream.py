import pyrealsense2 as rs
import cv2
import numpy as np
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ==========================================================
# 1. CONFIGURAÇÃO DA CÂMERA D435 (Foco em RGB para OBS)
# ==========================================================
pipeline = rs.pipeline()
config = rs.config()

# Trava exclusivamente na câmera secundária (D435)
config.enable_device('141722078588')

# Apenas RGB ativado (640x480 a 30 FPS é o limite seguro para USB 2.0)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

print("Iniciando câmera D435 para o OBS...")
pipeline.start(config)

# ==========================================================
# 2. SERVIDOR HTTP MJPEG (Nativo para OBS Studio)
# ==========================================================
class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            while True:
                try:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    # Converte frame para imagem OpenCV
                    img = np.asanyarray(color_frame.get_data())
                    
                    # Comprime a imagem em JPEG para streaming leve
                    ret, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    
                    # Envia pela rede
                    self.wfile.write(b'--jpgboundary\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(jpg.size))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b'\r\n')
                except Exception as e:
                    break
        else:
            self.send_response(404)
            self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Lida com requests em threads separadas."""

if __name__ == '__main__':
    PORT = 8080
    server = ThreadedHTTPServer(('0.0.0.0', PORT), CamHandler)
    print(f"🎬 Servidor OBS Online!")
    print(f"👉 Coloque no OBS: http://IP_DO_ROBO:{PORT}/cam.mjpg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pipeline.stop()
        server.socket.close()