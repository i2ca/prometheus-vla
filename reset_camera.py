import pyrealsense2 as rs
import time

print("🔍 Procurando câmera RealSense embutida...")
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("❌ Nenhuma câmera encontrada! O barramento USB perdeu a câmera.")
else:
    for dev in devices:
        nome = dev.get_info(rs.camera_info.name)
        print(f"⚡ Disparando hardware reset em: {nome}")
        dev.hardware_reset()
    
    print("✅ Comando enviado! Aguarde 10 segundos para o G1 religar a câmera internamente...")
