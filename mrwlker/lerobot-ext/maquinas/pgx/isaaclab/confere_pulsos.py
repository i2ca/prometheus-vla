"""Qual porta ZMQ e o punho ESQUERDO? Mexe so o braco esquerdo e ve qual imagem muda."""
import sys, time
sys.path.insert(0, "/home/mrwlker/DEV/prometheus-vla/mrwlker/lerobot-ext/maquinas/pgx")
import numpy as np
from roda_politica_g1_isaaclab import CameraZMQ, abre_dds

cams = {"55556 (rotulada ESQ)": CameraZMQ("a", 55556, "127.0.0.1"),
        "55557 (rotulada DIR)": CameraZMQ("b", 55557, "127.0.0.1")}
for c in cams.values(): c.start()
estado, pub, msg, crc, ge, gd, reset_cena, _ = abre_dds()
t = time.time()
while estado["corpo"] is None and time.time() - t < 20: time.sleep(0.3)
if estado["corpo"] is None:
    raise SystemExit("sem lowstate: o simulador esta rodando?")
ls = estado["corpo"]; msg.mode_pr = 0; msg.mode_machine = ls.mode_machine
for j in range(29):
    msg.motor_cmd[j].mode = 1; msg.motor_cmd[j].q = ls.motor_state[j].q
while any(c.le() is None for c in cams.values()): time.sleep(0.3)

for junta, rotulo, alvo in ((16, "OMBRO ESQUERDO (roll)", 0.9), (23, "OMBRO DIREITO (roll)", -0.9)):
    antes = {k: c.le().astype(float) for k, c in cams.items()}
    t0 = time.time()
    while time.time() - t0 < 8:
        msg.motor_cmd[junta].q = alvo
        msg.crc = crc.Crc(msg); pub.Write(msg); time.sleep(0.033)
    depois = {k: c.le().astype(float) for k, c in cams.items()}
    print(f"mexendo {rotulo}:")
    for k in cams:
        print(f"   {k}: mudanca media de pixel {np.abs(depois[k]-antes[k]).mean():7.2f}")
    # devolve a junta
    msg.motor_cmd[junta].q = ls.motor_state[junta].q
    t0 = time.time()
    while time.time() - t0 < 5:
        msg.crc = crc.Crc(msg); pub.Write(msg); time.sleep(0.033)
for c in cams.values(): c.para()
