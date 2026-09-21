"""Tira de quadros da camera da cabeca, para ver o que a politica esta fazendo."""
import sys, time
sys.path.insert(0, "/home/mrwlker/DEV/prometheus-vla/lerobot-ext/pgx")
import numpy as np
from PIL import Image
from roda_politica_g1_isaaclab import CameraZMQ
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
espera = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
saida = sys.argv[3] if len(sys.argv) > 3 else "/home/mrwlker/testes_g1/tira.png"
c = CameraZMQ("cam_left_high", 55555, "127.0.0.1"); c.start()
t = time.time()
while c.le() is None and time.time() - t < 15: time.sleep(0.3)
qs = []
for i in range(n):
    q = c.le()
    if q is not None: qs.append(q)
    if i < n - 1: time.sleep(espera)
c.para()
Image.fromarray(np.hstack(qs)).resize((450 * len(qs), 340)).save(saida)
print(f"{len(qs)} quadros, {espera:g} s entre eles -> {saida}")
