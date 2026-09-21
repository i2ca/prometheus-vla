"""Rastreia o centro do cubo vermelho e a garra ao longo do tempo. Mede se a tarefa anda."""
import sys, time
sys.path.insert(0, "/home/mrwlker/DEV/prometheus-vla/lerobot-ext/pgx")
import numpy as np
from roda_politica_g1_isaaclab import CameraZMQ, abre_dds

SEG = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
c = CameraZMQ("cam_left_high", 55555, "127.0.0.1"); c.start()
estado, *_ = abre_dds()
t = time.time()
while (c.le() is None or estado["corpo"] is None) and time.time() - t < 20: time.sleep(0.3)

def cubo(rgb):
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    m = (r > 120) & (r - g > 60) & (r - b > 60)
    if m.sum() < 40: return None, 0
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())), int(m.sum())

hist = []
t0 = time.time()
while time.time() - t0 < SEG:
    p, area = cubo(c.le())
    gd = estado["garra_d"].states[0].q if estado["garra_d"] else -1
    if p: hist.append((time.time() - t0, p[0], p[1], area, gd))
    time.sleep(2.0)
c.para()
H = np.array(hist)
print(f"{'t(s)':>5} {'x(px)':>7} {'y(px)':>7} {'area':>6} {'garra_dir':>10}")
for l in H[::max(1, len(H)//12)]:
    print(f"{l[0]:>5.0f} {l[1]:>7.0f} {l[2]:>7.0f} {l[3]:>6.0f} {l[4]:>10.2f}")
print(f"\ndeslocamento do cubo: {np.hypot(H[-1,1]-H[0,1], H[-1,2]-H[0,2]):.1f} px"
      f" | maior variacao: x {H[:,1].max()-H[:,1].min():.0f} px, y {H[:,2].max()-H[:,2].min():.0f} px")
print(f"garra direita: min {H[:,4].min():.2f}  max {H[:,4].max():.2f}")
