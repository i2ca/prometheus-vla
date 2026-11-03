"""Ψ0 ajustado (v24, multicâmera, cintura) em malha fechada no MuJoCo, repetindo a pega da caneca, com vídeo ao vivo em MJPEG.

Mesmo laço e mesmo limitador de ./rodar_mujoco.py; o modelo real responde pelo servidor (porta --porta). A cada episódio a caneca
cai numa posição nova da região de treino da v24. Esquerda: câmera global; direita: as 3 imagens que a policy recebeu no último bloco.
Uso (na atena): MUJOCO_GL=egl python ao_vivo.py --porta 8780 --http 8800   ->   http://127.0.0.1:8800/
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, threading, time, multiprocessing as mp
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, mujoco, cv2
from rodar_mujoco import Sim, CENA, JUNTAS, CINTURA, POSE_HE, monta, pede, render_multicam

QUADRO = [None]


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/frame.jpg"):   # um JPEG por requisição: a página faz polling, sem multipart
            b = QUADRO[0] or b""; self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            try: self.wfile.write(b)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
        try:
            while True:
                if QUADRO[0] is not None: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(QUADRO[0]) + QUADRO[0] + b"\r\n")
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError): pass
    def log_message(self, *a): pass


def publica(rend, d, imagens, txt):
    rend.update_scene(d, camera="global_view"); canvas = np.zeros((720, 1280, 3), np.uint8); canvas[:, :960] = rend.render()
    for i, n in enumerate(("head", "left_wrist", "right_wrist")):
        if imagens: canvas[240 * i:240 * (i + 1), 960:] = cv2.resize(imagens[n], (320, 240)); cv2.putText(canvas, n, (968, 240 * i + 20), 0, 0.6, (255, 255, 0), 2)
    for i, t in enumerate(txt): cv2.putText(canvas, t, (12, 28 + 28 * i), 0, 0.75, (255, 255, 255), 2)
    QUADRO[0] = cv2.imencode(".jpg", canvas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def episodio(a, n, rng, ultimo):
    x, y, yaw = rng.uniform(*a.x), rng.uniform(*a.y), 90 + rng.uniform(-a.yaw, a.yaw)
    sim = Sim(CENA); m, d = sim.m, sim.d
    monta(sim, dict(zip(JUNTAS, POSE_HE)), (x, y), a.afasta, yaw_copo=np.radians(yaw), render=False)
    rend = mujoco.Renderer(m, 720, 960); s = __import__("socket").create_connection(("127.0.0.1", a.porta))
    J = JUNTAS + CINTURA; alvo = sim.q(J).copy(); z0 = d.xpos[sim.copo][2]; n_sub = int(round((1 / 30) / m.opt.timestep))
    ALVO, SUB, INC, imagens = [], [], [], None
    for b in range(a.blocos):
        publica(rend, d, imagens, [f"psi0 ft v24 ckpt_10000 | ep {n} | bloco {b}/{a.blocos} | pensando...", ultimo])
        imagens = render_multicam(CENA, d.qpos.copy(), a.afasta)
        r = pede(s, imagens, sim.q(J).astype(np.float32), "", 0 if b == 0 else a.executa)
        passo_anterior = ALVO[-1] - ALVO[-2] if len(ALVO) > 1 else np.zeros_like(alvo); acoes = []
        for acao in r["acoes"][:, :len(J)]:
            passo = np.clip(acao - alvo, -2.5 / 30, 2.5 / 30); passo = np.clip(passo, passo_anterior - 35 / 900, passo_anterior + 35 / 900)
            alvo = alvo + passo; passo_anterior = passo; acoes.append(alvo.copy())
        for k in range(a.executa):
            t0 = time.time(); alvo = acoes[k]; sim.alvo(J, alvo); sim.passo(n_sub); ALVO.append(alvo.copy())
            SUB.append(1000 * float(d.xpos[sim.copo][2] - z0)); INC.append(float(np.degrees(np.arccos(np.clip(d.xmat[sim.copo][8], -1, 1)))))
            publica(rend, d, imagens, [f"psi0 ft v24 ckpt_10000 | ep {n} | bloco {b}/{a.blocos} | lat {r['lat_s']:.2f}s", f"caneca subiu {SUB[-1]:.0f} mm | inclinacao {INC[-1]:.1f} graus", ultimo])
            time.sleep(max(0, 1 / 30 - (time.time() - t0)))   # tempo real
    s.close(); cod = d.xpos[m.body("coador").id]; erro = 1000 * float(np.linalg.norm(d.xpos[sim.copo][:2] - np.array([cod[0] - 0.006, cod[1] - 0.020])))
    return f"ep {n-0}: subiu max {max(SUB):.0f} mm, final {SUB[-1]:.0f} mm, inclin max {max(INC):.1f}, erro XY {erro:.0f} mm | caneca ({x:.3f},{y:.3f})"


def main():
    mp.set_start_method("spawn")   # filhos de câmera não herdam o contexto EGL do renderizador da vista
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--porta", type=int, default=8780); ap.add_argument("--http", type=int, default=8800)
    ap.add_argument("--blocos", type=int, default=30); ap.add_argument("--executa", type=int, default=15); ap.add_argument("--afasta", type=float, default=0.10)
    ap.add_argument("--x", type=float, nargs=2, default=[0.26, 0.27]); ap.add_argument("--y", type=float, nargs=2, default=[-0.18, -0.16]); ap.add_argument("--yaw", type=float, default=20)
    ap.add_argument("--semente", type=int, default=int(time.time())); a = ap.parse_args()
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", a.http), H).serve_forever, daemon=True).start()
    rng = np.random.default_rng(a.semente); ultimo = "(sem episodio concluido)"; n = 0
    while True:
        ultimo = episodio(a, n, rng, ultimo); print(ultimo, flush=True); n += 1


if __name__ == "__main__":
    main()
