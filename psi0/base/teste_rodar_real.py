"""Teste sem robô nem GPU: robô falso + servidor falso falando o protocolo do servidor_task3.py. Rode: python teste_rodar_real.py"""
import argparse, pickle, socket, struct, threading, tempfile
import numpy as np
import rodar_real as R

N = 28; nomes = [f"j{i}.q" for i in range(N)]
class Robo:
    def __init__(s): s.q = np.zeros(N); s.enviado = []
    def get_observation(s): return {**{n: float(v) for n, v in zip(nomes, s.q)}, "head_camera": np.random.randint(0, 255, (480, 848, 3), np.uint8)}
    def send_action(s, a): s.enviado.append(a); s.q = np.array([a[n] for n in nomes])
def servidor(srv, vistos):
    c, _ = srv.accept()
    while True:
        try: n = struct.unpack("!I", c.recv(4, socket.MSG_WAITALL))[0]
        except struct.error: return
        buf = b""
        while len(buf) < n: buf += c.recv(n - len(buf))
        q = pickle.loads(buf); vistos.append(q)
        b = pickle.dumps({"acoes": np.full((30, 31), 1.0, np.float32), "lat_s": 0.1}); c.sendall(struct.pack("!I", len(b)) + b)
srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1); vistos = []
threading.Thread(target=servidor, args=(srv, vistos), daemon=True).start()
def a(enviar): return argparse.Namespace(out=tempfile.mkdtemp(), blocos=3, executa=15, fps=1000, cintura=True, punho="preto", bgr=False, enviar=enviar)
s = socket.create_connection(srv.getsockname())
r = Robo(); R.rodar(r, s, a(False), nomes, log=lambda *x: None)
assert not r.enviado and len(vistos) == 3                      # dry-run não envia nada
q = vistos[0]; assert q["primeiro"] and vistos[1]["k"] == 15 and q["state"].shape == (31,)
assert q["images"]["head"].shape == (480, 640, 3) and q["images"]["left_wrist"].max() == 0
import builtins; builtins.input = lambda *_: ""
r = Robo(); R.rodar(r, s, a(True), nomes, log=lambda *x: None)
q = np.array([[x[n] for n in nomes] for x in r.enviado])
assert len(q) == 45 and np.abs(np.diff(q, axis=0)).max() <= R.MAX_VEL + 1e-6 and q[0].max() <= R.MAX_VEL + 1e-6   # rampa a partir do estado medido
print("ok: dry-run mudo, protocolo certo, velocidade limitada, parte do estado medido")
