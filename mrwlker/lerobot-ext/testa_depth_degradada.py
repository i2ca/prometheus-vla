#!/usr/bin/env python
"""
A profundidade degradada do MuJoCo derruba o modelo?
=====================================================
Mesmo episódio, mesmo servidor, duas entradas: profundidade MÉTRICA (como o
modelo viu no treino) e profundidade DEGRADADA (como o cliente ao vivo entrega,
via `--depth-legado`).

Motiva este teste: o replay contra o dataset dá erro de ~2°, mas ao vivo no
MuJoCo o robô não vai até a xícara. A diferença entre os dois caminhos é
sobretudo a profundidade — o `converte_depth_legado` do cliente quantiza
0-2000 mm em 255 degraus (~7,8 mm) E SATURA em 2 m, enquanto o dataset chega a
2300 mm. Se o erro explodir na coluna degradada, achamos a causa; se não
mudar, a causa está em outro lugar (malha fechada, enquadramento da cena).

Uso (na athena, servidor no ar):
  python testa_depth_degradada.py --episodios=100,300 --root=<dataset>
"""
import sys

import zmq
import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def degrada(mm: np.ndarray) -> np.ndarray:
    """Exatamente o caminho do cliente ao vivo: mm → cinza 8 bits → mm.

    O servidor de câmera do MuJoCo publica cinza 0-255 para 0-2000 mm; o
    `converte_depth_legado` desfaz multiplicando por 2000/255. As duas pontas
    juntas são uma quantização com saturação, e é isso que se reproduz aqui.
    """
    cinza = np.clip(mm.astype(np.float32) * (255.0 / 2000.0), 0, 255).astype(np.uint8)
    return (cinza.astype(np.float32) * (2000.0 / 255.0)).astype(np.uint16)


def _u8(t):
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def roda(ds, juntas, sock, ep, passo, degradar):
    idx = [i for i, e in enumerate(ds.hf_dataset["episode_index"]) if int(e) == ep]
    prev, grav, travadas = [], [], 0
    for k in range(0, len(idx) - 1, passo):
        am = ds[idx[k]]
        obs = {n: float(am["observation.state"][i]) for i, n in enumerate(juntas)}
        obs["head_camera"] = _u8(am["observation.images.head_camera"])
        obs["right_wrist_camera"] = _u8(am["observation.images.right_wrist_camera"])
        d = am["observation.images.head_camera_depth"].numpy().squeeze().astype(np.uint16)
        obs["head_camera_depth"] = degrada(d) if degradar else d

        sock.send(msgpack.packb(
            {"obs": obs, "obs_step": k, "actions_per_chunk": passo, "want_debug": False},
            default=m.encode))
        r = msgpack.unpackb(sock.recv(), object_hook=m.decode, raw=False)
        if "error" in r:
            raise SystemExit(f"servidor: {r['error']}")
        chunk = np.asarray(r["chunk_np"], dtype=np.float32)
        travadas += int(r.get("travadas", 0))
        for j in range(min(len(chunk), len(idx) - 1 - k)):
            alvo = ds[idx[k + j]]["action"].numpy()
            grav.append(alvo[0] if alvo.ndim > 1 else alvo)
            prev.append(chunk[j])
    return np.array(prev), np.array(grav), travadas


def main():
    eps, root, passo = [100, 300], None, 32
    server, port = "127.0.0.1", 5600
    for a in sys.argv[1:]:
        if a.startswith("--episodios="):
            eps = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--root="):
            root = a.split("=", 1)[1]
        elif a.startswith("--passo="):
            passo = int(a.split("=", 1)[1])
        elif a.startswith("--server="):
            server = a.split("=", 1)[1]

    ds = LeRobotDataset("x/y", root=root)
    juntas = ds.meta.features["observation.state"]["names"]
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 120000)
    sock.connect(f"tcp://{server}:{port}")

    # Grupos na ordem do dataset real: braços 0-13, cintura 14, mãos 15-28.
    grupos = {"braco esq": range(0, 7), "braco dir": range(7, 14),
              "cintura": range(14, 15), "mao dir": range(22, 29)}

    print(f"{'episódio':>9} {'profundidade':>14} {'erro médio':>12} {'braço dir':>11} {'mão dir':>10} {'travadas':>9}")
    print("-" * 72)
    for ep in eps:
        for degradar in (False, True):
            p, g, tr = roda(ds, juntas, sock, ep, passo, degradar)
            err = np.abs(p - g)
            geral = err.mean()
            bd = err[:, list(grupos["braco dir"])].mean()
            md = err[:, list(grupos["mao dir"])].mean()
            rot = "DEGRADADA" if degradar else "métrica"
            print(f"{ep:>9} {rot:>14} {geral:>9.4f} rad {bd:>8.4f} {md:>10.4f} {tr:>9}")
        print()


if __name__ == "__main__":
    main()
