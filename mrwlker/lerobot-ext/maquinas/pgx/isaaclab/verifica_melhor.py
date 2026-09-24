#!/usr/bin/env python
"""Repete a melhor configuração da varredura, gravando vídeo e ouvindo a recompensa do simulador.

A varredura diz QUE algo aconteceu (55 px de deslocamento, área 1,86, garra fechada). Isto diz
O QUE aconteceu: grava a câmera da cabeça durante várias tentativas, marca o instante de cada
evento no log e guarda o mp4 para assistir.

    python ~/testes_g1/verifica_melhor.py --tentativas 8 --segundos 60
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/mrwlker/DEV/prometheus-vla/mrwlker/lerobot-ext/maquinas/pgx")
import av
import numpy as np

SAIDA = Path.home() / "testes_g1" / "verificacao"
PONTE = "/home/mrwlker/sobe_politica.sh"
CKPT = str(Path.home() / "ckpts_remendados" / "binabik-ai__act_PickPlaceRedBlock")
EXTRA = ["--redimensiona", "224", "--passos-acao", "10"]
FPS_VIDEO = 10


def cubo(rgb):
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    m = (r > 120) & (r - g > 60) & (r - b > 60)
    if m.sum() < 40:
        return None, 0
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]), int(m.sum())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tentativas", type=int, default=8)
    p.add_argument("--segundos", type=float, default=60.0)
    args = p.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)

    from roda_politica_g1_isaaclab import CameraZMQ, abre_dds
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    cam = CameraZMQ("cam_left_high", 55555, "127.0.0.1")
    cam.start()
    estado, *_resto, reset_cena, _subs = abre_dds()
    premios = {"max": 0.0, "n": 0}

    def ouve_premio(m):
        try:
            d = json.loads(m.data)
            v = float(max(d.get("rewards", [0.0])))
            premios["max"] = max(premios["max"], v)
            premios["n"] += 1
        except Exception:                                          # noqa: BLE001
            pass

    ChannelSubscriber("rt/rewards_state", String_).Init(ouve_premio, 10)

    log = SAIDA / "politica.log"
    with open(log, "w") as fl:
        proc = subprocess.Popen([PONTE, "--politica", CKPT, "--saida", str(SAIDA / "corrida"), *EXTRA],
                                stdout=fl, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    limite = time.time() + 600
    while time.time() < limite and "laço infinito" not in log.read_text(errors="ignore"):
        if proc.poll() is not None:
            raise SystemExit("a ponte morreu ao subir; veja " + str(log))
        time.sleep(5)
    print(f"[{datetime.now():%H:%M:%S}] política de pé", flush=True)

    while cam.le() is None:
        time.sleep(0.5)
    cont = av.open(str(SAIDA / "tentativas.mp4"), mode="w")
    fluxo = cont.add_stream("libx264", rate=FPS_VIDEO)
    alt, larg = cam.le().shape[:2]
    fluxo.width, fluxo.height, fluxo.pix_fmt = larg - larg % 2, alt - alt % 2, "yuv420p"

    resumo = []
    t_video = 0.0
    for n in range(args.tentativas):
        reset_cena()
        time.sleep(4.0)
        premios["max"] = 0.0
        p0, a0 = cubo(cam.le())
        t0 = time.time()
        pico, area_pico, t_pico = 0.0, a0, 0.0
        prox_quadro = 0.0
        while time.time() - t0 < args.segundos:
            q = cam.le()
            if time.time() - t0 >= prox_quadro:
                vq = av.VideoFrame.from_ndarray(np.ascontiguousarray(q[:fluxo.height, :fluxo.width]), format="rgb24")
                for pac in fluxo.encode(vq):
                    cont.mux(pac)
                prox_quadro += 1 / FPS_VIDEO
                t_video += 1 / FPS_VIDEO
            pt, ar = cubo(q)
            if pt is not None and p0 is not None:
                d = float(np.linalg.norm(pt - p0))
                if d > pico:
                    pico, area_pico, t_pico = d, ar, time.time() - t0
            time.sleep(0.05)
        linha = {"tentativa": n + 1, "pico_px": round(pico, 1), "em_s": round(t_pico, 1),
                 "area_no_pico": round(area_pico / max(a0, 1), 2),
                 "recompensa_max": round(premios["max"], 3),
                 "video_em_s": round(t_video - args.segundos, 1)}
        resumo.append(linha)
        print(f"[{datetime.now():%H:%M:%S}] {linha}", flush=True)

    for pac in fluxo.encode():
        cont.mux(pac)
    cont.close()
    cam.para()
    proc.terminate()

    with open(SAIDA / "RESUMO.md", "w") as f:
        f.write(f"# Verificação do `act_oficial_lead10` — {datetime.now():%d/%m %H:%M}\n\n")
        f.write(f"{args.tentativas} tentativas de {args.segundos:g} s. Vídeo: `tentativas.mp4` "
                f"(a coluna *começa em* diz onde procurar cada tentativa no vídeo).\n\n")
        f.write("| tentativa | pico (px) | no instante | área no pico | recompensa | começa em (s) |\n|---|---|---|---|---|---|\n")
        for l in resumo:
            f.write(f"| {l['tentativa']} | {l['pico_px']} | {l['em_s']} s | {l['area_no_pico']} | "
                    f"{l['recompensa_max']} | {l['video_em_s']} |\n")
        bons = [l for l in resumo if l["pico_px"] > 25]
        f.write(f"\n**{len(bons)} de {len(resumo)}** tentativas moveram o cubo mais de 25 px.\n")
    print(f"resumo em {SAIDA / 'RESUMO.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
