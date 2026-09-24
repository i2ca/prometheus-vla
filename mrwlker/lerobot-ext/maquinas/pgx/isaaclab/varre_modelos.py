#!/usr/bin/env python
"""Varre checkpoints publicados contra o simulador da Unitree, sem ninguém olhando.

Para cada checkpoint: baixa, remenda o config se precisar, sobe a ponte, dá três tentativas
(reset a cada 2 min) e mede o que importa — o cubo saiu do lugar? a garra fechou? Escreve o
resultado em `~/testes_g1/RESULTADOS.md` DEPOIS DE CADA MODELO, para que uma interrupção no meio
não perca o que já foi medido.

    python ~/testes_g1/varre_modelos.py                 # a lista inteira
    python ~/testes_g1/varre_modelos.py --minutos 3     # mais rápido, para conferir a mecânica
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/mrwlker/DEV/prometheus-vla/lerobot-ext/pgx")
import numpy as np

CASA = Path.home()
SAIDA = CASA / "testes_g1"
RESULTADOS = SAIDA / "RESULTADOS.md"
CKPTS = CASA / "ckpts_remendados"
PONTE = "/home/mrwlker/sobe_politica.sh"
SOBE_SIM = "/home/mrwlker/sobe_sim_unitree.sh"

# (apelido, repo no HF, argumentos extras da ponte)
CONFIGS = [
    # O melhor de hoje, e duas variações de quanto ele executa antes de replanejar.
    ("act_oficial_lead50", "binabik-ai/act_PickPlaceRedBlock", ["--redimensiona", "224"]),
    ("act_oficial_lead10", "binabik-ai/act_PickPlaceRedBlock", ["--redimensiona", "224", "--passos-acao", "10"]),
    ("act_oficial_ensemble", "binabik-ai/act_PickPlaceRedBlock", ["--redimensiona", "224", "--ensemble", "0.01"]),
    # Os que ainda não foram testados.
    ("act_baseline_30k", "RooibosT/Sim_act_dex1_baseline", []),
    ("pi05_107demo_abs", "RooibosT/Sim_pi05_expert_only_absolute_107demo", []),
    ("pi05_107demo_rel", "RooibosT/Sim_pi05_expert_only_relative_107demo", []),
    ("pi05_abs", "RooibosT/Sim_pi05_expert_only_absolute", []),
    ("pi05_rel", "RooibosT/Sim_pi05_expert_only_relative", []),
]


def agora() -> str:
    return datetime.now().strftime("%H:%M:%S")


def prepara(repo: str) -> str:
    """Baixa e, se o checkpoint não carregar como está, devolve uma cópia remendada.

    Dois remendos conhecidos, os dois porque o treino usou um LeRobot diferente do nosso:
      * `return_uint8_images` no config das π0.5 do RooibosT — campo que a nossa PI05Config não tem;
      * `binabik_act`, que é ACT do LeRobot mais um resize para 224 embutido no pré-processador.
        Vira `act` puro, e o resize passa a ser feito pela ponte com `--redimensiona 224`.
    """
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(repo))
    cfg = json.load(open(d / "config.json"))
    precisa = "return_uint8_images" in cfg or cfg.get("type") == "binabik_act"
    if not precisa:
        return repo

    CKPTS.mkdir(parents=True, exist_ok=True)
    dest = CKPTS / repo.replace("/", "__")
    dest.mkdir(exist_ok=True)
    for f in d.iterdir():
        alvo = dest / f.name
        if alvo.is_symlink() or alvo.exists():
            alvo.unlink()
        if f.name == "config.json":
            c = dict(cfg)
            c.pop("return_uint8_images", None)
            if c.get("type") == "binabik_act":
                c["type"] = "act"
                for k in [k for k in list(c) if k.startswith("resize_")]:
                    c.pop(k)
            json.dump(c, open(alvo, "w"), indent=2)
        elif f.name == "policy_preprocessor.json":
            pp = json.load(open(f))
            pp["steps"] = [s for s in pp["steps"] if s["registry_name"] != "binabik_resize_with_pad"]
            json.dump(pp, open(alvo, "w"), indent=2)
        else:
            alvo.symlink_to(f.resolve())
    # o normalizador guarda o índice do passo no nome; tirar um passo desloca a numeração
    pp = json.load(open(dest / "policy_preprocessor.json"))
    idx = [i for i, s in enumerate(pp["steps"]) if s["registry_name"] == "normalizer_processor"]
    if idx:
        certo = dest / f"policy_preprocessor_step_{idx[0]}_normalizer_processor.safetensors"
        if not certo.exists():
            for f in dest.iterdir():
                if "normalizer_processor.safetensors" in f.name and "postprocessor" not in f.name:
                    certo.symlink_to(f.resolve())
                    break
    return str(dest)


def sim_vivo() -> bool:
    return subprocess.run(["pgrep", "-f", "sim_mai[n].py"], capture_output=True).returncode == 0


def garante_sim() -> None:
    """O simulador tem que estar de pé e com as três câmeras publicando."""
    if sim_vivo():
        return
    print(f"[{agora()}] simulador fora do ar — subindo", flush=True)
    with open(CASA / "unitree_sim.log", "w") as log:
        subprocess.Popen([SOBE_SIM], stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    limite = time.time() + 600
    while time.time() < limite:
        s = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
        if all(str(p) in s for p in (55555, 55556, 55557)):
            time.sleep(10)
            return
        time.sleep(10)
    raise RuntimeError("o simulador não subiu em 10 min")


def mede(minutos: float, reset, cams, estado):
    """Roda o relógio medindo cubo e garra, com reset a cada 2 min (três tentativas)."""
    def cubo(rgb):
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        m = (r > 120) & (r - g > 60) & (r - b > 60)
        if m.sum() < 40:
            return None, 0
        ys, xs = np.nonzero(m)
        return np.array([xs.mean(), ys.mean()]), int(m.sum())

    tentativas = []
    for t in range(3):
        reset()
        time.sleep(4.0)
        p0, a0 = cubo(cams["cam_left_high"].le())
        t0 = time.time()
        desloc_max, garra_min, areas = 0.0, 9.9, []
        while time.time() - t0 < minutos * 60 / 3:
            p, a = cubo(cams["cam_left_high"].le())
            if p is not None and p0 is not None:
                desloc_max = max(desloc_max, float(np.linalg.norm(p - p0)))
                areas.append(a)
            gd = estado["garra_d"]
            if gd is not None and len(gd.states):
                garra_min = min(garra_min, float(gd.states[0].q))
            time.sleep(1.5)
        tentativas.append({
            "deslocamento_px": round(desloc_max, 1),
            "garra_mais_fechada": round(garra_min, 2),
            "area_final_sobre_inicial": round(float(np.mean(areas[-5:]) / max(a0, 1)), 2) if areas else 0.0,
        })
        print(f"    tentativa {t + 1}: {tentativas[-1]}", flush=True)
    return tentativas


def escreve(nome, repo, args_extra, tentativas, obs=""):
    cabecalho = not RESULTADOS.exists()
    with open(RESULTADOS, "a") as f:
        if cabecalho:
            f.write("# Varredura de checkpoints no simulador da Unitree\n\n"
                    "Cada modelo teve **três tentativas**, com reset da cena entre elas.\n\n"
                    "* `deslocamento_px` — quanto o cubo andou na imagem da câmera da cabeça. "
                    "Abaixo de ~10 px é ruído.\n"
                    "* `garra_mais_fechada` — menor valor atingido pela garra direita (0 fechada, "
                    "5,4 aberta). Se não desce de ~3, ela nunca tentou agarrar.\n"
                    "* `area_final_sobre_inicial` — área vermelha no fim dividida pela do começo; "
                    "bem acima de 1 costuma ser cubo tombado ou mais perto da câmera.\n\n")
        f.write(f"\n## {nome}\n\n`{repo}`"
                + (f" com `{' '.join(args_extra)}`" if args_extra else "")
                + f" — {datetime.now():%d/%m %H:%M}\n\n")
        if obs:
            f.write(f"{obs}\n\n")
        if tentativas:
            f.write("| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |\n")
            f.write("|---|---|---|---|\n")
            for i, t in enumerate(tentativas, 1):
                f.write(f"| {i} | {t['deslocamento_px']} | {t['garra_mais_fechada']} | "
                        f"{t['area_final_sobre_inicial']} |\n")
            melhor = max(t["deslocamento_px"] for t in tentativas)
            fechou = min(t["garra_mais_fechada"] for t in tentativas)
            veredito = ("**mexeu o cubo e fechou a garra** — olhar de perto"
                        if melhor > 25 and fechou < 1.5 else
                        "mexeu o cubo, mas não fechou a garra" if melhor > 25 else
                        "fechou a garra, mas não mexeu o cubo" if fechou < 1.5 else
                        "não fez nada")
            f.write(f"\n{veredito}\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--minutos", type=float, default=6.0, help="por modelo, dividido em 3 tentativas")
    args = p.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)

    from roda_politica_g1_isaaclab import CAMERAS, CameraZMQ, abre_dds

    print(f"[{agora()}] varredura de {len(CONFIGS)} configurações, {args.minutos:g} min cada", flush=True)
    garante_sim()
    cams = {n: CameraZMQ(n, porta, "127.0.0.1") for n, porta in CAMERAS.items()}
    for c in cams.values():
        c.start()
    estado, _pub, _msg, _crc, _ge, _gd, reset_cena, _subs = abre_dds()
    limite = time.time() + 60
    while estado["corpo"] is None and time.time() < limite:
        time.sleep(0.5)

    for nome, repo, extra in CONFIGS:
        print(f"\n[{agora()}] ===== {nome} ({repo})", flush=True)
        proc = None
        try:
            garante_sim()
            caminho = prepara(repo)
            log = SAIDA / f"{nome}.log"
            with open(log, "w") as fl:
                proc = subprocess.Popen([PONTE, "--politica", caminho, "--saida", str(SAIDA / nome), *extra],
                                        stdout=fl, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            # a π0.5 leva uns minutos para carregar 7,5 GB
            limite = time.time() + 900
            pronto = False
            while time.time() < limite:
                txt = log.read_text(errors="ignore")
                if "laço infinito" in txt:
                    pronto = True
                    break
                if "Traceback" in txt or "❌" in txt:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(10)
            if not pronto:
                cauda = "\n".join(log.read_text(errors="ignore").splitlines()[-3:])
                escreve(nome, repo, extra, [], obs=f"não subiu:\n\n```\n{cauda}\n```")
                print(f"[{agora()}] {nome}: não subiu", flush=True)
                continue
            tentativas = mede(args.minutos, reset_cena, cams, estado)
            escreve(nome, repo, extra, tentativas)
        except Exception as e:                                    # noqa: BLE001
            escreve(nome, repo, extra, [], obs=f"erro na varredura: `{type(e).__name__}: {e}`")
            print(f"[{agora()}] {nome}: {type(e).__name__}: {e}", flush=True)
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                time.sleep(12)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            time.sleep(8)

    for c in cams.values():
        c.para()
    print(f"[{agora()}] varredura terminada — resultados em {RESULTADOS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
