"""Avalia um checkpoint do ajuste fino em malha fechada: N posições da caneca sorteadas com semente diferente das demos,
cada uma com ../base/rodar_mujoco.py --ft. Sucesso = caneca a mais de 50 mm do tampo no fim, 2+ elos da mão em contato,
inclinação máxima abaixo de 15°. Servidor (../task3/servidor_task3.py com PSI0_SEM_CORPO=1) escutando em --porta.
Uso: MUJOCO_GL=egl python avalia.py saida/aval_ckpt4000 --porta 8779 --n 10 --semente 99
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

AQUI = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("--porta", type=int, default=8779); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--semente", type=int, default=99); ap.add_argument("--blocos", type=int, default=20)
    ap.add_argument("--x", type=float, nargs=2, default=[0.24, 0.28]); ap.add_argument("--y", type=float, nargs=2, default=[-0.22, -0.07])
    ap.add_argument("--yaw", type=float, default=20)
    ap.add_argument("--cintura", action="store_true", help="checkpoint com a cintura (v3): mesa a 0,15, região x 0,24-0,30, y -0,22 a -0,04")
    ap.add_argument("--humano", action="store_true", help="com --cintura: demos de movimento humano (v4), mesa a 0,10")
    a = ap.parse_args(); afasta = "0.05"
    if a.cintura: afasta, a.x, a.y = ("0.10" if a.humano else "0.15"), [0.24, 0.30], [-0.22, -0.04]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.semente); res = []
    for i in range(a.n):
        x, y, yaw = rng.uniform(*a.x), rng.uniform(*a.y), 90 + rng.uniform(-a.yaw, a.yaw)
        ep = out / f"ep{i:02d}"
        p = subprocess.run([sys.executable, str(AQUI.parent / "base" / "rodar_mujoco.py"), "--out", str(ep), "--ft", "--executa", "15",
                            "--blocos", str(a.blocos), "--afasta", afasta, "--porta", str(a.porta), "--copo", f"{x:.4f}", f"{y:.4f}",
                            "--yaw-copo", f"{yaw:.1f}"] + (["--cintura"] if a.cintura else []), capture_output=True, text=True)
        linha = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if not linha: print(p.stdout[-2000:], p.stderr[-2000:]); raise SystemExit(f"ep{i:02d} falhou")
        r = dict(json.loads(linha[-1]), xy=[round(x, 4), round(y, 4)], yaw=round(yaw, 1))
        r["sucesso"] = bool(r["subida_final_mm"] > 50 and r["dedos_final"] >= 2 and r["inclinacao_max_deg"] < 15); res.append(r)
        print(f"ep{i:02d} xy=({x:.3f},{y:.3f}) yaw={yaw:.0f} subida_final={r['subida_final_mm']:.0f}mm max={r['subida_max_mm']:.0f}mm "
              f"dedos={r['dedos_final']} incl_max={r['inclinacao_max_deg']:.1f} sucesso={r['sucesso']}", flush=True)
        (out / "resumo.json").write_text(json.dumps(res, indent=1))
    print(f"sucesso {sum(r['sucesso'] for r in res)}/{len(res)}")


if __name__ == "__main__":
    main()
