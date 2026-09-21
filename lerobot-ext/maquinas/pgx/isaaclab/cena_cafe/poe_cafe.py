#!/usr/bin/env python3
"""Poe o copo ou o coador numa posicao da mesa, com o simulador rodando.

As coordenadas do mundo nesta cena sao pouco intuitivas (-4,18 / -3,90), entao a linha de
comando fala em CENTIMETROS VISTOS DO ROBO:

    frente  quanto a frente do robo, em cm  (o robo olha para o -Y do mundo)
    lado    positivo = ESQUERDA do robo     (a esquerda do robo e o +X do mundo)

    ~/move_coador.sh                 # so mostra onde estao os dois
    ~/move_coador.sh 20 2            # 20 cm a frente, 2 cm a esquerda
    ~/move_coador.sh 25 -4 --giro 30 # girado 30 graus
    ~/move_coador.sh --altura 81.5   # so mexe na altura (cm), mantem x e y
    ~/move_coador.sh --mundo -4.18 -3.90

O arquivo escrito e `~/cena_cafe.json`. O simulador le ele a cada volta do laco
(`tools/cena_cafe_vivo.py`), entao o objeto se move na hora, sem reiniciar. Reiniciar tambem
funciona: a cena nasce com a mesma pose.
"""
import argparse
import importlib.util
import sys

# Carrego o `pose_cafe.py` PELO CAMINHO, e nao como `tasks.common_scene.pose_cafe`: o
# `tasks/__init__.py` do simulador puxa toml/isaaclab, que so existem no conda do IsaacSim.
# Assim este comando roda em qualquer python3 do sistema.
_spec = importlib.util.spec_from_file_location(
    "pose_cafe", "/home/mrwlker/DEV/unitree_sim_isaaclab/tasks/common_scene/pose_cafe.py")
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)
ARQUIVO, TAMPO = _pc.ARQUIVO, _pc.TAMPO
do_robo, escreve, le, para_mundo = _pc.do_robo, _pc.escreve, _pc.le, _pc.para_mundo


def mostra(nome: str) -> None:
    p = le(nome)
    frente, lado = do_robo(p["x"], p["y"])
    banda = "esquerda" if lado >= 0 else "direita"
    print(f"{nome:7s} frente {frente * 100:6.1f} cm | {abs(lado) * 100:5.1f} cm para a {banda}"
          f" | altura {p['z'] * 100:5.1f} cm | giro {p['giro']:.0f} graus"
          f"    (mundo x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("objeto", choices=["copo", "coador"])
    ap.add_argument("frente", nargs="?", type=float, help="cm a frente do robo")
    ap.add_argument("lado", nargs="?", type=float, help="cm para a esquerda (negativo = direita)")
    ap.add_argument("--altura", type=float, help="altura em cm (padrao: o tampo da mesa)")
    ap.add_argument("--giro", type=float, help="graus em torno do eixo vertical")
    ap.add_argument("--mundo", nargs=2, type=float, metavar=("X", "Y"),
                    help="coordenada do mundo em metros, em vez de frente/lado")
    args = ap.parse_args()

    p = le(args.objeto)
    if args.mundo:
        p["x"], p["y"] = args.mundo
    elif args.frente is not None:
        if args.lado is None:
            ap.error("faltou o lado: `frente lado`, os dois em cm")
        p["x"], p["y"] = para_mundo(args.frente / 100.0, args.lado / 100.0)
    elif args.altura is None and args.giro is None:
        for n in ("copo", "coador"):
            mostra(n)
        print(f"\narquivo: {ARQUIVO}   (tampo da mesa em {TAMPO * 100:.0f} cm)")
        return 0

    if args.altura is not None:
        p["z"] = args.altura / 100.0
    if args.giro is not None:
        p["giro"] = args.giro

    frente, lado = do_robo(p["x"], p["y"])
    if not 0.10 <= frente <= 0.60 or abs(lado) > 0.40:
        print(f"AVISO: frente {frente * 100:.0f} cm / lado {lado * 100:.0f} cm cai fora da faixa "
              f"que o braco alcanca na mesa (frente 10-60 cm, lado +-40 cm). Vou pôr assim mesmo.")
    if abs(p["z"] - TAMPO) > 0.15:
        print(f"AVISO: altura {p['z'] * 100:.0f} cm esta longe do tampo ({TAMPO * 100:.0f} cm).")

    escreve(args.objeto, p)
    mostra(args.objeto)
    print("aplicado no simulador em ate meio segundo (se ele estiver de pe).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
