#!/usr/bin/env python
"""O `rt/lowcmd` move mesmo o G1 do simulador MuJoCo? Sem política no meio.

Escrito em 16/09 porque, no teste do `pi05_unitree_g1`, a política emitia ação COM variação
(juntas oscilando 1,0 a 1,7 rad no `acoes.csv`) e mesmo assim o robô parecia parado no painel.
Isso tem duas explicações possíveis, e este script separa as duas:

  * o comando não chega/não é aplicado no simulador → o medido fica plano;
  * o comando é aplicado e o movimento simplesmente não aparece na câmera de terceira
    pessoa → o medido acompanha o comandado.

Comanda UMA junta (por padrão o cotovelo esquerdo, índice 18) com uma senoide larga e imprime,
lado a lado, o valor comandado e o `q` medido no `rt/lowstate`. As outras 28 juntas ficam
seguradas na pose inicial, como no teste real.

    python maquinas/pgx/testa_dds_sim.py                  # cotovelo esquerdo, ±0.8 rad
    python maquinas/pgx/testa_dds_sim.py --junta 20 --amplitude 1.0
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO_ENV = "lerobot/unitree-g1-mujoco"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--junta", type=int, default=18, help="índice da junta (18 = cotovelo esquerdo)")
    p.add_argument("--amplitude", type=float, default=0.8, help="rad")
    p.add_argument("--periodo", type=float, default=2.0, help="s por ciclo")
    p.add_argument("--segundos", type=float, default=6.0)
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    raiz = Path(snapshot_download(REPO_ENV))
    spec = importlib.util.spec_from_file_location("g1_mujoco_env", raiz / "env.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["g1_mujoco_env"] = modulo
    spec.loader.exec_module(modulo)
    # Sem câmera: aqui só interessa a junta. Evita também disputar a porta 5555.
    env = modulo.make_env(n_envs=1, use_async_envs=False, cameras=[], publish_images=False, onscreen=False)
    env.reset()

    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
    from unitree_sdk2py.utils.crc import CRC

    from lerobot.robots.unitree_g1.config_unitree_g1 import _DEFAULT_KD, _DEFAULT_KP

    pub = ChannelPublisher("rt/lowcmd", hg_LowCmd)
    pub.Init()
    estado = {"msg": None}
    sub = ChannelSubscriber("rt/lowstate", hg_LowState)
    sub.Init(lambda m: estado.__setitem__("msg", m), 1)

    limite = time.time() + 10.0
    while estado["msg"] is None:
        env.step()
        if time.time() > limite:
            raise SystemExit("❌ nenhum `rt/lowstate` em 10 s")
    lowstate = estado["msg"]

    msg, crc = unitree_hg_msg_dds__LowCmd_(), CRC()
    msg.mode_pr = 0
    msg.mode_machine = lowstate.mode_machine
    for j in range(29):
        msg.motor_cmd[j].mode = 1
        msg.motor_cmd[j].kp = _DEFAULT_KP[j]
        msg.motor_cmd[j].kd = _DEFAULT_KD[j]
        msg.motor_cmd[j].q = lowstate.motor_state[j].q
        msg.motor_cmd[j].qd = 0.0
        msg.motor_cmd[j].tau = 0.0

    q0 = lowstate.motor_state[args.junta].q
    print(f">>> junta {args.junta} | q inicial {q0:+.3f} | kp {_DEFAULT_KP[args.junta]} "
          f"kd {_DEFAULT_KD[args.junta]} | senoide ±{args.amplitude} rad a cada {args.periodo}s")

    dt, passos_fisica = 1.0 / 30.0, 8
    n = int(args.segundos * 30)
    erro_max, medidos = 0.0, []
    for passo in range(n):
        t = passo * dt
        alvo = q0 + args.amplitude * np.sin(2 * np.pi * t / args.periodo)
        msg.motor_cmd[args.junta].q = float(alvo)
        msg.crc = crc.Crc(msg)
        pub.Write(msg)
        for _ in range(passos_fisica):
            env.step()
        medido = estado["msg"].motor_state[args.junta].q
        medidos.append(medido)
        erro_max = max(erro_max, abs(alvo - medido))
        if passo % 10 == 0:
            print(f"    t={t:4.1f}s  comandado {alvo:+.3f}  medido {medido:+.3f}  erro {alvo - medido:+.3f}")

    medidos = np.array(medidos)
    amplitude_medida = medidos.max() - medidos.min()
    print(f">>> amplitude comandada {2 * args.amplitude:.3f} rad | medida {amplitude_medida:.3f} rad "
          f"| erro máximo de rastreio {erro_max:.3f} rad")
    if amplitude_medida < 0.05:
        print("❌ a junta NÃO se moveu: o `rt/lowcmd` não está sendo aplicado no simulador")
    elif amplitude_medida < args.amplitude:
        print("⚠️  moveu, mas bem menos que o comandado: ganho baixo ou algo segurando a junta")
    else:
        print("✅ a junta seguiu o comando: o caminho DDS → simulador funciona")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
