#!/usr/bin/env python
"""GR00T N1.6 no simulador de G1 da NVIDIA, dentro da NOSSA cabine (painel 8090).

O `rollout_policy.py` deles roda N episódios e reinicia sozinho no fim de cada
um, sem painel. Aqui é a mesma cabine do `pontes/cabine/`: caixa de tarefa,
câmeras e estado no navegador, e o MCP (`mcp_prometheus.py`) funciona igual,
porque ele só fala com a cabine. O modelo é o servidor deles
(`run_gr00t_server.py`, porta 5555); o simulador é o MuJoCo deles, com o
controle de corpo inteiro deles.

    # terminal 1 — o modelo (ambiente .venv-gb10)
    cd ~/DEV/Isaac-GR00T && source .venv-gb10/bin/activate && \\
      GR00T_ATTN=sdpa python gr00t/eval/run_gr00t_server.py \\
        --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \\
        --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper

    # terminal 2 — simulador com janela + cabine em http://localhost:8090
    cd ~/DEV/Isaac-GR00T && GR00T_SIM_JANELA=1 MUJOCO_GL=glfw \\
      gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \\
      ~/DEV/prometheus-vla/lerobot-ext/pontes/groot/roda_groot_interativo.py

Os botões da cabine, aqui:
  tarefa            reinicia o episódio e roda com a frase enviada
  parar             congela o robô (não manda mais ação) até a próxima tarefa
  mudar xícara (ç)  reinicia o episódio: o robosuite sorteia a cena de novo

O reinício é LEVE (`hard_reset=False`): resorteia os objetos no mesmo modelo, e
a janela do MuJoCo continua aberta. Depende do remendo em
`gr00trobosuite/robosuite/environments/base.py`, que antes destruía a janela
em todo reset.

A frase vai no campo `annotation.*` da observação, por onde o ambiente entrega a
instrução ao modelo. Este checkpoint só viu UMA frase (`FRASE_TREINO`); outras
são fora da distribuição — servem para ver o quanto o texto pesa.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

os.environ.setdefault("GR00T_SIM_JANELA", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # lerobot-ext

from gr00t.eval.rollout_policy import get_gym_env  # noqa: E402
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402

from pontes.cabine.servidor import Cabine, sobe  # noqa: E402

FRASE_TREINO = "pick up the apple, walk left and place the apple on the plate."


def algum(v) -> bool:
    """Verdadeiro se qualquer elemento (em qualquer nível de aninhamento) for."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return any(algum(x) for x in v)
    return bool(v)


def troca_frase(obs: dict, frase: str) -> dict:
    """Põe a frase em todo campo `annotation.*`, no mesmo formato que veio."""
    for k, v in obs.items():
        if not k.startswith("annotation"):
            continue
        if isinstance(v, np.ndarray):
            # Sem herdar o dtype: um `<U60` cortaria uma frase mais longa sem avisar.
            obs[k] = np.array([frase] * len(v), dtype=object if v.dtype == object else None)
        elif isinstance(v, (list, tuple)):
            obs[k] = type(v)([frase] * len(v))
        else:
            obs[k] = frase
    return obs


def publica_cameras(cabine: Cabine, obs: dict) -> None:
    """Toda chave `video.*` da observação vira uma câmera no painel."""
    for k, v in obs.items():
        if k.startswith("video."):
            img = np.asarray(v)
            while img.ndim > 3:          # (env, tempo, H, W, 3) -> o quadro mais recente
                img = img[-1]
            cabine.publica_quadro(k[len("video."):], np.ascontiguousarray(img.astype(np.uint8)))


class EspiaoCameras(gym.Wrapper):
    """Publica as câmeras a CADA `cada` passos do simulador, por dentro do
    `MultiStepWrapper`. Por fora dele só se vê um quadro a cada consulta ao
    modelo (20 passos + a inferência), e o painel ficava picotado."""

    def __init__(self, env, cabine: Cabine, cada: int = 2):
        super().__init__(env)
        self.cabine, self.cada, self.n = cabine, cada, 0

    def step(self, acao):
        r = self.env.step(acao)
        self.n += 1
        if self.n % self.cada == 0:
            publica_cameras(self.cabine, r[0])
        return r


def monta_env(cena: str, cabine: Cabine, acoes_por_consulta: int):
    """A mesma pilha do `create_eval_env` deles, sem vídeo e com o espião."""
    if "LMCafe" in cena:
        # A cena do café é nossa: registra os ids dela no gym antes do `make`.
        import pontes.groot.cena_cafe.cena_cafe  # noqa: F401

    def fn():
        env = get_gym_env(cena, 0, 1)
        # Reset LEVE: resorteia a posição dos objetos no mesmo modelo, sem
        # recarregar a cena — e sem destruir a janela do MuJoCo (ver o remendo
        # em robosuite/environments/base.py). O hard reset deles recarregava
        # tudo e a janela fechava e reabria a cada episódio.
        env.unwrapped.base_env.hard_reset = False
        env = EspiaoCameras(env, cabine)
        # Episódio "infinito" e sem término por sucesso: quem reinicia é a cabine.
        return MultiStepWrapper(env, video_delta_indices=np.array([0]),
                                state_delta_indices=np.array([0]),
                                n_action_steps=acoes_por_consulta,
                                max_episode_steps=10**9, terminate_on_success=False)
    return gym.vector.SyncVectorEnv([fn])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cena", default="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
                   help="qualquer uma das 36 cenas `gr00tlocomanip_g1_sim/*_G1_gear_wbc`, ou a "
                        "nossa `gr00tlocomanip_g1_sim/LMCafeXicaraCoador_G1_gear_wbc`")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--porta-modelo", type=int, default=5555)
    p.add_argument("--porta", type=int, default=8090, help="porta da cabine")
    p.add_argument("--acoes-por-consulta", type=int, default=20,
                   help="n_action_steps: o mesmo 20 da receita deles")
    p.add_argument("--nome-modelo", default="GR00T-N1.6-G1-PnPAppleToPlate (NVIDIA)",
                   help="só o rótulo na cabine (ex.: o N1.7 do servidor_n17.py)")
    p.add_argument("--ja-roda", action="store_true",
                   help="começa executando a frase do treino, sem esperar a cabine")
    args = p.parse_args()

    cabine = Cabine()
    sobe(cabine, args.porta, "0.0.0.0")
    print(f"cabine: http://localhost:{args.porta}/")

    env = monta_env(args.cena, cabine, args.acoes_por_consulta)
    politica = PolicyClient(host=args.host, port=args.porta_modelo)
    obs, _ = env.reset()
    politica.reset()
    publica_cameras(cabine, obs)
    if args.ja_roda:
        cabine.define_tarefa(FRASE_TREINO)

    visto, frase, rodando = 0, "", False
    episodio, passos, sucesso, inferencia_ms = 1, 0, False, 0
    sucessos, episodios_com_tarefa = 0, 0
    resultado = f"pronto. A frase do treino é: {FRASE_TREINO!r}"

    def reinicia():
        nonlocal obs, episodio, passos, sucesso
        obs, _ = env.reset()
        politica.reset()
        episodio += 1
        passos, sucesso = 0, False

    print("pronto. Mande a tarefa pela cabine.")
    while True:
        texto, seq = cabine.tarefa()
        if seq != visto and texto:
            visto = seq
            if rodando or passos:
                reinicia()
            frase, rodando = texto, True
            episodios_com_tarefa += 1
            resultado = f"executando: {frase}"
        if cabine.consome_pedido_copo():
            reinicia()
            if rodando:
                episodios_com_tarefa += 1
        if rodando and cabine.parada_pedida():
            rodando = False
            resultado = f"parado a pedido, no passo {passos}"

        if rodando:
            t0 = time.perf_counter()
            acoes, _ = politica.get_action(troca_frase(obs, frase))
            inferencia_ms = round((time.perf_counter() - t0) * 1000)
            obs, _, term, trunc, info = env.step(acoes)
            passos += args.acoes_por_consulta
            suc = info.get("success")
            if suc is not None and algum(suc) and not sucesso:
                sucesso = True
                sucessos += 1
                resultado = f"SUCESSO no passo {passos}: {frase}"
            if algum(term) or algum(trunc):
                episodio += 1
                passos, sucesso = 0, False
        else:
            time.sleep(0.05)

        if not rodando:
            publica_cameras(cabine, obs)
        cabine.publica_estado({
            "fase": "rodando" if rodando else "parado",
            "passo": passos,
            "motor": args.nome_modelo,
            "ultimo_resultado": resultado,
            "frase_ativa": frase,
            "frase_do_treino": FRASE_TREINO,
            "episodio": episodio,
            "sucesso_neste_episodio": sucesso,
            "sucessos": f"{sucessos} de {episodios_com_tarefa} episódios com tarefa",
            "inferencia_ms": inferencia_ms,
            "cena": args.cena,
        })


if __name__ == "__main__":
    main()
