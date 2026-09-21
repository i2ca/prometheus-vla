#!/usr/bin/env python
"""Cérebro da cabine — MuJoCo de um lado, tarefa em texto do outro.

    python -m pontes.cabine.cerebro                      # motor roteirizado
    python -m pontes.cabine.cerebro --motor wla --pesos /data/.../steps_2000_model.safetensors

Abre `http://<ip>:8090/`, escreve a tarefa na caixa (ou manda pelo MCP) e o robô
executa. O painel mostra as duas câmeras que a política vê e o estado das juntas.

── O motor é trocável, e isso é o ponto ────────────────────────────────────
`--motor roteirizado` usa o especialista do `demo_pega_copo_mujoco.py`: o mesmo
que gravou os 1.828 episódios, confiável, e que NÃO precisa de checkpoint. Serve
para a demonstração da super IA existir antes de a rede estar boa.
`--motor wla` põe a UnifoLM-WLA no lugar. A casca — painel, MCP, tarefas — é a
mesma nos dois casos, de propósito: o dia em que a rede ficar boa, troca-se uma
palavra na linha de comando.

── O roteador de texto é burro, e está escrito para ser ────────────────────
Hoje ele procura palavra-chave e escolhe entre três roteiros. Isso NÃO é
compreensão de linguagem; é um mapa de mesa. Quem entende texto de verdade é a
VLA, e no `--motor wla` a frase vai inteira para o campo `lang` do modelo, sem
passar por aqui. O roteador só existe para o motor roteirizado ter o que fazer.

── Por que o simulador roda numa thread e não no processo do servidor ──────
O laço do MuJoCo com IK é síncrono e demora ~18 s por episódio. Se ele rodasse
na thread do HTTP, o painel congelaria durante a execução — exatamente quando
alguém está olhando. Aqui o laço é dono da sua thread e só deposita quadros.
"""
from __future__ import annotations

import os
import sys

# O backend do GL é escolhido no import do mujoco, antes de qualquer argparse.
# EGL renderiza offscreen sem abrir janela, que é o que a cabine quer: quem
# assiste, assiste pelo navegador.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")   # ipopt multithread come 600% de CPU

import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

import mujoco  # noqa: E402

from pontes.cabine.servidor import Cabine, sobe  # noqa: E402


# Câmeras publicadas no painel. As duas primeiras são os `image_roles` do treino
# (`head_left` e `cam_wrist_right` no `dados_prometheus.yaml`) — é o que a
# política enxerga. A `global_view` é só para quem assiste entender a cena.
# `head_camera_antiga` está aqui de propósito: o dataset `nosso_sim_pega`, que é
# 97% do que treinou a rede, foi gravado com a pose errada. Enquanto for assim, é
# ELA que casa com o que a política viu — a `head_camera` (pose real, corrigida
# em 21/09) fica ao lado para dar para comparar as duas no painel.
CAMERAS = {
    "head_camera": (480, 848),
    "head_camera_antiga": (480, 848),
    "right_wrist_camera": (480, 640),
    "global_view": (480, 640),
}
# Qual delas alimenta a política. Câmeras que não existem na cena são ignoradas.
CAMERA_DA_POLITICA = os.environ.get("CABINE_CAM_CABECA", "head_camera_antiga")


class MotorRoteirizado:
    """Especialista do `demo_pega_copo_mujoco.py`, sem rede neural nenhuma."""

    # Palavra-chave → roteiro. Ordem importa: "põe" ganha de "pega" numa frase
    # que tem as duas ("pegue a xícara e ponha no filtro" é o roteiro completo).
    MAPA = (
        (("completo", "café", "cafe", "tudo"), "completo"),
        (("filtro", "coador", "dripper", "ponha", "poe", "põe", "place", "put"), "poe"),
        (("xícara", "xicara", "caneca", "copo", "cup", "pegue", "pega", "pick"), "pega"),
    )

    def __init__(self):
        import demo_pega_copo_mujoco as demo

        # O demo faz um remendo no módulo do IK antes de expor a classe (ver o
        # comentário dele por quê); importar `G1_29_ArmIK` direto do módulo
        # original pula esse remendo e quebra.
        self.demo = demo
        self.cena = demo.Cena(com_janela=False)
        self.ik = demo.G1_29_ArmIK(
            visualization=False, travar_punho_dir=False,
            postura_ref=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0], float),
        )
        self.descanso = np.array([0.42, -0.26, self.cena.z_copo + 0.22])
        self.cena.define_pose_partida(self.ik, self.descanso)
        self.cena.descanso = self.descanso
        self.rng = np.random.default_rng(0)

    def roteiro_de(self, texto: str) -> str | None:
        t = (texto or "").lower()
        for chaves, roteiro in self.MAPA:
            if any(c in t for c in chaves):
                return roteiro
        return None

    def executa(self, texto: str, n: int, ao_passo, dorme: float = 0.0) -> str:
        roteiro = self.roteiro_de(texto)
        if roteiro is None:
            return (f"não sei fazer {texto!r}. O motor roteirizado conhece: pegar a "
                    f"xícara, pôr no filtro, ou o roteiro completo.")
        self.demo.episodio(self.cena, self.ik, self.rng, n, dorme=dorme,
                           ao_passo=ao_passo, tarefa=roteiro)
        return f"executei o roteiro {roteiro!r}"


class MotorWLA:
    """UnifoLM-WLA no laço. Esqueleto: a desnormalização ainda falta.

    O que já está certo aqui é a FORMA do que o modelo espera — veio da leitura
    do `QwenMMDiT.predict_action`. O que falta é o caminho de volta: a saída é
    normalizada em `minmax_q` no espaço unificado de 54 dims, e precisa das
    MESMAS estatísticas do treino para virar ângulo de junta. É o inverso do
    `pontes/wla/converte_dataset_wla.py`.
    """

    def __init__(self, pesos: Path):
        raise NotImplementedError(
            "MotorWLA ainda não fecha o laço: falta desnormalizar a saída com as "
            f"estatísticas do treino. Pesos que seriam usados: {pesos}"
        )


class Cerebro:
    def __init__(self, cabine: Cabine, motor, passos_por_quadro: int = 8,
                 dorme: float = 0.0):
        self.cabine = cabine
        self.motor = motor
        # Segundos de sono por passo. Zero = o mais rápido que a CPU deixa, e um
        # episódio inteiro passa em ~4 s — ótimo para testar, ruim para gravar
        # vídeo, porque o movimento fica picotado. Para o vídeo da super IA, use
        # algo como 0.004 (o passo do MuJoCo), que dá tempo quase real.
        self.dorme = float(dorme)
        self.passos_por_quadro = passos_por_quadro
        self.cena = motor.cena
        self.renderizadores = {
            nome: mujoco.Renderer(self.cena.m, alt, larg)
            for nome, (alt, larg) in CAMERAS.items()
            if mujoco.mj_name2id(self.cena.m, mujoco.mjtObj.mjOBJ_CAMERA, nome) >= 0
        }
        self._parar = threading.Event()
        self._contador = 0
        self._fase = "parado"
        self._ultimo_resultado = ""

    # ── publicação ──────────────────────────────────────────────────────
    def _publica(self, fase: str, q) -> None:
        self._contador += 1
        if self._contador % self.passos_por_quadro:
            return
        for nome, r in self.renderizadores.items():
            try:
                r.update_scene(self.cena.d, camera=nome)
                self.cabine.publica_quadro(nome, r.render())
            except Exception:
                pass          # um render que falha não pode derrubar o laço
        self.cabine.publica_estado({
            "fase": fase,
            "passo": self._contador,
            "juntas_braco": [round(float(v), 4) for v in np.atleast_1d(q)],
            "copo": [round(float(v), 3) for v in self.cena.d.qpos[self.cena.copo_qpos:
                                                                 self.cena.copo_qpos + 3]],
            "motor": type(self.motor).__name__,
            "ultimo_resultado": self._ultimo_resultado,
        })

    def _ao_passo(self, fase, q):
        self._fase = fase
        self._publica(fase, q)
        if self._parar.is_set() or self.cabine.parada_pedida():
            # `episodio` não tem cancelamento; levantar é o jeito honesto de
            # sair no meio sem deixar o MuJoCo num estado pela metade.
            raise KeyboardInterrupt("parada pedida pela cabine")

    # ── laço principal ──────────────────────────────────────────────────
    def roda(self) -> None:
        visto = -1
        n = 0
        self._publica("ocioso", np.zeros(14))
        while True:
            texto, seq = self.cabine.tarefa()
            if seq == visto or not texto:
                self._publica("ocioso", self.cena.d.qpos[self.cena.q_braco])
                time.sleep(0.1)
                continue
            visto = seq
            n += 1
            self._parar.clear()
            self._ultimo_resultado = f"executando: {texto}"
            try:
                self._ultimo_resultado = self.motor.executa(
                    texto, n, self._ao_passo, dorme=self.dorme)
            except KeyboardInterrupt as e:
                self._ultimo_resultado = f"interrompido: {e}"
            except Exception as e:
                self._ultimo_resultado = f"falhou: {e}"
                traceback.print_exc()
            print(f"[cabine] {texto!r} → {self._ultimo_resultado}")

    def para(self) -> None:
        self._parar.set()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--porta", type=int, default=8090)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--motor", choices=("roteirizado", "wla"), default="roteirizado")
    p.add_argument("--pesos", type=Path, default=None,
                   help="checkpoint .safetensors, obrigatório com --motor wla")
    p.add_argument("--tarefa", default="", help="já começa executando esta frase")
    p.add_argument("--dorme", type=float, default=0.0, metavar="SEGUNDOS",
                   help="sono por passo; 0.004 dá movimento quase em tempo real, "
                        "que é o que o vídeo pede. O padrão 0 corre solto.")
    args = p.parse_args()

    print(f"⏳ montando o motor {args.motor}…")
    if args.motor == "wla":
        if args.pesos is None:
            p.error("--motor wla precisa de --pesos")
        motor = MotorWLA(args.pesos)
    else:
        motor = MotorRoteirizado()

    cabine = Cabine()
    cerebro = Cerebro(cabine, motor, dorme=args.dorme)
    sobe(cabine, args.porta, args.host)
    print(f"✅ cabine no ar: http://<ip>:{args.porta}/")
    print(f"   câmeras: {', '.join(cerebro.renderizadores)}")
    if args.tarefa:
        cabine.define_tarefa(args.tarefa)
    try:
        cerebro.roda()
    except KeyboardInterrupt:
        print("\nsaindo.")


if __name__ == "__main__":
    main()
