#!/usr/bin/env python
"""
Especialista roteirizado no MuJoCo — pega a xícara e põe no alvo sorteado.
================================================================================
Isto é o PASSO ZERO do gerador de dados: antes de gravar milhares de episódios,
olhar um deles e responder "o movimento que eu vou gravar se parece com o que eu
demonstraria?" e "isso trava em algum lugar?".

Por isso ele não grava nada. Só mostra.

    python demo_pega_copo_mujoco.py                 # com janela, episódios em laço
    python demo_pega_copo_mujoco.py --episodios=3
    python demo_pega_copo_mujoco.py --sem-janela    # teste rápido, sem tela

── Três decisões de projeto, e o que cada uma custa ─────────────────────────

1. CARREGA O XML DIRETO, sem o `SimulatorFactory`. Sem DDS, sem ponte Unitree,
   sem publicação ZMQ de câmera. Sobe em segundos e não existe socket que possa
   travar — o que importa quando a pergunta é justamente "isso entra em laço
   infinito?".

2. CINEMÁTICA PURA: as juntas do braço são escritas direto em `qpos` e o passo é
   `mj_forward`, não `mj_step`. Não há controle de torque, não há ganho para
   ajustar, e o robô não pode cair. Em troca, não há física de contato: este
   script mostra a TRAJETÓRIA, não a pega.

3. A PEGA É ROTEIRIZADA: a partir da fase `fechar`, a xícara é teleportada para
   a palma a cada quadro. Para treinar ação e junta — que é o que se quer aqui —
   isso basta, porque o rótulo é a trajetória do braço. Não sustenta nenhuma
   afirmação sobre força de preensão ou escorregamento.

── As travas contra laço infinito ───────────────────────────────────────────
Cada fase tem orçamento de passos E teste de progresso: se a distância até o
alvo não cair por `PACIENCIA` passos seguidos, o episódio é abortado com o
motivo impresso. Sem isso, uma pose fora do alcance faz o IK devolver sempre a
mesma solução e o laço roda para sempre sem nada na tela mudando — que é
exatamente o modo de falha que se quer descobrir agora, e não depois de deixar
gerando a noite toda.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# O IK vive na árvore do lerobot-ext e resolve o URDF por caminho relativo a ela.
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from robot.unitree_g1.robot_control import g1_arm_ik as _ik_mod  # noqa: E402

# O módulo resolve os assets como `robot/unitree_g1/assets`, diretório que NÃO
# EXISTE neste checkout — o URDF e as malhas moram em `lerobot-ext/assets/g1`.
# Do jeito que está, `G1_29_ArmIK()` levanta FileNotFoundError para qualquer
# chamador. Corrigir aqui, e não no módulo, porque ele é compartilhado com o
# teleop e mudar o caminho por baixo dele é o tipo de coisa que quebra outra
# coisa numa terça-feira.
_ik_mod.ASSETS_DIR = AQUI / "assets"
G1_29_ArmIK = _ik_mod.G1_29_ArmIK  # noqa: E402

CENA = AQUI.parent / "unitree-g1-mujoco" / "assets" / "scene_43dof.xml"

# Ordem que o IK reduzido devolve: 7 do braço esquerdo, 7 do direito.
JUNTAS_BRACO = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Vetor de fechamento da mão direita, copiado do `teleop/unitree_g1/keyboard_g1_arm.py`
# (`RIGHT_HAND_CLOSED_TARGETS`, mapeado do ROS2). O SINAL importa: no polegar os
# alvos são NEGATIVOS e nos dedos POSITIVOS. Fechar tudo com o mesmo valor
# positivo dobra o polegar para fora e a mão abre em vez de fechar — foi o que
# aconteceu na primeira versão deste script.
MAO_FECHADA = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -1.5,
    "right_hand_thumb_2_joint": -1.5,
    "right_hand_index_0_joint": 1.5,
    "right_hand_index_1_joint": 1.5,
    "right_hand_middle_0_joint": 1.5,
    "right_hand_middle_1_joint": 1.5,
}

REJEITA = False        # ligado por --rejeita-colisao
PASSOS_NOMINAIS = 150  # duração de referência de uma fase, para o perfil de velocidade


# ── Onde cada objeto nasce ────────────────────────────────────────────────
# O robô olha para +x, e para ele +y é a ESQUERDA. A XÍCARA nasce sempre à
# direita e o COADOR sempre à esquerda: é o braço direito que age, então pegar
# do lado dele, atravessar o corpo e pousar do outro lado é a estrutura que se
# repete em todo episódio, e o que varia é só onde, dentro de cada metade.
#
# AS DUAS CAIXAS SÃO MEDIDAS, não escolhidas. Varredura de 04/09, `--tarefa=pega`,
# 2 a 3 tentativas por célula (a fase que falha, entre parênteses):
#
#          y=-0,06  -0,10  -0,14  -0,18   -0,22
#   x=0,40    ok    MESA   MESA   MESA    MESA     ← antebraço entra no tampo
#   x=0,44    ok    MESA    ok     ok     MESA
#   x=0,45    ok    MESA    ok     ok    ALCANCE
#   x=0,46    ok     ok     ok     ok    ALCANCE
#   x=0,47    ok     ok     ok   ALCANCE ALCANCE
#   x=0,48    ok     ok     ok   ALCANCE ALCANCE
#   x=0,50  ALCANCE ALCANCE ALCANCE ALCANCE ALCANCE
#
# MESA    = `descer` bloqueado por colisão com o tampo (o cotovelo não tem para
#           onde ir a x baixo com o braço aberto para a direita).
# ALCANCE = "fora do alcance mesmo com a cintura". Não é orçamento de passos:
#           subir o da `descer` de 700 para 1400 só troca a mensagem
#           "estourou 700 passos" por "fora do alcance" nas mesmas células.
#
# Daí a caixa da xícara ser ESTREITA em x. Alargá-la é tentador e já foi
# tentado: a faixa antiga (0,40–0,50) é a razão de a taxa de aproveitamento ser
# ~47% — metade dos episódios nascia fora do alcance do braço.
#
# O COADOR não tem esse problema: a mão chega nele com 4,5 cm de tolerância e
# vinda de cima, e a mesma varredura com `--tarefa=poe` deu 10 de 12 células
# boas em x ∈ [0,42, 0,50] × y ∈ [0,10, 0,22]. Por isso a caixa dele é larga.
X_COPO = (0.45, 0.48)      # varredura acima
Y_DIREITA = (0.06, 0.16)   # MÓDULO; a xícara entra com sinal negativo
X_ALVO = (0.42, 0.50)
Y_ESQUERDA = (0.10, 0.22)  # o coador, positivo


def perfil_humano(passo, nominal=PASSOS_NOMINAIS, piso=0.18):
    """Sino de velocidade: devagar no início, rápido no meio, devagar no fim.

    Ganho constante dá aproximação EXPONENCIAL — salto no primeiro passo e uma
    cauda infinita chegando —, que é o oposto de como uma pessoa move o braço.
    Movimento humano tem perfil de jerk mínimo, e a derivada dele é este sino
    (30τ²(1-τ)², normalizado). Com ele o braço acelera saindo, cruza o espaço
    livre depressa e desacelera ao encostar na xícara.

    O piso existe porque a fase termina por TOLERÂNCIA e não por tempo: sem ele
    o ganho tenderia a zero na cauda e a fase nunca fecharia.
    """
    tau = min(passo / max(1.0, nominal), 1.0)
    sino = 30.0 * tau * tau * (1.0 - tau) ** 2 / 1.875
    return piso + (1.0 - piso) * sino
FPS = 30.0
# 200, e não 60: com o perfil de jerk mínimo o começo do movimento é lento DE
# PROPÓSITO, e a melhora por passo cai abaixo do limiar de 0,1 mm sem que nada
# esteja travado. Com paciência curta a fase abortava a 2,4 cm do alvo — perto,
# devagar e declarada morta.
PACIENCIA = 200
# 2,5 cm. A xícara tem 7,8 cm de diâmetro e os dedos fecham POR CONTATO, então
# esse resto não muda a pega — e com a correção lenta (que tirou o zigue-zague)
# a convergência fica mais demorada, e episódios paravam exatamente na fronteira
# dos 2,0 cm, perto e declarados mortos.
TOLERANCIA = 0.025


# Penetração, em metros, a partir da qual o antebraço "bateu" na mesa.
# Zero significaria marcar qualquer toque — ver o comentário no
# `bateu_na_mesa`.
MARGEM_MESA = 0.005

def matriz(pos, R=None):
    """4x4 homogênea a partir de posição e rotação."""
    T = np.eye(4)
    T[:3, 3] = pos
    if R is not None:
        T[:3, :3] = R
    return T


class Cena:
    """O modelo, os índices que interessam e as conversões de referencial."""

    def __init__(self, com_janela: bool):
        self.m = mujoco.MjModel.from_xml_path(str(CENA))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

        # MESA ABAIXO DO COTOVELO. Medido na cena original: cotovelo em z=0,898
        # e tampo em z=0,900 — na MESMA altura, e por isso o antebraço raspa a
        # mesa em toda aproximação. Descer 5 cm dá folga para o cotovelo passar
        # por cima. A mesa é corpo estático (sem junta), então mexer em
        # `body_pos` basta e não precisa tocar no XML, que é compartilhado.
        b_mesa = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "mesa")
        b_cot = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
        if b_mesa >= 0 and b_cot >= 0:
            folga = 0.05
            topo_atual = float(self.d.xpos[b_mesa][2] + 0.02)
            topo_novo = float(self.d.xpos[b_cot][2]) - folga
            self.m.body_pos[b_mesa][2] += (topo_novo - topo_atual)
            mujoco.mj_forward(self.m, self.d)

        self.q_braco = np.array(
            [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in JUNTAS_BRACO]
        )
        # Dedos: fechar a mão é cosmético aqui, mas sem isso a garra atravessa a
        # xícara na tela e a gravação parece errada para quem assiste.
        self.q_dedos = np.array([
            self.m.jnt_qposadr[j] for j in range(self.m.njnt)
            if "hand" in self.m.joint(j).name and "right" in self.m.joint(j).name
        ])

        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "junta_livre_bloco")
        self.copo_qpos = self.m.jnt_qposadr[jid]
        self.copo_dof = self.m.jnt_dofadr[jid]

        self.id_pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.id_punho = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        # Referencial do tronco: é o "corpo" com que a rolagem da mão se alinha
        # (ver `alinha_rolagem`). Vem do `torso_link` e não da pelve porque é
        # ele que o `aponta_cintura` gira.
        self.id_tronco = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

        # PONTO DE PEGA = centro da pinça, e não o punho. O punho fica em
        # x=0,200 e as pontas dos dedos em x=0,365: mandar o PUNHO até a xícara
        # faz a mão inteira passar 12 cm além dela, e na tela o braço atravessa
        # o objeto sem pegar. A média das três pontas segue a mão em qualquer
        # pose, sem precisar de deslocamento fixo.
        self.id_dedos = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("right_hand_thumb_2_link",
                      "right_hand_index_1_link",
                      "right_hand_middle_1_link")
        ]

        # ALTURA DA MESA, lida do modelo em vez de chutada. A caixa está em
        # z=0,88 com meia-altura 0,02, então o tampo é 0,90; o copo é um
        # cilindro de meia-altura 0,048, então apoiado o centro dele fica em
        # 0,948. O valor 1,10 que o `reset_cup` usa deixa a xícara FLUTUANDO
        # 15 cm acima da mesa — some no ar assim que a física roda.
        # Juntas da mão na ordem do vetor acima, e os geoms de cada dedo — o
        # fechamento por contato precisa saber qual geom pertence a qual junta.
        self.mao = []
        for nome, alvo in MAO_FECHADA.items():
            jid_m = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, nome)
            if jid_m < 0:
                continue
            corpo = self.m.jnt_bodyid[jid_m]
            geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == corpo]
            self.mao.append({"adr": self.m.jnt_qposadr[jid_m], "alvo": alvo, "geoms": set(geoms)})

        self.geom_copo = {
            g for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] == self.m.jnt_bodyid[jid]
        }

        # Cintura. No robô real o `use_waist_yaw=True` põe o `kWaistYaw` no
        # vetor de ação, e o dataset foi gravado assim — se o especialista da
        # simulação mantém o tronco travado, ele gera demonstração de um robô
        # que não é o seu.
        jw = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")
        self.adr_cintura = self.m.jnt_qposadr[jw]
        self.lim_cintura = 0.5     # bem menos que os ±2,6 do modelo: girar o
                                   # tronco todo tira as câmeras da cena

        # O X de destino. `body_mocapid` traduz o id do body para o índice na
        # tabela `d.mocap_pos`, que é densa e só tem os bodies mocap — usar o
        # id do body direto aponta para a linha errada.
        b_marca = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "marca_alvo")
        if b_marca < 0:
            raise RuntimeError(
                "body 'marca_alvo' não existe em scene_43dof.xml — sem ele o X "
                "não entra nas câmeras e o dataset sai sem o alvo visível.")
        self.mocap_alvo = int(self.m.body_mocapid[b_marca])

        self.geom_mesa = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "geometria_mesa")
        # O cilindro de colisão no topo do coador. `-1` se a cena não o tiver
        # (versões antigas do XML), e aí `bateu_no_coador` só devolve False.
        self.geom_coador = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "coador_boca")
        def _geoms_de(pred):
            return {g for g in range(self.m.ngeom)
                    if pred(self.m.body(self.m.geom_bodyid[g]).name)}

        self.geoms_dir = _geoms_de(
            lambda n: n.startswith("right_") and
            any(k in n for k in ("shoulder", "elbow", "wrist", "hand")))
        # Para AUTO-COLISÃO, do cotovelo para baixo. O ombro é preso no tronco:
        # o contato entre eles é estrutural e aparece em `d.ncon` o tempo todo.
        # Contá-lo faz todo episódio abortar por "auto-colisao" sem que nada
        # esteja errado — e esconde a colisão de verdade, que é o cotovelo e o
        # antebraço entrando no corpo.
        # Cotovelo, punho E MÃO contra a mesa. A mão entrou de volta agora que
        # a aproximação é VERTICAL: descendo reto sobre a xícara os dedos param
        # nela, e não há motivo para atravessar o tampo. Com a aproximação
        # diagonal de antes isso era impossível de satisfazer — a mão chegava
        # rasante e encostava na mesa antes do copo.
        # Cotovelo e punho. A MÃO fica de fora, e não por preguiça: a xícara
        # está APOIADA no tampo, então envolvê-la obriga os dedos a descerem até
        # a altura da mesa. Medido: com a mão incluída, 5 de 5 episódios morrem
        # na descida, porque a restrição não tem solução — o obstáculo está no
        # destino, não no caminho. Quem não pode atravessar é o braço.
        self.geoms_antebraco = _geoms_de(
            lambda n: n.startswith("right_") and
            any(k in n for k in ("elbow", "wrist")))
        self.geoms_dir_livre = _geoms_de(
            lambda n: n.startswith("right_") and
            any(k in n for k in ("elbow", "wrist", "hand")))
        self.geoms_tronco = _geoms_de(
            lambda n: n in ("torso_link", "pelvis", "waist_yaw_link",
                            "waist_roll_link", "logo_link", "head_link")
            or n.startswith("left_"))

        self.geoms_robo = {
            g for g in range(self.m.ngeom)
            if g not in self.geom_copo and g != self.geom_mesa
        }

        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "geometria_mesa")
        # `geom_xpos` (mundo), não `geom_pos` (relativo ao corpo): o corpo da
        # mesa está em z=0,88 e o geom tem deslocamento zero dentro dele, então
        # `geom_pos` devolve 0,02 e a xícara nasce enterrada no chão.
        self.z_mesa = float(self.d.geom_xpos[gid][2] + self.m.geom_size[gid][2])
        gid_c = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "colisao_copo")
        self.meio_copo = float(self.m.geom_size[gid_c][1])
        self.z_copo = self.z_mesa + self.meio_copo

        # Altura em que a xícara POUSA quando o alvo é o coador: a boca do
        # funil (18,5 cm da base, medido na malha) mais a meia altura da
        # xícara. Sem isto ela era pousada no tampo e ATRAVESSAVA o coador —
        # e nenhum geom de colisão conserta isso, porque esta simulação roda
        # `mj_forward` e nunca `mj_step`: contato é detectado, nunca resolvido.
        self.z_boca_coador = self.z_mesa + 0.185 + self.meio_copo

        # Pose inicial do braço e orientação de referência do punho, guardadas
        # com o robô ainda na pose "home". Duas coisas dependem disso:
        #
        #  - reiniciar o braço entre episódios. Sem isso o episódio seguinte
        #    começa na pose torta onde o anterior falhou, e falha por herança —
        #    o que faz parecer que o alvo é que era ruim.
        #  - a orientação do alvo do IK. Usar a rotação ATUAL do punho como
        #    referência funciona enquanto o braço está bem posicionado e vira
        #    alvo impossível quando não está, e o casadi então cospe
        #    "Error in Opti::solve" a cada passo.
        self.q_braco0 = self.d.qpos[self.q_braco].copy()
        self.R_ref = self.d.xmat[self.id_punho].reshape(3, 3).copy()

        # EIXO DA PALMA, medido e não assumido: a direção que vai do punho ao
        # centro da pinça, expressa no referencial LOCAL do punho. Com ela dá
        # para construir, a cada passo, a rotação que aponta a palma para um
        # ponto do mundo — em vez de carregar uma orientação fixa e chegar na
        # xícara de lado.
        v = self.ponto_pega() - self.punho_no_mundo()
        self.v_palma = self.R_ref.T @ (v / (np.linalg.norm(v) + 1e-9))

        self.viewer = None
        if com_janela:
            # JANELA COM BARRA DE TÍTULO. No Wayland o GLFW depende do libdecor
            # para desenhar decoração do lado do cliente, e aqui o plugin GTK
            # existe mas falha ao iniciar:
            #
            #   Failed to load plugin 'libdecor-gtk.so': failed to init
            #   No plugins found, falling back on no decorations
            #
            # O resultado é uma janela sem minimizar, maximizar nem fechar — não
            # dá nem para tirar da frente. Pedindo X11 (via XWayland), quem
            # desenha a barra é o gerenciador de janelas, e ela aparece. O hint
            # tem que vir ANTES do `glfw.init()` que o viewer faz lá dentro.
            try:
                import glfw
                glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
            except Exception as erro:
                print(f"   (sem forçar X11: {erro})")

            self.viewer = mujoco.viewer.launch_passive(self.m, self.d,
                                                       show_left_ui=False, show_right_ui=False)
            self.viewer.cam.azimuth, self.viewer.cam.elevation = 130, -25
            self.viewer.cam.distance = 1.8
            self.viewer.cam.lookat = np.array([0.25, 0.0, 1.0])

    # ── referenciais ────────────────────────────────────────────────────────
    def orienta_para(self, ponto, de=None):
        """Rotação do punho que faz a palma apontar para `ponto`.

        Construída como a MENOR rotação que leva a direção atual da palma até a
        direção desejada, aplicada sobre a orientação de referência. Menor
        rotação porque o eixo em torno da própria palma fica livre (há uma
        família de soluções), e girar só o necessário mantém o movimento
        contínuo em vez de fazer o punho rolar sem motivo entre um passo e outro.
        """
        # `de` é DE ONDE se olha. Usar a posição atual do punho faz a palma
        # apontar ao longo da diagonal por onde o braço está chegando — e a mão
        # pega a xícara torta. Passando o DESTINO da fase, a orientação é a que
        # a mão terá quando estiver no lugar: vindo de cima do copo, a palma
        # aponta reto para baixo, e a descida é vertical.
        origem = self.punho_no_mundo() if de is None else np.asarray(de, dtype=float)
        alvo = np.asarray(ponto, dtype=float) - origem
        n = np.linalg.norm(alvo)
        if n < 1e-6:
            return self.R_ref
        b = alvo / n
        a = self.R_ref @ self.v_palma
        a = a / (np.linalg.norm(a) + 1e-9)

        eixo = np.cross(a, b)
        sen = np.linalg.norm(eixo)
        cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if sen < 1e-8:
            return self.R_ref if cos > 0 else -self.R_ref
        eixo = eixo / sen
        ang = np.arctan2(sen, cos)
        K = np.array([[0, -eixo[2], eixo[1]],
                      [eixo[2], 0, -eixo[0]],
                      [-eixo[1], eixo[0], 0]])
        R_al = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        return self.alinha_rolagem(R_al @ self.R_ref)

    def R_lateral(self, ponto):
        """Palma na HORIZONTAL, mirando `ponto`. Pega pela lateral da caneca.

        Diferença para o `R_yaw`: lá a palma vai para -Z (pega POR CIMA, que é
        como se pega uma tampa, não uma caneca). Aqui a palma fica no plano da
        mesa e aponta para o objeto, que é como uma mão humana envolve a
        lateral de uma caneca.

        Continua sendo "só yaw" no sentido que você pediu: a palma nunca sobe
        nem desce (pitch zero, porque a direção alvo tem componente z nula) e a
        mão não rola em torno do próprio eixo (roll zero, porque a rotação
        aplicada é a MÍNIMA entre duas direções horizontais). O único grau que
        varia é o giro em torno da vertical.
        """
        d = np.asarray(ponto, dtype=float) - self.punho_no_mundo()
        d[2] = 0.0                      # <- isto é o que zera o pitch
        n = np.linalg.norm(d)
        if n < 1e-6:
            return self.R_ref
        b = d / n
        a = self.R_ref @ self.v_palma
        a = a / (np.linalg.norm(a) + 1e-9)

        eixo = np.cross(a, b)
        sen = float(np.linalg.norm(eixo))
        cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if sen < 1e-8:
            return self.R_ref if cos > 0 else -self.R_ref
        eixo = eixo / sen
        ang = np.arctan2(sen, cos)
        K = np.array([[0, -eixo[2], eixo[1]],
                      [eixo[2], 0, -eixo[0]],
                      [-eixo[1], eixo[0], 0]])
        R_al = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        return R_al @ self.R_ref

    def R_yaw(self, ponto):
        """Palma reta para BAIXO, girando só no yaw para encarar `ponto`.

        Pitch e roll ficam em zero por construção: primeiro leva-se o eixo da
        palma exatamente para -Z do mundo (é isso que zera pitch e roll), e só
        depois se gira em torno do PRÓPRIO -Z, que é o yaw. Nenhuma dessas duas
        rotações introduz inclinação.

        A `orienta_para` fazia diferente: apontava a palma direto para o alvo,
        e como o alvo quase nunca está exatamente abaixo do punho, a mão chegava
        inclinada — era isso que aparecia como "pegando errado o copo".
        """
        # Passo 1: palma para baixo.
        a = self.R_ref @ self.v_palma
        a = a / (np.linalg.norm(a) + 1e-9)
        b = np.array([0.0, 0.0, -1.0])
        eixo = np.cross(a, b)
        sen = float(np.linalg.norm(eixo))
        cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if sen < 1e-8:
            R = self.R_ref.copy() if cos > 0 else -self.R_ref
        else:
            eixo = eixo / sen
            ang = np.arctan2(sen, cos)
            K = np.array([[0, -eixo[2], eixo[1]],
                          [eixo[2], 0, -eixo[0]],
                          [-eixo[1], eixo[0], 0]])
            R = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)) @ self.R_ref

        # Passo 2: yaw FIXO no "para frente" do tronco.
        #
        # `ponto` é ignorado de propósito. A versão anterior girava o yaw para
        # encarar o alvo, e isso é justamente o que faz a mão mudar de
        # orientação ao longo do episódio — o pedido é que ela fique RETA
        # sempre. Com o yaw preso ao tronco, a mão acompanha só o giro da
        # cintura (que o `aponta_cintura` faz) e nada mais.
        frente_tronco = self.d.xmat[self.id_tronco].reshape(3, 3)[:, 0]
        ponto = self.punho_no_mundo() + np.array(
            [frente_tronco[0], frente_tronco[1], 0.0])

        # yaw em torno do Z do mundo
        #
        # A referência do yaw é a coluna do punho MAIS PERPENDICULAR à palma, e
        # não a coluna 0. Medido nesta cena: `v_palma` é [0.991, 0.133, 0.0] no
        # frame local, ou seja a coluna 0 É o eixo da palma. Depois do passo 1
        # ela aponta para -Z, e tirar o yaw dela é degenerado — `arctan2` sobre
        # duas componentes quase nulas devolve ângulo aleatório. Era esse o bug
        # da primeira versão do `R_yaw`, que dava 0 episódios salvos em 8.
        j = int(np.argmin(np.abs(self.v_palma / (np.linalg.norm(self.v_palma) + 1e-9))))
        d = np.asarray(ponto, dtype=float)[:2] - self.punho_no_mundo()[:2]
        if np.linalg.norm(d) < 1e-6:
            return R
        alvo_yaw = np.arctan2(d[1], d[0])
        ref = R[:, j]
        yaw_atual = np.arctan2(ref[1], ref[0])
        return gira_em_torno(R, np.array([0.0, 0.0, 1.0]), alvo_yaw - yaw_atual)

    def R_reta(self):
        """Punho alinhado com o TRONCO, sem inclinação — o "0, 0, 0" da mão.

        A `R_ref` é medida da pose de descanso do braço e carrega a inclinação
        natural dela. Para carregar a xícara sem tombar, o que se quer é a mão
        na mesma orientação do corpo: os eixos do punho paralelos aos do
        `torso_link`, com a palma apontando para baixo.

        Construída a partir dos eixos do tronco e não de ângulos fixos porque o
        `aponta_cintura` gira o tronco durante o episódio — com ângulos fixos a
        mão ficaria reta em relação ao mundo e torta em relação ao corpo.
        """
        T = self.d.xmat[self.id_tronco].reshape(3, 3)
        # Palma para baixo: leva o eixo da palma até -Z do mundo, mantendo o
        # resto alinhado ao tronco.
        palma_local = self.v_palma / (np.linalg.norm(self.v_palma) + 1e-9)
        a = T @ palma_local
        b = np.array([0.0, 0.0, -1.0])
        eixo = np.cross(a, b)
        sen = np.linalg.norm(eixo)
        cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if sen < 1e-8:
            return T if cos > 0 else -T
        eixo = eixo / sen
        ang = np.arctan2(sen, cos)
        K = np.array([[0, -eixo[2], eixo[1]],
                      [eixo[2], 0, -eixo[0]],
                      [-eixo[1], eixo[0], 0]])
        R_al = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        return R_al @ T

    def alinha_rolagem(self, R):
        """Fixa o giro da mão EM TORNO DA PALMA no mesmo ângulo do tronco.

        `orienta_para` resolve só para onde a palma APONTA. Sobra um grau de
        liberdade: girar a mão em torno do próprio eixo da palma não muda para
        onde ela olha. Antes esse grau ficava solto — vinha do que a rotação
        mínima acidentalmente produzisse, e o `resolve_livre` ainda o sorteava
        entre ±0,8 rad. O resultado é a mão pegando a xícara com a pinça
        rolada num ângulo diferente a cada episódio.

        Aqui ele passa a ser DETERMINADO: gira-se em torno da palma até que o
        eixo lateral da mão fique perpendicular ao "para frente" do tronco —
        ou seja, a pinça abre no mesmo plano em que o corpo está virado. Como
        o `aponta_cintura` já gira o tronco na direção do alvo, a mão passa a
        acompanhar essa rotação em vez de ignorá-la.
        """
        palma = R @ self.v_palma
        palma = palma / (np.linalg.norm(palma) + 1e-9)

        # "Para frente" do tronco, projetado no plano perpendicular à palma.
        frente = self.d.xmat[self.id_tronco].reshape(3, 3)[:, 0]
        alvo = frente - np.dot(frente, palma) * palma
        n = np.linalg.norm(alvo)
        if n < 1e-6:
            return R          # palma paralela ao eixo do tronco: nada a fixar
        alvo = alvo / n

        # Eixo lateral atual da mão, no mesmo plano.
        lat = R[:, 1] - np.dot(R[:, 1], palma) * palma
        m = np.linalg.norm(lat)
        if m < 1e-6:
            return R
        lat = lat / m

        ang = np.arctan2(float(np.dot(np.cross(lat, alvo), palma)),
                         float(np.dot(lat, alvo)))

        # Alinhar um eixo a outro tem SEMPRE duas soluções, a 180° uma da
        # outra, e as duas satisfazem "mesmo ângulo do corpo". Uma delas põe a
        # mão de cabeça para baixo — foi o que apareceu na tela como o robô
        # metendo a mão POR BAIXO da xícara para pegar. Escolhe-se a que fica
        # mais perto da orientação que o `orienta_para` produziu, que é a
        # rotação MÍNIMA a partir da pose de referência: assim o alinhamento
        # com o tronco acontece sem virar a mão.
        melhor, escolhida = None, R
        for cand in (ang, ang + np.pi):
            Rc = gira_em_torno(R, palma, cand)
            prox = float(np.trace(Rc.T @ R))     # 3 = idêntica, -1 = oposta
            if melhor is None or prox > melhor:
                melhor, escolhida = prox, Rc
        return escolhida

    def T_pelvis(self):
        """Pose da pelve no mundo. O IK trabalha no referencial da raiz do URDF,
        e a cena põe o robô num lugar arbitrário — sem esta conversão o alvo vai
        para o lugar errado e o IK 'converge' para uma pose sem sentido."""
        R = self.d.xmat[self.id_pelvis].reshape(3, 3)
        return matriz(self.d.xpos[self.id_pelvis], R)

    def para_pelvis(self, T_mundo):
        return np.linalg.inv(self.T_pelvis()) @ T_mundo

    def punho_no_mundo(self):
        return np.array(self.d.xpos[self.id_punho])

    def ponto_pega(self):
        """Centro da pinça: média das pontas do polegar, indicador e médio."""
        return np.mean([self.d.xpos[i] for i in self.id_dedos], axis=0)

    # ── cena ────────────────────────────────────────────────────────────────
    def sorteia_copo(self, rng):
        """Xícara em algum lugar da METADE DIREITA da mesa (y negativo).

        Antes ela caía em y ∈ [-0,09, +0,09] e o coador em qualquer um dos dois
        lados, então as duas metades da tarefa mudavam de mão de episódio para
        episódio. Fixando xícara à direita e coador à esquerda (`sorteia_alvo`),
        todo episódio tem a mesma ESTRUTURA — pegar do lado do braço que age,
        atravessar o corpo, pousar do outro lado — e o que varia é só onde,
        dentro de cada metade. É o que se quer numa demonstração: variação de
        posição, não de roteiro.

        AFASTADO DO CORPO em x, e a faixa é apertada: ver a varredura no
        `X_COPO`. Abaixo dela o cotovelo não tem para onde ir (o par
        `torso_link ↔ right_elbow_link`, e depois o próprio tampo, bloqueiam a
        descida); acima, o ponto sai do alcance. A mesa vai de x=0,08 a 0,88 —
        sobra tampo de sobra, o que falta é braço.
        """
        pos = np.array([rng.uniform(*X_COPO),
                        -rng.uniform(*Y_DIREITA),
                        self.z_copo])
        self.d.qpos[self.copo_qpos:self.copo_qpos + 3] = pos
        self.d.qpos[self.copo_qpos + 3:self.copo_qpos + 7] = [1, 0, 0, 0]
        self.d.qvel[self.copo_dof:self.copo_dof + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        return pos

    def sorteia_alvo(self, rng):
        """Coador em algum lugar da METADE ESQUERDA da mesa (y positivo).

        Estava embutido no `episodio` como `rng.choice([-1, 1]) * ...`, que
        sorteava o LADO junto com a posição. Agora o lado é fixo — ver
        `sorteia_copo`. O z é o do TAMPO: a xícara pousa no PRATO DA BASE do
        coador, embaixo do funil, que é onde ela fica num suporte de verdade.
        Uma versão anterior pousava na BOCA do funil, em cima; errado.
        """
        # COADOR FIXO (18/09). Ele sorteava dentro de X_ALVO x Y_ESQUERDA, e para a
        # tarefa `pega` — que termina ao LEVANTAR a xicara e nunca chega ao coador —
        # isso so acrescenta um objeto que anda pela mesa de episodio para episodio,
        # sem nada no roteiro que dependa de onde ele esta. Ruido visual puro no
        # dataset. Fixo no centro das duas faixas, ele vira parte estavel do cenario.
        # Para voltar a sortear, e so descomentar a linha de baixo.
        # return np.array([rng.uniform(*X_ALVO), rng.uniform(*Y_ESQUERDA), self.z_copo])
        return np.array([sum(X_ALVO) / 2.0, sum(Y_ESQUERDA) / 2.0, self.z_copo])

    def poe_copo(self, pos):
        # O tampo como piso. Sem isto a xícara acompanha a mão para dentro da
        # mesa no fim do `pousar` — em cinemática pura nada a impede, e na tela
        # ela some dentro do móvel um instante antes de ser solta.
        pos = np.asarray(pos, dtype=float).copy()
        pos[2] = max(float(pos[2]), self.z_copo)
        self.d.qpos[self.copo_qpos:self.copo_qpos + 3] = pos
        self.d.qvel[self.copo_dof:self.copo_dof + 6] = 0

    def desenha_alvo(self, alvo):
        """Move o X (body mocap `marca_alvo`) para o destino sorteado.

        Um X deitado na mesa, não um disco no ar: duas barras finas cruzadas a
        45 graus, na altura do tampo. O disco flutuante de uma versão anterior
        ficava na altura do CENTRO do copo e parecia um objeto da cena; um X
        rente à mesa se lê como marcação de destino, que é o que ele é.

        ── Por que mocap, e não `viewer.user_scn` ──────────────────────────
        Esta função ANTES desenhava no `user_scn` e saía cedo quando não havia
        janela. O `user_scn` é decoração do viewer: os renderizadores
        offscreen chamam `update_scene`, que remonta a cena a partir do
        MODELO, e nada do `user_scn` chega lá. O resultado é que o X não
        aparecia em NENHUM quadro gravado — e sem janela nem na tela.

        Isso não era um detalhe cosmético: o alvo é sorteado a cada episódio,
        hoje dentro da metade esquerda da mesa (`sorteia_alvo`). Sem o coador
        na imagem, o dataset mostra a mesma cena com ações diferentes, e o
        destino fica impossível de inferir. Um dataset assim não é difícil, é
        ambíguo. (O lado deixou de ser sorteado, mas a posição dentro dele
        não — o marcador continua sendo a única pista de para onde ir.)
        """
        self.d.mocap_pos[self.mocap_alvo] = (
            alvo[0], alvo[1], self.z_mesa + 0.002)
        # `mj_forward` para o marcador já estar no lugar certo se alguém
        # renderizar antes do próximo passo de física.
        mujoco.mj_forward(self.m, self.d)

    def bateu_no_coador(self):
        """Braço direito dentro do coador.

        Sem isto o especialista não sabe que o coador existe: as duas outras
        checagens leem pares de geoms fixos (mesa×antebraço, braço×tronco) e
        nenhuma delas vê o `coador_boca`. O resultado na tela era a mão
        atravessando o funil no caminho do pouso.
        """
        if self.geom_coador < 0:
            return False
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            if g1 == self.geom_coador and g2 in self.geoms_dir_livre:
                return True
            if g2 == self.geom_coador and g1 in self.geoms_dir_livre:
                return True
        return False

    def pose_colide(self):
        """Alguma parte do braço direito onde não pode estar: cotovelo ou punho
        na mesa, cotovelo/punho/mão dentro do tronco, ou o braço dentro do
        coador. A mão perto do tampo é permitida — é onde a xícara está."""
        # `bateu_no_coador()` está de volta. Ele ficou desligado enquanto o
        # transporte corria a 14 cm acima do alvo (18,8 cm do tampo), porque
        # isso cai DENTRO do bloco do coador (15,6 a 19,5 cm): o destino era
        # ele próprio uma colisão e o braço travava em 5 de 8 episódios. Com o
        # transporte a 24 cm o destino passou para 28,8 cm, acima do bloco, e
        # a checagem volta a fazer sentido — agora o braço desvia do coador
        # durante todo o movimento em vez de atravessá-lo.
        return self.bateu_na_mesa() or self.auto_colisao()

    def pares_colisao(self):
        """Quem está tocando quem — para depurar sem adivinhar."""
        out = []
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            if (g1 in self.geoms_dir_livre and g2 in self.geoms_tronco) or \
               (g2 in self.geoms_dir_livre and g1 in self.geoms_tronco):
                out.append((self.m.body(self.m.geom_bodyid[g1]).name,
                            self.m.body(self.m.geom_bodyid[g2]).name))
        return sorted(set(out))

    def auto_colisao(self):
        """Braço direito tocando o próprio corpo.

        Em cinemática pura nada impede o braço de atravessar o tronco, e foi o
        que apareceu na tela. Aqui só se olha o par que importa — geom do braço
        direito contra geom do tronco, da cintura ou do lado esquerdo — porque
        contato entre elos VIZINHOS do mesmo braço é normal e marcaria tudo.
        """
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            if g1 in self.geoms_dir_livre and g2 in self.geoms_tronco:
                return True
            if g2 in self.geoms_dir_livre and g1 in self.geoms_tronco:
                return True
        return False

    def aponta_cintura(self, alvo, fracao=1.0):
        """Gira o tronco na direção do alvo, como no robô real com
        `use_waist_yaw=True`. Ponto longe ou de lado => o tronco vira e leva o
        ombro para perto, que é como o braço alcança o que não alcançaria parado.

        O IK foi montado sobre um robô REDUZIDO só com as juntas do braço, então
        ele não sabe que a cintura girou. Isso é erro de modelo — e a malha
        externa o absorve, porque ela corrige pelo que o MuJoCo MEDE, não pelo
        que o IK acredita. Sem a malha externa, mexer na cintura mandaria o
        braço para o lugar errado.
        """
        yaw = float(np.clip(np.arctan2(alvo[1], alvo[0]), -self.lim_cintura, self.lim_cintura))
        atual = float(self.d.qpos[self.adr_cintura])
        self.d.qpos[self.adr_cintura] = atual + fracao * (yaw - atual)

    def bateu_na_mesa(self):
        """Algum geom do robô tocando o tampo. Não impede — marca. Episódio que
        raspa a mesa não serve como demonstração."""
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            # Só o braço DIREITO. O esquerdo fica na pose pendurada e encosta
            # no tampo o tempo todo — contá-lo faz a rejeição de colisão
            # bloquear no passo zero de todo episódio, e o braço que interessa
            # nunca chega a se mover.
            # A MÃO pode chegar perto do tampo — é onde a xícara está, e
            # bloquear isso torna a pega impossível. O que não pode entrar na
            # mesa é o ANTEBRAÇO e o cotovelo, que foi o que apareceu na tela.
            if (g1 == self.geom_mesa and g2 in self.geoms_antebraco) or \
               (g2 == self.geom_mesa and g1 in self.geoms_antebraco):
                # MARGEM: só conta se ENTRAR na mesa, não se encostar. O
                # `contact.dist` é negativo quando há penetração; zero é toque
                # exato. Sem esta folga, um roçar de fração de milímetro conta
                # como colisão, e o `resolve_livre` responde jogando o cotovelo
                # PARA CIMA para escapar — que é o movimento estranho que
                # aparece na tela. Com 5 mm de tolerância o braço pode passar
                # rente ao tampo, como um braço humano passa.
                if self.d.contact[c].dist < -MARGEM_MESA:
                    return True
        return False

    def pousa_copo(self, xy, z=None):
        """Apoia o copo em (x, y), na altura `z` (padrão: o tampo).

        Sem isto ele fica onde a mão soltou — no ar, porque em cinemática pura
        não há gravidade. O `z` existe para pousar na boca do coador em vez do
        tampo; ver `z_boca_coador`."""
        z = self.z_copo if z is None else float(z)
        self.d.qpos[self.copo_qpos:self.copo_qpos + 3] = [xy[0], xy[1], z]
        self.d.qpos[self.copo_qpos + 3:self.copo_qpos + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)

    def define_pose_partida(self, ik, ponto):
        """Calcula por IK uma pose de prontidão ACIMA da mesa e a adota como
        início de todo episódio.

        O keyframe `home` do modelo deixa os braços pendurados, e com a mesa na
        altura do cotovelo isso põe as duas mãos DENTRO do tampo — medido: 11
        contatos mesa↔mão já na pose inicial. Começar dali faz a rejeição de
        colisão bloquear no passo zero (o braço já está em contato e qualquer
        movimento parece piorar), e o episódio morre antes de existir.
        """
        q = self.q_braco0.copy()
        T_esq = self.para_pelvis(matriz(self.d.xpos[self.id_pelvis] + np.array([0.05, 0.30, 0.35])))
        correcao = np.zeros(3)
        for _ in range(120):
            T = self.para_pelvis(matriz(ponto + correcao, self.R_ref))
            sol, _ = ik.solve_ik(T_esq, T, q)
            q = q + 0.2 * (np.asarray(sol) - q)
            self.d.qpos[self.q_braco] = q
            self.abre_mao()
            mujoco.mj_forward(self.m, self.d)
            erro = ponto - self.ponto_pega()
            # Aqui a correção pode ser rápida: é cálculo de pose única, fora do
            # laço de controle, e ninguém vê o transiente.
            correcao = np.clip(correcao + 0.15 * erro, -0.25, 0.25)
            if np.linalg.norm(erro) < 0.02:
                break
        self.q_braco0 = q.copy()
        if self.bateu_na_mesa() or self.auto_colisao():
            print("   ⚠️  a pose de partida ainda toca algo — suba o ponto de descanso")
        return q

    def parte_com_copo(self, ik, ponto, palma, passos=150, tol=0.02):
        """Braço DIRETO no fim do `levantar`: mão em `ponto`, dedos fechados na
        xícara, sem nunca ter feito a pega.

        É o estado inicial da segunda metade da tarefa (`--tarefa=poe`), e
        substitui a versão anterior, que executava a pega inteira em silêncio só
        para chegar aqui. Aquilo funcionava mas custava ~60% do tempo de cada
        episódio, e descartava o episódio inteiro quando a pega falhava — para
        um dataset em que a pega não aparece em quadro nenhum.

        A pose não é inventada: é a MESMA construção do `define_pose_partida` —
        IK mais um integrador de correção, que é o que absorve o erro de modelo
        da cintura (o IK não sabe que ela girou) — só que mirando o ponto de
        transporte em vez do de descanso, e com a xícara na mão.

        A xícara é fechada com o deslocamento `palma` aplicado, e não centrada
        na pinça. Durante a pega de verdade a mão desce sobre a xícara com esse
        offset (ver o `palma` do `episodio`), então os dedos param em ângulos
        que correspondem a uma xícara descentrada; centrá-la aqui fecharia os
        dedos um pouco mais. Só depois de fechada ela vai para o centro da
        pinça, que é onde o `executa_fase` a mantém durante o transporte.

        Devolve `(q, ok)`. `ok=False` quando o ponto não é alcançável ou quando
        nenhum dedo encosta — o episódio é descartado como qualquer outro.
        """
        q = self.q_braco0.copy()
        self.aponta_cintura(ponto, fracao=1.0)
        R = self.R_lateral(ponto)
        T_esq = self.para_pelvis(matriz(self.d.xpos[self.id_pelvis] + np.array([0.05, 0.30, 0.35])))
        correcao = np.zeros(3)
        erro = np.full(3, np.inf)
        for _ in range(passos):
            T = self.para_pelvis(matriz(ponto + correcao, R))
            sol, _ = ik.solve_ik(T_esq, T, q)
            q = q + 0.2 * (np.asarray(sol) - q)
            self.d.qpos[self.q_braco] = q
            self.abre_mao()
            # A xícara acompanha a mão durante a busca: sem isso ela fica na
            # mesa e o `fecha_mao_ate_tocar` do fim não acha o que agarrar.
            self.poe_copo(self.ponto_pega() - palma)
            mujoco.mj_forward(self.m, self.d)
            erro = ponto - self.ponto_pega()
            correcao = np.clip(correcao + 0.15 * erro, -0.25, 0.25)
            if np.linalg.norm(erro) < tol:
                break
        if np.linalg.norm(erro) >= tol:
            print(f"   partida     FALHOU — ponto de transporte fora do alcance "
                  f"({np.linalg.norm(erro) * 100:.1f} cm)")
            return q, False
        n_dedos = self.fecha_mao_ate_tocar()
        self.poe_copo(self.ponto_pega())
        mujoco.mj_forward(self.m, self.d)
        print(f"   partida      {'ok ' if n_dedos else 'FALHOU'} — já com a xícara "
              f"na mão, {n_dedos} de {len(self.mao)} dedos encostados")
        return q, n_dedos > 0

    def reset_braco(self, ik=None):
        """Braço na pose inicial, mão aberta, cintura reta. Também zera o ponto
        de partida do IK: ele faz warm start da última solução, e começar do
        lugar errado leva o otimizador para um mínimo local esquisito — foi o
        que fez o episódio 2 falhar por herança do episódio 1."""
        self.d.qpos[self.q_braco] = self.q_braco0
        self.d.qpos[self.adr_cintura] = 0.0
        self.abre_mao()
        mujoco.mj_forward(self.m, self.d)
        if ik is not None:
            ik.init_data = self.q_braco0.copy()
        return self.q_braco0.copy()

    def abre_mao(self):
        for j in self.mao:
            self.d.qpos[j["adr"]] = 0.0

    def fecha_mao_ate_tocar(self, passos=45, ao_passo=None):
        """Fecha cada junta até encostar na xícara e para ali.

        Em cinemática pura nada impede o dedo de atravessar o objeto — cravar a
        pose fechada faz a mão fechar DENTRO da xícara, que na tela é
        exatamente "não está pegando". Aqui cada junta avança um incremento por
        vez e congela quando um geom dela aparece num contato com o copo. O
        `mj_forward` já roda a detecção de colisão, então isso não custa física:
        custa varrer `d.ncon`.

        Devolve quantos dedos encostaram — zero é pega falhada, e é o sinal que
        o gerador vai usar para descartar o episódio.
        """
        travadas = [False] * len(self.mao)
        for k in range(1, passos + 1):
            for i, j in enumerate(self.mao):
                if not travadas[i]:
                    self.d.qpos[j["adr"]] = j["alvo"] * (k / passos)
            mujoco.mj_forward(self.m, self.d)

            tocando = set()
            for c in range(self.d.ncon):
                g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
                if g1 in self.geom_copo:
                    tocando.add(g2)
                elif g2 in self.geom_copo:
                    tocando.add(g1)
            # Só a partir de 25% do curso. A mão desce SOBRE a xícara, então
            # indicador e médio já estão encostando nela quando o fechamento
            # começa — congelar no primeiro contato os deixa parados em zero, e
            # na tela só o polegar fecha. Este piso obriga cada dedo a envolver
            # o objeto antes de poder travar.
            if k > passos * 0.25:
                for i, j in enumerate(self.mao):
                    if not travadas[i] and (j["geoms"] & tocando):
                        travadas[i] = True

            if ao_passo is not None:
                ao_passo("fechar", self.d.qpos[self.q_braco].copy())
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(1.0 / FPS)
            if all(travadas):
                break
        return sum(travadas)

    def mostra(self):
        mujoco.mj_forward(self.m, self.d)
        if self.viewer is not None:
            self.viewer.sync()

    def viva(self):
        return self.viewer is None or self.viewer.is_running()


def gira_em_torno(R, eixo_mundo, ang):
    """Rotaciona R em torno de um eixo do mundo (Rodrigues)."""
    e = eixo_mundo / (np.linalg.norm(eixo_mundo) + 1e-9)
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    return (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)) @ R


def resolve_livre(cena, ik, T_esq, pos_alvo, R_base, q_atual, ganho, fechar_mao,
                  tentativas=12):
    """Uma pose que alcança o alvo E não colide, procurada no espaço nulo.

    O braço tem 7 juntas para 6 graus de liberdade da mão: para a MESMA pose de
    punho existe uma família de configurações, que diferem essencialmente por
    onde fica o cotovelo. É isso que se explora aqui.

    A primeira tentativa é o de sempre — warm start da pose atual, que dá
    continuidade ao movimento. Se a solução colide, as tentativas seguintes
    reiniciam o otimizador de sementes perturbadas, e o ipopt cai em outro ramo
    da família. A primeira livre de colisão ganha.

    Por que isto e não uma restrição dentro do casadi: a restrição exigiria a
    DISTÂNCIA de colisão e o gradiente dela dentro do otimizador (hpp-fcl
    diferenciável), que é outro projeto. Aqui o teste de colisão é o do MuJoCo,
    completo, com a malha real de todos os elos — e ele já está carregado.

    Devolve (q, True) se achou pose livre; (q_atual, False) se todas colidem.
    """
    guarda = q_atual.copy()
    # A pose da mão é 6D, mas apontar a palma para o copo prende só DUAS
    # direções — o giro EM TORNO do eixo da palma continua livre, e é um grau
    # de liberdade a mais para escapar do tronco sem mudar para onde a mão
    # olha. A busca usa os dois: reconfiguração do cotovelo (espaço nulo das
    # juntas) e rolagem da mão (espaço nulo da tarefa).
    palma_mundo = R_base @ cena.v_palma
    for t in range(tentativas):
        R_try = R_base if t == 0 else gira_em_torno(
            R_base, palma_mundo, np.random.uniform(-0.8, 0.8))
        T_alvo = cena.para_pelvis(matriz(pos_alvo, R_try))

        if t == 0:
            semente = q_atual
        else:
            # VARREDURA DIRIGIDA, não sorteio. O par que colide é sempre
            # `torso_link ↔ right_elbow_link`, e a junta que AFASTA o cotovelo
            # do corpo é a abertura do ombro (`right_shoulder_roll`, índice 8).
            # Sortear os dois sinais desperdiça metade das tentativas indo na
            # direção que piora; aqui o sinal alterna e a amplitude cresce, de
            # modo que as primeiras tentativas já são as mais promissoras.
            passo_abre = 0.20 * ((t + 1) // 2)
            sinal = 1.0 if t % 2 else -1.0
            ruido = np.zeros_like(q_atual)
            ruido[8] = sinal * passo_abre                    # right_shoulder_roll
            ruido[9] = np.random.uniform(-0.15, 0.15)        # right_shoulder_yaw
            ruido[10] = np.random.uniform(-0.15, 0.15)       # right_elbow
            semente = q_atual + ruido

        ik.init_data = np.asarray(semente, dtype=float)
        sol, _ = ik.solve_ik(T_esq, T_alvo, semente)
        cand = q_atual + ganho * (np.asarray(sol) - q_atual)

        cena.d.qpos[cena.q_braco] = cand
        if not fechar_mao:
            cena.abre_mao()
        mujoco.mj_forward(cena.m, cena.d)
        if not cena.pose_colide():
            return cand, True

    # Guarda quem colidiu na ÚLTIMA tentativa, antes de desfazer — depois do
    # revert a informação some, e o relatório fica dizendo "colidiu com nada".
    cena.ultimos_pares = cena.pares_colisao() or [("mesa", "?")]
    cena.d.qpos[cena.q_braco] = guarda
    mujoco.mj_forward(cena.m, cena.d)
    return guarda, False


def executa_fase(cena, ik, nome, alvo, q_atual, fechar_mao, copo_na_mao,
                 max_passos, ganho, dorme, tol=TOLERANCIA, foco=None, ao_passo=None):
    """Leva o CENTRO DA PINÇA até `alvo` (no mundo), girando a cintura junto.

    `ganho` é a velocidade: quanto da solução do IK entra por passo. Fases de
    espaço livre podem ser rápidas; descer sobre a xícara e pousar não — na
    versão anterior tudo usava 0,25 e o movimento ficava com cara de teletransporte,
    inútil como demonstração.

    O teste de progresso continua sendo o que impede o laço infinito: alvo fora
    do alcance faz o IK devolver sempre a mesma solução, a distância congela, e
    sem isto o laço roda para sempre com a tela parada.
    """
    melhor, parado, bloqueios = np.inf, 0, 0
    correcao = np.zeros(3)
    desvio = np.zeros(3)      # desvio para cima quando a colisão bloqueia
    # Sem foco, a mão vai RETA: alinhada ao tronco, sem inclinação. A
    # `R_ref` é a orientação do punho na pose de descanso, e ela já vem
    # inclinada — era isso que aparecia na tela como a xícara sendo carregada
    # torta. `R_reta` monta a orientação a partir dos eixos do próprio tronco.
    # Sem foco a mão também vai RETA (palma para baixo, só yaw): é o mesmo
    # critério das fases com foco, só que mirando o próprio destino da fase.
    R = cena.R_lateral(alvo) if foco is None else None
    R_travado = None
    T_esq = cena.para_pelvis(matriz(cena.d.xpos[cena.id_pelvis] + np.array([0.05, 0.30, 0.35])))
    raspou = colidiu = False

    for passo in range(max_passos):
        if not cena.viva():
            return q_atual, False, "janela fechada", False

        # CINTURA: entra junto com o braço, um pouco por passo. Se o ponto está
        # longe ou de lado, o tronco gira e leva o ombro para perto — que é como
        # o robô real alcança, e o que o dataset gravou com `use_waist_yaw=True`.
        cena.aponta_cintura(alvo, fracao=0.06)

        # Palma sempre voltada para o foco da fase (o centro da xícara na
        # aproximação, o X ao pousar). Recalculada a cada passo: enquanto o
        # braço se move, a direção certa muda.
        # A orientação persegue o foco só ENQUANTO o braço está longe. Perto do
        # alvo, recalcular a cada passo faz o alvo se mexer junto com a mão —
        # perseguição de alvo móvel, que é outra fonte de zigue-zague. A menos
        # de 8 cm ela congela no último valor.
        # Mira CONTÍNUA, congelada perto do alvo (a "opção 1"). Calcular a
        # orientação a partir do destino da fase deixa a descida perfeitamente
        # vertical, mas é mais restritivo: medido, a taxa de episódios completos
        # caiu de 6/6 para 2/4, porque poses que o braço alcança na diagonal ele
        # não alcança reto. Fica a diagonal e ficam os episódios.
        if foco is not None and (R_travado is None or melhor > 0.08):
            # SEM `de=alvo`, apesar do que o docstring do `orienta_para`
            # recomenda. Medido em 03/09: passar o destino faz a palma apontar
            # reto para baixo — que é o que se quer visualmente — mas exige uma
            # pose de punho que este braço NÃO alcança nas posições em que a
            # xícara é sorteada. Resultado: 1 episódio salvo em 8, com as
            # falhas em `sobre copo — fora do alcance mesmo com a cintura`.
            # A inclinação da mão na chegada não é defeito, é a folga que o IK
            # usa para alcançar.
            R_travado = cena.R_lateral(foco)
        R_passo = R_travado if foco is not None else R
        candidato, livre = resolve_livre(cena, ik, T_esq, alvo + correcao + desvio,
                                         R_passo, q_atual,
                                         ganho * perfil_humano(passo), fechar_mao)

        # PASSO COM REJEIÇÃO DE COLISÃO. Detectar não basta: em cinemática pura
        # nada impede o braço de atravessar a mesa nem o próprio tronco, e foi o
        # que apareceu na tela. Aqui a pose candidata é escrita, o
        # `mj_forward` roda a detecção, e se houver interpenetração ela é
        # DESFEITA — o braço fica onde estava.
        #
        # Só desfazer trava: o IK devolveria a mesma solução ruim para sempre.
        # Por isso, quando bloqueia, o alvo ganha um desvio para CIMA, que é
        # para onde há espaço livre acima da mesa. O braço então contorna por
        # cima em vez de insistir contra o obstáculo.
        anterior = q_atual.copy()
        bate_mesa = bate_corpo = False
        if not livre:
            # Nenhuma das configurações do espaço nulo serve neste alvo: o
            # obstáculo está no caminho da MÃO, não do cotovelo. Aí a saída é
            # mudar o alvo — subir e afastar — até abrir passagem.
            bate_mesa, bate_corpo = cena.bateu_na_mesa(), cena.auto_colisao()
            raspou |= bate_mesa
            colidiu |= bate_corpo

        # REJEITAR a pose que colide só funciona se o IK souber do obstáculo.
        # Ele não sabe: devolve a mesma solução proibida a cada passo, a
        # rejeição a desfaz, e o movimento trava — medido, o episódio morre na
        # descida sobre a xícara. Evitar colisão de verdade é restrição dentro
        # do otimizador (casadi), não filtro depois dele.
        #
        # Por padrão, então, apenas MARCA: o episódio completa e sai rotulado
        # como sujo, e o gerador descarta. Com `--rejeita-colisao` o filtro
        # entra, para quem quiser ver onde ele trava.
        if not livre:
            cena.d.qpos[cena.q_braco] = anterior
            mujoco.mj_forward(cena.m, cena.d)
            # Escape para CIMA e para FORA. Só subir não resolve quando o
            # obstáculo é o próprio tronco: a xícara centrada obriga o braço a
            # passar na frente do corpo, e a saída é afastar o cotovelo para o
            # lado direito (y negativo) enquanto sobe.
            fuga = np.array([0.0, -0.012, 0.015]) if bate_corpo else np.array([0.0, 0.0, 0.02])
            desvio = np.clip(desvio + fuga, [-0.02, -0.20, 0.0], [0.02, 0.02, 0.20])
            bloqueios += 1
            if bloqueios > 3 * PACIENCIA:
                detalhe = " " + str(getattr(cena, "ultimos_pares", [])[:3])
                return (anterior, False,
                        f"bloqueado por {'mesa' if bate_mesa else 'auto-colisao'} "
                        f"a {melhor * 100:.1f} cm{detalhe}", True)
            if dorme:
                time.sleep(1.0 / FPS)
            continue

        # Passa-baixa na pose comandada. Quando a busca no espaço nulo troca de
        # ramo entre um passo e outro, a solução salta; misturar com a anterior
        # transforma o salto numa transição e tira o tremor sem atrasar o
        # movimento de forma perceptível.
        q_atual = 0.65 * q_atual + 0.35 * candidato
        # Alívio lento do desvio: uma vez livre, volta a mirar o alvo de verdade.
        desvio *= 0.97
        cena.mostra()
        if copo_na_mao:
            cena.poe_copo(cena.ponto_pega())
            cena.mostra()

        erro = alvo - cena.ponto_pega()
        # 0,05 e não 0,15. Esta correção é um INTEGRADOR, e o ganho do braço
        # varia ao longo da fase por causa do perfil de velocidade: integrador
        # rápido sobre planta de ganho variável oscila, e a oscilação aparece na
        # tela como o zigue-zague ao chegar na xícara. Devagar ela ainda anula o
        # erro de referencial, só que sem sobressalto.
        correcao = np.clip(correcao + 0.05 * erro, -0.25, 0.25)
        dist = float(np.linalg.norm(erro))

        if dist < tol:
            aviso = (" [raspou a mesa]" if raspou else "") + (" [auto-colisao]" if colidiu else "")
            return q_atual, True, f"{dist * 100:.1f} cm em {passo} passos{aviso}", raspou or colidiu
        if dist < melhor - 2e-5:
            melhor, parado = dist, 0
        else:
            parado += 1
            if parado > PACIENCIA:
                return q_atual, False, (f"parou a {melhor * 100:.1f} cm "
                                        f"(fora do alcance mesmo com a cintura)"), True
        if ao_passo is not None:
            ao_passo(nome, q_atual)
        if dorme:
            time.sleep(1.0 / FPS)

    return q_atual, False, f"estourou {max_passos} passos, ficou a {melhor * 100:.1f} cm", True


# A fase que fecha a primeira metade. Ver `episodio`.
FIM_DA_PEGA = "levantar"
TAREFAS = ("completo", "pega", "poe")


def episodio(cena, ik, rng, n, dorme, ao_passo=None, tarefa="completo"):
    q = cena.reset_braco(ik)
    copo = cena.sorteia_copo(rng)
    alvo = cena.sorteia_alvo(rng)
    cena.desenha_alvo(alvo)

    # A pinça entra um pouco além do centro do copo, na direção da palma: mirar
    # o centro exato deixa a xícara na PONTA dos dedos, e ela escorrega para
    # fora quando o braço levanta. Este deslocamento a encaixa mais fundo.
    palma = cena.punho_no_mundo() - cena.ponto_pega()
    palma = palma / (np.linalg.norm(palma) + 1e-9) * 0.025

    # DUAS alturas, e a distinção não é cosmética. `alto` é a aproximação da
    # PEGA: 14 cm acima da xícara, e subir isso põe o ponto fora do alcance do
    # braço — medido, 26 cm aqui faz a fase `sobre copo` falhar com "fora do
    # alcance mesmo com a cintura".
    # `alto_carga` é a altura de TRANSPORTE, e essa precisa passar do funil do
    # coador, cujo topo está a 19,5 cm do tampo. O fundo da xícara fica
    # exatamente `alto_carga` acima do tampo (a meia-altura da xícara entra na
    # conta do alvo e sai na do fundo, e as duas se cancelam), então 22 cm dão
    # 2,5 cm de folga. 26 cm foram testados e são pior: a descida mais íngreme
    # faz o antebraço bater na mesa e o `pousar` falhou em 4 de 5.
    # 17 cm: 14 mais os 3 pedidos para folgar a entrada no coador.
    # ── Alturas ────────────────────────────────────────────────────────────
    # `alto` é a aproximação da PEGA. Fica em 14 cm: medido, subir aqui põe o
    # ponto fora do alcance e a fase `sobre copo` falha com "fora do alcance
    # mesmo com a cintura".
    alto = np.array([0.0, 0.0, 0.14])

    # `alto_carga` é a altura de TRANSPORTE, com a xícara na mão. O fundo dela
    # viaja exatamente `alto_carga` acima do tampo (a meia-altura entra na conta
    # do alvo e sai na do fundo), e o topo do funil do coador está a 19,5 cm —
    # então 24 cm passam por cima com folga em vez de atravessar o funil.
    alto_carga = np.array([0.0, 0.0, 0.24])

    # ENTRADA LATERAL TENTADA E REVERTIDA (04/09). Medido na malha, o funil
    # ocupa de 8,8 a 19,5 cm com raio até 4,7 cm, e abaixo disso não há nada —
    # então descer AO LADO e entrar na horizontal por baixo dele evitaria o
    # funil por geometria, sem depender de detecção de colisão. Não funciona
    # com este braço: com 12 cm de afastamento em y, 0 de 8 episódios; com
    # 6 cm, 1 de 8. As falhas são `transportar`/`descer lado` "fora do alcance"
    # — o ponto lateral sai do envelope. É a quinta abordagem geométrica que
    # falha pelo mesmo motivo, e a conclusão é que resolver isto exige um
    # PLANEJADOR (cuRobo/MoveIt), não waypoints escolhidos à mão.
    z_entrada = cena.z_mesa + 0.08          # a mão para 8 cm acima do tampo

    descanso = cena.descanso

    RAPIDO, LENTO = 0.16, 0.07

    print(f"\n── episódio {n} [{tarefa}] "
          f"| copo ({copo[0]:.2f}, {copo[1]:.2f}) "
          f"| coador ({alvo[0]:.2f}, {alvo[1]:.2f})")

    fases = [
        # nome,        destino,            mão,   copo,  passos, ganho
        # nome,       destino,        mão,   copo,  passos, ganho,  foco da palma
        ("partida",    descanso,       False, False, 500, RAPIDO, None),
        # TENTATIVA DE APROXIMAÇÃO LATERAL REVERTIDA (04/09). Substituir estas
        # duas fases por um ponto de pré-pega ao lado do copo falhou: com 12,5
        # cm de afastamento, 0 de 8 episódios; com 6,2 cm, 1 de 8. Em ambos a
        # fase "ao lado" estourava o orçamento de passos — o ponto lateral cai
        # fora do que o braço alcança nessa faixa da mesa.
        # Para a pega lateral funcionar é preciso um PLANEJADOR de trajetória
        # (OMPL/MoveIt/cuRobo), não waypoints escolhidos à mão: o que existe
        # aqui é um seguidor de waypoints com IK e teste de colisão, e ele não
        # tem como achar o caminho sozinho.
        ("sobre copo", copo + alto,    False, False, 600, RAPIDO, copo),
        ("descer",     copo + palma,   False, False, 700, LENTO,  copo),
        ("fechar",     copo,           True,  False, 0,   0,      None),
        # `foco=None` nas três fases COM A XÍCARA NA MÃO. Com um foco, a
        # `orienta_para` re-aponta a palma a cada passo e a mão vai girando
        # durante o transporte, levando a xícara junto — na tela ela chega
        # tombando. Sem foco, a orientação fica travada na `R_ref` e a xícara
        # viaja reta do começo ao fim, que é o que se quer.
        ("levantar",   copo + alto_carga, True, True, 500, LENTO,  None),
        # TRAZ PARA PERTO DO CORPO antes de ir ao suporte. Sem esta fase o
        # braço vai do copo direto ao coador pelo caminho mais curto, que é uma
        # diagonal esticada — na tela parece que ele "leva direto", sem recolher
        # o braço. Aqui a xícara vem primeiro para o meio do corpo, na altura de
        # transporte, e só então segue para o alvo.
        # O x é o do DESCANSO, que é uma pose sabidamente alcançável (foi
        # calculada por IK no início do episódio); usar um valor arbitrário mais
        # perto do tronco cai na faixa onde o cotovelo bate no corpo.
        ("recolher",   np.array([cena.descanso[0], (copo[1] + alvo[1]) / 2.0,
                                 alvo[2]]) + alto_carga, True, True, 500, RAPIDO, None),
        ("transportar", alvo + alto_carga, True, True, 600, RAPIDO, None),
        # A MÃO para a 8 cm DO TAMPO, e não desce até encostar. É altura
        # absoluta, medida da mesa: `alvo[2]` já está 4,8 cm acima dela (meia
        # altura da xícara), então somar 8 ali daria 12,8 cm.
        # A xícara é posta no lugar pelo `pousa_copo` no `soltar`, que escreve
        # a posição dela direto — a mão não precisa levá-la até o fim, e descer
        # até o tampo é o que fazia o antebraço raspar a mesa.
        ("pousar",     np.array([alvo[0], alvo[1], z_entrada]) + palma,
                                       True,  True,  700, LENTO,  None),
        ("soltar",     alvo,           False, False, 0,   0,      None),
        ("recuar",     alvo + alto_carga, False, False, 400, RAPIDO, None),
        ("voltar",     descanso,       False, False, 400, RAPIDO, None),
    ]

    # ── As duas metades ────────────────────────────────────────────────────
    # `tarefa="pega"`  roda de `partida` até `levantar` e PARA ali, com a
    #                  xícara no ar.
    # `tarefa="poe"`   roda de `recolher` até `voltar`. NÃO faz a pega: o braço
    #                  já nasce no fim do `levantar`, com a xícara na mão, pelo
    #                  `parte_com_copo`. O episódio é só o transporte.
    #
    # Uma versão anterior executava a pega em silêncio (`ao_passo=None`) para
    # chegar a esse estado. O estado saía bom, mas custava ~60% do tempo de cada
    # episódio e derrubava o episódio inteiro quando a pega falhava — para
    # gravar um dataset onde a pega não aparece em quadro nenhum.
    if tarefa not in TAREFAS:
        raise ValueError(f"tarefa desconhecida: {tarefa!r} (use {TAREFAS})")
    corte = [f[0] for f in fases].index(FIM_DA_PEGA) + 1
    if tarefa == "pega":
        fases = fases[:corte]
    elif tarefa == "poe":
        fases = fases[corte:]
        q, ok = cena.parte_com_copo(ik, copo + alto_carga, palma)
        if not ok:
            return False

    sujo = False
    for nome, destino, mao, copo_junto, orcamento, ganho, foco in fases:
        if nome == "fechar":
            n_dedos = cena.fecha_mao_ate_tocar(ao_passo=ao_passo)
            ok = n_dedos > 0
            print(f"   {nome:<12} {'ok ' if ok else 'FALHOU'} — "
                  f"{n_dedos} de {len(cena.mao)} dedos encostaram")
            if not ok:
                return False
            continue
        if nome == "soltar":
            cena.abre_mao()
            cena.pousa_copo(alvo[:2], alvo[2])
            cena.mostra()
            print(f"   {nome:<12} ok  — copo apoiado na mesa em "
                  f"({alvo[0]:.2f}, {alvo[1]:.2f})")
            continue

        # Segurando a xícara, os dedos fechados deslocam o centro da pinça em
        # relação à mão aberta com que os alvos foram calculados: exigir 2 cm aí
        # é exigir o que a geometria não permite, e a fase falha a 3 cm para
        # sempre. As fases com o copo na mão ganham folga.
        # Tolerância por propósito da fase. Pegar e pousar exigem precisão;
        # recuar e voltar são só sair do caminho, e cobrar 2 cm ali faz a fase
        # falhar por nada — foi o que derrubou um episódio com o X do lado
        # oposto do corpo, onde o braço chega esticado.
        if nome in ("recuar", "voltar", "partida"):
            tol = 0.05
        elif copo_junto:
            tol = 0.045
        else:
            tol = TOLERANCIA
        q, ok, motivo, ruim = executa_fase(cena, ik, nome, destino, q, mao, copo_junto,
                                           orcamento, ganho, dorme, tol, foco, ao_passo)
        sujo |= ruim
        print(f"   {nome:<12} {'ok ' if ok else 'FALHOU'} — {motivo}")
        if not ok:
            return False

    if sujo:
        print("   ⚠️  episódio COMPLETO mas SUJO (raspou a mesa ou o próprio corpo) "
              "— descartar na geração de dados")
    return not sujo


def main():
    args = sys.argv[1:]
    global REJEITA
    REJEITA = "--rejeita-colisao" in args
    com_janela = "--sem-janela" not in args
    n_eps = 0
    for a in args:
        if a.startswith("--episodios="):
            n_eps = int(a.split("=", 1)[1])
    semente = 0
    for a in args:
        if a.startswith("--semente="):
            semente = int(a.split("=", 1)[1])
    tarefa = "completo"
    for a in args:
        if a.startswith("--tarefa="):
            tarefa = a.split("=", 1)[1]

    if not CENA.exists():
        print(f"❌ cena não encontrada: {CENA}")
        sys.exit(1)

    print(f"⏳ carregando {CENA.name}...")
    cena = Cena(com_janela)
    print("⏳ montando o IK (Pinocchio, braço reduzido)...")
    ik = G1_29_ArmIK(
        visualization=False,
        # Punho direito preso: sem isto o solver o gira durante o movimento.
        travar_punho_dir=False,
        # Cotovelo DIREITO dobrado (índice 10 = right_elbow_joint, faixa
        # -1.047 a +2.094, zero = esticado). A regularização do IK puxa para
        # esta postura; com o padrão de zeros ela puxava para braço reto, que
        # era a causa do cotovelo esticado — não a busca no espaço nulo.
        postura_ref=np.array([0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 1.0, 0, 0, 0], dtype=float),
    )
    print("✅ pronto.\n"
          f"   janela: {'sim' if com_janela else 'não'} | "
          f"episódios: {n_eps or 'em laço'} | tolerância: {TOLERANCIA * 100:.0f} cm")

    # Ponto de descanso: à direita, acima da mesa, longe do tronco. É daqui que
    # todo episódio parte e para onde ele volta.
    DESCANSO = np.array([0.42, -0.26, cena.z_copo + 0.22])
    print("⏳ calculando a pose de partida...")
    cena.define_pose_partida(ik, DESCANSO)
    cena.descanso = DESCANSO
    print("✅ pose de partida pronta.")

    rng = np.random.default_rng(semente)
    n, sucessos = 0, 0
    try:
        while cena.viva() and (n_eps == 0 or n < n_eps):
            n += 1
            sucessos += bool(episodio(cena, ik, rng, n, dorme=com_janela,
                                      tarefa=tarefa))
    except KeyboardInterrupt:
        print("\ninterrompido.")
    finally:
        print(f"\n== {sucessos} de {n} episódios completos ==")
        if cena.viewer is not None:
            cena.viewer.close()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # o ipopt multithread afunda o FPS
    main()
