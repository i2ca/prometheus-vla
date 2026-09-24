# PROMETHEUS 17/09/2026 — de onde sai a pose do copo e do coador na cena do cafe.
#
# Existe por dois motivos descobertos ao subir a cena pela primeira vez:
#
# 1. O COPO NASCIA DEITADO. A malha `copo_texturizado.obj` e Y-PARA-CIMA (altura 0,092 m no eixo
#    Y, base em y=0), enquanto o IsaacLab e Z-para-cima. Com rotacao identidade ele entra tombado.
#    O nosso MuJoCo ja corrigia isso com euler="1.5708 -1.5708 0" no geom visual; aqui a correcao
#    equivalente e o quaternion de +90 graus em X, que leva o Y da malha para o Z do mundo.
#    A malha do coador ja e Z-para-cima (altura 0,1948 m com escala 0,195, base em z=0), entao
#    ele nao precisa de correcao nenhuma.
#
# 2. NAO DAVA PARA MOVER NADA SEM EDITAR CODIGO. Agora a pose dos dois objetos mora num JSON
#    (`~/cena_cafe.json`), lido tanto na hora de criar a cena quanto AO VIVO pelo
#    `tools/cena_cafe_vivo.py`. Quem escreve o JSON e o `~/move_coador.sh` / `~/move_copo.sh`.
#
# O JSON guarda coordenadas do MUNDO em metros, que e o que o simulador consome direto — a
# conversao amigavel ("20 cm a frente do robo") fica na linha de comando, nao aqui, para nao
# haver duas unidades circulando dentro do simulador.
import json
import math
import os

ARQUIVO = os.environ.get("CENA_CAFE", os.path.expanduser("~/cena_cafe.json"))

# Base do robo nesta cena (`pickplace_cafe_g1_29dof_dex3_joint_env_cfg.py`): (-4.2, -3.7, 0.76),
# girado -90 graus em Z. Logo, visto do robo: FRENTE = -Y do mundo, ESQUERDA = +X do mundo.
BASE_ROBO = (-4.2, -3.7)

# Tampo da mesa: MEDIDO na caixa do prim `/World/envs/env_0/PackingTable` (x -5.537..-3.063,
# y -4.581..-3.819, z -0.200..0.794) e confirmado pelo coador que tinha tombado — o ponto mais
# baixo da malha dele parou justamente em 0.7945. O cubo vermelho deles nasce em 0.84 com 6 cm de
# aresta, base em 0.81, ou seja, 1,6 cm acima do tampo: cai um tiquinho e assenta.
TAMPO = 0.794

# Arranjo escolhido em 17/09 olhando a cena com a janela aberta. O COPO fica onde o cubo
# vermelho deles nascia (33 cm a frente, 5 cm a direita do robo) — faixa de alcance que ja
# sabemos que funciona. O COADOR fica 45 cm a frente e 25 cm a esquerda: fora do caminho do
# braco direito, que e quem pega o copo, e ainda dentro da mesa (que vai ate 88 cm de frente).
# Estes numeros sao o PADRAO: valem mesmo sem o `~/cena_cafe.json`, e o arquivo, quando existe,
# manda por cima.
PADRAO = {
    "copo":   {"x": -4.25, "y": -4.03,  "z": TAMPO + 0.002, "giro": 0.0},
    "coador": {"x": -3.95, "y": -4.15,  "z": TAMPO,         "giro": 0.0},
}

# Quaternion (w, x, y, z) que poe cada malha de pe. Ver motivo 1 no topo.
ENDIREITA = {
    "copo":   (0.7071067811865476, 0.7071067811865476, 0.0, 0.0),   # +90 graus em X
    "coador": (1.0, 0.0, 0.0, 0.0),                                  # ja nasce de pe
}


def le(nome: str) -> dict:
    """Pose do objeto: o que estiver no JSON, completado pelo padrao."""
    alvo = dict(PADRAO[nome])
    try:
        with open(ARQUIVO) as f:
            alvo.update(json.load(f).get(nome, {}))
    except (OSError, ValueError):
        pass                      # sem arquivo, ou arquivo pela metade: fica o padrao
    return alvo


def escreve(nome: str, pose: dict) -> dict:
    """Grava a pose de um objeto preservando a do outro."""
    try:
        with open(ARQUIVO) as f:
            tudo = json.load(f)
    except (OSError, ValueError):
        tudo = {}
    tudo[nome] = pose
    with open(ARQUIVO, "w") as f:
        json.dump(tudo, f, indent=2)
        f.write("\n")
    return tudo


def quaternion(nome: str, giro_graus: float):
    """Correcao de eixo da malha, depois girada `giro_graus` em torno do Z DO MUNDO."""
    a = math.radians(giro_graus) / 2.0
    c, s = math.cos(a), math.sin(a)
    w, x, y, z = ENDIREITA[nome]
    # q_giro (c,0,0,s) vezes q_endireita (w,x,y,z)
    return (c * w - s * z,
            c * x - s * y,
            c * y + s * x,
            c * z + s * w)


def pose_mundo(nome: str):
    """(posicao, quaternion) prontos para o `init_state` ou para `write_root_pose_to_sim`."""
    p = le(nome)
    return [float(p["x"]), float(p["y"]), float(p["z"])], list(quaternion(nome, float(p["giro"])))


def do_robo(x: float, y: float):
    """Mundo -> (frente, lado) em metros, do ponto de vista do robo. Lado positivo = esquerda."""
    return BASE_ROBO[1] - y, x - BASE_ROBO[0]


def para_mundo(frente: float, lado: float):
    """(frente, lado) em metros, do ponto de vista do robo -> mundo."""
    return BASE_ROBO[0] + lado, BASE_ROBO[1] - frente
