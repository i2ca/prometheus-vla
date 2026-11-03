"""Gera cena_task3.xml a partir de cena_cafe_medidas.xml: o arranjo da Task 3 do Ψ0 ("Pick bottle, turn and pour into cup"),
como nos episódios gravados deles. Mesa branca à frente com uma garrafa d'água de 500 ml à direita; mesa de madeira à
direita do robô (o modelo gira ~83° para ela) com bandeja azul, caneca verde, dois copinhos e um brinquedo.
O pelve ganha juntas de base (x, y, z, yaw) para o cliente emular o controlador de pernas (AMO) a partir das ações do modelo.
Uso: python gera_cena_task3.py  (--amo-fisico gera cena_task3_amo.xml: pelve livre, pés colidindo, pernas pelo AMO;
--objeto-lab troca a garrafa pelo copo do lab e grava *_copo.xml)
"""
import sys
from pathlib import Path
import mujoco

AQUI = Path(__file__).resolve().parent
J = mujoco.mjtJoint; G = mujoco.mjtGeom
s = mujoco.MjSpec.from_file(str(AQUI / "cena_cafe_medidas.xml"))
# --objeto-lab: teste de generalização com o NOSSO copo (CAN-001, cerâmica branca 90 x Ø80, alça, 312 g, atrito 1, como em
# g1-cup-grasp/scripts/gera_cena_medidas.py) no lugar da garrafa; o corpo vira "garrafa" para o cliente, a auditoria e o reel.
OBJETO_LAB = "--objeto-lab" in sys.argv
for n in ("chaleira", "base_eletrica", "coador", "pote", "tampa", "scoop") + (() if OBJETO_LAB else ("copo",)):
    s.delete(s.body(n))

mesa = s.body("mesa"); mesa.pos = [0.569, -0.239, mesa.pos[2] + 0.03]   # mesa branca e tampo a 0,78 como na câmera real (camera/fit_cena.py):
mesa.quat = [0.99941, 0, 0, -0.03445]                                        # canto frente-esquerdo em (0,354, 0,277), girada -3,95°
for g in mesa.geoms:
    if g.name == "table_visual": g.rgba = [0.90, 0.90, 0.88, 1]
    if g.name == "tampo":   # a cena do café colide com uma caixa maciça do chão ao tampo; aqui as mãos começam abaixo do tampo
        g.rgba = [0, 0, 0, 0]; g.size = [0.25, 0.5, 0.02]; g.pos = [0, 0, 0.355]   # e se apoiavam nessa parede invisível: só o tampo colide
    elif g.name != "table_visual": g.contype = 1; g.conaffinity = 1   # pernas
# Dex3: frictionloss 0,1 da classe finger_motor (unitree-g1-mujoco) com o kp 1,5 do deploy cria zona morta de 0,067 rad e os dedos
# distais seguem 10-26% do alvo (no real 72-102%). O MJCF oficial (unitree_ros) não tem fricção nos dedos; 0,03 é calibração
# pelo degrau em espaço livre (real_vs_sim/dex3_fric.py: 78/92/95% do degrau de 0,1/0,3/0,6 rad em 0,5 s).
s.find_default("finger_motor").joint.frictionloss = 0.03
for j in s.joints:
    if "_hand_" in j.name: j.frictionloss = 0.03   # o spec já tem o valor da classe copiado em cada junta
for g in s.worldbody.geoms:
    if g.name.startswith("marker_"): g.rgba = [0, 0, 0, 0]   # quadrados coloridos da cena do café: não existem na Task 3 real


def corpo(nome, pos, livre=True):
    b = s.worldbody.add_body(name=nome, pos=pos)
    if livre: b.add_freejoint(name=f"{nome}_livre")
    return b


# garrafa PET de 500 ml: Ø65, 205 de altura, rótulo azul, tampa azul; corpo transparente (alpha baixo, sem branco)
# (x, y) da garrafa retroprojetada nos quadros reais do ep. 0 (camera/fit_cena.py); a mediana de fecho da mão pela FK dá (0,389, -0,097)
PET = [0.80, 0.90, 0.97, 0.22]
if OBJETO_LAB:   # em pé onde a mão real fecha (mediana da FK nos 80 eps, fk_pega.py), alça para +x (longe do robô, fora da palma e dos dedos)
    copo = s.body("copo"); copo.name = "garrafa"; copo.pos = [0.348, -0.089, 0.78]; copo.quat = [1, 0, 0, 0]
    copo.first_joint().name = "garrafa_livre"
gar = corpo("garrafa_pet" if OBJETO_LAB else "garrafa", [0.37, -0.14, 0.78])
gar.add_geom(name="garrafa_corpo", type=G.mjGEOM_CYLINDER, size=[0.0325, 0.080, 0], pos=[0, 0, 0.080], rgba=PET, mass=0.40)
gar.add_geom(name="garrafa_rotulo", type=G.mjGEOM_CYLINDER, size=[0.0330, 0.020, 0], pos=[0, 0, 0.095], rgba=[0.12, 0.30, 0.78, 1], mass=0.005, contype=0, conaffinity=0)
gar.add_geom(name="garrafa_ombro", type=G.mjGEOM_ELLIPSOID, size=[0.0325, 0.0325, 0.025], pos=[0, 0, 0.160], rgba=PET, mass=0.02)
gar.add_geom(name="garrafa_gargalo", type=G.mjGEOM_CYLINDER, size=[0.0135, 0.012, 0], pos=[0, 0, 0.190], rgba=PET, mass=0.01)
gar.add_geom(name="garrafa_tampa", type=G.mjGEOM_CYLINDER, size=[0.0150, 0.007, 0], pos=[0, 0, 0.198], rgba=[0.10, 0.35, 0.80, 1], mass=0.005)

# aparência do 1o quadro (ablação visual, results/psi0-task3-sim/ablacao_visual): com o xadrez azul, a luz saturando o tampo e a
# garrafa azul, o modelo pedia yaw alvo de -21° no 1o bloco (real: -6,7°) e girava antes de pegar. Garrafa como a real (PET
# acinzentado, tampa branca, rótulo branco estreito com faixa azul fina), chão escuro liso e luz a 0,6.
for n in ("garrafa_corpo", "garrafa_ombro", "garrafa_gargalo"): s.geom(n).rgba = [0.62, 0.64, 0.66, 0.45]
s.geom("garrafa_tampa").rgba = [0.93, 0.93, 0.95, 1]
s.geom("garrafa_rotulo").rgba = [0.92, 0.92, 0.94, 1]; s.geom("garrafa_rotulo").size = [0.033, 0.016, 0]
gar.add_geom(name="garrafa_faixa", type=G.mjGEOM_CYLINDER, size=[0.0332, 0.004, 0], pos=[0, 0, 0.113], rgba=[0.12, 0.30, 0.78, 1], contype=0, conaffinity=0, mass=0.001)
s.geom("floor").material = ""; s.geom("floor").rgba = [0.12, 0.12, 0.12, 1]
for l in s.lights: l.diffuse = [x * 0.6 for x in l.diffuse]
s.worldbody.add_geom(name="movel_fundo", type=G.mjGEOM_BOX, size=[0.25, 0.8, 0.40], pos=[1.15, -0.2, 0.40], rgba=[0.45, 0.32, 0.20, 1], contype=0, conaffinity=0)   # madeira atrás da mesa (faixa marrom no alto do quadro real)
s.visual.headlight.ambient = [x * 0.6 for x in s.visual.headlight.ambient]; s.visual.headlight.diffuse = [x * 0.6 for x in s.visual.headlight.diffuse]

# mesa de madeira à direita (o robô fica de frente para ela depois de girar). Posições pela FK dos 80 episódios com a base
# integrada como no cliente (vx 0,6 m/s do oficial, caneca_80eps.py): no despejo a boca da garrafa fica na mediana
# (0,348, -1,133) e o pelve em (0,137, -0,938), virado ~-81°. Objetos da mesa: (+0,35, -0,533) para a caneca cair sob a boca.
# Mesa: borda de perto 10 cm à frente do pelve (y = -1,038). No despejo, quadril e tronco na altura do tampo ficam 7 cm (p50)
# a 14 cm (p90) à frente do pelve e a boca 24,5 cm (replay/frente_corpo.py); com a borda no p50 do corpo o robô atravessa o tampo.
# Bandeja e brinquedo vão mais 4 cm em -y para não ficar pendurados na borda nova (a caneca continua sob a boca).
DX, DY = 0.35, -0.533
m2 = corpo("mesa2", [-0.10 + DX, -0.938 - 0.10 - 0.25, 0], livre=False)
m2.add_geom(name="tampo2", type=G.mjGEOM_BOX, size=[0.40, 0.25, 0.01], pos=[0, 0, 0.74], rgba=[0.52, 0.34, 0.18, 1])
for i, (x, y) in enumerate([(-0.37, -0.22), (-0.37, 0.22), (0.37, -0.22), (0.37, 0.22)]):
    m2.add_geom(name=f"perna2_{i}", type=G.mjGEOM_BOX, size=[0.02, 0.02, 0.365], pos=[x, y, 0.365], rgba=[0.15, 0.12, 0.10, 1])
band = corpo("bandeja", [0.00 + DX, -0.62 + DY - 0.04, 0.75], livre=False)
band.add_geom(name="bandeja", type=G.mjGEOM_BOX, size=[0.20, 0.15, 0.0075], pos=[0, 0, 0.0075], rgba=[0.10, 0.20, 0.62, 1])
can = corpo("caneca", [0.00 + DX, -0.60 + DY, 0.765])   # caneca CAN-001 (90 x Ø80), verde clara como a dos episódios
can.add_geom(name="caneca_vis", type=G.mjGEOM_MESH, meshname="m_caneca", rgba=[0.80, 0.86, 0.70, 1], contype=0, conaffinity=0, group=1, density=0)
can.add_geom(name="caneca_alca_vis", type=G.mjGEOM_MESH, meshname="m_caneca_alca", rgba=[0.95, 0.72, 0.62, 1], contype=0, conaffinity=0, group=1, density=0)
can.add_geom(name="caneca_col", type=G.mjGEOM_CYLINDER, size=[0.040, 0.045, 0], pos=[0, 0, 0.045], group=3, rgba=[0, 0, 0, 0], mass=0.31)
for nome, xy, cor in (("copinho_amarelo", [0.14 + DX, -0.72 + DY], [0.95, 0.78, 0.10, 1]), ("copinho_laranja", [0.08 + DX, -0.72 + DY], [0.95, 0.42, 0.10, 1])):
    c = corpo(nome, [*xy, 0.765]); c.add_geom(type=G.mjGEOM_CYLINDER, size=[0.025, 0.028, 0], pos=[0, 0, 0.028], rgba=cor, mass=0.02)
bri = corpo("brinquedo", [0.12 + DX, -0.53 + DY - 0.04, 0.765])
bri.add_geom(type=G.mjGEOM_CYLINDER, size=[0.045, 0.006, 0], pos=[0, 0, 0.006], rgba=[0.15, 0.62, 0.30, 1], mass=0.02)
bri.add_geom(type=G.mjGEOM_ELLIPSOID, size=[0.035, 0.035, 0.022], pos=[0, 0, 0.014], rgba=[0.95, 0.55, 0.30, 1], mass=0.02)

# garrafa como a dos episódios: Kirkland de 500 ml VAZIA e amassada (o despejo deles é encenado, nada sai dela), 25 g no total
# (mesma do replay do ep. 0, results/psi0-task3-sim/gemeo). Atrito PET liso em laminado/bandeja ~0,25. condim 6 liga atrito de
# giro e de rolamento: a garrafa amassada não rola como cilindro perfeito; sem isso o corpo deitado rolava até cair da mesa.
tot = sum(g.mass for g in gar.geoms)
for g in gar.geoms:
    g.mass = g.mass * 0.025 / tot; g.condim = 6; g.friction = [0.25, 0.005, 0.002]
for n in ("tampo", "tampo2", "bandeja"): s.geom(n).friction = [0.25, 0.005, 0.002]
if OBJETO_LAB: s.delete(gar)
for g in s.geoms:   # câmeras D435 de punho são do G1 do laboratório; o robô dos episódios do Ψ0 não tem (aparecem na câmera da cabeça)
    if "_d435_" in g.name: g.rgba = [0, 0, 0, 0]
AMO_FISICO = "--amo-fisico" in sys.argv
for b in s.bodies:   # pernas idealizadas (AMO emulado): sem colisão, senão os pés afundam no chão ao baixar o pelve e o empurrão inclina o tronco
    if any(k in b.name for k in ("hip", "knee", "ankle")):
        for g in b.geoms:
            # --amo-fisico: só as 4 esferas de cada pé colidem, como no g1.xml do AMO público (OpenTeleVision/AMO 34caaf9, linhas 85-88;
            # lá as malhas de colisão das pernas estão comentadas)
            if not (AMO_FISICO and "ankle_roll" in b.name and g.type == G.mjGEOM_SPHERE): g.contype = 0; g.conaffinity = 0
            # --amo-fisico: as malhas das pernas batem nas mesas (bit 2), mas não no chão, entre si nem no resto do robô. Sem isso o joelho
            # direito atravessava a perna da mesa de madeira em 506 quadros de t3_af1 (até 38 mm); a perna real é sólida.
            if AMO_FISICO and g.type == G.mjGEOM_MESH and g.group != 1: g.contype = 2
if AMO_FISICO:
    for n in ("mesa", "mesa2"):
        for g in s.body(n).geoms:
            if g.conaffinity: g.conaffinity |= 2
pel = s.body("pelvis")
if AMO_FISICO:   # pelve livre; as pernas e a cintura são do AMO (amo/pernas_mujoco.py), com física e contato dos pés
    pel.add_freejoint(name="pelvis_livre")
    for lado in ("left", "right"):   # effort do hip roll no URDF oficial (g1_body29_hand14.urdf) e no g1.xml público: 139, não 88
        s.joint(f"{lado}_hip_roll_joint").actfrcrange = [-139, 139]; s.actuator(f"{lado}_hip_roll").ctrlrange = [-139, 139]
else:            # base emulada: o cliente posiciona estas juntas a cada passo (AMO ideal)
    for nome, tipo, eixo in (("base_x", J.mjJNT_SLIDE, [1, 0, 0]), ("base_y", J.mjJNT_SLIDE, [0, 1, 0]), ("base_z", J.mjJNT_SLIDE, [0, 0, 1]), ("base_yaw", J.mjJNT_HINGE, [0, 0, 1])):
        pel.add_joint(name=nome, type=tipo, axis=eixo, damping=1e4)

saida = AQUI / (("cena_task3_amo" if AMO_FISICO else "cena_task3") + ("_copo" if OBJETO_LAB else "") + ".xml")
xml = s.to_xml().replace(str(AQUI / "assets"), "assets")
saida.write_text(xml)
m = mujoco.MjModel.from_xml_path(str(saida)); print("ok", saida.name, "nq", m.nq, "nbody", m.nbody)
