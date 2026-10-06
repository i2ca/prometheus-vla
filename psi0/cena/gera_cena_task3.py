"""Gera cena_task3.xml a partir de cena_cafe_medidas.xml: o arranjo da Task 3 do Ψ0 ("Pick bottle, turn and pour into cup"),
como nos episódios gravados deles. Mesa branca à frente com uma garrafa d'água de 500 ml à direita; mesa de madeira à
direita do robô (o modelo gira ~83° para ela) com bandeja azul, caneca verde, dois copinhos e um brinquedo.
O pelve ganha juntas de base (x, y, z, yaw) para o cliente emular o controlador de pernas (AMO) a partir das ações do modelo.
Uso: python gera_cena_task3.py
"""
from pathlib import Path
import mujoco

AQUI = Path(__file__).resolve().parent
J = mujoco.mjtJoint; G = mujoco.mjtGeom
s = mujoco.MjSpec.from_file(str(AQUI / "cena_cafe_medidas.xml"))
for n in ("chaleira", "base_eletrica", "coador", "pote", "tampa", "scoop", "copo"):
    s.delete(s.body(n))

mesa = s.body("mesa"); mesa.pos[0] += 0.15   # mesa branca a 35 cm do pelve, como nos episódios
for g in mesa.geoms:
    if g.name == "table_visual": g.rgba = [0.90, 0.90, 0.88, 1]
    if g.name == "tampo": g.rgba = [0, 0, 0, 0]   # caixa de colisão do chão ao tampo: não aparece


def corpo(nome, pos, livre=True):
    b = s.worldbody.add_body(name=nome, pos=pos)
    if livre: b.add_freejoint(name=f"{nome}_livre")
    return b


# garrafa PET de 500 ml: Ø65, 205 de altura, rótulo azul, tampa branca
gar = corpo("garrafa", [0.47, -0.13, 0.75])
gar.add_geom(name="garrafa_corpo", type=G.mjGEOM_CYLINDER, size=[0.0325, 0.080, 0], pos=[0, 0, 0.080], rgba=[0.78, 0.86, 0.95, 0.55], mass=0.40)
gar.add_geom(name="garrafa_rotulo", type=G.mjGEOM_CYLINDER, size=[0.0330, 0.020, 0], pos=[0, 0, 0.095], rgba=[0.12, 0.30, 0.78, 1], mass=0.005, contype=0, conaffinity=0)
gar.add_geom(name="garrafa_ombro", type=G.mjGEOM_ELLIPSOID, size=[0.0325, 0.0325, 0.025], pos=[0, 0, 0.160], rgba=[0.78, 0.86, 0.95, 0.55], mass=0.02)
gar.add_geom(name="garrafa_gargalo", type=G.mjGEOM_CYLINDER, size=[0.0135, 0.012, 0], pos=[0, 0, 0.190], rgba=[0.78, 0.86, 0.95, 0.6], mass=0.01)
gar.add_geom(name="garrafa_tampa", type=G.mjGEOM_CYLINDER, size=[0.0150, 0.007, 0], pos=[0, 0, 0.198], rgba=[0.95, 0.95, 0.95, 1], mass=0.005)

# mesa de madeira à direita (o robô fica de frente para ela depois de girar)
m2 = corpo("mesa2", [-0.10, -0.62, 0], livre=False)
m2.add_geom(name="tampo2", type=G.mjGEOM_BOX, size=[0.40, 0.25, 0.01], pos=[0, 0, 0.74], rgba=[0.52, 0.34, 0.18, 1])
for i, (x, y) in enumerate([(-0.37, -0.22), (-0.37, 0.22), (0.37, -0.22), (0.37, 0.22)]):
    m2.add_geom(name=f"perna2_{i}", type=G.mjGEOM_BOX, size=[0.02, 0.02, 0.365], pos=[x, y, 0.365], rgba=[0.15, 0.12, 0.10, 1])
band = corpo("bandeja", [0.00, -0.62, 0.75], livre=False)
band.add_geom(name="bandeja", type=G.mjGEOM_BOX, size=[0.20, 0.15, 0.0075], pos=[0, 0, 0.0075], rgba=[0.10, 0.20, 0.62, 1])
can = corpo("caneca", [0.00, -0.60, 0.765])   # caneca CAN-001 (90 x Ø80), verde clara como a dos episódios
can.add_geom(name="caneca_vis", type=G.mjGEOM_MESH, meshname="m_caneca", rgba=[0.80, 0.86, 0.70, 1], contype=0, conaffinity=0, group=1, density=0)
can.add_geom(name="caneca_alca_vis", type=G.mjGEOM_MESH, meshname="m_caneca_alca", rgba=[0.95, 0.72, 0.62, 1], contype=0, conaffinity=0, group=1, density=0)
can.add_geom(name="caneca_col", type=G.mjGEOM_CYLINDER, size=[0.040, 0.045, 0], pos=[0, 0, 0.045], group=3, rgba=[0, 0, 0, 0], mass=0.31)
for nome, xy, cor in (("copinho_amarelo", [0.14, -0.72], [0.95, 0.78, 0.10, 1]), ("copinho_laranja", [0.08, -0.72], [0.95, 0.42, 0.10, 1])):
    c = corpo(nome, [*xy, 0.765]); c.add_geom(type=G.mjGEOM_CYLINDER, size=[0.025, 0.028, 0], pos=[0, 0, 0.028], rgba=cor, mass=0.02)
bri = corpo("brinquedo", [0.12, -0.53, 0.765])
bri.add_geom(type=G.mjGEOM_CYLINDER, size=[0.045, 0.006, 0], pos=[0, 0, 0.006], rgba=[0.15, 0.62, 0.30, 1], mass=0.02)
bri.add_geom(type=G.mjGEOM_ELLIPSOID, size=[0.035, 0.035, 0.022], pos=[0, 0, 0.014], rgba=[0.95, 0.55, 0.30, 1], mass=0.02)

pel = s.body("pelvis")   # base emulada: o cliente posiciona estas juntas a cada passo (AMO ideal)
for nome, tipo, eixo in (("base_x", J.mjJNT_SLIDE, [1, 0, 0]), ("base_y", J.mjJNT_SLIDE, [0, 1, 0]), ("base_z", J.mjJNT_SLIDE, [0, 0, 1]), ("base_yaw", J.mjJNT_HINGE, [0, 0, 1])):
    pel.add_joint(name=nome, type=tipo, axis=eixo, damping=1e4)

xml = s.to_xml().replace(str(AQUI / "assets"), "assets")
(AQUI / "cena_task3.xml").write_text(xml)
m = mujoco.MjModel.from_xml_path(str(AQUI / "cena_task3.xml")); print("ok nq", m.nq, "nbody", m.nbody)
