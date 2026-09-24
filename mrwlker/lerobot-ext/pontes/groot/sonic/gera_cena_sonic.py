#!/usr/bin/env python
"""Monta a cena do café para o simulador MuJoCo do SONIC (GR00T-WholeBodyControl).

    python pontes/groot/sonic/gera_cena_sonic.py
    SONIC_CENA=<caminho impresso> python gear_sonic/scripts/run_sim_loop.py ...

O simulador do SONIC carrega um XML só (`ROBOT_SCENE`), e o dele tem apenas o G1
no chão. Aqui vai a MESMA cena de duas mesas do `cena_cafe/cena_cafe.py` (a do
simulador da maçã), com as mesmas medidas e posições:

    MESA DO COADOR (y=1,5)          MESA PRINCIPAL (tampo 1,58 x 0,74 m, topo a 0,684 m)
    ┌──────────────┐    ┌─────────────────────────────────────────────┐  +x
    │    coador    │    │  chaleira                        pote+tampa │
    │              │    │               xícara   garrafa    scoop     │
    └──────────────┘    └─────────────────────────────────────────────┘
            +y (esquerda)                robô                -y (direita)

mais a GARRAFA PET — o objeto que o `SII-Linzy/groot-g1-sonic-grab-bottle`
aprendeu a pegar, à frente e à direita, como nos vídeos de avaliação dele. Os
utensílios são os de `cena_cafe/objetos/` (rode `cena_cafe/gera_objetos.py` antes).
As mesas são caixas com as medidas da `lab_table` do robocasa, não a malha dela.

Os `model.xml` dos objetos não podem ser incluídos juntos: todos chamam o corpo
de `object` e os sites de `bottom_site`. Então cada um é COPIADO para dentro da
cena com prefixo no nome, caminhos de malha absolutos e as classes de default
(`collision`) desdobradas nos próprios geoms.

O XML sai na pasta do modelo do G1 deles, porque ele inclui o
`g1_29dof_with_hand.xml` por caminho relativo e herda o `meshdir` dele.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

OBJETOS = Path(__file__).resolve().parents[1] / "cena_cafe" / "objetos"
G1 = Path.home() / "DEV/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1"
SAIDA = G1 / "scene_cafe_sonic.xml"

# Medidas da `lab_table` do robocasa: tampo 1,17 x 0,74 m, 2,6 cm de espessura,
# topo a 0,684 m. A principal é esticada 1,35x na largura, como na cena de lá.
MESA_TOPO = 0.684
MESA_ESPESSURA = 0.026
MESAS = {
    # nome: (centro x, centro y, meia-profundidade x, meia-largura y)
    "mesa": (0.5, 0.0, 0.37, 0.585 * 1.35),
    "mesa_coador": (0.5, 1.5, 0.37, 0.585),
}

# nome: (x, y, livre?)  — o robô está na origem, olhando para +x; y>0 é a esquerda.
# As mesmas posições do `cena_cafe.py` (centro das faixas de sorteio de lá).
LAYOUT = {
    "xicara":   (0.38, 0.06, True),
    "coador":   (0.30, 1.50, False),
    "chaleira": (0.66, 0.45, False),
    "pote":     (0.66, -0.42, False),
    "scoop":    (0.40, -0.30, True),
}
GARRAFA_XY = (0.40, -0.12)


def garrafa() -> str:
    """Garrafa PET de 500 ml: corpo azul translúcido, rótulo, tampa preta."""
    z = MESA_TOPO + 0.10
    x, y = GARRAFA_XY
    return f"""
    <body name="garrafa" pos="{x} {y} {z:.3f}">
      <freejoint name="garrafa_livre"/>
      <geom name="garrafa_corpo" type="cylinder" size="0.032 0.09" rgba="0.35 0.65 0.95 0.55"
            mass="0.3" friction="1.2 0.05 0.002" condim="4"/>
      <geom name="garrafa_rotulo" type="cylinder" size="0.0325 0.03" pos="0 0 -0.02"
            rgba="0.2 0.5 0.9 1" contype="0" conaffinity="0" mass="0"/>
      <geom name="garrafa_gargalo" type="cylinder" size="0.014 0.012" pos="0 0 0.10"
            rgba="0.35 0.65 0.95 0.55" mass="0.005"/>
      <geom name="garrafa_tampa" type="cylinder" size="0.016 0.01" pos="0 0 0.12"
            rgba="0.05 0.05 0.05 1" mass="0.005"/>
    </body>"""


def mesa(nome, cx, cy, mx, my) -> str:
    """Tampo de madeira e duas colunas de metal, no desenho da `lab_table`."""
    mz = MESA_ESPESSURA / 2
    h = MESA_TOPO - MESA_ESPESSURA
    pernas = "".join(
        f'\n      <geom type="box" size="0.04 0.04 {h / 2:.3f}" '
        f'pos="0 {sy * (my - 0.1):.3f} {-h / 2 - mz:.3f}" material="metal_mesa"/>'
        f'\n      <geom type="box" size="{mx - 0.05:.3f} 0.03 0.012" '
        f'pos="0 {sy * (my - 0.1):.3f} {-MESA_TOPO + mz + 0.012:.3f}" material="metal_mesa"/>'
        for sy in (-1, 1))
    return f"""
    <body name="{nome}" pos="{cx} {cy} {MESA_TOPO - mz:.4f}">
      <geom name="{nome}_tampo" type="box" size="{mx} {my} {mz}" material="madeira_mesa"
            friction="1 0.05 0.002"/>{pernas}
    </body>"""


def copia_objeto(nome: str, x: float, y: float, livre: bool, assets: ET.Element) -> str:
    pasta = OBJETOS / nome
    raiz = ET.parse(pasta / "model.xml").getroot()
    # O default `collision` do objeto, para desdobrar nos geoms.
    padrao = {}
    for d in raiz.iter("default"):
        if d.get("class") == "collision":
            g = d.find("geom")
            padrao = dict(g.attrib) if g is not None else {}
    for a in raiz.find("asset"):
        a = copy.deepcopy(a)
        a.set("name", f"{nome}__{a.get('name')}")
        if a.get("file"):
            a.set("file", str((pasta / a.get("file")).resolve()))
        if a.get("texture"):
            a.set("texture", f"{nome}__{a.get('texture')}")
        assets.append(a)
    corpo_ext = raiz.find("worldbody/body")
    h = abs(float(corpo_ext.find("site[@name='bottom_site']").get("pos").split()[2]))
    corpo = ET.Element("body", name=nome, pos=f"{x} {y} {MESA_TOPO + h + 0.002:.4f}")
    if livre:
        ET.SubElement(corpo, "freejoint", name=f"{nome}_livre")
    for g in corpo_ext.find("body").findall("geom"):
        g = copy.deepcopy(g)
        if g.get("class") == "collision":
            del g.attrib["class"]
            for k, v in padrao.items():
                g.attrib.setdefault(k, v)
        for chave in ("mesh", "material"):
            if g.get(chave):
                g.set(chave, f"{nome}__{g.get(chave)}")
        if not livre:
            g.attrib.pop("mass", None)
        corpo.append(g)
    return ET.tostring(corpo, encoding="unicode")


def main():
    assets = ET.Element("asset")
    corpos = [mesa(n, *v) for n, v in MESAS.items()] + [garrafa()]
    for nome, (x, y, livre) in LAYOUT.items():
        corpos.append(copia_objeto(nome, x, y, livre, assets))
    # A tampa em cima do pote (fixa, como na cena do robocasa).
    corpos.append(copia_objeto("tampa", LAYOUT["pote"][0], LAYOUT["pote"][1], False, assets))
    # sobe a tampa pela altura do pote (10 cm)
    ultimo = ET.fromstring(corpos[-1])
    x, y, z = map(float, ultimo.get("pos").split())
    ultimo.set("pos", f"{x} {y} {z + 0.10:.4f}")
    corpos[-1] = ET.tostring(ultimo, encoding="unicode")

    extra = "".join(ET.tostring(a, encoding="unicode") for a in assets)
    xml = f"""<mujoco model="g1_43dof cafe (gerado por prometheus-vla/lerobot-ext/pontes/groot/sonic/gera_cena_sonic.py)">
  <include file="g1_29dof_with_hand.xml"/>
  <statistic center="0.3 0 0.6" extent="2.0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="madeira_mesa" rgba="0.66 0.50 0.34 1" specular="0.2"/>
    <material name="metal_mesa" rgba="0.75 0.75 0.78 1" specular="0.6" shininess="0.6"/>
    {extra}
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <light pos="0.6 0 2.0" dir="0 0 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <site name="com_marker" pos="0.1 0 0" size="0.05" rgba="1 0 0 1" type="sphere"/>{"".join(corpos)}
  </worldbody>
  <default>
    <geom friction="1.0"/>
  </default>
</mujoco>
"""
    SAIDA.write_text(xml)
    print(SAIDA)


if __name__ == "__main__":
    main()
