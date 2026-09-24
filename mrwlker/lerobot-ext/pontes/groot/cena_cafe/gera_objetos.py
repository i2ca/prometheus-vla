#!/usr/bin/env python
"""Converte os utensílios do café para objetos do robocasa (o simulador de G1 da NVIDIA).

    ~/miniforge3/envs/prometheus-vla/bin/python pontes/groot/cena_cafe/gera_objetos.py

Cada objeto vira `objetos/<nome>/` com o formato que o `MJCFObject` do robocasa lê
(o mesmo da `apple_0` deles):
    model.xml               malha visual + colisão + os três sites que o robocasa usa
    visual/modelo.obj       com UV, e a textura ao lado
    colisao/*.obj           cascos convexos (o MuJoCo só colide malha convexa)

O QUE CADA OBJETO PRECISA SABER
-------------------------------
`rot`     euler XYZ intrínseco (o do MuJoCo) que põe o objeto EM PÉ com Z para cima. Os `.glb` de
          ~/Downloads vêm em Y para cima (padrão glTF); os que já estavam no
          `unitree-g1-mujoco` herdam a rotação que a nossa cena usava.
`altura`  altura REAL em metros. Os `.glb` saem de gerador imagem-3D normalizados
          para ~1 m, então a escala é chute informado; ajuste aqui e rode de novo.
`cascos`  como fatiar a colisão ao longo de Z. Um casco só serve para objeto
          "cheio" (pote, chaleira). O coador precisa de dois — base e funil —
          senão o casco único vira um cone maciço e a xícara não entra embaixo.

A origem fica no CENTRO da caixa envolvente, como na `apple_0`: o robocasa pousa o
objeto na mesa pelo `bottom_site`, e ele fica em -altura/2.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import trimesh

AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "objetos"
DOWNLOADS = Path.home() / "Downloads"
NOSSO = Path(__file__).resolve().parents[4] / "unitree-g1-mujoco" / "assets"

Y_PARA_Z = (np.pi / 2, 0.0, 0.0)

OBJETOS = {
    # nome: fonte, rotação, altura real (m), fatias de colisão em fração da altura
    # A xícara e o coador são os da NOSSA cena (mesmo tamanho e rotação de lá).
    "xicara":   dict(fonte=NOSSO / "copo_texturizado.obj", rot=(np.pi / 2, -np.pi / 2, 0.0),
                     altura=None, escala=1.0, cascos=[(0.0, 1.0)], massa=0.15),
    "coador":   dict(fonte=NOSSO / "coador.obj", rot=(0.0, 0.0, 0.0),
                     altura=None, escala=0.195, cascos=[(0.0, 0.06), (0.45, 1.0)], massa=0.4),
    "chaleira": dict(fonte=DOWNLOADS / "chaleira.glb", rot=Y_PARA_Z,
                     altura=0.22, cascos=[(0.0, 1.0)], massa=0.8),
    "pote":     dict(fonte=DOWNLOADS / "pote.glb", rot=Y_PARA_Z,
                     altura=0.10, cascos=[(0.0, 1.0)], massa=0.3),
    # Tampa do pote: a altura sai da boca do pote (~16 cm de diâmetro), porque a
    # malha tem 0,46 de altura para 1,0 de diâmetro.
    "tampa":    dict(fonte=DOWNLOADS / "tampa.glb", rot=Y_PARA_Z,
                     altura=0.074, cascos=[(0.0, 1.0)], massa=0.08),
    "scoop":    dict(fonte=DOWNLOADS / "scoop.glb", rot=Y_PARA_Z,
                     altura=0.02, cascos=[(0.0, 1.0)], massa=0.03),
}


def carrega(fonte: Path) -> trimesh.Trimesh:
    m = trimesh.load(fonte, force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise SystemExit(f"{fonte}: não virou uma malha única")
    return m


def converte(nome: str, cfg: dict) -> dict:
    m = carrega(cfg["fonte"])
    # "rxyz": euler INTRÍNSECO, a convenção do `euler=` do MuJoCo (a rotação da
    # xícara veio de lá). Com "sxyz" ela saía deitada, com a alça para cima.
    R = trimesh.transformations.euler_matrix(*cfg["rot"], axes="rxyz")
    m.apply_transform(R)
    if cfg.get("altura"):
        esc = cfg["altura"] / float(m.extents[2])
    else:
        esc = cfg["escala"]
    m.apply_scale(esc)
    m.apply_translation(-m.bounds.mean(axis=0))          # origem no centro da caixa

    pasta = SAIDA / nome
    if pasta.exists():
        shutil.rmtree(pasta)
    (pasta / "visual").mkdir(parents=True)
    (pasta / "colisao").mkdir()

    # Visual, com a textura. O trimesh escreve .obj + .mtl + a imagem.
    m.export(pasta / "visual" / "modelo.obj", include_texture=True)
    texturas = sorted((pasta / "visual").glob("*.png")) + sorted((pasta / "visual").glob("*.jpg"))

    # Colisão: um casco convexo por fatia de altura.
    z0, h = m.bounds[0, 2], float(m.extents[2])
    cascos = []
    for k, (a, b) in enumerate(cfg["cascos"]):
        v = m.vertices[(m.vertices[:, 2] >= z0 + a * h - 1e-9) & (m.vertices[:, 2] <= z0 + b * h + 1e-9)]
        casco = trimesh.convex.convex_hull(v)
        f = pasta / "colisao" / f"casco_{k}.obj"
        casco.export(f)
        cascos.append(f.name)

    raio = float(np.max(np.linalg.norm(m.vertices[:, :2], axis=1)))
    xml = monta_xml(nome, texturas[0].name if texturas else None, cascos, h, raio, cfg["massa"])
    (pasta / "model.xml").write_text(xml)
    return {"tamanho_m": m.extents.round(3).tolist(), "escala": round(esc, 4),
            "cascos": len(cascos), "textura": bool(texturas)}


def monta_xml(nome, textura, cascos, h, raio, massa) -> str:
    tex = (f'    <texture type="2d" name="tex_{nome}" file="visual/{textura}"/>\n'
           f'    <material name="mat_{nome}" texture="tex_{nome}" specular="0.3" shininess="0.3"/>\n'
           if textura else
           f'    <material name="mat_{nome}" rgba="1 1 1 1" specular="0.3" shininess="0.3"/>\n')
    malhas = "".join(f'    <mesh file="colisao/{c}" name="{nome}_col_{i}"/>\n' for i, c in enumerate(cascos))
    geoms = "".join(f'        <geom class="collision" type="mesh" mesh="{nome}_col_{i}"/>\n'
                    for i in range(len(cascos)))
    # Massa pedida, dividida igualmente entre os cascos: densidade sobre o volume
    # de um casco convexo daria massa demais para objeto oco (xícara, pote).
    return f"""<mujoco model="{nome}">
  <asset>
    <mesh file="visual/modelo.obj" name="{nome}_vis"/>
{tex}{malhas}  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom type="mesh" mesh="{nome}_vis" conaffinity="0" contype="0" group="1" material="mat_{nome}"/>
{geoms}      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {-h / 2:.5f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {h / 2:.5f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{raio:.5f} 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
  <default>
    <default class="visual">
      <geom group="1" density="100"/>
    </default>
    <default class="collision">
      <geom group="0" mass="{massa / max(1, len(cascos)):.4f}" solimp="0.95 0.98 0.001" solref="0.004 1" rgba="1 1 1 0.25" friction="1.2 0.05 0.002"/>
    </default>
  </default>
</mujoco>
"""


if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    for nome, cfg in OBJETOS.items():
        print(f"{nome:9s}", converte(nome, cfg))
