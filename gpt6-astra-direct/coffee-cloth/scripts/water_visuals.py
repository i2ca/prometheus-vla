"""Overlays visuais pro surrogate WaterTransfer. Visualização do estado do
surrogate, NÃO hidrodinâmica medida nem partículas de fluido.

Modelo requisitado: claude/claude-fable-5 (worker; orquestrador gpt-6-astra).

O que é desenhado (geoms transientes na MjvScene, sem tocar no modelo físico):
- Água no receiver: cilindro semitransparente com altura derivada de
  water.receiver_ml pela área interna do copo (mesmas dimensões do cilindro
  do surrogate: radius/bottom/top de WaterTransfer). O nível acompanha o eixo
  do receiver; a superfície livre NÃO é corrigida pra gravidade se o receiver
  tombar.
- Jato: cápsula vertical a partir de latest['lip'] quando
  latest['flow_ml_s'] > 0.01, terminando no plano da boca do filtro se
  latest['capture'], senão no tampo da mesa (ou no chão, se o pé do jato cair
  fora da mesa). A largura cresce com a vazão só como indicação visual.

O que NÃO é desenhado, de propósito:
- Líquido dentro do copo fonte inclinado: recorte correto de um volume contra
  um cilindro tombado exige primitivas cortadas por plano, que a MjvScene não
  tem; uma aproximação por elipsoide vazaria pelas paredes. Omitido.
- Água retida no filtro de pano (filter_ml): fora do escopo pedido.

Uso: chamar add_water_visuals(renderer, sim, water, latest) DEPOIS de
renderer.update_scene(...) e ANTES de renderer.render(). update_scene zera a
cena a cada frame, então os geoms não vazam entre frames.
"""
import numpy as np
import mujoco

FILL_RGBA = np.array([0.25, 0.50, 0.85, 0.55], dtype=np.float32)
STREAM_RGBA = np.array([0.45, 0.65, 0.95, 0.65], dtype=np.float32)
FILL_RADIUS = 0.0365      # face interna da parede fica em 0.037; folga anti z-fighting
MIN_VISIBLE_ML = 0.05
MIN_FLOW_ML_S = 0.01
STREAM_RADIUS_MIN = 0.0012
STREAM_RADIUS_MAX = 0.0040


def _add_cylinder(scene, pos, mat, radius, half_height, rgba):
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([radius, radius, half_height]),
                        np.asarray(pos, dtype=np.float64),
                        np.asarray(mat, dtype=np.float64).reshape(9),
                        rgba)
    scene.ngeom += 1
    return True


def _add_capsule(scene, p_from, p_to, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.eye(3).reshape(9), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.asarray(p_from, dtype=np.float64),
                         np.asarray(p_to, dtype=np.float64))
    scene.ngeom += 1
    return True


def add_water_visuals(renderer, sim, water, latest):
    """Anexa os overlays de água na cena já atualizada do renderer.

    renderer: mujoco.Renderer após update_scene() e antes de render().
    sim: objeto com .m (MjModel) e .d (MjData).
    water: instância de WaterTransfer (fonte da verdade dos volumes).
    latest: último dict retornado por water.step(); {} antes do primeiro step.
    """
    scene = renderer.scene
    m, d = sim.m, sim.d
    latest = latest or {}

    # Água no receiver
    if water.receiver_ml > MIN_VISIBLE_ML:
        inner_area = np.pi * water.radius ** 2
        height = min(water.receiver_ml * 1e-6 / inner_area, water.top - water.bottom)
        body = m.body('receiver').id
        rot = d.xmat[body].reshape(3, 3)
        center = d.xpos[body] + rot @ np.array([0., 0., water.bottom + height / 2])
        if np.all(np.isfinite(center)):
            _add_cylinder(scene, center, rot, FILL_RADIUS, height / 2, FILL_RGBA)

    # Jato vertical do surrogate
    flow = float(latest.get('flow_ml_s', 0.) or 0.)
    lip = latest.get('lip')
    if flow > MIN_FLOW_ML_S and lip is not None:
        lip = np.asarray(lip, dtype=np.float64)
        if np.all(np.isfinite(lip)):
            if latest.get('capture'):
                end_z = d.xpos[m.body('filter_ring').id][2]
            else:
                tampo = m.geom('tampo').id
                top = d.geom_xpos[tampo]
                half = m.geom_size[tampo]
                on_table = (abs(lip[0] - top[0]) <= half[0]
                            and abs(lip[1] - top[1]) <= half[1])
                end_z = top[2] + half[2] if on_table else 0.
            if lip[2] > end_z + 1e-3:
                width = STREAM_RADIUS_MIN + (STREAM_RADIUS_MAX - STREAM_RADIUS_MIN) \
                    * np.sqrt(min(flow / water.max_flow_ml_s, 1.))
                _add_capsule(scene, lip, [lip[0], lip[1], end_z], width, STREAM_RGBA)
