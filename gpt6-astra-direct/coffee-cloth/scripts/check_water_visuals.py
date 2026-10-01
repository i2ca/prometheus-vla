"""Smoke check standalone dos overlays de água (water_visuals.py).

Não roda o trial nem toca em física: carrega a cena, posa os corpos com
mj_forward e verifica geometria e renderização dos overlays em casos
sintéticos. Modelo requisitado: claude/claude-fable-5.

Uso:
  MUJOCO_GL=egl ../g1-cup-grasp/.venv/bin/python check_water_visuals.py \
      --scene ../scene/coffee-006.xml [--out ../results/claude-visuals-001]
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
import types
from pathlib import Path
import numpy as np
import mujoco
from liquid import WaterTransfer
from water_visuals import add_water_visuals, FILL_RADIUS

ap = argparse.ArgumentParser()
ap.add_argument('--scene', type=Path, default=Path(__file__).resolve().parents[1] / 'scene/coffee-006.xml')
ap.add_argument('--out', type=Path, default=None)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(args.scene))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
sim = types.SimpleNamespace(m=m, d=d)
renderer = mujoco.Renderer(m, 360, 640)

filter_z = d.xpos[m.body('filter_ring').id][2]
receiver_pos = d.xpos[m.body('receiver').id].copy()
checks = []
images = {}


def check(name, ok, detail=''):
    checks.append({'name': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS' if ok else 'FAIL'), name, detail)


def frame(water, latest, camera='coffee_closeup'):
    renderer.update_scene(d, camera=camera)
    before = renderer.scene.ngeom
    add_water_visuals(renderer, sim, water, latest)
    added = [renderer.scene.geoms[i] for i in range(before, renderer.scene.ngeom)]
    img = renderer.render().copy()
    return added, img


# Caso 1: sem água e sem fluxo, nada é desenhado
water = WaterTransfer()
lip_capture = [receiver_pos[0] + 0.01, receiver_pos[1], filter_z + 0.12]
added, img = frame(water, {'flow_ml_s': 0., 'lip': lip_capture, 'capture': False})
check('empty_no_overlay', len(added) == 0, f'{len(added)} geoms adicionados')
check('empty_render_finite', np.all(np.isfinite(img)))
added, _ = frame(water, {})
check('empty_latest_dict', len(added) == 0)

# Caso 2: receiver cheio + fluxo capturado -> preenchimento e jato
water.receiver_ml = 350.
latest = {'flow_ml_s': 15., 'lip': lip_capture, 'capture': True}
added, img = frame(water, latest)
check('full_flow_two_geoms', len(added) == 2, f'{len(added)} geoms')
if len(added) == 2:
    fill, stream = added
    exp_h = 350e-6 / (np.pi * water.radius ** 2)
    exp_center = receiver_pos + [0., 0., water.bottom + exp_h / 2]
    check('fill_position', np.allclose(fill.pos, exp_center, atol=1e-6),
          f'pos {np.round(fill.pos, 4).tolist()} esperado {np.round(exp_center, 4).tolist()}')
    check('fill_inside_cup', fill.size[0] <= FILL_RADIUS + 1e-9
          and water.bottom + exp_h <= water.top + 1e-9,
          f'raio {fill.size[0]:.4f} topo local {water.bottom + exp_h:.4f}')
    mid = (np.asarray(lip_capture) + [lip_capture[0], lip_capture[1], filter_z]) / 2
    check('stream_midpoint_capture', np.allclose(stream.pos, mid, atol=1e-6),
          f'pos {np.round(stream.pos, 4).tolist()} esperado {np.round(mid, 4).tolist()}')
    check('stream_finite', np.all(np.isfinite(stream.pos)) and np.all(np.isfinite(stream.mat)))
check('full_render_finite', np.all(np.isfinite(img)))
images['full_capture.png'] = img

# Caso 3: jato perdido sobre a mesa termina no tampo (z=0.75)
lip_miss = [0.50, -0.30, 1.00]
added, img = frame(water, {'flow_ml_s': 5., 'lip': lip_miss, 'capture': False})
streams = [g for g in added if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE]
check('miss_on_table', len(streams) == 1
      and np.isclose(streams[0].pos[2], (1.00 + 0.75) / 2, atol=1e-6),
      f'z médio {streams[0].pos[2]:.4f}' if streams else 'sem jato')
images['miss_table.png'] = img

# Caso 4: jato perdido fora da mesa termina no chão (z=0)
added, _ = frame(water, {'flow_ml_s': 5., 'lip': [1.50, 0., 1.00], 'capture': False})
streams = [g for g in added if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE]
check('miss_off_table_floor', len(streams) == 1
      and np.isclose(streams[0].pos[2], 0.5, atol=1e-6),
      f'z médio {streams[0].pos[2]:.4f}' if streams else 'sem jato')

# Caso 5: dois frames seguidos, geoms não vazam (update_scene zera a cena)
n1 = None
for _ in range(2):
    frame(water, latest)
    n1 = n1 or renderer.scene.ngeom
check('no_leak_across_frames', renderer.scene.ngeom == n1,
      f'{n1} vs {renderer.scene.ngeom}')

# Caso 6: cena cheia é respeitada (guard de maxgeom, sem estourar)
renderer.update_scene(d, camera='coffee_closeup')
scene = renderer.scene
saved = scene.ngeom
scene.ngeom = scene.maxgeom
add_water_visuals(renderer, sim, water, latest)
check('maxgeom_guard', scene.ngeom == scene.maxgeom)
scene.ngeom = saved

failed = [c['name'] for c in checks if not c['pass']]
report = {
    'model_requested': 'claude/claude-fable-5',
    'model_evidence': 'sem evidência verificável de backend em runtime; identidade auto-reportada pelo harness como Claude Fable 5, orquestrador gpt-6-astra',
    'mujoco_version': mujoco.__version__,
    'mujoco_gl': os.environ.get('MUJOCO_GL'),
    'scene': str(args.scene.resolve()),
    'scope': 'overlays visuais do surrogate; sem física, sem CFD, sem líquido no copo fonte inclinado',
    'checks': checks,
    'all_pass': not failed,
}
print(json.dumps({'all_pass': report['all_pass'], 'failed': failed}))

if args.out:
    import cv2
    args.out.mkdir(parents=True, exist_ok=False)
    for name, img in images.items():
        cv2.imwrite(str(args.out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    report['images'] = sorted(images)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
renderer.close()
raise SystemExit(1 if failed else 0)
