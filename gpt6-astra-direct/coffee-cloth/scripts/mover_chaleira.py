"""Cria uma fonte com a chaleira deslocada, para testar se o pipeline generaliza.

Nada aqui muda o modelo: so' reescreve o estado de integracao salvo, mexendo no
qpos da junta livre da chaleira e deixando a fisica reassentar.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, mujoco

origem, destino = Path(sys.argv[1]), Path(sys.argv[2])
dx, dy, dyaw = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

r = json.loads((origem / 'report.json').read_text())
m = mujoco.MjModel.from_xml_path(r['scene'])
d = mujoco.MjData(m)
ck = np.load(origem / 'continuation.npz')
for n in ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'bvh_aabb']:
    getattr(m, n)[:] = ck[n]
mujoco.mj_setConst(m, mujoco.MjData(m))
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)

adr = m.jnt_qposadr[m.joint('chaleira_livre').id]
antes = d.qpos[adr:adr+3].copy()
d.qpos[adr] += dx
d.qpos[adr+1] += dy
if dyaw:
    q = d.qpos[adr+3:adr+7].copy()
    giro = np.array([np.cos(np.deg2rad(dyaw)/2), 0, 0, np.sin(np.deg2rad(dyaw)/2)])
    res = np.zeros(4); mujoco.mju_mulQuat(res, giro, q); d.qpos[adr+3:adr+7] = res
# base eletrica acompanha a chaleira, senao ela flutua fora do apoio
badr = m.jnt_qposadr[m.joint('base_livre').id] if any(
    m.joint(j).name == 'base_livre' for j in range(m.njnt)) else None
if badr is not None:
    d.qpos[badr] += dx; d.qpos[badr+1] += dy
d.qvel[:] = 0
for _ in range(int(2. / m.opt.timestep)):   # deixa assentar 2 s
    mujoco.mj_step(m, d)
depois = d.qpos[adr:adr+3].copy()

destino.mkdir(parents=True, exist_ok=False)
for f in origem.iterdir():
    if f.is_file() and f.name != 'continuation.npz':
        shutil.copy2(f, destino / f.name)
integracao = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
mujoco.mj_getState(m, d, integracao, mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(destino / 'continuation.npz',
                    **{k: ck[k] for k in ck.files if k != 'integration'},
                    integration=integracao)
print(json.dumps({'origem': str(origem), 'destino': str(destino),
                  'pedido_mm': [dx*1000, dy*1000], 'giro_deg': dyaw,
                  'chaleira_antes': antes.round(4).tolist(),
                  'chaleira_depois': depois.round(4).tolist(),
                  'deslocamento_real_mm': ((depois-antes)*1000).round(1).tolist()}, indent=2))
