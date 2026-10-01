"""Sintetiza pegas envolvendo a barra vertical da alca da chaleira.

A pega validada ate aqui e' uma pinca por cima: polegar enganchado no topo da
alca e punho 27 cm acima da base da chaleira. No despejo isso poe a mao a
1,24 m, acima do ombro do G1 (1,12 m). A pega humana envolve a barra vertical
na altura do meio. O otimizador de pinca (plan_kettle_grasp_v3) nao encontra
essa pega: 0 de 40 com 6,5 mm e com 4 mm de folga termica.

Estrategia: separar a mao do braco. O robo fica parado numa pose de braco
qualquer e a CHALEIRA e' movida para cada pose relativa candidata; os dedos
fecham junta a junta ate encostar na alca, como uma mao real. Tudo o que
depende so' da relacao mao-chaleira (contato, oposicao, folga ao metal
quente, colisao da mao aberta) sai daqui. O braco e' resolvido depois pelo
plan_kettle_grasp_v3 --fixed-hand usando a melhor pega como referencia.

Cinematica da Dex3 esquerda (referencial do punho): indicador e medio saem em
+x, separados 57 mm em z, e flexionam em torno de z (angulo negativo fecha);
o polegar sai para -y e fecha com angulo positivo. Para envolver uma barra
vertical o eixo z do punho fica paralelo a barra.
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--arm-ref', type=Path, default=Path('results-claude/pega-referencia-validada'))
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--hot-clearance', type=float, default=.004)
ap.add_argument('--contact', type=float, default=.0006, help='distancia que conta como encostou')
ap.add_argument('--top', type=int, default=8)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)

src = json.loads((a.source/'report.json').read_text())
m = mujoco.MjModel.from_xml_path(src['scene']); d = mujoco.MjData(m)
ref = json.loads((a.arm_ref/'report.json').read_text())
base = np.array(ref['grasp']['qpos'])

K = m.body('chaleira').id
W = m.body('left_wrist_yaw_link').id
kadr = m.jnt_qposadr[m.joint('chaleira_livre').id]
handle = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')]
hot = m.geom('chaleira_hot_body').id
kettle_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == K and m.geom_contype[g]]
hand_geoms = [g for g in range(m.ngeom) if m.geom_contype[g]
              and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand', 'left_wrist'))]
def link_geoms(nome):
    return [g for g in hand_geoms if m.body(int(m.geom_bodyid[g])).name == nome]
TIPS = {'thumb': link_geoms('left_hand_thumb_2_link'),
        'index': link_geoms('left_hand_index_1_link'),
        'middle': link_geoms('left_hand_middle_1_link')}
PROX = {'thumb': link_geoms('left_hand_thumb_1_link'),
        'index': link_geoms('left_hand_index_0_link'),
        'middle': link_geoms('left_hand_middle_0_link')}
JN = ['left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
      'left_hand_index_0_joint', 'left_hand_index_1_joint',
      'left_hand_middle_0_joint', 'left_hand_middle_1_joint']
HA = np.array([m.jnt_qposadr[m.joint(n).id] for n in JN])
RANGE = np.array([m.jnt_range[m.joint(n).id] for n in JN])
# direcao de fechamento: indicador/medio negativo, polegar 1 e 2 positivo
CLOSE_TO = np.array([0., RANGE[1, 1], RANGE[2, 1], RANGE[3, 0], RANGE[4, 0], RANGE[5, 0], RANGE[6, 0]])

# Geometria da barra no referencial da chaleira, pelos vertices das malhas.
d.qpos[:] = base; mujoco.mj_kinematics(m, d)
Rk0, pk0 = d.xmat[K].reshape(3, 3).copy(), d.xpos[K].copy()
verts = []
for g in handle:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    Rg = d.geom_xmat[g].reshape(3, 3)
    verts.append((Rk0.T @ (d.geom_xpos[g][:, None] + Rg @ v.T - pk0[:, None])).T)
verts = np.vstack(verts)
bar = verts[(verts[:, 2] > .07) & (verts[:, 2] < .16)]
BAR = {'x': (float(bar[:, 0].min()), float(bar[:, 0].max())),
       'y': (float(bar[:, 1].min()), float(bar[:, 1].max())),
       'z': (float(verts[:, 2].min()), float(verts[:, 2].max()))}
bx, by = np.mean(BAR['x']), np.mean(BAR['y'])
print('barra no referencial da chaleira (mm):',
      {k: [round(v*1000, 1) for v in val] for k, val in BAR.items()}, flush=True)

# Pose do punho fixa (robo parado); a chaleira e' posta em Wpose * T_kw^-1.
Rw, pw = d.xmat[W].reshape(3, 3).copy(), d.xpos[W].copy()


def poe_chaleira(R_kw, p_kw):
    """R_kw, p_kw: pose do PUNHO no referencial da chaleira."""
    Rk = Rw @ R_kw.T
    pk = pw - Rk @ p_kw
    d.qpos[kadr:kadr+3] = pk
    q = Rotation.from_matrix(Rk).as_quat()  # x y z w
    d.qpos[kadr+3:kadr+7] = [q[3], q[0], q[1], q[2]]


def dist(gs, alvo, lim=.05):
    return min(mujoco.mj_geomDistance(m, d, g, h, lim, None) for g in gs for h in alvo)


def normal(gs):
    best, ft = None, np.zeros(6)
    for g in gs:
        for h in handle:
            f = np.zeros(6)
            s = mujoco.mj_geomDistance(m, d, g, h, .05, f)
            if best is None or s < best:
                best, ft = s, f
    n = ft[3:] - ft[:3]
    return n / max(np.linalg.norm(n), 1e-9) * (1 if best >= 0 else -1), best


def fecha(t0):
    """Fecha os dedos junta a junta ate encostar; devolve hand_q."""
    q = np.zeros(7); q[0] = t0
    d.qpos[HA] = q; mujoco.mj_kinematics(m, d)
    dedos = {'thumb': (1, 2), 'index': (3, 4), 'middle': (5, 6)}
    for dedo, (j0, j1) in dedos.items():
        parado0 = parado1 = False
        for u in np.linspace(0, 1, 41)[1:]:
            if not parado0:
                q[j0] = CLOSE_TO[j0] * u
            if not parado1:
                q[j1] = CLOSE_TO[j1] * u
            d.qpos[HA] = q; mujoco.mj_kinematics(m, d)
            if dist(TIPS[dedo], handle) <= a.contact or dist(TIPS[dedo], kettle_geoms) <= 0:
                parado1 = True
            if dist(PROX[dedo], handle) <= a.contact:
                parado0 = True
            if min(dist(TIPS[dedo] + PROX[dedo], [hot]), 1) < a.hot_clearance:
                # recua um passo e para: nao encosta no metal quente
                q[j0] = CLOSE_TO[j0] * (u - 1/40) if not parado0 else q[j0]
                q[j1] = CLOSE_TO[j1] * (u - 1/40) if not parado1 else q[j1]
                break
            if parado0 and parado1:
                break
    d.qpos[HA] = q; mujoco.mj_kinematics(m, d)
    return q


import time
_t0 = time.monotonic(); _n = 0
import collections; rej = collections.Counter()
resultados = []
alturas = np.linspace(BAR['z'][0] + .045, BAR['z'][1] - .045, 4)
for s, theta, zh, xb, yb, t0 in itertools.product(
        (1., -1.), np.deg2rad(np.arange(0, 360, 20)), alturas,
        (.060, .080, .100, .120, .140, .160), (-.020, -.030, -.040, -.050, -.060), np.deg2rad((-60., -30., 0., 30., 60.))):
    ex = np.array([np.cos(theta), np.sin(theta), 0.])
    ez = np.array([0., 0., s])
    ey = np.cross(ez, ex)
    R_kw = np.column_stack([ex, ey, ez])
    p_kw = np.array([bx, by, zh]) - R_kw @ np.array([xb, yb, 0.])
    _n += 1
    if _n in (5, 50) or _n % 200 == 0:
        print(f'  pose {_n}: {(time.monotonic()-_t0)/_n*1000:.0f} ms por pose, {len(resultados)} validas', flush=True)
    d.qpos[:] = base
    poe_chaleira(R_kw, p_kw)
    q_aberta = np.zeros(7); q_aberta[0] = t0
    d.qpos[HA] = q_aberta; mujoco.mj_kinematics(m, d)
    # mao aberta nao pode estar dentro da chaleira nem colada no metal quente
    if dist(hand_geoms, kettle_geoms, .01) < .001:
        rej['aberta_colide'] += 1; continue
    if dist(hand_geoms, [hot], .02) < a.hot_clearance:
        rej['aberta_quente'] += 1; continue
    q = fecha(t0)
    gaps = {k: dist(v, handle) for k, v in TIPS.items()}
    if max(gaps.values()) > .0015 or min(gaps.values()) < -.0005:
        rej['pontas:' + ','.join(k for k, v in gaps.items() if v > .0015 or v < -.0005)] += 1; continue
    quente = dist(hand_geoms, [hot], .05)
    if quente < a.hot_clearance:
        rej['fechada_quente'] += 1; continue
    penetra = min(dist(hand_geoms, [g], .01) for g in kettle_geoms if g not in handle and g != hot)
    if penetra < 0:
        rej['penetra_corpo'] += 1; continue
    n_t, _ = normal(TIPS['thumb']); n_i, _ = normal(TIPS['index']); n_m, _ = normal(TIPS['middle'])
    opos = [float(n_t @ n_i), float(n_t @ n_m)]
    if max(opos) > -.3:
        rej['oposicao'] += 1; continue
    # quao envolvente: os dedos terminam do lado da barra oposto a palma?
    pal = Rk0.T @ (d.xpos[W] - d.xpos[K])
    resultados.append({
        's': s, 'theta_deg': float(np.rad2deg(theta)), 'altura_mm': float(zh*1000),
        'barra_na_mao_mm': [xb*1000, yb*1000], 'thumb0_deg': float(np.rad2deg(t0)),
        'hand_q': q.tolist(), 'pontas_mm': {k: round(v*1000, 2) for k, v in gaps.items()},
        'oposicao': opos, 'quente_mm': float(quente*1000),
        'punho_na_chaleira_mm': (R_kw.T @ np.zeros(3) + p_kw*1000).tolist(),
        'qpos': d.qpos.copy().tolist()})

print('rejeicoes:', dict(rej.most_common()), flush=True)
resultados.sort(key=lambda r: (max(r['oposicao']), -r['quente_mm']))
print(f'{len(resultados)} pegas envolventes validas', flush=True)
for r in resultados[:a.top]:
    print(f"  theta {r['theta_deg']:5.0f} s {r['s']:+.0f} altura {r['altura_mm']:5.1f} "
          f"barra {r['barra_na_mao_mm']} t0 {r['thumb0_deg']:4.0f} | pontas {r['pontas_mm']} "
          f"opos {[round(x, 2) for x in r['oposicao']]} quente {r['quente_mm']:.1f} "
          f"punho {np.round(r['punho_na_chaleira_mm'], 0)}", flush=True)

# Cada pega vira um relatorio de referencia no formato que o
# plan_kettle_grasp_v3 --seed ... --fixed-hand consome.
for i, r in enumerate(resultados[:a.top]):
    pasta = a.out / f'ref-{i}'; pasta.mkdir()
    (pasta/'report.json').write_text(json.dumps({
        'pass': True, 'scene': src['scene'], 'origem': 'synth_wrap_grasp.py', 'pega': {
            k: v for k, v in r.items() if k != 'qpos'},
        'grasp': {'qpos': r['qpos'], 'joint_names': ref['grasp']['joint_names'], 'q': ref['grasp']['q'],
                  'hand_joint_names': JN, 'hand_q': r['hand_q']}}, indent=1))
(a.out/'report.json').write_text(json.dumps({
    'barra_m': BAR, 'validas': len(resultados),
    'resultados': [{k: v for k, v in r.items() if k != 'qpos'} for r in resultados]}, indent=1))
