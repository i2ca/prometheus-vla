"""Refit the handle grasp with geometric elbow limits in the current scene.

The historical grip supplies a hand shape and object-relative pose hint only.
Its old elbow configuration is not accepted as a valid current robot posture.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--seed', type=Path, default=Path('results/loadable-refined-002'))
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--seeds', type=int, default=6)
# O planejador de pega mira 7 mm de folga do metal quente e aprova a partir de
# 6 mm, mas o planejador de conexao recusa qualquer estado abaixo de 8 mm. Os
# seis candidatos saiam com exatamente 0,007 m e morriam 1 mm depois. Pedir
# mais folga aqui e' ficar mais conservador, nao menos.
ap.add_argument('--hot-clearance', type=float, default=.007)
# O alvo da palma e' a pose historica transportada para a chaleira atual. Com a
# chaleira 14 mm deslocada, essa pose encosta a mao no corpo quente. Deslocar o
# alvo alguns milimetros permite procurar uma pega mais alta na alca, mais longe
# do metal, sem mexer em portao nenhum.
ap.add_argument('--goal-offset', type=float, nargs=3, default=[0., 0., 0.])
# O alvo da palma e' preso com peso 1000 e a orientacao com 30. Com a chaleira
# deslocada, essa pose historica e' justamente a que encosta no metal quente.
# Afrouxar o vinculo deixa o otimizador procurar outra pose de palma que ainda
# oponha os dedos na alca; os portoes de oposicao, folga e anatomia continuam.
# A cintura estava travada em +-12 graus tambem aqui, como nos planejadores do
# botao. Com a chaleira 14 mm mais longe, o braco trabalha perto do limite de
# alcance e a mao se fecha numa postura que aproxima o polegar do metal. Deixar
# o tronco ajudar e' usar a capacidade real do robo (o modelo permite +-29,79).
# Porta a pega para a mao DIREITA. Motivo medido: levando as duas palmas a
# mesma pose de pega, a direita fica a 32,34 mm do corpo quente e a esquerda a
# 10,67 mm. A alca aponta para o lado esquerdo do robo, entao a direita a
# alcanca pelo outro lado. Com 32 mm de folga o milimetro perdido no
# assentamento deixa de importar.
ap.add_argument('--hand-side', type=str, default='left', choices=['left', 'right'])
ap.add_argument('--waist-rp-deg', type=float, default=12.)
ap.add_argument('--goal-weight', type=float, default=1000.)
ap.add_argument('--goal-rot-weight', type=float, default=30.)
# O objetivo so' pedia que as tres pontas chegassem perto da alca; qual lado
# cada dedo toca ficava por conta do acaso, e o portao de oposicao julgava
# depois. Com a mao espelhada isso passou a falhar sempre: polegar e indicador
# caiam do mesmo lado (cosseno +0,8). Este termo poe a oposicao dentro do
# objetivo, usando as mesmas normais de contato que o portao usa.
ap.add_argument('--oppose-weight', type=float, default=0.)
ap.add_argument('--oppose-target', type=float, default=-.5,
                help='cosseno desejado entre a normal do polegar e a dos outros dedos')
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
for p in [Path(__file__), Path('scripts/elbow_anatomy.py')]:
    shutil.copy2(p, a.out/p.name)
r = json.loads((a.source/'report.json').read_text())
assert r['pass']
s = G1Sim(r['scene'])
m, d = s.m, s.d
ck = np.load(a.source/'continuation.npz')
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)
old = json.loads((a.seed/'report.json').read_text())
om = mujoco.MjModel.from_xml_path(old['scene'])
_P = a.hand_side + '_'
od = mujoco.MjData(om)
od.qpos[:] = old['grasp']['qpos']
mujoco.mj_forward(om, od)
oj, op = om.body('chaleira').id, om.body('left_wrist_yaw_link').id
jar = m.body('chaleira').id
palm = m.body(_P+'wrist_yaw_link').id
# "right" aqui e' o braco que NAO pega, so fica preso onde esta.
right = m.body(('left_' if a.hand_side == 'right' else 'right_')+'wrist_yaw_link').id
alignment = d.xmat[jar].reshape(3, 3)@od.xmat[oj].reshape(3, 3).T
goal = d.xpos[jar]+alignment@(od.xpos[op]-od.xpos[oj])+np.array(a.goal_offset)
goal_R = alignment@od.xmat[op].reshape(3, 3)
other_goal = d.xpos[right].copy()
arm_names = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'] + [n.replace('right_', 'left_') for n in ARM_JOINTS] + list(ARM_JOINTS)
hand_names = [n.replace('right_', _P) for n in HAND_JOINTS]
ik = ArmIK(s, arm_names+hand_names, palm=_P+'wrist_yaw_link')
bounds = ik.bounds.copy()
bounds[0] = [-.7, .7]
bounds[1:3] = np.deg2rad([[-a.waist_rp_deg, a.waist_rp_deg]]*2)
anatomy = ElbowAnatomy(m)
anatomy.bound_search(m, ik.joint_names, bounds)
for j in [7, 14]:
    bounds[j:j+3] = np.deg2rad([[-60, 60], [-45, 45], [-30, 30]])
base = d.qpos[ik.qa].copy()
# A semente historica de dedos e' da mao ESQUERDA. As duas maos sao espelhadas:
# comparando as faixas das juntas no proprio modelo, todas tem sinal invertido
# entre os lados, menos thumb_0, que e' simetrica. Entregar a semente crua para
# a direita colocava os dedos fora de faixa, e a oposicao do indicador nunca
# fechava. Aqui a semente e' espelhada antes de virar chute inicial.
_semente = np.array(old['grasp']['hand_q'], dtype=float)
if a.hand_side == 'right':
    _espelho = np.array([1., -1., -1., -1., -1., -1., -1.])
    _semente = _semente * _espelho
base[17:] = _semente
historical = base.copy()
for name, value in zip(old['grasp']['joint_names'], old['grasp']['q']):
    historical[ik.joint_names.index(name)] = value
hg = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith((_P+'hand', _P+'wrist'))]
tips = [next(g for g in hg if m.body(int(m.geom_bodyid[g])).name == _P+'hand_'+n+'_link') for n in ['thumb_2', 'index_1', 'middle_1']]
handle = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')]
hot = m.geom('chaleira_hot_body').id
table = m.geom('tampo').id
rng = np.random.default_rng(20260920)
results, states = [], []
for seed in range(a.seeds):
    q = (base if seed % 2 == 0 else historical).copy()
    if seed > 1:
        q[:10] += rng.normal(0, .6, 10)
    q = np.clip(q, bounds[:, 0]+1e-8, bounds[:, 1]-1e-8)
    ref = q.copy()
    avoid = set()
    for retry in range(8):
        pairs = sorted(avoid)
        def normais_de_contato():
            """Normal do par mais proximo entre cada ponta e a alca, como no portao."""
            saida = []
            for g in tips:
                melhor, pontos = None, np.zeros(6)
                for h in handle:
                    pts = np.zeros(6)
                    dist = mujoco.mj_geomDistance(m, ik.d, g, h, .1, pts)
                    if melhor is None or dist < melhor:
                        melhor, pontos = dist, pts.copy()
                n = (pontos[3:] - pontos[:3]) * np.sign(melhor if melhor else 1.)
                saida.append(n / max(np.linalg.norm(n), 1e-9))
            return saida

        def residual(x):
            p, R = ik.fk(x)
            gaps = [min(mujoco.mj_geomDistance(m, ik.d, g, h, .1, None) for h in handle) for g in tips]
            oposicao = []
            if a.oppose_weight:
                # Sem o fator de proximidade o otimizador aprende a fugir da alca:
                # longe nao ha contato, nao ha normal, e a penalidade some. O peso
                # so' vale quando a ponta ja esta perto o suficiente para tocar.
                ns = normais_de_contato()
                perto = [min(1., max(0., (.015 - g) / .015)) for g in gaps]
                oposicao = [max(0., float(ns[0] @ n) - a.oppose_target) * min(perto[0], perto[i + 1])
                            for i, n in enumerate(ns[1:])]
            thermal = [min(0, mujoco.mj_geomDistance(m, ik.d, g, hot, .02, None)-a.hot_clearance) for g in hg]
            obstacles = [min(0, mujoco.mj_geomDistance(m, ik.d, g, h, .02, None)-.003) for g, h in pairs]
            table_gaps = [min(0, mujoco.mj_geomDistance(m, ik.d, g, table, .03, None)-.023) for g in hg]
            return np.r_[(p-goal)*a.goal_weight, Rotation.from_matrix(R@goal_R.T).as_rotvec()*a.goal_rot_weight,
                         (ik.d.xpos[right]-other_goal)*300,
                         (np.array(gaps)-.0008)*1500,
                         np.array(thermal)*5000, np.array(obstacles)*5000,
                         np.array(table_gaps)*5000, np.array(oposicao)*a.oppose_weight,
                         anatomy.penalty(ik.d), (x-ref)*.05]
        fit = least_squares(residual, q, bounds=bounds.T, max_nfev=120)
        q = fit.x
        p, R = ik.fk(q)
        mujoco.mj_forward(m, ik.d)
        bad = []
        for c in ik.d.contact:
            if c.dist >= 0:
                continue
            gs = [int(c.geom1), int(c.geom2)]
            bn = [m.body(int(m.geom_bodyid[g])).name for g in gs]
            if any(n.startswith(('left_', 'right_', 'torso', 'waist', 'pelvis', 'head')) for n in bn):
                bad.append(bn)
                avoid.add(tuple(sorted(gs)))
        if not bad:
            break
    closest = []
    normals = []
    for g in tips:
        choices = []
        for h in handle:
            pts = np.zeros(6)
            distance = mujoco.mj_geomDistance(m, ik.d, g, h, .1, pts)
            choices.append((distance, pts))
        distance, pts = min(choices, key=lambda x: x[0])
        n = (pts[3:]-pts[:3])*np.sign(distance)
        n /= max(np.linalg.norm(n), 1e-9)
        closest.append(float(distance))
        normals.append(n)
    hot_gap = min(mujoco.mj_geomDistance(m, ik.d, g, hot, 1, None) for g in hg)
    table_gap = min(mujoco.mj_geomDistance(m, ik.d, g, table, 1, None) for g in hg)
    opposition = [float(normals[0]@n) for n in normals[1:]]
    error = float(np.linalg.norm(p-goal))
    passed = (anatomy.valid(ik.d, tolerance_deg=.2) and not bad and
              hot_gap >= a.hot_clearance-.001 and table_gap >= .022 and error < .005 and
              min(closest) >= 0 and max(closest) < .0018 and max(opposition) < -.3)
    row = {'seed': seed, 'pass': bool(passed), 'tip_gaps_m': closest,
           'opposition_cosines': opposition, 'hot_clearance_m': hot_gap,
           'table_clearance_m': table_gap, 'palm_error_m': error,
           'elbow_flexion_deg': anatomy.angles(ik.d).tolist(), 'contacts': bad,
           'q': q[:17].tolist(), 'hand_q': q[17:].tolist(), 'palm_R': R.tolist()}
    results.append(row)
    states.append(ik.d.qpos.copy())
    print(row, flush=True)
result = {'model': 'gpt-6-astra', 'source': str(a.source), 'scene': r['scene'],
          'historical_pose_hint': str(a.seed), 'joint_names': arm_names,
          'hand_names': hand_names, 'results': results,
          'scope': 'offline grasp geometry with corrected elbow band; no load support proven'}
(a.out/'report.json').write_text(json.dumps(result, indent=2))
np.savez_compressed(a.out/'candidates.npz', qpos=states)
