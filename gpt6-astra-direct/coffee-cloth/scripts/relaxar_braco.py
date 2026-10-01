"""Baixa o braco que nao esta trabalhando para uma pose de descanso.

O braco direito ficava congelado na pose em que largou a colher: rolamento do
ombro no limite da junta (-129 graus), umero girado 127 graus e cotovelo quase
todo fechado (137 graus de flexao real), com a mao erguida na frente do peito
durante a fervura, a pega, o despejo e a devolucao. Uma pessoa baixa o braco
depois de largar a colher.

Procura uma pose de descanso ao lado do corpo sem contato, confere o caminho
em espaco de juntas e executa com os mesmos motores e o mesmo modelo de agua e
calor dos outros executores. Tudo o que nao e' o braco fica no alvo anterior.

uso: relaxar_braco.py --source <estado> --out <pasta> [--side right]
"""
import argparse, itertools, json, shutil, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from elbow_anatomy import ElbowAnatomy
from brew_state import BrewState
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--side', default='right', choices=['left', 'right', 'both'])
ap.add_argument('--seconds', type=float, default=5.)
ap.add_argument('--hold', type=float, default=1.5)
ap.add_argument('--clearance', type=float, default=.01, help='folga minima do braco a qualquer coisa no caminho')
# A etapa do botao congelava as juntas durante a fervura com a cintura girada
# 1,0 rad e o braco esquerdo erguido em cima da chaleira. Estas opcoes deixam o
# descanso tambem desfazer a torcao do tronco e esperar a agua ferver na postura
# relaxada (o modelo de calor segue rodando aqui).
# folga das maos as proprias pernas e pelve: com 18 mm a mao direita pendurada
# bateu no quadril quando o tronco inclinou para servir o cafe
ap.add_argument('--body-clearance', type=float, default=.035)
ap.add_argument('--table-clearance', type=float, default=.03)
ap.add_argument('--waist-to', type=float, nargs=3, default=None, help='yaw roll pitch da cintura, graus')
ap.add_argument('--until-boil', action='store_true')
ap.add_argument('--max-wait', type=float, default=420.)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, a.out/Path(__file__).name)

r = json.loads((a.source/'report.json').read_text())
s = G1Sim(r['scene']); m, d = s.m, s.d
brew = BrewState(m, a.source)
ck = np.load(a.source/'continuation.npz')
water = KettleWater()
coupler = LiquidMassCoupler(m, float(ck['dry_kettle_kg']))
for n, v in json.loads((a.source/'liquid-state.json').read_text()).items():
    setattr(water, n, v)
for n in ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'bvh_aabb']:
    getattr(m, n)[:] = ck[n]
mujoco.mj_setConst(m, mujoco.MjData(m))
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)
s.kp[:], s.kd[:] = ck['kp'], ck['kd']
target = ck['last_target'].copy()

# 'both': durante a prensa o braco ocioso fica preso no mundo enquanto a
# cintura gira 57 graus, entao as juntas dele mudam; quando o tronco volta ao
# centro ele sobra torto (umero 78 graus erguido para tras). Relaxar os dois
# juntos com a cintura resolve.
SIDES = ['left', 'right'] if a.side == 'both' else [a.side]
ARM = ['shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint',
       'elbow_joint', 'wrist_roll_joint', 'wrist_pitch_joint', 'wrist_yaw_joint']
NOMES = [sd+'_'+n for sd in SIDES for n in ARM]
P = SIDES[0] + '_'
if a.waist_to is not None:
    NOMES += ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
ADR = np.array([m.jnt_qposadr[m.joint(n).id] for n in NOMES])
LIM = np.array([m.jnt_range[m.joint(n).id] for n in NOMES])
# O ombro (pitch e roll) encosta no tronco por construcao; a checagem fica com
# umero, antebraco e mao.
# Se a cintura mexe, os dois bracos giram junto e os dois entram na checagem.
LADOS = tuple(sd+'_' for sd in SIDES) if a.waist_to is None else ('left_', 'right_')
BRACO = [b for b in range(m.nbody) if m.body(b).name.startswith(tuple(L+x for L in LADOS for x in ('shoulder_yaw', 'elbow', 'wrist', 'hand')))]
OMBRO = [b for b in range(m.nbody) if m.body(b).name.startswith(tuple(L+x for L in LADOS for x in ('shoulder_pitch', 'shoulder_roll')))]
BRACO_G = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] in BRACO]
OUTROS_G = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] not in BRACO + OMBRO]
anatomia = ElbowAnatomy(m)
lado = 0 if a.side == 'left' else 1
sinal = -1. if a.side == 'right' else 1.   # rolamento para fora do corpo
inicio = d.qpos[ADR].copy()
scratch = mujoco.MjData(m)


def folga(q):
    scratch.qpos[:] = d.qpos; scratch.qpos[ADR] = q
    mujoco.mj_kinematics(m, scratch)
    return min(mujoco.mj_geomDistance(m, scratch, g, h, .05, None) for g in BRACO_G for h in OUTROS_G)


# Pose de descanso escolhida pelo que ela e' no corpo, nao por angulo de junta:
# os mesmos angulos sao 'braco caido' para uma orientacao do tronco e 'braco
# erguido para tras' para outra. Criterio no referencial do tronco: umero
# quase vertical, mao ao lado da coxa, nada atras do corpo.
TORSO = m.body('torso_link').id
TAMPO = m.geom('tampo').id
nw = 3 if a.waist_to is not None else 0
def medida_braco(q, sd):
    scratch.qpos[:] = d.qpos; scratch.qpos[ADR] = q; mujoco.mj_kinematics(m, scratch)
    Rt = scratch.xmat[TORSO].reshape(3, 3)
    o = scratch.xpos[m.body(sd+'_shoulder_roll_link').id]; c = scratch.xpos[m.body(sd+'_elbow_link').id]
    u = Rt.T @ (c - o); u /= np.linalg.norm(u)
    mao = Rt.T @ (scratch.xpos[m.body(sd+'_wrist_yaw_link').id] - scratch.xpos[TORSO])
    return float(np.rad2deg(np.arccos(np.clip(-u[2], -1, 1)))), mao, float(anatomia.angles(scratch)[0 if sd == 'left' else 1])
descanso = inicio.copy()
if nw:
    descanso[-3:] = np.deg2rad(a.waist_to)
flex_descanso = {}
for bi, sd in enumerate(SIDES):
    sg = 1. if sd == 'left' else -1.
    melhor = None
    for pitch, roll, elbow in itertools.product(range(-20, 41, 5), (6., 12., 18., 24.), (62., 70., 78.)):
        q = descanso.copy()
        q[bi*7:(bi+1)*7] = np.deg2rad([pitch, sg*roll, 0., elbow, 0., 0., 0.])
        if np.any(q < LIM[:, 0]) or np.any(q > LIM[:, 1]):
            continue
        elev, mao, flex = medida_braco(q, sd)
        custo = abs(elev - 8)/10 + abs(flex - 20)/15 + max(0., -mao[0] - .03)*40 + abs(roll - 12)/20
        if melhor is not None and custo >= melhor[0]:
            continue
        if folga(q) < a.clearance:
            continue
        scratch.qpos[:] = d.qpos; scratch.qpos[ADR] = q; mujoco.mj_kinematics(m, scratch)
        maos = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith((sd+'_hand', sd+'_wrist'))]
        corpo = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('pelvis', sd+'_hip', sd+'_knee'))]
        if min(mujoco.mj_geomDistance(m, scratch, g, h, .1, None) for g in maos for h in corpo) < a.body_clearance:
            continue
        # a conexao da chaleira exige 22 mm das duas maos ao tampo ja' na pose
        # inicial; com 21,3 mm ela nem comecava a abrir os dedos
        if min(mujoco.mj_geomDistance(m, scratch, g, TAMPO, .1, None) for g in maos) < a.table_clearance:
            continue
        melhor =(custo, q[bi*7:(bi+1)*7].copy(), elev, flex, mao)
    assert melhor, f'nenhuma pose de descanso sem contato para {sd}'
    descanso[bi*7:(bi+1)*7] = melhor[1]
    flex_descanso[sd] = {'flexao_cotovelo_deg': round(melhor[3], 1), 'elevacao_umero_deg': round(melhor[2], 1),
                         'mao_no_tronco_mm': (melhor[4]*1000).round(0).tolist()}
assert folga(descanso) >= a.clearance, 'poses de descanso se colidem entre si'

def caminho_livre(pontos, n=80):
    pior = 1.
    for i in range(len(pontos)-1):
        for u in np.linspace(0, 1, n):
            pior = min(pior, folga(pontos[i]*(1-u) + pontos[i+1]*u))
    return pior

# A rota direta varre a quina da mesa: ombro indo para tras e braco descendo
# ao mesmo tempo passam a mao por dentro do tampo. RRT-connect nas 7 juntas,
# depois atalhos aleatorios para tirar os zigue-zagues.
rng = np.random.default_rng(20260922)
def livre(q):
    return bool(np.all(q >= LIM[:, 0]) and np.all(q <= LIM[:, 1])) and folga(q) >= a.clearance
def aresta(q1, q2):
    n = max(2, int(np.max(np.abs(q2-q1))/.04)+1)
    return all(livre(q1*(1-u)+q2*u) for u in np.linspace(0, 1, n)[1:])
def estende(arvore, alvo, passo=.2):
    nos, pais = arvore
    i = int(np.argmin(np.linalg.norm(np.array(nos)-alvo, axis=1)))
    dq = alvo-nos[i]; dist = np.linalg.norm(dq)
    novo = alvo if dist <= passo else nos[i]+dq/dist*passo
    if not aresta(nos[i], novo):
        return None
    nos.append(novo); pais.append(i); return len(nos)-1
def ramo(arvore, i):
    nos, pais = arvore; out = []
    while i != -1:
        out.append(nos[i]); i = pais[i]
    return out
assert livre(inicio) or folga(inicio) > 0, 'o braco ja comeca em contato'
arvores = [([inicio], [-1]), ([descanso], [-1])]
rota = [inicio, descanso] if aresta(inicio, descanso) else None
for it in range(4000):
    if rota:
        break
    A, B = arvores[it % 2], arvores[1 - it % 2]
    amostra = descanso if it % 2 == 0 and rng.random() < .2 else rng.uniform(LIM[:, 0], LIM[:, 1])
    ia = estende(A, amostra)
    if ia is None:
        continue
    for _ in range(60):
        ib = estende(B, A[0][ia])
        if ib is None:
            break
        if np.linalg.norm(B[0][ib]-A[0][ia]) < 1e-6:
            ra, rb = ramo(A, ia), ramo(B, ib)
            caminho = ra[::-1] + rb[1:]
            rota = caminho if np.allclose(caminho[0], inicio) else caminho[::-1]
            break
assert rota, 'nenhuma rota sem colisao para o braco'
for _ in range(300):   # atalhos
    if len(rota) < 3:
        break
    i, j = sorted(rng.choice(len(rota), 2, replace=False))
    if j - i > 1 and aresta(rota[i], rota[j]):
        rota = rota[:i+1] + rota[j:]
pior_rota = min(folga(rota[i]*(1-u)+rota[i+1]*u) for i in range(len(rota)-1) for u in np.linspace(0, 1, 30))
print(f'descanso {np.rad2deg(descanso).round(1)} {flex_descanso} | '
      f'rota com {len(rota)-1} trecho(s), folga minima {pior_rota*1000:.1f} mm', flush=True)
assert pior_rota > 0, 'nenhuma rota sem colisao para o braco'

dt = m.opt.timestep
def smooth(x):
    x = min(max(x, 0.), 1.); return x*x*(3-2*x)
duracoes = np.array([max(np.max(np.abs(rota[i+1]-rota[i])), 1e-6) for i in range(len(rota)-1)])
tempos = np.r_[0, np.cumsum(duracoes)] / duracoes.sum() * a.seconds
objetos = [m.body(n).id for n in ['chaleira', 'coador', 'copo', 'pote', 'tampa', 'scoop', 'base_eletrica']
           if any(m.body(b).name == n for b in range(m.nbody))]
obj0 = d.xpos[objetos].copy()
rows, states, liquid = [], [], []
pior_contato, bad = 0., None
fervida_em = None
passos = int((a.seconds + (a.max_wait if a.until_boil else a.hold))/dt)
for k in range(passos):
    t = k*dt
    i = min(int(np.searchsorted(tempos, t, side='right')-1), len(rota)-2)
    u = smooth((t-tempos[i])/(tempos[i+1]-tempos[i])) if t < a.seconds else 1.
    target[ADR] = rota[i]*(1-u) + rota[i+1]*u if t < a.seconds else rota[-1]
    erro = target[s.qadr]-d.qpos[s.qadr]
    tau = s.kp*erro - s.kd*d.qvel[s.vadr] + d.qfrc_bias[s.vadr]
    d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    mujoco.mj_step(m, d)
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in BRACO) != (b2 in BRACO) and b1 not in OMBRO and b2 not in OMBRO and c.dist < 0:
            pior_contato = max(pior_contato, -c.dist)
            bn = [m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
            if -c.dist > .0005:
                bad = {'reason': 'arm contact', 'bodies': bn, 'depth_m': float(-c.dist), 't': t}
    if k % 10 == 0:
        brew.step(d, .02, water)
        bs = [m.body(n).id for n in ['chaleira', 'coador', 'copo']]
        flow = water.step(.02, *sum(([d.xpos[b], d.xmat[b].reshape(3, 3)] for b in bs), []))
        coupler.apply(d, water)
        liquid.append({'t': float(d.time), **flow})
    if k % 16 == 0:
        states.append(d.qpos.copy())
        rows.append({'t': float(d.time), 'phase': 'relax_arm' if t < a.seconds else
                     ('heat_water' if a.until_boil and fervida_em is None else 'rest_arm'),
                     'temperature_C': float(brew.heater.temperature_C)})
    if a.until_boil and fervida_em is None and brew.heater.last_event == 'automatic_boil_cutoff' \
            and brew.heater.temperature_C > 99:
        fervida_em = t
    if a.until_boil and fervida_em is not None and t > max(a.seconds, fervida_em) + a.hold:
        break
    if bad or s.warnings():
        bad = bad or {'reason': 'numerical warning', 'warnings': s.warnings()}
        break

deriva_obj = float(np.max(np.linalg.norm(d.xpos[objetos]-obj0, axis=1)))
if a.until_boil and fervida_em is None and not bad:
    bad = {'reason': 'water did not boil', 'temperature_C': float(brew.heater.temperature_C),
           'heater_on': bool(brew.heater.on)}
if deriva_obj > .002 and not bad:
    bad = {'reason': 'object moved', 'max_m': deriva_obj}
erro_final = float(np.rad2deg(np.max(np.abs(d.qpos[ADR]-rota[-1]))))
report = {'model': 'claude', 'source': str(a.source), 'scene': r['scene'], 'pass': bad is None,
          'failure': bad, 'side': a.side, 'rest_pose_deg': np.rad2deg(descanso).tolist(),
          'rest_arms': flex_descanso, 'start_pose_deg': np.rad2deg(inicio).tolist(),
          'route_segments': len(rota)-1, 'route_min_clearance_m': float(pior_rota),
          'max_arm_contact_depth_m': pior_contato, 'max_object_drift_m': deriva_obj,
          'final_joint_error_deg': erro_final, 'coffee_completed': False,
          'waist_to_deg': a.waist_to, 'boiled_at_s': fervida_em,
          'heater': {'temperature_C': float(brew.heater.temperature_C), 'last_event': brew.heater.last_event},
          'water_state': asdict(water), 'rows': rows, 'liquid_samples': liquid}
(a.out/'report.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(a.out/'trajectory.npz', qpos=states)
integ = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
mujoco.mj_getState(m, d, integ, mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(a.out/'continuation.npz', integration=integ, last_target=target, kp=s.kp, kd=s.kd,
                    body_mass=m.body_mass, body_ipos=m.body_ipos, body_inertia=m.body_inertia,
                    body_iquat=m.body_iquat, bvh_aabb=m.bvh_aabb, dry_kettle_kg=float(ck['dry_kettle_kg']))
(a.out/'liquid-state.json').write_text(json.dumps(asdict(water), indent=2))
brew.save(a.out)
print({k: v for k, v in report.items() if k not in ('rows', 'liquid_samples', 'water_state')})
