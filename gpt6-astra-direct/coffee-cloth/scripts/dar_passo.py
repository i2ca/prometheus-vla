"""Da' um passo com o robo entre duas etapas da cadeia.

Motivo medido: na pega da chaleira a distancia ombro-punho fica entre 359 e
362 mm em todas as fases, e o maximo que o braco alcanca e' 359 mm. O braco
trabalha esticado no limite, o cotovelo nao tem para onde dobrar e a mao so'
se move girando o umero (41 graus ate a alca, 19 na subida). Uma pessoa
chega perto do balcao antes de pegar a chaleira.

A pelve do G1 nesta cena e' soldada ao mundo (sem junta livre), entao o passo
e' uma translacao rigida do robo: nova cena com a pelve deslocada e o mesmo
estado de juntas. E' uma abstracao de locomocao; o caminhar em si nao e'
simulado. Objetos, agua, po e aquecedor seguem exatamente como estavam.

uso: dar_passo.py <fonte> <destino> <dx_m> <dy_m>
"""
import json, re, shutil, sys
from pathlib import Path
import numpy as np, mujoco

fonte, destino = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
dx, dy = float(sys.argv[3]), float(sys.argv[4])
rep = json.loads((fonte / 'report.json').read_text())
cena = Path(rep['scene'])
texto_cena = cena.read_text()
inc = re.search(r'<include file="([^"]*robot\.xml)"', texto_cena)
robo = Path(inc.group(1))
texto_robo = robo.read_text()
m0 = re.search(r'<body name="pelvis" pos="([^"]+)"', texto_robo)
px, py, pz = map(float, m0.group(1).split())

destino.mkdir(parents=True, exist_ok=False)
for f in fonte.iterdir():
    if f.is_file():
        shutil.copy2(f, destino / f.name)
shutil.copy2(__file__, destino / Path(__file__).name)
(destino / 'robot.xml').write_text(texto_robo.replace(
    m0.group(0), f'<body name="pelvis" pos="{px+dx:.4f} {py+dy:.4f} {pz:.4f}"', 1))
(destino / 'scene.xml').write_text(texto_cena.replace(inc.group(1), str(destino / 'robot.xml'), 1))

m = mujoco.MjModel.from_xml_path(str(destino / 'scene.xml')); d = mujoco.MjData(m)
ck = np.load(fonte / 'continuation.npz')
for n in ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'bvh_aabb']:
    if n in ck.files and getattr(m, n).shape == ck[n].shape:
        getattr(m, n)[:] = ck[n]
mujoco.mj_setConst(m, mujoco.MjData(m))
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)

robo_corpos = ('left_', 'right_', 'torso', 'waist', 'pelvis', 'head')
colisoes = []
for c in d.contact[:d.ncon]:
    bn = [m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
    if any(n.startswith(robo_corpos) for n in bn) and not all(n.startswith(robo_corpos) for n in bn):
        colisoes.append({'corpos': bn, 'profundidade_mm': round(-c.dist * 1000, 2)})
mesa = m.geom('tampo').id
folga_mesa = min(mujoco.mj_geomDistance(m, d, g, mesa, .5, None) for g in range(m.ngeom)
                 if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(robo_corpos))

rep['scene'] = str(destino / 'scene.xml')
rep['passo'] = {'dx_m': dx, 'dy_m': dy, 'pelve_antes_m': [px, py, pz],
                'pelve_depois_m': [px+dx, py+dy, pz],
                'abstracao': 'translacao rigida do robo; locomocao nao simulada',
                'colisoes_depois_do_passo': colisoes, 'folga_robo_mesa_mm': round(folga_mesa*1000, 1)}
(destino / 'report.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep['passo'], indent=1, ensure_ascii=False))
