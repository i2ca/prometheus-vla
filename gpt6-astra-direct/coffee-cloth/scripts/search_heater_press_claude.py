"""Procura, entre os candidatos de alcance aprovados, um que realmente ligue a
chaleira.

O balancim tem rigidez 0,3 N.m/rad e o ponto de contato fica a 18 mm do eixo,
logo ligar (0,14 rad) exige cerca de 2,33 N no dedo. A prensa entregava 0,9 N
com a orientacao original, o que da' 0,054 rad e bate com o medido. A hipotese
aqui e' que a direcao de empurrar importa: so a componente normal a face do
balancim gera torque.
"""
import json
import subprocess
import sys
from pathlib import Path

REACH = Path('results/hr-claude-grid')
SOURCE = Path('results/spoon-release-v2-001')
OUT = Path('results/heater-search-claude-001')
PY = sys.executable

OUT.mkdir(exist_ok=True)
candidates = json.loads((REACH / 'report.json').read_text())['results']
aprovados = [i for i, r in enumerate(candidates) if r['pass']]
resumo = []

for i in aprovados:
    row = {'choice': i, 'pitch': candidates[i]['pitch_deg'], 'yaw': candidates[i]['yaw_deg']}
    conn = OUT / f'connect-{i}'
    code = subprocess.run(
        [PY, 'scripts/plan_heater_connection_v2.py', '--source', str(SOURCE),
         '--candidate', str(REACH), '--choice', str(i), '--direct-start',
         '--waist-rp-deg', '25', '--waist-yaw-rad', '1.0',
         '--approach-distance', '0.04', '--out', str(conn)],
        stdout=(OUT / f'connect-{i}.log').open('w'), stderr=subprocess.STDOUT).returncode
    ok = False
    if code == 0 and (conn / 'report.json').exists():
        ok = json.loads((conn / 'report.json').read_text()).get('pass', False)
    row['connect'] = ok
    if not ok:
        resumo.append(row)
        (OUT / 'summary.json').write_text(json.dumps(resumo, indent=2))
        print(row, flush=True)
        continue

    press = OUT / f'press-{i}'
    subprocess.run(
        [PY, 'scripts/execute_heater_press_v2.py', '--source', str(SOURCE),
         '--plan', str(conn), '--waist-rp-deg', '25', '--waist-yaw-rad', '1.0',
         '--press-depth', '0.030', '--press-trim', '0.009',
         '--boil', '--hold-joints-during-heating', '--out', str(press)],
        stdout=(OUT / f'press-{i}.log').open('w'), stderr=subprocess.STDOUT)
    if (press / 'report.json').exists():
        rep = json.loads((press / 'report.json').read_text())
        row['press_pass'] = rep.get('pass')
        row['failure'] = rep.get('failure')
        amostras = rep.get('samples') or rep.get('rows') or []
        angs = [s.get('switch') for s in amostras if isinstance(s, dict) and s.get('switch') is not None]
        forcas = [s.get('force') for s in amostras if isinstance(s, dict) and s.get('force') is not None]
        row['switch_max_rad'] = max(angs) if angs else None
        row['force_max_N'] = max(forcas) if forcas else None
    resumo.append(row)
    (OUT / 'summary.json').write_text(json.dumps(resumo, indent=2))
    print(row, flush=True)
    if row.get('press_pass'):
        print('LIGOU com o candidato', i, flush=True)
        break

(OUT / 'summary.json').write_text(json.dumps(resumo, indent=2))
print(json.dumps(resumo, indent=2), flush=True)
