"""Cadeia completa do cafe, do estado inicial ate a chaleira de volta na base.

Um processo MuJoCo por vez: esta maquina tem 14 GB e cada planejador come 1,6 GB.
Onde a busca e estocastica (pega da alca) o script tenta os candidatos aprovados
em ordem ate um deles sobreviver a conexao e ao levantamento, em vez de apostar
num indice fixo que muda a cada regeracao da cena.
"""
import json, subprocess, sys, time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PY_BIN = str(RAIZ.parent / 'g1-cup-grasp/.venv/bin/python')
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / 'results-claude/cadeia-completa'
INICIAL = 'results/spoon-release-v2-001'
# Reaproveita uma fervura ja validada quando passada como segundo argumento: a
# prensa mais o tempo de fervura custam 4 minutos e nao mudam entre tentativas
# da pega.
FERVURA_PRONTA = sys.argv[2] if len(sys.argv) > 2 else None


def rodar(nome, script, *args):
    destino = SAIDA / nome
    cmd = [PY_BIN, f'scripts/{script}.py', *map(str, args), '--out', str(destino)]
    inicio = time.monotonic()
    with (SAIDA / f'{nome}.log').open('w') as log:
        code = subprocess.run(cmd, cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT).returncode
    rep = destino / 'report.json'
    dados = json.loads(rep.read_text()) if rep.exists() else {'pass': False,
                                                              'failure': f'processo saiu com {code}'}
    print(f'  {nome:24} pass={str(dados.get("pass")):5} '
          f'{time.monotonic()-inicio:6.1f}s  {dados.get("failure") or ""}', flush=True)
    return destino, dados


def ordenar_por_esforco(fonte, pasta, rep):
    """Candidatos aprovados do menor ao maior esforco ponderado a partir da pose atual."""
    import numpy as np, mujoco
    sys.path.insert(0, str(RAIZ / 'scripts'))
    from postura_humana import pesos
    fr = json.loads((RAIZ / fonte / 'report.json').read_text())
    m = mujoco.MjModel.from_xml_path(fr['scene']); d = mujoco.MjData(m)
    ck = np.load(RAIZ / fonte / 'continuation.npz')
    mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
    nomes = rep['joint_names']; qa = [m.jnt_qposadr[m.joint(n).id] for n in nomes]
    Q = np.load(RAIZ / pasta / 'candidates.npz')['qpos']; W = pesos(nomes)
    ok = [i for i, x in enumerate(rep.get('results', [])) if x.get('pass')]
    return sorted(ok, key=lambda i: float(np.linalg.norm((Q[i][qa] - d.qpos[qa]) * W)))


SAIDA.mkdir(parents=True, exist_ok=True)
etapas = {}

if FERVURA_PRONTA:
    fervido = Path(FERVURA_PRONTA)
    etapas['fervura'] = str(fervido)
    print(f'1-3. fervura reaproveitada de {fervido}', flush=True)
else:
  print('0. baixar os dois bracos e centralizar o tronco depois de largar a colher', flush=True)
  relaxado, rr = rodar('00-relax-arm', 'relaxar_braco', '--source', INICIAL,
                       '--side', 'both', '--waist-to', 0, 0, 0)
  assert rr.get('pass'), 'bracos nao desceram'
  INICIAL = str(relaxado)
  # Prensa com postura: antes a cintura ia a 30 graus de giro e 25 de flexao e o
  # umero girava 107 graus na ida. Agora cintura limitada (6 graus, 0,35 rad),
  # orientacao do dedo livre (so' apontar para baixo), cotovelo junto ao corpo,
  # braco ocioso pendurado e protegido da mesa.
  CINTURA = ['--waist-rp-deg', 6, '--waist-yaw-rad', 0.35]
  print('1. alcancar o botao', flush=True)
  alcance, ra = rodar('01-heater-reach', 'plan_heater_reach_v2', '--source', INICIAL, *CINTURA,
                      '--grid', '30,0 40,0 50,0 60,0 70,0 80,0 90,0 60,30 60,-30 45,30 45,-30 75,30 75,-30 90,20 90,-20',
                      '--free-orientation', '--human-weights', '--fix-idle-arm', '--elbow-lateral-m', 0.10)
  ordem = ordenar_por_esforco(INICIAL, alcance, ra)
  print(f'   candidatos por esforco: {ordem}', flush=True)
  print('2. conectar ao botao', flush=True)
  conexao_botao = None
  for i in ordem:
      destino, rb = rodar(f'02-heater-connect-c{i}', 'plan_heater_connection_v2', '--source', INICIAL,
                          '--candidate', alcance, '--direct-start', *CINTURA, '--approach-distance', 0.04,
                          '--choice', i, '--human-weights', '--fix-idle-arm')
      if rb.get('pass'):
          conexao_botao = destino
          break
  assert conexao_botao is not None, 'conexao ao botao falhou'

  print('3. apertar o botao', flush=True)
  prensado, rf = rodar('03-heater-press', 'execute_heater_press_v2', '--source', INICIAL,
                       '--plan', conexao_botao, *CINTURA,
                       '--press-depth', 0.03, '--press-offset', 0, -0.006, 0,
                       '--human-weights', '--fix-idle-arm')
  assert rf.get('pass'), 'prensa falhou'
  # Antes a prensa esperava a fervura com as juntas congeladas: tronco torcido e
  # braco esquerdo erguido em cima da chaleira por quase 5 minutos. Agora o
  # braco desce, o tronco volta ao centro e a agua ferve nessa postura.
  print('3b. baixar o braco, endireitar o tronco e esperar ferver', flush=True)
  fervido, rd = rodar('03b-rest-until-boil', 'relaxar_braco', '--source', prensado,
                      '--side', 'both', '--waist-to', 0, 0, 0, '--until-boil')
  assert rd.get('pass'), 'fervura falhou'
  etapas['fervura'] = str(fervido)

print('4. planejar a pega da alca', flush=True)
pega, rp = rodar('04-kettle-grasp', 'plan_kettle_grasp_v3', '--source', fervido,
                 '--seed', 'results-claude/pega-referencia-validada', '--fixed-hand',
                 '--human-weights', '--elbow-comfort-deg', 50, '--fix-idle-arm',
                 '--hand-side', 'left', '--hot-clearance', 0.0088, '--waist-rp-deg', 12,
                 '--seeds', 6, '--open-check-factor', 0)
# Ordem por folga termica decrescente, nao por indice: o fechamento da mao
# consome de 2,8 a 5,7 mm de folga ate o polegar, entao o candidato com mais
# margem estatica e o que tem mais chance de sobreviver ao portao de 6 mm do
# executor.
aprovados = sorted((i for i, r in enumerate(rp.get('results', [])) if r.get('pass')),
                   key=lambda i: -rp['results'][i]['hot_clearance_m'])
print('   candidatos aprovados (folga fechada, mm): '
      + ', '.join(f'{i}:{rp["results"][i]["hot_clearance_m"]*1000:.1f}' for i in aprovados), flush=True)
assert aprovados, 'nenhuma pega valida'

levantado = None
for i in aprovados:
    print(f'5. candidato {i}: conexao e levantamento', flush=True)
    conexao, rc = rodar(f'05-kettle-connect-c{i}', 'plan_kettle_connection_v2', '--source', fervido,
                        '--candidate', pega, '--choice', i, '--direct-start', '--hand-side', 'left',
                        '--hot-clearance', 0.0065, '--waist-rp-deg', 12, '--open-factor', 0,
                        '--escape', 0, 0, 1, '--human-weights', '--fix-idle-arm')
    if not rc.get('pass'):
        continue
    destino, rl = rodar(f'06-kettle-lift-c{i}', 'execute_kettle_lift_v2', '--source', fervido,
                        '--plan', conexao, '--hand-side', 'left', '--waist-rp-deg', 12, '--human-weights', '--fix-idle-arm',
                        '--wrench-feedforward', '--freeze-grip-on-lift', '--object-feedback',
                        '--contact-frame', '--normal-force', 8, '--contact-target', 10)
    if rl.get('pass'):
        levantado, etapas['levantamento'] = destino, str(destino)
        print(f'   levantou {rl["max_lift_m"]*1000:.1f} mm com o candidato {i}', flush=True)
        break
assert levantado is not None, 'nenhum candidato sobreviveu ao levantamento'

print('7. planejar o despejo', flush=True)
plano_despejo, rd = rodar('07-pour-plan', 'plan_kettle_pour_v2', '--source', levantado, '--fix-idle-arm')
assert rd.get('pass'), 'planejamento do despejo falhou'

print('8. despejar', flush=True)
despejado, rj = rodar('08-pour', 'execute_coffee_pour_v2', '--source', levantado,
                      '--plan', plano_despejo, '--require-hot', '--duration', 20,
                      '--max-seconds', 320, '--return-offset-x', 0, '--return-lift', 0.03,
                      '--return-rate', 0.4, '--fix-idle-arm')
assert rj.get('pass'), 'despejo falhou'
etapas['despejo'] = str(despejado)

print('9. devolver a chaleira', flush=True)
final, rv = rodar('09-place', 'execute_coffee_place_v2', '--source', despejado,
                  '--freeze-torso-release', '--align-seconds', 2, '--lower-seconds', 4, '--fix-idle-arm')
assert rv.get('pass'), 'devolucao falhou'
etapas['final'] = str(final)

agua = rv.get('water_state', {})
resumo = {'inicial': INICIAL, 'etapas': etapas,
          'na_xicara_ml': agua.get('receiver_ml'), 'derramado_ml': agua.get('spilled_ml'),
          'retido_no_pano_ml': agua.get('cloth_retained_ml'),
          'restou_na_chaleira_ml': agua.get('source_ml'),
          'po_g': agua.get('coffee_g'),
          'subida_mm': rl['max_lift_m'] * 1000,
          'candidato_de_pega': i}
(SAIDA / 'resumo.json').write_text(json.dumps(resumo, indent=2))
print(json.dumps(resumo, indent=2), flush=True)
