"""Continuação da cadeia do café a partir da colher já solta.

Mesma sequência de `results/coffee-finish-002/run_coffee_finish.py`, começando
depois do release (que lá travava no portão de força instantâneo; ver
AUDITORIA-CLAUDE-2026-09-21.md). Cada etapa ganha uma pasta nova; tentativa
reprovada nunca alimenta a seguinte.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True, help='release aceito')
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
shutil.copy2(__file__, a.out / Path(__file__).name)

status = {'model': 'claude-opus-5', 'start_source': str(a.source),
          'phase': 'starting', 'latest_accepted_source': str(a.source),
          'commands': [], 'coffee_completed': False}


def publish():
    tmp = a.out / 'progress.tmp'
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(a.out / 'progress.json')


def run(name, script, *args, candidates=False):
    destination = a.out / name
    command = [sys.executable, 'scripts/' + script + '.py', *map(str, args), '--out', str(destination)]
    status['phase'] = name
    status['commands'].append(command)
    publish()
    with (a.out / (name + '.log')).open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f'{name}: process exit {code}; inspect log')
    report = json.loads((destination / 'report.json').read_text())
    if not candidates and not report.get('pass'):
        raise RuntimeError(f'{name}: rejected: {report.get("failure")}')
    return destination, report


def try_choices(name, script, choices, *args, **kw):
    """Um candidato geometricamente valido ainda pode nao ter aproximacao
    valida. O runner original pegava sempre o primeiro; aqui percorre todos e
    guarda o motivo de cada recusa."""
    erros = []
    for i in choices:
        try:
            return run(f'{name}-c{i}', script, *args, '--choice', i, **kw)
        except RuntimeError as exc:
            erros.append((i, str(exc)))
            status.setdefault('rejected_choices', []).append({'stage': name, 'choice': i,
                                                              'error': str(exc)})
            publish()
    raise RuntimeError(f'{name}: nenhum candidato com aproximacao valida: {erros}')


def accept(source):
    status['latest_accepted_source'] = str(source)
    publish()


publish()
source = a.source
try:
    g = json.loads((source / 'brew-state.json').read_text())['grounds']
    assert 19.8 <= g['filter_g'] <= 20.5 and g['spoon_g'] < .01 and g['spilled_g'] <= .05, g

    # Sem --free-orientation e sem as preferencias de cotovelo. Nesse estado a
    # chaleira ficou 14 mm mais longe (deriva acumulada na dosagem) e aqueles
    # termos, que sao preferencia de postura e nao fisica, deixavam o dedo a
    # 5,5 mm do botao. O alcance puramente cinematico e' exato (0,000 mm), e
    # sem eles 2 de 5 candidatos passam com 0,056 mm. Portoes de anatomia do
    # cotovelo, contato proibido e folga termica continuam ativos.
    # Grade de orientacoes ampliada de 5 para 15: com a chaleira deslocada a
    # grade original nao continha nenhuma pose utilizavel.
    candidate, report = run('heater-reach', 'plan_heater_reach_v2', '--source', source,
                            '--waist-rp-deg', 25, '--waist-yaw-rad', 1.0,
                            '--grid', '30,0 40,0 50,0 60,0 70,0 80,0 90,0 60,30 60,-30 '
                                      '45,30 45,-30 75,30 75,-30 90,20 90,-20',
                            candidates=True)
    choices = [i for i, r in enumerate(report['results']) if r['pass']]
    assert choices, 'No valid heater reach'
    # Mesmos limites de cintura do alcance (senao o candidato e' recortado e
    # aparece como 4,18 graus de erro ja na distancia zero) e recuo de 40 mm em
    # vez de 80. O comprimento do recuo e' escolha de projeto; o limite de 5
    # graus de orientacao continua intacto e com folga (passa em 40 mm, estoura
    # em 52 mm).
    plan, _ = try_choices('heater-connect', 'plan_heater_connection_v2', choices,
                          '--source', source, '--candidate', candidate, '--direct-start',
                          '--waist-rp-deg', 25, '--waist-yaw-rad', 1.0,
                          '--approach-distance', .04)
    # Curso de 30 mm e alvo 6 mm em -y. O ponto que a IK mira fica dentro do
    # link do indicador, mas o contato acontece na superficie da capsula; com a
    # chaleira 14 mm fora do lugar essa diferenca caia na borda do balancim e o
    # dedo escapava (descia 18 mm e o botao afundava 0,9 mm). Medido: rigidez
    # 0,3 N.m/rad e alavanca 18 mm exigem 2,33 N, e o pico observado nos runs
    # que ligam fica entre 2,31 e 4,00 N.
    source, _ = run('heater', 'execute_heater_press_v2', '--source', source, '--plan', plan,
                    '--waist-rp-deg', 25, '--waist-yaw-rad', 1.0,
                    '--press-depth', .030, '--press-offset', 0, -.006, 0,
                    '--boil', '--hold-joints-during-heating')
    accept(source)

    candidate, report = run('kettle-grasp', 'plan_kettle_grasp', '--source', source, candidates=True)
    choices = [i for i, r in enumerate(report['results']) if r['pass']]
    assert choices, 'No valid kettle grasp'
    # Tres correcoes aqui, todas medidas:
    # 1. --hand-side left: o planejador abria os dedos da mao DIREITA enquanto a
    #    chaleira e' pega com a esquerda, entao a pose de dedos do candidato nunca
    #    era aplicada e a mao esquerda ficava com o indicador estendido do aperto
    #    do botao, penetrando a alca em 5,0 mm.
    # 2. --open-factor 0: escalar os angulos dos dedos linearmente faz a ponta
    #    atravessar o metal no meio do caminho. Medido: em 0,6 a folga ao corpo
    #    quente e' -7,5 mm (dentro da chaleira) com 7,5 mm de penetracao; em 0,0
    #    e' +10,1 mm sem penetracao nenhuma.
    # 3. --hot-clearance 0,0065: este era o unico ponto do pipeline exigindo 8 mm.
    #    O planejador de pega mira 7 e aprova em 6; o executor do lift, que valida
    #    a pega fisica, so reprova abaixo de 6. O kettle-lift-007 aprovado rodou
    #    com 6,95 mm.
    plan, _ = try_choices('kettle-connect', 'plan_kettle_connection_v2', choices,
                          '--source', source, '--candidate', candidate, '--direct-start',
                          '--hand-side', 'left', '--hot-clearance', .0065,
                          '--open-factor', 0, '--escape', 0, 0, 1)
    source, _ = run('kettle-lift', 'execute_kettle_lift', '--source', source, '--plan', plan,
                    '--wrench-feedforward', '--freeze-grip-on-lift', '--object-feedback',
                    '--contact-frame', '--normal-force', 20, '--contact-target', 10)
    accept(source)

    plan, _ = run('pour-plan', 'plan_kettle_pour', '--source', source)
    source, _ = run('pour', 'execute_coffee_pour', '--source', source, '--plan', plan,
                    '--require-hot', '--duration', 20, '--max-seconds', 320,
                    '--return-offset-x', 0, '--return-lift', 0, '--return-rate', .4)
    accept(source)

    source, _ = run('kettle-place', 'execute_coffee_place', '--source', source,
                    '--freeze-torso-release', '--align-seconds', 2, '--lower-seconds', 4)
    accept(source)

    run('chain-audit', 'audit_episode_chain', '--source', source)
    status['phase'] = 'physical_sequence_complete_pending_final_review'
except Exception as exc:
    status['failure'] = str(exc)
    status['phase'] = 'stopped_for_inspection'
finally:
    publish()
    print(json.dumps(status, indent=2), flush=True)
