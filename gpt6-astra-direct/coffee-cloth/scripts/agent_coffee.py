"""Orquestrador agêntico do café: o modelo escolhe as primitivas.

Mesma ideia do IsaacSim_GeminiRobotics (modelo com visão decidindo por function
calling qual primitiva chamar), aplicada ao G1 com Dex3 no MuJoCo. A diferença
é que lá as ações iam por tópico ROS 2 para um FSM em tempo real; aqui cada
ferramenta é um script de `scripts/` que roda a etapa inteira e devolve
`report.json`, e o estado passa adiante por `continuation.npz`.

Honestidade sobre o escopo: as primitivas leem a pose dos objetos direto do
MuJoCo, como o repo original lê de TF. O modelo escolhe *qual* etapa e *quando*,
e vê a cena pela câmera da cabeça, mas não localiza nada por pixel.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types as gt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import licoes

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT.parent / 'g1-cup-grasp/.venv/bin/python')

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True, help='estado aceito inicial')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--model', default='agy/gemini-3.7-flash-high')
ap.add_argument('--base-url', default='http://127.0.0.1:20129')
ap.add_argument('--goal', default='Prepare o café: ferva a água, pegue a chaleira pela alça, '
                                 'despeje sobre o coador de pano e devolva a chaleira à base.')
ap.add_argument('--max-turns', type=int, default=12)
ap.add_argument('--max-minutes', type=float, default=60.)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)

estado = {'model': a.model, 'goal': a.goal, 'source_inicial': str(a.source),
          'fonte_atual': str(a.source), 'turnos': [], 'coffee_completed': False}


def publicar():
    tmp = a.out / 'progress.tmp'
    tmp.write_text(json.dumps(estado, indent=2, ensure_ascii=False))
    tmp.replace(a.out / 'progress.json')


_ultimo_run = {'dir': ''}


def rodar(nome, script, *args, aceita_falha=False):
    """Roda uma etapa e devolve (pasta, report)."""
    destino = a.out / nome
    # a licao precisa apontar para os arquivos que produziram os numeros, senao
    # nao da' para auditar depois de qual corrida ela saiu.
    _ultimo_run['dir'] = str(destino)
    cmd = [PY, f'scripts/{script}.py', *map(str, args), '--out', str(destino)]
    with (a.out / f'{nome}.log').open('w') as log:
        code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    rep_path = destino / 'report.json'
    if code or not rep_path.exists():
        return destino, {'pass': False, 'failure': {'reason': f'processo saiu com {code}'}}
    rep = json.loads(rep_path.read_text())
    if not aceita_falha and not rep.get('pass'):
        return destino, rep
    return destino, rep


# Ferramentas

def ferramenta_observar(_args):
    """Três câmeras, o mesmo conjunto instalado no robô real: cabeça e dois
    punhos. Para a pega da alça a de punho importa, porque vista do tronco o
    próprio braço oclui a alça."""
    destino, _ = rodar(f'obs-{len(estado["turnos"])}', 'observe_state',
                       '--source', estado['fonte_atual'], aceita_falha=True)
    obs = json.loads((destino / 'observation.json').read_text())
    visivel = {k: v for k, v in obs.items()
               if k not in ('posicoes_objetos', 'source', 'cameras', 'arquivos')}
    imagens = []
    for arquivo in obs.get('arquivos', ['view.png']):
        caminho = destino / arquivo
        if caminho.exists():
            imagens.append((arquivo.replace('.png', ''), caminho.read_bytes()))
    return visivel, imagens


def ferramenta_ligar_chaleira(args):
    """Parâmetros abertos: o modelo escolhe curso, deslocamento do alvo,
    liberdade de cintura e comprimento do recuo."""
    fonte = estado['fonte_atual']
    n = len(estado['turnos'])
    curso = float(args.get('curso_mm', 30)) / 1000
    desloc = float(args.get('desloc_y_mm', -6)) / 1000
    cintura = float(args.get('cintura_deg', 6))  # validado: prensa com cintura <= 6 graus
    recuo = float(args.get('recuo_mm', 40)) / 1000
    _todos_h = {'curso_mm': curso * 1000, 'desloc_y_mm': desloc * 1000,
                'cintura_deg': cintura, 'recuo_mm': recuo * 1000}
    # antes de alcancar o botao: bracos caidos e tronco no centro, que e' de onde
    # uma pessoa parte. Depois a prensa com postura (ver run_cadeia_completa.py).
    fonte, rr = rodar(f'heater-{n}-relax', 'relaxar_braco', '--source', fonte, '--side', 'both',
                      '--waist-to', 0, 0, 0, aceita_falha=True)
    if not rr.get('pass'):
        return {'sucesso': False, 'etapa': 'alcancar_botao', 'motivo': 'bracos nao desceram: ' + str(rr.get('failure')),
                'parametros_usados': _todos_h}, None
    fonte = str(fonte)
    cand, rep = rodar(f'heater-reach-{n}', 'plan_heater_reach_v2', '--source', fonte,
                      '--waist-rp-deg', cintura, '--waist-yaw-rad', 0.35,
                      '--free-orientation', '--human-weights', '--fix-idle-arm', '--elbow-lateral-m', 0.10,
                      '--grid', '30,0 40,0 50,0 60,0 70,0 80,0 90,0 60,30 60,-30 '
                                '45,30 45,-30 75,30 75,-30 90,20 90,-20',
                      aceita_falha=True)
    escolhas = [i for i, r in enumerate(rep.get('results', [])) if r.get('pass')]
    if not escolhas:
        menor = min((r.get('tip_error_m', 9) for r in rep.get('results', [])), default=9)
        return {'sucesso': False, 'etapa': 'alcancar_botao',
                'motivo': 'nenhuma pose de alcance valida',
                'medido': {'menor_erro_de_ponta_mm': round(menor * 1000, 3)},
                'parametros_usados': _todos_h}, None
    for i in escolhas:
        plano, rp = rodar(f'heater-connect-{n}-c{i}', 'plan_heater_connection_v2',
                          '--source', fonte, '--candidate', cand, '--choice', i,
                          '--direct-start', '--waist-rp-deg', cintura, '--waist-yaw-rad', 0.35,
                          '--approach-distance', recuo, '--human-weights', '--fix-idle-arm', aceita_falha=True)
        if rp.get('pass'):
            break
    else:
        ult = (rp.get('approach') or [{}])[-1]
        return {'sucesso': False, 'etapa': 'conectar_ao_botao',
                'motivo': 'nenhum candidato com aproximacao valida',
                'medido': {'ultimo_passo': ult},
                'parametros_usados': _todos_h}, None
    novo, rp = rodar(f'heater-{n}', 'execute_heater_press_v2', '--source', fonte, '--plan', plano,
                     '--waist-rp-deg', cintura, '--waist-yaw-rad', 0.35,
                     '--press-depth', curso, '--press-offset', 0, desloc, 0,
                     '--human-weights', '--fix-idle-arm', aceita_falha=True)
    if not rp.get('pass'):
        return {'sucesso': False, 'etapa': 'prensar',
                'motivo': str(rp.get('failure')),
                'medido': {'pico_de_forca_N': rp.get('peak_switch_force_N')},
                'parametros_usados': _todos_h}, None
    pico = rp.get('peak_switch_force_N', 0)
    # A prensa esperava a fervura com as juntas congeladas: tronco torcido e braco
    # erguido sobre a chaleira por ~5 min. Agora os bracos descem, o tronco
    # endireita e a agua ferve nessa postura.
    novo, rp = rodar(f'heater-{n}-descanso', 'relaxar_braco', '--source', novo, '--side', 'both',
                     '--waist-to', 0, 0, 0, '--until-boil', aceita_falha=True)
    if not rp.get('pass'):
        return {'sucesso': False, 'etapa': 'prensar', 'motivo': str(rp.get('failure')),
                'medido': {'pico_de_forca_N': pico}, 'parametros_usados': _todos_h}, None
    rp['peak_switch_force_N'] = pico
    estado['fonte_atual'] = str(novo)
    h = rp.get('heater', {})
    return {'sucesso': True, 'temperatura_C': round(h.get('temperature_C', 0), 3),
            'evento': h.get('last_event'),
            'pico_de_forca_N': round(rp.get('peak_switch_force_N', 0), 3)}, None


def ferramenta_pegar_chaleira(args):
    """Parâmetros abertos, incluindo qual mão pega a alça.

    O relatório de falha volta com os números medidos (folga ao metal quente,
    cossenos de oposição dos dedos), que é o que permite ao modelo procurar."""
    fonte = estado['fonte_atual']
    n = len(estado['turnos'])
    mao = 'right' if str(args.get('mao', 'esquerda')).startswith('dir') else 'left'
    folga = float(args.get('folga_alvo_mm', 7)) / 1000
    sementes = int(args.get('sementes', 12))
    oposicao = float(args.get('peso_oposicao', 0))
    cintura = float(args.get('cintura_deg', 12))  # validado: pega e levantamento com 12 graus
    abertura = float(args.get('abertura', 0))
    forca = float(args.get('forca_fechamento_N', 20))
    folga_con = float(args.get('folga_conexao_mm', 6.5)) / 1000
    orientacao = float(args.get('peso_orientacao', 8 if mao == 'right' else 30))
    # Todos os parâmetros da chamada, não só os da etapa que falhou. Antes cada
    # falha reportava só os seus, e ao creditar uma etapa anterior a lição saía
    # com os parâmetros errados: a pega aparecia "resolvida" com os parâmetros
    # do levantamento.
    _todos = {'mao': args.get('mao', 'esquerda'), 'folga_alvo_mm': folga * 1000,
              'sementes': sementes, 'peso_oposicao': oposicao, 'cintura_deg': cintura,
              'abertura': abertura, 'forca_fechamento_N': forca,
              'folga_conexao_mm': folga_con * 1000, 'peso_orientacao': orientacao}

    # v3 em vez de _right: o _right so mede a folga ao metal quente com a mao
    # FECHADA, e quem viaja ate a pega e' a mao aberta. Ele aprovava candidatos
    # inalcancaveis e a falha aparecia dois scripts depois, como
    # 'reverse approach invalid' na distancia zero.
    cand, rep = rodar(f'kettle-grasp-{n}', 'plan_kettle_grasp_v3', '--source', fonte,
                      '--hand-side', mao, '--hot-clearance', folga, '--waist-rp-deg', cintura,
                      '--open-check-factor', abertura,
                      # pega validada guardada no referencial da chaleira: so o
                      # braco e' resolvido, os dedos ficam na pose que levantou,
                      # despejou e devolveu. Taxa de ~2% para 50-100% com a
                      # chaleira deslocada 5 cm.
                      '--seed', 'results-claude/pega-referencia-validada', '--fixed-hand',
                      # postura: cotovelo dobrado a 50 graus e IK que prefere
                      # ombro e cotovelo a girar o umero e torcer o punho. Sem
                      # isto o cotovelo ficava parado no limite da junta e a mao
                      # andava so por rotacao do umero.
                      '--human-weights', '--elbow-comfort-deg', 50, '--fix-idle-arm',
                      '--seeds', sementes, '--oppose-weight', oposicao,
                      '--goal-rot-weight', orientacao, aceita_falha=True)
    resultados = rep.get('results', [])
    escolhas = [i for i, r in enumerate(resultados) if r.get('pass')]
    if not escolhas:
        # Reportar o candidato de maior folga era enganoso: o de maior folga e'
        # justamente a mao que fugiu da alca, com as pontas no teto da busca e
        # normais degeneradas. Aqui o criterio e' "mais perto de passar": as
        # tres pontas encostando e a pior oposicao a menor possivel.
        def encostando(r):
            return sum(1 for g in r.get('tip_gaps_m', []) if g < .005)

        tocaram = [r for r in resultados if encostando(r) == 3]
        candidatos = tocaram or resultados
        melhor = min(candidatos,
                     key=lambda r: (-encostando(r), max(r.get('opposition_cosines', [9]))),
                     default={})
        return {'sucesso': False, 'etapa': 'planejar_pega',
                'motivo': 'nenhuma pega valida na alca',
                'medido': {
                    'candidatos': len(resultados),
                    'com_os_tres_dedos_na_alca': len(tocaram),
                    'melhor_candidato': {
                        'dedos_encostando': encostando(melhor),
                        'folga_das_pontas_mm': [round(v * 1000, 2) for v in melhor.get('tip_gaps_m', [])],
                        'oposicao_dos_dedos': [round(v, 3) for v in melhor.get('opposition_cosines', [])],
                        'folga_ao_metal_quente_mm': round(melhor.get('hot_clearance_m', 0) * 1000, 2)},
                    'portao_oposicao': 'os dois cossenos precisam ficar abaixo de -0,3; '
                                       'o primeiro e polegar contra indicador, o segundo polegar contra medio',
                    'portao_das_pontas': 'as tres pontas precisam ficar entre 0 e 1,8 mm da alca ao mesmo tempo',
                    'portao_folga_quente': 'a folga ao metal quente precisa ficar acima da folga alvo pedida',
                    'leitura': 'ponta em 15 mm significa que o dedo nem chegou na alca; '
                               'oposicao positiva com dedos longe nao quer dizer nada'},
                'parametros_usados': _todos}, None
    for i in escolhas:
        plano, rp = rodar(f'kettle-connect-{n}-c{i}', 'plan_kettle_connection_v2',
                          '--source', fonte, '--candidate', cand, '--choice', i,
                          '--direct-start', '--hand-side', mao, '--hot-clearance', folga_con,
                          '--waist-rp-deg', cintura, '--human-weights', '--fix-idle-arm',
                          '--open-factor', abertura, '--escape', 0, 0, 1, aceita_falha=True)
        if rp.get('pass'):
            break
    else:
        return {'sucesso': False, 'etapa': 'aproximacao',
                'motivo': 'nenhum candidato com aproximacao valida',
                'medido': {'ultimo_passo': (rp.get('approach') or [{}])[-1]},
                'parametros_usados': _todos}, None
    novo, rp = rodar(f'kettle-lift-{n}', 'execute_kettle_lift_v2', '--source', fonte, '--plan', plano,
                     '--hand-side', mao, '--waist-rp-deg', cintura, '--human-weights', '--fix-idle-arm',
                     '--wrench-feedforward', '--freeze-grip-on-lift',
                     '--object-feedback', '--contact-frame', '--normal-force', forca,
                     '--contact-target', 10, aceita_falha=True)
    if not rp.get('pass'):
        aud = rp.get('clearance_audit', {})
        return {'sucesso': False, 'etapa': 'levantar',
                'motivo': str(rp.get('failure')),
                'medido': {'folga_minima_ao_metal_quente_mm': round(aud.get('hot_min_m', 0) * 1000, 4),
                           'portao_mm': 6.0,
                           'passos_abaixo_do_portao': aud.get('steps_under_gate')},
                'parametros_usados': _todos}, None
    estado['fonte_atual'] = str(novo)
    estado['mao_da_chaleira'] = mao
    return {'sucesso': True, 'subida_m': rp.get('lift_m'),
            'folga_minima_mm': round(rp.get('min_hot_clearance_m', 0) * 1000, 3)}, None


def ferramenta_servir(_args):
    fonte = estado['fonte_atual']
    plano, rp = rodar('pour-plan', 'plan_kettle_pour_v2', '--source', fonte, '--fix-idle-arm', aceita_falha=True)
    if not rp.get('pass'):
        return {'sucesso': False, 'motivo': str(rp.get('failure'))}, None
    # v2: a v1 tem hot_gap<.006 e max_penetration>.0005 fixos no codigo. O
    # levantamento validado segura a chaleira com 0,58 mm de penetracao, entao o
    # despejo reprovava em t=0 sem sequer inclinar a chaleira.
    novo, rp = rodar('pour', 'execute_coffee_pour_v2', '--source', fonte, '--plan', plano,
                     '--require-hot', '--duration', 20, '--max-seconds', 320,
                     # subir 3 cm ao endireitar: com 0 a chaleira girava rente ao
                     # coador e se apoiava nele quando comecava 5 cm deslocada em y
                     '--return-offset-x', 0, '--return-lift', 0.03, '--return-rate', .4, '--fix-idle-arm',
                     aceita_falha=True)
    if not rp.get('pass'):
        return {'sucesso': False, 'motivo': str(rp.get('failure'))}, None
    estado['fonte_atual'] = str(novo)
    w = rp.get('water_state', {})
    return {'sucesso': True, 'despejado_ml': round(w.get('discharged_ml', 0), 3),
            'na_xicara_ml': round(w.get('receiver_ml', 0), 3),
            'derramado_ml': round(w.get('spilled_ml', 0), 3)}, None


def ferramenta_devolver_chaleira(_args):
    fonte = estado['fonte_atual']
    novo, rp = rodar('kettle-place', 'execute_coffee_place_v2', '--source', fonte,
                     '--freeze-torso-release', '--align-seconds', 2, '--lower-seconds', 4, '--fix-idle-arm',
                     aceita_falha=True)
    if not rp.get('pass'):
        return {'sucesso': False, 'motivo': str(rp.get('failure'))}, None
    estado['fonte_atual'] = str(novo)
    return {'sucesso': True}, None


# Ordem das etapas dentro de cada ferramenta. Serve para creditar o que foi
# vencido: se a falha aconteceu em "levantar", então "planejar_pega" e
# "aproximacao" passaram, e isso precisa virar lição. Sem isto o episódio
# seguinte refaz a busca da pega do zero, que foi exatamente o que aconteceu
# entre o agent-cafe-010 e o 011.
ETAPAS = {
    'ligar_chaleira': ['alcancar_botao', 'conectar_ao_botao', 'prensar'],
    'pegar_chaleira': ['planejar_pega', 'aproximacao', 'levantar'],
    'servir_cafe': ['planejar_despejo', 'despejar'],
    'devolver_chaleira': ['apoiar'],
}


def _creditar_etapas_vencidas(nome_ferramenta, resultado, args, episodio):
    """Registra como resolvida toda etapa anterior à que falhou."""
    ordem = ETAPAS.get(nome_ferramenta, [])
    etapa_que_falhou = resultado.get('etapa')
    if etapa_que_falhou not in ordem:
        return
    vencidas = ordem[:ordem.index(etapa_que_falhou)]
    if not vencidas:
        return
    caderno = licoes._carregar()['licoes']
    for etapa in vencidas:
        abertos = [c for c in caderno
                   if c.startswith(f'{nome_ferramenta}/{etapa}/') and not caderno[c].get('solucao')]
        for chave in abertos:
            _, _, portao = chave.split('/', 2)
            licoes.registrar_solucao(
                nome_ferramenta, etapa, portao, args,
                {'nota': f'etapa vencida; a falha foi depois, em {etapa_que_falhou}',
                 'falha_seguinte': str(resultado.get('motivo'))[:200],
                 'medido_da_falha_seguinte': resultado.get('medido')},
                episodio, contexto=estado.get('estado_medido'), run=_ultimo_run['dir'])


def _registrar(nome_ferramenta, resultado, args):
    """Toda falha vira tentativa registrada; todo sucesso que vinha falhando
    vira solução. Automático, porque depender do modelo lembrar de anotar é
    justamente o que não funciona."""
    episodio = str(a.out.name)
    try:
        if resultado.get('sucesso') is False:
            chave = licoes.registrar_tentativa(
                nome_ferramenta, resultado.get('etapa'), resultado.get('motivo'),
                resultado.get('parametros_usados') or args, resultado.get('medido'),
                episodio, run=_ultimo_run['dir'])
            resultado['registrado_como'] = chave
            _creditar_etapas_vencidas(nome_ferramenta, resultado, args, episodio)
        elif resultado.get('sucesso') is True:
            # só registra solução para um problema que já existia no caderno
            abertos = [c for c in licoes._carregar()['licoes']
                       if c.startswith(nome_ferramenta + '/')
                       and not licoes._carregar()['licoes'][c].get('solucao')]
            for chave in abertos:
                _, etapa, portao = chave.split('/', 2)
                licoes.registrar_solucao(nome_ferramenta, etapa, portao, args,
                                         {k: v for k, v in resultado.items() if k != 'sucesso'},
                                         episodio, contexto=estado.get('estado_medido'),
                                         run=_ultimo_run['dir'])
            if abertos:
                resultado['licao_salva_para'] = abertos
    except Exception as exc:  # o caderno nunca pode derrubar o episódio
        print(f'   [licoes] falha ao registrar: {exc}', flush=True)
    return resultado


def ferramenta_consultar_licoes(args):
    """O modelo pergunta ao caderno antes de repetir o que já não funcionou."""
    achados = licoes.consultar(
        ferramenta=args.get('ferramenta'), etapa=args.get('etapa'),
        motivo=args.get('problema', ''))
    if not achados:
        return {'encontrado': 0, 'nota': 'nada registrado ainda para este problema'}, None
    return {'encontrado': len(achados), 'licoes': achados}, None


def ferramenta_concluir(args):
    estado['coffee_completed'] = bool(args.get('cafe_pronto'))
    estado['justificativa_final'] = args.get('justificativa', '')
    return {'registrado': True}, None


FERRAMENTAS = {
    'observar_cena': ferramenta_observar,
    'ligar_chaleira': ferramenta_ligar_chaleira,
    'pegar_chaleira': ferramenta_pegar_chaleira,
    'servir_cafe': ferramenta_servir,
    'devolver_chaleira': ferramenta_devolver_chaleira,
    'concluir': ferramenta_concluir,
    'consultar_licoes': ferramenta_consultar_licoes,
}

DECLARACOES = [
    gt.FunctionDeclaration(
        name='observar_cena',
        description='Renderiza a câmera da cabeça e devolve pó no filtro, temperatura, '
                    'se o aquecedor está ligado e o líquido em cada recipiente.',
        parameters=gt.Schema(type='OBJECT', properties={})),
    gt.FunctionDeclaration(
        name='ligar_chaleira',
        description='Alcança o botão da chaleira, pressiona e aguarda a fervura. O interruptor '
                    'só liga se o balancim girar mais de 0,14 rad com mais de 0,5 N.',
        parameters=gt.Schema(type='OBJECT', properties={
            'curso_mm': gt.Schema(type='NUMBER', description='quanto o dedo desce prensando (padrão 30)'),
            'desloc_y_mm': gt.Schema(type='NUMBER', description='desloca o alvo da prensa sobre o balancim (padrão -6)'),
            'cintura_deg': gt.Schema(type='NUMBER', description='liberdade de roll/pitch do tronco, vale para os tres planejadores da chaleira ao mesmo tempo. Medido: em 25 graus o segundo cosseno de oposicao vira +0,99 (os dois dedos empurram para o mesmo lado, nao e pega); em 12 a semente 1 passa de forma reproduzivel'),
            'recuo_mm': gt.Schema(type='NUMBER', description='comprimento do recuo planejado, limite de 5° de orientação (padrão 40)')})),
    gt.FunctionDeclaration(
        name='pegar_chaleira',
        description='Planeja uma pega na alça e levanta a chaleira. Exige água fervida. '
                    'Dois portões costumam barrar: a oposição dos dedos (os dois cossenos '
                    'precisam ficar abaixo de -0,3) e a folga ao metal quente, que reprova '
                    'abaixo de 6 mm durante a execução. O relatório de falha volta com os '
                    'dois números medidos.',
        parameters=gt.Schema(type='OBJECT', properties={
            'mao': gt.Schema(type='STRING', description="qual mão pega a alça: 'esquerda' ou 'direita'. "
                                                        'A direita está livre desde que largou a colher.'),
            'folga_alvo_mm': gt.Schema(type='NUMBER', description='folga do metal quente que o planejador persegue (padrão 7)'),
            'sementes': gt.Schema(type='NUMBER', description='quantas poses iniciais tentar (padrão 12)'),
            'peso_oposicao': gt.Schema(type='NUMBER', description='peso do termo que força os dedos a se oporem na alça; '
                                                                 '0 desliga, valores úteis entre 100 e 400'),
            'cintura_deg': gt.Schema(type='NUMBER', description='liberdade de roll/pitch do tronco (padrão 25)'),
            'abertura': gt.Schema(type='NUMBER', description='fator de abertura dos dedos na aproximação, 0 a 1. '
                                                            'Valores intermediários fazem a ponta atravessar o metal (padrão 0)'),
            'forca_fechamento_N': gt.Schema(type='NUMBER', description='força normal alvo ao fechar (padrão 20)'),
            'folga_conexao_mm': gt.Schema(type='NUMBER', description='folga mínima exigida no planejamento da aproximação (padrão 6,5)'),
            'peso_orientacao': gt.Schema(type='NUMBER', description='o quanto a palma é obrigada a manter a orientação de '
                                                                   'referência. Alto prende a pose herdada; baixo deixa o '
                                                                   'otimizador procurar outro jeito de envolver a alça. '
                                                                   'Padrão 30 na esquerda, 8 na direita, 0 solta de vez')})),
    gt.FunctionDeclaration(
        name='servir_cafe',
        description='Inclina a chaleira e despeja a água quente sobre o coador de pano, que '
                    'filtra para a xícara. Exige a chaleira na mão.',
        parameters=gt.Schema(type='OBJECT', properties={
            'duracao_s': gt.Schema(type='NUMBER', description='duração do despejo (padrão 20)')})),
    gt.FunctionDeclaration(
        name='devolver_chaleira',
        description='Apoia a chaleira de volta na base e solta.',
        parameters=gt.Schema(type='OBJECT', properties={})),
    gt.FunctionDeclaration(
        name='consultar_licoes',
        description='Consulta o caderno de lições: o que já foi tentado para um problema e, '
                    'se houver, a solução que funcionou, com os parâmetros exatos. '
                    'Use antes de repetir uma tentativa.',
        parameters=gt.Schema(type='OBJECT', properties={
            'ferramenta': gt.Schema(type='STRING', description='nome da ferramenta que falhou'),
            'etapa': gt.Schema(type='STRING', description='etapa que falhou, se souber'),
            'problema': gt.Schema(type='STRING', description='o motivo da falha, como veio no relatório')})),
    gt.FunctionDeclaration(
        name='concluir',
        description='Encerra o episódio declarando se o café ficou pronto e por quê. '
                    'Só use depois de ver líquido na xícara, ou quando não houver mais o que tentar.',
        parameters=gt.Schema(type='OBJECT', properties={
            'cafe_pronto': gt.Schema(type='BOOLEAN'),
            'justificativa': gt.Schema(type='STRING')}, required=['cafe_pronto'])),
]

SISTEMA = """Você comanda um Unitree G1 com mãos Dex3 preparando café coado em pano, em simulação física.

A cena já está montada: pó no coador de pano, a xícara embaixo dele, a chaleira sobre a base
elétrica e a colher já largada na mesa pela mão direita. **O estado real vem medido na abertura
da conversa e por `observar_cena`; não presuma temperatura nem volume.**

Cada ferramenta roda física de verdade e aceita parâmetros. Quando uma falha, o relatório volta
com `etapa` (onde morreu), `motivo` e `medido` (os números). **Use esses números para escolher
parâmetros diferentes na próxima tentativa.** Repetir a mesma chamada com os mesmos argumentos
não muda nada: a busca de pega é estocástica, mas a geometria não.

Duas coisas que ajudam a raciocinar sobre falha de pega:
- Se a oposição dos dedos não fecha, os dedos estão caindo do mesmo lado da alça. Existe um peso
  que força a oposição dentro do planejamento.
- Se a folga ao metal quente fica logo abaixo do portão, a mão está passando perto demais do
  corpo da chaleira. A geometria depende de qual mão pega e de quanta folga o planejador persegue.

`observar_cena` devolve três imagens: câmera da cabeça e as duas de punho, o mesmo conjunto
instalado no robô real. A de punho mostra a alça de perto, sem o braço ocluindo.

Você tem um caderno de lições que persiste entre episódios. As lições já
resolvidas e as tentativas que falharam chegam no começo da conversa, e
`consultar_licoes` busca por problema específico. Cada falha sua é registrada
automaticamente, e quando você resolve um problema que estava aberto, a solução
é salva com os números. Use isso: não repita parâmetro que já consta como
falho, e quando reencontrar um problema conhecido, comece pela solução que
funcionou.

Não declare café pronto sem ver líquido na xícara em `observar_cena`. Se esgotar as alternativas,
encerre explicando o que tentou. Responda sempre em português, curto, dizendo o que vai variar e
por quê."""

cliente = genai.Client(api_key='via-omniroute', http_options={'base_url': a.base_url})
# Abre o episódio com o que já se aprendeu. É isto que separa acumular de
# redescobrir: sem isto, cada run refaz a busca do zero.
_sabidas = licoes.resolvidas()
_abertura = [a.goal]

# Estado medido na abertura. Antes eu descrevia a cena no prompt ("800 ml de
# água fria") e isso era falso quando o episódio partia de um estado já
# fervido: o modelo aplicou a lição certa na condição errada e perdeu um turno
# de dez minutos. Melhor medir do que descrever.
try:
    _dest, _ = rodar('obs-inicial', 'observe_state', '--source', estado['fonte_atual'],
                     aceita_falha=True)
    _obs = json.loads((_dest / 'observation.json').read_text())
    _visivel = {k: v for k, v in _obs.items()
                if k not in ('posicoes_objetos', 'source', 'cameras', 'arquivos')}
    estado['estado_medido'] = _visivel
    _abertura.append('\n[ESTADO MEDIDO AGORA]\n' + json.dumps(_visivel, indent=1, ensure_ascii=False))
except Exception as _exc:  # medir é desejável, não obrigatório
    print(f'   [abertura] nao consegui medir o estado: {_exc}', flush=True)
if _sabidas:
    _abertura.append('\n[CADERNO DE LIÇÕES] Problemas que você já resolveu antes, '
                     'com os parâmetros que funcionaram:\n'
                     + json.dumps(_sabidas, indent=1, ensure_ascii=False))
_pendentes = [l for l in (licoes._resumir(v) for v in licoes._carregar()['licoes'].values())
              if not l['solucao']]
if _pendentes:
    _abertura.append('\n[CADERNO DE LIÇÕES] Problemas ainda abertos e o que já falhou neles. '
                     'Não repita estes parâmetros:\n'
                     + json.dumps(_pendentes, indent=1, ensure_ascii=False))
historico = [gt.Content(role='user', parts=[gt.Part.from_text(text='\n'.join(_abertura))])]
inicio = time.time()
publicar()

for turno in range(1, a.max_turns + 1):
    if (time.time() - inicio) / 60 > a.max_minutes:
        estado['parada'] = 'tempo esgotado'
        break
    resposta = cliente.models.generate_content(
        model=a.model, contents=historico,
        config=gt.GenerateContentConfig(
            system_instruction=SISTEMA, tools=[gt.Tool(function_declarations=DECLARACOES)],
            temperature=0.2, automatic_function_calling={'disable': True}))
    if not resposta.candidates:
        estado['parada'] = 'modelo nao respondeu'
        break
    partes = resposta.candidates[0].content.parts or []
    textos = [p.text for p in partes if p.text]
    chamadas = [p.function_call for p in partes if p.function_call]
    registro = {'turno': turno, 'texto': ' '.join(textos).strip()[:800],
                'chamadas': [c.name for c in chamadas]}
    print(f"[turno {turno}] {registro['texto'][:160]} -> {registro['chamadas']}", flush=True)
    historico.append(gt.Content(role='model', parts=partes))

    if not chamadas:
        estado['turnos'].append(registro)
        estado['parada'] = 'modelo parou de chamar ferramentas'
        publicar()
        break

    respostas = []
    for c in chamadas:
        fn = FERRAMENTAS.get(c.name)
        if fn is None:
            resultado, imagem = {'erro': f'ferramenta desconhecida {c.name}'}, None
        else:
            argumentos = dict(c.args or {})
            resultado, imagem = fn(argumentos)
            if c.name not in ('observar_cena', 'concluir', 'consultar_licoes'):
                resultado = _registrar(c.name, resultado, argumentos)
        print(f'   {c.name} -> {json.dumps(resultado, ensure_ascii=False)[:200]}', flush=True)
        registro.setdefault('resultados', []).append({c.name: resultado})
        respostas.append(gt.Part.from_function_response(name=c.name, response=resultado))
        for nome_cam, bytes_img in (imagem or []):
            respostas.append(gt.Part.from_text(text=f'[câmera {nome_cam}]'))
            respostas.append(gt.Part.from_bytes(data=bytes_img, mime_type='image/png'))
    historico.append(gt.Content(role='user', parts=respostas))
    estado['turnos'].append(registro)
    publicar()
    if estado['coffee_completed'] or any('concluir' in r for r in registro.get('chamadas', [])):
        break

publicar()
print(json.dumps({k: v for k, v in estado.items() if k != 'turnos'}, indent=2, ensure_ascii=False))
