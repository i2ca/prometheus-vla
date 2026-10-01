"""Caderno de lições do orquestrador: o que já foi tentado e o que resolveu.

A diferença entre buscar e acumular. Sem isto, cada episódio redescobre do zero:
num run o modelo achou que `peso_orientacao=0` na mão direita destravava a
oposição dos dedos, e no run seguinte foi testar a mão esquerda de novo.

Cada lição é indexada por uma **assinatura de problema**, que é onde a coisa
morreu, não a mensagem literal: `<ferramenta>/<etapa>/<portao>`. Assim uma
falha nova reconhece um problema antigo mesmo com números diferentes.

Uma tentativa é registrada sempre. Uma solução só é registrada quando a etapa
de fato passou, com os números que sustentam. É a mesma régua do resto do
projeto: não aceitar sucesso sem evidência.
"""
import json
import re
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO = RAIZ / 'licoes' / 'licoes.json'
LEGIVEL = RAIZ / 'licoes' / 'LICOES.md'

# Portões conhecidos, para a assinatura não depender do texto exato do erro.
PORTOES = [
    (r'hand clearance|hot_m|metal quente', 'folga_metal_quente'),
    (r'forbidden contact|contato proibido', 'contato_proibido'),
    (r'penetration|penetra', 'penetracao'),
    (r'elbow|cotovelo', 'anatomia_cotovelo'),
    (r'wrist (bounds|limits)|punho', 'limites_punho'),
    (r'spill|derram', 'derramamento'),
    (r'prop (drift|displacement)|deriva', 'deriva_de_props'),
    (r'table clearance|mesa', 'folga_mesa'),
    (r'nenhuma pega valida|oposicao|opposition', 'pega_sem_oposicao'),
    (r'aproximacao valida|reverse approach', 'aproximacao_invalida'),
    (r'alcance valida|reach', 'alcance_invalido'),
    (r'timed out|timeout', 'tempo_esgotado'),
    (r'numerical warning', 'aviso_numerico'),
    (r'sustained contact force|contact force', 'forca_de_contato'),
]


def portao_de(motivo: str) -> str:
    texto = (motivo or '').lower()
    for padrao, nome in PORTOES:
        if re.search(padrao, texto):
            return nome
    return 'outro'


def assinatura(ferramenta: str, etapa: str, motivo: str) -> str:
    return f'{ferramenta}/{etapa or "?"}/{portao_de(motivo)}'


def _carregar() -> dict:
    if ARQUIVO.exists():
        return json.loads(ARQUIVO.read_text())
    return {'versao': 1, 'licoes': {}}


def _salvar(dados: dict) -> None:
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARQUIVO.with_suffix('.tmp')
    tmp.write_text(json.dumps(dados, indent=2, ensure_ascii=False))
    tmp.replace(ARQUIVO)
    _escrever_legivel(dados)


def registrar_tentativa(ferramenta, etapa, motivo, parametros, medido, episodio='', run=''):
    """Toda falha entra, mesmo sem solução. É o 'documentou a tentativa'."""
    dados = _carregar()
    chave = assinatura(ferramenta, etapa, motivo)
    licao = dados['licoes'].setdefault(chave, {
        'problema': chave, 'primeira_vez': datetime.now().isoformat(timespec='seconds'),
        'tentativas': [], 'solucao': None})
    licao['tentativas'].append({
        'quando': datetime.now().isoformat(timespec='seconds'),
        'episodio': episodio, 'parametros': parametros, 'run': str(run),
        'medido': medido, 'motivo': str(motivo)[:300], 'resolveu': False})
    # não deixa crescer sem limite; guarda as mais recentes
    licao['tentativas'] = licao['tentativas'][-40:]
    _salvar(dados)
    return chave


def registrar_solucao(ferramenta, etapa_que_falhava, portao, parametros, evidencia,
                      episodio='', contexto=None, run=''):
    """Só é chamada quando a etapa passou de verdade, com números.

    `contexto` guarda o estado em que a solução funcionou. Sem isso a lição vira
    receita cega: o modelo aplicou os parâmetros certos da fervura num estado
    onde a água já estava a 100 °C e o aquecedor recusou ligar.
    """
    dados = _carregar()
    chave = f'{ferramenta}/{etapa_que_falhava or "?"}/{portao}'
    licao = dados['licoes'].setdefault(chave, {
        'problema': chave, 'primeira_vez': datetime.now().isoformat(timespec='seconds'),
        'tentativas': [], 'solucao': None})
    licao['solucao'] = {
        'quando': datetime.now().isoformat(timespec='seconds'), 'episodio': episodio,
        'parametros': parametros, 'evidencia': evidencia,
        # sem o diretorio da corrida a licao nao e' auditavel: ficam os numeros
        # sem os arquivos que os produziram.
        'run': str(run), 'valia_no_estado': contexto,
        'tentativas_ate_resolver': len(licao['tentativas'])}
    _salvar(dados)
    return chave


def consultar(ferramenta=None, etapa=None, motivo=None, chave=None):
    """Devolve o que já se sabe sobre este problema.

    Busca em três níveis, do específico para o amplo, porque exigir que o
    modelo acerte a assinatura exata é pedir demais: ele pergunta com as
    palavras dele. Na prática a primeira versão devolvia vazio mesmo com cinco
    problemas registrados para a ferramenta perguntada.
    """
    dados = _carregar()
    licoes_ = dados['licoes']

    if chave is None and ferramenta:
        chave = assinatura(ferramenta, etapa, motivo)
    if chave and chave in licoes_:
        return [_resumir(licoes_[chave])]

    # 2) mesmo portão, qualquer ferramenta
    alvo = portao_de(motivo or '')
    if alvo != 'outro':
        achados = [_resumir(v) for k, v in licoes_.items() if k.endswith('/' + alvo)]
        if achados:
            return achados

    # 3) tudo o que existe para aquela ferramenta
    if ferramenta:
        achados = [_resumir(v) for k, v in licoes_.items() if k.startswith(ferramenta + '/')]
        if achados:
            return achados

    return []


def resolvidas():
    dados = _carregar()
    return [_resumir(v) for v in dados['licoes'].values() if v.get('solucao')]


def _resumir(licao: dict) -> dict:
    tentativas = licao.get('tentativas', [])
    return {
        'problema': licao['problema'],
        'solucao': licao.get('solucao'),
        'tentativas_que_falharam': len(tentativas),
        'parametros_ja_tentados_sem_sucesso': [t.get('parametros') for t in tentativas[-8:]],
    }


def _escrever_legivel(dados: dict) -> None:
    linhas = ['# Lições do orquestrador do café', '',
              'Gerado por `scripts/licoes.py`. Uma solução só entra aqui depois que a',
              'etapa passou de verdade, com os números que sustentam.', '']
    resolvidas_, abertas = [], []
    for licao in dados['licoes'].values():
        (resolvidas_ if licao.get('solucao') else abertas).append(licao)
    linhas += ['## Resolvidos', '']
    if not resolvidas_:
        linhas += ['Nenhum ainda.', '']
    for licao in resolvidas_:
        s = licao['solucao']
        linhas += [f"### {licao['problema']}", '',
                   f"Resolvido em {s['quando']} depois de {s['tentativas_ate_resolver']} tentativas falhas.", '',
                   '```json', json.dumps({'parametros': s['parametros'],
                                          'evidencia': s['evidencia']}, indent=1, ensure_ascii=False),
                   '```', '']
    linhas += ['## Ainda abertos', '']
    if not abertas:
        linhas += ['Nenhum.', '']
    for licao in abertas:
        linhas += [f"### {licao['problema']}", '',
                   f"{len(licao['tentativas'])} tentativas, nenhuma resolveu. Últimas:", '']
        for t in licao['tentativas'][-5:]:
            linhas.append(f"- `{json.dumps(t.get('parametros'), ensure_ascii=False)}` "
                          f"-> {json.dumps(t.get('medido'), ensure_ascii=False)[:180]}")
        linhas.append('')
    LEGIVEL.write_text('\n'.join(linhas))
