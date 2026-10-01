"""Importa para o caderno as tentativas dos episódios já rodados.

O caderno começa com a história real em vez de vazio. Só entra o que está
gravado nos `progress.json`; nada é inventado.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import licoes

RAIZ = Path(__file__).resolve().parents[1]
n_tent = 0
n_sol = 0
for progresso in sorted(RAIZ.glob('results-claude/agent-cafe-*/progress.json')):
    episodio = progresso.parent.name
    dados = json.loads(progresso.read_text())
    for turno in dados.get('turnos', []):
        for resultado in turno.get('resultados', []):
            for ferramenta, valor in resultado.items():
                if not isinstance(valor, dict) or ferramenta in ('observar_cena', 'concluir'):
                    continue
                if valor.get('sucesso') is False:
                    licoes.registrar_tentativa(
                        ferramenta, valor.get('etapa'), valor.get('motivo'),
                        valor.get('parametros_usados') or {}, valor.get('medido'), episodio)
                    n_tent += 1
                elif valor.get('sucesso') is True and ferramenta == 'ligar_chaleira':
                    # a fervura é a única etapa que já passou de verdade
                    licoes.registrar_solucao(
                        'ligar_chaleira', 'prensar', 'forca_de_contato',
                        {'curso_mm': 30, 'desloc_y_mm': -6, 'cintura_deg': 25, 'recuo_mm': 40},
                        {k: v for k, v in valor.items() if k != 'sucesso'}, episodio)
                    n_sol += 1

print(f'tentativas importadas: {n_tent} | soluções: {n_sol}')
