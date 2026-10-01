"""Explicitly select an accepted physical checkpoint; preserve prior index."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--next', required=True)
a = ap.parse_args()
r = json.loads((a.source / 'report.json').read_text())
assert r.get('pass') and not (a.source / 'INVALIDATED.json').exists()
for name in ['continuation.npz', 'liquid-state.json', 'brew-state.json']:
    assert (a.source / name).is_file(), name
a.out.mkdir(exist_ok=False)
index = Path('assets/active-simulation.json')
previous = index.read_text()
(a.out / 'previous-active-simulation.json').write_text(previous)
active = json.loads(previous)
active.update(model='gpt-6-astra', scene=r['scene'],
              latest_accepted_stage=str(a.source / 'report.json'),
              latest_physical_checkpoint=str(a.source / 'continuation.npz'),
              liquid_checkpoint=str(a.source / 'liquid-state.json'),
              brew_checkpoint=str(a.source / 'brew-state.json'),
              stage=str(a.source), next_stage=a.next, coffee_completed=False)
index.write_text(json.dumps(active, indent=2))
brew = json.loads((a.source / 'brew-state.json').read_text())
entry = {'model': 'gpt-6-astra', 'source': str(a.source), 'next': a.next,
         'grounds': brew['grounds'], 'coffee_completed': False}
(a.out / 'milestone.json').write_text(json.dumps(entry, indent=2))
with Path('CHECKPOINT.md').open('a') as f:
    f.write('\n## Etapa aceita — gpt-6-astra\n' + json.dumps(entry, ensure_ascii=False) + '\n')
resume = Path('RESUME-NOW.md')
text = resume.read_text()
start = text.index('## Estado canônico')
end = text.index('## Correção crítica')
text = text[:start] + f'''## Estado canônico e trabalho atual

- Último checkpoint físico aceito: **{a.source}**. Usar continuation.npz, liquid-state.json e brew-state.json juntos.
- Próximo trabalho: **{a.next}**.
- Pó: pote {brew['grounds']['pot_g']:.6f} g, colher {brew['grounds']['spoon_g']:.6f} g, filtro {brew['grounds']['filter_g']:.6f} g, derramado {brew['grounds']['spilled_g']:.6f} g.
- Cena: {r['scene']}. Índice atualizado em assets/active-simulation.json. Histórico preservado em CHECKPOINT.md.
- Física da colher corrigida em spoon-contact-model001: solimp .99 .999 .0001 .5 2 / solref .004 1. Geometria/massa/atrito/motores iguais; aproximação rígida numérica, não borracha calibrada.
- Pega050 substitui049 (penetração excessiva detectada após sucesso cinemático). Pega050 e auditoria de profundidade004 passaram. Transporte003 passou, penetração máxima0.055mm. O executor atual limita a penetração mão/colher a0.2mm em cada passo.
- Ramo heaterphysical003 é independente; não mesclar seu estado quente com esta cadeia. Replanejar aquecimento depois da dosagem.
- Usuário pediu continuar até café pronto, sem Claude/subagentes. Não abrir janelas. Café ainda incompleto. Grafo MCP indisponível, fonte direta.

''' + text[end:]
resume.write_text(text)
print(entry)
