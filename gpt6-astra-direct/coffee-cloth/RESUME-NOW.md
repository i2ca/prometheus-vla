# Retomada imediata

Objetivo do usuário: continuar autonomamente até fazer café coado com pano no MuJoCo, com física e movimento plausível. Não usar Claude/subagentes, não abrir janelas sem pedido, preservar cada tentativa em results/. Modelo responsável: gpt-6-astra. Café AINDA NÃO pronto.

## Estado canônico e trabalho atual

- Último checkpoint físico aceito: **results/coffee-finish-002/spoon-place**. Usar continuation.npz, liquid-state.json e brew-state.json juntos.
- Próximo trabalho: **Dosagem concluída:19.8513g. coffee-finish-002 está ativo: soltura colher, planejamento/aquecimento, pega alça, despejo e retorno. Consultar progress.json e processos antes de retomar. Preservar os estados e não misturar ramos.**.
- Pó: pote 80.148680 g, colher 0.000000 g, filtro 19.851320 g, derramado 0.000000 g.
- Cena: /opt/i2ca/robotics-lab/coffee-cloth/results/spoon-contact-model-001/scene.xml. Índice atualizado em assets/active-simulation.json. Histórico preservado em CHECKPOINT.md.
- Física da colher corrigida em spoon-contact-model001: solimp .99 .999 .0001 .5 2 / solref .004 1. Geometria/massa/atrito/motores iguais; aproximação rígida numérica, não borracha calibrada.
- Pega050 substitui049 (penetração excessiva detectada após sucesso cinemático). Pega050 e auditoria de profundidade004 passaram. Transporte003 passou, penetração máxima0.055mm. O executor atual limita a penetração mão/colher a0.2mm em cada passo.
- Ramo heaterphysical003 é independente; não mesclar seu estado quente com esta cadeia. Replanejar aquecimento depois da dosagem.
- Usuário pediu continuar até café pronto, sem Claude/subagentes. Não abrir janelas. Café ainda incompleto. Grafo MCP indisponível, fonte direta.

## Correção crítica de cotovelo

NUNCA restaurar a regra antiga q_elbow>=0! Zero do encoder equivale a~81° de flexão. Q negativo pode ser flexão normal; q100° é extensão para trás. Usar `scripts/elbow_anatomy.py`, ângulo geométrico projetado entre âncoras shoulder_pitch/elbow/wrist_pitch, faixa5..145° e tolerância física2°. Motor e punhos60/45/30 continuam limitados. Limite de busca do encoder superior75° + gate geométrico; negativos mecânicos permitidos. Auditorias `elbow-anatomy-audit-001`, `elbow-trajectory-audit-001`: lid016 corrigido PASS, anteriores tinham extensão excessiva.

## Próximos passos

1. Conferir coleta dip004, executar uma transferência completa ao filtro e medir dose/derramamento.
2. Ajustar run_dosing_loop.py aos parâmetros validados da nova pega e acumular20g no mesmo episódio.
3. Integrar pressão do botão/aquecimento, pega da jarra pela alça, despejo quente, drenagem e retorno. Para200ml na xícara, considerar40ml retidos em20g de pó e5ml no pano (~245ml despejados).
4. Revalidar dinâmica da jarra e limites anatômicos nesta cena. Renderizar evidência final e preservar tentativas.

## Armadilhas e evidências

- prepared011 falso PASS invalidado (colher caiu durante5s de settle). `prepared_layout_gate.py` corrigido e teste em trajetórias009/011/012.013 no canto também estável mas pega não passou, não é cena canônica.
- whole-fit001 falso PASS invalidado: ArmIK.fk não atualiza contatos; novo fitter chama mj_forward. `INVALIDATED.json` e auditorias preservados. Seed antigo é só chute de braço, não prova.
- Tentativas anteriores de pinça duas pontas giravam colher~84° e produziam apenas0.068g/dose. Primeiro despejo histórico em spoon-right-dose002; não somar com novo episódio.
- Física: objetos livres, massa/COM/inércia acopladas, motores limitados. Pelve fixa; pó/água/calor reduzidos, sem CFD ou extração química; parâmetros de densidade/massa/atrito parcialmente estimados.
- MCP grafo indisponível (Transport closed); consultas de código por fonte direta, limitação registrada.
- Sem vídeo/janela abertos. Relatório e histórico completo em CHECKPOINT.md (append-only).
