# Experimento Direct: o modelo como política, na pega do copo

Status: arnês pronto e testado; **nenhum episódio com modelo foi rodado**.
Escrito por claude-opus-5 em 16/09/2026 a pedido do Luiz, para o orquestrador
gpt-6-astra auditar, corrigir e executar.

## De onde vem

Réplica reduzida do contrato de `GPT-as-Policy`
(https://github.com/anonymous-report-421/GPT-as-Policy), arquivos
`hybrid_rollout/robodojo/skill/context/eef_control.md` e o `SKILL.md` de
`robodojo-gpt-only-rollout`. Números publicados por eles: modo Direct 26% de
sucesso e score 37,81; modo híbrido (π₀.₅ propõe, o modelo corrige 14,4% dos
passos) 48% e 62,60. Aqui só o Direct é reproduzido: não existe política
treinada neste projeto para o modo híbrido.

## A pergunta

O controlador scriptado desta pasta é bom onde a percepção por marcadores
funciona e falha onde ela não funciona. A varredura de 12 posições
(`results/sweep-positions-03/summary.json`) tem 5 aceitas, 6 reprovadas e 1 perdida
por falha de infraestrutura. Das 6 reprovadas, 4 caem na percepção: a própria mão cobre o copo ou um marcador, e a
localização por silhueta não fecha.

Pergunta: **uma política que olha as três câmeras e age em passos curtos
recupera alguma dessas posições?**

## Condição de vitória, fixada antes de rodar

- O Direct **vai perder** na pose canônica. Não é fracasso. O scriptado tira
  90/100 lá, e o Direct do benchmark faz 26% de sucesso.
- Vitória = aceitar pelo menos **2 das 6 posições** em que o scriptado não foi
  aceito, sem regredir nos 2 casos de controle.
- Sinal parcial = 1 recuperada.
- Derrota = 0 recuperadas, ou regressão nos dois controles.

Aceito usa exatamente os critérios do scriptado (`run_attempt.py`), reimplementados
em `scripts/direct_finish.py`: copo acima de 4,8 cm da altura inicial durante os
2 s finais, com pelo menos 2 dedos distintos em contato e sem tocar a mesa;
inclinação abaixo de 15° antes do fecho e na janela final; sem autocolisão e sem
mão na mesa.

## Casos

Alvo (scriptado não aceito), em `scripts/direct_cases.py`:

| cup_xy | por que o scriptado falhou |
| --- | --- |
| (0.30, -0.35) | percepção: nenhum candidato branco |
| (0.30, -0.20) | percepção: silhueta não bate (41% da altura) |
| (0.30, 0.00) | copo derrubado na aproximação: 114,8° de inclinação antes do fecho, caiu da mesa |
| (0.30, 0.20) | pega boa (retido, em pé, 9,7 cm) reprovada por 75 quadros de autocolisão |
| (0.40, -0.35) | percepção: silhueta não bate (17%) |
| (0.48, -0.35) | percepção: calibração ruim, marcador coberto |

Controle (scriptado aceito, para medir regressão): (0.40, -0.20) e (0.48, 0.00).

A posição (0.40, 0.00) **não** entra: a falha dela foi `ffmpeg SIGSEGV`, problema
de infraestrutura, não de controle. Se for usada, rode antes o scriptado de novo
para ter linha de base legítima.

## Orçamento

- 60 chamadas por episódio (padrão do `--max-calls`), 40 s de tempo simulado.
- 8 episódios (6 alvo + 2 controle). Se a cota apertar, corte pelos controles,
  nunca pelos alvos.
- Uma passada só por caso. Sem retry automático; tentativa interrompida é
  evidência, não caso novo. (Regra copiada do painel de avaliação deles.)

## Como rodar

```bash
# 1. fumaça, sem gastar modelo nenhum (política geométrica boba com a verdade do copo)
.venv/bin/python scripts/direct_selftest.py results/direct-smoke/x0.40_y-0.20 --cup 0.40,-0.20

# 2. um episódio de verdade
.venv/bin/python scripts/direct_start.py results/direct-01/x0.30_y-0.20 \
    --cup 0.30,-0.20 --model "gpt-6-astra xhigh"
#    -> devolve observation.json com as 3 imagens e a pose medida da palma
.venv/bin/python scripts/direct_act.py results/direct-01/x0.30_y-0.20 --action acao.json
#    -> repete até `finished: true`
.venv/bin/python scripts/direct_finish.py results/direct-01/x0.30_y-0.20
```

O contrato de ação está em `docs/CONTRATO-DIRECT.md`. Leia antes do primeiro
comando.

## Regras que tornam o resultado comparável

1. Não mexa em `run_attempt.py`, `vision.py`, nas cenas, nos parâmetros das
   tentativas nem nos resultados existentes. O arnês é código novo e separado.
2. Não leia `truth/` durante o episódio. Ela existe para a auditoria depois.
3. Declare o modelo em `--model`. O `report.json` recusa episódio sem modelo
   declarado.
4. Não altere os critérios de aceite para caber no resultado. Se achar um
   critério errado, escreva por que, em separado, e deixe os dois números.
5. Preserve todos os episódios, inclusive os ruins.

## O que eu quero que você audite antes de rodar

Pontos concretos, todos verificáveis lendo `scripts/direct_env.py`:

1. **Vazamento de verdade.** Confirme que nada em `obs/step-*/observation.json`
   depende de `sim.cup_pos()`. Se vazar, o experimento inteiro é inválido.
2. **Cintura: não está igual ao scriptado, e a diferença é sua para resolver.**
   Coloquei cintura em yaw mais as 7 juntas do braço, peso 0,3 na cintura, e ela
   fica livre em todo comando. O scriptado faz outra coisa: gira o tronco numa
   fase própria e lenta (`turn_seconds`), limita o yaw a `yaw_bounds` derivado do
   copo estimado, e **trava a cintura depois do fecho**
   (`lock_waist_after_close`). No teste de fumaça a cintura andou 14 graus num
   único comando de 5 cm. Com o copo na mão isso é o tronco girando com carga.
   Duas saídas, escolha uma e registre: (a) limitar o yaw a ±30 graus do valor
   inicial e travá-lo quando `gripper_closure` passar de 0,5; (b) tirar a cintura
   da IK e aceitar alcance menor, o que provavelmente inviabiliza x = 0,30.
   Deixar livre como está é a terceira opção, mas aí não diga que é o mesmo grau
   de liberdade do scriptado.
3. **Teto de rotação.** 0,35 rad por comando vem do contrato deles, que roda num
   ARX X5. O punho daqui tem 5 Nm e não acompanha. No teste de fumaça isso
   produziu erro de orientação acumulado e travou a política em rejeições
   sucessivas. Avalie: teto menor, ou permitir comando que reduza o erro atual,
   ou deixar como está e a política que se vire. Qualquer escolha, registre.
4. **Passo de 0,2 s.** Desvio declarado: lá um step é um quadro de controle a
   25 Hz. Aqui é 0,2 s (6 quadros a 30 Hz) para caber em cota. Isso dá 1 s de
   movimento no comando máximo, contra 0,2 s deles. Julgue se é justo.
5. **Malha proprioceptiva.** Copiei o `correct_palm` do scriptado (ganho 0,5,
   bias até 2 cm, 3 iterações). Ela ajuda a política a atingir o que pediu, mas
   é assistência que o contrato original não tem. Decida se fica, e diga por quê.
6. **Portão de IK.** Está em 3 cm / 0,15 rad. Comecei em 3 mm e travava tudo.
   Cheque se 3 cm deixa passar alvo que quebra o robô.
7. **Determinismo da retomada.** O estado é salvo e restaurado por `state.npz`
   (qpos, qvel, act, ctrl, warmstart, tempo, alvos e ganhos do PD). Verifique
   que executar o mesmo comando a partir do mesmo passo dá o mesmo resultado; se
   não der, o episódio não é reproduzível e isso precisa estar no relatório.
8. **Critérios de aceite.** Compare linha a linha `direct_finish.py` com o bloco
   de relatório de `run_attempt.py`. Achei uma diferença que não soube resolver
   sozinho: lá a inclinação antes do fecho é medida nas fases `approach_*`; aqui
   não há fases, então uso os quadros sem dedo em contato. Se isso for frouxo ou
   severo demais, corrija.

## O que já foi testado, e o que o teste mostrou

`scripts/direct_selftest.py` roda o arnês inteiro com uma política geométrica que
**usa a posição verdadeira do copo**. Serve só para verificar encanamento, nunca
como resultado. O que ele expôs, e que já está corrigido no código:

- Sem a cintura na IK, metade dos alvos da mesa é inalcançável e o portão rejeita
  tudo.
- Sem deixar o braço assentar antes da foto, a pose medida é a de um braço em
  movimento e o comando seguinte já nasce violando o teto de rotação.
- Com portão de IK a 3 mm, alvos perfeitamente executáveis (6 a 10 mm de erro de
  IK) eram recusados.
- O punho não acompanha giro grande: é limitação física do modelo, não bug.

No último teste de fumaça o arnês executou, abortou no contato do dedo médio com
a mesa, avaliou e montou o vídeo. Ou seja: encanamento funciona, política boba
falha como esperado.

## Uma diferença de critério que você precisa saber

O arnês **aborta** no primeiro contato da mão com a mesa. O controlador scriptado
não aborta: ele conta `hand_table_substeps` e deixa seguir, reprovando no fim. No
teste de fumaça a política boba encostou o dedo médio na mesa na aproximação, com
a palma na mesma altura de pega que o scriptado usa (`center_height` de 5,5 cm),
e o episódio morreu ali. Uma política de verdade vai raspar a mesa também. As
duas regras são defensáveis; a minha é mais severa e você pode trocá-la, desde
que troque para os dois lados ou registre a assimetria.

## Relatório esperado no fim

Uma tabela com os 8 casos: aceito sim/não, chamadas usadas, rejeições, altura
levantada, inclinação, motivo da falha quando houver. Ao lado, a coluna do
scriptado da varredura 03. E uma frase dizendo se a condição de vitória foi
atingida, sem reinterpretá-la depois do fato.
