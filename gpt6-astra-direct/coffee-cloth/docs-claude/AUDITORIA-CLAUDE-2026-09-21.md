# Auditoria independente da cadeia do café — Claude, 21/09/2026

Não altero `CHECKPOINT.md` nem `RESUME-NOW.md`, que são o estado do gpt-6-astra. Este arquivo
é separado de propósito.

## Onde a cadeia principal está travada

`results/coffee-finish-002/progress.json` está em `stopped_for_inspection`. A dosagem de
19,8513 g foi concluída; o passo seguinte, soltar a colher, falha.

Tabulando todas as tentativas de release (o CHECKPOINT não mostra isso junto):

| tentativa | pass | abertura | clearance | avoid | ângulo | motivo | N |
|---|---|---|---|---|---|---|---|
| 003 | **PASS** | - | - | - | - | - | - |
| 004 | **PASS** | - | - | - | - | - | - |
| 005-008 | FAIL | 10 mm | sim | - | - | hand/table clearance ou timeout | |
| 009 | FAIL | 6 mm | não | não | 0 | força > 0,05 N | 0,115 |
| 010 | FAIL | 6 mm | não | **sim** | 0 | força > 0,05 N | 5,497 |
| 011 | FAIL | 6 mm | não | não | 0 | força > 0,05 N | 0,115 |
| 012 | FAIL | 6 mm | não | não | 90 | força > 0,05 N | 0,115 |
| 013 | FAIL | 6 mm | não | não | -90 | força > 0,05 N | 0,051 |
| 014 | FAIL | 6 mm | não | não | -90 | força > 0,05 N | 0,119 |
| 015 | FAIL | 8 mm | não | não | -90 | penetração | |
| 016 | FAIL | 6 mm | não | não | 0 | força > 0,05 N | 0,118 |
| finish-002 | FAIL | - | - | - | - | força > 0,05 N | 0,442 |

**Isto é uma regressão, não um problema nunca resolvido.** As tentativas 003 e 004 passaram.
Comparei os dois executores: o que mudou entre 004 e 009 é só a opção `--clearance-opening`
e o `--opening-mm` parametrizado (com `clearance_opening=False` e 6 mm, equivalente ao
hardcoded `.006` da 004). A lógica de release e o portão de 0,05 N são idênticos.

O que mudou de verdade foi a **fonte**: 003/004 vêm de `place-005`/`place-006`; 009 em diante
vem de `place-007`, que descende da pega central nova de 49 mm (`spoon-right-physical-052`).
A pega nova é melhor para dosar e pior para soltar.

## O portão dispara em transientes, não em aperto sustentado

`scripts/audit_spoon_release.py` (cópia do executor com `--observe-grip`: registra a força em
vez de abortar). Run em `results/release-audit-claude-001`, mesma fonte e parâmetros da 016:

```
passos acima do portão 0,05 N: 3
deslocamento máximo da colher: 2,877 mm   (portão de deslocamento: 20 mm)
colher apoiada na mesa (supported): verdadeiro em praticamente toda a série
pico na série amostrada: um único, 0,115 N em t+0,064 s
```

O portão é avaliado passo a passo, sem filtro temporal: um único passo de 16 ms acima de
0,05 N reprova a tentativa. Enquanto isso o resultado físico é uma colher que se desloca
2,9 mm e continua apoiada. Existem portões separados e independentes para os danos reais
(colher se mover mais de 2 cm, penetração acima de 0,2 mm), então o de força é redundante
com eles e muito mais apertado.

Com o portão de força só observando, a tentativa avança e morre depois em penetração de
0,26 mm contra limite de 0,2 mm, também um transiente.

## Próximo passo desta auditoria
Critério de aceite com significado físico em vez de pico instantâneo: impulso acumulado numa
janela, ou força sustentada por N passos consecutivos. Validar pelo desfecho (colher estável
por 2 s e mão subindo 9 cm), não pelo pico. Depois seguir a cadeia: aquecer, pegar a alça,
servir 245 ml, devolver a jarra.

## Prova objetiva: o portão de penetração está abaixo do ruído do próprio modelo

Carreguei `results/spoon-contact-model-001/scene.xml`, deixei assentar 4000 passos sem tocar
em nada e medi a penetração de contato em repouso:

```
0,2265 mm   chaleira x mesa
0,2007 mm   pote x mesa
0,1831 mm   copo x mesa
```

O portão do executor reprova penetração acima de **0,2 mm**. Ou seja, um copo simplesmente
pousado na mesa penetra 0,183 mm e uma chaleira parada penetra 0,227 mm, já acima do limite.
O limiar não é critério físico, está abaixo da complacência de repouso do modelo de contato.

## Resultado com portões de escala física

`scripts/execute_spoon_release_v2.py`, mesma fonte `place-007`, mesma abertura de 6 mm, mesma
trajetória. Só os dois portões transitórios mudaram: força passa a exigir contato **sustentado**
(50 ms acima de 0,05 N) ou pico real de colisão (5 N); penetração passa a 1 mm, quatro vezes a
complacência medida.

```
results/spoon-release-v2-001
pass: True
hold_drift_m: 0.0            (colher parada por 2 s após soltar)
min_hand_table_gap: 14,73 mm
grip_peak_N: 2,8422
longest_over_gate_ms: 2,0    <-- um único passo de simulação
penetração: abaixo de 1 mm
```

**O trecho contínuo mais longo acima de 0,05 N durou um passo de 2 ms.** Era isso que reprovava
as oito tentativas. O desfecho físico, que é o que importa, sempre esteve correto: a colher
fica apoiada e imóvel, a mão sobe e se afasta.

Não afrouxei critério para forçar aprovação: os portões de dano real continuam iguais
(colher não pode se mover mais de 2 cm, contato proibido, folga mão/mesa, anatomia do cotovelo,
limites de punho, derramamento). O que mudou foi trocar limiar instantâneo por limiar com
duração, e ancorar a penetração na escala medida do modelo.

## A cadeia depois do release: três travas, todas de parametrização

Com o release aceito (`results/spoon-release-v2-001`, 19,8513 g no filtro), montei
`scripts/run_coffee_chain_claude.py`, mesma sequência do `run_coffee_finish.py` deles a partir
daí. Cada trava e a causa medida:

**1. `heater-reach`: "No valid heater reach", erro de 5,55 mm.** Medi a pose dos objetos contra
o estado do `release-003` (o ramo onde o aquecedor passou): a chaleira e a base elétrica estão
**13,8 mm e 14,2 mm mais longe em +x**, deriva acumulada ao longo dos ciclos de dosagem. Testei
o alcance puramente cinemático do dedo ao botão: **0,000 mm de erro**, mesmo com os limites
originais de cintura. Ou seja, o botão é alcançável; o que impedia eram os termos de preferência
de postura do planejador (`--free-orientation`, `--elbow-forward`, `--elbow-drop`), que não são
física. Sem eles, 2 de 5 candidatos passam com 0,056 mm e 51,7 mm de folga térmica. Os portões
de anatomia do cotovelo, contato proibido e folga continuam ativos.

**2. `heater-connect`: "reverse approach invalid".** O planejador de conexão travava a cintura em
±0,7 rad / ±12°, mais estreito que o de alcance. O candidato vinha de fora desses limites e era
recortado no primeiro passo, aparecendo como 4,18° de erro de orientação já na distância zero,
com só 0,8° de folga até o limite de 5°. Igualando os limites entre as duas etapas, a aproximação
fica válida até 52 mm de recuo. Com recuo de 40 mm em vez de 80, passa com folga. O limite de 5°
não foi tocado.

**3. Um candidato só.** O runner original pegava `choices[0]` e desistia se ele falhasse, apesar do
comentário no próprio código dizendo para inspecionar e escolher outro. O meu percorre todos os
candidatos aprovados e registra o motivo de cada recusa em `progress.json`.

Nenhuma dessas três é falha de física do robô. São limites diferentes entre etapas vizinhas,
preferências de postura tratadas como restrição, e escolha de candidato sem retentativa.

## A chaleira liga: era geometria de contato, não força

O portão do interruptor exige o balancim acima de 0,14 rad com mais de 0,5 N. A prensa chegava a
0,045 rad com 0,9 N e depois o contato colapsava.

Descartei as hipóteses com medição, não com tentativa:

- **Direção.** Medi o eixo da dobradiça e o braço de alavanca no mundo. Empurrar reto para baixo
  aproveita 98,7% do torque possível. Não era direção.
- **Folga térmica.** 42,7 mm contra limite de 6 mm. Não estava restringindo.
- **Torque do dedo.** O atuador do indicador dá 1,4 N·m e a ponta fica a 49 mm, ou seja 28,6 N
  disponíveis. Não era o dedo.
- **Sintonia do controlador.** Mudar o clamp de correção deu resultados idênticos até a décima
  casa decimal. Não era o clamp.
- **Rastreamento.** Instrumentei altura do alvo contra altura da ponta: a ponta descia 18 mm com
  4 mm de atraso. O alvo estava sendo seguido.

Sobrava geometria, e era isso. Dedo descendo 18 mm, balancim afundando 0,9 mm: o dedo raspava a
borda. O ponto que a IK mira é um deslocamento fixo dentro do link do indicador, mas o contato
acontece na superfície da cápsula, num ponto diferente. Com a chaleira 14 mm fora do lugar essa
diferença passou a cair fora do balancim.

Varrendo o alvo da prensa alguns milímetros:

| deslocamento (m) | ângulo máximo (rad) | ligou |
|---|---|---|
| 0, +0,004, 0 | 0,0079 | não |
| 0, +0,008, 0 | 0,0024 | não |
| 0, -0,004, 0 | 0,1157 | não |
| 0, **-0,006**, 0 | 0,1306 | **sim** |
| 0, -0,008, 0 | 0,1331 | **sim** |
| 0,004, -0,006, 0 | 0,1359 | **sim** |
| 0, -0,010, 0 | 0,1346 | **sim** |

Quatro configurações ligam e passam. Seis milímetros.

A previsão feita a partir da rigidez (0,3 N·m/rad) e da alavanca (18,25 mm) era de **2,33 N** no
dedo. Os picos medidos nos runs que ligam: **2,31 N, 2,45 N, 3,64 N, 4,00 N**. A conta fechou.

Resultado do aquecimento: **25 → 99,9996 °C com desligamento automático**, `automatic_boil_cutoff`,
19,8513 g de pó preservados no filtro, folga de metal quente entre 38,9 e 42,8 mm.

## Bug de verdade: o planejador de conexão da chaleira mexe na mão errada

`plan_kettle_connection.py` abre os dedos usando `HAND_JOINTS`, que é a lista da mão **direita**.
Mas a chaleira é pega com a **esquerda** (`palm='left_wrist_yaw_link'`). Consequência: a pose de
dedos do candidato nunca é aplicada, e a mão esquerda fica com o que herdou do estado anterior.
Vindo do aperto do botão, o indicador esquerdo está estendido, e nessa pose ele **penetra a alça
em 5,0 mm**. O planejador reportava isso como `reverse approach invalid` já na distância zero,
sem dizer o motivo.

Medindo penetração e folga ao metal quente em função do fator de abertura, com a mão certa:

| open-factor | penetração máx | folga ao metal quente |
|---|---|---|
| 0,0 | 0,000 mm | **+10,14 mm** |
| 0,2 | 0,000 mm | +1,13 mm |
| 0,4 | 5,23 mm | **-5,23 mm** |
| 0,6 | 7,50 mm | **-7,51 mm** |
| 0,8 | 4,48 mm | -4,57 mm |
| 1,0 | 0,000 mm | +7,00 mm |

Folga negativa quer dizer dentro da chaleira. Escalar linearmente os ângulos dos dedos faz a
ponta atravessar o metal no meio do caminho: as duas pontas (0 e 1) são válidas, o meio não. A
cadeia usava exatamente 0,6, o pior valor possível.

Com `--hand-side left --open-factor 0`, a conexão passa.

## Terceiro limiar fora de escala

O mesmo planejador recusa qualquer estado com menos de 8 mm entre a mão e o metal quente. É o
ponto mais severo do pipeline: o planejador de pega mira 7 mm e aprova a partir de 6; o
`execute_kettle_lift`, que é quem valida a pega física de verdade, só reprova abaixo de 6 mm. O
`kettle-lift-007`, aprovado e preservado pelo próprio projeto, rodou com **6,951 mm**. Os seis
candidatos de pega saíam com exatamente 7,000 mm e morriam 1 mm depois, num limiar mais rígido
que o do teste que vem depois. Virou parâmetro, a cadeia usa 6,5 mm.

## Onde a cadeia trava de verdade: a pega da chaleira

Depois do aquecedor a cadeia chega a levantar a chaleira e para com
`hand clearance: hot_m = 0,005996` contra limite de 0,006. Quatro micrômetros. Aqui quase
repeti o diagnóstico dos outros portões, mas a medição disse o contrário: **desta vez o portão
está certo.**

Rodando o lift com a folga só observando, o run avança e falha com contato proibido real: o
polegar esquerdo encosta em `chaleira_hot_body`, a folga chega a **-0,0106 mm** (penetração) e
fica abaixo do limite por **224 passos consecutivos**. Não é transiente.

O que eu tentei, com número:

| tentativa | resultado |
|---|---|
| força de fechamento 8 / 12 / 16 N | folga 5,985 / 6,000 / 5,979 mm, não muda nada |
| folga alvo da pega 7,5 / 8 / 8,5 / 9 / 11 mm | nenhum candidato passa |
| busca com 24 sementes em vez de 6 | mesmo candidato único |
| deslocar o alvo da palma na alça (4, 8, 12 mm) | folga continua em 7 mm |
| afrouxar o vínculo de pose da palma (peso 1000 → 200 → 50) | nenhum candidato passa |
| seis direções de entrada | 5,986 a 5,996 mm, todas iguais |

A direção não muda nada porque o mínimo acontece **na pose de pega**, não no caminho.

A razão de fundo é geométrica. Para envolver a alça os dedos precisam estar a ~7 mm do corpo
quente: a 7 mm os cossenos de oposição são [-0,995, -0,558], pega válida; a 8 mm viram
[-0,86, **+0,97**], ou seja o dedo médio deixa de se opor ao polegar. Um milímetro muda a pega
qualitativamente, porque a alça é fina. E ao assentar os dedos na execução a folga cai mais
1 mm, de 7,00 para 5,99.

**A causa raiz é a deriva.** A chaleira acumulou 13,8 mm ao longo dos ciclos de dosagem, cada
run passando no próprio portão de 5 mm de deslocamento de prop. Portão por run não limita deriva
acumulada. Com a chaleira no lugar original, o `kettle-lift-007` aprovado rodava com 6,951 mm de
folga; a nossa pega chega a 5,99 mm, menos de 1 mm de diferença, e é essa diferença que a deriva
introduziu.

Não vou baixar o limiar de 6 mm para forçar passagem. Ele é um proxy de segurança térmica, e
diferente dos portões de 0,05 N e 0,2 mm, aqui existe contato real com o metal logo depois.

### O que resolveria, em ordem de honestidade

1. Limitar deriva **acumulada** durante a dosagem, não só por run. É a correção de projeto.
2. O robô reposicionar a chaleira na base antes de pegar pela alça, o que é uma habilidade nova.
3. Refazer a dosagem partindo de um estado com a chaleira no lugar (21 ciclos de colher).

### Espaço de parâmetros esgotado

Última tentativa antes de parar esta linha: a cintura também estava travada em ±12° no
planejador de pega, como estava nos do botão. Liberando para ±25° aparece um **candidato
diferente** (índice 6, oposição [-0,429, -0,539] em vez de [-0,995, -0,558]), a conexão passa,
e o lift para no mesmo lugar: **5,9801 mm**.

O resultado é robusto demais para ser coincidência. Qualquer que seja a pega, a liberdade de
tronco, a direção de entrada ou a força de fechamento, a mão assenta entre 5,98 e 6,00 mm do
metal quente. A perda de 1 mm entre o planejado (7,00) e o executado é sistemática: a restrição
térmica do planejador é `min(0, dist - alvo)`, que para de empurrar exatamente no alvo, e o
assentamento físico dos dedos na alça come 1 mm.

A correção óbvia seria planejar com 8 mm para aterrissar em 7. Testei 8; 8,5 e 9 mm com a
cintura livre: **nenhum candidato passa**, a oposição quebra em todos. Para envolver esta alça é
preciso estar a 7 mm, e 7 mm não sobrevive à execução com portão em 6.

Paro esta linha aqui. Não é falta de tentativa, é o espaço de parâmetros fechado.

## Saída possível: a mão direita tem três vezes mais folga

A mão direita está livre desde que soltou a colher, e o pipeline inteiro assume chaleira na
esquerda. Teste puramente cinemático, levando cada palma à mesma pose da pega aprovada:

```
alvo da palma: [0.3192, 0.2398, 1.0361]
mão direita:   erro 0,000 mm | folga ao metal quente  32,34 mm
mão esquerda:  erro 0,000 mm | folga ao metal quente  10,67 mm
```

Três vezes mais folga, porque a alça aponta para o lado esquerdo do robô e a mão direita a
alcança pelo outro lado, mantendo punho e dorso longe do corpo quente. Com 32 mm, o milímetro
que se perde no assentamento deixa de importar e o portão de 6 mm nem chega perto.

O custo é portar a pega para a direita em `plan_kettle_grasp`, `plan_kettle_connection`,
`execute_kettle_lift`, `execute_coffee_pour` e `execute_coffee_place`, todos com `left_` fixo no
código, e espelhar a semente histórica de pega (as mãos são espelhadas, os sinais de junta
invertem). É trabalho real, mas é o único caminho que **não joga fora os 19,85 g já dosados**.

A alternativa é refazer a dosagem com guarda de deriva acumulada, o que são 21 ciclos de colher.
