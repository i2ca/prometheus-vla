# Café, noite de 20 para 21/09/2026 — o que andou e onde parou

Trabalho autônomo em loop. Não mexi em `CHECKPOINT.md` nem `RESUME-NOW.md`, que são o estado do
gpt-6-astra. Detalhe completo com todos os números em `AUDITORIA-CLAUDE-2026-09-21.md`.

## Destravado

**Soltar a colher.** Estava reprovando havia 8 tentativas seguidas. Dois portões transitórios
eram a causa: força de contato avaliada passo a passo (o trecho contínuo mais longo acima do
limite durou **um passo de 2 ms**) e penetração com limiar de 0,2 mm, **abaixo da complacência de
repouso do próprio modelo** (copo parado na mesa penetra 0,183 mm, chaleira 0,227 mm). Com
limiares de escala física, passa com `hold_drift = 0,0`. Descobri também que 003 e 004 tinham
passado: era regressão vinda da pega central nova, não problema inédito.

**Apertar o botão e ferver.** Estava travado com o balancim em 0,045 rad contra 0,14 exigidos.
Descartei direção (98,7% do torque já era aproveitado), folga térmica (42,7 mm), torque do dedo
(28,6 N disponíveis), sintonia do controlador (resultados idênticos) e rastreamento (a ponta
descia 18 mm). Era geometria de contato: o dedo raspava a borda. **Seis milímetros** de
deslocamento no alvo resolveram. A previsão pela rigidez (0,3 N·m/rad) e alavanca (18,25 mm) era
de 2,33 N; os picos medidos nos runs que ligam foram 2,31 / 2,45 / 3,64 / 4,00 N.
Resultado: **25 → 99,9996 °C com desligamento automático**, 19,8513 g de pó preservados.

Vídeo de seis câmeras em `evidencias/cafe_aperta_botao_e_ferve.mp4` e `cafe_solta_colher.mp4`.

Outras travas removidas no caminho, todas de parametrização entre etapas vizinhas: limites de
cintura diferentes entre planejar e executar, grade de orientações com 5 poses, runner que
testava só o primeiro candidato, e um **bug de verdade**: o planejador de conexão da chaleira
abre os dedos da mão **direita** enquanto a chaleira é pega com a **esquerda**.

## Onde parou

`execute_kettle_lift` reprova com 5,98 mm de folga ao metal quente contra limite de 6 mm. Aqui o
portão **está certo**: com ele desligado o polegar encosta no metal de verdade (−0,0106 mm) por
224 passos consecutivos.

Esgotei o espaço de parâmetros: força de fechamento (8/12/16 N), folga alvo (7,5 a 11 mm), 24
sementes, deslocamento do alvo na alça, afrouxar o vínculo de pose da palma, liberar a cintura,
seis direções de entrada. **Todas dão entre 5,98 e 6,00 mm.** O motivo é que para envolver a alça
os dedos precisam estar a 7 mm do corpo quente (a 8 mm a oposição quebra: o dedo médio deixa de
se opor ao polegar) e o assentamento físico come 1 mm.

**Causa raiz: a chaleira acumulou 13,8 mm de deriva nos ciclos de dosagem**, cada run passando no
próprio portão de 5 mm de deslocamento. Portão por run não limita deriva acumulada. Com a
chaleira no lugar, o `kettle-lift-007` aprovado rodava com 6,951 mm.

## O caminho mais promissor, com número

A mão direita está livre desde que soltou a colher. Levando cada palma à mesma pose de pega:

```
mão direita:  folga ao metal quente 32,34 mm
mão esquerda: folga ao metal quente 10,67 mm
```

Portei o planejador de pega para a direita (`scripts/plan_kettle_grasp_right.py`) e os candidatos
já saem com **10 mm de folga** (contra teto de 7 mm na esquerda), dedos na alça a 0,8 mm e erro
de palma de 0,2 mm. Falta a oposição: o melhor candidato tem [-0,075, -0,975], e o portão exige
o máximo abaixo de -0,3. O par polegar-médio está ótimo; o indicador é que não fecha a pinça.

É problema da semente de dedos, que é uma pose de mão **esquerda** usada como chute inicial numa
mão espelhada. Espelhar a semente corretamente é o próximo passo, e é onde eu parei: fazer isso
errado produz uma pega plausível e falsa, que é justamente o tipo de coisa que essa auditoria
toda existe para evitar.

## Decisão que te espera

1. **Portar a pega para a mão direita.** Preserva os 19,85 g dosados. Exige espelhar a semente e
   adaptar `plan_kettle_connection`, `execute_kettle_lift`, `execute_coffee_pour` e
   `execute_coffee_place`, todos com `left_` fixo.
2. **Refazer a dosagem com guarda de deriva acumulada.** Corrige a causa raiz, mas são 21 ciclos
   de colher e joga fora o estado atual.
