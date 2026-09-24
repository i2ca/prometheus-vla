# O que dá para tirar do UnifoLM — e um buraco que achamos no caminho

> 13/09/2026. Pesquisa sobre a família UnifoLM da Unitree e o que dela melhora o
> nosso robô. O item mais urgente **não é** sobre a Unitree: é um buraco de
> realimentação no nosso próprio dataset, encontrado ao comparar os schemas.

---

## 1. ⚠️ 66 dimensões da nossa observação são zero constante

Medido nos 12.848 quadros de `white_cup_on_dripper_2026-08-11` e nos 956
episódios de simulação:

| canal | dims | dataset real | datasets de simulação |
|---|---:|---|---|
| `observation.state[15:29]` (juntas das mãos) | 14 | **0 de 14 variam** | 6 de 14 variam |
| `observation.left_hand_pressure` | 33 | **0 de 33** | **0 de 33** |
| `observation.right_hand_pressure` | 33 | **0 de 33** | **0 de 33** |

Enquanto isso, a **ação** da mão direita varia normalmente: desvio-padrão 0,734,
faixa -1,5 a 1,5, 56 valores distintos. Ou seja, **a mão é comandada e nunca
observada**.

### Por que isso importa

A política decide fechar a mão olhando só para as câmeras e para 15 juntas de
braço. Ela não tem como saber se a mão fechou, se pegou, ou se escorregou —
esses três estados produzem exatamente a mesma observação.

É a explicação mecânica do sintoma registrado em
[SESSAO_2026-08-18.md](SESSAO_2026-08-18.md): *"a garra fecha e não segura"*.
Não é o modelo que é fraco na preensão: **o laço de preensão não existe**. Nem
o FastWAM-D, nem o π0.5, nem o WLA-Base quando sair consertam isso, porque o
sinal não está no dado.

### E tem um agravante no co-treino

Repare na coluna da simulação: no MuJoCo o estado da mão **é** gravado (6 de 14),
no robô real não (0 de 14). Os configs `*_cotreino_sim_real.yaml` misturam os
dois. O modelo aprende a usar um canal que é informativo em simulação e zero
constante no robô — e no robô ele recebe uma entrada fora da distribuição em
que aprendeu, sem nada quebrar visivelmente.

### O que fazer

1. **Conferir se o SDK entrega o estado da Dex3.** O `dex3_g1_server.py` publica
   comando; a questão é se o `lowstate` da mão está sendo lido de volta e
   escrito no dataset. Se estiver disponível e só não estiver sendo gravado, é
   conserto de uma linha no gravador — e requer **regravar**.
2. **Idem para a pressão.** 66 dims zeradas em TODOS os datasets, inclusive o
   real, sugere que o tópico de tato nunca chegou ao gravador.
3. **Enquanto não regravar, desligue os canais mortos nos configs.** O
   `pi05depth` e o `fastwam_depth` projetam a pressão num token. Projetar uma
   constante gasta parâmetro, cria um token sem informação no prefixo, e faz o
   `pressure_proj` nascer aleatório sem nunca receber gradiente útil
   (ver [PI05_BASE.md](PI05_BASE.md) §2).
4. **No co-treino, zere o estado da mão também na simulação** até o real
   gravar. Igualar por baixo é pior que ter o sinal, e é melhor que treinar com
   um canal que muda de significado entre as duas fontes.

### Um detalhe a mais

`action[22]` (`R_thumb0`) é constante mesmo na ação: das 7 juntas da Dex3
direita, só 6 são comandadas. E as 6 se movem juntas — a mão está sendo pilotada
como garra binária, não como mão de 7 graus de liberdade.

---

## 2. 34 datasets novos no G1: 83,7 horas

Novos desde o levantamento de [DATASETS_EXTERNOS.md](DATASETS_EXTERNOS.md)
(09/09). São os `unitreerobotics/G1_WBT_*` — *Whole Body Teleoperation* —
liberados junto com o WLA-1.0:

```
34 datasets | 7.254 episódios | 9.039.599 quadros | 83,7 horas
  Inspire   15 datasets | 3.700 eps | 42,0 h
  Brainco   15 datasets | 3.400 eps | 40,6 h
  Dex1       1 dataset  |   154 eps |  1,1 h
```

Para comparar: o nosso real tem 12.848 quadros. Isso é **700× mais**.

LeRobot **v3.0**, 30 fps, Apache-2.0, sem gate — abrem direto, sem conversão de
formato.

### O schema é mais rico que o nosso

```
observation.images.head_stereo_left    [480, 640, 3]   ← par estéreo
observation.images.head_stereo_right   [480, 640, 3]
observation.images.wrist_left          [480, 640, 3]
observation.images.wrist_right         [480, 640, 3]
observation.state.ee_state             [12]   pose FK dos dois efetuadores
observation.state.hand_state           [12]   6 por mão, 0-1 aberto→fechado
observation.state.robot_q_current      [36]   7 (xyz + quat da raiz) + 29 juntas
action.ee_action                       [12]
action.hand_cmd                        [12]
action.robot_q_desired                 [36]
```

**Repare:** eles gravam `hand_state`. Nós não (§1).

### Quanto encaixa no nosso schema

`robot_q_current[7:36]` são as 29 juntas do corpo do G1 (12 pernas + 3 cintura +
14 braços) — que **não** são as nossas 29. A interseção:

| nossas dims | o que é | vem de |
|---|---|---|
| 0-13 | 14 juntas dos braços | `robot_q[7:36]`, subconjunto dos braços ✓ |
| 14 | yaw da cintura | `robot_q[7:36]`, cintura ✓ |
| 15-28 | 14 juntas Dex3 | **não existe** — Inspire/Brainco são 6 fechamentos normalizados |

**15 de 29 mapeiam exatamente.** As 14 das mãos não, e é aqui que o
`dex_retargeting` (já vendorizado) entra: converter fechamento de dedo
normalizado → ângulos de junta da Dex3 é exatamente o que ele faz. É a ponte
que transforma 83,7 horas de dado alheio em dado nosso.

### E o estéreo dá profundidade

Baixei um par de `G1_WBT_Inspire_Put_Drinks_Into_Fridge` e testei: são **RGB
coloridas** (não é o par IR de uma D435i), retificadas, e um `StereoSGBM` sem
ajuste nenhum já produz disparidade coerente — 42% dos pixels válidos, mediana
6,6 px, a garrafa destacada do fundo e o tampo da bancada num gradiente correto.

Não há calibração publicada, então isso é disparidade, não metro. Mas a escala é
recuperável **de graça**: `robot_q_current` dá a cinemática completa, os punhos
aparecem em boa parte dos quadros, e a posição 3D deles é conhecida por FK.
Casar disparidade medida com Z conhecido no pixel do punho deixa uma incógnita
só (`f·B`) contra milhares de restrições.

Isso fecha o buraco que a [DATASETS_EXTERNOS.md](DATASETS_EXTERNOS.md) apontava:
nenhuma fonte externa tinha profundidade. Agora tem 83,7 horas de onde tirar.

---

## 3. Normalizar por percentil, não por extremo

A spec do WLA usa `minmax_q` — 1º e 99º percentil — como padrão, e reserva
`minmax` (extremos) para quase nada. Medido no nosso dataset, nas dimensões que
não são constantes:

| | quanto do intervalo [-1, 1] o corpo do sinal (q01–q99) ocupa |
|---|---|
| `action` | **57%** |
| `observation.state` | **36%** |

Pior caso: `LShoulderYaw` no estado, onde min/max é `0,066..0,145` e q01/q99 é
`0,072..0,073` — **1%**. Um punhado de quadros de outlier está consumindo 99% da
faixa dinâmica dessa dimensão.

Com `min_max`, mais da metade da faixa que a rede tem para representar está
sendo gasta com cauda. Trocar para percentil é mudança de config, não de
arquitetura.

Duas regras que vêm junto na spec e valem copiar:

* **Só o xyz da pose é normalizado** (§11.1). Rotação em 6D fica crua —
  aplicar estatística de translação em rotação mistura geometrias.
* **Ao juntar tarefas, peso igual por tarefa, não por amostra** (§12.8). É
  exatamente a decisão que a [DATASETS_MULTITAREFA.md](DATASETS_MULTITAREFA.md)
  precisa tomar, e a spec dá as fórmulas de merge de média, desvio, extremos e
  quantis.

---

## 4. A ação deveria ser pose relativa, não junta

Tanto a spec do WLA quanto os datasets WBT usam **pose SE(3) do efetuador,
relativa ao estado atual**, com rotação em vetor de rotação — e não posição de
junta. O `ee_action` deles inclui a contribuição da cintura, calculada por FK
desde o link raiz.

O nosso ACT-D e o FastWAM-D predizem junta. A diferença não é cosmética:

* pose relativa é invariante à configuração de partida — a mesma ação "aproxime
  3 cm da caneca" vale de qualquer pose inicial, e em junta são números
  diferentes toda vez;
* é a representação que todo prior publicado usa (π0.5, GR00T, WLA), então é o
  que permite aproveitar peso pré-treinado;
* é o que torna os 83,7 h do §2 utilizáveis mesmo com mãos diferentes: o braço
  fala a mesma língua.

Temos o pinocchio e o FK pronto em
[`../robot/unitree_g1/robot_control/g1_arm_ik.py`](../robot/unitree_g1/robot_control/g1_arm_ik.py)
(`framesForwardKinematics` + `oMf`). O custo é reprocessar os datasets e treinar
de novo, não escrever cinemática.

---

## 5. A arquitetura do WLA-1.0, e o que ela valida

Da página do projeto: **WLA-1.0 = backbone do ER-Flow + um action expert MMDiT**,
treinado em ~2.500 h e 64 tarefas (54 de mesa, 10 de corpo inteiro).

Isso é convergente com o que vocês já escolheram no `pi0_depth`: flow matching
com um expert separado. A diferença é o que alimenta o expert — no π0.5 é
linguagem + imagem + estado; no WLA é isso mais **predição de região dinâmica
futura**, que é o modelo de mundo comprimido em tokens de máscara em vez de
vídeo inteiro.

O barato disso, comparado ao FastWAM-D: prever uma máscara de onde a cena vai
mudar custa muito menos que prever o vídeo com um DiT de 5 B. Se o objetivo do
world model é "saber o que vai se mover", a máscara pode ser suficiente — e
caberia numa GPU de verdade. É a ablação mais interessante que dá para desenhar
a partir do que eles publicaram.

---

## 6. ER-1 fora do laço, mas dentro do pipeline

O ER-1 roda em 2,92 GB e leva 854 ms por pergunta — não serve para controle a
30 Hz (§8 de [UNIFOLM_WMA.md](UNIFOLM_WMA.md)). Serve para três coisas offline:

1. **Auditoria de dataset.** Rodar sobre os episódios gravados e marcar aqueles
   em que o objeto da tarefa não aparece, ou aparece fora do campo — barato de
   fazer e pega demo ruim antes do treino.
2. **Régua para o grounding do FastWAM-D**, comparando na primeira metade dos
   episódios, onde o ER-1 é inequívoco.
3. **Rótulo automático de tarefa.** O nosso dataset tem UMA string de `task` nos
   24 episódios, e é a causa raiz de o condicionamento por linguagem não
   funcionar. O ER-1 descreve a cena; dá para gerar variação de comando a partir
   do que está de fato no quadro, em vez de inventar.

---

## Ordem que eu seguiria

| # | o quê | custo | por que primeiro |
|---|---|---|---|
| 1 | Conferir e consertar a gravação de estado da mão e da pressão (§1) | baixo + regravar | nada mais importa enquanto o laço de preensão não existir |
| 2 | Desligar os canais mortos nos configs (§1.3) | trivial | para de treinar projeção de constante hoje |
| 3 | Trocar para normalização por percentil (§3) | config | recupera 40-60% da faixa dinâmica |
| 4 | Trazer os `G1_WBT_*` com `dex_retargeting` (§2) | alto | 700× mais dado, e é o que resolve os 24 episódios |
| 5 | Ação em pose relativa (§4) | alto | destrava prior pré-treinado, incluindo o WLA quando sair |
