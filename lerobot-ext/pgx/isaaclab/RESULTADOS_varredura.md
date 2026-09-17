# Varredura de checkpoints no simulador da Unitree

Cada modelo teve **três tentativas**, com reset da cena entre elas.

* `deslocamento_px` — quanto o cubo andou na imagem da câmera da cabeça. Abaixo de ~10 px é ruído.
* `garra_mais_fechada` — menor valor atingido pela garra direita (0 fechada, 5,4 aberta). Se não desce de ~3, ela nunca tentou agarrar.
* `area_final_sobre_inicial` — área vermelha no fim dividida pela do começo; bem acima de 1 costuma ser cubo tombado ou mais perto da câmera.


## act_oficial_lead50

`binabik-ai/act_PickPlaceRedBlock` com `--redimensiona 224` — 16/09 20:50

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 0.7 | 0.37 | 1.0 |
| 2 | 0.3 | 1.47 | 1.01 |
| 3 | 1.4 | 1.96 | 0.99 |

fechou a garra, mas não mexeu o cubo

## act_oficial_lead10

`binabik-ai/act_PickPlaceRedBlock` com `--redimensiona 224 --passos-acao 10` — 16/09 20:57

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 1.9 | 1.72 | 1.03 |
| 2 | 55.4 | 0.0 | 1.86 |
| 3 | 4.5 | 5.4 | 0.67 |

**mexeu o cubo e fechou a garra** — olhar de perto

## act_oficial_ensemble

`binabik-ai/act_PickPlaceRedBlock` com `--redimensiona 224 --ensemble 0.01` — 16/09 21:03

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 3.4 | 0.0 | 1.0 |
| 3 | 2.1 | 0.0 | 0.97 |

fechou a garra, mas não mexeu o cubo

## act_baseline_30k

`RooibosT/Sim_act_dex1_baseline` — 16/09 21:11

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 3.8 | 0.0 | 1.0 |
| 2 | 0.9 | 4.54 | 0.99 |
| 3 | 10.1 | 2.94 | 1.32 |

fechou a garra, mas não mexeu o cubo

## pi05_107demo_abs

`RooibosT/Sim_pi05_expert_only_absolute_107demo` — 16/09 21:32

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 4.7 | 0.37 | 0.99 |
| 2 | 2.7 | 4.54 | 1.0 |
| 3 | 2.5 | 5.16 | 1.0 |

fechou a garra, mas não mexeu o cubo

## pi05_107demo_rel

`RooibosT/Sim_pi05_expert_only_relative_107demo` — 16/09 21:54

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 31.4 | 0.0 | 1.0 |
| 2 | 4.3 | 0.0 | 1.02 |
| 3 | 3.3 | 0.0 | 0.97 |

**mexeu o cubo e fechou a garra** — olhar de perto

## pi05_abs

`RooibosT/Sim_pi05_expert_only_absolute` — 16/09 22:20

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 2.9 | 2.45 | 1.0 |
| 2 | 2.1 | 0.12 | 1.01 |
| 3 | 2.0 | 0.0 | 1.0 |

fechou a garra, mas não mexeu o cubo

## pi05_rel

`RooibosT/Sim_pi05_expert_only_relative` — 16/09 22:35

| tentativa | deslocamento (px) | garra mais fechada | área final/inicial |
|---|---|---|---|
| 1 | 42.5 | 0.0 | 1.09 |
| 2 | 0.5 | 0.0 | 1.0 |
| 3 | 0.3 | 0.0 | 1.0 |

**mexeu o cubo e fechou a garra** — olhar de perto
