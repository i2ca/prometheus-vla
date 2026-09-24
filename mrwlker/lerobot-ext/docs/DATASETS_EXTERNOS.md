# Datasets externos para a tarefa do café

> Levantamento de 09/09/2026. O problema não é o tamanho da rede — é que 24
> episódios reais não sustentam clonagem de comportamento. Este documento lista
> o que existe publicado **no nosso robô** (G1 + Dex3), o que custa para usar, e
> o que NÃO serve.

---

## 0. A regra que decide tudo

`lerobot/datasets/aggregate.py:47` exige `fps`, `robot_type` e o dicionário de
`features` **idênticos** para juntar datasets, e `MultiLeRobotDataset` está
desligado neste fork (`lerobot/datasets/factory.py:115`). Ver
[DATASETS_MULTITAREFA.md](DATASETS_MULTITAREFA.md).

Consequência: **qualquer** dataset de fora só entra depois de convertido para o
nosso schema v2 exato — 29 dims, `head_camera` 848×480, `head_camera_depth`,
`right_wrist_camera` 224×224, `left/right_hand_pressure` 33. Não existe atalho
de "treinar com os dois ao mesmo tempo".

O nosso ponto de partida hoje:

| | episódios | quadros |
|---|---|---|
| real (`white_cup_on_dripper_2026-08-11`) | 27 (24 bons) | 12.848 |
| sim (MuJoCo, `gerar_dataset_mujoco.py`) | 275 | ~87 k |

---

## 0.1 "v2" aqui é o NOSSO schema, não o formato do LeRobot

Duas coisas diferentes carregam número de versão, e confundir as duas custa caro:

| | o que é | onde vive | onde estamos |
|---|---|---|---|
| **schema v2 do G1** | nossa decisão de robô: 29 dims (yaw do tronco) + câmera de pulso | [SCHEMA_G1_V2.md](SCHEMA_G1_V2.md) | v2 |
| **`codebase_version`** | formato de arquivos do LeRobot | `meta/info.json` | **v3.0** |

O formato já é v3.0 em tudo que é nosso — conferido no `meta/info.json` de
`white_cup_on_dripper_2026-08-11` e dos datasets de simulação — e
`CODEBASE_VERSION = "v3.0"` em `lerobot/datasets/dataset_metadata.py:60`. Não há
nada para migrar do nosso lado.

Do lado de fora, o formato decide quanto trabalho cada fonte dá:

| fonte | `codebase_version` | o que é preciso |
|---|---|---|
| `unitreerobotics/G1_Dex3_*` | **v3.0** | nada — abre direto |
| `USC-PSI-Lab/humanoid-everyday` | v2.1 | `lerobot/scripts/convert_dataset_v21_to_v30.py` antes de tudo |
| `agibot-world/AgiBotWorld2026` | v2.1 | idem (e a licença já o descarta) |

Ou seja: **os 2.851 episódios da Unitree já estão no formato certo**; o que falta
neles é o nosso schema de robô, e é isso que o `converte_dex3_unitree.py` faz.

---

## 1. Camada A — mesmo corpo, mesmas mãos, encaixa direto

### 1.1 Datasets oficiais da Unitree (`unitreerobotics/G1_Dex3_*`)

LeRobot v3.0, `robot_type: Unitree_G1`, **30 fps**, `action`/`observation.state`
de **28 dims**, 4 câmeras RGB de 640×480 (`cam_left_high`, `cam_right_high`,
`cam_left_wrist`, `cam_right_wrist`). Licença **Apache-2.0**, sem gate.

| dataset | eps | quadros | GB | cams | o que a tarefa É (do card da Unitree) |
|---|---:|---:|---:|:---:|---|
| **Pouring** | 311 | 121.587 | 4,3 | 4 | pegar a garrafa de água e **simular despejar num copo de vidro** (7×8 cm). Bimanual: a esquerda segura o copo, a direita despeja |
| **ObjectPlacement** | 210 | 98.266 | 3,5 | 4 | pegar pasta de dente e saco de lixo da mesa e **pôr dentro do cesto azul** |
| **ToastedBread** | 418 | 352.022 | 14,8 | 4 | pôr o pão da bandeja **na torradeira**; depois **entregar a torrada à pessoa** |
| **CameraPackaging** | 201 | 256.253 | 8,5 | 4 | encaixar a RealSense D-405 **na caixa** e fechar a tampa |
| BlockStacking | 301 | 281.196 | 9,2 | 4 | empilhar 3 cubos de 5 cm (vermelho, amarelo, azul) sobre a fita preta |
| ~~GraspSquare~~ | ~~301~~ | ~~281.196~~ | ~~9,2~~ | 4 | ⚠ **duplicata do BlockStacking** |
| PickApple | 201 | 152.569 | 3,0 | **2** | `Put the apple into the plate.` |
| PickBottle | 202 | 176.774 | 3,5 | **2** | `Put the bottle into the plate.` |
| PickTissue | 205 | 166.756 | 3,1 | **2** | `Put the tissue paper into the plate.` |
| PickDoll | 203 | 300.557 | 6,7 | **2** | `Put the doll into the plate.` |
| PickSnack | 200 | 163.487 | 3,4 | **2** | `Put the snack into the plate.` |
| PickCharger | 200 | 123.260 | 2,3 | **2** | `Put the charger into the plate.` |
| PickGum | 199 | 113.592 | 2,1 | **2** | `Put the gum into the plate.` |

**⚠ Os sete `Pick*` NÃO TÊM CÂMERA DE PULSO** — só `cam_left_high` e
`cam_right_high` (`robot_type: Unitree_G1_Dex3`, campanha de gravação diferente
das outras). A nossa `right_wrist_camera` é a que mostra a xícara de perto no
momento da pega; preenchê-la com constante nesses 1.410 episódios seria o mesmo
veneno da profundidade. Ou ficam de fora, ou a fase de co-treino roda sem a
câmera de pulso. O `converte_dex3_unitree.py` recusa esses datasets de propósito.

**⚠ `GraspSquare` é o `BlockStacking` republicado.** Mesma contagem de episódios,
mesma contagem de quadros, e o `data/chunk-000/file-000.parquet` dos dois tem
**exatamente 63.247.041 bytes**; os `meta/episodes` diferem em 5 bytes, que é o
comprimento da string de tarefa. Por cima, o rótulo do `GraspSquare` diz
`camera packaging`, que é a tarefa de um TERCEIRO dataset. Não baixe os dois, e
não confie na string de tarefa deste sem abrir um vídeo.

**Total real: 2.851 episódios / 2.306.319 quadros ≈ 21,4 h / 64,4 GB**, do MESMO
robô. Contra os nossos 24 episódios reais.

### 1.2 Humanoid Everyday (`USC-PSI-Lab/humanoid-everyday`)

Apache-2.0, sem gate, LeRobot v2.1, 30 fps. É o G1 de 29 DoF **com Dex3-1**, e
é a única fonte externa que tem **profundidade real** e tátil.

- subconjunto G1: **4.064 episódios / 1,78 M quadros / 247 tarefas**
- `action` 28 = `arm_joints` (14) + `hand_joints` (14) — o nosso espaço v1
- `observation.leg_joints` tem **15** valores (12 pernas + 3 da cintura) → o yaw
  do tronco existe no dado bruto
- `observation.depth.egocentric` 480×640 float32, valores até ~3261 → **milímetros**,
  a mesma convenção de [PROFUNDIDADE_NATIVA.md](PROFUNDIDADE_NATIVA.md)
- tem ainda LiDAR, IMU, odometria e `observation.tactile.values` (18×4)

A fatia de café, medida episódio a episódio no `meta/episodes.jsonl`:

| tarefa (G1) | eps | quadros |
|---|---:|---:|
| `Basic/pour_water_from_a_kettle_into_a_cup` | 40 | 20.565 |
| `Basic/pour_water_from_a_kettle_into_a_cup_g1` | 40 | 20.565 |
| `Locomanip/walk_towards_a_desk_and_pour_water_from_the_pot_to_a_cup` | 40 | 33.446 |
| `Locomanip/walk_towards_a_desk_and_pick_up_the_boiling_pot_from_its_base` | 40 | 23.146 |
| `Locomanip/walk_towards_a_desk_and_turn_on_the_boiling_pot` | 40 | 20.706 |
| `Basic/pour_sunflower_seeds_on_plate_g1` | 40 | 20.742 |
| `Articulated/close_a_kettle_lid_g1` | 40 | 10.098 |
| `Locomanip/pick_up_a_kettle_place_it_on_base_and_turn_on` | 3 | 5.785 |
| `Locomanip/place_kettle_near_coffee_maker` | 3 | 3.048 |
| `Locomanip/walk_towards_coffee_maker_and_close_lid` | 3 | 1.602 |
| **total** | **289** | **159.703** |

O subconjunto H1 (mãos Inspire) tem mais tarefas de chaleira
(`lift_a_kettle_from_base`, `place_the_kettle_on_its_base`,
`press_start_boiling_button`, `pour_water_into_a_cup`), mas é **outro corpo** —
serve só para pré-treino de visão/linguagem, não para o merge.

**Três armadilhas medidas:**

1. **Só uma câmera.** `observation.images.egocentric` 480×640. Não há câmera de
   pulso — a nossa `right_wrist_camera` teria que ir zerada, e uma feature de
   imagem constante é pior que ausente (a política aprende a ignorá-la ou, pior,
   a usá-la como pista de "isto é dado externo").
2. **Peso.** Um episódio de 506 quadros ocupa **123 MB** (profundidade e LiDAR
   crus dentro do parquet) — ~243 KB/quadro. A fatia de café dá **~39 GB**; o
   subconjunto G1 inteiro passaria de 400 GB. Baixar por episódio, nunca o repo.
3. **`instruction` está quebrado.** Os 8.949 episódios trazem a MESMA string
   (`"the robot uses its right hand to hold the eraser..."`). Use `meta/tasks.jsonl`
   (`task_index` → nome), e reescreva o nome em linguagem natural na conversão.
   O repo `USC-PSI-Lab/Humanoid-Everyday-G1` (só G1) é mais leve mas **perdeu a
   profundidade e o tátil** — para o nosso caso o repo grande é o que interessa.

---

### 1.3 Encaixe no desafio do café, etapa por etapa

Nosso roteiro é: **pegar a xícara → pôr no coador → pegar a chaleira → despejar**.
Nenhum episódio público faz isso. O que existe é gesto por gesto:

| etapa | melhor correspondência | quão perto |
|---|---|---|
| pegar a xícara | `Pick*` (`Put the X into the plate`), ObjectPlacement | perto: pega de objeto pequeno sobre mesa branca |
| **pôr no coador** | **CameraPackaging** (encaixar a câmera na cavidade da caixa) | é o único encaixe preciso em alvo com tolerância pequena |
| pegar a chaleira | Pouring (pega da garrafa), PickBottle | **médio**: garrafa de 350 ml pega pelo corpo ≠ chaleira pega pela alça |
| **despejar** | **Pouring** | é literalmente o gesto, e bimanual: esquerda segura o copo, direita inclina |
| ligar a cafeteira | ToastedBread (pôr o pão, mexer na torradeira) | interação com eletrodoméstico sobre a bancada |

O que **não** existe em lugar nenhum: coador, café, e a pega pela alça de uma
chaleira cheia. Isso continua tendo que sair das nossas demonstrações.

**Uma observação que vale mais do que parece:** a cena da Unitree é mesa branca,
parede clara e câmera de cabeça olhando de cima — visualmente **mais próxima do
nosso laboratório do que o nosso próprio MuJoCo** (mesa cinza chapada, chão
xadrez azul, sombra dura; ver [ARTEFATOS_SIM.md](ARTEFATOS_SIM.md)). Para
fechar o vão sim→real, 1.429 episódios reais de mesa branca provavelmente valem
mais que os 275 sintéticos.

**O que esperar honestamente.** Nenhum destes episódios ensina "place the white
cup on the dripper". O que eles compram é: (1) prior de manipulação do MESMO
corpo, 9 h em vez dos 13 min que temos; (2) variedade de cena e objeto, que é o
que impede a rede de decorar 24 trajetórias e travar no laço fechado; (3) várias
strings de tarefa, sem as quais o condicionamento por texto não tem no que se
condicionar. O ajuste fino para o café continua sendo nosso.

---

## 2. Compatibilidade: o que eu conferi

### 2.1 A ordem das juntas bate — bit a bit

Os `names` do `meta/info.json` da Unitree são exatamente os nossos enums:

```
kLeftShoulderPitch..kRightWristYaw      = G1_29_JointArmIndex        (dims 0–13)
kLeftHandThumb0,1,2, Middle0,1, Index0,1 = Dex3_1_Left_JointIndex     (robot/unitree_g1/g1_utils.py:202)
kRightHandThumb0,1,2, Index0,1, Middle0,1 = Dex3_1_Right_JointIndex   (robot/unitree_g1/g1_utils.py:213)
```

Inclusive a inversão do lado direito (índice antes do médio), que é a pegadinha
que quebraria tudo em silêncio. **Nenhuma permutação é necessária.**

De 28 → 29 é inserir o yaw do tronco na **dim 14** (ver
[SCHEMA_G1_V2.md](SCHEMA_G1_V2.md)):

- Unitree: 0.0 constante (tronco fixo na coleta) — honesto, é o que o robô fez
- Humanoid Everyday: o valor real está em `observation.leg_joints`. Quase certo
  que é o índice 12 (enum `G1_29_JointIndex`: 0–11 pernas, 12 `kWaistYaw`), mas
  **confirme num episódio `Locomanip` em que o tronco gira** antes de confiar

### 2.2 O que NÃO bate

| nossa feature | Unitree Dex3 | Humanoid Everyday |
|---|---|---|
| `head_camera` 848×480 | `cam_left_high` 640×480 — **outro FOV**, letterbox, não esticar | `egocentric` 640×480, idem |
| `head_camera_depth` | **não existe** | ✅ existe, em mm |
| `right_wrist_camera` 224×224 | `cam_right_wrist` 640×480 → redimensionar | **não existe** |
| `left/right_hand_pressure` (33) | não existe | tátil 18×4, layout diferente |
| dim 14 (yaw) | 0.0 | recuperável do `leg_joints` |

Nenhuma fonte externa fecha as três modalidades. Essa é a decisão real deste
documento, não a escolha do dataset.

### 2.3 O conversor, e a prova de que o merge passa

[`../converte_dex3_unitree.py`](../converte_dex3_unitree.py) faz a adaptação
inteira: 28 → 29 dims, encaixe das câmeras, profundidade e pressão preenchidas.

```bash
python converte_dex3_unitree.py \
    --origem unitreerobotics/G1_Dex3_Pouring_Dataset \
    --destino meu_dataset/ext_pouring --confere
```

O schema não é escrito no script: é **lido do `meta/info.json` do dataset real**,
para não existir uma terceira cópia que divirja. O `--confere` fecha o ciclo
chamando `features_equal_for_merge`.

Validado em 09/09/2026 contra uma origem sintética com o `meta/info.json` real da
Unitree (o download de 520 MB por arquivo de vídeo é lento demais para um teste):

```
braços iguais (0-13): True
dim 14 (yaw) = 0.0
mãos deslocadas (14-27 → 15-28): True
head_camera        (3, 480, 848)   right_wrist_camera (3, 224, 224)
features_equal_for_merge contra white_cup_on_dripper_2026-08-11: True
validate_all_metadata PASSOU — fps 30, robot_type unitree_g1_dex3, 12 features
```

`validate_all_metadata` é a função de `aggregate.py:121` — a mesma que recusou o
co-treino em 03/09. Passando ela, o merge passa.

**Uma descoberta do teste:** a profundidade gravada como zeros **não volta como
zero**. O decodificador do LeRobot aplica a quantização logarítmica inversa e
devolve **10,0 m uniformes** (o `video.depth_max` do `info.json`). É uma parede
plana a 10 metros, não um "sem dado" — mais um motivo para estes episódios só
entrarem em treino com profundidade desligada.

**Baixe na athena, não no notebook.** Cada arquivo de vídeo destes datasets tem
~520 MB e o link daqui rendeu ~250 kB/s: os 14,3 GB da mistura levariam ~16 h.

---

## 3. O buraco da profundidade — três saídas

**(a) Duas fases, sem profundidade primeiro.** Co-treino em RGB puro
(`pi05depth` com `use_depth_3d: false`, ou FastWAM-D com `depth_mode: off`) sobre
mistura externa + nossa; depois fine-tune com profundidade só nos nossos dados.
Mais barato, não inventa dado, e reaproveita o `pi05_cotreino_sim_real.yaml`
quase inteiro. **É por onde começar.**

**(b) Estimar profundidade métrica do RGB externo** (Depth Anything V2 metric,
UniDepth, Metric3Dv2) e preencher `head_camera_depth`. Dá o pipeline completo,
mas o ruído de um estimador não é o ruído da D435i — a política pode aprender a
distinguir as duas origens. Só depois de (a) funcionar.

**(c) Humanoid Everyday para profundidade, Unitree para pulso.** Mistura as duas
lacunas em vez de resolver uma. Não recomendo como primeiro passo.

Pressão: em qualquer caminho vai zerada, como já é o caso do dataset sim — e as
políticas do co-treino ficam com `use_pressure: false`, que é o que os configs
atuais já fazem.

---

## 4. Plano recomendado

**Etapa 0 — o conversor.** Um script (`converte_externo.py`), dois adaptadores
(Unitree, Humanoid Everyday), saída no nosso `features` v2 exato. Com `--dry-run`
comparando o dicionário de features contra o nosso, como o
`build_multitask_dataset.py` já faz. É aqui que mora todo o trabalho.

**Etapa 1 — mistura "café v1"**, tudo real, tudo no nosso robô, **e tudo com as
quatro câmeras** (a versão anterior deste plano incluía PickBottle e PickApple,
que não têm pulso — corrigido em 09/09):

| fonte | eps | quadros | por que |
|---|---:|---:|---|
| Unitree Pouring | 311 | 121.587 | o gesto de despejar, e bimanual |
| Unitree ObjectPlacement | 210 | 98.266 | objeto → dentro de recipiente |
| Unitree CameraPackaging | 201 | 256.253 | encaixe preciso numa cavidade — o mais perto de "copo no coador" |
| Unitree ToastedBread | 418 | 352.022 | interação com eletrodoméstico + entrega a pessoa |
| HE fatia café (G1) | 289 | 159.703 | chaleira, coador, cafeteira, com profundidade real |
| Unitree BlockStacking | 301 | 281.196 | opcional: pega precisa e pilha; engorda sem aproximar do café |
| **externo (sem o BlockStacking)** | **1.429** | **987.831** | ~9,1 h |
| nosso real | 24 | ~11 k | oversample 5–10× na amostragem |
| nosso sim | 275 | ~87 k | ver [ARTEFATOS_SIM.md](ARTEFATOS_SIM.md) antes de confiar nele |

~70 GB de download (31,1 GB Unitree + ~39 GB da fatia HE).

**Etapa 2 — treinar o π0.5** (já implementado, `policies/pi0_depth/`) na mistura,
depois fine-tune nos nossos dados. Ganho colateral: a mistura tem **várias strings
de tarefa**, e sem isso nenhum VLA aprende condicionamento por linguagem — é a
limitação que o `override_task` dos configs atuais esconde.

**Etapa 3 — não espere que dado resolva o travamento sozinho.** O diagnóstico
registrado no `pi05_cotreino_sim_real.yaml` é erro que se acumula: toda ação leva
o corpo para fora do que foi demonstrado e não há comportamento de recuperação.
Mais demonstrações limpas reduzem a taxa, não a causa. Vale gravar episódios que
**começam fora da pose ideal** e recuperam — é barato e ataca exatamente isso.

---

## 5. O que NÃO usar (e por quê)

| dataset | por que fica de fora |
|---|---|
| `unitreerobotics/Z1_Dual_Dex1_PourCoffee_Dataset` (331 eps, 443 k quadros) | literalmente "pour coffee", mas é braço Z1 + garra. Só como pré-treino de VLA |
| `lerobot/aloha_static_coffee` (50 eps) | cápsula de café em ALOHA, 50 fps, outro corpo |
| `agibot-world/AgiBotWorld2026` / `-Beta` | escala enorme, mas **CC-BY-NC-SA 4.0** (não comercial + share-alike, que contamina o derivado) e a Beta é *gated*. Robô AgiBot G2 |
| `x-humanoid-robomind/RoboMIND` | Apache-2.0 porém *gated*, e o humanoide é o Tien Kung |
| RoboCasa (`CoffeeSetupMug`, `CoffeeServeMug`, `CoffeePressButton`) | é MuJoCo, como o nosso sim, mas com braço Franka. Serve como **fonte de cenário e de assets** de cafeteira para o `gerar_dataset_mujoco.py`, não como trajetória |

---

## 6. Para acompanhar

**RoboTacDex** ([arXiv 2606.31836](https://arxiv.org/abs/2606.31836), junho/2026)
— Unitree G1, 6 k trajetórias, 19 tarefas, RGB **e profundidade** multi-vista
**e tátil**, com anotação semântica. É a única coisa publicada que fecha
exatamente o nosso stack de sensores. O artigo diz "will be open-sourced soon" e
não há repositório ainda. Vale checar de tempos em tempos.
