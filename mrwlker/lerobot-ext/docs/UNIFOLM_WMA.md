# UnifoLM da Unitree — o que dá para rodar hoje, e o que não dá

> 13/09/2026. Resumo em uma linha: **a política do UnifoLM-WLA-1.0 não foi
> publicada**; o que existe da família e fecha o laço de controle é o
> antecessor, o UnifoLM-WMA-0, e mesmo ele não é zero-shot no nosso robô.

---

## 1. O estado do WLA-1.0

O [`unitreerobotics/unifolm-wla`](https://github.com/unitreerobotics/unifolm-wla)
tem, na data desta nota, **cinco arquivos e nenhuma linha de código**: dois
READMEs, a licença e a spec de processamento de ação/estado. O plano de
open-source do próprio README:

| | o quê | estado |
|---|---|---|
| ✅ | `UnifoLM-ER-1` — 4 B, raciocínio corporificado sobre Qwen3-VL-4B | publicado 11/09 |
| ✅ | `UnifoLM-ER-Flow` — 4 B, + região dinâmica + tokens de ação discretos | publicado 11/09 |
| ❌ | **`UnifoLM-WLA-Base`** — a política de 6 B | **não publicado** |
| ❌ | **Post-Train Code** | **não publicado** |

E o ER-Flow, apesar de emitir ação, emite **token discreto de RVQ**. Os
decodificadores RVQ (um por componente: pose do efetuador, juntas da mão,
juntas do corpo inferior) **não estão no repositório**. Sem eles o token não
vira radiano, e o modelo é um preditor de região dinâmica — não uma política.

Conclusão prática: hoje não existe nada do WLA-1.0 que feche o laço de
controle. O que dá para testar do WLA-1.0 é percepção (§7).

## 2. O que foi baixado

| repositório | tamanho | onde | precisa de login |
|---|---:|---|---|
| `unitreerobotics/UnifoLM-ER-1` | 8,3 GB | `~/.cache/huggingface/hub` | não |
| `unitreerobotics/UnifoLM-WMA-0-Base` | 10,4 GB | — | **sim, gated** |
| `unitreerobotics/UnifoLM-WMA-0-Dual` | 16,7 GB | `~/.cache/huggingface/hub` | **sim, gated** |

Os dois do WMA estão com `gated: auto` — basta estar logado e clicar em
*Agree* na página; a aprovação é automática:

```bash
hf auth login
# abra e aceite:
#   https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual
#   https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base
hf download unitreerobotics/UnifoLM-WMA-0-Dual --exclude 'assets/*'
```

O código vive no submódulo [`../../unifolm-wma`](../../unifolm-wma), fixado no
commit em que foi vendorizado — mesma regra do `lerobot` e do
`unitree_sdk2_python`.

## 3. O que o WMA-0 é, e o que ele perde para o FastWAM-D

São primos: os dois preveem o vídeo do futuro e usam essa previsão para
decidir a ação. As diferenças que mudam o resultado:

| | FastWAM-D (nosso) | UnifoLM-WMA-0 |
|---|---|---|
| modelo de mundo | Wan2.2, 5 B, DiT | DynamiCrafter, UNet 3D, ~1,4 B |
| cabeça de ação | dentro do próprio DiT | `ConditionalUnet1D` separada (diffusion policy) |
| **profundidade** | **métrica, no latente** | **nenhuma** |
| **tato** | **pressão das duas mãos** | **nenhum** |
| câmeras | cabeça + pulso + depth | **uma só**, RGB |
| resolução | 224×224 | 320×512 |
| condicionamento | texto (CLIP do Wan) | texto + imagem (OpenCLIP) + etiqueta de embodiment |
| pesos treinados no G1 | não | **sim** (o Dual viu `G1_Pack_Camera`) |

A troca é direta: o WMA-0 traz pesos que já viram este robô, e cobra a
profundidade e o tato — as duas coisas que
[PROFUNDIDADE_NATIVA.md](PROFUNDIDADE_NATIVA.md) custou a acertar. Não é
substituto do FastWAM-D; é uma segunda opinião sobre a mesma pergunta.

## 4. ⚠️ A armadilha: a cabeça de ação nasce aleatória

O `agent_action_dim` e o `agent_state_dim` do WMA nascem em **16**. O nosso
schema (SCHEMA_G1_V2) tem **29**. O README da Unitree manda mudar os dois — e
não diz o que isso quebra.

`agent_action_dim` é o `input_dim` do `ConditionalUnet1D`, que **é** a cabeça
de ação. Mudando 16 → 29, as camadas de entrada e saída dela deixam de casar
com o checkpoint. E o `load_model_checkpoint` do `real_eval_server.py` carrega
com `strict=False`:

```python
model.load_state_dict(state_dict, strict=False)   # sem uma linha de aviso
```

Isso não é dedução — está no arquivo. Os tensores que fazem fronteira com a
ação no `unifolm_wma_dual.ckpt`:

```
(1024, 16)  state_projector.projector.0.weight
(1024, 16)  action_projector.projector.0.weight
  (32, 16)  model.diffusion_model.action_unet.proj_in_horizon.0.weight
  (16, 32)  model.diffusion_model.action_unet.proj_out_horizon.1.weight
  (32, 16)  model.diffusion_model.state_unet.proj_in_horizon.0.weight
  (16, 32)  model.diffusion_model.state_unet.proj_out_horizon.1.weight
```

Todo `16` ali vira `29` no nosso config. Seis tensores que não carregam — e
são justamente a entrada e a saída da ação.

Ou seja: **o WMA-0-Dual não é zero-shot no nosso robô.** O que transfere é o
modelo de mundo (UNet 3D, VAE, CLIP); a política tem que ser pós-treinada.

É a mesma armadilha de [PI05_BASE.md](PI05_BASE.md) §1.2, onde 657 de 812
tensores carregavam em silêncio e o π0.5 treinou do zero por semanas. O
contador que faltava lá está escrito aqui:

```bash
cd unifolm-wma
python ../lerobot-ext/wma/confere_checkpoint.py \
    --config ../lerobot-ext/config/wma/treino_g1_dex3.yaml \
    --ckpt ~/.cache/huggingface/hub/models--unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/*/unifolm_wma_dual.ckpt
```

Se a linha **"cabeça de ação (ConditionalUnet1D)"** aparecer no bloco do que
nasce aleatório, é isso — e é o esperado com 29 dimensões.

## 5. O WMA-0 não cabe no notebook

Medido no `unifolm_wma_dual.ckpt` (não estimado — é `torch.load` com `mmap`):

```
3646 tensores | 4,20 B parâmetros | todos float32 | 16,78 GB

  2516,9 M  10,07 GB  model              (UNet 3D + cabeça de ação)
   683,8 M   2,74 GB  embedder           (OpenCLIP de imagem)
   506,5 M   2,03 GB  dp_ema_model       (EMA da cabeça, também instanciada)
   354,0 M   1,42 GB  cond_stage_model   (OpenCLIP de texto)
    83,7 M   0,33 GB  first_stage_model  (VAE)
```

O `world_model_interaction.py` roda em **fp32** — não há flag de half — e faz
`model.cuda(gpu_no)` com tudo junto, EMA inclusa.

| | precisa | tem no notebook |
|---|---:|---:|
| VRAM (fp32, como o script roda) | 16,78 GB | **8,06 GB** |
| VRAM se fosse fp16 | 8,39 GB | **8,06 GB** |
| RAM (o `torch.load` acontece na CPU antes do `.cuda()`) | 16,78 GB | **14 GB** |

Não cabe por nenhum caminho: nem na GPU, nem em fp16 hipotético, nem na CPU.
Fazer caber exigiria mexer no código deles (half, offload, ou device
configurável) — e não é uma linha: o `.cuda()` está espalhado pelo
`image_guided_synthesis`.

**Este teste espera a athena.** Tudo o que vem antes dele já está pronto e
verificado aqui: env, checkpoint, configs e launchers.

## 6. Como rodar

### 6.1 Ambiente (separado, de propósito)

O WMA pede python 3.10.18 e pytorch-lightning; o `prometheus-vla` é 3.12 com
LeRobot 0.6.1. Juntar os dois troca um problema resolvido por um novo. Quem
conversa entre eles é a ponte ZMQ, não o `import`.

```bash
bash lerobot-ext/install-unifolm-wma.sh
```

**Os pins de torch do `pyproject.toml` deles não servem em GPU nova.** Eles
fixam `torch==2.3.1` e `xformers==0.0.27`, cujos wheels trazem kernels até
**sm_90**. O notebook (RTX 5070 Laptop) é **sm_120** — Blackwell. O sintoma
não é um erro de instalação: é

```
CUDA error: no kernel image is available for execution on the device
```

depois de carregar os pesos, e `torch.cuda.is_available()` responde `True` o
tempo todo. Confira a sua placa com:

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

O install resolve instalando o `xformers` pelo índice `cu128`, que arrasta o
torch e o torchvision casados, e depois `pip install -e . --no-deps` para o
pyproject não reintroduzir o 2.3.1 por cima.

**E o xformers não é opcional.** Sem ele, a atenção cruzada com imagem cai em
`modules/attention.py:128`, que é literalmente `assert 1 > 2`. O `try/except`
do topo do arquivo só desliga o caminho eficiente; o outro caminho não existe.

Duas coisas do README deles que o install NÃO faz, de propósito:

  * `conda install pinocchio=3.2.0` — nada em `src/` ou `scripts/evaluation/`
    importa pinocchio; é herança do `unitree_deploy`, que fala com o robô.
    Quem faz IK aqui é o `prometheus-vla`.
  * `pip install -e external/dlimp` — arrasta o TensorFlow, e o dlimp só serve
    para preparar dados do Open-X. Nada do treino ou da inferência o importa.

### 6.2 O primeiro teste: simulação interativa (na athena)

Este é o teste a fazer antes de qualquer treino, e o único que mede o modelo
com **todos** os tensores carregados: ele roda o exemplo DELES
(`unitree_g1_pack_camera`, G1 com garra, 16 dims) no checkpoint DELES. Nada
nasce aleatório, ao contrário do nosso config de 29 dims (§4).

Precisa de ~17 GB de VRAM (§5), então é na athena — não no notebook.

```bash
bash lerobot-ext/maquinas/athena/launch_wma_interacao.sh 0
```

Sai um mp4 do futuro previsto. A pergunta que ele responde: **o modelo de
mundo da Unitree entende uma cena deste robô?** Se entender, pós-treinar a
política tem chance. Se não, não tem — e você economizou o treino.

### 6.3 Converter o nosso dataset

O `prepare_data/prepare_training_data.py` da Unitree lê LeRobot **v2.1** (um
parquet e um mp4 por episódio). Os nossos datasets são **v3.0**: tudo
concatenado, delimitado por índice e timestamp em `meta/episodes/`. Rodar o
script deles aqui dá `FileNotFoundError` no primeiro `episode_000000.parquet`.

```bash
python pontes/wma/converte_dataset_wma.py \
    --origem meu_dataset/white_cup_on_dripper_2026-08-11 \
    --destino /data/wma_data \
    --nome white_cup_on_dripper \
    --confere
```

`--confere` reconta os quadros de cada mp4 recortado contra o comprimento do
episódio no parquet. **Use sempre.** Corte de vídeo por timestamp erra por um
quadro com facilidade, e um episódio com vídeo mais curto que o estado treina
imagem e ação fora de fase — sem erro nenhum. Medido nos 27 episódios do
`white_cup_on_dripper`: bate em todos.

Profundidade, tato e câmera de pulso ficam de fora — o WMA não tem entrada
para eles (§3).

### 6.4 Pós-treino

```bash
cd unifolm-wma
# ajuste `name` e `save_root` em scripts/train.sh e aponte --base para:
#   ../lerobot-ext/config/wma/treino_g1_dex3.yaml
bash scripts/train.sh
```

O config já vem com as quatro mudanças marcadas `# PROMETHEUS`: as duas
dimensões em 29, o `data_dir`, o dataset e o batch em 2 (5 B de difusão de
vídeo em 320×512×16 quadros não cabe em 8).

### 6.5 Inferência com o cliente que já existe

O servidor da Unitree é FastAPI; o nosso cliente é ZMQ e tem dentro dele o
laço de controle, a leitura das câmeras, o mosaico e a fila de ações.
Reescrevê-lo para HTTP seria jogar isso fora para ganhar nada, então a
`pontes/wma/ponte_wma.py` traduz:

```
cliente (seu PC) ──ZMQ 5600──> ponte ──HTTP 8000──> real_eval_server (athena)
```

```bash
# na athena:
bash lerobot-ext/maquinas/athena/launch_wma_server.sh 0 /data/train_output/wma_g1_dex3/checkpoints/last.ckpt
# no seu PC: o MESMO cliente de sempre
python init_lerobot_inference_fastwamd_client.py --host=<athena> --port=5600
```

A ponte descarta profundidade e tato (e diz quantas vezes) e não responde
`want_debug` — o painel de atenção lê tensores internos do DiT do Wan, que não
existem aqui.

## 7. Preparando o terreno para o WLA-Base

A spec de ação/estado
([`docs/robot_action_state_processing_en.md`](https://github.com/unitreerobotics/unifolm-wla/blob/main/docs/robot_action_state_processing_en.md))
saiu antes dos pesos, e é o contrato que o WLA-Base vai cobrar: **ação de 54
dimensões, estado de 60**, com máscara booleana dizendo quais módulos o robô
tem.

Onde o nosso schema de 29 entra — e onde ele não entra:

| slot do WLA | dim | o nosso | observação |
|---|---:|---|---|
| `[0:6]` efetuador esq. (ação) | 6 | 7 juntas do braço esq. | **exige FK**: a ação vira pose SE(3) RELATIVA, não junta. Temos o pinocchio em `robot/unitree_g1/robot_control/g1_arm_ik.py` |
| `[6:7]` garra esq. | 1 | — | máscara 0: temos mão, não garra |
| `[7:13]` mão dex. esq. | 6 | 7 juntas Dex3 | **a spec trunca nos 6 primeiros** |
| `[13:19]` efetuador dir. | 6 | 7 juntas do braço dir. | idem esquerdo |
| `[19:20]` garra dir. | 1 | — | máscara 0 |
| `[20:26]` mão dex. dir. | 6 | 7 juntas Dex3 | trunca |
| `[26:29]` cintura | 3 | 1 (só yaw) | a spec chama schema mais estreito que o slot de **inválido**. Roll/pitch estão travados no nosso robô, então o preenchimento honesto é escrever os valores travados |
| `[29:54]` torso, base, altura, pernas | 25 | — | máscara 0 (manipulação com o robô parado) |
| estado `[41:47]` inercial | 6 | gravidade + ω do IMU | **temos** no `lowstate`, e não usamos hoje |

Duas coisas que essa tabela revela e que não são óbvias:

1. **A truncagem da Dex3 é assimétrica.** A nossa ordem de juntas é
   `thumb_0,1,2 + middle_0,1 + index_0,1` na esquerda e
   `thumb_0,1,2 + index_0,1 + middle_0,1` na direita. Pegando os 6 primeiros,
   a mão esquerda perde o `index_1` e a direita perde o `middle_1` — dedos
   **diferentes** nas duas mãos. A assimetria é nossa, não da spec, e se
   resolve reordenando antes de mapear.

2. **A ação deixa de ser em junta.** `[0:6]` e `[13:19]` são pose relativa em
   SE(3) no referencial do efetuador atual, com rotação em vetor de rotação
   (`φ = θu`); o estado usa rotação-6D (as duas primeiras colunas de `R`).
   Converter não é reshape: é FK em cada quadro do dataset. Vale fazer de
   qualquer jeito — é a mesma representação que o π0.5 e o GR00T usam, e o
   FastWAM-D hoje não tem.

O resto do contrato que já casa: 30 fps alvo (o nosso é 30), chunk de 30,
metros, radianos, quaternion `xyzw`, euler `xyz`, e o frame da base com x para
frente, y para a esquerda, z para cima.

## 8. O ER-1 roda aqui — e acerta

O `UnifoLM-ER-1` não é política: é a metade de percepção do WLA-1.0, 4 B sobre
Qwen3-VL-4B, treinada em apontamento, detecção, trajetória 2D e QA espacial.
Em NF4 ocupa **2,92 GB** e roda no notebook.

Testado no quadro 0 do `white_cup_on_dripper` (848×480, a cena real do café):

| alvo | resposta do ER-1 | em pixel | onde está |
|---|---|---|---|
| `the white cup` | `[699, 254]` | (593, 122) | ~(595, 110) ✓ |
| `the coffee dripper stand` | `[345, 265]` | (293, 127) | ~(295, 95) ✓ |
| `the robot hand` | `[965, 181]` | (818, 87) | ~(800, 90) ✓ |

**⚠️ O ER-1 responde em 0–1000 normalizado, não em pixel.** A prova é o `965`
do terceiro: é maior que a largura da imagem que o processador entrega ao
modelo (728 px) e menor que 1000. A armadilha é que `699` **cabe** em 848 —
uma heurística do tipo "só reescala se estourar a imagem" deixa o ponto 100 px
ao lado da caneca e nada parece errado. Em
[`../grounding_er1.py`](../grounding_er1.py) a escala é decisão explícita
(`--pixels` inverte), não palpite.

```bash
python grounding_er1.py --dataset meu_dataset/white_cup_on_dripper_2026-08-11 \
                        --episodio 0 --alvo "the white cup"
```

### 8.1 E onde ele erra: a caneca vira o coador

Um quadro isolado é um teste fácil. Rodando o episódio 1 inteiro (713 quadros,
a tarefa completa) com dois alvos — `the white cup` e `the coffee dripper` —
aparece o modo de falha:

```
                        quadros em que os DOIS pontos caem no mesmo lugar (< 45 px)
  primeira metade  (0-340)      6%   (22/341)   ← a caneca está longe, na direita da mesa
  segunda metade   (360-712)   50%  (177/353)   ← a caneca já está embaixo do coador
  episódio inteiro             28%  (201/713)
```

*Medido em 14/09 perguntando em todos os quadros
(`resultados/unifolm_2026-09-13/er1/rastreio_ep1_v2.csv`). A primeira medição,
de 13/09, amostrava um quadro a cada 20 e tinha dado 0% e 72% — a amostragem
esparsa errou nas duas pontas.*

Depois que o robô coloca a caneca embaixo do coador, o ponto de `the white cup`
pula para cima do **coador** — que também é branco e também tem forma de copo —
em metade dos quadros. Não é um estado estável: são **52 trechos** separados de
colapso ao longo do episódio, o maior de 35 quadros (462-496); entre eles o
ponto volta para a caneca de verdade.

E não é só proximidade. O primeiro colapso é no **quadro 68**, com a caneca
ainda sozinha na direita da mesa, a centenas de pixels do coador — no instante
em que os dedos do robô chegam nela e a cobrem em parte. A oclusão pela mão
parece bastar para o ER-1 trocar de objeto. É uma hipótese tirada de um
quadro; confirmar exige cruzar os trechos de colapso com o contato da mão.

Não é bug do script: é ambiguidade real da cena, e o ER-1 não tem estado
temporal para desempatar (cada quadro é uma pergunta nova, §8). O que isso
quer dizer na prática:

  * **Como régua para o `grounding_fastwamd.py`, use a primeira metade.** Ali
    o ER-1 é inequívoco, e é uma comparação justa.
  * **Prompt importa.** `the white mug` ou `the cup the robot is holding`
    provavelmente desempatam; `the white cup` numa cena com dois objetos
    brancos em forma de copo, não.
  * **Ponto sempre existe.** Foram 713/713 quadros com resposta nos dois
    alvos — o modelo nunca disse "não vejo". Ausência de resposta não é o
    sinal de erro aqui; o sinal é o ponto ir para o lugar errado com a mesma
    confiança. Por isso o CSV de pontos é sempre gravado: sem os números, esse
    colapso passa despercebido assistindo ao vídeo. O vídeo de hoje avisa na
    legenda (`⚠ 1 e 2 no mesmo lugar`) quando os pontos ficam a menos de 45 px.

Por que isso importa para o FastWAM-D: o
[`../grounding_fastwamd.py`](../grounding_fastwamd.py) faz a mesma pergunta
medindo a atenção de linguagem do prior do Wan, e precisa contornar dois
sumidouros empilhados (87% da massa no padding). O ER-1 dá a resposta direta.
Se ele acerta a caneca nesta mesma cena e a atenção do Wan não, o buraco do
FastWAM-D é de **dados de fine-tune** — uma string de `task` só nos 24
episódios — e não de percepção.

## 9. Arquivos desta integração

| arquivo | para quê |
|---|---|
| [`../../unifolm-wma`](../../unifolm-wma) | submódulo com o código da Unitree, intocado |
| [`../install-unifolm-wma.sh`](../install-unifolm-wma.sh) | cria o env `unifolm-wma` (py 3.10) |
| [`../wma/converte_dataset_wma.py`](../wma/converte_dataset_wma.py) | nosso LeRobot v3.0 → formato WMA |
| [`../wma/confere_checkpoint.py`](../wma/confere_checkpoint.py) | quantos tensores REALMENTE carregam (§4) |
| [`../wma/ponte_wma.py`](../wma/ponte_wma.py) | ZMQ ↔ HTTP, para o cliente de sempre |
| [`../config/wma/treino_g1_dex3.yaml`](../config/wma/treino_g1_dex3.yaml) | pós-treino, 29 dims |
| [`../config/wma/inferencia_g1_dex3.yaml`](../config/wma/inferencia_g1_dex3.yaml) | inferência, 29 dims |
| [`../athena/launch_wma_interacao.sh`](../athena/launch_wma_interacao.sh) | simulação interativa (roda hoje) |
| [`../athena/launch_wma_server.sh`](../athena/launch_wma_server.sh) | servidor + ponte |
| [`../config/wma/interacao_exemplo_g1.yaml`](../config/wma/interacao_exemplo_g1.yaml) | simulação interativa com o exemplo deles (16 dims, tudo carrega) |
| [`../grounding_er1.py`](../grounding_er1.py) | apontamento com o ER-1 num quadro (§8) — **roda no notebook** |
| [`../rastreia_er1.py`](../rastreia_er1.py) | o mesmo sobre um episódio inteiro, saindo csv de pontos + vídeo (§8.1) |
| [`../renderiza_rastreio.py`](../renderiza_rastreio.py) | redesenha o vídeo a partir do csv, sem GPU, em H.264 que toca em WhatsApp/Drive |
