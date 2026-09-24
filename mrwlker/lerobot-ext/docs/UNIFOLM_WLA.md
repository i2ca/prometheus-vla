# UnifoLM-WLA — o que é, o que dá para usar hoje, e o que falta

O código deles está em `unifolm-wla/`, na raiz do repositório, como **submódulo** — a mesma
convenção do `lerobot`, do `unitree_sdk2_python` e do `unifolm-wma`. Nada nosso entra lá dentro;
a nossa leitura do código é este documento, porque o README deles não responde quase nada do que
segue.

    git submodule update --init unifolm-wla

Lido em 21/09/2026, Apache-2.0.

## Dá para treinar HOJE, com o nosso dataset — e isso é o que importa

Os pesos do **UnifoLM-WLA-1.0 (6 B)** não saíram, e provavelmente isso não importa. O que eles
publicaram em 11/09/2026 é o que basta:

| item | estado |
|---|---|
| **UnifoLM-ER-1** (8,9 GB) e **UnifoLM-ER-Flow** (9,7 GB) — pesos | **publicados** |
| **Código de treinar um *action expert* sobre os modelos ER** | **publicado** |
| UnifoLM-WLA-Base (pesos) | *coming soon* |
| Código de afinar a partir do WLA-Base, e LoRA | *coming soon* |

Conferido em 21/09 na API do HuggingFace: a conta `unitreerobotics` tem `UnifoLM-ER-1`,
`UnifoLM-ER-Flow`, `UnifoLM-WMA-0-Base`, `UnifoLM-WMA-0-Dual`, `UnifoLM-VLM-Base`,
`UnifoLM-VLA-Base` e `UnifoLM-VLA-Libero`. Não existe `UnifoLM-WLA-Base`.

**Mas o caminho publicado não depende dele.** O `unifolm-wla/docs/train_action_expert_en.md` chama-se
"Training an Action Expert **from Scratch**": baixa-se o ER-1 ou o ER-Flow como backbone
visão-linguagem, aponta-se `base_vlm` para a pasta local, listam-se os datasets num YAML, e
treina-se o especialista de ação do zero em cima dele.

E o carregador é **multi-fonte, com `ConcatDataset`** — os datasets deles e os NOSSOS entram na
mesma corrida, cada um com o seu bloco no YAML. É exatamente o co-treino que a gente vinha
tentando montar à mão.

### A pegadinha: o `examples/` não foi publicado

O guia manda rodar `examples/pretrain/train_files/run_multi_source_train_mmdit.sh`. Esse
diretório **não existe no repositório** — não há um único `.sh` aqui. Mas é só o script de
conveniência que falta; o que importa está publicado:

```bash
python -m unifolm_wla.training.train_unifolm_wla \
    --config_yaml unifolm_wla/config/training/unifolm_wla_pretrain_multisource.yaml
```

O `--config_yaml` tem como padrão um arquivo de `examples/` que também não existe, então passá-lo
explicitamente é obrigatório. A config publicada traz:

```yaml
base_vlm: Qwen/Qwen3-VL-4B-Instruct          # trocar pelo caminho local do ER-1
data_config_path: unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml
per_device_batch_size: 8   gradient_accumulation_steps: 4
max_train_steps: 200000    num_warmup_steps: 2000   lr_scheduler: cosine_with_min_lr
```

Há configs de DeepSpeed ZeRO-2 e ZeRO-3 em `unifolm-wla/unifolm_wla/config/deepseeds/`, incluindo variantes
com offload para CPU. Backbone de ~4 B em bf16 com ZeRO-2 cabe numa A100 de 80 GB sozinha.

### O que falta do NOSSO lado

Uma só coisa, e é trabalho conhecido: **converter o formato da ação**. Eles esperam pose de mão
em colunas nomeadas; nós temos uma coluna `action` única com 29 ângulos de junta. Ver a seção
"Ele lê LeRobot nativamente", abaixo.

## O espaço de ação: 54 dims, e por que isso importa para nós

`unifolm-wla/unifolm_wla/dataloader/multi_source_dataset/action_mapping.py` define o layout unificado:

```
ação (54)                                   estado (60)
[ 0: 6] esq xyz + rotvec                    [ 0: 9] esq xyz + rot6d
[ 6: 7] esq garra                           [ 9:10] esq garra
[ 7:13] esq fig6d          <- DEDOS         [10:16] esq fig6d
[13:19] dir xyz + rotvec                    [16:25] dir xyz + rot6d
[19:20] dir garra                           [25:26] dir garra
[20:26] dir fig6d                           [26:32] dir fig6d
[26:29] CINTURA (3)        <- roll/pitch/yaw
[29:32] tronco (3)
[32:35] base vx, vy, vw
[35:41] base rotvec
[41:42] altura
[42:54] pernas (6 + 6)
```

Duas coisas resolvem limitações que nós MEDIMOS na semana passada:

1. **`waist_joint` tem 3 dimensões.** O `EE_R6_G1` do UnifoLM-VLA-0, que testamos em 18/09, tinha
   os três também — mas a nossa ponte só conseguia comandar o yaw, e medimos a consequência: com
   a cintura travada o braço morre em x = 0,409, e a nossa xícara está em 0,465. Sete
   centímetros fora de alcance. Com roll e pitch livres o mesmo braço chega a 0,50 com 2 mm de
   erro, inclinando 30°. **A cintura não é detalhe: é o que põe o objeto ao alcance.**

2. **`fig6d`, 6 dimensões por mão, ALÉM da garra escalar.** O VLA-0 só tinha um número de garra
   (0 a 4,5) por mão, e nós tivemos que inventar um mapeamento para as 7 juntas da Dex3. Aqui há
   um canal próprio para a configuração dos dedos.

**Atenção:** o README fala em *"two-finger grippers and multiple five-finger dexterous hands"*.
A nossa **Dex3-1 tem TRÊS dedos e 7 juntas**, e não é nenhum dos dois. Nas configurações de
exemplo aparece `Dex1`, não `Dex3`. Como o `fig6d` entra, e se cabe a Dex3, é a primeira coisa a
descobrir antes de investir nisto.

## Ele lê LeRobot nativamente — e é aqui que o nosso dado esbarra

`dataloader/multi_source_dataset/lerobot_wrapper.py` consome dataset LeRobot direto, e um YAML
mapeia as colunas. Do `configs/unitree.yaml`:

```yaml
action_keys:
  left_ee_pose:  "action.left_ee_pose_gripper_base"
  right_ee_pose: "action.right_ee_pose_gripper_base"
  left_gripper:  "action.left_gripper"
  right_gripper: "action.right_gripper"
  waist_joint:   "action.waist_action_joint"
  left_leg:      "action.left_leg"
  right_leg:     "action.right_leg"
```

O obstáculo: eles esperam **POSE DE MÃO em colunas separadas e nomeadas**
(`ee_format: "xyz_rpy"`), e os nossos datasets têm uma coluna `action` única de **29 ângulos de
junta**. Converter é trabalho conhecido, não pesquisa — a cinemática direta que monta xyz+rot6d
a partir das nossas juntas já existe e está validada em
`lerobot-ext/unifolm/roda_unifolm_mujoco.py` (classe `Cinematica`), escrita em 18/09 justamente para a ponte
do VLA-0. O que falta é reescrever os parquet com as colunas no nome que eles esperam.

Outros parâmetros do exemplo: `target_fps: 30`, `chunk_size: 30` (1 s), `image_size: [336, 448]`,
`norm_type: "minmax_q"`, `arm_type` entre `dual` e `dual_with_legs`.

## Onde ler primeiro

- `unifolm-wla/docs/robot_action_state_processing_en.md` — como eles preparam ação e estado
- `unifolm-wla/docs/train_action_expert_en.md` — treinar o *action expert* sobre os modelos ER
- `unifolm-wla/unifolm_wla/training/train_unifolm_wla.py` — o ponto de entrada
- `unifolm-wla/unifolm_wla/dataloader/multi_source_dataset/action_mapping.py` — o layout de 54/60

## O que eu faria antes de investir

O UnifoLM-VLA-0, testado aqui em 18/09, **quase não olhava para a imagem**: apagar a cena
inteira (imagem toda preta) mudava a resposta dele em 0,085, e mover o copo 6,5 cm não mudava
nada. Antes de repetir a empreitada com um modelo maior, vale medir a mesma coisa no ER-1 — é
um teste de meia hora e decide se o resto vale o esforço.
