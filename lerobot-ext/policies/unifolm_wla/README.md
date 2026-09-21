# `unifolm_wla` — action expert da Unitree, a nossa cópia de trabalho

Copiado de `unifolm-wla/unifolm_wla/` (o submódulo na raiz, commit **f33d0e7**) em 21/09/2026.

## Por que uma cópia, se o submódulo está ali

Mesma razão de `pi0_depth` ser a nossa versão do π0 e de `act_depth` ser a nossa do ACT: o
submódulo é a **referência de upstream**, intocada, para dar `git diff` contra ela; esta pasta é
onde a gente adapta. Mexer dentro do submódulo perderia tudo na primeira atualização.

Para ver o que já divergiu:

```bash
diff -ru ../../../unifolm-wla/unifolm_wla . | head -50
```

## O que dá para treinar hoje

Os pesos do UnifoLM-WLA-1.0 (6 B) **não** foram publicados, mas isso não bloqueia: o caminho
documentado é treinar um *action expert* **do zero** sobre um modelo ER, e esses saíram.

```bash
# 1. baixar o backbone visão-linguagem (8,9 GB)
huggingface-cli download unitreerobotics/UnifoLM-ER-1 --local-dir /data/mrwlker/UnifoLM-ER-1

# 2. apontar `base_vlm` para ele em config/training/unifolm_wla_pretrain_multisource.yaml
#    (o padrao vem como Qwen/Qwen3-VL-4B-Instruct)

# 3. treinar
python -m policies.unifolm_wla.training.train_unifolm_wla \
    --config_yaml policies/unifolm_wla/config/training/unifolm_wla_pretrain_multisource.yaml
```

**O `--config_yaml` é obrigatório.** O padrão dele aponta para
`examples/SimplerEnv/train_files/unifolm_wla_cotrain_oxe.yaml`, e o diretório `examples/` **não
foi publicado** — não há um único `.sh` no repositório deles, incluindo o
`run_multi_source_train_mmdit.sh` que o guia manda rodar.

## O que falta do nosso lado: o formato da ação

O carregador é multi-fonte e lê **LeRobot nativamente** (`dataloader/multi_source_dataset/`),
com um YAML mapeando coluna→campo. Os datasets deles e os nossos entram na mesma corrida por
`ConcatDataset`. O obstáculo é outro: eles esperam **pose de mão em colunas nomeadas** e nós
temos uma coluna `action` única com 29 ângulos de junta.

| campo deles | de onde sai o nosso |
|---|---|
| `left_ee_pose`, `right_ee_pose` | cinemática direta das juntas 0–13 — **já pronta e validada** em `pontes/unifolm-vla/roda_unifolm_mujoco.py`, classe `Cinematica` |
| `left_gripper`, `right_gripper` | o mapeamento garra↔7 dedos da Dex3, no mesmo arquivo |
| `waist_joint` (3) | temos só `kWaistYaw`; roll e pitch entram em zero |
| pernas, base, altura | não temos — usar `arm_type: "dual"` em vez de `dual_with_legs` |

Layout unificado, de `dataloader/multi_source_dataset/action_mapping.py`: ação de **54** dims,
estado de **60** (a diferença é rot6d no estado contra rotvec na ação).

## Armadilhas de ARM já levantadas

| | |
|---|---|
| `flash_attn` | **não é problema**: `model/modules/vlm/QWen3.py:57` tenta importar e cai sozinho para `sdpa`. Foi o remendo que tivemos que fazer à mão no VLA-0 |
| `pipablepytorch3d==0.7.6` | está no `pyproject.toml` deles, exige Python <3.12 (o nosso é 3.12) — e **não é importado em lugar nenhum**. Pino fantasma, basta não instalar |
| `torch==2.8.0` | tem wheel aarch64; rodamos 2.11 sem problema |
| servidor de inferência | **não existe** neste repositório. O `unifolm-vla` trazia `run_real_eval_server.py`; aqui o laço de servir é nosso |

Análise completa em [`docs/UNIFOLM_WLA.md`](../../docs/UNIFOLM_WLA.md).
