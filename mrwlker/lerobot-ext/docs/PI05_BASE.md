# Partir do `lerobot/pi05_base` — o que estava errado e o que mudou

> 09/09/2026. Resumo em uma linha: **as corridas de π0.5 treinavam 3 bilhões de
> parâmetros com inicialização aleatória**, e a cópia local do π0.5 não
> conseguia carregar os pesos publicados nem que quisesse.

---

## 1. O diagnóstico

Duas coisas separadas, que juntas explicam o resultado das corridas anteriores.

### 1.1 Nenhum config ligava o checkpoint

`policies/pi0_depth/modeling_pi05.py` (versão antiga, linha 395):

```python
self.paligemma = PaliGemmaForConditionalGeneration(vlm_config_hf)
```

Isso constrói o modelo **a partir de um config, com pesos aleatórios** — não é
`from_pretrained`. E nenhum `config/train/pi05*.yaml` definia `pretrained_path`.
O `maquinas/athena/launch_pi05.sh` baixa o `google/paligemma-3b-pt-224`, mas o comentário
dele já dizia: "puxa **só pelo tokenizer**".

Compare com o FastWAM-D, que sempre partiu do Wan2.2 pré-treinado
(`policies/fastwam_depth/modeling_fastwam_depth.py:309`, com `strict=False` e o
`patch_embedding` ampliado). O π0.5 era a única política do repositório treinando
do zero.

### 1.2 E o código não conseguiria carregar os pesos

Medido chave a chave contra o `model.safetensors` do `lerobot/pi05_base`
(812 tensores; o cabeçalho do safetensors dá a lista sem baixar os 14,5 GB):

| implementação | tensores que casavam |
|---|---:|
| `lerobot.policies.pi05.PI05Pytorch` (upstream, já vendorizado aqui) | **812 / 812** |
| a cópia em `policies/pi0_depth/` | **657 / 812** |

As três divergências:

| # | o quê | tamanho |
|---|---|---:|
| 1 | **Torre visual com largura errada.** Checkpoint: `vision_tower...mlp.fc1` `[4304, 1152]` (SigLIP-So400m). Cópia: `[4096, 1152]`, porque montava o config dos defaults de `CONFIG_MAPPING["paligemma"]()` em vez do config real do PaliGemma-3B | **81 tensores** (as 27 camadas) |
| 2 | **Expert sem adaRMS.** Checkpoint: `input_layernorm.dense.{weight,bias}` `[3072, 1024]`. Cópia: `input_layernorm.weight` `[1024]`, por usar o `GemmaForCausalLM` de estoque em vez do `PiGemmaForCausalLM` | **74 órfãs + 37 sem peso** |
| 3 | `embed_tokens` do expert, que o upstream anula | 3 |

O item 2 não é detalhe de implementação: **adaRMS é como o π0.5 injeta o timestep
do flow matching**. A cópia injetava pelo `time_mlp`, que é o jeito do π0. O que
rodava com o nome `pi05depth` era, na arquitetura, um π0.

E o pior: com `strict=False` aquilo carregaria 657 de 812 **em silêncio**, sem a
torre visual e sem as normalizações do expert.

---

## 2. O que foi feito

`policies/pi0_depth/` deixou de ser uma cópia e passou a ser um **enxerto por
herança** sobre `lerobot.policies.pi05`:

```
              antes        depois
modeling      1.307    →     350   linhas
configuration   195    →     120
processor       179    →     131
```

O modelo é o do upstream com **dois tokens a mais no prefixo**:

```
[ imagens SigLIP | linguagem | nuvem de pontos | pressão ]   ← prefixo
[ ações ruidosas ]                                           ← sufixo (expert)
```

Nada toca a atenção, o expert ou o flow matching. Os tokens novos entram com
`att_mask = 0`, no mesmo bloco bidirecional das imagens e da linguagem — são
observação, não ação.

### O que ficou diferente do upstream, de propósito

| | por quê |
|---|---|
| `PI05DepthPytorch.embed_prefix` | os dois tokens do enxerto |
| `PI05DEPTHPolicy._preprocess_images` | a profundidade é declarada VISUAL no YAML (é como o dataset a carrega) mas não pode ir para o SigLIP |
| `Pi05DepthPrepare...Step.pad_state_to_max` | o π0.5 **escreve o estado dentro do prompt**, discretizado em 256 níveis. O `pi05_base` foi treinado com estado de 32; o nosso robô manda 29. Sem completar até 32, o prompt tem comprimento diferente do que o checkpoint viu |
| `pretrained_strict` | o PointNet e a projeção de pressão não existem no checkpoint |
| `from_pretrained` exige `config` | o `config.json` do `pi05_base` diz `type: pi05` e produziria um `PI05Config` sem `use_depth_3d` — o erro só apareceria adiante, como `AttributeError` no meio do treino |

### Verificação

```
checkpoint 812 tensores | nosso modelo 827
CARREGAM (chave e shape iguais): 812  →  100.0% do checkpoint
no checkpoint e SEM destino (unexpected): 0
no nosso e SEM peso (missing): 15   ← pointnet (10), pressure_proj (4),
                                       embed_tokens (1, amarrado ao lm_head)
shapes DIFERENTES: 0
```

Mais o teste do enxerto isolado: o prefixo cresce exatamente 1 token com
profundidade e 2 com profundidade + tato, as máscaras saem no formato certo, e o
gradiente chega no PointNet e na projeção de pressão.

---

## 3. Como rodar

`config/train/pi05_base_cotreino.yaml`:

```yaml
policy:
  type: pi05depth
  pretrained_path: lerobot/pi05_base
  pretrained_strict: false
  use_depth_3d: false      # primeira corrida sem profundidade, de propósito
  optimizer_lr: 1.0e-5     # fine-tune, não treino do zero
```

**A primeira corrida vai sem profundidade e sem tato de propósito.** O PointNet e
a pressão nascem aleatórios; ligá-los na mesma corrida em que se estreia o prior
soma duas variáveis para explicar um resultado. Esta corrida responde uma
pergunta só: partir de pesos treinados muda o travamento no laço fechado?

**O LR caiu de 2.5e-5 para 1e-5.** 2.5e-5 é o preset de PRÉ-TREINO do π0.5.
Partindo de pesos bons com 10 mil quadros, esse LR apaga o prior em algumas
centenas de passos — esquecimento catastrófico é o modo de falha nº 1 deste
plano. Pelo mesmo motivo, o caminho é **co-treino**, não fine-tune sequencial:
ver [DATASETS_EXTERNOS.md](DATASETS_EXTERNOS.md) §4.

### O que quebra

Os checkpoints de π0.5 das corridas anteriores **não abrem** na classe nova — a
arquitetura mudou (adaRMS, largura da torre visual). Não há o que salvar: eles
vieram de um modelo inicializado aleatoriamente.

O tokenizer do PaliGemma continua sendo repo fechado; o `~/.hf_token` do
`launch_pi05.sh` continua obrigatório.

---

## 4. Outros checkpoints já treinados no G1 (levantamento de 09/09/2026)

O `pi05_base` é um **prior de ação**, não um modelo que já faz tarefa no G1 — ele
nunca viu este robô. Fui procurar checkpoints que já tivessem tarefas do G1
treinadas, para testar antes de treinar qualquer coisa.

### 4.1 O que existe

| checkpoint | tarefas | corpo | dado | licença |
|---|---|---|---|---|
| `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` | 1 (maçã → prato) | G1 + **Dex3** (7+7) + cintura 3 | **simulação** (GR00T WholeBodyControl) | NVIDIA Open Model |
| `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` | pick & place | G1 | **simulação** (IsaacLab Arena) | **não comercial** |
| `nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation` | locomanipulação | G1 | **simulação** | **não comercial** |
| `unitreerobotics/UnifoLM-VLA-Base` | **13 tarefas reais numa política só** | G1 + **Dex1** (garra, 23 dims) | **real** | CC BY-NC-SA |
| `unitreerobotics/UnifoLM-WMA-0-Base` / `-Dual` | world model + ação | G1 e outros | real | CC BY-NC-SA |
| `ambitiousmangosteen/g1-dex3-*` | 1 cada (pick, put, cola, torrada) | G1 Dex3, mas ação de **20** dims e 1 câmera | real, comunidade | — |

As 13 tarefas do UnifoLM-VLA-Base (do `dataset_statistics.json`): empilhar blocos,
guardar estojo, limpar mesa, apagar quadro, inserir saco, **despejar remédio**,
guardar pingue-pongue, organizar ferramentas, limpar mesa a dois braços,
preparar fruta, dobrar toalha.

### 4.2 O achado que muda a escolha da base

O **`nvidia/GR00T-N1.7-3B`** — o modelo base, não um fine-tune — foi
pré-treinado com dados **reais do G1 com Dex3**. No
`experiment_cfg/dataset_statistics.json` dele existe o embodiment
`real_g1_relative_eef_relative_joints`:

```
state : left_arm 7 | right_arm 7 | left_hand 7 | right_hand 7 | waist 3 | eef 9+9
action: os mesmos + base_height_command 1 + navigate_command 3
```

`left_hand 7` e `right_hand 7` são exatamente as Dex3-1. E o
`embodiment_id.json` traz `unitree_g1_full_body_with_waist_height_nav_cmd` como
tag de primeira classe.

Ou seja: **o π0.5 nunca viu o nosso robô; o GR00T N1.7 viu, com as nossas mãos.**

### 4.3 O que isso custa

| | π0.5 (`lerobot/pi05_base`) | GR00T N1.7-3B |
|---|---|---|
| prior do nosso corpo | nenhum | **G1 + Dex3 real** |
| runtime | o nosso LeRobot, já pronto | repo `Isaac-GR00T`, servidor próprio |
| formato de dado | o nosso, v3.0 | LeRobot **v2** + `meta/modality.json` |
| profundidade e tato | o enxerto já está feito (§2) | teria de ser refeito |
| licença | permissiva | NVIDIA Open Model (o base permite uso comercial; os fine-tunes da Arena, não) |
| VRAM para fine-tune | roda na A100 | 40 GB+ recomendado |

Dois detalhes que ajudam: o Isaac-GR00T **consome dataset em formato LeRobot**
(v2 mais um `modality.json`), então o trabalho de dataset não se perde; e a
inferência dele é **servidor + cliente por ZMQ**, que é exatamente a arquitetura
que já usamos em `init_lerobot_inference_*`.

### 4.4 Recomendação

São dois passos independentes, e vale fazer os dois:

1. **Testar hoje, sem treinar nada:** subir o
   `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` no servidor do Isaac-GR00T com
   `--embodiment_tag UNITREE_G1`. É o único checkpoint público que já sai com o
   nosso corpo e as nossas mãos. **Não espere que ele faça a tarefa** — foi
   treinado em simulação. O que ele testa é a ponte: observação do robô →
   modelo → ação executável. Se isso fechar, a ponte serve para qualquer
   checkpoint depois.
2. **Decidir a base do treino:** continuar no π0.5 (o enxerto está pronto e
   validado, 812/812) ou migrar para o GR00T N1.7 (prior do nosso corpo, mas
   outro runtime e o enxerto de profundidade a refazer). Não dá para responder
   isso por medição sem rodar as duas — e a corrida do π0.5 já está pronta para
   sair.
