# Treinar o action expert do UnifoLM-WLA com os nossos dados

Tudo aqui é **configuração**. O código deles roda do submódulo `unifolm-wla/`, com duas linhas
de patch opt-in (ver `prometheus.patch`, mais abaixo) que sem variável de ambiente não mudam nada.

## Como subir

```bash
# na athena, como mrwlker
E=$HOME/miniconda3/envs/unifolm-wla          # ambiente separado, ver `pontes/wla/instala_wla.sh`
cd ~/DEV/prometheus-vla/unifolm-wla
screen -dmS wla bash -c "export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 \
    WLA_ATTN_BACKEND=native PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LD_PRELOAD=$E/lib/libstdc++.so.6 HF_HOME=\$HOME/.cache/huggingface; \
    $E/bin/accelerate launch --config_file unifolm_wla/config/deepseeds/zero2.yaml \
    --num_processes 1 --num_machines 1 -m unifolm_wla.training.train_unifolm_wla \
    --config_yaml /data/mrwlker/wla_prometheus.yaml 2>&1 | tee /data/mrwlker/wla_treino.log"
```

`dados_prometheus.yaml` vai para
`unifolm-wla/unifolm_wla/dataloader/multi_source_dataset/configs/`, e o `wla_prometheus.yaml`
para onde o `--config_yaml` apontar.

## As nove armadilhas, todas do lado deles

| o que quebra | por quê, e o que resolve |
|---|---|
| `python -m ...` direto | o trainer exige `accelerate launch`. O script que faria isso está em `examples/`, **que eles não publicaram** — não há um único `.sh` no repositório |
| `Framework QwenGR00T is not implemented` | o YAML de exemplo **deles** aponta para um framework inexistente. O publicado é `QwenMMDiT` |
| `VLM model ... not implemented` | o despacho é por NOME DE PASTA (`"Qwen3-VL" in vlm_name`). O ER-1 é um Qwen3-VL (`model_type: qwen3_vl`), só não se chama assim — resolvido com link simbólico `UnifoLM-ER-1-Qwen3-VL` |
| `Couldn't cast list<float> to float` | a garra deles é `float32` ESCALAR; a nossa saía como vetor de 1 elemento. Corrigido no conversor |
| `has no precollected_stats_path` | eles não calculam estatística, exigem pré-coletada. Apontamos para a deles: mesmo robô, mesmas unidades |
| vídeo de profundidade | desencontro de um quadro no decodificador (30,0333 pedido contra 30,0667 carregado). A feature foi removida — nenhuma receita nossa a usa |
| `CUDA out of memory` | com UM processo o ZeRO-2 não reparte nada, e o AdamW de 4 B pede ~48 GB. Resolvido congelando o VLM (`freeze_modules: qwen_vl_interface`), que é literalmente o que "treinar um action expert" significa |
| `DeepSpeedCPUAdam has no attribute ds_opt_adam` | a extensão C++ do offload não compila. Não foi preciso, depois do congelamento |
| `flash_attn_varlen_func is None` | ver o patch |

## O patch, em `../../pontes/wla/prometheus.patch`

Duas linhas, as duas **opt-in** — sem a variável de ambiente o comportamento é byte a byte o
original. Reaplicar depois de `git submodule update`.

**Atenção (`mmdit.py`).** O DiT pede FLASH ao despachante do diffusers, que chama
`flash_attn_varlen_func` — `None` quando o flash-attn não carrega. Na athena ele NÃO carrega: o
binário exige GLIBC 2.32 e a máquina tem 2.31. Com `WLA_ATTN_BACKEND=native` usa a atenção
nativa do PyTorch, caminho que o comentário do `forward` deles já prevê ("NATIVE runs in
whatever dtype the module is already in") e que o `QWen3.py` deles já faz para o VLM.

**wandb (`train_unifolm_wla.py`).** Estava fixo em `mode="offline"`, e argumento explícito vence
`WANDB_MODE` do ambiente — não havia como ligar a sincronização sem tocar nisto.

## O que os nossos dados perdem na conversão

O canal `fig6d`, 6 dims por mão, fica vazio: a Dex3 tem 3 dedos e 7 juntas, e o formato deles
fala em garra de 2 dedos ou mão de 5. A mão entra só pela abertura. Também não temos perna,
base nem altura — daí `arm_type: "dual"`.

E só declaramos **dois papéis de câmera**, `head_left` e `cam_wrist_right`, que são os que as
três fontes têm em comum. Quando o nosso robô ganhar estéreo e câmera na mão esquerda, é o
`image_roles` do `dados_prometheus.yaml` que muda.
