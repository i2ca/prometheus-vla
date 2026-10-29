#!/bin/bash
# Ajuste fino do Ψ0 base na pega da caneca (demos de gera_demos.py convertidas por para_lerobot.py).
# Receita = scripts/train/psi0/finetune-real-psi0.sh do código de jan/2026 (57b52b1), o mesmo com que a Task 3 foi treinada:
# esse código soma a posição em todos os tokens e usa a cabeça sem final_layer_norm, como os pesos do base
# (o HEAD do repo muda as duas coisas em silêncio; ver ../base/README.md).
# Uso (na máquina com GPU): PSI0_JAN=<checkout 57b52b1> PSI0_VENV=<venv do Psi0> PSI0_PESOS=<pesos do base> \
#     DADOS=<raiz dos datasets> bash treina.sh <nome do dataset> [passos] [lote]
set -e
DS=${1:?nome do dataset}; PASSOS=${2:-10000}; LOTE=${3:-32}
AQUI=$(dirname "$(readlink -f "$0")"); cd "$PSI0_JAN"
export PYTHONPATH="$PSI0_JAN/src" OMP_NUM_THREADS=8 TF_CPP_MIN_LOG_LEVEL=3 WANDB_MODE=disabled
"$PSI0_VENV/bin/torchrun" --nproc_per_node=1 --master_port=29511 "$AQUI/sdpa.py" scripts/train.py finetune_real_psi0_config \
  --seed=292285 --exp="ft-$DS" --train.name=finetune --train.data_parallel=ddp --train.mixed_precision=bf16 \
  --train.train_batch_size=$LOTE --train.max_checkpoints_to_keep=10 --train.gradient_accumulation_steps=1 \
  --train.learning_rate=1e-4 --train.max_training_steps=$PASSOS --train.warmup_ratio=None --train.warmup_steps=500 \
  --train.checkpointing_steps=${CKPT:-2000} --train.validation_steps=${VALIDA:-1000} --train.val_num_batches=${VAL_BATCHES:-20} --train.max_grad_norm=1.0 ${RESUME:+--train.resume_from_checkpoint="$RESUME"} \
  --train.lr_scheduler_type=cosine --train.lr_scheduler_kwargs.weight_decay=1e-6 --train.lr_scheduler_kwargs.betas 0.95 0.999 \
  --data.root_dir="$DADOS" --data.train_repo_ids="$DS" --data.val_repo_ids="${VAL_DS:-$DS}" \
  --data.transform.repack.pad-action-dim=36 --data.transform.repack.pad-state-dim=36 \
  --data.transform.repack.image_keys observation.images.head observation.images.left_wrist observation.images.right_wrist \
  --data.transform.field.stat-path=meta/stats_psi0.json --data.transform.field.stat-action-key=action \
  --data.transform.field.stat-state-key=states --data.transform.field.action_norm_type=bounds \
  --data.transform.field.no-use-norm-mask --data.transform.field.normalize-state \
  --data.transform.field.pad-action-dim=36 --data.transform.field.pad-state-dim=36 \
  --data.transform.model.img-aug --data.transform.model.resize.size 240 320 --data.transform.model.center_crop.size 240 320 \
  --model.model_name_or_path="$PSI0_PESOS/pre" --model.pretrained-action-header-path="$PSI0_PESOS/postpre" \
  --model.noise-scheduler=flow --model.train-diffusion-steps=1000 --model.n_conditions=0 \
  --model.action-chunk-size=30 --model.action-dim=36 --model.action-exec-horizon=30 --model.observation-horizon=1 \
  --model.odim=36 --model.view_feature_dim=2048 --model.no-tune-vlm --model.no-use_film --model.no-combined_temb \
  --model.rtc --model.max-delay=8
