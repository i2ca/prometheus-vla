#!/usr/bin/env python
"""
Quantos tensores do checkpoint do WMA realmente entram no nosso modelo
======================================================================
O `load_model_checkpoint` do `real_eval_server.py` carrega com
`strict=False`. Isso é necessário (o checkpoint tem chaves do EMA e do
otimizador que o modelo de inferência não tem) e é perigoso pelo mesmo
motivo: mudar `agent_action_dim` de 16 para 29 faz a cabeça de ação inteira
deixar de casar, e o carregamento continua **sem uma linha de aviso**.

É a mesma armadilha que fez o π0.5 treinar 3 B de parâmetros do zero por
semanas (PI05_BASE.md §1). Este script é o contador que faltava lá.

USO (no ambiente `unifolm-wma`, a partir da raiz do submódulo)

    cd unifolm-wma
    python ../lerobot-ext/wma/confere_checkpoint.py \
        --config ../lerobot-ext/config/wma/treino_g1_dex3.yaml \
        --ckpt  ~/.cache/huggingface/hub/models--unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/*/unifolm_wma_dual.ckpt

COMO LER A SAÍDA

    CARREGAM         chave e shape iguais  → prior de verdade
    SHAPE DIFERENTE  a chave existe nos dois lados e o tensor não cabe
                     → nasce ALEATÓRIO. É aqui que a cabeça de ação aparece
                       quando você mexe em agent_action_dim.
    SEM PESO         está no nosso modelo e não no checkpoint → aleatório
    SOBRANDO         está no checkpoint e não no nosso modelo → descartado
                     (esperado para `*_ema*` e estado do otimizador)
"""

import argparse
import sys
from collections import defaultdict

import torch
from omegaconf import OmegaConf


def carrega_state_dict(caminho: str) -> dict:
    bruto = torch.load(caminho, map_location="cpu", weights_only=False)
    if "state_dict" in bruto:
        return bruto["state_dict"]
    if "module" in bruto:
        return {k[16:]: v for k, v in bruto["module"].items()}
    return bruto


def familia(chave: str) -> str:
    """Agrupa por subsistema, que é a unidade em que se decide se dói."""
    if ".unet_head." in chave or chave.startswith("model.diffusion_model.unet_head"):
        return "cabeça de ação (ConditionalUnet1D)"
    if "stem_process" in chave:
        return "tokens de ação (SATokenProjector)"
    if chave.startswith("first_stage_model"):
        return "VAE"
    if chave.startswith("cond_stage_model"):
        return "CLIP de texto"
    if chave.startswith("embedder") or "image_proj" in chave:
        return "CLIP de imagem / resampler"
    if chave.startswith("model.diffusion_model"):
        return "modelo de mundo (UNet 3D)"
    return "outros"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lista", action="store_true",
                   help="lista as chaves de shape diferente, uma por linha")
    args = p.parse_args()

    from unifolm_wma.utils.utils import instantiate_from_config

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    modelo = instantiate_from_config(cfg.model)
    nosso = {k: tuple(v.shape) for k, v in modelo.state_dict().items()}

    ckpt = carrega_state_dict(args.ckpt)
    deles = {k: tuple(v.shape) for k, v in ckpt.items()}

    carregam, shape_diferente, sem_peso, sobrando = [], [], [], []
    for k, forma in nosso.items():
        if k not in deles:
            sem_peso.append(k)
        elif deles[k] != forma:
            shape_diferente.append((k, deles[k], forma))
        else:
            carregam.append(k)
    sobrando = [k for k in deles if k not in nosso]

    total = len(nosso)
    print(f"\ncheckpoint {len(deles)} tensores | nosso modelo {total}\n")
    print(f"CARREGAM (chave e shape iguais): {len(carregam)}"
          f"  →  {100 * len(carregam) / max(total, 1):.1f}% do nosso modelo")
    print(f"SHAPE DIFERENTE (nascem aleatórios): {len(shape_diferente)}")
    print(f"SEM PESO no checkpoint (nascem aleatórios): {len(sem_peso)}")
    print(f"SOBRANDO no checkpoint (descartados): {len(sobrando)}")

    aleatorios = defaultdict(int)
    for k, _, _ in shape_diferente:
        aleatorios[familia(k)] += 1
    for k in sem_peso:
        aleatorios[familia(k)] += 1
    if aleatorios:
        print("\nO que nasce aleatório, por subsistema:")
        for nome, n in sorted(aleatorios.items(), key=lambda x: -x[1]):
            print(f"  {n:5d}  {nome}")

    if args.lista and shape_diferente:
        print("\nChaves de shape diferente:")
        for k, ck, no in shape_diferente:
            print(f"  {k}\n      checkpoint {ck}  ×  nosso {no}")

    if shape_diferente:
        print("\n⚠️  Tensor de shape diferente não é aviso: é peso NOVO. Se a")
        print("    'cabeça de ação' aparecer acima, o checkpoint está dando o")
        print("    modelo de mundo e nada de política — você precisa pós-treinar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
