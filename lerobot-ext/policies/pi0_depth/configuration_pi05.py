#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""
PI05-Depth — configuração.
================================================================================
**Herda de `PI05Config` do LeRobot em vez de copiá-la.** Antes este arquivo era
uma cópia de uma versão antiga do π0.5 upstream, e a cópia envelheceu: faltavam
`use_relative_actions`, `relative_exclude_joints` e `action_feature_names`, e a
arquitetura correspondente não conseguia carregar o `lerobot/pi05_base` (ver o
cabeçalho de `modeling_pi05.py` para as três divergências medidas).

Aqui ficam SÓ os nossos acréscimos:

| campo | para quê |
|---|---|
| `use_depth_3d`, `pointnet_num_points`, `camera_intrinsics` | nuvem de pontos → 1 token no prefixo |
| `use_pressure`, `pressure_feature_dim` | tato das Dex3 → 1 token no prefixo |
| `scene_uncertainty_threshold`, `n_samples_uncertainty` | gate de incerteza na inferência |
| `override_task` | força um prompt fixo (só para depurar) |

Tudo o mais — `chunk_size`, `max_action_dim`, presets de otimizador, RTC,
`validate_features` — vem do upstream e acompanha as correções dele.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05depth")
@dataclass
class PI05DEPTHConfig(PI05Config):
    """π0.5 do LeRobot + profundidade métrica (PointNet) + tato (Dex3)."""

    # ── Geometria 3D ─────────────────────────────────────────────────────────
    # A profundidade NÃO passa pelo SigLIP: vira nuvem de pontos e depois um
    # único token no prefixo. Ver `PI05DepthPytorch.embed_prefix`.
    use_depth_3d: bool = True
    pointnet_num_points: int = 1024
    camera_intrinsics: dict = field(
        default_factory=lambda: {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    )

    # ── Tato / pressão ───────────────────────────────────────────────────────
    # 33 valores por mão, concatenados: 66.
    use_pressure: bool = True
    pressure_feature_dim: int = 66

    # ── Gate de incerteza ────────────────────────────────────────────────────
    # O π0.5 é flow matching, não VAE: não existe log_sigma para ler. A
    # incerteza é estimada rodando `n_samples_uncertainty` denoising passes com
    # ruídos iniciais diferentes e medindo o desvio entre os resultados. O
    # prefixo (VLM + SigLIP) é calculado UMA vez com KV-cache; só o expert roda
    # n vezes.
    #   0.0  → desligado (padrão, sem custo)
    #   0.10 → ponto de partida razoável para o G1
    scene_uncertainty_threshold: float = 0.0
    n_samples_uncertainty: int = 1  # vira 3 sozinho se o threshold for > 0

    # ── Checkpoint base ──────────────────────────────────────────────────────
    # Ao partir do `lerobot/pi05_base`, o PointNet e a projeção de pressão não
    # existem no checkpoint — nascem aleatórios, e é o esperado. Com
    # `strict=True` (o padrão do upstream) isso vira exceção antes do passo 1.
    # `False` aqui NÃO é esconder problema: o `from_pretrained` imprime tudo que
    # faltou e tudo que sobrou, e reclama do que não for enxerto conhecido.
    pretrained_strict: bool = False

    # ── Depuração ────────────────────────────────────────────────────────────
    # Força um prompt fixo, ignorando a `task` do dataset. DESLIGA o
    # multi-tarefa — use só para depurar.
    override_task: str | None = None

    def __post_init__(self):
        super().__post_init__()

        # Detecta YAML inconsistente antes de carregar 3 B de pesos.
        has_depth = any("depth" in k.lower() for k in self.input_features)
        if self.use_depth_3d and not has_depth:
            raise ValueError(
                "use_depth_3d=True mas nenhuma feature com 'depth' no nome está em "
                "input_features. Acrescente a feature ou coloque use_depth_3d=False."
            )
        if not self.use_depth_3d and has_depth:
            warnings.warn(
                "use_depth_3d=False mas uma feature de depth está em input_features. "
                "A câmera será carregada do disco e ignorada — considere tirá-la do YAML.",
                stacklevel=2,
            )

        has_pressure = (
            "observation.left_hand_pressure" in self.input_features
            or "observation.right_hand_pressure" in self.input_features
        )
        if self.use_pressure and not has_pressure:
            raise ValueError(
                "use_pressure=True mas as features de pressão não estão em input_features."
            )

        if self.scene_uncertainty_threshold > 0 and self.n_samples_uncertainty <= 1:
            self.n_samples_uncertainty = 3

    @property
    def rgb_image_features(self) -> dict:
        """As câmeras que vão para o SigLIP — ou seja, `image_features` menos a profundidade.

        Existe porque o `_preprocess_images` do upstream itera `image_features` e
        manda tudo para a torre visual. A profundidade é declarada como VISUAL no
        YAML (é assim que o dataset a carrega e a encoda como vídeo de 1 canal),
        mas o caminho dela no modelo é outro.
        """
        return {k: v for k, v in self.image_features.items() if "depth" not in k.lower()}

    @property
    def depth_image_features(self) -> dict:
        """As câmeras de profundidade — as que vão para o PointNet."""
        return {k: v for k, v in self.image_features.items() if "depth" in k.lower()}
