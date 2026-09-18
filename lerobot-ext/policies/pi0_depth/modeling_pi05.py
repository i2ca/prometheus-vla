#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""
PI05-Depth — π0.5 do LeRobot + profundidade métrica + tato, por HERANÇA.
================================================================================
Este arquivo era uma cópia de 1.307 linhas de uma versão antiga do π0.5. A cópia
envelheceu e deixou de ser compatível com os pesos publicados. Medido chave a
chave contra o `lerobot/pi05_base` (812 tensores) em 09/09/2026:

| | tensores que casavam |
|---|---|
| `lerobot.policies.pi05.PI05Pytorch` (upstream) | **812 / 812**, zero shapes diferentes |
| a cópia que estava aqui | **657 / 812** |

As três divergências da cópia antiga, para o caso de alguém tentar ressuscitá-la:

1. **Torre visual com largura errada — 81 tensores.** O checkpoint tem
   `vision_tower...mlp.fc1` de `[4304, 1152]` (SigLIP-So400m); a cópia montava
   `[4096, 1152]` porque construía o config a partir dos defaults de
   `CONFIG_MAPPING["paligemma"]()` em vez do config real do PaliGemma-3B. As 27
   camadas da torre visual não carregavam.
2. **Expert sem adaRMS — 74 chaves órfãs + 37 sem peso.** O checkpoint tem
   `input_layernorm.dense.{weight,bias}` de `[3072, 1024]`; a cópia tinha
   `input_layernorm.weight` de `[1024]`, porque usava o `GemmaForCausalLM` de
   estoque em vez do `PiGemmaForCausalLM`. **adaRMS é como o π0.5 injeta o
   timestep do flow matching** — sem ele o que estava rodando era um π0 com nome
   de π0.5.
3. `embed_tokens` do expert, que o upstream anula (3 chaves).

Com `strict=False` isso carregaria 657 de 812 EM SILÊNCIO, sem a torre visual e
sem as normalizações do expert. Pior do que não carregar.

── O que sobrou de nosso ────────────────────────────────────────────────────
O enxerto são dois tokens a mais no prefixo e nada mais:

    [ imagens SigLIP | linguagem | nuvem de pontos | pressão ]  ← prefixo
    [ ações ruidosas ]                                          ← sufixo (expert)

Mais o gate de incerteza e o `neutral_position`, que vivem na Policy. Nada disso
toca a atenção, o expert ou o flow matching — por isso o
`from_pretrained("lerobot/pi05_base", strict=False)` carrega 812/812 e reporta
como faltando apenas `model.pointnet.*` e `model.pressure_proj.*`, que são
novos mesmo.

É o mesmo padrão que o FastWAM-D já usa com o Wan2.2
(`policies/fastwam_depth/modeling_fastwam_depth.py:309`).

── Como usar o checkpoint base ──────────────────────────────────────────────
No YAML:

    policy:
      type: pi05depth
      pretrained_path: lerobot/pi05_base
      pretrained_strict: false     # o PointNet e a pressão não existem lá

`pretrained_strict` é nosso: sem ele o `load_state_dict` estoura ao ver os
enxertos. Ver `from_pretrained` no fim deste arquivo.
"""

from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import pad_vector  # noqa: F401  (reexport: o processor usa)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config
from lerobot.utils.constants import ACTION

from .configuration_pi05 import PI05DEPTHConfig
from .depth_encoder import PointNetEncoder, depth_to_pointcloud

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="PI05DEPTHPolicy")


class PI05DepthPytorch(PI05Pytorch):
    """O π0.5 do upstream com dois tokens a mais no prefixo.

    Os tokens extras chegam por `set_extra_inputs`, e não por argumento de
    `embed_prefix`, porque quem chama `embed_prefix` são o `forward` e o
    `sample_actions` do upstream — que não conhecem profundidade. Herdar sem
    reescrever esses dois métodos é o que mantém o modelo idêntico ao do
    checkpoint; o preço é guardar duas referências no objeto entre a chamada da
    Policy e a do modelo, sempre dentro do mesmo passo.
    """

    def __init__(self, config: PI05DEPTHConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        largura = get_gemma_config(config.paligemma_variant).width

        if config.use_depth_3d:
            self.pointnet = PointNetEncoder(output_dim=largura)

        if config.use_pressure:
            self.pressure_proj = nn.Sequential(
                nn.Linear(config.pressure_feature_dim, largura),
                nn.ReLU(),
                nn.Linear(largura, largura),
            )

        self._depth_images: list[Tensor] | None = None
        self._pressure: Tensor | None = None

    def set_extra_inputs(self, depth_images=None, pressure=None) -> None:
        """Guarda as entradas extras para o próximo `embed_prefix`."""
        self._depth_images = depth_images
        self._pressure = pressure

    def embed_prefix(self, images, img_masks, tokens, masks):
        """Prefixo do upstream + 1 token de nuvem de pontos + 1 token de pressão.

        O upstream devolve os três tensores já concatenados, então acrescentar é
        `torch.cat` no fim. Os tokens novos entram com `att_mask = 0`, isto é,
        no mesmo bloco bidirecional das imagens e da linguagem — que é onde eles
        pertencem: são observação, não ação.
        """
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)

        extras = []

        if self.config.use_depth_3d and self._depth_images:
            # Só a PRIMEIRA câmera de profundidade. Os intrínsecos são um
            # dicionário único no config, então uma segunda câmera precisaria de
            # outro conjunto — não é o caso hoje.
            nuvem = depth_to_pointcloud(
                self._depth_images[0],
                self.config.camera_intrinsics,
                num_points=self.config.pointnet_num_points,
            )
            extras.append(self._apply_checkpoint(self.pointnet, nuvem).unsqueeze(1))

        if self.config.use_pressure and self._pressure is not None:
            extras.append(self._apply_checkpoint(self.pressure_proj, self._pressure).unsqueeze(1))

        if not extras:
            return embs, pad_masks, att_masks

        extra = torch.cat(extras, dim=1).to(dtype=embs.dtype, device=embs.device)
        bsize, n_extra = extra.shape[0], extra.shape[1]

        embs = torch.cat([embs, extra], dim=1)
        pad_masks = torch.cat(
            [pad_masks, torch.ones(bsize, n_extra, dtype=pad_masks.dtype, device=pad_masks.device)],
            dim=1,
        )
        att_masks = torch.cat(
            [att_masks, torch.zeros(bsize, n_extra, dtype=att_masks.dtype, device=att_masks.device)],
            dim=1,
        )
        return embs, pad_masks, att_masks


class PI05DEPTHPolicy(PI05Policy):
    """A Policy do upstream, trocando o modelo pelo enxertado.

    Acrescenta: separação das câmeras de profundidade antes do SigLIP, extração
    da pressão, o `neutral_position` e o gate de incerteza.
    """

    config_class = PI05DEPTHConfig
    name = "pi05depth"

    def __init__(self, config: PI05DEPTHConfig, **kwargs):
        # NÃO chama `PI05Policy.__init__`: ele instancia `PI05Pytorch` na mão, e
        # deixá-lo construir 3 B de parâmetros só para substituir depois custa
        # tempo e memória à toa. O corpo abaixo é o dele, com uma linha trocada
        # (marcada) — se o upstream mudar esse __init__, esta é a parte a revisar.
        from lerobot.policies.pretrained import PreTrainedPolicy
        from lerobot.utils.import_utils import require_package

        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05DepthPytorch(config, rtc_processor=self.rtc_processor)  # ← única troca

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        # Pose neutra do robô, salva no checkpoint. O gate de incerteza mistura
        # a ação prevista com ela quando a cena está incerta. Injetada pelo
        # `run_train` via `policies/act_depth/neutral_position.py`; zeros até lá.
        dim_acao = config.output_features[ACTION].shape[0]
        self.register_buffer("neutral_position", torch.zeros(dim_acao))

        self._last_uncertainty = 0.0
        self.reset()

    # ── Entradas extras ──────────────────────────────────────────────────────

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list, list]:
        """Manda para o SigLIP só as câmeras de COR, e guarda as de profundidade.

        O `_preprocess_images` do upstream itera `config.image_features` e
        empurra tudo para a torre visual. A profundidade é declarada VISUAL no
        YAML — é assim que o dataset a carrega e a encoda como vídeo de 1 canal —
        mas o caminho dela é o PointNet. A troca temporária de `input_features`
        é o jeito de reaproveitar o método do upstream inteiro (redimensionamento
        com padding, normalização para [-1,1], câmeras ausentes) sem copiá-lo.
        """
        dispositivo = next(self.parameters()).device

        # A profundidade vai CRUA, em milímetros, para o PointNet: o
        # `make_dataset` pula as câmeras de profundidade ao carimbar as stats do
        # ImageNet (`datasets/factory.py`), então não há normalização a desfazer.
        self._depth_cache = [
            batch[k].to(device=dispositivo, dtype=torch.float32)
            for k in self.config.depth_image_features
            if k in batch
        ]

        self.model.set_extra_inputs(
            depth_images=self._depth_cache,
            pressure=getattr(self, "_pressure_cache", None),
        )

        chaves_depth = set(self.config.depth_image_features)
        if not chaves_depth:
            return super()._preprocess_images(batch)

        originais = self.config.input_features
        try:
            self.config.input_features = {
                k: v for k, v in originais.items() if k not in chaves_depth
            }
            return super()._preprocess_images(batch)
        finally:
            self.config.input_features = originais

    def _extract_pressure(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Concatena as duas mãos: 33 + 33 = 66."""
        esquerda = batch.get("observation.left_hand_pressure")
        direita = batch.get("observation.right_hand_pressure")
        if esquerda is None or direita is None:
            return None
        return torch.cat([esquerda, direita], dim=1)

    # ── Treino e inferência ──────────────────────────────────────────────────
    #
    # A ORDEM é o que faz isto funcionar sem reescrever o `forward` e o
    # `sample_actions` do upstream: os dois chamam `_preprocess_images(batch)`
    # antes de qualquer coisa, e `embed_prefix` bem depois. Então quem entrega
    # as entradas extras ao modelo é o próprio `_preprocess_images` — o método
    # que já recebe o batch e já separa a profundidade. A pressão é guardada
    # aqui um instante antes, porque `_preprocess_images` não a conhece.

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        self._pressure_cache = self._extract_pressure(batch)
        return super().forward(batch, reduction)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self._pressure_cache = self._extract_pressure(batch)
        acoes = super().predict_action_chunk(batch, **kwargs)

        limiar = self.config.scene_uncertainty_threshold
        if limiar > 0:
            acoes = self._aplicar_gate_incerteza(acoes, batch, **kwargs)
        return acoes

    def _aplicar_gate_incerteza(self, acoes: Tensor, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Mistura a ação com a pose neutra quando a cena está incerta.

        A incerteza é o desvio-padrão médio entre `n_samples_uncertainty`
        amostragens com ruídos iniciais diferentes. Custa n passadas do expert —
        o prefixo (VLM + SigLIP) é o mesmo e não se repete.
        """
        n = max(1, self.config.n_samples_uncertainty)
        if n <= 1:
            self._last_uncertainty = 0.0
            return acoes

        amostras = [acoes]
        for _ in range(n - 1):
            amostras.append(super().predict_action_chunk(batch, **kwargs))
        incerteza = torch.stack(amostras, dim=0).std(dim=0).mean().item()
        self._last_uncertainty = incerteza

        limiar = self.config.scene_uncertainty_threshold
        if incerteza <= limiar:
            return acoes

        excesso = (incerteza - limiar) / (limiar + 1e-6)
        alfa = float(min(max(excesso, 0.0), 1.0))
        neutra = self.neutral_position.to(device=acoes.device, dtype=acoes.dtype)
        logger.warning(
            f"[UncertaintyGate/PI05] incerteza={incerteza:.4f} > limiar={limiar:.4f} "
            f"→ misturando {alfa:.0%} da pose neutra"
        )
        return (1.0 - alfa) * acoes + alfa * neutra.view(1, 1, -1)

    # ── Carregar o checkpoint base ───────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool | None = None,
        **kwargs,
    ) -> T:
        """Como o do upstream, mas ciente do enxerto.

        Duas diferenças:

        1. **`strict` vem do config.** Carregando o `lerobot/pi05_base`, o
           PointNet e a projeção de pressão não existem no checkpoint. Com o
           `strict=True` do upstream isso estoura antes do passo 1.
        2. **Exige um `PI05DEPTHConfig`.** Sem `config`, o upstream lê o
           `config.json` do repositório — e o do `pi05_base` diz `type: pi05`,
           que constrói um `PI05Config` sem `use_depth_3d`. O erro só apareceria
           lá adiante, como AttributeError no meio do treino. Melhor recusar aqui.
        """
        if config is None:
            raise ValueError(
                "PI05DEPTHPolicy.from_pretrained precisa de `config`.\n"
                f"O config.json de {pretrained_name_or_path} declara `type: pi05` e "
                "produziria um PI05Config sem os nossos campos (use_depth_3d, "
                "use_pressure, ...). Passe o config do SEU YAML — que é o que o "
                "`run_train` faz quando você define `pretrained_path`."
            )
        if not isinstance(config, PI05DEPTHConfig):
            raise TypeError(
                f"config deve ser PI05DEPTHConfig, veio {type(config).__name__}. "
                "Confira o `type: pi05depth` no YAML."
            )

        if strict is None:
            strict = bool(getattr(config, "pretrained_strict", False))

        return super().from_pretrained(
            pretrained_name_or_path, config=config, strict=strict, **kwargs
        )


__all__ = ["PI05DEPTHConfig", "PI05DEPTHPolicy", "PI05DepthPytorch", "pad_vector"]
