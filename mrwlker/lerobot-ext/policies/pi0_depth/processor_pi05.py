#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""
PI05-Depth — pipeline de pré e pós-processamento.
================================================================================
**O prompt é parte do checkpoint.** O π0.5 não recebe o estado por uma projeção
linear: ele **escreve o estado dentro do prompt**, discretizado em 256 níveis:

    Task: place the white cup on the dripper, State: 128 131 96 ... ;\\nAction:

Se a nossa string sair diferente da que o `lerobot/pi05_base` viu no treino, os
pesos continuam carregando e o modelo continua rodando — só que lendo um formato
que nunca viu. Por isso este arquivo **herda** o passo do upstream em vez de
copiá-lo: a formatação, os bins e a ordem dos passos vêm de lá.

── As duas diferenças que mantemos, e por quê ───────────────────────────────

1. **`pad_state_to_max`.** O passo do upstream discretiza o estado com a
   dimensão que chegar. O `pi05_base` foi treinado com `observation.state` de
   **32** (está no `config.json` dele), então o prompt do checkpoint tem 32
   números. O nosso robô manda 29. Sem completar até 32, o prompt tem
   comprimento diferente do que o checkpoint viu — e o estado é a única coisa
   que diz ao modelo onde o braço está. Padding com zeros DEPOIS da
   normalização, que é onde 0 cai no meio da faixa [-1, 1].

2. **`override_task`.** Força um prompt fixo, ignorando a `task` do dataset.
   DESLIGA o multi-tarefa — existe só para depurar.

O resto (renomear observações, batch, normalizar, tokenizar com o PaliGemma,
mover para o dispositivo) é montado com as mesmas peças do upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .configuration_pi05 import PI05DEPTHConfig


@ProcessorStepRegistry.register(name="pi05depth_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05DepthPrepareStateTokenizerProcessorStep(Pi05PrepareStateTokenizerProcessorStep):
    """O passo do upstream, com o estado completado até `max_state_dim` e o
    prompt opcionalmente forçado.

    A discretização, o texto e a ordem continuam sendo os do upstream: aqui só
    mexemos no que ENTRA nele.
    """

    override_task: str | None = None
    pad_state_to_max: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.pad_state_to_max:
            observacao = transition.get(TransitionKey.OBSERVATION)
            estado = observacao.get(OBS_STATE) if observacao else None
            if estado is not None and estado.shape[-1] < self.max_state_dim:
                transition = transition.copy()
                transition[TransitionKey.OBSERVATION] = {
                    **observacao,
                    OBS_STATE: pad_vector(estado, self.max_state_dim),
                }

        if self.override_task is not None:
            extra = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            tarefas = extra.get(self.task_key)
            if tarefas is not None:
                transition = transition.copy()
                transition[TransitionKey.COMPLEMENTARY_DATA] = {
                    **extra,
                    self.task_key: [self.override_task] * len(tarefas),
                }

        return super().__call__(transition)


def make_pi05depth_pre_post_processors(
    config: PI05DEPTHConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Mesma pipeline do `make_pi05_pre_post_processors`, com o nosso passo.

    A ordem é a do upstream e não é negociável: o `NormalizerProcessorStep` TEM
    de vir antes do passo de estado+tokenizer, porque a discretização em 256
    bins assume o estado já normalizado em [-1, 1].
    """
    passos = make_default_policy_processor_steps(config, dataset_stats)

    entrada: list[ProcessorStep] = [
        passos.rename_observations,
        passos.add_batch_dim,
        passos.normalize,
        Pi05DepthPrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            override_task=config.override_task,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        passos.to_device,
    ]

    saida: list[ProcessorStep] = [
        passos.unnormalize,
        passos.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=entrada, output_steps=saida)
