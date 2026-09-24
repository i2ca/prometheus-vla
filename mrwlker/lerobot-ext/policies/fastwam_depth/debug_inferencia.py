#!/usr/bin/env python
"""Instrumentação de depuração do FastWAM-D, para o servidor de inferência.

Tudo aqui é ligado por fora e por tempo limitado: o `CapturaDebug` é um
contexto que troca dois métodos, colhe o que precisa durante UMA inferência e
devolve os originais no fim. Nada disso mora no modelo — a política de treino
não paga nada por isto existir, e o servidor só liga quando o cliente pede.

O que se consegue ver:

  `mapa_atencao()`  — onde o expert de ação está olhando na imagem. No MoT,
      as consultas de ação atendem a `[K de vídeo | K de ação]`
      (`wan/modular.py::_forward_action_cached`), então a fatia de vídeo dessa
      atenção é literalmente "que pedaço da cena decidiu esta ação".

  `mapa_profundidade()` — o mapa de profundidade já normalizado, exatamente
      como o VAE do Wan o recebeu. É o único jeito honesto de conferir se a
      profundidade chega certa na inferência: comparar milímetros crus com o
      que o modelo vê depois do log e do recorte de faixa.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from einops import rearrange

logger = logging.getLogger(__name__)


class CapturaDebug:
    """Colhe atenção e profundidade de uma inferência do FastWAM-D.

    Uso:

        captura = CapturaDebug(policy)
        with captura:
            chunk = policy.predict_action_chunk(batch)
        mapa = captura.mapa_atencao()      # [h, w] em [0, 1], ou None
    """

    def __init__(self, policy: Any):
        self.policy = policy
        self._grade: tuple[int, int, int] | None = None   # (T, H, W) de tokens
        self._atencao: np.ndarray | None = None           # [S_video]
        self._cross: list[np.ndarray] = []                # por bloco, [S_video, L_texto]
        self._patchify_original = None
        self._mixed_original = None
        self._cross_original = None

    # ── contexto ────────────────────────────────────────────────────────────
    def __enter__(self) -> "CapturaDebug":
        self._instala_gancho_grade()
        self._instala_gancho_atencao()
        self._instala_gancho_cross()
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def remove(self) -> None:
        if self._patchify_original is not None:
            self.policy._expert_video.patchify = self._patchify_original
            self._patchify_original = None
        if self._mixed_original is not None:
            from lerobot.policies.fastwam.wan.modular import MoTLayer

            MoTLayer._mixed_attention = self._mixed_original
            self._mixed_original = None
        if self._cross_original is not None:
            from lerobot.policies.fastwam.wan.video_dit import FastWAMAttentionBlock

            FastWAMAttentionBlock.apply_cross_attention = self._cross_original
            self._cross_original = None

    # ── ganchos ─────────────────────────────────────────────────────────────
    def _instala_gancho_grade(self) -> None:
        """Anota a grade de tokens do vídeo, lendo a SAÍDA do `patch_embedding`.

        Ler a saída em vez de calcular `latente / patch_size` evita repetir
        aqui uma conta que já é feita lá — e que muda se o patch mudar. A
        `patchify` já vem embrulhada pelo enxerto de profundidade; embrulhamos
        de novo por cima, então a ordem na saída é: enxerto primeiro, anotação
        depois.
        """
        expert = self.policy._expert_video
        original = expert.patchify
        self._patchify_original = original

        def patchify_anotada(x: torch.Tensor):
            saida = original(x)
            # [B, dim, T, H, W] — a grade é o que sobra depois do stride do patch.
            if saida.dim() == 5:
                self._grade = tuple(int(v) for v in saida.shape[2:])
            return saida

        expert.patchify = patchify_anotada

    def _instala_gancho_atencao(self) -> None:
        from lerobot.policies.fastwam.wan.modular import MoTLayer

        original = MoTLayer._mixed_attention
        self._mixed_original = original
        captura = self

        def mixed_com_captura(self_layer, q_cat, k_cat, v_cat, attention_mask):
            captura._anota_atencao(self_layer, q_cat, k_cat)
            return original(self_layer, q_cat, k_cat, v_cat, attention_mask)

        MoTLayer._mixed_attention = mixed_com_captura

    def _anota_atencao(self, camada: Any, q_cat: torch.Tensor, k_cat: torch.Tensor) -> None:
        """Guarda a atenção ação→vídeo desta chamada.

        Só o caminho de ação interessa. Ele se distingue sozinho: nele o `k_cat`
        é `[K de vídeo | K de ação]` e portanto mais longo que o `q_cat`, que só
        tem as consultas de ação. No prefill do vídeo os dois têm o mesmo
        comprimento e a chamada é ignorada.

        Guardamos SEM acumular: cada camada de cada passo de denoise sobrescreve
        a anterior, então o que sobra no fim é a última camada do último passo —
        a mais próxima da ação que de fato foi emitida.
        """
        n_video = int(k_cat.shape[1]) - int(q_cat.shape[1])
        if n_video <= 0:
            return

        try:
            n = int(camada.num_heads)
            q = rearrange(q_cat.detach(), "b s (n d) -> b n s d", n=n)[:, :, :, :].float()
            k = rearrange(k_cat.detach()[:, :n_video], "b s (n d) -> b n s d", n=n).float()
            escala = 1.0 / np.sqrt(q.shape[-1])
            # [B, n, S_q, n_video] — pequeno: S_q é o horizonte de ação (dezenas)
            # e n_video é a grade latente (~1k), então isto cabe folgado mesmo
            # com o modelo inteiro na GPU.
            pesos = torch.softmax(q @ k.transpose(-1, -2) * escala, dim=-1)
            mapa = pesos.mean(dim=(0, 1, 2))
            self._atencao = mapa.to(torch.float32).cpu().numpy()
        except Exception as erro:  # noqa: BLE001 - depuração nunca derruba a inferência
            logger.warning(f"[FastWAM-D debug] atenção não capturada: {erro}")

    # ── resultados ──────────────────────────────────────────────────────────
    def mapa_atencao(self, normalizar: bool = True) -> np.ndarray | None:
        """Atenção sobre a imagem, como `[h, w]`. `None` se não deu.

        A grade tem uma dimensão temporal; na inferência o expert de vídeo roda
        só sobre o primeiro quadro, então ela costuma ser 1. Quando for maior,
        colapsamos pela média — o interesse é espacial.

        Com `normalizar=False` os valores saem como probabilidades de atenção
        (somam ~1 na grade). É o que o servidor usa para subtrair a linha de
        base: normalizar por quadro ANTES da subtração fixaria o sumidouro em
        1,0 sempre e apagaria justamente a variação que interessa.
        """
        if self._atencao is None or self._grade is None:
            return None

        t, h, w = self._grade
        atencao = self._atencao
        if atencao.size != t * h * w:
            # A grade anotada é do último `patchify`; se por algum motivo não
            # casar com o que a atenção viu, é melhor não desenhar nada do que
            # desenhar um mapa remontado errado.
            logger.warning(
                f"[FastWAM-D debug] atenção com {atencao.size} tokens, grade {t}x{h}x{w} — descartada."
            )
            return None

        mapa = atencao.reshape(t, h, w).mean(axis=0)
        if not normalizar:
            return mapa
        faixa = float(mapa.max() - mapa.min())
        if faixa <= 0:
            return np.zeros_like(mapa)
        return (mapa - float(mapa.min())) / faixa

    def mapa_profundidade(self) -> np.ndarray | None:
        """A profundidade normalizada que o VAE recebeu, como `[H, W]` em [0, 1].

        Sai do mosaico montado pelo `monta_video_profundidade`, no primeiro
        quadro: mesma largura do mosaico de cor, com a fatia de cada câmera no
        lugar dela. 0 é "sem medida", 1 é o `depth_max`.
        """
        video = getattr(self.policy, "_ultimo_video_depth", None)
        if video is None:
            return None
        try:
            # [B, 3, T, H, W] → primeiro item do batch, primeiro canal, primeiro quadro
            quadro = video[0, 0, 0]
            return quadro.detach().to(torch.float32).cpu().numpy()
        except Exception as erro:  # noqa: BLE001
            logger.warning(f"[FastWAM-D debug] profundidade não capturada: {erro}")
            return None

    # ── grounding: onde o TEXTO olha ────────────────────────────────────────
    def _instala_gancho_cross(self) -> None:
        """Colhe a cross-attention vídeo→texto de cada bloco do expert de vídeo.

        No robô real isto sai de graça: o prefill do expert de vídeo roda de
        qualquer jeito para produzir a ação, e a única conta a mais é a matriz
        `[patches, tokens]`, que é pequena. Ver `grounding_fastwamd.py` para o
        raciocínio completo sobre a métrica.
        """
        from lerobot.policies.fastwam.wan.video_dit import FastWAMAttentionBlock

        original = FastWAMAttentionBlock.apply_cross_attention
        self._cross_original = original
        captura = self

        def cross_com_captura(self_bloco, x, context, context_mask=None):
            captura._anota_cross(self_bloco, x, context)
            return original(self_bloco, x, context, context_mask=context_mask)

        FastWAMAttentionBlock.apply_cross_attention = cross_com_captura

    def _anota_cross(self, bloco: Any, x: torch.Tensor, context: torch.Tensor) -> None:
        """Recalcula os pesos de cross-attention deste bloco.

        Recalcular é obrigatório: o `fastwam_masked_attention` chama SDPA, que
        devolve só o resultado — a matriz de pesos nunca existe como tensor. As
        contas abaixo são cópia fiel do `video_dit.py::apply_cross_attention`;
        esquecer o `norm_q` daria um mapa plausível e errado.

        Só o expert de VÍDEO interessa, e ele se distingue pelo comprimento da
        consulta: a grade de patches (centenas) contra o horizonte de ação (32).
        """
        if self._grade is None:
            return
        if int(x.shape[1]) != int(np.prod(self._grade)):
            return
        try:
            attn = bloco.cross_attn
            n, d = int(attn.num_heads), int(attn.head_dim)
            q = attn.norm_q(attn.q(x.to(attn.q.weight.dtype)))
            k = attn.norm_k(attn.k(context.to(attn.k.weight.dtype)))
            q = q.detach().view(q.shape[0], q.shape[1], n, d).permute(0, 2, 1, 3).float()
            k = k.detach().view(k.shape[0], k.shape[1], n, d).permute(0, 2, 1, 3).float()
            pesos = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(d), dim=-1)
            self._cross.append(pesos.mean(dim=(0, 1)).cpu().numpy())
        except Exception as erro:  # noqa: BLE001 - depuração nunca derruba a inferência
            logger.warning(f"[FastWAM-D debug] cross-attention não capturada: {erro}")

    def mapa_grounding(self, indices: list[int], n_real: int,
                       so_primeira_camera: bool = True) -> np.ndarray | None:
        """Onde os tokens de `indices` aterrissam na imagem, como `[h, w]`.

        O valor é CONTRASTE ESPACIAL: a atenção deste patch à palavra dividida
        pela atenção do patch médio à mesma palavra. 1,0 é "olha tanto quanto
        qualquer outro"; acima disso é onde a palavra aterrissa.

        A divisão pela média da coluna não é cosmética — é o que torna o mapa
        legível. A atenção crua aqui tem dois sumidouros empilhados: ~87% da
        massa vai para as posições de padding do prompt (o `encode_prompt` faz
        `mask = torch.ones_like(mask)`, então a cross-attention enxerga as
        posições vazias), e do resto os tokens estruturais (`</s>`, `▁the`)
        levam metade, em TODO patch. Um sumidouro puxa massa em todo lugar,
        então dividir pela média espacial o achata em 1,0 e ele sai sozinho.

        `so_primeira_camera` recorta a fatia da câmera da cabeça: o mosaico tem
        as duas câmeras lado a lado na largura, e devolver o mapa inteiro para
        desenhar sobre uma imagem só desloca tudo.
        """
        if not self._cross or self._grade is None or not indices:
            return None
        pesos = np.stack(self._cross).mean(axis=0)          # [S_video, L_texto]
        n_real = min(int(n_real), pesos.shape[1])
        indices = [i for i in indices if i < n_real]
        if not indices:
            return None

        reais = pesos[:, :n_real]
        contraste = reais / np.clip(reais.mean(axis=0, keepdims=True), 1e-12, None)
        t, h, w = self._grade
        mapa = contraste[:, indices].mean(axis=1).reshape(t, h, w).mean(axis=0)
        return mapa[:, : w // 2] if so_primeira_camera and w >= 2 else mapa
