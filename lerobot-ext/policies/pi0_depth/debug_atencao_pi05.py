#!/usr/bin/env python
"""Onde o π0.5 olha: atenção das AÇÕES sobre as câmeras, para o painel de debug.

Mesmo espírito do `policies/fastwam_depth/debug_inferencia.py`: tudo é ligado
por fora e por tempo limitado. O `CapturaAtencaoPI05` troca uma função durante
UMA inferência e devolve a original no fim — o modelo não muda, e quem não liga
o painel não paga nada.

O QUE É CAPTURADO

  No `sample_actions` do upstream, o prefixo (imagens + texto) roda uma vez e
  vira cache; depois, a cada passo de denoising, os `chunk_size` tokens de ação
  do expert atendem a `[prefixo | ações]` (`denoise_step`). O expert usa o
  `GemmaAttention` do transformers, que resolve `eager_attention_forward` no
  módulo na hora da chamada — por isso trocar essa função enxerga os pesos, que
  o upstream já calcula e joga fora.

  Só entram as chamadas com `q = chunk_size` e `k > chunk_size`: é a assinatura
  das ações atendendo ao cache. A passada do prefixo (q = k) fica de fora.

  A média é sobre lote 0, cabeças, as 50 consultas de ação, as 18 camadas do
  expert e os 10 passos de denoising. Cada linha do softmax soma 1, então a
  média também soma 1 — e a "massa" por grupo (cabeça, pulso, texto, ações) é a
  fração da atenção que foi para cada um.

A GRADE

  O SigLIP a 224×224 com patch 14 dá 16×16 = 256 tokens por câmera, na ordem de
  `config.image_features`. O número de tokens é LIDO de `embed_image` durante a
  captura, e não suposto. A câmera da cabeça (848×480) entra no modelo com
  `resize_with_pad_torch`: reduzida a 224×126 e com faixas pretas em cima e
  embaixo. O mosaico do painel é montado do mesmo jeito, então cada célula do
  mapa cai sobre o pedaço da imagem que o token realmente viu — faixa preta
  incluída.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CapturaAtencaoPI05:
    """Colhe a atenção ações → prefixo de UMA inferência do π0.5.

    Uso:

        with CapturaAtencaoPI05(policy) as captura:
            acao = policy.select_action(batch)     # só se a fila estava vazia
        resumo = captura.resumo()                  # None se nada foi capturado
    """

    def __init__(self, policy):
        self.policy = policy
        self._modulo = None
        self._original = None
        self._pwe = None
        self._tokens_por_imagem: list[int] = []
        self._soma = None
        self._n = 0

    def __enter__(self):
        from transformers.models.gemma import modeling_gemma

        chunk = int(self.policy.config.chunk_size)
        captura = self
        original = modeling_gemma.eager_attention_forward

        def eager_com_captura(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            saida, pesos = original(module, query, key, value, attention_mask, scaling,
                                    dropout=dropout, **kwargs)
            if pesos is not None and pesos.shape[-2] == chunk and pesos.shape[-1] > chunk:
                # [B, cabeças, chunk, K] → [K]: lote 0, média em cabeças e consultas.
                vetor = pesos[0].float().mean(dim=(0, 1)).detach()
                captura._soma = vetor if captura._soma is None else captura._soma + vetor
                captura._n += 1
            return saida, pesos

        self._modulo = modeling_gemma
        self._original = original
        modeling_gemma.eager_attention_forward = eager_com_captura

        # Quantos tokens cada imagem vira, lido e não suposto.
        self._pwe = self.policy.model.paligemma_with_expert
        embed_original = self._pwe.embed_image

        def embed_com_registro(img):
            emb = embed_original(img)
            captura._tokens_por_imagem.append(int(emb.shape[1]))
            return emb

        self._pwe.embed_image = embed_com_registro
        return self

    def __exit__(self, *exc):
        if self._modulo is not None:
            self._modulo.eager_attention_forward = self._original
        if self._pwe is not None:
            # A troca foi um atributo de INSTÂNCIA por cima do método da classe:
            # apagar devolve o método original.
            self._pwe.__dict__.pop("embed_image", None)
        return False

    def resumo(self) -> dict | None:
        """`{"mapa": [16, 16 × n_câmeras], "massa": {câmera: f, "texto": f, "acoes": f}}`."""
        if self._soma is None or self._n == 0 or not self._tokens_por_imagem:
            logger.warning("[π0.5 debug] nenhuma atenção capturada nesta inferência")
            return None

        chunk = int(self.policy.config.chunk_size)
        media = (self._soma / self._n).cpu().numpy()
        k_prefixo = media.shape[0] - chunk
        nomes = [chave.rsplit(".", 1)[-1] for chave in self.policy.config.image_features]

        mapas, massa, ini = [], {}, 0
        for nome, n in zip(nomes, self._tokens_por_imagem):
            lado = int(round(n ** 0.5))
            fatia = media[ini:ini + n]
            massa[nome] = float(fatia.sum())
            mapas.append(fatia.reshape(lado, lado) if lado * lado == n else np.zeros((lado, lado)))
            ini += n
        massa["texto"] = float(media[ini:k_prefixo].sum())
        massa["acoes"] = float(media[k_prefixo:].sum())
        return {"mapa": np.hstack(mapas).astype(np.float32), "massa": massa}


class LinhaDeBaseDaAtencao:
    """Subtrai o que acende em QUALQUER quadro, para sobrar o que é desta cena.

    O mapa cru de um VLM costuma ter um "attention sink": poucos tokens (em geral
    os primeiros de cada imagem) recebem peso alto em todo quadro, sem relação
    com o conteúdo. A base é a média dos mapas já vistos (cumulativa nos
    primeiros `aquecimento` quadros, exponencial depois); o relativo é o que
    passa dela. É o mesmo tratamento que o painel do FastWAM-D usa.
    """

    def __init__(self, alfa: float = 0.05, aquecimento: int = 10):
        self.alfa = alfa
        self.aquecimento = aquecimento
        self._base = None
        self.n = 0

    def payload(self, resumo: dict) -> dict:
        mapa = resumo["mapa"]
        cru = mapa / max(float(mapa.max()), 1e-12)
        if self._base is None:
            relativo = cru
        else:
            relativo = np.clip(cru - self._base, 0.0, None)
            pico = float(relativo.max())
            relativo = relativo / pico if pico > 0 else relativo

        if self._base is None:
            self._base = cru.copy()
        else:
            peso = 1.0 / (self.n + 1) if self.n < self.aquecimento else self.alfa
            self._base = (1.0 - peso) * self._base + peso * cru
        self.n += 1
        return {"attn": cru, "attn_rel": relativo, "attn_base_n": self.n}


def mosaico_como_o_modelo(obs: dict, chaves_imagem, lado: int = 224) -> np.ndarray | None:
    """As câmeras lado a lado, cada uma como o SigLIP a recebe.

    Réplica de `resize_with_pad_torch` (lerobot/policies/common/vla_utils.py):
    razão = max(w/224, h/224), tamanho truncado com `int`, borda preta centrada
    com o resto da divisão na borda de baixo/direita. Alinhar o mapa com uma
    imagem esticada colocaria o calor fora do objeto.
    """
    import cv2

    quadros = []
    for chave in chaves_imagem:
        img = obs.get(chave.rsplit(".", 1)[-1])
        tela = np.zeros((lado, lado, 3), dtype=np.uint8)
        if img is not None:
            img = np.asarray(img)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            h, w = img.shape[:2]
            razao = max(w / lado, h / lado)
            rh, rw = int(h / razao), int(w / razao)
            reduzida = cv2.resize(img[..., :3], (rw, rh), interpolation=cv2.INTER_LINEAR)
            topo, esq = (lado - rh) // 2, (lado - rw) // 2
            tela[topo:topo + rh, esq:esq + rw] = reduzida
        quadros.append(tela)
    return np.hstack(quadros) if quadros else None


def profundidade_mm_para_painel(depth) -> np.ndarray | None:
    """Profundidade em MILÍMETROS e 2D, venha de onde vier.

    Robô real: o `full_realsenser_server.py` publica uint16 em mm, 1 canal —
    passa direto. MuJoCo: o `sim/base_sim.py` publica cinza de 8 bits em 3
    canais, com 0-2000 mm espremidos em 0-255 (~8 mm por nível). Entregue cru,
    o painel morria indexando uma máscara de 3 canais — e, se não morresse,
    leria 255 como 0,255 m.
    """
    if depth is None:
        return None
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    if d.dtype == np.uint8:
        mm = d.astype(np.float32) * (2000.0 / 255.0)
        # 255 é o teto do recorte (tudo além de 2 m), não uma medida: sem isto a
        # nuvem ganha uma parede plana a 2 m que não existe na cena.
        mm[d == 255] = 0.0
        return mm
    return d.astype(np.float32)
