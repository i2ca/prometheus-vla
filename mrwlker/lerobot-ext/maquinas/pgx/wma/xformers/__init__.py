"""Shim do xformers para o UnifoLM-WMA na PGX (aarch64 + CUDA 13, sem wheel do xformers).

O `unifolm_wma/modules/attention.py` importa o xformers num try/except, mas o caminho sem
ele NÃO é uma alternativa: o `forward` comum tem `assert 1 > 2, "should setup xformers..."`
na cross-attention de imagem. A 2ª rodada na PGX, em 15/09, morreu nesse assert. O modelo
só roda pelo `efficient_forward`, e ele só é escolhido quando `import xformers` funciona.

Este pacote só expõe o que o WMA usa, `xformers.ops.memory_efficient_attention`, em cima do
`scaled_dot_product_attention` do torch. Ver `ops.py`.
"""

__version__ = "0.0.0+prometheus-sdpa-shim"
