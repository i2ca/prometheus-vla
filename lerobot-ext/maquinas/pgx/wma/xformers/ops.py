"""`memory_efficient_attention` com a semântica do xformers, sobre o SDPA do torch.

Todas as chamadas do UnifoLM-WMA (attention.py e conditional_unet1d.py) passam q/k/v em
3D, [B·H, N, D], e recebem [B·H, N, D] de volta. O `attn_bias` é None ou um tensor float
[B·H, N, M], convertido para `q.dtype` antes da chamada. No xformers esse bias é ADITIVO,
e um `attn_mask` float no SDPA também é, então a troca é exata. A escala padrão é 1/sqrt(D)
nos dois. O `op` escolhe kernel no xformers; aqui ele é aceito e ignorado.

Máscara booleana tem semântica OPOSTA entre os dois (no SDPA, True = pode atender). O WMA
não passa bool; se alguém passar, é erro, e não uma conversão silenciosa.
"""
import torch
import torch.nn.functional as F


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, op=None):
    if attn_bias is not None:
        if not isinstance(attn_bias, torch.Tensor):
            raise NotImplementedError(
                f"shim do xformers só aceita attn_bias tensor ou None, não {type(attn_bias).__name__}")
        if attn_bias.dtype == torch.bool:
            raise TypeError("attn_bias booleano tem semântica oposta no SDPA; o WMA passa float aditivo")
        attn_bias = attn_bias.to(query.dtype)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias,
                                          dropout_p=p, scale=scale)
