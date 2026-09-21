"""Stub do decord para a imagem do UnifoLM-WMA na PGX (aarch64, sem wheel do decord).

O `unifolm_wma/data/wma_data.py` importa `VideoReader` e `cpu` no topo, mas só os usa ao
ler vídeo de treino, no `__getitem__`. O modo de interação nunca chega lá. Se chegar, é
para falhar alto, e não para devolver quadro vazio.
"""


def cpu(_=0):
    return None


class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "decord é um stub nesta imagem (não há wheel aarch64). A leitura de vídeo "
            "de treino não funciona aqui; use a athena ou o notebook para treinar.")
