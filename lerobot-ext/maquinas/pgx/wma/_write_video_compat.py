"""`torchvision.io.write_video` de volta, para o UnifoLM-WMA na imagem NGC 26.08.

O torchvision 0.29 da imagem removeu `torchvision.io.write_video`, porque o vídeo passou
para o torchcodec. O `save_results` do `world_model_interaction.py` e o
`unifolm_wma/utils/save_video.py` gravam todo mp4 com essa função. Sem ela, a rodada
morre no FIM, depois de minutos de GPU. Esta é a mesma interface, reimplementada com
PyAV, e o submódulo da Unitree fica intacto.

O encoder é conferido antes de abrir o arquivo. Se o pedido não existir no FFmpeg
embutido no wheel do PyAV, usa o próximo da lista e AVISA, porque o codec do mp4 muda.
"""
import warnings
from fractions import Fraction

# `h264` é o nome que a Unitree passa; no FFmpeg o encoder de verdade é o libx264.
_NOME_REAL = {"h264": "libx264"}
_RESERVA = ("libx264", "h264", "libopenh264", "mpeg4")


def _encoder_disponivel(nome):
    import av

    try:
        av.codec.Codec(nome, "w")
        return True
    except Exception:
        return False


def write_video(filename, video_array, fps, video_codec="libx264", options=None,
                audio_array=None, audio_fps=None, audio_codec=None, audio_options=None):
    """Grava `video_array` [T, H, W, C] uint8 em `filename`, como o torchvision <0.29."""
    import av
    import numpy as np
    import torch

    if audio_array is not None:
        raise NotImplementedError("este write_video não grava áudio")

    if isinstance(video_array, torch.Tensor):
        quadros = video_array.detach().cpu().numpy()
    else:
        quadros = np.asarray(video_array)
    quadros = quadros.astype(np.uint8)

    pedido = _NOME_REAL.get(video_codec, video_codec)
    candidatos = [pedido] + [c for c in _RESERVA if c != pedido]
    nome = next((c for c in candidatos if _encoder_disponivel(c)), None)
    if nome is None:
        raise RuntimeError(f"nenhum encoder de vídeo disponível no PyAV entre {candidatos}")
    if nome != pedido:
        warnings.warn(f"write_video: encoder {pedido!r} indisponível no PyAV; gravando com {nome!r}")

    # yuv420p exige lado par; o torchvision antigo também cortava.
    _, altura, largura, _ = quadros.shape
    altura, largura = altura - altura % 2, largura - largura % 2

    with av.open(str(filename), mode="w") as saida:
        stream = saida.add_stream(nome, rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height = largura, altura
        stream.pix_fmt = "yuv420p"
        # `crf` e companhia são opções do libx264; em outro encoder elas não significam nada.
        if options and nome == "libx264":
            stream.options = {k: str(v) for k, v in options.items()}
        for quadro in quadros:
            q = av.VideoFrame.from_ndarray(np.ascontiguousarray(quadro[:altura, :largura]), format="rgb24")
            for pacote in stream.encode(q):
                saida.mux(pacote)
        for pacote in stream.encode():
            saida.mux(pacote)
