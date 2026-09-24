"""Acrescenta `write_video` ao `torchvision.io` da imagem, só se ele não existir.

Roda no build (ver Dockerfile). O `_write_video_compat.py` já tem que estar no
site-packages. Se um dia a imagem trouxer o `write_video` de volta, isto não faz nada.
"""
import pathlib

import torchvision.io as tio

if hasattr(tio, "write_video"):
    print("torchvision.io.write_video já existe; nada a fazer")
else:
    init = pathlib.Path(tio.__file__)
    init.write_text(
        init.read_text()
        + "\n\n# PROMETHEUS (lerobot-ext/pgx/wma): o torchvision 0.29 removeu write_video,\n"
        + "# e o UnifoLM-WMA grava todo mp4 com ele.\n"
        + "from _write_video_compat import write_video  # noqa: E402,F401\n"
    )
    print("write_video acrescentado em", init)
