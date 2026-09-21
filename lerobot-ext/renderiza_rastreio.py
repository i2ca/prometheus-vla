#!/usr/bin/env python
"""
Desenha o vídeo de um rastreio do ER-1 a partir do CSV de pontos
================================================================
Separado do `rastreia_er1.py` de propósito: a inferência leva ~10 min de GPU,
o desenho leva segundos. Mudar a cara do vídeo não pode custar outra rodada do
modelo.

O que sai:
* 2× a resolução da câmera (848×480 → 1696×960). A câmera da cabeça grava a
  ~775 kbps, então ampliar não inventa detalhe — mas deixa mira e texto
  nítidos, e aguenta a recompressão do WhatsApp;
* cada alvo com uma cor e um número; o nome fica só na legenda, então dois
  pontos no mesmo lugar não embaralham os rótulos;
* aviso quando dois alvos caem a menos de 45 px (a régua do UNIFOLM_WMA.md §8.1);
* H.264 High + yuv420p + faststart, direto do ffmpeg: toca em WhatsApp,
  Google Drive e navegador.

USO

    python renderiza_rastreio.py --dataset meu_dataset/white_cup_on_dripper_2026-08-11 \
        --episodio 1 --csv rastreio_ep1.csv --saida rastreio_ep1.mp4
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

ESCALA = 2
# RGB. Contorno preto em volta de tudo: a toalha da mesa é branca.
CORES = [(230, 57, 70), (255, 183, 3), (46, 196, 182), (131, 56, 236), (58, 134, 255)]
COLAPSO_PX = 45
FONTE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTE_NEGRITO = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def le_pontos(caminho: Path) -> tuple[dict[int, dict[str, tuple[float, float] | None]], list[str]]:
    """CSV quadro,alvo,x,y → {quadro: {alvo: (x, y) ou None}}. x vazio = sem resposta."""
    medidas: dict[int, dict] = {}
    alvos: list[str] = []
    with open(caminho) as f:
        for linha in csv.DictReader(f):
            alvo = linha["alvo"]
            if alvo not in alvos:
                alvos.append(alvo)
            pt = (float(linha["x"]), float(linha["y"])) if linha["x"] else None
            medidas.setdefault(int(linha["quadro"]), {})[alvo] = pt
    return medidas, alvos


def renderiza(fonte: Path, pontos_csv: Path, saida: Path, fps: float, titulo: str,
              alvos: list[str] | None = None, limite: int | None = None) -> int:
    """Desenha os pontos do CSV sobre o vídeo `fonte`. Devolve o número de quadros."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    medidas, alvos_csv = le_pontos(pontos_csv)
    alvos = alvos or alvos_csv

    cap = cv2.VideoCapture(str(fonte))
    larg = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limite:
        total = min(total, limite)
    W, H = larg * ESCALA, alt * ESCALA

    f_txt = ImageFont.truetype(FONTE, 26)
    f_neg = ImageFont.truetype(FONTE_NEGRITO, 26)
    f_tag = ImageFont.truetype(FONTE_NEGRITO, 22)

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-an", str(saida)],
        stdin=subprocess.PIPE)

    preto = (0, 0, 0, 255)
    ultimo: dict[str, tuple[float, float] | None] = {a: None for a in alvos}
    n = 0
    while n < total:
        ok, quadro = cap.read()
        if not ok:
            break
        # Com --passo > 1 os quadros do meio seguram a última resposta.
        for alvo, pt in medidas.get(n, {}).items():
            if alvo in ultimo:
                ultimo[alvo] = pt

        img = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)).resize(
            (W, H), Image.LANCZOS).convert("RGBA")
        camada = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(camada)

        # Faixa de cima.
        d.rectangle([0, 0, W, 48], fill=(0, 0, 0, 170))
        d.text((16, 9), titulo, font=f_neg, fill="white")
        d.text((W - 16, 9), f"quadro {n}/{total - 1}    {n / fps:4.1f} s",
               font=f_txt, fill="white", anchor="ra")

        # Miras.
        for i, alvo in enumerate(alvos):
            pt = ultimo[alvo]
            if pt is None:
                continue
            cor = CORES[i % len(CORES)] + (255,)
            x, y = pt[0] * ESCALA, pt[1] * ESCALA
            r = 26
            d.ellipse([x - r, y - r, x + r, y + r], outline=preto, width=9)
            d.ellipse([x - r, y - r, x + r, y + r], outline=cor, width=5)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                seg = [x + dx * (r + 4), y + dy * (r + 4), x + dx * (r + 18), y + dy * (r + 18)]
                d.line(seg, fill=preto, width=7)
                d.line(seg, fill=cor, width=3)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=cor, outline=preto, width=2)
            # Etiqueta: pares em cima, ímpares embaixo — não se sobrepõem no colapso.
            tx = min(x + r + 10, W - 40)
            ty = y - r - 30 if i % 2 == 0 else y + r + 4
            ty = max(54, min(ty, H - 34))
            d.rounded_rectangle([tx, ty, tx + 30, ty + 30], 6, fill=cor, outline=preto, width=2)
            d.text((tx + 15, ty + 15), str(i + 1), font=f_tag, fill="black", anchor="mm")

        # Legenda.
        linhas = [(CORES[i % len(CORES)], f"{i + 1}  {a}" + ("" if ultimo[a] else "   (sem resposta)"))
                  for i, a in enumerate(alvos)]
        for i in range(len(alvos)):
            for j in range(i + 1, len(alvos)):
                pi, pj = ultimo[alvos[i]], ultimo[alvos[j]]
                if pi and pj and math.dist(pi, pj) < COLAPSO_PX:
                    linhas.append((None, f"⚠  {i + 1} e {j + 1} no mesmo lugar"))
        largura = max(d.textlength(t, font=f_txt) for _, t in linhas) + 64
        altura = 14 + 38 * len(linhas)
        y0 = H - 14 - altura
        d.rounded_rectangle([14, y0, 14 + largura, H - 14], 10, fill=(0, 0, 0, 175))
        for k, (cor, texto) in enumerate(linhas):
            yl = y0 + 10 + 38 * k
            if cor is None:
                d.text((30, yl), texto, font=f_neg, fill=(255, 110, 110))
            else:
                d.ellipse([30, yl + 6, 50, yl + 26], fill=cor + (255,), outline=preto, width=2)
                d.text((62, yl), texto, font=f_txt, fill="white")

        ffmpeg.stdin.write(Image.alpha_composite(img, camada).convert("RGB").tobytes())
        n += 1

    cap.release()
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit("❌ ffmpeg falhou ao codificar o vídeo")
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodio", type=int, default=0)
    p.add_argument("--view", default="observation.images.head_camera")
    p.add_argument("--csv", required=True)
    p.add_argument("--saida", default="rastreio_er1.mp4")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from rastreia_er1 import corta_episodio

    tmp = Path(args.saida).with_suffix(".fonte.mp4")
    fonte, fps, _ = corta_episodio(Path(args.dataset), args.episodio, args.view, tmp)
    n = renderiza(fonte, Path(args.csv), Path(args.saida), fps,
                  f"UnifoLM-ER-1  ·  episódio {args.episodio}")
    tmp.unlink(missing_ok=True)
    print(f">>> {n} quadros → {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
