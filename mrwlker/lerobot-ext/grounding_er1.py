#!/usr/bin/env python
"""
Grounding com o UnifoLM-ER-1 — onde o TEXTO aterrissa, segundo a Unitree
=========================================================================
Irmão do `grounding_fastwamd.py`, respondendo a MESMA pergunta por outro
caminho. Aquele mede a atenção de linguagem do prior do Wan e precisa
contornar dois sumidouros empilhados (87% da massa no padding). Este
simplesmente PERGUNTA a um modelo treinado para apontar.

O ER-1 é a metade publicada do UnifoLM-WLA-1.0: 4 B sobre Qwen3-VL-4B,
treinado em apontamento, detecção, trajetória 2D e QA espacial. Não é
política — não sai ação daqui. Serve como linha de base honesta: se o ER-1
acerta a caneca e a atenção do Wan não, o problema do FastWAM-D é de dados de
fine-tune, não de percepção.

CABE EM 8 GB? Só quantizado. Em bf16 são ~8,9 GB de pesos, acima da VRAM do
notebook. Em NF4 (`--4bit`, padrão) caem para ~3 GB.

USO

    python grounding_er1.py --imagem quadro.png --alvo "the white cup"
    python grounding_er1.py --dataset meu_dataset/white_cup_on_dripper_2026-08-11 \
                            --episodio 0 --alvo "the white cup"

A saída é uma cópia da imagem com o ponto marcado, mais o texto cru do
modelo — que fica impresso de propósito: o formato do apontamento é do
modelo, não nosso, e quando ele muda é aqui que se vê.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

MODELO = "unitreerobotics/UnifoLM-ER-1"


def quadro_do_dataset(raiz: Path, episodio: int, saida: Path) -> Path:
    """Extrai o primeiro quadro de um episódio do nosso dataset v3.0."""
    import pandas as pd

    info = json.loads((raiz / "meta" / "info.json").read_text())
    eps = pd.read_parquet(raiz / "meta" / "episodes")
    ep = eps[eps["episode_index"] == episodio].iloc[0]
    view = "observation.images.head_camera"
    src = raiz / info["video_path"].format(
        video_key=view,
        chunk_index=int(ep[f"videos/{view}/chunk_index"]),
        file_index=int(ep[f"videos/{view}/file_index"]),
    )
    t0 = float(ep[f"videos/{view}/from_timestamp"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ss", f"{t0:.6f}", "-frames:v", "1", str(saida)], check=True)
    print(f">>> episódio {episodio}: {ep['tasks']}")
    return saida


def extrai_pontos(texto: str, larg: int, alt: int,
                  pixels_absolutos: bool = False) -> list[tuple[float, float]]:
    """Tira coordenadas do texto do modelo e as converte para pixel.

    O ER-1 aponta em JSON (`{"point_2d": [x, y]}`) numa escala **0-1000
    normalizada**, e não em pixels. Medido neste checkpoint, no quadro 0 do
    `white_cup_on_dripper`, 848x480:

        alvo              resposta     como 0-1000    onde está
        caneca branca     (699, 254)   (593, 122)     (595, 110)  ✓
        coador            (345, 265)   (293, 127)     (295,  95)  ✓
        mão do robô       (965, 181)   (818,  87)     (800,  90)  ✓

    O `965` é a prova: é maior que a largura da imagem que o processador
    entrega ao modelo (728 px) e menor que 1000.

    **Não tente adivinhar a escala por heurística.** A tentação é reescalar
    só quando a coordenada estoura a imagem — mas 699 cabe em 848, e o ponto
    sai 100 px ao lado da caneca sem nada parecer errado. Se um dia o modelo
    passar a responder em pixel, use `--pixels`; a escala é uma decisão, não
    um palpite.
    """
    pontos: list[tuple[float, float]] = []
    for bloco in re.findall(r"\{[^{}]*\}", texto):
        try:
            d = json.loads(bloco)
        except json.JSONDecodeError:
            continue
        for chave in ("point_2d", "point", "coordinate", "position"):
            v = d.get(chave)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                pontos.append((float(v[0]), float(v[1])))
    for x, y in re.findall(r"<point>\s*([\d.]+)[,\s]+([\d.]+)\s*</point>", texto):
        pontos.append((float(x), float(y)))
    if not pontos:
        for x, y in re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", texto):
            pontos.append((float(x), float(y)))

    if not pixels_absolutos:
        pontos = [(x / 1000 * larg, y / 1000 * alt) for x, y in pontos]
    return pontos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    fonte = p.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--imagem", help="png/jpg para analisar")
    fonte.add_argument("--dataset", help="raiz de um dataset LeRobot v3.0")
    p.add_argument("--episodio", type=int, default=0)
    p.add_argument("--alvo", default="the white cup",
                   help="o que procurar, em inglês (o modelo foi treinado em inglês)")
    p.add_argument("--pergunta", default=None,
                   help="prompt inteiro, no lugar do template de apontamento")
    p.add_argument("--saida", default="grounding_er1.png")
    p.add_argument("--4bit", dest="quatro_bits", action="store_true", default=True)
    p.add_argument("--bf16", dest="quatro_bits", action="store_false",
                   help="sem quantizar (~8,9 GB de VRAM)")
    p.add_argument("--pixels", action="store_true",
                   help="tratar a resposta como pixel absoluto em vez de 0-1000")
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, AutoModelForImageTextToText

    caminho = Path(args.imagem) if args.imagem else quadro_do_dataset(
        Path(args.dataset), args.episodio, Path(args.saida).with_name("_quadro.png"))
    imagem = Image.open(caminho).convert("RGB")
    larg, alt = imagem.size
    print(f">>> imagem {caminho} ({larg}x{alt})")

    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    if args.quatro_bits:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    print(f">>> carregando {MODELO} ({'NF4' if args.quatro_bits else 'bf16'})...")
    proc = AutoProcessor.from_pretrained(MODELO)
    modelo = AutoModelForImageTextToText.from_pretrained(MODELO, **kwargs).eval()
    print(f"    VRAM ocupada: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pergunta = args.pergunta or (
        f"Point to {args.alvo} in the image. "
        f"Respond with the pixel coordinates as JSON: "
        f'{{"point_2d": [x, y], "label": "{args.alvo}"}}')
    msgs = [{"role": "user", "content": [{"type": "image", "image": imagem},
                                         {"type": "text", "text": pergunta}]}]
    entradas = proc.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(modelo.device)

    with torch.inference_mode():
        saida = modelo.generate(**entradas, max_new_tokens=args.max_tokens,
                                do_sample=False)
    texto = proc.decode(saida[0][entradas["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()

    print(f"\n--- pergunta ---\n{pergunta}\n--- resposta crua ---\n{texto}\n")

    pontos = extrai_pontos(texto, larg, alt, args.pixels)
    if not pontos:
        print("⚠️  nenhuma coordenada reconhecida na resposta. O texto cru está acima;")
        print("    se o formato mudou, ajuste `extrai_pontos`.")
        return 1

    desenho = ImageDraw.Draw(imagem)
    for x, y in pontos:
        r = max(6, min(larg, alt) // 60)
        desenho.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=3)
        desenho.line([x - r * 2, y, x + r * 2, y], fill=(255, 0, 0), width=2)
        desenho.line([x, y - r * 2, x, y + r * 2], fill=(255, 0, 0), width=2)
        print(f"    ponto: ({x:.0f}, {y:.0f})  —  {x/larg*100:.0f}% da largura, "
              f"{y/alt*100:.0f}% da altura")
    imagem.save(args.saida)
    print(f"\n>>> {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
