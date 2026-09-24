#!/usr/bin/env python
"""
Rastreia objetos num episódio com o UnifoLM-ER-1 e sai um vídeo
================================================================
Roda o ER-1 quadro a quadro sobre um episódio do nosso dataset, grava onde ele
diz que cada alvo está num CSV e desenha o vídeo com `renderiza_rastreio.py`.

NÃO É UM RASTREADOR. Cada quadro é uma pergunta independente — o modelo não
sabe o que respondeu no quadro anterior, não tem estado, não faz associação
temporal. Por isso o ponto pula: dois quadros quase idênticos podem receber
respostas diferentes, e isso é informação, não defeito. Um ponto que fica
parado quando o objeto está parado é evidência de que o modelo está mesmo
vendo o objeto; um ponto que dança sobre uma mesa branca uniforme é evidência
do contrário.

O CUSTO. O ER-1 em NF4 leva ~0,85 s por pergunta nesta placa. Um episódio de
700 quadros com um alvo são ~10 minutos; com dois alvos, ~20. O `--passo`
existe para isso: com `--passo 5` ele pergunta a cada 5 quadros e o vídeo
segura a última resposta nos quadros do meio.

O CSV fica sempre (por padrão ao lado do mp4): é dele que se mede o colapso, e
é dele que se redesenha o vídeo sem rodar o modelo de novo:

    python renderiza_rastreio.py --dataset ... --episodio 1 --csv X.csv --saida X.mp4

USO

    python rastreia_er1.py --dataset meu_dataset/white_cup_on_dripper_2026-08-11 \
                           --episodio 0 --alvo "the white cup"

    # dois alvos, todos os quadros
    python rastreia_er1.py --dataset meu_dataset/... --episodio 1 \
        --alvo "the white cup" "the coffee dripper" --saida rastreio_ep1.mp4
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODELO = "unitreerobotics/UnifoLM-ER-1"


def corta_episodio(raiz: Path, episodio: int, view: str, saida: Path) -> tuple[Path, float, int]:
    """Recorta um episódio do vídeo concatenado do v3.0. Devolve (arquivo, fps, n_quadros)."""
    import pandas as pd

    info = json.loads((raiz / "meta" / "info.json").read_text())
    eps = pd.read_parquet(raiz / "meta" / "episodes")
    linha = eps[eps["episode_index"] == episodio]
    if linha.empty:
        raise SystemExit(f"❌ episódio {episodio} não existe (o dataset tem "
                         f"{info['total_episodes']})")
    ep = linha.iloc[0]
    pref = f"videos/{view}"
    if f"{pref}/from_timestamp" not in ep:
        raise SystemExit(f"❌ vista {view!r} não existe neste dataset")
    src = raiz / info["video_path"].format(
        video_key=view,
        chunk_index=int(ep[f"{pref}/chunk_index"]),
        file_index=int(ep[f"{pref}/file_index"]),
    )
    # `-ss` depois do `-i`: corte no quadro exato, e não no keyframe anterior.
    # crf 12: este arquivo é fonte do desenho, não pode perder mais qualidade.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ss", f"{float(ep[f'{pref}/from_timestamp']):.6f}",
                    "-to", f"{float(ep[f'{pref}/to_timestamp']):.6f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                    "-pix_fmt", "yuv420p", "-an", str(saida)], check=True)
    tarefa = ep["tasks"]
    if isinstance(tarefa, (list, tuple)) or hasattr(tarefa, "tolist"):
        tarefa = list(tarefa)[0]
    print(f">>> episódio {episodio}: {int(ep['length'])} quadros — {tarefa!r}")
    return saida, float(info["fps"]), int(ep["length"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodio", type=int, default=0)
    p.add_argument("--alvo", nargs="+", default=["the white cup"],
                   help="um ou mais alvos, em inglês")
    p.add_argument("--view", default="observation.images.head_camera")
    p.add_argument("--passo", type=int, default=1,
                   help="perguntar a cada N quadros (os do meio seguram a última resposta)")
    p.add_argument("--limite", type=int, default=None, help="só os N primeiros quadros")
    p.add_argument("--saida", default="rastreio_er1.mp4")
    p.add_argument("--csv", default=None,
                   help="quadro,alvo,x,y (x vazio = sem resposta); padrão: ao lado do mp4")
    p.add_argument("--bf16", dest="quatro_bits", action="store_false", default=True,
                   help="sem quantizar (~8,9 GB de VRAM)")
    args = p.parse_args()

    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    sys.path.insert(0, str(Path(__file__).parent))
    from grounding_er1 import extrai_pontos
    from renderiza_rastreio import renderiza

    saida = Path(args.saida)
    caminho_csv = Path(args.csv) if args.csv else saida.with_suffix(".csv")
    tmp = saida.with_suffix(".fonte.mp4")
    fonte, fps, n_total = corta_episodio(Path(args.dataset), args.episodio, args.view, tmp)

    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    if args.quatro_bits:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    print(f">>> carregando {MODELO} ({'NF4' if args.quatro_bits else 'bf16'})...")
    proc = AutoProcessor.from_pretrained(MODELO)
    modelo = AutoModelForImageTextToText.from_pretrained(MODELO, **kwargs).eval()
    print(f"    VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def pergunta(imagem: "Image.Image", alvo: str) -> list:
        q = (f'Point to {alvo} in the image. Respond with the pixel coordinates '
             f'as JSON: {{"point_2d": [x, y], "label": "{alvo}"}}')
        msgs = [{"role": "user", "content": [{"type": "image", "image": imagem},
                                             {"type": "text", "text": q}]}]
        e = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt").to(modelo.device)
        with torch.inference_mode():
            o = modelo.generate(**e, max_new_tokens=64, do_sample=False)
        texto = proc.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True)
        return extrai_pontos(texto, imagem.width, imagem.height)

    limite = min(n_total, args.limite) if args.limite else n_total
    print(f">>> {limite} quadros @ {fps:g} fps | {len(args.alvo)} alvo(s) | passo {args.passo}")

    cap = cv2.VideoCapture(str(fonte))
    sem_resposta = {a: 0 for a in args.alvo}
    n = 0
    t_ini = time.perf_counter()
    ms_modelo = 0.0
    n_perguntas = 0

    with open(caminho_csv, "w") as csv_saida:
        csv_saida.write("quadro,alvo,x,y\n")
        while n < limite:
            ok, quadro = cap.read()
            if not ok:
                break
            if n % args.passo == 0:
                img = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
                for alvo in args.alvo:
                    t0 = time.perf_counter()
                    pontos = pergunta(img, alvo)
                    ms_modelo += (time.perf_counter() - t0) * 1000
                    n_perguntas += 1
                    if pontos:
                        csv_saida.write(f'{n},"{alvo}",{pontos[0][0]:.1f},{pontos[0][1]:.1f}\n')
                    else:
                        csv_saida.write(f'{n},"{alvo}",,\n')
                        sem_resposta[alvo] += 1
                csv_saida.flush()
            n += 1
            if n % 20 == 0:
                print(f"    {n}/{limite} quadros | {ms_modelo/max(n_perguntas,1):.0f} ms por pergunta",
                      flush=True)
    cap.release()
    print(f">>> pontos em {caminho_csv}")

    dur = time.perf_counter() - t_ini
    print(f"\n>>> {n} quadros em {dur/60:.1f} min "
          f"({ms_modelo/max(n_perguntas,1):.0f} ms por pergunta, {n_perguntas} perguntas)")
    for alvo, faltas in sem_resposta.items():
        medidos = (n + args.passo - 1) // args.passo
        print(f"    {alvo!r}: {medidos - faltas}/{medidos} quadros com ponto"
              + (f"  ⚠️  {faltas} sem resposta" if faltas else ""))

    print(">>> desenhando o vídeo...")
    renderiza(fonte, caminho_csv, saida, fps, f"UnifoLM-ER-1  ·  episódio {args.episodio}",
              alvos=args.alvo, limite=limite)
    tmp.unlink(missing_ok=True)
    print(f">>> {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
