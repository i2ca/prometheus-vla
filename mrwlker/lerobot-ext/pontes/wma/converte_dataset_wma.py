#!/usr/bin/env python
"""
Converte nosso dataset LeRobot v3.0 para o formato do UnifoLM-WMA
==================================================================
O `prepare_data/prepare_training_data.py` da Unitree lê **v2.1**: um parquet
por episódio (`data/chunk-000/episode_000000.parquet`), um mp4 por episódio
(`videos/chunk-000/<view>/episode_000000.mp4`) e `meta/tasks.jsonl`.

Os nossos datasets são **v3.0**, que é outra coisa:

  * os episódios vivem TODOS no mesmo parquet, delimitados por
    `dataset_from_index`/`dataset_to_index` em `meta/episodes/`;
  * os vídeos vivem TODOS no mesmo mp4, delimitados por `from_timestamp`/
    `to_timestamp` da mesma tabela;
  * `tasks.parquet` no lugar de `tasks.jsonl`.

Rodar o script deles na nossa árvore dá `FileNotFoundError` no primeiro
`episode_000000.parquet`. Este aqui faz o mesmo trabalho lendo v3.0.

O QUE SAI (exatamente o que o `WMAData` espera)

    target_dir/
      ├── videos/<nome>/<view>/{0,1,...}.mp4        H.264, um por episódio
      ├── transitions/<nome>/{0,1,...}.h5           observation.state + action
      ├── transitions/<nome>/meta_data/stats.safetensors
      └── <nome>.csv                                uma linha por episódio

O QUE FICA PARA TRÁS, DE PROPÓSITO

  * **Profundidade.** O WMA-0 é RGB de uma câmera só (`wma_data.py` monta o
    vídeo de UMA coluna `data_dir` do csv). O `head_camera_depth` não tem
    para onde ir sem mexer no modelo. É a diferença que importa contra o
    FastWAM-D, e está documentada em docs/UNIFOLM_WMA.md §3.
  * **Pressão das mãos.** Mesmo motivo: não existe entrada para tato.
  * **Câmera de pulso**, a menos que você peça `--views` com ela. O treino
    só consome a vista principal; mandar duas só inflaria o csv.

USO

    python pontes/wma/converte_dataset_wma.py \
        --origem meu_dataset/white_cup_on_dripper_2026-08-11 \
        --destino /data/wma_data \
        --nome white_cup_on_dripper \
        --robo "Unitree G1 Robot with Dex3 Hands"

`--confere` refaz a contagem de quadros de cada mp4 cortado contra o
comprimento do episódio no parquet. Vale a pena: corte de vídeo por timestamp
erra por um quadro com facilidade, e um episódio com vídeo mais curto que o
estado treina o modelo com imagem e ação fora de fase — sem erro nenhum.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

COLUNAS_CSV = [
    "videoid", "contentUrl", "duration", "data_dir", "instruction",
    "dynamic_confidence", "dynamic_wording", "dynamic_source_category",
    "embodiment",
]


def conta_quadros(video: Path) -> int:
    """Número de quadros do mp4, contando de verdade (e não pelo cabeçalho).

    `-count_frames` é lento, mas o `nb_frames` do cabeçalho vem errado em
    arquivo recortado: ele é copiado do stream original.
    """
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(saida) if saida.isdigit() else -1


def corta_video(entrada: Path, saida: Path, t0: float, t1: float) -> None:
    """Recorta [t0, t1) do mp4 concatenado e reescreve em H.264.

    O `-ss` vem DEPOIS do `-i` de propósito. Antes do `-i` o ffmpeg busca o
    keyframe anterior e o corte sai deslocado por até um GOP inteiro; depois
    do `-i` ele decodifica desde o começo e corta no quadro exato. É mais
    lento e é a única forma de o quadro 0 do episódio ser o quadro 0 do vídeo.

    H.264 porque o `WMAData` decodifica com decord, e o nosso `vcodec` padrão
    (AV1) não abre lá.
    """
    saida.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(entrada),
         "-ss", f"{t0:.6f}", "-to", f"{t1:.6f}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "23",
         "-pix_fmt", "yuv420p", "-an", str(saida)],
        check=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origem", required=True, help="raiz do dataset LeRobot v3.0")
    p.add_argument("--destino", required=True, help="onde escrever o formato WMA")
    p.add_argument("--nome", required=True, help="nome do dataset no csv e nos configs")
    p.add_argument("--robo", default="Unitree G1 Robot with Dex3 Hands",
                   help="etiqueta de embodiment; o WMA condiciona nela")
    p.add_argument("--views", nargs="+", default=["observation.images.head_camera"],
                   help="vistas a exportar (o treino usa só a primeira)")
    p.add_argument("--confere", action="store_true",
                   help="valida contagem de quadros contra o comprimento do episódio")
    p.add_argument("--limite", type=int, default=None, help="só os N primeiros episódios")
    args = p.parse_args()

    origem = Path(args.origem)
    destino = Path(args.destino)

    info = json.loads((origem / "meta" / "info.json").read_text())
    versao = info.get("codebase_version", "?")
    if not versao.startswith("v3"):
        print(f"❌ {origem} está em {versao}. Este script lê v3.0; para v2.1 use o "
              f"prepare_training_data.py da própria Unitree.", file=sys.stderr)
        return 1

    for chave in ("action", "observation.state"):
        if chave not in info["features"]:
            print(f"❌ falta a feature {chave!r} no info.json", file=sys.stderr)
            return 1
    dim = info["features"]["action"]["shape"][0]
    print(f">>> {args.nome}: v3.0, {info['total_episodes']} episódios, "
          f"ação de {dim} dimensões, {info['fps']} fps")
    if dim > 16:
        print(f"    ⚠️  o config do WMA nasce com agent_action_dim/agent_state_dim = 16. "
              f"Com {dim} dimensões, ajuste os DOIS no yaml — ver docs/UNIFOLM_WMA.md §4.")

    episodios = pd.read_parquet(origem / "meta" / "episodes")
    if args.limite:
        episodios = episodios.head(args.limite)
    tarefas = pd.read_parquet(origem / "meta" / "tasks.parquet")
    # O tasks.parquet tem a string no ÍNDICE e o task_index na coluna.
    por_indice = {int(v): str(k) for k, v in tarefas["task_index"].items()}

    dir_videos = destino / "videos" / args.nome
    dir_trans = destino / "transitions" / args.nome
    dir_meta = dir_trans / "meta_data"
    for d in (dir_videos, dir_trans, dir_meta):
        d.mkdir(parents=True, exist_ok=True)

    dados = pd.read_parquet(origem / "data")
    linhas, todas_acoes, todos_estados = [], [], []
    divergencias = []

    for _, ep in episodios.iterrows():
        idx = int(ep["episode_index"])
        fatia = dados.iloc[int(ep["dataset_from_index"]):int(ep["dataset_to_index"])]
        acoes = torch.tensor(np.stack(fatia["action"].to_numpy()), dtype=torch.float32)
        estados = torch.tensor(np.stack(fatia["observation.state"].to_numpy()),
                               dtype=torch.float32)

        # `tasks` é lista em v3.0 (um episódio pode ter mais de uma). O WMA
        # condiciona numa string só, então fica a primeira — e o aviso, porque
        # silenciosamente jogar fora a segunda tarefa é como se perde
        # multi-tarefa sem perceber.
        tarefa = ep["tasks"]
        if isinstance(tarefa, (list, np.ndarray)):
            if len(tarefa) > 1:
                print(f"    ⚠️  episódio {idx} tem {len(tarefa)} tarefas; usando a 1ª")
            tarefa = str(tarefa[0])
        else:
            tarefa = por_indice.get(int(tarefa), str(tarefa))

        with h5py.File(dir_trans / f"{idx}.h5", "w") as h5:
            h5.create_dataset("observation.state", data=estados.numpy())
            h5.create_dataset("action", data=acoes.numpy())
            h5.attrs["action_type"] = "joint position"
            h5.attrs["state_type"] = "joint position"
            h5.attrs["robot_type"] = args.robo
        todas_acoes.append(acoes)
        todos_estados.append(estados)

        for view in args.views:
            pref = f"videos/{view}"
            if f"{pref}/from_timestamp" not in ep:
                print(f"❌ vista {view!r} não existe neste dataset", file=sys.stderr)
                return 1
            fonte = origem / info["video_path"].format(
                video_key=view,
                chunk_index=int(ep[f"{pref}/chunk_index"]),
                file_index=int(ep[f"{pref}/file_index"]),
            )
            alvo = dir_videos / view / f"{idx}.mp4"
            corta_video(fonte, alvo, float(ep[f"{pref}/from_timestamp"]),
                        float(ep[f"{pref}/to_timestamp"]))
            if args.confere:
                n = conta_quadros(alvo)
                if n != len(fatia):
                    divergencias.append((idx, view, n, len(fatia)))

            linhas.append({
                "videoid": idx, "contentUrl": "x", "duration": "x",
                "data_dir": f"{args.nome}/{view}", "instruction": tarefa,
                "dynamic_confidence": "x", "dynamic_wording": "x",
                "dynamic_source_category": "x", "embodiment": args.robo,
            })
        print(f"    episódio {idx}: {len(fatia)} quadros — {tarefa!r}")

    acoes = torch.cat(todas_acoes, dim=0)
    estados = torch.cat(todos_estados, dim=0)
    stats = {}
    for nome, t in (("action", acoes), ("observation.state", estados)):
        stats[f"{nome}/max"] = t.max(dim=0).values
        stats[f"{nome}/min"] = t.min(dim=0).values
        stats[f"{nome}/mean"] = t.mean(dim=0)
        stats[f"{nome}/std"] = t.std(dim=0)
    save_file(stats, str(dir_meta / "stats.safetensors"))

    # O `load_stats` do WMA procura info.json e episode_data_index.safetensors
    # no meta_data junto do stats. Faltando, ele só avisa; copiar o nosso
    # info.json mantém a origem rastreável de dentro do dataset convertido.
    shutil.copy(origem / "meta" / "info.json", dir_meta / "info.json")

    pd.DataFrame(linhas, columns=COLUNAS_CSV).to_csv(destino / f"{args.nome}.csv",
                                                     index=False)

    print(f"\n>>> {len(episodios)} episódios em {destino}")
    print(f"    csv: {destino / f'{args.nome}.csv'}")
    if args.confere:
        if divergencias:
            print(f"\n❌ {len(divergencias)} vídeos com contagem diferente do parquet:")
            for idx, view, n, esperado in divergencias[:10]:
                print(f"    episódio {idx} [{view}]: {n} quadros no mp4, "
                      f"{esperado} no parquet")
            print("    Imagem e ação fora de fase treinam sem reclamar. Conserte antes.")
            return 1
        print("    ✓ contagem de quadros bate com o parquet em todos os episódios")
    else:
        print("    (rode de novo com --confere antes de treinar de verdade)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
