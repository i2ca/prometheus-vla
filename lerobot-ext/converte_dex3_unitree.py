#!/usr/bin/env python
"""
Converte os datasets públicos do G1 + Dex3 para o NOSSO schema v2.
================================================================================
Os `unitreerobotics/G1_Dex3_*` são o mesmo robô e as mesmas mãos que os nossos
(ver `docs/DATASETS_EXTERNOS.md`): 30 fps, 28 dims, e a ordem das juntas bate
BIT A BIT com `Dex3_1_Left_JointIndex` / `Dex3_1_Right_JointIndex`
(`robot/unitree_g1/g1_utils.py:202` e `:213`), inversão do lado direito
incluída. O que NÃO bate são as câmeras, o yaw do tronco e as pressões — e é
disto que este script cuida.

    python converte_dex3_unitree.py --origem unitreerobotics/G1_Dex3_Pouring_Dataset \\
        --destino meu_dataset/ext_pouring --episodios 5 --confere

Depois, junte com o resto pelo caminho de sempre (`train/build_multitask_dataset.py`
ou `aggregate_datasets`).

── Por que o schema vem de um dataset de REFERÊNCIA, e não escrito aqui ──────
`features_equal_for_merge` (`lerobot/datasets/feature_utils.py:125`) compara o
dicionário inteiro das features não-vídeo, `names` incluído. Escrever o schema à
mão aqui criaria uma terceira cópia (já existem a do robô e a do
`gerar_dataset_mujoco.py`) que diverge na primeira correção — foi exatamente
esse tipo de divergência que custou reescrever o `info.json` de 275 episódios em
03/09. Então o schema é LIDO do `meta/info.json` do dataset real, e o `--confere`
fecha o ciclo chamando o próprio `features_equal_for_merge`.

── As três lacunas, e o que cada uma vira ───────────────────────────────────

| nossa feature            | origem                    | o que este script faz            |
|--------------------------|---------------------------|----------------------------------|
| `head_camera` 848×480    | `cam_left_high` 640×480   | encaixe (padrão: letterbox)      |
| `head_camera_depth`      | não existe                | **zeros → 10 m uniformes** ⚠     |
| `right_wrist_camera` 224 | `cam_right_wrist` 640×480 | corte central quadrado + resize  |
| dim 14 (yaw do tronco)   | não existe                | 0.0 (tronco fixo na coleta)      |
| pressão 2×33             | não existe                | zeros, como já faz o sim         |

⚠ **A PROFUNDIDADE É FALSA.** O quadro é gravado com zeros, e o decodificador
do LeRobot devolve isso como **10,0 m uniformes** — o `video.depth_max` da
quantização logarítmica (medido, não deduzido: um quadro convertido volta com
`min = max = 10.0`). Não é "faltando", é uma parede plana a 10 metros. Num treino
com `depth_mode: latent` ou `use_depth_3d: true` a política aprende que dado
externo tem geometria impossível. Estes episódios só servem na fase SEM
profundidade (`pi05depth` com `use_depth_3d: false`, FastWAM-D com
`depth_mode: off`). O `meta/info.json` não diz nada disso — por isso está aqui, e
por isso o script repete o aviso no fim de toda conversão.

── Encaixe da câmera da cabeça ──────────────────────────────────────────────
A origem é 4:3 (640×480) e o nosso destino é 16:9 (848×480). Três saídas, e
nenhuma é de graça:

  letterbox (padrão)  geometria preservada, barras cinza nas laterais (104 px
                      de cada lado). A rede vê 75% da largura útil.
  corta               corta o topo/base da origem e amplia até 848×480: mantém
                      a proporção, PERDE campo de visão vertical.
  estica              usa o quadro inteiro, DISTORCE a proporção (um copo fica
                      1,33× mais largo do que é).

Escolhi letterbox como padrão porque distorção de proporção é a única das três
que muda a APARÊNCIA DO OBJETO — e a xícara é justamente o que a política tem
que reconhecer. A barra cinza é um artefato constante, que a rede aprende a
ignorar; um copo achatado é um copo diferente.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

# ── Uma thread por biblioteca ─────────────────────────────────────────────
# Medido em 10/09/2026 com 16 conversores em paralelo: 1,3 MILHÃO de trocas de
# contexto por segundo, 15% de tempo de sistema contra 5% de usuário, CPU total
# em 20% e `iowait` em 0. Ou seja, as máquinas não estavam ocupadas — estavam
# brigando. Cada processo abria ~20 threads (OpenCV no resize, torch, e os
# pools do x264/x265), e 16 processos × 20 threads numa caixa de 128 núcleos
# viram escalonamento puro.
#
# O trabalho aqui é serial por natureza (um quadro de cada vez), então uma
# thread por processo é o certo; o paralelismo vem de rodar vários processos,
# não de várias threads dentro de um.
cv2.setNumThreads(1)

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.feature_utils import features_equal_for_merge  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# Features que o `LeRobotDataset.create` gera sozinho — passá-las de novo é erro.
AUTO = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

# De onde vem cada imagem nossa, no dataset da Unitree.
FONTE_IMAGEM = {
    "observation.images.head_camera": "observation.images.cam_left_high",
    "observation.images.right_wrist_camera": "observation.images.cam_right_wrist",
}


def carrega_schema(referencia: Path) -> tuple[dict, int, str]:
    """Lê o dicionário de features do dataset que vai receber o merge."""
    info = json.loads((referencia / "meta" / "info.json").read_text())
    feats = {k: v for k, v in info["features"].items() if k not in AUTO}
    return feats, int(info["fps"]), info["robot_type"]


def para_uint8_hwc(x) -> np.ndarray:
    """Normaliza o que o LeRobot devolver para uint8 em (H, W, C).

    Dependendo de `return_uint8` e do backend de vídeo, um quadro chega como
    tensor torch CHW float em [0,1], como tensor CHW uint8, ou como array HWC.
    Em vez de assumir um dos três, converte os três.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] < x.shape[-1]:
        x = np.transpose(x, (1, 2, 0))  # CHW → HWC
    if x.dtype != np.uint8:
        # float em [0,1] (o caminho normal) ou já em [0,255]
        escala = 255.0 if float(x.max()) <= 1.0 + 1e-6 else 1.0
        x = np.clip(x * escala, 0, 255).astype(np.uint8)
    return x


def encaixa(img: np.ndarray, alvo_hw: tuple[int, int], modo: str) -> np.ndarray:
    """Leva um quadro de origem para (H, W) do destino. Ver docstring do módulo."""
    ah, aw = alvo_hw
    h, w = img.shape[:2]
    if (h, w) == (ah, aw):
        return img
    if modo == "estica":
        return cv2.resize(img, (aw, ah), interpolation=cv2.INTER_AREA)
    if modo == "corta":
        # amplia até cobrir e corta o excedente (centro)
        esc = max(aw / w, ah / h)
        nw, nh = int(round(w * esc)), int(round(h * esc))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        y0, x0 = (nh - ah) // 2, (nw - aw) // 2
        return r[y0:y0 + ah, x0:x0 + aw]
    # letterbox: cabe inteiro, sobra vira barra cinza neutra
    esc = min(aw / w, ah / h)
    nw, nh = int(round(w * esc)), int(round(h * esc))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    fundo = np.full((ah, aw, img.shape[2]), 114, dtype=np.uint8)
    y0, x0 = (ah - nh) // 2, (aw - nw) // 2
    fundo[y0:y0 + nh, x0:x0 + nw] = r
    return fundo


def quadrado_central(img: np.ndarray, lado: int) -> np.ndarray:
    """Corte quadrado no centro + resize — para a câmera de pulso.

    A nossa de pulso é 224×224 nativa (outra lente); a da Unitree é 640×480. Não
    existe transformação que iguale as duas ópticas. O corte central pelo menos
    preserva a proporção e mantém o objeto no meio do quadro, que é onde ele
    está nas duas.
    """
    h, w = img.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    return cv2.resize(img[y0:y0 + m, x0:x0 + m], (lado, lado), interpolation=cv2.INTER_AREA)


def vinte_e_oito_para_29(v: np.ndarray) -> np.ndarray:
    """[14 braço | 7 mão esq | 7 mão dir] → [14 braço | yaw | 7 esq | 7 dir].

    A ordem das juntas da origem já é a nossa — conferida nome a nome contra o
    `meta/info.json` da Unitree. A única mudança é abrir espaço no índice 14
    para o yaw do tronco, que a coleta deles não tinha (tronco fixo): vai 0.0,
    que é o que o robô de fato fez.
    """
    saida = np.zeros(29, dtype=np.float32)
    saida[:14] = v[:14]
    saida[14] = 0.0
    saida[15:29] = v[14:28]
    return saida



# ==========================================================================
# LEITURA SEQUENCIAL DO VÍDEO
# ==========================================================================
# Por que não usar o `LeRobotDataset.__getitem__`, que seria o caminho óbvio:
# ele decodifica A PARTIR DO INÍCIO DO ARQUIVO a cada quadro pedido, e os
# vídeos da Unitree empacotam ~130 episódios num arquivo de 520 MB. Conferido
# no `meta/episodes`: o episódio 140 do Pouring começa no SEGUNDO 1938 do
# arquivo — pedir um quadro dele custa decodificar meia hora de vídeo.
#
# Medido em 10/09/2026:
#   __getitem__ (episódio 5, começo do arquivo) ....    54 quadros/s
#   __getitem__ (episódios do fundo do arquivo) ..... ~zero, CPU a 100%
#   PyAV sequencial ................................. 1.167 quadros/s
#
# Foi isto que travou a conversão das 10:2x: 32 processos a 100% de CPU sem
# produzir um byte, porque cada um redecodificava o mesmo vídeo sem parar.
#
# Aqui o vídeo é lido UMA VEZ, em ordem. Como os episódios estão dispostos em
# sequência dentro do arquivo, converter uma faixa contígua é uma passada
# linear: o `seek` só acontece ao abrir um arquivo novo, e mesmo assim só
# quando a faixa começa no meio dele.


class LeitorSequencial:
    """Um decodificador aberto por câmera, sempre andando para a frente."""

    def __init__(self, raiz: Path, camera: str):
        self.raiz = raiz
        self.camera = camera
        self.arquivo = None          # (chunk, file_index) aberto agora
        self.container = None
        self.fluxo = None
        self.iterador = None
        self.t_atual = -1.0          # timestamp do último quadro decodificado

    def _abre(self, chunk: int, file_index: int) -> None:
        self.fecha()
        caminho = (self.raiz / "videos" / self.camera /
                   f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4")
        self.container = av.open(str(caminho))
        self.fluxo = self.container.streams.video[0]
        # Uma thread: o paralelismo aqui vem de rodar vários processos, e
        # threads de decodificação dentro de 16 processos viraram 1,3 milhão de
        # trocas de contexto por segundo (ver comentário no topo do arquivo).
        self.fluxo.thread_count = 1
        self.iterador = self.container.decode(video=0)
        self.arquivo = (chunk, file_index)
        self.t_atual = -1.0

    def fecha(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None
            self.iterador = None

    def le(self, chunk: int, file_index: int, t_inicio: float, n: int) -> list[np.ndarray]:
        """Devolve os `n` quadros do episódio que começa em `t_inicio`."""
        if self.arquivo != (chunk, file_index):
            self._abre(chunk, file_index)

        # Salto grande (a faixa deste worker começa no meio do arquivo): vale um
        # `seek`, que vai para o keyframe anterior e de lá decodifica para a
        # frente. Salto pequeno — o caso normal, episódios contíguos — não
        # precisa de nada: o iterador já está no lugar certo.
        if t_inicio - self.t_atual > 5.0:
            self.container.seek(int(t_inicio / self.fluxo.time_base),
                                stream=self.fluxo, any_frame=False, backward=True)
            self.iterador = self.container.decode(video=0)
            self.t_atual = -1.0

        quadros = []
        margem = 1.0 / (2 * 30.0)     # meio quadro a 30 fps
        for frame in self.iterador:
            t = float(frame.pts * self.fluxo.time_base)
            self.t_atual = t
            if t < t_inicio - margem:
                continue              # ainda antes do episódio pedido
            quadros.append(frame.to_ndarray(format="rgb24"))
            if len(quadros) == n:
                break
        if len(quadros) != n:
            raise RuntimeError(
                f"{self.camera}: pedi {n} quadros a partir de {t_inicio:.3f}s em "
                f"chunk-{chunk:03d}/file-{file_index:03d}.mp4 e vieram {len(quadros)}")
        return quadros


def carrega_indice(raiz: Path, cameras: list[str]) -> dict[int, dict]:
    """Mapa episódio → onde ele está, lido do `meta/episodes`.

    Traz, por episódio: a faixa de linhas no parquet de dados
    (`dataset_from_index`/`dataset_to_index`) e, por câmera, em qual arquivo de
    vídeo ele está e em que instante começa.
    """
    arquivos = sorted((raiz / "meta" / "episodes").rglob("*.parquet"))
    if not arquivos:
        raise RuntimeError(f"{raiz} não tem meta/episodes — dataset incompleto?")
    colunas = ["episode_index", "dataset_from_index", "dataset_to_index", "length"]
    for cam in cameras:
        colunas += [f"videos/{cam}/chunk_index", f"videos/{cam}/file_index",
                    f"videos/{cam}/from_timestamp"]
    indice = {}
    for arq in arquivos:
        d = pq.read_table(arq, columns=colunas).to_pydict()
        for i, ep in enumerate(d["episode_index"]):
            indice[ep] = {
                "de": d["dataset_from_index"][i],
                "ate": d["dataset_to_index"][i],
                "n": d["length"][i],
                "video": {cam: (d[f"videos/{cam}/chunk_index"][i],
                                d[f"videos/{cam}/file_index"][i],
                                d[f"videos/{cam}/from_timestamp"][i])
                          for cam in cameras},
            }
    return indice


def carrega_estados(raiz: Path) -> dict[str, np.ndarray]:
    """Lê `observation.state` e `action` de todos os parquets, em ordem.

    ⚠ NÃO use `to_pylist()` aqui. A primeira versão disto fazia
    `np.asarray(tabela[c].to_pylist())`, e para o ToastedBread isso converte
    352 mil linhas de 28 floats em listas Python — cerca de 10 milhões de
    objetos. Medido em 10/09/2026: 23 dos 24 workers ficaram com ZERO episódios
    prontos porque estavam todos presos nesta função, e a vazão agregada
    despencou para 18 quadros/s enquanto um worker sozinho fazia 15,9.
    O `flatten().to_numpy()` faz a mesma coisa como uma cópia de memória.
    """
    import pyarrow as pa
    arquivos = sorted((raiz / "data").rglob("*.parquet"))
    partes = [pq.read_table(a, columns=["observation.state", "action"]) for a in arquivos]
    tabela = partes[0] if len(partes) == 1 else pa.concat_tables(partes)

    saida = {}
    for c in tabela.column_names:
        coluna = tabela[c].combine_chunks()
        plano = coluna.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        saida[c] = plano.reshape(len(coluna), -1)
    return saida


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origem", required=True,
                   help="repo_id do HF (ex.: unitreerobotics/G1_Dex3_Pouring_Dataset) "
                        "ou caminho local de um dataset já baixado")
    p.add_argument("--destino", required=True, help="diretório do dataset convertido")
    p.add_argument("--repo-id", default=None, help="repo_id do destino (padrão: nome do diretório)")
    p.add_argument("--referencia", default="meu_dataset/white_cup_on_dripper_2026-08-11",
                   help="dataset de onde o schema é copiado — o que vai receber o merge")
    p.add_argument("--episodios", type=int, default=None, help="converter só os N primeiros")
    # Recorte por faixa — é o que permite paralelizar. Medido em 10/09/2026: a
    # conversão anda a ~2,3 episódios-quadro/s por processo, e o gargalo NÃO é a
    # leitura (54 quadros/s pelo `__getitem__`, 1.167 pelo PyAV cru) e sim a
    # ESCRITA da profundidade falsa: TIFF de 814 kB por quadro mais encode x265
    # lossless de um mapa que é só zeros. Como o custo é por processo e a athena
    # tem 128 núcleos, dividir por faixas é o conserto barato.
    p.add_argument("--inicio", type=int, default=0, help="primeiro episódio da faixa (inclusivo)")
    p.add_argument("--fim", type=int, default=None, help="último episódio da faixa (exclusivo)")
    p.add_argument("--tarefa", default=None,
                   help="sobrescreve a string de tarefa (padrão: a do dataset de origem)")
    p.add_argument("--encaixe", choices=["letterbox", "corta", "estica"], default="letterbox")
    p.add_argument("--profundidade-lossless", action="store_true",
                   help="mantém o x265 lossless na profundidade (padrão: rápido — ver comentário)")
    p.add_argument("--escritores", type=int, default=4,
                   help="processos do AsyncImageWriter (0 = síncrono, o padrão do LeRobot)")
    p.add_argument("--escritores-threads", type=int, default=4,
                   help="threads por processo do AsyncImageWriter")
    p.add_argument("--confere", action="store_true",
                   help="ao fim, roda features_equal_for_merge contra a referência")
    args = p.parse_args()

    referencia = Path(args.referencia)
    if not (referencia / "meta" / "info.json").exists():
        print(f"❌ referência sem meta/info.json: {referencia}")
        return 1
    feats, fps, robot_type = carrega_schema(referencia)

    # SÓ OS METADADOS da origem. O `LeRobotDataset` não é mais usado para LER —
    # a leitura agora é sequencial, com PyAV (ver `LeitorSequencial`); ele fica
    # só para ESCREVER o destino.
    origem_local = Path(args.origem)
    raiz_src = origem_local if origem_local.exists() else None
    repo_src = origem_local.name if raiz_src else args.origem
    print(f"⏳ lendo metadados de {args.origem} ...")
    meta = LeRobotDatasetMetadata(repo_src, root=raiz_src)
    if meta.fps != fps:
        print(f"❌ fps não bate: origem {meta.fps}, referência {fps}. "
              "Reamostrar está fora do escopo deste script.")
        return 1

    faltando = [k for k in FONTE_IMAGEM.values() if k not in meta.features]
    if faltando:
        print(f"❌ a origem não tem {faltando}.\n"
              "   Os sete `G1_Dex3_Pick*` (Apple, Bottle, Charger, Gum, Snack, Tissue, Doll)\n"
              "   só têm as DUAS câmeras de cabeça — foram gravados noutra campanha\n"
              "   (`robot_type: Unitree_G1_Dex3`). Preencher a `right_wrist_camera` com uma\n"
              "   constante seria o mesmo veneno da profundidade, então o conversor recusa.\n"
              "   Para usá-los, rode a fase de co-treino sem a câmera de pulso.")
        return 1

    tarefa = args.tarefa or meta.tasks.index[0]
    ini = max(0, args.inicio)
    fim = meta.total_episodes if args.fim is None else min(args.fim, meta.total_episodes)
    if args.episodios is not None:
        fim = min(fim, ini + args.episodios)
    n_eps = fim - ini
    if n_eps <= 0:
        print(f"❌ faixa vazia: --inicio {ini} --fim {fim}")
        return 1
    destino = Path(args.destino)
    repo_id = args.repo_id or f"local/{destino.name}"
    print(f"✅ origem: {meta.total_episodes} episódios, {meta.total_frames} quadros\n"
          f"   tarefa: \"{tarefa}\"\n"
          f"   convertendo {n_eps} episódios [{ini}..{fim}) → {destino}\n")

    dst = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=feats, root=destino,
        robot_type=robot_type, use_videos=True,
        # ── Escrita das imagens em paralelo ───────────────────────────────
        # MEDIDO em 10/09/2026, por quadro:
        #     profundidade (TIFF de zeros) ....  0,7 ms
        #     RGB da cabeça (PNG 848×480) ..... 66,1 ms   ← 99% do custo
        #
        # O `add_frame` comprime o PNG de forma SÍNCRONA, então um worker fica
        # preso a ~15 quadros/s por causa disso — e 66 ms × 828 mil quadros são
        # 15 horas de compressão. Não era a profundidade, como eu supus por
        # três tentativas: o mapa de zeros é praticamente de graça.
        #
        # O `AsyncImageWriter` do LeRobot resolve e vinha DESLIGADO (o padrão de
        # `image_writer_processes`/`threads` é 0): com processos, a compressão
        # sai do laço e passa a acontecer em paralelo com a leitura do vídeo.
        image_writer_processes=args.escritores,
        image_writer_threads=args.escritores_threads,
        # Mesmo motivo do `gerar_dataset_mujoco.py`: o default é av1 e o
        # `features_equal_for_merge` compara `video.codec`.
        rgb_encoder=RGBEncoderConfig(vcodec="h264", extra_options={"threads": "2"}),
        # ── Profundidade: encoder RÁPIDO por padrão ────────────────────────
        # A referência usa `x265-params: lossless=1`, que é o certo para
        # profundidade REAL. Aqui o mapa é constante (zeros), e comprimir um
        # quadro constante sem perdas dá exatamente o mesmo resultado que
        # comprimi-lo com perdas — só que o lossless custa caro: medido em
        # 10/09/2026, a conversão inteira anda a 2,3 quadros/s enquanto a
        # LEITURA sozinha faz 54, e a diferença é quase toda encode de
        # profundidade.
        #
        # E isto NÃO quebra o merge: `features_equal_for_merge` ignora
        # `preset`, `crf`, `g`, `extra_options`, `fast_decode` e
        # `video_backend` (`lerobot/configs/video.py:51`). O que ele compara —
        # `vcodec` (hevc) e `pix_fmt` (gray12le) — continua igual.
        #
        # `--profundidade-lossless` volta ao comportamento da referência, para
        # quando a origem tiver profundidade de verdade.
        depth_encoder=(DepthEncoderConfig() if args.profundidade_lossless
                       else DepthEncoderConfig(preset="ultrafast",
                                               extra_options={"x265-params": "pools=2:frame-threads=1"})),
    )

    alvo = {k: (feats[k]["shape"][0], feats[k]["shape"][1]) for k in FONTE_IMAGEM}
    prof_hw = feats["observation.images.head_camera_depth"]["shape"][:2]
    prof_zero = np.zeros((*prof_hw, 1), dtype=np.uint16)
    zeros33 = np.zeros(33, dtype=np.float32)
    lado_pulso = feats["observation.images.right_wrist_camera"]["shape"][0]

    # ── Leitura: uma passada linear por vídeo ────────────────────────────────
    if raiz_src is None:
        print("❌ a leitura sequencial precisa do dataset em disco. "
              "Baixe primeiro e passe o caminho em --origem.")
        return 1

    cam_cabeca = FONTE_IMAGEM["observation.images.head_camera"].split(".")[-1]
    cam_pulso = FONTE_IMAGEM["observation.images.right_wrist_camera"].split(".")[-1]
    chaves_cam = [f"observation.images.{cam_cabeca}", f"observation.images.{cam_pulso}"]

    print("⏳ índice de episódios e vetores de estado ...", flush=True)
    indice = carrega_indice(raiz_src, chaves_cam)
    colunas = carrega_estados(raiz_src)
    estados_src, acoes_src = colunas["observation.state"], colunas["action"]

    leitores = {c: LeitorSequencial(raiz_src, c) for c in chaves_cam}
    total = 0
    try:
        for k, ep in enumerate(range(ini, fim), start=1):
            info_ep = indice[ep]
            n = info_ep["n"]
            de, ate = info_ep["de"], info_ep["ate"]

            quadros = {}
            for chave in chaves_cam:
                chunk, arq, t0 = info_ep["video"][chave]
                quadros[chave] = leitores[chave].le(chunk, arq, t0, n)

            for i in range(n):
                estado = vinte_e_oito_para_29(estados_src[de + i])
                acao = vinte_e_oito_para_29(acoes_src[de + i])
                cabeca = encaixa(quadros[chaves_cam[0]][i],
                                 alvo["observation.images.head_camera"], args.encaixe)
                pulso = quadrado_central(quadros[chaves_cam[1]][i], lado_pulso)
                dst.add_frame({
                    "action": acao,
                    "observation.state": estado,
                    "observation.images.head_camera": cabeca,
                    "observation.images.head_camera_depth": prof_zero,
                    "observation.images.right_wrist_camera": pulso,
                    "observation.left_hand_pressure": zeros33,
                    "observation.right_hand_pressure": zeros33,
                    "task": tarefa,
                })
            if ate - de != n:
                raise RuntimeError(f"episódio {ep}: length={n} mas as linhas do parquet "
                                   f"vão de {de} a {ate} ({ate - de})")
            dst.save_episode()
            total += n
            print(f"   ↳ episódio {ep} ({k}/{n_eps}) — {n} quadros ({total} no total)", flush=True)
    finally:
        for leitor in leitores.values():
            leitor.fecha()

    # Sem isto o último buffer de metadados não vai para o disco e o dataset não
    # abre depois — é o mesmo `finalize()` do `convert_dataset.py:179`.
    dst.finalize()
    print(f"\n✅ {n_eps} episódios, {total} quadros em {destino}")

    if args.confere:
        ref_feats, _, _ = carrega_schema(referencia)
        novo = {k: v for k, v in json.loads(
            (destino / "meta" / "info.json").read_text())["features"].items() if k not in AUTO}
        ok = features_equal_for_merge(ref_feats, novo)
        print(f"{'✅' if ok else '❌'} features_equal_for_merge contra {referencia.name}: {ok}")
        if not ok:
            for k in sorted(set(ref_feats) | set(novo)):
                if ref_feats.get(k) != novo.get(k):
                    print(f"   ≠ {k}\n     ref:  {ref_feats.get(k)}\n     novo: {novo.get(k)}")
            return 1

    print("\n⚠  A PROFUNDIDADE DESTE DATASET É FALSA: grava zeros, e o decodificador\n"
          "   devolve 10,0 m uniformes (o `video.depth_max`). Só use em treino com\n"
          "   profundidade DESLIGADA (`use_depth_3d: false` / `depth_mode: off`).\n"
          "   As pressões também são zeros — `use_pressure: false`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
