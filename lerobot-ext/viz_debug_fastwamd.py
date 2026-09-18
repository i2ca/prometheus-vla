#!/usr/bin/env python
"""Painel de depuração do FastWAM-D — uma janela, quatro quadrantes.

    ┌──────────────────────┬──────────────────────┐
    │ 1. atenção do DiT    │ 2. profundidade      │
    │    sobre o mosaico   │    (crua e a do      │
    │    das câmeras       │     modelo)          │
    ├──────────────────────┼──────────────────────┤
    │ 3. nuvem de pontos   │ 4. temperatura dos   │
    │    topo + lateral    │    motores do G1     │
    └──────────────────────┴──────────────────────┘

Uma janela só, e não quatro, porque o loop de controle roda a 30 Hz: cada
`cv2.imshow` extra é tempo roubado do ciclo, e quatro janelas soltas viram
quatro `waitKey`. Aqui é um `imshow` por ciclo, com os quadrantes desenhados
num canvas único.

Os quadrantes 1 e 2 vêm do SERVIDOR (só o modelo sabe onde olhou e o que
recebeu como profundidade depois da normalização); os quadrantes 3 e 4 são
locais, calculados do que o robô e a câmera entregam aqui.

Dois transportes para o MESMO desenho:

  `PainelDebug`  janela OpenCV local — exige um build do cv2 com GUI e um X.
  `PainelWeb`    o mesmo canvas servido por HTTP como MJPEG, para máquina sem
                 tela (SSH, PC de rack, ou o OpenCV headless do lerobot). Abra
                 `http://<ip>:8088/` no navegador — inclusive o do celular,
                 que é o jeito prático de acompanhar o robô REAL: de pé ao lado
                 dele, com a mão no botão de emergência.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Faixa métrica do painel de profundidade e da nuvem, em metros. É a mesma do
# `depth_min`/`depth_max` do treino: ver a cena com uma escala e o modelo com
# outra é o jeito mais fácil de tirar conclusão errada de um debug.
PROF_MIN_M = 0.05
PROF_MAX_M = 5.0

# Intrínsecos NOMINAIS da RealSense a 848x480 (mesmos do `configuration_act.py`).
# Os reais ainda não foram medidos no robô — `Scripts_Prometheus_int/
# print_camera_intrinsics.py` imprime os de verdade. Com estes, a nuvem tem a
# forma certa mas a escala absoluta pode andar alguns por cento.
INTRINSECOS_PADRAO = {"fx": 617.0, "fy": 617.0, "cx": 424.0, "cy": 240.0}

_VERDE = (90, 200, 90)
_AMARELO = (60, 200, 230)
_VERMELHO = (60, 60, 235)
_CINZA = (150, 150, 150)
_FUNDO = (24, 24, 24)


def _texto(canvas, txt, x, y, escala=0.45, cor=_CINZA, grosso=1):
    """Escreve na tela. Os rótulos deste módulo são ASCII de propósito.

    As fontes Hershey do OpenCV não têm acento nem travessão: qualquer caractere
    fora do ASCII vira `?` na tela. Por isso "Atencao", "-" no lugar de "–" e
    "+/-" no lugar de "±" — feio no código, legível no painel.
    """
    cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grosso, cv2.LINE_AA)


def _encaixa(imagem: np.ndarray, w: int, h: int) -> np.ndarray:
    """Redimensiona preservando a proporção, com barras escuras no que sobrar.

    O mosaico das câmeras é 2:1 e o quadrante é 4:3; esticar deformaria a cena e,
    junto com ela, o mapa de atenção desenhado por cima — que passaria a apontar
    para um lugar que não corresponde ao pixel real.
    """
    alt, larg = imagem.shape[:2]
    escala = min(w / larg, h / alt)
    nova = cv2.resize(imagem, (max(1, int(larg * escala)), max(1, int(alt * escala))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    y0 = (h - nova.shape[0]) // 2
    x0 = (w - nova.shape[1]) // 2
    canvas[y0:y0 + nova.shape[0], x0:x0 + nova.shape[1]] = nova
    return canvas


def _quadro_vazio(w: int, h: int, aviso: str) -> np.ndarray:
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    _texto(canvas, aviso, 14, h // 2, 0.5, (110, 110, 110))
    return canvas


# ══════════════════════════════════════════════════════════════════════════
# Quadrantes
# ══════════════════════════════════════════════════════════════════════════
def desenha_atencao(rgb_mosaico: np.ndarray | None, mapa: np.ndarray | None,
                    w: int, h: int, relativo: bool = False, base_n: int = 0,
                    nome: str = "DiT") -> np.ndarray:
    """Mosaico das câmeras com o mapa de atenção por cima.

    O mapa vem na grade de tokens do DiT (dezenas de células, não pixels), então
    ele é ampliado com interpolação linear — a mancha suave é honesta quanto à
    resolução real: cada célula é um token, e o token é um pedaço da imagem.
    """
    if rgb_mosaico is None:
        return _quadro_vazio(w, h, "1. Atencao - sem imagem")

    bgr = cv2.cvtColor(rgb_mosaico, cv2.COLOR_RGB2BGR) if rgb_mosaico.ndim == 3 else rgb_mosaico
    base = _encaixa(bgr, w, h)

    if mapa is None:
        _texto(base, "1. Atencao - aguardando o servidor", 10, 20, 0.42)
        return base

    # O calor passa pelo MESMO encaixe da imagem: redimensionar os dois com
    # geometrias diferentes desalinharia o mapa em relação à cena.
    calor_rgb = cv2.applyColorMap(
        (np.clip(cv2.resize(mapa.astype(np.float32), (rgb_mosaico.shape[1], rgb_mosaico.shape[0]),
                            interpolation=cv2.INTER_LINEAR), 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    colorido = _encaixa(calor_rgb, w, h)
    calor = cv2.cvtColor(colorido, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # 0,55 de imagem e 0,45 de calor: o suficiente para achar a região quente
    # sem perder de vista o objeto que está embaixo dela.
    mistura = cv2.addWeighted(base, 0.55, colorido, 0.45, 0)

    titulo = (f"1. Atencao do {nome} - desvio da linha de base" if relativo
              else f"1. Atencao do {nome} (cru - contem o sumidouro)")
    _texto(mistura, titulo, 10, 20, 0.45, (240, 240, 240), 1)
    if relativo and base_n < 10:
        # Com poucas amostras a base ainda é o próprio quadro; dizer isso evita
        # que alguém leia um mapa quase vazio como "o modelo não olha para nada".
        _texto(mistura, f"base aquecendo ({base_n} amostras)", 10, 36, 0.38, (150, 150, 150))
    # O pico é localizado no mapa original e trazido para a tela pelo encaixe,
    # em vez de procurado no colormap — no JET, vermelho e azul escuro têm
    # luminância parecida, e um argmax sobre o cinza cairia no lugar errado.
    #
    # E é procurado no MIOLO, ignorando o anel de tokens da borda. O attention
    # sink vive ali: a primeira coluna de cada linha da grade recebe peso alto em
    # TODO quadro, e a subtração da linha de base tira a maior parte, mas não
    # tudo — sobra o suficiente para o argmax cair na borda esquerda ou no canto
    # inferior mesmo quando o calor de verdade está na xícara. O marcador então
    # dizia "ele está olhando para o nada" com a mancha certa desenhada ao lado,
    # que é pior do que não ter marcador.
    miolo = mapa[1:-1, 1:-1] if min(mapa.shape) > 2 else mapa
    ph, pw = np.unravel_index(int(np.argmax(miolo)), miolo.shape)
    if miolo is not mapa:
        ph, pw = ph + 1, pw + 1
    escala = min(w / rgb_mosaico.shape[1], h / rgb_mosaico.shape[0])
    lado_w = int(rgb_mosaico.shape[1] * escala)
    lado_h = int(rgb_mosaico.shape[0] * escala)
    pico = (
        int((h - lado_h) // 2 + (ph + 0.5) / mapa.shape[0] * lado_h),
        int((w - lado_w) // 2 + (pw + 0.5) / mapa.shape[1] * lado_w),
    )
    cv2.circle(mistura, (int(pico[1]), int(pico[0])), 9, (255, 255, 255), 2)
    _texto(mistura, "pico (borda ignorada)", int(pico[1]) + 12, int(pico[0]) - 8, 0.4,
           (255, 255, 255))
    return mistura


def desenha_profundidade(depth_mm: np.ndarray | None, depth_modelo: np.ndarray | None,
                         w: int, h: int) -> np.ndarray:
    """Duas faixas: a profundidade crua da câmera e a que o modelo recebeu.

    Ver as duas lado a lado é o ponto do painel. A de cima é o que a RealSense
    mandou, em milímetros. A de baixo é depois do log, do recorte de faixa e do
    mosaico — se a de cima tem geometria e a de baixo está preta, a
    profundidade não está chegando ao modelo, e nenhuma métrica de treino
    contaria isso.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    meia = h // 2

    if depth_mm is not None:
        metros = np.squeeze(depth_mm).astype(np.float32) / 1000.0
        valido = (metros > PROF_MIN_M) & (metros < PROF_MAX_M)
        norm = np.zeros_like(metros)
        norm[valido] = (metros[valido] - PROF_MIN_M) / (PROF_MAX_M - PROF_MIN_M)
        colorido = cv2.applyColorMap((np.clip(1.0 - norm, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        colorido[~valido] = (40, 40, 40)   # sem medida: cinza, não "perto"
        canvas[:meia] = _encaixa(colorido, w, meia)
        medianos = metros[valido]
        faixa = (f"{medianos.min():.2f} a {medianos.max():.2f} m"
                 if medianos.size else "sem medida valida")
        _texto(canvas, f"2. Depth da camera  ({faixa})", 10, 20, 0.45, (240, 240, 240))
    else:
        _texto(canvas, "2. Depth da camera - sem dado", 10, 20, 0.45)

    if depth_modelo is not None:
        mapa = np.clip(np.squeeze(depth_modelo).astype(np.float32), 0.0, 1.0)
        colorido = cv2.applyColorMap((np.clip(1.0 - mapa, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        colorido[mapa <= 0.0] = (40, 40, 40)
        canvas[meia:] = _encaixa(colorido, w, h - meia)
        _texto(canvas, "   o que o modelo recebeu (mosaico normalizado)",
               10, meia + 20, 0.42, (240, 240, 240))
    else:
        _texto(canvas, "   modelo: aguardando servidor", 10, meia + 20, 0.42)

    cv2.line(canvas, (0, meia), (w, meia), (70, 70, 70), 1)
    return canvas


def desenha_nuvem(depth_mm: np.ndarray | None, intrinsecos: dict,
                  w: int, h: int, max_pontos: int = 6000) -> np.ndarray:
    """Nuvem de pontos em duas projeções: topo (XZ) e lateral (YZ).

    É a mesma projeção que alimenta o encoder de profundidade das outras
    políticas (`depth_to_pointcloud`), refeita aqui em numpy para não arrastar
    torch para o cliente — o PC que controla o robô não precisa de modelo
    nenhum carregado.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    meia = w // 2
    cv2.line(canvas, (meia, 0), (meia, h), (70, 70, 70), 1)
    _texto(canvas, "3. Nuvem - topo (XZ)", 10, 18, 0.42, (240, 240, 240))
    _texto(canvas, "lateral (YZ)", meia + 10, 18, 0.42, (240, 240, 240))

    if depth_mm is None:
        _texto(canvas, "sem profundidade", 10, h // 2, 0.5, (110, 110, 110))
        return canvas

    mapa = np.squeeze(depth_mm).astype(np.float32) / 1000.0
    alt, larg = mapa.shape
    fx, fy = float(intrinsecos["fx"]), float(intrinsecos["fy"])
    cx, cy = float(intrinsecos["cx"]), float(intrinsecos["cy"])
    # Os intrínsecos são da resolução nativa; se o quadro vier redimensionado,
    # eles têm que acompanhar, senão a nuvem sai esticada.
    escala_x = larg / (2.0 * cx)
    escala_y = alt / (2.0 * cy)
    fx, cx = fx * escala_x, cx * escala_x
    fy, cy = fy * escala_y, cy * escala_y

    ys, xs = np.nonzero((mapa > PROF_MIN_M) & (mapa < PROF_MAX_M))
    if xs.size == 0:
        _texto(canvas, "nenhum ponto na faixa", 10, h // 2, 0.5, (110, 110, 110))
        return canvas
    if xs.size > max_pontos:
        escolha = np.random.choice(xs.size, max_pontos, replace=False)
        xs, ys = xs[escolha], ys[escolha]

    z = mapa[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    def _plota(px, pz, x0, faixa_h, faixa_v, rotulo_h, rotulo_v):
        for a, b, cor_z in zip(px, pz, z):
            u = int((a - faixa_h[0]) / (faixa_h[1] - faixa_h[0]) * (meia - 20)) + x0 + 10
            v = int(h - 24 - (b - faixa_v[0]) / (faixa_v[1] - faixa_v[0]) * (h - 50))
            if x0 <= u < x0 + meia and 24 <= v < h:
                t = np.clip((cor_z - PROF_MIN_M) / (PROF_MAX_M - PROF_MIN_M), 0, 1)
                canvas[v, u] = (int(60 + 180 * t), int(220 - 120 * t), int(240 - 200 * t))
        _texto(canvas, rotulo_h, x0 + meia - 60, h - 8, 0.35, (110, 110, 110))
        _texto(canvas, rotulo_v, x0 + 6, 34, 0.35, (110, 110, 110))

    _plota(x, z, 0, (-1.5, 1.5), (PROF_MIN_M, PROF_MAX_M), "X +/-1.5 m", "Z 0-5 m")
    _plota(y, z, meia, (-1.0, 1.0), (PROF_MIN_M, PROF_MAX_M), "Y +/-1.0 m", "Z 0-5 m")
    _texto(canvas, f"{xs.size} pts", 10, h - 8, 0.35, (110, 110, 110))
    return canvas


def desenha_temperatura(temperaturas: dict[str, float] | None, w: int, h: int,
                        alerta: float = 60.0, atencao: float = 45.0) -> np.ndarray:
    """Barras de temperatura por junta, com os dois limiares marcados.

    Os limiares são conservadores de propósito: os motores do G1 aguentam mais,
    mas numa sessão longa de teleoperação/inferência o que interessa é ver a
    subida ANTES do desarme, não descobrir depois.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    _texto(canvas, "4. Temperatura dos motores", 10, 20, 0.45, (240, 240, 240))

    if not temperaturas:
        _texto(canvas, "sem telemetria (robo em --sim ou sem lowstate)",
               10, h // 2, 0.45, (110, 110, 110))
        return canvas

    itens = list(temperaturas.items())
    topo, base = 34, h - 16
    passo = max(9, (base - topo) // max(1, len(itens)))
    largura_barra = w - 150

    for i, (nome, valor) in enumerate(itens):
        y = topo + i * passo
        if y + 6 > base:
            _texto(canvas, f"... +{len(itens) - i} juntas", 10, base, 0.35, (110, 110, 110))
            break
        curto = nome.replace(".q", "").replace("_joint", "").replace("k", "", 1)[:18]
        _texto(canvas, curto, 8, y + 6, 0.32, (170, 170, 170))
        largura = int(np.clip(valor / 90.0, 0, 1) * largura_barra)
        cor = _VERMELHO if valor >= alerta else (_AMARELO if valor >= atencao else _VERDE)
        cv2.rectangle(canvas, (120, y), (120 + largura, y + max(4, passo - 3)), cor, -1)
        _texto(canvas, f"{valor:.0f}C", w - 28, y + 6, 0.32, cor)

    for limiar, cor in ((atencao, _AMARELO), (alerta, _VERMELHO)):
        x = 120 + int(limiar / 90.0 * largura_barra)
        cv2.line(canvas, (x, topo), (x, base), cor, 1)
    return canvas


# ══════════════════════════════════════════════════════════════════════════
# Painéis
# ══════════════════════════════════════════════════════════════════════════
def desenha_chunks(chunks: list | None, passo: int, lead: int, w: int, h: int,
                   juntas: slice = slice(7, 14)) -> np.ndarray:
    """As ações que o modelo JÁ decidiu e ainda não foram executadas.

    Cada chunk é uma previsão de `action_horizon` passos feita a partir de UMA
    observação, e o loop consome uma ação por ciclo. Este quadrante mostra o que
    está no buffer: onde o passo atual está dentro de cada chunk, quanto ainda
    resta, e em que ponto o `lead` dispara o próximo pedido.

    Por que ver isso importa: quando dois chunks se sobrepõem, o ensembling
    temporal está MEDIANDO duas previsões feitas com ~1 s de diferença. Se as
    duas linhas de uma mesma junta estiverem longe uma da outra na região
    sobreposta, a ação executada não é nenhuma das duas — é a média, e o robô
    faz um movimento que o modelo nunca previu. Com as linhas coladas, o
    ensembling está só suavizando, que é o que se quer dele.

    O recorte padrão é o braço DIREITO (índices 7..13 de `JUNTAS_G1`), que é o
    que executa a tarefa; o esquerdo não se move nas demonstrações.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    _texto(canvas, "4. Chunks no buffer - o que ja foi decidido", 10, 20, 0.45, (240, 240, 240))

    vivos = [c for c in (chunks or []) if c.get("chunk") is not None and len(c["chunk"])]
    if not vivos:
        _texto(canvas, "buffer vazio - esperando o servidor", 10, h // 2, 0.45, (110, 110, 110))
        return canvas

    # Janela horizontal: do passo atual até o fim do chunk mais longo. Fixar o
    # zero no passo atual (e não no início do chunk mais antigo) mantém a linha
    # do "agora" parada na esquerda — o olho compara as previsões, não persegue
    # um eixo que anda.
    fim = max(c["inicio"] + len(c["chunk"]) for c in vivos)
    horizonte = max(4, fim - passo)

    esq, dir_, topo, base = 34, w - 10, 34, h - 26
    largura, altura = dir_ - esq, base - topo
    x_de = lambda p: esq + int((p - passo) / horizonte * largura)  # noqa: E731

    # Faixa vertical: as ações destes chunks, com folga. Escala compartilhada
    # entre as juntas de propósito — o que interessa é comparar as previsões
    # entre si, e uma escala por junta esconderia justamente a divergência.
    amostras = np.concatenate([np.asarray(c["chunk"])[:, juntas].ravel() for c in vivos])
    lo, hi = float(amostras.min()), float(amostras.max())
    if hi - lo < 1e-3:
        lo, hi = lo - 0.05, hi + 0.05
    folga = (hi - lo) * 0.08
    lo, hi = lo - folga, hi + folga
    y_de = lambda v: base - int((v - lo) / (hi - lo) * altura)  # noqa: E731

    cv2.rectangle(canvas, (esq, topo), (dir_, base), (52, 52, 52), 1)
    for v in (lo, (lo + hi) / 2, hi):
        _texto(canvas, f"{v:+.2f}", 2, y_de(v) + 4, 0.32, (110, 110, 110))

    # Uma cor por chunk, do mais antigo (cinza) para o mais novo (verde): o mais
    # novo é o que vai sobreviver, o antigo é o que está saindo.
    cores = [(120, 120, 120), _AMARELO, _VERDE]
    for k, c in enumerate(vivos):
        cor = cores[min(k, len(cores) - 1)] if len(vivos) > 1 else _VERDE
        bloco = np.asarray(c["chunk"])[:, juntas]
        for j in range(bloco.shape[1]):
            pontos = [
                (x_de(c["inicio"] + i), y_de(float(bloco[i, j])))
                for i in range(bloco.shape[0])
                if c["inicio"] + i >= passo
            ]
            if len(pontos) > 1:
                cv2.polylines(canvas, [np.array(pontos, dtype=np.int32)], False, cor, 1,
                              cv2.LINE_AA)

    # "Agora" e o gatilho do lead. O gatilho é medido no chunk MAIS NOVO, que é
    # o que o loop olha para decidir se pede outra inferência (`restantes`).
    cv2.line(canvas, (esq, topo), (esq, base), (230, 230, 230), 1)
    _texto(canvas, "agora", esq + 4, base + 16, 0.34, (230, 230, 230))

    mais_novo = vivos[-1]
    p_gatilho = mais_novo["inicio"] + len(mais_novo["chunk"]) - lead
    if passo <= p_gatilho <= fim:
        x = x_de(p_gatilho)
        for y in range(topo, base, 6):                       # tracejado
            cv2.line(canvas, (x, y), (x, min(y + 3, base)), (200, 160, 60), 1)
        _texto(canvas, f"pede nova (lead {lead})", min(x + 5, w - 150), topo + 14, 0.34,
               (200, 160, 60))

    # O rodapé começa depois do rótulo "agora", que fica colado no eixo: os dois
    # em x=10 se sobrepõem e viram um borrão ilegível.
    restam = fim - passo
    legenda = "cinza=antigo verde=novo | " if len(vivos) > 1 else ""
    _texto(canvas, f"{legenda}{len(vivos)} chunk(s) | restam {restam} acoes",
           esq + 62, h - 8, 0.36, (150, 150, 150))
    return canvas


def _exige_opencv_com_gui() -> None:
    """Falha com instrução em vez de deixar a janela simplesmente não aparecer.

    O core do lerobot depende de `opencv-python-headless`, compilado com
    `GUI: NONE`. Com esse build, `cv2.namedWindow` levanta um erro genérico
    ("The function is not implemented") ou, dependendo da versão, não abre nada
    e o programa segue como se estivesse tudo bem — que é o pior desfecho:
    ninguém suspeita do OpenCV, todo mundo suspeita da câmera.

    Só o `PainelDebug` (janela local) precisa disto. O `PainelWeb` desenha no
    mesmo canvas e codifica com `cv2.imencode`, que existe no build headless —
    por isso ele funciona por SSH, sem X, sem DISPLAY.
    """
    if "GUI:                           NONE" not in cv2.getBuildInformation():
        return
    raise RuntimeError(
        "Este OpenCV é o build 'headless' (GUI: NONE) — ele não consegue abrir "
        "janela nenhuma.\n"
        "   Conserto (os dois pacotes instalam o mesmo módulo cv2; vence o último):\n"
        "       pip install --no-deps 'opencv-python>=4.9.0,<4.14.0'\n"
        "   Confira com:\n"
        "       python -c \"import cv2; print(cv2.getBuildInformation())\" | grep GUI\n"
        "   Ou, sem instalar nada: troque --v-debug por --v-web e abra no navegador."
    )


class PainelBase:
    """O estado do painel e o desenho dos quatro quadrantes, sem transporte.

    Existe para que a janela local e o painel web mostrem EXATAMENTE a mesma
    imagem: um único `compoe()`, duas maneiras de entregá-lo. Se o desenho
    morasse em cada um, "o que eu vi no navegador" e "o que eu vi na tela do
    laboratório" seriam duas coisas diferentes na hora de comparar uma corrida
    com outra.

    `atualiza`/`define_*` são baratos e podem ser chamados todo ciclo; os
    quadrantes que não receberam dado novo simplesmente repetem o último
    desenho.
    """

    def __init__(self, largura_quadrante: int = 480, altura_quadrante: int = 360,
                 intrinsecos: dict | None = None):
        if cv2 is None:
            raise RuntimeError("OpenCV não está instalado: pip install opencv-python")
        self.qw = largura_quadrante
        self.qh = altura_quadrante
        self.intrinsecos = intrinsecos or dict(INTRINSECOS_PADRAO)

        self._rgb_mosaico: np.ndarray | None = None
        self._attn: np.ndarray | None = None
        self._attn_cru: np.ndarray | None = None
        self._attn_e_relativo = False
        self._attn_base_n = 0
        self._depth_mm: np.ndarray | None = None
        self._depth_modelo: np.ndarray | None = None
        self._temperaturas: dict[str, float] | None = None
        self._chunks: list | None = None
        self._passo = 0
        self._lead = 0
        self._cabecalho = ""
        # Quem é dono do mapa do quadrante 1. O painel nasceu para o FastWAM-D e
        # também serve o π0.5 (`init_lerobot_inference_v3.py --v-web`). ASCII:
        # o `cv2.putText` não desenha acento nem letra grega.
        self.nome_atencao = "DiT"

    # ── entradas ────────────────────────────────────────────────────────────
    def define_imagens(self, rgb_mosaico=None, depth_mm=None) -> None:
        if rgb_mosaico is not None:
            self._rgb_mosaico = rgb_mosaico
        if depth_mm is not None:
            self._depth_mm = depth_mm

    def define_debug_servidor(self, payload: dict | None) -> None:
        if not payload:
            return
        # O relativo é o que responde "para onde ele olhou NESTE quadro": o mapa
        # cru é dominado pelo attention sink das primeiras colunas, que acende
        # igual em qualquer cena. Ver a nota em `LinhaDeBaseDaAtencao`, no
        # servidor. O cru fica guardado para quem quiser conferir.
        if payload.get("attn_rel") is not None:
            self._attn = np.asarray(payload["attn_rel"], dtype=np.float32)
            self._attn_e_relativo = True
            self._attn_base_n = int(payload.get("attn_base_n", 0))
        elif payload.get("attn") is not None:
            self._attn = np.asarray(payload["attn"], dtype=np.float32)
            self._attn_e_relativo = False
        if payload.get("attn") is not None:
            self._attn_cru = np.asarray(payload["attn"], dtype=np.float32)
        if payload.get("depth") is not None:
            self._depth_modelo = np.asarray(payload["depth"], dtype=np.float32)

    def define_temperaturas(self, temperaturas: dict[str, float] | None) -> None:
        if temperaturas:
            self._temperaturas = temperaturas

    def define_chunks(self, chunks: list | None, passo: int, lead: int) -> None:
        """Cópia RASA de propósito: a lista do loop é reconstruída a cada ciclo
        (o filtro que descarta chunk vencido), mas os arrays dentro dela não são
        escritos depois de criados — copiar os `chunk` a 30 Hz seria pagar
        memória por uma garantia que já existe."""
        self._chunks = list(chunks) if chunks else None
        self._passo = int(passo)
        self._lead = int(lead)

    def define_cabecalho(self, texto: str) -> None:
        self._cabecalho = texto

    # ── desenho ─────────────────────────────────────────────────────────────
    def compoe(self) -> np.ndarray:
        """Os quatro quadrantes e o cabeçalho, num canvas só."""
        q1 = desenha_atencao(self._rgb_mosaico, self._attn, self.qw, self.qh,
                             relativo=self._attn_e_relativo, base_n=self._attn_base_n,
                             nome=self.nome_atencao)
        q2 = desenha_profundidade(self._depth_mm, self._depth_modelo, self.qw, self.qh)
        q3 = desenha_nuvem(self._depth_mm, self.intrinsecos, self.qw, self.qh)
        # No simulador não existe telemetria de motor — o quadrante da
        # temperatura seria um retângulo com "sem telemetria" escrito. Ali cabe
        # o buffer de chunks, que é o que se quer olhar quando o robô é virtual.
        # No robô real a temperatura volta a ganhar: uma junta esquentando é
        # informação com consequência física, e o buffer está resumido no
        # cabeçalho ("chunks N | restam M").
        if self._temperaturas:
            q4 = desenha_temperatura(self._temperaturas, self.qw, self.qh)
        else:
            q4 = desenha_chunks(self._chunks, self._passo, self._lead, self.qw, self.qh)

        corpo = np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])
        cabecalho = np.full((28, corpo.shape[1], 3), (16, 16, 16), dtype=np.uint8)
        _texto(cabecalho, self._cabecalho or "FastWAM-D", 10, 19, 0.45, (210, 210, 210))
        return np.vstack([cabecalho, corpo])

    # ── ciclo de vida (cada transporte implementa o seu) ────────────────────
    def create(self) -> None:
        pass

    def show(self) -> bool:
        """Devolve False quando o operador pediu para encerrar."""
        return True

    def destroy(self) -> None:
        pass


class PainelDebug(PainelBase):
    """Uma janela OpenCV local com os quatro quadrantes.

    Uma janela só, e não quatro, porque o loop de controle roda a 30 Hz: cada
    `cv2.imshow` extra é tempo roubado do ciclo.
    """

    def __init__(self, nome: str = "FastWAM-D — Debug", **kwargs):
        super().__init__(**kwargs)
        self.nome = nome

    def create(self) -> None:
        _exige_opencv_com_gui()
        cv2.namedWindow(self.nome, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nome, self.qw * 2, self.qh * 2 + 28)

    def show(self) -> bool:
        """Desenha e devolve False quando o usuário fecha a janela ou tecla q/ESC."""
        cv2.imshow(self.nome, self.compoe())
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            return False
        # Janela fechada no X: o getWindowProperty vira <1.
        try:
            if cv2.getWindowProperty(self.nome, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False
        return True

    def destroy(self) -> None:
        try:
            cv2.destroyWindow(self.nome)
        except cv2.error:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Painel web (MJPEG) — para máquina sem tela
# ══════════════════════════════════════════════════════════════════════════
_PAGINA = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FastWAM-D — Debug</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#101010; color:#d8d8d8;
         font:14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  header { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
           padding:8px 12px; background:#181818; border-bottom:1px solid #2a2a2a; }
  #cab { color:#eaeaea; }
  #luz { width:9px; height:9px; border-radius:50%; background:#666; flex:0 0 auto; }
  #luz.on { background:#5cc45c; }
  #luz.off { background:#d05050; }
  main { padding:10px; }
  img { display:block; width:100%; height:auto; max-width:1400px; margin:0 auto;
        background:#181818; border:1px solid #2a2a2a; }
  small { color:#7a7a7a; }
</style>
<header>
  <span id="luz"></span>
  <span id="cab">conectando…</span>
  <small id="meta"></small>
</header>
<main><img id="v" alt="painel de debug"></main>
<script>
  const img = document.getElementById('v');
  // O <img> de MJPEG cai calado quando o stream corta (cliente fechado,
  // rede oscilando). Sem esta religada, a página fica com o último quadro
  // congelado na tela — o pior desfecho num painel de depuração: parece que
  // o robô parou, e foi só o socket.
  function liga() { img.src = 'stream.mjpg?t=' + Date.now(); }
  img.onerror = () => setTimeout(liga, 1000);
  liga();

  async function estado() {
    try {
      const r = await fetch('estado.json', { cache: 'no-store' });
      const e = await r.json();
      document.getElementById('cab').textContent = e.cabecalho || 'FastWAM-D';
      const idade = e.idade_s;
      document.getElementById('meta').textContent =
        `${e.fps_render.toFixed(1)} fps · ${e.clientes} espectador(es)` +
        (idade > 2 ? ` · quadro parado há ${idade.toFixed(0)} s` : '');
      document.getElementById('luz').className = idade > 2 ? 'off' : 'on';
    } catch (_) { document.getElementById('luz').className = 'off'; }
  }
  estado(); setInterval(estado, 1000);
</script>
"""


class PainelWeb(PainelBase):
    """Os mesmos quatro quadrantes, servidos por HTTP como MJPEG.

    Para quando a máquina que roda o loop de controle não tem tela — sessão por
    SSH, PC do robô num rack, ou o OpenCV headless do lerobot. Abra
    `http://<ip>:<porta>/` em qualquer navegador da LAN (inclusive o celular,
    o que é o modo útil de acompanhar o robô real: dá para ficar de pé ao lado
    dele, com a mão no botão de emergência, olhando o painel).

    A composição e a codificação JPEG rodam numa THREAD SEPARADA, e não no
    ciclo de controle: desenhar a nuvem de pontos é um laço em Python sobre
    milhares de pontos, e pagar isso a 30 Hz dentro do loop atrasaria o
    `send_action`. Aqui o loop só deposita os dados mais recentes (`define_*`)
    e segue; a thread desenha na cadência dela, e um quadro perdido no painel
    não custa nada.

    Sem autenticação e sem TLS, de propósito: é ferramenta de bancada em LAN
    fechada. Não exponha a porta para fora da rede do laboratório.
    """

    def __init__(self, porta: int = 8088, host: str = "0.0.0.0", fps: float = 10.0,
                 qualidade: int = 80, **kwargs):
        super().__init__(**kwargs)
        self.porta = int(porta)
        self.host = host
        self.periodo = 1.0 / max(0.5, float(fps))
        self.qualidade = int(qualidade)

        self._lock = threading.Lock()          # protege o estado dos quadrantes
        self._novo = threading.Condition()     # avisa os streams de quadro novo
        self._jpeg: bytes | None = None
        self._jpeg_em = 0.0
        self._seq = 0
        self._fps_render = 0.0
        self._clientes = 0
        self._parar = threading.Event()
        self._httpd = None
        self._thread_render: threading.Thread | None = None

    # ── entradas ────────────────────────────────────────────────────────────
    def define_imagens(self, rgb_mosaico=None, depth_mm=None) -> None:
        # Cópia da profundidade: o buffer da RealSense é reaproveitado pelo
        # driver a cada quadro, e a thread de desenho lê fora do ciclo. Sem a
        # cópia, o painel mostraria meio quadro velho e meio novo de vez em
        # quando — bem no quadrante que serve para julgar se a profundidade
        # está chegando. O mosaico já vem de um `resize`/`hstack`, é novo.
        if depth_mm is not None:
            depth_mm = np.array(depth_mm, copy=True)
        with self._lock:
            super().define_imagens(rgb_mosaico=rgb_mosaico, depth_mm=depth_mm)

    def define_debug_servidor(self, payload: dict | None) -> None:
        with self._lock:
            super().define_debug_servidor(payload)

    def define_temperaturas(self, temperaturas: dict[str, float] | None) -> None:
        with self._lock:
            super().define_temperaturas(temperaturas)

    def define_chunks(self, chunks: list | None, passo: int, lead: int) -> None:
        with self._lock:
            super().define_chunks(chunks, passo, lead)

    def define_cabecalho(self, texto: str) -> None:
        with self._lock:
            super().define_cabecalho(texto)

    # ── ciclo de vida ───────────────────────────────────────────────────────
    def create(self) -> None:
        painel = self

        class Manipulador(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):        # o servidor não polui o terminal
                pass                          # do loop de controle

            def _cabecalhos(self, tipo, tamanho=None, cache=False):
                self.send_response(200)
                self.send_header("Content-Type", tipo)
                if tamanho is not None:
                    self.send_header("Content-Length", str(tamanho))
                if not cache:
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()

            def do_GET(self):
                rota = self.path.split("?", 1)[0]
                if rota in ("/", "/index.html"):
                    corpo = _PAGINA.encode("utf-8")
                    self._cabecalhos("text/html; charset=utf-8", len(corpo))
                    self.wfile.write(corpo)
                elif rota == "/estado.json":
                    corpo = json.dumps(painel._estado()).encode("utf-8")
                    self._cabecalhos("application/json", len(corpo))
                    self.wfile.write(corpo)
                elif rota == "/quadro.jpg":
                    quadro = painel._ultimo_jpeg()
                    if quadro is None:
                        self.send_error(503, "sem quadro ainda")
                        return
                    self._cabecalhos("image/jpeg", len(quadro))
                    self.wfile.write(quadro)
                elif rota == "/stream.mjpg":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=quadro")
                self.end_headers()
                painel._entra_cliente()
                visto = -1
                try:
                    while not painel._parar.is_set():
                        quadro, visto = painel._espera_quadro(visto)
                        if quadro is None:
                            continue
                        self.wfile.write(b"--quadro\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(quadro)))
                        self.end_headers()
                        self.wfile.write(quadro)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass                      # aba fechada: normal, não é erro
                finally:
                    painel._sai_cliente()

        self._httpd = ThreadingHTTPServer((self.host, self.porta), Manipulador)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, name="painel-http",
                         daemon=True).start()
        self._thread_render = threading.Thread(target=self._laco_render,
                                               name="painel-render", daemon=True)
        self._thread_render.start()
        for url in self.urls():
            print(f"🖥️  Painel web: {url}")

    def urls(self) -> list[str]:
        """Endereços prováveis para abrir no navegador."""
        if self.host not in ("0.0.0.0", "::"):
            return [f"http://{self.host}:{self.porta}/"]
        enderecos = [f"http://localhost:{self.porta}/"]
        try:
            # Sem tráfego: o connect de UDP só escolhe a rota e revela o IP da
            # interface que sai para a LAN — que é o que o navegador de outra
            # máquina precisa digitar.
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            enderecos.append(f"http://{s.getsockname()[0]}:{self.porta}/")
            s.close()
        except OSError:
            pass
        return enderecos

    def show(self) -> bool:
        return not self._parar.is_set()

    def destroy(self) -> None:
        self._parar.set()
        with self._novo:
            self._novo.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ── interno ─────────────────────────────────────────────────────────────
    def _laco_render(self) -> None:
        ultimo = 0.0
        erros_vistos: set[str] = set()
        while not self._parar.is_set():
            inicio = time.monotonic()
            # Um quadrante que quebra (ex.: profundidade num formato inesperado)
            # não pode matar a thread: sem ela o painel congela no último quadro
            # até o fim da corrida, e reiniciar custa recarregar o modelo. O erro
            # é impresso uma vez por tipo e o laço tenta de novo no próximo ciclo.
            try:
                with self._lock:
                    canvas = self.compoe()
            except Exception as erro:  # noqa: BLE001
                chave = f"{type(erro).__name__}: {erro}"
                if chave not in erros_vistos:
                    erros_vistos.add(chave)
                    print(f"\n⚠️  painel: falha ao desenhar ({chave}) — seguindo")
                self._parar.wait(self.periodo)
                continue
            ok, buf = cv2.imencode(".jpg", canvas,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.qualidade])
            if ok:
                with self._novo:
                    self._jpeg = buf.tobytes()
                    self._jpeg_em = time.time()
                    self._seq += 1
                    gasto = time.monotonic() - inicio
                    # Média móvel: o custo do desenho varia com o número de
                    # pontos válidos na nuvem, e um número pulando não diz nada.
                    inst = 1.0 / max(gasto, 1e-3)
                    self._fps_render = inst if ultimo == 0.0 else 0.8 * ultimo + 0.2 * inst
                    ultimo = self._fps_render
                    self._novo.notify_all()
            sobra = self.periodo - (time.monotonic() - inicio)
            if sobra > 0:
                self._parar.wait(sobra)

    def _espera_quadro(self, visto: int, timeout: float = 2.0):
        with self._novo:
            if self._seq == visto:
                self._novo.wait(timeout)
            return self._jpeg, self._seq

    def _ultimo_jpeg(self) -> bytes | None:
        with self._novo:
            return self._jpeg

    def _entra_cliente(self) -> None:
        with self._novo:
            self._clientes += 1

    def _sai_cliente(self) -> None:
        with self._novo:
            self._clientes = max(0, self._clientes - 1)

    def _estado(self) -> dict:
        with self._lock:
            cabecalho = self._cabecalho
            temperaturas = dict(self._temperaturas or {})
        with self._novo:
            idade = time.time() - self._jpeg_em if self._jpeg_em else 1e9
            return {
                "cabecalho": cabecalho,
                "fps_render": round(self._fps_render, 2),
                "clientes": self._clientes,
                "idade_s": round(idade, 2),
                "temperaturas": temperaturas,
            }


class GrupoDePaineis(PainelBase):
    """Vários painéis com uma interface só (janela local E web, por exemplo).

    O loop de controle não deveria saber quantos painéis existem: ele chama
    `define_*` e `show()` uma vez, como se fosse um. `show()` devolve False se
    QUALQUER painel pediu para parar — fechar a janela local encerra a corrida,
    que é o comportamento que já existia.
    """

    def __init__(self, paineis):
        self.paineis = list(paineis)

    def define_imagens(self, **kwargs) -> None:
        for painel in self.paineis:
            painel.define_imagens(**kwargs)

    def define_debug_servidor(self, payload) -> None:
        for painel in self.paineis:
            painel.define_debug_servidor(payload)

    def define_temperaturas(self, temperaturas) -> None:
        for painel in self.paineis:
            painel.define_temperaturas(temperaturas)

    def define_chunks(self, chunks, passo: int, lead: int) -> None:
        for painel in self.paineis:
            painel.define_chunks(chunks, passo, lead)

    def define_cabecalho(self, texto: str) -> None:
        for painel in self.paineis:
            painel.define_cabecalho(texto)

    def create(self) -> None:
        for painel in self.paineis:
            painel.create()

    def destroy(self) -> None:
        for painel in self.paineis:
            painel.destroy()

    def show(self) -> bool:
        # Sem curto-circuito (`all` com lista, e não com gerador): mesmo que a
        # janela local já tenha pedido para parar, o painel web precisa do
        # `show()` dele para encerrar limpo.
        return all([painel.show() for painel in self.paineis])
