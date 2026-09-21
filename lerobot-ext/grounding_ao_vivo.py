#!/usr/bin/env python
"""
Grounding AO VIVO — a câmera do robô, sem controlar o robô
===========================================================
Lê o stream das câmeras do G1, roda o FastWAM-D em cada quadro e mostra, no
navegador, onde cada palavra da instrução está aterrissando na cena. O robô não
se mexe: nenhuma ação é enviada, e o bridge (`dex3_g1_server_v2.py`) nem
precisa estar no ar. Só as câmeras.

    NO ROBÔ (10.9.8.73)                     NA ATHENA
    full_realsenser_server.py  :5555  ──▶   grounding_ao_vivo.py  ──▶  :8089
    right_arm_realsense_server.py :5556 ─▶        (modelo de 6 B)      navegador

POR QUE ISTO É RÁPIDO
----------------------
A cross-attention texto→imagem é calculada no PREFILL do expert de vídeo
(`wan/modular.py:1837`), uma vez por quadro, antes de qualquer passo de
denoise. Como aqui não existe ação para produzir, os 10 passos de denoise do
expert de ação são desperdício puro: o script interrompe a inferência assim que
o prefill devolve o cache. Isso corta a maior parte do custo — sobra o VAE
codificando os quadros e os 30 blocos do vídeo.

No robô COM controle é diferente e nem precisa deste atalho: lá a inferência
completa acontece de qualquer jeito para gerar as ações, e o mapa pega carona
de graça no payload de debug do servidor.

O QUE VOCÊ VÊ
--------------
Vermelho é um pedaço da cena que olha para a palavra selecionada MAIS que o
pedaço médio; azul, menos. Não é atenção crua — atenção crua aqui tem dois
sumidouros empilhados (~87% da massa vai para o padding do prompt, e dos 13%
restantes `</s>` e `▁the` levam metade, em todo patch). Cada token é dividido
pela própria média espacial, então um sumidouro vira 1,0 uniforme e sai da
conta sozinho. Ver `grounding_fastwamd.py` para o raciocínio completo.

Clique nas palavras para trocar o que é mostrado. Vale qualquer palavra do
prompt, inclusive as abstratas — e é justamente comparar uma palavra de objeto
("cup") com uma abstrata ("instruction") que separa grounding de artefato.

O QUE ESTE MODO ASSUME
-----------------------
**Propriocepção constante.** Sem o bridge não há ângulo de junta, então o
estado vai como um vetor fixo. Isso é honesto para o que se mede aqui: a
propriocepção entra pelo `proprio_encoder`, que a pendura no FIM do contexto de
texto — depois dos tokens do prompt, que são os únicos que este mapa lê. Mas o
modelo nunca viu essa pose, então NÃO julgue ação por aqui; só o mapa.

**A câmera de pulso, se faltar, vira preto.** O mosaico do FastWAM-D é as duas
câmeras de cor lado a lado na largura, e a geometria depende disso. Sem a
:5556 a metade direita fica preta, o que é entrada fora da distribuição — o
mapa da metade da cabeça continua legível, mas o aviso aparece no painel.

USO
----
    bash maquinas/athena/launch_grounding_vivo.sh 2
    # e no navegador: http://10.9.8.252:8089/

Opções:
  --checkpoint=<PATH>  pretrained_model do FastWAM-D (obrigatório)
  --robo=<IP>          IP do robô (padrão: 10.9.8.73)
  --port-cam=<INT>     porta do stream da cabeça (padrão: 5555)
  --port-pulso=<INT>   porta do stream do pulso (padrão: 5556; 0 desliga)
  --porta=<INT>        porta do painel web (padrão: 8089)
  --task=<STR>         a instrução (padrão: "place the white cup on the dripper")
  --completo           NÃO corta no prefill: roda a inferência inteira. Mais
                       lento e sem ganho aqui — serve para conferir que o
                       atalho não muda o mapa.
  --device=<STR>       cuda / cuda:1 / cpu
  -h, --help           esta mensagem
"""

from __future__ import annotations

import os
import sys
import threading
import time

# zmq ANTES do torch — sem isso o processo morre em `Segmentation fault` sem
# imprimir uma linha. O libstdc++ do sistema não tem `GLIBCXX_3.4.29`, e quem
# carrega primeiro define qual vale para o processo inteiro. Ver maquinas/athena/README.
import zmq  # noqa: F401

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policies.fastwam_depth.debug_inferencia import CapturaDebug  # noqa: E402

ROBO_PADRAO = "10.9.8.73"
TAREFA_PADRAO = "place the white cup on the dripper"

# As 29 juntas na pose neutra. Ver a nota sobre propriocepção no cabeçalho:
# é uma constante honesta, não uma leitura.
ESTADO_NEUTRO = np.zeros(29, dtype=np.float32)


class _PrefillPronto(Exception):
    """Sinal interno: o prefill do vídeo acabou, não precisamos do denoise."""


# ─────────────────────────────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────────────────────────────
class MotorDeGrounding:
    """Carrega o modelo uma vez e devolve o mapa de um quadro."""

    def __init__(self, checkpoint: str, task: str, device: torch.device, completo: bool):
        from lerobot.policies.factory import make_pre_post_processors

        from policies.fastwam_depth.configuration_fastwam_depth import FastWAMDepthConfig
        from policies.fastwam_depth.modeling_fastwam_depth import FastWAMDepthPolicy

        print(f"⏳ Carregando FastWAM-D de: {checkpoint}")
        self.config = FastWAMDepthConfig.from_pretrained(checkpoint)
        self.policy = FastWAMDepthPolicy.from_pretrained(checkpoint, config=self.config)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, _ = make_pre_post_processors(
            policy_cfg=self.config, pretrained_path=checkpoint
        )
        self.task = task
        self.completo = completo
        self.prompt = self.config.prompt_template.format(task=task)

        tok = self.policy.model.tokenizer.tokenizer
        enc = tok(self.prompt, add_special_tokens=True, return_offsets_mapping=True)
        self.tokens = tok.convert_ids_to_tokens(enc["input_ids"])
        self.n_real = len(enc["input_ids"])

        print(f"✅ carregado — depth_mode={self.config.depth_mode} | "
              f"{self.n_real} tokens reais | corte no prefill: {not completo}")
        print(f"   prompt: {self.prompt}")

    def _instala_corte(self):
        """Interrompe a inferência assim que o prefill do vídeo devolve o cache.

        O `prefill_video_cache` roda os 30 blocos do expert de vídeo — que é
        onde a cross-attention com o texto acontece — e só DEPOIS começa o laço
        de denoise. Levantar daqui é o corte mais tardio que ainda pula todo o
        trabalho inútil, e não exige tocar no modelo.
        """
        from lerobot.policies.fastwam.wan.modular import MoT

        original = MoT.prefill_video_cache

        def prefill_e_para(self_mot, *a, **k):
            original(self_mot, *a, **k)
            raise _PrefillPronto

        MoT.prefill_video_cache = prefill_e_para
        return lambda: setattr(MoT, "prefill_video_cache", original)

    def mapa(self, head_rgb, pulso_rgb, head_depth, indices):
        """`([h, w] de contraste, ms)`. `head_*` em uint8/uint16, como a câmera manda."""
        raw = {"observation.state": torch.from_numpy(ESTADO_NEUTRO)}
        for chave, imagem in zip(self.config.rgb_feature_keys, (head_rgb, pulso_rgb)):
            raw[chave] = torch.from_numpy(np.ascontiguousarray(imagem)).permute(2, 0, 1).float().div(255.0)
        for chave in self.config.depth_feature_keys:
            raw[chave] = torch.from_numpy(np.ascontiguousarray(head_depth)).float().unsqueeze(0)
        raw["task"] = self.task

        batch = self.preprocessor(raw)
        batch.pop("action", None)

        restaura = None if self.completo else self._instala_corte()
        captura = CapturaDebug(self.policy)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode(), captura:
                try:
                    self.policy.predict_action_chunk(batch)
                except _PrefillPronto:
                    pass
        finally:
            if restaura is not None:
                restaura()
        ms = (time.perf_counter() - t0) * 1000.0
        return captura.mapa_grounding(indices, self.n_real), ms


# ─────────────────────────────────────────────────────────────────────
# Câmeras
# ─────────────────────────────────────────────────────────────────────
def abre_stream(ip: str, porta: int):
    """Stream compartilhado do servidor de câmera do robô.

    O `_SharedZMQStream` do lerobot entrega TODAS as imagens de um pacote de uma
    vez — é o que garante que a cor e a profundidade da cabeça sejam do mesmo
    instante. Abrir uma `ZMQCamera` por nome faria duas leituras do mesmo
    socket e poderia parear quadros de instantes diferentes.
    """
    from lerobot.cameras.zmq.camera_zmq import _SharedZMQStream

    fluxo = _SharedZMQStream(ip, porta, timeout_ms=5000)
    fluxo.start()
    return fluxo


def desenha(base_rgb, mapa, frase, ms, aviso):
    """Quadro composto: a imagem, o mapa por cima, e os números que importam."""
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    if mapa is not None:
        # Divergente em torno de 1,0: azul abaixo, vermelho acima. O JET do
        # painel de ação não serve aqui — ele não tem um meio neutro, e o "não
        # liga para esta palavra" acabaria pintado de verde-berrante.
        m = cv2.resize(mapa.astype(np.float32), (bgr.shape[1], bgr.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
        t = np.clip((m - 1.0) / 0.4, -1, 1)
        calor = np.zeros_like(bgr)
        calor[..., 2] = (np.clip(t, 0, 1) * 235).astype(np.uint8)          # vermelho (BGR)
        calor[..., 0] = (np.clip(-t, 0, 1) * 235).astype(np.uint8)          # azul
        calor[..., 1] = ((1 - np.abs(t)) * 120).astype(np.uint8)
        bgr = cv2.addWeighted(bgr, 0.62, calor, 0.38, 0)

    faixa = np.full((52, bgr.shape[1], 3), (18, 18, 20), dtype=np.uint8)
    cv2.putText(faixa, f'"{frase}"', (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA)
    detalhe = f"{ms:.0f} ms/quadro"
    if mapa is not None:
        detalhe += f"   contraste {mapa.min():.2f}-{mapa.max():.2f}x"
    cv2.putText(faixa, detalhe, (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 155, 165), 1, cv2.LINE_AA)
    if aviso:
        cv2.putText(faixa, aviso, (bgr.shape[1] - 8 - 9 * len(aviso), 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 170, 240), 1, cv2.LINE_AA)
    return np.vstack([faixa, bgr])


# ─────────────────────────────────────────────────────────────────────
# Painel
# ─────────────────────────────────────────────────────────────────────
_PAGINA = """<!doctype html><meta charset="utf-8">
<title>Grounding ao vivo — FastWAM-D</title>
<style>
 body{margin:0;background:#0d1015;color:#dce2ea;font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
 .w{max-width:1000px;margin:0 auto;padding:22px 18px 40px}
 h1{font-size:19px;font-weight:600;margin:0 0 2px}
 p.s{color:#8b95a5;font-size:13px;margin:0 0 16px}
 img{width:100%;border-radius:5px;display:block;background:#161a21}
 .t{display:flex;flex-wrap:wrap;gap:4px;margin-top:14px}
 button{font:12px ui-monospace,monospace;padding:5px 7px;border-radius:3px;cursor:pointer;
        background:#161a21;color:#dce2ea;border:1px solid #252b35}
 button.on{background:#2c2417;border-color:#d9a052;font-weight:600}
 button:hover{border-color:#39414e}
</style>
<div class="w">
<h1>Grounding ao vivo — FastWAM-D</h1>
<p class="s">Câmera do robô, sem controlar o robô. Vermelho = este pedaço da cena olha para a
palavra selecionada mais que o pedaço médio. Clique nas palavras.</p>
<img src="/stream" alt="Câmera do robô com o mapa de atenção de linguagem">
<div class="t" id="t"></div></div>
<script>
const TOKENS = __TOKENS__; let sel = new Set(__SEL__);
const cx = document.getElementById("t");
TOKENS.forEach((tk, i) => {
  const b = document.createElement("button");
  b.textContent = tk; b.className = sel.has(i) ? "on" : "";
  b.onclick = async () => {
    sel.has(i) ? sel.delete(i) : sel.add(i);
    if (!sel.size) sel.add(i);
    [...cx.children].forEach((c, j) => c.className = sel.has(j) ? "on" : "");
    await fetch("/frase?i=" + [...sel].join(","));
  };
  cx.appendChild(b);
});
</script>
"""


class Painel:
    """Servidor do painel: uma página, um stream MJPEG e a troca de palavra."""

    def __init__(self, porta: int, tokens: list[str], selecao: set[int]):
        self.porta = porta
        self.tokens = tokens
        self.selecao = selecao
        self.quadro: bytes | None = None
        self.versao = 0
        self.cond = threading.Condition()
        self._httpd = None

    def publica(self, imagem_bgr) -> None:
        ok, buf = cv2.imencode(".jpg", imagem_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self.cond:
            self.quadro = buf.tobytes()
            self.versao += 1
            self.cond.notify_all()

    def sobe(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        painel = self

        class Manipulador(BaseHTTPRequestHandler):
            def log_message(self, *_):        # o servidor não polui o terminal
                pass

            def do_GET(self):
                if self.path.startswith("/frase"):
                    from urllib.parse import parse_qs, urlparse

                    bruto = parse_qs(urlparse(self.path).query).get("i", [""])[0]
                    novos = {int(v) for v in bruto.split(",") if v.strip().isdigit()}
                    if novos:
                        painel.selecao = novos
                    self.send_response(204); self.end_headers()
                elif self.path == "/" or self.path.startswith("/index"):
                    import json

                    corpo = (_PAGINA
                             .replace("__TOKENS__", json.dumps(painel.tokens))
                             .replace("__SEL__", json.dumps(sorted(painel.selecao)))
                             ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(corpo)))
                    self.end_headers()
                    self.wfile.write(corpo)
                elif self.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
                self.end_headers()
                visto = -1
                try:
                    while True:
                        with painel.cond:
                            while painel.versao <= visto or painel.quadro is None:
                                painel.cond.wait(timeout=1.0)
                            quadro, visto = painel.quadro, painel.versao
                        self.wfile.write(b"--quadro\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(quadro)))
                        self.end_headers()
                        self.wfile.write(quadro + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass                      # aba fechada: normal, não é erro

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.porta), Manipulador)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    checkpoint = None
    robo, port_cam, port_pulso, porta = ROBO_PADRAO, 5555, 5556, 8089
    task, completo, device_str = TAREFA_PADRAO, False, None

    for arg in sys.argv[1:]:
        if arg in ("-h", "--help"):
            print(__doc__); return
        elif arg.startswith("--checkpoint="): checkpoint = arg.split("=", 1)[1]
        elif arg.startswith("--robo="):       robo = arg.split("=", 1)[1]
        elif arg.startswith("--port-cam="):   port_cam = int(arg.split("=", 1)[1])
        elif arg.startswith("--port-pulso="): port_pulso = int(arg.split("=", 1)[1])
        elif arg.startswith("--porta="):      porta = int(arg.split("=", 1)[1])
        elif arg.startswith("--task="):       task = arg.split("=", 1)[1]
        elif arg == "--completo":             completo = True
        elif arg.startswith("--device="):     device_str = arg.split("=", 1)[1]
        else:
            print(f"❌ opção desconhecida: {arg}  (use --help)"); sys.exit(2)

    if not checkpoint:
        print("❌ --checkpoint é obrigatório."); sys.exit(2)

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️  Device: {device}")
    motor = MotorDeGrounding(checkpoint, task, device, completo)

    # Abre com as palavras do objeto: é o que a pessoa veio ver. Procurar pelo
    # nome do token evita depender de índices, que mudam com a `--task`.
    selecao = {i for i, t in enumerate(motor.tokens) if t in ("▁white", "▁cup")}
    painel = Painel(porta, motor.tokens, selecao or {motor.n_real - 1})
    painel.sobe()
    print(f"\n🌐 Painel em http://<ip-desta-maquina>:{porta}/")

    print(f"📷 Conectando em {robo}:{port_cam}" + (f" e :{port_pulso}" if port_pulso else ""))
    cabeca = abre_stream(robo, port_cam)
    pulso = abre_stream(robo, port_pulso) if port_pulso else None

    versao_cabeca = 0
    n = 0
    try:
        while True:
            try:
                quadros = {}
                for nome in ("head_camera", "head_camera_depth"):
                    quadros[nome], versao_cabeca = cabeca.get_frame(nome, versao_cabeca - 1, 5000)
                versao_cabeca += 1
            except TimeoutError:
                print(f"⏳ sem quadro de {robo}:{port_cam} — o servidor de câmera está no ar?")
                continue

            head = quadros["head_camera"]
            if head.ndim == 3 and head.shape[2] == 3:
                head = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
            depth = np.squeeze(quadros["head_camera_depth"])

            aviso = ""
            if pulso is not None:
                try:
                    bruto, _ = pulso.get_frame("right_wrist_camera", -1, 200)
                    pulso_rgb = cv2.cvtColor(bruto, cv2.COLOR_BGR2RGB)
                except (TimeoutError, KeyError):
                    pulso_rgb = np.zeros_like(head); aviso = "sem camera de pulso"
            else:
                pulso_rgb = np.zeros_like(head); aviso = "sem camera de pulso"

            indices = sorted(painel.selecao)
            mapa, ms = motor.mapa(head, pulso_rgb, depth, indices)
            frase = " ".join(motor.tokens[i].replace("▁", "") for i in indices)
            painel.publica(desenha(head, mapa, frase, ms, aviso))

            n += 1
            if n % 10 == 0:
                print(f"[{n:05d}] {ms:6.0f} ms/quadro  ({1000.0 / max(ms, 1):.1f} q/s)  {frase!r}")
    except KeyboardInterrupt:
        print("\nencerrando.")
    finally:
        cabeca.stop()
        if pulso is not None:
            pulso.stop()


if __name__ == "__main__":
    main()
