#!/usr/bin/env python
"""
Gerador de dataset no MuJoCo — especialista roteirizado → formato LeRobot.
================================================================================
Roda o mesmo especialista do `demo_pega_copo_mujoco.py`, mas gravando. Cada
episódio é salvo ASSIM QUE TERMINA (`save_episode`, que já encoda os vídeos), a
cena é sorteada de novo e o próximo começa.

    python gerar_dataset_mujoco.py --episodios=50
    python gerar_dataset_mujoco.py --episodios=200 --repo-id=Mrwlker/sim_copo_2026-09-02
    python gerar_dataset_mujoco.py --episodios=5 --com-janela     # ver enquanto grava

── Uma tarefa ou duas ──────────────────────────────────────────────────────
`--tarefa` escolhe QUAL PEDAÇO do roteiro vira dataset:

    --tarefa=pega       partida → ... → levantar   ("pick up the white cup")
    --tarefa=poe        recolher → ... → voltar    ("place ... on the dripper")
    --tarefa=completo   o roteiro inteiro (padrão)

Em `poe` a pega ainda RODA, sem gravar, só para o braço chegar a um estado
inicial que a física de fato produz — ver o comentário no `episodio`.

Os dois podem ser gravados AO MESMO TEMPO, em processos separados, desde que
tenham `--repo-id` e `--semente` diferentes (o `grava_duplo.sh` faz isso). Cada
processo tem seu MuJoCo, seus renderizadores e seu diretório; o que eles
disputam é GPU e CPU, não arquivo.

── Por que importa do demo em vez de copiar ────────────────────────────────
A trajetória tem, hoje, seis coisas ajustadas contra sintomas que só aparecem na
tela: perfil de jerk mínimo, busca de pose livre no espaço nulo, mira da palma,
fechamento por contato, cintura, tolerância por fase. Duplicar isso aqui criaria
duas versões que divergem na primeira correção. O demo ganhou um `ao_passo`, e o
gravador é só o que se pendura nele.

── O esquema é o do dataset REAL, de propósito ─────────────────────────────
As features saem idênticas às de `meu_dataset/white_cup_on_dripper_2026-08-11`,
incluindo as duas de pressão. O `aggregate_datasets` do LeRobot compara com
`features_equal_for_merge`: `fps`, `robot_type` e o dicionário de features. Para
features NÃO-vídeo a igualdade é total — `names` incluído. Para features de
vídeo ele ignora `preset`, `crf`, `g`, `backend`, `fast_decode` e
`extra_options`, mas NÃO ignora `video.codec`.

DUAS ARMADILHAS, as duas cobradas em 03/09 com 275 episódios já gravados:

  1. `names` gerado por enum. Estava `[m.name for m in G1_29_JointIndex]`, que
     começa pelas pernas, enquanto o `ao_passo` escreve braços em 0-13. Ver o
     comentário do `NOMES_29` abaixo.

  2. CODEC. O `video.codec` declarado no `monta_features` NÃO manda: quem
     encoda é o `LeRobotDataset`, com os defaults da versão instalada. O real
     (08/2026) saiu em `h264`; a gravação de 03/09 saiu em `av1`, porque o
     default do LeRobot mudou no meio. O merge recusou e foi preciso reencodar
     1,6 GB de vídeo. CORRIGIDO em 04/09: o `create` agora passa
     `RGBEncoderConfig(vcodec="h264")` explícito. Ainda assim, antes de
     gravar dataset novo pra juntar com um antigo, CONFIRA no `meta/info.json`
     do que saiu — é o único lugar que diz a verdade.

E as pressões vão ZERADAS: esta simulação não tem sensor tátil. É dado falso com
forma certa, e quem for usar tátil precisa saber disso — está aqui, no
`meta/info.json` não estaria.

── O que este dataset NÃO é ────────────────────────────────────────────────
Cinemática, não dinâmica: os contatos são detectados, não resolvidos, e a
xícara é presa à pinça enquanto agarrada. As trajetórias de junta e as imagens
são coerentes entre si; força de preensão e escorregamento não existem aqui.
"""

from __future__ import annotations

import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import mujoco  # noqa: E402

import demo_pega_copo_mujoco as expert  # noqa: E402
from lerobot.configs.video import RGBEncoderConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# O texto da tarefa vai em CADA quadro (`"task"`), e é por ele que a política
# condicionada por linguagem separa uma coisa da outra. Os dois datasets são
# gravados apartados, mas os textos precisam ser distintos de qualquer forma:
# se um dia forem juntos, "pegar" e "pousar" partem de imagens parecidas (mão
# perto da xícara) e só a frase os distingue.
TAREFAS = {
    "completo": "place the white cup on the dripper",
    "pega": "pick up the white cup",
    "poe": "place the white cup on the dripper",
}
FPS = 30

# Resoluções idênticas às do dataset real — ver o `meta/info.json` dele.
CAMS = {
    "head_camera": (480, 848),
    "head_camera_depth": (480, 848),
    "right_wrist_camera": (224, 224),
}


# Os 29 nomes NA ORDEM EM QUE O `ao_passo` ESCREVE, que é a ordem do dataset
# real — copiados do `meta/info.json` dele, não gerados por enum.
#
# Aqui morava `[m.name for m in G1_29_JointIndex]`, e estava ERRADO: aquele enum
# começa pelas 12 juntas das pernas, então a dimensão 0 era rotulada
# `kLeftHipPitch` enquanto o `ao_passo` escreve ali `kLeftShoulderPitch`. Os
# VALORES sempre estiveram no layout do real (braços 0-13, cintura 14, mão
# direita 22-28); só o rótulo mentia.
#
# Não era cosmético: `features_equal_for_merge` compara o dicionário inteiro
# das features não-vídeo, `names` incluído, e o `aggregate_datasets` recusava o
# co-treino sim+real por causa disso. Custou reescrever o `info.json` de um
# dataset de 275 episódios já gravado.
NOMES_29 = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q",
    "kLeftElbow.q", "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
    "kRightElbow.q", "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
    "kWaistYaw.q",
    "left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q",
    "left_hand_thumb_2_joint.q", "left_hand_middle_0_joint.q",
    "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
    "left_hand_index_1_joint.q",
    "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q",
    "right_hand_thumb_2_joint.q", "right_hand_index_0_joint.q",
    "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
    "right_hand_middle_1_joint.q",
]


def monta_features():
    """Esquema idêntico ao do dataset gravado no robô real."""
    nomes = NOMES_29
    f = {
        "action": {"dtype": "float32", "shape": (29,), "names": nomes},
        "observation.state": {"dtype": "float32", "shape": (29,), "names": nomes},
        # `is_depth_map` NÃO é enfeite de metadado: o `meta.depth_keys` o lê e
        # manda a câmera para o pipeline de profundidade (TIFF de 16 bits e
        # encoder que quantiza métrica de 1 canal). Aqui ele está CERTO, porque
        # o MuJoCo entrega profundidade métrica de verdade — ao contrário do
        # `realsense_server.py`, que publica cinza de 3 canais e por isso tem a
        # flag desligada no caminho do robô real.
        "observation.images.head_camera": {
            "dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False, "video.height": 480, "video.width": 848,
                     "video.channels": 3, "video.codec": "hevc", "video.fps": FPS,
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "has_audio": False},
        },
        "observation.images.head_camera_depth": {
            "dtype": "video", "shape": (480, 848, 1), "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "video.height": 480, "video.width": 848,
                     "video.channels": 1, "video.codec": "hevc", "video.fps": FPS,
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": True,
                     "has_audio": False},
        },
        "observation.images.right_wrist_camera": {
            "dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False, "video.height": 224, "video.width": 224,
                     "video.channels": 3, "video.codec": "hevc", "video.fps": FPS,
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "has_audio": False},
        },
        "observation.left_hand_pressure": {
            "dtype": "float32", "shape": (33,),
            "names": [f"left_hand_pressure_{i}" for i in range(33)]},
        "observation.right_hand_pressure": {
            "dtype": "float32", "shape": (33,),
            "names": [f"right_hand_pressure_{i}" for i in range(33)]},
    }
    return f


class Cameras:
    """Os três renderizadores offscreen do MuJoCo.

    Criados uma vez e reaproveitados: `mujoco.Renderer` aloca contexto OpenGL, e
    abrir um por quadro é a diferença entre gravar em minutos e em horas.
    """

    def __init__(self, modelo):
        self.r_cor = {}
        for nome, (h, w) in CAMS.items():
            if nome.endswith("_depth"):
                continue
            self.r_cor[nome] = mujoco.Renderer(modelo, h, w)
        h, w = CAMS["head_camera_depth"]
        self.r_prof = mujoco.Renderer(modelo, h, w)
        self.r_prof.enable_depth_rendering()

    def captura(self, dados):
        saida = {}
        for nome, r in self.r_cor.items():
            r.update_scene(dados, camera=nome)
            saida[nome] = r.render().astype(np.uint8)

        # A profundidade do MuJoCo vem em METROS, float32. O dataset real está em
        # MILÍMETROS uint16 — foi assim que a RealSense publicou (medido: 480x848,
        # 0 a 2677 mm). Converter aqui, e não no treino, mantém as duas fontes na
        # mesma unidade e faz o `depth_unit: mm` dos configs valer para as duas.
        self.r_prof.update_scene(dados, camera="head_camera_depth")
        metros = self.r_prof.render()
        mm = np.clip(metros * 1000.0, 0, 65535).astype(np.uint16)
        saida["head_camera_depth"] = mm[..., None]
        return saida


def _parada_limpa(_sig, _frame):
    """SIGTERM vira KeyboardInterrupt.

    Escrito depois de perder 555 episódios: um `kill` comum chegou durante a
    escrita do parquet, que guarda TODO o índice num rodapé no fim do arquivo.
    Interrompido no meio, o arquivo fica com os dados e sem o rodapé — e o
    pyarrow não lê nada, nem as partes íntegras. Os 7 GB de vídeo sobreviveram
    e ficaram inúteis, porque o que amarra quadro a estado estava no parquet.
    
    Com este tratador, `kill` (TERM) desmonta pelo mesmo caminho do Ctrl+C: o
    episódio em curso é abandonado e o `finally` fecha o que já estava salvo.
    Contra `kill -9` não há defesa — esse não passa pelo processo.
    """
    raise KeyboardInterrupt


# ── ESPIA (18/09) ───────────────────────────────────────────────────────────
# O `--com-janela` NAO funciona nesta maquina: o visualizador do MuJoCo e os tres
# renderizadores offscreen disputam o contexto GL e a gravacao TRAVA em
# "renderizadores offscreen...", tanto com o backend padrao quanto com MUJOCO_GL=glfw.
# Medido duas vezes em 18/09. Como headless grava normal, a vista ao vivo sai por
# aqui: a cada episodio salvo, o ultimo quadro da camera da cabeca vira um PNG, e
# uma pagina com auto-refresh o mostra. Custa uma escrita de imagem por episodio.
#
# Escreve em arquivo temporario e RENOMEIA: sem isso o navegador pega o PNG pela
# metade e pisca em cinza.
def _espia(raiz_espia, imagem, n, salvos, descartados, fase=""):
    try:
        from PIL import Image
        raiz_espia.mkdir(parents=True, exist_ok=True)
        tmp = raiz_espia / "ultimo.tmp.jpg"
        # JPEG e nao PNG: isto roda A CADA 10 PASSOS, dentro do laco. PNG de
        # 480x848 custa ~20 ms e derrubaria o ritmo da gravacao; JPEG 70 custa ~3 ms.
        Image.fromarray(imagem).save(tmp, quality=70)
        tmp.replace(raiz_espia / "ultimo.jpg")
        # A pagina so e escrita UMA vez: ela se atualiza sozinha por JS, trocando a
        # `src` da imagem a cada 300 ms com um parametro anti-cache. Um `meta refresh`
        # recarregaria o documento inteiro e mostraria 1 quadro a cada 3 s, piscando.
        # Os contadores vem de `estado.json`, lido no mesmo intervalo.
        import json as _json
        (raiz_espia / "estado.json").write_text(_json.dumps(
            {"episodio": n, "salvos": salvos, "descartados": descartados, "fase": str(fase)}))
        pagina = raiz_espia / "index.html"
        if not pagina.exists():
            pagina.write_text("""<!doctype html><meta charset=utf-8><title>gravando</title>
<body style="margin:0;background:#15161a;color:#e8e8ea;font:14px system-ui,sans-serif;text-align:center">
<p id=t style="padding:8px 0;margin:0">a espera do primeiro quadro...</p>
<img id=q style="max-width:100%;height:auto">
<script>
const q=document.getElementById('q'), t=document.getElementById('t');
setInterval(()=>{q.src='ultimo.jpg?'+Date.now();},300);
setInterval(()=>{fetch('estado.json?'+Date.now()).then(r=>r.json()).then(e=>{
  t.textContent='episodio '+e.episodio+'  |  '+e.salvos+' salvos  |  '+e.descartados
              +' descartados  |  fase: '+e.fase;}).catch(()=>{});},1000);
</script></body>""")
    except Exception:            # noqa: BLE001 - o espia NUNCA derruba a gravacao
        pass


def main():
    signal.signal(signal.SIGTERM, _parada_limpa)
    args = sys.argv[1:]
    n_eps = 20
    repo_id = "Mrwlker/sim_copo_mujoco"
    raiz = None
    tarefa = "completo"
    semente = 0
    com_janela = "--com-janela" in args
    for a in args:
        if a.startswith("--episodios="):
            n_eps = int(a.split("=", 1)[1])
        elif a.startswith("--repo-id="):
            repo_id = a.split("=", 1)[1]
        elif a.startswith("--root="):
            raiz = a.split("=", 1)[1]
        elif a.startswith("--tarefa="):
            tarefa = a.split("=", 1)[1]
        elif a.startswith("--semente="):
            semente = int(a.split("=", 1)[1])
    if tarefa not in TAREFAS:
        print(f"❌ --tarefa={tarefa} não existe (use {'|'.join(TAREFAS)})")
        sys.exit(1)
    # As DUAS sementes, e as duas importam:
    #  - a legada (`np.random.seed`) alimenta o ruído do espaço nulo dentro do
    #    `resolve_livre`, que é o que faz duas tentativas da mesma pose darem
    #    resultados diferentes;
    #  - a do `default_rng` sorteia a CENA (posição da xícara e do coador).
    # Só a primeira estava ligada ao `--semente`, e a da cena era um `0` cravado
    # no código. Efeito: toda corrida via `grava_dataset.sh` gerava exatamente a
    # mesma sequência de cenas, com semente ou sem — e duas gravações
    # simultâneas gravariam as mesmas posições nos dois datasets. Agora as duas
    # sementes saem daqui.
    np.random.seed(semente)
    rng = np.random.default_rng(semente)

    raiz = Path(raiz) if raiz else AQUI / "meu_dataset" / repo_id.split("/")[-1]
    # Retomar em vez de recusar: escrito depois de uma corrida de 600 morrer
    # (sem traceback — os processos em bg do terminal caíram juntos) aos 113
    # episódios. `LeRobotDataset.resume` é a API própria pra isso — carrega os
    # metadados do que já existe e devolve um writer que continua a numeração
    # de episódio de onde parou. Sem ela, a alternativa era gravar num
    # diretório novo e juntar depois com o `aggregate_datasets`, que teria que
    # reler e recodificar tudo que já estava pronto.
    retomando = raiz.exists()

    print(f"⏳ carregando a cena (janela: {'sim' if com_janela else 'não'})...")
    cena = expert.Cena(com_janela=com_janela)
    # REINSTALA depois de criar o viewer. O `launch_passive` instala handlers
    # próprios, e o que foi registrado antes dele é sobrescrito — medido: com o
    # registro só no topo do `main`, `kill -TERM` não parava nada e o processo
    # seguia gravando. `kill -INT` funcionava porque cai no tratamento nativo
    # do Python. Registrando de novo aqui, os dois desmontam limpo.
    signal.signal(signal.SIGTERM, _parada_limpa)
    signal.signal(signal.SIGINT, _parada_limpa)
    print("⏳ montando o IK...")
    ik = expert.G1_29_ArmIK(
        visualization=False,
        # Punho direito preso: sem isto o solver o gira durante o movimento.
        travar_punho_dir=False,
        # Cotovelo DIREITO dobrado (índice 10 = right_elbow_joint, faixa
        # -1.047 a +2.094, zero = esticado). A regularização do IK puxa para
        # esta postura; com o padrão de zeros ela puxava para braço reto, que
        # era a causa do cotovelo esticado — não a busca no espaço nulo.
        postura_ref=np.array([0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 1.0, 0, 0, 0], dtype=float),
    )
    print("⏳ pose de partida...")
    DESCANSO = np.array([0.42, -0.26, cena.z_copo + 0.22])
    cena.define_pose_partida(ik, DESCANSO)
    cena.descanso = DESCANSO

    print("⏳ renderizadores offscreen...")
    cams = Cameras(cena.m)

    if retomando:
        ds = LeRobotDataset.resume(repo_id=repo_id, root=raiz)
        print(f"✅ retomando {raiz} — {ds.meta.total_episodes} episódios, "
              f"{ds.meta.total_frames} quadros já no disco\n")
    else:
        ds = LeRobotDataset.create(
            repo_id=repo_id, fps=FPS, features=monta_features(),
            root=raiz, robot_type="unitree_g1_dex3", use_videos=True,
            # H264 CRAVADO, e é o conserto da armadilha nº 2 lá de cima. O
            # default do `RGBEncoderConfig` é `libsvtav1`, e foi ele que fez a
            # gravação de 03/09 sair em `av1` enquanto o dataset real (08/2026)
            # está em `h264`. O `features_equal_for_merge` compara `video.codec`
            # e recusou o merge; custou reencodar 1,6 GB de vídeo depois de
            # gravado. A profundidade não precisa de nada: o
            # `DepthEncoderConfig` já usa `hevc`, que é o que o real tem.
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        )
        print(f"✅ dataset em {raiz}\n"
              f"   tarefa: {tarefa} — \"{TAREFAS[tarefa]}\" | semente: {semente}\n")

    zeros33 = np.zeros(33, dtype=np.float32)
    salvos = descartados = 0
    t0 = time.time()

    try:
        for n in range(1, n_eps + 1):
            quadros = []

            def ao_passo(fase, q_braco, _c=cena, _cams=cams, _buf=quadros):
                # O ESTADO é lido do MuJoCo depois do passo, e a AÇÃO é o mesmo
                # vetor: em cinemática pura a junta comandada É a junta atingida.
                # Num gerador com dinâmica os dois teriam que ser separados —
                # está escrito aqui para ninguém assumir o contrário depois.
                estado = np.zeros(29, dtype=np.float32)
                for i, nome in enumerate(expert.JUNTAS_BRACO):
                    jid = mujoco.mj_name2id(_c.m, mujoco.mjtObj.mjOBJ_JOINT, nome)
                    estado[i] = _c.d.qpos[_c.m.jnt_qposadr[jid]]
                estado[14] = _c.d.qpos[_c.adr_cintura]
                for k, j in enumerate(_c.mao):
                    if 22 + k < 29:
                        estado[22 + k] = _c.d.qpos[j["adr"]]
                imgs = _cams.captura(_c.d)
                _buf.append((estado, imgs))
                if len(_buf) % 10 == 0:
                    _espia(raiz.parent / "espia", imgs["head_camera"],
                           n, salvos, descartados, fase)

            ok = expert.episodio(cena, ik, rng, n, dorme=com_janela,
                                 ao_passo=ao_passo, tarefa=tarefa)

            if not ok or len(quadros) < 30:
                descartados += 1
                print(f"   ↳ descartado ({len(quadros)} quadros)\n")
                continue

            for estado, imgs in quadros:
                ds.add_frame({
                    "action": estado,
                    "observation.state": estado,
                    "observation.left_hand_pressure": zeros33,
                    "observation.right_hand_pressure": zeros33,
                    **{f"observation.images.{k}": v for k, v in imgs.items()},
                    "task": TAREFAS[tarefa],
                })
            # SALVA AGORA, não no fim. `save_episode` encoda os vídeos do
            # episódio e esvazia o buffer; deixar para o fim significaria manter
            # todos os quadros de todos os episódios em memória — a 30 Hz, com
            # três câmeras, isso estoura antes do décimo.
            ds.save_episode()
            salvos += 1
            dt = time.time() - t0
            print(f"   ↳ salvo ({len(quadros)} quadros) — "
                  f"{salvos} salvos, {descartados} descartados, "
                  f"{dt / max(salvos, 1):.0f}s por episódio\n")
    except KeyboardInterrupt:
        print("\ninterrompido — o que já foi salvo está no disco.")
    finally:
        print(f"== {salvos} episódios salvos, {descartados} descartados ==")
        print(f"   {raiz}")
        # Só apaga se o diretório nasceu NESTA corrida e ficou vazio. Numa
        # retomada (`retomando`), `salvos == 0` só quer dizer que esta chamada
        # não acrescentou nada — apagar destruiria os episódios de corridas
        # anteriores que já estavam lá.
        if salvos == 0 and not retomando and raiz.exists():
            shutil.rmtree(raiz, ignore_errors=True)
            print("   (nada salvo — diretório removido)")


if __name__ == "__main__":
    main()
