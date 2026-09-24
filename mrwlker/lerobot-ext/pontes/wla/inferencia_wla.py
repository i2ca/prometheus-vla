#!/usr/bin/env python
"""Inferência da UnifoLM-WLA — o caminho de volta que eles não publicaram.

O repositório deles só tem treino. `predict_action` existe e recebe texto, mas
devolve ação NORMALIZADA no espaço unificado de 54 dims, relativa à pose atual
da mão. Este arquivo é o inverso exato do `converte_dataset_wla.py`: pega o que
o modelo cospe e devolve pose absoluta da mão, garra e cintura.

    python -m pontes.wla.inferencia_wla --testa      # confere a matemática sem o modelo

── A cadeia, lida do código deles ──────────────────────────────────────────
No `single_source_dataset.py` a ação do braço é `T_rel = T_atual⁻¹ @ T_futuro`
(`se3_utils.compute_relative_actions`), depois normalizada com `relative_stats`
em **zscore** — porque o nosso `dados_prometheus.yaml` diz `rel_norm_type:
zscore`. Garra e cintura NÃO são relativas: saem de `stats.json` em `minmax_q`.
O normalizador deles é `(valor - offset) / escala`, então o inverso é
`valor = normalizado * escala + offset`, e a pose absoluta volta com
`T_futuro = T_atual @ T_rel`.

O estado tem uma assimetria que engana: a rotação da mão vai como **rot6d** no
estado e como **rotvec** na ação, e no estado **só o xyz é normalizado** — o
rot6d passa cru (ver `_normalize_state_unified`). Normalizar o rot6d aqui daria
um estado que o modelo nunca viu.

── Uma bandeira morta ──────────────────────────────────────────────────────
`merge_left_right_ee_stats: true` está no nosso YAML e não faz nada: o campo é
declarado no `config.py` deles e não é lido em lugar nenhum. Esquerda e direita
usam estatísticas separadas, e é assim que este arquivo faz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
_PASTA = RAIZ / "unifolm-wla/unifolm_wla/dataloader/multi_source_dataset"
_PAI = "_wla_leve"


def _importa(nome: str):
    """Carrega UM arquivo deles, sem passar pelo `__init__` do pacote.

    Um `import unifolm_wla.dataloader...` normal executa o `__init__` da cadeia,
    que puxa o `lerobot_wrapper` e morre com `ImportError` em
    `get_hf_features_from_features` — o LeRobot instalado aqui é mais novo que o
    que eles esperam. Os arquivos de que precisamos (contas de SE(3), fatias e
    estatísticas) não dependem de nada disso.

    Eles usam import relativo (`from .config import ...`), então não basta
    carregar pelo caminho: é preciso um pacote-pai de mentira apontando para a
    pasta, senão dá "attempted relative import with no known parent package".
    """
    import importlib.util
    import types

    if _PAI not in sys.modules:
        pai = types.ModuleType(_PAI)
        pai.__path__ = [str(_PASTA)]
        sys.modules[_PAI] = pai

    alvo = f"{_PAI}.{nome}"
    if alvo in sys.modules:
        return sys.modules[alvo]
    espec = importlib.util.spec_from_file_location(alvo, _PASTA / f"{nome}.py")
    mod = importlib.util.module_from_spec(espec)
    sys.modules[alvo] = mod           # antes do exec, para o import relativo achar
    espec.loader.exec_module(mod)
    return mod


_am = _importa("action_mapping")
_se3 = _importa("se3_utils")
_st = _importa("stats_utils")

SLICES, STATE_SLICES = _am.SLICES, _am.STATE_SLICES
UNIFIED_DIM, STATE_DIM = _am.UNIFIED_DIM, _am.STATE_DIM
matrix_to_rot6d, pose_to_se3 = _se3.matrix_to_rot6d, _se3.pose_to_se3
rotvec_to_matrix, se3_inverse = _se3.rotvec_to_matrix, _se3.se3_inverse
se3_to_xyz_rotvec = _se3.se3_to_xyz_rotvec
get_normalizer, load_relative_stats, load_stats = (
    _st.get_normalizer, _st.load_relative_stats, _st.load_stats)

STATS = RAIZ / "unifolm-wla/unifolm_wla/dataloader/multi_source_dataset/stats"

# Chaves do nosso `dados_prometheus.yaml`, âncora `*nosso`.
CHAVES = {
    "ee_dir":   ("right_ee_pose_gripper_base", "zscore"),          # relative_stats
    "ee_esq":   ("left_ee_pose_gripper_base", "zscore"),           # relative_stats
    "garra_dir": ("action.right_gripper", "minmax_q"),             # stats
    "garra_esq": ("action.left_gripper", "minmax_q"),
    "cintura":  ("action.waist_action_joint", "minmax_q"),
}
ESTADO_CHAVES = {
    "ee_dir":    ("observation.state.right_ee_pose_gripper_base", "minmax_q"),
    "ee_esq":    ("observation.state.left_ee_pose_gripper_base", "minmax_q"),
    "garra_dir": ("observation.state.right_gripper", "minmax_q"),
    "garra_esq": ("observation.state.left_gripper", "minmax_q"),
    "cintura":   ("observation.state.waist_state_joint", "minmax_q"),
}


class Escalas:
    """Os (offset, escala) de cada campo, lidos dos MESMOS arquivos do treino."""

    def __init__(self, pasta: Path = STATS):
        self.stats = load_stats(Path(pasta) / "stats.json")
        self.rel = load_relative_stats(Path(pasta) / "relative_stats.json")
        self.acao = {}
        for nome, (chave, tipo) in CHAVES.items():
            fonte = self.rel if nome.startswith("ee_") else self.stats
            self.acao[nome] = get_normalizer(fonte, chave, tipo)
        self.estado = {
            nome: get_normalizer(self.stats, chave, tipo)
            for nome, (chave, tipo) in ESTADO_CHAVES.items()
        }

    @staticmethod
    def aplica(v, par):
        offset, escala = par
        return (np.asarray(v, np.float32) - offset) / escala

    @staticmethod
    def desfaz(v, par):
        offset, escala = par
        return np.asarray(v, np.float32) * escala + offset


def monta_estado(esc: Escalas, T_esq, T_dir, garra_esq, garra_dir, cintura) -> np.ndarray:
    """Vetor de 60 dims como o dataloader deles monta, já normalizado.

    Atenção ao que NÃO é normalizado: o rot6d da mão passa cru, porque o
    `_normalize_state_unified` só toca no xyz dos campos de pose.
    """
    e = np.zeros(STATE_DIM, np.float32)
    for T, fatia, chave in ((T_esq, STATE_SLICES["left_xyz_rot6d"], "ee_esq"),
                            (T_dir, STATE_SLICES["right_xyz_rot6d"], "ee_dir")):
        if T is None:
            continue
        i = fatia.start
        offset, escala = esc.estado[chave]
        e[i:i + 3] = (T[:3, 3] - offset[:3]) / escala[:3]
        e[i + 3:i + 9] = matrix_to_rot6d(T[:3, :3])
    for g, fatia, chave in ((garra_esq, STATE_SLICES["left_gripper"], "garra_esq"),
                            (garra_dir, STATE_SLICES["right_gripper"], "garra_dir")):
        if g is not None:
            e[fatia] = esc.aplica([g], esc.estado[chave])
    if cintura is not None:
        e[STATE_SLICES["waist_joint"]] = esc.aplica(cintura, esc.estado["cintura"])
    return e


def mascara_estado(tem_esq: bool = True) -> np.ndarray:
    m = np.zeros(STATE_DIM, bool)
    m[STATE_SLICES["right_xyz_rot6d"]] = True
    m[STATE_SLICES["right_gripper"]] = True
    m[STATE_SLICES["waist_joint"]] = True
    if tem_esq:
        m[STATE_SLICES["left_xyz_rot6d"]] = True
        m[STATE_SLICES["left_gripper"]] = True
    return m


def mascara_acao(tem_esq: bool = True) -> np.ndarray:
    m = np.zeros(UNIFIED_DIM, bool)
    m[SLICES["right_xyz_rotvec"]] = True
    m[SLICES["right_gripper"]] = True
    m[SLICES["waist_joint"]] = True
    if tem_esq:
        m[SLICES["left_xyz_rotvec"]] = True
        m[SLICES["left_gripper"]] = True
    return m


def desnormaliza(esc: Escalas, acao_norm: np.ndarray, T_esq, T_dir) -> dict:
    """(T, 54) normalizado → poses ABSOLUTAS da mão, garra e cintura.

    `acao_norm` é o que sai do `predict_action`, para UM exemplo do lote.
    """
    a = np.atleast_2d(np.asarray(acao_norm, np.float32))
    saida = {}
    for lado, chave, fatia_ee, fatia_g, T_atual in (
        ("esq", "ee_esq", SLICES["left_xyz_rotvec"], SLICES["left_gripper"], T_esq),
        ("dir", "ee_dir", SLICES["right_xyz_rotvec"], SLICES["right_gripper"], T_dir),
    ):
        if T_atual is None:
            continue
        rel = esc.desfaz(a[:, fatia_ee], esc.acao[chave])          # (T, 6) xyz+rotvec
        poses = np.stack([
            T_atual @ pose_to_se3(r[:3], rotvec_to_matrix(r[3:])) for r in rel
        ])
        saida[f"pose_{lado}"] = poses
        saida[f"garra_{lado}"] = esc.desfaz(a[:, fatia_g], esc.acao[f"garra_{lado}"])[:, 0]
    saida["cintura"] = esc.desfaz(a[:, SLICES["waist_joint"]], esc.acao["cintura"])
    return saida


# ══════════════════════════════════════════════════════════════════════════
# O modelo
# ══════════════════════════════════════════════════════════════════════════
# Tudo aqui imita o que o dataloader deles entregou no treino. Cada número tem
# origem no `config.yaml` salvo na pasta do treino ou no `dados_prometheus.yaml`:
#   · `image_size: [336, 448]` é (altura, largura) — o `TF.resize` do dataloader
#     recebe [target_h, target_w]. O pulso do nosso dataset é 224x224 e VIRA
#     336x448 no treino, esticado; a inferência estica igual.
#   · papéis na ordem do `image_roles` do YAML, que é a ordem das tags no prompt.
#   · `robot_type: "unitree"` escolhe o embedding de corpo (`embodiment_types`).
#   · `arm_type: "dual"` escolhe o texto de modo de controle "só braços" no prompt.
IMG_HW = (336, 448)
PAPEIS = ("head_left", "cam_wrist_right")
TAREFA_TREINO = "pick up the white cup"     # a do `nosso_sim_pega`, 97% dos quadros


class PoliticaWLA:
    """A UnifoLM-WLA treinada, com a mesma entrada e a mesma saída do treino.

    `age()` recebe o que o robô vê e o que ele está fazendo, e devolve o
    pedaço de trajetória (30 passos, `action_horizon`) em pose ABSOLUTA da mão.
    A IK fica de fora de propósito: quem chama decide quantos passos executar
    antes de perguntar de novo.
    """

    def __init__(self, pasta_modelo: Path, base_vlm: Path, dispositivo: str = "cuda",
                 pesos: str = "final_model/model.safetensors"):
        import os
        # O DiT deles despacha a atenção para `flash_varlen` por padrão, e o
        # flash_attn não existe no aarch64 da GB10 (nem roda na athena, GLIBC
        # 2.31): a chamada vira `'NoneType' object is not callable` no meio da
        # primeira predição. O `prometheus.patch` deixa escolher o backend por
        # esta variável; `native` é o SDPA do PyTorch. Tem que estar definida
        # ANTES do import do `mmdit`, que lê na carga do módulo.
        os.environ.setdefault("WLA_ATTN_BACKEND", "native")
        import torch
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        sys.path.insert(0, str(RAIZ / "unifolm-wla"))
        from unifolm_wla.model.framework.base_framework import build_framework
        from unifolm_wla.model.framework.share_tools import apply_config_compat

        pasta_modelo = Path(pasta_modelo)
        cfg = OmegaConf.load(pasta_modelo / "config.yaml")
        cfg.framework.qwenvl.base_vlm = str(base_vlm)
        # O treino usou flash_attention_2, que não existe para o aarch64 da GB10.
        # SDPA é a mesma conta com outro kernel; muda o tempo, não o resultado.
        cfg.framework.qwenvl.attn_implementation = "sdpa"
        cfg = apply_config_compat(cfg)

        self.torch = torch
        self.vla = build_framework(cfg)
        estado = load_file(str(pasta_modelo / pesos))
        faltou, sobrou = self.vla.load_state_dict(estado, strict=False)
        # O checkpoint é o state_dict INTEIRO, VLM congelado incluído. Chave
        # faltando no action_model quer dizer pesos aleatórios na cabeça de ação
        # — o modelo roda e cospe lixo. Melhor morrer aqui.
        ruins = [k for k in faltou if "action_model" in k]
        if ruins:
            raise RuntimeError(f"{len(ruins)} pesos do action_model não carregaram, ex.: {ruins[:3]}")
        self.carga = {"faltou": len(faltou), "sobrou": len(sobrou), "total": len(estado)}
        del estado
        self.vla.to(dispositivo).eval()       # eval: sem o ruído de estado do treino
        self.esc = Escalas()
        self.espiao = EspiaoAtencao(self.vla)

    def age(self, imagens: dict, texto: str, T_esq, T_dir, garra_esq, garra_dir,
            cintura3) -> dict:
        from PIL import Image

        h, w = IMG_HW
        fotos = [Image.fromarray(np.asarray(imagens[p], np.uint8)).resize((w, h), Image.BILINEAR)
                 for p in PAPEIS]
        exemplo = {
            "image": fotos,
            "image_roles": list(PAPEIS),
            "lang": texto,
            "state": monta_estado(self.esc, T_esq, T_dir, garra_esq, garra_dir, cintura3),
            "state_mask": mascara_estado(T_esq is not None).astype(np.float32),
            "action_mask": mascara_acao(T_esq is not None).astype(np.float32),
            "arm_type": "dual",
            "robot_type": "unitree",
        }
        self.espiao.comeca()
        with self.torch.no_grad():
            saida = self.vla.predict_action([exemplo])
        mapas = self.espiao.mapas()
        norm = np.asarray(saida["normalized_actions"])[0]          # (30, 54)
        out = desnormaliza(self.esc, norm, T_esq, T_dir)
        # As imagens exatamente como entraram no modelo (336x448), com o mapa.
        out["atencao"] = {
            papel: sobrepoe(np.asarray(foto), mapa)
            for papel, foto, mapa in zip(PAPEIS, fotos, mapas)
        }
        return out


# ══════════════════════════════════════════════════════════════════════════
# Atenção — para OLHAR, não para mudar o resultado
# ══════════════════════════════════════════════════════════════════════════
class EspiaoAtencao:
    """Para onde a cabeça de ação olhou, pedaço a pedaço de cada imagem.

    QUAL atenção: a do DiT (a cabeça de ação), e não a do VLM. No MMDiT deles os
    30 tokens de ação e os ~centenas de tokens do VLM entram numa atenção
    CONJUNTA; a linha "ação → token de imagem" é literalmente o quanto cada passo
    da trajetória consultou cada pedaço de imagem. É a atenção que decide o
    movimento. A do VLM diria o que o modelo de linguagem achou relevante, o que
    é outra pergunta.

    COMO, sem mexer no código deles: o processador chama `dispatch_attention_fn`
    pelo nome importado no módulo `mmdit`. Troco esse nome por um invólucro que,
    quando ligado, calcula softmax(QKᵀ/√d) com a MESMA máscara e guarda só as
    linhas de ação contra as colunas de texto/imagem — e depois chama a função
    original, com as mesmas entradas. A saída do modelo não muda: o cálculo extra
    é jogado fora depois de promediado. Custa uns milissegundos por camada.

    O que se guarda: média sobre cabeças, sobre os 30 passos de ação e sobre as
    16 camadas do ÚLTIMO passo de difusão (o que produz a ação entregue; os três
    anteriores ainda operam sobre ação quase toda ruído).
    """

    CAMADAS = 16

    def __init__(self, vla):
        import math
        import torch
        from unifolm_wla.model.modules.action_model.DiT_modules import mmdit as mm

        self.torch, self.mm, self.vla = torch, mm, vla
        self.ligado = False
        self._pesos = []
        self._seq_acao = None
        self._chamada = 0
        passos = int(getattr(vla.action_model, "num_inference_timesteps", 4) or 4)
        self._pula = self.CAMADAS * (passos - 1)
        self._entradas = None
        espiao = self

        call_orig = mm.QwenDoubleStreamAttnProcessor2_0.__call__
        disp_orig = mm.dispatch_attention_fn

        # `functools.wraps` NÃO é enfeite. O `Attention.forward` do diffusers lê a
        # ASSINATURA do `__call__` do processador para decidir quais kwargs
        # repassar, e joga fora o resto com um aviso. Sem o `wraps`, a assinatura
        # vista era `(attn, hidden_states, *a, **k)` e ele descartava o
        # `image_rotary_emb` e o `encoder_hidden_states_mask` em TODAS as 384
        # chamadas: a rede rodava sem RoPE. Medido em 22/09: o deslocamento
        # previsto em 30 passos caiu de 1,6 cm para 0,2 cm — o robô "lento que
        # não chega". O `wraps` põe `__wrapped__`, e o `inspect.signature` segue.
        import functools

        @functools.wraps(call_orig)
        def call(proc, attn, hidden_states, *a, **k):
            espiao._seq_acao = hidden_states.shape[1]
            return call_orig(proc, attn, hidden_states, *a, **k)

        @functools.wraps(disp_orig)
        def dispatch(q, k, v, *a, **kw):
            espiao._chamada += 1
            # Só o ÚLTIMO passo de difusão. As chamadas dos passos anteriores
            # não entram na média, então nem se calcula: isso é o grosso do
            # custo (4 passos x 16 camadas de einsum extra).
            if espiao.ligado and espiao._seq_acao and espiao._chamada > espiao._pula:
                with torch.no_grad():
                    sa = espiao._seq_acao
                    w = torch.einsum("bshd,bthd->bhst", q[:, :sa].float(), k.float())
                    w = w / math.sqrt(q.shape[-1])
                    m = kw.get("attn_mask")
                    if m is not None:
                        w = w.masked_fill(~m.bool(), float("-inf"))
                    w = w.softmax(-1)[..., sa:]              # ação → texto/imagem
                    espiao._pesos.append(w.mean(dim=(1, 2))[0].cpu())
            return disp_orig(q, k, v, *a, **kw)

        mm.QwenDoubleStreamAttnProcessor2_0.__call__ = call
        mm.dispatch_attention_fn = dispatch

        # O que o VLM recebeu: precisamos do `input_ids` (onde estão os tokens de
        # imagem) e do `image_grid_thw` (a grade de cada imagem).
        interface = vla.qwen_vl_interface
        build_orig = interface.build_qwenvl_inputs

        def build(*a, **k):
            r = build_orig(*a, **k)
            espiao._entradas = r
            return r

        interface.build_qwenvl_inputs = build
        cfg = interface.model.config
        self.id_imagem = getattr(cfg, "image_token_id", None)
        self.funde = getattr(getattr(cfg, "vision_config", cfg), "spatial_merge_size", 2)

    def comeca(self):
        self._pesos.clear()
        self._chamada = 0
        self.ligado = True

    def mapas(self) -> list:
        """Um mapa (h, w) normalizado para [0, 1] por imagem, na ordem dos papéis."""
        self.ligado = False
        if not self._pesos or self._entradas is None:
            return []
        w = self.torch.stack(self._pesos[-self.CAMADAS:]).mean(0).numpy()   # (L_txt,)
        ids = self._entradas["input_ids"][0].cpu().numpy()
        pos = np.flatnonzero(ids == self.id_imagem)
        grades = self._entradas["image_grid_thw"].cpu().numpy()
        saida, i = [], 0
        for t, h, ww in grades:
            h2, w2 = int(h) // self.funde, int(ww) // self.funde
            n = int(t) * h2 * w2
            m = w[pos[i:i + n]].reshape(int(t), h2, w2).mean(0)
            i += n
            m = m - m.min()
            saida.append(m / (m.max() + 1e-12))
        return saida


def sobrepoe(img: np.ndarray, mapa: np.ndarray, alfa: float = 0.55) -> np.ndarray:
    """Mapa de calor por cima da imagem que o modelo viu (já no tamanho do treino)."""
    from PIL import Image
    import matplotlib.cm as cm

    h, w = img.shape[:2]
    m = np.asarray(Image.fromarray((mapa * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    cor = (cm.inferno(m)[..., :3] * 255).astype(np.float32)
    return (img.astype(np.float32) * (1 - alfa) + cor * alfa).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
# Teste da matemática, sem modelo e sem GPU
# ══════════════════════════════════════════════════════════════════════════
def testa() -> int:
    """Ida e volta: pose absoluta → relativa → normalizada → de volta.

    Se isto fecha, o único elo que resta é carregar os pesos: a conversão de
    unidade, a ordem das fatias e as estatísticas estão certas.
    """
    rng = np.random.default_rng(7)
    esc = Escalas()
    erros = []

    def se3_aleatorio():
        R = rotvec_to_matrix(rng.normal(0, 0.4, 3))
        return pose_to_se3(rng.normal([0.4, -0.1, 1.0], 0.05), R)

    for _ in range(200):
        T_atual = se3_aleatorio()
        T_futuro = se3_aleatorio()
        garra = float(rng.uniform(0, 1))
        cintura = rng.normal(0, 0.2, 3).astype(np.float32)

        # ── ida: exatamente o que o dataloader deles faz ──────────────
        rel = se3_to_xyz_rotvec(se3_inverse(T_atual) @ T_futuro[None])[0]
        a = np.zeros(UNIFIED_DIM, np.float32)
        a[SLICES["right_xyz_rotvec"]] = esc.aplica(rel, esc.acao["ee_dir"])
        a[SLICES["right_gripper"]] = esc.aplica([garra], esc.acao["garra_dir"])
        a[SLICES["waist_joint"]] = esc.aplica(cintura, esc.acao["cintura"])

        # ── volta ─────────────────────────────────────────────────────
        v = desnormaliza(esc, a[None], None, T_atual)
        erros.append((
            np.abs(v["pose_dir"][0] - T_futuro).max(),
            abs(v["garra_dir"][0] - garra),
            np.abs(v["cintura"][0] - cintura).max(),
        ))

    e = np.array(erros)
    print(f"{len(e)} sorteios, pior caso:")
    print(f"   pose da mão   {e[:, 0].max():.3e}   (metros / elementos da matriz)")
    print(f"   garra         {e[:, 1].max():.3e}")
    print(f"   cintura       {e[:, 2].max():.3e} rad")

    # ── estado: só o xyz é normalizado, o rot6d passa cru ─────────────
    T = se3_aleatorio()
    est = monta_estado(esc, None, T, None, 0.3, np.zeros(3))
    i = STATE_SLICES["right_xyz_rot6d"].start
    rot6d_cru = matrix_to_rot6d(T[:3, :3])
    ok_rot = np.abs(est[i + 3:i + 9] - rot6d_cru).max()
    print(f"\nestado de 60 dims montado; rot6d passou cru (erro {ok_rot:.1e})")
    print(f"   xyz normalizado -> {est[i:i + 3].round(3)}  (deve ficar perto de [-1, 1])")

    pior = e.max()
    print("\n✅ a volta fecha" if pior < 1e-4 else f"\n❌ erro grande: {pior:.3e}")
    return 0 if pior < 1e-4 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--testa", action="store_true", help="confere a ida e volta da matemática")
    args = p.parse_args()
    if args.testa:
        raise SystemExit(testa())
    p.print_help()


if __name__ == "__main__":
    main()
