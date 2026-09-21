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
        saida[f"garra_{lado}"] = esc.desfaz(a[:, fatia_g], esc.acao[f"garra_{lado[0]}sq"
                                            if lado == "esq" else "garra_dir"])[:, 0]
    saida["cintura"] = esc.desfaz(a[:, SLICES["waist_joint"]], esc.acao["cintura"])
    return saida


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
