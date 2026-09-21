#!/usr/bin/env python
"""
Junta o dataset REAL (white_cup_on_dripper) com o SIM (sim_copo_mujoco) num
dataset só, para co-treino do FastWAM-D.

Roda DEPOIS que `gerar_dataset_mujoco.py` terminar — ler o dataset sim
enquanto ele ainda está gravando dá `ArrowInvalid: Parquet magic bytes not
found`, porque o `save_episode` só fecha o rodapé do parquet ao finalizar o
chunk. Não é corrupção, é ler um arquivo no meio da escrita.

    python juntar_sim_real.py

── Por que o REAL vem PRIMEIRO na lista ───────────────────────────────────
`aggregate_datasets` reindexed episódios por deslocamento sequencial, na
ordem da lista: o primeiro dataset mantém os índices originais (0..26), o
segundo é deslocado a partir daí (27..). Com o real primeiro, os episódios
0-26 do resultado são exatamente os do real — e o filtro de episódios ruins
do config de treino (`episodes: [1..24]`, excluindo 0/25/26) continua válido
sem tradução nenhuma.
"""
import os
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

AQUI = Path(__file__).resolve().parent

# Caminhos por variável de ambiente: o merge roda tanto aqui (teste local) —
# onde os três datasets moram sob `meu_dataset/` — quanto na athena, onde o
# real já vive em `/data/mrwlker/meu_dataset/...` (cópia própria, não a do
# hercules — ver comentário em maquinas/athena/launch_fastwamd_lora.sh) e o sim chega
# por rsync no mesmo padrão. Sem variável, cai no layout local.
REAL_REPO = os.environ.get("REAL_REPO", "Mrwlker/white_cup_on_dripper_2026-08-11")
REAL_ROOT = Path(os.environ["REAL_ROOT"]) if "REAL_ROOT" in os.environ \
    else AQUI / "meu_dataset" / "white_cup_on_dripper_2026-08-11"
SIM_REPO = os.environ.get("SIM_REPO", "Mrwlker/sim_copo_mujoco_2026-09-03")
SIM_ROOT = Path(os.environ["SIM_ROOT"]) if "SIM_ROOT" in os.environ \
    else AQUI / "meu_dataset" / "sim_copo_mujoco_2026-09-03"

SAIDA_REPO = os.environ.get("SAIDA_REPO", "Mrwlker/cotreino_copo_sim_real_2026-09-03")
SAIDA_ROOT = Path(os.environ["SAIDA_ROOT"]) if "SAIDA_ROOT" in os.environ \
    else AQUI / "meu_dataset" / "cotreino_copo_sim_real_2026-09-03"


def confere_esquema():
    """Aborta ANTES de gastar tempo agregando se algo não bater.

    `aggregate_datasets` já recusa por dentro se os `features` não forem
    idênticos, mas o erro lá é uma asserção genérica — aqui a diferença exata
    aparece, campo a campo.
    """
    real = LeRobotDatasetMetadata(REAL_REPO, root=REAL_ROOT)
    sim = LeRobotDatasetMetadata(SIM_REPO, root=SIM_ROOT)

    print(f"real: {real.total_episodes} episódios, {real.total_frames} quadros")
    print(f"sim:  {sim.total_episodes} episódios, {sim.total_frames} quadros")

    problemas = []
    if real.fps != sim.fps:
        problemas.append(f"fps difere: real={real.fps} sim={sim.fps}")
    if real.robot_type != sim.robot_type:
        problemas.append(f"robot_type difere: real={real.robot_type!r} sim={sim.robot_type!r}")

    rk, sk = set(real.features), set(sim.features)
    if rk != sk:
        problemas.append(f"chaves só no real: {rk - sk}\n   chaves só no sim: {sk - rk}")
    for k in rk & sk:
        if real.features[k] != sim.features[k]:
            problemas.append(f"feature '{k}' difere:\n   real: {real.features[k]}\n   sim:  {sim.features[k]}")

    if problemas:
        print("\n❌ esquemas não batem — aggregate_datasets recusaria de qualquer forma:")
        for p in problemas:
            print("  -", p)
        raise SystemExit(1)
    print("✅ esquemas idênticos (fps, robot_type, features)\n")
    return real.total_episodes, sim.total_episodes


def main():
    if SAIDA_ROOT.exists():
        print(f"❌ {SAIDA_ROOT} já existe. Apague antes de juntar de novo.")
        raise SystemExit(1)

    n_real, n_sim = confere_esquema()

    print(f"⏳ agregando {REAL_REPO} + {SIM_REPO} → {SAIDA_REPO} ...")
    aggregate_datasets(
        repo_ids=[REAL_REPO, SIM_REPO],
        aggr_repo_id=SAIDA_REPO,
        roots=[REAL_ROOT, SIM_ROOT],
        aggr_root=SAIDA_ROOT,
    )
    print(f"✅ {SAIDA_ROOT}")
    print(f"   real: episódios 0..{n_real - 1} (mesmos índices de origem — foi o")
    print(f"         PRIMEIRO da lista, então não houve deslocamento)")
    print(f"   sim:  episódios {n_real}..{n_real + n_sim - 1}")
    print(f"   total: {n_real + n_sim} episódios")
    print()
    print("   O config de treino do pi05 exclui 0, 25 e 26 do real (episódio 26")
    print("   tem 6 quadros, tomada abortada; 0 e 25 foram julgados ruins na")
    print("   revisão de 20/08 — ver fastwamdepth_white_cup_on_dripper.yaml).")
    print("   Esses índices não mudaram na agregação: usar `episodes: [1..24]")
    print(f"   + {n_real}..{n_real + n_sim - 1}]` no YAML de co-treino.")


if __name__ == "__main__":
    main()
