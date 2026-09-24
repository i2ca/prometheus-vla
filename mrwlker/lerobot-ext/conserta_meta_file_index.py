#!/usr/bin/env python
"""
Conserta `meta/episodes/{chunk,file}_index` quando ele aponta para arquivos que
não existem.

── O sintoma ──────────────────────────────────────────────────────────────
`aggregate_datasets` monta o caminho do metadado de origem a partir DESSAS
COLUNAS (`aggregate.py:673-683`), não da listagem do diretório:

    FileNotFoundError: .../meta/episodes/chunk-000/file-001.parquet

── A causa ────────────────────────────────────────────────────────────────
O `cotreino_copo_sim_real_2026-09-03` saiu de um `aggregate_datasets` anterior
(`juntar_sim_real.py`, 03/09). Naquela corrida o contador de rotação de arquivo
avançou para 1 e carimbou `file_index = 1` em 162 das 302 linhas — mas o
`append_or_create_parquet_file` gravou TODAS as 302 no `file-000.parquet`. A
coluna ficou descrevendo um arquivo que nunca foi criado.

Nada se perdeu: as 302 linhas estão lá, `episode_index` vai de 0 a 301 sem
buraco, e todo arquivo de `data/` referenciado existe. Por isso o TREINO lê o
dataset sem reclamar — o `LeRobotDataset` varre o diretório — e só a agregação
quebra, que é a única coisa que confia na coluna.

── O conserto ─────────────────────────────────────────────────────────────
Para cada parquet de `meta/episodes/` no disco, carimba nas linhas dele o
chunk e o file do PRÓPRIO CAMINHO. O disco vira a verdade, que é o que a
coluna deveria ter dito desde o começo. Guarda `.bak` e troca com `os.replace`,
que é atômico — pode rodar com treino lendo o dataset.

    python conserta_meta_file_index.py <raiz> [--aplicar]

Sem `--aplicar` só relata.
"""
import os, re, sys, glob, shutil
import pyarrow as pa
import pyarrow.parquet as pq

PADRAO = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet$")


def main(raiz: str, aplicar: bool) -> int:
    arqs = sorted(glob.glob(os.path.join(raiz, "meta/episodes/*/*.parquet")))
    if not arqs:
        print(f"❌ nenhum meta/episodes/*.parquet em {raiz}")
        return 1

    total_erradas = 0
    for arq in arqs:
        m = PADRAO.search(arq)
        chunk_real, file_real = int(m.group(1)), int(m.group(2))
        t = pq.read_table(arq)
        cs = t.column("meta/episodes/chunk_index").to_pylist()
        fs = t.column("meta/episodes/file_index").to_pylist()
        erradas = sum(1 for c, f in zip(cs, fs) if (c, f) != (chunk_real, file_real))
        rel = os.path.relpath(arq, raiz)
        print(f"  {rel}: {t.num_rows} linhas, deveria ser ({chunk_real},{file_real}), "
              f"{erradas} erradas")
        if not erradas:
            continue
        total_erradas += erradas
        if not aplicar:
            continue

        novo = t
        for col, val in (("meta/episodes/chunk_index", chunk_real),
                         ("meta/episodes/file_index", file_real)):
            i = novo.schema.get_field_index(col)
            novo = novo.set_column(i, col,
                                   pa.array([val] * novo.num_rows, type=novo.column(i).type))
        bak = arq + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(arq, bak)
        tmp = arq + ".tmp"
        pq.write_table(novo, tmp)
        os.replace(tmp, arq)          # atômico: seguro com treino lendo
        print(f"    ✔ corrigido (backup em {os.path.basename(bak)})")

    if total_erradas and not aplicar:
        print(f"\n{total_erradas} linhas erradas. Rode de novo com --aplicar.")
    elif not total_erradas:
        print("\n✅ nada a consertar.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], "--aplicar" in sys.argv))
