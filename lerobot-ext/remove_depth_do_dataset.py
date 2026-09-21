#!/usr/bin/env python
"""
Tira `observation.images.head_camera_depth` do caminho de leitura de um dataset.

    python remove_depth_do_dataset.py <raiz> [--aplicar] [--reverter]

── POR QUE ISTO EXISTE ────────────────────────────────────────────────────
`aggregate_datasets` copia vídeo por arquivo, mas quando um arquivo de origem
cabe no que já está aberto ele APPENDA — e appendar mp4 codificado através da
fronteira entre dois datasets produz um arquivo cujo índice de seek só vale
até a emenda. Medido no `cotreino_completo_2026-09-11`:

    file-017.mp4  27.806 s, 834.189 quadros, 1.150 episódios
    seek em   0 s → 1 quadro     seek em  1.000 s → 0 quadros
    seek em 100 s → 1 quadro     seek em 20.000 s → 0 quadros

A emenda está em ~202 s: é onde acabam os 10 últimos episódios de profundidade
do nosso dataset e começam os 1.140 do externo. Seek antes disso funciona,
depois não. O treino morre com

    FrameTimestampError: No frames could be decoded from ...file-017.mp4
    in the timestamp range [9943.43, 9943.43]

O RGB não sofre porque os arquivos dele já chegam perto do limite de 100 MB e
a rotação abre arquivo novo (4 nossos + 24 do ext = 28, nenhum concatenado). A
profundidade dos externos é zeros puros, comprime a quase nada, e por isso
coube tudo num arquivo só — cruzando a fronteira.

── POR QUE REMOVER, E NÃO CONSERTAR O VÍDEO ───────────────────────────────
Porque este dado não vale o conserto. 84 % da profundidade deste dataset é
FALSA — os externos gravam zeros, que o decodificador devolve como 10,0 m
uniformes. Nenhuma corrida pode ligar `use_depth_3d` aqui. O `LeRobotDataset`
decodifica todo `video_key` que existe nas features, use o treino ou não:
então hoje o dataset paga decodificação de 848x480 por quadro para jogar o
resultado fora, e ainda quebra.

── O QUE ISTO FAZ ─────────────────────────────────────────────────────────
`video_keys` é derivado de `features` (`dataset_metadata.py:391`). Tirar a
chave do `meta/info.json` basta para o leitor nunca mais abrir esses mp4.

Os vídeos NÃO são apagados: a pasta é renomeada para `.sem_uso`, e o
`--reverter` desfaz tudo. As colunas `videos/...head_camera_depth/*` do
`meta/episodes` ficam onde estão — o leitor itera sobre `video_keys`, não
sobre as colunas, então elas são ignoradas.
"""
import json, os, shutil, sys
from pathlib import Path

CHAVE = "observation.images.head_camera_depth"


def main(raiz: Path, aplicar: bool, reverter: bool) -> int:
    info_p = raiz / "meta/info.json"
    if not info_p.exists():
        print(f"❌ {info_p} não existe"); return 1
    info = json.loads(info_p.read_text())
    vid = raiz / "videos" / CHAVE
    guardado = raiz / "videos" / (CHAVE + ".sem_uso")
    bak = info_p.with_suffix(".json.com_depth")

    if reverter:
        if not bak.exists():
            print("❌ não há backup para reverter"); return 1
        if aplicar:
            shutil.copy2(bak, info_p)
            if guardado.exists():
                guardado.rename(vid)
            print("✔ revertido: profundidade de volta nas features")
        else:
            print("(seco) reverteria info.json e a pasta de vídeos")
        return 0

    tem_feature = CHAVE in info.get("features", {})
    print(f"  feature presente: {tem_feature}")
    print(f"  pasta de vídeos:  {'existe' if vid.exists() else 'ausente'}"
          f"{' (já guardada)' if guardado.exists() else ''}")
    if not tem_feature and not vid.exists():
        print("\n✅ nada a fazer."); return 0
    if not aplicar:
        print("\nRode de novo com --aplicar."); return 0

    if not bak.exists():
        shutil.copy2(info_p, bak)
    info["features"].pop(CHAVE, None)
    info_p.write_text(json.dumps(info, indent=4))
    if vid.exists() and not guardado.exists():
        vid.rename(guardado)
    restantes = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    print(f"\n✔ aplicado. video_keys agora: {restantes}")
    print(f"  backup do info.json: {bak.name}")
    print(f"  vídeos guardados em: {guardado.name}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1]), "--aplicar" in sys.argv, "--reverter" in sys.argv))
