"""Remove a feature de profundidade de um dataset LeRobot — nao a usamos e o decodificador
tropeça nela (desencontro de timestamp de um quadro)."""
import json, shutil, sys, glob
from pathlib import Path
import pandas as pd
CHAVE = "observation.images.head_camera_depth"
d = Path(sys.argv[1])
v = d / "videos" / CHAVE
if v.exists(): shutil.rmtree(v); print("  videos de profundidade removidos")
i = json.load(open(d/"meta"/"info.json"))
if i["features"].pop(CHAVE, None): json.dump(i, open(d/"meta"/"info.json","w"), indent=4); print("  info.json limpo")
p = d/"meta"/"stats.json"
if p.exists():
    s = json.load(open(p))
    if s.pop(CHAVE, None): json.dump(s, open(p,"w")); print("  stats.json limpo")
for f in glob.glob(str(d/"meta"/"episodes"/"**"/"*.parquet"), recursive=True):
    df = pd.read_parquet(f); fora = [c for c in df.columns if CHAVE in c]
    if fora: df.drop(columns=fora).to_parquet(f, index=False); print(f"  {len(fora)} colunas fora de {Path(f).name}")
print("✅", d)
