import json, sys, numpy as np
E = sys.argv[1]; FPS = 30
samples, cut = [], {}
import glob, re
for f in sorted(glob.glob(f"{E}/actions/samples-*.json")):   # chamada recusada nao gera amostras: pula o buraco
    k = int(re.search(r"samples-(\d+)", f).group(1))
    samples += json.load(open(f)); cut[k] = len(samples)
z0 = json.load(open(f"{E}/truth/step-000.json"))["cup"][2]
for k in [int(x) for x in sys.argv[2].split(",")]:
    S = samples[:cut[k]]; hold = S[-2 * FPS:]
    before = max([x["cup_tilt_deg"] for x in S if not x["fingers"]] or [0])
    ret = all(x["cup"][2] > z0 + 0.048 and len(x["fingers"]) >= 2 and not x["cup_table"] for x in hold)
    th = float(np.mean([x["cup_tilt_deg"] for x in hold]))
    h = (hold[-1]["cup"][2] - z0) * 100
    print(f"se encerrasse apos a chamada {k}: altura {h:.1f} cm, retida 2 s {ret}, inclin. antes do fecho {before:.1f}, na janela {th:.1f} -> ACEITO={ret and before < 15 and th < 15}")
