"""Converte os GLB do Luiz (chaleira, tampa, coador, pote, scoop; todos normalizados a ~1 m no maior eixo, y para cima)
em malhas para o MuJoCo: OBJ visual em z para cima com a textura PNG, escala pela dimensao real declarada em
assets/dimensions.json (maior eixo em metros), e pecas convexas de colisao por CoACD (a chaleira e o pote sao ocos e a
alca precisa ser pegavel; casco convexo unico fecharia a boca e a alca). Saida em assets/mesh/<nome>/.
Uso: ../g1-cup-grasp/.venv/bin/python scripts/convert_glb.py [nome ...]"""
import argparse, json, sys, os, hashlib, shutil
import numpy as np, trimesh, coacd
from PIL import Image
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dims = json.load(open(os.path.join(ROOT, "assets", "dimensions.json")))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("names", nargs="*")
parser.add_argument("--out-root", type=Path, required=True, help="Fresh immutable asset generation directory")
args = parser.parse_args()
names = args.names or list(dims)
unknown = set(names) - set(dims)
if unknown: parser.error(f"Unknown objects: {sorted(unknown)}")
args.out_root.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, args.out_root / "convert_glb.py")
shutil.copy2(os.path.join(ROOT, "assets", "dimensions.json"), args.out_root / "dimensions.json")
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # y-up (glTF) -> z-up (MuJoCo)
for n in names:
    s = trimesh.load(os.path.join(ROOT, "assets", "glb", f"{n}.glb"), force="scene"); g = list(s.geometry.values())[0]
    spec = dims[n]; V0 = g.vertices @ R.T; ext = V0.max(0) - V0.min(0)
    if "fit" in spec:
        # escala anisotropica: z pela altura real e xy pelo diametro caracteristico (percentil do raio, para ignorar alcas e hastes)
        f = spec["fit"]; sz = f["height_m"] / ext[2]
        c = (V0[:, :2].max(0) + V0[:, :2].min(0)) / 2; rad = np.hypot(V0[:, 0] - c[0], V0[:, 1] - c[1])
        if "band" in f:
            # ancorar o diametro numa FAIXA de altura, que e como o desenho cota. O percentil sobre a
            # malha inteira ancorou a jarra no lugar errado: a base saiu com 100,8 mm de raio quando
            # o JAR-001 da 55, ou seja a peca ficou com quase o dobro da largura real.
            zb = V0[:, 2] - V0[:, 2].min(); alt = V0[:, 2].max() - V0[:, 2].min()
            sel = (zb >= f["band"][0] * alt) & (zb <= f["band"][1] * alt)
            # Anchor BOTH center and radius in the measured base band. The
            # bounding box of the entire object is shifted by handle/spout.
            c = (V0[sel, :2].max(0) + V0[sel, :2].min(0)) / 2
            rad = np.linalg.norm(V0[:, :2] - c, axis=1)
            d_glb = 2 * np.percentile(rad[sel], f.get("radius_percentile", 90))
        else:
            d_glb = 2 * np.percentile(rad, f.get("radius_percentile", 50))
        sxy = f["diameter_m"] / d_glb
        v = V0 * [sxy, sxy, sz]; scale = [float(sxy), float(sxy), float(sz)]
    else:
        axis = {"x": 0, "y": 1, "z": 2}[spec["axis"]]; sc = spec["size_m"] / ext[axis]; v = V0 * sc; scale = float(sc)
    center_xy = c * sxy if "fit" in spec else (v[:, :2].max(0) + v[:, :2].min(0)) / 2
    v -= [*center_xy, v[:, 2].min()]   # origem: centro xy, base em z=0
    if "shorten_cone" in spec:
        # o modelo 3D do coador tem o saco de tecido mais fundo que o objeto real (SUP-001: 70 mm); comprime verticalmente
        # so os vertices do saco (dentro do cone, longe da haste) para a ponta subir ate a profundidade real
        sc = spec["shorten_cone"]; top = v[:, 2].max()
        cx, cy = sc.get("axis_xy", [0.0, 0.0]); rad = np.hypot(v[:, 0] - cx, v[:, 1] - cy)
        haste = (v[:, 0] < sc["x_min"]) & (np.abs(v[:, 1]) < sc.get("haste_half_y", 0.012))   # a haste fina nao e o saco
        sel = (v[:, 2] > sc["z_min"]) & (rad < sc["r_max"]) & (~haste)
        depth_now = top - v[sel][:, 2].min(); k = sc["depth_m"] / depth_now
        v[sel, 2] = top - (top - v[sel, 2]) * k
        print(f"   {n}: saco encurtado de {depth_now*1000:.0f} para {sc['depth_m']*1000:.0f} mm ({sel.sum()} vertices)")
    out = str(args.out_root / n); os.makedirs(out, exist_ok=False)
    m = trimesh.Trimesh(v, g.faces, visual=g.visual, process=False)
    tex = g.visual.material.baseColorTexture; tex.save(os.path.join(out, f"{n}_tex.png"))
    # OBJ visual com uv (MuJoCo le v/vt/f)
    uv = g.visual.uv
    with open(os.path.join(out, f"{n}_visual.obj"), "w") as f:
        for p in v: f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in uv: f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for ia, ib, ic in g.faces + 1: f.write(f"f {ia}/{ia} {ib}/{ib} {ic}/{ic}\n")
    # colisao: CoACD
    # Fresh directory: never delete or overwrite historical collision pieces.
    mesh = coacd.Mesh(v.astype(np.float64), g.faces.astype(np.int64))
    parts = coacd.run_coacd(mesh, threshold=spec.get("coacd_threshold", 0.05), max_convex_hull=spec.get("max_parts", 24), preprocess_mode="auto", mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3)
    for i, (pv, pf) in enumerate(parts):
        trimesh.Trimesh(pv, pf).export(os.path.join(out, f"{n}_col{i:02d}.obj"))
    info = {"model": "gpt-6-astra", "source_sha256": hashlib.sha256(Path(ROOT, "assets", "glb", f"{n}.glb").read_bytes()).hexdigest(), "base_center_source_xy": c.tolist() if "fit" in spec else None, "source": f"assets/glb/{n}.glb", "scale": scale, "extents_m": (v.max(0) - v.min(0)).tolist(), "collision_parts": len(parts), "dimension_spec": spec}
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=2)
    print(n, "escala", np.round(scale, 4), "extensao m", np.round(v.max(0) - v.min(0), 4), "pecas de colisao", len(parts), flush=True)
