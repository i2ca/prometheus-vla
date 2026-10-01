"""Fit supplied kettle GLB to drawing landmarks; preserve each asset generation.

This is a declared geometric approximation, not a scan of the real object.
Base band and upper rim band are explicit assumptions. Original UVs are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import coacd
import numpy as np
import trimesh


def fit(vertices, dimensions):
    v = vertices.copy()
    v[:, 2] -= v[:, 2].min()
    v[:, 2] *= dimensions['altura_total'] / 1000 / np.ptp(v[:, 2])
    h = v[:, 2].max()
    base = v[:, 2] <= .08 * h
    upper = v[:, 2] >= .90 * h
    center = (v[base, :2].min(0) + v[base, :2].max(0)) / 2
    v[:, :2] -= center
    base_scale = dimensions['diametro_base'] / 1000 / np.ptp(v[base, :2], axis=0)
    # Handle and spout extend along X; Y measures the upper circular body.
    upper_scale = dimensions['diametro_externo_superior'] / 1000 / np.ptp(v[upper, 1])
    blend = np.clip((v[:, 2] / h - .08) / .82, 0, 1)
    scales = base_scale[None, :] * (1 - blend[:, None]) + upper_scale * blend[:, None]
    v[:, :2] *= scales
    # Extend only the outer handle region. Body geometry and base stay unchanged.
    body_radius = ((1 - blend) * dimensions['diametro_base'] +
                   blend * dimensions['diametro_externo_superior']) / 2000
    excess = np.maximum(v[:, 0] - body_radius, 0)
    target_max = v[:, 0].min() + dimensions['largura_total'] / 1000
    candidates = excess > 1e-9
    stretch = np.min((target_max - v[candidates, 0]) / excess[candidates])
    v[:, 0] += stretch * excess
    return v, {'base_band_fraction': [0, .08], 'upper_band_fraction': [.90, 1],
               'source_base_center_xy': center.tolist(), 'base_xy_scale': base_scale.tolist(),
               'upper_xy_scale': float(upper_scale), 'handle_extra_stretch': float(stretch)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.out / Path(__file__).name)
    refs = json.loads((root / 'assets/dimensions-reference.json').read_text())
    spec = refs['objects']['jarra']
    source = root / 'assets/glb/chaleira.glb'
    scene = trimesh.load(source, force='scene')
    mesh = list(scene.geometry.values())[0]
    original = mesh.vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    v, mapping = fit(original, spec['dimensions_mm'])
    mesh.visual.material.baseColorTexture.save(a.out / 'chaleira_tex.png')
    with (a.out / 'chaleira_visual.obj').open('x') as f:
        for p in v: f.write('v ' + ' '.join(f'{x:.9f}' for x in p) + '\n')
        for p in mesh.visual.uv: f.write('vt ' + ' '.join(f'{x:.9f}' for x in p) + '\n')
        for face in mesh.faces + 1: f.write('f ' + ' '.join(f'{i}/{i}' for i in face) + '\n')
    info = {'model': 'gpt-6-astra', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'drawing': spec, 'mapping': mapping, 'limitations': [
                'Four drawing dimensions do not determine the complete shape.',
                'Upper diameter measured across Y in the upper ten percent of height.',
                'Handle thickness and opening are inherited/deformed, not dimensioned in JAR-001.',
                'Collision geometry requires separate clearance verification.'],
            'measured_visual_mm': {'height': float(np.ptp(v[:, 2])*1000),
                'width_total': float(np.ptp(v[:, 0])*1000),
                'base_xy': (np.ptp(v[v[:, 2] <= .08*v[:, 2].max(), :2], axis=0)*1000).tolist(),
                'upper_y': float(np.ptp(v[v[:, 2] >= .90*v[:, 2].max(), 1])*1000)}}
    (a.out / 'landmarks.json').write_text(json.dumps(info, indent=2))
    parts = coacd.run_coacd(coacd.Mesh(v, mesh.faces.astype(np.int64)), threshold=.02,
        max_convex_hull=48, preprocess_mode='auto', mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3)
    for i, (pv, pf) in enumerate(parts):
        trimesh.Trimesh(pv, pf).export(a.out / f'chaleira_col{i:02d}.obj')
    info['collision_parts'] = len(parts)
    (a.out / 'info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
