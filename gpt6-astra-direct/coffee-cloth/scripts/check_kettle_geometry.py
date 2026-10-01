"""Independent exported-mesh dimensions and finite sphere clearance checks."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    asset = a.asset.resolve()
    info = json.loads((asset / 'info.json').read_text())
    v = trimesh.load(asset / 'chaleira_visual.obj', process=False).vertices
    h = np.ptp(v[:, 2]); base = v[:, 2] <= v[:, 2].min()+.08*h
    upper = v[:, 2] >= v[:, 2].min()+.90*h
    measured = np.r_[h, np.ptp(v[:, 0]), np.ptp(v[base, :2], axis=0), np.ptp(v[upper, 1])]*1000
    expected = np.array([232, 159, 110, 110, 90])
    xml = ET.Element('mujoco')
    assets = ET.SubElement(xml, 'asset')
    world = ET.SubElement(xml, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -1 1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'camera', name='side', pos='.03 -.65 .116', xyaxes='1 0 0 0 0 1', fovy='30')
    ET.SubElement(assets, 'texture', name='tex', type='2d', file=str(asset/'chaleira_tex.png'))
    ET.SubElement(assets, 'material', name='mat', texture='tex')
    ET.SubElement(assets, 'mesh', name='visual', file=str(asset/'chaleira_visual.obj'))
    ET.SubElement(world, 'geom', name='visual', type='mesh', mesh='visual', material='mat',
                  contype='0', conaffinity='0', group='1')
    for i in range(info['collision_parts']):
        name = f'chaleira_col{i:02d}'
        ET.SubElement(assets, 'mesh', name=name, file=str(asset/f'{name}.obj'))
        ET.SubElement(world, 'geom', name=name, type='mesh', mesh=name, group='3', rgba='.5 .6 .7 1')
    probe = ET.SubElement(world, 'body', name='probe', mocap='true', pos='0 0 1')
    ET.SubElement(probe, 'geom', name='probe', type='sphere', size='.009', group='4')
    ET.ElementTree(xml).write(a.out/'probe.xml')
    m = mujoco.MjModel.from_xml_path(str(a.out/'probe.xml')); d = mujoco.MjData(m)
    gs = [m.geom(f'chaleira_col{i:02d}').id for i in range(info['collision_parts'])]
    pg = m.geom('probe').id
    # Sideways entry, perpendicular to the handle's XZ plane. A free point
    # beyond the outside of the handle does not count as an opening.
    # These search bounds are recorded, not a claim of exhaustive grasp feasibility.
    xs = np.arange(.060, .0981, .002)
    zs = np.arange(.045, .2001, .003)
    ys = np.linspace(-.060, .060, 25)
    passes = []; best = None
    for x in xs:
        for z in zs:
            clearance = .1
            for y in ys:
                d.mocap_pos[0] = [x, y, z]; mujoco.mj_forward(m, d)
                clearance = min(clearance, *(mujoco.mj_geomDistance(m, d, pg, g, .1, None) for g in gs))
            row = {'x_m': float(x), 'z_m': float(z), 'clearance_m': float(clearance)}
            if best is None or clearance > best['clearance_m']: best = row
            if clearance >= .001: passes.append(row)
    d.mocap_pos[0] = [0, 0, 1]; mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, 480, 640)
    for name, group in [('visual', 1), ('collision', 3)]:
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[group] = 1
        renderer.update_scene(d, camera='side', scene_option=opt)
        cv2.imwrite(str(a.out/f'{name}.png'), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    renderer.close()
    report = {'model': 'gpt-6-astra', 'measured_mm': measured.tolist(), 'expected_mm': expected.tolist(),
              'dimension_order': ['height', 'width', 'base_x', 'base_y', 'upper_y'],
              'landmarks_pass_0_1mm': bool(np.all(abs(measured-expected)<.1)),
              'sphere_radius_m': .009, 'required_clearance_m': .001,
              'search_x_m': [float(xs[0]), float(xs[-1])], 'search_z_m': [float(zs[0]), float(zs[-1])],
              'path_y_m': [-.06, .06], 'path_sample_spacing_m': .005,
              'passing_paths': passes, 'best_path': best,
              'limitations': ['Discrete sphere sweep is not swept-volume proof or full-hand collision check.',
                              'Search can include exterior space: inspect visual/collision silhouette before accepting a handle path.',
                              'Dimensional landmarks do not prove complete surface fidelity.']}
    (a.out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='passing_paths'}))
    print('passing paths',len(passes))


if __name__ == '__main__': main()
