"""Export legacy post-step recordings as causal observation -> next-command pairs.

Author: gpt-6-astra, 2026-09-16. Does not modify source recordings.
Run with .venv-lerobot/bin/python. Output must not exist.
Legacy row i is (s[i+1], command[i]); export (s[i+1], command[i+1]).
The final observation has no recorded future command and is omitted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from build_lerobot_dataset import read_video


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episodes', nargs='+', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--repo-id', default='local/g1-cup-causal-pilot')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Preserving existing output: {args.out}')
    first = json.loads((args.episodes[0] / 'meta.json').read_text())
    names, fps = first['names'], first['fps']
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError('Expected 29 unique joint names')
    sources = []
    for ep in args.episodes:
        meta = json.loads((ep / 'meta.json').read_text())
        state, action, times = [np.load(ep / f'{k}.npy') for k in ('observation.state', 'action', 'timestamp')]
        n = meta['frames']
        if meta['names'] != names or meta['fps'] != fps or n < 2:
            raise ValueError(f'Incompatible metadata: {ep}')
        if state.shape != (n, 29) or action.shape != (n, 29) or times.shape != (n,):
            raise ValueError(f'Incompatible array shapes: {ep}')
        if state.dtype != np.float32 or action.dtype != np.float32:
            raise ValueError(f'Expected float32 state/action: {ep}')
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.allclose(times, np.arange(n) / fps, atol=1e-5):
            raise ValueError(f'Invalid values/timestamps: {ep}')
        hashes = {str(p.relative_to(ep)): digest(p) for p in sorted(ep.rglob('*')) if p.is_file()}
        sources.append((ep, meta, state, action, hashes))
    features = {
        'action': {'dtype': 'float32', 'shape': (29,), 'names': names},
        'observation.state': {'dtype': 'float32', 'shape': (29,), 'names': names},
    }
    for key, shape, dtype in (
        ('head_camera', (480, 848, 3), 'video'),
        ('head_camera_depth', (480, 848, 1), 'image'),
        ('right_wrist_camera', (224, 224, 3), 'video'),
    ):
        features[f'observation.images.{key}'] = {'dtype': dtype, 'shape': shape, 'names': ['height', 'width', 'channels']}
    ds = LeRobotDataset.create(args.repo_id, fps, features, root=args.out, robot_type='unitree_g1_dex3',
                               use_videos=True, image_writer_threads=2, vcodec='h264')
    provenance = {'author_model': 'gpt-6-astra', 'status': 'causal-export pilot; not physical-robot validation',
                  'alignment': 'observation raw[i], action raw[i+1]; last raw observation omitted',
                  'time_origin': 'export t=0 corresponds to raw simulation approximately 1/fps after reset',
                  'omitted': first['omitted'], 'depth': first['depth'],
                  'camera_calibration': 'simulated extrinsics; not verified against physical head camera',
                  'episodes': []}
    for ep, meta, state, action, hashes in sources:
        n = len(state)
        head = read_video(ep / 'head_camera.mp4', 848, 480)
        wrist = read_video(ep / 'right_wrist_camera.mp4', 224, 224)
        if len(head) != n or len(wrist) != n:
            raise ValueError(f'Video frame count mismatch: {ep}')
        for i in range(n - 1):
            with Image.open(ep / 'head_depth' / f'frame-{i:06d}.png') as source:
                if source.mode != 'I;16' or source.size != (848, 480):
                    raise ValueError(f'Invalid depth frame: {ep}, {i}')
                depth = source.copy()
            ds.add_frame({'action': action[i + 1], 'observation.state': state[i],
                          'observation.images.head_camera': head[i],
                          'observation.images.right_wrist_camera': wrist[i],
                          'observation.images.head_camera_depth': depth, 'task': meta['task']})
        ds.save_episode()
        provenance['episodes'].append({'source': str(ep.resolve()), 'raw_frames': n, 'export_frames': n - 1,
                                       'cup_xy': meta.get('cup_xy'), 'cup_yaw_deg': meta.get('cup_yaw_deg'),
                                       'raw_sha256': hashes})
        del head, wrist
    ds.finalize()
    (args.out / 'meta' / 'sim_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'output': str(args.out), 'episodes': len(sources), 'alignment': provenance['alignment']}))


if __name__ == '__main__':
    main()
